#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Game.py
@Time    :   2023/07/14 11:06:59
@Author  :   Zhou Zihao
@Version :   1.0
@Desc    :   None
'''

import os, json, sys
import gymnasium as gym
import numpy as np
import torch
import cv2
import base64
import time
from typing import Optional

import env
import algos
import skill
import utils
from teacher_policy import TeacherPolicy

# Trajectory logger (graceful fallback if unavailable)
try:
    from simulator.trajectory_logger import (
        TrajectoryLogger, StepRecord, EpisodeRecord
    )
    _TRAJ_LOGGER_AVAILABLE = True
except ImportError:
    _TRAJ_LOGGER_AVAILABLE = False

# Live visualizer event bus (graceful fallback if viz/ unavailable).
# event_bus imports no web deps, so this works without fastapi/uvicorn.
# Events are only emitted when the FastAPI server has initialised the bus
# (VIZ._ready == True), so notebook / CLI training pays ~zero overhead.
try:
    from viz.event_bus import VIZ as _VIZ_BUS
    _VIZ_BUS_OK = True
except Exception:
    _VIZ_BUS_OK = False


def _encode_frame_b64(rgb: np.ndarray) -> Optional[str]:
    """Encode an RGB frame as a PNG data-URI for the live dashboard."""
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")
    except Exception:
        return None

# Action name map for on-frame annotation
_ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle/open",
    6: "done",
}
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS  = 1
_LINE       = cv2.LINE_AA
_PAD        = 6
_LINE_H     = 18


def _annotate_frame(frame: np.ndarray, step: int, action: int,
                    ep_reward: float, plan: str = "",
                    success=None) -> np.ndarray:
    """Overlay step info, action name and symbolic plan on a BGR frame."""
    out = frame.copy()
    action_label = _ACTION_NAMES.get(int(action), f"#{action}")
    plan_short   = (plan[:50] + "…") if len(plan) > 53 else plan
    lines = [
        f"Step {step:3d}  reward {ep_reward:+.2f}",
        f"Action : {action_label}",
    ]
    if plan_short:
        lines.append(f"Plan   : {plan_short}")

    y = _LINE_H
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(out, (0, y - h - _PAD), (w + 2*_PAD, y + _PAD), (0, 0, 0), -1)
        cv2.putText(out, line, (_PAD, y), _FONT, _FONT_SCALE,
                    (255, 255, 255), _THICKNESS, _LINE)
        y += _LINE_H

    if success is not None:
        banner = " SUCCESS! " if success else " FAILED "
        bcolor = (0, 200, 0) if success else (0, 0, 200)
        H = out.shape[0]
        (bw, bh), _ = cv2.getTextSize(banner, _FONT, 0.8, 2)
        x0 = (out.shape[1] - bw) // 2
        cv2.rectangle(out, (x0-8, H-bh-20), (x0+bw+8, H-4), bcolor, -1)
        cv2.putText(out, banner, (x0, H-8), _FONT, 0.8, (255, 255, 255), 2, _LINE)

    return out

# Reflection system — optional, graceful degradation when disabled
try:
    from memory.reflection import EpisodeTrajectory
    from memory.memory_buffer import ReflectionMemory
    _REFLECTION_AVAILABLE = True
except ImportError:
    _REFLECTION_AVAILABLE = False

prefix = os.path.dirname(os.path.abspath(__file__))
task_info_json = os.path.join(prefix, "prompt/task_info.json")


class _NoTeacher:
    """
    Uniform no-op teacher used for the PPO-only phase (use_teacher_policy=False).

    Returns a uniform action distribution every step so the kickstarting
    target is uninformative (and the runner also sets iter_with_ks=0, making
    kickstarting fully inert). It has no planner and never intervenes — a true
    PPO baseline with zero LLM / planner overhead.

    The interface mirrors exactly the parts of TeacherPolicy that Game.collect(),
    Game.evaluate(), and Game.train() touch: __call__, reset(), the `planner`
    attribute, and the `intervention_stats` property.
    """

    def __init__(self, action_space):
        self.action_space = action_space
        self.planner = None

    def reset(self):
        pass

    def __call__(self, obs):
        return np.ones(self.action_space, dtype=np.float32) / self.action_space

    @property
    def intervention_stats(self):
        return {
            "total_interventions":   0,
            "stuck_events":          0,
            "oscillation_events":    0,
            "repeated_plan_events":  0,
            "failed_inter_events":   0,
        }


class Game:
    def __init__(self, args, training=True, exp_config=None):
        # Experiment config gates which components are active (see
        # experiment_config.py). None → full legacy behavior (teacher always on).
        self.exp_config = exp_config

        # init seed
        self.seed = args.seed
        self.setup_seed(args.seed)
        
        # init env
        self.load_task_info(args.task, args.frame_stack, args.offline_planner, args.soft_planner)

        # init logger
        self.logger = utils.create_logger(args, training)
        
        # init policy
        if args.loaddir:
            model_dir = os.path.join(args.logdir, args.policy, args.task, args.loaddir, args.loadmodel)
            policy = torch.load(model_dir, weights_only=False)  # torch>=2.6 requires explicit opt-in for full objects
        else:
            policy = None
        self.device = args.device
        self.batch_size = args.batch_size
        self.recurrent = args.recurrent
        # self.student_policy = policy
        self.student_policy = algos.PPO(policy, 
                                        self.obs_space,
                                        self.action_space,
                                        self.device, 
                                        self.logger.dir, 
                                        batch_size=self.batch_size, 
                                        recurrent=self.recurrent)
        
        # init buffer
        self.gamma = args.gamma
        self.lam = args.lam
        self.buffer = algos.Buffer(self.gamma, self.lam, self.device)

        # other settings
        self.n_itr = args.n_itr
        self.traj_per_itr = args.traj_per_itr
        self.num_eval = args.num_eval
        self.eval_interval = args.eval_interval
        self.save_interval = args.save_interval
        self.total_steps = 0

        # Reflection system (disabled by default, activated via set_reflection_system)
        self._reflector = None
        self._reflection_memory = None
        self._episode_count = 0

        # Trajectory logger — full step-level introspection
        self._traj_logger: Optional["TrajectoryLogger"] = None
        if _TRAJ_LOGGER_AVAILABLE:
            self._traj_logger = TrajectoryLogger(
                log_root=self.logger.dir,
                enabled=True,
            )
            print(f"[Game] TrajectoryLogger active → {self.logger.dir}/trajectory/")

        # Bug 6: configurable health-check thresholds
        self.planner_failure_threshold:    float = getattr(args, "planner_failure_threshold",    0.8)
        self.reflection_failure_threshold: float = getattr(args, "reflection_failure_threshold", 0.9)
        self.parser_rejection_threshold:   float = getattr(args, "parser_rejection_threshold",   0.5)

        # Intervention / autonomy tracking
        self._prev_intervention_count = 0
        
        
    def set_reflection_system(self, reflector, memory=None) -> None:
        """
        Activate the reflection-memory system.

        Parameters
        ----------
        reflector : QwenReflector
            Generates reflections from episode trajectories.
        memory : ReflectionMemory, optional
            Buffer that stores and retrieves reflections.
            A default buffer (maxlen=20) is created automatically if None.

        After calling this, the planner will have reflection context injected
        into each LLM query starting from the second episode.

        Example
        -------
        from memory.reflection import QwenReflector
        from memory.memory_buffer import ReflectionMemory

        reflector = QwenReflector(backend='ollama', model='qwen2.5:7b')
        memory    = ReflectionMemory(maxlen=20, top_k=5)
        game.set_reflection_system(reflector, memory)
        """
        if not _REFLECTION_AVAILABLE:
            print("[Game] WARNING: memory module not found. Reflection disabled.")
            return

        self._reflector = reflector
        self._reflection_memory = memory or ReflectionMemory(maxlen=20, top_k=5)

        # Wire memory into the planner
        if hasattr(self.teacher_policy, "planner"):
            self.teacher_policy.planner.set_reflection_memory(self._reflection_memory)
            print("[Game] Reflection memory wired into planner.")

    def _check_training_health(self, itr: int) -> bool:
        """
        Bug 6: Sanity checks before continuing training.

        Returns True if training is healthy, False if it should stop early.
        Issues [WARNING] logs for each detected problem.
        """
        healthy = True
        planner = getattr(self.teacher_policy, "planner", None)

        # Planner failure rate
        if planner is not None and hasattr(planner, "planner_stats"):
            ps = planner.planner_stats
            fallback_rate = ps.get("offline_fallback_rate", 0) + ps.get("explore_fallback_rate", 0)
            if fallback_rate > self.planner_failure_threshold:
                print(
                    f"[WARNING] Planner failure rate exceeded threshold at itr {itr}: "
                    f"{fallback_rate:.1%} > {self.planner_failure_threshold:.1%}. "
                    "Real Qwen inference may not be working."
                )
                healthy = False

            rejection_rate = (
                ps.get("parser_rejections", 0) / max(ps.get("planner_calls", 1), 1)
            )
            if rejection_rate > self.parser_rejection_threshold:
                print(
                    f"[WARNING] Excessive parser rejection frequency at itr {itr}: "
                    f"{rejection_rate:.1%} > {self.parser_rejection_threshold:.1%}."
                )
                healthy = False

            # Log planner state to stdout
            print(
                f"[PlannerStats itr={itr}] "
                f"calls={ps['planner_calls']} "
                f"online_ok={ps['online_success']} "
                f"offline_fb={ps['offline_fallback']} "
                f"explore_fb={ps['explore_fallback']} "
                f"cache_hits={ps['cache_hits']} "
                f"parser_rej={ps['parser_rejections']}"
            )

        # Reflection failure rate
        if self._reflector is not None:
            rs = self._reflector.stats
            total = max(rs.get("generated", 0) + rs.get("failed", 0), 1)
            fail_rate = rs.get("failed", 0) / total
            if fail_rate > self.reflection_failure_threshold:
                print(
                    f"[WARNING] Reflection system unstable at itr {itr}: "
                    f"failure rate {fail_rate:.1%} > {self.reflection_failure_threshold:.1%}."
                )
                healthy = False

        return healthy

    def _generate_and_store_reflection(self, trajectory: "EpisodeTrajectory") -> None:
        """
        Generate a reflection from a completed trajectory and store it.
        Called at the end of each collect() episode when the reflection
        system is active. Silently skips when reflector is offline/unavailable.
        """
        if self._reflector is None or self._reflection_memory is None:
            return
        reflection = self._reflector.reflect(trajectory)
        if reflection:
            added = self._reflection_memory.add_memory(
                reflection=reflection,
                episode_id=self._episode_count,
                success=trajectory.success,
                ep_len=trajectory.ep_len,
                total_reward=trajectory.total_reward,
                failure_type=trajectory.classify_failure(),   # → reflection cluster (req 11)
            )
            if added:
                print(
                    f"[Game] Reflection stored (ep {self._episode_count}, "
                    f"{'success' if trajectory.success else 'failure'}): "
                    f"{reflection[:80]}"
                )

    def setup_seed(self, seed):
        # setup seed for Numpy, Torch and LLM, not for env
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        
    def load_task_info(self, task, frame_stack, offline, soft):
        print(f"[INFO]: resetting the task: {task}")
        with open(task_info_json, 'r') as f:
            task_info = json.load(f)
        task = task.lower()
        
        env_fn = utils.make_env_fn(task_info[task]['configurations'], 
                                   render_mode="rgb_array", 
                                   frame_stack = frame_stack)
        self.env = utils.WrapEnv(env_fn)
        self.obs_space = utils.get_obss_preprocessor(self.env.observation_space)[0]
        self.action_space = self.env.action_space.n
        self.max_ep_len = self.env.unwrapped.max_steps

        self.task   = task                       # store for step-loop failure detection
        prefix      = task_info[task]['description'] + task_info[task]['example']
        avoid_lava  = (task == "lavadoorkey")   # lava-aware navigation for LavaDoorKey
        self.teacher_policy = TeacherPolicy(
            task, offline, soft, prefix,
            self.action_space,
            self.env.unwrapped.agent_view_size,
            avoid_lava=avoid_lava,
        )

        # Phase-gating: ppo_only (use_teacher_policy=False) trains with NO teacher.
        # Swap the planner-backed teacher for a uniform no-op so PPO learns as a
        # pure baseline. All other component gates (epochs, lr, entropy,
        # iter_with_ks, reflection wiring) are applied by experiment_runner.build_game().
        if self.exp_config is not None and not self.exp_config.use_teacher_policy:
            self.teacher_policy = _NoTeacher(self.action_space)
            print("[Game] use_teacher_policy=False → PPO-only baseline (no teacher).")

            
    def train(self):
        start_time = time.time()
        for itr in range(self.n_itr):
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            ## collecting ##
            sample_start = time.time()
            self.buffer.clear()
            n_traj = self.traj_per_itr
            for _ in range(n_traj):
                self.collect()
            while len(self.buffer) < self.batch_size * 2:
                self.collect()
                n_traj += 1
            total_steps = len(self.buffer)    
            samp_time = time.time() - sample_start
            print("{:.2f} s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            recent_success_rate = sum(r > 0 for r in self.buffer.ep_returns) / max(n_traj, 1)
            mean_losses = self.student_policy.update_policy(self.buffer, recent_success_rate)
            opt_time = time.time() - optimizer_start
            try:
                print("{:.2f} s to optimizer| loss {:6.3f}, entropy {:6.3f}, kickstarting {:6.3f}.".format(opt_time, mean_losses[0], mean_losses[1], mean_losses[2]))
            except:
                print(mean_losses)

            ## evaluate ##
            if itr % self.eval_interval == 0 and itr > 0:
                evaluate_start = time.time()
                eval_returns = []
                eval_lens = []
                eval_success = []
                for i in range(self.num_eval):
                    eval_outputs = self.evaluate(itr, record_frames=False, deterministic=True)
                    eval_returns.append(eval_outputs[0])
                    eval_lens.append(eval_outputs[1])
                    eval_success.append(eval_outputs[2])
                eval_time = time.time() - evaluate_start
                print("{:.2f} s to evaluate.".format(eval_time))
            
            if itr % self.save_interval == 0 and itr > 0:
                self.student_policy.save(str(itr))

            # Bug 6: health check every eval interval
            if itr % self.eval_interval == 0 and itr > 0:
                self._check_training_health(itr)
            
            ## log ##
            if self.logger is not None:
                avg_len = np.mean(self.buffer.ep_lens)
                avg_reward = np.mean(self.buffer.ep_returns)
                std_reward = np.std(self.buffer.ep_returns)
                success_rate = sum(i > 0 for i in self.buffer.ep_returns) / n_traj
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Return (train)', round(avg_reward,2)) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Episode Length (train)', round(avg_len,2)) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Success Rate (train)', round(success_rate,2)) + "\n")
                if itr % self.eval_interval == 0 and itr > 0:
                    avg_eval_reward = np.mean(eval_returns)
                    avg_eval_len = np.mean(eval_lens)
                    eval_success_rate = np.sum(eval_success) / self.num_eval
                    sys.stdout.write("| %25s | %15s |" % ('Return (eval)', round(avg_eval_reward,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Episode Length (eval) ', round(avg_eval_len,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Success Rate (eval) ', round(eval_success_rate,2)) + "\n")
                    self.logger.add_scalar("Test/Return", avg_eval_reward, itr)
                    self.logger.add_scalar("Test/Eplen", avg_eval_len, itr)
                    self.logger.add_scalar("Test/Success Rate", eval_success_rate, itr)
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.flush()

                self.logger.add_scalar("Train/Return Mean", avg_reward, itr)
                self.logger.add_scalar("Train/Return Std", std_reward, itr)
                self.logger.add_scalar("Train/Eplen", avg_len, itr)
                self.logger.add_scalar("Train/Success Rate", success_rate, itr)
                self.logger.add_scalar("Train/Loss", mean_losses[0], itr)
                self.logger.add_scalar("Train/Mean Entropy", mean_losses[1], itr)
                self.logger.add_scalar("Train/Kickstarting Loss", mean_losses[2], itr)
                self.logger.add_scalar("Train/Policy Loss", mean_losses[3], itr)
                self.logger.add_scalar("Train/Value Loss", mean_losses[4], itr)
                self.logger.add_scalar("Train/Kickstarting Coef", self.student_policy.ks_coef, itr)

                # Planner fallback + cache invalidation metrics
                planner = getattr(self.teacher_policy, "planner", None)
                if planner is not None and hasattr(planner, "planner_stats"):
                    ps = planner.planner_stats
                    self.logger.add_scalar("Planner/Online Success Rate",   ps["online_success_rate"],   itr)
                    self.logger.add_scalar("Planner/Offline Fallback Rate", ps["offline_fallback_rate"], itr)
                    self.logger.add_scalar("Planner/Explore Fallback Rate", ps["explore_fallback_rate"], itr)
                    self.logger.add_scalar("Planner/Cache Hit Rate",        ps["cache_hit_rate"],        itr)
                    self.logger.add_scalar("Planner/Parser Rejections",     ps["parser_rejections"],     itr)
                    self.logger.add_scalar("Planner/Cache Invalidations",
                                           ps.get("cache_invalidations", 0), itr)

                # Intervention / failure detector metrics
                iv_stats = getattr(self.teacher_policy, "intervention_stats", None)
                if iv_stats is not None:
                    total_iv = iv_stats["total_interventions"]
                    iv_this_itr = total_iv - self._prev_intervention_count
                    self._prev_intervention_count = total_iv
                    self.logger.add_scalar("Intervention/Total",            total_iv,                         itr)
                    self.logger.add_scalar("Intervention/This Iteration",   iv_this_itr,                      itr)
                    self.logger.add_scalar("Intervention/Stuck Events",     iv_stats["stuck_events"],         itr)
                    self.logger.add_scalar("Intervention/Oscillation",      iv_stats["oscillation_events"],   itr)
                    self.logger.add_scalar("Intervention/Repeated Plan",    iv_stats["repeated_plan_events"], itr)
                    self.logger.add_scalar("Intervention/Failed Interact",  iv_stats["failed_inter_events"],  itr)
                    # Autonomy proxy: fraction of steps WITHOUT intervention
                    traj_steps = max(len(self.buffer), 1)
                    autonomy = max(0.0, 1.0 - iv_this_itr / max(n_traj, 1))
                    self.logger.add_scalar("Intervention/PPO Autonomy Proxy", autonomy, itr)

                # Reflection system metrics
                if self._reflection_memory is not None:
                    self.logger.add_scalar(
                        "Reflection/Memory Size", len(self._reflection_memory), itr
                    )
                    rs = self._reflection_memory.stats
                    self.logger.add_scalar("Reflection/Total Added", rs["total_added"], itr)
                    self.logger.add_scalar("Reflection/Total Success", rs["total_success"], itr)
                    self.logger.add_scalar("Reflection/Total Failure", rs["total_failure"], itr)
                    self.logger.add_scalar("Reflection/Deduplicated", rs["total_deduplicated"], itr)
                if self._reflector is not None:
                    rrs = self._reflector.stats
                    self.logger.add_scalar("Reflection/Generated", rrs["generated"], itr)
                    self.logger.add_scalar("Reflection/Validated", rrs["validated"], itr)
                    self.logger.add_scalar("Reflection/Rejected", rrs["rejected"], itr)
                    self.logger.add_scalar("Reflection/Failed", rrs["failed"], itr)
                
        self.student_policy.save()

        # Save reflection memory alongside model checkpoint
        if self._reflection_memory is not None:
            mem_path = os.path.join(self.logger.dir, "reflection_memory.json")
            self._reflection_memory.save(mem_path)


    def collect(self):
        '''
        collect episodic data.
        '''
        with torch.no_grad():
            obs = self.env.reset()
            done = False
            ep_len = 0
            ep_reward = 0.0

            # reset student policy
            mask = torch.FloatTensor([1]).to(self.device)
            states = self.student_policy.model.init_states(self.device) if self.recurrent else None

            # reset teacher policy
            self.teacher_policy.reset()

            # init reflection trajectory tracker
            track_reflection = (
                _REFLECTION_AVAILABLE
                and self._reflector is not None
                and self._reflection_memory is not None
            )
            trajectory = EpisodeTrajectory() if (track_reflection and _REFLECTION_AVAILABLE) else None

            # ── begin trajectory logger episode ──────────────────────────────
            self._episode_count += 1
            ep_id = self._episode_count
            tl = self._traj_logger
            # Unified record: begin_episode() returns the SAME EpisodeRecord that
            # log_step() mutates, so per-episode flags (key/door detected, picked,
            # opened, goal) are written — eliminates the prior two-record bug.
            _ep_rec = tl.begin_episode(ep_id, env_name=getattr(self, "task", "")) \
                      if tl is not None else None
            if _ep_rec is None and _TRAJ_LOGGER_AVAILABLE:
                _ep_rec = EpisodeRecord(episode=ep_id, env_name=getattr(self, "task", ""))

            # Snapshot CUMULATIVE counters at episode start so the per-episode
            # values written below are true deltas (not running totals).
            _base_counts = self._counter_snapshot()

            # ── live visualizer wiring (active only under the viz server) ──────
            viz_on = _VIZ_BUS_OK and getattr(_VIZ_BUS, "_ready", False)
            viz_stop = getattr(self, "_viz_stop", None)
            viz_hz = 20.0
            viz_total_itr = self.n_itr
            if self.exp_config is not None:
                viz_hz = float(getattr(self.exp_config, "_viz_speed_hz", 20.0))
                viz_total_itr = getattr(self.exp_config, "total_iterations", self.n_itr)
            viz_itr = getattr(self, "_current_itr", 0)

            while not done and ep_len < self.max_ep_len:
                # Honor a viz Stop request mid-episode for responsive control.
                if viz_stop is not None and viz_stop.is_set():
                    break

                # get action from student policy
                dist, value, states = self.student_policy(
                    torch.Tensor(obs).to(self.device), mask, states
                )
                action    = dist.sample()
                log_probs = dist.log_prob(action)
                action    = action.to("cpu").numpy()

                # PPO entropy for logging
                try:
                    ppo_entropy = float(dist.entropy().mean().item())
                    ppo_value   = float(value.mean().item())
                except Exception:
                    ppo_entropy = -1.0
                    ppo_value   = -1.0

                # get action from teacher policy
                teacher_probs = self.teacher_policy(obs[0])

                # gather symbolic context for logging
                planner  = getattr(self.teacher_policy, "planner", None)
                obs_text = plan_str = ""
                if planner is not None:
                    try:
                        obs_text  = planner.mediator.RL2LLM(obs[0])
                        last_plan = getattr(planner, "dialogue_user", "")
                        plan_str  = last_plan.split("\n")[1] if "\n" in last_plan else last_plan
                    except Exception:
                        pass

                # interact with env
                next_obs, reward, done, info = self.env.step(action)
                step_reward = float(reward.squeeze()) if hasattr(reward, "squeeze") else float(reward)

                # record obs text + plan for reflection (with REAL reward)
                if trajectory is not None:
                    trajectory.add_step(obs_text, plan_str, step_reward)
                ep_reward  += step_reward

                # ── live step event for the dashboard (throttled) ─────────────
                if viz_on:
                    primitive = int(action.squeeze()) if hasattr(action, "squeeze") else int(action[0])
                    frame = _encode_frame_b64(self.env.unwrapped.get_mask_render())
                    _VIZ_BUS.emit_throttled("step", {
                        "iteration": viz_itr,
                        "total_itr": viz_total_itr,
                        "episode":   ep_id,
                        "step":      ep_len,
                        "obs_text":  obs_text,
                        "plan":      plan_str,
                        "action":    _ACTION_NAMES.get(primitive, f"#{primitive}"),
                        "reward":    step_reward,
                        "ep_reward": ep_reward,
                        "frame":     frame,
                    }, max_rate_hz=viz_hz)

                # ── log step to trajectory logger ─────────────────────────────
                if tl is not None and _TRAJ_LOGGER_AVAILABLE:
                    primitive = int(action.squeeze()) if hasattr(action, "squeeze") else int(action[0])
                    _ACTION_MAP = {0:"turn_left",1:"turn_right",2:"move_forward",
                                   3:"pick_up",4:"drop",5:"toggle_open",6:"wait"}
                    # extract agent position and carrying state from obs
                    try:
                        agent_map = obs[0, :, :, 3]
                        apos = np.argwhere(agent_map != 4)
                        apos = tuple(int(x) for x in apos[0]) if len(apos) > 0 else (0,0)
                        adir = int(obs[0, apos[0], apos[1], 3]) if len(apos) == 2 else 0

                        # Extract actual carrying state from observation (object type at agent position)
                        carrying_obj_type = int(obs[0, apos[0], apos[1], 0])
                        _CARRYING_MAP = {1:"nothing", 5:"key", 6:"ball", 7:"box"}
                        holding = _CARRYING_MAP.get(carrying_obj_type, "unknown")
                    except Exception:
                        apos, adir = (0, 0), 0
                        holding = "nothing"

                    iv_stats = getattr(self.teacher_policy, "intervention_stats", {})
                    try:
                        _t_act = int(np.argmax(np.asarray(teacher_probs, dtype=float)))
                    except Exception:
                        _t_act = -1
                    step_rec = StepRecord(
                        episode   = ep_id,
                        step      = ep_len,
                        agent_pos = apos,
                        obs       = obs_text,
                        holding   = holding,
                        plan      = plan_str,
                        subgoal   = plan_str.split(",")[0].strip() if plan_str else "",
                        action    = _ACTION_MAP.get(primitive, f"#{primitive}"),
                        action_id = primitive,
                        teacher_action = _t_act,
                        reward    = step_reward,
                        progress  = step_reward > 0,
                        ppo_entropy = ppo_entropy,
                        ppo_value   = ppo_value,
                        intervention = (
                            iv_stats.get("total_interventions", 0) >
                            getattr(self, "_prev_iv_count", 0)
                        ),
                    )
                    tl.log_step(step_rec)
                    self._prev_iv_count = iv_stats.get("total_interventions", 0)

                    # Log task events
                    if primitive == 3 and step_reward >= 0:
                        tl.log_event("key_picked_attempt", step=ep_len)
                    if primitive == 5:
                        if step_reward > 0:
                            tl.log_event("door_opened_SUCCESS", step=ep_len)
                            if _ep_rec: _ep_rec.door_opened = True
                        else:
                            tl.log_event("toggle_attempt_failed", step=ep_len)
                            # ColoredDoorKey: failed toggle while holding a key
                            # means the key colour does not match the door.
                            _task = getattr(self, "task", "")
                            if _task == "coloreddoorkey" and holding not in ("nothing", "unknown"):
                                if _ep_rec: _ep_rec.wrong_key_for_door = True
                                tl.log_event("wrong_key_colour_toggle", step=ep_len)

                    # LavaDoorKey: detect lava-death termination.
                    # Lava death = episode done + zero reward + not a timeout
                    # + last action was move_forward (primitive==2).
                    _task = getattr(self, "task", "")
                    if (_task == "lavadoorkey" and done and step_reward == 0
                            and primitive == 2 and ep_len < self.max_ep_len - 1):
                        if _ep_rec: _ep_rec.lava_collision = True
                        tl.log_event("lava_death", step=ep_len)
                        # Tell teacher so it injects lava-avoidance context
                        # into the next planning call.
                        if hasattr(self.teacher_policy, "report_lava_death"):
                            self.teacher_policy.report_lava_death()

                # store in PPO buffer
                self.buffer.store(
                    obs, action, reward,
                    value.to("cpu").numpy(),
                    log_probs.to("cpu").numpy(),
                    teacher_probs,
                )
                obs     = next_obs
                ep_len += 1

            if done:
                value = 0.
            else:
                value = self.student_policy(
                    torch.Tensor(obs).to(self.device), mask, states
                )[1].to("cpu").item()
            self.buffer.finish_path(last_val=value)

            # ── live episode-end event for the dashboard ──────────────────────
            if viz_on:
                _VIZ_BUS.emit("episode_end", {
                    "episode":      ep_id,
                    "ep_len":       ep_len,
                    "total_reward": ep_reward,
                    "success":      ep_reward > 0,
                    "frame":        _encode_frame_b64(self.env.unwrapped.get_mask_render()),
                })

            # ── reflection trigger (Phase 3 only) ─────────────────────────────
            # Reflect whenever ANY of:
            #   • the FailureDetector fired during THIS episode (a DETECTED
            #     failure) — this is the fix: previously detected failures only
            #     reflected if they happened to land on the every-3rd cadence;
            #   • the episode succeeded (capture what worked);
            #   • periodic fallback (every 3rd) for uneventful episodes.
            # Phase-gated: `trajectory` is non-None only when reflection is active
            # (Phase 3 wires the reflector), so Phase 2 never reflects.
            # Ollama load stays bounded: the reflector has a 30s timeout and skips
            # gracefully when busy, so failure-triggered reflection cannot stall.
            refl_this_ep = 0
            if trajectory is not None:
                trajectory.finish(success=(ep_reward > 0))
                iv_now = self._counter_snapshot().get("interventions", 0)
                failure_detected = (iv_now - _base_counts.get("interventions", 0)) > 0
                # Reflect on success ALWAYS; on a DETECTED FAILURE but rate-limited
                # to >= reflection_frequency episodes since the last reflection, so a
                # working reflector (~seconds/call on CPU) cannot stall training when
                # most episodes have detected failures.
                _gap   = getattr(self.exp_config, "reflection_frequency", 3) if self.exp_config is not None else 3
                _since = self._episode_count - getattr(self, "_last_reflect_ep", -(10 ** 9))
                if (ep_reward > 0) or (failure_detected and _since >= _gap):
                    _g0 = self._reflector.stats.get("generated", 0) if self._reflector else 0
                    self._generate_and_store_reflection(trajectory)
                    _g1 = self._reflector.stats.get("generated", 0) if self._reflector else 0
                    refl_this_ep = max(0, _g1 - _g0)
                    self._last_reflect_ep = self._episode_count

            # ── end trajectory logger episode ─────────────────────────────────
            # All counts below are PER-EPISODE deltas (cumulative snapshot at start
            # vs now). Cumulative phase totals live in results/<phase>/llm_stats.json.
            if tl is not None and _ep_rec is not None and _TRAJ_LOGGER_AVAILABLE:
                _now_counts = self._counter_snapshot()
                _d = {k: max(0, _now_counts[k] - _base_counts[k]) for k in _now_counts}

                _ep_rec.iteration_id  = int(getattr(self, "_current_itr", -1))
                _ep_rec.total_reward  = ep_reward
                _ep_rec.total_steps   = ep_len
                _ep_rec.success       = ep_reward > 0
                _ep_rec.timeout       = ep_len >= self.max_ep_len
                _ep_rec.termination_reason = "success" if ep_reward > 0 else \
                                             ("timeout" if ep_len >= self.max_ep_len else "done")
                # Per-episode planner counters (deltas)
                _ep_rec.llm_calls           = _d["online_success"]                 # successful online LLM plan calls this episode
                _ep_rec.cache_hits          = _d["cache_hits"]                      # plan-cache hits this episode
                _ep_rec.cache_invalidations = _d["cache_invalidations"]
                _ep_rec.plans_generated     = max(0, _d["planner_calls"] - _d["cache_hits"])  # cache-misses → fresh plans
                # Per-episode intervention counters (deltas)
                _ep_rec.interventions      = _d["interventions"]
                _ep_rec.oscillation_events = _d["oscillation"]
                _ep_rec.stuck_events       = _d["stuck"]
                # Per-episode reflections actually generated
                _ep_rec.reflections        = refl_this_ep

                # ── Calculate per-episode execution funnel metrics ────────────────────
                # Calculate funnel metrics for THIS episode using trajectory data
                try:
                    from experiment_runner import _calculate_episode_funnel
                    # Get trajectory indices for this episode
                    traj = list(self.buffer.traj_idx)
                    if len(traj) >= 2:
                        ep_idx = len(traj) - 2  # Current episode is the last one
                        s, e = traj[ep_idx], traj[ep_idx + 1]
                        act_arr = np.array(self.buffer.actions).astype(int)
                        obs_l = self.buffer.obs
                        funnel = _calculate_episode_funnel(s, e, act_arr, obs_l)
                        # Populate episode record with funnel metrics
                        _ep_rec.key_visible_steps = funnel["key_visible_steps"]
                        _ep_rec.key_adjacent_steps = funnel["key_adjacent_steps"]
                        _ep_rec.key_picked_steps = funnel["key_picked_steps"]
                        _ep_rec.door_adjacent_steps = funnel["door_adjacent_steps"]
                        _ep_rec.door_facing_steps = funnel["door_facing_steps"]
                        _ep_rec.door_opened_steps = funnel["door_opened_steps"]
                except Exception as e_funnel:
                    # If funnel calculation fails, just skip it (doesn't break training)
                    pass

                tl.end_episode(_ep_rec)
                tl.flush()
        
        
    def _counter_snapshot(self) -> dict:
        """Read the current CUMULATIVE counters (planner / intervention / reflector).

        Used at episode start and end to compute per-episode deltas for the
        trajectory logger. Pure read — does not mutate any subsystem.
        Returns zeros for any subsystem not present (e.g. ppo_only has no planner).
        """
        planner = getattr(self.teacher_policy, "planner", None)
        ps = planner.planner_stats if planner is not None else {}
        iv = getattr(self.teacher_policy, "intervention_stats", {}) or {}
        rg = getattr(self._reflector, "stats", {}) if self._reflector is not None else {}
        return {
            "online_success":      int(ps.get("online_success", 0)),
            "cache_hits":          int(ps.get("cache_hits", 0)),
            "planner_calls":       int(ps.get("planner_calls", 0)),
            "cache_invalidations": int(ps.get("cache_invalidations", 0)),
            "interventions":       int(iv.get("total_interventions", 0)),
            "oscillation":         int(iv.get("oscillation_events", 0)),
            "stuck":               int(iv.get("stuck_events", 0)),
            "generated":           int(rg.get("generated", 0)),
        }

    def _get_plan_text(self) -> str:
        """Return current symbolic plan string from planner (best-effort)."""
        try:
            planner = self.teacher_policy.planner
            raw = getattr(planner, "dialogue_user", "")
            if "\n" in raw:
                return raw.split("\n")[1].strip()
            cache = getattr(planner, "plans_cache", {})
            if cache:
                return list(cache.values())[-1]
        except Exception:
            pass
        return ""

    def evaluate(self, itr=None, seed=None, record_frames=True,
                 deterministic=False, teacher_policy=False, annotate=True):
        """
        Run one evaluation episode.

        Parameters
        ----------
        record_frames : save video to log dir (annotated when annotate=True)
        annotate      : overlay action name + symbolic plan on each video frame
        """
        with torch.no_grad():
            # init env
            seed = seed if seed else np.random.randint(1000000)
            obs = self.env.reset(seed)
            done = False
            ep_len = 0
            ep_return = 0.
            last_action = 0

            if teacher_policy:
                self.teacher_policy.reset()
            else:
                mask   = torch.Tensor([1.]).to(self.device)
                states = self.student_policy.model.init_states(self.device) if self.recurrent else None

            # init video directory
            if record_frames:
                img_array = []
                raw_init = self.env.unwrapped.get_mask_render()
                init_bgr = cv2.cvtColor(raw_init, cv2.COLOR_RGB2BGR)
                img_array.append(
                    _annotate_frame(init_bgr, 0, 0, 0.0, plan="(start)") if annotate else init_bgr
                )
                dir_name = 'teacher video' if teacher_policy else 'video'
                dir_path = os.path.join(self.logger.dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)

            while not done and ep_len < self.max_ep_len:
                if teacher_policy:
                    probs = self.teacher_policy(obs[0])
                    if probs is None:
                        raise ValueError("Teacher policy returned None")
                    if deterministic:
                        action = np.argmax(probs)
                    else:
                        action = np.random.choice(self.action_space, p=probs)
                else:
                    dist, value, states = self.student_policy(
                        torch.Tensor(obs).to(self.device), mask, states
                    )
                    if deterministic:
                        action = torch.argmax(dist.probs).unsqueeze(0).to("cpu").numpy()
                    else:
                        action = dist.sample().to("cpu").numpy()

                last_action = int(action) if np.isscalar(action) else int(action.flat[0])

                obs, reward, done, info = self.env.step(action)
                ep_return += float(reward.squeeze())
                ep_len    += 1

                if record_frames:
                    raw = self.env.unwrapped.get_mask_render()
                    bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                    if annotate:
                        plan = self._get_plan_text()
                        # show success/failure banner on terminal frame
                        term_success = (ep_return > 0) if done else None
                        bgr = _annotate_frame(bgr, ep_len, last_action,
                                              ep_return, plan=plan,
                                              success=term_success)
                    img_array.append(bgr)

            ep_success = 1 if ep_return > 0 else 0

            # save video
            if record_frames and img_array:
                h, w = img_array[-1].shape[:2]
                video_name = "%s-%s.avi" % (itr, seed) if itr else "%s.avi" % seed
                video_path = os.path.join(dir_path, video_name)
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    fps=3,
                    frameSize=(w, h),
                )
                for frame in img_array:
                    writer.write(frame)
                writer.release()

            return ep_return, ep_len, ep_success
    
        
if __name__ == '__main__':
    pass


