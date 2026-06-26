#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
experiment_runner.py — Research pipeline for LLM4Teach ablation studies

Provides:
    build_experiment_config   (re-exported from experiment_config)
    run_phase(config, ...)    — train one phase, return history dict
    evaluate_phase(game, ...) — eval-only run, return metrics dict
    save_results(...)         — persist metrics as JSON + CSV row
    aggregate_results(...)    — combine all phases into summary.csv
    run_research_pipeline(...) — execute all five phases automatically

All metric keys that do not apply to a disabled component are logged as 0,
so downstream analysis code never encounters missing keys.
"""

import os
import sys
import csv
import json
import time
import numpy as np
from types import SimpleNamespace
from typing import Optional, Dict, Any

from experiment_config import (
    ExperimentConfig, build_experiment_config, PHASE_ORDER
)

# Visualizer event bus — optional
try:
    from viz.event_bus import VIZ as _VIZ
    _VIZ_OK = True
except ImportError:
    _VIZ_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Metric template — every key is always present (disabled → 0)
# ─────────────────────────────────────────────────────────────────────────────

# Counter metrics that are CUMULATIVE in the underlying subsystems. For each we
# emit a per-iteration *_delta and a cumulative *_total. The base name used in the
# snapshot dict is the key here; legacy aliases (below) are kept for back-compat.
_COUNTER_KEYS = [
    "planner_consultations",   # plan() calls per step (NOT LLM calls) — was "planner_calls"
    "cache_hits",
    "cache_misses",
    "online_success",          # actual successful online LLM requests
    "interventions",
    "memory_retrievals",
    "cache_invalidations",
    "reflections_generated",
]

# Legacy metric name → snapshot base name (kept as *_total aliases for back-compat).
_LEGACY_ALIASES = {
    "planner_calls":         "planner_consultations",
    "cache_hits":            "cache_hits",
    "cache_misses":          "cache_misses",
    "online_success":        "online_success",
    "interventions":         "interventions",
    "memory_retrievals":     "memory_retrievals",
    "cache_invalidations":   "cache_invalidations",
    "reflections_generated": "reflections_generated",
}


def _zero_metrics() -> dict:
    m = dict(
        # Core performance (per-iteration)
        success_rate=0.0,
        average_reward=0.0,
        episode_length=0.0,
        entropy=0.0,
        # Loss breakdown (per-iteration means from PPO.update_policy)
        total_loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        ks_loss=0.0,            # KS (distillation) loss magnitude
        # Teacher / kickstarting
        teacher_agreement=0.0,
        kickstarting_coef=0.0,  # KS decay schedule (ks_coef over iterations)
        entropy_coef=0.0,       # entropy coefficient (hyperparameter)
        ks_applied=0,           # 1 if KS loss is actually added this iter, else 0
        # Per-action-type teacher agreement (student executed == teacher argmax)
        navigation_agreement=0.0,   # over steps where teacher wants left/right/forward
        pickup_agreement=0.0,       # over steps where teacher wants PICKUP(3)
        toggle_agreement=0.0,       # over steps where teacher wants TOGGLE(5)
        # Execution counts (teacher requested vs student executed vs succeeded)
        teacher_requested_pickup=0,
        student_executed_pickup=0,
        successful_pickups=0,       # student pickup that actually grabbed a key
        teacher_requested_toggle=0,
        student_executed_toggle=0,
        successful_door_opens=0,    # toggle that caused closed→open door state transition
        # 3-stage door chain counters (episodes per iteration)
        ep_reached_door_with_key=0, # ep where agent held key AND saw door
        ep_opened_door=0,           # ep where door state actually changed closed→open
        ep_reached_goal=0,          # ep where agent reached goal (reward > 0)
        # 6-step execution funnel (step counts per iteration)
        key_visible_steps=0,        # steps with key on floor in view
        key_adjacent_steps=0,       # steps where agent is 1 cell from a key
        key_picked_steps=0,         # steps where agent is carrying a key
        door_adjacent_steps=0,      # steps where door is directly in front
        door_facing_steps=0,        # same as door_adjacent (in-front = facing)
        door_opened_steps=0,        # steps where toggle caused closed→open
        # Planner (legacy cumulative aliases — equal to *_total)
        planner_calls=0,        # alias of planner_consultations_total
        cache_hits=0,
        cache_misses=0,
        online_success=0,
        cache_invalidations=0,
        reflection_calls=0,
        # Memory
        memory_retrievals=0,
        state_memory_hits=0,
        episode_memory_hits=0,
        episode_memory_writes=0,
        episode_memory_reads=0,
        state_memory_writes=0,
        state_memory_reads=0,
        # Replanning
        replans=0,
        replan_attempts=0,
        successful_replans=0,
        replan_failures=0,
        # Failure / intervention
        interventions=0,
        failure_counts=0,
        failure_class_distribution={},
        # Reflection
        reflections_generated=0,
        reflections_used=0,
        reflection_cluster_distribution={},
    )
    # Explicit per-iteration (*_delta) and cumulative (*_total) counter columns.
    for _k in _COUNTER_KEYS:
        m[f"{_k}_delta"] = 0
        m[f"{_k}_total"] = 0
    # New instrumentation columns (stable schema; populated in _extract_metrics).
    for _k in _NEW_LOG_KEYS:
        m[_k] = 0
    m["iteration_id"] = 0
    m["ppo_loss"] = 0.0
    m["overall_teacher_agreement"] = 0.0
    return m


# Instrumentation-only per-iteration columns added for PPO-learning / planner /
# reflection diagnostics (all read-only; computed in _extract_metrics).
_NEW_LOG_KEYS = [
    "teacher_requested_navigation", "student_executed_navigation",
    "pickup_attempts", "toggle_attempts",
    "pickup_success_rate", "toggle_success_rate", "door_open_success_rate",
    "prompt_injections_count", "retrieved_reflections_count",
    "key_never_found", "key_visible_but_navigation_failed",
    "key_acquired_door_interaction_failed", "wrong_key_for_door", "timeout_failure",
    "planner_requested_pickup_at_key", "planner_requested_navigation_at_key",
    "planner_requested_toggle_at_door", "planner_requested_navigation_at_door",
    "unique_symbolic_states_seen", "cache_hit_rate",
    # 3-stage door chain — separates "can't reach door" / "can't open door" / "can't reach goal"
    "ep_reached_door_with_key", "ep_opened_door", "ep_reached_goal",
    # 6-step execution funnel — per-iteration step counts
    "key_visible_steps", "key_adjacent_steps", "key_picked_steps",
    "door_adjacent_steps", "door_facing_steps", "door_opened_steps",
]


def _phase_counter_snapshot(game) -> dict:
    """Read the CUMULATIVE counters from the live subsystems (pure read).

    Returns one int per name in _COUNTER_KEYS. Used at iteration start and end to
    derive per-iteration deltas. Does not mutate planner/teacher/reflector/memory.
    """
    planner = getattr(getattr(game, "teacher_policy", None), "planner", None)
    ps = planner.planner_stats if (planner is not None and hasattr(planner, "planner_stats")) else {}
    iv = getattr(getattr(game, "teacher_policy", None), "intervention_stats", {}) or {}
    refl = getattr(game, "_reflector", None)
    rs = refl.stats if refl is not None else {}
    mem = getattr(game, "_reflection_memory", None)
    ms = mem.stats if mem is not None else {}

    pc = int(ps.get("planner_calls", 0))
    ch = int(ps.get("cache_hits", 0))
    return {
        "planner_consultations": pc,
        "cache_hits":            ch,
        "cache_misses":          max(0, pc - ch),
        "online_success":        int(ps.get("online_success", 0)),
        "interventions":         int(iv.get("total_interventions", 0)),
        "memory_retrievals":     int(ms.get("total_retrievals", 0)),
        "cache_invalidations":   int(ps.get("cache_invalidations", 0)),
        "reflections_generated": int(rs.get("generated", 0)),
    }


def _check_metric_invariants(m: dict, itr: int) -> None:
    """Warn (never crash) if per-iteration counter invariants are violated."""
    try:
        hits, miss, consult = m["cache_hits_delta"], m["cache_misses_delta"], m["planner_consultations_delta"]
        if hits + miss != consult:
            print(f"[Runner][WARN][itr {itr}] cache_hits_delta+cache_misses_delta "
                  f"({hits}+{miss}) != planner_consultations_delta ({consult})")
        if m["online_success_delta"] > miss:
            print(f"[Runner][WARN][itr {itr}] online_success_delta ({m['online_success_delta']}) "
                  f"> cache_misses_delta ({miss})")
        if m["memory_retrievals_delta"] > miss:
            print(f"[Runner][WARN][itr {itr}] memory_retrievals_delta ({m['memory_retrievals_delta']}) "
                  f"> cache_misses_delta ({miss})")
        # Interaction-execution invariants (warn-only).
        def _w(cond, msg):
            if cond:
                print(f"[Runner][WARN][itr {itr}] {msg}")
        _w(m.get("successful_pickups", 0) > m.get("pickup_attempts", 0),
           f"successful_pickups ({m.get('successful_pickups')}) > pickup_attempts ({m.get('pickup_attempts')})")
        _w(m.get("successful_door_opens", 0) > m.get("toggle_attempts", 0),
           f"successful_door_opens ({m.get('successful_door_opens')}) > toggle_attempts ({m.get('toggle_attempts')})")
        _trp = m.get("teacher_requested_pickup", 0) or 0
        _trt = m.get("teacher_requested_toggle", 0) or 0
        _w(m.get("successful_pickups", 0) > _trp and _trp > 0,
           f"successful_pickups ({m.get('successful_pickups')}) > teacher_requested_pickup ({_trp})")
        _w(m.get("planner_requested_pickup_at_key", 0) > _trp and _trp > 0,
           f"planner_requested_pickup_at_key ({m.get('planner_requested_pickup_at_key')}) > teacher_requested_pickup ({_trp})")
        _w(m.get("planner_requested_toggle_at_door", 0) > _trt and _trt > 0,
           f"planner_requested_toggle_at_door ({m.get('planner_requested_toggle_at_door')}) > teacher_requested_toggle ({_trt})")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Game factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_args(cfg: ExperimentConfig, savedir: str,
               task: str = "SimpleDoorKey",
               device: str = "cpu",
               seed: int = 42,
               has_live_llm: bool = False) -> SimpleNamespace:
    """Build an args namespace compatible with Game.__init__.

    offline_planner logic:
      - use_planner=False (ppo_only)          → offline_planner=True  (no teacher at all)
      - use_planner=True  + no live LLM       → offline_planner=True  (use pre-computed plans)
      - use_planner=True  + QwenLLM provided  → offline_planner=False (online Qwen inference)
    """
    offline_planner = (not cfg.use_planner) or (not has_live_llm)
    return SimpleNamespace(
        seed=seed,
        task=task,
        frame_stack=1,
        offline_planner=offline_planner,
        soft_planner=False,
        logdir="log",
        policy="ppo",
        loaddir=None,
        loadmodel="acmodel",
        savedir=savedir,
        device=device,
        batch_size=cfg.batch_size,
        recurrent=False,
        gamma=0.99,
        lam=0.95,
        n_itr=cfg.total_iterations,
        traj_per_itr=cfg.episodes_per_iteration,
        num_eval=cfg.num_eval,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.save_interval,
    )


def _ollama_available(url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Quick reachability probe for a local Ollama server (used to decide the
    reflector backend: Ollama primary, offline backup)."""
    try:
        import requests
        return requests.get(url.rstrip("/") + "/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def build_game(cfg: ExperimentConfig,
               task: str = "SimpleDoorKey",
               device: str = "cpu",
               seed: int = 42,
               qwen_llm=None,
               reflector_backend: str = None,
               reflector_model: str = None,
               reflector_api_key: str = None) -> "Game":
    """
    Instantiate a Game object wired to the given ExperimentConfig.

    Parameters
    ----------
    cfg      : ExperimentConfig controlling which components are active
    task     : MiniGrid task name
    device   : 'cpu' or 'cuda'
    seed     : random seed
    qwen_llm : optional QwenLLM instance for online planning
    reflector_backend : 'ollama' (primary, online) | 'offline' (backup) for the
        episode reflector. If None, Ollama is used when reachable, otherwise it
        falls back to 'offline' (no network). An explicit 'ollama' that is not
        reachable also degrades to 'offline' so we never block on a dead endpoint.
    reflector_model   : optional model name override for the reflector
        (defaults to the reflector's own qwen2.5:7b for Ollama).
    reflector_api_key : unused (kept for signature compatibility).

    Returns
    -------
    Configured Game instance ready for run_phase().
    """
    # Import here to avoid circular imports at module level
    from Game import Game

    savedir  = cfg.checkpoint_path
    has_llm  = (qwen_llm is not None) and cfg.use_planner
    args     = _make_args(cfg, savedir, task=task, device=device, seed=seed,
                          has_live_llm=has_llm)

    # Log planning mode so it's explicit in output
    if not cfg.use_planner:
        print(f"[Runner] Phase '{cfg.name}': teacher DISABLED (ppo_only)")
    elif has_llm:
        print(f"[Runner] Phase '{cfg.name}': ONLINE planning — {type(qwen_llm).__name__}")
    else:
        print(f"[Runner] Phase '{cfg.name}': OFFLINE planning — pre-computed plans")

    # Game now accepts exp_config to gate components
    game = Game(args, training=True, exp_config=cfg)

    # ── Wire online LLM if provided ──────────────────────────────────────────
    if has_llm:
        try:
            game.teacher_policy.planner.set_llm(qwen_llm)
            print(f"[Runner] Online LLM wired into planner.")
        except Exception as e:
            print(f"[Runner] WARNING: could not wire LLM: {e}")

    # ── Apply PPO epoch override ──────────────────────────────────────────────
    game.student_policy.epochs = cfg.ppo_epochs

    # ── Apply learning rate override ──────────────────────────────────────────
    for pg in game.student_policy.optimizer.param_groups:
        pg["lr"] = cfg.learning_rate

    # ── Apply entropy coefficient override ───────────────────────────────────
    game.student_policy.entropy_coef = cfg.entropy_coef

    # ── Apply kickstarting overrides ─────────────────────────────────────────
    if not cfg.use_kickstarting:
        game.student_policy.iter_with_ks = 0
    else:
        game.student_policy.iter_with_ks              = cfg.iter_with_ks
        game.student_policy.ks_coef                   = cfg.kickstarting_coef_initial
        game.student_policy.ks_coef_minimum           = cfg.kickstarting_coef_minimum
        game.student_policy.ks_coef_descent           = cfg.kickstarting_coef_descent

    # ── Wire reflection system if needed ─────────────────────────────────────
    if cfg.use_reflection and cfg.use_episode_memory:
        try:
            from memory.reflection import QwenReflector
            from memory.memory_buffer import ReflectionMemory
            # Reflector backend: Ollama is the PRIMARY (online) reflector; OFFLINE
            # is the automatic backup. If Ollama isn't reachable (e.g. Kaggle / CPU
            # box with no server) we degrade to offline so we never block on a dead
            # endpoint. An explicit reflector_backend overrides the default but an
            # unreachable 'ollama' still falls back to offline.
            rb = reflector_backend if reflector_backend is not None else "ollama"
            if rb == "ollama" and not _ollama_available():
                print("[Runner] Ollama not reachable — reflector falling back to 'offline' (backup).")
                rb = "offline"
            # Reflector model: if not given, REUSE THE PLANNER'S model (which the
            # host has already pulled, e.g. Kaggle pulls qwen2.5:3b). The old hard
            # default of qwen2.5:7b caused EVERY reflector call to fail with
            # "model not found" when only 3b was pulled (generated=0, failed=N).
            rm = reflector_model
            if rm is None and qwen_llm is not None:
                rm = getattr(qwen_llm, "model", None)
            reflector = QwenReflector(backend=rb, model=rm)
            memory    = ReflectionMemory(maxlen=20, top_k=5)
            game.set_reflection_system(reflector, memory)
            print(f"[Runner] Reflection system wired (reflector backend={rb}, "
                  f"model={getattr(reflector, 'model', '?')}).")
        except Exception as e:
            print(f"[Runner] WARNING: could not wire reflection: {e}")

    # ── Record the active LLM modes (print + persistent per-run log line) ─────
    planner_mode = (
        "ONLINE (Ollama LLM)" if has_llm
        else ("OFFLINE (pre-computed plans)" if cfg.use_planner else "DISABLED (ppo_only)")
    )
    reflector_mode = "none"
    if cfg.use_reflection and cfg.use_episode_memory:
        reflector_mode = getattr(getattr(game, "_reflector", None), "backend", "none")
    _mode_msg = (f"[LLM-MODE] phase='{cfg.name}' | planner={planner_mode} | "
                 f"reflector={reflector_mode}")
    print(_mode_msg)
    try:
        _log_dir = getattr(getattr(game, "logger", None), "dir", None)
        if _log_dir:
            with open(os.path.join(_log_dir, "llm_mode.log"), "a", encoding="utf-8") as _f:
                _f.write(_mode_msg + "\n")
    except Exception:
        pass

    return game


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_episode_funnel(s, e, act_arr, obs_l, DIRECTION={0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}):
    """
    Calculate execution funnel metrics for a SINGLE episode.

    s, e: start and end indices for this episode
    act_arr: all actions array
    obs_l: all observations array

    Returns dict with funnel metrics for this episode only.
    """
    def _grid(o):
        return (o[0] if getattr(o, "ndim", 2) == 4 else o)

    def _carry(o):
        g = _grid(o)
        ap = np.argwhere(g[:, :, 3] != 4)
        return 1 if len(ap) == 0 else int(g[ap[0][0], ap[0][1], 0])

    def _door_visible(o):
        return bool(np.any(_grid(o)[:, :, 0] == 4))

    funnel = {
        "key_visible_steps": 0,
        "key_adjacent_steps": 0,
        "key_picked_steps": 0,
        "door_adjacent_steps": 0,
        "door_facing_steps": 0,
        "door_opened_steps": 0,
    }

    # Only loop through THIS episode's timesteps
    for t in range(s, e):
        o = obs_l[t]
        g = _grid(o)
        ap = np.argwhere(g[:, :, 3] != 4)
        if len(ap) == 0:
            continue
        ar, ac = int(ap[0][0]), int(ap[0][1])
        agent_dir = int(g[ar, ac, 3])

        carrying_t = _carry(o)
        carrying_key = carrying_t in (5, 6, 7)

        # Key on floor visible
        keys_on_floor = [(int(k[0]), int(k[1])) for k in np.argwhere(g[:, :, 0] == 5)
                        if not (int(k[0]) == ar and int(k[1]) == ac)]
        if keys_on_floor:
            funnel["key_visible_steps"] += 1
            if any(abs(kr - ar) + abs(kc - ac) == 1 for kr, kc in keys_on_floor):
                funnel["key_adjacent_steps"] += 1

        if carrying_key:
            funnel["key_picked_steps"] += 1

        # Door in front
        dx, dy = DIRECTION.get(agent_dir, (0, 0))
        fr, fc = ar + dx, ac + dy
        h, w = g.shape[:2]
        if 0 <= fr < h and 0 <= fc < w and int(g[fr, fc, 0]) == 4:
            funnel["door_adjacent_steps"] += 1
            funnel["door_facing_steps"] += 1

        # Door opened this step
        if t < e - 1 and int(act_arr[t]) == 5:
            def _door_state(obs):
                g_tmp = _grid(obs)
                doors = np.argwhere(g_tmp[:, :, 0] == 4)
                return {(int(d[0]), int(d[1]), int(g_tmp[d[0], d[1], 2])) for d in doors}

            bc = {(r, c) for r, c, s2 in _door_state(obs_l[t]) if s2 in (1, 2)}
            ao = {(r, c) for r, c, s2 in _door_state(obs_l[t + 1]) if s2 == 0}
            if bc & ao:
                funnel["door_opened_steps"] += 1

    return funnel


def _extract_metrics(game, buffer, n_traj, mean_losses, itr) -> dict:
    """Extract per-iteration metrics from live training state."""
    m = _zero_metrics()

    # Core performance
    ep_rets = list(buffer.ep_returns)
    ep_lens = list(buffer.ep_lens)
    m["success_rate"]    = sum(r > 0 for r in ep_rets) / max(n_traj, 1)
    m["average_reward"]  = float(np.mean(ep_rets)) if ep_rets else 0.0
    m["episode_length"]  = float(np.mean(ep_lens)) if ep_lens else 0.0

    # Loss breakdown — PPO.update_policy returns
    # [total_loss, entropy_loss, kickstarting_loss, policy_loss, value_loss]
    ml = list(mean_losses) if mean_losses is not None else []
    m["total_loss"]  = float(ml[0]) if len(ml) > 0 else 0.0
    m["entropy"]     = float(ml[1]) if len(ml) > 1 else 0.0   # entropy loss/value
    m["ks_loss"]     = float(ml[2]) if len(ml) > 2 else 0.0   # KS loss magnitude
    m["policy_loss"] = float(ml[3]) if len(ml) > 3 else 0.0
    m["value_loss"]  = float(ml[4]) if len(ml) > 4 else 0.0
    m["ppo_loss"]    = m["policy_loss"]   # alias (clipped surrogate policy loss)

    sp = getattr(game, "student_policy", None)
    m["entropy_coef"] = float(getattr(sp, "entropy_coef", 0.0))
    # KS is actually applied to the loss only while iter < iter_with_ks (0 in ppo_only).
    m["ks_applied"] = int(getattr(sp, "iter", 0) < getattr(sp, "iter_with_ks", 0))

    # Teacher agreement — only meaningful when a real planner/teacher is active.
    # In ppo_only the "teacher" is a uniform stub, so scoring against it yields a
    # chance-level (~1/n_actions) value that is NOT agreement → report None (N/A).
    use_planner = bool(getattr(getattr(game, "exp_config", None), "use_planner", True))
    if not use_planner:
        m["teacher_agreement"] = None
    else:
        try:
            tp_arr  = np.array(buffer.teacher_probs)
            act_arr = np.array(buffer.actions).astype(int)
            if tp_arr.ndim == 2 and tp_arr.shape[1] > 1:
                ta = np.argmax(tp_arr, axis=1)
                m["teacher_agreement"] = float(np.mean(ta == act_arr))
        except Exception:
            pass

    # ── Action-type agreement, interaction counts, failure breakdown, planner
    #    critical-state accuracy — ALL read-only from the buffer (no behavior change).
    try:
        tp_arr  = np.array(buffer.teacher_probs)
        act_arr = np.array(buffer.actions).astype(int)
        rew_arr = np.array(buffer.rewards).reshape(-1)
        obs_l   = buffer.obs
        traj    = list(buffer.traj_idx)
        if tp_arr.ndim == 2 and tp_arr.shape[1] > 1 and len(act_arr) == len(tp_arr):
            targ    = np.argmax(tp_arr, axis=1)
            nav_set = np.isin(targ, [0, 1, 2])

            def _agree(mask):
                n = int(mask.sum())
                return (float(np.mean(act_arr[mask] == targ[mask])) if n > 0 else None)

            if use_planner:   # teacher-relative metrics only meaningful with a real teacher
                m["navigation_agreement"] = _agree(nav_set)
                m["pickup_agreement"]     = _agree(targ == 3)
                m["toggle_agreement"]     = _agree(targ == 5)
                m["teacher_requested_navigation"] = int(nav_set.sum())
                m["teacher_requested_pickup"]     = int((targ == 3).sum())
                m["teacher_requested_toggle"]     = int((targ == 5).sum())
            else:
                m["navigation_agreement"] = m["pickup_agreement"] = m["toggle_agreement"] = None

            # Student-executed counts / attempts (PPO acts — always meaningful).
            m["student_executed_navigation"] = int(np.isin(act_arr, [0, 1, 2]).sum())
            m["student_executed_pickup"]     = int((act_arr == 3).sum())
            m["student_executed_toggle"]     = int((act_arr == 5).sum())
            m["pickup_attempts"]             = m["student_executed_pickup"]
            m["toggle_attempts"]             = m["student_executed_toggle"]
            # read-only grounded obs decoders
            def _grid(o):  return (o[0] if getattr(o, "ndim", 2) == 4 else o)
            def _carry(o):
                g = _grid(o); ap = np.argwhere(g[:, :, 3] != 4)
                return 1 if len(ap) == 0 else int(g[ap[0][0], ap[0][1], 0])
            def _key_on_floor(o):
                g = _grid(o); ap = np.argwhere(g[:, :, 3] != 4); keys = np.argwhere(g[:, :, 0] == 5)
                if len(ap) == 0: return len(keys) > 0
                ar, ac = int(ap[0][0]), int(ap[0][1])
                return any(not (int(k[0]) == ar and int(k[1]) == ac) for k in keys)
            def _door_visible(o): return bool(np.any(_grid(o)[:, :, 0] == 4))

            def _door_state(o):
                """Return set of (row, col, state) for all door cells. state: 0=open 1=closed 2=locked."""
                g = _grid(o)
                doors = np.argwhere(g[:, :, 0] == 4)
                return {(int(d[0]), int(d[1]), int(g[d[0], d[1], 2])) for d in doors}

            # True door-open: closed/locked → open transition on the step the agent toggled.
            # Breaks the failure chain into 3 distinct cases:
            #   can't reach door  /  reaches door but can't open it  /  opens door but misses goal
            succ_door_open = 0
            for t in range(len(act_arr) - 1):
                if int(act_arr[t]) != 5:   # only count on actual toggle action
                    continue
                before = _door_state(obs_l[t])
                after  = _door_state(obs_l[t + 1])
                # Any door that was closed(1) or locked(2) before and is now open(0)
                before_closed = {(r, c) for r, c, s in before if s in (1, 2)}
                after_open    = {(r, c) for r, c, s in after  if s == 0}
                if before_closed & after_open:
                    succ_door_open += 1
            m["successful_door_opens"] = succ_door_open

            # Planner critical-state primitive accuracy (teacher only)
            if use_planner:
                pk_pick = pk_nav = dr_tog = dr_nav = 0
                for t in range(len(act_arr)):
                    carrying = _carry(obs_l[t]) in (5, 6, 7)
                    if _key_on_floor(obs_l[t]) and not carrying:
                        if   targ[t] == 3:            pk_pick += 1
                        elif targ[t] in (0, 1, 2):    pk_nav  += 1
                    if _door_visible(obs_l[t]) and carrying:
                        if   targ[t] == 5:            dr_tog  += 1
                        elif targ[t] in (0, 1, 2):    dr_nav  += 1
                m["planner_requested_pickup_at_key"]      = pk_pick
                m["planner_requested_navigation_at_key"]  = pk_nav
                m["planner_requested_toggle_at_door"]     = dr_tog
                m["planner_requested_navigation_at_door"] = dr_nav

            # Successful pickups: carrying None→key transition right after a student pickup
            succ_pick = 0
            for s, e in zip(traj[:-1], traj[1:]):
                for t in range(s, e - 1):
                    if int(act_arr[t]) == 3 and _carry(obs_l[t]) not in (5, 6, 7) \
                       and _carry(obs_l[t + 1]) in (5, 6, 7):
                        succ_pick += 1
            m["successful_pickups"] = succ_pick

            # Interaction success rates
            m["pickup_success_rate"]    = (succ_pick / m["pickup_attempts"]) if m["pickup_attempts"] else 0.0
            m["toggle_success_rate"]    = (m["successful_door_opens"] / m["toggle_attempts"]) if m["toggle_attempts"] else 0.0
            m["door_open_success_rate"] = m["toggle_success_rate"]

            # Per-episode failure breakdown → tally per iteration (mirrors _diagnose)
            # Also tracks the 3-stage door chain: reached_door / opened_door / reached_goal
            fb = {"key_never_found": 0, "key_visible_but_navigation_failed": 0,
                  "key_acquired_door_interaction_failed": 0, "wrong_key_for_door": 0,
                  "timeout_failure": 0}
            ep_reached_door  = 0   # key picked AND door visible at some point
            ep_opened_door   = 0   # door state actually changed closed→open
            ep_reached_goal  = 0   # reward > 0 (goal cell reached)
            for s, e in zip(traj[:-1], traj[1:]):
                ep_reward = float(rew_arr[s:e].sum())
                if ep_reward > 0:
                    ep_reached_goal += 1

                key_det = key_pick = door_reached = door_opened = False
                for t in range(s, e):
                    if _key_on_floor(obs_l[t]):                    key_det      = True
                    if _carry(obs_l[t]) in (5, 6, 7):              key_pick     = True
                    if key_pick and _door_visible(obs_l[t]):       door_reached = True
                    if t < e - 1 and int(act_arr[t]) == 5:
                        before_c = {(r, c) for r, c, s2 in _door_state(obs_l[t])     if s2 in (1, 2)}
                        after_o  = {(r, c) for r, c, s2 in _door_state(obs_l[t + 1]) if s2 == 0}
                        if before_c & after_o:
                            door_opened = True

                if door_reached: ep_reached_door += 1
                if door_opened:  ep_opened_door  += 1

                if ep_reward > 0:
                    continue   # success — not a failure bucket
                if key_pick:    fb["key_acquired_door_interaction_failed"] += 1
                elif key_det:   fb["key_visible_but_navigation_failed"]    += 1
                else:           fb["key_never_found"]                      += 1

            for _k, _v in fb.items():
                m[_k] = _v
            # 3-stage door chain (absolute episode counts this iteration)
            m["ep_reached_door_with_key"] = ep_reached_door
            m["ep_opened_door"]           = ep_opened_door
            m["ep_reached_goal"]          = ep_reached_goal

            # ✓ Funnel metrics are now calculated PER-EPISODE (see _calculate_episode_funnel)
            # They are populated in the per-episode loop when creating trajectory/episode records
    except Exception:
        pass

    # Kickstarting coef
    m["kickstarting_coef"] = float(getattr(game.student_policy, "ks_coef", 0.0))

    # Cumulative counter TOTALS (planner / intervention / reflection / memory).
    # Phase-cumulative; run_phase derives per-iteration *_delta from these.
    snap = _phase_counter_snapshot(game)
    for base in _COUNTER_KEYS:
        m[f"{base}_total"] = snap[base]
    # Legacy cumulative aliases (back-compat with existing consumers / plots).
    for legacy, base in _LEGACY_ALIASES.items():
        m[legacy] = snap[base]

    # Episode-memory writes (distinct from retrievals)
    ref_mem = getattr(game, "_reflection_memory", None)
    if ref_mem is not None:
        m["episode_memory_writes"] = ref_mem.stats.get("total_added", 0)

    # ── Reflection-effectiveness + planner cache-quality (real, populated only
    #    when the relevant subsystem is active) ────────────────────────────────
    planner_obj = getattr(getattr(game, "teacher_policy", None), "planner", None)
    if planner_obj is not None and hasattr(planner_obj, "planner_stats"):
        ps2 = planner_obj.planner_stats
        m["prompt_injections_count"]     = int(ps2.get("prompt_injections", 0))
        m["cache_hit_rate"]              = float(ps2.get("cache_hit_rate", 0.0))
        m["unique_symbolic_states_seen"] = len(getattr(planner_obj, "plans_dict", {}) or {})
    if ref_mem is not None:
        m["retrieved_reflections_count"] = int(ref_mem.stats.get("total_reflections_retrieved", 0))
    m["overall_teacher_agreement"] = m["teacher_agreement"]   # explicit alias

    # State memory metrics
    sm = getattr(game, "_state_memory", None)
    if sm is not None:
        sms = sm.stats
        m["state_memory_writes"]   = sms.get("total_added", 0)
        m["state_memory_reads"]    = sms.get("total_retrieved", 0)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# run_phase — main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(
    cfg: ExperimentConfig,
    game=None,
    task: str = "SimpleDoorKey",
    device: str = "cpu",
    seed: int = 42,
    qwen_llm=None,
    reflector_backend: str = None,
    reflector_model: str = None,
    reflector_api_key: str = None,
    verbose: bool = True,
    plot_every: int = 50,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Train one experiment phase according to cfg.

    Parameters
    ----------
    cfg              : ExperimentConfig for this phase
    game             : pre-built Game instance (built via build_game if None)
    task / device / seed / qwen_llm : passed to build_game when game=None
    verbose          : print progress every 10 iterations
    plot_every       : how often to call progress_callback(history, itr)
    progress_callback: optional fn(history, itr) for live plotting

    Returns
    -------
    history : dict with keys:
        itr, success_rate, average_reward, episode_length, entropy,
        teacher_agreement, kickstarting_coef, planner_calls, reflections_generated,
        interventions, episode_memory_writes, state_memory_writes, total_steps,
        eval_itr, eval_success, eval_reward, eval_ep_len
    """
    import torch

    if game is None:
        game = build_game(cfg, task=task, device=device, seed=seed,
                          qwen_llm=qwen_llm,
                          reflector_backend=reflector_backend,
                          reflector_model=reflector_model,
                          reflector_api_key=reflector_api_key)

    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    os.makedirs(cfg.results_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(cfg.summary())
    print(f"{'='*60}\n")

    # History tracking
    history = dict(
        itr=[], success_rate=[], average_reward=[], episode_length=[],
        entropy=[], teacher_agreement=[], kickstarting_coef=[],
        total_loss=[], policy_loss=[], value_loss=[], ks_loss=[],
        entropy_coef=[], ks_applied=[],
        navigation_agreement=[], pickup_agreement=[], toggle_agreement=[],
        teacher_requested_pickup=[], student_executed_pickup=[], successful_pickups=[],
        teacher_requested_toggle=[], student_executed_toggle=[], successful_door_opens=[],
        planner_calls=[], cache_hits=[], cache_misses=[], online_success=[],
        cache_invalidations=[], memory_retrievals=[],
        reflections_generated=[], interventions=[],
        episode_memory_writes=[], state_memory_writes=[], total_steps=[],
        eval_itr=[], eval_success=[], eval_reward=[], eval_ep_len=[],
    )
    # Per-iteration (*_delta) and cumulative (*_total) counter histories.
    for _k in _COUNTER_KEYS:
        history[f"{_k}_delta"] = []
        history[f"{_k}_total"] = []
    # New instrumentation columns + join key.
    for _k in (_NEW_LOG_KEYS + ["iteration_id", "ppo_loss", "overall_teacher_agreement"]):
        history[_k] = []

    start = time.time()
    metrics_log = []   # per-iteration metric dicts for save_results

    for itr in range(cfg.total_iterations):
        # propagate current iteration to Game so collect() can annotate step events
        game._current_itr = itr

        # check stop signal from visualizer
        _stop = getattr(cfg, "_viz_stop", None)
        if _stop is not None and _stop.is_set():
            break

        # Snapshot cumulative counters at iteration START → per-iteration deltas.
        _base_counts = _phase_counter_snapshot(game)

        # ── Collect trajectories ──────────────────────────────────────────────
        game.buffer.clear()
        n_traj = game.traj_per_itr
        for _ in range(n_traj):
            game.collect()
        while len(game.buffer) < game.batch_size * 2:
            game.collect()
            n_traj += 1
        game.total_steps += len(game.buffer)

        # ── Update policy ─────────────────────────────────────────────────────
        recent_sr = sum(r > 0 for r in game.buffer.ep_returns) / max(n_traj, 1)
        mean_losses = game.student_policy.update_policy(game.buffer, recent_sr)

        # ── Extract metrics ───────────────────────────────────────────────────
        m = _extract_metrics(game, game.buffer, n_traj, mean_losses, itr)
        m["itr"]          = itr
        m["iteration_id"] = itr        # explicit join key (== itr)
        m["total_steps"]  = game.total_steps

        # Per-iteration deltas = (cumulative total now) − (snapshot at iter start).
        # _extract_metrics already populated *_total from the end-of-iter snapshot.
        for base in _COUNTER_KEYS:
            m[f"{base}_delta"] = max(0, m[f"{base}_total"] - _base_counts[base])
        _check_metric_invariants(m, itr)

        metrics_log.append(m)

        _hist_keys = ["itr", "success_rate", "average_reward", "episode_length",
                      "entropy", "teacher_agreement", "kickstarting_coef",
                      "total_loss", "policy_loss", "value_loss", "ks_loss",
                      "entropy_coef", "ks_applied",
                      "navigation_agreement", "pickup_agreement", "toggle_agreement",
                      "teacher_requested_pickup", "student_executed_pickup", "successful_pickups",
                      "teacher_requested_toggle", "student_executed_toggle", "successful_door_opens",
                      "planner_calls", "cache_hits", "cache_misses", "online_success",
                      "cache_invalidations", "memory_retrievals",
                      "reflections_generated", "interventions",
                      "episode_memory_writes", "state_memory_writes", "total_steps"]
        _hist_keys += [f"{k}_delta" for k in _COUNTER_KEYS]
        _hist_keys += [f"{k}_total" for k in _COUNTER_KEYS]
        _hist_keys += _NEW_LOG_KEYS + ["iteration_id", "ppo_loss", "overall_teacher_agreement"]
        for key in _hist_keys:
            history[key].append(m.get(key, 0))

        # ── Evaluate ──────────────────────────────────────────────────────────
        if itr % cfg.eval_interval == 0 and itr > 0:
            evals = [game.evaluate(record_frames=False)
                     for _ in range(cfg.num_eval)]
            e_succ = float(np.mean([r[2] for r in evals]))
            e_ret  = float(np.mean([r[0] for r in evals]))
            e_len  = float(np.mean([r[1] for r in evals]))
            history["eval_itr"].append(itr)
            history["eval_success"].append(e_succ)
            history["eval_reward"].append(e_ret)
            history["eval_ep_len"].append(e_len)
            if verbose:
                print(f"  [EVAL itr={itr}] "
                      f"success={e_succ:.2f} reward={e_ret:.3f} len={e_len:.1f}")

        # ── Save checkpoint ───────────────────────────────────────────────────
        if itr % cfg.save_interval == 0 and itr > 0:
            game.student_policy.save(str(itr))

        # ── Console log (per-iteration DELTAS by default; totals on a 2nd line) ──
        if verbose and (itr % 10 == 0 or itr == cfg.total_iterations - 1):
            elapsed = time.time() - start
            status = "✅" if m["success_rate"] > 0 else "❌"
            _ta = m.get("teacher_agreement")
            _ta_str = f"{_ta:.2f}" if isinstance(_ta, (int, float)) and _ta == _ta else "n/a"
            _ks_flag = "" if m.get("ks_applied", 0) else "(off)"
            print(
                f"[{cfg.name}] [{itr:4d}/{cfg.total_iterations}] "
                f"succ={m['success_rate']:.2f} ret={m['average_reward']:6.3f} "
                f"len={m['episode_length']:5.1f} ks_coef={m['kickstarting_coef']:.3f}{_ks_flag} "
                f"ksL={m['ks_loss']:.3f} ent={m['entropy']:.3f} ent_c={m['entropy_coef']:.3f} "
                f"ta={_ta_str} iv={m['interventions_delta']} "
                f"online={m['online_success_delta']} {status} {elapsed:.0f}s"
            )
            print(
                f"[{cfg.name}]   totals: consult={m['planner_consultations_total']} "
                f"hits={m['cache_hits_total']} miss={m['cache_misses_total']} "
                f"online={m['online_success_total']} iv={m['interventions_total']} "
                f"retr={m['memory_retrievals_total']}"
            )
            def _ag(x): return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "n/a"
            print(
                f"[{cfg.name}]   agree: nav={_ag(m['navigation_agreement'])} "
                f"pick={_ag(m['pickup_agreement'])} tog={_ag(m['toggle_agreement'])} | "
                f"pickup req/exec/ok={m['teacher_requested_pickup']}/"
                f"{m['student_executed_pickup']}/{m['successful_pickups']} | "
                f"toggle req/exec/open={m['teacher_requested_toggle']}/"
                f"{m['student_executed_toggle']}/{m['successful_door_opens']}"
            )
            print(
                f"[{cfg.name}]   rates: pickup_sr={m['pickup_success_rate']:.2f} "
                f"door_sr={m['door_open_success_rate']:.2f} | "
                f"states_seen={m['unique_symbolic_states_seen']} cache_hit={m['cache_hit_rate']:.4f} | "
                f"inject={m['prompt_injections_count']} retr_refl={m['retrieved_reflections_count']} | "
                f"fail[never={m['key_never_found']} nav={m['key_visible_but_navigation_failed']} "
                f"door={m['key_acquired_door_interaction_failed']}]"
            )

        # ── Visualizer iteration-end event ───────────────────────────────────
        if _VIZ_OK and _VIZ._ready:
            _VIZ.emit("iteration_end", {
                "iteration":    itr,
                "total_itr":    cfg.total_iterations,
                "success_rate": m["success_rate"],
                "avg_reward":   m["average_reward"],
                "ep_len":       m["episode_length"],
                "ks_coef":      m["kickstarting_coef"],
                "entropy":      m["entropy"],
                "interventions": m["interventions"],
                "planner_calls": m["planner_calls"],
                "total_steps":  game.total_steps,
            })

        # ── Progress callback (live plotting) ─────────────────────────────────
        if progress_callback is not None and itr % plot_every == 0 and itr > 1:
            try:
                progress_callback(history, itr)
            except Exception:
                pass

    # ── Final save ────────────────────────────────────────────────────────────
    game.student_policy.save()
    save_results(cfg, history, metrics_log)
    save_llm_stats(cfg, game)   # per-phase planner/cache/reflection counters

    # Mirror the per-EPISODE trajectory metrics.csv into results/<phase>/episodes.csv
    # (so it ships in the download zip and is joinable via episode_id/iteration_id).
    try:
        import shutil
        _tl = getattr(game, "_traj_logger", None)
        _src = os.path.join(_tl.log_root, "metrics.csv") if _tl is not None else None
        if _src and os.path.exists(_src):
            _dst = os.path.join(cfg.results_path, "episodes.csv")
            shutil.copyfile(_src, _dst)
            print(f"[Runner] Per-episode metrics mirrored → {_dst}")
    except Exception as _e:
        print(f"[Runner] (episodes.csv mirror skipped: {_e})")

    final_sr = float(np.mean(history["success_rate"][-20:])) if history["success_rate"] else 0.0
    print(f"\n[{cfg.name}] Training complete. "
          f"steps={game.total_steps:,} | "
          f"final_success_rate(last 20)={final_sr:.3f}")

    history["_game"] = game   # so caller can reuse game for evaluate_phase
    return history


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_phase
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_phase(
    game,
    cfg: ExperimentConfig,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation episodes (epsilon=0) and return metrics dict.

    Parameters
    ----------
    game         : trained Game instance
    cfg          : ExperimentConfig (used for metadata)
    n_episodes   : number of evaluation episodes
    deterministic: use argmax action (no exploration noise)

    Returns
    -------
    metrics dict with success_rate, avg_reward, avg_episode_length,
    teacher_agreement (teacher upper bound), phase, name.
    """
    import numpy as np
    results = [game.evaluate(record_frames=False, deterministic=deterministic)
               for _ in range(n_episodes)]
    ret  = [r[0] for r in results]
    lens = [r[1] for r in results]
    succ = [r[2] for r in results]

    # Teacher upper bound
    teacher_results = [
        game.evaluate(teacher_policy=True, record_frames=False, deterministic=False)
        for _ in range(min(n_episodes, 10))
    ]
    t_succ = [r[2] for r in teacher_results]
    t_ret  = [r[0] for r in teacher_results]

    return dict(
        phase=cfg.phase,
        name=cfg.name,
        n_episodes=n_episodes,
        success_rate=float(np.mean(succ)),
        avg_reward=float(np.mean(ret)),
        avg_episode_length=float(np.mean(lens)),
        std_reward=float(np.std(ret)),
        teacher_success_rate=float(np.mean(t_succ)),
        teacher_avg_reward=float(np.mean(t_ret)),
        student_teacher_gap=float(np.mean(t_succ)) - float(np.mean(succ)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# save_results / aggregate_results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    cfg: ExperimentConfig,
    history: dict,
    metrics_log: Optional[list] = None,
) -> None:
    """
    Persist training history and metrics to disk.

    Writes:
      results/<name>/history.json       — full per-iteration history
      results/<name>/metrics.csv        — per-iteration CSV
      results/<name>/eval_metrics.json  — evaluation episodes summary
    """
    os.makedirs(cfg.results_path, exist_ok=True)

    # Full history JSON
    hist_clean = {k: v for k, v in history.items() if k != "_game"}
    with open(os.path.join(cfg.results_path, "history.json"), "w") as f:
        json.dump(hist_clean, f, indent=2)

    # Per-iteration CSV
    if metrics_log:
        csv_path = os.path.join(cfg.results_path, "metrics.csv")
        fieldnames = list(metrics_log[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_log:
                flat = {k: (json.dumps(v) if isinstance(v, dict) else v)
                        for k, v in row.items()}
                writer.writerow(flat)

    print(f"[Runner] Results saved → {cfg.results_path}/")


def save_llm_stats(cfg: ExperimentConfig, game) -> None:
    """
    Persist the FINAL cumulative planner / reflector / memory counters for a phase
    to results/<name>/llm_stats.json.

    This is the durable per-phase record for the ablation experiments:
      planner_calls, cache_hits, cache_misses, online_success (real LLM calls),
      cache_invalidations, reflections generated/validated, memory retrievals.
    """
    os.makedirs(cfg.results_path, exist_ok=True)
    out = {"phase": cfg.phase, "name": cfg.name}

    planner = getattr(getattr(game, "teacher_policy", None), "planner", None)
    if planner is not None and hasattr(planner, "planner_stats"):
        ps = dict(planner.planner_stats)
        ps["cache_misses"] = max(0, ps.get("planner_calls", 0) - ps.get("cache_hits", 0))
        out["planner"] = ps

    reflector = getattr(game, "_reflector", None)
    if reflector is not None:
        out["reflector"] = {"backend": getattr(reflector, "backend", "?"), **reflector.stats}

    ref_mem = getattr(game, "_reflection_memory", None)
    if ref_mem is not None:
        out["memory"] = ref_mem.stats

    path = os.path.join(cfg.results_path, "llm_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Console echo so the counts show in the Kaggle cell output too.
    if "planner" in out:
        p = out["planner"]
        print(f"[Runner] LLM stats ({cfg.name}): planner_calls={p.get('planner_calls',0)} "
              f"cache_hits={p.get('cache_hits',0)} cache_misses={p.get('cache_misses',0)} "
              f"online_success={p.get('online_success',0)} "
              f"prompt_injections={p.get('prompt_injections',0)} "
              f"invalidations={p.get('cache_invalidations',0)}")
    if "memory" in out:
        print(f"[Runner]   reflections_added={out['memory'].get('total_added',0)} "
              f"retrievals={out['memory'].get('total_retrievals',0)}")
    print(f"[Runner] LLM stats saved → {path}")


def save_eval_results(cfg: ExperimentConfig, eval_metrics: dict) -> None:
    """Save evaluate_phase() output to results/<name>/eval_metrics.json."""
    os.makedirs(cfg.results_path, exist_ok=True)
    path = os.path.join(cfg.results_path, "eval_metrics.json")
    with open(path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"[Runner] Eval metrics saved → {path}")


def aggregate_results(results_dir: str = "results") -> Optional[str]:
    """
    Scan results/<phase>/eval_metrics.json for each phase and produce
    results/summary.csv.

    Returns the path to summary.csv, or None if no phase results found.
    """
    import glob

    rows = []
    for phase_name in PHASE_ORDER:
        path = os.path.join(results_dir, phase_name, "eval_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append(m)

    if not rows:
        print("[Runner] No phase results found. Run at least one phase first.")
        return None

    # Statistical comparisons between consecutive phases
    for i in range(1, len(rows)):
        prev, curr = rows[i-1], rows[i]
        abs_imp = curr["success_rate"] - prev["success_rate"]
        pct_imp = (abs_imp / max(prev["success_rate"], 1e-6)) * 100
        curr["vs_prev_absolute_improvement"] = round(abs_imp, 4)
        curr["vs_prev_pct_improvement"]       = round(pct_imp, 2)

    # Write summary CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    os.makedirs(results_dir, exist_ok=True)
    fieldnames = [
        "phase", "name", "success_rate", "avg_reward", "avg_episode_length",
        "std_reward", "teacher_success_rate", "student_teacher_gap",
        "vs_prev_absolute_improvement", "vs_prev_pct_improvement",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row.setdefault("vs_prev_absolute_improvement", "—")
            row.setdefault("vs_prev_pct_improvement",       "—")
            writer.writerow(row)

    print(f"[Runner] Summary written → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# run_research_pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_research_pipeline(
    task:          str   = "SimpleDoorKey",
    device:        str   = "cpu",
    seed:          int   = 42,
    qwen_llm             = None,
    phase_configs: dict  = None,
    phases:        list  = None,
    verbose:       bool  = True,
    eval_episodes: int   = 20,
    progress_callback    = None,
    reflector_backend: str = None,
    reflector_model:   str = None,
    reflector_api_key: str = None,
) -> Dict[str, Any]:
    """
    Automatically execute all five ablation phases in order.

    Parameters
    ----------
    task          : MiniGrid task name
    device        : 'cpu' or 'cuda'
    seed          : reproducibility seed
    qwen_llm      : QwenLLM instance for online planning (phases 2-5)
    phase_configs : per-phase override dicts, e.g.
                    {"ppo_only": {"total_iterations": 500}}
    phases        : list of phase names to run (default: all 5)
    verbose       : print training progress
    eval_episodes : episodes for post-phase evaluation
    progress_callback : optional fn(history, itr) for live plotting
    reflector_backend / reflector_model / reflector_api_key :
        episode-reflector config. Default: Ollama when reachable, else offline
        backup (reflector_api_key is unused — DashScope is not used here).

    Returns
    -------
    all_results : dict mapping phase_name → {"history": ..., "eval": ...}
    """
    if phases is None:
        phases = list(PHASE_ORDER)

    all_results = {}

    print(f"\n{'#'*60}")
    print(f"# LLM4Teach Research Pipeline — {len(phases)} phases")
    print(f"# Task: {task} | Device: {device} | Seed: {seed}")
    print(f"{'#'*60}\n")

    for phase_name in phases:
        cfg = build_experiment_config(
            phase_name,
            phase_configs=phase_configs,
        )
        print(f"\n{'━'*60}")
        print(f"  PHASE {cfg.phase}: {phase_name.upper()}")
        print(f"{'━'*60}")

        # Only pass qwen_llm to phases that actually use the planner
        # (None → offline pre-computed plans; non-None → online inference)
        phase_qwen = qwen_llm if cfg.use_planner else None

        history = run_phase(
            cfg,
            task=task,
            device=device,
            seed=seed,
            qwen_llm=phase_qwen,
            reflector_backend=reflector_backend,
            reflector_model=reflector_model,
            reflector_api_key=reflector_api_key,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        game = history.pop("_game", None)

        eval_metrics = {}
        if game is not None:
            eval_metrics = evaluate_phase(game, cfg, n_episodes=eval_episodes)
            save_eval_results(cfg, eval_metrics)
            print(f"\n  [{phase_name}] Eval: "
                  f"success={eval_metrics['success_rate']:.2f} "
                  f"reward={eval_metrics['avg_reward']:.3f}")

        all_results[phase_name] = {"history": history, "eval": eval_metrics}

    # Aggregate into summary CSV
    summary_path = aggregate_results()
    all_results["_summary_path"] = summary_path

    print(f"\n{'#'*60}")
    print(f"# Pipeline complete.")
    if summary_path:
        print(f"# Summary: {summary_path}")
    print(f"{'#'*60}\n")

    return all_results
