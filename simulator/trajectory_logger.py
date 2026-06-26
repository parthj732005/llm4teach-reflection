#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
simulator/trajectory_logger.py
───────────────────────────────
Consolidated logging system for LLM-guided PPO + Reflexion MiniGrid.

Creates ONLY 4 files under log_root:
    training.log       — human-readable episode summaries + events
    trajectories.jsonl — one JSON per step, append-only, streamable
    prompts.log        — all LLM prompts + responses, chronological
    metrics.csv        — per-episode numbers for plotting
    debug.log          — severe/special events only (optional, stays small)

Design principles:
    - append-only, never rewritten
    - zero extra LLM calls
    - CPU-lightweight
    - grep-friendly
    - easy to parse later with pandas / jq
"""

import os
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    episode:          int   = 0
    step:             int   = 0
    obs:              str   = ""
    holding:          str   = "nothing"
    plan:             str   = ""
    subgoal:          str   = ""
    skill:            str   = ""
    action:           str   = ""
    action_id:        int   = -1
    teacher_action:   int   = -1   # teacher argmax this step (for per-episode agreement)
    reward:           float = 0.0
    progress:         bool  = False
    dist_to_target:   float = -1.0
    intervention:     bool  = False
    reflection:       bool  = False
    oscillation:      bool  = False
    ppo_entropy:      float = -1.0
    ppo_value:        float = -1.0
    agent_pos:        tuple = (0, 0)


@dataclass
class EpisodeRecord:
    episode:             int   = 0
    iteration_id:        int   = -1   # iteration this episode belongs to (join key)
    env_name:            str   = ""
    total_reward:        float = 0.0
    total_steps:         int   = 0
    success:             bool  = False
    key_detected:        bool  = False
    key_picked:          bool  = False
    door_detected:       bool  = False
    door_opened:         bool  = False
    lava_collision:      bool  = False
    wrong_key_for_door:  bool  = False
    timeout:             bool  = False
    goal_reached:        bool  = False
    interventions:       int   = 0
    reflections:         int   = 0
    replans:             int   = 0
    cache_invalidations: int   = 0
    llm_calls:           int   = 0
    oscillation_events:  int   = 0
    stuck_events:        int   = 0
    plans_generated:     int   = 0
    cache_hits:          int   = 0
    avg_ppo_entropy:     float = -1.0
    failure_type:        str   = ""
    termination_reason:  str   = ""
    # Per-episode action-specific teacher agreement (accumulated in log_step)
    nav_requested:       int   = 0
    nav_matched:         int   = 0
    pickup_requested:    int   = 0
    pickup_matched:      int   = 0
    toggle_requested:    int   = 0
    toggle_matched:      int   = 0
    pickup_attempts:     int   = 0   # student action == 3
    toggle_attempts:     int   = 0   # student action == 5
    # Per-episode execution funnel (step counts)
    key_visible_steps:   int   = 0
    key_adjacent_steps:  int   = 0
    key_picked_steps:    int   = 0
    door_adjacent_steps: int   = 0
    door_facing_steps:   int   = 0
    door_opened_steps:   int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based failure diagnosis — zero LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose(ep: EpisodeRecord) -> tuple:
    """
    Returns (failure_type, reason_str) from counters alone.
    Pure rule-based. No LLM.
    """
    if ep.success:
        return "none", "episode_succeeded"
    if ep.timeout and ep.key_picked and not ep.door_opened:
        return "execution_failure", "key_acquired_door_interaction_failed_before_timeout"
    if ep.timeout and ep.key_detected and not ep.key_picked:
        return "execution_failure", "key_visible_but_navigation_failed_to_reach_it"
    if ep.timeout and not ep.key_detected:
        return "exploration_failure", "key_never_found_exploration_insufficient"
    if ep.cache_invalidations > 0 and not ep.success:
        return "planner_failure", "symbolic_instability_cache_invalidated"
    if ep.oscillation_events > 2:
        return "execution_failure", "persistent_navigation_oscillation"
    if ep.wrong_key_for_door and not ep.success:
        return "execution_failure", "wrong_key_colour_used_for_locked_door"
    if ep.lava_collision:
        return "execution_failure", "lava_collision_navigation_path_issue"
    if ep.timeout:
        return "timeout", "episode_exceeded_step_limit"
    return "unknown", "failure_cause_unclear_from_counters"


# ─────────────────────────────────────────────────────────────────────────────
# CSV columns
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "episode_id", "iteration_id", "episode", "reward", "success", "steps",
    "llm_calls", "interventions", "reflections",
    "cache_invalidations", "oscillation_events", "stuck_events",
    "plans_generated", "cache_hits",
    "key_detected", "key_picked", "door_detected", "door_opened", "goal_reached",
    "lava_collision", "wrong_key_for_door", "timeout",
    # per-episode action-specific teacher agreement + interaction execution
    "navigation_agreement", "pickup_agreement", "toggle_agreement",
    "teacher_requested_navigation", "teacher_requested_pickup", "teacher_requested_toggle",
    "pickup_attempts", "toggle_attempts", "successful_pickups", "successful_door_opens",
    "avg_ppo_entropy", "failure_type", "termination_reason",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main logger
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryLogger:
    """
    Writes to exactly 4 append-only files:
        training.log       — human readable
        trajectories.jsonl — one JSON line per step
        prompts.log        — all LLM I/O
        metrics.csv        — per-episode numbers
        debug.log          — severe events only
    """

    def __init__(self, log_root: str, enabled: bool = True):
        self.log_root = log_root
        self.enabled  = enabled
        if not enabled:
            return

        os.makedirs(log_root, exist_ok=True)

        # Open all files in append mode
        self._train_f  = open(os.path.join(log_root, "training.log"),      "a", encoding="utf-8", buffering=1)
        self._traj_f   = open(os.path.join(log_root, "trajectories.jsonl"),"a", encoding="utf-8", buffering=1)
        self._prompt_f = open(os.path.join(log_root, "prompts.log"),       "a", encoding="utf-8", buffering=1)
        self._debug_f  = open(os.path.join(log_root, "debug.log"),         "a", encoding="utf-8", buffering=1)

        # CSV — write header only if file is new/empty
        csv_path = os.path.join(log_root, "metrics.csv")
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self._csv_raw = open(csv_path, "a", newline="", encoding="utf-8")
        self._csv     = csv.DictWriter(self._csv_raw, fieldnames=_CSV_FIELDS)
        if write_header:
            self._csv.writeheader()
            self._csv_raw.flush()

        # Per-episode in-memory state
        self._ep: Optional[EpisodeRecord] = None
        self._ep_ppo_entropies: List[float] = []

    # ── Episode lifecycle ──────────────────────────────────────────────────────

    def begin_episode(self, episode_id: int, env_name: str = "") -> Optional[EpisodeRecord]:
        """Start a new episode record and RETURN it.

        Callers should use the returned object as the single source of truth for
        the episode (log_step() mutates this same object). Returning it lets the
        caller set its own fields on the SAME record instead of a separate copy —
        this is what eliminates the prior two-record bug.
        """
        if not self.enabled:
            return None
        self._ep = EpisodeRecord(episode=episode_id, env_name=env_name)
        self._ep_ppo_entropies = []
        return self._ep

    def end_episode(self, ep: EpisodeRecord) -> None:
        if not self.enabled or self._ep is None:
            return

        # Fill average PPO entropy
        if self._ep_ppo_entropies:
            ep.avg_ppo_entropy = round(
                sum(self._ep_ppo_entropies) / len(self._ep_ppo_entropies), 4
            )

        # Rule-based diagnosis
        ep.failure_type, ep.termination_reason = _diagnose(ep)

        # ── Write to training.log ─────────────────────────────────────────────
        lines = [
            f"\n[Episode {ep.episode}]",
            f"  Env          : {ep.env_name}",
            f"  Reward       : {ep.total_reward:.4f}",
            f"  Success      : {ep.success}",
            f"  Steps        : {ep.total_steps}",
            f"  Termination  : {ep.termination_reason}",
            f"",
            f"  Key Detected : {ep.key_detected}",
            f"  Key Picked   : {ep.key_picked}",
            f"  Door Detected: {ep.door_detected}",
            f"  Door Opened  : {ep.door_opened}",
            f"  Lava Hit     : {ep.lava_collision}",
            f"  Wrong Key    : {ep.wrong_key_for_door}",
            f"  Goal Reached : {ep.goal_reached}",
            f"",
            f"  LLM Calls    : {ep.llm_calls}",
            f"  Interventions: {ep.interventions}",
            f"  Reflections  : {ep.reflections}",
            f"  Cache Inval. : {ep.cache_invalidations}",
            f"  Cache Hits   : {ep.cache_hits}",
            f"  Plans Gen.   : {ep.plans_generated}",
            f"  Oscillations : {ep.oscillation_events}",
            f"  Stuck Events : {ep.stuck_events}",
            f"",
            f"  Failure Type : {ep.failure_type}",
            f"  PPO Entropy  : {ep.avg_ppo_entropy:.4f}" if ep.avg_ppo_entropy >= 0 else "  PPO Entropy  : n/a",
            f"{'─'*40}",
        ]
        self._train_f.write("\n".join(lines) + "\n")

        # ── Write to metrics.csv ──────────────────────────────────────────────
        def _ag(matched, requested):
            return round(matched / requested, 4) if requested > 0 else ""
        row = {
            "episode_id":          ep.episode,
            "iteration_id":        ep.iteration_id,
            "episode":             ep.episode,
            "reward":              round(ep.total_reward, 6),
            "success":             int(ep.success),
            "steps":               ep.total_steps,
            "llm_calls":           ep.llm_calls,
            "interventions":       ep.interventions,
            "reflections":         ep.reflections,
            "cache_invalidations": ep.cache_invalidations,
            "oscillation_events":  ep.oscillation_events,
            "stuck_events":        ep.stuck_events,
            "plans_generated":     ep.plans_generated,
            "cache_hits":          ep.cache_hits,
            "key_detected":        int(ep.key_detected),
            "key_picked":          int(ep.key_picked),
            "door_detected":       int(ep.door_detected),
            "door_opened":         int(ep.door_opened),
            "goal_reached":        int(ep.goal_reached),
            "lava_collision":      int(ep.lava_collision),
            "wrong_key_for_door":  int(ep.wrong_key_for_door),
            "timeout":             int(ep.timeout),
            "navigation_agreement": _ag(ep.nav_matched, ep.nav_requested),
            "pickup_agreement":     _ag(ep.pickup_matched, ep.pickup_requested),
            "toggle_agreement":     _ag(ep.toggle_matched, ep.toggle_requested),
            "teacher_requested_navigation": ep.nav_requested,
            "teacher_requested_pickup":     ep.pickup_requested,
            "teacher_requested_toggle":     ep.toggle_requested,
            "pickup_attempts":     ep.pickup_attempts,
            "toggle_attempts":     ep.toggle_attempts,
            "successful_pickups":  int(ep.key_picked),
            "successful_door_opens": int(ep.door_opened),
            "avg_ppo_entropy":     ep.avg_ppo_entropy,
            "failure_type":        ep.failure_type,
            "termination_reason":  ep.termination_reason,
            # Per-episode execution funnel
            "key_visible_steps":   ep.key_visible_steps,
            "key_adjacent_steps":  ep.key_adjacent_steps,
            "key_picked_steps":    ep.key_picked_steps,
            "door_adjacent_steps": ep.door_adjacent_steps,
            "door_facing_steps":   ep.door_facing_steps,
            "door_opened_steps":   ep.door_opened_steps,
        }
        self._csv.writerow(row)
        self._csv_raw.flush()

        self._ep = None
        self._ep_ppo_entropies = []

    # ── Step logging ───────────────────────────────────────────────────────────

    def log_step(self, rec: StepRecord) -> None:
        """Write one compact JSON line to trajectories.jsonl."""
        if not self.enabled:
            return
        # Collect PPO entropy for episode average
        if rec.ppo_entropy >= 0:
            self._ep_ppo_entropies.append(rec.ppo_entropy)
        # Auto-update episode record task flags
        if self._ep is not None:
            ep = self._ep
            if "key" in rec.obs.lower():
                ep.key_detected = True
            if rec.holding not in ("nothing", ""):
                ep.key_picked = True
            if "door" in rec.obs.lower():
                ep.door_detected = True
            if rec.action_id == 5 and rec.reward > 0:
                ep.door_opened  = True
                ep.goal_reached = True
            # Per-episode action-specific teacher agreement (teacher argmax vs student)
            _ta, _sa = rec.teacher_action, rec.action_id
            if _ta in (0, 1, 2):
                ep.nav_requested += 1
                ep.nav_matched   += int(_sa == _ta)
            elif _ta == 3:
                ep.pickup_requested += 1
                ep.pickup_matched   += int(_sa == 3)
            elif _ta == 5:
                ep.toggle_requested += 1
                ep.toggle_matched   += int(_sa == 5)
            if _sa == 3:
                ep.pickup_attempts += 1
            if _sa == 5:
                ep.toggle_attempts += 1
            if rec.intervention:
                ep.interventions += 1
            if rec.oscillation:
                ep.oscillation_events += 1
        # Write compact JSON line
        entry = {
            "ep":         rec.episode,
            "step":       rec.step,
            "obs":        rec.obs,
            "hold":       rec.holding,
            "plan":       rec.plan,
            "subgoal":    rec.subgoal,
            "skill":      rec.skill,
            "action":     rec.action,
            "act_id":     rec.action_id,
            "rew":        round(rec.reward, 4),
            "prog":       int(rec.progress),
            "dist":       round(rec.dist_to_target, 2) if rec.dist_to_target >= 0 else -1,
            "interv":     int(rec.intervention),
            "reflect":    int(rec.reflection),
            "osc":        int(rec.oscillation),
            "entropy":    round(rec.ppo_entropy, 4) if rec.ppo_entropy >= 0 else -1,
            "val":        round(rec.ppo_value,   4) if rec.ppo_value   >= 0 else -1,
            "pos":        list(rec.agent_pos),
        }
        self._traj_f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # ── Events ────────────────────────────────────────────────────────────────

    def log_event(self, event: str, episode: int = 0, step: int = 0) -> None:
        """Append a single-line event to training.log."""
        if not self.enabled:
            return
        self._train_f.write(
            f"  [EVENT ep={episode} step={step}] {event}\n"
        )

    # ── Interventions ─────────────────────────────────────────────────────────

    def log_intervention(self, reason: str, episode: int = 0, step: int = 0,
                         evidence: dict = None) -> None:
        if not self.enabled:
            return
        ev_str = " | ".join(f"{k}={v}" for k, v in (evidence or {}).items())
        self._train_f.write(
            f"  [INTERVENTION ep={episode} step={step}] "
            f"reason={reason} {ev_str}\n"
        )
        # Also write to trajectories.jsonl as a flag
        flag = {
            "ep": episode, "step": step,
            "EVENT": "intervention", "reason": reason,
            **(evidence or {}),
        }
        self._traj_f.write(json.dumps(flag, separators=(",", ":")) + "\n")

    # ── LLM prompts ───────────────────────────────────────────────────────────

    def log_llm_prompt(self, tag: str, episode: int, step: int,
                       system: str = "", user: str = "",
                       raw_response: str = "",
                       parsed_plan: str = "") -> None:
        """
        Append one prompt block to prompts.log.
        This is the ONLY prompt file. No per-episode prompt files.
        """
        if not self.enabled:
            return
        ts = time.strftime("%H:%M:%S")
        block = (
            f"\n[{ts} | EP {episode:04d} | STEP {step:04d} | {tag.upper()}]\n"
            f"{'─'*50}\n"
            f"SYSTEM:\n{system.strip()}\n\n"
            f"USER:\n{user.strip()}\n\n"
            f"RAW RESPONSE:\n{raw_response.strip()}\n\n"
            f"PARSED:\n{parsed_plan.strip()}\n"
            f"{'═'*50}\n"
        )
        self._prompt_f.write(block)
        # Count LLM calls
        if self._ep is not None:
            if tag == "planner":
                self._ep.llm_calls += 1
            elif tag == "reflection":
                self._ep.reflections += 1

    # ── Reflection ────────────────────────────────────────────────────────────

    def log_reflection(self, text: str, episode: int, success: bool) -> None:
        """Log a generated reflection to training.log."""
        if not self.enabled:
            return
        label = "SUCCESS" if success else "FAILURE"
        self._train_f.write(
            f"  [REFLECTION ep={episode} {label}] {text[:120]}\n"
        )
        if self._ep is not None:
            self._ep.reflections += 1

    # ── Debug ─────────────────────────────────────────────────────────────────

    def log_debug(self, msg: str, episode: int = 0, step: int = 0) -> None:
        """
        Write to debug.log only for SEVERE events.
        Examples: repeated impossible plans, contradictory observations,
        hallucination detected, repeated cache invalidation loop.
        Keep this file small.
        """
        if not self.enabled:
            return
        ts = time.strftime("%H:%M:%S")
        self._debug_f.write(f"[{ts} | ep={episode} step={step}] {msg}\n")

    # ── Cache / planner events ─────────────────────────────────────────────────

    def log_cache_invalidation(self, obs_text: str, episode: int,
                               failures: int, confidence: float) -> None:
        if not self.enabled:
            return
        self._train_f.write(
            f"  [CACHE_INVALIDATED ep={episode}] "
            f"obs='{obs_text[:50]}' failures={failures} conf={confidence:.2f}\n"
        )
        if self._ep is not None:
            self._ep.cache_invalidations += 1
        if failures > 8:  # severe repeated invalidation → debug
            self.log_debug(
                f"Repeated cache invalidation: obs='{obs_text[:50]}' "
                f"failures={failures}", episode=episode
            )

    def log_replan(self, obs_text: str, episode: int, step: int,
                   old_plan: str, new_plan: str, reason: str) -> None:
        if not self.enabled:
            return
        self._train_f.write(
            f"  [REPLAN ep={episode} step={step}] "
            f"reason={reason} | '{old_plan[:40]}' → '{new_plan[:40]}'\n"
        )
        if self._ep is not None:
            self._ep.replans += 1

    def log_plan_created(self, plan: str, episode: int, step: int,
                         reason: str = "new_obs") -> None:
        if not self.enabled:
            return
        self._train_f.write(
            f"  [PLAN ep={episode} step={step}] "
            f"reason={reason} | '{plan[:60]}'\n"
        )
        if self._ep is not None:
            self._ep.plans_generated += 1

    # ── Flush / close ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        if not self.enabled:
            return
        for f in (self._train_f, self._traj_f, self._prompt_f,
                  self._csv_raw, self._debug_f):
            try:
                f.flush()
            except Exception:
                pass

    def close(self) -> None:
        if not self.enabled:
            return
        for f in (self._train_f, self._traj_f, self._prompt_f,
                  self._csv_raw, self._debug_f):
            try:
                f.close()
            except Exception:
                pass
