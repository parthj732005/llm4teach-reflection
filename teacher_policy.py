#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
teacher_policy.py — Failure-triggered intervention teacher policy

Architecture philosophy:
  - PPO is the PRIMARY controller (handles normal movement)
  - LLM/planner intervenes ONLY when failure detectors fire
  - Skills verify their own success (grounded execution)
  - Teacher weight decays naturally via PPO's ks_coef

New in this version:
  A. FailureDetector — detects stuck, oscillation, repeated plan,
     failed interaction, no-progress
  B. Threshold-triggered LLM intervention — not every step
  C. Cache invalidation on repeated failure
  D. Execution verification hooks via skill.failure_reason
  E. Intervention statistics tracking
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from collections import deque

from planner import Planner
from skill import GoTo_Goal, Explore, Pickup, Drop, Toggle, Wait
from mediator import IDX_TO_SKILL, IDX_TO_OBJECT


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

class FailureDetector:
    """
    Lightweight per-episode failure detector.

    Tracks:
      - Stuck: position unchanged for N steps
      - Oscillation: left-right action alternation
      - Repeated plan: same plan string N times in a row
      - Failed interaction: toggle/pickup returned failure_reason
      - No-progress: distance to last-known goal object unchanged

    Intervention is triggered when ANY detector exceeds its threshold.
    """

    def __init__(
        self,
        stuck_threshold       = 20,   # raised: tolerate longer stuck periods
        oscillation_window    = 16,   # raised: require sustained oscillation
        repeated_plan_limit   = 999,  # disabled: repeated plans are NORMAL
        failed_interact_limit = 5,    # raised: tolerate more failed interactions
        env_type              = "simpledoorkey",
    ):
        self.stuck_threshold       = stuck_threshold
        self.oscillation_window    = oscillation_window
        self.repeated_plan_limit   = repeated_plan_limit
        self.failed_interact_limit = failed_interact_limit
        self.env_type              = env_type.lower()

        # For ColoredDoorKey: track wrong-key attempts separately with a lower
        # threshold (2 mismatches) so the planner intervenes quickly.
        # For LavaDoorKey/SimpleDoorKey: not used.
        self._wrong_key_limit      = 2 if self.env_type == "coloreddoorkey" else 999
        self._wrong_key_count      = 0

        # Track lava deaths so intervention context can reference them.
        self._lava_death_count     = 0

        # Progress tracking — repeated navigation toward target is NOT a failure
        self._min_dist_seen = float('inf')   # best distance to goal seen
        self._progress_window = deque(maxlen=30)  # recent distance readings

        # Internal state
        self._pos_history    = deque(maxlen=stuck_threshold)
        self._action_history = deque(maxlen=oscillation_window)
        self._plan_history   = deque(maxlen=repeated_plan_limit)
        self._failed_interactions = 0

        # Diagnostics
        self.last_trigger_reason = ""

    def reset(self):
        self._pos_history.clear()
        self._action_history.clear()
        self._plan_history.clear()
        self._failed_interactions = 0
        self._wrong_key_count     = 0
        self.last_trigger_reason  = ""
        self._min_dist_seen       = float('inf')
        self._progress_window.clear()

    def update(self, agent_pos, action, plan_str):
        """Update detector state with latest step info."""
        if agent_pos is not None:
            self._pos_history.append(tuple(agent_pos))
        if action is not None:
            self._action_history.append(int(action))
        if plan_str:
            self._plan_history.append(plan_str.strip())

    def report_failed_interaction(self, reason: str = ""):
        """Call when Toggle/Pickup fails to actually change object state."""
        self._failed_interactions += 1
        if reason:
            self.last_trigger_reason = reason
        # Track colour-mismatch attempts separately for ColoredDoorKey.
        if "does not match" in reason:
            self._wrong_key_count += 1

    def report_lava_death(self):
        """Call from Game.py when a lava-death termination is detected."""
        self._lava_death_count += 1
        self.last_trigger_reason = "lava_death"

    def reset_interaction_count(self):
        self._failed_interactions = 0

    # ── Individual detectors ──────────────────────────────────────────────────

    def is_stuck(self):
        if len(self._pos_history) < self.stuck_threshold:
            return False
        return len(set(self._pos_history)) <= 2

    def is_oscillating(self):
        if len(self._action_history) < self.oscillation_window:
            return False
        actions  = list(self._action_history)
        lr_pairs = sum(
            1 for i in range(len(actions) - 1)
            if (actions[i] == 0 and actions[i+1] == 1)
            or (actions[i] == 1 and actions[i+1] == 0)
        )
        # Require 75% of window to be L-R alternation — not just half
        return lr_pairs >= int(self.oscillation_window * 0.75)

    def is_repeating_plan(self):
        # Disabled — repeated plans are normal and desirable for PPO stability
        return False

    def is_making_progress(self, agent_pos) -> bool:
        """
        Return True if the agent's position has changed recently.
        Used to suppress interventions when the agent is still moving.
        Checks whether the agent visited at least 3 distinct positions
        in the last 30 steps — a position-diversity proxy for movement.
        """
        if agent_pos is None:
            return True   # can't tell — don't intervene
        pos_tuple = (int(agent_pos[0]), int(agent_pos[1]))
        self._progress_window.append(pos_tuple)
        # Require at least 3 distinct positions in the window to count as progress
        if len(self._progress_window) >= 10:
            return len(set(self._progress_window)) >= 3
        return True

    def has_repeated_failed_interactions(self):
        return self._failed_interactions >= self.failed_interact_limit

    # ── Composite trigger ─────────────────────────────────────────────────────

    def should_intervene(self, agent_pos=None):
        """
        Return (should_intervene: bool, reason: str).

        IMPORTANT: only trigger when there is persistent failure WITH no progress.
        Temporary execution instability should NOT trigger intervention.
        """
        making_progress = self.is_making_progress(agent_pos)

        # Stuck: no position change AND no progress trend
        if self.is_stuck() and not making_progress:
            return True, "stuck"

        # Oscillation: must be severe (75% L-R) AND no progress
        if self.is_oscillating() and not making_progress:
            return True, "oscillation"

        # Repeated failed interactions: only if truly repeated impossible actions
        if self.has_repeated_failed_interactions():
            return True, "failed_interaction"

        # ColoredDoorKey: wrong key colour — intervene quickly (threshold=2)
        if self._wrong_key_count >= self._wrong_key_limit:
            return True, "wrong_key_colour"

        # LavaDoorKey: lava death — always intervene immediately
        if self._lava_death_count > 0:
            return True, "lava_death"

        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER POLICY
# ══════════════════════════════════════════════════════════════════════════════

class TeacherPolicy:
    """
    Failure-triggered intervention teacher policy with plan execution lifecycle.

    Normal operation:
      PPO is primary controller. Teacher provides kickstarting signal.
      The ACTIVE PLAN persists until success, failure, or timeout.
      Planner is NOT queried every step — only on new observations
      or when FailureDetector fires.

    Intervention mode (when FailureDetector fires):
      - Invalidates bad cached plan via planner.report_failure()
      - Forces fresh LLM query with grounded failure context
      - Cooldown suppresses re-intervention for N steps

    Parameters
    ----------
    task              : task name (e.g. "SimpleDoorKey")
    offline           : use offline planner (no LLM calls)
    soft              : soft planner mode (probability-weighted skills)
    prefix            : task description prefix for LLM
    action_space      : number of primitive actions
    agent_view_size   : MiniGrid agent view size
    avoid_lava        : pass True for LavaDoorKey to block lava in GoTo
    intervention_cooldown : steps to suppress re-intervention after one fires
    """

    def __init__(self, task, offline, soft, prefix,
                 action_space, agent_view_size, avoid_lava=False,
                 intervention_cooldown=30):
        self.planner         = Planner(task, offline, soft, prefix)
        self.agent_view_size = agent_view_size
        self.action_space    = action_space
        self.avoid_lava      = avoid_lava

        # Persistent Explore instance — MUST survive across timesteps.
        # Creating a new Explore() every step was the root cause of broken
        # exploration: visited tracking, oscillation detection, and
        # no-progress counters were all reset to empty every single step.
        self._explore_skill  = Explore(agent_view_size)

        # Failure detection — same conservative thresholds for all envs;
        # env-specific failure modes (wrong_key, lava_death) handled separately.
        self._detector      = FailureDetector(
            stuck_threshold       = 20,
            oscillation_window    = 16,
            repeated_plan_limit   = 999,  # disabled
            failed_interact_limit = 5,
            env_type              = task,
        )
        self._last_obs_text = ""
        self._last_agent_pos = None

        # Plan execution lifecycle
        self._active_plan_text  = ""
        self._plan_step_count   = 0
        self._plan_max_steps    = 80   # raised: allow plan to run longer

        # Intervention cooldown — longer to prevent spam
        self._cooldown_steps    = intervention_cooldown
        self._cooldown_remaining= 0

        # Intervention tracking
        self._intervention_count  = 0
        self._stuck_count         = 0
        self._oscillation_count   = 0
        self._repeated_plan_count = 0
        self._failed_inter_count  = 0

    # ── Public API for Game.py / reflection system ────────────────────────────

    def report_lava_death(self):
        """
        Called by Game.py when a lava-death termination is detected for
        LavaDoorKey.  Immediately triggers intervention on the next planning
        call so the LLM gets lava-avoidance context injected.
        """
        self._detector.report_lava_death()

    @property
    def intervention_stats(self):
        return {
            "total_interventions":   self._intervention_count,
            "stuck_events":          self._stuck_count,
            "oscillation_events":    self._oscillation_count,
            "repeated_plan_events":  self._repeated_plan_count,
            "failed_inter_events":   self._failed_inter_count,
        }

    def get_skill_name(self, skill):
        try:
            return IDX_TO_SKILL[skill["action"]] + " " + IDX_TO_OBJECT[skill["object"]]
        except (AttributeError, KeyError):
            return "None"

    def reset(self):
        self.skill              = None
        self.skill_list         = []
        self.skill_teminated    = False
        self._detector.reset()
        self._last_obs_text     = ""
        self._active_plan_text  = ""
        self._plan_step_count   = 0
        self._cooldown_remaining= 0
        self.planner.reset()
        # Reset persistent explore skill for new episode
        self._explore_skill.reset()

    # ── Skill instantiation ───────────────────────────────────────────────────

    def skill2teacher(self, skill):
        skill_action = skill['action']
        if skill_action == 0:
            # Return the PERSISTENT explore instance — do NOT create a new one.
            # Creating a new Explore() every step resets visited/oscillation
            # tracking and was the root cause of broken exploration behavior.
            return self._explore_skill
        elif skill_action == 1:
            return GoTo_Goal(skill['coordinate'], avoid_lava=self.avoid_lava)
        elif skill_action == 2:
            return Pickup(skill['object'], avoid_lava=self.avoid_lava)
        elif skill_action == 3:
            return Drop(skill['object'])
        elif skill_action == 4:
            return Toggle(skill['object'])
        elif skill_action == 6:
            return Wait()
        else:
            assert False, f"invalid skill action index: {skill_action}"

    # ── Skill execution with verification ────────────────────────────────────

    def get_action(self, skill_list, obs):
        """
        Execute the next skill step from skill_list.

        Verifies skill termination — if a skill sets failure_reason,
        reports it to the failure detector.
        """
        terminated = True
        action     = None

        while action is None and terminated and len(skill_list) > 0:
            skill   = skill_list.pop(0)
            teacher = self.skill2teacher(skill)
            action, terminated = teacher(obs)

            # Execution verification: check if skill reported a failure
            failure_reason = getattr(teacher, "failure_reason", "")
            if failure_reason:
                self._detector.report_failed_interaction(failure_reason)

        if action is None:
            # Fall back to Explore rather than WAIT — Explore avoids lava and
            # provides a useful navigation signal even when GoTo can't find a path.
            explore_action, _ = self._explore_skill(obs)
            action = explore_action if explore_action is not None else 6

        action = np.array(
            [i == action for i in range(self.action_space)],
            dtype=np.float32,
        )
        return action

    # ── Main call ─────────────────────────────────────────────────────────────

    def __call__(self, obs):
        """
        Return teacher action probability vector for PPO kickstarting.

        Execution lifecycle:
          1. Decode current obs text (planner cache key).
          2. Tick intervention cooldown.
          3. Update failure detectors.
          4. Decide whether to replan:
               - observation changed → natural replan
               - plan lifetime exceeded → forced replan
               - failure detector fired AND cooldown elapsed → intervention
          5. Get plan from planner.
          6. Execute skill step and return one-hot action vector.
        """
        # ── Decode observation ────────────────────────────────────────────────
        try:
            obs_text = self.planner.mediator.RL2LLM(obs)
        except Exception:
            obs_text = ""

        # ── Tick cooldown ─────────────────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ── Update detectors ──────────────────────────────────────────────────
        try:
            agent_map = obs[:, :, 3] if len(obs.shape) == 3 else obs[0, :, :, 3]
            agent_pos = np.argwhere(agent_map != 4)
            agent_pos = agent_pos[0] if len(agent_pos) > 0 else None
        except Exception:
            agent_pos = None

        self._last_agent_pos = agent_pos

        last_plan = getattr(self.planner, "dialogue_user", "")
        if "\n" in last_plan:
            last_plan = last_plan.split("\n")[1]
        self._detector.update(agent_pos, None, last_plan)

        # ── Decide whether to replan ──────────────────────────────────────────
        obs_changed    = (obs_text != self._active_plan_text)
        plan_timed_out = (self._plan_step_count >= self._plan_max_steps)

        # Pass agent_pos so detector can check progress before firing
        intervene, reason = self._detector.should_intervene(agent_pos)
        do_intervene   = intervene and (self._cooldown_remaining == 0) and bool(obs_text)

        if obs_changed:
            # New state → natural replan — report success on old plan if any
            if self._active_plan_text:
                self.planner.report_success(self._active_plan_text)
            self._active_plan_text = obs_text
            self._plan_step_count  = 0
            self._detector.reset()
        elif plan_timed_out:
            # Timeout alone is NOT a failure — the plan ran its course naturally
            # Only invalidate if the observation hasn't changed at all (truly stuck)
            if obs_text == self._active_plan_text:
                print(f"[TeacherPolicy] Plan lifetime reached ({self._plan_max_steps} steps)"
                      f" — allowing natural replan: '{obs_text[:50]}'")
                # Soft replan: do NOT invalidate cache, just reset timer
                self._plan_step_count = 0
            else:
                self._plan_step_count = 0
        elif do_intervene:
            self._handle_intervention(obs_text, reason)

        self._last_obs_text    = obs_text
        self._plan_step_count += 1

        # ── Get plan from planner ─────────────────────────────────────────────
        skill_list, probs = self.planner(obs)

        # ── Execute and return action vector ──────────────────────────────────
        action = np.zeros(self.action_space)
        for skills, prob in zip(skill_list, probs):
            action += self.get_action(list(skills), obs) * prob

        return action

    def _handle_intervention(self, obs_text: str, reason: str):
        """
        Process an intervention event:
          - Report failure to planner (may invalidate cache)
          - Inject failure context into mediator for richer LLM prompt
          - Reset the failure detector to avoid repeated interventions
          - Log statistics
        """
        self._intervention_count += 1

        # Tally by reason
        if reason == "stuck":
            self._stuck_count += 1
        elif reason == "oscillation":
            self._oscillation_count += 1
        elif reason == "repeated_plan":
            self._repeated_plan_count += 1
        elif reason == "failed_interaction":
            self._failed_inter_count += 1
        elif reason == "wrong_key_colour":
            self._failed_inter_count += 1
        elif reason == "lava_death":
            self._stuck_count += 1

        print(
            f"[TeacherPolicy] Intervention #{self._intervention_count} "
            f"— reason: {reason} | obs: '{obs_text[:55]}'"
        )

        # Tell planner this plan is failing → may invalidate cache
        self.planner.report_failure(obs_text)

        # Build env-specific failure context for the LLM prompt.
        task = getattr(self._detector, "env_type", "simpledoorkey")
        if reason == "wrong_key_colour":
            failure_ctx = (
                "Tried to open door with the wrong key colour — it did not open. "
                "Drop the held key, find the key whose colour matches the door, pick it up, then open the door."
            )
        elif reason == "lava_death":
            failure_ctx = (
                "Agent stepped on lava and the episode ended. "
                "Lava tiles are lethal — always navigate around them. Choose a path that avoids all lava tiles."
            )
        elif reason == "stuck" and task == "lavadoorkey":
            failure_ctx = (
                "Agent is stuck — possibly blocked by lava. Navigate around lava tiles and find an alternative path."
            )
        elif reason == "failed_interaction" and task == "coloreddoorkey":
            failure_ctx = (
                "Door interaction failed — verify the held key colour matches the door colour before opening."
            )
        else:
            failure_ctx = f"Previously {reason} at this state. Try a different approach."

        # Inject failure context into mediator so LLM gets richer prompt
        mediator = getattr(self.planner, "mediator", None)
        if mediator is not None and hasattr(mediator, "set_failure_context"):
            mediator.set_failure_context(failure_ctx)

        # Reset env-specific counters so they don't re-trigger immediately
        if reason == "wrong_key_colour":
            self._detector._wrong_key_count = 0
        if reason == "lava_death":
            self._detector._lava_death_count = 0

        # Reset detector + start cooldown so it doesn't fire again immediately
        self._detector.reset()
        self._cooldown_remaining = self._cooldown_steps
        self._plan_step_count    = 0   # restart plan lifetime after intervention
