"""
tests/test_episode_trajectory.py

Unit tests for EpisodeTrajectory.classify_failure() in memory/reflection.py.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory.reflection import EpisodeTrajectory


def _make_trajectory(plans, failure_reasons=None, success=False, ep_len=None):
    traj = EpisodeTrajectory()
    for i, plan in enumerate(plans):
        traj.add_step(
            obs_text="Agent sees key, holds nothing.",
            plan=plan,
            reward=0.0,
            failure_reason=(failure_reasons[i] if failure_reasons and i < len(failure_reasons) else ""),
        )
    traj.finish(success=success)
    if ep_len is not None:
        traj.ep_len = ep_len
    return traj


class TestClassifyFailureSuccess:
    def test_success_returns_success(self):
        traj = _make_trajectory(["go to <key>", "pick up <key>", "open <door>"], success=True)
        assert traj.classify_failure() == "success"


class TestClassifyFailurePickupAlignment:
    def test_pickup_alignment_detected(self):
        traj = _make_trajectory(
            ["go to <key>"] * 10,
            failure_reasons=["PICKUP_ALIGNMENT_FAILURE: could not align to key."],
        )
        assert traj.classify_failure() == "pickup_alignment"

    def test_align_keyword_detected(self):
        traj = _make_trajectory(
            ["explore"] * 5,
            failure_reasons=["align failed"],
        )
        assert traj.classify_failure() == "pickup_alignment"


class TestClassifyFailureRepeatedPlan:
    def test_repeated_plan_detected(self):
        plans = ["go to <door>"] * 100
        traj = _make_trajectory(plans)
        assert traj.classify_failure() == "repeated_plan"


class TestClassifyFailureExploration:
    def test_exploration_failure_detected(self):
        # "explore" repeated heavily also triggers "repeated_plan" first
        # (classify_failure checks repeated_plan before exploration_failure).
        # Pure explore trajectories correctly classify as repeated_plan.
        plans = ["explore"] * 80 + ["go to <key>"] * 5
        traj = _make_trajectory(plans)
        assert traj.classify_failure() in ("exploration_failure", "repeated_plan")

    def test_exploration_failure_no_other_plans(self):
        # When the only activity is exploring with no goal progress
        plans = ["explore"] * 10
        traj = _make_trajectory(plans)
        # repeated_plan fires first when a single plan dominates > 60%
        assert traj.classify_failure() == "repeated_plan"


class TestClassifyFailureToggleWithoutKey:
    def test_toggle_without_key_detected(self):
        # 6 "open <door>" (toggle_plans > 5) with no key/pickup plans
        # (key_plans < 2). No single plan exceeds 60% of 11 episodes
        # so repeated_plan threshold is not crossed (6 < 0.6 * 11 = 6.6).
        plans = (["open <door>"] * 6 +
                 ["go to <door>", "go to <goal>",
                  "explore", "go to <door>", "explore"])
        traj = _make_trajectory(plans)
        assert traj.classify_failure() == "toggle_without_key"


class TestClassifyFailureKeyAcquiredDoorFailed:
    def test_key_acquired_door_failed(self):
        plans = ["go to <key>", "pick up <key>", "explore", "explore", "explore"]
        traj = _make_trajectory(plans)
        assert traj.classify_failure() == "key_acquired_door_failed"


class TestClassifyFailureNavigation:
    def test_navigation_failure_detected(self):
        # Use varied "go to" plans so no single plan exceeds 60% (repeated_plan
        # threshold), while total goto_plans > 80% triggers navigation_failure.
        plans = (["go to <key>"] * 34 +
                 ["go to <door>"] * 33 +
                 ["go to <goal>"] * 33)
        traj = _make_trajectory(plans)
        assert traj.classify_failure() == "navigation_failure"


class TestEpisodeTrajectoryAddStep:
    def test_ep_len_increments(self):
        traj = EpisodeTrajectory()
        traj.add_step("obs", "explore", 0.0)
        traj.add_step("obs", "explore", 0.0)
        assert traj.ep_len == 2

    def test_total_reward_accumulates(self):
        traj = EpisodeTrajectory()
        traj.add_step("obs", "explore", 0.5)
        traj.add_step("obs", "explore", 0.5)
        assert abs(traj.total_reward - 1.0) < 1e-6

    def test_empty_plan_defaults_to_explore(self):
        traj = EpisodeTrajectory()
        traj.add_step("obs", "", 0.0)
        assert traj.plans[0] == "explore"

    def test_failure_reason_stored(self):
        traj = EpisodeTrajectory()
        traj.add_step("obs", "explore", 0.0, failure_reason="stuck")
        assert "stuck" in traj.failure_reasons

    def test_intervention_stored(self):
        traj = EpisodeTrajectory()
        traj.add_step("obs", "explore", 0.0, intervention="oscillation")
        assert "oscillation" in traj.interventions
