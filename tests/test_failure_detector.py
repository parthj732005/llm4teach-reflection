"""
tests/test_failure_detector.py

Unit tests for FailureDetector in teacher_policy.py.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock

# Stub out all heavy dependencies before importing teacher_policy
def _mock_module(name):
    mod = MagicMock()
    sys.modules[name] = mod
    return mod

for _mod in [
    "gymnasium", "gymnasium.spaces",
    "minigrid", "minigrid.core", "minigrid.core.actions",
    "torch", "torch.nn", "torch.nn.functional",
    "cv2", "base64", "requests",
]:
    sys.modules.setdefault(_mod, MagicMock())

# Make torch.nn accessible as attribute of torch mock
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]

from teacher_policy import FailureDetector


class TestFailureDetectorStuck:
    def test_not_stuck_initially(self):
        fd = FailureDetector(stuck_threshold=5)
        fd.reset()
        intervene, _ = fd.should_intervene(agent_pos=None)
        assert not intervene

    def test_stuck_when_same_position_repeated(self):
        fd = FailureDetector(stuck_threshold=5, oscillation_window=100)
        fd.reset()
        for _ in range(6):
            fd.update(agent_pos=[3, 4], action=2, plan_str="go to <door>")
        # Force progress window to show no movement
        for _ in range(10):
            fd._progress_window.append((3, 4))
        assert fd.is_stuck()

    def test_not_stuck_when_moving(self):
        fd = FailureDetector(stuck_threshold=5)
        fd.reset()
        positions = [[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3]]
        for pos in positions:
            fd.update(agent_pos=pos, action=2, plan_str="go to <key>")
        assert not fd.is_stuck()

    def test_stuck_with_two_positions(self):
        fd = FailureDetector(stuck_threshold=4, oscillation_window=100)
        fd.reset()
        for i in range(5):
            fd.update(agent_pos=[1, 1] if i % 2 == 0 else [1, 2],
                      action=0, plan_str="explore")
        for _ in range(10):
            fd._progress_window.append((1, 1))
        assert fd.is_stuck()


class TestFailureDetectorOscillation:
    def test_oscillation_detected(self):
        fd = FailureDetector(oscillation_window=8)
        fd.reset()
        for _ in range(8):
            fd.update(agent_pos=[2, 2], action=0, plan_str="explore")
            fd.update(agent_pos=[2, 2], action=1, plan_str="explore")
        assert fd.is_oscillating()

    def test_no_oscillation_uniform_actions(self):
        fd = FailureDetector(oscillation_window=8)
        fd.reset()
        for _ in range(16):
            fd.update(agent_pos=[2, 2], action=2, plan_str="go to <key>")
        assert not fd.is_oscillating()

    def test_oscillation_requires_75_percent(self):
        fd = FailureDetector(oscillation_window=8)
        fd.reset()
        # Only 50% L-R pairs — should NOT trigger
        actions = [0, 1, 0, 1, 2, 2, 2, 2]
        for a in actions:
            fd.update(agent_pos=[2, 2], action=a, plan_str="explore")
        assert not fd.is_oscillating()


class TestFailureDetectorInteraction:
    def test_failed_interaction_triggers(self):
        fd = FailureDetector(failed_interact_limit=3)
        fd.reset()
        fd.report_failed_interaction("door locked")
        fd.report_failed_interaction("door locked")
        fd.report_failed_interaction("door locked")
        assert fd.has_repeated_failed_interactions()

    def test_failed_interaction_below_threshold(self):
        fd = FailureDetector(failed_interact_limit=5)
        fd.reset()
        fd.report_failed_interaction("door locked")
        fd.report_failed_interaction("door locked")
        assert not fd.has_repeated_failed_interactions()

    def test_reset_clears_interaction_count(self):
        fd = FailureDetector(failed_interact_limit=2)
        fd.reset()
        fd.report_failed_interaction("fail")
        fd.report_failed_interaction("fail")
        assert fd.has_repeated_failed_interactions()
        fd.reset()
        assert not fd.has_repeated_failed_interactions()

    def test_wrong_key_tracked_separately(self):
        fd = FailureDetector(env_type="coloreddoorkey")
        fd.reset()
        fd.report_failed_interaction("key does not match door color")
        fd.report_failed_interaction("key does not match door color")
        intervene, reason = fd.should_intervene(agent_pos=None)
        assert intervene
        assert reason == "wrong_key_colour"


class TestFailureDetectorLava:
    def test_lava_death_triggers_immediately(self):
        fd = FailureDetector(env_type="lavadoorkey")
        fd.reset()
        fd.report_lava_death()
        intervene, reason = fd.should_intervene(agent_pos=None)
        assert intervene
        assert reason == "lava_death"


class TestFailureDetectorReset:
    def test_full_reset_clears_all_state(self):
        fd = FailureDetector(stuck_threshold=3, failed_interact_limit=2)
        fd.reset()
        for _ in range(4):
            fd.update(agent_pos=[1, 1], action=0, plan_str="explore")
        fd.report_failed_interaction("fail")
        fd.report_failed_interaction("fail")
        fd.reset()
        assert not fd.has_repeated_failed_interactions()
        assert fd.last_trigger_reason == ""
        assert len(fd._pos_history) == 0
