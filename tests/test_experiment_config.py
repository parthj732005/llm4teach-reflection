"""
tests/test_experiment_config.py

Unit tests for ExperimentConfig and build_experiment_config.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from experiment_config import ExperimentConfig, build_experiment_config, PHASE_ORDER


class TestExperimentConfigDefaults:
    def test_default_construction(self):
        cfg = ExperimentConfig()
        assert cfg.use_planner is True
        assert cfg.use_reflection is False
        assert cfg.use_episode_memory is False
        assert cfg.use_kickstarting is True
        assert cfg.use_teacher_policy is True

    def test_default_hyperparameters(self):
        cfg = ExperimentConfig()
        assert cfg.total_iterations == 1000
        assert cfg.kickstarting_coef_initial == 1.0
        assert cfg.kickstarting_coef_minimum == 0.15
        assert cfg.learning_rate == 3e-4


class TestExperimentConfigValidation:
    def test_kickstarting_requires_teacher(self):
        cfg = ExperimentConfig(use_kickstarting=True, use_teacher_policy=False)
        assert cfg.use_kickstarting is False

    def test_episode_memory_requires_reflection(self):
        cfg = ExperimentConfig(use_episode_memory=True, use_reflection=False)
        assert cfg.use_episode_memory is False

    def test_state_memory_requires_reflection(self):
        cfg = ExperimentConfig(use_state_memory=True, use_reflection=False)
        assert cfg.use_state_memory is False

    def test_mid_episode_replanning_requires_planner(self):
        cfg = ExperimentConfig(use_mid_episode_replanning=True, use_planner=False)
        assert cfg.use_mid_episode_replanning is False

    def test_failure_classifier_requires_failure_detector(self):
        cfg = ExperimentConfig(use_failure_classifier=True, use_failure_detector=False)
        assert cfg.use_failure_classifier is False

    def test_valid_combination_unchanged(self):
        cfg = ExperimentConfig(
            use_teacher_policy=True,
            use_kickstarting=True,
            use_reflection=True,
            use_episode_memory=True,
        )
        assert cfg.use_kickstarting is True
        assert cfg.use_episode_memory is True


class TestExperimentConfigPaths:
    def test_checkpoint_path(self):
        cfg = ExperimentConfig(phase=3, name="reflection")
        assert "phase3" in cfg.checkpoint_path
        assert "reflection" in cfg.checkpoint_path

    def test_results_path(self):
        cfg = ExperimentConfig(name="planner")
        assert "planner" in cfg.results_path


class TestExperimentConfigSummary:
    def test_summary_returns_string(self):
        cfg = ExperimentConfig()
        s = cfg.summary()
        assert isinstance(s, str)
        assert "ON" in s or "OFF" in s


class TestBuildExperimentConfig:
    def test_ppo_only_preset(self):
        cfg = build_experiment_config("ppo_only")
        assert cfg.use_teacher_policy is False
        assert cfg.use_planner is False
        assert cfg.use_kickstarting is False
        assert cfg.phase == 1

    def test_planner_preset(self):
        cfg = build_experiment_config("planner")
        assert cfg.use_planner is True
        assert cfg.use_kickstarting is True
        assert cfg.use_reflection is False
        assert cfg.phase == 2

    def test_reflection_preset(self):
        cfg = build_experiment_config("reflection")
        assert cfg.use_planner is True
        assert cfg.use_reflection is True
        assert cfg.use_episode_memory is True
        assert cfg.phase == 3

    def test_overrides_applied(self):
        cfg = build_experiment_config("ppo_only", overrides={"total_iterations": 42})
        assert cfg.total_iterations == 42

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            build_experiment_config("nonexistent_phase")

    def test_phase_order_complete(self):
        for name in PHASE_ORDER:
            cfg = build_experiment_config(name)
            assert cfg is not None

    def test_name_case_insensitive(self):
        cfg = build_experiment_config("PPO_ONLY")
        assert cfg.use_planner is False
