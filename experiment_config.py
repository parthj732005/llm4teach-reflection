#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
experiment_config.py — Unified ablation configuration for LLM4Teach

Every major component is controlled by a boolean flag.
No source-code edits are needed between experiments — change the
ExperimentConfig and the entire pipeline adapts.

Usage
-----
    from experiment_config import ExperimentConfig, build_experiment_config

    cfg = build_experiment_config("planner")   # Phase 2
    cfg = build_experiment_config("full")      # Phase 5  (identical to original)
    cfg = ExperimentConfig(use_planner=True, total_iterations=500)
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # ── Component toggles ────────────────────────────────────────────────────
    use_planner:               bool = True
    """Enable LLM-based symbolic planner + teacher kickstarting signal."""

    use_reflection:            bool = False
    """Generate episode-level reflections with Qwen after each episode."""

    use_episode_memory:        bool = False
    """Store/retrieve reflections in ReflectionMemory for planner context."""

    use_state_memory:          bool = False
    """Store/retrieve state-specific failure patterns in StateMemory."""

    use_mid_episode_replanning: bool = False
    """Trigger LLM replan mid-episode when FailureDetector fires."""

    use_failure_classifier:    bool = False
    """Use MidEpisodeFailureClassifier for richer failure labels.
    When False, a generic 'NAVIGATION_FAILURE' label is used."""

    use_failure_detector:      bool = True
    """Run FailureDetector every step (stuck/oscillation/interact checks).
    When False, the detector never fires — no interventions occur."""

    use_teacher_policy:        bool = True
    """Enable teacher policy. When False, PPO trains without any teacher
    signal (uniform distribution returned — kickstarting has no effect)."""

    use_kickstarting:          bool = True
    """Add kickstarting loss to PPO gradient.  Requires use_teacher_policy.
    When False, iter_with_ks=0 on the PPO object."""

    # ── Training hyperparameters ─────────────────────────────────────────────
    total_iterations:          int   = 1000
    episodes_per_iteration:    int   = 10
    ppo_epochs:                int   = 3
    batch_size:                int   = 128
    learning_rate:             float = 3e-4
    entropy_coef:              float = 0.05

    kickstarting_coef_initial: float = 1.0
    kickstarting_coef_descent: float = 0.005
    kickstarting_coef_minimum: float = 0.15
    iter_with_ks:              int   = 3000

    reflection_frequency:      int   = 3    # reflect every N episodes
    replan_budget:             int   = 2    # max mid-episode replans/episode

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_interval:             int   = 50
    num_eval:                  int   = 10
    save_interval:             int   = 200

    # ── Experiment metadata ───────────────────────────────────────────────────
    name:           str = "full"
    phase:          int = 5
    checkpoint_dir: str = "checkpoints"
    results_dir:    str = "results"
    warm_start:     bool = False   # reuse previous phase checkpoint when True

    def __post_init__(self):
        # Enforce logical consistency
        if self.use_kickstarting and not self.use_teacher_policy:
            self.use_kickstarting = False
        if self.use_episode_memory and not self.use_reflection:
            self.use_episode_memory = False
        if self.use_state_memory and not self.use_reflection:
            self.use_state_memory = False
        if self.use_mid_episode_replanning and not self.use_planner:
            self.use_mid_episode_replanning = False
        if self.use_failure_classifier and not self.use_failure_detector:
            self.use_failure_classifier = False

    @property
    def checkpoint_path(self) -> str:
        import os
        return os.path.join(self.checkpoint_dir, f"phase{self.phase}_{self.name}")

    @property
    def results_path(self) -> str:
        import os
        return os.path.join(self.results_dir, self.name)

    def summary(self) -> str:
        lines = [f"ExperimentConfig — phase={self.phase} name='{self.name}'"]
        lines.append(f"  Components : "
                     f"teacher={'ON' if self.use_teacher_policy else 'OFF'} | "
                     f"planner={'ON' if self.use_planner else 'OFF'} | "
                     f"ks={'ON' if self.use_kickstarting else 'OFF'} | "
                     f"reflect={'ON' if self.use_reflection else 'OFF'} | "
                     f"ep_mem={'ON' if self.use_episode_memory else 'OFF'} | "
                     f"st_mem={'ON' if self.use_state_memory else 'OFF'} | "
                     f"replan={'ON' if self.use_mid_episode_replanning else 'OFF'} | "
                     f"fail_det={'ON' if self.use_failure_detector else 'OFF'} | "
                     f"fail_cls={'ON' if self.use_failure_classifier else 'OFF'}")
        lines.append(f"  Training   : itr={self.total_iterations} | "
                     f"eps/itr={self.episodes_per_iteration} | "
                     f"epochs={self.ppo_epochs} | "
                     f"batch={self.batch_size} | lr={self.learning_rate}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Per-phase preset builder
# ─────────────────────────────────────────────────────────────────────────────

# Default per-phase training schedules — override in the notebook via PHASE_CONFIGS
_PHASE_DEFAULTS = {
    "ppo_only": dict(
        phase=1, total_iterations=1000, ppo_epochs=5,
        use_teacher_policy=False, use_planner=False,
        use_kickstarting=False, use_failure_detector=False,
    ),
    "planner": dict(
        phase=2, total_iterations=1000, ppo_epochs=5,
        use_teacher_policy=True, use_planner=True,
        use_kickstarting=True, use_failure_detector=True,
    ),
    "reflection": dict(
        phase=3, total_iterations=1500, ppo_epochs=8,
        use_teacher_policy=True, use_planner=True,
        use_kickstarting=True, use_failure_detector=True,
        use_reflection=True, use_episode_memory=True,
    ),
    "replanning": dict(
        phase=4, total_iterations=2000, ppo_epochs=10,
        use_teacher_policy=True, use_planner=True,
        use_kickstarting=True, use_failure_detector=True,
        use_reflection=True, use_episode_memory=True,
        use_mid_episode_replanning=True,
    ),
    "full": dict(
        phase=5, total_iterations=2000, ppo_epochs=10,
        use_teacher_policy=True, use_planner=True,
        use_kickstarting=True, use_failure_detector=True,
        use_reflection=True, use_episode_memory=True,
        use_state_memory=True, use_mid_episode_replanning=True,
        use_failure_classifier=True,
    ),
}

PHASE_ORDER = ["ppo_only", "planner", "reflection", "replanning", "full"]


def build_experiment_config(
    name: str,
    overrides: Optional[dict] = None,
    phase_configs: Optional[dict] = None,
) -> ExperimentConfig:
    """
    Build an ExperimentConfig by name.

    Parameters
    ----------
    name : str
        One of 'ppo_only', 'planner', 'reflection', 'replanning', 'full'.
    overrides : dict, optional
        Any field overrides applied on top of the phase defaults.
        E.g. overrides={"total_iterations": 500, "batch_size": 64}
    phase_configs : dict, optional
        Per-phase schedule overrides (from the notebook's PHASE_CONFIGS dict).
        Merged with the phase defaults before applying `overrides`.

    Returns
    -------
    ExperimentConfig
    """
    name = name.lower().strip()
    if name not in _PHASE_DEFAULTS:
        raise ValueError(
            f"Unknown experiment name '{name}'. "
            f"Valid names: {list(_PHASE_DEFAULTS.keys())}"
        )

    # Start with phase defaults
    kwargs = dict(_PHASE_DEFAULTS[name])
    kwargs["name"] = name

    # Apply per-phase schedule from notebook (e.g. PHASE_CONFIGS["planner"])
    if phase_configs is not None and name in phase_configs:
        kwargs.update(phase_configs[name])

    # Apply explicit overrides last
    if overrides:
        kwargs.update(overrides)

    return ExperimentConfig(**kwargs)
