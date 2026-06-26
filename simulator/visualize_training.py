#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
simulator/visualize_training.py — Matplotlib visualization utilities for LLM4Teach

Provides reusable plotting functions for:
  - Reward / episode-length convergence curves
  - Success rate over training iterations
  - Reflection memory statistics
  - Planner output distributions
  - PPO loss components

These functions read TensorBoard event files from the log/ directory and
return matplotlib Figure objects for use in notebooks or the dashboard.

Usage
-----
from simulator.visualize_training import (
    load_tb_scalars,
    plot_reward_curve,
    plot_success_rate,
    plot_reflection_stats,
    plot_ppo_losses,
)

fig = plot_reward_curve("log/ppo/SimpleDoorKey/train-0")
fig.savefig("reward_curve.png")
"""

import os
import json
from typing import Optional, Dict, List, Tuple, Any

try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend safe for notebooks and server
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False
    plt = None

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


# ── TensorBoard reader ────────────────────────────────────────────────────────

def load_tb_scalars(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load scalar data from a TensorBoard event file.

    Returns a dict: tag → list of (step, value) tuples.
    Returns empty dict if tensorboard is not installed or no events found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    if not os.path.isdir(log_dir):
        return {}

    ea = EventAccumulator(log_dir)
    ea.Reload()
    result: Dict[str, List[Tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]
    return result


def _unzip(pairs: List[Tuple[int, float]]):
    """Unzip [(step, val), ...] → (steps_list, vals_list)."""
    if not pairs:
        return [], []
    steps, vals = zip(*pairs)
    return list(steps), list(vals)


def _smooth(values: List[float], window: int = 10) -> List[float]:
    """Simple moving average smoothing."""
    if not _NP or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same").tolist()


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_reward_curve(
    log_dir: str,
    smooth_window: int = 10,
    title: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """
    Plot training and evaluation reward curves from a TensorBoard log directory.
    Returns a matplotlib Figure or None if matplotlib/data is unavailable.
    """
    if not _MPL:
        return None

    scalars = load_tb_scalars(log_dir)
    fig, ax = plt.subplots(figsize=(10, 4))

    for tag, label, color in [
        ("Train/Return Mean", "Train reward (mean)", "steelblue"),
        ("Test/Return",       "Eval reward",          "darkorange"),
    ]:
        if tag in scalars:
            steps, vals = _unzip(scalars[tag])
            smoothed = _smooth(vals, smooth_window)
            ax.plot(steps, vals, alpha=0.3, color=color)
            ax.plot(steps, smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.set_title(title or f"Reward Curve — {os.path.basename(log_dir)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_success_rate(
    log_dir: str,
    smooth_window: int = 10,
    title: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """Plot training and evaluation success rates."""
    if not _MPL:
        return None

    scalars = load_tb_scalars(log_dir)
    fig, ax = plt.subplots(figsize=(10, 4))

    for tag, label, color in [
        ("Train/Success Rate", "Train success rate", "steelblue"),
        ("Test/Success Rate",  "Eval success rate",  "darkorange"),
    ]:
        if tag in scalars:
            steps, vals = _unzip(scalars[tag])
            smoothed = _smooth(vals, smooth_window)
            ax.plot(steps, vals, alpha=0.3, color=color)
            ax.plot(steps, smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Success Rate — {os.path.basename(log_dir)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ppo_losses(
    log_dir: str,
    title: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """Plot PPO loss components (total, policy, value, entropy, kickstarting)."""
    if not _MPL:
        return None

    scalars = load_tb_scalars(log_dir)
    tags = {
        "Train/Loss":              ("Total loss",            "black"),
        "Train/Policy Loss":       ("Policy loss",           "steelblue"),
        "Train/Value Loss":        ("Value loss",            "darkorange"),
        "Train/Mean Entropy":      ("Entropy",               "green"),
        "Train/Kickstarting Loss": ("Kickstarting loss",     "red"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    available = [k for k in tags if k in scalars]
    for i, tag in enumerate(available[:6]):
        ax = axes[i]
        steps, vals = _unzip(scalars[tag])
        label, color = tags[tag]
        ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title or f"PPO Losses — {os.path.basename(log_dir)}", fontsize=13)
    fig.tight_layout()
    return fig


def plot_reflection_stats(
    log_dir: str,
    title: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """
    Plot reflection system metrics from TensorBoard logs.
    Shows memory size, generated/validated/rejected counts over training.
    """
    if not _MPL:
        return None

    scalars = load_tb_scalars(log_dir)
    reflection_tags = [k for k in scalars if k.startswith("Reflection/")]

    if not reflection_tags:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No reflection data found in logs",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Reflection Statistics")
        return fig

    n = len(reflection_tags)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ["steelblue", "green", "red", "darkorange", "purple", "grey"]
    for i, tag in enumerate(sorted(reflection_tags)):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        steps, vals = _unzip(scalars[tag])
        ax.plot(steps, vals, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_title(tag.replace("Reflection/", ""))
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title or "Reflection System Statistics", fontsize=13)
    fig.tight_layout()
    return fig


def plot_reflection_memory(
    memory_json_path: str,
    title: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """
    Visualize the contents of a saved reflection_memory.json file.
    Shows success/failure breakdown and episode reward distribution.
    """
    if not _MPL or not os.path.exists(memory_json_path):
        return None

    with open(memory_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Memory buffer is empty",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    successes = sum(1 for e in entries if e.get("success"))
    failures  = len(entries) - successes
    rewards   = [e.get("total_reward", 0) for e in entries if e.get("total_reward") is not None]
    ep_lens   = [e.get("ep_len", 0) for e in entries if e.get("ep_len") is not None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Success / failure pie
    axes[0].pie(
        [successes, failures],
        labels=["Success", "Failure"],
        colors=["#4CAF50", "#F44336"],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[0].set_title("Episode Outcomes in Memory")

    # Reward distribution
    if rewards and _NP:
        axes[1].hist(rewards, bins=15, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("Total Reward")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Reward Distribution")
        axes[1].grid(True, alpha=0.3)

    # Episode length distribution
    if ep_lens and _NP:
        axes[2].hist(ep_lens, bins=15, color="darkorange", edgecolor="white")
        axes[2].set_xlabel("Episode Length")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Episode Length Distribution")
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title or f"Reflection Memory — {len(entries)} entries", fontsize=13)
    fig.tight_layout()
    return fig


def plot_training_overview(
    log_dir: str,
    memory_json_path: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """
    Combined overview: reward curve + success rate + reflection memory size.
    Returns a single wide figure suitable for dashboards.
    """
    if not _MPL:
        return None

    scalars = load_tb_scalars(log_dir)
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── panel 1: reward ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for tag, label, color in [
        ("Train/Return Mean", "Train", "steelblue"),
        ("Test/Return",       "Eval",  "darkorange"),
    ]:
        if tag in scalars:
            steps, vals = _unzip(scalars[tag])
            ax1.plot(steps, _smooth(vals), label=label, color=color, linewidth=2)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Return")
    ax1.set_title("Reward Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── panel 2: success rate ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for tag, label, color in [
        ("Train/Success Rate", "Train", "steelblue"),
        ("Test/Success Rate",  "Eval",  "darkorange"),
    ]:
        if tag in scalars:
            steps, vals = _unzip(scalars[tag])
            ax2.plot(steps, _smooth(vals), label=label, color=color, linewidth=2)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.05); ax2.set_title("Success Rate")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── panel 3: reflection memory growth ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if "Reflection/Memory Size" in scalars:
        steps, vals = _unzip(scalars["Reflection/Memory Size"])
        ax3.plot(steps, vals, color="purple", linewidth=2)
        ax3.set_xlabel("Iteration"); ax3.set_ylabel("# Reflections Stored")
        ax3.set_title("Reflection Memory Growth"); ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Reflection disabled or\nno data yet",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_title("Reflection Memory")

    fig.suptitle(f"Training Overview — {os.path.basename(log_dir)}", fontsize=14)
    return fig
