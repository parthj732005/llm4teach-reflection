#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
simulator/live_dashboard.py — Live Streamlit training dashboard for LLM4Teach

Launch
------
    streamlit run simulator/live_dashboard.py

Features
--------
- Auto-refreshing reward / success-rate curves (reads TensorBoard logs)
- PPO loss component plots
- Reflection memory size and statistics
- Planner prompt / output log inspection
- Reflection summary browser
- Episode map viewer (last saved video thumbnail)
- Configurable auto-refresh interval
- All settings configurable from the sidebar
"""

import os
import sys
import json
import time
from pathlib import Path

# Make sure the project root is on the path when launched from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import streamlit as st
    _ST = True
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

from simulator.visualize_training import (
    load_tb_scalars,
    plot_reward_curve,
    plot_success_rate,
    plot_ppo_losses,
    plot_reflection_stats,
    plot_reflection_memory,
    plot_training_overview,
    _unzip,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM4Teach Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 LLM4Teach — Live Training Dashboard")
st.caption("Symbolic RL with Qwen reflection-memory. Refresh to update.")


# ── Sidebar settings ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    log_root = st.text_input("Log root directory", value=str(_ROOT / "log"))
    task_choices = ["SimpleDoorKey", "LavaDoorKey", "ColoredDoorKey", "TwoDoor"]
    task = st.selectbox("Task", task_choices, index=0)
    policy = st.selectbox("Policy", ["ppo"], index=0)

    # List available run directories for selected task
    task_log_dir = os.path.join(log_root, policy, task)
    run_dirs = []
    if os.path.isdir(task_log_dir):
        run_dirs = sorted(
            [d for d in os.listdir(task_log_dir)
             if os.path.isdir(os.path.join(task_log_dir, d))],
            reverse=True,
        )
    run_label = st.selectbox("Run", run_dirs if run_dirs else ["(none found)"])
    log_dir = os.path.join(task_log_dir, run_label)

    st.divider()
    st.subheader("Display options")
    smooth_window = st.slider("Reward curve smoothing", 1, 50, 10)
    show_overview = st.checkbox("Show overview panel", value=True)
    show_losses   = st.checkbox("Show PPO losses", value=True)
    show_reflect  = st.checkbox("Show reflection stats", value=True)

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_sec  = st.slider("Refresh interval (s)", 5, 120, 30)
    if auto_refresh:
        st.info(f"Refreshing every {refresh_sec}s")
        time.sleep(refresh_sec)
        st.rerun()

    if st.button("🔄 Refresh now"):
        st.rerun()


# ── Helper: safe figure display ───────────────────────────────────────────────

def show_fig(fig, use_container_width=True):
    if fig is not None:
        st.pyplot(fig, use_container_width=use_container_width)
        plt.close(fig)
    else:
        st.info("No data available yet — start training or select a valid run directory.")


# ── Main content ──────────────────────────────────────────────────────────────

if not os.path.isdir(log_dir):
    st.warning(f"Log directory not found: `{log_dir}`")
    st.info(
        "Start a training run to populate logs:\n\n"
        "```\npython main.py train --task SimpleDoorKey --savedir my_run\n```"
    )
    st.stop()

scalars = load_tb_scalars(log_dir)

# ── 1. Overview ───────────────────────────────────────────────────────────────

if show_overview:
    st.subheader("📊 Training Overview")
    memory_path = os.path.join(log_dir, "reflection_memory.json")
    fig_ov = plot_training_overview(
        log_dir,
        memory_json_path=memory_path if os.path.exists(memory_path) else None,
    )
    show_fig(fig_ov)

st.divider()

# ── 2. Detailed reward / success ──────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Reward Curve")
    fig_r = plot_reward_curve(log_dir, smooth_window=smooth_window)
    show_fig(fig_r)

with col2:
    st.subheader("✅ Success Rate")
    fig_s = plot_success_rate(log_dir, smooth_window=smooth_window)
    show_fig(fig_s)

st.divider()

# ── 3. PPO losses ─────────────────────────────────────────────────────────────

if show_losses:
    st.subheader("📉 PPO Loss Components")
    fig_l = plot_ppo_losses(log_dir)
    show_fig(fig_l)
    st.divider()

# ── 4. Reflection statistics ──────────────────────────────────────────────────

if show_reflect:
    st.subheader("🧠 Reflection System Statistics")

    # TensorBoard reflection metrics
    fig_rs = plot_reflection_stats(log_dir)
    show_fig(fig_rs)

    # JSON memory browser
    memory_path = os.path.join(log_dir, "reflection_memory.json")
    if os.path.exists(memory_path):
        st.subheader("📚 Reflection Memory Browser")
        col_a, col_b = st.columns([2, 1])

        with col_a:
            with open(memory_path, "r", encoding="utf-8") as f:
                mem_data = json.load(f)
            entries = mem_data.get("entries", [])
            st.write(f"**{len(entries)} reflections stored**")

            filter_success = st.checkbox("Show successful episodes only", value=False)
            if filter_success:
                entries = [e for e in entries if e.get("success")]

            for i, entry in enumerate(reversed(entries[:50])):
                label = "✅ SUCCESS" if entry.get("success") else "❌ FAILURE"
                ep_id = entry.get("episode_id", "?")
                reward = entry.get("total_reward")
                reward_str = f" | reward={reward:.2f}" if reward is not None else ""
                with st.expander(f"[Ep {ep_id}] {label}{reward_str}"):
                    st.write(entry.get("reflection", "(no reflection text)"))

        with col_b:
            fig_mem = plot_reflection_memory(memory_path)
            show_fig(fig_mem, use_container_width=True)

        st.divider()
    else:
        st.info(
            "No `reflection_memory.json` found in this run's log directory.\n\n"
            "Enable reflection at train time:\n"
            "```\npython main.py train ... --reflection --reflection_backend ollama\n```"
        )

# ── 5. Bug 4+5 — Planner health debug panel ──────────────────────────────────

st.subheader("🩺 Planner Health Diagnostics")
planner_tags = [k for k in scalars if k.startswith("Planner/")]

if planner_tags:
    p_cols = st.columns(min(len(planner_tags), 4))
    for i, tag in enumerate(sorted(planner_tags)):
        steps, vals = _unzip(scalars[tag])
        if steps:
            latest_val = vals[-1]
            label = tag.replace("Planner/", "")
            p_cols[i % 4].metric(label, f"{latest_val:.3f}", f"step {steps[-1]}")

    # Online vs offline planner state indicator
    online_tag = "Planner/Online Success Rate"
    offline_tag = "Planner/Offline Fallback Rate"
    if online_tag in scalars and offline_tag in scalars:
        _, online_vals  = _unzip(scalars[online_tag])
        _, offline_vals = _unzip(scalars[offline_tag])
        latest_online  = online_vals[-1]  if online_vals  else 0
        latest_offline = offline_vals[-1] if offline_vals else 0
        if latest_online > 0.5:
            st.success(f"✅ Online planner ACTIVE — success rate {latest_online:.1%}")
        elif latest_offline > 0.5:
            st.warning(
                f"⚠️ Planner mostly using OFFLINE fallback ({latest_offline:.1%}). "
                "Check that Ollama is running and inference works."
            )
        else:
            st.info("Planner state: mixed online/offline")

    # Parser rejection rate warning
    rej_tag = "Planner/Parser Rejections"
    if rej_tag in scalars:
        _, rej_vals = _unzip(scalars[rej_tag])
        if rej_vals and rej_vals[-1] > 10:
            st.warning(
                f"⚠️ High parser rejection count: {rej_vals[-1]:.0f}. "
                "Invalid symbolic objects may be contaminating PPO supervision."
            )

    if _MPL:
        rate_tags = [t for t in planner_tags if "Rate" in t]
        if rate_tags:
            fig_ph, ax = plt.subplots(figsize=(10, 3))
            colors = ["green", "orange", "red", "steelblue", "purple"]
            for j, tag in enumerate(sorted(rate_tags)):
                steps, vals = _unzip(scalars[tag])
                ax.plot(steps, vals, label=tag.replace("Planner/", ""), color=colors[j % len(colors)], linewidth=2)
            ax.set_xlabel("Iteration"); ax.set_ylabel("Rate")
            ax.set_ylim(0, 1.05); ax.set_title("Planner Online/Fallback Rates")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            fig_ph.tight_layout()
            show_fig(fig_ph)
else:
    st.info(
        "No planner health metrics found. "
        "These are logged automatically during online training."
    )

st.divider()

# ── 6. Episode replay viewer ──────────────────────────────────────────────────

st.subheader("🎬 Episode Replay Viewer")
st.caption(
    "Browse annotated evaluation videos saved during training. "
    "Each frame shows the **primitive action** and **symbolic plan** taken."
)

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

_video_dirs = {
    "student": os.path.join(log_dir, "video"),
    "teacher": os.path.join(log_dir, "teacher video"),
    "render":  os.path.join(log_dir, "render_output"),
}

_found_videos: list = []
for _vdir in _video_dirs.values():
    if os.path.isdir(_vdir):
        for _f in sorted(os.listdir(_vdir), reverse=True):
            if _f.endswith(".avi"):
                _found_videos.append(os.path.join(_vdir, _f))

if _found_videos and _CV2_OK and _MPL:
    _selected_video = st.selectbox(
        "Select video",
        _found_videos,
        format_func=lambda p: os.path.relpath(p, log_dir),
    )

    _cap = _cv2.VideoCapture(_selected_video)
    _frames: list = []
    while True:
        _ret, _frm = _cap.read()
        if not _ret:
            break
        _frames.append(_cv2.cvtColor(_frm, _cv2.COLOR_BGR2RGB))
    _cap.release()

    if _frames:
        st.write(f"**{len(_frames)} frames** — `{os.path.basename(_selected_video)}`")
        _idx = st.slider("Frame", 0, len(_frames) - 1, 0, key="frame_slider")
        _fig_v, _ax_v = plt.subplots(figsize=(5, 5))
        _ax_v.imshow(_frames[_idx])
        _ax_v.axis("off")
        _fig_v.tight_layout(pad=0)
        st.pyplot(_fig_v, use_container_width=False)
        plt.close(_fig_v)

        if st.button("▶ Auto-play (all frames)"):
            _ph = st.empty()
            for _fi, _fr in enumerate(_frames):
                _ff, _aa = plt.subplots(figsize=(5, 5))
                _aa.imshow(_fr)
                _aa.axis("off")
                _ff.tight_layout(pad=0)
                _ph.pyplot(_ff, use_container_width=False)
                plt.close(_ff)
                time.sleep(0.18)
    else:
        st.warning("Could not decode video file.")
elif not _found_videos:
    st.info(
        "No evaluation videos yet. They are saved automatically every "
        "`eval_interval` iterations, or generate them on demand:\n\n"
        "```\npython main.py render --task SimpleDoorKey "
        "--loaddir run2 --loadmodel acmodel --n_render 3\n```"
    )
else:
    st.info("Install `opencv-python` to preview videos here.")

st.divider()

# ── 7. Raw scalar browser ─────────────────────────────────────────────────────

with st.expander("🔍 Raw scalar data browser"):
    if scalars:
        tag = st.selectbox("Select scalar tag", sorted(scalars.keys()))
        steps, vals = _unzip(scalars[tag])
        if steps and _MPL:
            fig_raw, ax = plt.subplots(figsize=(10, 3))
            ax.plot(steps, vals, linewidth=1.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(tag.split("/")[-1])
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)
            fig_raw.tight_layout()
            show_fig(fig_raw)
        if steps:
            latest_step, latest_val = steps[-1], vals[-1]
            st.metric(f"Latest value ({tag})", f"{latest_val:.4f}", f"step {latest_step}")
    else:
        st.info("No TensorBoard scalars found. Is training running?")

# ── 8. Footer ─────────────────────────────────────────────────────────────────

st.caption(
    f"Log dir: `{log_dir}` | "
    f"Last refreshed: {time.strftime('%H:%M:%S')} | "
    "LLM4Teach reflection-memory system"
)
