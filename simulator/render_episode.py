#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
simulator/render_episode.py — Live visual episode renderer for LLM4Teach

Shows the MiniGrid DoorKey environment in a real-time window with overlay text:
  • Current primitive action (turn left / turn right / forward / pick up / …)
  • Current symbolic plan from the planner (go to <key>, open <door>, …)
  • Step counter, cumulative reward, episode success / failure banner

Usage (standalone)
------------------
    python main.py render --task SimpleDoorKey --loaddir run2 --loadmodel acmodel

Usage (from Python)
-------------------
    from simulator.render_episode import run_render
    run_render(game, n_episodes=3, fps=4, show_window=True)
"""

import os
import sys
import time
import numpy as np
import cv2

from pathlib import Path

# ── Action label map (MiniGrid integer → human-readable) ──────────────────────

ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle / open",
    6: "done",
}


# ── Frame annotation helpers ───────────────────────────────────────────────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS  = 1
_LINE       = cv2.LINE_AA
_PAD        = 6          # pixels of padding for text background box
_LINE_H     = 18         # line height in pixels


def _text_lines(frame: np.ndarray, lines: list[str], origin_y: int,
                fg=(255, 255, 255), bg=(0, 0, 0)) -> None:
    """Draw a stack of text lines starting at origin_y, with a dark background."""
    y = origin_y
    for line in lines:
        if not line:
            y += _LINE_H
            continue
        (w, h), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _THICKNESS)
        # background rect
        cv2.rectangle(frame,
                      (0, y - h - _PAD),
                      (w + 2 * _PAD, y + _PAD),
                      bg, -1)
        cv2.putText(frame, line, (_PAD, y), _FONT, _FONT_SCALE, fg, _THICKNESS, _LINE)
        y += _LINE_H


def annotate_frame(frame: np.ndarray, step: int, action: int,
                   reward: float, ep_reward: float,
                   plan: str = "", success: bool = None) -> np.ndarray:
    """
    Return a copy of *frame* (H×W×3 uint8 BGR) with overlay:
      top-left  → step, primitive action, plan
      bottom    → success / failure banner when episode is done
    """
    out = frame.copy()
    action_label = ACTION_NAMES.get(int(action), f"action={action}")
    plan_short   = (plan[:52] + "…") if len(plan) > 55 else plan

    top_lines = [
        f"Step {step:3d}  |  reward: {ep_reward:+.2f}",
        f"Primitive: {action_label}",
    ]
    if plan_short:
        top_lines.append(f"Plan: {plan_short}")

    _text_lines(out, top_lines, origin_y=_LINE_H)

    # Banner at bottom when episode ends
    if success is not None:
        banner_text = " SUCCESS! " if success else " FAILED "
        banner_color = (0, 200, 0) if success else (0, 0, 200)
        h = out.shape[0]
        (w, bh), _ = cv2.getTextSize(banner_text, _FONT, 0.8, 2)
        x0 = (out.shape[1] - w) // 2
        cv2.rectangle(out, (x0 - 8, h - bh - 20), (x0 + w + 8, h - 4), banner_color, -1)
        cv2.putText(out, banner_text, (x0, h - 8),
                    _FONT, 0.8, (255, 255, 255), 2, _LINE)

    return out


# ── Episode renderer ───────────────────────────────────────────────────────────

def _get_plan_text(teacher_policy) -> str:
    """Extract current symbolic plan string from planner (best-effort)."""
    try:
        planner = teacher_policy.planner
        # planner.dialogue_user holds the last formatted prompt; plan is 2nd line
        raw = getattr(planner, "dialogue_user", "")
        if "\n" in raw:
            return raw.split("\n")[1].strip()
        # Alternative: last cache entry
        cache = getattr(planner, "plans_cache", {})
        if cache:
            return list(cache.values())[-1]
    except Exception:
        pass
    return ""


def render_episode(game, fps: int = 4, show_window: bool = True,
                   save_path: str = None, teacher_policy: bool = False,
                   seed: int = None) -> dict:
    """
    Render one episode with real-time window and/or video save.

    Parameters
    ----------
    game          : Game instance (already initialised, no active training)
    fps           : playback frames per second (for window pacing & video)
    show_window   : display a live cv2 window
    save_path     : if given, save annotated .avi to this path
    teacher_policy: use teacher instead of student policy
    seed          : env seed (random if None)

    Returns dict with ep_return, ep_len, ep_success, frames_saved.
    """
    import torch

    env   = game.env
    dev   = game.device
    seed  = seed if seed is not None else int(np.random.randint(1_000_000))

    obs   = env.reset(seed)
    done  = False
    ep_len   = 0
    ep_reward = 0.0

    if teacher_policy:
        game.teacher_policy.reset()
        mask   = None
        states = None
    else:
        mask   = torch.Tensor([1.]).to(dev)
        states = game.student_policy.model.init_states(dev) if game.recurrent else None

    annotated_frames: list[np.ndarray] = []
    frame_delay_ms = max(1, int(1000 / fps))

    window_name = "LLM4Teach — DoorKey Live View  (press Q to quit)"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)

    last_action = 0
    last_plan   = ""
    success     = None

    with torch.no_grad():
        while not done and ep_len < game.max_ep_len:
            # ── choose action ─────────────────────────────────────────────────
            if teacher_policy:
                probs = game.teacher_policy(obs[0])
                if probs is None:
                    break
                last_action = int(np.argmax(probs))
                last_plan   = _get_plan_text(game.teacher_policy)
            else:
                dist, value, states = game.student_policy(
                    torch.Tensor(obs).to(dev), mask, states
                )
                last_action = int(torch.argmax(dist.probs).item())
                last_plan   = _get_plan_text(game.teacher_policy) \
                              if hasattr(game, "teacher_policy") else ""

            # ── step env ──────────────────────────────────────────────────────
            obs, reward, done, info = env.step(
                np.array([last_action]) if not teacher_policy else last_action
            )
            r = float(reward.squeeze()) if hasattr(reward, "squeeze") else float(reward)
            ep_reward += r
            ep_len    += 1

            # ── get frame ─────────────────────────────────────────────────────
            raw_frame = env.get_mask_render()          # H×W×3  RGB
            bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            ann = annotate_frame(
                bgr_frame, ep_len, last_action, r, ep_reward,
                plan=last_plan,
                success=(True if (done and ep_reward > 0) else
                         False if done else None),
            )
            annotated_frames.append(ann)

            if show_window:
                cv2.imshow(window_name, ann)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord("q") or key == 27:   # Q or ESC → quit early
                    break

    # ── final frame with result banner ────────────────────────────────────────
    if annotated_frames:
        success = ep_reward > 0
        last_ann = annotate_frame(
            annotated_frames[-1], ep_len, last_action, 0.0, ep_reward,
            plan=last_plan, success=success,
        )
        annotated_frames.append(last_ann)
        if show_window:
            cv2.imshow(window_name, last_ann)
            cv2.waitKey(max(frame_delay_ms, 1500))   # hold result banner 1.5 s

    if show_window:
        cv2.destroyWindow(window_name)

    # ── save video ────────────────────────────────────────────────────────────
    frames_saved = 0
    if save_path and annotated_frames:
        h, w = annotated_frames[0].shape[:2]
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            (w, h),
        )
        for f in annotated_frames:
            writer.write(f)
        writer.release()
        frames_saved = len(annotated_frames)
        print(f"[render] Saved {frames_saved} frames → {save_path}")

    return {
        "ep_return":  ep_reward,
        "ep_len":     ep_len,
        "ep_success": int(ep_reward > 0),
        "frames_saved": frames_saved,
    }


# ── Multi-episode runner (called by main.py) ──────────────────────────────────

def run_render(game, n_episodes: int = 3, fps: int = 4,
               show_window: bool = True, save_dir: str = None,
               teacher_policy: bool = False) -> None:
    """
    Render *n_episodes* episodes back-to-back and print a summary.
    Saves annotated .avi files to *save_dir* if provided.
    """
    results = []
    for ep_idx in range(n_episodes):
        seed = ep_idx * 1337 + 42
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            policy_label = "teacher" if teacher_policy else "student"
            save_path = os.path.join(save_dir, f"ep{ep_idx:03d}_{policy_label}.avi")

        print(f"\n{'='*60}")
        print(f"  Episode {ep_idx+1}/{n_episodes}  |  seed={seed}  |  "
              f"policy={'teacher' if teacher_policy else 'student'}")
        print(f"{'='*60}")

        result = render_episode(
            game,
            fps=fps,
            show_window=show_window,
            save_path=save_path,
            teacher_policy=teacher_policy,
            seed=seed,
        )
        results.append(result)
        status = "✅ SUCCESS" if result["ep_success"] else "❌ FAILED"
        print(f"  {status} | return={result['ep_return']:+.3f} | len={result['ep_len']}")

    # Summary
    print(f"\n{'─'*60}")
    print(f"  SUMMARY over {n_episodes} episodes:")
    print(f"    Mean return  : {np.mean([r['ep_return']  for r in results]):+.3f}")
    print(f"    Mean ep-len  : {np.mean([r['ep_len']     for r in results]):.1f}")
    print(f"    Success rate : {np.mean([r['ep_success'] for r in results]):.0%}")
    print(f"{'─'*60}\n")
