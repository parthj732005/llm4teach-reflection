#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
skill/goto_goal.py — A* navigation skill with grounded verification

Improvements over original:
  - Lava-aware navigation (configurable per task)
  - Stuck detection: if position unchanged for N steps → terminate
  - Oscillation detection: repeated left-right → terminate
  - Repeated-state detection: same (pos, dir) revisited too often → terminate
  - Exposes failure_reason for upstream intervention logic
"""

import numpy as np
from collections import deque
from .base_skill import BaseSkill

DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

# Tasks where lava should block navigation
LAVA_AWARE_TASKS = {"lavadoorkey"}


def check_go_through(pos, maps, avoid_lava=False):
    """
    Return True if the agent can walk onto cell (x, y).

    avoid_lava=True  → lava (type 9) is treated as impassable wall.
    avoid_lava=False → lava is traversable (original behaviour for non-lava tasks).
    """
    x, y = pos
    width, height, _ = maps.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    obj = maps[x, y, 0]
    if obj == 9 and avoid_lava:     # lava — blocked in lava environments
        return False
    return (
        obj in (1, 8)               # empty or goal
        or obj == 9                 # lava (only if avoid_lava=False)
        or (obj == 4 and maps[x, y, 2] == 0)   # open door
    )


def get_neighbors(pos_and_dir, maps, avoid_lava=False):
    """
    Return all reachable next states from (x, y, direction).
    Each state costs 1 (turning and stepping both cost 1).
    """
    x, y, direction = pos_and_dir
    next_dir_left  = direction - 1 if direction > 0 else 3
    next_dir_right = direction + 1 if direction < 3 else 0
    neighbor_list  = [(x, y, next_dir_left), (x, y, next_dir_right)]

    forward_x, forward_y = DIRECTION[direction]
    new_x, new_y = x + forward_x, y + forward_y
    if check_go_through((new_x, new_y), maps, avoid_lava=avoid_lava):
        neighbor_list.append((new_x, new_y, direction))

    return neighbor_list


class GoTo_Goal(BaseSkill):
    """
    Navigate to the cell adjacent to target_pos and face it.

    Parameters
    ----------
    target_pos  : (row, col) target grid position
    avoid_lava  : if True, treat lava cells as impassable (use for LavaDoorKey)
    stuck_threshold : number of recent positions to track for stuck detection
    """

    def __init__(self, target_pos, avoid_lava=False, stuck_threshold=10):
        self.target_pos     = target_pos
        self.avoid_lava     = avoid_lava
        self.message        = "none"
        self.failure_reason = ""

        # Stuck / oscillation detection
        self._stuck_threshold = stuck_threshold
        self._pos_history     = deque(maxlen=stuck_threshold)
        self._state_history   = deque(maxlen=stuck_threshold)   # (pos, dir)

    def plan(self):
        """
        Run A* from current agent state to any of 4 goal states
        (adjacent to target, facing it).

        Returns list of (x, y, dir) nodes if path found, else None.
        """
        start_node = (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(self.agent_dir),
        )
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        goal_states = [
            (tx - 1, ty,     0),   # approach from left,  facing right
            (tx,     ty - 1, 1),   # approach from above, facing down
            (tx + 1, ty,     2),   # approach from right, facing left
            (tx,     ty + 1, 3),   # approach from below, facing up
        ]

        open_list   = {start_node}
        closed_list = set()
        g           = {start_node: 0}
        parents     = {start_node: start_node}

        while open_list:
            # Pick cheapest node
            n = min(open_list, key=lambda v: g[v])

            if n in goal_states:
                # Reconstruct path
                path = []
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.append(start_node)
                path.reverse()
                return path

            for m in get_neighbors(n, self.obs, avoid_lava=self.avoid_lava):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + 1
                elif g.get(m, 1e9) > g[n] + 1:
                    g[m] = g[n] + 1
                    parents[m] = n
                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        self.message = "no path found"
        return None

    def _check_stuck(self):
        """
        Return True if agent is stuck (no position change) or
        oscillating (only 2 distinct positions in history).
        """
        if len(self._pos_history) < self._stuck_threshold:
            return False
        unique_positions = set(self._pos_history)
        return len(unique_positions) <= 2

    def _check_oscillating(self):
        """
        Return True if agent is spinning in place
        (same (pos, dir) repeatedly).
        """
        if len(self._state_history) < self._stuck_threshold:
            return False
        unique_states = set(self._state_history)
        return len(unique_states) <= 3

    def __call__(self, obs):
        self.unpack_obs(obs)

        # ── Stuck / oscillation detection ────────────────────────────────────
        current_pos   = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        current_state = (current_pos, int(self.agent_dir))
        self._pos_history.append(current_pos)
        self._state_history.append(current_state)

        if self._check_stuck():
            self.failure_reason = (
                f"GoTo: stuck/oscillating at {current_pos} "
                f"(pos repeated in last {self._stuck_threshold} steps)"
            )
            return None, True   # signal failure → teacher can re-plan

        if self._check_oscillating():
            self.failure_reason = (
                f"GoTo: direction oscillation detected at {current_pos}"
            )
            return None, True

        # ── A* pathfinding ────────────────────────────────────────────────────
        path = self.plan()

        if path is None:
            self.failure_reason = f"GoTo: no path to target {self.target_pos}"
            return None, False  # not terminated — parent skill may retry

        if len(path) == 1:
            return None, True   # already at goal

        # Compute next action from first two path nodes
        cur_dir  = int(path[0][2])
        next_dir = int(path[1][2])
        angle    = (cur_dir - next_dir) % 4

        if angle == 1:
            action = 0   # turn LEFT
        elif angle == 3:
            action = 1   # turn RIGHT
        elif angle == 0:
            action = 2   # move FORWARD
        else:
            self.failure_reason = "GoTo: invalid path step (double-turn)"
            return None, True

        return action, False
