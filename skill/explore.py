#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
skill/explore.py — Frontier-based exploration skill with anti-oscillation

Improvements over original:
  - Visited cell tracking with revisit penalty
  - Anti-oscillation: penalises repeated left-right turning
  - Frontier scoring: prefer directions toward cells adjacent to unseen area
  - No-progress detection: terminates if no new cells found for N steps
  - Exposes failure_reason for upstream intervention
"""

import numpy as np
from collections import deque
from .base_skill import BaseSkill, DIRECTION

# Lava object index
OBJ_LAVA = 9


class Explore(BaseSkill):
    """
    Frontier-based exploration skill.

    Parameters
    ----------
    agent_view_size     : MiniGrid agent view size (default 7)
    no_progress_limit   : steps without new unseen cells before giving up
    oscillation_window  : recent action window for oscillation detection
    """

    def __init__(self, agent_view_size,
                 no_progress_limit=25,
                 oscillation_window=8):
        assert agent_view_size >= 3
        self.agent_view_size   = agent_view_size
        self.no_progress_limit = no_progress_limit
        self.message           = "none"
        self.failure_reason    = ""

        # Frontier / progress tracking — persistent across episode
        self._visited            = set()     # (row, col) cells ever visited
        self._global_seen        = set()     # cells that have been observed
        self._prev_unseen_count  = None
        self._no_progress_steps  = 0
        self._total_steps        = 0

        # Anti-oscillation — persistent across episode
        self._action_history     = deque(maxlen=oscillation_window)
        self._oscillation_window = oscillation_window

        # Position history for repeated-cell detection
        self._pos_history        = deque(maxlen=16)

    def reset(self):
        """
        Reset exploration state for a new episode.
        Call this at episode start, NOT between steps.
        The whole point of persistent state is that it survives between steps.
        """
        self._visited.clear()
        self._global_seen.clear()
        self._prev_unseen_count  = None
        self._no_progress_steps  = 0
        self._total_steps        = 0
        self._action_history.clear()
        self._pos_history.clear()
        self.failure_reason      = ""
        self.message             = "none"

    # ── Room boundary ─────────────────────────────────────────────────────────

    def get_room_boundary(self):
        """Find the inner dimensions of the room (wall-bounded area)."""
        width  = self.map.shape[0]
        height = self.map.shape[1]
        self.botX, self.botY = width, height
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.map[i, j] not in (2, 4):
                    pass
                elif self.botX == width and self.map[i, j + 1] in (2, 4):
                    self.botX = i + 1
                elif self.botY == height and self.map[i + 1, j] in (2, 4):
                    self.botY = j + 1
                if self.botX != width and self.botY != height:
                    break

    # ── View slices ───────────────────────────────────────────────────────────

    def get_view(self, agent_dir, agent_pos=None):
        """Return the grid slice the agent can currently see."""
        agent_pos = agent_pos if agent_pos is not None else self.agent_pos
        if agent_dir == 0:
            topX = int(agent_pos[0])
            topY = int(agent_pos[1]) - self.agent_view_size // 2
        elif agent_dir == 1:
            topX = int(agent_pos[0]) - self.agent_view_size // 2
            topY = int(agent_pos[1])
        elif agent_dir == 2:
            topX = int(agent_pos[0]) - self.agent_view_size + 1
            topY = int(agent_pos[1]) - self.agent_view_size // 2
        elif agent_dir == 3:
            topX = int(agent_pos[0]) - self.agent_view_size // 2
            topY = int(agent_pos[1]) - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        topX = max(0, topX)
        topY = max(0, topY)
        botX = min(topX + self.agent_view_size, self.botX)
        botY = min(topY + self.agent_view_size, self.botY)
        return self.map[topX:botX, topY:botY]

    def get_grid_slice(self, agent_dir, agent_pos=None):
        """Return the grid slice NOT currently visible (exploration target)."""
        agent_pos = agent_pos if agent_pos is not None else self.agent_pos
        topX, topY = 0, 0
        botX = self.botX
        botY = self.botY

        if agent_dir == 0:
            topX = int(agent_pos[0]) + self.agent_view_size // 2 + 1
        elif agent_dir == 1:
            topY = int(agent_pos[1]) + self.agent_view_size // 2 + 1
        elif agent_dir == 2:
            botX = int(agent_pos[0]) - self.agent_view_size // 2
        elif agent_dir == 3:
            botY = int(agent_pos[1]) - self.agent_view_size // 2
        else:
            assert False, "invalid agent direction"

        return self.map[topX:botX, topY:botY]

    # ── Obstacle / hazard checks ──────────────────────────────────────────────

    def object_forward(self, agent_dir, agent_pos=None):
        """
        Check what type of obstacle is directly ahead.
        Returns:
          0 = clear path
          1 = wall or locked door
          2 = pickable object (key, ball, box)
          3 = lava
        """
        pos = agent_pos if agent_pos is not None else self.agent_pos
        dx, dy = DIRECTION[agent_dir]
        x, y   = int(pos[0]) + dx, int(pos[1]) + dy
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:
            fwd_obj = self.map[x, y]
            if fwd_obj == OBJ_LAVA:
                return 3
            if fwd_obj in (2, 4):       # wall or door
                return 1
            if fwd_obj in (5, 6, 7):    # key, ball, box
                return 2
        return 0

    def count_unseen_grid(self, agent_dir, agent_pos=None):
        """Count cells with value 0 (unseen) in the slice behind the agent."""
        grid = self.get_grid_slice(agent_dir, agent_pos)
        if grid.size == 0:
            return 0
        return int(np.count_nonzero(grid == 0))

    # ── Oscillation detection ─────────────────────────────────────────────────

    def _is_oscillating(self):
        """
        Return True if the recent action history shows repeated
        left-right turning (action 0 and 1 alternating).
        """
        if len(self._action_history) < self._oscillation_window:
            return False
        actions  = list(self._action_history)
        lr_pairs = sum(
            1 for i in range(len(actions) - 1)
            if (actions[i] == 0 and actions[i + 1] == 1)
            or (actions[i] == 1 and actions[i + 1] == 0)
        )
        return lr_pairs >= self._oscillation_window // 2

    # ── Frontier scoring ──────────────────────────────────────────────────────

    def _score_direction(self, agent_dir):
        """
        Score a candidate direction based on:
          - unseen cells in that direction (primary)
          - lava penalty
          - oscillation penalty (avoid direction that caused oscillation)
        Returns a non-negative score; higher is better.
        """
        obstacle = self.object_forward(agent_dir)
        if obstacle == 1:          # wall/door — can't go this way
            return -1
        if obstacle == 3:          # lava ahead — heavy penalty
            return -2

        unseen = self.count_unseen_grid(agent_dir)

        # Oscillation penalty: penalise turning in the opposite direction
        # of the most recent turn if already oscillating
        turn_penalty = 0
        if len(self._action_history) >= 2:
            last_action = list(self._action_history)[-1]
            if (last_action == 0 and agent_dir == (self.agent_dir + 1) % 4):
                turn_penalty = 1    # just turned left, penalise right turn
            elif (last_action == 1 and agent_dir == (self.agent_dir - 1) % 4):
                turn_penalty = 1

        return unseen - turn_penalty * 0.5

    # ── Main call ─────────────────────────────────────────────────────────────

    def _count_unseen_in_direction(self, candidate_dir) -> int:
        """
        Count truly unseen (value==0) cells in the slice the agent would
        see if facing candidate_dir.  Also subtracts recently-visited
        positions to bias toward unvisited regions.
        """
        grid = self.get_grid_slice(candidate_dir)
        if grid.size == 0:
            return 0
        unseen = int(np.count_nonzero(grid == 0))
        # Revisit penalty: if forward cell in this direction is already visited
        dx, dy = DIRECTION[candidate_dir]
        fwd_r  = int(self.agent_pos[0]) + dx
        fwd_c  = int(self.agent_pos[1]) + dy
        if (fwd_r, fwd_c) in self._visited:
            unseen = max(0, unseen - 2)
        return unseen

    def _repeated_position(self) -> bool:
        """True if the agent has returned to the same cell recently."""
        if len(self._pos_history) < 8:
            return False
        current = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        recent  = list(self._pos_history)[-8:]
        return recent.count(current) >= 3

    def _borders_unseen(self, r, c) -> bool:
        """
        True if cell (r, c) is a frontier edge — i.e. it directly borders an
        unseen (value==0) cell. Used to bias exploration toward frontiers
        (req 7): prioritize unexplored regions rather than wandering inside
        already-seen space.
        """
        w, h = self.map.shape[0], self.map.shape[1]
        for dx, dy in DIRECTION.values():
            nr, nc = r + dx, c + dy
            if 0 <= nr < w and 0 <= nc < h and self.map[nr, nc] == 0:
                return True
        return False

    def __call__(self, obs):
        self.unpack_obs(obs)
        self.get_room_boundary()
        self._total_steps += 1

        current_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        self._visited.add(current_pos)
        self._pos_history.append(current_pos)

        # ── Track unseen cells for progress ───────────────────────────────────
        total_unseen = int(np.count_nonzero(self.map == 0))

        if self._prev_unseen_count is not None:
            if total_unseen < self._prev_unseen_count:
                # New cells discovered — reset counter
                self._no_progress_steps = 0
            else:
                self._no_progress_steps += 1
        self._prev_unseen_count = total_unseen

        # No-progress termination — expose failure_reason for intervention
        if self._no_progress_steps >= self.no_progress_limit:
            self.failure_reason = (
                f"explore_no_progress:{self._no_progress_steps}_steps"
            )
            return None, True

        # ── All cells explored ─────────────────────────────────────────────────
        if total_unseen == 0:
            self.message = "no unseen grid"
            return None, True

        # ── Break oscillation first ────────────────────────────────────────────
        if self._is_oscillating() or self._repeated_position():
            # Force a forward move to break the cycle
            fwd = self.object_forward(self.agent_dir)
            if fwd == 0:
                action = 2
                self._action_history.append(action)
                return action, False
            # Can't go forward — pick the direction with most unseen cells
            best_escape = max(
                range(4),
                key=lambda d: (
                    self._count_unseen_in_direction(d)
                    if self.object_forward(d) == 0 else -1
                )
            )
            diff = (best_escape - self.agent_dir) % 4
            action = 2 if diff == 0 else (1 if diff == 1 else 0)
            self._action_history.append(action)
            return action, False

        # ── Avoid pickable object directly ahead ──────────────────────────────
        if self.object_forward(self.agent_dir) == 2:
            action = 1 if self.object_forward((self.agent_dir - 1) % 4) in (1, 2) else 0
            self._action_history.append(action)
            return action, False

        # ── Score all 4 candidate directions ──────────────────────────────────
        scores = {}
        for candidate_dir in range(4):
            obs_type = self.object_forward(candidate_dir)
            if obs_type in (1, 3):          # wall or lava — skip
                continue
            unseen = self._count_unseen_in_direction(candidate_dir)
            # Turn cost: prefer directions requiring fewer turns
            turn_cost = min(
                (candidate_dir - self.agent_dir) % 4,
                (self.agent_dir - candidate_dir) % 4,
            )
            # Oscillation penalty
            turn_penalty = 0
            if len(self._action_history) >= 2:
                last = list(self._action_history)[-1]
                if (last == 0 and candidate_dir == (self.agent_dir + 1) % 4) or \
                   (last == 1 and candidate_dir == (self.agent_dir - 1) % 4):
                    turn_penalty = 1
            # Frontier bonus (req 7): prefer stepping onto a frontier edge — a
            # cell that itself borders unseen space — so exploration pushes into
            # unexplored regions instead of re-covering seen area.
            fwd_r = int(self.agent_pos[0]) + DIRECTION[candidate_dir][0]
            fwd_c = int(self.agent_pos[1]) + DIRECTION[candidate_dir][1]
            frontier_bonus = 0.5 if self._borders_unseen(fwd_r, fwd_c) else 0.0
            scores[candidate_dir] = (
                unseen - turn_cost * 0.1 - turn_penalty * 0.5 + frontier_bonus
            )

        if not scores:
            self.failure_reason = "explore_all_blocked"
            return None, True

        best_dir = max(scores, key=scores.__getitem__)

        # ── Convert best direction to primitive action ─────────────────────────
        diff = (best_dir - self.agent_dir) % 4
        if diff == 0:
            action = 2   # FORWARD
        elif diff == 1:
            action = 1   # turn RIGHT
        elif diff == 3:
            action = 0   # turn LEFT
        else:
            action = 0   # 180° — start turning left

        self._action_history.append(action)
        return action, False
