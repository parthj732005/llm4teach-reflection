#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
skill/base_skill.py — Grounded closed-loop primitive skill controllers

Every skill verifies pre-conditions before acting and post-conditions
after acting.  They are stateful: called once per timestep, returning
ONE primitive action each time until the skill terminates or fails.

Return convention: (action, terminated)
  action     : int primitive action index, or None if nothing to do
  terminated : True  → skill is done (success OR failure)
               False → skill still running, call again next timestep
"""

import numpy as np

# Direction vectors: agent_dir → (dx, dy)
DIRECTION = {
    0: [1, 0],   # East  / right
    1: [0, 1],   # South / down
    2: [-1, 0],  # West  / left
    3: [0, -1],  # North / up
}

# MiniGrid object type indices
OBJ_EMPTY  = 1
OBJ_WALL   = 2
OBJ_FLOOR  = 3
OBJ_DOOR   = 4
OBJ_KEY    = 5
OBJ_BALL   = 6
OBJ_BOX    = 7
OBJ_GOAL   = 8
OBJ_LAVA   = 9
OBJ_AGENT  = 10

# MiniGrid door state indices
DOOR_OPEN   = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2

PICKABLE_OBJECTS = (OBJ_KEY, OBJ_BALL, OBJ_BOX)
CLEAR_CELLS      = (OBJ_EMPTY, OBJ_FLOOR, OBJ_GOAL)


class BaseSkill:
    """Shared observation unpacking for all skill controllers."""

    def __init__(self):
        pass

    def unpack_obs(self, obs):
        """
        Decode raw observation array into agent state variables.

        Sets:
            self.obs        — raw (H, W, C) observation (last 4 channels)
            self.agent_pos  — np.array [row, col]
            self.agent_dir  — int 0-3
            self.map        — (H, W) object-type grid, agent cell set to 10
            self.carrying   — object type the agent is holding (1=nothing)
        """
        if len(obs.shape) == 4:
            obs = obs[0, :, :, -4:]
        self.obs       = obs
        agent_map      = obs[:, :, 3]
        self.agent_pos = np.argwhere(agent_map != 4)[0]
        self.agent_dir = int(obs[self.agent_pos[0], self.agent_pos[1], 3])
        self.map       = obs[:, :, 0].copy()
        # Object at agent's cell encodes what is being carried
        self.carrying  = int(self.map[self.agent_pos[0], self.agent_pos[1]])
        self.map[self.agent_pos[0], self.agent_pos[1]] = OBJ_AGENT

    def _fwd_pos(self):
        """Return (row, col) of the cell directly in front of the agent."""
        dx, dy = DIRECTION[self.agent_dir]
        return int(self.agent_pos[0] + dx), int(self.agent_pos[1] + dy)

    def _cell(self, r, c):
        """
        Return (obj_type, obj_color, obj_state) at (r, c).
        Returns (-1,-1,-1) if out of bounds.
        """
        h, w = self.obs.shape[:2]
        if 0 <= r < h and 0 <= c < w:
            return (int(self.obs[r, c, 0]),
                    int(self.obs[r, c, 1]),
                    int(self.obs[r, c, 2]))
        return (-1, -1, -1)

    def _fwd_cell(self):
        """Return (obj_type, obj_color, obj_state) of the cell in front."""
        r, c = self._fwd_pos()
        return self._cell(r, c)

    def _is_carrying(self):
        return self.carrying in PICKABLE_OBJECTS


# ─────────────────────────────────────────────────────────────────────────────
# Grounded Pickup
# ─────────────────────────────────────────────────────────────────────────────

class Pickup(BaseSkill):
    """
    Grounded pickup controller with in-episode alignment recovery (req 14).

    Skill instances are EPHEMERAL — teacher_policy.skill2teacher() recreates the
    skill every timestep — so this controller is fully STATELESS: it decides the
    single best primitive action from the CURRENT observation each call. No
    cross-step or cross-episode state is kept; recovery uses only what is visible
    in the current observation.

    Decision order each call:
      1. Already carrying a key            → success.
      2. Key directly in front (adjacent)  → PICKUP.
      3. Key visible but not adjacent      → take one step toward it (align via
                                             reused GoTo_Goal A*), retry next step.
      4. Pickup mis-aligned / no path      → set PICKUP_ALIGNMENT_FAILURE and
                                             soft-fail so the planner / FailureDetector
                                             pipeline takes over.
      5. No key visible at all             → soft-fail (planner re-plans / explores).

    Parameters
    ----------
    avoid_lava : treat lava as impassable when navigating to the key
                 (pass True for LavaDoorKey).
    """

    def __init__(self, target_obj=None, avoid_lava=False):
        self.avoid_lava     = avoid_lava
        self.success        = False
        self.failure_reason = ""

    def _find_nearest_key(self):
        """
        Return (row, col) of the nearest on-floor key in the current observation,
        or None if no key is visible. Uses only the current obs (no memory of
        previous episodes). The agent's own cell is encoded as OBJ_AGENT by
        unpack_obs(), so a carried key is never matched here.

        For ColoredDoorKey (multiple keys) this returns the nearest key; the
        planner's GoTo already routes to the correct colored key, so recovery
        only needs a robust fallback.
        """
        keys = np.argwhere(self.map == OBJ_KEY)
        if len(keys) == 0:
            return None
        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        dists  = [abs(int(k[0]) - ar) + abs(int(k[1]) - ac) for k in keys]
        return tuple(int(v) for v in keys[int(np.argmin(dists))])

    def __call__(self, obs):
        self.unpack_obs(obs)

        # 1. Already holding the key → done.
        if self._is_carrying():
            self.success = True
            return None, True

        obj_type, _, _ = self._fwd_cell()

        # 2. Key directly in front (adjacent + facing) → pick it up.
        if obj_type in PICKABLE_OBJECTS:
            return 3, False

        # 3. Recovery: key not adjacent. Navigate toward the nearest visible key,
        #    aligning to be adjacent and facing it, then retry pickup next step.
        key_pos = self._find_nearest_key()
        if key_pos is None:
            # 5. No key visible — soft fail so the planner re-plans / explores.
            self.failure_reason = "Pickup: no key visible to approach."
            return None, False

        from .goto_goal import GoTo_Goal      # lazy import avoids circular import
        nav = GoTo_Goal(key_pos, avoid_lava=self.avoid_lava)
        action, _ = nav(obs)
        if action is not None:
            return action, False             # step toward / align to the key

        # 4. Could not align (no path or stuck) → alignment failure. Surfaces to
        #    FailureDetector.report_failed_interaction() (limited retries then the
        #    normal failure-handling pipeline escalates).
        self.failure_reason = "PICKUP_ALIGNMENT_FAILURE: could not align to key."
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Grounded Drop
# ─────────────────────────────────────────────────────────────────────────────

class Drop(BaseSkill):
    """
    Grounded drop controller.

    Verifies agent is carrying something before dropping,
    then confirms the carried object is gone after.
    """

    MAX_RETRIES = 2

    def __init__(self, target_obj=None):
        self._state   = "check"
        self._retries = 0
        self.success  = False
        self.failure_reason = ""

    def __call__(self, obs):
        self.unpack_obs(obs)

        if self._state == "check":
            if not self._is_carrying():
                self.failure_reason = "Drop: not carrying anything"
                return None, True
            self._state = "verify"
            return 4, False              # DROP

        elif self._state == "verify":
            if not self._is_carrying():
                self.success = True
                return None, True        # confirmed drop
            self._retries += 1
            if self._retries >= self.MAX_RETRIES:
                self.failure_reason = (
                    f"Drop: still carrying after {self._retries} attempts"
                )
                return None, True
            self._state = "check"
            return 4, False

        return None, True


# ─────────────────────────────────────────────────────────────────────────────
# Grounded Toggle (open/close door)
# ─────────────────────────────────────────────────────────────────────────────

class Toggle(BaseSkill):
    """
    Grounded toggle (door) controller (req 8).

    Skill instances are EPHEMERAL (recreated each timestep), so this controller
    is STATELESS — it validates the open pre-conditions from the current
    observation and issues a single TOGGLE only when ALL hold:

      * adjacency      — a door is in the cell directly in front of the agent
      * correct key    — for a LOCKED door, the agent is carrying a key whose
                         color matches the door (for single-color envs the key
                         always matches, so the check is a no-op; for
                         ColoredDoorKey a mismatched key is rejected)
      * valid action   — the door is not already open

    On a successful open the environment terminates the episode, so a single
    issued TOGGLE is sufficient. Any unmet pre-condition terminates the skill
    with failure_reason set (feeding the FailureDetector pipeline) instead of
    wasting a toggle.

    Exposes:
        self.success        — True if the door is already open / opens
        self.failure_reason — set when a pre-condition is not met
    """

    def __init__(self, target_obj=None):
        self.success        = False
        self.failure_reason = ""

    def _find_nearest_door(self):
        """Return (row, col) of the nearest closed/locked door, or None."""
        doors = np.argwhere(self.map == OBJ_DOOR)
        if len(doors) == 0:
            return None
        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        # Prefer closed/locked doors (state 1 or 2); skip already-open ones.
        candidates = [
            d for d in doors
            if int(self.obs[int(d[0]), int(d[1]), 2]) in (DOOR_CLOSED, DOOR_LOCKED)
        ]
        if not candidates:
            return None
        dists = [abs(int(d[0]) - ar) + abs(int(d[1]) - ac) for d in candidates]
        return tuple(int(v) for v in candidates[int(np.argmin(dists))])

    def __call__(self, obs):
        self.unpack_obs(obs)
        obj_type, door_color, state = self._fwd_cell()

        # Adjacency + facing: door must be directly in front.
        # If not, navigate to be adjacent and facing the door first.
        if obj_type != OBJ_DOOR:
            door_pos = self._find_nearest_door()
            if door_pos is not None:
                from .goto_goal import GoTo_Goal
                nav = GoTo_Goal(door_pos)
                action, _ = nav(obs)
                if action is not None:
                    return action, False   # move toward door, retry next step
            self.failure_reason = (
                f"Toggle: no door in front (cell type={obj_type}) "
                "and no reachable door found."
            )
            return None, True

        # Already open → the interaction is effectively complete.
        if state == DOOR_OPEN:
            self.success = True
            return None, True

        # Correct key: a LOCKED door only opens with the matching key in hand.
        if state == DOOR_LOCKED:
            if not self._is_carrying():
                self.failure_reason = (
                    "Toggle: locked door in front but agent holds no key."
                )
                return None, True
            carried_color = int(self.obs[self.agent_pos[0], self.agent_pos[1], 1])
            if door_color != carried_color:
                self.failure_reason = (
                    f"Toggle: held key color ({carried_color}) does not match "
                    f"locked door color ({door_color})."
                )
                return None, True

        # All pre-conditions satisfied → open the door.
        return 5, False


# ─────────────────────────────────────────────────────────────────────────────
# Wait (no-op)
# ─────────────────────────────────────────────────────────────────────────────

class Wait(BaseSkill):
    """No-operation skill. Used when mediator parser cannot identify an action."""

    def __init__(self, target_obj=None):
        pass

    def __call__(self, obs):
        return 6, True
