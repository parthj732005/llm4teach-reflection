#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mediator.py
@Time    :   2023/05/16 10:22:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import re
import copy
from abc import ABC, abstractmethod
def get_minigrid_words():
    colors = ["red", "green", "blue", "yellow", "purple", "grey"]
    objects = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "box",
        "key",
        "ball",
        "door",
        "goal",
        "agent",
        "lava",
    ]

    verbs = [
        "pick",
        "avoid",
        "get",
        "find",
        "put",
        "use",
        "open",
        "go",
        "fetch",
        "reach",
        "unlock",
        "traverse",
    ]

    extra_words = [
        "up",
        "the",
        "a",
        "at",
        ",",
        "square",
        "and",
        "then",
        "to",
        "of",
        "rooms",
        "near",
        "opening",
        "must",
        "you",
        "matching",
        "end",
        "hallway",
        "object",
        "from",
        "room",
    ]

    all_words = colors + objects + verbs + extra_words
    assert len(all_words) == len(set(all_words))
    return {word: i for i, word in enumerate(all_words)}

# Map of agent direction, 0: East; 1: South; 2: West; 3: North
DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
} 
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

# Map of skill names to integers
SKILL_TO_IDX = {"explore": 0, "go to object": 1, "pickup": 2, "drop": 3, "toggle": 4}
IDX_TO_SKILL = dict(zip(SKILL_TO_IDX.values(), SKILL_TO_IDX.keys()))




def _relative_direction(agent_dir, agent_pos, obj_pos):
    """
    Return a compact relational label for an object's direction
    relative to the agent.

    Returns one of: "ahead", "left", "right", "behind".

    agent_dir: 0=East 1=South 2=West 3=North
    """
    dr = int(obj_pos[0]) - int(agent_pos[0])
    dc = int(obj_pos[1]) - int(agent_pos[1])

    # Rotate (dr, dc) into agent-relative frame
    # agent_dir 0 (East):  forward=+r, right=-c, left=+c
    # agent_dir 1 (South): forward=+c, right=+r, left=-r
    # agent_dir 2 (West):  forward=-r, right=+c, left=-c
    # agent_dir 3 (North): forward=-c, right=-r, left=+r
    ad = int(agent_dir)
    if   ad == 0: fwd, lat = dr,  -dc
    elif ad == 1: fwd, lat = dc,   dr
    elif ad == 2: fwd, lat = -dr,  dc
    else:         fwd, lat = -dc, -dr

    if abs(fwd) >= abs(lat):
        return "ahead" if fwd >= 0 else "behind"
    return "right" if lat > 0 else "left"


class Base_Mediator(ABC):
    """The base class for Base_Mediator."""

    def __init__(self, soft):
        super().__init__()
        self.soft = soft
        self.obj_coordinate = {}
        # Optional: recent failure context injected by TeacherPolicy
        self._failure_context: str = ""

    def set_failure_context(self, context: str) -> None:
        """
        Inject a recent-failure string that will be appended to the
        rich description.  Called by TeacherPolicy on intervention.
        """
        self._failure_context = context

    def clear_failure_context(self) -> None:
        self._failure_context = ""

    # ── Core RL → language translation (UNCHANGED — keeps cache keys stable) ──

    def RL2LLM(self, obs, color_info=True):
        """
        Standard observation-to-text conversion.

        Output format is KEPT IDENTICAL to the original so that
        planner cache keys are stable across versions.
        """
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map  = obs[:, :, 3]
        agent_pos  = np.argwhere(agent_map != 4)[0]
        agent_dir  = agent_map[agent_pos[0], agent_pos[1]]

        key_list  = np.argwhere(obs_object == 5)
        door_list = np.argwhere(obs_object == 4)

        carrying = "nothing"
        if len(key_list):
            for key in key_list:
                i, j = key
                if color_info:
                    color = obs[i, j, 1]
                    obj   = f"{IDX_TO_COLOR[color]} key"
                else:
                    obj = "key"
                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, "
                    self.obj_coordinate[obj] = (i, j)

        if len(door_list):
            for door in door_list:
                i, j = door
                if color_info:
                    color = obs[i, j, 1]
                    obj   = f"{IDX_TO_COLOR[color]} door"
                else:
                    obj = "door"
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i, j)

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        context = f"Agent sees {context}"
        return context

    # ── Rich relational description (used for LLM prompts only) ──────────────

    def RL2LLM_rich(self, obs, color_info=True, include_lava=False):
        """
        Enhanced grounded description with directional relations.

        Examples:
          "Door ahead. Key to your left. Holding nothing.
           Lava to your right. North unexplored."

        Used in LLM prompts (NOT as cache key).
        DO NOT call this for planner.plan() cache lookup.
        """
        if len(obs.shape) == 4:
            obs = obs[0, :, :, -4:]
        obs_object = copy.deepcopy(obs[:, :, 0])
        agent_map  = obs[:, :, 3]
        agent_pos  = np.argwhere(agent_map != 4)[0]
        agent_dir  = int(agent_map[agent_pos[0], agent_pos[1]])

        key_list  = np.argwhere(obs_object == 5)
        door_list = np.argwhere(obs_object == 4)
        lava_list = np.argwhere(obs_object == 9) if include_lava else []

        parts    = []
        carrying = "nothing"

        # Keys
        for key in key_list:
            i, j = key
            color_name = IDX_TO_COLOR[obs[i, j, 1]] if color_info else ""
            label      = f"{color_name} key".strip()
            if (agent_pos == key).all():
                carrying = label
            else:
                rel = _relative_direction(agent_dir, agent_pos, key)
                parts.append(f"Key {rel}")
                self.obj_coordinate[label] = (i, j)

        # Doors
        for door in door_list:
            i, j       = door
            color_name = IDX_TO_COLOR[obs[i, j, 1]] if color_info else ""
            label      = f"{color_name} door".strip()
            door_state = obs[i, j, 2]
            state_str  = {0: "open", 1: "closed", 2: "locked"}.get(int(door_state), "")
            rel        = _relative_direction(agent_dir, agent_pos, door)
            parts.append(f"{label.capitalize()} {rel} ({state_str})")
            self.obj_coordinate[label] = (i, j)

        # Lava (only for lava environments)
        if include_lava and len(lava_list):
            nearest = min(lava_list,
                          key=lambda p: abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]))
            rel = _relative_direction(agent_dir, agent_pos, nearest)
            parts.append(f"Lava {rel}")

        # Unseen area
        unseen_count = int(np.count_nonzero(obs_object == 0))
        if unseen_count > 0:
            parts.append(f"{unseen_count} unseen cells remain")

        if not parts:
            parts.append("Nothing visible")

        parts.append(f"Holding {carrying}")

        description = ". ".join(parts) + "."

        # Append recent failure context if set
        if self._failure_context:
            description += f" Recent issue: {self._failure_context}"

        return description

    def LLM2RL(self, plans, probs):
        if self.soft:
            skill_list = [self.parser(plan) for plan in plans]
        else:
            plan = np.random.choice(plans, p=probs)
            skill_list = [self.parser(plan)]
            probs = [1.]
                
        return skill_list, probs
    
    def reset(self):
        self.obj_coordinate = {}

class SimpleDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs, color_info=False):
        return super().RL2LLM(obs, color_info=False)

    def RL2LLM_rich(self, obs, color_info=False, include_lava=False):
        return super().RL2LLM_rich(obs, color_info=False, include_lava=False)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                # Unknown verb → fallback to explore (never WAIT)
                act = SKILL_TO_IDX["explore"]
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                # Target coordinate unavailable → explore instead of WAIT
                act = SKILL_TO_IDX["explore"]
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)

        return skill_list


class ColoredDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        return super().RL2LLM(obs)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                act = SKILL_TO_IDX["explore"]  # never WAIT
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                act = SKILL_TO_IDX["explore"]  # never WAIT
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
                
            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list
    
class TwoDoor_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        carrying = "nothing"
        if len(key_list):
            for key in key_list:
                i, j = key
                obj = "key"

                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, " 
                    self.obj_coordinate[obj] = (i,j)

        if len(door_list):
            n = 1
            for door in door_list:
                i, j = door
                obj = f"door{n}"
                n += 1
                
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i,j)

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        
        context = f"Agent sees {context}"
        return context
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                act = SKILL_TO_IDX["explore"]  # never WAIT
            # object:
            try:
                if "door1" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door1"]
                elif "door2" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door2"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                act = SKILL_TO_IDX["explore"]  # never WAIT
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list


if __name__ == "__main__":
    word = get_minigrid_words()