#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import os, requests
from typing import Any, Optional
from mediator import *
from utils import global_param

from abc import ABC, abstractmethod

# ── import robust helpers from qwen_llm (graceful fallback if missing) ────────
# These are only used in the ONLINE Qwen path of query_codex().
# Offline planner mode is completely unaffected.
try:
    from utils.qwen_llm import (
        MINIGRID_SYSTEM_PROMPT as _SYSTEM_PROMPT,
        CORRECTION_PROMPT      as _CORRECTION_PROMPT,
        robust_parse_plan      as _robust_parse,
    )
    _QWEN_UTILS = True
except ImportError:
    _QWEN_UTILS = False
    _SYSTEM_PROMPT     = ""
    _CORRECTION_PROMPT = ""
    _robust_parse      = None

# ── strict symbolic parser (graceful fallback if missing) ────────────────────
try:
    from utils.symbolic_parser import strict_parse as _strict_parse
    _STRICT_PARSER = True
except ImportError:
    _STRICT_PARSER = False
    _strict_parse  = None

# ── reflection cluster classifier (graceful fallback if missing) ─────────────
# Used only to PRIORITIZE which stored reflections to inject (req 11/12).
# Never changes plan-stability behavior (req 13).
try:
    from memory.reflection import classify_cluster as _CLUSTER_FN
except Exception:
    _CLUSTER_FN = None

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self, offline=True, soft=False, prefix='', env_type='simpledoorkey'):
        super().__init__()
        self.offline = offline
        self.soft = soft
        self.prompt_prefix = prefix
        self._env_type = env_type.lower()
        self.plans_dict = {}
        self.mediator = None

        self.dialogue_system = ''
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = False

        # Bug 4: explicit experiment-state tracking
        self._stats = {
            "planner_calls":      0,
            "online_success":     0,
            "offline_fallback":   0,
            "explore_fallback":   0,
            "cache_hits":         0,
            "parser_rejections":  0,
            "cache_invalidations": 0,
            "prompt_injections":  0,   # times reflection text was injected into an LLM prompt
        }

        # Failure-aware cache invalidation — confidence-based, not count-based
        # Maps obs_text → {"failures": int, "successes": int, "confidence": float}
        self._plan_confidence:        dict  = {}
        self._INVALIDATION_THRESHOLD: int   = 5         # invalidate after 5 sustained failures
        self._CONFIDENCE_DECAY:       float = 0.35  # each failure decays confidence faster
        self._CONFIDENCE_RECOVERY:    float = 0.25  # each success recovers confidence
        self._epsilon_diversity:      float = 0.0   # disabled: stability over diversity

        if not offline:
            self.llm_model = "qwen2.5:3b"
            # Bug 1 fix: native Ollama endpoint
            self.llm_url   = "http://localhost:11434/api/chat"
            self.plans_dict = {}
        
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
        self.mediator.reset()
        # if not self.offline:
        #     self.online_planning("reset")
        
    def init_llm(self):
        self.dialogue_system += self.prompt_prefix

        ## set system part
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': self.llm_model, "messages":[{"role": "system", "content": self.prompt_prefix}]}
                response = requests.post(self.llm_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to initialize: status code {response.status_code}"                
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to initialize: {e}")

    def report_failure(self, obs_text: str) -> None:
        """
        Confidence-based cache invalidation.
        Each failure decays confidence. Only invalidates after sustained
        evidence of failure — tolerates temporary execution instability.
        Repeated navigation / temporary oscillation should NOT reach here.
        """
        entry = self._plan_confidence.setdefault(
            obs_text, {"failures": 0, "successes": 0, "confidence": 1.0}
        )
        entry["failures"]   += 1
        entry["confidence"] -= self._CONFIDENCE_DECAY
        entry["confidence"]  = max(0.0, entry["confidence"])

        # Only invalidate when BOTH failure count is high AND confidence is low
        if (entry["failures"] >= self._INVALIDATION_THRESHOLD
                and entry["confidence"] <= 0.1
                and obs_text in self.plans_dict):
            print(
                f"[Planner] Cache invalidated for obs '{obs_text[:55]}' "
                f"after {entry['failures']} failures, confidence={entry['confidence']:.2f}"
            )
            del self.plans_dict[obs_text]
            entry["failures"]    = 0
            entry["confidence"]  = 1.0
            self._stats["cache_invalidations"] += 1

    def report_success(self, obs_text: str) -> None:
        """Recover confidence after a successful execution — prevents over-invalidation."""
        entry = self._plan_confidence.setdefault(
            obs_text, {"failures": 0, "successes": 0, "confidence": 1.0}
        )
        entry["successes"]  += 1
        entry["confidence"] += self._CONFIDENCE_RECOVERY
        entry["confidence"]  = min(1.0, entry["confidence"])

    def reset_failure_count(self, obs_text: str) -> None:
        """Reset confidence tracking after a successful execution."""
        self._plan_confidence.pop(obs_text, None)

    def set_reflection_memory(self, memory) -> None:
        """
        Inject a ReflectionMemory instance so the planner can prepend
        strategic reflections to each LLM query.

        Pass None to remove a previously injected memory.
        Offline planner mode is completely unaffected — reflection context
        is only used in the ONLINE Qwen path of query_codex().
        """
        self._reflection_memory = memory

    def set_llm(self, llm) -> None:
        """
        Inject a QwenLLM instance for online planning.

        Call after creating the Game / planner:
            from utils.qwen_llm import QwenLLM
            qwen = QwenLLM(backend='ollama')
            game.teacher_policy.planner.set_llm(qwen)

        Pass None to remove a previously injected LLM.
        Offline planner mode is unaffected — set_llm() is only used
        when the planner needs to call an external LLM.
        """
        self._qwen_llm = llm

    def _build_prompt_with_reflection(self, obs_text: str) -> str:
        """
        Build a grounded, constraint-aware prompt for the LLM.

        Uses RL2LLM_rich() if available to produce relational state summaries
        instead of raw object lists. Adds task implications and failure context.
        The obs_text (cache key) is NOT modified — only the LLM prompt is enriched.
        """
        # --- Build grounded state description ---
        grounded = obs_text  # fallback to raw cache key text
        mediator = getattr(self, "mediator", None)
        last_obs = getattr(self, "_last_raw_obs", None)
        if mediator is not None and last_obs is not None and hasattr(mediator, "RL2LLM_rich"):
            try:
                include_lava = getattr(self, "_include_lava", False)
                color_info   = not isinstance(mediator, SimpleDoorKey_Mediator)
                grounded = mediator.RL2LLM_rich(last_obs, color_info=color_info,
                                                 include_lava=include_lava)
            except Exception:
                grounded = obs_text

        # --- Add task implications ---
        implications = self._infer_task_implications(obs_text)

        # --- Add reflection memory (cluster-prioritized retrieval, req 12) ---
        mem = getattr(self, "_reflection_memory", None)
        current_cluster = self._current_cluster(obs_text)
        mem_context = (
            mem.get_context(current_cluster=current_cluster)
            if mem is not None else ""
        )

        # --- Assemble compact prompt ---
        parts = []
        if mem_context:
            # Reflection text actually reached the LLM prompt → count an injection.
            self._stats["prompt_injections"] += 1
            parts.append(f"Past experience:\n{mem_context}")
        parts.append(f"Current state:\n{grounded}")
        if implications:
            parts.append(f"Task constraints:\n{implications}")
        return "\n\n".join(parts)

    def _current_cluster(self, obs_text: str):
        """
        Infer the reusable-strategy cluster most relevant to the current state,
        used to PRIORITIZE which stored reflections are injected (req 11/12).

        Best-effort and side-effect free: prefers the mediator's recent failure
        context, then falls back to the grounded state. Returns None when nothing
        specific applies (memory then falls back to success-rate / frequency
        ordering). This never alters replanning behavior (req 13).
        """
        mediator = getattr(self, "mediator", None)
        failure_ctx = getattr(mediator, "_failure_context", "") if mediator is not None else ""
        if failure_ctx and _CLUSTER_FN is not None:
            c = _CLUSTER_FN(failure_ctx)
            if c != "GENERAL":
                return c

        state = self._parse_obs_state(obs_text)
        if state["door_visible"] and not state["holding_key"]:
            return "KEY_FIRST"
        if state["holding_key"] and state["door_visible"]:
            return "DIRECT_NAVIGATION"
        if state["key_visible"] and not state["holding_key"]:
            return "PICKUP_ALIGNMENT"
        if state["nothing_visible"] and state["holding_nothing"]:
            return "EXPLORE_FRONTIER"
        return None

    def _infer_task_implications(self, obs_text: str) -> str:
        """
        Derive symbolic task implications from the observation text.
        These are constraint reminders injected into the LLM prompt.
        """
        txt = obs_text.lower()
        hints = []

        # Door visible but no key held → cannot open yet
        if "door" in txt and "holds <nothing>" in txt:
            hints.append("Door cannot be opened yet — find and pick up the key first.")

        # Door visible and key held → ready to open
        # Match bare <key> (SimpleDoorKey) AND any <color key> (ColoredDoorKey)
        _holds_key = "holds <key>" in txt or any(
            f"holds <{c} key>" in txt
            for c in ["red", "green", "blue", "yellow", "purple", "grey"]
        )
        if "door" in txt and _holds_key:
            hints.append("Holding key — go to door and open it.")

        # Colored key/door matching hint
        for color in ["red", "green", "blue", "yellow", "purple", "grey"]:
            if f"{color} door" in txt and f"holds <{color} key>" not in txt:
                if f"{color} key" in txt:
                    hints.append(f"Need {color} key to open {color} door — pick it up first.")
                else:
                    hints.append(f"Need {color} key for {color} door — explore to find it.")

        # Nothing visible → must explore
        if "nothing" in txt and "holds <nothing>" in txt:
            hints.append("No objects visible — explore to find key and door.")

        # LavaDoorKey-specific hints
        if getattr(self, "_env_type", "") == "lavadoorkey":
            if "lava" in txt:
                hints.append("Lava tiles are visible — navigate around them, never step on lava.")
            if "lava ahead" in txt:
                hints.append("Lava is directly ahead — turn left or right to avoid it.")

        # ColoredDoorKey-specific hints
        if getattr(self, "_env_type", "") == "coloreddoorkey":
            # Wrong key warning: holding a key but door colour doesn't match
            import re as _re
            held_match = _re.search(r"holds <(\w+) key>", txt)
            door_match = _re.search(r"<(\w+) door>", txt)
            if held_match and door_match:
                held_col = held_match.group(1)
                door_col = door_match.group(1)
                if held_col != door_col:
                    hints.append(
                        f"Holding {held_col} key but door is {door_col} — "
                        f"drop this key, find the {door_col} key and pick it up."
                    )

        # Failure context from mediator
        mediator = getattr(self, "mediator", None)
        if mediator is not None:
            fc = getattr(mediator, "_failure_context", "")
            if fc:
                hints.append(f"Recent failure: {fc} — change strategy.")

        return "\n".join(f"- {h}" for h in hints) if hints else ""

    def _apply_strict_parser(self, plan: str) -> str:
        """
        Run the strict symbolic parser on a plan string.
        Returns the validated plan or 'explore' as a safe fallback.
        Logs rejections and retries are handled by the caller.
        """
        if not _STRICT_PARSER or _strict_parse is None:
            return plan
        result = _strict_parse(plan)
        if result is None:
            print(f"[SymbolicParser] Plan rejected, fallback to 'explore': '{plan[:60]}'")
            return "explore"
        return result

    def query_codex(self, prompt_text: str) -> str:
        """
        Query the LLM for a single symbolic plan string.

        Two execution paths
        ───────────────────
        A) Qwen path  (active when set_llm() has been called)
           Uses QwenLLM.call_with_retry() which:
             1. Sends MINIGRID_SYSTEM_PROMPT + observation text
             2. Runs robust_parse_plan() (4-pass tolerant extractor)
             3. On parse failure: retries once with CORRECTION_PROMPT
             4. Returns (plan_or_None, raw_response)
           If both attempts fail:
             - checks self.plans_dict for an offline fallback entry
             - if found: returns the highest-probability offline plan
             - if not:   returns 'explore' (safe no-op)
           NEVER crashes training.

        B) Legacy local-server path  (no QwenLLM injected)
           Preserved for backward compatibility with vicuna-33b / chatglm
           setups.  Uses robust_parse_plan() if available, then falls
           back to the original regex.  Fixes the original dict-vs-string
           regex bug.

        Note: offline planner mode never calls query_codex() because
        plan() returns from self.plans_dict before reaching this method.
        """
        # ── Path A: Qwen / modern LLM ────────────────────────────────────────
        if getattr(self, "_qwen_llm", None) is not None:
            augmented_prompt = self._build_prompt_with_reflection(prompt_text)
            plan, _ = self._qwen_llm.call_with_retry(
                user_prompt=augmented_prompt,
                system_prompt=_SYSTEM_PROMPT if _QWEN_UTILS else self.prompt_prefix,
                correction_prompt=_CORRECTION_PROMPT if _QWEN_UTILS else None,
            )
            if plan is not None:
                plan = self._apply_strict_parser(plan)
                self._stats["online_success"] += 1
                return plan

            # Both Qwen attempts failed — Bug 4: log as fallback, not silent
            print(f"[Qwen] Falling back to offline planner for obs: '{prompt_text[:60]}'")
            self._stats["offline_fallback"] += 1
            if prompt_text in self.plans_dict:
                offline_plans, _ = self.plans_dict[prompt_text]
                fallback = offline_plans[0] if offline_plans else "explore"
                print(f"[Qwen] Offline fallback plan: {fallback}")
                return fallback
            self._stats["explore_fallback"] += 1
            return "explore"

        # ── Path B: legacy local-server (vicuna / chatglm / raw Ollama) ──────
        server_error_cnt = 0
        while server_error_cnt < 5:
            try:
                headers = {"Content-Type": "application/json"}
                if self.llm_model == "chatglm_Turbo":
                    data = {
                        "model":  self.llm_model,
                        "prompt": [{"role": "user",
                                    "content": self.prompt_prefix + prompt_text}],
                    }
                else:  # vicuna-33b, qwen2.5 via raw Ollama, or any OpenAI-compat
                    data = {
                        "model":    self.llm_model,
                        "messages": [{"role": "user", "content": prompt_text}],
                    }

                response = requests.post(
                    self.llm_url, headers=headers, json=data, timeout=60
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")

                result = response.json()
                # Extract text string from response dict.
                # Original code incorrectly ran regex on the dict object.
                if isinstance(result, dict) and "choices" in result:
                    result_text = result["choices"][0]["message"]["content"]
                elif isinstance(result, dict) and "response" in result:
                    # Some Ollama versions return {"response": "..."}
                    result_text = result["response"]
                else:
                    result_text = str(result)

                # Try robust parser first, then fall back to original regex
                plan = _robust_parse(result_text) if _QWEN_UTILS else None
                if plan is None:
                    m = re.search(
                        r"Action[s]?\s*:\s*\{([^}]*)\}",
                        result_text, re.I | re.M,
                    )
                    plan = m.group(1).strip() if m else None

                if plan:
                    return plan

                print(
                    f"[LLM] Could not parse plan from response: "
                    f"'{result_text[:120]}'"
                )
                server_error_cnt += 1

            except Exception as exc:
                server_error_cnt += 1
                print(f"[query_codex] attempt {server_error_cnt}/5 failed: {exc}")

        print("[query_codex] WARNING: all retries exhausted, defaulting to 'explore'")
        return "explore"
        
    def _sanitize_plans_dict(self) -> None:
        """
        Bug 3 fix: filter offline plans_dict entries through strict_parse.

        Removes invalid symbolic objects (e.g. <handle>, 'with <key>') from
        pre-seeded offline plans. Redistributes probability to valid entries
        or falls back to 'explore' if all plans for an obs are invalid.

        Called once by subclass __init__ after plans_dict is populated.
        """
        if not _STRICT_PARSER or _strict_parse is None:
            return
        cleaned: dict = {}
        for obs_text, (plans, probs) in self.plans_dict.items():
            valid_plans, valid_probs = [], []
            for plan, prob in zip(plans, probs):
                validated = _strict_parse(plan)
                if validated is not None:
                    valid_plans.append(validated)
                    valid_probs.append(prob)
                else:
                    print(
                        f"[SymbolicParser] Removed invalid offline plan "
                        f"'{plan[:60]}' for obs '{obs_text[:50]}'"
                    )
                    self._stats["parser_rejections"] += 1
            if not valid_plans:
                valid_plans = ["explore"]
                valid_probs = [1.0]
                print(
                    f"[SymbolicParser] All plans invalid for obs "
                    f"'{obs_text[:50]}' — fallback to explore"
                )
            # Renormalize probabilities
            total = sum(valid_probs)
            valid_probs = [p / total for p in valid_probs]
            cleaned[obs_text] = (valid_plans, valid_probs)
        self.plans_dict = cleaned

    @property
    def planner_stats(self) -> dict:
        s = self._stats
        total = max(s["planner_calls"], 1)
        return {
            **s,
            "online_success_rate":   s["online_success"]   / total,
            "offline_fallback_rate": s["offline_fallback"]  / total,
            "explore_fallback_rate": s["explore_fallback"]  / total,
            "cache_hit_rate":        s["cache_hits"]         / total,
        }

    def plan(self, text, n_ask=1):
        """
        n_ask=1: single LLM call per new observation.

        Rationale: PPO needs STABLE symbolic priors, not diverse sampling.
        The original n_ask=10 was designed for a static teacher with no PPO
        exploration. The current architecture already has PPO exploration,
        skills, and intervention diversity — additional LLM sampling is harmful.

        Cache hits are returned immediately — repetition is NORMAL and desirable.
        """
        self._stats["planner_calls"] += 1

        if text in self.plans_dict:
            self._stats["cache_hits"] += 1
            plans, probs = self.plans_dict[text]
            # NO epsilon-diversity sampling — stability over diversity
            return plans, probs

        # --- Single LLM call for new observation ---
        print(f"new obs: {text}")
        raw_plan = self.query_codex(text)

        # Step 1: Symbolic grammar validation
        if _STRICT_PARSER and _strict_parse is not None:
            validated = _strict_parse(raw_plan)
            if validated is None:
                print(f"[SymbolicParser] Plan rejected: '{raw_plan[:60]}' → 'explore'")
                self._stats["parser_rejections"] += 1
                validated = "explore"
                self._stats["explore_fallback"] += 1
        else:
            validated = raw_plan

        # Step 2: Semantic grounding validation + repair
        repaired = self._validate_and_repair_plan(validated, text)

        # Step 3: Add stochasticity — reduce planner overconfidence
        plans, probs = self._diversify_plan(text, repaired)

        self.plans_dict[text] = (plans, probs)

        for k, v in self.plans_dict.items():
            print(f"{k}:{v}")

        return plans, probs

    # ── Grounded state parser ─────────────────────────────────────────────────

    @staticmethod
    def _parse_obs_state(obs_text: str) -> dict:
        """
        Parse symbolic observation into a structured grounded state dict.
        Used by the semantic validator to check plan feasibility.
        """
        import re as _re
        txt = obs_text.lower()
        m = _re.search(r"holds <([^>]+)>", txt)
        holding_obj     = m.group(1).strip() if m else "nothing"
        holding_key     = bool(holding_obj) and "key" in holding_obj
        holding_nothing = holding_obj in ("nothing", "")
        sees_part       = txt.split("holds")[0] if "holds" in txt else txt
        key_visible     = ("<key>" in sees_part or
                           any(f"<{c} key>" in sees_part
                               for c in ["red","green","blue","yellow","purple","grey"]))
        door_visible    = "door" in sees_part
        nothing_visible = "<nothing>" in sees_part
        return {
            "holding_key":     holding_key,
            "holding_nothing": holding_nothing,
            "key_visible":     key_visible,
            "door_visible":    door_visible,
            "nothing_visible": nothing_visible,
            "holding_obj":     holding_obj,
        }

    def _validate_and_repair_plan(self, plan: str, obs_text: str) -> str:
        """
        Validate and REPAIR a plan against the current grounded state.

        Instead of rejecting plans, this repairs them token by token:
          R1. go to <key> when key not visible    → replace with explore
          R2. pick up <key> when already holding  → remove token
          R3. go to <door> when door not visible  → prefix with explore
          R4. open <door> without key             → inject pickup or explore
          R5. nothing visible + holding nothing   → force explore
          R6. deduplicate consecutive tokens
        """
        state = self._parse_obs_state(obs_text)

        _is_colored = getattr(self, 'env_type', '').lower() == 'coloreddoorkey'

        if _is_colored:
            print("OBS     :", obs_text)
            print("PLAN IN :", plan)

        # R5 — nothing visible and holding nothing → must explore
        if state["nothing_visible"] and state["holding_nothing"]:
            return "explore"

        tokens = [t.strip() for t in plan.split(",") if t.strip()]
        if not tokens:
            return "explore"

        repaired = []
        need_explore_prefix = False

        for tok in tokens:
            tl = tok.lower()

            # R2: already holding key → skip redundant pickup
            if "pick up" in tl and "key" in tl and state["holding_key"]:
                print(f"[SemanticValidator] Removed redundant pickup (already holding): '{tok}'")
                continue

            # R1: go to key but key not visible
            if "go to" in tl and "key" in tl and not state["key_visible"] and not state["holding_key"]:
                print(f"[SemanticValidator] Key not visible — replacing with explore: '{tok}'")
                need_explore_prefix = True
                continue

            # R3: go to door but door not visible
            if "go to" in tl and "door" in tl and not state["door_visible"]:
                print(f"[SemanticValidator] Door not visible — adding explore prefix: '{tok}'")
                need_explore_prefix = True
                continue

            # R4: open door without holding key
            if "open" in tl and "door" in tl and not state["holding_key"]:
                if state["key_visible"]:
                    print(f"[SemanticValidator] Injecting key pickup before open door.")
                    # Extract actual key token — may be colored (e.g. <red key>)
                    import re as _re2
                    _km = _re2.search(r"<((?:\w+ )?key)>", obs_text.lower())
                    _key_tok = f"<{_km.group(1)}>" if _km else "<key>"
                    if not any("go to" in r.lower() and "key" in r.lower() for r in repaired):
                        repaired.append(f"go to {_key_tok}")
                    if not any("pick up" in r.lower() and "key" in r.lower() for r in repaired):
                        repaired.append(f"pick up {_key_tok}")
                    repaired.append(tok)
                else:
                    print(f"[SemanticValidator] Can't open door — no key → explore.")
                    need_explore_prefix = True
                continue

            repaired.append(tok)

        # R6: deduplicate consecutive identical tokens
        deduped = []
        for tok in repaired:
            if not deduped or tok.lower() != deduped[-1].lower():
                deduped.append(tok)

        if need_explore_prefix and not deduped:
            return "explore"
        if need_explore_prefix:
            deduped = ["explore"] + deduped
        if not deduped:
            return "explore"

        result = ", ".join(deduped)
        if result != plan:
            print(f"[SemanticValidator] '{plan[:55]}' → '{result[:55]}'")

        if _is_colored:
            print("PLAN OUT:", result)
            if "<key>" in result and "<key>" not in plan:
                print("*** bare <key> INJECTED — was not in original plan ***")

        return result

    def _diversify_plan(self, obs_text: str, plan: str) -> tuple:
        """
        Add stochasticity to prevent deterministic plan loops.
        Returns (plans, probs).
        Reduces primary plan weight when it has repeatedly failed.
        Gives PPO meaningful diversity without flooding it with noise.
        """
        failures = self._plan_confidence.get(obs_text, {}).get("failures", 0)
        if plan == "explore":
            return ["explore"], [1.0]
        if failures >= 3:
            print(f"[Planner] Plan failed {failures}x — boosting explore weight.")
            return [plan, "explore"], [0.40, 0.60]
        if failures >= 1:
            return [plan, "explore"], [0.70, 0.30]
        return [plan, "explore"], [0.85, 0.15]

    def _filter_impossible_plans(self, obs_text: str, counts: dict) -> dict:
        """Kept for backward compatibility — logic moved to _validate_and_repair_plan."""
        return counts
    
    def __call__(self, obs):
        # Store raw obs so _build_prompt_with_reflection can access RL2LLM_rich
        self._last_raw_obs = obs
        # COORDINATE-FREE (req 5): the cache key is the symbolic RL2LLM() text
        # ("Agent sees <key>, holds <nothing>.") — no coordinates. Grid positions
        # live only in mediator.obj_coordinate and are consumed by skills for
        # navigation; they are NEVER stored in the plan cache / reflection memory
        # nor sent to the LLM (the LLM prompt uses relational RL2LLM_rich).
        text = self.mediator.RL2LLM(obs)       # cache key — unchanged
        plans, probs = self.plan(text)
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        return skill_list, probs
    
    

class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix, env_type='simpledoorkey'):
        super().__init__(offline, soft, prefix, env_type=env_type)
        self.mediator = SimpleDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door>, holds <nothing>."    : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."     : [["go to <key>, pick up <key>", "pick up <key>"], [0.98, 0.02]],
                # Bug 3 fix: removed <handle> and "with <key>" — invalid symbolic entities.
                # _sanitize_plans_dict() below also catches any residual invalid entries.
                "Agent sees <nothing>, holds <key>."     : [["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>"], [0.68, 0.22, 0.04, 0.04, 0.02]],
                "Agent sees <door>, holds <key>."        : [["go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>"], [0.92, 0.06, 0.02]],
                "Agent sees <key>, <door>, holds <nothing>." : [["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, explore"], [0.84, 0.08, 0.04, 0.04]],
            }
        self._sanitize_plans_dict()
    

class LavaDoorKey_Planner(SimpleDoorKey_Planner):
    """
    Planner for LavaDoorKey — identical to SimpleDoorKey except:
      - _include_lava=True  → RL2LLM_rich() surfaces lava direction in the prompt
      - _env_type='lavadoorkey' → lava-avoidance hints injected via
        _infer_task_implications()
    SimpleDoorKey offline plan dict and mediator are reused unchanged.
    """
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix, env_type='lavadoorkey')
        # Wire lava into the rich prompt builder
        self._include_lava = True


class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix, env_type='coloreddoorkey')
        self.mediator = ColoredDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {  # noqa: E501
                "Agent sees <nothing>, holds <nothing>."       : [["explore"],[1]],
                "Agent sees <nothing>, holds <color1 key>."    : [["explore","go to east"], [0.94,0.06]],
                "Agent sees <color1 key>, holds <nothing>."    : [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 door>, holds <nothing>."   : [["explore"],[1.0]],
                "Agent sees <color1 door>, holds <color1 key>.": [["go to <color1 door>, open <color1 door>","open <color1 door>"],[0.72,0.28]],
                "Agent sees <color1 door>, holds <color2 key>.": [["explore", "go to <color2 key>"],[0.98,0.02]],
                "Agent sees <color1 key>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 key>, <color2 key>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.81,0.19]],
                "Agent sees <color1 key>, <color2 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.73,0.27]],
                "Agent sees <color1 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.84,0.16]],
                "Agent sees <color1 key>, <color1 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.79,0.21]],
                "Agent sees <color1 key>, <color2 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>", "go to <color2 door>, open <color2 door>"],[0.71,0.29]],
                "Agent sees <color1 key>, <color2 key>, <color2 door>, holds <nothing>.": [["go to <color2 key>, pick up <color2 key>","pick up <color2 key>","go to <color1 key>, pick up <color1 key>"],[0.72,0.24,0.04]],
                "Agent sees <color1 key>, <color2 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>", "pick up <color1 key>"],[0.94,0.06]],
            }
        # ColoredDoorKey uses color1/color2 placeholder tokens which are NOT
        # in VALID_OBJECTS. Sanitization is skipped here — color expansion
        # happens in plan() before any LLM call or cache lookup.

    def plan(self, text):
        pattern= r'\b(blue|green|grey|purple|red|yellow)\b'
        color_words = re.findall(pattern, text)

        # Deduplicate while preserving first-occurrence order.
        # list(set(...)) destroys order — two episodes with the same
        # colors in different positions would map color1/color2 differently,
        # making cached plans inconsistent across episodes.
        seen = {}
        color_words = [seen.setdefault(w, w) for w in color_words if w not in seen]
        color_index =['color1','color2']
        if color_words != []:
            for i in range(len(color_words)):
                text = text.replace(color_words[i], color_index[i])

        plans, probs = super().plan(text)

        plans = str(plans)
        for i in range(len(color_words)):
            plans = plans.replace(color_index[i], color_words[i])
        plans = eval(plans)

        return plans, probs


class TwoDoor_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = TwoDoor_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door1>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <nothing>, holds <key>."     : [["explore"], [1.0]],
                "Agent sees <door1>, holds <key>."        : [["go to <door1>, open <door1>"], [1.0]],
                "Agent sees <key>, <door1>, holds <nothing>." : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, <door1>, <door2>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <key>.": [["go to <door1>, open <door1>", "go to <door2>, open <door2>"], [0.5, 0.5]],
            }
        self._sanitize_plans_dict()


def Planner(task, offline=True, soft=False, prefix=''):
    t = task.lower()
    if t in ("simpledoorkey", "simpledoorkey_large"):
        planner = SimpleDoorKey_Planner(offline, soft, prefix, env_type=t)
    elif t == "lavadoorkey":
        planner = LavaDoorKey_Planner(offline, soft, prefix)
    elif t == "coloreddoorkey":
        planner = ColoredDoorKey_Planner(offline, soft, prefix)
    elif t == "twodoor":
        planner = TwoDoor_Planner(offline, soft, prefix)
    else:
        raise ValueError(f"Unknown task: '{task}'")
    return planner
                                                            
                                                            