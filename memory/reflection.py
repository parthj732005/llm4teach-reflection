#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
memory/reflection.py — Qwen episode reflector for LLM4Teach

Generates compressed strategic reflections from completed episode trajectories.
Uses Qwen2.5-7B-Instruct (recommended) via Ollama or DashScope.

This module is COMPLETELY INDEPENDENT of the planner LLM:
  - No shared model weights
  - No shared chat history
  - No shared dialogue state
  - No shared conversation memory

The reflector summarizes ONLY verified episode facts.
It MUST NOT hallucinate hidden objects, coordinates, or speculative world state.
"""

import re
from typing import List, Optional, Dict, Any


# ── Valid environment entities ────────────────────────────────────────────────
VALID_COLORS  = {"red", "green", "blue", "yellow", "purple", "grey"}
VALID_OBJECTS = {"key", "door", "goal", "wall", "lava", "agent", "nothing"}

# Patterns that indicate hallucination or invalid reflection content.
# Reflection memory must store reusable STRATEGIES only — never coordinates,
# room locations, exact paths, or object positions (req 5, 6, 15).
_HALLUCINATION_PATTERNS = [
    re.compile(r"\b\d+\s*,\s*\d+\b"),                          # bare coordinates like 7,3
    re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)"),                    # (7,3) style
    # Location-specific language — bans positions/rooms/corners (req 15)
    re.compile(r"\b(upper|lower|top|bottom)[-\s]?(left|right)\b", re.I),  # "upper-right"
    re.compile(r"\bcorner\b", re.I),                          # "near the ... corner"
    re.compile(r"\b(row|column|col)\s*\d+\b", re.I),          # "row 3", "col 7"
    re.compile(r"\b(location|position|coordinate[s]?)\b", re.I),
    # Exact-path / directional instructions (req 15: no exact paths) —
    # e.g. "go north then east", "move south". Reusable strategies never name
    # absolute compass directions; those encode a specific route or location.
    re.compile(r"\b(north|south|east|west)(ward|ern)?\b", re.I),
    re.compile(r"\bhidden\s+room\b", re.I),
    re.compile(r"\bsecret\b|\bconcealed\b|\binvisible\b", re.I),
    re.compile(r"\bbookshelf\b|\bexit\b|\bstaircase\b|\bwindow\b", re.I),
    re.compile(r"\bemergency\b|\bescape\s+route\b", re.I),
    re.compile(r"\bprobably\s+(in|at|near|behind)\b", re.I),
    re.compile(r"\bthere\s+(is|are|should\s+be)\s+(probably|likely)\b", re.I),
]


# ── Reflection clusters (req 11) ──────────────────────────────────────────────
# Each cluster is a REUSABLE corrective STRATEGY, not an environment-specific
# situation. Multiple distinct failures map to the same cluster when they share
# the same corrective strategy (e.g. "toggled a locked door" and "tried to open
# the door without a key" both → KEY_FIRST).
CLUSTER_LABELS = (
    "KEY_FIRST",          # pick up the key before attempting to open the door
    "EXPLORE_FRONTIER",   # search unexplored frontier regions when no target is visible
    "DIRECT_NAVIGATION",  # after locating the target, navigate directly to it
    "PICKUP_ALIGNMENT",   # approach and face the key before picking it up
    "PLAN_PERSISTENCE",   # keep the working plan; avoid thrashing between strategies
    "SUCCESS",            # reinforce a behaviour that solved the task
    "GENERAL",            # fallback when no specific strategy applies
)

# EpisodeTrajectory.classify_failure() label → cluster
_FAILURE_TYPE_TO_CLUSTER = {
    "success":                  "SUCCESS",
    "toggle_without_key":       "KEY_FIRST",
    "key_acquired_door_failed": "DIRECT_NAVIGATION",
    "exploration_failure":      "EXPLORE_FRONTIER",
    "navigation_failure":       "DIRECT_NAVIGATION",
    "repeated_plan":            "PLAN_PERSISTENCE",
    "pickup_alignment":         "PICKUP_ALIGNMENT",
    "unknown_failure":          "GENERAL",
}


def classify_cluster(text: str, failure_type: Optional[str] = None) -> str:
    """
    Map a reflection to a reusable-strategy cluster label (req 11).

    Preference order:
      1. Explicit failure_type (from EpisodeTrajectory.classify_failure()).
      2. Keyword inference from the reflection text.
      3. GENERAL fallback.

    Returns one of CLUSTER_LABELS.
    """
    if failure_type:
        ft = failure_type.strip().lower()
        if ft in _FAILURE_TYPE_TO_CLUSTER:
            return _FAILURE_TYPE_TO_CLUSTER[ft]
        if "pickup" in ft or "align" in ft:
            return "PICKUP_ALIGNMENT"
        if "key" in ft and "door" in ft:
            return "KEY_FIRST"
        # Known failure type but no mapping — use text scan but NEVER return SUCCESS
        # for an explicitly non-success episode (prevents Qwen's "succeeded" wording
        # from mislabeling failure reflections as SUCCESS).
        _is_failure = (ft != "success")
    else:
        _is_failure = False

    t = (text or "").lower()
    if any(k in t for k in ("before attempting to open", "before opening the door",
                            "before the door", "without the key", "without holding the key",
                            "pick up the key first", "hold the key", "locked door")):
        return "KEY_FIRST"
    if any(k in t for k in ("align", "face the key", "adjacent to the key",
                            "approach the key", "next to the key")):
        return "PICKUP_ALIGNMENT"
    if any(k in t for k in ("explore", "unseen", "unexplored", "frontier", "search")):
        return "EXPLORE_FRONTIER"
    if any(k in t for k in ("navigate directly", "go straight", "directly to",
                            "shortest path", "navigate to the door", "navigate to the target")):
        return "DIRECT_NAVIGATION"
    if any(k in t for k in ("same plan", "repeated the", "stuck in a loop", "kept trying",
                            "maintain the plan", "keep the plan", "persist", "thrash")):
        return "PLAN_PERSISTENCE"
    # Only assign SUCCESS when the episode actually succeeded — never for failure episodes
    # even if Qwen's reflection text contains words like "succeeded" or "worked".
    if not _is_failure and any(k in t for k in ("succeeded", "success", "worked",
                                                 "completed the task", "opened the door")):
        return "SUCCESS"
    return "GENERAL"


# ── Reflection system prompt ──────────────────────────────────────────────────
REFLECTION_SYSTEM_PROMPT = (
    "You are an execution diagnostician for a MiniGrid RL agent.\n\n"
    "Your job: identify the SPECIFIC execution mistake and give ONE corrective strategy.\n"
    "Write in EXACTLY 2 sentences. No more.\n\n"

    "Sentence 1 — What went wrong (be specific about the failure mechanism):\n"
    "  GOOD: 'Agent approached the door and toggled it 8 times without holding the key.'\n"
    "  GOOD: 'Agent explored the same corridor repeatedly without reaching unseen areas.'\n"
    "  GOOD: 'Agent picked up the key but failed to navigate to the door before timeout.'\n"
    "  BAD:  'Agent failed this episode.'  ← too vague\n"
    "  BAD:  'The task was not completed.'  ← no diagnosis\n\n"

    "Sentence 2 — ONE concrete corrective strategy:\n"
    "  GOOD: 'Next time: pick up the key BEFORE approaching the door.'\n"
    "  GOOD: 'Next time: after picking up the key, navigate directly to the door.'\n"
    "  GOOD: 'Next time: explore unseen areas by turning toward open space.'\n"
    "  BAD:  'Try a different approach.'  ← too vague\n\n"

    "STRICT RULES:\n"
    "1. NO coordinates, NO grid positions\n"
    "2. NO speculation about unseen objects\n"
    "3. NO markdown, no bullets, no headers\n"
    "4. ONLY reference objects actually seen: key, door, lava, wall\n"
    "5. Focus on EXECUTION mechanics, not high-level outcomes\n"
    "6. If episode succeeded: describe WHY it worked in 1 sentence only\n"
)


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeTrajectory:
    """
    Collects and stores symbolic data from a single episode for reflection.

    Only stores text — no tensors, no numpy arrays, no raw observations.
    """

    def __init__(self):
        self.observations: List[str]   = []   # text obs from mediator.RL2LLM
        self.plans: List[str]          = []   # planner symbolic outputs
        self.rewards: List[float]      = []
        self.failure_reasons: List[str]= []   # semantic reasons from skills
        self.interventions: List[str]  = []   # intervention events from TeacherPolicy
        self.success: bool             = False
        self.ep_len: int               = 0
        self.total_reward: float       = 0.0

    def add_step(self, obs_text: str, plan: str, reward: float,
                 failure_reason: str = "", intervention: str = "") -> None:
        self.observations.append(obs_text)
        self.plans.append(plan if plan else "explore")
        self.rewards.append(float(reward))
        self.ep_len += 1
        self.total_reward += float(reward)
        if failure_reason:
            self.failure_reasons.append(failure_reason)
        if intervention:
            self.interventions.append(intervention)

    def finish(self, success: bool) -> None:
        self.success = success

    def classify_failure(self) -> str:
        """
        Classify the primary failure mode based on trajectory patterns.
        Returns a short label used in the reflection prompt.
        """
        if self.success:
            return "success"

        # Pickup alignment failures, when skill failure signals are available.
        if any("pickup_alignment" in r.lower() or "align" in r.lower()
               for r in self.failure_reasons):
            return "pickup_alignment"

        seen_plans: dict = {}
        for p in self.plans:
            seen_plans[p] = seen_plans.get(p, 0) + 1

        # Repetitive plan → repeated plan failure
        if seen_plans and max(seen_plans.values()) > self.ep_len * 0.6:
            return "repeated_plan"

        # Only explore in plans → exploration failure
        explore_count = sum(1 for p in self.plans if p.strip() == "explore")
        if explore_count > self.ep_len * 0.7:
            return "exploration_failure"

        # Door-only plans without key → interaction failure
        toggle_plans = sum(1 for p in self.plans if "open" in p or "toggle" in p)
        key_plans    = sum(1 for p in self.plans if "pick up" in p or "key" in p)
        if toggle_plans > 5 and key_plans < 2:
            return "toggle_without_key"

        # Key acquired but door never opened — dominant failure (39% of episodes)
        # Detect: agent had a "pick up <key>" plan succeed, then timed out without opening door
        key_pickup_idx = next(
            (i for i, p in enumerate(self.plans) if "pick up" in p and "key" in p), None
        )
        door_open_any = any("open" in p or "toggle" in p for p in self.plans)
        if key_pickup_idx is not None and not door_open_any:
            return "key_acquired_door_failed"

        # Navigation: GoTo repeated many times
        goto_plans = sum(1 for p in self.plans if "go to" in p)
        if goto_plans > self.ep_len * 0.8:
            return "navigation_failure"

        return "unknown_failure"

    def to_prompt(self) -> str:
        """
        Format trajectory as an execution-diagnosis prompt.
        Includes failure classification and corrective-action request.
        """
        status       = "SUCCESS" if self.success else "FAILURE"
        failure_type = self.classify_failure()

        seen_obs: dict = {}
        for o in self.observations:
            seen_obs[o] = seen_obs.get(o, 0) + 1
        seen_plans: dict = {}
        for p in self.plans:
            seen_plans[p] = seen_plans.get(p, 0) + 1

        lines = [
            f"Episode result: {status}",
            f"Failure type: {failure_type}",
            f"Episode length: {self.ep_len} steps",
            f"Total reward: {self.total_reward:.2f}",
            "",
            "Observations seen (most frequent first):",
        ]
        for obs, cnt in sorted(seen_obs.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  - {obs}  (repeated x{cnt})")

        lines += ["", "Plans executed (most frequent first):"]
        for plan, cnt in sorted(seen_plans.items(), key=lambda x: -x[1])[:4]:
            lines.append(f"  - {plan}  (x{cnt})")

        # Add failure-specific diagnostic hint
        if failure_type == "repeated_plan":
            lines += ["", "Note: The agent was stuck executing the same plan repeatedly."]
        elif failure_type == "toggle_without_key":
            lines += ["", "Note: The agent tried to open the door without picking up the key first."]
        elif failure_type == "exploration_failure":
            lines += ["", "Note: The agent spent most time exploring without finding key objects."]
        elif failure_type == "navigation_failure":
            lines += ["", "Note: The agent navigated repeatedly without completing interactions."]

        # Add skill-level failure reasons if present
        if self.failure_reasons:
            reason_counts: dict = {}
            for r in self.failure_reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            lines += ["", "Skill failure signals:"]
            for r, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"  - {r}  (x{cnt})")

        # Add intervention events if present
        if self.interventions:
            lines += ["", f"Intervention events: {len(self.interventions)} total"]
            unique_events = list(dict.fromkeys(self.interventions))[:3]
            for e in unique_events:
                lines.append(f"  - {e}")

        lines += [
            "",
            "In exactly 2 sentences:",
            "Sentence 1: What specific execution mistake caused this outcome?",
            "Sentence 2: ONE concrete corrective action for next time.",
            "ONLY describe what actually happened. Do NOT speculate.",
        ]
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return self.ep_len == 0


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_reflection(text: str) -> bool:
    """
    Reject reflections that contain hallucinations or invalid content.
    Returns True if the reflection passes all checks.
    """
    if not text or len(text.strip()) < 15:
        return False
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(text):
            return False
    return True


def clean_reflection(raw: str) -> str:
    """Strip markdown, trim whitespace, keep first 3 sentences."""
    text = re.sub(r"```[^`]*```", " ", raw, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = text.strip()
    # Keep at most 3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3]).strip()


# ══════════════════════════════════════════════════════════════════════════════
# QWEN REFLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class QwenReflector:
    """
    Qwen LLM wrapper for episode reflection generation.

    Completely independent of the planner LLM — no shared state.
    Recommended model: Qwen2.5-7B-Instruct for richer reflections.

    Supported backends
    ------------------
    'offline'   : No LLM calls. reflect() always returns None.
                  Use when reflection is disabled or no LLM is available.
    'ollama'    : Local Qwen via Ollama (free, recommended).
                  Run: ollama pull qwen2.5:7b && ollama serve
    'dashscope' : Alibaba Cloud Qwen API (free tier available).
                  pip install dashscope
    """

    BACKENDS = ("offline", "ollama", "dashscope")

    def __init__(
        self,
        backend: str = "offline",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.4,
        max_tokens: int = 200,
    ) -> None:
        assert backend in self.BACKENDS, (
            f"backend must be one of {self.BACKENDS}, got '{backend}'"
        )
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stats: Dict[str, int] = {
            "generated": 0,
            "validated": 0,
            "rejected": 0,
            "failed": 0,
            "skipped_empty": 0,
        }

        if backend == "ollama":
            import requests as _req
            self._requests = _req
            self.model = model or "qwen2.5:7b"
            self.ollama_url = ollama_url.rstrip("/")
            # BUG FIX: use NATIVE /api/chat endpoint (not /v1/chat/completions)
            self.chat_url = f"{self.ollama_url}/api/chat"
            self._check_ollama()

        elif backend == "dashscope":
            try:
                import dashscope as _ds
            except ImportError:
                raise ImportError(
                    "Run: pip install dashscope\n"
                    "Get a free key at: https://bailian.console.aliyun.com/"
                )
            assert api_key, (
                "Dashscope backend requires an API key.\n"
                "Get a FREE key at: https://bailian.console.aliyun.com/"
            )
            self.model = model or "qwen-plus"
            self.api_key = api_key
            self._ds = _ds

        else:  # offline
            self.model = "offline"

    # ── connectivity check ────────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        try:
            r = self._requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                # EXACT match (incl. tag). A substring/base match falsely reported
                # qwen2.5:7b as "found" when only qwen2.5:3b was pulled → every
                # /api/chat call then 404'd.
                if self.model in names or f"{self.model}:latest" in names:
                    print(f"[QwenReflector] Ollama ✅  model '{self.model}' found.")
                else:
                    print(f"[QwenReflector] ⚠️  Model '{self.model}' NOT pulled "
                          f"(will 404 on every call). Available: {names}")
                    print(f"[QwenReflector]     Run: ollama pull {self.model}")
        except Exception as exc:
            print(f"[QwenReflector] Could not verify Ollama: {exc}")

    # ── low-level call ────────────────────────────────────────────────────────

    def _call(self, user_prompt: str) -> Optional[str]:
        """Send a single call to the reflector LLM. Returns raw text or None."""
        if self.backend == "offline":
            return None

        if self.backend == "ollama":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            try:
                # Shorter timeout (30s) — reflection is not realtime critical.
                # Planner gets priority. If Ollama is busy, skip this reflection.
                resp = self._requests.post(self.chat_url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
            except self._requests.exceptions.Timeout:
                print("[QwenReflector] Timeout (30s) — Ollama busy with planner. Skipping reflection.")
                self._stats["failed"] += 1
                return None
            except Exception as exc:
                print(f"[QwenReflector] Ollama call failed: {exc}")
                self._stats["failed"] += 1
                return None

        if self.backend == "dashscope":
            from dashscope import Generation
            self._ds.api_key = self.api_key
            resp = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                result_format="message",
            )
            if resp.status_code == 200:
                return resp.output.choices[0]["message"]["content"]
            print(f"[QwenReflector] Dashscope error {resp.status_code}: {resp.message}")
            return None

        return None

    # ── public API ────────────────────────────────────────────────────────────

    def reflect(self, trajectory: EpisodeTrajectory) -> Optional[str]:
        """
        Generate a validated strategic reflection from an episode trajectory.

        Returns
        -------
        str  : Validated 1-3 sentence plain-English reflection.
        None : If backend is offline, generation failed, or reflection
               failed validation (hallucination detected).
        """
        if self.backend == "offline":
            return None

        if trajectory.is_empty():
            self._stats["skipped_empty"] += 1
            return None

        prompt = trajectory.to_prompt()
        raw = self._call(prompt)

        if raw is None:
            self._stats["failed"] += 1
            return None

        self._stats["generated"] += 1
        reflection = clean_reflection(raw)

        if validate_reflection(reflection):
            self._stats["validated"] += 1
            print(f"[QwenReflector] ✓ Reflection: {reflection[:100]}")
            return reflection
        else:
            self._stats["rejected"] += 1
            print(
                f"[QwenReflector] ✗ Reflection rejected (hallucination detected): "
                f"{reflection[:80]}"
            )
            return None

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0
