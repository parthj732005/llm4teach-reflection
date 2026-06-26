#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
utils/qwen_llm.py  —  Qwen LLM wrapper for LLM4Teach
======================================================
Replaces the original ChatGLM / vicuna-33b / OpenAI backend.

Backends
--------
  'offline'   No LLM calls.  Uses pre-computed plans in planner.py dicts.
              Zero setup, works immediately.

  'ollama'    Local Qwen2.5 via Ollama (free, no API key, ~1 GB disk).
              Install : https://ollama.ai
              Pull    : ollama pull qwen2.5:3b
              Start   : ollama serve

              Uses the NATIVE Ollama /api/chat endpoint (not the OpenAI-compat
              /v1/chat/completions which returns 404 on older Ollama versions).

  'dashscope' Alibaba Cloud Qwen API (free tier: 2M tokens/month).
              Key     : https://bailian.console.aliyun.com/
              Install : pip install dashscope

Bug fixes in this version
--------------------------
1. FIXED: _call_ollama() now uses the NATIVE /api/chat endpoint
   (was using /v1/chat/completions → 404 on most Ollama versions).
   Response key is now message.content, not choices[0].message.content.

2. FIXED: startup validation now verifies actual inference works,
   not just that the server is reachable.

3. ADDED: explicit [Qwen] log messages for every state transition:
   Connected / Inference succeeded / Retry triggered / Falling back.

4. ADDED: _stats dict tracking calls, successes, retries, failures.

5. ADDED: verify_inference() — run a test call to confirm LLM works
   before training starts.
"""

import re
import requests
from typing import Optional, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MINIGRID_SYSTEM_PROMPT = (
    "You are a strategic MiniGrid planning agent.\n\n"
    "Output ONLY a compact symbolic action plan — no explanations, no markdown.\n\n"

    "ALLOWED ACTIONS:\n"
    "  explore\n"
    "  go to <key>  |  go to <door>  |  go to <goal>\n"
    "  pick up <key>\n"
    "  open <door>\n"
    "  drop <key>\n\n"

    "CRITICAL VISIBILITY RULE (most important):\n"
    "  You can ONLY use 'go to <X>' or 'pick up <X>' for objects that appear in 'Agent sees ...'.\n"
    "  If an object is NOT listed in the current observation → you CANNOT navigate to it.\n"
    "  If objects are NOT visible → use 'explore' first to find them.\n\n"

    "TASK CONSTRAINTS:\n"
    "1. LOCKED door requires matching key held — CANNOT open without key.\n"
    "2. Door not held key → find key first. Door visible + key held → go open it.\n"
    "3. Nothing visible → explore.\n"
    "4. Already holding key → NEVER generate 'pick up <key>' again.\n"
    "5. Object not in 'Agent sees ...' → NEVER generate 'go to <object>'.\n"
    "6. Previous strategy failed → change approach, do NOT repeat same plan.\n"
    "7. Prefer shortest valid plan — no redundant steps.\n\n"

    "STATE → CORRECT PLAN EXAMPLES:\n"
    "  'sees <nothing>, holds <nothing>'         → explore\n"
    "  'sees <door>, holds <nothing>'            → explore\n"
    "  'sees <key>, holds <nothing>'             → go to <key>, pick up <key>\n"
    "  'sees <key>, <door>, holds <nothing>'     → go to <key>, pick up <key>, go to <door>, open <door>\n"
    "  'sees <door>, holds <key>'                → go to <door>, open <door>\n"
    "  'sees <nothing>, holds <key>'             → explore\n\n"

    "OUTPUT FORMAT: comma-separated symbolic actions only. No text.\n\n"
    "GOOD:  go to <key>, pick up <key>, go to <door>, open <door>\n"
    "BAD:   go to <key>  (when key not in 'Agent sees')\n"
    "BAD:   pick up <key>  (when already holding key)\n"
    "BAD:   open <door>  (when not holding key)\n"
    "BAD:   turn left, move forward\n"
)

CORRECTION_PROMPT = (
    "Your previous response was not a valid symbolic plan.\n"
    "Remember the key constraint: a locked door requires the key held first.\n"
    "Output ONLY comma-separated symbolic actions. Example:\n"
    "  go to <key>, pick up <key>, go to <door>, open <door>"
)


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST PARSER
# ══════════════════════════════════════════════════════════════════════════════

def robust_parse_plan(text: str) -> Optional[str]:
    """
    Robustly extract a symbolic MiniGrid action plan from an LLM response.

    Four-pass strategy (stops at first success):
    Pass 1 — Action:{plan} exact format
    Pass 2 — Symbolic tokens with <object> references
    Pass 3 — Bare 'explore' keyword
    Pass 4 — Heuristic line scan
    """
    if not text or not text.strip():
        return None

    cleaned = re.sub(r"```[^`]*```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", cleaned)
    cleaned = cleaned.strip()

    # Pass 1
    m = re.search(r"Action[s]?\s*:\s*\{([^}]*)\}", cleaned, re.I | re.M)
    if m:
        plan = m.group(1).strip()
        if plan:
            return plan

    # Pass 2
    SYMBOLIC = re.compile(
        r"((?:go\s+to|pick\s+up|open|drop)\s+<[^>\n]+>)", re.I
    )
    found = SYMBOLIC.findall(cleaned)
    if found:
        seen: set = set()
        deduped = []
        for raw_action in found:
            norm = re.sub(r"\s+", " ", raw_action.strip().lower())
            if norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        return ", ".join(deduped)

    # Pass 3
    if re.search(r"\bexplore\b", cleaned, re.I):
        for line in cleaned.splitlines():
            line_core = re.sub(r"^[-*•\d.):\s]+", "", line).strip()
            if re.fullmatch(r"explore[.,!?]?", line_core, re.I):
                return "explore"
        has_object_verbs = re.search(
            r"\b(go\s+to|pick\s+up|open|drop)\b", cleaned, re.I
        )
        if not has_object_verbs:
            return "explore"

    # Pass 4
    SENTENCE_WORDS = re.compile(
        r"\b(should|would|must|will|need|the agent|because|therefore|"
        r"first\s+you|then\s+you|after\s+that|finally|i\s+will|"
        r"you\s+need|let\s+me)\b",
        re.I,
    )
    ACTION_VERBS = re.compile(r"\b(go\s+to|pick\s+up|open|drop|explore)\b", re.I)
    for line in cleaned.splitlines():
        line_core = re.sub(r"^[-*•\d.):\s]+", "", line).strip()
        if (
            ACTION_VERBS.search(line_core)
            and not SENTENCE_WORDS.search(line_core)
            and len(line_core) <= 200
        ):
            plan = re.sub(r"[.!?]$", "", line_core).strip()
            if plan:
                return plan

    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class QwenLLM:
    """
    Qwen LLM client for online planning.

    BUG FIX: _call_ollama() now uses the NATIVE /api/chat endpoint
    instead of the OpenAI-compat /v1/chat/completions (which caused 404).
    """

    BACKENDS = ("offline", "ollama", "dashscope")

    def __init__(
        self,
        backend: str = "offline",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_predict: int = 64,
    ) -> None:
        assert backend in self.BACKENDS, (
            f"backend must be one of {self.BACKENDS}, got '{backend}'"
        )
        self.backend = backend
        self.temperature = temperature
        self.num_predict = num_predict

        # Stats: track every call outcome for experiment validity
        self._stats: Dict[str, int] = {
            "calls": 0,
            "successes": 0,
            "retries": 0,
            "failures": 0,
            "fallbacks": 0,
        }

        if backend == "dashscope":
            try:
                import dashscope as _ds
            except ImportError:
                raise ImportError(
                    "Run:  pip install dashscope\n"
                    "Then get a free key at https://bailian.console.aliyun.com/"
                )
            assert api_key, (
                "Dashscope backend requires an API key.\n"
                "Get a FREE key at: https://bailian.console.aliyun.com/"
            )
            self.api_key = api_key
            self.model = model or "qwen-turbo"
            self._ds = _ds

        elif backend == "ollama":
            self.model = model or "qwen2.5:3b"
            self.ollama_url = ollama_url.rstrip("/")
            # NATIVE Ollama endpoint — works on ALL Ollama versions
            self.chat_url = f"{self.ollama_url}/api/chat"
            self._check_ollama()

    # ── startup validation ────────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        """
        Verify Ollama is running and the model is available.
        Uses /api/tags (native endpoint, always present).
        """
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                base = self.model.split(":")[0]
                if any(base in n for n in names):
                    print(f"[Qwen] Connected successfully — model '{self.model}' found.")
                else:
                    print(f"[Qwen] ⚠️  Model '{self.model}' not yet pulled.")
                    print(f"[Qwen]     Run:  ollama pull {self.model}")
            else:
                print(f"[Qwen] Ollama returned HTTP {r.status_code}.")
        except requests.exceptions.ConnectionError:
            print("[Qwen] ❌  Ollama is NOT running.")
            print("[Qwen]    Install  :  https://ollama.ai")
            print("[Qwen]    Start    :  ollama serve")
            print(f"[Qwen]    Pull     :  ollama pull {self.model}")
        except Exception as exc:
            print(f"[Qwen] Could not check Ollama: {exc}")

    def verify_inference(self) -> bool:
        """
        Run a test call to confirm inference actually works end-to-end.
        Call this before training begins.

        Returns True if inference succeeded, False otherwise.
        Logs [Qwen] Inference succeeded or [Qwen] Inference FAILED.
        """
        if self.backend == "offline":
            print("[Qwen] Backend is offline — no inference verification needed.")
            return True
        try:
            raw = self.call(
                system_prompt="You are a test agent. Reply with exactly: explore",
                user_prompt="Test. Reply: explore",
            )
            plan = robust_parse_plan(raw)
            if plan:
                print(f"[Qwen] Inference succeeded — test response: '{raw[:60]}'")
                return True
            else:
                print(f"[Qwen] Inference FAILED — could not parse test response: '{raw[:60]}'")
                return False
        except Exception as exc:
            print(f"[Qwen] Inference FAILED with exception: {exc}")
            return False

    # ── low-level call ────────────────────────────────────────────────────────

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send one chat request. Raises RuntimeError on failure.
        """
        if self.backend == "offline":
            raise RuntimeError(
                "QwenLLM(backend='offline') makes no LLM calls."
            )
        if self.backend == "dashscope":
            return self._call_dashscope(system_prompt, user_prompt)
        if self.backend == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _call_dashscope(self, system_prompt: str, user_prompt: str) -> str:
        from dashscope import Generation
        self._ds.api_key = self.api_key
        resp = Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            result_format="message",
        )
        if resp.status_code == 200:
            return resp.output.choices[0]["message"]["content"]
        raise RuntimeError(f"Dashscope error {resp.status_code}: {resp.message}")

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """
        BUG FIX: Use NATIVE /api/chat endpoint.

        Previous code used /v1/chat/completions (OpenAI-compat) which returns
        404 on Ollama versions that do not have the compat layer enabled.

        Native /api/chat response format:
          { "message": { "role": "assistant", "content": "..." }, "done": true }

        NOT choices[0].message.content (that is OpenAI format).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            # Native /api/chat response: data["message"]["content"]
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama not reachable at {self.ollama_url}. "
                "Start it with: ollama serve"
            )
        except KeyError as exc:
            # Unexpected response shape — log full body for debugging
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:300]
            raise RuntimeError(
                f"Unexpected Ollama response shape (missing key {exc}): {body}"
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}")

    # ── high-level call with retry ────────────────────────────────────────────

    def call_with_retry(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        correction_prompt: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Call LLM with robust plan extraction and one automatic retry.

        Returns (plan_or_None, last_raw_response).
        plan is None only when BOTH attempts fail to parse.

        Explicit log messages:
          [Qwen] Connected successfully
          [Qwen] Inference succeeded
          [Qwen] Retry triggered
          [Qwen] Falling back to offline planner
        """
        sys_p  = system_prompt     or MINIGRID_SYSTEM_PROMPT
        corr_p = correction_prompt or CORRECTION_PROMPT

        self._stats["calls"] += 1

        # ── first attempt ──────────────────────────────────────────────────
        try:
            raw = self.call(system_prompt=sys_p, user_prompt=user_prompt)
        except Exception as exc:
            print(f"[Qwen] LLM call failed on first attempt: {exc}")
            self._stats["failures"] += 1
            return None, ""

        preview = raw[:120].replace("\n", " ")
        print(f"[Qwen] Raw response: {preview}{'…' if len(raw) > 120 else ''}")

        plan = robust_parse_plan(raw)
        if plan is not None:
            print(f"[Qwen] Inference succeeded — plan: {plan}")
            self._stats["successes"] += 1
            return plan, raw

        # ── retry ──────────────────────────────────────────────────────────
        print("[Qwen] Retry triggered — first response could not be parsed")
        self._stats["retries"] += 1
        retry_user = f"{user_prompt}\n\n{corr_p}"
        try:
            raw2 = self.call(system_prompt=sys_p, user_prompt=retry_user)
        except Exception as exc:
            print(f"[Qwen] LLM call failed on retry: {exc}")
            self._stats["failures"] += 1
            return None, raw

        preview2 = raw2[:120].replace("\n", " ")
        print(f"[Qwen] Retry raw response: {preview2}{'…' if len(raw2) > 120 else ''}")

        plan2 = robust_parse_plan(raw2)
        if plan2 is not None:
            print(f"[Qwen] Inference succeeded (retry) — plan: {plan2}")
            self._stats["successes"] += 1
            return plan2, raw2

        print("[Qwen] Falling back to offline planner — both attempts failed")
        self._stats["failures"] += 1
        self._stats["fallbacks"] += 1
        return None, raw2

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0

    # ── parsing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def robust_parse_plan(text: str) -> Optional[str]:
        return robust_parse_plan(text)

    @staticmethod
    def parse_plan(text: str) -> Optional[str]:
        return robust_parse_plan(text)

    # ── info helper ───────────────────────────────────────────────────────────

    @staticmethod
    def api_key_guide() -> None:
        print("""
╔══════════════════════════════════════════════════════════╗
║              QWEN LLM  —  SETUP GUIDE                   ║
╠══════════════════════════════════════════════════════════╣
║  OPTION A  offline  (default, zero setup)               ║
║    Pre-computed plans, no internet, no key.             ║
║    Set: args.offline_planner = True                     ║
╠══════════════════════════════════════════════════════════╣
║  OPTION B  ollama   (local model, NO key needed)        ║
║    1. Install  →  https://ollama.ai                     ║
║    2. Pull     →  ollama pull qwen2.5:3b                ║
║    3. Start    →  ollama serve                          ║
║    Use: QwenLLM(backend='ollama', model='qwen2.5:3b')   ║
╠══════════════════════════════════════════════════════════╣
║  OPTION C  dashscope  (Alibaba Cloud, free tier)        ║
║    1. Sign up  →  https://bailian.console.aliyun.com/   ║
║    2. Create API key  (free 2M tokens/month)            ║
║    3. pip install dashscope                             ║
║    Use: QwenLLM(backend='dashscope', api_key='sk-...')  ║
╚══════════════════════════════════════════════════════════╝
""")
