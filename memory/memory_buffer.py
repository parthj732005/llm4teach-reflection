#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
memory/memory_buffer.py — Reflection memory buffer for LLM4Teach

Stores episode reflections as reusable STRATEGY lessons and formats them for
injection into planner prompts.

Design (requirements 9–12, 15):
  - Hash-based deduplication: a duplicate reflection raises an existing entry's
    `frequency` instead of creating a new entry (req 9).
  - Each entry stores: reflection text, cluster label, frequency, and success
    statistics (success_count / failure_count) (req 10).
  - Cluster labels are reusable corrective strategies covering the major failure
    categories of the environment (req 11, see memory.reflection.CLUSTER_LABELS).
  - Retrieval is ordered by: (1) cluster relevance, (2) success rate,
    (3) frequency (req 11). Top-k entries are injected into the planner context;
    the planner remains responsible for deciding whether to change its plan
    (req 12).
  - Coordinate / position / path content is rejected on write (req 5, 6, 15) via
    memory.reflection.validate_reflection().

Memory entries contain ONLY plain text + scalar metadata — no tensors, no arrays,
no coordinates. All entries are JSON-serializable.
"""

import hashlib
import json
import os
import re
import time
from typing import Optional, List, Dict, Any

from .reflection import classify_cluster, validate_reflection


# ── text normalization + hashing (req 9) ──────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation for hashing."""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .!?,;:'\"")


def _hash(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


class ReflectionMemory:
    """
    Bounded store of reflection strategy lessons with hash dedup + clustering.

    Usage
    -----
    memory = ReflectionMemory(maxlen=20, top_k=5)
    memory.add_memory("Pick up the key before opening the door.", success=True)
    context_str = memory.get_context(current_cluster="KEY_FIRST")  # → planner prompt
    memory.save("log/reflections.json")
    """

    def __init__(
        self,
        maxlen: int = 20,
        success_only: bool = False,
        token_budget: int = 500,
        top_k: int = 5,
    ) -> None:
        self.maxlen = maxlen
        self.success_only = success_only
        self.token_budget = token_budget
        self.top_k = top_k

        self._buffer: List[Dict[str, Any]] = []      # entries (insertion order)
        self._index: Dict[str, Dict[str, Any]] = {}  # hash → entry (dedup)

        # Retrieval introspection (req: verify retrieval quality). Each
        # get_context() call records what it selected; set verbose_retrieval=True
        # to also print a one-line trace.
        self.verbose_retrieval: bool = False
        self._last_retrieval: List[Dict[str, Any]] = []
        self._stats: Dict[str, int] = {
            "total_added":        0,   # new unique entries created
            "total_merged":       0,   # duplicates merged (frequency bumped)
            "total_deduplicated": 0,   # alias of total_merged (back-compat logging)
            "total_filtered":     0,   # dropped by success-only filter
            "total_rejected":     0,   # dropped by coordinate/invalid guard
            "total_success":      0,
            "total_failure":      0,
            "total_retrievals":   0,   # get_context() calls that returned ≥1 entry
            "total_reflections_retrieved": 0,  # LOGGING-ONLY: Σ N reflections returned across calls
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _success_rate(entry: Dict[str, Any]) -> float:
        freq = max(int(entry.get("frequency", 1)), 1)
        return int(entry.get("success_count", 0)) / freq

    @staticmethod
    def _public_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Serializable view of an entry (drop internal bookkeeping keys)."""
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def _evict_if_needed(self) -> None:
        """Enforce maxlen, keeping the dedup index in sync on eviction."""
        while len(self._buffer) > self.maxlen:
            old = self._buffer.pop(0)
            h = old.get("_hash")
            if h and self._index.get(h) is old:
                del self._index[h]

    # ── write ─────────────────────────────────────────────────────────────────

    def add_memory(
        self,
        reflection: str,
        episode_id: Optional[int] = None,
        success: bool = False,
        ep_len: Optional[int] = None,
        total_reward: Optional[float] = None,
        cluster: Optional[str] = None,
        failure_type: Optional[str] = None,
    ) -> bool:
        """
        Add a reflection.

        Returns True only when a NEW unique entry is created. Returns False when
        the reflection is empty, filtered (success-only), rejected (coordinate /
        invalid content), or merged into an existing entry (frequency bumped).
        """
        if not reflection or not reflection.strip():
            return False
        text = reflection.strip()

        # Success-only mode: skip failure episodes.
        if self.success_only and not success:
            self._stats["total_filtered"] += 1
            return False

        # Coordinate / position / path guard (req 5, 6, 15): every write is
        # filtered, not just LLM-generated reflections.
        if not validate_reflection(text):
            self._stats["total_rejected"] += 1
            return False

        h = _hash(text)

        # Hash-based dedup (req 9): duplicates raise frequency, no new entry.
        if h in self._index:
            entry = self._index[h]
            entry["frequency"] += 1
            if success:
                entry["success_count"] += 1
                self._stats["total_success"] += 1
            else:
                entry["failure_count"] += 1
                self._stats["total_failure"] += 1
            entry["last_episode_id"] = episode_id
            entry["timestamp"] = time.time()
            self._stats["total_merged"]       += 1
            self._stats["total_deduplicated"] += 1
            return False

        # New unique entry (req 10): text, cluster, frequency, success stats.
        if cluster is None:
            cluster = classify_cluster(text, failure_type)
        entry = {
            "reflection":      text,
            "cluster":         cluster,
            "frequency":       1,
            "success_count":   1 if success else 0,
            "failure_count":   0 if success else 1,
            "episode_id":      episode_id,
            "last_episode_id": episode_id,
            "ep_len":          ep_len,
            "total_reward":    total_reward,
            "timestamp":       time.time(),
            "_hash":           h,
        }
        self._buffer.append(entry)
        self._index[h] = entry
        self._evict_if_needed()

        self._stats["total_added"] += 1
        if success:
            self._stats["total_success"] += 1
        else:
            self._stats["total_failure"] += 1
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def get_context(
        self,
        current_cluster: Optional[str] = None,
        max_entries: Optional[int] = None,
    ) -> str:
        """
        Return a formatted top-k block of strategy lessons for prompt injection.

        Ordering (req 11/12): cluster relevance to `current_cluster` first, then
        success rate, then frequency. The planner decides whether to act on these
        — this only supplies context (req 12, 13).
        """
        if not self._buffer:
            return ""

        k = max_entries if max_entries is not None else self.top_k

        def sort_key(e: Dict[str, Any]):
            cluster_match = 1 if (current_cluster and e.get("cluster") == current_cluster) else 0
            return (cluster_match, self._success_rate(e), int(e.get("frequency", 1)))

        ranked = sorted(self._buffer, key=sort_key, reverse=True)[:k]

        # Record the retrieval decision for inspection (req: retrieval quality).
        self._last_retrieval = [
            {
                "query_cluster":      current_cluster,
                "retrieved_cluster":  e.get("cluster", "GENERAL"),
                "reflection":         e["reflection"],
                "cluster_match":      bool(current_cluster and e.get("cluster") == current_cluster),
                "success_rate":       round(self._success_rate(e), 3),
                "frequency":          int(e.get("frequency", 1)),
            }
            for e in ranked
        ]
        if self._last_retrieval:
            self._stats["total_retrievals"] += 1
            self._stats["total_reflections_retrieved"] += len(self._last_retrieval)
        if self.verbose_retrieval and self._last_retrieval:
            top = self._last_retrieval[0]
            print(f"[MemRetrieval] query_cluster={current_cluster} → "
                  f"{top['retrieved_cluster']} (match={top['cluster_match']}, "
                  f"succ={top['success_rate']:.0%}, x{top['frequency']}): "
                  f"{top['reflection'][:70]}")

        header = "Strategic memory from past episodes (reusable strategies):"
        lines = [header]
        char_budget = self.token_budget * 4   # ~4 chars per token
        used = len(header)

        for e in ranked:
            sr = self._success_rate(e)
            label = (f"[{e.get('cluster', 'GENERAL')} | seen x{int(e.get('frequency', 1))} "
                     f"| succ {sr:.0%}]")
            line = f"  {label} {e['reflection']}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the n most recently touched entries as plain dicts (debug)."""
        return [self._public_entry(e) for e in self._buffer[-n:]]

    def cluster_distribution(self) -> Dict[str, int]:
        """Count of entries per cluster label."""
        dist: Dict[str, int] = {}
        for e in self._buffer:
            c = e.get("cluster", "GENERAL")
            dist[c] = dist.get(c, 0) + 1
        return dist

    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return len(self._buffer) > 0

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the buffer to JSON (internal keys dropped)."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            "config": {
                "maxlen":       self.maxlen,
                "success_only": self.success_only,
                "token_budget": self.token_budget,
                "top_k":        self.top_k,
            },
            "stats":   self._stats,
            "clusters": self.cluster_distribution(),
            "entries": [self._public_entry(e) for e in self._buffer],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[ReflectionMemory] Saved {len(self._buffer)} entries → {path}")

    def load(self, path: str) -> None:
        """
        Load a saved buffer. Backward-compatible with the legacy schema
        (entries with only {reflection, success, ...}): missing fields default to
        frequency=1, cluster inferred, and success/failure counts from `success`.
        """
        if not os.path.exists(path):
            print(f"[ReflectionMemory] No file at '{path}' — starting fresh.")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        for raw in data.get("entries", []):
            text = (raw.get("reflection") or "").strip()
            if not text:
                continue
            h = _hash(text)
            if h in self._index:
                continue
            success_legacy = bool(raw.get("success", False))
            entry = {
                "reflection":      text,
                "cluster":         raw.get("cluster") or classify_cluster(text),
                "frequency":       int(raw.get("frequency", 1)),
                "success_count":   int(raw.get("success_count",
                                               1 if success_legacy else 0)),
                "failure_count":   int(raw.get("failure_count",
                                               0 if success_legacy else 1)),
                "episode_id":      raw.get("episode_id"),
                "last_episode_id": raw.get("last_episode_id", raw.get("episode_id")),
                "ep_len":          raw.get("ep_len"),
                "total_reward":    raw.get("total_reward"),
                "timestamp":       raw.get("timestamp", time.time()),
                "_hash":           h,
            }
            self._buffer.append(entry)
            self._index[h] = entry
            loaded += 1
        self._evict_if_needed()

        for k, v in data.get("stats", {}).items():
            if isinstance(v, (int, float)):
                self._stats[k] = self._stats.get(k, 0) + v

        print(f"[ReflectionMemory] Loaded {loaded} entries from {path}")

    def clear(self) -> None:
        """Clear all entries and dedup state."""
        self._buffer.clear()
        self._index.clear()

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        s: Dict[str, Any] = dict(self._stats)
        s["cluster_distribution"] = self.cluster_distribution()
        return s

    def summary(self) -> str:
        """Human-readable one-line summary of buffer state."""
        s = self._stats
        return (
            f"ReflectionMemory: {len(self._buffer)}/{self.maxlen} entries | "
            f"added={s['total_added']} merged={s['total_merged']} "
            f"rejected={s['total_rejected']} filtered={s['total_filtered']} | "
            f"clusters={self.cluster_distribution()}"
        )
