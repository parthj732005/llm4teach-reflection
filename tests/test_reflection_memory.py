"""
tests/test_reflection_memory.py

Unit tests for ReflectionMemory in memory/memory_buffer.py.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory.memory_buffer import ReflectionMemory


VALID_REFLECTION = "Agent approached the door without the key and toggled it repeatedly. Next time: pick up the key before approaching the door."
VALID_SUCCESS = "Agent picked up the key and navigated directly to the door, opening it successfully."
COORDINATE_REFLECTION = "Agent moved to position (7, 3) and tried the door."
DIRECTION_REFLECTION = "Agent should move north then east to find the key."


class TestReflectionMemoryAdd:
    def test_add_valid_reflection(self):
        mem = ReflectionMemory(maxlen=10)
        added = mem.add_memory(VALID_REFLECTION, success=False)
        assert added is True
        assert len(mem._buffer) == 1

    def test_add_success_reflection(self):
        mem = ReflectionMemory(maxlen=10)
        added = mem.add_memory(VALID_SUCCESS, success=True)
        assert added is True

    def test_reject_coordinate_reflection(self):
        mem = ReflectionMemory(maxlen=10)
        added = mem.add_memory(COORDINATE_REFLECTION, success=False)
        assert added is False
        assert len(mem._buffer) == 0

    def test_reject_direction_reflection(self):
        mem = ReflectionMemory(maxlen=10)
        added = mem.add_memory(DIRECTION_REFLECTION, success=False)
        assert added is False

    def test_reject_empty_reflection(self):
        mem = ReflectionMemory(maxlen=10)
        assert mem.add_memory("") is False
        assert mem.add_memory("   ") is False


class TestReflectionMemoryDeduplication:
    def test_duplicate_bumps_frequency(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.add_memory(VALID_REFLECTION, success=False)
        assert len(mem._buffer) == 1
        assert mem._buffer[0]["frequency"] == 2

    def test_near_duplicate_deduped(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.add_memory(VALID_REFLECTION.upper(), success=False)
        assert len(mem._buffer) == 1

    def test_distinct_reflections_stored_separately(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.add_memory(VALID_SUCCESS, success=True)
        assert len(mem._buffer) == 2


class TestReflectionMemoryEviction:
    def test_eviction_respects_maxlen(self):
        mem = ReflectionMemory(maxlen=3)
        for i in range(5):
            mem.add_memory(f"Agent failed step {i}. Next time: try approach {i}.", success=False)
        assert len(mem._buffer) <= 3

    def test_oldest_entry_evicted(self):
        mem = ReflectionMemory(maxlen=2)
        mem.add_memory("Agent failed step 0. Next time: try approach 0.", success=False)
        mem.add_memory("Agent failed step 1. Next time: try approach 1.", success=False)
        mem.add_memory("Agent failed step 2. Next time: try approach 2.", success=False)
        # Internal key is "reflection", not "text"
        texts = [e["reflection"] for e in mem._buffer]
        assert not any("step 0" in t for t in texts)


class TestReflectionMemoryRetrieval:
    def test_get_context_returns_string(self):
        mem = ReflectionMemory(maxlen=10, top_k=3)
        mem.add_memory(VALID_REFLECTION, success=False)
        ctx = mem.get_context()
        assert isinstance(ctx, str)

    def test_get_context_empty_when_no_entries(self):
        mem = ReflectionMemory(maxlen=10)
        ctx = mem.get_context()
        assert ctx == ""

    def test_get_context_with_cluster(self):
        mem = ReflectionMemory(maxlen=10, top_k=3)
        mem.add_memory(VALID_REFLECTION, success=False, cluster="KEY_FIRST")
        ctx = mem.get_context(current_cluster="KEY_FIRST")
        assert len(ctx) > 0

    def test_stats_track_retrievals(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.get_context()
        assert mem._stats["total_retrievals"] >= 1


class TestReflectionMemoryStats:
    def test_stats_track_success_failure(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.add_memory(VALID_SUCCESS, success=True)
        assert mem._stats["total_failure"] >= 1
        assert mem._stats["total_success"] >= 1

    def test_stats_track_rejected(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(COORDINATE_REFLECTION, success=False)
        assert mem._stats["total_rejected"] >= 1

    def test_stats_track_deduplicated(self):
        mem = ReflectionMemory(maxlen=10)
        mem.add_memory(VALID_REFLECTION, success=False)
        mem.add_memory(VALID_REFLECTION, success=False)
        assert mem._stats["total_merged"] >= 1
