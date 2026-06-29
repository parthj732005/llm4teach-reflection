"""
tests/test_reflection_validation.py

Unit tests for validate_reflection() hallucination guard in memory/reflection.py.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory.reflection import validate_reflection


class TestValidReflections:
    def test_clean_failure_reflection(self):
        text = "Agent approached the door without the key and failed. Next time: pick up the key first."
        assert validate_reflection(text) is True

    def test_clean_success_reflection(self):
        text = "Agent picked up the key and opened the door successfully."
        assert validate_reflection(text) is True

    def test_explore_strategy(self):
        text = "Agent explored the same area repeatedly. Next time: search unseen frontier areas."
        assert validate_reflection(text) is True

    def test_navigation_strategy(self):
        text = "Agent had the key but failed to navigate to the door. Navigate directly to the door after pickup."
        assert validate_reflection(text) is True


class TestCoordinateRejection:
    def test_bare_coordinates_rejected(self):
        assert validate_reflection("Agent moved to 7, 3 and toggled.") is False

    def test_parenthesis_coordinates_rejected(self):
        assert validate_reflection("Agent was at (3, 4) when it failed.") is False

    def test_row_column_rejected(self):
        assert validate_reflection("Agent was in row 3 col 7.") is False


class TestDirectionRejection:
    def test_north_rejected(self):
        assert validate_reflection("Move north to find the key.") is False

    def test_south_rejected(self):
        assert validate_reflection("Go south then east.") is False

    def test_upper_left_rejected(self):
        assert validate_reflection("Key is in the upper-left corner.") is False

    def test_corner_rejected(self):
        assert validate_reflection("Agent was near the corner.") is False


class TestSpeculationRejection:
    def test_probably_rejected(self):
        assert validate_reflection("The key is probably in the hidden room.") is False

    def test_hidden_rejected(self):
        assert validate_reflection("There is a hidden room behind the wall.") is False

    def test_secret_rejected(self):
        assert validate_reflection("Find the secret passage.") is False


class TestEdgeCases:
    def test_empty_string_rejected(self):
        assert validate_reflection("") is False

    def test_whitespace_only_rejected(self):
        assert validate_reflection("   ") is False

    def test_none_type(self):
        assert validate_reflection(None) is False
