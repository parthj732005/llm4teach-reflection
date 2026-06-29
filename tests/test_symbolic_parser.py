"""
tests/test_symbolic_parser.py

Unit tests for symbolic plan parser in utils/symbolic_parser.py.
No GPU, no Ollama, no MiniGrid required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.symbolic_parser import (
    validate_plan,
    parse_and_validate,
    normalize_plan,
    tokenize_plan,
    validate_action_token,
    VALID_OBJECTS,
)


class TestNormalizePlan:
    def test_lowercase(self):
        assert normalize_plan("GO TO <KEY>") == "go to <key>"

    def test_strip_whitespace(self):
        assert normalize_plan("  explore  ") == "explore"

    def test_collapse_spaces(self):
        assert normalize_plan("go  to  <key>") == "go to <key>"

    def test_strip_trailing_punctuation(self):
        assert normalize_plan("explore.") == "explore"


class TestTokenizePlan:
    def test_single_action(self):
        assert tokenize_plan("explore") == ["explore"]

    def test_multiple_actions(self):
        tokens = tokenize_plan("go to <key>, pick up <key>, open <door>")
        assert len(tokens) == 3

    def test_empty_plan(self):
        assert tokenize_plan("") == []


class TestValidateActionToken:
    def test_explore_valid(self):
        ok, err = validate_action_token("explore")
        assert ok

    def test_go_to_key_valid(self):
        ok, err = validate_action_token("go to <key>")
        assert ok

    def test_pick_up_key_valid(self):
        ok, err = validate_action_token("pick up <key>")
        assert ok

    def test_open_door_valid(self):
        ok, err = validate_action_token("open <door>")
        assert ok

    def test_colored_key_valid(self):
        ok, err = validate_action_token("go to <red key>")
        assert ok

    def test_colored_door_valid(self):
        ok, err = validate_action_token("open <blue door>")
        assert ok

    def test_invalid_object_rejected(self):
        ok, err = validate_action_token("go to <handle>")
        assert not ok

    def test_coordinate_rejected(self):
        ok, err = validate_action_token("go to position 7 3")
        assert not ok

    def test_unknown_verb_rejected(self):
        ok, err = validate_action_token("push <key>")
        assert not ok

    def test_empty_token_rejected(self):
        ok, err = validate_action_token("")
        assert not ok


class TestValidatePlan:
    def test_valid_full_plan(self):
        ok, errors = validate_plan("go to <key>, pick up <key>, open <door>")
        assert ok
        assert errors == []

    def test_explore_only_valid(self):
        ok, errors = validate_plan("explore")
        assert ok

    def test_empty_plan_invalid(self):
        ok, errors = validate_plan("")
        assert not ok

    def test_invalid_object_in_plan(self):
        ok, errors = validate_plan("go to <handle>")
        assert not ok
        assert len(errors) > 0

    def test_mixed_valid_invalid(self):
        ok, errors = validate_plan("explore, go to <handle>")
        assert not ok


class TestParseAndValidate:
    def test_valid_plan_returned(self):
        result = parse_and_validate("go to <key>, pick up <key>, open <door>")
        assert result is not None
        assert "go to <key>" in result

    def test_invalid_plan_returns_none(self):
        result = parse_and_validate("go to <handle>, push <button>")
        assert result is None

    def test_deduplication_applied(self):
        result = parse_and_validate("explore, explore, explore")
        assert result == "explore"

    def test_empty_returns_none(self):
        result = parse_and_validate("")
        assert result is None

    def test_mixed_drops_invalid_keeps_valid(self):
        result = parse_and_validate("explore, go to <handle>")
        assert result == "explore"


class TestValidObjects:
    def test_basic_objects_present(self):
        assert "key" in VALID_OBJECTS
        assert "door" in VALID_OBJECTS
        assert "goal" in VALID_OBJECTS

    def test_colored_objects_present(self):
        assert "red key" in VALID_OBJECTS
        assert "blue door" in VALID_OBJECTS

    def test_invalid_objects_absent(self):
        assert "handle" not in VALID_OBJECTS
        assert "button" not in VALID_OBJECTS
        assert "room" not in VALID_OBJECTS
