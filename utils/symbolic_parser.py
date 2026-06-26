#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
utils/symbolic_parser.py — Strict symbolic plan grammar validator for LLM4Teach

BUG FIX: Explicit object whitelist — <handle>, <button>, <room>, coordinates,
and any entity not in the allowed set are now REJECTED with explicit logs.

Formal grammar
--------------
PLAN   := ACTION ("," ACTION)*
ACTION := "explore"
        | "go to" OBJECT
        | "pick up" OBJECT
        | "open" OBJECT
        | "drop" OBJECT

OBJECT := "<" NOUN ">"
NOUN   := BASE_OBJ | COLOR " " BASE_OBJ
COLOR  := "red" | "green" | "blue" | "yellow" | "purple" | "grey"
BASE_OBJ := "key" | "door" | "goal" | "door1" | "door2"

ALLOWED objects (exact whitelist — nothing else passes):
  <key>  <door>  <goal>
  <red key>  <green key>  <blue key>  <yellow key>  <purple key>  <grey key>
  <red door>  <green door>  <blue door>  <yellow door>  <purple door>  <grey door>
  <door1>  <door2>

REJECTED (with explicit log):
  <handle>  <button>  <room>  <exit>  (7,3)  any free-form text
"""

import re
from typing import List, Optional, Tuple


# ── Strict object whitelist ───────────────────────────────────────────────────

VALID_COLORS      = frozenset(["red", "green", "blue", "yellow", "purple", "grey"])
VALID_BASE_OBJECTS = frozenset(["key", "door", "goal", "door1", "door2"])
VALID_ACTION_VERBS = frozenset(["explore", "go to", "pick up", "open", "drop"])


def _build_valid_objects() -> frozenset:
    """Build complete whitelist of allowed <object> names."""
    objs = set(VALID_BASE_OBJECTS)
    for color in VALID_COLORS:
        for base in ("key", "door"):
            objs.add(f"{color} {base}")
    return frozenset(objs)


VALID_OBJECTS = _build_valid_objects()


def _is_valid_object(name: str) -> bool:
    """Return True iff name is in the strict object whitelist."""
    return name.strip().lower() in VALID_OBJECTS


# ── Normalizer ────────────────────────────────────────────────────────────────

def normalize_plan(plan: str) -> str:
    text = plan.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text).strip()
    return text


# ── Token parser ──────────────────────────────────────────────────────────────

_OBJECT_RE = re.compile(r"<([^>]+)>")


def tokenize_plan(plan: str) -> List[str]:
    tokens = []
    for part in plan.split(","):
        part = normalize_plan(part)
        if part:
            tokens.append(part)
    return tokens


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_action_token(token: str) -> Tuple[bool, str]:
    """
    Validate one action token against the strict grammar.
    Returns (is_valid, error_message).

    BUG FIX: Every <object> is checked against VALID_OBJECTS.
    Unknown objects like <handle>, <button>, <room> are REJECTED
    with an explicit [SymbolicParser] log line.
    """
    token = normalize_plan(token)

    if not token:
        return False, "empty token"

    # Coordinates are never allowed
    if re.search(r"\d", token):
        return False, f"coordinates not allowed: '{token}'"

    # 'explore' needs no object
    if token == "explore":
        return True, ""

    # Must have an <object> reference
    obj_match = _OBJECT_RE.search(token)
    if obj_match is None:
        return False, f"missing <object> in token: '{token}'"

    obj_name = obj_match.group(1).strip().lower()

    # STRICT whitelist check — reject anything not in VALID_OBJECTS
    if not _is_valid_object(obj_name):
        # Explicit rejection log as required
        print(f"[SymbolicParser] Rejected invalid object: <{obj_name}>")
        return False, (
            f"<{obj_name}> is not a valid symbolic object. "
            f"Allowed: {sorted(VALID_OBJECTS)}"
        )

    # Extract and validate the action verb
    verb_part = token[: token.index("<")].strip()
    matched = any(
        verb_part == v or verb_part.endswith(v)
        for v in VALID_ACTION_VERBS
        if v != "explore"
    )
    if not matched:
        return False, f"unknown action verb '{verb_part}' in: '{token}'"

    return True, ""


def validate_plan(plan: str) -> Tuple[bool, List[str]]:
    """
    Validate a full plan string. Returns (is_valid, errors).
    is_valid is True only when errors is empty.
    """
    if not plan or not plan.strip():
        return False, ["empty plan"]

    tokens = tokenize_plan(plan)
    if not tokens:
        return False, ["no tokens after tokenization"]

    errors = []
    for tok in tokens:
        ok, err = validate_action_token(tok)
        if not ok:
            errors.append(err)

    return len(errors) == 0, errors


def deduplicate_plan(tokens: List[str]) -> List[str]:
    seen: set = set()
    result = []
    for tok in tokens:
        norm = normalize_plan(tok)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def parse_and_validate(raw_plan: str) -> Optional[str]:
    """
    Full pipeline: normalize → tokenize → validate → deduplicate → reassemble.

    BUG FIX: invalid tokens (including <handle>, coordinates, unknown objects)
    are dropped with explicit log lines. Returns None if no valid tokens remain.
    Callers MUST fall back to 'explore' on None.
    """
    if not raw_plan:
        return None

    normed = normalize_plan(raw_plan)
    tokens = tokenize_plan(normed)

    valid_tokens = []
    for tok in tokens:
        ok, err = validate_action_token(tok)
        if ok:
            valid_tokens.append(tok)
        else:
            print(f"[SymbolicParser] Dropped invalid token '{tok}': {err}")

    if not valid_tokens:
        print(f"[SymbolicParser] Plan '{raw_plan[:80]}' → no valid tokens → rejected")
        return None

    deduped = deduplicate_plan(valid_tokens)
    return ", ".join(deduped)


def strict_parse(raw_plan: str) -> Optional[str]:
    """
    Public entry point for strict validation.
    Returns validated plan string or None (caller falls back to 'explore').
    """
    return parse_and_validate(raw_plan)
