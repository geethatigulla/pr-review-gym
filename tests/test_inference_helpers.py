"""Offline tests for the LLM baseline's parsing and action-mapping helpers."""
from __future__ import annotations

import pytest

from inference import (
    _client_side_duplicate,
    _decision_to_action,
    _parse_confidence,
    _parse_llm_json,
    add_line_numbers,
    word_overlap_ratio,
)


def test_add_line_numbers() -> None:
    assert add_line_numbers("a\nb") == "   1 | a\n   2 | b"


@pytest.mark.parametrize(
    "raw",
    [
        '{"done_reviewing": true}',
        '```json\n{"done_reviewing": true}\n```',
        '```\n{"done_reviewing": true}\n```',
    ],
)
def test_parse_llm_json_strips_fences(raw: str) -> None:
    assert _parse_llm_json(raw) == {"done_reviewing": True}


def test_parse_llm_json_non_object_means_done() -> None:
    assert _parse_llm_json("[1, 2]") == {"done_reviewing": True}


def test_decision_to_action_comment_and_done() -> None:
    action = _decision_to_action(
        {"line_number": "4", "comment": "bug", "severity": "major", "category": "logic"}
    )
    assert action.review_comment is not None
    assert action.review_comment.line_number == 4
    assert _decision_to_action({"done_reviewing": True}).done_reviewing is True


def test_decision_to_action_invalid_enum_falls_back_to_done() -> None:
    action = _decision_to_action(
        {"line_number": 1, "comment": "x", "severity": "huge", "category": "bug"}
    )
    assert action.done_reviewing is True


def test_confidence_parsing() -> None:
    base = {"line_number": 1, "comment": "c", "category": "bug"}
    assert _parse_confidence({**base, "confidence": 0.4}) == 0.4
    assert _parse_confidence({**base, "confidence": 7}) == 1.0
    assert _parse_confidence({**base, "confidence": "nan?"}) == 1.0
    assert _parse_confidence(base) == 1.0


def test_duplicate_detection() -> None:
    snaps = [(4, "bug", "off by one in the range upper bound")]
    assert _client_side_duplicate(4, "bug", "anything", snaps)
    assert _client_side_duplicate(9, "bug", "off by one in the range upper bound", snaps)
    assert not _client_side_duplicate(9, "bug", "division by zero on empty list", snaps)
    assert word_overlap_ratio("", "") == 1.0
