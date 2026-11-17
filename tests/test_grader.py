"""Grader edge cases and score interval checks."""
from __future__ import annotations

import pytest

from server.grader import grade_comments, strict_open_unit_score, unit_interval_score
from server.models import ReviewComment


def test_unit_interval_score_bounds() -> None:
    assert unit_interval_score(-1.0) == 0.0
    assert unit_interval_score(2.0) == 1.0
    assert 0.0 <= unit_interval_score(0.3333333) <= 1.0


def test_grade_empty_expected_issues() -> None:
    g = grade_comments([], [])
    perfect = strict_open_unit_score(1.0)
    assert g["score"] == perfect
    assert g["sub_score_1"] == perfect
    assert g["sub_score_2"] == perfect


def test_grade_task_scores_strict_open_interval() -> None:
    g = grade_comments([], [])
    for key in ("score", "sub_score_1", "sub_score_2"):
        v = float(g[key])
        assert 0.0 < v < 1.0, f"{key}={v} must be strictly inside (0, 1)"


def test_grade_all_false_positives() -> None:
    issues = [
        {
            "line_number": 10,
            "category": "bug",
            "severity": "critical",
            "keywords": ["uniquekeywordxyz"],
            "synonym_group": None,
            "description": "n/a",
        }
    ]
    comments = [
        ReviewComment(
            line_number=99,
            comment="unrelated noise",
            severity="minor",
            category="style",
        )
    ]
    g = grade_comments(comments, issues)
    assert 0.0 < g["score"] < 1.0
    assert "sub_score_1" in g and "sub_score_2" in g


def test_grade_perfect_keyword_line() -> None:
    issues = [
        {
            "line_number": 5,
            "category": "bug",
            "severity": "major",
            "keywords": ["off-by-one"],
            "synonym_group": "off_by_one",
            "description": "fencepost",
        }
    ]
    comments = [
        ReviewComment(
            line_number=5,
            comment="off-by-one fence post boundary",
            severity="major",
            category="bug",
        )
    ]
    g = grade_comments(comments, issues)
    assert g["found_issues"] == 1
    assert g["score"] > 0.5
