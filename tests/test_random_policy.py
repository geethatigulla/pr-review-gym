"""Random policy on hard task must stay below rubric ceiling."""
from __future__ import annotations

import statistics

import pytest

from baseline import run_episode


def test_hard_random_policy_mean_below_rubric() -> None:
    scores = [run_episode("hard", 20_000 + i) for i in range(40)]
    mean = statistics.mean(scores)
    assert mean < 0.15, f"mean score {mean} should be < 0.15 for random policy"
