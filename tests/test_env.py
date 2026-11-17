"""Tests for ``CodeReviewEnv`` reset/step, bounds, and truncation."""
from __future__ import annotations

import pytest

from server.env import MAX_EPISODE_STEPS, CodeReviewEnv
from server.models import CodeReviewAction, ReviewComment


def test_reset_easy_medium_hard() -> None:
    for task in ("easy", "medium", "hard"):
        env = CodeReviewEnv()
        rr = env.reset(task_name=task, seed=42, scenario_index=0)
        assert rr.observation.task_name == task
        assert rr.observation.diff
        assert "expected_issues" in env.scenario


def test_legacy_task_alias() -> None:
    env = CodeReviewEnv()
    env.reset(task_name="find-bugs", seed=1, scenario_index=0)
    assert env.scenario["id"]


def test_reset_total_reward_strict_open_interval() -> None:
    env = CodeReviewEnv()
    env.reset("easy", seed=0, scenario_index=0)
    tr = env.state().total_reward
    assert 0.0 < tr < 1.0


def test_step_rewards_bounded() -> None:
    env = CodeReviewEnv()
    env.reset("easy", seed=0, scenario_index=0)
    for _ in range(10):
        r = env.step(
            CodeReviewAction(
                review_comment=ReviewComment(
                    line_number=1,
                    comment="the and mostly perhaps wording clarity",
                    severity="minor",
                    category="style",
                )
            )
        )
        assert 0.0 < r.reward < 1.0
        if r.done:
            break


def test_truncation_max_steps() -> None:
    env = CodeReviewEnv()
    env.reset("easy", seed=99, scenario_index=0)
    for _ in range(MAX_EPISODE_STEPS + 2):
        r = env.step(CodeReviewAction(done_reviewing=False))
        if r.done:
            assert r.truncated is True
            assert r.info.get("truncated") is True
            break
    else:
        pytest.fail("expected truncation before loop exhaustion")


def test_done_reviewing_sets_done() -> None:
    env = CodeReviewEnv()
    env.reset("easy", seed=1, scenario_index=0)
    r = env.step(CodeReviewAction(done_reviewing=True))
    assert r.done
    assert r.truncated is False
