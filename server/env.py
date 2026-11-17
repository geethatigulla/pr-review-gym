"""Core environment logic for Code Review Env."""
from typing import Any, Optional

from server.grader import grade_comments, strict_open_unit_score, unit_interval_score
from server.models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    ReviewComment,
    ResetResult,
    StepResult,
)
from server.tasks import get_scenario

MAX_COMMENTS = 10  # max review comments per episode
MAX_EPISODE_STEPS = 30  # max environment steps (turns) per episode


def _clip_step_reward(raw: float) -> float:
    """Clamp non-positive deltas to 0, then map to ``(0.0, 1.0)`` for wire/OpenEnv validators."""
    return strict_open_unit_score(unit_interval_score(max(0.0, raw)))


class CodeReviewEnv:
    """Stateful code review environment."""

    def __init__(self) -> None:
        self.task_name: str = "easy"
        self.scenario: dict[str, Any] = {}
        self.comments: list[ReviewComment] = []
        self.step_count: int = 0
        self.done: bool = False
        self.total_reward: float = 0.0
        self._last_feedback: Optional[str] = None
        self._truncated: bool = False

    def reset(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
        scenario_index: Optional[int] = None,
    ) -> ResetResult:
        """Reset the environment with a new scenario.

        Args:
            task_name: Canonical ``easy`` / ``medium`` / ``hard`` or legacy alias.
            seed: Optional RNG seed for scenario sampling.
            scenario_index: If set, pick this scenario index modulo pool size (deterministic).
        """
        self.task_name = task_name
        self.scenario = get_scenario(
            task_name, seed=seed, scenario_index=scenario_index
        )
        self.comments = []
        self.step_count = 0
        self.done = False
        expected = self.scenario.get("expected_issues", [])
        self.total_reward = float(grade_comments([], expected)["score"])
        self._last_feedback = None
        self._truncated = False
        return ResetResult(observation=self._make_observation())

    def step(self, action: CodeReviewAction) -> StepResult:
        """Process one agent action."""
        if self.done:
            return StepResult(
                observation=self._make_observation(),
                reward=_clip_step_reward(0.0),
                done=True,
                truncated=False,
                info={"message": "Episode already finished."},
            )

        if "expected_issues" not in self.scenario:
            self._last_feedback = (
                "No scenario loaded. Call POST /reset with the same "
                "`X-OpenEnv-Session-Id` header you use for POST /step, "
                "use WebSocket /ws, or run in-process (see baseline.py / inference.py)."
            )
            return StepResult(
                observation=self._make_observation(),
                reward=_clip_step_reward(0.0),
                done=False,
                truncated=False,
                info={"error": "not_reset", "message": self._last_feedback},
            )

        expected_issues: list = self.scenario["expected_issues"]

        self.step_count += 1
        step_start_total = self.total_reward
        raw_reward = 0.0
        self._last_feedback = None
        self._truncated = False

        if action.review_comment is not None and not action.done_reviewing:
            comment = action.review_comment
            is_duplicate = any(
                c.line_number == comment.line_number for c in self.comments
            )
            if is_duplicate:
                self._last_feedback = (
                    "Duplicate comment — you already commented on this line. "
                    "Find a DIFFERENT line or set done_reviewing to true."
                )
                raw_reward = -0.05
            else:
                self.comments.append(comment)
                grade = grade_comments(self.comments, expected_issues)
                new_score = float(grade["score"])
                previous_total = self.total_reward
                raw_reward = max(0.0, new_score - previous_total)
                self.total_reward = new_score

        if action.done_reviewing or len(self.comments) >= MAX_COMMENTS:
            self.done = True
        elif self.step_count >= MAX_EPISODE_STEPS:
            self.done = True
            self._truncated = True

        if self.done:
            final = grade_comments(self.comments, expected_issues)
            self.total_reward = float(final["score"])
            if raw_reward == 0.0:
                raw_reward = self.total_reward - step_start_total
            reward = _clip_step_reward(raw_reward)
            info: dict[str, Any] = {
                "comments_submitted": len(self.comments),
                "total_reward": round(self.total_reward, 4),
                "rl_reward": round(raw_reward, 4),
                "sub_score_1": float(final["sub_score_1"]),
                "sub_score_2": float(final["sub_score_2"]),
                "truncated": self._truncated,
            }
            return StepResult(
                observation=self._make_observation(),
                reward=reward,
                done=True,
                truncated=self._truncated,
                info=info,
            )

        reward = _clip_step_reward(raw_reward)
        last = grade_comments(self.comments, expected_issues)
        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=False,
            truncated=False,
            info={
                "comments_submitted": len(self.comments),
                "total_reward": round(self.total_reward, 4),
                "rl_reward": round(raw_reward, 4),
                "sub_score_1": float(last["sub_score_1"]),
                "sub_score_2": float(last["sub_score_2"]),
                "truncated": False,
            },
        )

    def state(self) -> CodeReviewState:
        """Return full internal state including expected_issue_count from scenario."""
        expected_issues = self.scenario.get("expected_issues", [])
        return CodeReviewState(
            task_name=self.task_name,
            scenario_id=self.scenario.get("id", ""),
            diff=self.scenario.get("diff", ""),
            file_name=self.scenario.get("file_name", ""),
            language=self.scenario.get("language", ""),
            pr_title=self.scenario.get("pr_title", ""),
            pr_description=self.scenario.get("pr_description", ""),
            comments=list(self.comments),
            step=self.step_count,
            done=self.done,
            total_reward=self.total_reward,
            expected_issue_count=len(expected_issues),
        )

    def _make_observation(self) -> CodeReviewObservation:
        """Build the agent-visible observation for the current step."""
        remaining_comments = MAX_COMMENTS - len(self.comments)
        return CodeReviewObservation(
            task_name=self.task_name,
            diff=self.scenario.get("diff", ""),
            file_name=self.scenario.get("file_name", ""),
            language=self.scenario.get("language", "python"),
            pr_title=self.scenario.get("pr_title", ""),
            pr_description=self.scenario.get("pr_description", ""),
            comments_so_far=list(self.comments),
            remaining_comments=max(0, remaining_comments),
            step_feedback=self._last_feedback,
            truncated=self._truncated,
        )
