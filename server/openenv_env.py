"""OpenEnv Environment adapter: wraps CodeReviewEnv for HTTPEnvServer / WebSocket clients."""
from __future__ import annotations

from typing import Any, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from pydantic import Field

from server.env import CodeReviewEnv
from server.grader import strict_open_unit_score
from server.models import CodeReviewAction, CodeReviewObservation, ReviewComment


class CodeReviewActionAPI(Action):
    review_comment: Optional[ReviewComment] = None
    done_reviewing: bool = False


class CodeReviewObservationAPI(Observation):
    task_name: str = ""
    diff: str = ""
    file_name: str = ""
    language: str = ""
    pr_title: str = ""
    pr_description: str = ""
    comments_so_far: List[ReviewComment] = Field(default_factory=list)
    remaining_comments: int = 0
    step_feedback: Optional[str] = None
    truncated: bool = False


class CodeReviewServerState(State):
    task_name: str = ""
    scenario_id: str = ""
    diff: str = ""
    file_name: str = ""
    language: str = ""
    pr_title: str = ""
    pr_description: str = ""
    comments: List[ReviewComment] = Field(default_factory=list)
    done: bool = False
    total_reward: float = 0.0
    expected_issue_count: int = 0


class CodeReviewOpenEnvironment(Environment):
    """Single-session code review env compatible with OpenEnv servers and clients."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._inner = CodeReviewEnv()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="code-review-env",
            description=(
                "RL environment for reviewing Python and JavaScript PR diffs: find bugs, "
                "security issues, and quality problems with partial-credit grading."
            ),
            version="1.0.0",
            author="hackathon-team",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CodeReviewObservationAPI:
        """Load a scenario; pass ``task_name``, optional ``seed`` / ``scenario_index`` for reproducibility."""
        task_name = kwargs.get("task_name") or "easy"
        eff_seed = seed if seed is not None else kwargs.get("seed")
        scenario_index = kwargs.get("scenario_index")
        rr = self._inner.reset(
            task_name=task_name,
            seed=eff_seed,
            scenario_index=scenario_index,
        )
        # No transition yet; use strict-open (0, 1) so validators never see reward == 0.0.
        return self._wrap_obs(
            rr.observation, reward=strict_open_unit_score(0.0), done=False
        )

    def step(
        self,
        action: CodeReviewActionAPI,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservationAPI:
        """Apply one review action; observation includes ``reward``, ``done``, and ``truncated`` when applicable."""
        inner = CodeReviewAction(
            review_comment=action.review_comment,
            done_reviewing=action.done_reviewing,
        )
        sr = self._inner.step(inner)
        return self._wrap_obs(
            sr.observation, reward=sr.reward, done=sr.done, truncated=sr.truncated
        )

    @property
    def state(self) -> CodeReviewServerState:
        s = self._inner.state()
        return CodeReviewServerState(
            episode_id=s.scenario_id or None,
            step_count=s.step,
            task_name=s.task_name,
            scenario_id=s.scenario_id,
            diff=s.diff,
            file_name=s.file_name,
            language=s.language,
            pr_title=s.pr_title,
            pr_description=s.pr_description,
            comments=list(s.comments),
            done=s.done,
            total_reward=s.total_reward,
            expected_issue_count=s.expected_issue_count,
        )

    def _wrap_obs(
        self,
        obs: CodeReviewObservation,
        reward: Optional[float],
        done: bool,
        truncated: bool = False,
    ) -> CodeReviewObservationAPI:
        return CodeReviewObservationAPI(
            task_name=obs.task_name,
            diff=obs.diff,
            file_name=obs.file_name,
            language=obs.language,
            pr_title=obs.pr_title,
            pr_description=obs.pr_description,
            comments_so_far=list(obs.comments_so_far),
            remaining_comments=obs.remaining_comments,
            step_feedback=obs.step_feedback,
            truncated=truncated or obs.truncated,
            reward=reward,
            done=done,
        )
