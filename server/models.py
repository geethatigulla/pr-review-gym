"""Pydantic models for the Code Review environment."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ReviewComment(BaseModel):
    """A single review comment submitted by the agent."""
    line_number: int = Field(..., description="Line number the comment refers to")
    comment: str = Field(..., description="The review comment text")
    severity: str = Field(
        "minor",
        description="Issue severity",
        pattern="^(critical|major|minor|suggestion)$",
    )
    category: str = Field(
        "bug",
        description="Issue category",
        pattern="^(bug|security|performance|style|logic)$",
    )


class CodeReviewAction(BaseModel):
    """Action the agent takes each step."""
    review_comment: Optional[ReviewComment] = Field(
        None, description="A review comment, or null if done"
    )
    done_reviewing: bool = Field(
        False, description="Set true when the agent is finished reviewing"
    )


class CodeReviewObservation(BaseModel):
    """What the agent sees each step."""
    task_name: str = Field(..., description="Name of the review task / scenario type")
    diff: str = Field(..., description="Unified diff of the change under review")
    file_name: str = Field(..., description="Primary file path referenced by the diff")
    language: str = Field(..., description="Programming language of the changed file")
    pr_title: str = Field(..., description="Pull request title")
    pr_description: str = Field(..., description="Pull request description body")
    comments_so_far: List[ReviewComment] = Field(
        default_factory=list,
        description="Review comments the agent has submitted this episode",
    )
    remaining_comments: int = Field(
        ..., description="How many additional comments the agent may still submit"
    )
    step_feedback: Optional[str] = Field(
        None, description="Optional feedback on the last action (e.g., duplicate comment)"
    )
    truncated: bool = Field(
        False,
        description="True when the episode ended due to step budget (timeout), not natural completion",
    )


class CodeReviewState(BaseModel):
    """Full internal state for state() endpoint."""
    task_name: str = Field(..., description="Name of the review task / scenario type")
    scenario_id: str = Field(..., description="Unique identifier for the loaded scenario")
    diff: str = Field(..., description="Unified diff of the change under review")
    file_name: str = Field(..., description="Primary file path referenced by the diff")
    language: str = Field(..., description="Programming language of the changed file")
    pr_title: str = Field(..., description="Pull request title")
    pr_description: str = Field(..., description="Pull request description body")
    comments: List[ReviewComment] = Field(
        ..., description="All review comments submitted so far this episode"
    )
    step: int = Field(..., description="Environment step count since last reset")
    done: bool = Field(..., description="Whether the episode has ended")
    total_reward: float = Field(..., description="Cumulative reward for the episode")
    expected_issue_count: int = Field(
        ..., description="Number of labeled issues in the scenario rubric"
    )


class StepResult(BaseModel):
    """Returned by step()."""
    observation: CodeReviewObservation = Field(
        ..., description="Observation after applying the action"
    )
    reward: float = Field(..., description="Reward for this transition")
    done: bool = Field(..., description="Whether the episode is finished")
    truncated: bool = Field(
        False,
        description="True if the episode stopped because of max steps (timeout budget)",
    )
    info: dict = Field(
        default_factory=dict,
        description="Auxiliary diagnostics (scores, messages, etc.)",
    )


class ResetResult(BaseModel):
    """Returned by reset()."""
    observation: CodeReviewObservation = Field(
        ..., description="Initial observation after loading a scenario"
    )