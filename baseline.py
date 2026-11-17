#!/usr/bin/env python3
"""Deterministic random-action baseline over in-process ``CodeReviewEnv``.

Prints per-task final rubric scores with **fixed seeds** (no network / no API calls).
For the HTTP API server, run: ``uvicorn server.app:app --host 0.0.0.0 --port 8000``.
"""
from __future__ import annotations

import random
from typing import List

from server.env import CodeReviewEnv
from server.models import CodeReviewAction, ReviewComment

# Benign vocabulary — avoids accidental hits on security / bug synonym lists.
_SAFE_WORDS: List[str] = [
    "the",
    "and",
    "mostly",
    "perhaps",
    "wording",
    "clarity",
    "structure",
    "layout",
    "commentary",
    "paragraph",
]


def _random_action(rng: random.Random, max_line: int) -> CodeReviewAction:
    line_number = rng.randint(1, max(1, max_line))
    text = " ".join(rng.choice(_SAFE_WORDS) for _ in range(10))
    return CodeReviewAction(
        review_comment=ReviewComment(
            line_number=line_number,
            comment=text,
            severity=rng.choice(["minor", "suggestion"]),
            category=rng.choice(["style", "bug"]),
        ),
        done_reviewing=False,
    )


def run_episode(task_name: str, seed: int, n_random_steps: int = 10) -> float:
    """Run up to ``n_random_steps`` random comments then ``done_reviewing``."""
    rng = random.Random(seed)
    env = CodeReviewEnv()
    env.reset(task_name=task_name, seed=seed, scenario_index=0)
    max_line = len(env.scenario.get("diff", "").splitlines())
    for _ in range(n_random_steps):
        result = env.step(_random_action(rng, max_line))
        if result.done:
            return float(env.state().total_reward)
    final = env.step(CodeReviewAction(done_reviewing=True))
    assert final.done
    return float(env.state().total_reward)


def main() -> None:
    # Fixed seeds per rubric / README reproducibility.
    seeds = {"easy": 10_001, "medium": 10_002, "hard": 10_003}
    print("code-review-env baseline (random actions, fixed seeds)", flush=True)
    for task, sd in seeds.items():
        score = run_episode(task, sd)
        print(f"{task}: score={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
