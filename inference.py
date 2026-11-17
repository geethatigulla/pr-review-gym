"""
Baseline inference script for Code Review OpenEnv.
Runs an LLM agent against the code review environment.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from server.env import CodeReviewEnv
from server.grader import grade_comments, strict_open_unit_score, unit_interval_score
from server.models import CodeReviewAction, ReviewComment
from server.tasks import TASKS as SCENARIO_REGISTRY

# --- Environment variables (read with os.getenv) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME")
MIN_SUBMIT_CONFIDENCE = float(os.getenv("MIN_SUBMIT_CONFIDENCE", "0") or 0)
MAX_CONFIDENCE_RETRIES = 2
TARGET_IMPACT_ISSUES_COMPREHENSIVE = 5


def _resolve_api_key() -> str:
    hf = os.getenv("HF_TOKEN")
    key = hf or os.getenv("API_KEY")
    if not key:
        raise ValueError(
            "HF_TOKEN or API_KEY must be set (no default for HF_TOKEN; API_KEY is used as fallback)."
        )
    return key


# --- Config constants ---
BENCHMARK = "code-review-env"
MAX_STEPS = 12
MAX_TOKENS = 800
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a senior software engineer performing a thorough code review on a pull request.

STRATEGY:
1. First, read the entire diff carefully before commenting.
2. Prioritize issues by severity: find CRITICAL issues first, then MAJOR, then MINOR.
3. Look for these categories: bugs, security vulnerabilities, logic errors, performance issues, style problems.
4. Each comment must address a DIFFERENT issue on a DIFFERENT line. Never repeat yourself.
5. Be precise about line numbers — point to the exact line with the problem.
6. For the **hard** task (full review; the user message states the task id), run structured passes before done_reviewing: (1) security and trust boundaries, (2) correctness including async/Promise mistakes and validation, (3) business logic and order of operations, (4) performance (N+1 I/O, unnecessary full sorts), (5) error handling and opaque catches. Keep surfacing distinct issues on new lines until the comment budget is exhausted or you truly have nothing credible left — do not stop after a single finding.
7. When you cannot find any more distinct issues, use done_reviewing.
8. Include "confidence" (0.0–1.0) for each comment: how sure you are that this is a real, distinct issue worth raising.

COMMON ISSUES TO LOOK FOR:
- Security: hardcoded secrets, SQL injection, XSS, path traversal, weak crypto, SSRF
- Bugs: wrong variable names, off-by-one errors, missing return statements, type errors
- Logic: missing validation, incorrect order of operations, edge cases not handled
- Performance: N+1 queries, unnecessary sorting, loading all data into memory

RESPONSE FORMAT — respond with ONLY a raw JSON object, no markdown, no backticks, no explanation:
{
  "line_number": <int>,
  "comment": "<explain what's wrong and how to fix it>",
  "severity": "<critical|major|minor|suggestion>",
  "category": "<bug|security|performance|style|logic>",
  "confidence": <float 0.0-1.0>,
  "done_reviewing": false
}

When finished, respond with ONLY:
{"done_reviewing": true}
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    short = (action or "")[:80].replace("\n", " ")
    err = "null" if error is None else error.replace("\n", " ")
    print(
        f"[STEP] step={step} action={short} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}",
        flush=True,
    )


def _rl_reward_from_step_result(result: Any) -> Optional[float]:
    info = getattr(result, "info", None)
    if isinstance(info, dict) and "rl_reward" in info:
        try:
            return float(info["rl_reward"])
        except (TypeError, ValueError):
            return None
    return None


def _observation_as_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def add_line_numbers(diff_text: str) -> str:
    """Prepend 1-based line numbers to each line of the diff for clearer references."""
    lines = diff_text.splitlines()
    numbered: List[str] = []
    for i, line in enumerate(lines, start=1):
        numbered.append(f"{i:4d} | {line}")
    return "\n".join(numbered)


_WORD_TOKENS_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _word_set(text: str) -> set[str]:
    return set(_WORD_TOKENS_RE.findall(text.lower()))


def word_overlap_ratio(a: str, b: str) -> float:
    """Jaccard similarity on word token sets: |A ∩ B| / |A ∪ B|."""
    sa, sb = _word_set(a), _word_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _snapshots_from_obs(obs_dict: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """(line_number, category, comment text) for comments already accepted by the env."""
    out: List[Tuple[int, str, str]] = []
    for c in obs_dict.get("comments_so_far") or []:
        if isinstance(c, dict):
            ln = int(c.get("line_number", 0))
            cat = str(c.get("category", ""))
            txt = str(c.get("comment", ""))
        else:
            ln, cat, txt = c.line_number, c.category, c.comment
        out.append((ln, cat, txt))
    return out


def _client_side_duplicate(
    line_number: int,
    category: str,
    comment_text: str,
    snapshots: List[Tuple[int, str, str]],
    overlap_threshold: float = 0.8,
) -> bool:
    for ln, _cat, prev_txt in snapshots:
        if ln == line_number:
            return True
        if word_overlap_ratio(comment_text, prev_txt) > overlap_threshold:
            return True
    return False


def _scenario_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sid = state.get("scenario_id") or ""
    if not sid:
        return None
    for scenarios in SCENARIO_REGISTRY.values():
        for sc in scenarios:
            if sc.get("id") == sid:
                return sc
    return None


def _comments_for_grade(raw: Any) -> List[ReviewComment]:
    out: List[ReviewComment] = []
    for c in raw or []:
        if isinstance(c, dict):
            try:
                out.append(ReviewComment(**c))
            except (ValidationError, TypeError, ValueError):
                continue
        elif isinstance(c, ReviewComment):
            out.append(c)
    return out


async def _fetch_state_dict(env_wrapper: Any) -> Optional[Dict[str, Any]]:
    getter = getattr(env_wrapper, "state", None)
    if getter is None:
        return None
    st = await getter()
    if hasattr(st, "model_dump"):
        return st.model_dump()
    if isinstance(st, dict):
        return st
    return None


async def _issues_progress(
    env_wrapper: Any, obs_dict: Dict[str, Any]
) -> Tuple[int, int]:
    comments = _comments_for_grade(obs_dict.get("comments_so_far"))
    state_dict = await _fetch_state_dict(env_wrapper)
    scenario: Optional[Dict[str, Any]] = None
    if state_dict:
        scenario = _scenario_from_state(state_dict)
    if scenario is None:
        inner = getattr(env_wrapper, "env", None)
        if inner is not None and hasattr(inner, "scenario"):
            scenario = getattr(inner, "scenario", None)
    if scenario and scenario.get("expected_issues") is not None:
        g = grade_comments(comments, scenario["expected_issues"])
        return int(g["found_issues"]), int(g["total_issues"])
    total = 0
    if state_dict:
        total = int(state_dict.get("expected_issue_count", 0) or 0)
    return 0, total


def _task_impact_guidance(task_name: str) -> str:
    if task_name == "hard":
        return (
            f"Aim for at most the {TARGET_IMPACT_ISSUES_COMPREHENSIVE} most impactful, distinct issues "
            "— prioritize severity and real risk over minor nits."
        )
    if task_name == "medium":
        return (
            "Prioritize the highest-risk security issues first; prefer fewer, sharper findings "
            "over many low-signal comments."
        )
    if task_name == "easy":
        return (
            "Focus on clear, demonstrable bugs first; report the worst problems before smaller issues."
        )
    return "Find issues one at a time, starting with the most critical."


def build_first_user_message(obs_dict: Dict[str, Any]) -> str:
    fname = obs_dict.get("file_name", "")
    lang = obs_dict.get("language", "")
    title = obs_dict.get("pr_title", "")
    desc = obs_dict.get("pr_description", "")
    task_name = str(obs_dict.get("task_name", "") or "")
    diff = obs_dict.get("diff", "") or ""
    numbered = add_line_numbers(diff)
    impact = _task_impact_guidance(task_name)
    comprehensive_block = ""
    if task_name == "hard":
        comprehensive_block = (
            "\n\nTask type: **hard** (full pass expected).\n"
            "After each comment you will get feedback — continue with a different line and category "
            "until security, logic, performance, and robustness have each been considered, "
            "or you have no remaining comment budget."
        )
    return (
        "Review this pull request.\n\n"
        f"Task: {task_name or '(unspecified)'}\n"
        f"File: {fname} ({lang})\n"
        f"PR Title: {title}\n"
        f"PR Description: {desc}\n\n"
        "Code to review:\n"
        f"{numbered}\n\n"
        "There are multiple issues in this code. Find them one at a time, starting with the most critical.\n"
        f"{impact}{comprehensive_block}"
    )


def budget_reminder(step: int, max_steps: int, obs_dict: Dict[str, Any]) -> str:
    remaining = int(obs_dict.get("remaining_comments", 0) or 0)
    steps_left = max(0, max_steps - step + 1)
    task_name = str(obs_dict.get("task_name", "") or "")
    extra = ""
    if task_name == "hard" and remaining > 0:
        extra = (
            " For the hard task, prefer another comment on a new issue if you still have budget "
            "and have not yet covered security, logic, performance, and error-handling angles."
        )
    return (
        f"Budget: review step {step} of at most {max_steps} "
        f"({steps_left} step(s) left including this one). "
        f"You may submit at most {remaining} more comment(s) before the comment cap."
        f"{extra}"
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)


def _parse_llm_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    else:
        text = _FENCE_RE.sub("", text)
    decision = json.loads(text)
    if not isinstance(decision, dict):
        return {"done_reviewing": True}
    return decision


def call_llm_messages(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    """Single completion; `messages` must not yet include this assistant turn."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        decision = _parse_llm_json(raw)
        return decision, raw
    except Exception as exc:  # network/API/JSON errors end the episode gracefully
        print(f"[WARN] LLM call failed ({type(exc).__name__}): {exc}", file=sys.stderr, flush=True)
        return {"done_reviewing": True}, '{"done_reviewing":true}'


def _is_comment_decision(decision: Dict[str, Any]) -> bool:
    if decision.get("done_reviewing"):
        return False
    return "line_number" in decision and "category" in decision and "comment" in decision


def _parse_confidence(decision: Dict[str, Any]) -> float:
    """Self-reported confidence for comment actions; missing field => 1.0 (no gating)."""
    if not _is_comment_decision(decision):
        return 1.0
    v = decision.get("confidence")
    if v is None:
        return 1.0
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, c))


def obtain_decision_with_client_retries(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float,
    snapshots: List[Tuple[int, str, str]],
) -> Dict[str, Any]:
    """
    Call the LLM; append assistant turns to `messages`.
    If the model proposes a client-side duplicate, nudge once; if it still duplicates,
    append the second assistant reply and return done_reviewing.
    """
    decision, raw = call_llm_messages(client, messages, temperature)
    messages.append({"role": "assistant", "content": raw})

    if not _is_comment_decision(decision):
        return decision

    try:
        ln = int(decision["line_number"])
        cat = str(decision["category"])
        comment_txt = str(decision.get("comment", ""))
    except (TypeError, ValueError, KeyError):
        return decision

    if not _client_side_duplicate(ln, cat, comment_txt, snapshots):
        return decision

    messages.append(
        {
            "role": "user",
            "content": (
                "You're about to repeat yourself (same line or nearly the same issue). "
                "Find a DIFFERENT line and a DIFFERENT issue."
            ),
        },
    )
    snapshots_plus_rejected = list(snapshots)
    snapshots_plus_rejected.append((ln, cat, comment_txt))

    decision2, raw2 = call_llm_messages(client, messages, temperature)
    messages.append({"role": "assistant", "content": raw2})

    if not _is_comment_decision(decision2):
        return decision2

    try:
        ln2 = int(decision2["line_number"])
        cat2 = str(decision2["category"])
        txt2 = str(decision2.get("comment", ""))
    except (TypeError, ValueError, KeyError):
        return {"done_reviewing": True}

    if _client_side_duplicate(ln2, cat2, txt2, snapshots_plus_rejected):
        return {"done_reviewing": True}
    return decision2


def _decision_to_action(decision: Dict[str, Any]) -> CodeReviewAction:
    if decision.get("done_reviewing"):
        return CodeReviewAction(done_reviewing=True)
    try:
        return CodeReviewAction(
            review_comment=ReviewComment(
                line_number=int(decision.get("line_number", 1)),
                comment=str(decision.get("comment", "")),
                severity=str(decision.get("severity", "minor")),
                category=str(decision.get("category", "bug")),
            ),
            done_reviewing=False,
        )
    except (ValidationError, TypeError, ValueError):
        return CodeReviewAction(done_reviewing=True)


def _episode_score(rewards: List[float]) -> float:
    total = sum(rewards) if rewards else 0.0
    return strict_open_unit_score(unit_interval_score(total))


class EnvWrapper:
    """Async wrapper around server.env.CodeReviewEnv."""

    def __init__(self) -> None:
        self.env = CodeReviewEnv()

    async def reset(self, task_name: str = "easy"):
        return await asyncio.to_thread(self.env.reset, task_name=task_name)

    async def step(self, action: CodeReviewAction):
        return await asyncio.to_thread(self.env.step, action)

    async def state(self) -> Dict[str, Any]:
        return await asyncio.to_thread(lambda: self.env.state().model_dump())

    async def close(self) -> None:
        return None


async def run_task(client: OpenAI, env_wrapper: Any, task_name: str) -> Tuple[float, int, int]:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    consecutive_nonpositive = 0
    last_found, last_total = 0, 0

    try:
        reset_out = await env_wrapper.reset(task_name=task_name)
        obs = getattr(reset_out, "observation", reset_out)
        obs_dict = _observation_as_dict(obs)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_first_user_message(obs_dict)},
        ]

        for step in range(1, MAX_STEPS + 1):
            temp = 0.2 if step <= 3 else 0.5
            snapshots = _snapshots_from_obs(obs_dict)

            messages.append(
                {"role": "user", "content": budget_reminder(step, MAX_STEPS, obs_dict)}
            )

            decision = obtain_decision_with_client_retries(
                client, messages, temp, snapshots
            )

            conf_attempts = 0
            while (
                MIN_SUBMIT_CONFIDENCE > 0.0
                and _is_comment_decision(decision)
                and _parse_confidence(decision) < MIN_SUBMIT_CONFIDENCE
                and conf_attempts < MAX_CONFIDENCE_RETRIES
            ):
                print(
                    f"[CONF] step={step} confidence={_parse_confidence(decision):.2f} "
                    f"< {MIN_SUBMIT_CONFIDENCE:.2f}; re-querying",
                    flush=True,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last proposal had confidence below {MIN_SUBMIT_CONFIDENCE:.2f}. "
                            "Only submit a comment if you are at least that confident it is a real, "
                            "distinct issue; otherwise respond with "
                            '{"done_reviewing": true} or propose a stronger issue with honest confidence.'
                        ),
                    }
                )
                decision = obtain_decision_with_client_retries(
                    client, messages, temp, snapshots
                )
                conf_attempts += 1

            if (
                MIN_SUBMIT_CONFIDENCE > 0.0
                and _is_comment_decision(decision)
                and _parse_confidence(decision) < MIN_SUBMIT_CONFIDENCE
            ):
                print(
                    f"[CONF] step={step} giving up after retries; forcing done_reviewing",
                    flush=True,
                )
                decision = {"done_reviewing": True}

            action = _decision_to_action(decision)
            action_preview = json.dumps(decision, separators=(",", ":"))

            step_error: Optional[str] = None
            reward = 0.0
            done = False
            try:
                result = await env_wrapper.step(action)
                obs = getattr(result, "observation", None)
                reward = float(result.reward if result.reward is not None else 0.0)
                done = bool(result.done)
                rl_reward = _rl_reward_from_step_result(result)
                if obs is not None:
                    obs_dict = _observation_as_dict(obs)
            except Exception as exc:
                step_error = str(exc)
                done = True
                rl_reward = None

            if step_error is None:
                last_found, last_total = await _issues_progress(env_wrapper, obs_dict)

            if rl_reward is not None:
                if rl_reward > 0.0:
                    consecutive_nonpositive = 0
                else:
                    consecutive_nonpositive += 1
            elif reward > 0.0:
                consecutive_nonpositive = 0
            else:
                consecutive_nonpositive += 1

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_preview, reward, done, step_error)

            if step_error is None and consecutive_nonpositive >= 3 and not done:
                forced = CodeReviewAction(done_reviewing=True)
                forced_preview = '{"done_reviewing":true}'
                try:
                    result = await env_wrapper.step(forced)
                    obs = getattr(result, "observation", None)
                    r2 = float(result.reward if result.reward is not None else 0.0)
                    d2 = bool(result.done)
                    if obs is not None:
                        obs_dict = _observation_as_dict(obs)
                except Exception as exc:
                    step_error = str(exc)
                    r2 = 0.0
                    d2 = True
                rewards.append(r2)
                forced_step = step + 1
                steps_taken = forced_step
                log_step(forced_step, forced_preview, r2, d2, step_error)
                break

            if done:
                break

            if action.review_comment is not None and not action.done_reviewing:
                dup = (rl_reward is not None and round(rl_reward, 4) == -0.05) or (
                    isinstance(obs_dict.get("step_feedback"), str)
                    and "Duplicate comment" in (obs_dict.get("step_feedback") or "")
                )
                if dup:
                    fb = (
                        "That was a duplicate. You already commented on that area. "
                        "Find a DIFFERENT issue on a DIFFERENT line."
                    )
                elif (rl_reward is not None and rl_reward > 0.0) or (
                    rl_reward is None and reward > 0.0
                ):
                    fb = (
                        f"Good find. Reward: {reward:.2f}. Issues found: {last_found}/{last_total}. "
                        "Find the next issue."
                    )
                else:
                    fb = (
                        "That comment didn't match a known issue. Be more precise. "
                        f"Issues found so far: {last_found}/{last_total}."
                    )
                messages.append({"role": "user", "content": fb})

    finally:
        score = _episode_score(rewards)
        try:
            st = await env_wrapper.state()
            d: Any = st.model_dump() if hasattr(st, "model_dump") else st
            if isinstance(d, dict) and d.get("total_reward") is not None:
                # ``total_reward`` is already rubric output in (0, 1); avoid remapping.
                score = float(d["total_reward"])
        except Exception:
            pass
        log_end(score >= 0.1, steps_taken, score, rewards)

    return score, last_found, last_total


async def main() -> None:
    api_key = _resolve_api_key()
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    results: List[Tuple[str, float, int, int]] = []

    if IMAGE_NAME:
        from openenv.core.generic_client import GenericEnvClient

        env = await GenericEnvClient.from_docker_image(IMAGE_NAME)
        try:
            for i, task in enumerate(TASKS):
                s, fnd, tot = await run_task(client, env, task)
                results.append((task, s, fnd, tot))
                # Long gaps between steps/tasks let the WS idle; servers often close
                # with 1011 keepalive ping timeout. Reconnect before the next task.
                if i + 1 < len(TASKS):
                    await env.disconnect()
                    await env.connect()
        finally:
            await env.close()
    else:
        env = EnvWrapper()
        try:
            for task in TASKS:
                s, fnd, tot = await run_task(client, env, task)
                results.append((task, s, fnd, tot))
        finally:
            await env.close()

    if results:
        print("[SUMMARY]", flush=True)
        total_score = 0.0
        for name, sc, fnd, tot in results:
            print(
                f"{name}: score={sc:.2f} ({fnd}/{tot} issues)",
                flush=True,
            )
            total_score += sc
        avg = total_score / len(results)
        print(f"average: {avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
