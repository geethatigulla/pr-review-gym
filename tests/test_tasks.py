"""Scenario-pool integrity: rubric labels must point at real lines and valid enums."""
from __future__ import annotations

import pytest

from server.grader import SYNONYM_GROUPS
from server.tasks import TASK_ALIASES, TASKS, get_all_task_names, get_scenario, resolve_task_name

SEVERITIES = {"critical", "major", "minor", "suggestion"}
CATEGORIES = {"bug", "security", "performance", "style", "logic"}
ALL_SCENARIOS = [(task, sc) for task, pool in TASKS.items() for sc in pool]


def test_canonical_task_names() -> None:
    assert get_all_task_names() == ["easy", "medium", "hard"]
    assert set(TASKS) == set(get_all_task_names())


def test_scenario_ids_are_unique() -> None:
    ids = [sc["id"] for _, sc in ALL_SCENARIOS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("task,scenario", ALL_SCENARIOS, ids=[sc["id"] for _, sc in ALL_SCENARIOS])
def test_expected_issues_are_well_formed(task: str, scenario: dict) -> None:
    for key in ("id", "file_name", "language", "pr_title", "pr_description", "diff"):
        assert scenario[key], f"{scenario['id']} missing {key}"
    n_lines = len(scenario["diff"].splitlines())
    assert scenario["expected_issues"]
    for issue in scenario["expected_issues"]:
        assert 1 <= issue["line_number"] <= n_lines
        assert issue["severity"] in SEVERITIES
        assert issue["category"] in CATEGORIES
        assert issue["keywords"]
        assert issue["description"]
        group = issue.get("synonym_group")
        assert group is None or group in SYNONYM_GROUPS


def test_aliases_resolve() -> None:
    for alias, canonical in TASK_ALIASES.items():
        assert resolve_task_name(alias) == canonical


def test_unknown_task_raises() -> None:
    with pytest.raises(ValueError):
        resolve_task_name("nope")


def test_get_scenario_is_reproducible() -> None:
    assert get_scenario("hard", seed=7)["id"] == get_scenario("hard", seed=7)["id"]
    pool = TASKS["medium"]
    assert get_scenario("medium", scenario_index=len(pool) + 1)["id"] == pool[1]["id"]
