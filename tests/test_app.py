"""HTTP API tests: session-aware reset/step/state, validation and error codes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import SESSION_HEADER, app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _reset(client: TestClient, **body) -> tuple[str, dict]:
    r = client.post("/reset", json=body)
    assert r.status_code == 200, r.text
    sid = r.headers[SESSION_HEADER]
    assert sid
    return sid, r.json()


def test_root_and_tasks(client: TestClient) -> None:
    assert client.get("/").json()["tasks"] == "/tasks"
    assert client.get("/tasks").json() == {"tasks": ["easy", "medium", "hard"]}
    assert client.get("/health").status_code == 200


def test_reset_returns_observation_and_session(client: TestClient) -> None:
    _, body = _reset(client, task_name="medium", scenario_index=0)
    obs = body["observation"]
    assert obs["task_name"] == "medium"
    assert obs["diff"]
    assert obs["remaining_comments"] == 10
    assert body["done"] is False
    assert 0.0 < body["reward"] < 1.0


def test_reset_unknown_task_is_400(client: TestClient) -> None:
    r = client.post("/reset", json={"task_name": "does-not-exist"})
    assert r.status_code == 400
    assert "Unknown task" in r.json()["detail"]


def test_step_requires_session_header(client: TestClient) -> None:
    r = client.post("/step", json={"action": {"done_reviewing": True}})
    assert r.status_code == 400


def test_step_unknown_session_is_404(client: TestClient) -> None:
    r = client.post(
        "/step",
        json={"action": {"done_reviewing": True}},
        headers={SESSION_HEADER: "no-such-session"},
    )
    assert r.status_code == 404


def test_step_invalid_action_is_422(client: TestClient) -> None:
    sid, _ = _reset(client, task_name="easy", scenario_index=0)
    r = client.post(
        "/step",
        json={
            "action": {
                "review_comment": {
                    "line_number": 4,
                    "comment": "x",
                    "severity": "bogus",
                    "category": "bug",
                }
            }
        },
        headers={SESSION_HEADER: sid},
    )
    assert r.status_code == 422


def test_full_episode_over_http(client: TestClient) -> None:
    sid, _ = _reset(client, task_name="easy", scenario_index=0)
    headers = {SESSION_HEADER: sid}

    good = {
        "review_comment": {
            "line_number": 4,
            "comment": "off-by-one: range upper bound len(items) + 1 raises IndexError",
            "severity": "critical",
            "category": "bug",
        }
    }
    r = client.post("/step", json={"action": good}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is False
    assert body["observation"]["remaining_comments"] == 9
    assert len(body["observation"]["comments_so_far"]) == 1

    state = client.get("/state", headers=headers).json()
    assert state["step_count"] == 1
    assert len(state["comments"]) == 1
    mid_score = state["total_reward"]

    r = client.post("/step", json={"action": {"done_reviewing": True}}, headers=headers)
    assert r.json()["done"] is True

    # Finished sessions stay readable until they expire.
    final = client.get("/state", headers=headers).json()
    assert final["done"] is True
    assert final["total_reward"] == pytest.approx(mid_score)


def test_sessions_are_isolated(client: TestClient) -> None:
    sid_a, _ = _reset(client, task_name="easy", scenario_index=0)
    sid_b, _ = _reset(client, task_name="hard", scenario_index=1)
    assert sid_a != sid_b
    client.post("/step", json={"action": {"done_reviewing": True}}, headers={SESSION_HEADER: sid_a})
    assert client.get("/state", headers={SESSION_HEADER: sid_a}).json()["done"] is True
    st_b = client.get("/state", headers={SESSION_HEADER: sid_b}).json()
    assert st_b["done"] is False
    assert st_b["task_name"] == "hard"


def test_state_without_header_is_empty(client: TestClient) -> None:
    r = client.get("/state")
    assert r.status_code == 200
    assert r.json()["step_count"] == 0
