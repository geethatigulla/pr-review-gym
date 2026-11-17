"""FastAPI server exposing the Code Review environment over HTTP/WebSocket (OpenEnv).

OpenEnv's stock ``/reset``, ``/step`` and ``/state`` HTTP routes are stateless (a new
environment per request). This module replaces them with session-aware handlers keyed
by the ``X-OpenEnv-Session-Id`` header so a multi-step episode can be driven over plain
HTTP. Idle sessions expire after ``SESSION_TTL_SEC`` seconds and at most
``MAX_SESSIONS`` live at once. The WebSocket ``/ws`` route is left untouched.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Hashable

from fastapi import Body, FastAPI, Header, HTTPException, Response
from fastapi.routing import APIRoute
from openenv.core.env_server import create_fastapi_app
from openenv.core.env_server.serialization import deserialize_action, serialize_observation
from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse

from server.cache import TTLCache
from server.openenv_env import (
    CodeReviewActionAPI,
    CodeReviewObservationAPI,
    CodeReviewOpenEnvironment,
    CodeReviewServerState,
)
from server.tasks import get_all_task_names, resolve_task_name

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-OpenEnv-Session-Id"
SESSION_TTL_SEC = float(os.environ.get("SESSION_TTL_SEC", "1800"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "1000"))


@dataclass
class _Session:
    env: CodeReviewOpenEnvironment
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _close_session(key: Hashable, session: _Session) -> None:
    try:
        session.env.close()
    except Exception:  # pragma: no cover - close() is best effort
        logger.exception("Failed to close expired session %s", key)


def _strip_openenv_session_routes(application: FastAPI) -> None:
    """Remove stateless ``POST /reset``, ``POST /step`` and ``GET /state``."""
    replaced = {("/reset", "POST"), ("/step", "POST"), ("/state", "GET")}
    application.router.routes = [
        r
        for r in application.router.routes
        if not (
            isinstance(r, APIRoute)
            and any((r.path, m) in replaced for m in getattr(r, "methods", set()))
        )
    ]


app = create_fastapi_app(
    lambda: CodeReviewOpenEnvironment(),
    CodeReviewActionAPI,
    CodeReviewObservationAPI,
)
_strip_openenv_session_routes(app)
app.state.code_review_sessions = TTLCache(
    default_ttl_sec=SESSION_TTL_SEC,
    on_expire=_close_session,
    max_size=MAX_SESSIONS,
)


def _sessions() -> TTLCache:
    return app.state.code_review_sessions


def _require_session(session_id: str | None) -> _Session:
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {SESSION_HEADER} header; send the value returned by POST /reset.",
        )
    session = _sessions().get(session_id, touch=True)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown or expired {SESSION_HEADER}; call POST /reset first.",
        )
    return session


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
async def session_reset(
    response: Response,
    request: ResetRequest = Body(default_factory=ResetRequest),
    x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> ResetResponse:
    """Start an episode. Reuse the returned ``X-OpenEnv-Session-Id`` header on later calls.

    Body fields: ``task_name`` (``easy`` / ``medium`` / ``hard`` or a legacy alias),
    optional ``seed`` and ``scenario_index`` for reproducible scenarios.
    """
    kwargs: dict[str, Any] = request.model_dump(exclude_unset=True)
    task_name = kwargs.get("task_name")
    if task_name is not None:
        try:
            resolve_task_name(str(task_name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    sessions = _sessions()
    sid = x_openenv_session_id or str(uuid.uuid4())
    stale = sessions.pop(sid)
    if stale is not None:
        _close_session(sid, stale)

    session = _Session(env=CodeReviewOpenEnvironment())
    async with session.lock:
        observation = await asyncio.to_thread(session.env.reset, **kwargs)
    sessions.set(sid, session)
    response.headers[SESSION_HEADER] = sid
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
async def session_step(
    request: StepRequest = Body(...),
    x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> StepResponse:
    """Apply one action to the episode identified by ``X-OpenEnv-Session-Id``.

    Finished episodes stay readable through ``GET /state`` until the session expires;
    further steps on them return ``done=true`` without changing the score.
    """
    session = _require_session(x_openenv_session_id)
    try:
        action = deserialize_action(request.action, CodeReviewActionAPI)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with session.lock:
        if request.timeout_s is not None:
            observation = await asyncio.to_thread(session.env.step, action, request.timeout_s)
        else:
            observation = await asyncio.to_thread(session.env.step, action)
    return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=CodeReviewServerState, tags=["State Management"])
async def session_state(
    x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> CodeReviewServerState:
    """Full episode state for a session (an empty state when no header is sent)."""
    if not x_openenv_session_id:
        return CodeReviewOpenEnvironment().state
    session = _require_session(x_openenv_session_id)
    return session.env.state


@app.get("/")
def root() -> dict:
    """Landing page for browsers; the OpenEnv factory does not register ``/``."""
    return {
        "service": "openenv-code-review-env",
        "message": (
            f"Use POST /reset then POST /step with header {SESSION_HEADER} "
            "(same id for the whole episode). WebSocket /ws is also supported."
        ),
        "docs": "/docs",
        "tasks": "/tasks",
    }


@app.get("/tasks")
def tasks() -> dict:
    """List canonical task ids ``easy``, ``medium``, ``hard``."""
    return {"tasks": get_all_task_names()}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
