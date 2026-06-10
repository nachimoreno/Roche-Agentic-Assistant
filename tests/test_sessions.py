"""
Per-user session + history HTTP tests with the *real* auth flow (cookies),
real Session/User repositories, and ownership enforcement. A no-op assistant
is injected so chat ownership can be checked without Groq.
"""

from __future__ import annotations

import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

import api
from db import create_all, make_engine
from repositories import SessionRepository, UserRepository


class _NoopAssistant:
    """Asserts it is never reached when ownership should have blocked a call."""

    def __init__(self):
        self.calls = 0

    def handle(self, *a, **k):
        self.calls += 1
        raise AssertionError("assistant.handle should not run for a blocked request")

    def handle_stream(self, *a, **k):
        self.calls += 1
        raise AssertionError("assistant.handle_stream should not run for a blocked request")


@pytest.fixture
def ctx(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/sessions.db")
    create_all(engine)
    api.app.state.sessions = SessionRepository(engine)
    api.app.state.users = UserRepository(engine)
    api.app.state.assistant = _NoopAssistant()
    api.app.dependency_overrides.clear()  # use the real cookie-based auth
    return SessionRepository(engine)


def _user_client(email: str) -> TestClient:
    """A client with its own cookie jar, registered + logged in as `email`."""
    c = TestClient(api.app)
    r = c.post("/api/auth/register", json={"email": email, "password": "supersecret"})
    assert r.status_code == 201
    return c


# ---------------------------------------------------------------------------
# Auth required
# ---------------------------------------------------------------------------

def test_session_endpoints_require_auth(ctx):
    anon = TestClient(api.app)
    assert anon.get("/api/sessions").status_code == 401
    assert anon.post("/api/sessions").status_code == 401
    assert anon.get("/api/sessions/whatever/messages").status_code in (401, 422)


# ---------------------------------------------------------------------------
# Per-user isolation
# ---------------------------------------------------------------------------

def test_sessions_are_listed_per_user(ctx):
    a = _user_client("a@roche.com")
    b = _user_client("b@roche.com")

    a.post("/api/sessions", json={"title": "Centrifuge cleaning"})
    a.post("/api/sessions", json={"title": "Onboarding"})

    a_list = a.get("/api/sessions").json()
    assert {s["title"] for s in a_list} == {"Centrifuge cleaning", "Onboarding"}
    # B sees none of A's sessions.
    assert b.get("/api/sessions").json() == []


def test_cannot_access_another_users_session(ctx):
    a = _user_client("owner@roche.com")
    b = _user_client("intruder@roche.com")
    sid = a.post("/api/sessions", json={"title": "Private"}).json()["id"]

    # 404 (not 403) so existence isn't leaked.
    assert b.get(f"/api/sessions/{sid}/messages").status_code == 404
    assert b.patch(f"/api/sessions/{sid}", json={"title": "hax"}).status_code == 404
    assert b.delete(f"/api/sessions/{sid}").status_code == 404
    assert b.post(f"/api/sessions/{sid}/chat", json={"message": "hi"}).status_code == 404
    assert b.post(f"/api/sessions/{sid}/chat/stream", json={"message": "hi"}).status_code == 404
    assert api.app.state.assistant.calls == 0  # never reached the assistant


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def test_messages_returns_persisted_history(ctx):
    a = _user_client("hist@roche.com")
    sid = a.post("/api/sessions").json()["id"]

    # Seed turns directly through the repo (same engine the API uses).
    from uuid import UUID
    ctx.append_turn(UUID(sid), role="user", content="How do I clean it?", language="english")
    ctx.append_turn(UUID(sid), role="assistant", content="Use isopropyl.", language="english")

    msgs = a.get(f"/api/sessions/{sid}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "How do I clean it?"
    assert msgs[1]["content"] == "Use isopropyl."


# ---------------------------------------------------------------------------
# Rename / delete
# ---------------------------------------------------------------------------

def test_rename_and_soft_delete(ctx):
    a = _user_client("crud@roche.com")
    sid = a.post("/api/sessions", json={"title": "old"}).json()["id"]

    renamed = a.patch(f"/api/sessions/{sid}", json={"title": "new title"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "new title"
    assert a.get("/api/sessions").json()[0]["title"] == "new title"

    assert a.delete(f"/api/sessions/{sid}").status_code == 204
    assert a.get("/api/sessions").json() == []
    # Deleted session is no longer reachable.
    assert a.get(f"/api/sessions/{sid}/messages").status_code == 404
