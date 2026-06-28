"""
HTTP-layer tests for the runtime-settings surface (`/api/settings`).

Like test_api.py, these exercise the FastAPI surface with fakes injected into
`app.state` — the lifespan is never run. A duck-typed agent/llm/docs stand in
for the real components so we can assert that an update actually mutates the
running objects (live tuning) and persists through the repository.
"""

from __future__ import annotations

import os

# `api`/`settings` instantiate Settings() at import time, which needs a key.
os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

import api
from db import create_all, make_engine
from repositories import SessionRepository, SettingsRepository, UserRepository
from runtime_settings import RuntimeSettings
from settings import Settings


# ---------------------------------------------------------------------------
# Duck-typed stand-ins for the live components RuntimeSettings tunes.
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self):
        self.top_k = 4
        self.max_tokens = 1024
        self.temperature = 0.0
        self.min_dense = 0.30
        self.min_lexical = 0.85
        self.warn_dense = 0.45


class _FakeLLM:
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"


class _FakeDocs:
    def __init__(self):
        self.hybrid = True


def _build_runtime(engine):
    """A RuntimeSettings bound to fresh fakes + a real repo on `engine`."""
    repo = SettingsRepository(engine)
    runtime = RuntimeSettings(
        agent=_FakeAgent(),
        llm=_FakeLLM(),
        docs=_FakeDocs(),
        repo=repo,
        settings=Settings(),
    )
    runtime.load_persisted()
    return runtime, repo


@pytest.fixture
def ctx(tmp_path):
    """TestClient (no lifespan) with settings wiring + a per-test admin helper."""
    engine = make_engine(f"sqlite:///{tmp_path}/settings.db")
    create_all(engine)
    api.app.state.users = UserRepository(engine)
    api.app.state.sessions = SessionRepository(engine)
    runtime, repo = _build_runtime(engine)
    api.app.state.settings_repo = repo
    api.app.state.runtime_settings = runtime
    client = TestClient(api.app)

    def make_admin(email="admin@roche.com"):
        r = client.post(
            "/api/auth/register",
            json={"email": email, "password": "password123", "display_name": "A"},
        )
        assert r.status_code == 201
        api.app.state.users.set_role(api.app.state.users.get_by_email(email).id, "admin")
        return r.json()

    try:
        yield client, runtime, engine, make_admin
    finally:
        api.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def test_settings_hidden_from_non_admin(ctx):
    client, _runtime, _engine, _make_admin = ctx
    client.post(
        "/api/auth/register",
        json={"email": "plain@roche.com", "password": "password123"},
    )
    assert client.get("/api/settings").status_code == 404
    assert client.put("/api/settings", json={"values": {"top_k": 7}}).status_code == 404


def test_settings_page_hidden_from_non_admin(ctx):
    client, *_ = ctx
    client.post(
        "/api/auth/register",
        json={"email": "plain2@roche.com", "password": "password123"},
    )
    assert client.get("/settings").status_code == 404
    assert client.get("/announcements").status_code == 404


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def test_get_settings_returns_grouped_defaults(ctx):
    client, _runtime, _engine, make_admin = ctx
    make_admin()
    body = client.get("/api/settings").json()
    ids = [g["id"] for g in body["groups"]]
    assert ids == ["confidence", "retrieval", "llm", "demo"]
    params = {p["key"]: p for g in body["groups"] for p in g["params"]}
    assert params["retrieval_min_dense"]["value"] == pytest.approx(0.30)
    assert params["top_k"]["value"] == 4
    assert params["retrieval_mode"]["value"] == "hybrid"
    # metadata the UI relies on
    assert params["retrieval_mode"]["choices"] == ["hybrid", "dense"]
    assert params["seed_demo_feedback"]["live"] is False


# ---------------------------------------------------------------------------
# Write — live application
# ---------------------------------------------------------------------------

def test_update_applies_to_agent_live(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    r = client.put("/api/settings", json={"values": {"retrieval_min_dense": 0.42}})
    assert r.status_code == 200
    assert runtime._agent.min_dense == pytest.approx(0.42)
    params = {p["key"]: p for g in r.json()["groups"] for p in g["params"]}
    assert params["retrieval_min_dense"]["value"] == pytest.approx(0.42)


def test_update_toggles_retrieval_mode(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    client.put("/api/settings", json={"values": {"retrieval_mode": "dense"}})
    assert runtime._docs.hybrid is False
    client.put("/api/settings", json={"values": {"retrieval_mode": "hybrid"}})
    assert runtime._docs.hybrid is True


def test_update_model_name(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    client.put("/api/settings", json={"values": {"model_name": "llama-3.1-8b-instant"}})
    assert runtime._llm.model == "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Write — validation
# ---------------------------------------------------------------------------

def test_numeric_values_are_clamped(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    client.put("/api/settings", json={"values": {"top_k": 999, "retrieval_min_dense": 5}})
    assert runtime._agent.top_k == 20            # clamped to max
    assert runtime._agent.min_dense == pytest.approx(1.0)


def test_invalid_enum_is_rejected(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    r = client.put("/api/settings", json={"values": {"retrieval_mode": "banana"}})
    assert r.status_code == 422
    assert runtime._docs.hybrid is True          # unchanged


def test_unknown_keys_are_ignored(ctx):
    client, runtime, _engine, make_admin = ctx
    make_admin()
    r = client.put("/api/settings", json={"values": {"nonsense": 1, "top_k": 7}})
    assert r.status_code == 200
    assert runtime._agent.top_k == 7


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_override_persists_across_restart(ctx):
    client, _runtime, engine, make_admin = ctx
    make_admin()
    client.put("/api/settings", json={"values": {"retrieval_warn_dense": 0.55, "top_k": 9}})
    # Simulate a restart: a brand-new RuntimeSettings + fresh fakes on same DB.
    fresh, _repo = _build_runtime(engine)
    assert fresh._agent.warn_dense == pytest.approx(0.55)
    assert fresh._agent.top_k == 9
