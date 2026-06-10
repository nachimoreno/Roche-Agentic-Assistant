"""
HTTP-layer tests for `api.py`.

These exercise the FastAPI surface only — routing, request validation,
response mapping, and error handling — with a fake Assistant injected into
`app.state`. No Groq, no Chroma, no ingest: the lifespan (which would call
`build_assistant`) is deliberately not triggered, so `TestClient(app)` is
constructed without the `with` context manager and we set the assistant
ourselves.
"""

from __future__ import annotations

import json
import os

# `api` instantiates Settings() at import time, which requires a Groq key.
# Provide a dummy so importing the module never reaches the network.
os.environ.setdefault("GROQ_API_KEY", "test-key")

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import api
from agent import Citation
from auth import get_current_user
from conversation_layer import AnalysisResult
from db import User, create_all, make_engine
from orchestrator import Response, StreamDone, StreamMeta, StreamToken
from repositories import SessionRepository, UserRepository

# A fixed authenticated user for the HTTP-surface tests (auth is overridden,
# not exercised here — see test_auth.py / test_sessions.py for the real flow).
_FAKE_USER = User(email="tester@roche.com", password_hash="x", display_name="Tester")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeAssistant:
    """Records calls and returns a canned Response (or raises)."""

    def __init__(self, response: Response | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.calls: list[tuple] = []

    def handle(self, session_id, message, **kwargs):
        self.calls.append((session_id, message))
        if self._raises is not None:
            raise self._raises
        return self._response


def _question_response() -> Response:
    return Response(
        text="Use a 70 percent isopropyl alcohol wipe.",
        analysis=AnalysisResult(language="english", type="question", emotion=None),
        citations=[Citation(source="06_cleaning_lab_devices.md", section="Centrifuges")],
    )


def _feedback_response() -> Response:
    return Response(
        text="Thanks, I've recorded your feedback.",
        analysis=AnalysisResult(language="english", type="feedback", emotion="confused"),
        citations=[],
    )


@pytest.fixture
def client(tmp_path):
    """TestClient with no lifespan run.

    Real Session/User repos (temp file DB) back the ownership checks, auth is
    overridden to a fixed user, and the assistant is injected per test.
    """
    engine = make_engine(f"sqlite:///{tmp_path}/api.db")
    create_all(engine)
    api.app.state.sessions = SessionRepository(engine)
    api.app.state.users = UserRepository(engine)
    api.app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    try:
        yield TestClient(api.app)
    finally:
        api.app.dependency_overrides.clear()


def _inject(assistant) -> None:
    api.app.state.assistant = assistant


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def test_create_session_returns_valid_uuid(client):
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    body = resp.json()
    # Should parse as a UUID — i.e. a real id, not an empty string.
    UUID(body["id"])


def test_create_session_ids_are_unique(client):
    a = client.post("/api/sessions").json()["id"]
    b = client.post("/api/sessions").json()["id"]
    assert a != b


# ---------------------------------------------------------------------------
# Chat — happy paths
# ---------------------------------------------------------------------------

def test_chat_returns_text_and_mapped_citations(client):
    _inject(FakeAssistant(response=_question_response()))
    sid = str(UUID(int=1))

    resp = client.post(f"/api/sessions/{sid}/chat", json={"message": "How do I clean it?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Use a 70 percent isopropyl alcohol wipe."
    assert body["language"] == "english"
    assert body["type"] == "question"
    assert body["emotion"] is None
    assert body["citations"] == [
        {"source": "06_cleaning_lab_devices.md", "section": "Centrifuges"}
    ]


def test_chat_surfaces_feedback_emotion(client):
    _inject(FakeAssistant(response=_feedback_response()))
    sid = str(UUID(int=2))

    body = client.post(
        f"/api/sessions/{sid}/chat", json={"message": "This doc is confusing"}
    ).json()

    assert body["type"] == "feedback"
    assert body["emotion"] == "confused"
    assert body["citations"] == []


def test_chat_forwards_parsed_uuid_and_message_to_assistant(client):
    fake = FakeAssistant(response=_question_response())
    _inject(fake)
    sid = str(UUID(int=3))

    client.post(f"/api/sessions/{sid}/chat", json={"message": "hello there"})

    assert len(fake.calls) == 1
    passed_sid, passed_msg = fake.calls[0]
    assert isinstance(passed_sid, UUID)           # endpoint parses the path param
    assert str(passed_sid) == sid
    assert passed_msg == "hello there"


# ---------------------------------------------------------------------------
# Chat — validation & error handling
# ---------------------------------------------------------------------------

def test_chat_rejects_malformed_session_id(client):
    _inject(FakeAssistant(response=_question_response()))
    resp = client.post("/api/sessions/not-a-uuid/chat", json={"message": "hi"})
    assert resp.status_code == 422
    assert "session_id" in resp.json()["detail"]


def test_chat_requires_message_field(client):
    _inject(FakeAssistant(response=_question_response()))
    sid = str(UUID(int=4))
    resp = client.post(f"/api/sessions/{sid}/chat", json={})
    # Pydantic body validation kicks in before our handler.
    assert resp.status_code == 422


def test_chat_handler_error_returns_generic_500_without_leaking(client):
    secret = "boom: postgres://user:pa55w0rd@db/internal"
    _inject(FakeAssistant(raises=RuntimeError(secret)))
    sid = str(UUID(int=5))

    resp = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail == "Internal error handling the request."
    # The raw exception text must not reach the client.
    assert "pa55w0rd" not in detail
    assert secret not in detail


# ---------------------------------------------------------------------------
# Static / routing
# ---------------------------------------------------------------------------

def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_route_is_gone(client):
    # Regression guard: the broken /dashboard route was removed.
    assert client.get("/dashboard").status_code == 404


# ---------------------------------------------------------------------------
# Streaming chat (SSE)
# ---------------------------------------------------------------------------

class _StatusError(Exception):
    """Stand-in for a provider error carrying an HTTP status_code."""

    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


class FakeStreamingAssistant:
    """Yields canned stream events; or raises `error` mid-stream after meta."""

    def __init__(self, tokens, citations, *, error: Exception | None = None):
        self._tokens = tokens
        self._citations = citations
        self._error = error
        self.calls: list[tuple] = []

    def handle_stream(self, session_id, message, **kwargs):
        self.calls.append((session_id, message))
        yield StreamMeta(
            analysis=AnalysisResult(language="english", type="question", emotion=None)
        )
        if self._error is not None:
            raise self._error
        for t in self._tokens:
            yield StreamToken(text=t)
        yield StreamDone(text="".join(self._tokens), citations=self._citations)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data-dict) frames."""
    frames = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        frames.append((event, data))
    return frames


def test_stream_emits_meta_tokens_then_done(client):
    cites = [Citation(source="06_cleaning.md", section="Centrifuges")]
    _inject(FakeStreamingAssistant(["Use ", "isopropyl."], cites))
    sid = str(UUID(int=10))

    resp = client.post(f"/api/sessions/{sid}/chat/stream", json={"message": "how?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert events[0] == "meta"
    assert events[-1] == "done"
    assert events.count("token") == 2

    meta = frames[0][1]
    assert meta["language"] == "english" and meta["type"] == "question"
    text = "".join(d["text"] for e, d in frames if e == "token")
    assert text == "Use isopropyl."
    done = frames[-1][1]
    assert done["citations"] == [{"source": "06_cleaning.md", "section": "Centrifuges"}]


def test_stream_rejects_malformed_session_id(client):
    _inject(FakeStreamingAssistant(["x"], []))
    resp = client.post("/api/sessions/not-a-uuid/chat/stream", json={"message": "hi"})
    assert resp.status_code == 422


def test_stream_error_emits_internal_category_without_leaking(client):
    _inject(FakeStreamingAssistant(["x"], [], error=RuntimeError("secret token=abc123")))
    sid = str(UUID(int=11))

    resp = client.post(f"/api/sessions/{sid}/chat/stream", json={"message": "hi"})
    # Stream starts 200; the failure surfaces as an SSE error frame.
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["category"] == "internal"
    assert frames[-1][1]["detail"] == "Internal error handling the request."
    assert "abc123" not in resp.text


def test_stream_error_categorizes_auth_failure(client):
    _inject(FakeStreamingAssistant(["x"], [], error=_StatusError(401, "Invalid API Key")))
    sid = str(UUID(int=12))

    resp = client.post(f"/api/sessions/{sid}/chat/stream", json={"message": "hi"})
    frames = _parse_sse(resp.text)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["category"] == "auth"
    assert "authentication" in frames[-1][1]["detail"].lower()
    assert "Invalid API Key" not in resp.text


# ---------------------------------------------------------------------------
# _error_category
# ---------------------------------------------------------------------------

def test_error_category_maps_status_codes():
    assert api._error_category(_StatusError(401))[0] == "auth"
    assert api._error_category(_StatusError(403))[0] == "auth"
    assert api._error_category(_StatusError(429))[0] == "rate_limit"


def test_error_category_defaults_to_internal():
    cat, detail = api._error_category(RuntimeError("anything"))
    assert cat == "internal"
    assert detail == "Internal error handling the request."
