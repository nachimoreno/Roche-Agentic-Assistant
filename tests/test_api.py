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

import os

# `api` instantiates Settings() at import time, which requires a Groq key.
# Provide a dummy so importing the module never reaches the network.
os.environ.setdefault("GROQ_API_KEY", "test-key")

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import api
from agent import Citation
from conversation_layer import AnalysisResult
from orchestrator import Response


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
def client():
    """TestClient with no lifespan run; assistant injected per test."""
    return TestClient(api.app)


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
    UUID(body["session_id"])


def test_create_session_ids_are_unique(client):
    a = client.post("/api/sessions").json()["session_id"]
    b = client.post("/api/sessions").json()["session_id"]
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
