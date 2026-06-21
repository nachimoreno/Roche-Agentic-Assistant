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
from repositories import (
    AnnouncementRepository,
    FeedbackRepository,
    SessionRepository,
    UserRepository,
)

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
        citations=[Citation(
            source="06_cleaning_lab_devices.md",
            section="Centrifuges",
            title="Cleaning Laboratory Devices",
        )],
        follow_ups=["How often should I clean the rotor?"],
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
    api.app.state.feedback = FeedbackRepository(engine)
    api.app.state.announcements = AnnouncementRepository(engine)
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


def _session_with_pending_user_turn(client) -> str:
    """Create an owned session and seed an unanswered user turn.

    Mirrors the real abort flow: handle_stream persists the user turn up front,
    then the stream is abandoned before the assistant turn is written — so at
    abort time the most recent turn is the user's.
    """
    sid = client.post("/api/sessions").json()["id"]
    api.app.state.sessions.append_turn(UUID(sid), role="user", content="A question?")
    return sid


def test_append_partial_message_persists_assistant_turn(client):
    sid = _session_with_pending_user_turn(client)
    resp = client.post(
        f"/api/sessions/{sid}/messages",
        json={"content": "Partial answer kept after Stop", "language": "english"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "assistant"

    rows = client.get(f"/api/sessions/{sid}/messages").json()
    assert [r["content"] for r in rows] == ["A question?", "Partial answer kept after Stop"]


def test_append_partial_message_rejects_empty_content(client):
    sid = _session_with_pending_user_turn(client)
    resp = client.post(f"/api/sessions/{sid}/messages", json={"content": "   "})
    assert resp.status_code == 422


def test_append_partial_message_404_for_unowned_session(client):
    resp = client.post(
        f"/api/sessions/{UUID(int=99)}/messages", json={"content": "x"}
    )
    assert resp.status_code == 404


def test_append_partial_message_rejects_when_no_pending_user_turn(client):
    # No unanswered user turn → a client must not be able to forge an assistant
    # turn (which would feed back into model history). Covers empty sessions...
    sid = client.post("/api/sessions").json()["id"]
    resp = client.post(f"/api/sessions/{sid}/messages", json={"content": "forged"})
    assert resp.status_code == 409

    # ...and stacking a second, different assistant turn after a legitimate one.
    sid2 = _session_with_pending_user_turn(client)
    assert client.post(
        f"/api/sessions/{sid2}/messages", json={"content": "real partial"}
    ).status_code == 201
    stacked = client.post(
        f"/api/sessions/{sid2}/messages", json={"content": "a different forged turn"}
    )
    assert stacked.status_code == 409


def test_append_partial_message_is_idempotent_for_duplicate_assistant_turn(client):
    # Race: the stream finished and the server persisted the full turn just as
    # the client aborted and posted the identical text it kept. The endpoint
    # must not create a second assistant turn (also covers a client retry).
    sid = _session_with_pending_user_turn(client)
    payload = {"content": "Full answer the server already saved", "language": "english"}

    first = client.post(f"/api/sessions/{sid}/messages", json=payload)
    assert first.status_code == 201
    second = client.post(f"/api/sessions/{sid}/messages", json=payload)
    assert second.status_code == 201

    rows = client.get(f"/api/sessions/{sid}/messages").json()
    assert [r["content"] for r in rows] == [
        "A question?",
        "Full answer the server already saved",
    ]


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
        {
            "source": "06_cleaning_lab_devices.md",
            "section": "Centrifuges",
            "title": "Cleaning Laboratory Devices",
            "url": None,
        }
    ]
    assert body["follow_ups"] == ["How often should I clean the rotor?"]


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


def test_service_worker_served_at_root_scope(client):
    # The SW must be reachable at /sw.js (root scope) with the header that lets
    # it control the whole app, so the PWA can install + cache the shell.
    resp = client.get("/sw.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert resp.headers.get("service-worker-allowed") == "/"


def test_dashboard_route_is_gone(client):
    # Regression guard: the broken /dashboard route was removed.
    assert client.get("/dashboard").status_code == 404


# ---------------------------------------------------------------------------
# Timestamp serialisation — must carry an explicit UTC offset so the browser
# does not render it in local time (the "2h ago right after sending" bug).
# ---------------------------------------------------------------------------

def test_iso_utc_stamps_naive_datetime_as_utc():
    from datetime import datetime, timezone

    naive = datetime(2026, 6, 16, 17, 7, 8)        # as SQLite returns it
    out = api._iso_utc(naive)
    assert out.endswith("+00:00")
    assert datetime.fromisoformat(out) == naive.replace(tzinfo=timezone.utc)


def test_iso_utc_converts_aware_datetime_to_utc():
    from datetime import datetime, timedelta, timezone

    aware = datetime(2026, 6, 16, 19, 7, 8, tzinfo=timezone(timedelta(hours=2)))
    out = api._iso_utc(aware)
    assert out.endswith("+00:00")
    assert datetime.fromisoformat(out) == datetime(
        2026, 6, 16, 17, 7, 8, tzinfo=timezone.utc
    )


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

    def __init__(self, tokens, citations, *, error: Exception | None = None,
                 turn_id=None, follow_ups=None):
        self._tokens = tokens
        self._citations = citations
        self._error = error
        self._turn_id = turn_id
        self._follow_ups = follow_ups or []
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
        yield StreamDone(
            text="".join(self._tokens),
            citations=self._citations,
            turn_id=self._turn_id,
            follow_ups=self._follow_ups,
        )


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
    cites = [Citation(
        source="06_cleaning.md", section="Centrifuges", title="Cleaning Guide"
    )]
    _inject(FakeStreamingAssistant(
        ["Use ", "isopropyl."], cites, follow_ups=["What about the lid?"]
    ))
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
    assert done["citations"] == [
        {"source": "06_cleaning.md", "section": "Centrifuges", "title": "Cleaning Guide", "url": None}
    ]
    assert done["follow_ups"] == ["What about the lid?"]


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


# ---------------------------------------------------------------------------
# turn_id surfaced to the client (needed so the UI can rate an answer)
# ---------------------------------------------------------------------------

def test_chat_surfaces_turn_id(client):
    tid = UUID(int=42)
    _inject(FakeAssistant(response=Response(
        text="ok",
        analysis=AnalysisResult(language="english", type="question", emotion=None),
        citations=[],
        turn_id=tid,
    )))
    body = client.post(
        f"/api/sessions/{UUID(int=1)}/chat", json={"message": "hi"}
    ).json()
    assert body["turn_id"] == str(tid)


def test_stream_done_includes_turn_id(client):
    tid = UUID(int=55)
    _inject(FakeStreamingAssistant(["x"], [], turn_id=tid))
    resp = client.post(f"/api/sessions/{UUID(int=10)}/chat/stream", json={"message": "how?"})
    done = _parse_sse(resp.text)[-1][1]
    assert done["turn_id"] == str(tid)


def test_messages_include_turn_id(client):
    sid = _session_with_pending_user_turn(client)
    client.post(f"/api/sessions/{sid}/messages", json={"content": "answer", "language": "english"})
    rows = client.get(f"/api/sessions/{sid}/messages").json()
    assert rows and all(r.get("id") for r in rows)
    UUID(rows[0]["id"])   # each id is a real UUID


# ---------------------------------------------------------------------------
# Explicit ratings — POST /sessions/{sid}/turns/{tid}/feedback
# ---------------------------------------------------------------------------

class FakeRatingAssistant:
    """Records record_rating calls; optionally raises."""

    def __init__(self, raises: Exception | None = None):
        self._raises = raises
        self.calls: list[dict] = []

    def record_rating(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return object()


def _owned_session(client) -> str:
    """A session owned by the fixture's authenticated user."""
    return client.post("/api/sessions").json()["id"]


def test_rate_turn_records_rating(client):
    fake = FakeRatingAssistant()
    _inject(fake)
    sid = _owned_session(client)
    tid = str(UUID(int=7))

    resp = client.post(
        f"/api/sessions/{sid}/turns/{tid}/feedback",
        json={"rating": "down", "comment": "wrong section"},
    )
    assert resp.status_code == 201
    assert len(fake.calls) == 1
    assert fake.calls[0]["rating"] == "down"
    assert fake.calls[0]["comment"] == "wrong section"


def test_rate_turn_rejects_bad_rating(client):
    _inject(FakeRatingAssistant())
    sid = _owned_session(client)
    resp = client.post(
        f"/api/sessions/{sid}/turns/{UUID(int=7)}/feedback", json={"rating": "sideways"}
    )
    assert resp.status_code == 422


def test_rate_turn_rejects_malformed_turn_id(client):
    _inject(FakeRatingAssistant())
    sid = _owned_session(client)
    resp = client.post(
        f"/api/sessions/{sid}/turns/not-a-uuid/feedback", json={"rating": "up"}
    )
    assert resp.status_code == 422


def test_rate_turn_404_for_unowned_session(client):
    _inject(FakeRatingAssistant())
    resp = client.post(
        f"/api/sessions/{UUID(int=99)}/turns/{UUID(int=7)}/feedback", json={"rating": "up"}
    )
    assert resp.status_code == 404


def test_rate_turn_404_when_assistant_rejects_turn(client):
    # Assistant raises ValueError when the turn isn't a rateable assistant turn.
    _inject(FakeRatingAssistant(raises=ValueError("no such assistant turn")))
    sid = _owned_session(client)
    resp = client.post(
        f"/api/sessions/{sid}/turns/{UUID(int=7)}/feedback", json={"rating": "up"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Admin gate + analytics endpoints
#
# `require_admin` resolves the session cookie via the real `get_current_user`
# (a direct call, not a Depends), so the dependency override above does NOT
# apply here — these tests exercise the genuine register → cookie → role path.
# ---------------------------------------------------------------------------

def _register(client, email="scientist@roche.com"):
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123", "display_name": "T"},
    )
    assert r.status_code == 201
    return r.json()


def _promote_current(client, email):
    users: UserRepository = api.app.state.users
    user = users.get_by_email(email)
    users.set_role(user.id, "admin")


def test_analytics_404_for_regular_user(client):
    _register(client)
    assert client.get("/api/analytics/summary").status_code == 404
    assert client.get("/api/analytics/hotspots").status_code == 404
    assert client.get("/api/analytics/trend").status_code == 404
    assert client.get("/admin").status_code == 404


def test_analytics_ok_for_admin(client):
    _register(client, "it-admin@roche.com")
    _promote_current(client, "it-admin@roche.com")

    s = client.get("/api/analytics/summary")
    assert s.status_code == 200
    body = s.json()
    assert body["total"] == 0 and body["negative_rate"] == 0.0

    h = client.get("/api/analytics/hotspots?dimension=process&days=30")
    assert h.status_code == 200 and h.json() == []

    t = client.get("/api/analytics/trend?days=7")
    assert t.status_code == 200 and t.json() == []

    page = client.get("/admin")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Feedback" in page.text


def test_analytics_rejects_bad_dimension(client):
    _register(client, "it-admin2@roche.com")
    _promote_current(client, "it-admin2@roche.com")
    assert client.get("/api/analytics/hotspots?dimension=user").status_code == 422


# ---------------------------------------------------------------------------
# Announcements
# ---------------------------------------------------------------------------

def test_announcement_empty_when_none(client):
    assert client.get("/api/announcement").json() == {"id": None, "message": None}


def test_admin_publishes_and_users_see_announcement(client):
    _register(client, "ann-admin@roche.com")
    _promote_current(client, "ann-admin@roche.com")
    pub = client.put("/api/announcement", json={"message": "Lab 3 closed Friday."})
    assert pub.status_code == 200 and pub.json()["message"] == "Lab 3 closed Friday."
    got = client.get("/api/announcement").json()
    assert got["message"] == "Lab 3 closed Friday." and got["id"]


def test_regular_user_cannot_publish_announcement(client):
    _register(client, "ann-user@roche.com")            # not promoted
    r = client.put("/api/announcement", json={"message": "not an admin"})
    assert r.status_code == 404                          # surface hidden from non-admins
    assert client.get("/api/announcement").json()["message"] is None


def test_blank_message_takes_down_announcement(client):
    _register(client, "ann-admin2@roche.com")
    _promote_current(client, "ann-admin2@roche.com")
    client.put("/api/announcement", json={"message": "Temporary notice"})
    assert client.get("/api/announcement").json()["message"] == "Temporary notice"
    down = client.put("/api/announcement", json={"message": "   "})
    assert down.status_code == 200 and down.json()["message"] is None
    assert client.get("/api/announcement").json()["message"] is None


def test_register_never_accepts_role(client):
    # A forged role in the register body must be ignored, not honored.
    r = client.post(
        "/api/auth/register",
        json={"email": "sneaky@roche.com", "password": "password123", "role": "admin"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "user"
    assert client.get("/api/analytics/summary").status_code == 404


def test_admin_allowlist_promotes_on_register(client, monkeypatch):
    monkeypatch.setattr(api._settings, "admin_emails", "boss@roche.com, Other@x.com")
    body = _register(client, "boss@roche.com")
    assert body["role"] == "admin"
    assert client.get("/api/analytics/summary").status_code == 200


def test_admin_allowlist_promotes_on_login(client, monkeypatch):
    _register(client, "late-admin@roche.com")
    assert client.get("/api/analytics/summary").status_code == 404

    # Allowlisted afterwards — the next login promotes.
    monkeypatch.setattr(api._settings, "admin_emails", "late-admin@roche.com")
    r = client.post(
        "/api/auth/login",
        json={"email": "late-admin@roche.com", "password": "password123"},
    )
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert client.get("/api/analytics/summary").status_code == 200


# ---------------------------------------------------------------------------
# Voice input — POST /api/transcribe (Groq Whisper)
#
# `transcribe_audio` is mocked throughout: the fast suite never makes a live
# Groq call. The endpoint reads the upload, enforces the 25 MB guard, and
# returns the transcribed text.
# ---------------------------------------------------------------------------

def test_transcribe_requires_auth():
    # No get_current_user override → the real dependency runs. Without a
    # session cookie the request is rejected before any transcription happens.
    api.app.dependency_overrides.pop(get_current_user, None)
    unauthed = TestClient(api.app)
    resp = unauthed.post(
        "/api/transcribe",
        files={"audio": ("clip.webm", b"audio-bytes", "audio/webm")},
    )
    assert resp.status_code == 401


def test_transcribe_rejects_oversized_upload(client, monkeypatch):
    # The guard must fire purely on size, before any transcription is attempted.
    called = {"n": 0}

    def _fake_transcribe(**kwargs):
        called["n"] += 1
        return "should not be reached"

    monkeypatch.setattr(api, "transcribe_audio", _fake_transcribe)

    oversized = b"\x00" * (api._MAX_AUDIO_BYTES + 1)
    resp = client.post(
        "/api/transcribe",
        files={"audio": ("big.webm", oversized, "audio/webm")},
    )
    assert resp.status_code == 413
    assert called["n"] == 0   # rejected before reaching the transcriber


def test_transcribe_happy_path_returns_text(client, monkeypatch):
    captured = {}

    def _fake_transcribe(**kwargs):
        captured.update(kwargs)
        return "  Compare PD-1 versus PD-L1 inhibitors.  "

    monkeypatch.setattr(api, "transcribe_audio", _fake_transcribe)

    resp = client.post(
        "/api/transcribe",
        files={"audio": ("recording.webm", b"some-webm-bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    # The endpoint returns whatever the (mocked) transcriber produced.
    assert resp.json() == {"text": "  Compare PD-1 versus PD-L1 inhibitors.  "}
    # The recorded clip's bytes and filename are forwarded to the transcriber.
    assert captured["audio"] == b"some-webm-bytes"
    assert captured["filename"] == "recording.webm"


def test_transcribe_rejects_empty_upload(client, monkeypatch):
    monkeypatch.setattr(api, "transcribe_audio", lambda **kw: "unused")
    resp = client.post(
        "/api/transcribe",
        files={"audio": ("empty.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 422
