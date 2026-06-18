"""
api.py
------
FastAPI HTTP server for the Roche Scientist Assistant.

Run from the project root:
    uvicorn src.api:app --reload
    # or directly:
    python src/api.py

The Assistant is assembled by `main.build_assistant`, the single composition
root shared with the CLI — so the API and CLI always wire up the same stack
(including the configured DocumentSource, e.g. Google Drive).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

# Make sibling modules importable whether invoked as
# `python src/api.py` or `uvicorn src.api:app` from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from datetime import datetime, timedelta, timezone

from auth import get_current_user, hash_password, require_admin, verify_password
from db import User, utcnow
from llm import LLMAuthError, transcribe_audio
from logging_setup import setup_logging
from main import build_assistant, build_drive_source, build_engine
from orchestrator import Assistant, StreamDone, StreamMeta, StreamToken
from repositories import (
    EmailTakenError,
    FeedbackRepository,
    SessionRepository,
    UserRepository,
)
from settings import Settings

load_dotenv()
_settings = Settings()
setup_logging(
    level=_settings.log_level,
    fmt=_settings.log_format,
    log_file=_settings.log_file,
)

logger = logging.getLogger(__name__)

# Application logs are routed to a file (settings.log_file) to keep the console
# clean. Startup status — notably whether Google Drive connected — still needs
# to reach the operator's terminal, so it gets a dedicated stdout logger.
# propagate=True keeps a copy in the log file too, via the root handler.
_console = logging.getLogger("roche.startup")
if not _console.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)


def _report_drive_status(settings: Settings) -> None:
    """Probe the Google Drive integration at startup and report to the console.

    Logs whether Drive is disabled, skipped, connected, or failed — and on
    failure, why. This is informational only: `CompositeSource` keeps ingestion
    running on local docs even when Drive is down, so this never blocks startup.
    """
    if settings.document_source == "local":
        _console.info("google_drive: disabled (document_source='local')")
        return

    if not settings.drive_folder_id:
        _console.info(
            "google_drive: skipped — document_source=%r but DRIVE_FOLDER_ID is "
            "not set; serving local docs only.",
            settings.document_source,
        )
        return

    try:
        count = build_drive_source(settings).check_connection()
    except Exception as exc:
        _console.error(
            "google_drive: FAILED to connect to folder %s — %s: %s",
            settings.drive_folder_id,
            type(exc).__name__,
            exc,
        )
        return

    _console.info(
        "google_drive: OK — connected to folder %s (%d+ files visible).",
        settings.drive_folder_id,
        count,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine(_settings)
    app.state.users = UserRepository(engine)
    app.state.sessions = SessionRepository(engine)
    app.state.feedback = FeedbackRepository(engine)
    _report_drive_status(_settings)
    try:
        app.state.assistant = build_assistant(_settings, engine=engine)
    except LLMAuthError:
        logger.error(
            "startup.auth.failed: Groq rejected the API key. Fix GROQ_API_KEY "
            "in .env (a single valid key) and restart the server."
        )
        raise
    yield


app = FastAPI(title="Roche Scientist Assistant", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=_settings.session_secret,
    session_cookie=_settings.session_cookie,
    https_only=_settings.session_https_only,
    same_site="lax",
)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_static / "index.html"))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class CitationOut(BaseModel):
    source: str
    section: str
    title: Optional[str] = None        # human-readable doc title for display
    url: Optional[str] = None          # click-through link to the source doc


class ChatResponse(BaseModel):
    text: str
    language: str
    type: str
    emotion: Optional[str] = None
    citations: list[CitationOut]
    turn_id: Optional[str] = None      # the assistant turn, for rating
    follow_ups: list[str] = []         # suggested next questions (chips)


class SessionItemOut(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class MessageOut(BaseModel):
    id: str                            # the turn id, so the UI can rate it
    role: str
    content: str
    language: Optional[str] = None
    created_at: str


class AppendPartialRequest(BaseModel):
    content: str
    language: Optional[str] = None


class RateRequest(BaseModel):
    rating: str                        # "up" | "down"
    comment: Optional[str] = None


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None
    role: Optional[str] = None         # "user" | "admin" — drives the dashboard link


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role or "user",
    )


def _iso_utc(dt: datetime) -> str:
    """Serialize a stored timestamp as an explicit-UTC ISO string.

    Timestamps are written in UTC, but SQLite returns them tz-naive, so a bare
    `.isoformat()` has no offset and browsers parse it as *local* time (showing
    e.g. "2h ago" right after sending). Stamping UTC makes the client parse it
    correctly regardless of the viewer's timezone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _maybe_promote_admin(users: UserRepository, user: User) -> User:
    """Promote an allowlisted email to admin (ADMIN_EMAILS in .env).

    This is the only path that grants the role — `register`/`login` request
    bodies never carry one. Promotion is one-way here; revocation is a manual
    `set_role` (deliberate: removing an email from the allowlist shouldn't
    silently demote an admin mid-investigation).
    """
    if user.role != "admin" and user.email in _settings.admin_email_set:
        promoted = users.set_role(user.id, "admin")
        if promoted is not None:
            return promoted
    return user


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=UserOut, status_code=201)
def register(req: RegisterRequest, request: Request):
    """Create an account and start a session (auto-login)."""
    if not _EMAIL_RE.match(req.email.strip()):
        raise HTTPException(status_code=422, detail="Please provide a valid email address.")
    if len(req.password) < _settings.min_password_length:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at least {_settings.min_password_length} characters.",
        )

    users: UserRepository = request.app.state.users
    try:
        user = users.create(
            email=req.email,
            password_hash=hash_password(req.password),
            display_name=req.display_name,
        )
    except EmailTakenError:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    user = _maybe_promote_admin(users, user)
    request.session["user_id"] = str(user.id)
    return _user_out(user)


@app.post("/api/auth/login", response_model=UserOut)
def login(req: LoginRequest, request: Request):
    users: UserRepository = request.app.state.users
    user = users.get_by_email(req.email)
    # Generic error + always-run verify avoids leaking which emails exist.
    if user is None or not verify_password(user.password_hash, req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = _maybe_promote_admin(users, user)
    request.session["user_id"] = str(user.id)
    return _user_out(user)


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


# ---------------------------------------------------------------------------
# Voice input — server-side speech-to-text
# ---------------------------------------------------------------------------

# Guard against oversized uploads. Voice prompts are short; 25 MB is already
# generous and matches Groq's transcription size limit.
_MAX_AUDIO_BYTES = 25 * 1024 * 1024


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Transcribe a recorded voice clip to text (Groq Whisper).

    The browser records with MediaRecorder and POSTs the clip here, so voice
    input no longer depends on the browser's Web Speech API (which errors out
    in Edge / corporate networks / the desktop app).
    """
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio upload.")
    if len(data) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio clip too large.")

    try:
        text = transcribe_audio(
            api_key=_settings.groq_api_key,
            audio=data,
            filename=audio.filename or "recording.webm",
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean 502 to the client
        logger.error("api.transcribe.failed", extra={"error": str(exc)})
        raise HTTPException(status_code=502, detail="Transcription failed.")

    return {"text": text}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _parse_sid(session_id: str) -> UUID:
    try:
        return UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format")


def _session_item(s) -> SessionItemOut:
    return SessionItemOut(id=str(s.id), title=s.title, created_at=_iso_utc(s.created_at))


def _require_owned(request: Request, sid: UUID, user: User):
    """Return the user's session or 404 (404, not 403, to avoid leaking ids)."""
    s = request.app.state.sessions.get(sid)
    if s is None or s.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    return s


def _check_access(request: Request, sid: UUID, user: User) -> None:
    """Allow a new (nonexistent) session, but reject another user's session."""
    s = request.app.state.sessions.get(sid)
    if s is not None and s.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Session not found")


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions", response_model=list[SessionItemOut])
def list_sessions(request: Request, user: User = Depends(get_current_user)):
    return [_session_item(s) for s in request.app.state.sessions.list_sessions(str(user.id))]


@app.post("/api/sessions", response_model=SessionItemOut, status_code=201)
def create_session(
    request: Request,
    body: Optional[CreateSessionRequest] = None,
    user: User = Depends(get_current_user),
):
    s = request.app.state.sessions.create(
        user_id=str(user.id), title=(body.title if body else None)
    )
    return _session_item(s)


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_messages(session_id: str, request: Request, user: User = Depends(get_current_user)):
    sid = _parse_sid(session_id)
    _require_owned(request, sid, user)
    return [
        MessageOut(
            id=str(t.id),
            role=t.role,
            content=t.content,
            language=t.language,
            created_at=_iso_utc(t.created_at),
        )
        for t in request.app.state.sessions.messages(sid)
    ]


@app.post("/api/sessions/{session_id}/messages", response_model=MessageOut, status_code=201)
def append_partial_message(
    session_id: str,
    body: AppendPartialRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Persist a partial assistant answer after the client aborted the stream.

    When the user hits Stop (or the connection drops) the server-side
    generator is abandoned before it can persist the assistant turn, but the
    UI keeps the partial text on screen. The client calls this so the stored
    history matches what the user actually saw.

    This is a narrow, abort-recovery endpoint, not a general "write an
    assistant message" API. The only legitimate state for it is right after a
    streamed user turn that the server never got to answer, so it accepts a
    write only when the most recent turn is an unanswered user turn. That
    blocks a client from forging assistant turns (which feed back into model
    history) at arbitrary points — empty sessions, stacked assistant turns, or
    mid-conversation injection are all rejected with 409.

    Idempotent: if the stream finished server-side in the small window before
    the client aborted, the assistant turn is already persisted with the full
    text — which, since every token reached the client, is identical to what
    the client kept. In that case (and on a client retry) return the existing
    turn instead of writing a duplicate.
    """
    sid = _parse_sid(session_id)
    _require_owned(request, sid, user)
    if not body.content.strip():
        raise HTTPException(status_code=422, detail="content must not be empty")
    sessions = request.app.state.sessions
    recent = sessions.recent_turns(sid, n=1)
    last = recent[-1] if recent else None
    if last is not None and last.role == "assistant" and last.content == body.content:
        # Race or retry: the full turn is already stored. Return it, no dup.
        t = last
    elif last is not None and last.role == "user":
        # The expected abort-recovery case: answer the pending user turn.
        t = sessions.append_turn(
            sid, role="assistant", content=body.content, language=body.language
        )
    else:
        # No unanswered user turn to attach to → reject rather than forge one.
        raise HTTPException(
            status_code=409,
            detail="no unanswered user turn to attach a partial answer to",
        )
    return MessageOut(
        id=str(t.id),
        role=t.role,
        content=t.content,
        language=t.language,
        created_at=_iso_utc(t.created_at),
    )


@app.patch("/api/sessions/{session_id}", response_model=SessionItemOut)
def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    sid = _parse_sid(session_id)
    _require_owned(request, sid, user)
    request.app.state.sessions.rename(sid, body.title)
    return _session_item(request.app.state.sessions.get(sid))


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, request: Request, user: User = Depends(get_current_user)):
    sid = _parse_sid(session_id)
    _require_owned(request, sid, user)
    request.app.state.sessions.soft_delete_session(sid)


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(
    session_id: str,
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Send a message in a session and receive the assistant's reply."""
    sid = _parse_sid(session_id)
    _check_access(request, sid, user)

    assistant: Assistant = request.app.state.assistant
    try:
        resp = assistant.handle(sid, req.message, user_id=str(user.id))
    except Exception:
        # Log the detail server-side; don't leak internals to the client.
        logger.exception("api.chat.failed", extra={"session_id": str(sid)})
        raise HTTPException(
            status_code=500, detail="Internal error handling the request."
        )

    return ChatResponse(
        text=resp.text,
        language=resp.analysis.language,
        type=resp.analysis.type,
        emotion=resp.analysis.emotion,
        citations=[
            CitationOut(source=c.source, section=c.section, title=c.title, url=c.url)
            for c in resp.citations
        ],
        turn_id=str(resp.turn_id) if resp.turn_id else None,
        follow_ups=resp.follow_ups,
    )


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _error_category(exc: Exception) -> tuple[str, str]:
    """Map an exception to a (category, client-safe detail) pair.

    Categorized by HTTP status when the provider exposes one (groq's API
    errors carry `status_code`), so the UI can distinguish a misconfigured
    assistant from a transient hiccup. Never returns internal detail text.
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "auth", "The assistant is not configured correctly (authentication failed)."
    if status == 429:
        return "rate_limit", "The assistant is busy right now. Please wait a moment and try again."
    return "internal", "Internal error handling the request."


@app.post("/api/sessions/{session_id}/chat/stream")
def chat_stream(
    session_id: str,
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Same as /chat, but streams the reply token-by-token over SSE.

    Event sequence: `meta` (language/type/emotion) -> `token`* (text deltas)
    -> `done` (citations). On failure an `error` event is emitted instead of
    leaking the exception. Auth + ownership are checked before streaming starts
    so they surface as real HTTP status codes, not SSE frames.
    """
    sid = _parse_sid(session_id)
    _check_access(request, sid, user)

    assistant: Assistant = request.app.state.assistant
    uid = str(user.id)

    def event_stream():
        try:
            for ev in assistant.handle_stream(sid, req.message, user_id=uid):
                if isinstance(ev, StreamMeta):
                    yield _sse("meta", {
                        "language": ev.analysis.language,
                        "type": ev.analysis.type,
                        "emotion": ev.analysis.emotion,
                    })
                elif isinstance(ev, StreamToken):
                    yield _sse("token", {"text": ev.text})
                elif isinstance(ev, StreamDone):
                    yield _sse("done", {
                        "citations": [
                            {"source": c.source, "section": c.section, "title": c.title, "url": c.url}
                            for c in ev.citations
                        ],
                        "turn_id": str(ev.turn_id) if ev.turn_id else None,
                        "follow_ups": ev.follow_ups,
                    })
        except Exception as exc:
            category, detail = _error_category(exc)
            logger.exception(
                "api.chat_stream.failed",
                extra={"session_id": str(sid), "category": category},
            )
            yield _sse("error", {"category": category, "detail": detail})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Feedback (explicit ratings)
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/turns/{turn_id}/feedback", status_code=201)
def rate_turn(
    session_id: str,
    turn_id: str,
    body: RateRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Record a thumb up/down (+ optional comment) on an assistant answer.

    Ownership is checked here; the assistant validates that the turn is an
    assistant turn in the session. Idempotent per turn (re-rating updates).
    """
    sid = _parse_sid(session_id)
    try:
        tid = UUID(turn_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid turn_id format")
    if body.rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
    _require_owned(request, sid, user)

    assistant: Assistant = request.app.state.assistant
    try:
        assistant.record_rating(
            session_id=sid, turn_id=tid, rating=body.rating, comment=body.comment
        )
    except ValueError:
        # Turn doesn't exist / isn't an assistant turn in this session.
        raise HTTPException(status_code=404, detail="Answer not found")
    except Exception:
        logger.exception("api.rate.failed", extra={"session_id": str(sid)})
        raise HTTPException(status_code=500, detail="Internal error handling the request.")

    return {"ok": True}


# ---------------------------------------------------------------------------
# Analytics (admin-only)
# ---------------------------------------------------------------------------

_pages = Path(__file__).parent / "pages"


def _since(days: Optional[int]) -> Optional[datetime]:
    """A UTC cutoff `days` ago, or None for all time. Clamped to >= 1."""
    if days is None:
        return None
    return utcnow() - timedelta(days=max(1, days))


@app.get("/admin", include_in_schema=False)
def admin_page(user: User = Depends(require_admin)):
    """The analytics dashboard. Lives outside the /static mount so the page
    itself (not just its data) is behind the admin gate."""
    return FileResponse(str(_pages / "admin.html"))


@app.get("/api/analytics/summary")
def analytics_summary(
    request: Request,
    days: Optional[int] = None,
    user: User = Depends(require_admin),
):
    feedback: FeedbackRepository = request.app.state.feedback
    return feedback.summary(since=_since(days), tenant_id=user.tenant_id)


@app.get("/api/analytics/hotspots")
def analytics_hotspots(
    request: Request,
    dimension: str = "process",
    days: Optional[int] = None,
    limit: int = 10,
    user: User = Depends(require_admin),
):
    feedback: FeedbackRepository = request.app.state.feedback
    try:
        return feedback.hotspots(
            dimension=dimension,
            since=_since(days),
            tenant_id=user.tenant_id,
            limit=max(1, min(limit, 50)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/analytics/trend")
def analytics_trend(
    request: Request,
    days: Optional[int] = None,
    user: User = Depends(require_admin),
):
    feedback: FeedbackRepository = request.app.state.feedback
    return feedback.trend(since=_since(days), tenant_id=user.tenant_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_settings.host, port=_settings.port)
