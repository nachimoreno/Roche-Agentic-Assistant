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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from auth import get_current_user, hash_password, verify_password
from db import User
from llm import LLMAuthError
from logging_setup import setup_logging
from main import build_assistant, build_engine
from orchestrator import Assistant, StreamDone, StreamMeta, StreamToken
from repositories import EmailTakenError, SessionRepository, UserRepository
from settings import Settings

load_dotenv()
_settings = Settings()
setup_logging(
    level=_settings.log_level,
    fmt=_settings.log_format,
    log_file=_settings.log_file,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine(_settings)
    app.state.users = UserRepository(engine)
    app.state.sessions = SessionRepository(engine)
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


class ChatResponse(BaseModel):
    text: str
    language: str
    type: str
    emotion: Optional[str] = None
    citations: list[CitationOut]


class SessionItemOut(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class MessageOut(BaseModel):
    role: str
    content: str
    language: Optional[str] = None
    created_at: str


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


def _user_out(user: User) -> UserOut:
    return UserOut(id=str(user.id), email=user.email, display_name=user.display_name)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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

    request.session["user_id"] = str(user.id)
    return _user_out(user)


@app.post("/api/auth/login", response_model=UserOut)
def login(req: LoginRequest, request: Request):
    users: UserRepository = request.app.state.users
    user = users.get_by_email(req.email)
    # Generic error + always-run verify avoids leaking which emails exist.
    if user is None or not verify_password(user.password_hash, req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

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
# Session helpers
# ---------------------------------------------------------------------------

def _parse_sid(session_id: str) -> UUID:
    try:
        return UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format")


def _session_item(s) -> SessionItemOut:
    return SessionItemOut(id=str(s.id), title=s.title, created_at=s.created_at.isoformat())


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
            role=t.role,
            content=t.content,
            language=t.language,
            created_at=t.created_at.isoformat(),
        )
        for t in request.app.state.sessions.messages(sid)
    ]


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
            CitationOut(source=c.source, section=c.section)
            for c in resp.citations
        ],
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
                            {"source": c.source, "section": c.section}
                            for c in ev.citations
                        ],
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_settings.host, port=_settings.port)
