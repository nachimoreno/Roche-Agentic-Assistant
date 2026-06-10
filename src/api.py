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
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

# Make sibling modules importable whether invoked as
# `python src/api.py` or `uvicorn src.api:app` from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import new_id
from llm import LLMAuthError
from logging_setup import setup_logging
from main import build_assistant
from orchestrator import Assistant, StreamDone, StreamMeta, StreamToken
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
    try:
        app.state.assistant = build_assistant(_settings)
    except LLMAuthError:
        logger.error(
            "startup.auth.failed: Groq rejected the API key. Fix GROQ_API_KEY "
            "in .env (a single valid key) and restart the server."
        )
        raise
    yield


app = FastAPI(title="Roche Scientist Assistant", lifespan=lifespan)

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


class SessionOut(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions", response_model=SessionOut, status_code=201)
def create_session():
    """Create a new conversation session and return its ID."""
    return SessionOut(session_id=str(new_id()))


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, req: ChatRequest, request: Request):
    """Send a message in a session and receive the assistant's reply."""
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    assistant: Assistant = request.app.state.assistant
    try:
        resp = assistant.handle(sid, req.message)
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
def chat_stream(session_id: str, req: ChatRequest, request: Request):
    """Same as /chat, but streams the reply token-by-token over SSE.

    Event sequence: `meta` (language/type/emotion) -> `token`* (text deltas)
    -> `done` (citations). On failure an `error` event is emitted instead of
    leaking the exception.
    """
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    assistant: Assistant = request.app.state.assistant

    def event_stream():
        try:
            for ev in assistant.handle_stream(sid, req.message):
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
