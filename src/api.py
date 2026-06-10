"""
api.py
------
FastAPI HTTP server for the Roche Scientist Assistant.

Run from the project root:
    uvicorn src.api:app --reload
    # or directly:
    python src/api.py
"""
from __future__ import annotations

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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import RAGAgent
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from document_source import LocalMarkdownSource
from embeddings import FastEmbedProvider
from llm import GroqClient
from logging_setup import setup_logging
from orchestrator import Assistant
from repositories import FeedbackRepository, SessionRepository
from retrieval import DocumentStore
from settings import Settings
from vector_store import ChromaVectorStore

load_dotenv()
_settings = Settings()
setup_logging(
    level=_settings.log_level,
    fmt=_settings.log_format,
    log_file=_settings.log_file,
)


def _build(settings: Settings) -> Assistant:
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    create_all(engine)

    llm = GroqClient(api_key=settings.groq_api_key, model=settings.model_name)
    embedder = FastEmbedProvider(model_name=settings.embedding_model)
    vector_store = ChromaVectorStore(
        path=settings.chroma_path,
        collection_name=f"roche_{embedder.name.replace('/', '_')}",
    )
    source = LocalMarkdownSource(path=settings.docs_path)
    docs = DocumentStore(
        source=source,
        embedder=embedder,
        vector_store=vector_store,
        manifest_path=f"{settings.chroma_path}/manifest.json",
    )
    docs.ingest()

    return Assistant(
        conversation_layer=ConversationLayer(llm=llm),
        rag_agent=RAGAgent(document_store=docs, llm=llm, top_k=settings.top_k),
        session_repo=SessionRepository(engine),
        feedback_repo=FeedbackRepository(engine),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assistant = _build(_settings)
    yield


app = FastAPI(title="Roche Scientist Assistant", lifespan=lifespan)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_static / "index.html"))


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(str(_static / "dashboard.html"))


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


class TurnOut(BaseModel):
    role: str
    content: str
    language: Optional[str] = None


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
        raise HTTPException(
            status_code=422, detail="Invalid session_id format"
        )

    assistant: Assistant = request.app.state.assistant
    try:
        resp = assistant.handle(sid, req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_settings.host, port=_settings.port)
