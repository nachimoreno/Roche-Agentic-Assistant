"""
End-to-end orchestrator tests.

Two flavors:
- A `live` flavor exercising real Groq + real retrieval against the
  on-disk corpus.
- A `fake` flavor using a hand-rolled `FakeLLMClient` to prove the
  interface seams are real (no Groq dependency).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from agent import RAGAgent
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from document_source import LocalMarkdownSource
from embeddings import SentenceTransformersProvider
from llm import GroqClient, LLMClient
from orchestrator import Assistant
from repositories import FeedbackRepository, SessionRepository
from retrieval import DocumentStore
from settings import Settings
from vector_store import ChromaVectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = REPO_ROOT / "data" / "docs"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def doc_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chroma_orch")
    embedder = SentenceTransformersProvider()
    store = ChromaVectorStore(path=str(tmp), collection_name="test_orchestrator")
    docs = DocumentStore(
        source=LocalMarkdownSource(DOCS_PATH),
        embedder=embedder,
        vector_store=store,
        manifest_path=str(tmp / "manifest.json"),
    )
    docs.ingest()
    return docs


# ---------------------------------------------------------------------------
# Fake LLM — proves the seams are real (no Groq dependency).
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Deterministic stand-in for `LLMClient` used in interface-swap tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "history": list(history)})

        schema_title = schema.get("title", "")
        # Conversation-layer schema: classify by simple keyword heuristics.
        if "AnalysisResult" in schema_title:
            language = "german" if any(
                w in user.lower() for w in ("wie", "ich", "können", "system")
            ) else "english"
            if any(w in user.lower() for w in ("confusing", "frustrat", "hate", "love", "great")):
                emotion = "confused" if "confusing" in user.lower() else "frustrated"
                return {"language": language, "type": "feedback", "emotion": emotion}
            return {"language": language, "type": "question"}

        # RAG-agent schema.
        return {
            "text": "Use a 70 percent isopropyl alcohol wipe.",
            "citations": [
                {"source": "06_cleaning_lab_devices.md", "section": "Centrifuges"}
            ],
        }


# ---------------------------------------------------------------------------
# Fake-LLM end-to-end — fast, no API calls.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_assistant(engine, doc_store):
    llm: LLMClient = FakeLLMClient()
    cl = ConversationLayer(llm=llm)
    agent = RAGAgent(document_store=doc_store, llm=llm, top_k=3)
    sessions = SessionRepository(engine)
    feedback = FeedbackRepository(engine)
    assistant = Assistant(
        conversation_layer=cl,
        rag_agent=agent,
        session_repo=sessions,
        feedback_repo=feedback,
    )
    return assistant, sessions, feedback


def test_question_path_persists_turns_and_returns_citations(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")

    assert resp.analysis.type == "question"
    assert resp.text
    assert len(resp.citations) >= 1

    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user", "assistant"]


def test_feedback_path_writes_feedback_row(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    resp = assistant.handle(
        sid, "This onboarding doc is really confusing."
    )

    assert resp.analysis.type == "feedback"
    rows = feedback_repo.list()
    assert len(rows) == 1
    assert rows[0].emotion == "confused"
    assert rows[0].language == "english"


def test_multi_turn_history_grows_in_order(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    assistant.handle(sid, "How do I clean the centrifuge?")
    assistant.handle(sid, "And where do I log the clean?")

    turns = sessions.recent_turns(sid, n=10)
    roles = [t.role for t in turns]
    assert roles == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# Live end-to-end — exercises real Groq.
# ---------------------------------------------------------------------------

@pytest.fixture
def live_assistant(engine, doc_store):
    settings = Settings()
    llm = GroqClient(api_key=settings.groq_api_key, model=settings.model_name)
    cl = ConversationLayer(llm=llm)
    agent = RAGAgent(document_store=doc_store, llm=llm, top_k=settings.top_k)
    sessions = SessionRepository(engine)
    feedback = FeedbackRepository(engine)
    return Assistant(
        conversation_layer=cl,
        rag_agent=agent,
        session_repo=sessions,
        feedback_repo=feedback,
    ), sessions, feedback


@pytest.mark.live
def test_live_english_question(live_assistant):
    assistant, _, _ = live_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")
    assert resp.analysis.type == "question"
    assert resp.analysis.language == "english"
    assert resp.text
    assert resp.citations


@pytest.mark.live
def test_live_german_question(live_assistant):
    assistant, _, _ = live_assistant
    sid = new_id()
    resp = assistant.handle(sid, "Wie kann ich den Probenbestand prüfen?")
    assert resp.analysis.type == "question"
    assert resp.analysis.language == "german"
    assert resp.citations


@pytest.mark.live
def test_live_agent_describes_its_capabilities(live_assistant):
    assistant, _, _ = live_assistant
    sid = new_id()
    resp = assistant.handle(sid, "What can you do?")
    assert resp.analysis.type == "question"
    text = resp.text.lower()
    # The answer should mention at least a couple of real capabilities.
    hits = sum(
        keyword in text
        for keyword in (
            "feedback",
            "instrument",
            "stock",
            "cleaning",
            "incident",
            "onboard",
            "language",
        )
    )
    assert hits >= 2, f"capability answer mentioned too few features: {resp.text!r}"


@pytest.mark.live
def test_live_english_feedback_is_persisted(live_assistant):
    assistant, _, feedback_repo = live_assistant
    sid = new_id()
    resp = assistant.handle(
        sid, "This onboarding doc is really confusing and frustrating."
    )
    assert resp.analysis.type == "feedback"
    rows = feedback_repo.list()
    assert len(rows) == 1
    assert rows[0].language == "english"
