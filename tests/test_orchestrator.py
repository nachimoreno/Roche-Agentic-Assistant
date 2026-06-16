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
from attribution import AttributionResolver
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from document_source import LocalMarkdownSource
from embeddings import FastEmbedProvider
from llm import GroqClient, LLMClient
from orchestrator import Assistant, StreamDone, StreamMeta, StreamToken
from repositories import FeedbackRepository, SessionRepository
from retrieval import DocumentStore
from settings import Settings
from vector_store import ChromaVectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent
# Offline markdown corpus fixture (production ingests from Google Drive instead).
DOCS_PATH = REPO_ROOT / "tests" / "fixtures" / "docs"


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
    embedder = FastEmbedProvider()
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
                # A complaint that also asks something ("...how do I X?") embeds
                # an answerable question.
                contains_question = "how" in user.lower() or "?" in user
                return {
                    "language": language,
                    "type": "feedback",
                    "emotion": emotion,
                    "contains_question": contains_question,
                }
            return {"language": language, "type": "question"}

        # RAG-agent schema.
        return {
            "text": "Use a 70 percent isopropyl alcohol wipe.",
            "citations": [
                {"source": "06_cleaning_lab_devices.md", "section": "Centrifuges"}
            ],
        }

    def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.calls.append({"system": system, "user": user, "history": list(history)})
        # Prose, then the delimiter, then the JSON citations tail — chunked
        # the way a real token stream would arrive.
        yield "Use a 70 percent "
        yield "isopropyl alcohol wipe."
        yield "\n---CITATIONS---\n"
        yield '[{"source": "06_cleaning_lab_devices.md", "section": "Centrifuges"}]'


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
        attribution=AttributionResolver(doc_store),
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


def test_question_response_exposes_turn_id_and_persists_citations(engine, fake_assistant):
    from sqlmodel import Session as DbSession, select
    from db import TurnCitation

    assistant, _, _ = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")

    # The assistant turn id is surfaced so the UI can rate the answer.
    assert resp.turn_id is not None
    # Its citations are persisted (the link feedback later uses for attribution).
    with DbSession(engine) as db:
        cites = db.exec(
            select(TurnCitation).where(TurnCitation.turn_id == resp.turn_id)
        ).all()
    assert len(cites) >= 1
    assert cites[0].source == "06_cleaning_lab_devices.md"
    assert cites[0].rank == 0


def test_citations_resolve_process_from_front_matter(fake_assistant):
    # The fixture 06_cleaning_lab_devices.md carries front-matter
    # process: equipment-cleaning / department: lab-operations.
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")
    cits = sessions.citations_for_turn(resp.turn_id)
    assert cits and cits[0].source == "06_cleaning_lab_devices.md"
    assert cits[0].process == "equipment-cleaning"
    assert cits[0].department == "lab-operations"


def test_record_rating_attributes_via_citation_split(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")

    entry = assistant.record_rating(session_id=sid, turn_id=resp.turn_id, rating="down")
    assert entry.attribution_method == "citation"

    rows = feedback_repo.attributions_for(entry.id)
    assert rows, "citation attribution should write at least one row"
    assert abs(sum(r.weight for r in rows) - 1.0) < 1e-9
    assert {r.process for r in rows} == {"equipment-cleaning"}
    assert all(r.method == "citation" for r in rows)


def test_nlp_feedback_attributed_via_embedding(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    assistant.handle(sid, "This onboarding doc is really confusing.")

    nlp = [r for r in feedback_repo.list() if r.source == "nlp"]
    assert len(nlp) == 1
    assert nlp[0].attribution_method == "embedding"
    rows = feedback_repo.attributions_for(nlp[0].id)
    assert len(rows) == 1 and rows[0].method == "embedding"
    assert rows[0].weight == 1.0


def test_record_rating_writes_explicit_feedback_with_classified_comment(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")

    entry = assistant.record_rating(
        session_id=sid, turn_id=resp.turn_id, rating="down", comment="this is confusing"
    )
    assert entry.rating == "down"
    assert entry.source == "explicit"
    assert entry.comment == "this is confusing"
    # The comment is classified for sentiment (fake maps "confusing" -> confused).
    assert entry.emotion == "confused"
    assert any(r.source == "explicit" for r in feedback_repo.list())


def test_record_rating_without_comment_defaults_emotion_neutral(fake_assistant):
    assistant, _, _ = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")
    entry = assistant.record_rating(session_id=sid, turn_id=resp.turn_id, rating="up")
    assert entry.rating == "up"
    assert entry.emotion == "neutral"
    assert entry.comment is None


def test_record_rating_is_idempotent_per_turn(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")
    assistant.record_rating(session_id=sid, turn_id=resp.turn_id, rating="up")
    assistant.record_rating(session_id=sid, turn_id=resp.turn_id, rating="down")
    explicit = [r for r in feedback_repo.list() if r.source == "explicit"]
    assert len(explicit) == 1 and explicit[0].rating == "down"


def test_record_rating_rejects_unknown_or_user_turn(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    sessions.get_or_create(sid)
    user_turn = sessions.append_turn(sid, "user", "hi", language="english")

    with pytest.raises(ValueError):
        assistant.record_rating(session_id=sid, turn_id=new_id(), rating="up")   # unknown
    with pytest.raises(ValueError):
        assistant.record_rating(session_id=sid, turn_id=user_turn.id, rating="up")  # not assistant


def test_record_rating_rejects_bad_value(fake_assistant):
    assistant, _, _ = fake_assistant
    sid = new_id()
    resp = assistant.handle(sid, "How do I clean the centrifuge?")
    with pytest.raises(ValueError):
        assistant.record_rating(session_id=sid, turn_id=resp.turn_id, rating="meh")


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


def test_classifier_receives_prior_turns_on_followup(fake_assistant):
    """Routing must be context-aware: a short follow-up like "yes, that's
    the one" is only interpretable against the previous turn, so the
    conversation layer must be fed the prior turns (not just the bare
    message). Regression test for follow-ups mis-routed to feedback."""
    assistant, _, _ = fake_assistant
    sid = new_id()
    assistant.handle(sid, "How do I clean the centrifuge?")

    llm = assistant._cl._llm
    llm.calls.clear()
    assistant.handle(sid, "yes, that is the one")

    classify_user = llm.calls[0]["user"]
    assert "CONVERSATION SO FAR" in classify_user
    assert "centrifuge" in classify_user
    assert "LATEST MESSAGE" in classify_user
    assert classify_user.rstrip().endswith("yes, that is the one")


def test_feedback_with_embedded_question_answers_and_logs(fake_assistant):
    # A complaint that also asks a real question must be answered (RAG path)
    # AND still recorded as feedback.
    assistant, sessions, feedback_repo = fake_assistant
    sid = new_id()
    resp = assistant.handle(
        sid, "this onboarding is so confusing, how do I clean the centrifuge?"
    )

    assert resp.analysis.type == "feedback"
    assert resp.analysis.contains_question is True
    assert len(resp.citations) >= 1          # it actually answered the question
    assert "isopropyl" in resp.text.lower()
    assert len(feedback_repo.list()) == 1    # feedback still logged

    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert "isopropyl" in turns[-1].content.lower()


def test_stream_feedback_with_embedded_question_streams_answer(fake_assistant):
    assistant, _, feedback_repo = fake_assistant
    sid = new_id()
    events = list(assistant.handle_stream(
        sid, "this is frustrating, how do I clean the centrifuge?"
    ))

    assert events[0].analysis.type == "feedback"
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert len(done.citations) >= 1          # answered, not just acked
    assert len(feedback_repo.list()) == 1    # feedback still logged


# ---------------------------------------------------------------------------
# Streaming (handle_stream) — fast, no API calls.
# ---------------------------------------------------------------------------

def test_stream_question_emits_meta_tokens_done_and_persists(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    events = list(assistant.handle_stream(sid, "How do I clean the centrifuge?"))

    # First event is meta, last is done.
    assert isinstance(events[0], StreamMeta)
    assert events[0].analysis.type == "question"
    done = events[-1]
    assert isinstance(done, StreamDone)

    tokens = [e for e in events if isinstance(e, StreamToken)]
    full = "".join(e.text for e in tokens)
    assert "isopropyl" in full
    # The sentinel and JSON tail must never leak as prose.
    assert "---CITATIONS---" not in full
    assert "source" not in full

    assert done.text == full
    assert len(done.citations) == 1
    assert done.citations[0].source == "06_cleaning_lab_devices.md"

    # Assistant turn persisted with the assembled prose (not the raw stream).
    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[-1].content == full


def test_stream_aborted_midway_does_not_persist_assistant_turn(fake_assistant):
    # When the client hits Stop / disconnects, the SSE generator is closed,
    # raising GeneratorExit mid-stream. The server must NOT persist the
    # assistant turn here — the client saves the partial it kept on screen via
    # POST /api/sessions/{id}/messages. Persisting in both places would dup.
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    gen = assistant.handle_stream(sid, "How do I clean the centrifuge?")

    assert isinstance(next(gen), StreamMeta)        # consume meta
    assert isinstance(next(gen), StreamToken)        # consume first token
    gen.close()                                      # abort while mid-stream

    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user"]       # no assistant turn written


def test_handle_stamps_user_id_and_sets_title(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    assistant.handle(sid, "How do I clean the centrifuge?", user_id="user-1")

    session = sessions.get(sid)
    assert session.user_id == "user-1"
    assert session.title == "How do I clean the centrifuge?"
    # The session shows up in that user's list, not another's.
    assert [s.id for s in sessions.list_sessions("user-1")] == [sid]
    assert sessions.list_sessions("user-2") == []


def test_title_set_once_from_first_message(fake_assistant):
    assistant, sessions, _ = fake_assistant
    sid = new_id()
    assistant.handle(sid, "First question about cleaning?", user_id="u")
    assistant.handle(sid, "A second, different question?", user_id="u")
    assert sessions.get(sid).title == "First question about cleaning?"


def test_stream_feedback_acks_and_persists_feedback(fake_assistant):
    assistant, sessions, feedback_repo = fake_assistant
    sid = new_id()
    events = list(assistant.handle_stream(sid, "This onboarding doc is really confusing."))

    assert isinstance(events[0], StreamMeta)
    assert events[0].analysis.type == "feedback"
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.citations == []
    assert done.text  # an acknowledgement was streamed

    rows = feedback_repo.list()
    assert len(rows) == 1
    assert rows[0].emotion == "confused"

    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user", "assistant"]


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
        attribution=AttributionResolver(doc_store),
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
def test_live_stream_question_yields_clean_prose_and_citations(live_assistant):
    assistant, sessions, _ = live_assistant
    sid = new_id()
    events = list(assistant.handle_stream(sid, "How do I clean the centrifuge?"))

    assert isinstance(events[0], StreamMeta)
    assert events[0].analysis.type == "question"

    tokens = [e for e in events if isinstance(e, StreamToken)]
    full = "".join(e.text for e in tokens)
    assert full.strip()
    # The real model must not leak the delimiter into the streamed prose.
    assert "---CITATIONS---" not in full

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.text == full
    assert done.citations  # grounded answer should cite at least one source
    # Persisted assistant turn matches the assembled prose.
    assert sessions.recent_turns(sid, n=10)[-1].content == full


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
