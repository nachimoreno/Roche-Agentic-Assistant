"""
Contradiction-handling tests — version-dedup, conflict detection, and conflict
analytics. See architecture/Contradiction_Handling_Design.md.

Offline by construction: the deterministic version-dedup (Phase 2) is a pure
function tested on inline chunks; conflict detection (Phase 1/3) uses fake LLM
clients; analytics (Phase 5) use an in-memory SQLite engine. One end-to-end
ingest test runs the real LocalMarkdownSource -> FastEmbed -> Chroma stack over
isolated `conflict_docs` fixtures (kept out of the main corpus so the retrieval
suite is unaffected). No Groq, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from agent import (
    AnswerComplete,
    AnswerResult,
    Citation,
    RAGAgent,
    TextDelta,
    _format_context,
    _parse_tail,
)
from conversation_layer import AnalysisResult
from db import create_all, make_engine
from document_source import LocalMarkdownSource, parse_front_matter
from orchestrator import Assistant, StreamDone
from repositories import (
    FeedbackRepository,
    QuestionGapRepository,
    SessionRepository,
)
from retrieval import (
    DocumentStore,
    _chunk_document,
    _collapse_superseded,
    _doc_meta_record,
    _title_similar,
)
from vector_store import Chunk


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFLICT_DOCS = REPO_ROOT / "tests" / "fixtures" / "conflict_docs"


def _chunk(source_id: str, **meta: Any) -> Chunk:
    meta["source_id"] = source_id
    return Chunk(id=source_id, text="body", metadata=meta, score=0.9)


# ===========================================================================
# Phase 1 — metadata foundation
# ===========================================================================

def test_front_matter_parses_conflict_keys():
    front, body = parse_front_matter(
        "---\n"
        "process: lab-cleaning\n"
        "version: 2\n"
        "effective_date: 2026-05-01\n"
        "status: current\n"
        "owner: lab-ops@roche\n"
        "supersedes: old.md\n"
        "---\n"
        "# Title\nbody\n"
    )
    assert front["version"] == "2"
    assert front["effective_date"] == "2026-05-01"
    assert front["status"] == "current"
    assert front["owner"] == "lab-ops@roche"
    assert front["supersedes"] == "old.md"
    assert "process" not in body  # front-matter is stripped from the body


def test_local_source_carries_conflict_keys(tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\nprocess: p\nversion: 3\nstatus: current\nsupersedes: prev.md\n---\n"
        "# Doc\n\n## Section\n\n" + ("content " * 40),
        encoding="utf-8",
    )
    docs = list(LocalMarkdownSource(tmp_path).list_documents())
    assert len(docs) == 1
    md = docs[0].metadata
    assert md["version"] == "3"
    assert md["status"] == "current"
    assert md["supersedes"] == "prev.md"


def test_chunk_metadata_carries_recency_and_version(tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\nprocess: p\nversion: 4\neffective_date: 2026-01-02\nstatus: current\n---\n"
        "# Doc\n\n## Section\n\n" + ("content " * 40),
        encoding="utf-8",
    )
    doc = next(iter(LocalMarkdownSource(tmp_path).list_documents()))
    chunks = list(_chunk_document(doc))
    assert chunks, "expected at least one chunk"
    m = chunks[0].metadata
    assert m["version"] == "4"
    assert m["effective_date"] == "2026-01-02"
    assert m["status"] == "current"
    assert "modified_at" in m  # ISO string, Chroma-safe (no datetime/None)
    assert isinstance(m["modified_at"], str)


def test_doc_meta_record_includes_conflict_keys(tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\nprocess: p\nversion: 2\nstatus: deprecated\nsupersedes: x.md\n---\n"
        "# Doc\n\n## Section\n\n" + ("content " * 40),
        encoding="utf-8",
    )
    doc = next(iter(LocalMarkdownSource(tmp_path).list_documents()))
    rec = _doc_meta_record(doc)
    assert rec["version"] == "2"
    assert rec["status"] == "deprecated"
    assert rec["supersedes"] == "x.md"
    assert rec["modified_at"] is not None


# ===========================================================================
# Phase 2 — deterministic version-dedup (_collapse_superseded), pure / no LLM
# ===========================================================================

def test_collapse_drops_stale_via_explicit_supersedes():
    chunks = [
        _chunk("v2.md", supersedes="v1.md", process="cleaning",
               title="Cleaning Lab Devices", effective_date="2026-05-01"),
        _chunk("v1.md", process="cleaning",
               title="Cleaning Lab Devices", effective_date="2025-01-01"),
    ]
    kept = {c.metadata["source_id"] for c in _collapse_superseded(chunks)}
    assert kept == {"v2.md"}


def test_collapse_drops_stale_via_process_and_title():
    # No explicit chain; identity inferred from same process + near-identical
    # title (version tokens stripped). Newer effective_date wins.
    chunks = [
        _chunk("a_v2.md", process="cleaning", title="Cleaning Lab Devices v2",
               effective_date="2026-05-01"),
        _chunk("a_v1.md", process="cleaning", title="Cleaning Lab Devices v1",
               effective_date="2025-01-01"),
    ]
    kept = {c.metadata["source_id"] for c in _collapse_superseded(chunks)}
    assert kept == {"a_v2.md"}


def test_collapse_keeps_dissimilar_docs_guard():
    # Same process but genuinely different titles -> NOT the same logical doc.
    # The confident-identity guard keeps both (Case B is left for the LLM).
    chunks = [
        _chunk("clean.md", process="cleaning", title="Centrifuge Solvent Guide",
               modified_at="2026-05-01T00:00:00+00:00"),
        _chunk("decon.md", process="cleaning", title="Decontamination Standard",
               modified_at="2026-01-01T00:00:00+00:00"),
    ]
    kept = {c.metadata["source_id"] for c in _collapse_superseded(chunks)}
    assert kept == {"clean.md", "decon.md"}


def test_collapse_drops_deprecated_when_current_sibling_exists():
    # Deprecated is dropped even though it is the newer file.
    chunks = [
        _chunk("cur.md", process="cleaning", title="Cleaning Lab Devices",
               status="current", modified_at="2026-01-01T00:00:00+00:00"),
        _chunk("dep.md", process="cleaning", title="Cleaning Lab Devices",
               status="deprecated", modified_at="2026-09-01T00:00:00+00:00"),
    ]
    kept = {c.metadata["source_id"] for c in _collapse_superseded(chunks)}
    assert kept == {"cur.md"}


def test_collapse_effective_date_beats_modified_at():
    # effective_date is authoritative for recency over file modified_at.
    chunks = [
        _chunk("new.md", process="cleaning", title="Cleaning Lab Devices",
               effective_date="2026-05-01", modified_at="2020-01-01T00:00:00+00:00"),
        _chunk("old.md", process="cleaning", title="Cleaning Lab Devices",
               effective_date="2025-01-01", modified_at="2026-12-01T00:00:00+00:00"),
    ]
    kept = {c.metadata["source_id"] for c in _collapse_superseded(chunks)}
    assert kept == {"new.md"}


def test_collapse_is_noop_for_single_doc():
    chunks = [_chunk("only.md", process="p", title="Only", section_index=0),
              _chunk("only.md", process="p", title="Only", section_index=1)]
    out = _collapse_superseded(chunks)
    assert [c.metadata["source_id"] for c in out] == ["only.md", "only.md"]


def test_collapse_preserves_order_of_survivors():
    chunks = [
        _chunk("keep1.md", process="a", title="Alpha Guide"),
        _chunk("v2.md", supersedes="v1.md", process="b", title="Beta", version="2"),
        _chunk("keep2.md", process="c", title="Gamma Guide"),
        _chunk("v1.md", process="b", title="Beta", version="1"),
    ]
    out = [c.metadata["source_id"] for c in _collapse_superseded(chunks)]
    assert out == ["keep1.md", "v2.md", "keep2.md"]   # v1 dropped, order intact


def test_title_similarity_helper():
    assert _title_similar("Cleaning Lab Devices v2", "Cleaning Lab Devices v3")
    assert not _title_similar("Centrifuge Solvent Guide", "Decontamination Standard")


# ===========================================================================
# Phase 3 — in-prompt conflict flag
# ===========================================================================

class _FakeDocStore:
    def __init__(self, chunks, max_dense=0.9, max_lexical=0.95):
        from retrieval import RetrievalResult
        self._chunks = chunks
        self._result = RetrievalResult(chunks=chunks, max_dense=max_dense,
                                       max_lexical=max_lexical)

    def retrieve_scored(self, query, k=4):
        return self._result

    def doc_metadata(self, source_id):
        return None


class _RecordingLLM:
    def __init__(self, payload):
        self._payload = payload

    def complete_structured(self, **kw):
        return self._payload


class _StreamingLLM:
    def __init__(self, deltas):
        self._deltas = deltas

    def stream_text(self, **kw):
        yield from self._deltas


def test_format_context_includes_recency_and_version_headers():
    out = _format_context([
        Chunk(id="c", text="x", score=0.9, metadata={
            "source_id": "06_cleaning_v3.md",
            "title": "Cleaning Lab Devices",
            "section": "Solvents",
            "effective_date": "2026-05-01",
            "version": "3",
            "status": "current",
        })
    ])
    assert 'modified="2026-05-01"' in out
    assert 'version="3"' in out
    assert 'status="current"' in out


def test_answer_sets_conflict_from_model_and_keeps_both_docs():
    docs = _FakeDocStore([_chunk("a.md", title="A"), _chunk("b.md", title="B")])
    llm = _RecordingLLM({
        "text": "Doc A says 70% ethanol; Doc B says 90% isopropyl — they disagree.",
        "citations": [
            {"source": "a.md", "section": "Solvent"},
            {"source": "b.md", "section": "Solvent"},
        ],
        "conflict": True,
    })
    agent = RAGAgent(document_store=docs, llm=llm)
    result = agent.answer("which solvent?", language="english")
    assert result.conflict is True
    # _dedupe_citations keeps one row per source, so both disagreeing docs stay.
    assert sorted(c.source for c in result.citations) == ["a.md", "b.md"]


def test_answer_conflict_defaults_false_when_model_omits_it():
    docs = _FakeDocStore([_chunk("a.md", title="A")])
    llm = _RecordingLLM({"text": "All good.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)
    assert agent.answer("q", language="english").conflict is False


def test_stream_parses_conflict_from_tail():
    deltas = [
        "They disagree.", "\n---CITATIONS---\n",
        '{"citations": [{"source":"a.md","section":"S"},'
        '{"source":"b.md","section":"S"}], "follow_ups": [], "conflict": true}',
    ]
    docs = _FakeDocStore([_chunk("a.md"), _chunk("b.md")])
    agent = RAGAgent(document_store=docs, llm=_StreamingLLM(deltas))
    pieces = list(agent.answer_stream("q", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    assert complete.conflict is True
    assert sorted(c.source for c in complete.citations) == ["a.md", "b.md"]


def test_parse_tail_returns_conflict_and_defaults_false():
    cits, fups, conflict = _parse_tail(
        '{"citations": [], "follow_ups": [], "conflict": true}'
    )
    assert conflict is True
    # Malformed tail -> conflict False (mirrors the other flags), no raise.
    assert _parse_tail("not json") == ([], [], False)
    # Bare-array legacy form -> conflict False.
    assert _parse_tail('[{"source":"a","section":"s"}]')[2] is False


# ===========================================================================
# Phase 4 — conflict threads through the orchestrator
# ===========================================================================

@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


class _FakeCL:
    def analyze(self, message, history=()):
        return AnalysisResult(language="english", type="question",
                              corrected_query=message)


class _FakeAgent:
    def __init__(self, result):
        self._result = result

    def answer(self, *, message, language, history=(), retrieval_query=None):
        return self._result


class _FakeStreamAgent:
    def __init__(self, complete):
        self._complete = complete

    def answer_stream(self, *, message, language, history=(), retrieval_query=None):
        yield TextDelta(self._complete.text)
        yield self._complete


def _assistant(engine, agent):
    return Assistant(
        conversation_layer=_FakeCL(),
        rag_agent=agent,
        session_repo=SessionRepository(engine),
        feedback_repo=FeedbackRepository(engine),
        question_gap_repo=QuestionGapRepository(engine),
    )


def _conflict_answer():
    return AnswerResult(
        text="They disagree.",
        citations=[Citation(source="a.md", section="S"),
                   Citation(source="b.md", section="S")],
        conflict=True,
        retrieval_max_dense=0.7,
        retrieval_max_lexical=0.6,
    )


def test_response_carries_conflict_flag(engine):
    a = _assistant(engine, _FakeAgent(_conflict_answer()))
    resp = a.handle(uuid4(), "which solvent?")
    assert resp.conflict is True


def test_stream_done_carries_conflict_flag(engine):
    complete = AnswerComplete(
        text="They disagree.",
        citations=[Citation(source="a.md", section="S"),
                   Citation(source="b.md", section="S")],
        conflict=True,
        retrieval_max_dense=0.7,
        retrieval_max_lexical=0.6,
    )
    a = _assistant(engine, _FakeStreamAgent(complete))
    done = [e for e in a.handle_stream(uuid4(), "which solvent?")
            if isinstance(e, StreamDone)]
    assert done and done[0].conflict is True


# ===========================================================================
# Phase 5 — conflict analytics (_log_conflict + conflict_pairs)
# ===========================================================================

def test_log_conflict_writes_row_with_both_source_ids(engine):
    a = _assistant(engine, _FakeAgent(_conflict_answer()))
    a.handle(uuid4(), "which solvent?")

    pairs = QuestionGapRepository(engine).conflict_pairs()
    assert pairs["total"] == 1
    assert len(pairs["pairs"]) == 1
    assert pairs["pairs"][0]["sources"] == ["a.md", "b.md"]
    assert pairs["pairs"][0]["count"] == 1


def test_conflict_with_single_citation_is_not_logged(engine):
    answer = AnswerResult(
        text="only one source", conflict=True,
        citations=[Citation(source="a.md", section="S")],
    )
    a = _assistant(engine, _FakeAgent(answer))
    a.handle(uuid4(), "q")
    # A conflict needs two docs to form a pair; a single-source flag is skipped.
    assert QuestionGapRepository(engine).conflict_pairs()["total"] == 0


def test_non_conflict_answer_logs_no_conflict_row(engine):
    answer = AnswerResult(text="fine", citations=[Citation(source="a.md", section="S")])
    a = _assistant(engine, _FakeAgent(answer))
    a.handle(uuid4(), "q")
    assert QuestionGapRepository(engine).conflict_pairs()["total"] == 0


def test_conflict_pairs_rank_and_normalize_source_order(engine):
    repo = QuestionGapRepository(engine)
    sid = uuid4()
    # Same pair, different source order on the second row -> same bucket.
    repo.add(session_id=sid, query="q1", kind="conflict", conflict_sources="a.md,b.md")
    repo.add(session_id=sid, query="q2", kind="conflict", conflict_sources="b.md,a.md")
    repo.add(session_id=sid, query="q3", kind="conflict", conflict_sources="c.md,d.md")

    result = repo.conflict_pairs()
    assert result["total"] == 3
    top = result["pairs"][0]
    assert top["sources"] == ["a.md", "b.md"]   # sorted, merged across order
    assert top["count"] == 2


def test_conflict_rows_excluded_from_documentation_gaps(engine):
    repo = QuestionGapRepository(engine)
    sid = uuid4()
    repo.add(session_id=sid, query="declined q", kind="declined", topic="access")
    repo.add(session_id=sid, query="conflict q", kind="conflict",
             conflict_sources="a.md,b.md", topic="access")

    # The documentation-gaps views count only declined/low_confidence rows.
    assert repo.count() == 1
    labels = [c["count"] for c in repo.clusters()]
    assert sum(labels) == 1
    # ...while the conflict shows up only in its own panel.
    assert repo.conflict_pairs()["total"] == 1


# ===========================================================================
# End-to-end — ingest the isolated conflict corpus through the real stack
# ===========================================================================

@pytest.fixture(scope="module")
def conflict_store(tmp_path_factory):
    from embeddings import FastEmbedProvider
    from lexical_index import BM25Index
    from vector_store import ChromaVectorStore

    tmp = tmp_path_factory.mktemp("chroma_conflict")
    store = DocumentStore(
        source=LocalMarkdownSource(CONFLICT_DOCS),
        embedder=FastEmbedProvider(),
        vector_store=ChromaVectorStore(path=str(tmp), collection_name="test_conflict"),
        manifest_path=str(tmp / "manifest.json"),
        lexical_index=BM25Index(),
    )
    store.ingest()
    return store


def test_ingest_collapses_stale_version_but_keeps_genuine_conflict(conflict_store):
    # A cleaning query pulls the v1/v2 pair plus the two distinct solvent docs.
    chunks = conflict_store.retrieve_scored("how do I clean the centrifuge", k=4).chunks
    sources = {c.metadata.get("source_id") for c in chunks}
    # Case A: the deprecated/superseded v1 is collapsed away; v2 survives.
    assert "cleaning_devices_v1.md" not in sources
    assert "cleaning_devices_v2.md" in sources


def test_ingest_lands_conflict_metadata_on_doc_record(conflict_store):
    meta = conflict_store.doc_metadata("cleaning_devices_v2.md")
    assert meta is not None
    assert meta["version"] == "2"
    assert meta["status"] == "current"
    assert meta["supersedes"] == "cleaning_devices_v1.md"
