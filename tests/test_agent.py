"""
Unit tests for the RAG agent — context formatting and the answer() seam.

No Groq, no Chroma: a fake DocumentStore feeds canned chunks and a fake
LLMClient records what the agent sends and returns a canned payload. These
lock in the prompt-assembly contract (language, context, history) and the
parsing of the structured reply into AnswerResult.
"""

from __future__ import annotations

from typing import Any, Sequence

from agent import (
    AnswerComplete,
    AnswerResult,
    Citation,
    RAGAgent,
    TextDelta,
    Turn,
    _clean_citation_value,
    _format_context,
    _overlap,
    _parse_citations,
    _retrieval_query,
)
from capabilities import CAPABILITIES
from vector_store import Chunk


def _chunk(
    text: str,
    *,
    source_id: str | None = None,
    section: str | None = None,
    title: str | None = None,
) -> Chunk:
    meta: dict[str, Any] = {}
    if source_id is not None:
        meta["source_id"] = source_id
    if section is not None:
        meta["section"] = section
    if title is not None:
        meta["title"] = title
    return Chunk(id="c", text=text, metadata=meta, score=0.9)


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

def test_format_context_empty_returns_placeholder():
    assert _format_context([]) == "(no relevant documentation found)"


def test_format_context_includes_source_and_section_header():
    out = _format_context([_chunk("wipe it down", source_id="06_cleaning.md", section="Centrifuges")])
    assert '[source="06_cleaning.md" section="Centrifuges"]' in out
    assert "wipe it down" in out


def test_format_context_shows_human_title_for_readability():
    # The title is shown as document="..." for the model's readability, but the
    # cited key (source="...") stays the stable id so attribution is unaffected.
    out = _format_context([
        _chunk("wipe it down", source_id="06_cleaning.md",
               section="Centrifuges", title="Cleaning Laboratory Devices")
    ])
    assert '[source="06_cleaning.md" document="Cleaning Laboratory Devices" section="Centrifuges"]' in out


def test_format_context_omits_section_when_absent():
    out = _format_context([_chunk("body text", source_id="06_cleaning.md")])
    assert '[source="06_cleaning.md"]' in out
    assert "section=" not in out


def test_format_context_falls_back_to_unknown_source():
    out = _format_context([_chunk("orphan text")])
    assert '[source="unknown"]' in out


def test_format_context_joins_multiple_chunks_with_separator():
    out = _format_context([
        _chunk("first", source_id="a.md"),
        _chunk("second", source_id="b.md"),
    ])
    assert "\n\n---\n\n" in out
    assert out.index("first") < out.index("second")


# ---------------------------------------------------------------------------
# Fakes for RAGAgent.answer
# ---------------------------------------------------------------------------

class FakeDocumentStore:
    def __init__(self, chunks, meta=None):
        self._chunks = chunks
        # source_id -> {"title", "process", "department"}, mirroring the real
        # DocumentStore.doc_metadata used for citation title enrichment.
        self._meta = meta or {}
        self.calls: list[tuple] = []

    def retrieve(self, query, k=4):
        self.calls.append((query, k))
        return self._chunks

    def doc_metadata(self, source_id):
        return self._meta.get(source_id)


class RecordingLLM:
    def __init__(self, payload):
        self._payload = payload
        self.last: dict[str, Any] = {}

    def complete_structured(self, *, system, user, schema, history=(),
                            temperature=0.0, max_tokens=1024) -> dict[str, Any]:
        self.last = {
            "system": system, "user": user, "schema": schema,
            "history": list(history), "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._payload


class StreamingLLM:
    """Yields the given deltas from stream_text; complete_structured unused."""

    def __init__(self, deltas):
        self._deltas = deltas
        self.last: dict[str, Any] = {}

    def stream_text(self, *, system, user, history=(),
                    temperature=0.0, max_tokens=1024):
        self.last = {"system": system, "user": user, "history": list(history)}
        for d in self._deltas:
            yield d

    def complete_structured(self, **kw):  # pragma: no cover - guard
        raise AssertionError("answer_stream must not call complete_structured")


# ---------------------------------------------------------------------------
# RAGAgent.answer
# ---------------------------------------------------------------------------

def _make_agent(chunks, payload, **kw):
    docs = FakeDocumentStore(chunks)
    llm = RecordingLLM(payload)
    agent = RAGAgent(document_store=docs, llm=llm, **kw)
    return agent, docs, llm


def test_answer_retrieves_with_configured_top_k():
    agent, docs, _ = _make_agent(
        [_chunk("x", source_id="a.md")],
        {"text": "ok", "citations": []},
        top_k=7,
    )
    agent.answer("how do I clean it?", language="english")
    assert docs.calls == [("how do I clean it?", 7)]


# ---------------------------------------------------------------------------
# _retrieval_query — context-aware search for follow-ups
# ---------------------------------------------------------------------------

def test_retrieval_query_is_bare_message_without_history():
    assert _retrieval_query("how do I clean the centrifuge?", []) == (
        "how do I clean the centrifuge?"
    )


def test_retrieval_query_anchors_followup_on_prior_user_turn():
    history = [
        Turn(role="user", content="how do I clean the centrifuge?"),
        Turn(role="assistant", content="Use a 70% isopropyl wipe."),
    ]
    q = _retrieval_query("what about the rotor?", history)
    assert "how do I clean the centrifuge?" in q
    assert q.endswith("what about the rotor?")


def test_retrieval_query_uses_most_recent_user_turn():
    history = [
        Turn(role="user", content="first topic"),
        Turn(role="assistant", content="..."),
        Turn(role="user", content="second topic"),
        Turn(role="assistant", content="..."),
    ]
    q = _retrieval_query("yes", history)
    assert "second topic" in q
    assert "first topic" not in q


def test_answer_uses_history_anchored_query_for_retrieval():
    agent, docs, _ = _make_agent(
        [_chunk("x", source_id="a.md")],
        {"text": "ok", "citations": []},
    )
    history = [Turn(role="user", content="how do I clean the centrifuge?")]
    agent.answer("yes, that one", language="english", history=history)
    query, _k = docs.calls[0]
    assert "how do I clean the centrifuge?" in query
    assert "yes, that one" in query


def test_answer_builds_prompt_with_language_and_context():
    agent, _, llm = _make_agent(
        [_chunk("use isopropyl", source_id="06_cleaning.md", section="Centrifuges")],
        {"text": "ok", "citations": []},
    )
    agent.answer("question", language="german")
    assert "german" in llm.last["system"]
    assert '[source="06_cleaning.md" section="Centrifuges"]' in llm.last["system"]
    assert "use isopropyl" in llm.last["system"]
    assert llm.last["user"] == "question"


def test_answer_passes_history_as_role_content_dicts():
    agent, _, llm = _make_agent(
        [_chunk("x", source_id="a.md")],
        {"text": "ok", "citations": []},
    )
    history = [Turn(role="user", content="earlier q"), Turn(role="assistant", content="earlier a")]
    agent.answer("follow up", language="english", history=history)
    assert llm.last["history"] == [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]


def test_answer_parses_payload_into_answer_result_with_citations():
    agent, _, _ = _make_agent(
        [_chunk("use isopropyl", source_id="06_cleaning.md", section="Centrifuges")],
        {
            "text": "Use a 70% isopropyl wipe.",
            "citations": [{"source": "06_cleaning.md", "section": "Centrifuges"}],
        },
    )
    result = agent.answer("how do I clean it?", language="english")
    assert isinstance(result, AnswerResult)
    assert result.text == "Use a 70% isopropyl wipe."
    assert len(result.citations) == 1
    assert result.citations[0].source == "06_cleaning.md"
    assert result.citations[0].section == "Centrifuges"


def test_answer_uses_placeholder_context_when_no_chunks():
    agent, _, llm = _make_agent([], {"text": "I don't have that.", "citations": []})
    agent.answer("obscure question", language="english")
    assert "(no relevant documentation found)" in llm.last["system"]


# ---------------------------------------------------------------------------
# Self-knowledge — CAPABILITIES is always injected, even with empty context
# ---------------------------------------------------------------------------

def test_answer_injects_capabilities_even_with_empty_context():
    # A capability question with no retrieved chunks: the agent must still hand
    # the model its self-knowledge, so the answer never depends on retrieval.
    agent, _, llm = _make_agent([], {"text": "...", "citations": []})
    agent.answer("what can you do?", language="english")
    system = llm.last["system"]
    assert "(no relevant documentation found)" in system   # context truly empty
    assert CAPABILITIES.as_prompt_block() in system        # but self-knowledge present
    # ServiceNow is described as not-yet, never advertised as a capability.
    can_part, _, cannot_part = system.partition("What you cannot do yet:")
    assert "ServiceNow" in cannot_part
    assert "ServiceNow" not in can_part


def test_answer_stream_injects_capabilities_even_with_empty_context():
    agent, _, llm = _stream_agent(["ok.", "\n---CITATIONS---\n", "[]"], chunks=[])
    list(agent.answer_stream("what can you do?", language="english"))
    system = llm.last["system"]
    assert "(no relevant documentation found)" in system
    assert CAPABILITIES.as_prompt_block() in system


# ---------------------------------------------------------------------------
# Streaming helpers — _overlap and _parse_citations
# ---------------------------------------------------------------------------

def test_overlap_detects_partial_sentinel_suffix():
    sentinel = "\n---CITATIONS---"
    # A buffer ending in the start of the sentinel must be held back.
    assert _overlap("answer\n---CIT", sentinel) == len("\n---CIT")
    # No overlap when the suffix isn't a sentinel prefix.
    assert _overlap("just words", sentinel) == 0


def test_parse_citations_valid_and_garbage():
    good = _parse_citations('[{"source": "a.md", "section": "S"}]')
    assert len(good) == 1 and good[0].source == "a.md"
    assert _parse_citations("") == []
    assert _parse_citations("not json at all") == []


# ---------------------------------------------------------------------------
# RAGAgent.answer_stream
# ---------------------------------------------------------------------------

def _stream_agent(deltas, chunks=None):
    docs = FakeDocumentStore(chunks if chunks is not None else [_chunk("ctx", source_id="a.md")])
    llm = StreamingLLM(deltas)
    return RAGAgent(document_store=docs, llm=llm), docs, llm


def test_answer_stream_splits_prose_from_citations_across_chunks():
    # Sentinel deliberately split across deltas to exercise the hold-back.
    deltas = [
        "Use a 70% ", "isopropyl wipe.", "\n---CIT", "ATIONS---\n",
        '[{"source": "06_cleaning.md", ', '"section": "Centrifuges"}]',
    ]
    agent, _, _ = _stream_agent(deltas)
    pieces = list(agent.answer_stream("how do I clean it?", language="english"))

    text = "".join(p.text for p in pieces if isinstance(p, TextDelta))
    completes = [p for p in pieces if isinstance(p, AnswerComplete)]

    assert text == "Use a 70% isopropyl wipe."
    assert "---CITATIONS---" not in text          # sentinel never leaks
    assert "source" not in text                   # JSON tail never leaks
    assert len(completes) == 1
    assert completes[0].text == "Use a 70% isopropyl wipe."
    assert completes[0].citations[0].source == "06_cleaning.md"
    assert completes[0].citations[0].section == "Centrifuges"


def test_answer_stream_terminal_event_is_last():
    deltas = ["hello.", "\n---CITATIONS---\n", "[]"]
    agent, _, _ = _stream_agent(deltas)
    pieces = list(agent.answer_stream("q", language="english"))
    assert isinstance(pieces[-1], AnswerComplete)
    assert all(isinstance(p, TextDelta) for p in pieces[:-1])


def test_answer_stream_without_sentinel_yields_all_prose_no_citations():
    deltas = ["The model ", "forgot the delimiter."]
    agent, _, _ = _stream_agent(deltas)
    pieces = list(agent.answer_stream("q", language="english"))
    text = "".join(p.text for p in pieces if isinstance(p, TextDelta))
    complete = pieces[-1]
    assert text == "The model forgot the delimiter."
    assert isinstance(complete, AnswerComplete)
    assert complete.text == "The model forgot the delimiter."
    assert complete.citations == []


def test_answer_stream_builds_streaming_prompt_with_language_and_context():
    agent, _, llm = _stream_agent(
        ["ok.", "\n---CITATIONS---\n", "[]"],
        chunks=[_chunk("use isopropyl", source_id="06_cleaning.md", section="Centrifuges")],
    )
    list(agent.answer_stream("question", language="french"))
    assert "french" in llm.last["system"]
    assert "---CITATIONS---" in llm.last["system"]      # streaming output contract
    assert '[source="06_cleaning.md" section="Centrifuges"]' in llm.last["system"]


# ---------------------------------------------------------------------------
# Citation sanitization — _clean_citation_value strips leaked header syntax
# ---------------------------------------------------------------------------

def test_clean_citation_value_strips_leading_document_key():
    assert _clean_citation_value('document="Cleaning Guide"') == "Cleaning Guide"


def test_clean_citation_value_strips_leading_source_and_section_keys():
    assert _clean_citation_value("source=06_cleaning.md") == "06_cleaning.md"
    assert _clean_citation_value("section=Centrifuges") == "Centrifuges"


def test_clean_citation_value_strips_section_heading_merged_on_title():
    assert (
        _clean_citation_value("Cleaning Laboratory Devices §Centrifuges")
        == "Cleaning Laboratory Devices"
    )


def test_clean_citation_value_strips_trailing_merged_key():
    assert (
        _clean_citation_value('Cleaning Guide section="Centrifuges"')
        == "Cleaning Guide"
    )


def test_clean_citation_value_strips_surrounding_quotes():
    assert _clean_citation_value('"Just A Title"') == "Just A Title"
    assert _clean_citation_value("'Just A Title'") == "Just A Title"


def test_clean_citation_value_leaves_clean_value_untouched():
    assert _clean_citation_value("06_cleaning.md") == "06_cleaning.md"


def test_citation_validator_normalises_leaked_header_syntax():
    # Whatever shape the model emits, the cited key and section come out clean.
    c = Citation(source='document="06_cleaning.md"', section='Centrifuges section="x"')
    assert c.source == "06_cleaning.md"
    assert c.section == "Centrifuges"


# ---------------------------------------------------------------------------
# Title enrichment — citations carry the human title while source stays the id
# ---------------------------------------------------------------------------

def test_answer_enriches_citation_title_from_doc_metadata():
    docs = FakeDocumentStore(
        [_chunk("use isopropyl", source_id="clean-001", section="Centrifuges",
                title="Cleaning Laboratory Devices")],
        meta={"clean-001": {
            "title": "Cleaning Laboratory Devices",
            "process": "equipment-cleaning",
            "department": "lab-operations",
        }},
    )
    llm = RecordingLLM({
        "text": "Wipe it down.",
        "citations": [{"source": "clean-001", "section": "Centrifuges"}],
    })
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("how do I clean it?", language="english")
    c = result.citations[0]
    assert c.source == "clean-001"                       # id kept for attribution
    assert c.title == "Cleaning Laboratory Devices"      # human title for display


def test_answer_stream_enriches_citation_title_from_doc_metadata():
    deltas = [
        "Wipe it down.", "\n---CITATIONS---\n",
        '[{"source": "clean-001", "section": "Centrifuges"}]',
    ]
    docs = FakeDocumentStore(
        [_chunk("ctx", source_id="clean-001")],
        meta={"clean-001": {"title": "Cleaning Laboratory Devices"}},
    )
    llm = StreamingLLM(deltas)
    agent = RAGAgent(document_store=docs, llm=llm)

    pieces = list(agent.answer_stream("how do I clean it?", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    c = complete.citations[0]
    assert c.source == "clean-001"
    assert c.title == "Cleaning Laboratory Devices"


def test_enrich_titles_leaves_title_none_for_unknown_source():
    # Missing-title fallback: an id the store doesn't know stays title=None so
    # the UI can fall back to `source`.
    docs = FakeDocumentStore(
        [_chunk("ctx", source_id="unknown-001")],
        meta={},
    )
    llm = RecordingLLM({
        "text": "ok",
        "citations": [{"source": "unknown-001", "section": "S"}],
    })
    agent = RAGAgent(document_store=docs, llm=llm)
    result = agent.answer("q", language="english")
    assert result.citations[0].source == "unknown-001"
    assert result.citations[0].title is None
