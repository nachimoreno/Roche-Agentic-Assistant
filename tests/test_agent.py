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
    RetrievalInfo,
    TextDelta,
    Turn,
    _clean_citation_value,
    _clean_follow_ups,
    _decline_message,
    _is_capability_question,
    _format_context,
    _overlap,
    _parse_citations,
    _parse_tail,
    _retrieval_query,
)
from capabilities import CAPABILITIES
from retrieval import RetrievalResult
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
    def __init__(self, chunks, meta=None, max_dense=0.9, max_lexical=0.95):
        self._chunks = chunks
        # source_id -> {"title", "process", "department"}, mirroring the real
        # DocumentStore.doc_metadata used for citation title enrichment.
        self._meta = meta or {}
        # Top retriever scores returned by retrieve_scored, both on a [0, 1]
        # scale (dense cosine, normalised BM25). Default high so the off-domain
        # guardrail does not trip — tests that exercise the guardrail pass low
        # values explicitly.
        self._max_dense = max_dense
        self._max_lexical = max_lexical
        self.calls: list[tuple] = []

    def retrieve(self, query, k=4):
        self.calls.append((query, k))
        return self._chunks

    def retrieve_scored(self, query, k=4):
        self.calls.append((query, k))
        return RetrievalResult(
            chunks=self._chunks,
            max_dense=self._max_dense,
            max_lexical=self._max_lexical,
        )

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
    block = CAPABILITIES.as_prompt_block()
    assert block in system                                 # but self-knowledge present
    # The ServiceNow incident skill is wired, so within the capability block it
    # reads as a current capability. Partition the block itself, not the whole
    # system prompt — the operational guidance below also names ServiceNow (as a
    # place to point scientists), which is unrelated to the can/cannot split.
    can_part, _, cannot_part = block.partition("What you cannot do yet:")
    assert "ServiceNow" in can_part
    assert "ServiceNow" not in cannot_part


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
    # Retrieval transparency: a single RetrievalInfo leads, then prose deltas,
    # then the terminal AnswerComplete.
    assert isinstance(pieces[0], RetrievalInfo)
    assert all(isinstance(p, TextDelta) for p in pieces[1:-1])


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
            "url": "https://drive.google.com/file/d/clean-001/view",
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
    assert c.url == "https://drive.google.com/file/d/clean-001/view"  # click-through link


def test_answer_dedupes_citations_to_one_row_per_document():
    docs = FakeDocumentStore(
        [_chunk("ctx", source_id="book-001")],
        meta={"book-001": {"title": "Booking Laboratory Instruments"}},
    )
    # The model cites several chunks of the same document — different sections,
    # but all resolving to the same title and click-through URL. They collapse
    # to one row; a genuinely different document stays.
    llm = RecordingLLM({
        "text": "Use the booking portal.",
        "citations": [
            {"source": "book-001", "section": "Finding an Instrument"},
            {"source": "book-001", "section": "Making a Reservation"},
            {"source": "incident-001", "section": "How to report"},
        ],
    })
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("how do I book?", language="english")
    sources = [c.source for c in result.citations]
    assert sources == ["book-001", "incident-001"]      # one row per document
    # First occurrence wins, so the representative section is kept.
    assert result.citations[0].section == "Finding an Instrument"


# ---------------------------------------------------------------------------
# Off-domain guardrail — decline deterministically when retrieval is too weak
# ---------------------------------------------------------------------------

def test_off_domain_query_declines_without_calling_llm():
    # Both retrievers weak → declined before the LLM is ever invoked.
    docs = FakeDocumentStore([_chunk("irrelevant")], max_dense=0.20, max_lexical=0.30)
    llm = RecordingLLM({"text": "should not be used", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("how do I bake a cake?", language="english")
    assert result.text == _decline_message("english")
    assert result.citations == []
    assert llm.last == {}                                # LLM was never called


def test_strong_dense_prevents_decline_even_with_weak_lexical():
    # A real question the embedder matches (dense above threshold) is answered
    # even if BM25 is weak — both signals must be low to decline.
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             max_dense=0.50, max_lexical=0.30)
    llm = RecordingLLM({"text": "Here you go.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("a real question", language="english")
    assert result.text == "Here you go."
    assert llm.last != {}                                # LLM was called


def test_strong_lexical_prevents_decline_even_with_weak_dense():
    # An exact-keyword hit (BM25 above threshold) is answered even if the
    # embedder scores it low — protects part numbers / SOP codes.
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             max_dense=0.20, max_lexical=0.90)
    llm = RecordingLLM({"text": "Here you go.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("SOP-12345", language="english")
    assert result.text == "Here you go."
    assert llm.last != {}                                # LLM was called


def test_decline_message_is_localized_with_english_fallback():
    assert "Labordokumentation" in _decline_message("german")
    assert "laboratoire" in _decline_message("french")
    assert "laboratorio" in _decline_message("italian")
    # Unsupported/unknown languages fall back to English.
    assert _decline_message("spanish") == _decline_message("english")


def test_stream_off_domain_declines_without_calling_llm():
    docs = FakeDocumentStore([_chunk("irrelevant")], max_dense=0.20, max_lexical=0.30)
    llm = StreamingLLM(["should ", "not ", "be ", "used"])
    agent = RAGAgent(document_store=docs, llm=llm)

    pieces = list(agent.answer_stream("what's the weather?", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    assert complete.text == _decline_message("english")
    assert complete.citations == []
    assert complete.follow_ups == []
    # The decline prose was emitted as a delta; the canned LLM tokens were not.
    text = "".join(p.text for p in pieces if isinstance(p, TextDelta))
    assert text == _decline_message("english")


# ---------------------------------------------------------------------------
# Capability questions bypass the off-domain guardrail
#
# A self-question ("what can you do?") has no match in the lab corpus, so both
# retrievers score low and the guardrail would wrongly decline it. The agent
# must instead reach the LLM, which answers from the injected capabilities
# block. Without this exemption the self-knowledge feature is silently broken.
# ---------------------------------------------------------------------------

def test_capability_question_bypasses_off_domain_decline():
    # Weak retrieval (would normally decline) BUT it's a capability question →
    # the LLM is still called and the capabilities block is injected.
    docs = FakeDocumentStore([_chunk("irrelevant")], max_dense=0.10, max_lexical=0.20)
    llm = RecordingLLM({"text": "I can help with lab operations.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("what can you do?", language="english")
    assert result.text == "I can help with lab operations."   # not the decline
    assert result.text != _decline_message("english")
    assert llm.last != {}                                      # LLM WAS called
    assert CAPABILITIES.as_prompt_block() in llm.last["system"]


def test_stream_capability_question_bypasses_off_domain_decline():
    docs = FakeDocumentStore([_chunk("irrelevant")], max_dense=0.10, max_lexical=0.20)
    llm = StreamingLLM(["I help ", "with lab ops.", "\n---CITATIONS---\n", "[]"])
    agent = RAGAgent(document_store=docs, llm=llm)

    pieces = list(agent.answer_stream("who are you?", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    assert complete.text == "I help with lab ops."             # not the decline
    assert complete.text != _decline_message("english")
    assert CAPABILITIES.as_prompt_block() in llm.last["system"]


def test_is_capability_question_matches_supported_languages():
    assert _is_capability_question("What can you do?")
    assert _is_capability_question("who are you")
    assert _is_capability_question("Tell me about your capabilities.")
    assert _is_capability_question("Was kannst du?")           # German
    assert _is_capability_question("Que peux-tu faire ?")      # French
    assert _is_capability_question("Cosa puoi fare?")          # Italian
    # Operational questions are NOT treated as capability questions.
    assert not _is_capability_question("How do I clean the centrifuge?")
    assert not _is_capability_question("How do I book an instrument?")
    assert not _is_capability_question("")


# ---------------------------------------------------------------------------
# Context-aware follow-up suggestions
# ---------------------------------------------------------------------------

def test_answer_returns_cleaned_follow_ups():
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")])
    llm = RecordingLLM({
        "text": "Here you go.",
        "citations": [],
        "follow_ups": [
            "How do I cancel a booking?",
            "  How do I cancel a booking?  ",     # duplicate after strip
            "What if it's already reserved?",
            "How far ahead can I book?",
            "A fourth one beyond the cap",         # dropped — capped at 3
        ],
    })
    agent = RAGAgent(document_store=docs, llm=llm)
    result = agent.answer("how do I book?", language="english")
    assert result.follow_ups == [
        "How do I cancel a booking?",
        "What if it's already reserved?",
        "How far ahead can I book?",
    ]


def test_answer_stream_parses_follow_ups_from_object_tail():
    deltas = [
        "Use the portal.", "\n---CITATIONS---\n",
        '{"citations": [{"source": "a.md", "section": "Booking"}], '
        '"follow_ups": ["How do I cancel?", "Can I book two at once?"]}',
    ]
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             meta={"a.md": {"title": "Booking"}})
    llm = StreamingLLM(deltas)
    agent = RAGAgent(document_store=docs, llm=llm)

    pieces = list(agent.answer_stream("how do I book?", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    assert [c.source for c in complete.citations] == ["a.md"]
    assert complete.follow_ups == ["How do I cancel?", "Can I book two at once?"]


def test_off_domain_answer_has_no_follow_ups():
    docs = FakeDocumentStore([_chunk("x")], max_dense=0.2, max_lexical=0.30)
    llm = RecordingLLM({"text": "unused", "citations": [], "follow_ups": ["x"]})
    agent = RAGAgent(document_store=docs, llm=llm)
    result = agent.answer("bake a cake?", language="english")
    assert result.follow_ups == []


def test_clean_follow_ups_dedupes_strips_and_caps():
    assert _clean_follow_ups(["a", " a ", "", "b", "c", "d"]) == ["a", "b", "c"]
    assert _clean_follow_ups(None) == []


def test_parse_tail_accepts_object_array_and_malformed():
    cits, fups, contra = _parse_tail(
        '{"citations": [{"source":"a","section":"s"}], "follow_ups": ["q1"],'
        ' "contradiction": true}'
    )
    assert [c.source for c in cits] == ["a"] and fups == ["q1"] and contra is True
    # Bare array (legacy form / model ignored the object instruction).
    cits, fups, contra = _parse_tail('[{"source":"a","section":"s"}]')
    assert [c.source for c in cits] == ["a"] and fups == [] and contra is False
    # Garbage degrades to empty, no raise.
    assert _parse_tail("not json at all") == ([], [], False)


# ---------------------------------------------------------------------------
# retrieval_query override (spell-corrected query drives retrieval)
# ---------------------------------------------------------------------------

def test_answer_retrieves_on_override_query_but_answers_original():
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")])
    llm = RecordingLLM({"text": "ok", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    agent.answer(
        "how do I bok an instrumnet?",
        language="english",
        retrieval_query="how do I book an instrument?",
    )
    # Retrieval ran on the corrected query...
    assert docs.calls[-1][0] == "how do I book an instrument?"
    # ...but the LLM still got the original (typo'd) message.
    assert llm.last["user"] == "how do I bok an instrumnet?"


def test_answer_retrieves_on_message_when_no_override():
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")])
    llm = RecordingLLM({"text": "ok", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    agent.answer("clean the centrifuge", language="english")
    assert docs.calls[-1][0] == "clean the centrifuge"


# ---------------------------------------------------------------------------
# Low-confidence ("verify this") flag
# ---------------------------------------------------------------------------

def test_answer_flags_low_confidence_when_grounding_is_weak():
    # Dense above the decline floor (0.40) but below the warn threshold (0.45):
    # answered, but flagged for the user to verify.
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             max_dense=0.42, max_lexical=0.90)
    llm = RecordingLLM({"text": "Here you go.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("q", language="english")
    assert result.low_confidence is True


def test_answer_confident_when_match_is_strong():
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             max_dense=0.70, max_lexical=0.90)
    llm = RecordingLLM({"text": "Here you go.", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("q", language="english")
    assert result.low_confidence is False


def test_declined_answer_is_not_flagged_low_confidence():
    # Off-domain queries are declined outright, not flagged as a weak answer.
    # Both signals weak on the normalized [0, 1] scale (dense < 0.40 and
    # lexical < 0.85) so the guardrail declines before low-confidence applies.
    docs = FakeDocumentStore([_chunk("x")], max_dense=0.20, max_lexical=0.30)
    llm = RecordingLLM({"text": "unused", "citations": []})
    agent = RAGAgent(document_store=docs, llm=llm)

    result = agent.answer("bake a cake?", language="english")
    assert result.low_confidence is False


def test_answer_stream_flags_low_confidence():
    deltas = ["Here you go.", "\n---CITATIONS---\n", '{"citations": [], "follow_ups": []}']
    docs = FakeDocumentStore([_chunk("ctx", source_id="a.md")],
                             max_dense=0.42, max_lexical=0.90)
    llm = StreamingLLM(deltas)
    agent = RAGAgent(document_store=docs, llm=llm)

    pieces = list(agent.answer_stream("q", language="english"))
    complete = next(p for p in pieces if isinstance(p, AnswerComplete))
    assert complete.low_confidence is True


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
    assert result.citations[0].url is None
