"""
Unit tests for the RAG agent — context formatting and the answer() seam.

No Groq, no Chroma: a fake DocumentStore feeds canned chunks and a fake
LLMClient records what the agent sends and returns a canned payload. These
lock in the prompt-assembly contract (language, context, history) and the
parsing of the structured reply into AnswerResult.
"""

from __future__ import annotations

from typing import Any, Sequence

from agent import AnswerResult, RAGAgent, Turn, _format_context
from vector_store import Chunk


def _chunk(text: str, *, source_id: str | None = None, section: str | None = None) -> Chunk:
    meta: dict[str, Any] = {}
    if source_id is not None:
        meta["source_id"] = source_id
    if section is not None:
        meta["section"] = section
    return Chunk(id="c", text=text, metadata=meta, score=0.9)


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

def test_format_context_empty_returns_placeholder():
    assert _format_context([]) == "(no relevant documentation found)"


def test_format_context_includes_source_and_section_header():
    out = _format_context([_chunk("wipe it down", source_id="06_cleaning.md", section="Centrifuges")])
    assert "[source: 06_cleaning.md §Centrifuges]" in out
    assert "wipe it down" in out


def test_format_context_omits_section_when_absent():
    out = _format_context([_chunk("body text", source_id="06_cleaning.md")])
    assert "[source: 06_cleaning.md]" in out
    assert "§" not in out


def test_format_context_falls_back_to_unknown_source():
    out = _format_context([_chunk("orphan text")])
    assert "[source: unknown]" in out


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
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls: list[tuple] = []

    def retrieve(self, query, k=4):
        self.calls.append((query, k))
        return self._chunks


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


def test_answer_builds_prompt_with_language_and_context():
    agent, _, llm = _make_agent(
        [_chunk("use isopropyl", source_id="06_cleaning.md", section="Centrifuges")],
        {"text": "ok", "citations": []},
    )
    agent.answer("question", language="german")
    assert "german" in llm.last["system"]
    assert "[source: 06_cleaning.md §Centrifuges]" in llm.last["system"]
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
