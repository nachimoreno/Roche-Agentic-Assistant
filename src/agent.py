"""
agent.py
--------
RAG agent for the Roche Scientist Assistant.

Composes the LLM provider seam with the DocumentStore. Knows nothing
about Groq, Anthropic, ChromaDB, or sentence-transformers directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator, Sequence, Union

from pydantic import BaseModel, Field

from llm import LLMClient
from retrieval import DocumentStore


logger = logging.getLogger(__name__)

# Sentinel the streaming model emits between its prose answer and the JSON
# citations block. Kept on its own line so it is easy to split on.
_CITATION_SENTINEL = "\n---CITATIONS---"


# ---------------------------------------------------------------------------
# Result schema — the LLM is asked to return this exact shape.
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    source: str = Field(description="Source document identifier (filename or id).")
    section: str = Field(description="Section heading within the source.")


class AnswerResult(BaseModel):
    text: str = Field(description="The answer to show the scientist.")
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Turn — the minimal shape the agent needs from chat history.
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str        # "user" | "assistant"
    content: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_HEAD = """\
You are the Roche Scientist Assistant. You help scientists find accurate
operational information from internal documentation.

## About yourself

If the scientist asks what you can do, how you can help, or anything
about your own capabilities, describe yourself using the following
summary (and the capabilities document in the context if present):

- You answer operational questions grounded in internal lab documentation
  sourced from the team's Google Drive: onboarding and access, navigating
  internal applications, incident reporting, instrument booking and
  calibration, sample stock, ordering chemicals and consumables, cleaning,
  decontamination and disinfection, waste management, lab sharing, campus
  and facilities, and virtual session troubleshooting.
- You read your documentation live from Google Drive, so answers reflect
  the latest versions uploaded there.
- You point scientists to the right internal application when the action
  lives there. You do not perform actions in other applications.
- You record feedback for IT, detect sentiment, and support English,
  German, French, and Italian.
- Your chat history is persisted, so the conversation continues across
  devices.
- You cannot yet create ServiceNow incidents or act inside other internal
  applications — those are planned.

For capability questions, you may answer from the summary above even
when context is empty. For every other type of question, follow the
rules below strictly.

## Rules

1. For operational questions, answer ONLY from the provided context
   chunks. If the answer is not in the context, say you do not have that
   information and suggest where the scientist could look (for example,
   the Application Catalog or ServiceNow).
2. Respond in {language}. If the context is in English, translate your
   answer into {language} naturally.
3. Be concise and direct. Scientists are often standing in a lab — give
   them the answer first, then the explanation if needed.
4. Cite the sources you used. Every claim that comes from a context chunk
   must be backed by a citation. For capability questions answered from
   the summary above, citations may be empty.
5. Do not invent procedures, product names, or values that are not in the
   context.

## Context

{context}
"""


_JSON_OUTPUT = """
## Output

Return ONLY a JSON object with two fields:
- "text": your answer in {language}
- "citations": a list of objects with "source" (the source_id) and
  "section" (the section heading), one for each context chunk you relied on.
"""


_STREAM_OUTPUT = """
## Output

Write your complete answer in {language} as plain prose (no JSON, no
markdown code fences). When the answer is finished, output a line containing
exactly:
---CITATIONS---
and then a JSON array of objects with "source" (the source_id) and "section"
(the section heading), one per context chunk you relied on (an empty array []
if you used none). Output nothing after the JSON array.
"""


# Non-streaming path (JSON mode) and streaming path (prose + delimiter) share
# the same head; only the output contract differs.
_SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_HEAD + _JSON_OUTPUT
_STREAM_SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_HEAD + _STREAM_OUTPUT


# ---------------------------------------------------------------------------
# Streaming events — what `answer_stream` yields.
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    """A piece of prose answer text, safe to forward to the client."""

    text: str


@dataclass
class AnswerComplete:
    """Terminal event: the full prose answer plus parsed citations."""

    text: str
    citations: list[Citation]


StreamPiece = Union[TextDelta, AnswerComplete]


class RAGAgent:
    def __init__(
        self,
        document_store: DocumentStore,
        llm: LLMClient,
        *,
        top_k: int = 4,
        max_tokens: int = 1024,
    ) -> None:
        self._docs = document_store
        self._llm = llm
        self._top_k = top_k
        self._max_tokens = max_tokens

    def answer(
        self,
        message: str,
        language: str,
        history: Sequence[Turn] = (),
    ) -> AnswerResult:
        chunks = self._docs.retrieve(message, k=self._top_k)
        context = _format_context(chunks)
        system = _SYSTEM_PROMPT_TEMPLATE.format(language=language, context=context)

        payload = self._llm.complete_structured(
            system=system,
            user=message,
            schema=AnswerResult.model_json_schema(),
            history=[{"role": t.role, "content": t.content} for t in history],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        result = AnswerResult.model_validate(payload)
        logger.info(
            "agent.answered",
            extra={
                "language": language,
                "chunks_used": len(chunks),
                "citations": len(result.citations),
            },
        )
        return result

    def answer_stream(
        self,
        message: str,
        language: str,
        history: Sequence[Turn] = (),
    ) -> Iterator[StreamPiece]:
        """Stream the answer.

        Yields `TextDelta` events as prose arrives, then a single terminal
        `AnswerComplete` carrying the full text and the citations parsed from
        the post-delimiter JSON tail. The `---CITATIONS---` sentinel and the
        JSON after it are never leaked to the caller as prose.
        """
        chunks = self._docs.retrieve(message, k=self._top_k)
        context = _format_context(chunks)
        system = _STREAM_SYSTEM_PROMPT_TEMPLATE.format(language=language, context=context)

        deltas = self._llm.stream_text(
            system=system,
            user=message,
            history=[{"role": t.role, "content": t.content} for t in history],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )

        splitter = _ProseSplitter()
        prose: list[str] = []
        for delta in deltas:
            emit = splitter.feed(delta)
            if emit:
                prose.append(emit)
                yield TextDelta(emit)
        tail = splitter.finish()
        if tail:
            prose.append(tail)
            yield TextDelta(tail)

        citations = _parse_citations(splitter.citation_tail)
        logger.info(
            "agent.streamed",
            extra={
                "language": language,
                "chunks_used": len(chunks),
                "citations": len(citations),
            },
        )
        yield AnswerComplete(text="".join(prose), citations=citations)


def _overlap(text: str, sentinel: str) -> int:
    """Length of the longest suffix of `text` that is a prefix of `sentinel`.

    Used to hold back the tail of a buffer that might be the start of the
    sentinel split across stream chunks, so a partial sentinel is never
    emitted as prose.
    """
    for k in range(min(len(text), len(sentinel)), 0, -1):
        if text[-k:] == sentinel[:k]:
            return k
    return 0


class _ProseSplitter:
    """Splits a token stream into prose (before sentinel) and a JSON tail."""

    def __init__(self, sentinel: str = _CITATION_SENTINEL) -> None:
        self._sentinel = sentinel
        self._pending = ""        # un-emitted prose buffer
        self._tail = ""           # everything after the sentinel
        self._seen = False

    def feed(self, delta: str) -> str:
        """Accept a raw delta; return prose text safe to emit now (maybe "")."""
        if self._seen:
            self._tail += delta
            return ""

        self._pending += delta
        idx = self._pending.find(self._sentinel)
        if idx != -1:
            before = self._pending[:idx]
            self._tail = self._pending[idx + len(self._sentinel):]
            self._pending = ""
            self._seen = True
            return before

        # Hold back any suffix that could be the start of the sentinel.
        keep = _overlap(self._pending, self._sentinel)
        cut = len(self._pending) - keep
        emit, self._pending = self._pending[:cut], self._pending[cut:]
        return emit

    def finish(self) -> str:
        """Flush remaining prose (only non-empty when no sentinel was seen)."""
        out, self._pending = self._pending, ""
        return out

    @property
    def citation_tail(self) -> str:
        return self._tail


def _parse_citations(tail: str) -> list[Citation]:
    tail = tail.strip()
    if not tail:
        return []
    try:
        data = json.loads(tail)
        return [Citation.model_validate(c) for c in data]
    except Exception:
        logger.warning(
            "agent.stream.citations_unparseable", extra={"tail": tail[:200]}
        )
        return []


def _format_context(chunks) -> str:
    if not chunks:
        return "(no relevant documentation found)"
    blocks = []
    for c in chunks:
        source = c.metadata.get("source_id", "unknown")
        section = c.metadata.get("section", "")
        header = f"[source: {source} §{section}]" if section else f"[source: {source}]"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)
