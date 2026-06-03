"""
agent.py
--------
RAG agent for the Roche Scientist Assistant.

Composes the LLM provider seam with the DocumentStore. Knows nothing
about Groq, Anthropic, ChromaDB, or sentence-transformers directly.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pydantic import BaseModel, Field

from llm import LLMClient
from retrieval import DocumentStore


logger = logging.getLogger(__name__)


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

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Roche Scientist Assistant. You help scientists find accurate
operational information from internal documentation.

## Rules

1. Answer ONLY from the provided context chunks. If the answer is not in
   the context, say you do not have that information and suggest where the
   scientist could look (for example, the Application Catalog or
   ServiceNow).
2. Respond in {language}. If the context is in English, translate your
   answer into {language} naturally.
3. Be concise and direct. Scientists are often standing in a lab — give
   them the answer first, then the explanation if needed.
4. Cite the sources you used. Every claim that comes from a context chunk
   must be backed by a citation.
5. Do not invent procedures, product names, or values that are not in the
   context.

## Context

{context}

## Output

Return ONLY a JSON object with two fields:
- "text": your answer in {language}
- "citations": a list of objects with "source" (the source_id) and
  "section" (the section heading), one for each context chunk you relied on.
"""


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
