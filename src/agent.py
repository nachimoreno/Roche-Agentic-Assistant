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
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator

from capabilities import CAPABILITIES
from llm import LLMClient
from retrieval import DocumentStore, RetrievalResult


logger = logging.getLogger(__name__)


# Deterministic decline shown when a query is clearly off-domain (see
# RAGAgent._off_domain). Localised for the four supported languages so the
# guardrail can short-circuit without an LLM call yet still answer in the
# scientist's language; anything else falls back to English.
_DECLINE_MESSAGES = {
    "english": (
        "I don't have information on that in the lab documentation. For help, "
        "check the Application Catalog or raise a request in ServiceNow."
    ),
    "german": (
        "Dazu habe ich keine Informationen in der Labordokumentation. Für "
        "Unterstützung sehen Sie im Application Catalog nach oder erstellen Sie "
        "eine Anfrage in ServiceNow."
    ),
    "french": (
        "Je n'ai pas d'information à ce sujet dans la documentation du "
        "laboratoire. Pour obtenir de l'aide, consultez l'Application Catalog "
        "ou créez une demande dans ServiceNow."
    ),
    "italian": (
        "Non ho informazioni in merito nella documentazione di laboratorio. "
        "Per assistenza, consulta l'Application Catalog o crea una richiesta in "
        "ServiceNow."
    ),
}


def _decline_message(language: str) -> str:
    return _DECLINE_MESSAGES.get(
        language.strip().lower(), _DECLINE_MESSAGES["english"]
    )


# Phrasings that signal a question about the assistant itself ("what can you
# do?", "who are you?") rather than about lab operations. These are answered
# from the injected capabilities block (see `_SYSTEM_PROMPT_HEAD` and
# capabilities.py), NOT from retrieval — so they must bypass the off-domain
# guardrail, which would otherwise decline them: a meta-question has no match in
# the lab corpus and scores low on both retrievers. Matching is deliberately
# lenient across the four supported languages; a false positive merely lets a
# weak-retrieval query reach the LLM (harmless — the prompt still grounds
# operational answers in context), while a false negative wrongly declines a
# self-question, so we err toward matching.
_CAPABILITY_PATTERNS = re.compile(
    r"""
      \bwhat\s+(can|are)\s+you\b              # what can you do / what are you
    | \bwho\s+are\s+you\b
    | \bwhat\s+do\s+you\s+do\b
    | \bhow\s+(can|do)\s+you\s+help\b
    | \bcan\s+you\s+help\s+me\s+with\b
    | \byour\s+(capabilities|abilities|features|skills|functions)\b
    | \bwas\s+kannst\s+du\b                    # German
    | \bwer\s+bist\s+du\b
    | \bdeine\s+(fähigkeiten|funktionen)\b
    | \bque\s+(peux|pouvez)[-\s]?(?:tu|vous)\b  # French
    | \bqui\s+(es[-\s]?tu|êtes[-\s]?vous)\b
    | \b(tes|vos)\s+capacités\b
    | \bcosa\s+(puoi|sai)\s+fare\b             # Italian
    | \bchi\s+sei\b
    | \ble\s+tue\s+(capacità|funzioni)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_capability_question(message: str) -> bool:
    """True when the message asks about the assistant itself, not lab ops.

    Used to exempt self-questions from the off-domain decline so they reach the
    LLM, which answers them from the capabilities block. See `_CAPABILITY_PATTERNS`.
    """
    return bool(_CAPABILITY_PATTERNS.search(message or ""))


# Sentinel the streaming model emits between its prose answer and the JSON
# citations block. Kept on its own line so it is easy to split on.
_CITATION_SENTINEL = "\n---CITATIONS---"


# ---------------------------------------------------------------------------
# Result schema — the LLM is asked to return this exact shape.
# ---------------------------------------------------------------------------

def _clean_citation_value(v: str) -> str:
    """Strip context-header syntax the model sometimes copies verbatim.

    The retrieval context labels each chunk
    `[source="id" document="Title" section="Sec"]`. Models occasionally echo
    that syntax into a citation field — e.g. a source of `document="Cleaning
    Guide"` or `Title §Section`. We normalise back to the bare value so the
    join key (and any title fallback) stays clean. This is defense-in-depth:
    titles for display are enriched server-side from the source id, not taken
    from what the model emits here.
    """
    v = v.strip()
    # Drop a leading `document=` / `section=` / `source=` key the model copied.
    m = re.match(r'^\s*(?:document|section|source)\s*=\s*(.*)$', v, re.IGNORECASE)
    if m:
        v = m.group(1).strip()
    # Drop a trailing key the model merged in (e.g. `Title section="Sec"`).
    v = re.split(r'\s+(?:document|section)\s*=', v, maxsplit=1, flags=re.IGNORECASE)[0]
    # Drop a `§section` heading merged onto the title.
    v = v.split(" §", 1)[0]
    # Strip matching surrounding quotes.
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        v = v[1:-1].strip()
    return v


class Citation(BaseModel):
    source: str = Field(description="Source document identifier (filename or id).")
    section: str = Field(description="Section heading within the source.")
    # Human-readable document title for display, enriched server-side from
    # `source`'s metadata after the model returns. `source` stays the stable
    # source_id so attribution still joins on it; the UI falls back to `source`
    # when no title is known.
    title: Optional[str] = Field(
        default=None, description="Human-readable title for display (server-set)."
    )
    # Deep link to the source document, enriched server-side from `source`'s
    # metadata (e.g. a Google Drive webViewLink). None when the source has no
    # URL (local markdown), in which case the UI renders the citation as plain
    # text rather than a link.
    url: Optional[str] = Field(
        default=None, description="Link to the source document (server-set)."
    )

    @field_validator("source", "section", mode="before")
    @classmethod
    def _normalise(cls, v):
        return _clean_citation_value(v) if isinstance(v, str) else v


class AnswerResult(BaseModel):
    text: str = Field(description="The answer to show the scientist.")
    citations: list[Citation] = Field(default_factory=list)
    follow_ups: list[str] = Field(
        default_factory=list,
        description=(
            "2-3 short follow-up questions the scientist might naturally ask "
            "next, grounded in the same documentation."
        ),
    )
    # Set server-side (not by the model): True when retrieval was weak enough
    # that the answer should carry a "verify this" caution badge.
    low_confidence: bool = Field(default=False)
    # Set server-side (not by the model): True when the off-domain guardrail
    # declined the query (deterministic message, no LLM call, no citations).
    # Distinguishes a real decline from a normal weakly-grounded answer so the
    # orchestrator can log it as a documentation gap. See `_off_domain`.
    declined: bool = Field(default=False)
    # Top per-retriever scores for this turn (both on a [0, 1] scale), surfaced
    # so the orchestrator can persist them on a gap row for later tuning.
    # None on paths that never retrieved (e.g. capability questions short-circuit).
    retrieval_max_dense: Optional[float] = Field(default=None)
    retrieval_max_lexical: Optional[float] = Field(default=None)


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
about your own capabilities, answer from this summary — it is the
authoritative description of yourself:

{capabilities}

For capability questions, you may answer from the summary above even
when the context is empty, and you need no citations. For every other
type of question, follow the rules below strictly.

## Rules

1. For operational questions, answer ONLY from the provided context
   chunks. Handle a missing answer in one of two ways:
   a. If the request is ambiguous — for example the context holds several
      documents or instruments that could match (different cobas analyzers,
      multiple handbooks) — ask ONE short clarifying question naming the
      options, rather than guessing. Do not ask more than one.
   b. If nothing in the context is relevant, say you do not have that
      information and suggest where to look (for example, the Application
      Catalog or ServiceNow). Do not ask a clarifying question in this case.
   A clarifying question needs no citations.
2. Respond in {language}. If the context is in English, translate your
   answer into {language} naturally.
3. Be concise and direct. Scientists are often standing in a lab — give
   them the answer first, then the explanation if needed.
4. Cite the sources you used. Every claim that comes from a context chunk
   must be backed by a citation. For capability questions answered from
   the summary above, citations may be empty.
5. Do not invent procedures, product names, or values that are not in the
   context.
6. If the message is only a greeting or thanks (for example "hi" or "thank
   you"), reply in one warm line and offer a concrete example of what to ask
   — such as how to clean a centrifuge or how to request access to a lab
   application. No citations.
7. If the request is clearly outside what you cover — general world facts,
   public-web lookups, or anything unrelated to lab operations — say in one
   line that it is outside what you cover and point to where to look (for
   example the Application Catalog or ServiceNow). No citations.

## Context

{context}
"""


_JSON_OUTPUT = """
## Output

Each context chunk starts with a header like
[source="some-id" document="Some Title" section="Some Section"]. When you
cite, copy the source="..." identifier into "source" and the section="..."
value into "section" — never merge them, and never put the document="..."
title into "source".

Return ONLY a JSON object with three fields:
- "text": your answer in {language}
- "citations": a list of objects with "source" (the source="..." identifier)
  and "section" (the section heading), one for each context chunk you relied on.
- "follow_ups": a list of 2-3 short follow-up questions, in {language}, that the
  scientist might naturally ask next — each grounded in the same documentation
  and phrased as a complete question. Use an empty list if none fit.
"""


_STREAM_OUTPUT = """
## Output

Each context chunk starts with a header like
[source="some-id" document="Some Title" section="Some Section"]. Copy the
source="..." identifier into "source" and the section="..." value into
"section" — never merge them, and never put the document="..." title into
"source".

Write your complete answer in {language} as plain prose (no JSON, no
markdown code fences). When the answer is finished, output a line containing
exactly:
---CITATIONS---
and then a single JSON object with two fields:
- "citations": a list of objects with "source" (the source="..." identifier)
  and "section" (the section heading), one per context chunk you relied on (an
  empty list [] if you used none).
- "follow_ups": a list of 2-3 short follow-up questions, in {language}, that the
  scientist might naturally ask next — each grounded in the same documentation
  and phrased as a complete question (an empty list [] if none fit).
Output nothing after the JSON object.
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
    follow_ups: list[str] = field(default_factory=list)
    low_confidence: bool = False
    declined: bool = False
    retrieval_max_dense: Optional[float] = None
    retrieval_max_lexical: Optional[float] = None


StreamPiece = Union[TextDelta, AnswerComplete]


class RAGAgent:
    def __init__(
        self,
        document_store: DocumentStore,
        llm: LLMClient,
        *,
        top_k: int = 4,
        max_tokens: int = 1024,
        min_dense: float = 0.40,
        min_lexical: float = 0.85,
        warn_dense: float = 0.45,
    ) -> None:
        self._docs = document_store
        self._llm = llm
        self._top_k = top_k
        self._max_tokens = max_tokens
        # Top dense cosine below this (but above min_dense, so still answered)
        # marks the answer low-confidence for a "verify this" UI badge.
        self._warn_dense = warn_dense
        # Off-domain guardrail thresholds (both on a [0, 1] scale). A query is
        # declined deterministically (no LLM call, no citations) only when BOTH
        # retrievers are weak — dense cosine alone overlaps in/out of corpus, so
        # both signals are required to avoid wrongly rejecting real questions.
        # See `_off_domain` and settings.py for how these were calibrated.
        self._min_dense = min_dense
        self._min_lexical = min_lexical

    @property
    def document_store(self) -> DocumentStore:
        """The backing store — lets callers ingest new documents at runtime."""
        return self._docs

    def _off_domain(self, retrieval: RetrievalResult) -> bool:
        """True when retrieval is too weak to answer — clearly off-domain.

        Requires the top dense cosine AND the top normalised-BM25 score to both
        fall below their thresholds. In dense-only mode `max_lexical` is 0.0, so
        the gate reduces to the dense check alone.
        """
        return (
            retrieval.max_dense < self._min_dense
            and retrieval.max_lexical < self._min_lexical
        )

    def _low_confidence(self, retrieval: RetrievalResult) -> bool:
        """True when an answer is given but grounding is weak — flag to verify.

        Only meaningful once `_off_domain` has passed (so the query cleared the
        decline floor). The top retrieved chunk being only a moderate match
        means the answer may be loosely grounded, so the UI shows a caution.
        """
        return retrieval.max_dense < self._warn_dense

    def answer(
        self,
        message: str,
        language: str,
        history: Sequence[Turn] = (),
        retrieval_query: Optional[str] = None,
    ) -> AnswerResult:
        # Retrieve on `retrieval_query` (e.g. a spell-corrected version) when
        # given, so typos don't starve retrieval; the LLM still answers the
        # original `message`, which it understands despite typos.
        retrieval = self._docs.retrieve_scored(
            _retrieval_query(retrieval_query or message, history), k=self._top_k
        )
        # Capability/meta questions are answered from the injected capabilities
        # block, not retrieval, so they bypass the off-domain decline.
        if self._off_domain(retrieval) and not _is_capability_question(message):
            logger.info(
                "agent.declined_off_domain",
                extra={
                    "language": language,
                    "max_dense": round(retrieval.max_dense, 3),
                    "max_lexical": round(retrieval.max_lexical, 2),
                },
            )
            return AnswerResult(
                text=_decline_message(language),
                citations=[],
                declined=True,
                retrieval_max_dense=retrieval.max_dense,
                retrieval_max_lexical=retrieval.max_lexical,
            )

        chunks = retrieval.chunks
        context = _format_context(chunks)
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            language=language,
            context=context,
            capabilities=CAPABILITIES.as_prompt_block(),
        )

        payload = self._llm.complete_structured(
            system=system,
            user=message,
            schema=AnswerResult.model_json_schema(),
            history=[{"role": t.role, "content": t.content} for t in history],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        result = AnswerResult.model_validate(payload)
        result.citations = _dedupe_citations(result.citations)
        result.follow_ups = _clean_follow_ups(result.follow_ups)
        result.low_confidence = self._low_confidence(retrieval)
        result.retrieval_max_dense = retrieval.max_dense
        result.retrieval_max_lexical = retrieval.max_lexical
        self._enrich_titles(result.citations)
        logger.info(
            "agent.answered",
            extra={
                "language": language,
                "chunks_used": len(chunks),
                "citations": len(result.citations),
                "follow_ups": len(result.follow_ups),
                "low_confidence": result.low_confidence,
            },
        )
        return result

    def answer_stream(
        self,
        message: str,
        language: str,
        history: Sequence[Turn] = (),
        retrieval_query: Optional[str] = None,
    ) -> Iterator[StreamPiece]:
        """Stream the answer.

        Yields `TextDelta` events as prose arrives, then a single terminal
        `AnswerComplete` carrying the full text and the citations parsed from
        the post-delimiter JSON tail. The `---CITATIONS---` sentinel and the
        JSON after it are never leaked to the caller as prose.

        `retrieval_query` (e.g. a spell-corrected message) drives retrieval when
        given, so typos don't starve it; the LLM still answers `message`.
        """
        retrieval = self._docs.retrieve_scored(
            _retrieval_query(retrieval_query or message, history), k=self._top_k
        )
        # Capability/meta questions are answered from the injected capabilities
        # block, not retrieval, so they bypass the off-domain decline.
        if self._off_domain(retrieval) and not _is_capability_question(message):
            logger.info(
                "agent.declined_off_domain",
                extra={
                    "language": language,
                    "max_dense": round(retrieval.max_dense, 3),
                    "max_lexical": round(retrieval.max_lexical, 2),
                },
            )
            message_text = _decline_message(language)
            yield TextDelta(message_text)
            yield AnswerComplete(
                text=message_text,
                citations=[],
                declined=True,
                retrieval_max_dense=retrieval.max_dense,
                retrieval_max_lexical=retrieval.max_lexical,
            )
            return

        chunks = retrieval.chunks
        context = _format_context(chunks)
        system = _STREAM_SYSTEM_PROMPT_TEMPLATE.format(
            language=language,
            context=context,
            capabilities=CAPABILITIES.as_prompt_block(),
        )

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

        citations, follow_ups = _parse_tail(splitter.citation_tail)
        citations = _dedupe_citations(citations)
        self._enrich_titles(citations)
        low_confidence = self._low_confidence(retrieval)
        logger.info(
            "agent.streamed",
            extra={
                "language": language,
                "chunks_used": len(chunks),
                "citations": len(citations),
                "follow_ups": len(follow_ups),
                "low_confidence": low_confidence,
            },
        )
        yield AnswerComplete(
            text="".join(prose),
            citations=citations,
            follow_ups=follow_ups,
            low_confidence=low_confidence,
            retrieval_max_dense=retrieval.max_dense,
            retrieval_max_lexical=retrieval.max_lexical,
        )

    def _enrich_titles(self, citations: list[Citation]) -> None:
        """Populate each citation's display `title` and `url` from doc metadata.

        `source` stays the source_id the model cited (attribution joins on it);
        `title` is the human-readable name shown in the UI and `url` is the
        click-through link to the source document. We look both up
        authoritatively from the store rather than trusting anything the model
        emitted, and leave them ``None`` when unknown so the UI can fall back to
        `source` and render plain text. No-op if the store can't resolve
        metadata.
        """
        lookup = getattr(self._docs, "doc_metadata", None)
        if not callable(lookup):
            return
        for c in citations:
            meta = lookup(c.source) or {}
            c.title = meta.get("title")
            c.url = meta.get("url") or None


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


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Collapse citations to one row per source document, preserving order.

    The model routinely cites several chunks of the same document — sometimes
    the same section, sometimes different sections — but every chunk of a
    document resolves to the same title and the same click-through URL, so
    listing them separately is pure redundancy to the scientist. We keep the
    first citation per `source` (retaining its section as a representative
    location) so the list shows each document once and citation numbering stays
    stable.
    """
    seen: set[str] = set()
    out: list[Citation] = []
    for c in citations:
        if c.source in seen:
            continue
        seen.add(c.source)
        out.append(c)
    return out


def _clean_follow_ups(items, limit: int = 3) -> list[str]:
    """Normalise model-suggested follow-ups: strip, de-dup, cap at `limit`."""
    out: list[str] = []
    for f in items or []:
        s = str(f).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


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


def _parse_tail(tail: str) -> tuple[list[Citation], list[str]]:
    """Parse the streamed post-sentinel tail into citations and follow-ups.

    The current contract is a JSON object {"citations": [...],
    "follow_ups": [...]}, but a bare citations array (the previous format, or a
    model that ignored the object instruction) is also accepted so a malformed
    tail degrades to "citations only, no follow-ups" rather than losing both.
    """
    tail = tail.strip()
    if not tail:
        return [], []
    try:
        data = json.loads(tail)
    except Exception:
        logger.warning("agent.stream.tail_unparseable", extra={"tail": tail[:200]})
        return [], []
    try:
        if isinstance(data, list):
            return [Citation.model_validate(c) for c in data], []
        citations = [Citation.model_validate(c) for c in data.get("citations", [])]
        return citations, _clean_follow_ups(data.get("follow_ups", []))
    except Exception:
        logger.warning("agent.stream.tail_invalid", extra={"tail": tail[:200]})
        return [], []


def _retrieval_query(message: str, history: Sequence[Turn]) -> str:
    """Build the vector-search query for this turn.

    Follow-ups like "yes, that one" or "what about the rotor?" carry no
    retrievable terms on their own, so retrieval on the bare message finds
    nothing. We prepend the most recent prior *user* turn to anchor the
    search on the topic under discussion; the current message comes last so
    its terms still dominate for genuinely new questions.
    """
    prev_user = next(
        (t.content for t in reversed(history) if t.role == "user"), ""
    )
    return f"{prev_user}\n{message}" if prev_user else message


def _format_context(chunks) -> str:
    if not chunks:
        return "(no relevant documentation found)"
    blocks = []
    for c in chunks:
        source = c.metadata.get("source_id", "unknown")
        title = c.metadata.get("title")
        section = c.metadata.get("section", "")
        # Show the human title (document="...") for readability so the model can
        # name the document in its prose, but keep source="..." as the stable id
        # the model is told to cite — attribution joins on that, and titles for
        # display are enriched server-side from it. Title and section are kept
        # as clearly separate quoted fields so the model never merges them.
        parts = [f'source="{source}"']
        if title:
            parts.append(f'document="{title}"')
        if section:
            parts.append(f'section="{section}"')
        header = "[" + " ".join(parts) + "]"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)
