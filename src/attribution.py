"""
attribution.py
--------------
Resolve *which document(s)* a piece of feedback concerns, so analytics can
roll negative feedback up by document / process / department.

Two strategies, in priority order (see Feedback_Analytics_Design.md §8):

1. citation   — deterministic. A rated answer's cited docs; blame is split
                across them, each weight = 1/N.
2. embedding  — inferred. Orphan/volunteered feedback with no citation: the
                nearest chunk in the vector store lends its process, weight 1.0.

The resolver wraps the retrieval layer (a `DocumentStore`) but only needs two
methods from it — `doc_metadata(source_id)` and `retrieve(query, k)` — so it is
trivially fakeable in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

@dataclass
class AttributionRow:
    source: str
    section: Optional[str]
    process: Optional[str]
    department: Optional[str]
    weight: float
    method: str                       # "citation" | "embedding"
    distance: Optional[float] = None  # embedding distance, when method="embedding"


@dataclass
class AttributionResult:
    method: str                       # "citation" | "embedding" | "none"
    rows: list[AttributionRow]


# Citation tuple shape passed in by callers: (source, section, process, department).
Citation4 = tuple[str, Optional[str], Optional[str], Optional[str]]


class DocLookup(Protocol):
    def doc_metadata(self, source_id: str) -> Optional[dict[str, Optional[str]]]: ...
    def retrieve(self, query: str, k: int = ...): ...


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class AttributionResolver:
    def __init__(self, store: DocLookup) -> None:
        self._store = store

    def doc_meta(self, source_id: str) -> tuple[Optional[str], Optional[str]]:
        """(process, department) for a document id, or (None, None)."""
        meta = self._store.doc_metadata(source_id)
        if not meta:
            return None, None
        return meta.get("process"), meta.get("department")

    def resolve_from_citations(
        self, citations: Sequence[Citation4]
    ) -> AttributionResult:
        """Split blame equally across the cited documents (weight = 1/N)."""
        if not citations:
            return AttributionResult(method="none", rows=[])
        n = len(citations)
        weight = 1.0 / n
        rows = [
            AttributionRow(
                source=source,
                section=section,
                process=process,
                department=department,
                weight=weight,
                method="citation",
            )
            for (source, section, process, department) in citations
        ]
        return AttributionResult(method="citation", rows=rows)

    def resolve_from_text(self, text: str) -> AttributionResult:
        """Attribute free-text feedback to the nearest document in the store."""
        text = (text or "").strip()
        if not text:
            return AttributionResult(method="none", rows=[])
        chunks = self._store.retrieve(text, k=1)
        if not chunks:
            return AttributionResult(method="none", rows=[])
        top = chunks[0]
        meta = getattr(top, "metadata", {}) or {}
        # `score` is 1 - cosine_distance (higher = closer); store the distance.
        score = getattr(top, "score", None)
        distance = (1.0 - score) if isinstance(score, (int, float)) else None
        row = AttributionRow(
            source=meta.get("source_id", "unknown"),
            section=meta.get("section"),
            process=meta.get("process"),
            department=meta.get("department"),
            weight=1.0,
            method="embedding",
            distance=distance,
        )
        return AttributionResult(method="embedding", rows=[row])
