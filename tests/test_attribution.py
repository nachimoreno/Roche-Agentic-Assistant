"""
AttributionResolver unit tests.

Uses a hand-rolled fake store (no embeddings, no Chroma) so the split-weight
and nearest-doc logic is exercised in isolation and fast.
"""

from __future__ import annotations

from dataclasses import dataclass

from attribution import AttributionResolver


@dataclass
class _FakeChunk:
    metadata: dict
    score: float


class _FakeStore:
    """Implements the two methods AttributionResolver needs."""

    def __init__(self, docs: dict, nearest: _FakeChunk | None):
        self._docs = docs
        self._nearest = nearest
        self.queries: list[str] = []

    def doc_metadata(self, source_id):
        return self._docs.get(source_id)

    def retrieve(self, query, k=1):
        self.queries.append(query)
        return [self._nearest] if self._nearest is not None else []


def test_doc_meta_returns_process_and_department():
    store = _FakeStore({"a.md": {"process": "onboarding", "department": "it"}}, None)
    resolver = AttributionResolver(store)
    assert resolver.doc_meta("a.md") == ("onboarding", "it")
    assert resolver.doc_meta("missing.md") == (None, None)


def test_resolve_from_citations_splits_weight_evenly():
    resolver = AttributionResolver(_FakeStore({}, None))
    result = resolver.resolve_from_citations([
        ("a.md", "Intro", "onboarding", "it"),
        ("b.md", "Setup", "instrument-booking", "lab-operations"),
        ("c.md", None, "onboarding", "it"),
    ])
    assert result.method == "citation"
    assert len(result.rows) == 3
    assert all(abs(r.weight - 1 / 3) < 1e-9 for r in result.rows)
    assert abs(sum(r.weight for r in result.rows) - 1.0) < 1e-9
    # Aggregating by process: onboarding gets 2/3, booking 1/3.
    by_proc: dict[str, float] = {}
    for r in result.rows:
        by_proc[r.process] = by_proc.get(r.process, 0.0) + r.weight
    assert abs(by_proc["onboarding"] - 2 / 3) < 1e-9


def test_resolve_from_citations_empty_is_none():
    resolver = AttributionResolver(_FakeStore({}, None))
    result = resolver.resolve_from_citations([])
    assert result.method == "none" and result.rows == []


def test_resolve_from_text_uses_nearest_doc():
    nearest = _FakeChunk(
        metadata={"source_id": "04_booking_instruments.md", "section": "Reserving",
                  "process": "instrument-booking", "department": "lab-operations"},
        score=0.82,
    )
    resolver = AttributionResolver(_FakeStore({}, nearest))
    result = resolver.resolve_from_text("the booking tool keeps failing")

    assert result.method == "embedding"
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.source == "04_booking_instruments.md"
    assert row.process == "instrument-booking"
    assert row.weight == 1.0
    assert abs(row.distance - (1.0 - 0.82)) < 1e-9


def test_resolve_from_text_empty_or_no_hit_is_none():
    assert AttributionResolver(_FakeStore({}, None)).resolve_from_text("x").method == "none"
    nearest = _FakeChunk(metadata={"source_id": "a.md"}, score=0.5)
    assert AttributionResolver(_FakeStore({}, nearest)).resolve_from_text("   ").method == "none"
