"""
tests/test_lexical_index.py
---------------------------
Unit tests for BM25Index and the RRF fusion helper. Pure, deterministic — no
embeddings or Drive calls.
"""

from __future__ import annotations

from lexical_index import BM25Index, _tokenize
from retrieval import _reciprocal_rank_fusion
from vector_store import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, text=text, metadata={"source_id": cid}, score=0.0)


def test_tokenize_is_lowercase_and_unicode():
    assert _tokenize("Calibration DRIFT-2024") == ["calibration", "drift", "2024"]
    # Accented (German) terms survive as whole tokens.
    assert "lösungsmittel" in _tokenize("Lösungsmittel entsorgen")


def test_exact_token_match_ranks_first():
    idx = BM25Index()
    idx.index([
        _chunk("waste", "Dispose of solvent waste in the yellow bin per SOP-4471."),
        _chunk("onboard", "New employees request badge access on day one."),
        _chunk("booking", "Book the centrifuge through the instrument calendar."),
    ])
    hits = idx.search("SOP-4471", k=3)
    assert hits, "expected a lexical match for the exact code"
    assert hits[0].id == "waste"


def test_search_drops_non_matches():
    idx = BM25Index()
    idx.index([_chunk("a", "centrifuge cleaning procedure")])
    # No shared terms -> no results, rather than a zero-score chunk.
    assert idx.search("quarterly budget forecast", k=4) == []


def test_empty_index_and_empty_query_return_nothing():
    idx = BM25Index()
    assert idx.search("anything", k=4) == []
    idx.index([_chunk("a", "some text")])
    assert idx.search("", k=4) == []


def test_rrf_rewards_agreement_across_lists():
    a, b, c = _chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")
    dense = [a, b, c]      # a best by dense
    lexical = [b, a, c]    # b best by lexical, a close second
    fused = _reciprocal_rank_fusion([dense, lexical], k=3)
    # 'a' (ranks 1 & 2) and 'b' (ranks 2 & 1) both beat 'c' (ranks 3 & 3).
    assert {fused[0].id, fused[1].id} == {"a", "b"}
    assert fused[2].id == "c"


def test_rrf_includes_items_unique_to_one_list():
    dense = [_chunk("a", "x")]
    lexical = [_chunk("b", "y")]
    fused = _reciprocal_rank_fusion([dense, lexical], k=5)
    assert {ch.id for ch in fused} == {"a", "b"}
