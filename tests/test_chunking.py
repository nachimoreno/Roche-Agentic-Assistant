"""
tests/test_chunking.py
----------------------
Unit tests for the chunker in retrieval.py. Pure string logic, no embedding.

Regression focus: DOCX/PDF-extracted text has no Markdown `##` headings and no
blank-line paragraph breaks (only single newlines). Such documents previously
collapsed into one oversized chunk that the embedder truncated, making most of
their content unretrievable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from document_source import SourceDocument
from retrieval import _MAX_CHUNK_CHARS, _atomize, _chunk_document, _split_long


def _doc(content: str) -> SourceDocument:
    return SourceDocument(
        id="d1",
        title="Doc",
        content=content,
        modified_at=datetime.now(tz=timezone.utc),
    )


def test_split_long_returns_short_text_unchanged():
    assert _split_long("a short paragraph", _MAX_CHUNK_CHARS) == ["a short paragraph"]


def test_split_long_never_exceeds_max_for_single_newline_text():
    # DOCX-style: long body joined only by single newlines, no blank lines.
    body = "\n".join(f"Line {i} with some descriptive content." for i in range(200))
    chunks = _split_long(body, _MAX_CHUNK_CHARS)
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)


def test_split_long_hard_slices_unbroken_run():
    # No separators at all — must still be sliced to <= max_chars.
    body = "x" * (_MAX_CHUNK_CHARS * 3 + 17)
    chunks = _split_long(body, _MAX_CHUNK_CHARS)
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)
    assert "".join(chunks) == body


def test_atomize_prefers_blank_lines_then_newlines():
    assert _atomize("short", 100) == ["short"]


def test_chunk_document_splits_docx_like_text_into_many_chunks():
    # Single-newline prose with no Markdown structure, well over one chunk.
    body = "\n".join(f"Paragraph {i}: handling and storage details here." for i in range(150))
    chunks = list(_chunk_document(_doc(body)))
    assert len(chunks) > 1
    assert all(len(c.text) <= _MAX_CHUNK_CHARS for c in chunks)
    # Metadata still falls back to the doc title when there is no H2 heading.
    assert chunks[0].metadata["section"] == "Doc"
