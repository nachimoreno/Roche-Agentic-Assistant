"""
Unit tests for `DocumentStore.add_document` — the live, single-document ingest
behind the "add to knowledge base" upload feature.

Fully offline: a fake embedder, vector store and lexical index stand in for the
real ONNX model / Chroma / BM25, so we assert the ingest *wiring* (chunk → embed
→ upsert → manifest → lexical rebuild) without loading anything heavy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from document_source import SourceDocument
from retrieval import DocumentStore


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self):
        self.upserts: list[list[str]] = []
        self.deleted: list[list[str]] = []
        self._all: list[tuple] = []

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upserts.append(list(ids))
        self._all.extend(zip(ids, documents))

    def delete(self, *, ids):
        self.deleted.append(list(ids))

    def get_all(self):
        return self._all


class _FakeLexical:
    def __init__(self):
        self.index_calls = 0

    def index(self, items):
        self.index_calls += 1


def _doc(*, id="doc-1", text="Some lab procedure text. " * 20):
    return SourceDocument(
        id=id,
        title="Lab SOP",
        content=text,
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"process": "cleaning", "department": "lab", "url": "http://x"},
    )


def _make(tmp_path, lexical=None):
    store = _FakeStore()
    ds = DocumentStore(
        source=None,  # add_document never touches the source
        embedder=_FakeEmbedder(),
        vector_store=store,
        manifest_path=str(tmp_path / "manifest.json"),
        lexical_index=lexical,
    )
    return ds, store


def test_add_document_writes_chunks_and_returns_count(tmp_path):
    ds, store = _make(tmp_path)
    n = ds.add_document(_doc())
    assert n > 0
    assert store.upserts and len(store.upserts[0]) == n


def test_add_document_records_metadata_for_citations(tmp_path):
    ds, _ = _make(tmp_path)
    ds.add_document(_doc())
    meta = ds.doc_metadata("doc-1")
    assert meta is not None
    assert meta["title"] == "Lab SOP"
    assert meta["url"] == "http://x"


def test_add_document_rebuilds_lexical_index(tmp_path):
    lex = _FakeLexical()
    ds, _ = _make(tmp_path, lexical=lex)
    ds.add_document(_doc())
    assert lex.index_calls == 1


def test_add_document_without_lexical_index_is_fine(tmp_path):
    ds, store = _make(tmp_path, lexical=None)
    assert ds.add_document(_doc()) > 0  # dense-only path must not error


def test_re_adding_same_id_replaces_prior_chunks(tmp_path):
    ds, store = _make(tmp_path)
    ds.add_document(_doc(id="doc-1", text="first version of the document. " * 20))
    ds.add_document(_doc(id="doc-1", text="second version of the document. " * 20))
    # The second add must delete the first add's chunks (re-upload, not duplicate).
    assert store.deleted, "re-upload should delete prior chunks"
