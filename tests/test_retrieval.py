"""
Retrieval tests.

Ingests the real `data/docs/` through the production stack
(LocalMarkdownSource -> FastEmbedProvider -> ChromaVectorStore)
into a temp directory, then asserts semantic queries return the expected
source documents. No API calls — runs entirely on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from document_source import LocalMarkdownSource
from embeddings import FastEmbedProvider
from lexical_index import BM25Index
from retrieval import DocumentStore
from vector_store import ChromaVectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent
# A frozen copy of the markdown corpus, kept as a test fixture. The production
# app no longer ships local docs (it ingests from Google Drive), but these
# tests need a deterministic, offline corpus to assert retrieval behaviour.
DOCS_PATH = REPO_ROOT / "tests" / "fixtures" / "docs"


@pytest.fixture(scope="module")
def document_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chroma")
    embedder = FastEmbedProvider()
    store = ChromaVectorStore(
        path=str(tmp), collection_name="test_retrieval"
    )
    docs = DocumentStore(
        source=LocalMarkdownSource(DOCS_PATH),
        embedder=embedder,
        vector_store=store,
        manifest_path=str(tmp / "manifest.json"),
    )
    report = docs.ingest()
    assert report.documents_seen >= 6
    assert report.chunks_written > 0
    return docs


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("how do I clean the centrifuge", "06_cleaning_lab_devices.md"),
        ("safe alcohol percentage for cleaning a laptop",
         "06_cleaning_lab_devices.md"),
        ("how can I check sample stock", "05_checking_sample_stock.md"),
        ("book an instrument time slot", "04_booking_instruments.md"),
        ("biological spill cleanup procedure", "07_decontamination.md"),
        ("my virtual session keeps disconnecting",
         "08_virtual_session_troubleshooting.md"),
        ("how do I request access to a lab application",
         "01_onboarding_access_requests.md"),
        ("create an incident in ServiceNow", "03_incident_reporting.md"),
        ("what can you help me with", "00_assistant_capabilities.md"),
        ("what can the assistant do", "00_assistant_capabilities.md"),
    ],
)
def test_semantic_retrieval(document_store, query, expected_source):
    chunks = document_store.retrieve(query, k=3)
    sources = {c.metadata.get("source_id") for c in chunks}
    assert expected_source in sources, (
        f"expected {expected_source} in top-3 for query {query!r}, "
        f"got {sources}"
    )


def test_multilingual_query_finds_english_doc(document_store):
    # German query against an English corpus.
    chunks = document_store.retrieve(
        "Wie kann ich den Probenbestand prüfen?", k=3
    )
    sources = {c.metadata.get("source_id") for c in chunks}
    assert "05_checking_sample_stock.md" in sources


def test_reingest_is_idempotent(document_store):
    # Re-running ingest on unchanged sources should not write new chunks.
    second = document_store.ingest()
    assert second.chunks_written == 0
    assert second.documents_reindexed == 0


# ---------------------------------------------------------------------------
# Hybrid retrieval (dense + BM25)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chroma_hybrid")
    store = ChromaVectorStore(path=str(tmp), collection_name="test_hybrid")
    docs = DocumentStore(
        source=LocalMarkdownSource(DOCS_PATH),
        embedder=FastEmbedProvider(),
        vector_store=store,
        manifest_path=str(tmp / "manifest.json"),
        lexical_index=BM25Index(),
    )
    docs.ingest()
    return docs


def test_hybrid_promotes_exact_keyword_match(hybrid_store):
    # "ServiceNow" is a distinctive keyword; BM25 should pull the incident-
    # reporting doc to the very top, where dense-only ranks it lower.
    chunks = hybrid_store.retrieve("ServiceNow incident", k=3)
    assert chunks[0].metadata.get("source_id") == "03_incident_reporting.md"


def test_hybrid_preserves_semantic_recall(hybrid_store):
    # A paraphrase with no shared keywords must still work — the dense half of
    # the hybrid carries it.
    chunks = hybrid_store.retrieve("safe alcohol percentage for cleaning a laptop", k=3)
    sources = {c.metadata.get("source_id") for c in chunks}
    assert "06_cleaning_lab_devices.md" in sources
