"""
retrieval.py
------------
Document ingestion and similarity search.

`DocumentStore` is pure composition over the three provider interfaces
(`DocumentSource`, `EmbeddingProvider`, `VectorStore`). It has no
knowledge of ChromaDB, sentence-transformers, or the local filesystem —
swap any of those for a different implementation and this class is
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from document_source import DocumentSource, SourceDocument
from embeddings import EmbeddingProvider
from lexical_index import LexicalIndex
from vector_store import Chunk, VectorStore


logger = logging.getLogger(__name__)


_MAX_CHUNK_CHARS = 1200
_MIN_CHUNK_CHARS = 80

# Hybrid retrieval tuning. We pull a wider candidate pool from each retriever
# than the final k so fusion has signal to work with, then blend the two
# rankings with Reciprocal Rank Fusion. _RRF_K is the standard RRF constant; a
# larger value flattens the contribution of top ranks.
_HYBRID_POOL = 20
_RRF_K = 60

# Bump when chunking logic changes so previously-ingested documents are
# re-chunked even though their content hash is unchanged. Without this, the
# manifest's hash check would treat a doc as "unchanged" and keep stale chunks
# produced by the old logic.
_CHUNK_SCHEME_VERSION = 2


@dataclass
class IngestReport:
    documents_seen: int
    documents_reindexed: int
    chunks_written: int


@dataclass
class _DocChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class DocumentStore:
    def __init__(
        self,
        source: DocumentSource,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        manifest_path: str | Path = ".chroma/manifest.json",
        lexical_index: Optional[LexicalIndex] = None,
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._store = vector_store
        self._manifest_path = Path(manifest_path)
        # When set, retrieval is hybrid (dense + BM25). When None, dense-only.
        self._lexical = lexical_index
        # source_id -> {"process", "department", "title"} for citation→process
        # attribution. Populated on every ingest for *all* docs seen (including
        # unchanged ones that skip re-embedding).
        self._doc_meta: dict[str, dict[str, Optional[str]]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self) -> IngestReport:
        manifest = self._load_manifest()
        new_manifest: dict[str, dict[str, Any]] = {}
        seen = 0
        reindexed = 0
        chunks_written = 0

        for doc in self._source.list_documents():
            seen += 1
            # Record process/department for every doc seen, even if its content
            # is unchanged and we skip re-embedding below — the citation→process
            # lookup must cover the whole corpus, not just docs touched this run.
            self._doc_meta[doc.id] = {
                "process": doc.metadata.get("process"),
                "department": doc.metadata.get("department"),
                "title": doc.title,
            }
            content_hash = _hash(doc.content)
            entry = {
                "hash": content_hash,
                "scheme": _CHUNK_SCHEME_VERSION,
                "modified_at": doc.modified_at.isoformat(),
                "chunk_ids": [],
            }

            prior = manifest.get(doc.id)
            if (
                prior
                and prior.get("hash") == content_hash
                and prior.get("scheme") == _CHUNK_SCHEME_VERSION
            ):
                # Unchanged content *and* same chunking scheme — keep prior chunks.
                entry["chunk_ids"] = prior.get("chunk_ids", [])
                new_manifest[doc.id] = entry
                continue

            # Stale or new — delete prior chunks then re-embed.
            if prior:
                self._store.delete(ids=prior.get("chunk_ids", []))

            doc_chunks = list(_chunk_document(doc))
            if not doc_chunks:
                new_manifest[doc.id] = entry
                continue

            embeddings = self._embedder.embed([c.text for c in doc_chunks])
            self._store.upsert(
                ids=[c.chunk_id for c in doc_chunks],
                embeddings=embeddings,
                documents=[c.text for c in doc_chunks],
                metadatas=[c.metadata for c in doc_chunks],
            )
            entry["chunk_ids"] = [c.chunk_id for c in doc_chunks]
            new_manifest[doc.id] = entry
            reindexed += 1
            chunks_written += len(doc_chunks)
            logger.info(
                "ingest.doc",
                extra={"doc_id": doc.id, "chunks": len(doc_chunks)},
            )

        # Documents that disappeared from the source — drop their vectors.
        for stale_id, prior in manifest.items():
            if stale_id not in new_manifest:
                self._store.delete(ids=prior.get("chunk_ids", []))
                logger.info("ingest.dropped", extra={"doc_id": stale_id})

        self._save_manifest(new_manifest)

        # Rebuild the lexical index from the full corpus (not just docs touched
        # this run) so BM25 and dense retrieval always see the same chunks.
        if self._lexical is not None:
            self._lexical.index(self._store.get_all())

        report = IngestReport(
            documents_seen=seen,
            documents_reindexed=reindexed,
            chunks_written=chunks_written,
        )
        logger.info(
            "ingest.done",
            extra={
                "documents_seen": seen,
                "documents_reindexed": reindexed,
                "chunks_written": chunks_written,
            },
        )
        return report

    def doc_metadata(self, source_id: str) -> Optional[dict[str, Optional[str]]]:
        """Process/department/title for a document id, or None if unknown.

        Populated during `ingest`; the key is `SourceDocument.id`, which is also
        what the agent emits as `Citation.source`.
        """
        return self._doc_meta.get(source_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        embedding = self._embedder.embed([query])[0]

        # Dense-only path (no lexical index configured).
        if self._lexical is None:
            chunks = self._store.query(embedding=embedding, k=k)
            logger.info(
                "retrieval.done",
                extra={"mode": "dense", "k": k, "returned": len(chunks)},
            )
            return chunks

        # Hybrid: blend a wider dense + BM25 candidate pool with RRF.
        pool = max(k, _HYBRID_POOL)
        dense = self._store.query(embedding=embedding, k=pool)
        lexical = self._lexical.search(query, k=pool)
        fused = _reciprocal_rank_fusion([dense, lexical], k=k)
        logger.info(
            "retrieval.done",
            extra={
                "mode": "hybrid",
                "k": k,
                "dense": len(dense),
                "lexical": len(lexical),
                "returned": len(fused),
            },
        )
        return fused

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "manifest.unreadable", extra={"path": str(self._manifest_path)}
            )
            return {}

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    result_lists: Iterable[list[Chunk]], k: int, rrf_k: int = _RRF_K
) -> list[Chunk]:
    """Combine several ranked chunk lists into one via Reciprocal Rank Fusion.

    RRF scores each chunk by 1/(rrf_k + rank) summed across the lists it
    appears in, so a chunk ranked highly by either retriever surfaces — and one
    ranked by both is reinforced. It works on ranks, not raw scores, so the
    incomparable scales of cosine similarity and BM25 never need normalising.
    """
    fused: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            fused[chunk.id] += 1.0 / (rrf_k + rank)
            chunk_by_id.setdefault(chunk.id, chunk)

    ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    out: list[Chunk] = []
    for cid in ranked[:k]:
        c = chunk_by_id[cid]
        out.append(Chunk(id=c.id, text=c.text, metadata=dict(c.metadata), score=fused[cid]))
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _chunk_document(doc: SourceDocument):
    sections = _split_by_h2(doc.content)
    for section_index, (heading, body) in enumerate(sections):
        body = body.strip()
        if len(body) < _MIN_CHUNK_CHARS:
            continue
        sub_chunks = _split_long(body, _MAX_CHUNK_CHARS)
        for chunk_index, text in enumerate(sub_chunks):
            chunk_id = f"{doc.id}::{section_index}::{chunk_index}"
            metadata = {
                "source_id": doc.id,
                "title": doc.title,
                "section": heading or doc.title,
                "section_index": section_index,
                "chunk_index": chunk_index,
            }
            # Carry process/department onto each chunk so the embedding fallback
            # can read them off the nearest chunk for orphan feedback.
            for key in ("process", "department"):
                value = doc.metadata.get(key)
                if value is not None:
                    metadata[key] = value
            yield _DocChunk(chunk_id=chunk_id, text=text, metadata=metadata)


def _split_by_h2(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    # Capture any preamble before the first heading.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end].strip()))
    return sections


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # Break into the smallest natural units we can, then greedily pack them
    # back up to max_chars. _atomize guarantees every unit is <= max_chars, so
    # text without blank-line structure (DOCX/PDF-extracted prose) still chunks
    # properly instead of collapsing into one oversized, un-embeddable blob.
    units = _atomize(text, max_chars)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for unit in units:
        if buffer and buffer_len + len(unit) + 2 > max_chars:
            chunks.append("\n\n".join(buffer))
            buffer = [unit]
            buffer_len = len(unit)
        else:
            buffer.append(unit)
            buffer_len += len(unit) + 2
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _atomize(segment: str, max_chars: int) -> list[str]:
    """Break `segment` into units each <= max_chars.

    Splits on the coarsest separator available (blank lines, then single
    newlines), recursing into pieces that are still too long. A run with no
    newlines at all is hard-sliced on character count as a last resort.
    """
    if len(segment) <= max_chars:
        return [segment]
    for sep in ("\n\n", "\n"):
        if sep in segment:
            parts = [p.strip() for p in segment.split(sep) if p.strip()]
            if len(parts) > 1:
                out: list[str] = []
                for part in parts:
                    out.extend(_atomize(part, max_chars))
                return out
    return [segment[i : i + max_chars] for i in range(0, len(segment), max_chars)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
