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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_source import DocumentSource, SourceDocument
from embeddings import EmbeddingProvider
from vector_store import Chunk, VectorStore


logger = logging.getLogger(__name__)


_MAX_CHUNK_CHARS = 1200
_MIN_CHUNK_CHARS = 80

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
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._store = vector_store
        self._manifest_path = Path(manifest_path)

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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        embedding = self._embedder.embed([query])[0]
        chunks = self._store.query(embedding=embedding, k=k)
        logger.info(
            "retrieval.done",
            extra={"k": k, "returned": len(chunks)},
        )
        return chunks

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
            yield _DocChunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "source_id": doc.id,
                    "title": doc.title,
                    "section": heading or doc.title,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                },
            )


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
