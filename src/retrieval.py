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
from datetime import datetime, timezone
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
#
# v3: contradiction-handling metadata (modified_at/version/effective_date/
# status/supersedes) is now carried onto chunks; front-matter is stripped before
# hashing, so adding those keys doesn't change a doc's content hash — bumping the
# scheme forces the one-time corpus re-embed that lands the new chunk metadata.
_CHUNK_SCHEME_VERSION = 3

# Metadata keys that drive contradiction handling — recency/version signals
# carried from doc front-matter onto each chunk (alongside process/department)
# so the deterministic version-dedup resolver and the in-prompt conflict
# detector can both read them. See Contradiction_Handling_Design.md §5–6.
_CONFLICT_META_KEYS = ("version", "effective_date", "status", "owner", "supersedes")


@dataclass
class IngestReport:
    documents_seen: int
    documents_reindexed: int
    chunks_written: int


@dataclass
class RetrievalResult:
    """Retrieved chunks plus the raw top scores from each retriever.

    `chunks` is the fused, ranked result (RRF scores in hybrid mode). The two
    score fields expose the *un-fused* top signal from each retriever so a
    caller can gauge confidence: both are in [0, 1] — `max_dense` is the best
    cosine similarity, and `max_lexical` is the best normalised BM25 "match
    completeness" (0.0 in dense-only mode, or when no query term is in the
    corpus). Both are corpus-size stable, so thresholds over them don't drift as
    the corpus grows. These are kept separate from the fused list because RRF
    scores reflect rank, not relevance, and so make a poor confidence signal on
    their own.
    """

    chunks: list[Chunk]
    max_dense: float
    max_lexical: float


@dataclass
class _DocChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _doc_meta_record(doc: SourceDocument) -> dict[str, Optional[str]]:
    """The citation/resolution metadata stored per source_id on every ingest.

    `process`/`department`/`title`/`url` drive citation→process attribution and
    clickable links; `modified_at` plus the front-matter recency keys
    (`version`/`effective_date`/`status`/`owner`/`supersedes`) feed the
    contradiction resolver. `modified_at` is normalised to an ISO string so the
    record is plain-JSON (matching what's stored on chunk metadata).
    """
    record: dict[str, Optional[str]] = {
        "process": doc.metadata.get("process"),
        "department": doc.metadata.get("department"),
        "title": doc.title,
        # Deep link to the source (e.g. Drive webViewLink); used to make
        # citations clickable. Empty/absent for sources without URLs.
        "url": doc.metadata.get("url"),
        "modified_at": doc.modified_at.isoformat() if doc.modified_at else None,
    }
    for key in _CONFLICT_META_KEYS:
        record[key] = doc.metadata.get(key)
    return record


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
        # When set, a BM25 index is available for hybrid retrieval. The `_hybrid`
        # flag selects the active mode: keeping the index built but toggling the
        # flag lets the /settings surface switch dense <-> hybrid at runtime
        # without rebuilding. Dense-only when no index was supplied.
        self._lexical = lexical_index
        self._hybrid = lexical_index is not None
        # source_id -> citation/resolution metadata (process, department, title,
        # url, modified_at + conflict keys; see `_doc_meta_record`) for
        # citation→process attribution and the contradiction resolver. Populated
        # on every ingest for *all* docs seen (including unchanged ones that skip
        # re-embedding).
        self._doc_meta: dict[str, dict[str, Optional[str]]] = {}

    @property
    def hybrid(self) -> bool:
        """Whether hybrid (dense + BM25) retrieval is active. Only True when a
        lexical index is available. Settable at runtime from /settings."""
        return self._hybrid and self._lexical is not None

    @hybrid.setter
    def hybrid(self, on: bool) -> None:
        self._hybrid = bool(on)

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
            # Record metadata for every doc seen, even if its content is
            # unchanged and we skip re-embedding below — the citation→process
            # lookup (and the conflict resolver) must cover the whole corpus, not
            # just docs touched this run.
            self._doc_meta[doc.id] = _doc_meta_record(doc)
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

    def add_document(self, doc: SourceDocument) -> int:
        """Ingest a single document into the live index, now — without a full
        re-scan of the source.

        Chunks, embeds and upserts the document, records it in the manifest and
        the citation metadata map, and rebuilds the lexical index so BM25 sees
        it too. Returns the number of chunks written. Used by the "add document"
        feature so an uploaded file becomes answerable immediately; because the
        file is also persisted to the source (Drive), the next startup `ingest()`
        keeps it rather than dropping it as stale.
        """
        manifest = self._load_manifest()

        # Replace any prior chunks for this id (re-upload of the same file).
        prior = manifest.get(doc.id)
        if prior:
            self._store.delete(ids=prior.get("chunk_ids", []))

        doc_chunks = list(_chunk_document(doc))
        if doc_chunks:
            embeddings = self._embedder.embed([c.text for c in doc_chunks])
            self._store.upsert(
                ids=[c.chunk_id for c in doc_chunks],
                embeddings=embeddings,
                documents=[c.text for c in doc_chunks],
                metadatas=[c.metadata for c in doc_chunks],
            )

        manifest[doc.id] = {
            "hash": _hash(doc.content),
            "scheme": _CHUNK_SCHEME_VERSION,
            "modified_at": doc.modified_at.isoformat(),
            "chunk_ids": [c.chunk_id for c in doc_chunks],
        }
        self._save_manifest(manifest)

        self._doc_meta[doc.id] = _doc_meta_record(doc)

        # Rebuild the lexical index from the full corpus so BM25 and dense
        # retrieval stay in lockstep (cheap at MVP scale).
        if self._lexical is not None:
            self._lexical.index(self._store.get_all())

        logger.info(
            "ingest.add_document",
            extra={"doc_id": doc.id, "chunks": len(doc_chunks)},
        )
        return len(doc_chunks)

    def doc_metadata(self, source_id: str) -> Optional[dict[str, Optional[str]]]:
        """Citation/resolution metadata for a document id, or None if unknown.

        Includes process/department/title/url plus recency keys (modified_at,
        version, effective_date, status, owner, supersedes). Populated during
        `ingest`; the key is `SourceDocument.id`, which is also what the agent
        emits as `Citation.source`.
        """
        return self._doc_meta.get(source_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        return self.retrieve_scored(query, k=k).chunks

    def retrieve_scored(self, query: str, k: int = 4) -> RetrievalResult:
        """Retrieve `k` chunks plus the top score from each retriever.

        Same ranking as `retrieve`, but also surfaces `max_dense` (best cosine
        similarity) and `max_lexical` (best BM25 score) so a caller can gauge
        retrieval confidence — e.g. to decline clearly off-domain queries.
        """
        embedding = self._embedder.embed([query])[0]

        # Dense-only path (no lexical index, or hybrid switched off at runtime).
        if self._lexical is None or not self._hybrid:
            chunks = self._store.query(embedding=embedding, k=k)
            # Best dense score is the retrieval-confidence signal; capture it
            # before version-dedup, which may drop the top chunk as a stale copy.
            max_dense = chunks[0].score if chunks else 0.0
            chunks = _collapse_superseded(chunks)
            logger.info(
                "retrieval.done",
                extra={"mode": "dense", "k": k, "returned": len(chunks)},
            )
            return RetrievalResult(
                chunks=chunks,
                max_dense=max_dense,
                max_lexical=0.0,
            )

        # Hybrid: blend a wider dense + BM25 candidate pool with RRF.
        pool = max(k, _HYBRID_POOL)
        dense = self._store.query(embedding=embedding, k=pool)
        lexical = self._lexical.search(query, k=pool)
        # Collapse superseded versions on the final fused ranking so a stale copy
        # never competes with its current replacement in the context (Case A —
        # see Contradiction_Handling_Design.md §7). max_dense/max_lexical come
        # from the un-collapsed pools, so dropping a stale top hit doesn't
        # understate confidence.
        fused = _collapse_superseded(_reciprocal_rank_fusion([dense, lexical], k=k))
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
        return RetrievalResult(
            chunks=fused,
            max_dense=dense[0].score if dense else 0.0,
            max_lexical=lexical[0].score if lexical else 0.0,
        )

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
# Version-dedup — Case A (stale-version) resolution
#
# Before the LLM sees the context, collapse chunks belonging to a *superseded*
# version of a document so a stale copy never competes with its current
# replacement. This resolves the Google-Drive "multiple versions" problem
# (architecture §2.2) deterministically and shrinks what reaches the in-prompt
# conflict detector down to genuine Case-B disagreements. Pure functions over
# already-fetched chunk metadata — no LLM, no Chroma, fully unit-testable.
# See Contradiction_Handling_Design.md §7.
# ---------------------------------------------------------------------------

# Tokens like "v2" / "version 3" / "rev. 4" are stripped before comparing
# titles, so "Cleaning Lab Devices v2" and "Cleaning Lab Devices v3" normalise
# to the same identity.
_VERSION_TOKEN_RE = re.compile(r"\b(?:v|ver|version|rev|revision)\.?\s*\d+\b", re.IGNORECASE)
# Title similarity at/above which two same-process docs are treated as versions
# of one logical document. Deliberately high: the confident-identity guard errs
# toward *not* collapsing, so an uncertain pair is left for conflict detection
# rather than silently dropped (Contradiction_Handling_Design.md §7, §13).
_TITLE_SIMILARITY = 0.8


def _collapse_superseded(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks of any document a newer version supersedes.

    Groups chunks by *logical-document identity* and, within each group, keeps
    only the chunks of the newest doc. Identity is taken only when **confident**:
    an explicit `supersedes` chain, or an identical `process` plus a near-
    identical normalised title. When identity is uncertain the docs are left
    untouched for in-prompt conflict detection to treat as a potential genuine
    disagreement — we never silently drop a doc we can't prove is a stale
    version. Recency is decided by `effective_date`, falling back to
    `modified_at`; a doc explicitly `status: deprecated` is always dropped when a
    non-deprecated sibling exists.

    Preserves input order of the surviving chunks. A no-op (returns the input
    list) when nothing is superseded, so the common case is free.
    """
    if len(chunks) < 2:
        return list(chunks)

    sources = _source_snapshots(chunks)
    if len(sources) < 2:
        return list(chunks)  # all chunks from one doc — nothing to collapse

    # An explicit `supersedes: X` declares direction outright: X is stale and is
    # forced out whenever its superseder is also present, regardless of dates.
    explicitly_stale: set[str] = set()
    uf = _UnionFind(sources.keys())
    for sid, meta in sources.items():
        for target in _supersedes_targets(meta.get("supersedes")):
            if target in sources:
                uf.union(sid, target)
                explicitly_stale.add(target)

    # Heuristic identity (confident only): same process + near-identical title.
    sids = list(sources)
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            if _same_logical_doc(sources[a], sources[b]):
                uf.union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for sid in sources:
        groups[uf.find(sid)].append(sid)

    keep: set[str] = set()
    for members in groups.values():
        if len(members) == 1:
            keep.add(members[0])
            continue
        # Prefer a doc no present sibling explicitly supersedes; among those pick
        # the newest current one. Fall back to the whole group if the chain
        # somehow marked every member stale (cyclic/ill-formed front-matter).
        candidates = [m for m in members if m not in explicitly_stale] or members
        keep.add(max(candidates, key=lambda m: _recency_key(sources[m])))

    if len(keep) == len(sources):
        return list(chunks)  # nothing dropped
    return [c for c in chunks if c.metadata.get("source_id") in keep]


def _source_snapshots(chunks: list[Chunk]) -> dict[str, dict[str, Any]]:
    """One metadata snapshot per source_id (first chunk wins; all agree)."""
    out: dict[str, dict[str, Any]] = {}
    for c in chunks:
        sid = c.metadata.get("source_id")
        if not sid or sid in out:
            continue
        m = c.metadata
        out[sid] = {
            "process": m.get("process"),
            "title": m.get("title") or "",
            "status": str(m.get("status") or "").strip().lower(),
            "effective_date": m.get("effective_date"),
            "modified_at": m.get("modified_at"),
            "supersedes": m.get("supersedes"),
        }
    return out


def _same_logical_doc(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two docs are confidently the same logical document.

    Requires an identical (non-empty) `process` and a near-identical normalised
    title. The strict bar is intentional — see `_TITLE_SIMILARITY`.
    """
    pa, pb = a.get("process"), b.get("process")
    if not pa or pa != pb:
        return False
    return _title_similar(a.get("title") or "", b.get("title") or "")


def _normalize_title(title: str) -> str:
    """Lowercase, strip version tokens, collapse to alphanumeric words."""
    t = _VERSION_TOKEN_RE.sub(" ", title.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _title_similar(a: str, b: str) -> bool:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= _TITLE_SIMILARITY


def _supersedes_targets(value: Any) -> list[str]:
    """Source ids a `supersedes` value names (comma/space separated)."""
    if not value:
        return []
    return [t.strip() for t in re.split(r"[,\s]+", str(value)) if t.strip()]


def _recency_key(meta: dict[str, Any]) -> tuple[bool, float]:
    """Sort key for picking the surviving version: (is_current, recency).

    A non-deprecated doc always outranks a deprecated sibling; among equals the
    later effective_date (else modified_at) wins.
    """
    is_current = meta.get("status") != "deprecated"
    return (is_current, _recency_seconds(meta))


def _recency_seconds(meta: dict[str, Any]) -> float:
    dt = _parse_dt(meta.get("effective_date")) or _parse_dt(meta.get("modified_at"))
    return dt.timestamp() if dt is not None else 0.0


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO date or datetime to a UTC-aware datetime, else None.

    Naive inputs (a bare `effective_date: 2026-05-01`) are pinned to UTC so they
    compare consistently against tz-aware `modified_at` timestamps.
    """
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    for candidate in (s, f"{s}T00:00:00+00:00"):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return None


class _UnionFind:
    """Tiny union-find over source ids for grouping logical-document versions."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {i: i for i in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path-compress so repeated finds stay near-flat.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


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
            # Recency signal for version-dedup + the in-prompt conflict detector;
            # an ISO string keeps chunk metadata plain-JSON / Chroma-safe (no
            # datetime, no None). effective_date front-matter overrides it.
            if doc.modified_at is not None:
                metadata["modified_at"] = doc.modified_at.isoformat()
            # Carry process/department (embedding-fallback attribution) plus the
            # contradiction-handling recency/version keys onto each chunk so the
            # resolver and the LLM can read them off the chunk directly.
            for key in ("process", "department", *_CONFLICT_META_KEYS):
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
