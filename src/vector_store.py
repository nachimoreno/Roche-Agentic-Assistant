"""
vector_store.py
---------------
Vector store seam.

`VectorStore` is the interface; `ChromaVectorStore` is the default
implementation. When the system outgrows Chroma, a `PgVectorStore` (writing
to the same Postgres as the rest of the app) or `QdrantVectorStore` slot in
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A retrieved document chunk with provenance metadata."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(self, *, embedding: list[float], k: int = 4) -> list[Chunk]: ...

    def delete(self, *, ids: list[str]) -> None: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Default `VectorStore` backed by a persistent ChromaDB collection."""

    def __init__(self, path: str, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store.opened",
            extra={"path": path, "collection": collection_name},
        )

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, *, embedding: list[float], k: int = 4) -> list[Chunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        chunks: list[Chunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity in [0, 1]
            score = 1.0 - float(dist)
            chunks.append(Chunk(id=cid, text=doc, metadata=dict(meta or {}), score=score))
        return chunks

    def delete(self, *, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
