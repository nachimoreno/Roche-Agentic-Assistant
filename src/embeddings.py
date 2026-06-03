"""
embeddings.py
-------------
Embedding provider seam.

`EmbeddingProvider` is the interface every consumer talks to.
`SentenceTransformersProvider` is the default implementation (local, free,
multilingual). Swapping to a hosted API (Cohere, Voyage, OpenAI) later is
one new class.
"""

from __future__ import annotations

import logging
from typing import Protocol


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def name(self) -> str: ...


class SentenceTransformersProvider:
    """Default `EmbeddingProvider` backed by a local sentence-transformers model."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "embeddings.loaded",
            extra={"model": model_name, "dimension": self._dimension},
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name
