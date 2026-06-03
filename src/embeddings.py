"""
embeddings.py
-------------
Embedding provider seam.

`EmbeddingProvider` is the interface every consumer talks to.
`FastEmbedProvider` is the default implementation (local, free,
multilingual, ONNX-backed — no PyTorch dependency, so it works on
Python 3.13 + Intel macOS where PyTorch wheels are not available).

Swapping to a hosted API (Cohere, Voyage, OpenAI) later is one new class.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def name(self) -> str: ...


class FastEmbedProvider:
    """Default `EmbeddingProvider` backed by a local ONNX model via fastembed."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

        # Probe the output dimension with a tiny embed so we don't have to
        # rely on version-specific metadata APIs.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dimension = int(np.asarray(probe).shape[0])

        logger.info(
            "embeddings.loaded",
            extra={"model": model_name, "dimension": self._dimension},
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]
        return [_normalize(v).tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm
