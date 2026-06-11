"""
lexical_index.py
----------------
Lexical (sparse) retrieval seam, the keyword-matching counterpart to the dense
`VectorStore`. Dense embeddings capture meaning but blur exact tokens — SOP
codes, part numbers, app names, acronyms. BM25 matches those literally, so
fusing the two (see `retrieval.DocumentStore`) covers both failure modes.

`LexicalIndex` is the interface; `BM25Index` is the default implementation — a
compact, dependency-free Okapi BM25 held in memory. The corpus is small and
rebuilt from the `VectorStore` at startup, so a separate persistence format
would be needless complexity. When it outgrows memory, a `TantivyIndex` or a
Postgres full-text index slots in behind the same Protocol.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Protocol

from vector_store import Chunk


logger = logging.getLogger(__name__)


# Okapi BM25 free parameters. k1 controls term-frequency saturation, b controls
# length normalisation — these are the long-standing defaults.
_K1 = 1.5
_B = 0.75

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase unicode word tokens. Unicode-aware so German/French/Italian
    accented terms tokenise the same way the multilingual embedder sees them."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class LexicalIndex(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int = 4) -> list[Chunk]: ...


class BM25Index:
    """In-memory Okapi BM25 over a fixed chunk corpus.

    `index()` rebuilds the whole index (cheap for this corpus size); `search()`
    returns the top-k chunks by BM25 score, dropping non-matches (score 0).
    """

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter] = []        # term frequencies per chunk
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._tf = [Counter(toks) for toks in tokenized]
        self._doc_len = [len(toks) for toks in tokenized]
        n = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0

        df: Counter = Counter()
        for toks in tokenized:
            df.update(set(toks))
        # Smoothed idf (the "+1" keeps it positive even for very common terms).
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        logger.info("lexical.indexed", extra={"chunks": n, "terms": len(self._idf)})

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if not self._chunks:
            return []
        terms = _tokenize(query)
        if not terms:
            return []

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._doc_len[i]
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                score += idf * (freq * (self._k1 + 1)) / denom
            if score > 0.0:
                scored.append((score, i))

        scored.sort(key=lambda s: s[0], reverse=True)
        results: list[Chunk] = []
        for score, i in scored[:k]:
            c = self._chunks[i]
            results.append(Chunk(id=c.id, text=c.text, metadata=dict(c.metadata), score=score))
        return results
