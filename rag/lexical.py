"""BM25 lexical retrieval, for comparison against and fusion with the dense index."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

BM25_K1 = 1.5
BM25_B = 0.75

_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves
know think really going lot say said thing things way kind sort like get got make
""".split())

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{1,}")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens, optionally without stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25Index:
    """An in-memory BM25 index over passage text."""

    def __init__(self, passages: list[dict], k1: float = BM25_K1,
                 b: float = BM25_B, drop_stopwords: bool = True):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.drop_stopwords = drop_stopwords

        docs = [tokenize(p["text"], drop_stopwords) for p in passages]
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0

        postings: dict[str, list[tuple[int, int]]] = {}
        for row, doc in enumerate(docs):
            for term, freq in Counter(doc).items():
                postings.setdefault(term, []).append((row, freq))

        self.postings = {
            term: (np.array([r for r, _ in plist], dtype=np.int32),
                   np.array([f for _, f in plist], dtype=np.float32))
            for term, plist in postings.items()
        }
        self.idf = {
            term: math.log(1.0 + (self.n - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, (rows, _) in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage against the query."""
        out = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self.lengths / self.avg_length)
        for term in tokenize(query, self.drop_stopwords):
            entry = self.postings.get(term)
            if entry is None:
                continue
            rows, freqs = entry
            out[rows] += self.idf[term] * freqs * (self.k1 + 1.0) / (
                freqs + norm[rows]
            )
        return out

    def rank(self, query: str, top_k: int | None = None) -> np.ndarray:
        """Row indices, best first."""
        order = np.argsort(-self.scores(query))
        return order if top_k is None else order[:top_k]


def reciprocal_rank_fusion(rankings: list[np.ndarray], n: int,
                           k: int = 60) -> np.ndarray:
    """Fuse several rankings into one score per row."""
    fused = np.zeros(n, dtype=np.float32)
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            fused[row] += 1.0 / (k + rank)
    return fused
