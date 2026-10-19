"""Build and query the evidence vector index.

Evidence is embedded once and persisted to data/index/, so a fact-check run loads
vectors from disk instead of re-embedding the source every time. The corpus here is
small (tens of passages), so similarity is exact cosine over a dense matrix — no
approximate-NN library needed, and nothing to go stale.

Depends on sentence-transformers and numpy; no network, no LLM server.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EMBED_MODEL, EVIDENCE_DIR, INDEX_DIR, MIN_SIMILARITY, TOP_K

_VECTORS = "evidence_vectors.npy"
_META = "evidence_meta.json"


def load_evidence(evidence_dir: Path = EVIDENCE_DIR) -> list[dict]:
    """Read every evidence JSON, flattening passages and tagging provenance."""
    passages: list[dict] = []
    for path in sorted(evidence_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for p in doc.get("passages", []):
            passages.append({
                **p,
                "source": doc.get("source", path.stem),
                "publisher": doc.get("publisher", ""),
                "source_url": doc.get("url", ""),
                "updated": doc.get("updated", ""),
                "source_file": path.name,
            })
    if not passages:
        raise FileNotFoundError(f"No evidence passages found in {evidence_dir}")
    return passages


def _embed_text(passage: dict) -> str:
    """What actually gets embedded.

    The question and topic are included alongside the answer text: a transcript
    claim ("the vaccine makes you magnetic") is lexically closer to the myth being
    addressed than to the CDC's denial of it.
    """
    return " ".join(filter(None, [
        passage.get("claim_addressed", ""),
        passage.get("topic", ""),
        passage.get("text", ""),
    ]))


def build(evidence_dir: Path = EVIDENCE_DIR, index_dir: Path = INDEX_DIR,
          model_name: str = EMBED_MODEL, verbose: bool = True) -> int:
    """Embed every evidence passage and persist vectors + metadata.

    Returns the number of passages indexed.
    """
    from sentence_transformers import SentenceTransformer

    passages = load_evidence(evidence_dir)
    if verbose:
        print(f"Embedding {len(passages)} passage(s) with {model_name}...")

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [_embed_text(p) for p in passages],
        normalize_embeddings=True,     # so a dot product is cosine similarity
        show_progress_bar=False,
    ).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / _VECTORS, vectors)
    (index_dir / _META).write_text(
        json.dumps({"model": model_name, "passages": passages},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved index: {index_dir / _VECTORS} ({vectors.shape})")
    return len(passages)


class EvidenceIndex:
    """A loaded evidence index, ready to retrieve against."""

    def __init__(self, vectors: np.ndarray, passages: list[dict], model_name: str):
        self.vectors = vectors
        self.passages = passages
        self.model_name = model_name
        self._model = None

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "EvidenceIndex":
        """Load a prebuilt index, with a clear error if it is missing."""
        vec_path, meta_path = index_dir / _VECTORS, index_dir / _META
        if not vec_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index in {index_dir}. Build it first:\n"
                f"    python -m rag --build-index"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(np.load(vec_path), meta["passages"], meta["model"])

    def _encoder(self):
        """Lazily load the embedding model — only needed to embed queries."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, top_k: int = TOP_K,
               min_similarity: float = MIN_SIMILARITY) -> list[dict]:
        """Return the most similar passages, each with its similarity score."""
        q = self._encoder().encode([query], normalize_embeddings=True)
        scores = (self.vectors @ q[0].T).ravel()
        order = np.argsort(-scores)[:top_k]
        return [
            {**self.passages[i], "similarity": round(float(scores[i]), 4)}
            for i in order
            if scores[i] >= min_similarity
        ]
