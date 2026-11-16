"""Build and query the evidence vector index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    EMBED_MODEL,
    EXPAND_NEIGHBOURS,
    INDEX_DIR,
    MIN_SIMILARITY,
    PASSAGE_DIR,
    RELATIVE_CUTOFF,
    TOPIC_RANK_BONUS,
    SCOPE_ANSWERED_SIMILARITY,
    TOP_K,
)
from .corpus import build_passages, load_passages

_VECTORS = "evidence_vectors.npy"
_META = "evidence_meta.json"


def load_evidence(passage_dir: Path = PASSAGE_DIR) -> list[dict]:
    """Read every extracted passage JSONL."""
    passages = load_passages(passage_dir)
    if not passages:
        raise FileNotFoundError(
            f"No evidence passages in {passage_dir}.\n"
            "Extract them from the source PDFs first:\n"
            "    python -m rag --build-index"
        )
    return passages


def _embed_text(passage: dict) -> str:
    """What actually gets embedded: the passage text, nothing else."""
    return passage.get("text", "")


def build(passage_dir: Path = PASSAGE_DIR, index_dir: Path = INDEX_DIR,
          model_name: str = EMBED_MODEL, extract: bool = True,
          verbose: bool = True) -> int:
    """Extract passages from the source PDFs, embed them, persist the index."""
    from sentence_transformers import SentenceTransformer

    if extract:
        if verbose:
            print("Extracting passages from source PDFs:")
        build_passages(out_dir=passage_dir, verbose=verbose)
        if verbose:
            print()

    passages = load_evidence(passage_dir)
    if verbose:
        print(f"Embedding {len(passages)} passage(s) with {model_name}...")

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [_embed_text(p) for p in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / _VECTORS, vectors)
    (index_dir / _META).write_text(
        json.dumps({
            "model": model_name,
            "dim": int(vectors.shape[1]),
            "count": len(passages),
            "passage_ids": [p["id"] for p in passages],
        }, ensure_ascii=False, indent=2),
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
        self._router = None
        self.last_scope: str | None = None
        self._by_position = {
            (p.get("source_file"), p.get("ordinal")): i
            for i, p in enumerate(passages)
        }

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR,
             passage_dir: Path = PASSAGE_DIR) -> "EvidenceIndex":
        """Load the vectors and rejoin them with passage text from the JSONL."""
        vec_path, meta_path = index_dir / _VECTORS, index_dir / _META
        if not vec_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index in {index_dir}. Build it first:\n"
                f"    python -m rag --build-index"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path)

        by_id = {p["id"]: p for p in load_passages(passage_dir)}
        ids = meta["passage_ids"]
        missing = [i for i in ids if i not in by_id]
        if missing or len(ids) != vectors.shape[0]:
            raise RuntimeError(
                f"Index is stale: {len(missing)} of {len(ids)} passage(s) no longer "
                f"exist in {passage_dir}.\nRebuild it:\n"
                f"    python -m rag --build-index"
            )
        return cls(vectors, [by_id[i] for i in ids], meta["model"])

    def _encoder(self):
        """Lazily load the embedding model — only needed to embed queries."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _rank(self, query: str) -> np.ndarray:
        """Similarity of every passage to the query, highest first."""
        q = self._encoder().encode([query], normalize_embeddings=True)
        return (self.vectors @ q[0].T).ravel()

    def router(self):
        """The topic router, built on first use from this index's passages."""
        if self._router is None:
            from .router import TopicRouter
            self._router = TopicRouter.build(self.passages,
                                             model=self._encoder())
        return self._router

    def search(self, query: str, top_k: int = TOP_K,
               min_similarity: float = MIN_SIMILARITY,
               answered_similarity: float = SCOPE_ANSWERED_SIMILARITY,
               relative_cutoff: float = RELATIVE_CUTOFF,
               topic_bonus: float = TOPIC_RANK_BONUS,
               expand: int = EXPAND_NEIGHBOURS,
               skip: set[str] | None = None,
               route: bool = True) -> list[dict]:
        """Return the most similar passages, each with its similarity score."""
        scores = self._rank(query)
        true_sim = scores.copy()
        skip = skip or set()

        if route:
            base = scores
            if topic_bonus:
                topical = np.array(
                    [p.get("topic") not in (None, "general") for p in self.passages],
                    dtype=bool,
                )
                base = base + topic_bonus * topical

            tiers = self.router().scopes(query, self.passages)
            scores = np.where(tiers[-1][1], base, -1.0)
            self.last_scope = tiers[-1][0]
            for label, mask in tiers:
                candidate = np.where(mask, base, -1.0)
                usable = [
                    i for i in np.argsort(-candidate)[:top_k]
                    if candidate[i] >= answered_similarity
                    and self.passages[i]["id"] not in skip
                ]
                if usable:
                    scores = candidate
                    self.last_scope = label
                    break

        floor = min_similarity
        if relative_cutoff and len(scores):
            best = float(scores.max())
            if best > 0:
                floor = max(floor, relative_cutoff * best)

        chosen: list[dict] = []
        seen: set[int] = set()
        for i in np.argsort(-scores):
            if len(chosen) >= top_k:
                break
            if scores[i] < floor:
                break
            if self.passages[i]["id"] in skip or i in seen:
                continue
            seen.add(int(i))
            chosen.append({**self.passages[i],
                           "similarity": round(float(true_sim[i]), 4),
                           "matched": True})

        if not expand:
            return chosen

        out: list[dict] = []
        for hit in chosen:
            group = [hit]
            for offset in range(-expand, expand + 1):
                if offset == 0:
                    continue
                j = self._neighbour(hit, offset)
                if j is None or j in seen:
                    continue
                seen.add(j)
                group.append({**self.passages[j],
                              "similarity": hit["similarity"],
                              "matched": False})
            group.sort(key=lambda p: p.get("ordinal", 0))
            out.extend(group)
        return out

    def _neighbour(self, passage: dict, offset: int) -> int | None:
        """Row index of the chunk `offset` positions from `passage`, if any."""
        want = passage.get("ordinal")
        if want is None:
            return None
        return self._by_position.get(
            (passage.get("source_file"), want + offset)
        )
