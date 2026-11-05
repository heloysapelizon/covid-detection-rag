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
    """Read every extracted passage JSONL.

    All evidence comes from the same place now: passages extracted from the source
    PDFs by `corpus.py`. There is no hand-written evidence file to keep in sync, so
    a document is added by dropping its PDF in and re-extracting.
    """
    passages = load_passages(passage_dir)
    if not passages:
        raise FileNotFoundError(
            f"No evidence passages in {passage_dir}.\n"
            "Extract them from the source PDFs first:\n"
            "    python -m rag --build-index"
        )
    return passages


def _embed_text(passage: dict) -> str:
    """What actually gets embedded: the passage text, nothing else.

    Retrieval has to match a claim spoken in a transcript against what a passage
    actually says, so only the passage's own words belong in its vector.

    Two things were tried and dropped. A static `claim_addressed` label (what a
    document is nominally about) pulled every passage in a document toward the
    same text regardless of content. The `section` heading fares no better: only
    22% of passages sit under a real heading, the rest inheriting running headers
    like "Vaccines 2023, 11, 1693" or "May 24, 2022" that the heading regex
    mistakes for structure. Measured over six real claims, text alone retrieved
    the right evidence 5/6 against 4/6 with the heading prepended, at the same
    mean score (0.713 vs 0.715).

    The heading is still stored on each passage — it is useful for auditing where
    a hit came from — it simply is not embedded.
    """
    return passage.get("text", "")


def build(passage_dir: Path = PASSAGE_DIR, index_dir: Path = INDEX_DIR,
          model_name: str = EMBED_MODEL, extract: bool = True,
          verbose: bool = True) -> int:
    """Extract passages from the source PDFs, embed them, persist the index.

    Extraction runs first by default so one command covers the whole chain:
    PDF -> passages -> vectors. Pass extract=False to re-embed the committed
    passages without needing the PDFs present.

    Returns the number of passages indexed.
    """
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
        normalize_embeddings=True,     # so a dot product is cosine similarity
        show_progress_bar=False,
    ).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / _VECTORS, vectors)
    # Passage ids only, never the text. The JSONL files are the single source of
    # truth; a second copy here would silently diverge from them whenever a PDF is
    # re-extracted without rebuilding, and would double the stored corpus.
    # Order matters: row i of the vector matrix is passage_ids[i].
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
        # Which scope tier answered the most recent search: a topic name, "general"
        # or "all". Useful for auditing how often routing had to widen.
        self.last_scope: str | None = None
        # (source_file, ordinal) -> row, so neighbour lookup is O(1).
        self._by_position = {
            (p.get("source_file"), p.get("ordinal")): i
            for i, p in enumerate(passages)
        }

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR,
             passage_dir: Path = PASSAGE_DIR) -> "EvidenceIndex":
        """Load the vectors and rejoin them with passage text from the JSONL.

        The index stores ids, not text, so this reads the passages back and orders
        them to match the vector rows. A mismatch means the passages changed since
        the index was built, which is reported rather than silently mis-ranked.
        """
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
        """Return the most similar passages, each with its similarity score.

        With expand > 0 each hit also brings its neighbouring chunks from the same
        document. Small chunks retrieve precisely but can cut a finding in half:
        measured on a research paper, top-3 over 55-word chunks found the paper's
        actual conclusion for 5 of 7 real transcript queries, and 7 of 7 once
        neighbours were included. Neighbours carry the matched chunk's score and
        are marked, so the model sees context without it competing for rank.

        `skip` excludes passage ids already shown — that is what lets a caller ask
        for the next-best evidence after a NO_EVIDENCE verdict.

        `min_similarity` filters which passages are returned; `answered_similarity`
        is the separate, higher bar a scope tier must clear to stop the search
        widening. See rag/config.py for why they are not the same number.

        `relative_cutoff` additionally drops passages scoring below that fraction
        of the best in-scope hit, so a claim with one strong match returns one
        passage rather than being padded to `top_k`. Pass 0 to disable it.
        """
        scores = self._rank(query)
        # The unmodified cosine, kept for reporting: the rank bonus below must
        # influence ORDER only, since a reported similarity is compared against
        # MIN_SIMILARITY, FLAG_MIN_SIMILARITY and the relative cutoff.
        true_sim = scores.copy()
        skip = skip or set()

        if route:
            # Walk the routed scopes narrowest-first and keep the first that
            # answers the claim. Out-of-scope passages are pushed below every
            # threshold rather than removed, so row indices still line up with
            # the vector matrix and neighbour expansion keeps working.
            #
            # Prefer topic-specific passages on a near tie: every tier carries the
            # general documents, so a general passage sharing the claim's
            # vocabulary can otherwise outrank the topic passage that answers it.
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
            # No tier cleared `answered_similarity`, so `scores` holds the
            # broadest tier as initialised above. That is deliberate: the weaker
            # evidence it contains is still returned if it clears
            # `min_similarity`, so raising the widening floor can only add
            # evidence, never withhold it. Previously this case kept whichever
            # tier happened to be last in the loop, which was the same thing only
            # by accident.

        # Relative cutoff, applied after the tier is chosen so the reference
        # point is the best hit actually in scope. A fixed top_k pads a claim
        # that has one good passage with three mediocre ones; this drops them,
        # which matters because every returned passage goes into the prompt and
        # gives the model another chance to flag on topical resemblance.
        # Returning fewer passages is the whole point, so this never raises the
        # count above top_k.
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

        # Add each hit's neighbours from the same source document, in reading
        # order, so the model receives a continuous passage rather than fragments.
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
        """Row index of the chunk `offset` positions from `passage`, if any.

        Neighbours must come from the same source document: the index holds every
        document's passages concatenated, so stepping past a document boundary
        would splice in unrelated text.
        """
        want = passage.get("ordinal")
        if want is None:
            return None
        return self._by_position.get(
            (passage.get("source_file"), want + offset)
        )
