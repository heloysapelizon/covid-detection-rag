"""Route a claim to the topics worth searching, and decide which are general.

Retrieval over the whole corpus lets an unrelated document outrank the right one:
once the ICH shedding paper contributed 74 of 90 passages, a claim about vaccine
magnetism retrieved ICH study-design text instead of the CDC's answer. Narrowing
the scope before ranking removes that competition.

Two decisions, both derived from the corpus rather than declared:

  Routing — each topic's `addresses` text in sources.json is embedded once; a
  claim is routed to every topic it resembles above a threshold. Soft, so a claim
  touching two narratives searches both.

  Generality — a document whose passages spread across many topics is general and
  always stays in scope. Measured as Shannon entropy over the topic its passages
  most resemble. Observed: the CDC myths sheet 1.88 across 5 topics (8 distinct
  myths), the ICH guidance 1.31, the single-subject infertility review 0.79.
  Nothing is hardcoded as general; a document earns the flag by covering ground.

Depends on sentence-transformers and numpy; no network, no LLM.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .config import (
    EMBED_MODEL,
    EVIDENCE_DIR,
    GENERAL_ENTROPY_MIN,
    GENERAL_TOP_SHARE_MAX,
    ROUTE_FLOOR,
    ROUTE_GAMMA,
    ROUTE_TOP_N,
)


def topic_descriptions(evidence_dir: Path = EVIDENCE_DIR) -> dict[str, str]:
    """Topic name -> the `addresses` text that defines what it covers."""
    manifest = json.loads(
        (evidence_dir / "sources.json").read_text(encoding="utf-8")
    )
    return {
        name: spec.get("addresses", name)
        for name, spec in manifest["topics"].items()
    }


def general_files(evidence_dir: Path = EVIDENCE_DIR) -> set[str]:
    """Filenames declared general-purpose — searched for every claim."""
    path = evidence_dir / "sources.json"
    if not path.exists():
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {f["filename"] for f in manifest.get("general_files", [])}


def topic_entropy(labels: list[str]) -> float:
    """Shannon entropy (bits) of a topic assignment — how spread out it is.

    0 means every passage resembles one topic; higher means the document ranges
    across several.
    """
    if not labels:
        return 0.0
    n = len(labels)
    return -sum(
        (c / n) * math.log2(c / n) for c in Counter(labels).values()
    )


def classify_documents(
    passages: list[dict],
    model=None,
    entropy_min: float = GENERAL_ENTROPY_MIN,
    top_share_max: float = GENERAL_TOP_SHARE_MAX,
) -> dict[str, dict]:
    """Decide, per source document, whether it is general or topic-specific.

    Each passage is matched to the topic description it most resembles. A document
    is general when its passages spread widely — high entropy AND no single topic
    dominating — which is what distinguishes a multi-myth FAQ from a paper that
    merely mentions neighbouring subjects.

    Returns {source_file: {general, entropy, top_topic, top_share, counts}}.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)

    names = list(topic_descriptions())
    topic_vecs = model.encode(
        [topic_descriptions()[n] for n in names], normalize_embeddings=True
    )
    passage_vecs = model.encode(
        [p["text"] for p in passages], normalize_embeddings=True
    )
    nearest = (passage_vecs @ topic_vecs.T).argmax(axis=1)

    per_doc: dict[str, list[str]] = {}
    for passage, idx in zip(passages, nearest):
        per_doc.setdefault(passage["source_file"], []).append(names[idx])

    out: dict[str, dict] = {}
    for doc, labels in per_doc.items():
        counts = Counter(labels)
        top_topic, top_count = counts.most_common(1)[0]
        top_share = top_count / len(labels)
        entropy = topic_entropy(labels)
        out[doc] = {
            "general": entropy >= entropy_min and top_share <= top_share_max,
            "entropy": round(entropy, 3),
            "top_topic": top_topic,
            "top_share": round(top_share, 3),
            "counts": dict(counts),
        }
    return out


class TopicRouter:
    """Routes a claim to the topics whose evidence is worth searching."""

    def __init__(self, descriptions: dict[str, str], vectors: np.ndarray,
                 general_docs: set[str], model=None):
        self.names = list(descriptions)
        self.descriptions = descriptions
        self.vectors = vectors
        self.general_docs = general_docs
        self._model = model

    @classmethod
    def build(cls, passages: list[dict], model=None,
              evidence_dir: Path = EVIDENCE_DIR) -> "TopicRouter":
        """Embed the topic descriptions and read which documents are general.

        Generality is *declared* in the manifest (`general_files`), not inferred
        from passage statistics. A general document is searched for every claim
        whatever it routes to, so its breadth has to be a property of what the
        document is for — an authority myth sheet answers many narratives by
        design. `classify_documents` remains available as a diagnostic for
        spotting a document whose content has drifted from its declared scope.
        """
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBED_MODEL)

        descriptions = topic_descriptions(evidence_dir)
        vectors = model.encode(
            [descriptions[n] for n in descriptions], normalize_embeddings=True
        )
        general = general_files(evidence_dir)
        return cls(descriptions, vectors, general, model=model)

    def route(self, claim: str, min_similarity: float = ROUTE_FLOOR,
              top_n: int = ROUTE_TOP_N) -> list[tuple[str, float]]:
        """Candidate topics for this claim, closest first.

        An ordered list, not a single choice: the caller tries the first scope and
        moves to the next topic if it yields nothing usable. Topics below
        `min_similarity` are not candidates. An empty list means no topic resembles
        the claim, which the caller must read as "search everything" rather than
        "no evidence" — an obliquely phrased claim is a routing miss, not an
        absence of evidence.
        """
        q = self._encoder().encode([claim], normalize_embeddings=True)
        scores = (self.vectors @ q[0].T).ravel()
        order = np.argsort(-scores)[:top_n]
        return [
            (self.names[i], round(float(scores[i]), 4))
            for i in order
            if scores[i] >= min_similarity
        ]

    def scopes(self, claim: str, passages: list[dict],
               gamma: float = ROUTE_GAMMA, **kwargs) -> list[tuple[str, np.ndarray]]:
        """Scopes to try for this claim, narrowest first, broadest last.

        Returns (label, mask) pairs. The caller searches the first scope and falls
        through to the next when the evidence it finds is not usable — the ordered
        equivalent of the confidence gate in Sun et al. (arXiv:2609.29282), where a
        route is accepted only when confident and otherwise widened.

        Tiers:
          "<topic>"         one per candidate at or above gamma, in descending
            similarity, each also including the general documents — a general
            document is in scope for every claim, so every tier carries them
          "general"         general documents alone
          "all"             the whole corpus, always last

        Widening is never skipped. A mis-route hands the model off-topic evidence
        and risks calling a true statement false; searching too widely only costs
        rank competition.
        """
        routed = self.route(claim, **kwargs)
        general = np.array(
            [p.get("source_file") in self.general_docs for p in passages],
            dtype=bool,
        )

        # Only topics at or above gamma lead with their own scope. Each carries
        # the general documents: a general document is in scope for every claim
        # by definition, so no tier withholds them.
        confident = [(n, sc) for n, sc in routed if sc >= gamma]

        tiers: list[tuple[str, np.ndarray]] = []
        for name, _score in confident:
            in_topic = np.array(
                [p.get("topic") == name for p in passages], dtype=bool
            )
            mask = in_topic | general
            if mask.any():
                tiers.append((name, mask))

        if general.any():
            tiers.append(("general", general))
        tiers.append(("all", np.ones(len(passages), dtype=bool)))
        return tiers

    def scope_mask(self, claim: str, passages: list[dict],
                   **kwargs) -> np.ndarray:
        """The first scope for this claim — kept for callers wanting one mask.

        `scopes()` is the fuller interface; this is its first tier.
        """
        tiers = self.scopes(claim, passages, **kwargs)
        return tiers[0][1]

    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model
