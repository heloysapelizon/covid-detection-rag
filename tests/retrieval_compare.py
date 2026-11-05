"""Compare dense, BM25 and hybrid retrieval on the labelled phrases.

    python tests/retrieval_compare.py            # headline comparison
    python tests/retrieval_compare.py --tune     # also sweep BM25 k1/b/stopwords

Metric is the rank of the best passage containing a labelled phrase, over the flat
index with no routing. That is config-independent: it does not depend on top_k, on
a similarity floor, or on which passages a particular configuration happened to
return, so it is comparable across retrieval methods in a way recall@k is not.
Lower mean rank is better; `top4` counts phrases a top-4 retrieval would surface.

The labels carry a known selection bias (docs/chunking-experiments.md): the phrases
were chosen by reading passages the dense retriever surfaced, so they favour dense.
The bias is worth stating rather than correcting for, because the measured gap is
an order of magnitude and the bias cannot plausibly account for that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import EvidenceIndex  # noqa: E402
from rag.lexical import BM25Index, reciprocal_rank_fusion  # noqa: E402

LABELS = Path(__file__).resolve().parent / "chunk_eval_labels.json"
# Rankings are truncated before fusion: RRF over the whole corpus would let a
# passage ranked 2000th by one method still contribute. 200 is deep enough that
# anything retrievable is included.
FUSE_DEPTH = 200


def labelled_items() -> list[dict]:
    items = json.loads(LABELS.read_text(encoding="utf-8"))["items"]
    return [i for i in items if i.get("relevant_evidence")]


def phrase_ranks(order: np.ndarray, texts: list[str],
                 phrases: list[str]) -> list[int | None]:
    """Rank (1-based) of the first passage containing each phrase."""
    out = []
    for phrase in phrases:
        needle = phrase.lower()
        out.append(
            next((i + 1 for i, row in enumerate(order) if needle in texts[row]),
                 None)
        )
    return out


def summarise(name: str, ranks: list[int]) -> str:
    a = np.array(ranks)
    return (f"{name:14}{len(a):5}{a.mean():9.1f}{np.median(a):8.0f}"
            f"{(a <= 4).sum():7}{(a <= 10).sum():7}{(a <= 20).sum():7}")


HEADER = f"{'method':14}{'n':>5}{'mean':>9}{'median':>8}{'top4':>7}{'top10':>7}{'top20':>7}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true",
                    help="sweep BM25 k1/b and stopword removal")
    args = ap.parse_args()

    index = EvidenceIndex.load()
    texts = [p["text"].lower() for p in index.passages]
    items = labelled_items()
    n = len(index.passages)
    print(f"index  : {n} passages")
    print(f"labels : {len(items)} windows, "
          f"{sum(len(i['relevant_evidence']) for i in items)} phrases\n")

    # Dense rankings are reused across every BM25 configuration.
    dense_orders = [np.argsort(-index._rank(i["window_text"])) for i in items]

    def evaluate(bm25: BM25Index) -> dict[str, list[int]]:
        got: dict[str, list[int]] = {"dense": [], "bm25": [], "hybrid": []}
        for item, dense_order in zip(items, dense_orders):
            bm25_order = bm25.rank(item["window_text"])
            fused = np.argsort(-reciprocal_rank_fusion(
                [dense_order[:FUSE_DEPTH], bm25_order[:FUSE_DEPTH]], n
            ))
            for method, order in (("dense", dense_order),
                                  ("bm25", bm25_order),
                                  ("hybrid", fused)):
                for rank in phrase_ranks(order, texts, item["relevant_evidence"]):
                    if rank is not None:
                        got[method].append(rank)
        return got

    print("=== default BM25 (k1=1.5, b=0.75, stopwords removed) ===")
    print(HEADER)
    results = evaluate(BM25Index(index.passages))
    for method, ranks in results.items():
        print(summarise(method, ranks))

    if args.tune:
        print("\n=== BM25 parameter sweep (bm25 / hybrid mean rank) ===")
        print(f"{'k1':>5}{'b':>6}{'stop':>6}{'bm25_mean':>11}"
              f"{'hyb_mean':>10}{'hyb_top4':>10}")
        for drop in (True, False):
            for k1 in (0.9, 1.2, 1.5, 2.0):
                for b in (0.3, 0.5, 0.75, 0.9):
                    got = evaluate(BM25Index(index.passages, k1=k1, b=b,
                                             drop_stopwords=drop))
                    bm = np.array(got["bm25"])
                    hy = np.array(got["hybrid"])
                    print(f"{k1:5.1f}{b:6.2f}{str(drop):>6}{bm.mean():11.1f}"
                          f"{hy.mean():10.1f}{(hy <= 4).sum():10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
