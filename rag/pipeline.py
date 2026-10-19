"""Orchestration: segment a transcript, retrieve evidence, adjudicate, report.

For each claim window: retrieve the nearest evidence passages, ask the local model
whether the claim asserts something they contradict, and for anything flagged,
attach the passage text that debunks it.
"""
from __future__ import annotations

import textwrap

from .config import CLAIM_STRIDE_WORDS, CLAIM_WINDOW_WORDS, MIN_SIMILARITY, TOP_K
from .index import EvidenceIndex
from .llm import adjudicate


def segment_claims(
    transcript: str,
    window: int = CLAIM_WINDOW_WORDS,
    stride: int = CLAIM_STRIDE_WORDS,
) -> list[str]:
    """Split a transcript into overlapping word windows.

    Whisper output is largely unpunctuated, so sentence splitting is unreliable.
    Overlapping windows mean a claim spanning a boundary still appears whole in at
    least one window.
    """
    words = transcript.split()
    if len(words) <= window:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[i:i + window])
        for i in range(0, max(1, len(words) - window + stride), stride)
    ]


def check_transcript(
    transcript: str,
    index: EvidenceIndex,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    verbose: bool = True,
) -> list[dict]:
    """Fact-check one transcript. Returns one result per claim window.

    Each result carries the window text, what was retrieved, the model's verdict,
    and — when flagged — the evidence passage that debunks it.
    """
    claims = segment_claims(transcript)
    if verbose:
        print(f"{len(claims)} claim window(s) to check...")

    results = []
    for n, claim in enumerate(claims, 1):
        passages = index.search(claim, top_k=top_k, min_similarity=min_similarity)
        verdict = adjudicate(claim, passages)

        # Resolve the passage the model cited, so the debunking text travels with
        # the flag. Fall back to the best-scoring passage if the id is unusable.
        cited = next(
            (p for p in passages if p["id"] == verdict.get("evidence_id")),
            passages[0] if passages else None,
        )

        results.append({
            "window": n,
            "claim_text": claim,
            "verdict": verdict["verdict"],
            "flagged_phrase": verdict.get("claim", ""),
            "reason": verdict.get("reason", ""),
            "evidence_id": cited["id"] if cited else "",
            "debunk_text": cited["text"] if cited else "",
            "evidence_source": (
                f"{cited.get('publisher', '')} — {cited.get('source', '')}"
                f" ({cited.get('updated', '')})" if cited else ""
            ),
            "evidence_url": cited.get("source_url", "") if cited else "",
            "similarity": cited.get("similarity") if cited else None,
            "retrieved_ids": [p["id"] for p in passages],
        })

        if verbose:
            mark = {"FAKE_NEWS": "FLAG", "ERROR": "err "}.get(verdict["verdict"], "ok  ")
            print(f"  [{n}/{len(claims)}] {mark} {verdict['verdict']}")

    return results


def print_report(results: list[dict], meta: dict | None = None) -> None:
    """Print the flagged claims with the evidence that refutes each."""
    flagged = [r for r in results if r["verdict"] == "FAKE_NEWS"]
    errors = [r for r in results if r["verdict"] == "ERROR"]

    print("\n" + "=" * 78)
    if meta:
        print(f"{meta.get('title', '')[:76]}")
        print(f"{meta.get('channel', '')}  |  {meta.get('url', '')}")
    print("=" * 78)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if errors:
        print(f"\n{len(errors)} window(s) errored — first: {errors[0]['reason'][:120]}")

    if not flagged:
        print("\nNo claims flagged as fake news.")
        return

    print(f"\n{len(flagged)} claim(s) flagged as FAKE NEWS:\n")
    for i, r in enumerate(flagged, 1):
        print(f"--- FLAG {i} " + "-" * 62)
        phrase = r["flagged_phrase"] or "(model returned no phrase)"
        print("  FALSE PHRASE:")
        for line in textwrap.wrap(phrase, 70):
            print(f"    {line}")
        if r["reason"]:
            print("\n  WHY:")
            for line in textwrap.wrap(r["reason"], 70):
                print(f"    {line}")
        print(f"\n  DEBUNKED BY: {r['evidence_id']}"
              f"   (similarity {r['similarity']})")
        print(f"  SOURCE: {r['evidence_source']}")
        if r["evidence_url"]:
            print(f"          {r['evidence_url']}")
        print("  EVIDENCE SAYS:")
        for line in textwrap.wrap(r["debunk_text"], 70):
            print(f"    {line}")
        print()
