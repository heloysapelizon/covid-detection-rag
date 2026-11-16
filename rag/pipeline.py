"""Orchestration: segment a transcript, retrieve evidence, adjudicate, report."""
from __future__ import annotations

import textwrap

from . import config
from .config import CLAIM_STRIDE_WORDS, CLAIM_WINDOW_WORDS, MIN_SIMILARITY, TOP_K
from .index import EvidenceIndex
from .llm import adjudicate, adjudicate_parametric, build_query
from .manifest import document_sources
from .websearch import (
    is_authority,
    rank_by_authority,
    search as web_search,
    should_search,
)

_WEB_VERDICTS = {"FAKE_NEWS": "FAKE_NEWS_WEB",
                 "NOT_FAKE_NEWS": "NOT_FAKE_NEWS_WEB"}


def segment_claims(
    transcript: str,
    window: int = CLAIM_WINDOW_WORDS,
    stride: int = CLAIM_STRIDE_WORDS,
) -> list[str]:
    """Split a transcript into overlapping word windows."""
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
    web_fetch_fn=None,
    published: str | None = None,
) -> list[dict]:
    """Fact-check one transcript. Returns one result per claim window."""
    from .temporal import parse_date
    claim_date = parse_date(published)
    claims = segment_claims(transcript)
    sources = document_sources()
    if verbose:
        print(f"{len(claims)} claim window(s) to check...")

    results = []
    for n, claim in enumerate(claims, 1):
        passages = index.search(claim, top_k=top_k, min_similarity=min_similarity)
        if config.TIME_CHECK:
            passages = [{**p, "updated": (sources.get(p.get("source_file", ""))
                                          or {}).get("updated", "")}
                        for p in passages]
        verdict = adjudicate(claim, passages, claim_date=claim_date)

        origin = "corpus"
        web_why = ""
        if web_fetch_fn is not None:
            go, web_why = should_search(
                claim, passages,
                verdict=verdict["verdict"],
                addresses=verdict.get("addresses"),
                stance=verdict.get("stance"),
            )
            if go:
                query = build_query(claim)
                web = rank_by_authority(
                    web_search(query, max_results=top_k, fetch_fn=web_fetch_fn)
                )
                web_why = f'{web_why}; query="{query}"'
                if web:
                    passages = web
                    origin = "web"
                    verdict = adjudicate(claim, passages, claim_date=claim_date)
                    verdict["verdict"] = _WEB_VERDICTS.get(verdict["verdict"],
                                                           verdict["verdict"])
                if verbose:
                    print(f"  [{n}/{len(claims)}] web fallback: "
                          f"{len(web)} result(s) for \"{query}\"")

        cited = next(
            (p for p in passages if p["id"] == verdict.get("evidence_id")),
            passages[0] if passages else None,
        )

        doc = (sources.get(cited.get("source_file", ""))
               if cited and origin == "corpus" else None)

        results.append({
            "window": n,
            "claim_text": claim,
            "verdict": verdict["verdict"],
            "addresses": verdict.get("addresses", ""),
            "stance": verdict.get("stance", ""),
            "flagged_phrase": verdict.get("claim", ""),
            "reason": verdict.get("reason", ""),
            "evidence_id": cited["id"] if cited else "",
            "debunk_text": cited["text"] if cited else "",
            "evidence_source": (
                f"{(doc or {}).get('publisher') or cited.get('publisher', '')}"
                f" — {(doc or {}).get('title') or cited.get('source_file', '')}"
                f" ({(doc or {}).get('updated', '')})" if cited else ""
            ),
            "evidence_url": (
                (doc or {}).get("source_url", "") or cited.get("source_url", "")
                if cited else ""
            ),
            "similarity": cited.get("similarity") if cited else None,
            "retrieved_ids": [p["id"] for p in passages],
            "evidence_origin": origin,
            "retrieved_at": cited.get("retrieved_at", "") if cited else "",
            "web_decision": web_why,
        })

        if verbose:
            mark = {"FAKE_NEWS": "FLAG", "ERROR": "err "}.get(verdict["verdict"], "ok  ")
            print(f"  [{n}/{len(claims)}] {mark} {verdict['verdict']}")

    return overlap_veto(results) if config.OVERLAP_VETO else results


_FLAGS = {"FAKE_NEWS", "FAKE_NEWS_WEB"}
_CLEARS = {"NOT_FAKE_NEWS", "NOT_FAKE_NEWS_WEB"}


def _grams(text: str, k: int = 6) -> set[str]:
    import re
    w = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def overlap_veto(results: list[dict]) -> list[dict]:
    """Withdraw a flag that an overlapping window explicitly cleared."""
    shingles = [_grams(r.get("claim_text", "")) for r in results]
    out = []
    for i, r in enumerate(results):
        r = {**r, "verdict_raw": r.get("verdict_raw", r["verdict"])}
        if r["verdict"] in _FLAGS:
            mine = _grams(r.get("flagged_phrase", ""))
            clearing = [results[j]["window"] for j in range(len(results))
                        if j != i and mine & shingles[j]
                        and results[j]["verdict"] in _CLEARS]
            if clearing:
                r["verdict"] = ("NOT_FAKE_NEWS_WEB" if r["verdict"] == "FAKE_NEWS_WEB"
                                else "NOT_FAKE_NEWS")
                r["reason"] = (f"Overridden (overlap veto): window(s) {clearing} "
                               f"contain the same words and cleared them. "
                               f"{r.get('reason', '')}")
        out.append(r)
    return out


def print_report(results: list[dict], meta: dict | None = None) -> None:
    """Print the flagged claims with the evidence that refutes each."""
    flagged = [r for r in results if r["verdict"] == "FAKE_NEWS"]
    flagged_web = [r for r in results if r["verdict"] == "FAKE_NEWS_WEB"]
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

    web = [r for r in results if r.get("evidence_origin") == "web"]
    if web:
        print(f"\n{len(web)} window(s) answered on WEB evidence, not the "
              f"reference corpus:")
        for r in web:
            domain = r["evidence_url"].split("//")[-1].split("/")[0] or "?"
            print(f"  window {r['window']}: {r['verdict']}  [{domain}]"
                  f"  retrieved {r.get('retrieved_at', '?')}")
        print("  Web sources are not licence-vetted, DOI-resolved or "
              "authorship-screened.")
        print("  Authority domains rank first but no result is discarded, so "
              "evidence here")
        print("  may be a blog or forum. Treat as leads and exclude from corpus "
              "metrics.")

    if errors:
        print(f"\n{len(errors)} window(s) errored — first: {errors[0]['reason'][:120]}")

    if not flagged and not flagged_web:
        print("\nNo claims flagged as fake news.")
        return

    if flagged:
        print(f"\n{len(flagged)} claim(s) flagged as FAKE NEWS:\n")
        _print_flags(flagged)

    if flagged_web:
        print("\n" + "=" * 78)
        print(f"{len(flagged_web)} claim(s) flagged on WEB evidence — reported "
              "separately.")
        print("The reference corpus had no evidence for these. Web sources are "
              "not")
        print("licence-vetted, DOI-resolved or authorship-screened; treat as "
              "leads,")
        print("not as results, and exclude from corpus precision figures.")
        print("=" * 78 + "\n")
        _print_flags(flagged_web)


def _print_flags(flagged: list[dict]) -> None:
    """Print one block per flagged claim, with the evidence that refutes it."""
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
        if r.get("evidence_origin") == "web":
            tier = ("recognised authority domain"
                    if is_authority(r["evidence_url"]) else "NOT an authority "
                    "domain — weigh accordingly")
            print(f"          [web evidence, {tier}]")
            print(f"          [retrieved {r.get('retrieved_at', 'unknown')}]")
        print("  EVIDENCE SAYS:")
        for line in textwrap.wrap(r["debunk_text"], 70):
            print(f"    {line}")
        print()


def check_transcript_parametric(transcript: str, verbose: bool = False,
                                published: str | None = None) -> list[dict]:
    """Parametric-only condition: the same claim windows, adjudicated without retrieval
    (rag.llm.adjudicate_parametric)."""
    from .temporal import parse_date
    claim_date = parse_date(published)
    claims = segment_claims(transcript)
    results = []
    for n, claim in enumerate(claims, 1):
        verdict = adjudicate_parametric(claim, claim_date=claim_date)
        results.append({
            "window": n,
            "claim_text": claim,
            "verdict": verdict["verdict"],
            "addresses": "",
            "stance": verdict.get("stance", ""),
            "flagged_phrase": verdict.get("claim", ""),
            "reason": verdict.get("reason", ""),
            "evidence_id": "",
            "debunk_text": "",
            "evidence_source": "",
            "evidence_url": "",
            "similarity": None,
            "retrieved_ids": [],
            "evidence_origin": "none",
            "retrieved_at": "",
            "web_decision": "",
        })
        if verbose:
            mark = {"FAKE_NEWS": "FLAG", "ERROR": "err "}.get(verdict["verdict"], "ok  ")
            print(f"  [{n}/{len(claims)}] {mark} {verdict['verdict']}")
    return overlap_veto(results) if config.OVERLAP_VETO else results
