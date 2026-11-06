"""Orchestration: segment a transcript, retrieve evidence, adjudicate, report.

For each claim window: retrieve the nearest evidence passages, ask the local model
whether the claim asserts something they contradict, and for anything flagged,
attach the passage text that debunks it.
"""
from __future__ import annotations

import textwrap

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

# Verdicts reached on web evidence rather than corpus evidence. Kept distinct so
# they can never be summed into the corpus figures: every corpus passage is
# licence-vetted, DOI-resolved and authorship-screened, and a web page is none of
# those. See docs/adjudication-experiments.md.
_WEB_VERDICTS = {"FAKE_NEWS": "FAKE_NEWS_WEB",
                 "NOT_FAKE_NEWS": "NOT_FAKE_NEWS_WEB"}


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
    web_fetch_fn=None,
) -> list[dict]:
    """Fact-check one transcript. Returns one result per claim window.

    Each result carries the window text, what was retrieved, the model's verdict,
    and — when flagged — the evidence passage that debunks it.
    """
    claims = segment_claims(transcript)
    sources = document_sources()
    if verbose:
        print(f"{len(claims)} claim window(s) to check...")

    results = []
    for n, claim in enumerate(claims, 1):
        passages = index.search(claim, top_k=top_k, min_similarity=min_similarity)
        verdict = adjudicate(claim, passages)

        # Web fallback, decided on the ADJUDICATED verdict rather than on an
        # empty retrieval: the corpus almost always returns something, so
        # "retrieval found nothing" never fires, while the relevance gate does
        # correctly report that nothing addresses the claim. Adjudication has
        # already run, so this costs no extra model call.
        #
        # Inert without a fetch function, so a run with no web access behaves
        # exactly as before.
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
                # The claim window cannot be used as a search query directly: a
                # 140-word unpunctuated passage returns nothing. build_query
                # reduces it to the query a fact-checker would type.
                query = build_query(claim)
                web = rank_by_authority(
                    web_search(query, max_results=top_k, fetch_fn=web_fetch_fn)
                )
                web_why = f'{web_why}; query="{query}"'
                if web:
                    # Re-adjudicate against the web evidence. The corpus verdict
                    # is discarded: it was NO_EVIDENCE by construction, which is
                    # what sent us here.
                    passages = web
                    origin = "web"
                    verdict = adjudicate(claim, passages)
                    # Relabel so a web-sourced verdict is never mistaken for a
                    # corpus one. NO_EVIDENCE and ERROR keep their names: they
                    # assert nothing, so there is no provenance to qualify.
                    verdict["verdict"] = _WEB_VERDICTS.get(verdict["verdict"],
                                                           verdict["verdict"])
                if verbose:
                    print(f"  [{n}/{len(claims)}] web fallback: "
                          f"{len(web)} result(s) for \"{query}\"")

        # Resolve the passage the model cited, so the debunking text travels with
        # the flag. Fall back to the best-scoring passage if the id is unusable.
        cited = next(
            (p for p in passages if p["id"] == verdict.get("evidence_id")),
            passages[0] if passages else None,
        )

        # Citation details live once in sources.json, keyed by source_file, rather
        # than on every passage. Resolved here so a flag still cites the publisher
        # of record. A web passage has no manifest entry — its source_file is a
        # URL — so it carries its own publisher and url instead.
        doc = (sources.get(cited.get("source_file", ""))
               if cited and origin == "corpus" else None)

        results.append({
            "window": n,
            "claim_text": claim,
            "verdict": verdict["verdict"],
            # Whether the model judged the evidence to address the claim at all.
            # Recorded because it is the field that distinguishes "the evidence
            # refutes this" from "the evidence is silent on this" — without it a
            # NO_EVIDENCE row cannot be audited after the fact.
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
            # Provenance of the evidence behind this verdict: "corpus" or "web".
            # Reporting and metrics must split on this, never pool it.
            "evidence_origin": origin,
            "retrieved_at": cited.get("retrieved_at", "") if cited else "",
            "web_decision": web_why,
        })

        if verbose:
            mark = {"FAKE_NEWS": "FLAG", "ERROR": "err "}.get(verdict["verdict"], "ok  ")
            print(f"  [{n}/{len(claims)}] {mark} {verdict['verdict']}")

    return results


def print_report(results: list[dict], meta: dict | None = None) -> None:
    """Print the flagged claims with the evidence that refutes each."""
    flagged = [r for r in results if r["verdict"] == "FAKE_NEWS"]
    # Reported separately, never pooled with the corpus flags: the evidence
    # behind them has not been through the corpus's licence, DOI and authorship
    # screening (docs/corpus-inclusion.md).
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

    # Every web-sourced verdict is named here, not only the flags. A
    # NOT_FAKE_NEWS_WEB otherwise appears as a bare count, leaving the reader
    # unable to tell that a claim was cleared on a web page rather than on the
    # vetted corpus — and the whole provenance design rests on that being
    # visible.
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
            # Authority is a ranking signal, not a filter, so a cited web
            # passage may come from any domain. Saying which it is lets the
            # reader weigh the citation instead of having to look it up.
            tier = ("recognised authority domain"
                    if is_authority(r["evidence_url"]) else "NOT an authority "
                    "domain — weigh accordingly")
            print(f"          [web evidence, {tier}]")
            print(f"          [retrieved {r.get('retrieved_at', 'unknown')}]")
        print("  EVIDENCE SAYS:")
        for line in textwrap.wrap(r["debunk_text"], 70):
            print(f"    {line}")
        print()



def check_transcript_parametric(transcript: str, verbose: bool = False) -> list[dict]:
    """Parametric-only condition: the same claim windows, adjudicated without
    retrieval (rag.llm.adjudicate_parametric).

    Rows carry the same keys as check_transcript(), with every evidence field
    empty and evidence_origin "none", so the same scoring code reads both.
    """
    claims = segment_claims(transcript)
    results = []
    for n, claim in enumerate(claims, 1):
        verdict = adjudicate_parametric(claim)
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
    return results
