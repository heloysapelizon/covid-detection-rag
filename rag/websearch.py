"""Web-search fallback for claims the reference corpus does not cover.

The corpus is deliberately small and curated (docs/corpus-inclusion.md), so some
narratives have no evidence in it at all — graphene oxide, microchips, 5G. For
those, retrieval returns nothing and the verdict is NO_EVIDENCE, which is honest
but unhelpful: the system has no opinion on exactly the claims most in need of
one.

This module retrieves from the web when, and only when, the corpus has nothing.

## Why web evidence is kept separate

Every corpus passage is licence-vetted, DOI-resolved and authorship-screened; a
paper was withdrawn on authorship grounds alone (docs/corpus-inclusion.md). Web
pages have none of that. Merging them into the same evidence list would silently
lower the standard behind every citation, so:

  - web verdicts use distinct labels (FAKE_NEWS_WEB / NOT_FAKE_NEWS_WEB)
  - they are excluded from the headline precision and recall figures
  - every result carries its URL and the time it was fetched, because pages
    change and a thesis figure has to be reproducible
  - results are cached to disk, so a re-run reproduces the same evidence rather
    than whatever the web says today (cleared before a measurement run)

## Why the trigger is not simply "retrieval was empty"

Measured over the 81 labelled windows, the relative cutoff leaves 4 windows with
no evidence. Only one is a genuine coverage gap (a 5G claim). The other three are
a sponsor read, conversational filler and scheduling chatter:

    "No, no, no, we're gonna do a series with you if you don't mind."

There is no claim there to check, so searching the web for it would spend a
network call to confirm that small talk is not misinformation. The trigger
therefore requires a *claim* as well as an absence of evidence, which is what
`should_search` decides using the existing stance pass.

## No topic restriction, deliberately

A domain-relevance check was built and **rejected on measurement**. Scoring each
window against a description of the study's subject (best 14-word fragment, since
whole-window scoring collapses to ~0.00 on 140 words of speech) separates
on-domain from off-domain windows cleanly in general — on-domain labelled windows
score 0.52-0.68 — but on the four empty windows it scores:

    0.1786  "mask on planes you can tell they're scared"      filler
    0.1121  crypto sponsor read
    0.0997  "2,000 studies ... effects of 5G on our systems"   REAL coverage gap
    0.0886  "SB 276, anything you want to say"                chatter

The genuine coverage gap scores **below the sponsor read**, because that window
discusses 5G without tying it to vaccination, so no threshold separates them. A
topic gate would have silently removed the trigger's only real case.

Beyond that failure, restricting to on-topic claims is the wrong scope: a system
that detects misinformation should not decline to check a false claim because it
is about 5G rather than vaccines. The cost is a wasted search on the occasional
sponsor read, which is cheap and visible in the output.

## Caching: on by default, cleared for a measurement run

A cache fixes the evidence at whatever the web said when the entry was written,
so a figure quoted from a warm cache is not a figure about the web today. That
is why a measurement run clears it first (`--clear-cache`).

It is on by default because the opposite risk dominates everywhere else: when
comparing two prompts or re-running a transcript, differing search results would
make the comparison measure the search engine rather than the thing under test.

The staleness is recorded rather than hidden — a cached row keeps its ORIGINAL
`retrieved_at`, the entry records `cached_at` separately, and preflight prints
the entry count so a warm-cache run cannot pass for a fresh one. `WEB_CACHE=0`
disables it. See docs/web-search.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .config import PROJECT_ROOT

# Authority domains are preferred in ranking but not required: general search was
# chosen deliberately, because the narratives that reach this fallback (graphene,
# microchips, 5G) are precisely the ones health authorities have not written
# about. Restricting to an allowlist would return nothing for them, which is the
# situation this module exists to fix. Provenance is handled by labelling the
# verdicts separately instead.
# Matched as domain suffixes, so "nih.gov" also covers "pubmed.ncbi.nlm.nih.gov"
# and "gov.br" covers every Brazilian government subdomain. The list was revised
# against domains DuckDuckGo actually returns for these narratives — a first
# version written from assumption would have classed itu.int and news.un.org as
# non-authority, which is most of what comes back for a 5G claim.
AUTHORITY_DOMAINS = (
    # Health and standards bodies
    "who.int", "cdc.gov", "ema.europa.eu", "fda.gov", "nih.gov",
    "cochrane.org", "gov.br", "anvisa.gov.br", "ecdc.europa.eu",
    "itu.int", "un.org", "europa.eu", "nhs.uk", "gov.uk",
    # Peer-reviewed publishers
    "nature.com", "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
    "sciencedirect.com", "springer.com", "wiley.com", "cell.com",
    "sciencemag.org", "science.org", "plos.org", "frontiersin.org", "mdpi.com",
    # Universities and research institutes
    "edu", "ac.uk",
    # Wire services and dedicated fact-checkers
    "reuters.com", "apnews.com", "afp.com", "factcheck.org", "fullfact.org",
    "snopes.com", "aosfatos.org", "politifact.com", "healthfeedback.org",
    "lupa.uol.com.br", "g1.globo.com",
)


def should_search(claim: str, passages: list[dict],
                  verdict: str | None = None,
                  addresses: str | None = None,
                  stance: str | None = None) -> tuple[bool, str]:
    """Whether to fall back to the web for this claim. Returns (decision, why).

    The trigger is the ADJUDICATED verdict, not an empty retrieval. Two
    conditions, both required:

      1. The corpus had nothing that addresses the claim — `verdict` is
         NO_EVIDENCE, or the model answered `addresses` "no".
      2. The window actually asserts something: `stance` is "asserting".

    Condition 1 was previously "retrieval returned no passage", which is close to
    unreachable: over 2,488 passages sharing a vaccine vocabulary, something
    almost always clears MIN_SIMILARITY. Measured on 171 windows collected
    specifically for narratives the corpus does not cover, only 1.8% retrieved
    nothing — a turbo-cancer claim gets general mRNA text at 0.399, a 5G claim
    gets myocarditis passages at 0.524. "Retrieval found no passage" is not the
    same question as "the corpus cannot answer this", and the relevance gate in
    rag/llm.py already decides the second one. See docs/web-search.md.

    Condition 2 exists because a window with no claim in it needs no evidence:
    of the four empty-retrieval windows in the v1 labels, three were a sponsor
    read, conversational filler and scheduling chatter.

    Called after `adjudicate`, so both fields are already available and no extra
    model call is needed.
    """
    no_evidence = verdict == "NO_EVIDENCE" or (addresses or "").lower() == "no"
    if not no_evidence:
        if passages:
            return False, f"corpus evidence addressed the claim (verdict={verdict})"
        return False, "no verdict available; not searching"

    if stance is None:
        return False, "no stance available; not searching"
    if stance != "asserting":
        return False, f"stance={stance!r}: nothing asserted to check"

    return True, f"corpus has no evidence addressing the claim (verdict={verdict})"


def search(query: str, max_results: int = 4, fetch_fn=None) -> list[dict]:
    """Retrieve web passages for a claim the corpus does not cover.

    Returns a list shaped like corpus passages — id, text, source_file,
    publisher, similarity — so `adjudicate` and the report path need no special
    case, plus `source_url` and `retrieved_at` for provenance. `origin` is set to
    "web" on every row so a caller can never mistake one for corpus evidence.

    `fetch_fn(query, max_results) -> list[dict]` performs the actual search and
    is injected rather than hardcoded: the search backend is whatever the host
    environment provides, and keeping it out of this module means the provenance
    and gating logic is testable offline.

    Results are cached on disk (see `cache_get`/`cache_put`). The cached entry
    keeps the ORIGINAL `retrieved_at`, so a replayed run reports when the
    evidence was actually fetched rather than when it was reused.
    """
    if fetch_fn is None:
        return []

    backend = getattr(fetch_fn, "__name__", "fetch")
    hit = cache_get(backend, query, max_results)
    if hit is not None:
        return hit

    raw = fetch_fn(query, max_results) or []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    results = []
    for n, item in enumerate(raw[:max_results], 1):
        url = item.get("url", "")
        results.append({
            "id": f"WEB-{n}",
            "text": item.get("text", "") or item.get("snippet", ""),
            "source_file": url,
            "source_url": url,
            "publisher": item.get("publisher", "") or _domain(url),
            "title": item.get("title", ""),
            "retrieved_at": stamp,
            "origin": "web",
            # Authority rank, not a similarity: web results arrive ordered by the
            # search engine, and calling it `similarity` would invite comparison
            # with cosine scores from the corpus, which it is not comparable to.
            "authority": is_authority(url),
            "similarity": None,
        })

    cache_put(backend, query, max_results, results)
    return results


# --- cache -----------------------------------------------------------------
#
# One JSON file per (backend, query, max_results). Enabled by default so a
# re-run costs no API credits and so prompt comparisons see IDENTICAL evidence:
# without it, a prompt variant could win or lose because the search engine
# returned different pages, which would measure the engine rather than the
# prompt.
#
# The cost is that a cached run reports what the web said when the entry was
# written, not now. `retrieved_at` carries the original timestamp precisely so
# that is visible in the results, and a measurement run clears the cache first
# (`--clear-cache`) so its evidence is fetched fresh.
# WEB_CACHE_DIR gives a run its own cache, so two runs searching the web at the
# same time (e.g. two models in parallel) cannot clear each other's evidence
# with --clear-cache. Default: the shared data/web_cache/.
CACHE_DIR = Path(os.environ["WEB_CACHE_DIR"]).resolve() if os.environ.get(
    "WEB_CACHE_DIR") else PROJECT_ROOT / "data" / "web_cache"
CACHE_ENABLED = os.environ.get("WEB_CACHE", "1") not in ("0", "false", "False")


def _cache_key(backend: str, query: str, max_results: int) -> str:
    raw = f"{backend}\x00{query.strip().lower()}\x00{max_results}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def cache_path(backend: str, query: str, max_results: int) -> Path:
    return CACHE_DIR / f"{_cache_key(backend, query, max_results)}.json"


def cache_get(backend: str, query: str, max_results: int) -> list[dict] | None:
    """Return cached results, or None on a miss or any read problem."""
    if not CACHE_ENABLED:
        return None
    p = cache_path(backend, query, max_results)
    try:
        entry = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    results = entry.get("results")
    return results if isinstance(results, list) else None


def cache_put(backend: str, query: str, max_results: int,
              results: list[dict]) -> None:
    """Store results. A cache write must never break a run, so errors pass."""
    if not CACHE_ENABLED:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path(backend, query, max_results).write_text(
            json.dumps({"backend": backend, "query": query,
                        "max_results": max_results,
                        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                        "results": results},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError:
        pass


def cache_clear() -> int:
    """Delete every cached entry. Returns how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def cache_stats() -> dict:
    """How many entries are cached, for preflight reporting."""
    files = list(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
    return {"enabled": CACHE_ENABLED, "entries": len(files),
            "dir": str(CACHE_DIR)}


TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY_ENV = "TAVILY_API_KEY"

# "basic" costs 1 API credit per request; "advanced" costs 2 and returns longer
# extracted snippets. Basic is used because the snippet only has to be long
# enough for the relevance gate to judge, and the free tier is 1,000 credits a
# month against a fallback that fires on a few windows per video.
TAVILY_DEPTH = "basic"


def _api_key_from_env(name: str) -> str | None:
    """Read a key from the environment, falling back to the project .env.

    The collect package loads .env itself; rag did not, so a key written there
    was silently ignored and the adapter returned no results with no indication
    why. Checked here rather than at import so the file is read only when a
    backend actually needs a key.
    """
    value = os.environ.get(name)
    if value:
        return value
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip("'\"") or None
    except OSError:
        return None
    return None


def tavily_fetch(query: str, max_results: int = 4,
                 api_key: str | None = None,
                 search_depth: str = TAVILY_DEPTH,
                 timeout: float = 20.0) -> list[dict]:
    """A `fetch_fn` backed by Tavily, read from TAVILY_API_KEY.

    Tavily is built for retrieval-augmented use: it returns cleaned extracted
    text per result rather than a rendered SERP, so no page fetching or HTML
    stripping is needed. That is the one concrete advantage over the DuckDuckGo
    adapter, whose `body` field is a short meta description.

    It is NOT the default, and the reasoning is in docs/web-search.md: the
    DuckDuckGo backend needs no key and follows published practice for this exact
    task (Tian et al., arXiv:2409.00009), which makes it citable. Tavily is a
    paid service outside that literature, so a result produced with it rests on a
    different footing.

    Returns the same shape every `fetch_fn` returns — url/title/text — plus
    `score`, Tavily's own relevance number. Failures return [] rather than
    raising: the web tier is a fallback, so an outage or an exhausted quota
    should leave the verdict at NO_EVIDENCE rather than end a transcript run.
    """
    key = api_key or _api_key_from_env(TAVILY_KEY_ENV)
    if not key:
        return []

    try:
        import requests
    except ImportError:
        return []

    try:
        response = requests.post(
            TAVILY_URL,
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                # Authority preference rather than a hard filter: the narratives
                # that reach this fallback are the ones health authorities have
                # not published on, so restricting to them would return nothing
                # exactly when it is needed. See AUTHORITY_DOMAINS.
                "include_domains": list(AUTHORITY_DOMAINS),
                "include_domains_mode": "prefer",
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Network error, bad key, exhausted quota, or an unexpected body.
        return []

    out = []
    for item in (payload.get("results") or [])[:max_results]:
        out.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "text": item.get("content", "") or item.get("raw_content", ""),
            "score": item.get("score"),
        })
    return out


def duckduckgo_fetch(query: str, max_results: int = 4) -> list[dict]:
    """A `fetch_fn` backed by DuckDuckGo, via the `ddgs` package.

    Chosen over Brave, Serper, Tavily and Exa because it needs no API key, no
    card and no per-query budget, and because it is the backend used by Tian et
    al., "Web Retrieval Agents for Evidence-Based Misinformation Detection"
    (arXiv:2409.00009) — the one study measuring web retrieval for an offline
    LLM, which makes the choice citable rather than merely convenient. At the
    volume this fallback fires (roughly 2 windows in 81) a paid API buys nothing.

    `ddgs` returns title/href/body; `search` expects title/url/text, so the keys
    are mapped here. Failures return [] rather than raising: the web tier is a
    fallback, and a search outage should leave the verdict at NO_EVIDENCE rather
    than crash a transcript run.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    try:
        hits = DDGS().text(query, max_results=max_results) or []
    except Exception:
        # Rate limits, network errors and upstream HTML changes all surface here.
        return []

    return [
        {
            "url": h.get("href", ""),
            "title": h.get("title", ""),
            "text": h.get("body", ""),
        }
        for h in hits
    ]


def _domain(url: str) -> str:
    """Host part of a URL, without scheme or www."""
    host = url.split("//")[-1].split("/")[0].lower()
    return host[4:] if host.startswith("www.") else host


def is_authority(url: str) -> bool:
    """Whether a URL's host is, or sits under, a listed authority domain.

    Suffix match on dot boundaries, so "pubmed.ncbi.nlm.nih.gov" matches
    "nih.gov" while "notnih.gov" does not.
    """
    host = _domain(url)
    return any(host == d or host.endswith("." + d) for d in AUTHORITY_DOMAINS)


def rank_by_authority(results: list[dict]) -> list[dict]:
    """Stable sort putting authority domains first, order otherwise preserved.

    General search with authority preference, rather than an allowlist: see
    AUTHORITY_DOMAINS for why.
    """
    return sorted(results, key=lambda r: not r.get("authority", False))


# Backend registry. DuckDuckGo is the default because it needs no key and follows
# published practice for this task; Tavily is available for runs where cleaned
# extracted text matters more than citability. See docs/web-search.md.
BACKENDS = {
    "duckduckgo": duckduckgo_fetch,
    "tavily": tavily_fetch,
}

DEFAULT_BACKEND = "duckduckgo"


def get_backend(name: str | None = None):
    """Resolve a `fetch_fn` by name, or None to disable the fallback.

    Reads WEB_SEARCH_BACKEND when `name` is not given, so a run can switch
    backend without a code change. An unknown name raises rather than silently
    falling back, since which backend produced a verdict is part of its
    provenance.
    """
    import os
    chosen = name or os.environ.get("WEB_SEARCH_BACKEND") or DEFAULT_BACKEND
    if chosen in ("none", "off", ""):
        return None
    if chosen not in BACKENDS:
        raise ValueError(
            f"unknown web-search backend {chosen!r}; "
            f"choose from {', '.join(sorted(BACKENDS))}, or 'none'"
        )
    return BACKENDS[chosen]
