"""Web-search fallback for claims the reference corpus does not cover."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .config import PROJECT_ROOT

AUTHORITY_DOMAINS = (
    "who.int", "cdc.gov", "ema.europa.eu", "fda.gov", "nih.gov",
    "cochrane.org", "gov.br", "anvisa.gov.br", "ecdc.europa.eu",
    "itu.int", "un.org", "europa.eu", "nhs.uk", "gov.uk",
    "nature.com", "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
    "sciencedirect.com", "springer.com", "wiley.com", "cell.com",
    "sciencemag.org", "science.org", "plos.org", "frontiersin.org", "mdpi.com",
    "edu", "ac.uk",
    "reuters.com", "apnews.com", "afp.com", "factcheck.org", "fullfact.org",
    "snopes.com", "aosfatos.org", "politifact.com", "healthfeedback.org",
    "lupa.uol.com.br", "g1.globo.com",
)


def should_search(claim: str, passages: list[dict],
                  verdict: str | None = None,
                  addresses: str | None = None,
                  stance: str | None = None) -> tuple[bool, str]:
    """Whether to fall back to the web for this claim. Returns (decision, why)."""
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
    """Retrieve web passages for a claim the corpus does not cover."""
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
            "authority": is_authority(url),
            "similarity": None,
        })

    cache_put(backend, query, max_results, results)
    return results


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

TAVILY_DEPTH = "basic"


def _api_key_from_env(name: str) -> str | None:
    """Read a key from the environment, falling back to the project .env."""
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
    """A `fetch_fn` backed by Tavily, read from TAVILY_API_KEY."""
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
                "include_domains": list(AUTHORITY_DOMAINS),
                "include_domains_mode": "prefer",
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
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
    """A `fetch_fn` backed by DuckDuckGo, via the `ddgs` package."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    try:
        hits = DDGS().text(query, max_results=max_results) or []
    except Exception:
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
    """Whether a URL's host is, or sits under, a listed authority domain."""
    host = _domain(url)
    return any(host == d or host.endswith("." + d) for d in AUTHORITY_DOMAINS)


def rank_by_authority(results: list[dict]) -> list[dict]:
    """Stable sort putting authority domains first, order otherwise preserved."""
    return sorted(results, key=lambda r: not r.get("authority", False))


BACKENDS = {
    "duckduckgo": duckduckgo_fetch,
    "tavily": tavily_fetch,
}

DEFAULT_BACKEND = "duckduckgo"


def get_backend(name: str | None = None):
    """Resolve a `fetch_fn` by name, or None to disable the fallback."""
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
