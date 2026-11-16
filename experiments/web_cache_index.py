"""Index a web-search cache without its page text: queries, URLs, titles, times.

The cache (data/web_cache*/) stores the text of every page a run retrieved,
which is third-party content and is not redistributed. This writes what is
needed to audit a run instead: which query was made, when, and which pages
came back.

    python experiments/web_cache_index.py data/web_cache
    python experiments/web_cache_index.py data/results/v1/web_cache_after_<model>_tavily.tar.gz
    python experiments/web_cache_index.py SRC --out data/results/v1/web_cache_index_<name>.json

SRC is a cache directory or a .tar.gz archive of one. Without --out the index
is written next to SRC as web_cache_index_<name>.json.
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

KEEP = ("id", "source_url", "title", "publisher", "retrieved_at", "authority")


def _entries(src: Path):
    """Yield (name, parsed JSON) for every cache file in a directory or archive."""
    if src.is_dir():
        for f in sorted(src.glob("*.json")):
            yield f.name, json.loads(f.read_text(encoding="utf-8"))
        return
    with tarfile.open(src, "r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            name = Path(member.name).name
            # macOS archives carry "._x.json" resource forks beside each file
            if not member.isfile() or not name.endswith(".json") or name.startswith("._"):
                continue
            yield name, json.load(tar.extractfile(member))


def build_index(src: Path) -> dict:
    entries = []
    for name, entry in _entries(src):
        results = entry.get("results") or []
        entries.append({
            "file": name,
            "backend": entry.get("backend"),
            "query": entry.get("query"),
            "max_results": entry.get("max_results"),
            "cached_at": entry.get("cached_at"),
            "results": [{k: r.get(k) for k in KEEP} for r in results],
        })
    entries.sort(key=lambda e: (e["query"] or "", e["file"]))
    return {
        "description": ("Web-search cache index: every query a run made, when, and "
                        "which pages it retrieved. Page text is not included."),
        "source": src.name,
        "entries": len(entries),
        "pages": sum(len(e["results"]) for e in entries),
        "queries": entries,
    }


def default_out(src: Path) -> Path:
    name = src.name
    for suffix in (".tar.gz", ".tgz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("web_cache_"):
        name = name[len("web_cache_"):]
    elif name == "web_cache":
        name = "local"
    return src.parent / f"web_cache_index_{name}.json"


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(args[0])
    out = Path(args[args.index("--out") + 1]) if "--out" in args else default_out(src)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1
    index = build_index(src)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"{index['entries']} queries, {index['pages']} pages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
