"""Choose balanced subsets from a collected candidate pool.

    python tests/select_dev.py --group mirror    --n 50 --write
    python tests/select_dev.py --group uncovered --n 21 --write
    python tests/select_dev.py --group mirror    --n 50          # dry run

A collection returns far more candidates than are needed: 135 mirror and 158
uncovered for targets of 50 and 21. Transcribing all of them wastes hours, and
taking the first N inherits the pool's imbalances. Selection balances the axes
that affect what can be measured, and is deterministic — candidates are ordered
by SHA-1 of their video id within each bucket, so re-running gives the same
subset with no stored seed and no sensitivity to input order.

Rationale for the criteria is in docs/dev-sets.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
YOUTUBE = ROOT / "data" / "youtube"

# The second dev collection, and the two groups it splits into. They are kept
# apart because they have different jobs: the mirror group repeats the v1 query
# set, so retrieval figures measured on it transfer to the test set, while the
# uncovered group is deliberately unrepresentative and exists only to exercise
# the web-search fallback. Pooling them would let the unrepresentative half
# dominate every tuning figure — the raw collection was 39 mirror against 158
# uncovered.
POOLS: dict[str, Path] = {
    "mirror": YOUTUBE / "covid_vaccine_videos_dev2_mirror_more.json",
    "uncovered": YOUTUBE / "covid_vaccine_videos_dev2_uncovered.json",
}
OUTPUTS: dict[str, Path] = {
    "mirror": YOUTUBE / "covid_vaccine_videos_dev2_mirror_selected.json",
    "uncovered": YOUTUBE / "covid_vaccine_videos_dev2_uncovered_selected.json",
}

# Ids already used by the test set or the first dev collection. A candidate in
# here must never be selected: the test videos carry the final result, and the
# first dev pool was the exclusion basis for this collection.
EXCLUDE = YOUTUBE / "exclude_dev2.json"

# At most this many videos from one channel. Several clips from one broadcaster
# share a presenter, format and stock phrasing, so they behave as one
# observation rather than several.
MAX_PER_CHANNEL = 2

# Preferred duration band in seconds — a tiebreak, not a filter. Long enough to
# yield several claim windows, short enough to transcribe quickly and to sit
# under the transcription cap in transcribe/config.py.
PREFERRED_MIN_S = 60
PREFERRED_MAX_S = 12 * 60


def load_excluded() -> set[str]:
    """Video ids that must not be selected."""
    if not EXCLUDE.exists():
        return set()
    data = json.loads(EXCLUDE.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return {r["video_id"] for r in data["items"]}
    return {r["video_id"] if isinstance(r, dict) else r for r in data}


def queries_for(group: str) -> list[str]:
    """The query list a group balances across."""
    from collect.config import (
        SEARCH_QUERIES_V3_MIRROR,
        SEARCH_QUERIES_V3_UNCOVERED,
    )
    return {
        "mirror": list(SEARCH_QUERIES_V3_MIRROR),
        "uncovered": list(SEARCH_QUERIES_V3_UNCOVERED),
    }[group]


def _rank_key(record: dict) -> tuple:
    """Order within a query bucket: preferred duration first, then stable hash."""
    duration = record.get("duration_s") or 0
    in_band = PREFERRED_MIN_S <= duration <= PREFERRED_MAX_S
    return (
        0 if in_band else 1,
        hashlib.sha1(record["video_id"].encode()).hexdigest(),
    )


def select(pool: list[dict], queries: list[str], n: int,
           max_per_channel: int = MAX_PER_CHANNEL,
           excluded: set[str] | None = None) -> list[dict]:
    """Pick n candidates, round-robin across queries, spread across channels.

    Round-robin rather than proportional, so a query that returned many
    candidates cannot crowd out one that returned few — the raw pools are uneven
    (10 videos for one mirror query against 29 for another).

    Two passes: the first respects the per-channel cap, the second relaxes it
    only if the target cannot otherwise be met. A shortfall is visible in
    `report` rather than silently reallocated.
    """
    excluded = excluded or set()
    by_query: dict[str, list[dict]] = defaultdict(list)
    for record in pool:
        if record["video_id"] in excluded:
            continue
        query = (record.get("query") or "").strip()
        if query in queries:
            by_query[query].append(record)
    for records in by_query.values():
        records.sort(key=_rank_key)

    chosen: list[dict] = []
    channel_count: Counter = Counter()
    cursor = {q: 0 for q in by_query}

    for relax in (False, True):
        while len(chosen) < n:
            progressed = False
            for query in queries:
                if query not in by_query or len(chosen) >= n:
                    continue
                while cursor[query] < len(by_query[query]):
                    candidate = by_query[query][cursor[query]]
                    cursor[query] += 1
                    channel = candidate.get("channel", "")
                    if not relax and channel_count[channel] >= max_per_channel:
                        continue
                    chosen.append(candidate)
                    channel_count[channel] += 1
                    progressed = True
                    break
            if not progressed:
                break
        if len(chosen) >= n:
            break
        cursor = {q: 0 for q in by_query}
        seen = {r["video_id"] for r in chosen}
        for query in by_query:
            by_query[query] = [r for r in by_query[query]
                               if r["video_id"] not in seen]
    return chosen[:n]


def report(selected: list[dict], pool: list[dict], queries: list[str]) -> str:
    lines = [f"pool {len(pool)} candidates -> selected {len(selected)}", ""]
    counts = Counter(r.get("query") for r in selected)
    pool_counts = Counter(r.get("query") for r in pool)
    lines.append("per query:")
    for query in queries:
        lines.append(f"  {counts.get(query, 0):3}  (pool {pool_counts.get(query, 0):3})"
                     f"  {query}")
    channels = Counter(r.get("channel", "") for r in selected)
    lines.append("")
    lines.append(f"channels: {len(channels)} distinct, "
                 f"max {max(channels.values()) if channels else 0} per channel")
    durations = [(r.get("duration_s") or 0) / 60 for r in selected]
    if durations:
        cap = PREFERRED_MAX_S / 60
        lines.append(f"duration: median {statistics.median(durations):.1f} min, "
                     f"{sum(1 for d in durations if d <= cap)}/{len(durations)} "
                     f"under the {cap:.0f}-min transcription cap")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(POOLS), required=True,
                    help="which group of the second dev collection to select")
    ap.add_argument("--n", type=int, required=True, help="how many to select")
    ap.add_argument("--write", action="store_true",
                    help="write the selection; otherwise print and exit")
    args = ap.parse_args()

    pool_path = POOLS[args.group]
    if not pool_path.exists():
        print(f"no candidate pool at {pool_path}")
        return 1
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    queries = queries_for(args.group)
    excluded = load_excluded()

    selected = select(pool, queries, args.n, excluded=excluded)
    print(f"group: {args.group}   excluding {len(excluded)} already-used id(s)")
    print(report(selected, pool, queries))

    leaked = {r["video_id"] for r in selected} & excluded
    if leaked:
        print(f"\nERROR: {len(leaked)} selected id(s) are in the exclusion list")
        return 1

    if args.write:
        out = OUTPUTS[args.group]
        out.write_text(
            json.dumps(selected, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
        print(f"transcribe it with:\n  python -m transcribe {out.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
