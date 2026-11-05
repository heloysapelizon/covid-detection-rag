"""orchestration: gather candidates, complete their metadata, filter, report.

every candidate is completed with a videos.list pass. That is also what makes the duration filter possible, and it costs
1 quota unit per 50 videos against the 100 units each search already spent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATASET_NAME, MAX_DURATION_S, SEARCH_QUERIES
from .data import parse_links_file, save_dataset, to_records
from .filters import apply_all
from .api import HttpError, fetch_videos_by_id, search_videos


def gather_from_search(client, queries: list[str] = SEARCH_QUERIES) -> dict[str, str]:
    """Search every query, returning {video_id: query} for the first match.

    First-wins keeps a video's original provenance when it matches several queries.
    A failing query is reported and skipped rather than ending the run.
    """
    provenance: dict[str, str] = {}
    for q in queries:
        try:
            for v in search_videos(client, q):
                provenance.setdefault(v.video_id, q)
            print(f"  searched: {q!r:36}  unique so far: {len(provenance)}")
        except HttpError as e:
            print(f"  SEARCH ERROR for {q!r}: {e}")
    return provenance


def gather_from_links(links_file: Path) -> dict[str, str]:
    """Parse a links .txt, returning {video_id: provenance}."""
    ids, bad = parse_links_file(links_file)
    print(f"  {links_file.name}: {len(ids)} unique id(s)")
    if bad:
        print(f"  {len(bad)} line(s) not recognised:")
        for b in bad[:10]:
            print(f"    {b[:70]}")
    return {vid: f"link_list:{links_file.stem}" for vid in ids}


def load_excluded_ids(path: Path) -> set[str]:
    """Video ids to keep out of a collection run.

    Accepts either a ground-truth label file (an object with an `items` list) or
    a plain JSON list of ids. Returns an empty set when the file is absent, so a
    run without exclusions still works.
    """
    if not path or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return {r["video_id"] for r in data["items"] if r.get("video_id")}
    if isinstance(data, list):
        return {r["video_id"] if isinstance(r, dict) else r for r in data}
    return set()


def collect(
    client,
    queries: list[str] | None = None,
    links_file: Path | None = None,
    name: str = DATASET_NAME,
    max_duration_s: int = MAX_DURATION_S,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Build and save a filtered dataset. Returns the saved records.

    `exclude_ids` drops candidates before metadata is fetched. This is what keeps
    a second collection run from leaking the held-out test videos into a dev set:
    the v2 queries overlap the v1 ones in subject matter, so YouTube returns many
    of the same videos, and without this they would silently reappear in the set
    used for tuning.
    """
    provenance: dict[str, str] = {}

    if queries:
        print(f"Searching {len(queries)} query(ies) "
              f"({len(queries) * 100} quota units):")
        provenance.update(gather_from_search(client, queries))

    if links_file:
        print("Reading link list:")
        for vid, src in gather_from_links(links_file).items():
            provenance.setdefault(vid, src)

    if not provenance:
        print("No candidates gathered.")
        return []

    if exclude_ids:
        overlap = set(provenance) & exclude_ids
        if overlap:
            print(f"\nExcluding {len(overlap)} candidate(s) already in the "
                  f"held-out set:")
            for vid in sorted(overlap):
                print(f"    {vid}  ({provenance[vid]})")
            for vid in overlap:
                del provenance[vid]
        if not provenance:
            print("Every candidate was excluded.")
            return []

    ids = list(provenance)
    print(f"\nFetching full metadata for {len(ids)} video(s) "
          f"({(len(ids) + 49) // 50} quota unit(s))...")
    videos, missing = fetch_videos_by_id(client, ids, query="")
    for v in videos:
        v.query = provenance[v.video_id]
    if missing:
        print(f"  {len(missing)} id(s) returned nothing "
              f"(deleted/private/typo): {missing}")

    records = apply_all(to_records(videos), max_duration_s)

    json_path, csv_path = save_dataset(records, name)
    print(f"\nSaved {len(records)} videos:\n  {json_path}\n  {csv_path}")
    return records


def print_summary(records: list[dict]) -> None:
    """Print channel and provenance breakdowns for a saved dataset."""
    if not records:
        return
    df = pd.DataFrame(records)
    print(f"\nUnique channels: {df['channel'].nunique()}")

    durations = df["duration_s"].dropna()
    if len(durations):
        print(f"Total runtime: {durations.sum() / 3600:.1f} h "
              f"(median {durations.median() / 60:.0f} min per video)")

    print("\nVideos per query:")
    print(df["query"].value_counts().to_string())
