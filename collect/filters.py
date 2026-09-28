"""removes duplicated videos and videos with duration over an hour"""
from __future__ import annotations

from .config import MAX_DURATION_S


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    """Remove repeated video_ids, keeping the first occurrence"""
    seen: set[str] = set()
    kept: list[dict] = []
    for r in records:
        vid = r["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        kept.append(r)
    return kept, len(records) - len(kept)


def drop_long(
    records: list[dict],
    max_duration_s: int = MAX_DURATION_S,
) -> tuple[list[dict], int, int]:
    """remove videos longer than max_duration_s."""
    kept: list[dict] = []
    dropped = 0
    unknown = 0
    for r in records:
        d = r.get("duration_s")
        if not d: # None or 0 (live streams report P0D)
            unknown += 1
            kept.append(r)
        elif d > max_duration_s:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped, unknown


def apply_all(
    records: list[dict],
    max_duration_s: int = MAX_DURATION_S,
    verbose: bool = True,
) -> list[dict]:
    """Run every filter in order and report what each removed."""
    total = len(records)

    records, n_dupes = dedupe(records)
    records, n_long, n_unknown = drop_long(records, max_duration_s)

    if verbose:
        print(f"Filtering {total} record(s):")
        print(f"  duplicates removed:        {n_dupes}")
        print(f"  over {max_duration_s // 60} min removed:      {n_long}")
        if n_unknown:
            print(f"  kept, duration unknown:    {n_unknown}")
        print(f"  remaining:                 {len(records)}")
    return records
