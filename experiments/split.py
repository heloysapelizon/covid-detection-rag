"""Internal A/B split of the v1 labelled windows, for transfer checks."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = (Path(__file__).resolve().parent.parent / "data" / "labels" / "windows"
          / "chunk_eval_labels.json")

_TOPIC_ORDER = (
    "blood_clots",
    "myocarditis",
    "infertility",
    "natural_immunity",
    "general",
)

SPLITS = ("dev", "test")


def load_items(path: Path = LABELS) -> list[dict]:
    """The labelled windows, as stored."""
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _video_topic(windows: list[dict]) -> str:
    """The topic a video is counted under: its most common non-general label."""
    topical = Counter(
        w["topic"] for w in windows if w.get("topic") and w["topic"] != "general"
    )
    return topical.most_common(1)[0][0] if topical else "general"


def assign_split(items: list[dict] | None = None) -> dict[str, str]:
    """video_id -> "dev" or "test". Deterministic, stratified by topic."""
    items = items if items is not None else load_items()

    by_video: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_video[item["video_id"]].append(item)

    topic_of = {v: _video_topic(ws) for v, ws in by_video.items()}

    assignment: dict[str, str] = {}
    for topic in _TOPIC_ORDER:
        videos = sorted(
            (v for v, t in topic_of.items() if t == topic),
            key=lambda v: hashlib.sha1(v.encode()).hexdigest(),
        )
        for i, video in enumerate(videos):
            assignment[video] = SPLITS[i % 2]

    for video in by_video:
        assignment.setdefault(video, SPLITS[0])
    return assignment


def split_items(items: list[dict] | None = None) -> dict[str, list[dict]]:
    """{"dev": [...], "test": [...]} over the labelled windows."""
    items = items if items is not None else load_items()
    assignment = assign_split(items)
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for item in items:
        out[assignment[item["video_id"]]].append(item)
    return out


def describe(items: list[dict] | None = None) -> str:
    """A short table of what each side holds, for printing in eval output."""
    parts = []
    for name, rows in split_items(items).items():
        topical = [r for r in rows if r.get("topic") and r["topic"] != "general"]
        counts = dict(sorted(Counter(r["topic"] for r in topical).items()))
        parts.append(
            f"  {name:5} windows={len(rows):3} videos={len({r['video_id'] for r in rows}):3} "
            f"topical={len(topical):3} general={len(rows) - len(topical):3}\n"
            f"        {counts}"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    print(describe())
