"""Internal A/B split of the v1 labelled windows, for transfer checks.

NOT the project's dev/test split. That one is in tests/datasets.py and separates
two different COLLECTIONS: the v2 dev videos (tuning) from the v1 test videos
(final report). This module splits the v1 labels *against themselves*, so both
of its halves are tuning material and neither is held out in the project sense.
It answers only "does a threshold chosen on one half transfer to the other?",
which is weaker than a real held-out figure but still catches overfitting to a
particular set of windows — it is what showed the weighted-fusion gain in
docs/retrieval-methods.md was noise.

The halves are named `dev` and `test` because that is what the sweep code reads;
read them as "half A" and "half B".

    from tests.split import assign_split, SPLITS

Hyperparameters (ROUTE_GAMMA, SCOPE_ANSWERED_SIMILARITY, FLAG_MIN_SIMILARITY)
were chosen by reading a threshold table off all 81 labelled windows and picking
the knee. That is selection on the evaluation set: the reported figure is then
partly a measure of the choice, not only of the system.

This module fixes a split so a threshold can be chosen on `dev` and reported on
`test`, which no tuning decision has seen.

Two properties matter and neither is satisfied by a random row split:

  Split by VIDEO, not by window. Windows from one video overlap by 70 words and
  share a speaker, vocabulary and topic, so the same claim often appears in two
  consecutive windows. Splitting rows would put near-duplicates on both sides and
  make `test` a paraphrase of `dev`.

  Stratify by TOPIC. The corpus has four topics with 6-13 labelled windows each.
  An unstratified split left blood_clots at 1 dev / 8 test and natural_immunity
  at 5 / 1, so neither half could measure routing for both.

The assignment is deterministic — videos are ordered by SHA-1 of their id within
each topic and alternated — so it is stable across runs and machines without
storing a list of ids. It is not random: there is no seed to vary. Re-deriving
the split after adding labels will reshuffle it, so a reported `test` figure
should name the label-file state it was computed against.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = Path(__file__).resolve().parent / "chunk_eval_labels.json"

# Ordered so the smallest topics are assigned first: with an odd number of
# videos the earlier topics get the extra one, and the scarce topics are where
# that matters.
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
    """The topic a video is counted under: its most common non-general label.

    A video that makes claims about two topics is assigned to its dominant one,
    so it lands wholly on one side of the split rather than being divided.
    """
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

    # Any topic not named in _TOPIC_ORDER (a label added later) still gets split
    # rather than silently dropped.
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
