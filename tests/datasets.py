"""Which videos belong to which split.

    python tests/datasets.py          # split sizes and a leakage check

Three sets, with different jobs:

  TEST   the 124 v1 videos, hand-labelled true/false in
         data/labels/ground_truth.json (119 after exclusions). Used ONCE, for
         the final reported result. No parameter may be chosen by looking at it.

  DEV    the mirror group of the second dev collection — the six v1 queries
         repeated, so retrieval figures transfer to the test set. Every tuning
         decision happens here.

  FBACK  the uncovered group of the same collection: narratives the evidence
         corpus does not cover. Deliberately unrepresentative, and used only to
         exercise the web-search fallback, never to choose a parameter.

Why separate collections rather than a split of the 124: 45 of those videos had
been used to choose seven hyperparameters, so any figure over them partly
measures that tuning. See docs/dev-sets.md for the collection and selection, and
for why the first dev set was replaced.

tests/split.py is a different thing — an internal A/B split of the v1 labels,
useful for checking whether a threshold transfers, but both of its halves are
tuning material.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Single definition, so the rule deciding which transcripts enter an experiment
# cannot drift between the pipeline and the eval scripts.
from transcribe.config import is_usable  # noqa: E402
LABELS = ROOT / "data" / "labels"
YOUTUBE = ROOT / "data" / "youtube"

TEST_LABELS = LABELS / "ground_truth.json"
# The tuning set: the mirror group of the second dev collection, which repeats
# the v1 query set so figures transfer to the test set. The uncovered group is
# tracked separately because it is deliberately unrepresentative and is used only
# to exercise the web-search fallback — never to choose a parameter.
# See docs/dev-sets.md.
DEV_SELECTED = YOUTUBE / "covid_vaccine_videos_dev2_mirror_selected.json"
DEV_DATASET = YOUTUBE / "covid_vaccine_videos_dev2_mirror_selected_transcribed.json"
DEV_LABELS = ROOT / "tests" / "chunk_eval_labels_dev2.json"

FALLBACK_SELECTED = YOUTUBE / "covid_vaccine_videos_dev2_uncovered_selected.json"
FALLBACK_DATASET = (
    YOUTUBE / "covid_vaccine_videos_dev2_uncovered_selected_transcribed.json"
)

# Videos excluded from evaluation even though they are labelled. Recorded here
# rather than deleted from the label file, so the reason survives.
EXCLUDED: dict[str, str] = {
    "RnWIrve2yv0": "off scope: CNBC interview with the CEO of Microchip "
                   "Technology Inc about semiconductor supply chains",
    "pQl_p4GtCOo": "off scope: cancer/sugar debunk, no vaccine content",
    "rJgxDRsg1Jo": "unlabellable: teaser deferring all substance to another video",
    # Byte-identical transcripts; the second of each pair is dropped so a video
    # is not scored twice.
    "kSU7Eyv0_Ds": "duplicate of gQfeAj5w-yc (same CBS package, two channels)",
    "-9U3HTacDP8": "duplicate of Z6qlcRGzh-Q (same Sinclair interview, two stations)",
}


def test_videos(include_excluded: bool = False) -> list[dict]:
    """The held-out test set: hand-labelled videos from the v1 collection."""
    if not TEST_LABELS.exists():
        return []
    items = json.loads(TEST_LABELS.read_text(encoding="utf-8"))["items"]
    if include_excluded:
        return items
    return [r for r in items if r["video_id"] not in EXCLUDED]


def _transcribed(path: Path) -> list[dict]:
    """Usable transcripts from a transcribed dataset file.

    Accepts "ok_capped" as usable: dev transcripts are capped at
    TRANSCRIBE_MAX_S by design, and a cap is a deliberate truncation rather than
    a failure.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("videos", [])
    return [r for r in rows
            if is_usable(r.get("transcript_status")) and r.get("transcript")]


def dev_videos() -> list[dict]:
    """The tuning set: the transcribed mirror group."""
    return _transcribed(DEV_DATASET)


def fallback_videos() -> list[dict]:
    """The uncovered group, for web-fallback measurement only."""
    return _transcribed(FALLBACK_DATASET)


def check_leakage() -> list[str]:
    """Video ids present in both splits. Must be empty."""
    test_ids = {r["video_id"] for r in test_videos(include_excluded=True)}
    dev_ids = {r["video_id"] for r in dev_videos()}
    fb_ids = {r["video_id"] for r in fallback_videos()}
    return sorted((dev_ids | fb_ids) & test_ids | (dev_ids & fb_ids))


def describe() -> str:
    test = test_videos()
    dev = dev_videos()
    lines = [
        f"TEST  {len(test):3} videos  ({len(EXCLUDED)} excluded)  "
        f"labels: {TEST_LABELS.name}",
    ]
    if test:
        fake = sum(1 for r in test if r["label_fake_news"])
        lines.append(f"      {fake} fake_news / {len(test) - fake} not_fake_news")
    for label, selected_path, rows in (
        ("DEV  ", DEV_SELECTED, dev),
        ("FBACK", FALLBACK_SELECTED, fallback_videos()),
    ):
        sel = (len(json.loads(selected_path.read_text(encoding="utf-8")))
               if selected_path.exists() else 0)
        if rows:
            lines.append(f"{label} {len(rows):3} videos transcribed  "
                         f"({sel} selected)")
        elif sel:
            lines.append(f"{label} {0:3} videos transcribed  "
                         f"({sel} selected, awaiting transcription)")
        else:
            lines.append(f"{label}   -  NOT YET SELECTED")
    leak = check_leakage()
    lines.append(f"LEAK  {'none' if not leak else str(len(leak)) + ' -> ' + str(leak)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
