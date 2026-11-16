"""Which videos belong to which split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transcribe.config import is_usable  # noqa: E402
LABELS = ROOT / "data" / "labels"
YOUTUBE = ROOT / "data" / "youtube"

TEST_LABELS = LABELS / "ground_truth.json"
DEV_SELECTED = YOUTUBE / "covid_vaccine_videos_dev2_mirror_selected.json"
DEV_DATASET = YOUTUBE / "covid_vaccine_videos_dev2_mirror_selected_transcribed.json"
DEV_LABELS = LABELS / "windows" / "chunk_eval_labels_dev2.json"
DEV_VIDEO_LABELS = LABELS / "dev_ground_truth.json"

FALLBACK_SELECTED = YOUTUBE / "covid_vaccine_videos_dev2_uncovered_selected.json"
FALLBACK_DATASET = (
    YOUTUBE / "covid_vaccine_videos_dev2_uncovered_selected_transcribed.json"
)

EXCLUDED: dict[str, str] = {
    "RnWIrve2yv0": "off scope: CNBC interview with the CEO of Microchip "
                   "Technology Inc about semiconductor supply chains",
    "pQl_p4GtCOo": "off scope: cancer/sugar debunk, no vaccine content",
    "rJgxDRsg1Jo": "unlabellable: teaser deferring all substance to another video",
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
    """Usable transcripts from a transcribed dataset file."""
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


def check_content_leakage(threshold: float = 0.4) -> list[tuple[str, str, float]]:
    """Dev/fallback videos whose transcript largely repeats a test transcript."""
    def shingles(text: str, k: int = 8) -> set[str]:
        w = text.lower().split()
        return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k))}

    transcripts = YOUTUBE / "covid_vaccine_videos_transcribed.json"
    if not transcripts.exists():
        return []
    test_ids = {r["video_id"] for r in test_videos(include_excluded=True)}
    test = {r["video_id"]: shingles(r["transcript"])
            for r in json.loads(transcripts.read_text(encoding="utf-8"))
            if r["video_id"] in test_ids and r.get("transcript")}
    handled = set()
    if DEV_VIDEO_LABELS.exists():
        handled = {i["video_id"] for i in json.loads(
            DEV_VIDEO_LABELS.read_text(encoding="utf-8"))["items"]
            if i.get("exclude")}
    hits = []
    for r in dev_videos() + fallback_videos():
        if r["video_id"] in handled:
            continue
        mine = shingles(r["transcript"])
        for tid, theirs in test.items():
            share = len(mine & theirs) / max(1, min(len(mine), len(theirs)))
            if share >= threshold:
                hits.append((r["video_id"], tid, round(share, 2)))
    return hits


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
    dup = check_content_leakage()
    lines.append(f"DUP   {'none' if not dup else str(len(dup)) + ' -> ' + str(dup)}"
                 "  (transcript overlap with test, unexcluded)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
