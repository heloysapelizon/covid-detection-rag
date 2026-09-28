"""Merge transcripts back into the dataset records and write the result."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def build_enriched(dataset: list[dict], results: dict) -> list[dict]:
    """Return dataset records with transcript columns appended.

    Record order and every original field are preserved; videos absent from
    results are marked "not_processed" rather than dropped.
    """
    enriched = []
    for d in dataset:
        r = results.get(d["video_id"], {})
        row = dict(d)
        row["transcript"] = r.get("transcript", "")
        row["transcript_status"] = r.get("transcript_status", "not_processed")
        row["detected_language"] = r.get("meta", {}).get("language", "")
        row["audio_duration_s"] = r.get("meta", {}).get("duration_s")
        enriched.append(row)
    return enriched


def save_enriched(
    enriched: list[dict],
    dataset_path: Path,
    out_dir: Path = DATA_DIR,
) -> tuple[Path, Path]:
    """Write <dataset>_transcribed.json/.csv, replacing them if they exist.

    One file per dataset, overwritten on every run — the checkpoint is what
    accumulates progress, so the output is always a fresh projection of it.
    Written via a temp file then moved, so an interrupt cannot leave a partial
    file behind. The source dataset file is never modified.
    """
    stem = f"{dataset_path.stem}_transcribed"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_json = json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    tmp_json.replace(json_path)

    tmp_csv = csv_path.with_suffix(".csv.tmp")
    pd.DataFrame(enriched).to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    return json_path, csv_path


def print_summary(enriched: list[dict]) -> None:
    """Print usable-transcript counts and the status breakdown."""
    df = pd.DataFrame(enriched)
    ok = df[df["transcript_status"] == "ok"]

    print(f"\nUsable transcripts: {len(ok)}/{len(df)}")
    if len(ok):
        words = ok["transcript"].str.split().str.len()
        print(f"Median transcript length: {words.median():.0f} words")
    print("\nStatus breakdown:")
    print(df["transcript_status"].value_counts().to_string())
