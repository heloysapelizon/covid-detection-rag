"""All filesystem I/O: datasets in, checkpoints both ways, results out.

Grouped together because these share one concern — reading and writing JSON/CSV
under DATA_DIR — and one reliability rule: every write goes to a temp file and is
then moved into place, so an interrupt can never leave a half-written file.

Depends only on the standard library and pandas, so it imports fast and can be
exercised without yt-dlp, ffmpeg, or a Whisper model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CHECKPOINT_PATH, DATA_DIR

# --------------------------------------------------------------------------
# Atomic write helpers
# --------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    """Write JSON via a temp file, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV via a temp file, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


# --------------------------------------------------------------------------
# Datasets (read-only)
# --------------------------------------------------------------------------


def list_datasets() -> list[str]:
    """Dataset filenames available in DATA_DIR, checkpoints excluded."""
    return sorted(
        p.name for p in DATA_DIR.glob("*.json")
        if not p.name.startswith("transcripts_checkpoint")
    )


def resolve_dataset_path(name: str) -> Path:
    """Turn a user-supplied dataset name into a path.

    Accepts a bare name ("covid_vaccine_videos"), a filename with extension, or
    a full/relative path. Bare names are looked up inside DATA_DIR.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return DATA_DIR / candidate.name


def load_dataset(name: str) -> tuple[Path, list[dict]]:
    """Return (path, records) for a dataset JSON.

    Raises FileNotFoundError listing the available datasets, since a typo'd name
    is the most likely failure here.
    """
    path = resolve_dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}\n"
            f"Available in {DATA_DIR}:\n  " + "\n  ".join(list_datasets())
        )

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} is not a non-empty list of records")
    if "video_id" not in records[0]:
        raise ValueError(f"{path} records have no video_id field")

    return path, records


# --------------------------------------------------------------------------
# Checkpoints (read + write)
# --------------------------------------------------------------------------


def checkpoint_path_for(dataset_path: Path) -> Path:
    """Checkpoint file for a dataset.

    The original single-dataset checkpoint keeps its name so existing progress is
    still found; any other dataset gets its own file, so runs over different
    datasets do not overwrite each other's results.
    """
    if dataset_path.name.startswith("covid_vaccine_videos") and dataset_path.parent == DATA_DIR:
        return CHECKPOINT_PATH
    return DATA_DIR / f"transcripts_checkpoint_{dataset_path.stem}.json"


def load_checkpoint(path: Path) -> dict:
    """Return {video_id: result} from a previous run, or empty."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, results: dict) -> None:
    """Persist results so an interrupt loses at most one video's work."""
    _write_json(path, results)


# --------------------------------------------------------------------------
# Results (write)
# --------------------------------------------------------------------------


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
        row["segments"] = r.get("segments", [])
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
    The source dataset file is never modified.
    """
    stem = f"{dataset_path.stem}_transcribed"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    _write_json(json_path, enriched)
    # Segments are nested lists and stay JSON-only; the CSV is for inspection.
    _write_csv(csv_path, [{k: v for k, v in r.items() if k != "segments"} for r in enriched])

    return json_path, csv_path
