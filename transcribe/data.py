"""All filesystem I/O: datasets in, checkpoints both ways, results out."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CHECKPOINT_PATH, DATA_DIR


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


def list_datasets() -> list[str]:
    """Dataset filenames available in DATA_DIR, checkpoints excluded."""
    return sorted(
        p.name for p in DATA_DIR.glob("*.json")
        if not p.name.startswith("transcripts_checkpoint")
    )


def resolve_dataset_path(name: str) -> Path:
    """Turn a user-supplied dataset name into a path."""
    candidate = Path(name)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return DATA_DIR / candidate.name


def load_dataset(name: str) -> tuple[Path, list[dict]]:
    """Return (path, records) for a dataset JSON."""
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


_LEGACY_DATASET = "covid_vaccine_videos"


def checkpoint_path_for(dataset_path: Path) -> Path:
    """Checkpoint file for a dataset."""
    if dataset_path.parent == DATA_DIR and dataset_path.stem == _LEGACY_DATASET:
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


def build_enriched(dataset: list[dict], results: dict) -> list[dict]:
    """Return dataset records with transcript columns appended."""
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
    """Write <dataset>_transcribed.json/.csv, replacing them if they exist."""
    stem = f"{dataset_path.stem}_transcribed"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    _write_json(json_path, enriched)
    _write_csv(csv_path, [{k: v for k, v in r.items() if k != "segments"} for r in enriched])

    return json_path, csv_path
