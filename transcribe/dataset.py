"""Load the collected video dataset."""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR


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

    Raises FileNotFoundError with the available options listed, since a typo'd
    name is the most likely failure here.
    """
    path = resolve_dataset_path(name)
    if not path.exists():
        available = sorted(
            p.name for p in DATA_DIR.glob("*.json")
            if p.name != "transcripts_checkpoint.json"
        )
        raise FileNotFoundError(
            f"No dataset at {path}\n"
            f"Available in {DATA_DIR}:\n  " + "\n  ".join(available)
        )

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} is not a non-empty list of records")
    if "video_id" not in records[0]:
        raise ValueError(f"{path} records have no video_id field")

    return path, records
