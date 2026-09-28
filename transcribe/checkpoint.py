"""Persist progress so an interrupted run resumes instead of starting over."""
from __future__ import annotations

import json
from pathlib import Path

from .config import CHECKPOINT_PATH, DATA_DIR


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
    """Write results atomically so an interrupt can't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
