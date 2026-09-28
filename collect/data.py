"""All filesystem I/O: link lists in, datasets out.

Grouped by concern — reading and writing under DATA_DIR — and by one reliability
rule: every write goes to a temp file and is then moved into place, so an interrupt
can never leave a half-written dataset.

Depends only on the standard library and pandas, so it imports fast and can be used
without an API key or network access.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DATASET_NAME
from .models import Video

# watch?v=, youtu.be/, /shorts/, /embed/, /live/, or a bare id.
_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/(?:shorts|embed|live)/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),
]

# write helpers
def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def parse_links_file(path: Path) -> tuple[list[str], list[str]]:
    """Read a .txt of one link (or bare id) per line.

    Returns (video_ids, unrecognised_lines). Blank lines and #-comments are
    ignored, inline comments stripped, ids deduplicated in file order.
    """
    if not path.exists():
        raise FileNotFoundError(f"No links file at {path}")

    ids: list[str] = []
    seen: set[str] = set()
    bad: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for pat in _ID_PATTERNS:
            m = pat.search(line)
            if m:
                if m.group(1) not in seen:
                    seen.add(m.group(1))
                    ids.append(m.group(1))
                break
        else:
            bad.append(raw.strip())
    return ids, bad


# --------------------------------------------------------------------------
# Datasets (read + write)
# --------------------------------------------------------------------------


def dataset_paths(name: str = DATASET_NAME, out_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """The (json, csv) paths for a dataset name."""
    return out_dir / f"{name}.json", out_dir / f"{name}.csv"


def load_records(path: Path) -> list[dict]:
    """Read a dataset JSON as a list of records."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a list of records")
    return records


def to_records(videos: list[Video]) -> list[dict]:
    """Convert Video objects to plain dicts."""
    return [asdict(v) for v in videos]


def save_dataset(
    records: list[dict],
    name: str = DATASET_NAME,
    out_dir: Path = DATA_DIR,
) -> tuple[Path, Path]:
    """Write <name>.json and <name>.csv, replacing them if they exist."""
    json_path, csv_path = dataset_paths(name, out_dir)
    _write_json(json_path, records)
    _write_csv(csv_path, records)
    return json_path, csv_path
