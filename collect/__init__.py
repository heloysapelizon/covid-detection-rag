"""YouTube collection pipeline for the COVID-19 vaccine video dataset.
    python -m collect
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and editors only; no runtime import
    from .collector import collect, gather_from_links, gather_from_search, print_summary
    from .data import (
        dataset_paths,
        load_records,
        parse_links_file,
        save_dataset,
        to_records,
    )
    from .filters import apply_all, dedupe, drop_long
    from .main import main
    from .models import Video
    from .api import build_client, fetch_videos_by_id, parse_duration, search_videos

# public name -> module that defines it
_EXPORTS = {
    "Video": "models",
    "apply_all": "filters",
    "dedupe": "filters",
    "drop_long": "filters",
    "dataset_paths": "data",
    "load_records": "data",
    "parse_links_file": "data",
    "save_dataset": "data",
    "to_records": "data",
    "build_client": "api",
    "fetch_videos_by_id": "api",
    "parse_duration": "api",
    "search_videos": "api",
    "collect": "collector",
    "gather_from_links": "collector",
    "gather_from_search": "collector",
    "print_summary": "collector",
    "main": "main",
}

__all__ = [
    "Video",
    "apply_all",
    "build_client",
    "collect",
    "dataset_paths",
    "dedupe",
    "drop_long",
    "fetch_videos_by_id",
    "gather_from_links",
    "gather_from_search",
    "load_records",
    "main",
    "parse_duration",
    "parse_links_file",
    "print_summary",
    "save_dataset",
    "search_videos",
    "to_records",
]


def __getattr__(name: str):
    """Resolve a public name to its module on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
