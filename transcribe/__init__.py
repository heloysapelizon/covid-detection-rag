"""Whisper transcription pipeline for the COVID-19 vaccine video dataset."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import download_audio, ffmpeg_available
    from .main import main
    from .transcriber import print_summary, run
    from .whisper import load_model, transcribe_audio
    from .data import (
        build_enriched,
        checkpoint_path_for,
        list_datasets,
        load_checkpoint,
        load_dataset,
        resolve_dataset_path,
        save_checkpoint,
        save_enriched,
    )

_EXPORTS = {
    "download_audio": "audio",
    "ffmpeg_available": "audio",
    "load_model": "whisper",
    "transcribe_audio": "whisper",
    "build_enriched": "data",
    "checkpoint_path_for": "data",
    "list_datasets": "data",
    "load_checkpoint": "data",
    "load_dataset": "data",
    "resolve_dataset_path": "data",
    "save_checkpoint": "data",
    "save_enriched": "data",
    "run": "transcriber",
    "print_summary": "transcriber",
    "main": "main",
}

__all__ = [
    "build_enriched",
    "checkpoint_path_for",
    "download_audio",
    "ffmpeg_available",
    "list_datasets",
    "load_checkpoint",
    "load_dataset",
    "load_model",
    "main",
    "print_summary",
    "resolve_dataset_path",
    "run",
    "save_checkpoint",
    "save_enriched",
    "transcribe_audio",
]


def __getattr__(name: str):
    """Resolve a public name to its module on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
