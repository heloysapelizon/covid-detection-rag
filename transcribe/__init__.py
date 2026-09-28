"""Whisper transcription pipeline for the COVID-19 vaccine video dataset.

Modules are grouped by what they depend on, so a failure or a slow import stays
contained and each layer can be used without the ones above it:

    config       settings and paths — no dependencies
    data         all disk I/O: datasets in, checkpoints both ways, results out
                 (stdlib + pandas; every write is atomic)
    audio        yt-dlp + ffmpeg: YouTube video id -> 16kHz mono WAV
    whisper      faster-whisper: load a model, turn a WAV into text
    transcriber  the per-video loop and the run report
    main         argument parsing and exit codes

Dependencies point one way only: main -> transcriber -> {audio, whisper, data} ->
config. Nothing imports the layer above it.

Names are re-exported lazily (PEP 562), so `from transcribe.data import
load_dataset` does not pull in yt-dlp or faster-whisper. Import the submodule you
need directly to keep it that way.

Run it with:

    python -m transcribe covid_vaccine_videos
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and editors only; no runtime import
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

# public name -> module that defines it
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
