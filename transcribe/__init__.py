"""Whisper transcription pipeline for the COVID-19 vaccine video dataset.

Each stage lives in its own module:

    config       settings and paths
    dataset      load a collected dataset JSON
    model        load the Whisper model
    download     video id -> 16kHz mono WAV
    transcriber  WAV -> text, plus the command-line entry point
    checkpoint   resume support
    runner       the per-video loop
    output       merge transcripts into records and save

Run it with:

    python -m transcribe covid_vaccine_videos
"""
from .checkpoint import checkpoint_path_for, load_checkpoint, save_checkpoint
from .dataset import load_dataset, resolve_dataset_path
from .download import download_audio
from .model import load_model
from .output import build_enriched, print_summary, save_enriched
from .runner import run
from .transcriber import main, transcribe_audio

__all__ = [
    "checkpoint_path_for",
    "load_checkpoint",
    "save_checkpoint",
    "load_dataset",
    "resolve_dataset_path",
    "download_audio",
    "load_model",
    "build_enriched",
    "print_summary",
    "save_enriched",
    "run",
    "main",
    "transcribe_audio",
]
