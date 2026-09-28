"""Transcribe one audio file, and the command-line entry point.

    python -m transcribe covid_vaccine_videos
    python -m transcribe covid_vaccine_videos.json
    python -m transcribe data/youtube/disinformation_dozen.json

A bare name is looked up in data/youtube/. Progress is checkpointed after every
video, so re-running the same dataset resumes where it stopped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from faster_whisper import WhisperModel

from .checkpoint import checkpoint_path_for, load_checkpoint
from .config import DATA_DIR, LANG_PROBABILITY_MIN, MODEL_SIZE, TARGET_LANGUAGE
from .dataset import load_dataset
from .model import load_model
from .output import build_enriched, print_summary, save_enriched
from .runner import run


def transcribe_audio(model: WhisperModel, wav_path: Path) -> tuple[str, str, dict]:
    """Transcribe one WAV file.

    Returns (text, status, meta) where meta carries the detected language and
    probability for auditing which videos were dropped and why.

    Language is detected before transcription, so non-target audio costs a
    detection pass rather than full inference.
    """
    try:
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        meta = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_s": round(info.duration, 1),
        }

        if info.language != TARGET_LANGUAGE:
            return "", f"skipped_language ({info.language})", meta
        if info.language_probability < LANG_PROBABILITY_MIN:
            return "", f"skipped_language_uncertain ({info.language_probability:.2f})", meta

        # segments is a generator; consuming it is what actually runs inference.
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, ("ok" if text else "empty"), meta

    except Exception as e:
        return "", f"transcribe_error: {type(e).__name__}", {}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def list_datasets() -> None:
    names = sorted(
        p.name for p in DATA_DIR.glob("*.json")
        if not p.name.startswith("transcripts_checkpoint")
    )
    print(f"Datasets in {DATA_DIR}:")
    for n in names:
        print(f"  {n}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m transcribe",
        description="Transcribe a collected video dataset with Whisper.",
    )
    parser.add_argument("dataset", nargs="?",
                        help="dataset name or path (e.g. covid_vaccine_videos)")
    parser.add_argument("--model", default=MODEL_SIZE,
                        help=f"Whisper model size (default: {MODEL_SIZE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N videos this run")
    parser.add_argument("--list", action="store_true",
                        help="list available datasets and exit")
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild output from the checkpoint without transcribing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_datasets()
        return 0

    if not args.dataset:
        parser.print_usage()
        print("\nerror: a dataset is required (or use --list)", file=sys.stderr)
        return 2

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH — install it with: brew install ffmpeg",
              file=sys.stderr)
        return 1

    try:
        dataset_path, dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    checkpoint_file = checkpoint_path_for(dataset_path)
    results = load_checkpoint(checkpoint_file)

    print(f"Dataset:    {dataset_path}  ({len(dataset)} videos)")
    print(f"Checkpoint: {checkpoint_file}  ({len(results)} done)")

    if not args.summary_only:
        model = load_model(args.model)
        results = run(model, dataset, results, checkpoint_file, limit=args.limit)

    enriched = build_enriched(dataset, results)
    json_path, csv_path = save_enriched(enriched, dataset_path)
    print(f"\nSaved:\n  {json_path}\n  {csv_path}")
    print_summary(enriched)
    return 0
