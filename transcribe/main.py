"""
    python -m transcribe covid_vaccine_videos
    python -m transcribe covid_vaccine_videos.json
    python -m transcribe data/youtube/disinformation_dozen.json

Progress is checkpointed after every video, so re-running the same dataset resumes where it stopped.

"""
from __future__ import annotations

import argparse
import sys

from .audio import ffmpeg_available
from .config import DATA_DIR, MODEL_SIZE
from .transcriber import print_summary, run
from .whisper import load_model
from .data import (
    build_enriched,
    checkpoint_path_for,
    list_datasets,
    load_checkpoint,
    load_dataset,
    save_enriched,
)


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
        print(f"Datasets in {DATA_DIR}:")
        for name in list_datasets():
            print(f"  {name}")
        return 0

    if not args.dataset:
        parser.print_usage()
        print("\nerror: a dataset is required (or use --list)", file=sys.stderr)
        return 2

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
        # Only the transcribing path needs ffmpeg; --summary-only must not fail on it.
        if not ffmpeg_available():
            print("error: ffmpeg not found on PATH — install it with: brew install ffmpeg",
                  file=sys.stderr)
            return 1

        model = load_model(args.model)
        results = run(model, dataset, results, checkpoint_file, limit=args.limit)

    enriched = build_enriched(dataset, results)
    json_path, csv_path = save_enriched(enriched, dataset_path)
    print(f"\nSaved:\n  {json_path}\n  {csv_path}")
    print_summary(enriched)
    return 0
