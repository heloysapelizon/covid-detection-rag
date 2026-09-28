"""Command-line entry point.

    python -m collect                          # search the configured queries
    python -m collect --links disinformation_dozen.txt
    python -m collect --links disinformation_dozen.txt --no-search
    python -m collect --name my_dataset --max-duration 1800

Writes data/youtube/<name>.json and .csv — the dataset `transcribe` reads.

Argument parsing and exit codes only; the work lives in collector.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DATA_DIR, DATASET_NAME, MAX_DURATION_S, SEARCH_QUERIES
from .collector import collect, print_summary
from .api import build_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m collect",
        description="Collect COVID-19 vaccine videos into a filtered dataset.",
    )
    parser.add_argument("--links", metavar="FILE", default=None,
                        help="a .txt of video links to include "
                             "(bare name looked up in data/youtube/)")
    parser.add_argument("--no-search", action="store_true",
                        help="skip the search queries; use --links only")
    parser.add_argument("--name", default=DATASET_NAME,
                        help=f"output dataset name (default: {DATASET_NAME})")
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION_S,
                        metavar="SECONDS",
                        help=f"drop videos longer than this (default: {MAX_DURATION_S})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_search and not args.links:
        print("error: --no-search needs --links, or there is nothing to collect",
              file=sys.stderr)
        return 2

    links_file = None
    if args.links:
        links_file = Path(args.links)
        if not links_file.is_absolute() and links_file.parent == Path("."):
            links_file = DATA_DIR / links_file.name
        if not links_file.exists():
            print(f"error: no links file at {links_file}", file=sys.stderr)
            return 1

    try:
        client = build_client()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    records = collect(
        client,
        queries=None if args.no_search else SEARCH_QUERIES,
        links_file=links_file,
        name=args.name,
        max_duration_s=args.max_duration,
    )
    print_summary(records)
    return 0 if records else 1
