"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DATA_DIR, DATASET_NAME, MAX_DURATION_S, SEARCH_QUERIES
from .collector import collect, load_excluded_ids, print_summary
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
    parser.add_argument("--exclude", metavar="FILE", default=None,
                        help="JSON file of video ids to skip — a ground-truth "
                             "label file or a plain list. Use it to keep the "
                             "held-out test videos out of a dev collection")
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

    exclude_ids: set[str] = set()
    if args.exclude:
        exclude_path = Path(args.exclude)
        if not exclude_path.exists():
            print(f"error: no exclude file at {exclude_path}", file=sys.stderr)
            return 1
        exclude_ids = load_excluded_ids(exclude_path)
        print(f"Excluding {len(exclude_ids)} video id(s) from {exclude_path}")

    records = collect(
        client,
        queries=None if args.no_search else SEARCH_QUERIES,
        links_file=links_file,
        name=args.name,
        max_duration_s=args.max_duration,
        exclude_ids=exclude_ids,
    )
    print_summary(records)
    return 0 if records else 1
