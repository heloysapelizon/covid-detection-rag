"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import (
    DATA_DIR,
    DATASET_NAME,
    INDEX_DIR,
    LLAMA_URL,
    MIN_SIMILARITY,
    TOP_K,
)
from transcribe.config import is_usable  # noqa: E402

from .index import EvidenceIndex, build
from .llm import health
from .pipeline import check_transcript, print_report
from .websearch import get_backend


def _start_hint() -> str:
    """How to start whichever server this run is configured against."""
    if os.environ.get("LLM_SERVER", "").lower() == "ollama" or "11434" in LLAMA_URL:
        return "  ollama serve"
    return "  llama-server -m model.gguf -c 8192 --port 8080"


def load_transcripts() -> dict[str, dict]:
    """Return {video_id: record} for videos that have a usable transcript."""
    transcribed = DATA_DIR / f"{DATASET_NAME}.json"
    if transcribed.exists():
        records = json.loads(transcribed.read_text(encoding="utf-8"))
        out = {r["video_id"]: r for r in records
               if is_usable(r.get("transcript_status")) and r.get("transcript")}
        if out:
            return out

    checkpoint = DATA_DIR / "transcripts_checkpoint.json"
    if not checkpoint.exists():
        return {}
    cp = json.loads(checkpoint.read_text(encoding="utf-8"))
    meta: dict[str, dict] = {}
    for name in ("covid_vaccine_videos.json",):
        p = DATA_DIR / name
        if p.exists():
            meta.update({r["video_id"]: r
                         for r in json.loads(p.read_text(encoding="utf-8"))})
    return {
        vid: {**meta.get(vid, {"video_id": vid}), "transcript": v["transcript"]}
        for vid, v in cp.items()
        if is_usable(v.get("transcript_status")) and v.get("transcript")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="Fact-check a transcript against vectorized evidence "
                    "using a local llama.cpp model.",
    )
    parser.add_argument("video_id", nargs="?",
                        help="video id to fact-check (see --list)")
    parser.add_argument("--build-manifest", action="store_true",
                        help="regenerate data/evidence/sources.json from the "
                             "DOCUMENTS declaration in rag/manifest.py, then exit")
    parser.add_argument("--no-extract", action="store_true",
                        help="with --build-index, embed the committed passages "
                             "without re-reading the source PDFs (which are "
                             "gitignored, so this is the path on a clone)")
    parser.add_argument("--build-index", action="store_true",
                        help="embed the evidence corpus and exit")
    parser.add_argument("--check-server", action="store_true",
                        help="check the llama.cpp server and exit")
    parser.add_argument("--list", action="store_true",
                        help="list videos with usable transcripts and exit")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"evidence passages per claim (default: {TOP_K})")
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY,
                        help=f"retrieval floor (default: {MIN_SIMILARITY})")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="only check the first N claim windows")
    parser.add_argument("--json", metavar="FILE", default=None,
                        help="write full results to this JSON file")
    parser.add_argument("--web", default="none", metavar="BACKEND",
                        help="web-search fallback for claims the corpus cannot "
                             "address: duckduckgo, tavily, or none (default). "
                             "tavily needs TAVILY_API_KEY")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-window progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.build_manifest:
        from .manifest import build as build_manifest
        build_manifest()
        return 0

    if args.build_index:
        n = build(extract=not args.no_extract)
        print(f"Indexed {n} evidence passage(s) into {INDEX_DIR}")
        return 0

    if args.check_server:
        ok, detail = health()
        print(("OK   " if ok else "FAIL ") + detail)
        if not ok:
            print(f"\nStart the server this run is configured for "
                  f"({LLAMA_URL}):\n{_start_hint()}")
        return 0 if ok else 1

    transcripts = load_transcripts()

    if args.list:
        if not transcripts:
            print("No usable transcripts found. Run transcribe first.")
            return 1
        print(f"{len(transcripts)} video(s) with usable transcripts:")
        for vid, r in transcripts.items():
            words = len(r["transcript"].split())
            print(f"  {vid}  {words:6d}w  {r.get('channel', '?')[:22]:22} "
                  f"{r.get('title', '?')[:40]}")
        return 0

    if not args.video_id:
        build_parser().print_usage()
        print("\nerror: a video_id is required (or use --list)", file=sys.stderr)
        return 2

    if args.video_id not in transcripts:
        print(f"error: no usable transcript for {args.video_id!r}. "
              f"Use --list to see available videos.", file=sys.stderr)
        return 1

    ok, detail = health()
    if not ok:
        print(f"error: {detail}", file=sys.stderr)
        print(f"Start the server this run is configured for ({LLAMA_URL}):\n"
              f"{_start_hint()}", file=sys.stderr)
        return 1

    try:
        index = EvidenceIndex.load()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    record = transcripts[args.video_id]
    transcript = record["transcript"]

    print(f"model server : {detail}")
    print(f"evidence     : {len(index.passages)} passage(s), {index.model_name}")
    print(f"video        : {args.video_id}  "
          f"({len(transcript.split())} words)\n")

    if args.limit:
        from .pipeline import segment_claims
        windows = segment_claims(transcript)[:args.limit]
        transcript = " ".join(windows[0].split()) if len(windows) == 1 \
            else " ".join(w.split()[:45] for w in windows)

    try:
        web_fetch_fn = get_backend(args.web)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if web_fetch_fn is not None and not args.quiet:
        print(f"web fallback : {args.web or 'duckduckgo'}")

    results = check_transcript(
        transcript, index,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        verbose=not args.quiet,
        web_fetch_fn=web_fetch_fn,
    )
    print_report(results, meta=record)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps({
            "video_id": args.video_id,
            "title": record.get("title", ""),
            "channel": record.get("channel", ""),
            "url": record.get("url", ""),
            "evidence_model": index.model_name,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Full results written to {out}")

    return 0
