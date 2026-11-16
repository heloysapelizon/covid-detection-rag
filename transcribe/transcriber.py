from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

from .config import is_usable, TRANSCRIBE_MAX_S
from .audio import download_audio
from .whisper import transcribe_audio
from .data import save_checkpoint


def run(
    model: WhisperModel,
    dataset: list[dict],
    results: dict,
    checkpoint_file: Path,
    limit: int | None = None,
    resegment: bool = False,
    max_transcribe_s: int = TRANSCRIBE_MAX_S,
) -> dict:
    """Transcribe every dataset video absent from `results`."""
    def pending(vid: str) -> bool:
        r = results.get(vid)
        if r is None:
            return True
        return resegment and is_usable(r.get("transcript_status")) and "segments" not in r

    todo = [d for d in dataset if pending(d["video_id"])]
    if limit is not None:
        todo = todo[:limit]

    if not todo:
        print("Nothing to do — every video is already in the checkpoint.")
        return results

    print(f"{len(todo)} video(s) to process "
          f"({len(results)} already done of {len(dataset)}).\n")

    start = time.time()
    for n, d in enumerate(todo, 1):
        vid = d["video_id"]
        print(f"[{n}/{len(todo)}] {d.get('title', '')[:60]}")

        wav, dl_status = download_audio(vid, transcribe_max_s=max_transcribe_s)
        if wav is None:
            results[vid] = {"transcript": "", "transcript_status": dl_status,
                            "meta": {}, "segments": []}
            print(f"    -> {dl_status}")
        else:
            text, status, meta, segments = transcribe_audio(model, wav)
            if status == "ok" and dl_status.startswith("ok_capped"):
                status = dl_status
            results[vid] = {
                "transcript": text,
                "transcript_status": status,
                "meta": meta,
                "segments": segments,
            }
            wav.unlink(missing_ok=True)
            print(f"    -> {status}  ({len(text.split())} words, {len(segments)} segments, "
                  f"{meta.get('duration_s', 0):.0f}s audio)")

        save_checkpoint(checkpoint_file, results)

        if n % 10 == 0 and n < len(todo):
            rate = (time.time() - start) / n
            print(f"\n  --- {n}/{len(todo)} done, "
                  f"~{(len(todo) - n) * rate / 60:.0f} min remaining ---\n")

    print(f"\nFinished {len(todo)} video(s) in {(time.time() - start) / 60:.1f} min.")
    return results


def print_summary(enriched: list[dict]) -> None:
    """Print usable-transcript counts and the status breakdown."""
    df = pd.DataFrame(enriched)
    ok = df[df["transcript_status"].apply(is_usable)]

    print(f"\nUsable transcripts: {len(ok)}/{len(df)}")
    if len(ok):
        words = ok["transcript"].str.count(r"\S+")
        print(f"Median transcript length: {words.median():.0f} words")
    print("\nStatus breakdown:")
    print(df["transcript_status"].value_counts().to_string())
