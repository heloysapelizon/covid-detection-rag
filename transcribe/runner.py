"""Drive the per-video loop: download, transcribe, checkpoint, clean up."""
from __future__ import annotations

import time
from pathlib import Path

from faster_whisper import WhisperModel

from .checkpoint import save_checkpoint
from .download import download_audio


def run(
    model: WhisperModel,
    dataset: list[dict],
    results: dict,
    checkpoint_file: Path,
    limit: int | None = None,
) -> dict:
    """Transcribe every dataset video absent from `results`.

    Checkpoints after each video and deletes each audio file immediately, so
    disk use stays flat and an interrupt loses at most one video's work.
    Returns the updated results dict (also mutated in place).
    """
    # Imported here, not at module scope: transcriber imports run() for its CLI,
    # so a top-level import in either direction would be circular.
    from .transcriber import transcribe_audio

    todo = [d for d in dataset if d["video_id"] not in results]
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

        wav, status = download_audio(vid)
        if wav is None:
            results[vid] = {"transcript": "", "transcript_status": status, "meta": {}}
            print(f"    -> {status}")
        else:
            text, status, meta = transcribe_audio(model, wav)
            results[vid] = {
                "transcript": text,
                "transcript_status": status,
                "meta": meta,
            }
            wav.unlink(missing_ok=True)
            print(f"    -> {status}  ({len(text.split())} words, "
                  f"{meta.get('duration_s', 0):.0f}s audio)")

        save_checkpoint(checkpoint_file, results)

        if n % 10 == 0 and n < len(todo):
            rate = (time.time() - start) / n
            print(f"\n  --- {n}/{len(todo)} done, "
                  f"~{(len(todo) - n) * rate / 60:.0f} min remaining ---\n")

    print(f"\nFinished {len(todo)} video(s) in {(time.time() - start) / 60:.1f} min.")
    return results
