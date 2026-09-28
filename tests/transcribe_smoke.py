"""Exploratory: transcribe a single video end to end.

smoke test for faster whisper transcription

    python tests/transcribe_smoke.py                    # first untranscribed video
    python tests/transcribe_smoke.py u1MIKNhPz24        # a specific video
    python tests/transcribe_smoke.py u1MIKNhPz24 medium # with a different model

Does not write to the checkpoint or any dataset — output goes to stdout only.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"
AUDIO_DIR = Path(tempfile.gettempdir()) / "whisper_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "small"
COMPUTE_TYPE = "int8"

if not shutil.which("ffmpeg"):
    sys.exit("ffmpeg not found on PATH — brew install ffmpeg")


def pick_video_id() -> tuple[str, str]:
    """Return (video_id, title) — from argv, else the first untranscribed video."""
    if len(sys.argv) > 1:
        return sys.argv[1], ""

    saved = sorted(DATA_DIR.glob("covid_vaccine_videos_*.json"))
    if not saved:
        sys.exit(f"No dataset in {DATA_DIR} — run video_collection.ipynb first.")
    dataset = json.loads(saved[-1].read_text(encoding="utf-8"))

    checkpoint = DATA_DIR / "transcripts_checkpoint.json"
    done = set()
    if checkpoint.exists():
        done = set(json.loads(checkpoint.read_text(encoding="utf-8")))

    for d in dataset:
        if d["video_id"] not in done:
            return d["video_id"], d.get("title", "")
    sys.exit("Every video in the dataset is already in the checkpoint.")


def download_audio(video_id: str) -> Path | None:
    """Download audio as 16kHz mono WAV; None if the download fails."""
    out_stem = AUDIO_DIR / video_id
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_stem) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"  download failed: {type(e).__name__}: {str(e)[:200]}")
        return None
    wav = out_stem.with_suffix(".wav")
    return wav if wav.exists() else None


video_id, title = pick_video_id()
print(f"video: {video_id}  {title[:60]}")
print(f"  https://www.youtube.com/watch?v={video_id}")
print(f"  model: {MODEL_SIZE} ({COMPUTE_TYPE}, cpu)\n")

t0 = time.time()
wav = download_audio(video_id)
if wav is None:
    sys.exit(1)
print(f"  downloaded {wav.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s")

t0 = time.time()
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
print(f"  model loaded in {time.time() - t0:.1f}s")

t0 = time.time()
segments, info = model.transcribe(
    str(wav), beam_size=5, vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
)
print(f"  detected: {info.language} (p={info.language_probability:.3f}), "
      f"{info.duration:.0f}s audio")

text = " ".join(seg.text.strip() for seg in segments).strip()
elapsed = time.time() - t0
print(f"  transcribed in {elapsed:.1f}s "
      f"({info.duration / elapsed:.1f}x realtime), {len(text.split())} words\n")

print(text[:600] + ("..." if len(text) > 600 else ""))

wav.unlink(missing_ok=True)
