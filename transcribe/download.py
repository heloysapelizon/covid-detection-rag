"""Download a video's audio as the 16 kHz mono WAV Whisper expects."""
from __future__ import annotations

from pathlib import Path

import yt_dlp

from .config import AUDIO_DIR, MAX_DURATION_S


def download_audio(video_id: str, max_duration_s: int = MAX_DURATION_S) -> tuple[Path | None, str]:
    """Download one video's audio as 16kHz mono WAV.

    Returns (path, status). path is None when the download failed; status is
    "ok" or a short reason suitable for the transcript_status column.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = AUDIO_DIR / video_id
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_stem) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        # Whisper wants 16kHz mono; converting here avoids a second ffmpeg pass.
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration") or 0
            if duration > max_duration_s:
                return None, f"skipped_too_long ({duration // 60}min)"
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "private" in msg:
            return None, "unavailable_private"
        if "removed" in msg or "not available" in msg:
            return None, "unavailable_removed"
        if "age" in msg:
            return None, "unavailable_age_restricted"
        return None, "download_error"
    except Exception as e:
        return None, f"download_error: {type(e).__name__}"

    wav = out_stem.with_suffix(".wav")
    if not wav.exists():
        return None, "download_error_no_file"
    return wav, "ok"
