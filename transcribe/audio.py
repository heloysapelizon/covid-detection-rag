"""Everything that touches yt-dlp and ffmpeg: getting audio off YouTube."""
from __future__ import annotations

import shutil
from pathlib import Path

import yt_dlp

from .config import AUDIO_DIR, MAX_DURATION_S, TRANSCRIBE_MAX_S


def ffmpeg_available() -> bool:
    """Whether ffmpeg is on PATH. yt-dlp needs it to produce WAV."""
    return shutil.which("ffmpeg") is not None


def download_audio(
    video_id: str,
    max_duration_s: int = MAX_DURATION_S,
    transcribe_max_s: int = TRANSCRIBE_MAX_S,
) -> tuple[Path | None, str]:
    """Download one video's audio as 16kHz mono WAV."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = AUDIO_DIR / video_id
    pp_args = ["-ar", "16000", "-ac", "1"]
    if transcribe_max_s:
        pp_args += ["-t", str(int(transcribe_max_s))]
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
        "postprocessor_args": pp_args,
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    duration = None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration") if info is not None else None
            if duration is not None and duration > max_duration_s:
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
    if transcribe_max_s and duration and duration > transcribe_max_s:
        return wav, (f"ok_capped ({transcribe_max_s // 60}min of "
                     f"{duration // 60}min)")
    return wav, "ok"
