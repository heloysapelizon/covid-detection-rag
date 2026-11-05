"""Everything that touches yt-dlp and ffmpeg: getting audio off YouTube.

The external-tool boundary of the pipeline, and the only part that depends on the
network. Kept separate so a yt-dlp or ffmpeg problem is isolated to one module,
and so the rest of the package imports without them.
"""
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
    """Download one video's audio as 16kHz mono WAV.

    Returns (path, status). path is None when the download failed; status is
    "ok" or a short reason suitable for the transcript_status column. Failures
    are returned rather than raised, so one bad video cannot end a long run.

    `transcribe_max_s` trims the WAV to its first N seconds, discarding the rest
    (0 keeps the whole video). The trim happens here, in ffmpeg, rather than
    after transcription: cutting the audio means Whisper never loads the
    remainder, so the saving is in inference time and not only in storage.
    Status becomes "ok_capped (<n>min of <total>min)" when a trim occurred, so
    the dataset records which transcripts are partial.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = AUDIO_DIR / video_id
    pp_args = ["-ar", "16000", "-ac", "1"]
    if transcribe_max_s:
        # -t limits output duration. Placed with the other audio args so it
        # applies to the WAV that FFmpegExtractAudio writes.
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
