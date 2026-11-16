"""Settings and paths shared by the transcription pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"

MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
LANG_PROBABILITY_MIN = 0.5
WORD_TIMESTAMPS = False

AUDIO_DIR = Path(tempfile.gettempdir()) / "whisper_audio"
MAX_DURATION_S = 3600

TRANSCRIBE_MAX_S = 12 * 60

CHECKPOINT_PATH = DATA_DIR / "transcripts_checkpoint.json"


def is_usable(status: str | None) -> bool:
    """Whether a transcript_status means the transcript can be used."""
    return bool(status) and status.startswith("ok")
