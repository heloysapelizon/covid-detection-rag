"""Settings and paths shared by the transcription pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"

# Whisper settings
MODEL_SIZE = "small"        # tiny | base | small | medium | large-v3
COMPUTE_TYPE = "int8"       # int8 is fastest on CPU; float32 slightly more accurate
DEVICE = "cpu"              # CTranslate2 has no MPS backend; CPU is the only option on Mac
LANG_PROBABILITY_MIN = 0.5  # below this confidence, treat detection as unreliable
WORD_TIMESTAMPS = False     # True adds per-word timings inside each segment (slower)

# Download settings
AUDIO_DIR = Path(tempfile.gettempdir()) / "whisper_audio"
MAX_DURATION_S = 3600

# Progress is checkpointed here so an interrupted run can resume.
CHECKPOINT_PATH = DATA_DIR / "transcripts_checkpoint.json"
