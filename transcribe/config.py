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

# Transcribe at most this many seconds per video, discarding the remainder.
# 0 disables the cap and transcribes whole videos.
#
# For the dev set the sweeps only need enough speech to yield labelled windows,
# not complete transcripts. Measured on the 124 v1 videos: a 12-minute cap
# leaves 104 of them untouched and still keeps 74% of all words, while cutting
# total audio from 781 to 612 minutes (22% less inference).
#
# A cap rather than sampling slices across the video, deliberately: the claim
# segmentation in rag/pipeline.py relies on overlapping windows, and a
# transcript assembled from disjoint slices would place a boundary mid-argument.
# A claim severed from its own negation reads as an assertion, which is the
# documented false-positive case (see CLAIM_WINDOW_WORDS in rag/config.py), so
# gaps would manufacture exactly the error the eval is meant to detect.
#
# Known cost, stated because it biases the dev set: topical claims are not
# front-loaded. On the v1 labels the median labelled topical window sits 33% of
# the way into its video, and only 15 of 36 fall within the first quarter — so
# capping does lose real claims in long podcasts. It is acceptable here because
# the cap is for the DEV set, where the measurement is chunk size and routing
# rather than per-video recall, and because the thin topics this collection
# targets are mostly short news clips. The held-out TEST set
# (data/labels/ground_truth.json) was transcribed in full and must stay that
# way.
TRANSCRIBE_MAX_S = 12 * 60

# Progress is checkpointed here so an interrupted run can resume.
CHECKPOINT_PATH = DATA_DIR / "transcripts_checkpoint.json"


def is_usable(status: str | None) -> bool:
    """Whether a transcript_status means the transcript can be used.

    "ok" is a complete transcript; "ok_capped (...)" is the first
    TRANSCRIBE_MAX_S of a longer video. Both are usable — the cap is a
    deliberate truncation, not a failure — so every consumer should test this
    rather than `status == "ok"`, which silently discards capped videos.
    """
    return bool(status) and status.startswith("ok")
