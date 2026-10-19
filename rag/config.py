"""Settings and paths shared by the RAG pipeline."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# The transcribed dataset to fact-check.
DATASET_NAME = "covid_vaccine_videos_transcribed"

# --- Local model server --------------------------------------------------
# Any server exposing an OpenAI-compatible /v1/chat/completions works:
#   Ollama       ollama serve                                  (port 11434)
#   llama.cpp    llama-server -m model.gguf -c 8192 --port 8080
LLAMA_URL = "http://localhost:11434"
# Ollama needs a real model name; llama-server ignores it.
# 14b scores 11/12 on tests/stance_eval.py vs 9/12 for 7b — but still only 2/5 on
# whole videos, so verdicts are candidates for review, not final labels.
LLAMA_MODEL = "qwen2.5:14b"
REQUEST_TIMEOUT_S = 300      # local generation can be slow
TEMPERATURE = 0.0            # deterministic: this is classification, not writing

# llama.cpp enforces a GBNF grammar during sampling, making non-conforming output
# unrepresentable. Ollama ignores `grammar` but supports JSON mode instead, so the
# request adapts to whichever server is in use.
USE_GRAMMAR = False          # True only when pointing at llama.cpp
USE_JSON_MODE = True         # Ollama / OpenAI-style response_format

# --- Retrieval -----------------------------------------------------------
# Runs locally via sentence-transformers; small, fast, good enough for short claims.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4                    # evidence passages retrieved per claim
MIN_SIMILARITY = 0.25        # below this, treat as "no relevant evidence"

# --- Claim segmentation --------------------------------------------------
# Whisper output is largely unpunctuated, so transcripts are split into
# overlapping word windows rather than sentences.
#
# Windows are deliberately wide: a claim severed from its own negation reads as
# an assertion, which produced false positives on debunking speakers at 60 words
# ("the genetic changes are the type of thing our body does all the time" is
# reassurance, but only if the sentence that follows is in the same window).
# Heavy overlap means a claim spanning a boundary still appears whole somewhere.
CLAIM_WINDOW_WORDS = 140
CLAIM_STRIDE_WORDS = 70
