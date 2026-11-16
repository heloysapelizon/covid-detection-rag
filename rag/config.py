"""Settings and paths shared by the RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"

RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
RESULTS_VERSION = os.environ.get("RESULTS_VERSION", "v1")
RESULTS_DIR = RESULTS_ROOT / RESULTS_VERSION
SMOKE_DIR = RESULTS_ROOT / "smoke"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

PDF_DIR = EVIDENCE_DIR / "pdf"
PASSAGE_DIR = EVIDENCE_DIR / "passages"

CHUNK_WORDS = 55
CHUNK_STRIDE = 35

DATASET_NAME = "covid_vaccine_videos_transcribed"

_SERVER = os.environ.get("LLM_SERVER", "llamacpp").lower()
_IS_OLLAMA = _SERVER == "ollama"
_IS_OPENROUTER = _SERVER in ("openrouter", "open_router", "or")
_IS_LLAMACPP = not (_IS_OLLAMA or _IS_OPENROUTER)
IS_REMOTE = _IS_OPENROUTER

OPENROUTER_URL = "https://openrouter.ai/api/v1"
LLAMA_URL = os.environ.get(
    "LLAMA_URL",
    OPENROUTER_URL if _IS_OPENROUTER
    else "http://localhost:11434" if _IS_OLLAMA
    else "http://localhost:8080",
)
_DEFAULT_MODEL = ("qwen/qwen3.6-27b" if _IS_OPENROUTER else "qwen2.5:14b")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", _DEFAULT_MODEL)
REQUEST_TIMEOUT_S = int(os.environ.get(
    "LLM_TIMEOUT_S", "600" if _IS_OPENROUTER else "300"))
TEMPERATURE = 0.0

_CONSTRAINT = os.environ.get("LLM_CONSTRAINT", "").lower()

LLM_REASONING = os.environ.get("LLM_REASONING", "").strip().lower()
if LLM_REASONING not in ("", "off", "low", "medium", "high"):
    raise ValueError(f"LLM_REASONING must be off, low, medium or high, not {LLM_REASONING!r}")


def reasoning_tag() -> str:
    """Filename suffix for the reasoning setting: '' by default, '_nothink' off."""
    if not LLM_REASONING:
        return ""
    return "_nothink" if LLM_REASONING == "off" else f"_reasoning-{LLM_REASONING}"
USE_GRAMMAR = (_CONSTRAINT == "grammar") if _CONSTRAINT else _IS_LLAMACPP
USE_JSON_SCHEMA = (_CONSTRAINT == "json_schema") if _CONSTRAINT else _IS_OPENROUTER
USE_JSON_MODE = (_CONSTRAINT == "json_object") if _CONSTRAINT else _IS_OLLAMA

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MIN_SIMILARITY = 0.25

RELATIVE_CUTOFF = 0.0

TOPIC_RANK_BONUS = 0.03

SCOPE_ANSWERED_SIMILARITY = MIN_SIMILARITY

FLAG_MIN_SIMILARITY = 0.0

EXPAND_NEIGHBOURS = 1


ROUTE_GAMMA = 0.30
ROUTE_FLOOR = 0.30
ROUTE_TOP_N = 3


GENERAL_ENTROPY_MIN = 2.60
GENERAL_TOP_SHARE_MAX = 0.60

CLAIM_WINDOW_WORDS = 140
CLAIM_STRIDE_WORDS = 70

TIME_CHECK = os.environ.get("TIME_CHECK", "0") in ("1", "true", "True")

OVERLAP_VETO = os.environ.get("OVERLAP_VETO", "0") in ("1", "true", "True")

WHOLE_TRANSCRIPT = False
