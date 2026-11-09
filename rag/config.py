"""Settings and paths shared by the RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"

# Experiment results, grouped by protocol version (data/results/README.md):
#   v1     first test round: earlier prompts, no time check, evidence arm only
#   v2     current protocol: rules_v2 prompts, time check, both arms, verifier
#   smoke  short checks (--limit N), kept apart so they never count as a run
# New runs go to RESULTS_VERSION; set it to write elsewhere (e.g. v3).
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
RESULTS_VERSION = os.environ.get("RESULTS_VERSION", "v2")
RESULTS_DIR = RESULTS_ROOT / RESULTS_VERSION
SMOKE_DIR = RESULTS_ROOT / "smoke"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Reference corpus: source PDFs (gitignored) and the passages extracted from them
# (committed, so the corpus is reproducible without redistributing the PDFs).
PDF_DIR = EVIDENCE_DIR / "pdf"
PASSAGE_DIR = EVIDENCE_DIR / "passages"

# Evidence chunking. Measured gold-evidence recall (top-3 containing the paper's
# actual conclusion) across two document types:
#
#   chunk   research paper   CDC FAQ   paper mean top-1
#    55w        5/7            4/4         0.7422
#    90w        7/7            4/4         0.7182
#   140w        7/7            4/4         0.7029
#   220w        7/7            4/4         0.6965
#
# Small chunks score highest on similarity but fragment findings — 55w matched
# citation fragments instead of the conclusion on 2 of 7 queries. Rather than
# trade precision for recall by enlarging the chunk, retrieval keeps 55w and
# pulls the neighbouring chunks (EXPAND_NEIGHBOURS), which reaches 7/7 while
# keeping the sharper scores.
#
# Note the CDC FAQ is insensitive to this setting (4/4 everywhere): its Q&A pairs
# are already claim-sized, which is why tuning on it alone was misleading.
CHUNK_WORDS = 55
CHUNK_STRIDE = 35

# The transcribed dataset to fact-check.
DATASET_NAME = "covid_vaccine_videos_transcribed"

# --- Model server --------------------------------------------------------
# Any server exposing an OpenAI-compatible /v1/chat/completions works, so the
# same client code serves all of them. `LLM_SERVER` selects one:
#
#   llamacpp    llama-server -m model.gguf -c 8192 --port 8080   (default)
#   ollama      ollama serve                                      port 11434
#   openrouter  https://openrouter.ai/api/v1  — remote, needs OPENROUTER_API_KEY
#
# The thesis compares four models across two deployment kinds:
#
#   local   Gemma 4 12B, gpt-oss-20b          (llama.cpp or Ollama)
#   remote  Qwen3.6-27B, gpt-oss-120b          (OpenRouter)
#
# **Remote is not a drop-in substitute**: transcript text leaves the machine,
# which the local path avoids, and the output constraint differs (see below). The
# run metadata records server, model and constraint for every run so figures from
# different backends are never pooled (tests/run_experiment.py).
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
# Ollama and OpenRouter need a real model name; llama-server ignores it and
# serves whatever GGUF it was started with.
_DEFAULT_MODEL = ("qwen/qwen3.6-27b" if _IS_OPENROUTER else "qwen2.5:14b")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", _DEFAULT_MODEL)
# Remote calls cross a network and can queue behind a provider's rate limit, so
# they get longer than a local forward pass needs.
REQUEST_TIMEOUT_S = int(os.environ.get(
    "LLM_TIMEOUT_S", "600" if _IS_OPENROUTER else "300"))
TEMPERATURE = 0.0            # deterministic: this is classification, not writing

# How the six-field verdict contract is enforced, strongest first:
#
#   GBNF grammar (llama.cpp)  enforced during sampling — a response missing a
#                             required field is unrepresentable.
#   json_schema (OpenRouter)  provider validates against the schema, so the
#                             same six fields in the same order are guaranteed.
#   json_object (Ollama)      valid JSON of any shape only. A large evidence
#                             block was observed dropping required keys
#                             entirely (docs/adjudication-experiments.md).
#
# Field ORDER matters, not just presence: `addresses` is answered before
# `verdict` so relevance is decided before the verdict for a left-to-right
# decoder. All three mechanisms preserve that order.
# LLM_CONSTRAINT overrides the per-backend default: grammar | json_schema | json_object.
# gpt-oss on llama.cpp needs json_schema: a GBNF grammar forces "{" as the first
# token, skipping the harmony channel header, and llama-server's harmony parser
# then rejects every response with HTTP 500.
_CONSTRAINT = os.environ.get("LLM_CONSTRAINT", "").lower()
USE_GRAMMAR = (_CONSTRAINT == "grammar") if _CONSTRAINT else _IS_LLAMACPP
USE_JSON_SCHEMA = (_CONSTRAINT == "json_schema") if _CONSTRAINT else _IS_OPENROUTER
USE_JSON_MODE = (_CONSTRAINT == "json_object") if _CONSTRAINT else _IS_OLLAMA

# --- Retrieval -----------------------------------------------------------
# Runs locally via sentence-transformers; small, fast, good enough for short claims.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4                    # evidence passages retrieved per claim
MIN_SIMILARITY = 0.25        # below this, treat as "no relevant evidence"

# Relative retrieval cutoff: keep a passage only if it scores at least this
# fraction of the best hit for the same claim, so a claim with one clearly-best
# passage is not padded out to TOP_K with weaker ones. 0 disables it.
#
# Disabled. Labelled-phrase recall on dev is identical (0.667) at every ratio
# from 0.80 to 0.95 and with the cutoff off, so the dev set gives no basis for a
# value. See docs/routing-experiments.md.
RELATIVE_CUTOFF = 0.0

# Additive bonus for topic-specific passages when ranking inside a routed tier.
# General documents are in scope for every claim, so without this a general
# passage sharing a claim's vocabulary can outrank the topic passage that answers
# it. Affects ORDER only — index.search reports the unmodified cosine, because a
# reported similarity feeds MIN_SIMILARITY and the flag floor.
# Chosen mid-plateau on the dev labels; see docs/routing-experiments.md.
TOPIC_RANK_BONUS = 0.03

# Separate floor for the decision "did this scope tier answer the claim?",
# distinct from MIN_SIMILARITY, which decides "is this passage worth showing?".
# A tier whose best hit falls below this is treated as not having answered, and
# the search widens to the next scope.
#
# Set equal to MIN_SIMILARITY, which makes it a no-op: a tier that returns any
# showable passage counts as having answered. The dev set gives no basis for a
# higher value — recall is identical at every floor from 0.30 to 0.65, and only 5
# of 80 windows retrieve anything different across that whole range. See
# docs/routing-experiments.md.
SCOPE_ANSWERED_SIMILARITY = MIN_SIMILARITY

# Similarity floor for flags: below it, a FAKE_NEWS verdict is downgraded to
# NO_EVIDENCE. 0 disables the override.
#
# Disabled. On the dev set every floor removes correct flags and no false
# positives — the false positives sit at high similarity and are stance and
# specificity errors, which a threshold cannot reach. See
# docs/adjudication-experiments.md.
FLAG_MIN_SIMILARITY = 0.0

# Neighbouring chunks included around each hit. Small chunks retrieve precisely
# but can cut a finding in half; pulling the adjacent chunks restores the
# surrounding argument. Measured on a research paper, top-3 over 55-word chunks
# found the paper's conclusion for 5 of 7 real transcript queries, and 7 of 7
# with neighbours included. 0 disables expansion.
EXPAND_NEIGHBOURS = 1


# --- Topic routing -------------------------------------------------------
# A claim is searched only against the topics it resembles, plus any document
# judged general. Without this, a large single-subject paper outranks the right
# evidence: with 74 of 90 passages from the ICH shedding guidance, a magnetism
# claim retrieved ICH study-design text instead of the CDC's answer.
# Routing returns an ORDERED list of candidate topics, tried in turn: if the
# first scope yields no usable evidence the next-closest topic is tried, and the
# broad scope is the last resort. This keeps the soft multi-topic behaviour while
# adopting the confidence gate from Sun et al., "Asymmetric Dynamic Routing"
# (arXiv:2609.29282) — accept a narrow scope only when confident, otherwise fall
# back to a deliberately broader one, "reducing the risk of routing uncertain
# queries to either extreme".
#
# Falling back matters more than routing precisely. A mis-route is a silent false
# positive: the model is handed off-topic evidence and may call a true statement
# false. Searching too widely only costs rank competition.
#
# Their gamma = 0.75 does not transfer (different embedding, different confidence
# measure) — see docs/routing-experiments.md. The shape does transfer: too low
# admits ambiguous claims, too high collapses to fallback.
#
# Confidence gate: only a topic scoring at or above this may lead with its own
# scope. Below it the claim falls through to general, then to the whole corpus.
#
# Equal to ROUTE_FLOOR, so every candidate clearing the floor leads a tier. A
# higher value discards correct routes much faster than it prevents mis-routes on
# the dev labels; the gate is kept because a corpus with more topics may need it.
# Derivation in docs/routing-experiments.md.
ROUTE_GAMMA = 0.30
ROUTE_FLOOR = 0.30          # below this a topic is not a candidate at all
ROUTE_TOP_N = 3             # at most this many topics in the ordered list


# A document is "general" — always in scope — when its passages spread across
# many topics rather than concentrating in one. Derived from the corpus, never
# declared: measured entropy over nearest-topic assignment was 1.88 for the CDC
# myths sheet (5 topics, 8 distinct myths), 1.31 for the ICH guidance, and 0.79
# for the single-subject infertility review.
GENERAL_ENTROPY_MIN = 2.60   # bits of spread required; see docs/routing-experiments.md
GENERAL_TOP_SHARE_MAX = 0.60 # and no single topic may exceed this share

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

# --- Time-bound statements -----------------------------------------------------
# On: dates reach the prompt (claim = video publish date, corpus passage =
# document date, web = undated), windows with time-bound wording get a TIME
# CHECK note, and a time-bound flag resting on later evidence is overridden.
# See rag/temporal.py.
TIME_CHECK = os.environ.get("TIME_CHECK", "0") in ("1", "true", "True")

# --- Overlap veto ----------------------------------------------------------------
# Claim windows overlap by half, so a flagged phrase normally appears in a second
# window with different surrounding context. On: a flag is withdrawn when another
# window containing the same words judged them NOT_FAKE_NEWS — the case of a myth
# quoted in one window and rebutted in the next. An abstention (NO_EVIDENCE,
# UNVERIFIABLE) never vetoes. Deterministic, no extra model calls.
OVERLAP_VETO = os.environ.get("OVERLAP_VETO", "0") in ("1", "true", "True")
