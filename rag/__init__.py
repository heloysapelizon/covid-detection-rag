"""RAG fact-checking pipeline for the COVID-19 vaccine video dataset.

Takes a transcript from `transcribe`, retrieves evidence from a vectorized corpus of
public-health sources, and asks a locally served llama.cpp model whether any claim
asserts something that evidence contradicts. Each flag carries the false phrase and
the passage that debunks it.

Modules are grouped by what they depend on, so a failure or a slow import stays
contained and each layer can be used without the ones above it:

    config     settings, paths and server URL — no dependencies
    index      embed and query the evidence corpus (sentence-transformers, numpy)
    llm        the llama.cpp HTTP boundary — one claim in, a verdict out
    pipeline   segment, retrieve, adjudicate, report
    main       argument parsing and exit codes

Dependencies point one way only: main -> pipeline -> {index, llm} -> config.

Names are re-exported lazily (PEP 562), so `from rag.index import build` does not
import requests, and `from rag.llm import health` does not load an embedding model.

Everything runs locally: no data leaves the machine.

Usage:

    llama-server -m model.gguf -c 8192 --port 8080   # in another terminal
    python -m rag --build-index
    python -m rag <video_id>
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and editors only; no runtime import
    from .index import EvidenceIndex, build, load_evidence
    from .llm import adjudicate, build_prompt, health
    from .main import main
    from .pipeline import check_transcript, print_report, segment_claims

_EXPORTS = {
    "EvidenceIndex": "index",
    "build": "index",
    "load_evidence": "index",
    "adjudicate": "llm",
    "build_prompt": "llm",
    "health": "llm",
    "check_transcript": "pipeline",
    "print_report": "pipeline",
    "segment_claims": "pipeline",
    "main": "main",
}

__all__ = [
    "EvidenceIndex",
    "adjudicate",
    "build",
    "build_prompt",
    "check_transcript",
    "health",
    "load_evidence",
    "main",
    "print_report",
    "segment_claims",
]


def __getattr__(name: str):
    """Resolve a public name to its module on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
