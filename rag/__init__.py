"""RAG fact-checking pipeline for the COVID-19 vaccine video dataset."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
