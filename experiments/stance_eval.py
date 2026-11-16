"""Exploratory: can the configured model tell asserting from denying?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import LLAMA_MODEL  # noqa: E402
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import adjudicate  # noqa: E402

CASES: list[tuple[str, str]] = [
    ("The COVID vaccine contains a microchip that tracks your movements.", "FAKE_NEWS"),
    ("There's no chip or anything like that in the COVID vaccine.", "NOT_FAKE_NEWS"),
    ("We are not being injected with chips.", "NOT_FAKE_NEWS"),
    ("The vaccine makes you magnetic because of the metals inside it.", "FAKE_NEWS"),
    ("The vaccine does not make you magnetic; it contains no metals.", "NOT_FAKE_NEWS"),
    ("COVID vaccines cause infertility in women.", "FAKE_NEWS"),
    ("There is no evidence that COVID vaccines cause infertility.", "NOT_FAKE_NEWS"),
    ("The mRNA vaccine permanently alters your DNA.", "FAKE_NEWS"),
    ("The mRNA never enters the nucleus, so it cannot change your DNA.", "NOT_FAKE_NEWS"),
    ("Vaccinated people shed spike protein onto those around them.", "FAKE_NEWS"),
    ("Shedding only happens with live-virus vaccines, which these are not.",
     "NOT_FAKE_NEWS"),
    ("Getting the vaccine gives you COVID because it has the live virus.", "FAKE_NEWS"),
]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    model = argv[0] if argv else LLAMA_MODEL

    index = EvidenceIndex.load()
    print(f"model    : {model}")
    print(f"evidence : {len(index.passages)} passage(s)\n")

    correct = 0
    by_kind = {"asserting": [0, 0], "denying": [0, 0]}

    for text, want in CASES:
        kind = "asserting" if want == "FAKE_NEWS" else "denying"
        passages = index.search(text)
        got = adjudicate(text, passages, model=model)
        ok = got["verdict"] == want
        correct += ok
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want:14} got={got['verdict']:14} "
              f"stance={got.get('stance', '') or '-':10} {text[:48]}")

    print(f"\noverall   {correct}/{len(CASES)}")
    for kind, (c, n) in by_kind.items():
        print(f"  {kind:10} {c}/{n}")

    fp = by_kind["denying"][1] - by_kind["denying"][0]
    if fp:
        print(f"\n{fp} denial(s) misread as assertion — this model would flag "
              f"debunking content as fake news.")
    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
