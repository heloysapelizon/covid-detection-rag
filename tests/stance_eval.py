"""Exploratory: can the configured model tell asserting from denying?

The RAG pipeline's hard problem is stance, not retrieval. The dataset contains
misinformation and debunking of the *same* myths in near-identical vocabulary, so a
model that reads topic instead of stance flags fact-checkers as fake news.

These are minimal unambiguous pairs — one sentence, no transcription noise, no
window boundaries. A model that fails here cannot be trusted on real transcripts,
and a model that passes still needs checking on them.

    python tests/stance_eval.py                  # uses rag/config.py settings
    python tests/stance_eval.py qwen2.5:14b      # try another Ollama model

Not part of the experiment: this measures the classifier, it does not produce data.

---------------------------------------------------------------------------
Findings so far (qwen2.5:7b, temperature 0, reproduced 13 days apart)
---------------------------------------------------------------------------
One-pass adjudication (claim + evidence together):  9/12
    asserting 6/6   denying 3/6
It flags "The vaccine does not make you magnetic; it contains no metals" as
FAKE_NEWS — topic overlap is read as assertion.

Four attempts to fix this by prompting, all on the same 12 pairs:

  1. Stance asked in isolation (no evidence shown)      6/12  (asserting 0/6)
  2. Same, with all few-shot examples removed          6/12  (asserting 0/6)
  3. Neutral-topic control sentences (cats, printers)   4/6   — and the errors
     were "neutral", not "denying"
  4. Reframed as grammatical polarity, explicitly
     instructing that accuracy is irrelevant           7/12  (positive 1/6)

Control 3 is the diagnosis: the denying bias appears only on COVID content. Asked
about cats, the model reasons correctly ("presents the claim that cats are mammals
as a fact"). Asked about vaccines, it answers a question nobody posed — "is this
statement true?" — and reports "denying" because the statement is false. Even
framed as pure grammar, "The mRNA vaccine permanently alters your DNA" (no negation
words at all) comes back negative.

So the failure is not prompt wording. A 7B model's safety training on COVID claims
overrides the instruction to report the speaker's position.

---------------------------------------------------------------------------
Model comparison, same 12 pairs
---------------------------------------------------------------------------
    qwen2.5:7b    9/12   asserting 6/6   denying 3/6
    qwen2.5:14b  11/12   asserting 6/6   denying 5/6
    MNLI cross-encoder (nli-deberta-v3-small, 70 MB)   7/8 on a subset,
        including 4/4 denials

The cross-encoder is the interesting result: a small trained classifier beats a
7B generative model here, because entailment is a classification task with no
instruction for safety training to override.

---------------------------------------------------------------------------
Why clean-sentence scores overstate real performance
---------------------------------------------------------------------------
qwen2.5:14b scores 11/12 here but only 2/5 on whole videos:

    London Real graphene     PASS  8 windows flagged (correct, true positive)
    Premier Health           PASS  0 flagged
    CNBC microchip debunk    FAIL  18 of 39 windows flagged
    CBS fact-check           FAIL  flagged "Will it cause females to be sterile?"
    Florida Medical "Myth #1" FAIL flagged "a belief that they were actually
                                   going to implant vaccination microchips"

The flagged phrases are the lesson. Real speech constantly quotes, asks about and
describes myths in order to reject them — a journalist's question and a doctor's
"some people believe X" both read as assertions to the model. Minimal pairs contain
no reported speech, so they miss the dominant real-world failure mode entirely.

Any eval used to justify a model choice must therefore include real transcript
windows, not just constructed sentences.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import LLAMA_MODEL  # noqa: E402
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import adjudicate  # noqa: E402

# (claim, expected verdict). Half assert a myth, half deny the same myth.
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
    by_kind = {"asserting": [0, 0], "denying": [0, 0]}  # [correct, total]

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

    # Denials misread as assertions are the damaging error: they flag the
    # fact-checkers in the dataset as the misinformation.
    fp = by_kind["denying"][1] - by_kind["denying"][0]
    if fp:
        print(f"\n{fp} denial(s) misread as assertion — this model would flag "
              f"debunking content as fake news.")
    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
