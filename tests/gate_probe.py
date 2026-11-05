"""Probe the relevance gate on claims the corpus does and does not cover.

    python tests/gate_probe.py            # all five cases
    python tests/gate_probe.py microchip  # one case

The point of the gate is that similarity cannot decide relevance. `microchip`
retrieves at ~0.58 — above FLAG_MIN_SIMILARITY — on passages about immunology
that say nothing about microchips, so only the `addresses` field prevents a
flag. `magnetism` is the control: it retrieves at a *lower* score and should
still flag, because the CDC sheet genuinely answers it.

Expected, with the gate working:

    microchip   NO_EVIDENCE    addresses=no
    graphene    NO_EVIDENCE    addresses=no
    real-clot   NO_EVIDENCE    addresses=no   (speaker asserts; corpus refutes
                                               the mechanism, so FAKE_NEWS is
                                               also defensible here)
    magnetism   FAKE_NEWS      addresses=yes
    real-fert   FAKE_NEWS      addresses=yes

A `FAKE_NEWS` on microchip or graphene means the gate is not holding: those are
accepted coverage gaps (docs/corpus-inclusion.md), so any verdict other than
NO_EVIDENCE is asserted without evidence that addresses the claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import FLAG_MIN_SIMILARITY  # noqa: E402
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import adjudicate, health  # noqa: E402

# Phrased as a speaker asserting the claim, since that is what the pipeline sees.
CASES: dict[str, str] = {
    "microchip": "they put a microchip in the vaccine so they can track "
                 "everywhere you go after you get the shot",
    "graphene": "the vials have been analysed and they contain graphene oxide "
                "which is a toxic nanomaterial",
    "real-clot": "the spike protein from the vaccine forms clots that cannot "
                 "be broken down by the body",
    "magnetism": "after the shot your arm becomes magnetic and a spoon will "
                 "stick to the injection site",
    "real-fert": "the covid vaccine causes infertility in young women by "
                 "attacking the placenta",
}


def main() -> int:
    ok, detail = health()
    if not ok:
        print(f"server unreachable: {detail}\n\nStart one with:\n  ollama serve")
        return 1
    print(f"server: {detail}")

    wanted = sys.argv[1:] or list(CASES)
    unknown = [w for w in wanted if w not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}")
        print(f"available: {', '.join(CASES)}")
        return 2

    index = EvidenceIndex.load()
    print(f"index : {len(index.passages)} passages")
    print(f"floor : FLAG_MIN_SIMILARITY = {FLAG_MIN_SIMILARITY}\n")

    for name in wanted:
        claim = CASES[name]
        passages = index.search(claim)
        matched = [p for p in passages if p.get("matched")]
        top = matched[0]["similarity"] if matched else None
        verdict = adjudicate(claim, passages)

        print("=" * 74)
        print(f"{name}   tier={index.last_scope}   top_sim={top}")
        for p in matched:
            print(f"   [{p['similarity']:.4f}] {p['source_file'][:26]:28}"
                  f"{p['text'][:60].replace(chr(10), ' ')}")
        print(f"  -> addresses={verdict.get('addresses')!r}  "
              f"stance={verdict.get('stance')!r}  "
              f"VERDICT={verdict['verdict']}")
        if verdict.get("claim"):
            print(f"     flagged phrase: {verdict['claim'][:70]}")
        print(f"     reason: {verdict.get('reason', '')[:150]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
