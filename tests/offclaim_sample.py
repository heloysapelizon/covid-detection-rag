"""Audit a random sample of OFF-CLAIM flags, to turn the window-precision bounds
of tests/window_metrics.py into an estimate.

    python tests/offclaim_sample.py export data/results/test_run_*_tavily.json   # writes the review sheet
    #   ... fill the `judgement` column of data/labels/offclaim_review.csv ...
    python tests/offclaim_sample.py score                                          # estimate + 95% CI

An off-claim flag is a FAKE_NEWS / FAKE_NEWS_WEB window in a fake video that
does not contain the annotated flagged_phrase. It may be a real false claim the
annotation did not list, or an error. window_metrics.py can only bound this.

Review is BLIND: the sheet shows the flagged phrase, its window and the text
around it, never the model, its reason, the evidence or whether the flag came
from the corpus or the web; data/labels/offclaim_key.json maps rows back. Judge
each row by the annotation rules in data/labels/ground_truth.json:
    Y  the flagged phrase is asserted as fact and is false or unsupportable
    N  it is not (debunking, hedged, anecdotal, accurate, mis-located, ...)
    ?  cannot tell (excluded from the estimate, and counted)
Identical flags (same video, window and phrase) from different runs are judged
once. The sample is drawn per run with a fixed seed, so it is reproducible.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.run_experiment import load_test_set  # noqa: E402
from tests.window_metrics import FLAG, norm, positive_windows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "data" / "labels" / "offclaim_review.csv"
KEY = ROOT / "data" / "labels" / "offclaim_key.json"
SEED = 20261106
PER_RUN = 50
CONTEXT_WORDS = 70


def off_claim_flags(videos: dict, labels: dict) -> list[tuple[str, dict]]:
    out = []
    for vid, r in videos.items():
        lab = labels.get(vid)
        if not lab or not lab["label_fake_news"]:
            continue
        pos = set(lab["_positive"] or [])
        out += [(vid, w) for w in r["results"]
                if w["verdict"] in FLAG and w["window"] not in pos]
    return out


def context(transcript: str, window_text: str) -> tuple[str, str]:
    words, win = transcript.split(), window_text.split()
    for i in range(len(words) - len(win) + 1):
        if words[i:i + len(win)] == win:
            return (" ".join(words[max(0, i - CONTEXT_WORDS):i]),
                    " ".join(words[i + len(win):i + len(win) + CONTEXT_WORDS]))
    return "", ""


def export(files: list[str]) -> None:
    if SHEET.exists():
        raise SystemExit(f"{SHEET} exists; delete it first (it may hold judgements)")
    labels = {v["video_id"]: v for v in load_test_set()}
    for v in labels.values():
        v["_positive"] = positive_windows(v) if v["label_fake_news"] else None
    rng = random.Random(SEED)
    items, key = {}, {"seed": SEED, "per_run": PER_RUN, "runs": {}, "rows": {}}
    for f in files:
        payload = json.loads(Path(f).read_text(encoding="utf-8"))
        flags = off_claim_flags(payload["videos"], labels)
        sample = rng.sample(flags, min(PER_RUN, len(flags)))
        key["runs"][f] = {"off_claim_flags": len(flags), "sampled": len(sample)}
        for vid, w in sample:
            ident = (vid, w["window"], norm(w["flagged_phrase"]))
            if ident not in items:
                items[ident] = (vid, w)
            key["rows"].setdefault("|".join(map(str, ident)), []).append(
                {"run": f, "verdict": w["verdict"], "origin": w["evidence_origin"],
                 "reason": w["reason"]})
    order = list(items)
    rng.shuffle(order)          # rows from different runs interleaved
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row", "judgement", "flagged_phrase", "window_text",
                     "context_before", "context_after", "note", "_id"])
        for n, ident in enumerate(order, 1):
            vid, w = items[ident]
            before, after = context(labels[vid]["transcript"], w["claim_text"])
            wr.writerow([n, "", w["flagged_phrase"], w["claim_text"], before, after, "",
                         "|".join(map(str, ident))])
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f, s in key["runs"].items():
        print(f"{Path(f).name}: {s['off_claim_flags']} off-claim flags, {s['sampled']} sampled")
    print(f"{len(order)} unique rows to review -> {SHEET}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def score() -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    with SHEET.open(newline="", encoding="utf-8-sig") as fh:
        judged = {r["_id"]: r["judgement"].strip().upper()[:1] for r in csv.DictReader(fh)}
    blank = sum(1 for j in judged.values() if j == "")
    if blank:
        print(f"WARNING: {blank} row(s) not judged yet\n")
    labels = {v["video_id"]: v for v in load_test_set()}
    for v in labels.values():
        v["_positive"] = positive_windows(v) if v["label_fake_news"] else None
    for f, info in key["runs"].items():
        tally = {"Y": 0, "N": 0, "?": 0, "": 0}
        by_origin = {}
        for ident, entries in key["rows"].items():
            for e in entries:
                if e["run"] != f:
                    continue
                j = judged.get(ident, "")
                tally[j if j in tally else "?"] += 1
                o = by_origin.setdefault(e["origin"], [0, 0])
                if j in ("Y", "N"):
                    o[0] += j == "Y"; o[1] += 1
        k, n = tally["Y"], tally["Y"] + tally["N"]
        lo, hi = wilson(k, n)
        # window-level counts for this run, as in window_metrics.py
        videos = json.loads(Path(f).read_text(encoding="utf-8"))["videos"]
        tp_w = off = neg = 0
        for vid, r in videos.items():
            lab = labels.get(vid)
            if not lab:
                continue
            flags = {w["window"] for w in r["results"] if w["verdict"] in FLAG}
            if not lab["label_fake_news"]:
                neg += len(flags)
                continue
            pos = set(lab["_positive"] or [])
            tp_w += len(flags & pos); off += len(flags - pos)
        tot = tp_w + off + neg
        est = lambda s: (tp_w + s * off) / tot  # noqa: E731
        print(f"== {Path(f).name}")
        print(f"   sample: {n} judged (Y {tally['Y']} / N {tally['N']}), {tally['?']} unsure, "
              f"{tally['']} blank, of {info['off_claim_flags']} off-claim flags")
        if n:
            print(f"   off-claim flags that are real false claims: {k / n:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
            for o, (ok, m) in sorted(by_origin.items()):
                print(f"     {o:6}: {ok}/{m}")
            print(f"   window precision: {est(k / n):.3f}  (95% CI {est(lo):.3f}-{est(hi):.3f}; "
                  f"bounds were {est(0):.3f}-{est(1):.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "score"):
        raise SystemExit(__doc__)
    export(sys.argv[2:]) if sys.argv[1] == "export" else score()
