"""Find the routing confidence threshold below which a claim should stay general.

Routing narrows retrieval to the topics a claim resembles. That helps when the
route is right and hurts badly when it is wrong — a mis-routed claim cannot reach
the evidence that would answer it. So below some confidence the router should
decline to commit and leave the scope general.

This measures where that line falls, against hand-assigned ground truth in
tests/chunk_eval_labels.json (the `topic` field). The collection query a video was
found by is deliberately NOT used: a video found under "covid vaccine infertility"
makes claims about microchips and DNA too, and many windows make no claim at all.

    python tests/route_eval.py
    python tests/route_eval.py --signal margin

Two confidence signals are swept:
  absolute   the top topic's similarity to the window
  margin     the gap between the top topic and the runner-up

Reported per threshold, counting only windows with a real topic label:
  routed     share of windows the router would commit on
  precision  of those, the share routed to the labelled topic
  missed     labelled-topic windows left general (routing declined)

And separately, for windows labelled "general" (filler, intros, no claim):
  correctly left general — the error this threshold exists to prevent.

Not part of the experiment: this measures the router, it does not produce data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import EvidenceIndex  # noqa: E402

LABELS = Path(__file__).resolve().parent / "chunk_eval_labels.json"
THRESHOLDS = (0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70, 0.75, 0.80)


def load_labelled(labels: Path | None = None) -> list[dict]:
    path = labels or LABELS
    if not path.exists():
        sys.exit(f"No labels at {path}")
    items = json.loads(path.read_text())["items"]
    return [i for i in items if i.get("topic")]


def measure(router, items: list[dict], signal: str) -> list[dict]:
    """Route each labelled window once, recording its confidence."""
    rows: list[dict] = []
    for item in items:
        # min_similarity=0 so nothing is filtered before the sweep.
        routed = router.route(item["window_text"], min_similarity=0.0, top_n=3)
        if not routed:
            rows.append({**item, "routed": None, "confidence": 0.0})
            continue
        top, top_score = routed[0]
        runner_up = routed[1][1] if len(routed) > 1 else 0.0
        rows.append({
            **item,
            "routed": top,
            "confidence": top_score if signal == "absolute"
                          else top_score - runner_up,
        })
    return rows


def sweep(rows: list[dict], signal: str) -> None:
    topical = [r for r in rows if r["topic"] != "general"]
    general = [r for r in rows if r["topic"] == "general"]

    print(f"\n=== signal: {signal} ===")
    print(f"  {len(topical)} window(s) with a topic, "
          f"{len(general)} labelled general\n")
    print(f"  {'thresh':>7} {'routed':>7} {'precision':>10} {'missed':>7}"
          f" {'general kept':>13}")
    print("  " + "-" * 50)

    for threshold in THRESHOLDS:
        committed = [r for r in topical if r["confidence"] >= threshold]
        right = sum(1 for r in committed if r["routed"] == r["topic"])
        # A "general" window is handled correctly when routing declines to commit.
        kept = sum(1 for r in general if r["confidence"] < threshold)

        precision = f"{right / len(committed):10.1%}" if committed else f"{'n/a':>10}"
        general_col = f"{kept:6d}/{len(general):<6d}" if general else f"{'n/a':>13}"
        print(f"  {threshold:7.2f} {len(committed):7d} {precision} "
              f"{len(topical) - len(committed):7d} {general_col}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python tests/route_eval.py",
        description="Measure the routing confidence threshold.",
    )
    ap.add_argument("--labels", default=None,
                    help="label file to score (default: chunk_eval_labels.json)")
    ap.add_argument("--signal", choices=("absolute", "margin", "both"),
                    default="both")
    args = ap.parse_args(argv)

    labels_path = Path(args.labels) if args.labels else LABELS
    items = load_labelled(labels_path)
    if not items:
        print(f"No item in {labels_path} has a `topic` label yet.\n"
              f"Fill in `topic` for some windows "
              f"(a topic name, or \"general\"), then re-run.", file=sys.stderr)
        return 1

    index = EvidenceIndex.load()
    router = index.router()
    print(f"index  : {len(index.passages)} passages")
    print(f"labels : {len(items)} window(s)")

    for signal in (("absolute", "margin") if args.signal == "both"
                   else (args.signal,)):
        sweep(measure(router, items, signal), signal)

    print("\n  per-window detail:")
    for r in measure(router, items, "absolute"):
        mark = "ok " if r["routed"] == r["topic"] else "   "
        print(f"    {mark} {r['video_id']} w{r['window']:<3} "
              f"label={r['topic'][:16]:16} routed={str(r['routed'])[:16]:16} "
              f"abs={r['confidence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
