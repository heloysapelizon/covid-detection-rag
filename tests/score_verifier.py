"""Score "evidence as a verifier": a flag needs both arms to agree.

    python tests/score_verifier.py PARAMETRIC.json RAG.json [--overlap-veto]

Takes one parametric run and one evidence run of the same model, dataset and
prompt (tests/run_experiment.py output) and scores four predictors at video
level, against the labels stored in the run files:

  parametric    any window flagged without evidence
  rag           any window flagged with evidence (corpus or web)
  video-AND     both arms flagged the video, possibly on different windows
  window-AND    some window was flagged by BOTH arms. The stricter reading:
                the evidence confirmed the same claim the model proposed.

This is an agreement rule over two existing runs, not a generate-then-verify
pipeline: the evidence arm never sees the parametric flag. Its recall is
bounded by the parametric arm's, so it can only trade recall for precision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FLAG = {"FAKE_NEWS", "FAKE_NEWS_WEB"}


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("run") or {}, payload["videos"]


def flagged_windows(video: dict) -> set[int]:
    return {r["window"] for r in video["results"] if r["verdict"] in FLAG}


def metrics(pred: dict[str, bool], label: dict[str, bool]) -> str:
    tp = sum(pred[v] and label[v] for v in pred)
    fp = sum(pred[v] and not label[v] for v in pred)
    fn = sum(not pred[v] and label[v] for v in pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return (f"TP {tp:3}  FP {fp:3}  FN {fn:3}   "
            f"P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    pm, par = load(Path(args[0]))
    rm, rag = load(Path(args[1]))
    if "--overlap-veto" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from rag.pipeline import overlap_veto
        for runs in (par, rag):
            for v in runs.values():
                v["results"] = overlap_veto(v["results"])
        print("(both runs re-scored with overlap veto)")

    # Pairing runs that differ in anything but the arm would credit that
    # difference to the verifier rule.
    for key in ("dataset",):
        if pm.get(key, "test") != rm.get(key, "test"):
            raise SystemExit(f"runs differ in {key}: {pm.get(key)} vs {rm.get(key)}")
    pp, rp = (pm.get("prompt") or {}).get("name"), (rm.get("prompt") or {}).get("name")
    if pp != rp:
        raise SystemExit(f"runs used different prompts: {pp} vs {rp}")
    model = (pm.get("settings") or {}).get("LLAMA_MODEL")
    if model != (rm.get("settings") or {}).get("LLAMA_MODEL"):
        raise SystemExit("runs used different models")

    common = sorted(set(par) & set(rag))
    print(f"model={model}  dataset={pm.get('dataset', 'test')}  prompt={pp}  "
          f"videos={len(common)} (parametric {len(par)}, rag {len(rag)})\n")
    label = {v: par[v]["label_fake_news"] for v in common}
    preds = {
        "parametric": {v: bool(flagged_windows(par[v])) for v in common},
        "rag": {v: bool(flagged_windows(rag[v])) for v in common},
        "video-AND": {v: bool(flagged_windows(par[v]))
                      and bool(flagged_windows(rag[v])) for v in common},
        "window-AND": {v: bool(flagged_windows(par[v]) & flagged_windows(rag[v]))
                       for v in common},
    }
    for name, pred in preds.items():
        print(f"  {name:11} {metrics(pred, label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
