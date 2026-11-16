"""Window- and claim-level metrics, next to the video-level score."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import segment_claims  # noqa: E402
from experiments.run_experiment import load_test_set  # noqa: E402

FLAG = {"FAKE_NEWS", "FAKE_NEWS_WEB"}
ABSTAIN = {"NO_EVIDENCE", "UNVERIFIABLE"}
TO_CORPUS = {"FAKE_NEWS_WEB": "NO_EVIDENCE", "NOT_FAKE_NEWS_WEB": "NO_EVIDENCE"}


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def positive_windows(video: dict) -> list[int] | None:
    """1-based windows containing a whole segment of the annotated phrase."""
    segs = [norm(s) for s in re.split(r"\.\.\.|…", video.get("flagged_phrase") or "")]
    segs = [s for s in segs if s]
    if not segs:
        return None
    wins = [norm(w) for w in segment_claims(video["transcript"])]
    pos = [i + 1 for i, w in enumerate(wins) if any(s in w for s in segs)]
    return pos or None


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(videos: dict, labels: dict, mapping: dict | None = None) -> list[str]:
    m = mapping or {}
    verdict = lambda w: m.get(w["verdict"], w["verdict"])  # noqa: E731
    out = []
    ids = [k for k in videos if k in labels]
    fake = [k for k in ids if labels[k]["label_fake_news"]]
    neg = [k for k in ids if not labels[k]["label_fake_news"]]

    out.append("video level (k = 1 is the reported rule; k > 1 is post hoc):")
    for k in (1, 2, 3, 5):
        pred = {v: sum(verdict(w) in FLAG for w in videos[v]["results"]) >= k for v in ids}
        tp = sum(pred[v] for v in fake); fp = sum(pred[v] for v in neg)
        p, r, f = prf(tp, fp, len(fake) - tp)
        out.append(f"  >= {k} flagged window(s): TP {tp:3} FP {fp:3}  P {p:.3f} R {r:.3f} F1 {f:.3f}")
    base = len(fake) / len(ids)
    out.append(f"  baseline 'all fake'      : P {base:.3f} R 1.000 F1 {2 * base / (1 + base):.3f}")

    nw = [w for v in neg for w in videos[v]["results"]]
    nf = sum(verdict(w) in FLAG for w in nw)
    na = sum(verdict(w) in ABSTAIN for w in nw)
    out.append(f"windows of NOT-fake videos ({len(neg)} videos, {len(nw)} windows, all negative):")
    out.append(f"  flagged {nf} -> false-positive rate {nf / len(nw):.3f}, specificity "
               f"{1 - nf / len(nw):.3f}; abstained {na / len(nw):.3f}")

    located, detected, flagged_elsewhere_only = 0, 0, 0
    tp_w = pos_total = off_claim = 0
    for v in fake:
        pos = labels[v]["_positive"]
        res = videos[v]["results"]
        flags = {w["window"] for w in res if verdict(w) in FLAG}
        if pos is None:
            off_claim += len(flags)
            continue
        located += 1
        pos_total += len(pos)
        hit = flags & set(pos)
        tp_w += len(hit)
        off_claim += len(flags - set(pos))
        if hit:
            detected += 1
        elif flags:
            flagged_elsewhere_only += 1
    out.append(f"annotated claim in fake videos ({located}/{len(fake)} phrases located):")
    out.append(f"  claim recall {detected}/{located} = {detected / located:.3f}   "
               f"(+{flagged_elsewhere_only} videos flagged only away from the annotated claim)")
    out.append(f"  window recall {tp_w}/{pos_total} = {tp_w / pos_total:.3f} (overlapping windows)")

    tot = tp_w + off_claim + nf
    if tot:
        out.append(f"window precision: {tot} flags = {tp_w} on the annotated claim + {off_claim} "
                   f"elsewhere in fake videos + {nf} in not-fake videos")
        out.append(f"  bounds {tp_w / tot:.3f} (off-claim flags wrong) .. "
                   f"{(tp_w + off_claim) / tot:.3f} (off-claim flags right)")
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
        args = [a for a in args if Path(a) != out_path]
    labels = {v["video_id"]: v for v in load_test_set()}
    for v in labels.values():
        v["_positive"] = positive_windows(v) if v["label_fake_news"] else None

    lines = []
    for f in args:
        payload = json.loads(Path(f).read_text(encoding="utf-8"))
        videos, meta = payload["videos"], payload.get("run", {})
        st = meta.get("settings", {})
        name = (f"{st.get('LLAMA_MODEL')}  condition={meta.get('condition', 'rag')}  "
                f"web={meta.get('web_backend')}  ({len(videos)} videos)")
        has_web = any(w["verdict"].endswith("_WEB") for r in videos.values() for w in r["results"])
        views = [("as run", None)] + ([("corpus only (derived)", TO_CORPUS)] if has_web else [])
        for view, mapping in views:
            lines += ["", f"=== {name} -- {view}"] + evaluate(videos, labels, mapping)
    text = "\n".join(lines).lstrip("\n")
    print(text)
    if out_path:
        out_path.write_text("```\n" + text + "\n```\n", encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
