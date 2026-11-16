"""Strip or restore the window text in windowed result files.

Each window of a windowed run (v1, v2) records its text in `claim_text`, so a
result file carries most of each video's transcript. The windows are a
deterministic function of the transcript (rag.pipeline.segment_claims), so the
text can be removed for publication and put back from the transcript dataset.

    python experiments/redact_text.py --strip   data/results/v1/test_run_*.json
    python experiments/redact_text.py --restore data/results/v1/test_run_*.json
    python experiments/redact_text.py --check   data/results/v1/test_run_*.json

--strip   sets every `claim_text` to "" (the window index is kept).
--restore recomputes `claim_text` from the transcripts of the test and dev sets.
--check   reports whether each file's windows match the transcripts exactly.

Whole-transcript runs (v3) are skipped: their rows are claims the model
extracted, not transcript windows, and cannot be recomputed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.datasets import DEV_DATASET, FALLBACK_DATASET  # noqa: E402
from rag.pipeline import segment_claims  # noqa: E402

TRANSCRIPTS = (
    ROOT / "data" / "youtube" / "covid_vaccine_videos_transcribed.json",
    DEV_DATASET,
    FALLBACK_DATASET,
)


def load_transcripts() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in TRANSCRIPTS:
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else rows.get("videos", [])
        for r in rows:
            if r.get("transcript"):
                out.setdefault(r["video_id"], r["transcript"])
    return out


def is_whole(payload: dict) -> bool:
    videos = payload.get("videos") or {}
    first = next(iter(videos.values()), {})
    return "claims" in first or bool(payload.get("run", {}).get("whole"))


def process(path: Path, mode: str, transcripts: dict[str, str]) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "videos" not in payload:
        return "skipped (not a results file)"
    if is_whole(payload):
        return "skipped (whole-transcript run)"
    changed = mismatched = missing = 0
    for vid, rec in payload["videos"].items():
        windows = None
        if mode in ("restore", "check"):
            text = transcripts.get(vid)
            if text is None:
                missing += 1
                continue
            windows = segment_claims(text)
        for w in rec["results"]:
            if mode == "strip":
                if w.get("claim_text"):
                    w["claim_text"] = ""
                    changed += 1
                continue
            i = w["window"] - 1
            expected = windows[i] if i < len(windows) else None
            if expected is None:
                mismatched += 1
            elif mode == "restore":
                if w.get("claim_text") != expected:
                    w["claim_text"] = expected
                    changed += 1
            elif w.get("claim_text") != expected:
                mismatched += 1
    if mode != "check":
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    parts = [f"{changed} windows changed"] if mode != "check" else []
    if mismatched:
        parts.append(f"{mismatched} windows differ from the transcript")
    if missing:
        parts.append(f"{missing} videos without a transcript")
    if mode == "check" and not mismatched and not missing:
        parts.append("all windows match the transcripts")
    return ", ".join(parts)


def main() -> int:
    args = sys.argv[1:]
    modes = [a for a in ("--strip", "--restore", "--check") if a in args]
    files = [Path(a) for a in args if not a.startswith("--")]
    if len(modes) != 1 or not files:
        print(__doc__, file=sys.stderr)
        return 2
    mode = modes[0].lstrip("-")
    transcripts = load_transcripts() if mode != "strip" else {}
    for f in files:
        print(f"{f}: {process(f, mode, transcripts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
