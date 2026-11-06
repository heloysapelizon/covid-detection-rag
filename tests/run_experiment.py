"""Run the pipeline over the held-out test set and score it against the labels.

    python tests/run_experiment.py --check          # preflight only
    python tests/run_experiment.py                  # corpus only
    python tests/run_experiment.py --web duckduckgo
    python tests/run_experiment.py --score          # score an existing run

This is the final experiment. It runs over the 119 labelled test videos
(data/labels/ground_truth.json), which no parameter was chosen by looking at —
every threshold was derived on separate development collections (docs/dev-sets.md).

A video is labelled at the VIDEO level ("does this video assert something false?")
while the pipeline decides per claim window, so the two are joined by the rule
below: a video is predicted fake news if any of its windows is flagged. That is
the rule the pipeline's own reporting already implies, and it is stated here
rather than chosen after seeing the results.

Progress is checkpointed per video, so an interrupted run resumes. Nothing is
overwritten: re-running skips videos already present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config as rag_config  # noqa: E402
from rag.config import (  # noqa: E402
    IS_REMOTE,
    LLAMA_MODEL,
    LLAMA_URL,
    USE_GRAMMAR,
    USE_JSON_SCHEMA,
)
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import health  # noqa: E402
from rag.pipeline import check_transcript, check_transcript_parametric  # noqa: E402
from rag.websearch import get_backend  # noqa: E402
from tests.datasets import EXCLUDED, TEST_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "data" / "youtube" / "covid_vaccine_videos_transcribed.json"
OUT_DIR = ROOT / "data" / "results"

# A video counts as predicted fake news when at least one window is flagged.
# FAKE_NEWS_WEB counts too: it is a flag, distinguished only by provenance, and
# is reported separately below as well.
FLAG_VERDICTS = {"FAKE_NEWS", "FAKE_NEWS_WEB"}


def load_test_set() -> list[dict]:
    """Labelled test videos joined with their transcripts."""
    labels = json.loads(TEST_LABELS.read_text(encoding="utf-8"))["items"]
    transcripts = {
        r["video_id"]: r
        for r in json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    }
    out = []
    for row in labels:
        vid = row["video_id"]
        if vid in EXCLUDED:
            continue
        rec = transcripts.get(vid)
        if not rec or not rec.get("transcript"):
            continue
        out.append({**row, "transcript": rec["transcript"]})
    return out


def preflight(web: str) -> list[str]:
    """Everything that must be true before a multi-hour run. Returns problems."""
    problems = []

    ok, detail = health()
    print(f"  model server   : {'OK ' if ok else 'FAIL'}  {detail}")
    if not ok:
        problems.append("model server unreachable")
    # Named precisely: the three mechanisms give different guarantees, and
    # mislabelling OpenRouter as "Ollama JSON mode" would understate its.
    if USE_GRAMMAR:
        constraint = "GBNF grammar, enforced during sampling (llama.cpp)"
    elif USE_JSON_SCHEMA:
        constraint = "json_schema, provider-validated (OpenRouter)"
    else:
        constraint = "json_object, valid JSON only — shape not enforced (Ollama)"
    print(f"  output constr. : {constraint}")
    if IS_REMOTE:
        print("  NOTE           : remote backend — transcript text leaves this "
              "machine")
    print(f"  model          : {LLAMA_MODEL} at {LLAMA_URL}")

    try:
        index = EvidenceIndex.load()
        print(f"  evidence index : OK   {len(index.passages)} passages")
    except Exception as e:
        print(f"  evidence index : FAIL {type(e).__name__}: {e}")
        problems.append("evidence index missing or stale — run --build-index")

    videos = load_test_set()
    print(f"  test videos    : {len(videos)} labelled with usable transcripts")
    if len(videos) < 100:
        problems.append(f"only {len(videos)} test videos resolved; expected ~119")

    try:
        backend = get_backend(web)
    except ValueError as e:
        print(f"  web fallback   : FAIL {e}")
        problems.append(str(e))
        backend = None
    else:
        print(f"  web fallback   : {web}"
              + ("" if backend else "  (corpus only)"))
        if backend is not None:
            # A backend that silently returns nothing is worse than one that
            # fails loudly: the run would complete with every uncovered claim
            # left at NO_EVIDENCE and no sign that the web tier never worked.
            probe = backend("covid vaccine safety", 1)
            if probe:
                print(f"                   probe OK -> {probe[0]['url'][:52]}")
            else:
                problems.append(
                    f"--web {web} returned nothing for a probe query; "
                    + ("set TAVILY_API_KEY (environment or .env)"
                       if web == "tavily" else "check network access")
                )

    windows = sum(max(1, (len(v["transcript"].split()) - 140) // 70 + 1)
                  for v in videos)
    # 17.7 s/window measured on CPU (M2 Pro, qwen2.5:14b Q4_K_M, Ollama) over a
    # 42-window smoke test. GPU offload is typically an order of magnitude faster,
    # so treat this as the pessimistic bound.
    # Both rates measured on this project's CPU setup (docs/web-search.md).
    # The web fallback costs a search round-trip plus a second adjudication on
    # the ~40% of windows that trigger it, so it is not the same rate.
    if web and web != "none":
        from rag.websearch import cache_stats
        cs = cache_stats()
        print(f"  web cache      : {'on' if cs['enabled'] else 'off'}, "
              f"{cs['entries']} entry(ies) — a non-empty cache means some "
              f"evidence is replayed, not refetched")
    # Medians measured on test-set windows (docs/web-search.md). The corpus-only
    # figure was first taken from the 3-video smoke test and read 17.7 s, which
    # was ~20% optimistic: that sample included a short video. 21.0 s is the
    # median over windows from five different test videos.
    # The web figure (23.9 s) was measured with DuckDuckGo; Tavily's REST call
    # is comparable, so it is used for both. The ratio to the corpus-only
    # median is what matters, and it comes from the same smoke test.
    sec = round(21.0 * (23.9 / 17.7), 1) if web and web != "none" else 21.0
    print(f"  claim windows  : ~{windows} "
          f"(~{windows * sec / 3600:.1f} h at {sec} s/window on CPU)")
    return problems


# Every setting that could change a verdict. Recorded with the results so a run
# can be diagnosed months later without guessing which configuration produced
# it — a figure from gamma=0.30 and one from gamma=0.60 are otherwise
# indistinguishable on disk.
_RECORDED_SETTINGS = (
    "CHUNK_WORDS", "CHUNK_STRIDE", "TOP_K", "EXPAND_NEIGHBOURS",
    "MIN_SIMILARITY", "RELATIVE_CUTOFF", "SCOPE_ANSWERED_SIMILARITY",
    "USE_JSON_SCHEMA", "IS_REMOTE",
    "TOPIC_RANK_BONUS", "ROUTE_GAMMA", "ROUTE_FLOOR", "ROUTE_TOP_N",
    "FLAG_MIN_SIMILARITY", "CLAIM_WINDOW_WORDS", "CLAIM_STRIDE_WORDS",
    "EMBED_MODEL", "LLAMA_MODEL", "LLAMA_URL", "TEMPERATURE",
    "USE_GRAMMAR", "USE_JSON_MODE",
)


def run_metadata(web: str, index) -> dict:
    """Configuration and provenance for this run."""
    import platform
    import subprocess
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "web_backend": web,
        "corpus_passages": len(index.passages),
        "labels_file": str(TEST_LABELS.relative_to(ROOT)),
        "flag_verdicts": sorted(FLAG_VERDICTS),
        "settings": {k: getattr(rag_config, k, None)
                     for k in _RECORDED_SETTINGS},
    }


def run(web: str, limit: int | None, out: Path, condition: str = "rag") -> dict:
    """Fact-check every test video, checkpointing after each."""
    index = EvidenceIndex.load()
    backend = get_backend(web)
    videos = load_test_set()
    if limit:
        videos = videos[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    done = payload.get("videos", {})
    meta = payload.get("run") or run_metadata(web, index)
    # A results file belongs to one condition; resuming it under another
    # would mix evidence-grounded and parametric verdicts in one figure.
    stored = meta.get("condition", "rag")
    if payload and stored != condition:
        raise SystemExit(f"{out} holds a {stored!r} run; refusing to resume "
                         f"it as {condition!r}")
    meta["condition"] = condition
    # A resumed run must not silently mix configurations.
    current = run_metadata(web, index)["settings"]
    if meta.get("settings") and meta["settings"] != current:
        changed = {k: (meta["settings"].get(k), current.get(k))
                   for k in current
                   if meta["settings"].get(k) != current.get(k)}
        print("WARNING: settings changed since this run started:")
        for k, (was, now) in changed.items():
            print(f"  {k}: {was} -> {now}")
        print("  Results already on disk were produced with the old values.\n")

    def save():
        out.write_text(
            json.dumps({"run": meta, "videos": done},
                       ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    todo = [v for v in videos if v["video_id"] not in done]
    print(f"{len(todo)} video(s) to run, {len(done)} already done\n")

    start = time.time()
    for n, video in enumerate(todo, 1):
        vid = video["video_id"]
        print(f"[{n}/{len(todo)}] {vid}  {video['title'][:52]}")
        if condition == "parametric":
            results = check_transcript_parametric(video["transcript"])
        else:
            results = check_transcript(
                video["transcript"], index, verbose=False, web_fetch_fn=backend,
            )
        flags = [r for r in results if r["verdict"] in FLAG_VERDICTS]
        done[vid] = {
            "label_fake_news": video["label_fake_news"],
            "confidence": video.get("confidence"),
            "narrative": video.get("narrative"),
            "predicted_fake_news": bool(flags),
            "windows": len(results),
            "flags": len(flags),
            "web_windows": sum(1 for r in results
                               if r.get("evidence_origin") == "web"),
            "results": results,
        }
        save()
        mark = "FLAG" if flags else "ok  "
        truth = "fake" if video["label_fake_news"] else "not "
        print(f"    {mark}  {len(flags)}/{len(results)} flagged   label={truth}")

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["minutes"] = round((time.time() - start) / 60, 1)
    save()
    print(f"\nfinished in {meta['minutes']} min")
    return done


def score(out: Path, summary_path: Path | None = None) -> None:
    """Confusion matrix and per-narrative breakdown against the labels.

    Writes the same text to summary_path as well as stdout, so the scored
    figures survive without the console log.
    """
    if not out.exists():
        # builtins.print: this function defines a local `print` below, which
        # makes the bare name local throughout and this line an UnboundLocalError.
        import builtins
        builtins.print(f"no results at {out}")
        return
    lines: list[str] = []

    def print(*args, **kwargs):           # noqa: A001 - capture, then emit
        import builtins
        text = " ".join(str(a) for a in args)
        lines.append(text)
        builtins.print(text, **kwargs)
    payload = json.loads(out.read_text(encoding="utf-8"))
    done = payload.get("videos", payload)   # tolerate the pre-metadata shape
    meta = payload.get("run") or {}
    if meta:
        st = meta.get("settings", {})
        print(f"run: {meta.get('started')}  commit {meta.get('git_commit')}  "
              f"web={meta.get('web_backend')}  condition={meta.get('condition', 'rag')}")
        print(f"     model={st.get('LLAMA_MODEL')} gamma={st.get('ROUTE_GAMMA')} "
              f"chunk={st.get('CHUNK_WORDS')}/{st.get('CHUNK_STRIDE')} "
              f"grammar={st.get('USE_GRAMMAR')}")
        print(f"     corpus={meta.get('corpus_passages')} passages\n")
    bad = [k for k, r in done.items()
           if not isinstance(r, dict) or "label_fake_news" not in r]
    if bad:
        print(f"{out} does not look like an experiment results file: "
              f"{len(bad)} of {len(done)} entries have no 'label_fake_news' "
              f"(first: {bad[0]!r}).")
        print("Experiment runs are written by this script; prompt A/B results "
              "are scored with tests/score_prompt_ab.py.")
        return

    tp = sum(1 for r in done.values()
             if r["label_fake_news"] and r["predicted_fake_news"])
    fp = sum(1 for r in done.values()
             if not r["label_fake_news"] and r["predicted_fake_news"])
    fn = sum(1 for r in done.values()
             if r["label_fake_news"] and not r["predicted_fake_news"])
    tn = sum(1 for r in done.values()
             if not r["label_fake_news"] and not r["predicted_fake_news"])

    print(f"scored {len(done)} video(s)\n")
    print("                  predicted")
    print("                fake    not")
    print(f"  label fake    {tp:4}   {fn:4}")
    print(f"  label not     {fp:4}   {tn:4}\n")

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(done) if done else 0.0
    print(f"  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}   "
          f"accuracy {acc:.3f}")

    web = sum(r.get("web_windows", 0) for r in done.values())
    print(f"\n  windows answered on web evidence: {web}")

    missed = [(v, r) for v, r in done.items()
              if r["label_fake_news"] and not r["predicted_fake_news"]]
    if missed:
        print(f"\n  false negatives ({len(missed)}):")
        for vid, r in missed:
            print(f"    {vid:12} [{r.get('confidence')}] {r.get('narrative')}")
    wrong = [(v, r) for v, r in done.items()
             if not r["label_fake_news"] and r["predicted_fake_news"]]
    if wrong:
        print(f"\n  false positives ({len(wrong)}):")
        for vid, r in wrong:
            print(f"    {vid:12} {r['flags']}/{r['windows']} windows flagged")

    if summary_path is None:
        summary_path = out.with_name(out.stem + "_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import builtins
    builtins.print(f"\nsummary written to {summary_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="none",
                    help="web fallback backend: duckduckgo, tavily, none")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N videos (for a smoke test)")
    ap.add_argument("--out", default=None, help="results file")
    ap.add_argument("--check", action="store_true",
                    help="run preflight checks and exit")
    ap.add_argument("--score", action="store_true",
                    help="score existing results without running")
    ap.add_argument("--clear-cache", action="store_true",
                    help="empty the web-search cache first, so this run's "
                         "evidence is fetched fresh (use for a measurement run)")
    ap.add_argument("--condition", default="rag", choices=("rag", "parametric"),
                    help="rag: retrieve evidence (default); parametric: no "
                         "retrieval, the model judges from its own knowledge")
    args = ap.parse_args()
    if args.condition == "parametric" and args.web != "none":
        ap.error("--condition parametric uses no evidence; drop --web")

    # The model is part of the filename: a four-model comparison writing to one
    # default path would have each run silently resume from, and overwrite,
    # another model's results.
    _slug = LLAMA_MODEL.replace("/", "-").replace(":", "-")
    out = Path(args.out) if args.out else (
        OUT_DIR / f"test_run_{_slug}_"
        f"{'parametric' if args.condition == 'parametric' else args.web}.json")

    if args.score:
        score(out)
        return 0

    if args.clear_cache:
        from rag.websearch import cache_clear
        print(f"web cache: cleared {cache_clear()} entry(ies)\n")

    print("preflight:")
    problems = preflight(args.web)
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall checks passed")
    if args.check:
        return 0

    print()
    print(f"condition: {args.condition}")
    run(args.web, args.limit, out, args.condition)
    print()
    score(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
