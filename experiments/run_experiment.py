"""Run the pipeline over the held-out test set and score it against the labels."""
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
from experiments.datasets import (  # noqa: E402
    DEV_DATASET, DEV_VIDEO_LABELS, EXCLUDED, TEST_LABELS,
)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "data" / "youtube" / "covid_vaccine_videos_transcribed.json"
OUT_DIR = rag_config.RESULTS_DIR
SMOKE_DIR = rag_config.SMOKE_DIR

FLAG_VERDICTS = {"FAKE_NEWS", "FAKE_NEWS_WEB"}


DATASETS = {
    "test": (TEST_LABELS, TRANSCRIPTS, 100),
    "dev": (DEV_VIDEO_LABELS, DEV_DATASET, 35),
}


def load_test_set(dataset: str = "test") -> list[dict]:
    """Labelled videos of one split, joined with their transcripts."""
    labels_path, transcripts_path, _ = DATASETS[dataset]
    if not labels_path.exists():
        raise SystemExit(f"no labels for --dataset {dataset}: {labels_path} "
                         f"does not exist")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["items"]
    rows = json.loads(transcripts_path.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("videos", [])
    transcripts = {r["video_id"]: r for r in rows}
    out = []
    for row in labels:
        vid = row["video_id"]
        if vid in EXCLUDED or row.get("exclude"):
            continue
        rec = transcripts.get(vid)
        if not rec or not rec.get("transcript"):
            continue
        out.append({**row, "transcript": rec["transcript"],
                    "published_at": rec.get("published_at")})
    return out


def preflight(web: str, dataset: str = "test") -> list[str]:
    """Everything that must be true before a multi-hour run. Returns problems."""
    problems = []

    ok, detail = health()
    print(f"  model server   : {'OK ' if ok else 'FAIL'}  {detail}")
    if not ok:
        problems.append("model server unreachable")
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

    videos = load_test_set(dataset)
    minimum = DATASETS[dataset][2]
    print(f"  {dataset} videos{' ' * (6 - len(dataset))} : {len(videos)} "
          f"labelled with usable transcripts")
    if len(videos) < minimum:
        problems.append(f"only {len(videos)} {dataset} videos resolved; "
                        f"expected at least {minimum}")

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
    if web and web != "none":
        from rag.websearch import cache_stats
        cs = cache_stats()
        print(f"  web cache      : {'on' if cs['enabled'] else 'off'}, "
              f"{cs['entries']} entry(ies) — a non-empty cache means some "
              f"evidence is replayed, not refetched")
    sec = round(21.0 * (23.9 / 17.7), 1) if web and web != "none" else 21.0
    print(f"  claim windows  : ~{windows} "
          f"(~{windows * sec / 3600:.1f} h at {sec} s/window on CPU)")
    return problems


_RECORDED_SETTINGS = (
    "CHUNK_WORDS", "CHUNK_STRIDE", "TOP_K", "EXPAND_NEIGHBOURS",
    "MIN_SIMILARITY", "RELATIVE_CUTOFF", "SCOPE_ANSWERED_SIMILARITY",
    "USE_JSON_SCHEMA", "IS_REMOTE",
    "TOPIC_RANK_BONUS", "ROUTE_GAMMA", "ROUTE_FLOOR", "ROUTE_TOP_N",
    "FLAG_MIN_SIMILARITY", "CLAIM_WINDOW_WORDS", "CLAIM_STRIDE_WORDS",
    "EMBED_MODEL", "LLAMA_MODEL", "LLAMA_URL", "TEMPERATURE",
    "USE_GRAMMAR", "USE_JSON_MODE", "LLM_REASONING", "WHOLE_TRANSCRIPT", "TIME_CHECK",
    "OVERLAP_VETO",
)


def prompt_fingerprint(name: str) -> dict:
    """Name and hash of the prompts in effect, so runs cannot be mixed."""
    import hashlib
    from rag import llm
    def h(obj) -> str:
        text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
        return hashlib.sha1(text.encode()).hexdigest()[:12]
    return {"name": name, "sha1": {
        "rag": h(llm.SYSTEM_PROMPT),
        "parametric": h(llm.PARAMETRIC_PROMPT),
        "contract": h([llm.VERDICT_GRAMMAR, llm.VERDICT_SCHEMA,
                       llm.PARAMETRIC_GRAMMAR, llm.PARAMETRIC_SCHEMA]),
    }}


def apply_prompt(name: str) -> None:
    """Swap the adjudication prompts for a named pair (experiments/prompt_variants.py)."""
    if name == "production":
        return
    from rag import llm
    from experiments.prompt_variants import PROMPT_PAIRS
    if name not in PROMPT_PAIRS:
        raise SystemExit(f"unknown --prompt {name!r}; "
                         f"choose from {sorted(PROMPT_PAIRS)}")
    pair = PROMPT_PAIRS[name]
    llm.SYSTEM_PROMPT = pair["rag"]
    llm.PARAMETRIC_PROMPT = pair["parametric"]
    for key, attr in (("rag_grammar", "VERDICT_GRAMMAR"),
                      ("rag_schema", "VERDICT_SCHEMA"),
                      ("parametric_grammar", "PARAMETRIC_GRAMMAR"),
                      ("parametric_schema", "PARAMETRIC_SCHEMA")):
        if key in pair:
            setattr(llm, attr, pair[key])


def run_metadata(web: str, index, dataset: str = "test",
                 prompt: str = "production") -> dict:
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
        "dataset": dataset,
        "labels_file": str(DATASETS[dataset][0].relative_to(ROOT)),
        "prompt": prompt_fingerprint(prompt),
        "flag_verdicts": sorted(FLAG_VERDICTS),
        "settings": {k: getattr(rag_config, k, None)
                     for k in _RECORDED_SETTINGS},
    }


def run(web: str, limit: int | None, out: Path, condition: str = "rag",
        dataset: str = "test", prompt: str = "production") -> dict:
    """Fact-check every video of a split, checkpointing after each."""
    index = EvidenceIndex.load()
    backend = get_backend(web)
    videos = load_test_set(dataset)
    if limit:
        videos = videos[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    done = payload.get("videos", {})
    meta = payload.get("run") or run_metadata(web, index, dataset, prompt)
    stored = meta.get("condition", "rag")
    if payload and stored != condition:
        raise SystemExit(f"{out} holds a {stored!r} run; refusing to resume "
                         f"it as {condition!r}")
    meta["condition"] = condition
    fp = prompt_fingerprint(prompt)
    if payload and meta.get("prompt", fp) != fp:
        raise SystemExit(f"{out} was run with prompt {meta['prompt']}; "
                         f"refusing to resume it with {fp}")
    meta["prompt"] = fp
    current = run_metadata(web, index, dataset, prompt)["settings"]
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
        whole = None
        if rag_config.WHOLE_TRANSCRIPT:
            from rag import whole as _whole
            whole = (_whole.check_video_parametric(
                         video["transcript"], published=video.get("published_at"))
                     if condition == "parametric" else
                     _whole.check_video_rag(video["transcript"], index,
                                            web_fetch_fn=backend,
                                            published=video.get("published_at")))
            results = whole["rows"]
        elif condition == "parametric":
            results = check_transcript_parametric(
                video["transcript"], published=video.get("published_at"))
        else:
            results = check_transcript(
                video["transcript"], index, verbose=False, web_fetch_fn=backend,
                published=video.get("published_at"),
            )
        flags = [r for r in results if r["verdict"] in FLAG_VERDICTS]
        done[vid] = {
            "label_fake_news": video["label_fake_news"],
            "confidence": video.get("confidence"),
            "narrative": video.get("narrative"),
            "predicted_fake_news": bool(flags),
            **({"model_fake_news": whole["model_fake_news"],
                "claims": whole["claims"], "claims_truncated": whole["truncated"]}
               if whole else {}),
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


def score(out: Path, summary_path: Path | None = None,
          overlap_veto: bool = False) -> None:
    """Confusion matrix and per-narrative breakdown against the labels."""
    if not out.exists():
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
    done = payload.get("videos", payload)
    if overlap_veto and ((payload.get("run") or {}).get("settings") or {}).get(
            "WHOLE_TRANSCRIPT"):
        import builtins
        builtins.print("--overlap-veto needs overlapping windows; this is a "
                       "whole-transcript run (claims), so it does not apply")
        return
    if overlap_veto:
        from rag.pipeline import overlap_veto as _veto
        for r in done.values():
            if isinstance(r, dict) and "results" in r:
                r["results"] = _veto(r["results"])
                r["flags"] = sum(1 for x in r["results"]
                                 if x["verdict"] in FLAG_VERDICTS)
                r["predicted_fake_news"] = r["flags"] > 0
        print("(re-scored with overlap veto)")
    meta = payload.get("run") or {}
    if meta:
        st = meta.get("settings", {})
        print(f"run: {meta.get('started')}  commit {meta.get('git_commit')}  "
              f"web={meta.get('web_backend')}  condition={meta.get('condition', 'rag')}")
        print(f"     dataset={meta.get('dataset', 'test')}  "
              f"prompt={(meta.get('prompt') or {}).get('name', 'unrecorded')}")
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
              "are scored with experiments/score_prompt_ab.py.")
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
    whole_rows = [r for r in done.values() if "model_fake_news" in r]
    if whole_rows:
        disagree = [r for r in whole_rows
                    if bool(r["model_fake_news"]) != r["predicted_fake_news"]]
        print(f"  whole transcript: {sum(r.get('claims', 0) for r in whole_rows)} claims "
              f"over {len(whole_rows)} videos; model's own fake_news disagrees with "
              f"the claim verdicts on {len(disagree)} video(s); "
              f"{sum(1 for r in whole_rows if r.get('claims_truncated'))} truncated "
              f"at the claim cap\n")
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
        summary_path = out.with_name(
            out.stem + ("_ov" if overlap_veto else "") + "_summary.txt")
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
    ap.add_argument("--dataset", default="test", choices=sorted(DATASETS),
                    help="test: the held-out set (default); dev: the tuning set, "
                         "where prompts are compared")
    ap.add_argument("--time-check", action="store_true",
                    help="date the claim and evidence, add a TIME CHECK note to "
                         "windows with time-bound wording, and override "
                         "time-bound flags resting on later evidence")
    ap.add_argument("--overlap-veto", action="store_true",
                    help="withdraw a flag an overlapping window cleared; with "
                         "--score, re-scores an existing run this way")
    ap.add_argument("--whole", action="store_true",
                    help="judge the whole transcript in one go (rag/whole.py) "
                         "instead of 140-word windows; uses the v1 flagging "
                         "instructions, so it takes no --prompt (--time-check works)")
    ap.add_argument("--prompt", default="production",
                    help="prompt pair from experiments/prompt_variants.py "
                         "(default production = rag/llm.py as is; base, rules, "
                         "rules_v2, stance_first, v0 are frozen variants)")
    args = ap.parse_args()
    if args.condition == "parametric" and args.web != "none":
        ap.error("--condition parametric uses no evidence; drop --web")

    _slug = LLAMA_MODEL.replace("/", "-").replace(":", "-")
    _arm = 'parametric' if args.condition == 'parametric' else args.web
    _suffix = ("" if args.dataset == "test" and args.prompt == "production"
               else f"_{args.prompt}")
    _suffix += rag_config.reasoning_tag()
    if args.whole:
        if args.prompt != "production":
            ap.error("--whole has fixed v1 instructions: drop --prompt")
        rag_config.WHOLE_TRANSCRIPT = True
        _suffix = rag_config.reasoning_tag() + "_whole"
    rag_config.TIME_CHECK = args.time_check or rag_config.TIME_CHECK
    if rag_config.TIME_CHECK:
        _suffix += "_tc"
    if not args.score:
        rag_config.OVERLAP_VETO = args.overlap_veto or rag_config.OVERLAP_VETO
        if rag_config.OVERLAP_VETO:
            _suffix += "_ov"
    out = Path(args.out) if args.out else (
        (SMOKE_DIR if args.limit else OUT_DIR)
        / f"{args.dataset}_run_{_slug}_{_arm}{_suffix}.json")
    apply_prompt(args.prompt)

    if args.score:
        score(out, overlap_veto=args.overlap_veto)
        return 0

    if args.clear_cache:
        from rag.websearch import cache_clear
        print(f"web cache: cleared {cache_clear()} entry(ies)\n")

    print("preflight:")
    problems = preflight(args.web, args.dataset)
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall checks passed")
    if args.check:
        return 0

    print()
    print(f"condition: {args.condition}   dataset: {args.dataset}   "
          f"prompt: {args.prompt}")
    run(args.web, args.limit, out, args.condition, args.dataset, args.prompt)
    print()
    score(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
