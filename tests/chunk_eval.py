"""Chunk-configuration sweep over full transcripts and the full evidence corpus.

Replaces an earlier version that used hand-picked transcript excerpts and a single
paper. Both of those biased the result: excerpts are cleaner than real windows, and
one document cannot compete for rank the way a real corpus does.

What this does, for each chunk configuration:

  1. Re-extract and re-embed EVERY PDF in data/evidence/pdf/ at that setting.
  2. Segment each labelled transcript in full, exactly as rag.pipeline does.
  3. Retrieve for every window and score against tests/chunk_eval_labels.json.

Scoring needs ground truth, so it comes from a hand-labelled file rather than
heuristics. Build or extend it with:

    python tests/chunk_eval.py --make-labels          # propose windows to label
    python tests/chunk_eval.py --make-labels --topic infertility

Then fill in `relevant_passage_ids` for each entry and run:

    python tests/chunk_eval.py                       # sweep default configs
    python tests/chunk_eval.py --configs 55/35,90/60
    python tests/chunk_eval.py --limit 5             # fewer labelled items

Metrics, per configuration:
  recall@k      labelled-relevant passages that appear in the top k
  precision@k   retrieved passages that are labelled relevant
  hit rate      labelled windows with at least one relevant passage retrieved
  coverage      windows retrieving anything above MIN_SIMILARITY

Because a passage id embeds its chunk boundaries, ids differ between
configurations. Labels therefore record the *text* of the relevant evidence and a
passage counts as relevant when it contains that text, which is comparable across
configurations.

Not part of the experiment: this measures retrieval, it does not produce data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Single definition, so the rule deciding which transcripts enter an
# experiment cannot drift between the pipeline and the eval scripts.
from transcribe.config import is_usable  # noqa: E402

from rag.config import (
    CHUNK_STRIDE,
    CHUNK_WORDS,
    CLAIM_STRIDE_WORDS,
    CLAIM_WINDOW_WORDS,
    EMBED_MODEL,
    EXPAND_NEIGHBOURS,
    MIN_SIMILARITY,
    PDF_DIR,
    TOP_K,
)
from rag.corpus import (

    _normalise,
    chunk,
    is_dropped,
    is_end,
    load_manifest,
    split_sections,
)
from rag.pipeline import segment_claims

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "youtube"
LABELS = Path(__file__).resolve().parent / "chunk_eval_labels.json"

DEFAULT_CONFIGS = [(55, 35), (90, 60), (140, 90), (220, 150)]


# ---------------------------------------------------------------- corpus
def build_corpus(words: int, stride: int, min_words: int = 40) -> list[dict]:
    """Extract and chunk every downloaded PDF at one configuration.

    Mirrors rag.corpus.extract_pdf but keeps results in memory, so a sweep does
    not overwrite the committed passages.
    """
    import pymupdf

    manifest = load_manifest()
    out: list[dict] = []
    # Topic documents, then the general ones — the same grouping as
    # rag.corpus.build_passages. Omitting the general group measured retrieval
    # against a corpus the pipeline never uses: the general documents are in
    # scope for every claim, so leaving them out both shrank the corpus by 997
    # passages and removed the passages most likely to compete on vocabulary.
    groups = [(t, spec["files"]) for t, spec in manifest["topics"].items()]
    groups.append(("general", manifest.get("general_files", [])))
    for topic, entries in groups:
        for entry in entries:
            pdf = PDF_DIR / entry["filename"]
            if not pdf.exists():
                continue
            doc = pymupdf.open(pdf)
            raw = "\n".join(p.get_text() for p in doc)
            doc.close()

            seen: dict[str, int] = {}
            ordinal = 0
            for heading, body in split_sections(raw):
                if is_end(heading):
                    break
                if is_dropped(heading):
                    continue
                body = _normalise(body)
                if len(body.split()) < 40:
                    continue
                key = heading or "front"
                occ = seen.get(key, 0)
                seen[key] = occ + 1
                suffix = "" if occ == 0 else f".{occ}"
                for i, piece in enumerate(chunk(body, words, stride)):
                    if len(piece.split()) < min_words:
                        continue
                    out.append({
                        "id": f"{pdf.stem}#{key}{suffix}:{i}".replace(" ", "_"),
                        "topic": topic,
                        "section": heading or "front matter",
                        "text": piece,
                        "source_file": pdf.name,
                        "ordinal": ordinal,
                    })
                    ordinal += 1
    return out


def embed(corpus: list[dict], model) -> np.ndarray:
    return model.encode(
        # Text only — matches rag.index._embed_text; see its docstring for why
        # the section heading is not embedded.
        [p["text"] for p in corpus],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def retrieve(
    query: str, corpus: list[dict], vectors: np.ndarray, model,
    top_k: int, min_similarity: float, expand: int,
) -> list[dict]:
    """Top-k with neighbour expansion — the same contract as EvidenceIndex."""
    q = model.encode([query], normalize_embeddings=True)
    scores = (vectors @ q[0].T).ravel()

    by_pos = {(p["source_file"], p["ordinal"]): i for i, p in enumerate(corpus)}
    hits: list[int] = []
    for i in np.argsort(-scores):
        if len(hits) >= top_k or scores[i] < min_similarity:
            break
        hits.append(int(i))

    chosen = list(hits)
    if expand:
        for i in hits:
            for off in range(-expand, expand + 1):
                j = by_pos.get((corpus[i]["source_file"],
                                corpus[i]["ordinal"] + off))
                if j is not None and j not in chosen:
                    chosen.append(j)
    return [{**corpus[i], "similarity": round(float(scores[i]), 4)}
            for i in chosen]


# ---------------------------------------------------------------- labels
def transcripts(dataset: str | None = None) -> dict[str, dict]:
    """Usable transcripts joined with their collected metadata.

    With `dataset`, reads <dataset>_transcribed.json directly — that is how the
    v2 dev set is loaded. Without it, reads the v1 checkpoint plus the v1
    collection, which is where the original 124 videos live.
    """
    if dataset:
        path = DATA / f"{dataset}_transcribed.json"
        if not path.exists():
            sys.exit(f"No transcribed dataset at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("videos", [])
        return {
            r["video_id"]: r for r in rows
            if is_usable(r.get("transcript_status")) and r.get("transcript")
        }
    cp = json.loads((DATA / "transcripts_checkpoint.json").read_text())
    meta = {x["video_id"]: x
            for x in json.loads((DATA / "covid_vaccine_videos.json").read_text())}
    return {
        vid: {**meta.get(vid, {"video_id": vid}), "transcript": v["transcript"]}
        for vid, v in cp.items()
        if is_usable(v.get("transcript_status")) and v.get("transcript")
    }


def make_labels(topic: str | None, per_video: int, limit: int,
                dataset: str | None = None, out: Path | None = None) -> None:
    """Propose transcript windows to label, with retrieval candidates shown.

    Writes a skeleton to tests/chunk_eval_labels.json. Each entry needs
    `relevant_evidence` filled in: short verbatim phrases from the passage that
    actually answers the window. Phrases, not ids, so labels survive re-chunking.
    """
    from sentence_transformers import SentenceTransformer

    # Candidates are proposed at the current default so the shown text matches
    # what the committed index holds; scoring later rebuilds per configuration.
    corpus = build_corpus(CHUNK_WORDS, CHUNK_STRIDE)
    if not corpus:
        sys.exit(f"No PDFs in {PDF_DIR}. Download sources first.")
    model = SentenceTransformer(EMBED_MODEL)
    vectors = embed(corpus, model)

    target = out or LABELS
    existing = (json.loads(target.read_text())
                if target.exists() else {"items": []})
    already = {(i["video_id"], i["window"]) for i in existing["items"]}

    pool = transcripts(dataset)
    if topic:
        pool = {k: v for k, v in pool.items() if topic in (v.get("query") or "")}

    items = list(existing["items"])
    for vid, rec in list(pool.items())[:limit]:
        windows = segment_claims(rec["transcript"])
        # Label the windows that retrieve most strongly: they are where a wrong
        # answer does real damage, and where configurations actually differ.
        scored = []
        for n, w in enumerate(windows):
            hits = retrieve(w, corpus, vectors, model, 3, MIN_SIMILARITY, 0)
            if hits:
                scored.append((hits[0]["similarity"], n, w, hits))
        scored.sort(reverse=True)
        for _, n, w, hits in scored[:per_video]:
            if (vid, n) in already:
                continue
            items.append({
                "video_id": vid,
                "channel": rec.get("channel", ""),
                "title": rec.get("title", "")[:70],
                "window": n,
                "window_text": w,
                "candidates": [
                    {"id": h["id"], "similarity": h["similarity"],
                     "source_file": h["source_file"], "text": h["text"][:200]}
                    for h in hits
                ],
                "relevant_evidence": [],
                "note": "FILL IN relevant_evidence: verbatim phrases from the "
                        "passage(s) that actually answer this window. Leave empty "
                        "if no evidence in the corpus addresses it.",
            })

    out_path = out or LABELS
    out_path.write_text(
        json.dumps({
            "description": "Hand-labelled ground truth for chunk evaluation. "
                           "relevant_evidence holds verbatim phrases, not passage "
                           "ids, so labels stay valid when chunk size changes.",
            "window_words": CLAIM_WINDOW_WORDS,
            "window_stride": CLAIM_STRIDE_WORDS,
            "items": items,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    todo = sum(1 for i in items if not i["relevant_evidence"])
    print(f"Wrote {out_path}")
    print(f"  {len(items)} item(s), {todo} still need relevant_evidence")


# ---------------------------------------------------------------- scoring
def score_config(
    words: int, stride: int, labelled: list[dict], model,
    top_k: int, expand: int,
) -> dict:
    """Run every labelled window against a corpus built at this configuration."""
    corpus = build_corpus(words, stride)
    vectors = embed(corpus, model)

    tp = fp = fn = 0
    hits_with_relevant = 0
    covered = 0
    scores: list[float] = []

    for item in labelled:
        wanted = [p.lower() for p in item["relevant_evidence"]]
        got = retrieve(item["window_text"], corpus, vectors, model,
                       top_k, MIN_SIMILARITY, expand)
        if got:
            covered += 1
            scores.append(got[0]["similarity"])

        relevant_hits = [
            g for g in got
            if any(w in g["text"].lower() for w in wanted)
        ]
        found = {w for w in wanted
                 if any(w in g["text"].lower() for g in got)}

        tp += len(relevant_hits)
        fp += len(got) - len(relevant_hits)
        fn += len(wanted) - len(found)
        hits_with_relevant += bool(relevant_hits)

    n = len(labelled)
    return {
        "passages": len(corpus),
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "hit_rate": hits_with_relevant / n if n else 0.0,
        "coverage": covered / n if n else 0.0,
        "mean_top1": float(np.mean(scores)) if scores else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python tests/chunk_eval.py",
        description="Sweep chunk configurations over full transcripts and the "
                    "full evidence corpus.",
    )
    ap.add_argument("--make-labels", action="store_true",
                    help="propose transcript windows to label, then exit")
    ap.add_argument("--topic", default=None,
                    help="with --make-labels, only transcripts whose collection "
                         "query contains this string")
    ap.add_argument("--per-video", type=int, default=2,
                    help="with --make-labels, windows per transcript (default 2)")
    ap.add_argument("--limit", type=int, default=12,
                    help="max transcripts to propose, or labelled items to score")
    ap.add_argument("--configs", default=None,
                    help="comma-separated words/stride pairs, e.g. 55/35,90/60")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--expand", type=int, default=EXPAND_NEIGHBOURS)
    ap.add_argument("--dataset", default=None,
                    help="transcribed dataset name to propose windows from "
                         "(default: the v1 checkpoint)")
    ap.add_argument("--labels", default=None,
                    help="label file to write or score against "
                         "(default: tests/chunk_eval_labels.json)")
    args = ap.parse_args(argv)

    if args.make_labels:
        make_labels(args.topic, args.per_video, args.limit,
                    dataset=args.dataset,
                    out=Path(args.labels) if args.labels else None)
        return 0

    labels_path = Path(args.labels) if args.labels else LABELS
    if not labels_path.exists():
        print(f"No labels at {labels_path}.\nCreate them first:\n"
              f"    python tests/chunk_eval.py --make-labels", file=sys.stderr)
        return 1

    data = json.loads(labels_path.read_text())
    labelled = [i for i in data["items"] if i["relevant_evidence"]][:args.limit]
    if not labelled:
        print(f"{labels_path} has no item with relevant_evidence filled in.\n"
              f"Label some windows, then re-run.", file=sys.stderr)
        return 1

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split("/"))
                   for c in args.configs.split(",")]

    from sentence_transformers import SentenceTransformer


    model = SentenceTransformer(EMBED_MODEL)

    pdfs = sorted(p.name for p in PDF_DIR.glob("*.pdf"))
    print(f"labelled windows : {len(labelled)}")
    print(f"evidence PDFs    : {len(pdfs)}")
    print(f"transcript window: {data['window_words']}w / "
          f"{data['window_stride']}s")
    print(f"retrieval        : top_k={args.top_k} expand={args.expand} "
          f"min_sim={MIN_SIMILARITY}")
    print(f"embeddings       : {EMBED_MODEL}\n")

    print(f"{'chunk':>11} {'psg':>5} {'recall':>7} {'prec':>7} "
          f"{'hit rate':>9} {'coverage':>9} {'mean top1':>10}")
    print("-" * 64)
    for words, stride in configs:
        r = score_config(words, stride, labelled, model, args.top_k, args.expand)
        print(f"{words:4d}w/{stride:3d}s {r['passages']:5d} "
              f"{r['recall']:7.3f} {r['precision']:7.3f} "
              f"{r['hit_rate']:9.3f} {r['coverage']:9.3f} "
              f"{r['mean_top1']:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
