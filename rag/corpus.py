"""Turn reference PDFs into retrievable passages."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import (
    CHUNK_STRIDE,
    CHUNK_WORDS,
    EVIDENCE_DIR,
    PASSAGE_DIR,
    PDF_DIR,
)

_DROP_SECTIONS = (
    "acknowledgement", "acknowledgment", "author contributions",
    "conflict of interest", "conflicts of interest", "competing interests",
    "funding", "data availability", "ethics", "consent", "abbreviations",
    "orcid", "copyright", "license", "licence", "editor", "peer review",
    "publisher's note", "supplementary", "appendix",
)

_END_SECTIONS = (
    "references", "bibliography", "literature cited", "works cited",
)

_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?([A-Z][A-Za-z0-9 \-–&',/()]{2,60})\s*:?\s*$"
)


def _normalise(text: str) -> str:
    """Repair the artefacts PDF text extraction leaves behind."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_dropped(heading: str) -> bool:
    """Whether this one section is administrative and should be skipped."""
    h = heading.strip().lower().rstrip(":")
    return any(h == pat or h.startswith(pat) for pat in _DROP_SECTIONS)


def is_end(heading: str) -> bool:
    """Whether this heading begins the back matter — stop extracting here."""
    h = heading.strip().lower().rstrip(":")
    return any(h == pat or h.startswith(pat) for pat in _END_SECTIONS)


def split_sections(raw: str) -> list[tuple[str, str]]:
    """Split raw text into (heading, body) pairs at every heading-shaped line."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in raw.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines)) for h, lines in sections]


def chunk(text: str, words: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE,
          tail_min: int = 40) -> list[str]:
    """Split text into overlapping word windows."""
    w = text.split()
    if len(w) <= words:
        return [" ".join(w)] if w else []
    pieces = [
        " ".join(w[i:i + words])
        for i in range(0, max(1, len(w) - words + stride), stride)
    ]
    if len(pieces) > 1 and len(pieces[-1].split()) < tail_min:
        pieces.pop()
        pieces[-1] = " ".join(w[(len(pieces) - 1) * stride:])
    return pieces


def extract_pdf(
    pdf_path: Path,
    topic: str,
    meta: dict,
    min_words: int = 40,
) -> list[dict]:
    """Extract section-filtered, chunked passages from one PDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    raw = "\n".join(page.get_text() for page in doc)
    n_pages = doc.page_count
    doc.close()

    passages: list[dict] = []
    seen_headings: dict[str, int] = {}

    for heading, body in split_sections(raw):
        if is_end(heading):
            break
        if is_dropped(heading):
            continue
        body = _normalise(body)
        if len(body.split()) < min_words:
            continue

        key = heading or "front"
        occurrence = seen_headings.get(key, 0)
        seen_headings[key] = occurrence + 1
        suffix = "" if occurrence == 0 else f".{occurrence}"

        for i, piece in enumerate(chunk(body)):
            if len(piece.split()) < min_words:
                continue
            passages.append({
                "id": f"{pdf_path.stem}#{key}{suffix}:{i}".replace(" ", "_"),
                "topic": topic,
                "section": heading or "front matter",
                "text": piece,
                "publisher": meta.get("publisher", ""),
                "source_file": pdf_path.name,
                "pages": n_pages,
                "ordinal": len(passages),
            })
    return passages


def load_manifest(evidence_dir: Path = EVIDENCE_DIR) -> dict:
    """Read sources.json, the map of topic -> papers."""
    path = evidence_dir / "sources.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_passages(
    pdf_dir: Path = PDF_DIR,
    out_dir: Path = PASSAGE_DIR,
    verbose: bool = True,
) -> int:
    """Extract every manifest PDF present on disk into data/evidence/passages/."""
    manifest = load_manifest()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    missing: list[str] = []
    groups = [(t, spec["files"]) for t, spec in manifest["topics"].items()]
    groups.append(("general", manifest.get("general_files", [])))

    for topic, entries in groups:
        for entry in entries:
            pdf = pdf_dir / entry["filename"]
            if not pdf.exists():
                missing.append(entry["filename"])
                continue
            passages = extract_pdf(pdf, topic, entry)
            out = out_dir / f"{pdf.stem}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for p in passages:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(passages)
            if verbose:
                print(f"  {topic:16} {pdf.name[:40]:40} {len(passages):4d} passages")

    if missing and verbose:
        print(f"\n  {len(missing)} PDF(s) in the manifest but not present:")
        for m in missing[:10]:
            print(f"    {m}")
    return total


def load_passages(passage_dir: Path = PASSAGE_DIR) -> list[dict]:
    """Read every extracted passage JSONL."""
    out: list[dict] = []
    if not passage_dir.exists():
        return out
    for path in sorted(passage_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out
