"""Generate data/evidence/sources.json from the topic groups declared below.

    python -m rag --build-manifest

Declared by hand (below): which filename belongs to which topic, and what each
topic covers. That is editorial judgement and cannot be derived.

Derived per document: the DOI is read out of the PDF's own text, then publisher,
title and date come from Crossref. Citations therefore resolve to the publisher of
record rather than to any private copy.

Licensing is not handled here. Only documents whose licence permits this use are
placed in data/evidence/pdf/ in the first place, and that folder is gitignored —
the committed passages are the redistributable part.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import EVIDENCE_DIR, PDF_DIR

# --------------------------------------------------------------------------
# Declared by hand: what each topic covers. `addresses` is the text the router
# embeds to decide which topics a claim should be searched against.
# --------------------------------------------------------------------------
TOPICS: dict[str, str] = {
    "infertility": (
        "Claims that COVID-19 vaccines cause infertility or sterility in women "
        "or men."
    ),
    "myocarditis": (
        "Claims about myocarditis and cardiac risk following mRNA COVID-19 "
        "vaccination."
    ),
    "blood_clots": (
        "Claims about blood clots and thrombosis following COVID-19 vaccination."
    ),
    "natural_immunity": (
        "Claims comparing natural infection-acquired immunity with "
        "vaccine-induced immunity."
    ),
}

# Declared by hand: filename -> topic. Add a PDF to data/evidence/pdf/, add its
# filename here, then regenerate.
# There is deliberately no `viral_shedding` topic. Shedding is not studied as an
# mRNA-vaccine phenomenon — it does not occur with non-live vaccines, which is
# itself the answer — so no topic-specific literature exists to fill it. Six
# searches found nothing admissible. The subject is covered by the general tier
# (the ICH shedding guidance and the CDC sheet), which is in scope for every
# claim, so shedding claims still reach evidence; they simply do not route to a
# topic of their own. Evidence topics are independent of the collection queries
# in collect/config.py: those are search strings for finding videos, and nothing
# in rag/ reads a video's `query` field.
#
# A document mapped to GENERAL is not a topic of its own: it is searched for every
# claim, whatever that claim routes to. Reserved for authority sources that answer
# many narratives by construction — the CDC myth sheet covers magnetism, shedding,
# fertility, DNA alteration, test positivity, mandates and catching COVID in three
# pages. Declared rather than inferred: its breadth is a property of what the
# document is for, not something to rediscover from passage statistics whenever the
# corpus changes.
GENERAL = "__general__"

DOCUMENTS: dict[str, str] = {
    "cdc_108736_DS1.pdf": GENERAL,

    # Broad mechanism/immunology study: covers fetal development, syncytin-1
    # cross-reactivity, transplacental antibody transfer and the spike-protein
    # immune response. Serves claims across several narratives, so general.
    "pbio.3001506.pdf": GENERAL,

    "fendo-15-1356938.pdf": "infertility",
    "dead036.pdf": "infertility",          # miscarriage risk, Human Reproduction

    "clc-45-691.pdf": "myocarditis",
    "clinpract-16-00077.pdf": "myocarditis",
    "vaccines-10-01316.pdf": "myocarditis",
    "cureus-0017-00000098208.pdf": "myocarditis",

    # Regulatory guidance on how viral-vector products behave in the body —
    # what shedding is, how it is measured, replication competence. Mechanism
    # rather than a single narrative, so general.
    "ich-viral-vector-shedding.pdf": GENERAL,

    # Broad safety/efficacy syntheses. Each pools many trials or cohorts across
    # vaccine types and adverse-event categories, so they bear on myocarditis,
    # clotting, side effects and efficacy claims alike — general, not one topic.
    "vaccines-09-00467-v2.pdf": GENERAL,   # 235 cites, RCT meta-analysis
    "vaccines-09-00989-v2.pdf": GENERAL,   # 110 cites, phase-3 comparison
    "vaccines-11-01118.pdf": GENERAL,      # real-world safety, ~5.1M people

    # Platform-level reviews covering mechanism of action, efficacy and the
    # adverse-event profile across every major vaccine type. Added to the general
    # tier when jcm-11-06272 was withdrawn (see below): like the syntheses above
    # they bear on clotting, myocarditis and efficacy claims alike rather than on
    # one narrative, so general rather than one topic.
    "s41392-022-00950-y.pdf": GENERAL,     # mRNA platform review, Nature STTT
    "IDR-S315727.pdf": GENERAL,            # all-platform efficacy/safety review

    # Thrombosis after vaccination. Each is a systematic review of a distinct
    # clotting syndrome — TTS, immune thrombocytopenia, and the coagulopathy
    # mechanism — so together they cover the narrative rather than one case series.
    "cath-27-10760296211048815.pdf": "blood_clots",   # 38 cites, TTS post hoc
    "vaccines-10-01444.pdf": "blood_clots",           # 24 cites, ITP cases
    "ijms-22-10791-v2.pdf": "blood_clots",            # 23 cites, coagulopathy

    "vaccines-11-01693.pdf": "natural_immunity",
    "s41598-023-47043.pdf": "natural_immunity",
    "s41541-022-00594-7.pdf": "natural_immunity",
    "kwac011.pdf": "infertility",

    # WITHDRAWN 2026-11-01: jcm-11-06272.pdf, "SARS-CoV-2 — The Role of Natural
    # Immunity: A Narrative Review" (Diani et al., 2022, doi:10.3390/jcm11216272).
    # Removed on provenance grounds: authors subsequently associated with vaccine
    # hesitancy advocacy. A reference corpus used to adjudicate misinformation has
    # to be defensible on who wrote it, not only on licence and citation count —
    # a debunking that cites a hesitancy-aligned source undermines the verdict it
    # supports, whatever the cited sentence happens to say.
    # It was the corpus's largest document at 367 passages (16% of 2301) and sat
    # in the general tier, so it was in scope for every claim. Replaced by the two
    # platform-level reviews added above rather than by another natural-immunity
    # paper, because its role here had become general-tier breadth.
    # Do not re-add without revisiting this decision.
}

# Documents whose PDF carries no DOI — a government page, a standards body PDF.
# Nothing to look up, so their citation details are stated.
NO_DOI: dict[str, dict] = {
    "cdc_108736_DS1.pdf": {
        "title": "Myths and Facts about COVID-19 Vaccines",
        "publisher": "Centers for Disease Control and Prevention (CDC)",
        "source_url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/facts.html",
        "updated": "2021-07-07",
    },
    "ich-viral-vector-shedding.pdf": {
        "title": "ICH Considerations: General Principles to Address Virus and "
                 "Vector Shedding",
        "publisher": "International Council for Harmonisation (ICH)",
        "source_url": "https://database.ich.org/sites/default/files/"
                      "ICH_Considerations_Viral-Vector_Shedding_2009.pdf",
        "updated": "2009-06-02",
    },
}

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]>\"]+)")


def doi_from_pdf(pdf: Path, scan_chars: int = 8000) -> str | None:
    """Read the DOI out of a PDF's own text, trimming trailing punctuation."""
    try:
        import pymupdf
        doc = pymupdf.open(pdf)
        text = "".join(page.get_text() for page in doc)[:scan_chars]
        doc.close()
    except Exception:
        return None
    m = _DOI_RE.search(text)
    return m.group(1).rstrip(".,;)") if m else None


def fetch_metadata(doi: str) -> dict:
    """Title, publisher and date for one DOI, from Crossref."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "TCC-research/1.0 (academic use)"}
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            message = json.loads(response.read().decode())["message"]
    except Exception:
        return {"source_url": f"https://doi.org/{doi}"}

    parts = (message.get("issued", {}).get("date-parts") or [[""]])[0]
    return {
        "title": (message.get("title") or [""])[0],
        "publisher": (message.get("container-title")
                      or [message.get("publisher", "")])[0],
        "source_url": f"https://doi.org/{doi}",
        "updated": "-".join(str(x) for x in parts if x != ""),
    }


def build(evidence_dir: Path = EVIDENCE_DIR, pdf_dir: Path = PDF_DIR,
          verbose: bool = True) -> dict:
    """Resolve every declared document and write sources.json."""
    manifest = {
        "description": (
            "Reference corpus manifest: which document covers which claim topic, "
            "and where it is published. Generated by "
            "`python -m rag --build-manifest` from the DOCUMENTS declaration in "
            "rag/manifest.py — edit that, not this file."
        ),
        "generated": time.strftime("%Y-%m-%d"),
        "topics": {
            name: {"addresses": text, "files": []}
            for name, text in TOPICS.items()
        },
    }

    manifest["general_files"] = []

    for filename, topic in DOCUMENTS.items():
        if topic != GENERAL and topic not in manifest["topics"]:
            raise ValueError(f"{filename}: unknown topic {topic!r}")

        if filename in NO_DOI:
            entry = {"filename": filename, **NO_DOI[filename]}
            note = "declared"
        else:
            pdf = pdf_dir / filename
            doi = doi_from_pdf(pdf) if pdf.exists() else None
            if doi:
                entry = {"filename": filename, "doi": doi, **fetch_metadata(doi)}
                note = doi
                time.sleep(0.3)        # courtesy rate limit
            else:
                entry = {"filename": filename}
                note = "no PDF present" if not pdf.exists() else "no DOI in PDF"

        if topic == GENERAL:
            entry["scope"] = "general"
            manifest["general_files"].append(entry)
        else:
            manifest["topics"][topic]["files"].append(entry)
        if verbose:
            print(f"  {topic:17} {filename[:34]:34} {note}")

    for spec in manifest["topics"].values():
        spec["files"].sort(key=lambda f: f["filename"])
    manifest["general_files"].sort(key=lambda f: f["filename"])

    out = evidence_dir / "sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        total = (sum(len(s["files"]) for s in manifest["topics"].values())
                 + len(manifest["general_files"]))
        print(f"\nWrote {out}: {total} document(s), "
              f"{len(manifest['general_files'])} general-purpose")
    return manifest


def document_sources(evidence_dir: Path = EVIDENCE_DIR) -> dict[str, dict]:
    """source_file -> its manifest entry, for resolving citations at report time.

    Per-document facts live here once, not repeated on every passage.
    """
    path = evidence_dir / "sources.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    out = {
        f["filename"]: {**f, "topic": topic}
        for topic, spec in manifest["topics"].items()
        for f in spec["files"]
    }
    out.update({
        f["filename"]: {**f, "topic": "general"}
        for f in manifest.get("general_files", [])
    })
    return out
