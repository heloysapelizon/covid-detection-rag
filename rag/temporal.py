"""Time-bound statements: dates for the claim and its evidence, and the guard."""
from __future__ import annotations

import re
from datetime import date

TIME_MARKERS = re.compile(
    r"\b("
    r"so far|thus far|to date|as of (?:now|today|yet|this|right now)|"
    r"at (?:this|the) (?:time|moment|point)|at present|currently|"
    r"(?:not|n't) (?:yet|seen|heard)|yet to|no (?:\w+ )?data (?:yet )?|"
    r"too early|early days|we (?:still )?(?:don't|do not) (?:yet )?know|"
    r"(?:haven't|have not|hasn't|has not) (?:yet )?(?:been )?(?:seen|heard|found|shown|reported)"
    r")\b",
    re.IGNORECASE,
)


def markers(text: str) -> list[str]:
    """Distinct time-bound phrases in `text`, in order of appearance."""
    seen: list[str] = []
    for m in TIME_MARKERS.finditer(text or ""):
        w = m.group(0).lower()
        if w not in seen:
            seen.append(w)
    return seen


def parse_date(value) -> date | None:
    """Best-effort ISO-ish date: '2021-01-21T01:16:46Z', '2023-2-16', '2022'."""
    if not value:
        return None
    m = re.match(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(value).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def time_note(claim: str, claim_date: date | None, parametric: bool) -> str:
    """The TIME CHECK note for a window with time-bound wording, else ''."""
    found = markers(claim)
    if not found:
        return ""
    when = claim_date.isoformat() if claim_date else "an unknown date"
    quoted = ", ".join(f'"{w}"' for w in found[:6])
    against = ("your knowledge extends past that date, so judge them against "
               "what was established at that date, not what was learned later"
               if parametric else
               "compare it with the date of each evidence passage: evidence "
               "published after the claim cannot show that a statement about "
               "what was known at the time was false")
    return (f"TIME CHECK: the CLAIM contains time-bound wording ({quoted}) and "
            f"was published on {when}. A statement such as \"so far\" or \"not "
            f"yet\" is about what was known when it was said; {against}.")


def guard(verdict: dict, passages: list[dict], claim_date: date | None) -> dict:
    """Override a time-bound FAKE_NEWS that rests on later evidence."""
    if verdict.get("verdict") != "FAKE_NEWS" or claim_date is None:
        return verdict
    phrase = verdict.get("claim", "")
    if not markers(phrase):
        return verdict
    cited = next((p for p in passages if p.get("id") == verdict.get("evidence_id")),
                 None)
    ev_date = parse_date((cited or {}).get("updated"))
    if ev_date is None or ev_date <= claim_date:
        return verdict
    return {**verdict, "verdict": "NO_EVIDENCE", "claim": "",
            "reason": (f"Overridden (time check): the flagged words are "
                       f"time-bound ({', '.join(markers(phrase))}) and the cited "
                       f"evidence ({ev_date}) postdates the claim ({claim_date}). "
                       f"{verdict.get('reason', '')}")}
