"""Whole-transcript adjudication: one judgement per video instead of per window.

The windowed pipeline (rag/pipeline.py) asks about 14 questions per video and
joins them with "any window flagged => video flagged". Measured on dev and test,
that join turns one nitpick in a long explainer into a false positive, and a
myth quoted in one window and rebutted in the next into a flag. Here the model
reads the whole transcript, finds the claims itself and judges each with the
whole context in view.

    parametric   1 call:  claims + stance + verdict + debunking evidence, from the
                 model's own knowledge.
    rag          call 1:  the claims and their stance, no judgement.
                 retrieval: corpus passages for each claim.
                 call 2:  every claim judged against its own passages.
                 web:     claims the corpus cannot address are searched (Tavily)
                          and judged again on that evidence, as in the windowed
                          pipeline (verdicts relabelled *_WEB).

The flagging instructions are the v1 ones (tests/prompt_variants.py V0_*),
verbatim wherever they do not refer to a single window, so this condition
differs from v1 in the unit of judgement, not in what counts as fake news.

With TIME_CHECK on (--time-check, rag/temporal.py) the transcript carries the
video's publish date and every corpus passage its document date (web passages
are marked undated), and a transcript with time-bound wording ("so far", "not
yet") gets a TIME CHECK note. Unlike the windowed pipeline there is NO
deterministic time guard here: the model sees the dates and decides. The
windowed guard overrode a time-bound flag whenever its evidence was dated after
the video; that is a rule about the evidence, not a judgement about the claim,
and with the whole transcript in view the model can weigh it itself.

The same guards as rag/llm.py apply per claim: FAKE_NEWS needs stance
"asserting"; in the rag arm a claim the evidence does not address is
NO_EVIDENCE, and a flag needs evidence above FLAG_MIN_SIMILARITY.

The video is predicted fake when at least one claim survives as a flag. The
model's own `fake_news` answer is recorded beside it (`model_fake_news`) so a
disagreement is visible rather than silently resolved.

Rows are written in the same shape as windowed results ("window" is the claim's
index), so tests/check_run.py and the scoring in tests/run_experiment.py read
them unchanged. Window-level tools do not apply: claims are not windows, and the
two arms extract different claims, so only video-level agreement (video-AND in
tests/score_verifier.py) is meaningful for the verifier.
"""
from __future__ import annotations

from .config import FLAG_MIN_SIMILARITY, USE_GRAMMAR, USE_JSON_MODE, USE_JSON_SCHEMA
from .llm import _chat, build_query
from .websearch import rank_by_authority, search as web_search

# Evidence per claim. Kept small (no neighbour expansion) because every claim's
# passages share one prompt with the whole transcript.
PASSAGES_PER_CLAIM = 3
MAX_CLAIMS = 25            # a transcript yielding more is truncated, and the row says so
# Web searches per video. The windowed v1 runs searched on ~40% of ~14 windows,
# about 6 per video; capped at the same so Tavily cost and the share of
# web-grounded verdicts stay comparable. Claims are taken in the model's order
# (most important first).
MAX_WEB_CLAIMS = 6

_FLAGS = {"FAKE_NEWS", "FAKE_NEWS_WEB"}

# --- v1 rules, shared ----------------------------------------------------------
# Verbatim from the v1 prompts, with "the CLAIM" made "a claim".
_STANCE_RULES = (
    "Key distinction: a speaker who DENIES a myth (\"vaccines do not cause "
    'infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n\n'
    "Before deciding, determine the speaker's STANCE toward the false idea. Look for "
    "negations and denials: \"isn't\", \"aren't\", \"not possible\", \"no reason to "
    'suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker '
    "is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though "
    "the topic matches. Topic overlap alone is never enough to flag.\n\n"
)
_STANCE_FIELD = ('"asserting" if the speaker presents the false idea as true, '
                 '"denying" if they say it is false, "neutral" if merely '
                 'describing or unclear')
_FIND_CLAIMS = (
    "First find the factual claims in the TRANSCRIPT: statements about the world "
    "that could be checked, including ones the speaker denies, questions or only "
    "reports. Copy each as a short phrase of at most 30 words, verbatim from the "
    f"TRANSCRIPT. List at most {MAX_CLAIMS}, most important first; skip greetings, "
    "advertising and small talk.\n\n"
)

WHOLE_PARAMETRIC_PROMPT = (
    "You are a fact-checking classifier. You are given the full TRANSCRIPT of a "
    "video. No evidence is provided: judge it from your own knowledge of the "
    "scientific and public-health consensus.\n\n"
    + _FIND_CLAIMS +
    "Then decide, for each claim, whether the speaker is asserting something that "
    "consensus contradicts, and finally whether the video contains fake news.\n\n"
    "Verdicts for each claim:\n"
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the '
    "scientific and public-health consensus contradicts. This includes stating that "
    "a vaccine contains an ingredient it does not contain, or that it has an effect "
    "it does not have. It counts even if the speaker is vague, hedging, or "
    "embedding the claim in a longer story.\n"
    '- "NOT_FAKE_NEWS": the speaker agrees with the consensus, or is quoting, '
    "reporting, questioning or debunking the false claim rather than asserting it.\n"
    '- "UNVERIFIABLE": you cannot determine from your own knowledge whether what the '
    "speaker says is true or false, or it makes no checkable factual claim. "
    "Use this rather than guessing.\n\n"
    + _STANCE_RULES +
    "Reply with a JSON object using EXACTLY these two keys, in this order:\n"
    '  "claims"    - a list with one object per claim, each with EXACTLY these keys '
    "in this order:\n"
    '      "phrase"   - the claim, copied verbatim from the TRANSCRIPT\n'
    f'      "stance"   - {_STANCE_FIELD}\n'
    '      "verdict"  - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE". Use '
    'FAKE_NEWS only when stance is "asserting".\n'
    '      "evidence" - for FAKE_NEWS, one sentence on what the consensus says that '
    "contradicts the claim; otherwise an empty string\n"
    '  "fake_news" - true if at least one claim is FAKE_NEWS, otherwise false\n\n'
    "Example:\n"
    '{"claims": [{"phrase": "the vaccine makes you magnetic", "stance": "asserting", '
    '"verdict": "FAKE_NEWS", "evidence": "COVID-19 vaccines contain no metals and '
    'cannot make a person magnetic."}, {"phrase": "vaccines do not cause '
    'infertility", "stance": "denying", "verdict": "NOT_FAKE_NEWS", "evidence": ""}], '
    '"fake_news": true}\n\n'
    "JSON only."
)

WHOLE_EXTRACT_PROMPT = (
    "You are given the full TRANSCRIPT of a video. Do not judge whether anything "
    "in it is true.\n\n"
    + _FIND_CLAIMS +
    "For each claim, determine the speaker's STANCE toward it. Look for negations "
    "and denials: \"isn't\", \"aren't\", \"not possible\", \"no reason to suspect\", "
    "\"there's no evidence\", \"that's a myth\", \"doesn't\".\n\n"
    "Reply with a JSON object with EXACTLY one key:\n"
    '  "claims" - a list with one object per claim, each with EXACTLY these keys in '
    "this order:\n"
    '      "phrase" - the claim, copied verbatim from the TRANSCRIPT\n'
    '      "stance" - "asserting" if the speaker presents it as true, "denying" if '
    'they say it is false, "neutral" if merely describing, asking or unclear\n\n'
    'Example:\n{"claims": [{"phrase": "the vaccine makes you magnetic", "stance": '
    '"denying"}]}\n\nJSON only.'
)

WHOLE_JUDGE_PROMPT = (
    "You are a fact-checking classifier. You are given the full TRANSCRIPT of a "
    "video and numbered CLAIMS taken from it, each followed by EVIDENCE from public "
    "health authorities retrieved for that claim.\n\n"
    "For each claim, decide whether the speaker is asserting something its evidence "
    "contradicts, and finally whether the video contains fake news. Use the "
    "TRANSCRIPT for context: who is speaking, and whether a claim is asserted, "
    "quoted or rebutted elsewhere.\n\n"
    "Verdicts for each claim:\n"
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the '
    "evidence contradicts. This includes stating that a vaccine contains an "
    "ingredient the evidence says it does not contain, or that it has an effect the "
    "evidence says it does not have. It counts even if the speaker is vague, "
    "hedging, or embedding the claim in a longer story.\n"
    '- "NOT_FAKE_NEWS": the speaker agrees with the evidence, or is quoting, '
    "reporting, questioning or debunking the false claim rather than asserting it.\n"
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says. '
    "Use this whenever the evidence is silent on the claim's specific subject, EVEN "
    "IF you personally believe the claim is false. Saying something is false without "
    "evidence that addresses it is not fact-checking, and an unsupported true answer "
    "is still an error here.\n\n"
    + _STANCE_RULES +
    "Reply with a JSON object using EXACTLY these two keys, in this order:\n"
    '  "judgements" - a list with one object per claim, in the order given, each '
    "with EXACTLY these keys in this order:\n"
    '      "claim"       - the claim\'s number\n'
    '      "addresses"   - "yes" or "no". Does the claim\'s EVIDENCE discuss its '
    "specific subject? Share of a general topic is not enough: evidence about "
    "vaccine ingredients does not address microchips. If the specific subject is "
    'absent, answer "no".\n'
    f'      "stance"      - {_STANCE_FIELD}\n'
    '      "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". If '
    '"addresses" is "no" you MUST answer "NO_EVIDENCE". Use FAKE_NEWS only when '
    'stance is "asserting".\n'
    '      "evidence_id" - the id of the evidence passage you relied on, or empty\n'
    '      "evidence"    - for FAKE_NEWS, one sentence on what that passage says '
    "that contradicts the claim; otherwise an empty string\n"
    '  "fake_news"  - true if at least one claim is FAKE_NEWS, otherwise false\n\n'
    "Example:\n"
    # The three v1 examples (fake, debunk, no evidence), as one reply.
    '{"judgements": [{"claim": 1, "addresses": "yes", "stance": "asserting", '
    '"verdict": "FAKE_NEWS", "evidence_id": "CDC-MAGNETIC", "evidence": "The CDC '
    'states vaccines are free from metals."}, {"claim": 2, "addresses": "yes", '
    '"stance": "denying", "verdict": "NOT_FAKE_NEWS", "evidence_id": "CDC-MAGNETIC", '
    '"evidence": ""}, {"claim": 3, "addresses": "no", "stance": "asserting", '
    '"verdict": "NO_EVIDENCE", "evidence_id": "", "evidence": ""}], '
    '"fake_news": true}\n\n'
    "JSON only."
)

# --- output contracts ------------------------------------------------------------
_STR = {"type": "string"}
_STANCE = {"type": "string", "enum": ["asserting", "denying", "neutral"]}


def _schema(name: str, item: dict, order: list[str], top: str, with_flag: bool) -> dict:
    props = {top: {"type": "array", "items": {
        "type": "object", "properties": item, "required": order,
        "additionalProperties": False}}}
    required = [top]
    if with_flag:
        props["fake_news"] = {"type": "boolean"}
        required.append("fake_news")
    return {"name": name, "strict": True, "schema": {
        "type": "object", "properties": props, "required": required,
        "additionalProperties": False}}


PARAMETRIC_SCHEMA = _schema(
    "whole_parametric",
    {"phrase": _STR, "stance": _STANCE,
     "verdict": {"type": "string", "enum": ["FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE"]},
     "evidence": _STR},
    ["phrase", "stance", "verdict", "evidence"], "claims", True)
EXTRACT_SCHEMA = _schema(
    "whole_extract", {"phrase": _STR, "stance": _STANCE},
    ["phrase", "stance"], "claims", False)
JUDGE_SCHEMA = _schema(
    "whole_judge",
    {"claim": {"type": "integer"}, "addresses": {"type": "string", "enum": ["yes", "no"]},
     "stance": _STANCE,
     "verdict": {"type": "string", "enum": ["FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE"]},
     "evidence_id": _STR, "evidence": _STR},
    ["claim", "addresses", "stance", "verdict", "evidence_id", "evidence"],
    "judgements", True)

# GBNF equivalents for llama.cpp, in the same field order.
_GBNF_COMMON = r'''
# Bounded on purpose (2026-11-11): unbounded ws let gemma-4-12b pad a reply with
# newlines until the token cap, and an unbounded array let it list 86 claims
# against an instruction of at most MAX_CLAIMS. {0,24} = MAX_CLAIMS - 1 more items.
stance ::= "\"asserting\"" | "\"denying\"" | "\"neutral\""
bool   ::= "true" | "false"
int    ::= [0-9]+
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]{0,16}
'''
PARAMETRIC_GRAMMAR = r'''
root   ::= "{" ws "\"claims\":" ws "[" ws (item (ws "," ws item){0,24})? ws "]" "," ws "\"fake_news\":" ws bool ws "}"
item   ::= "{" ws "\"phrase\":" ws string "," ws "\"stance\":" ws stance "," ws "\"verdict\":" ws verdict "," ws "\"evidence\":" ws string ws "}"
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"UNVERIFIABLE\""
''' + _GBNF_COMMON
EXTRACT_GRAMMAR = r'''
root   ::= "{" ws "\"claims\":" ws "[" ws (item (ws "," ws item){0,24})? ws "]" ws "}"
item   ::= "{" ws "\"phrase\":" ws string "," ws "\"stance\":" ws stance ws "}"
''' + _GBNF_COMMON
JUDGE_GRAMMAR = r'''
root   ::= "{" ws "\"judgements\":" ws "[" ws (item (ws "," ws item){0,24})? ws "]" "," ws "\"fake_news\":" ws bool ws "}"
item   ::= "{" ws "\"claim\":" ws int "," ws "\"addresses\":" ws yesno "," ws "\"stance\":" ws stance "," ws "\"verdict\":" ws verdict "," ws "\"evidence_id\":" ws string "," ws "\"evidence\":" ws string ws "}"
yesno  ::= "\"yes\"" | "\"no\""
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"NO_EVIDENCE\""
''' + _GBNF_COMMON


def _time_on() -> bool:
    from . import config
    return config.TIME_CHECK


def _transcript_block(transcript: str, claim_date, parametric: bool) -> str:
    """The TRANSCRIPT section; dated, with a TIME CHECK note, when TIME_CHECK is on."""
    if not _time_on():
        return f'TRANSCRIPT:\n"{transcript}"'
    from .temporal import time_note
    when = claim_date.isoformat() if claim_date else "date unknown"
    block = f'TRANSCRIPT (published {when}):\n"{transcript}"'
    note = time_note(transcript, claim_date, parametric=parametric)
    if note:
        # time_note speaks of "the CLAIM"; here the wording is in the transcript.
        block += "\n\n" + note.replace("the CLAIM contains", "the TRANSCRIPT contains")
    return block


def _call(system: str, user: str, grammar: str, schema: dict) -> tuple[dict | None, str]:
    return _chat(system, user, _url(), _model(), USE_GRAMMAR,
                 grammar if USE_GRAMMAR else None, USE_JSON_MODE,
                 use_json_schema=USE_JSON_SCHEMA, schema=schema)


def _url():
    from .config import LLAMA_URL
    return LLAMA_URL


def _model():
    from .config import LLAMA_MODEL
    return LLAMA_MODEL


def _row(n: int, phrase: str, verdict: str, **kw) -> dict:
    """One claim, in the windowed result shape."""
    return {"window": n, "claim_text": phrase, "verdict": verdict,
            "addresses": kw.get("addresses", ""), "stance": kw.get("stance", ""),
            "flagged_phrase": phrase if verdict in _FLAGS else "",
            "reason": kw.get("reason", ""), "evidence_id": kw.get("evidence_id", ""),
            "debunk_text": kw.get("debunk_text", ""),
            "evidence_source": kw.get("evidence_source", ""),
            "evidence_url": kw.get("evidence_url", ""),
            "similarity": kw.get("similarity"),
            "retrieved_ids": kw.get("retrieved_ids", []),
            "evidence_origin": kw.get("origin", "none"),
            "retrieved_at": kw.get("retrieved_at", ""),
            "web_decision": kw.get("web_decision", "")}


def _error(reason: str) -> dict:
    return {"rows": [_row(0, "", "ERROR", reason=reason)], "model_fake_news": None,
            "claims": 0, "truncated": False}


def _claims(parsed: dict | None, key: str) -> list[dict] | None:
    items = (parsed or {}).get(key)
    if not isinstance(items, list):
        return None
    return [c for c in items if isinstance(c, dict)][:MAX_CLAIMS]


def check_video_parametric(transcript: str, published: str | None = None) -> dict:
    """Whole-transcript parametric judgement. Returns rows + model_fake_news."""
    from .temporal import parse_date
    claim_date = parse_date(published)
    parsed, err = _call(WHOLE_PARAMETRIC_PROMPT,
                        _transcript_block(transcript, claim_date, parametric=True),
                        PARAMETRIC_GRAMMAR, PARAMETRIC_SCHEMA)
    claims = _claims(parsed, "claims")
    if claims is None:
        return _error(err or f"no claims list in reply: {str(parsed)[:150]}")
    rows = []
    for n, c in enumerate(claims, 1):
        verdict = str(c.get("verdict", "")).upper()
        stance = str(c.get("stance", "")).lower()
        reason = str(c.get("evidence", ""))
        if verdict not in {"FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE"}:
            verdict, reason = "ERROR", f"unrecognised verdict {verdict!r}"
        elif verdict == "FAKE_NEWS" and stance != "asserting":
            # The v1 stance guard (rag/llm.py adjudicate_parametric).
            verdict, reason = "NOT_FAKE_NEWS", (f"Overridden: FAKE_NEWS with stance "
                                                f"{stance!r}. {reason}")
        rows.append(_row(n, str(c.get("phrase", "")), verdict, stance=stance,
                         reason=reason))
    return {"rows": rows, "model_fake_news": parsed.get("fake_news"),
            "claims": len(claims),
            "truncated": len(parsed.get("claims") or []) >= MAX_CLAIMS}


def _evidence_block(claims: list[dict], passages: dict[int, list[dict]]) -> str:
    lines = ["CLAIMS AND THEIR EVIDENCE:"]
    for n, c in enumerate(claims, 1):
        lines.append(f'\n[{n}] "{c["phrase"]}"')
        ps = passages.get(n) or []
        if not ps:
            lines.append("  (no evidence retrieved)")
        for p in ps:
            when = (p.get("updated") or "undated") if _time_on() else ""
            lines.append(f"  [{p['id']}] ({p.get('publisher', '')}"
                         f"{', ' + when if when else ''}) "
                         f"states: {p.get('text', '')}")
    return "\n".join(lines)


def _judge(transcript: str, claims: list[dict], passages: dict[int, list[dict]],
           claim_date=None) -> tuple[dict[int, dict] | None, object, str]:
    user = (f"{_transcript_block(transcript, claim_date, parametric=False)}\n\n"
            f"{_evidence_block(claims, passages)}")
    parsed, err = _call(WHOLE_JUDGE_PROMPT, user, JUDGE_GRAMMAR, JUDGE_SCHEMA)
    items = _claims(parsed, "judgements")
    if items is None:
        return None, None, err or f"no judgements list: {str(parsed)[:150]}"
    by_n = {}
    for j in items:
        try:
            by_n[int(j.get("claim"))] = j
        except (TypeError, ValueError):
            continue
    return by_n, parsed.get("fake_news"), ""


def _guarded(j: dict | None, ps: list[dict]) -> tuple[str, str, str, str]:
    """(verdict, addresses, stance, reason) after the rag/llm.py guards."""
    if j is None:
        return "ERROR", "", "", "claim missing from the model's judgements"
    verdict = str(j.get("verdict", "")).upper()
    addresses = str(j.get("addresses", "")).lower()
    stance = str(j.get("stance", "")).lower()
    reason = str(j.get("evidence", ""))
    if verdict not in {"FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE"}:
        return "ERROR", addresses, stance, f"unrecognised verdict {verdict!r}"
    if addresses == "no" and verdict != "NO_EVIDENCE":
        return "NO_EVIDENCE", addresses, stance, (f"Overridden: addresses='no' but "
                                                  f"{verdict}. {reason}")
    top = max((p.get("similarity") or 0.0 for p in ps), default=0.0)
    if verdict == "FAKE_NEWS" and top < FLAG_MIN_SIMILARITY:
        return "NO_EVIDENCE", addresses, stance, (f"Overridden: best evidence at "
                                                  f"{top:.3f} < {FLAG_MIN_SIMILARITY}. "
                                                  f"{reason}")
    if verdict == "FAKE_NEWS" and stance != "asserting":
        return "NOT_FAKE_NEWS", addresses, stance, (f"Overridden: FAKE_NEWS with "
                                                    f"stance {stance!r}. {reason}")
    return verdict, addresses, stance, reason


def check_video_rag(transcript: str, index, web_fetch_fn=None,
                    published: str | None = None) -> dict:
    """Whole-transcript judgement against retrieved evidence. Returns rows + model_fake_news."""
    from .manifest import document_sources
    from .temporal import parse_date
    claim_date = parse_date(published)
    sources = document_sources()
    # Extraction does not judge truth, so it gets the plain transcript.
    parsed, err = _call(WHOLE_EXTRACT_PROMPT, f'TRANSCRIPT:\n"{transcript}"',
                        EXTRACT_GRAMMAR, EXTRACT_SCHEMA)
    claims = _claims(parsed, "claims")
    if claims is None:
        return _error(err or f"no claims list in reply: {str(parsed)[:150]}")
    claims = [{"phrase": str(c.get("phrase", "")), "stance": str(c.get("stance", ""))}
              for c in claims if str(c.get("phrase", "")).strip()]
    if not claims:
        return {"rows": [], "model_fake_news": False, "claims": 0, "truncated": False}
    truncated = len(parsed.get("claims") or []) >= MAX_CLAIMS  # at the cap

    passages = {n: index.search(c["phrase"], top_k=PASSAGES_PER_CLAIM, expand=0)
                for n, c in enumerate(claims, 1)}
    if _time_on():
        # Passages carry no date; the document's is in the manifest. Copied so
        # the index's own dicts are not mutated.
        passages = {n: [{**p, "updated": (sources.get(p.get("source_file", "")) or {})
                         .get("updated", "")} for p in ps]
                    for n, ps in passages.items()}
    judged, model_flag, err = _judge(transcript, claims, passages, claim_date)
    if judged is None:
        return _error(err)

    rows, origin = [], {n: "corpus" for n in passages}
    final = {n: _guarded(judged.get(n), passages[n]) for n in passages}
    web_why = {n: "" for n in passages}

    # Web fallback for asserted claims the corpus could not address, as in the
    # windowed pipeline: the trigger is the adjudicated verdict, not retrieval.
    if web_fetch_fn is not None:
        retry = [n for n, (v, a, s, _) in sorted(final.items())
                 if v == "NO_EVIDENCE" and s == "asserting"][:MAX_WEB_CLAIMS]
        web = {}
        for n in retry:
            query = build_query(claims[n - 1]["phrase"])
            found = rank_by_authority(web_search(query, max_results=PASSAGES_PER_CLAIM,
                                                 fetch_fn=web_fetch_fn))
            web_why[n] = f'corpus did not address the claim; query="{query}"'
            if found:
                web[n] = found
        if web:
            sub = [claims[n - 1] for n in web]
            sub_passages = {i: web[n] for i, n in enumerate(web, 1)}
            rejudged, web_flag, _ = _judge(transcript, sub, sub_passages, claim_date)
            # The model's own answer covers both calls: a claim it flags on web
            # evidence is part of its verdict on the video.
            if web_flag is True:
                model_flag = True
            for i, n in enumerate(web, 1):
                if rejudged is not None:
                    v, a, s, r = _guarded(rejudged.get(i), web[n])
                    if v in ("FAKE_NEWS", "NOT_FAKE_NEWS"):
                        v += "_WEB"
                    final[n] = (v, a, s, r)
                    judged[n] = {**(rejudged.get(i) or {})}
                    passages[n] = web[n]
                    origin[n] = "web"

    for n, c in enumerate(claims, 1):
        v, a, s, r = final[n]
        ps = passages[n]
        cited = next((p for p in ps if p["id"] == (judged.get(n) or {}).get("evidence_id")),
                     ps[0] if ps else None)
        doc = (sources.get((cited or {}).get("source_file", ""))
               if cited and origin[n] == "corpus" else None)
        rows.append(_row(
            n, c["phrase"], v, addresses=a, stance=s, reason=r,
            evidence_id=cited["id"] if cited else "",
            debunk_text=cited.get("text", "") if cited else "",
            evidence_source=(f"{(doc or {}).get('publisher') or (cited or {}).get('publisher', '')}"
                             f" — {(doc or {}).get('title') or (cited or {}).get('source_file', '')}"
                             if cited else ""),
            evidence_url=((doc or {}).get("source_url", "") or (cited or {}).get("source_url", "")
                          if cited else ""),
            similarity=cited.get("similarity") if cited else None,
            retrieved_ids=[p["id"] for p in ps], origin=origin[n],
            retrieved_at=(cited or {}).get("retrieved_at", ""),
            web_decision=web_why[n]))
    return {"rows": rows, "model_fake_news": model_flag, "claims": len(claims),
            "truncated": truncated}
