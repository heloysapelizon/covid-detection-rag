"""Local model client: ask a locally served model to adjudicate one claim."""
from __future__ import annotations

import json
import os
import re

import requests

from .config import (
    LLM_REASONING,
    FLAG_MIN_SIMILARITY,
    IS_REMOTE,
    LLAMA_MODEL,
    LLAMA_URL,
    PROJECT_ROOT,
    RESULTS_DIR,
    REQUEST_TIMEOUT_S,
    TEMPERATURE,
    USE_GRAMMAR,
    USE_JSON_MODE,
    USE_JSON_SCHEMA,
)

VERDICT_GRAMMAR = r'''
root   ::= "{" ws "\"addresses\":" ws yesno "," ws "\"stance\":" ws stance "," ws "\"verdict\":" ws verdict "," ws "\"claim\":" ws string "," ws "\"evidence_id\":" ws string "," ws "\"reason\":" ws string ws "}"
yesno   ::= "\"yes\"" | "\"no\""
stance  ::= "\"asserting\"" | "\"denying\"" | "\"neutral\""
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"NO_EVIDENCE\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
'''

VERDICT_SCHEMA = {
    "name": "verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "addresses": {"type": "string", "enum": ["yes", "no"]},
            "stance": {"type": "string",
                       "enum": ["asserting", "denying", "neutral"]},
            "verdict": {"type": "string",
                        "enum": ["FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE"]},
            "claim": {"type": "string"},
            "evidence_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["addresses", "stance", "verdict",
                     "claim", "evidence_id", "reason"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    'You are a fact-checking classifier. You are given EVIDENCE from public health authorities and a CLAIM transcribed from a video.\n'
    '\n'
    'Decide whether the speaker is asserting something the evidence contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the evidence contradicts. This includes stating that a vaccine contains an ingredient the evidence says it does not contain, or that it has an effect the evidence says it does not have. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the evidence, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says. Use this whenever the evidence is silent on the claim\'s specific subject, EVEN IF you personally believe the claim is false. Saying something is false without evidence that addresses it is not fact-checking, and an unsupported true answer is still an error here.\n'
    '\n'
    'Key distinction: a speaker who DENIES a myth ("vaccines do not cause infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches the evidence. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these six keys, in this order:\n'
    '  "addresses"   - "yes" or "no". Does the EVIDENCE discuss the specific subject of the CLAIM? Answer this FIRST, before deciding anything else. Share of a general topic is not enough: evidence about vaccine ingredients does not address microchips, and evidence about spike-protein immunology does not address blood clots. If the specific subject is absent, answer "no".\n'
    '  "stance"      - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". If "addresses" is "no" you MUST answer "NO_EVIDENCE" — you may only answer "FAKE_NEWS" or "NOT_FAKE_NEWS" when "addresses" is "yes". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "evidence_id" - the id of the evidence passage you relied on, e.g. "CDC-MAGNETIC"\n'
    '  "reason"      - one sentence explaining the verdict\n'
    '\n'
    'Examples:\n'
    '{"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker presents magnetism as real; the CDC states vaccines are free from metals."}\n'
    '{"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is debunking the magnetism myth, which agrees with the evidence."}\n'
    '{"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", "claim": "", "evidence_id": "", "reason": "The speaker claims the vaccine contains graphene oxide; the evidence does not discuss graphene oxide, so it cannot support a verdict either way."}\n'
    '\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)


def health(url: str = LLAMA_URL, timeout: float = 5.0) -> tuple[bool, str]:
    """Check whether the configured model server is reachable."""
    base = url.rstrip("/")
    models_url = (f"{base}/models" if base.endswith("/v1")
                  else f"{base}/v1/models")
    headers = {}
    if IS_REMOTE:
        key = _api_key_from_env()
        if not key:
            return False, ("OPENROUTER_API_KEY not set (checked environment "
                           "and .env)")
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(models_url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} from {models_url}"
        data = r.json().get("data") or []
        names = [m.get("id", "?") for m in data] if data else []
        if IS_REMOTE:
            if LLAMA_MODEL not in names:
                return False, (f"{LLAMA_MODEL!r} is not offered by the "
                               f"provider ({len(names)} models listed)")
            try:
                a = requests.get(f"{base}/key", headers=headers, timeout=timeout)
                if a.status_code != 200:
                    return False, (f"API key rejected (HTTP {a.status_code} "
                                   f"from {base}/key)")
                info = (a.json() or {}).get("data") or {}
                limit, used = info.get("limit"), info.get("usage")
                budget = ("unlimited" if limit is None
                          else f"{limit - (used or 0):.2f} of {limit} left")
            except requests.exceptions.RequestException as e:
                return False, f"cannot verify API key: {type(e).__name__}"
            return True, (f"{url} — {LLAMA_MODEL} offered, key valid "
                          f"(credit: {budget})")
        return True, (f"{url} — models: {', '.join(names)}" if names
                      else f"{url} — reachable, no model list reported")
    except requests.exceptions.RequestException as e:
        return False, f"{type(e).__name__}: cannot reach {url}"


def build_prompt(claim: str, passages: list[dict], claim_date=None,
                 dated: bool = False) -> str:
    """Render retrieved evidence and the claim into one user message."""
    from .temporal import time_note
    lines = ["EVIDENCE:"]
    for p in passages:
        when = p.get('updated', '') or ("undated" if dated else "")
        lines.append(f"[{p['id']}] ({p.get('publisher', '')}, {when})")
        lines.append(f"  addresses: {p.get('claim_addressed', '')}")
        lines.append(f"  states: {p.get('text', '')}")
    lines.append("")
    if dated:
        when = claim_date.isoformat() if claim_date else "date unknown"
        lines.append(f"CLAIM (transcribed speech, published {when}):\n\"{claim}\"")
        note = time_note(claim, claim_date, parametric=False)
        if note:
            lines += ["", note]
    else:
        lines.append(f"CLAIM (transcribed speech):\n\"{claim}\"")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a response, tolerating stray prose."""
    def as_dict(value):
        if isinstance(value, str):
            try:
                value = json.loads(value, strict=False)
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    try:
        found = as_dict(json.loads(text, strict=False))
        if found is not None:
            return found
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        chunk = m.group(0)
        for candidate in (chunk, chunk.replace('\\"', '"')):
            try:
                found = as_dict(json.loads(candidate, strict=False))
            except json.JSONDecodeError:
                continue
            if found is not None:
                return found
    return None


def _api_key_from_env(name: str = "OPENROUTER_API_KEY") -> str | None:
    """Read an API key from the environment, falling back to the project .env."""
    key = os.environ.get(name)
    if key:
        return key.strip()
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def _endpoint(url: str) -> str:
    """Chat-completions URL for a base that may or may not already end in /v1."""
    base = url.rstrip("/")
    return (f"{base}/chat/completions" if base.endswith("/v1")
            else f"{base}/v1/chat/completions")


def _chat_once(
    system: str,
    user: str,
    url: str,
    model: str,
    use_grammar: bool,
    grammar: str | None,
    use_json_mode: bool,
    use_json_schema: bool = USE_JSON_SCHEMA,
    schema: dict | None = None,
) -> tuple[dict | None, str]:
    """POST one chat completion. Returns (parsed_json, error_detail)."""
    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if use_grammar and grammar:
        payload["grammar"] = grammar
    if use_json_schema and schema:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": schema}
    elif use_json_mode:
        payload["response_format"] = {"type": "json_object"}
    if IS_REMOTE:
        payload["provider"] = {"require_parameters": True}
    if LLM_REASONING:
        payload["reasoning"] = ({"enabled": False} if LLM_REASONING == "off"
                                else {"effort": LLM_REASONING})

    headers = {}
    if IS_REMOTE:
        key = _api_key_from_env()
        if not key:
            return None, ("OPENROUTER_API_KEY not set (checked environment and "
                          ".env) — a remote run cannot proceed without it")
        headers["Authorization"] = f"Bearer {key}"

    try:
        r = requests.post(_endpoint(url), json=payload, headers=headers,
                          timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
        if "error" in body and "choices" not in body:
            err = body["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return None, f"provider error: {msg[:200]}"
        content = body["choices"][0]["message"].get("content") or ""
    except requests.exceptions.RequestException as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"unexpected response shape: {type(e).__name__}"

    parsed = _extract_json(content)
    if parsed is None:
        _log_failure(payload, body)
        return None, f"model did not return JSON: {content[:200]}"
    return parsed, ""


QUERY_PROMPT = (
    "You turn a passage of transcribed speech into a web search query.\n\n"
    "Identify the factual claim the speaker is making and write the search query "
    "a fact-checker would use to verify it. Keep it under 12 words. Use plain "
    "search keywords, no quotation marks and no operators.\n\n"
    'Reply with JSON: {"query": "..."}\n\n'
    "Examples:\n"
    '{"query": "do 5G networks spread COVID-19"}\n'
    '{"query": "covid vaccine graphene oxide content"}\n'
    '{"query": "covid vaccine microchip tracking"}\n'
    "JSON only."
)


def build_query(
    claim: str,
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_json_mode: bool = USE_JSON_MODE,
    max_words: int = 12,
) -> str:
    """Reduce a claim window to a web search query."""
    parsed, _ = _chat(QUERY_PROMPT, f'PASSAGE:\n"{claim}"',
                      url, model, False, None, use_json_mode)
    query = str((parsed or {}).get("query", "")).strip()
    if not query:
        query = " ".join(claim.split()[:max_words])
    return " ".join(query.split()[:max_words])


def adjudicate(
    claim: str,
    passages: list[dict],
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_grammar: bool = USE_GRAMMAR,
    use_json_mode: bool = USE_JSON_MODE,
    system_prompt: str | None = None,
    claim_date=None,
) -> dict:
    """Ask the local model to classify one claim against retrieved evidence."""
    if not passages:
        return {"verdict": "NO_EVIDENCE", "addresses": "no", "stance": "",
                "claim": "", "evidence_id": "",
                "reason": "No evidence passage passed the similarity threshold."}

    parsed, err = _chat(
        system_prompt or SYSTEM_PROMPT,
        build_prompt(claim, passages, claim_date, dated=_time_check()),
        url, model,
        use_grammar, VERDICT_GRAMMAR if use_grammar else None,
        use_json_mode,
        use_json_schema=USE_JSON_SCHEMA,
        schema=VERDICT_SCHEMA,
    )
    if parsed is None:
        return {"verdict": "ERROR", "addresses": "", "stance": "",
                "claim": "", "evidence_id": "", "reason": err}

    verdict = ""
    for key in ("verdict", "decision", "label", "classification", "result"):
        if parsed.get(key):
            verdict = str(parsed[key]).upper().strip().replace(" ", "_")
            break
    if verdict not in {"FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE"}:
        return {"verdict": "ERROR", "addresses": "", "stance": "", "claim": "", "evidence_id": "",
                "reason": f"unrecognised verdict {verdict!r} in {list(parsed)}"}

    stance = str(parsed.get("stance", "")).lower().strip()
    addresses = str(parsed.get("addresses", "")).lower().strip()

    if addresses == "no" and verdict in {"FAKE_NEWS", "NOT_FAKE_NEWS"}:
        return {
            "verdict": "NO_EVIDENCE",
            "addresses": addresses,
            "stance": stance,
            "claim": "",
            "evidence_id": "",
            "reason": (f"Overridden: model answered addresses='no' but returned "
                       f"{verdict}. {parsed.get('reason', '')}"),
        }

    top_sim = max((p.get("similarity") or 0.0 for p in passages), default=0.0)
    if verdict == "FAKE_NEWS" and top_sim < FLAG_MIN_SIMILARITY:
        return {
            "verdict": "NO_EVIDENCE",
            "addresses": addresses,
            "stance": stance,
            "claim": "",
            "evidence_id": "",
            "reason": (f"Overridden: FAKE_NEWS on best evidence at similarity "
                       f"{top_sim:.3f}, below the {FLAG_MIN_SIMILARITY} floor for "
                       f"grounding a flag. {parsed.get('reason', '')}"),
        }

    if verdict == "FAKE_NEWS" and stance in {"denying", "neutral"}:
        return {
            "verdict": "NOT_FAKE_NEWS",
            "addresses": addresses,
            "stance": stance,
            "claim": "",
            "evidence_id": str(parsed.get("evidence_id", "")),
            "reason": (f"Overridden: model said FAKE_NEWS but reported stance "
                       f"{stance!r}. {parsed.get('reason', '')}"),
        }

    result = {
        "verdict": verdict,
        "addresses": addresses,
        "stance": stance,
        "claim": str(parsed.get("claim", "")),
        "evidence_id": str(parsed.get("evidence_id", "")),
        "reason": str(parsed.get("reason", "")),
    }
    if _time_check():
        from .temporal import guard
        result = guard(result, passages, claim_date)
    return result


def _time_check() -> bool:
    from . import config
    return config.TIME_CHECK


_RETRY_NO_JSON = 2
_RETRY_TRANSPORT = 6
_RETRY_WAIT_S = 20
_TRANSPORT_ERRORS = ("ConnectionError", "HTTPError", "ReadTimeout",
                     "ConnectTimeout", "Timeout", "ChunkedEncodingError",
                     "SSLError")


def _chat(*args, **kwargs) -> tuple[dict | None, str]:
    """_chat_once with bounded retries; same signature and return value."""
    import sys
    import time
    no_json = transport = 0
    while True:
        parsed, err = _chat_once(*args, **kwargs)
        if parsed is not None:
            return parsed, err
        if err.startswith("model did not return JSON") and no_json < _RETRY_NO_JSON:
            no_json += 1
        elif err.startswith(_TRANSPORT_ERRORS) and transport < _RETRY_TRANSPORT:
            transport += 1
            time.sleep(_RETRY_WAIT_S)
        else:
            return parsed, err
        print(f"    retry: {err[:120]}", file=sys.stderr, flush=True)


FAILURE_DIR = RESULTS_DIR


def _log_failure(payload: dict, body: dict) -> None:
    import time
    slug = str(payload.get("model", "unknown")).replace("/", "-").replace(":", "-")
    path = FAILURE_DIR / f"llm_failures_{slug}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "request": payload, "response": body},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass


PARAMETRIC_PROMPT = (
    'You are a fact-checking classifier. You are given a CLAIM transcribed from a video. No evidence is provided: judge it from your own knowledge of the scientific and public-health consensus.\n'
    '\n'
    'Decide whether the speaker is asserting something that consensus contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the scientific and public-health consensus contradicts. This includes stating that a vaccine contains an ingredient it does not contain, or that it has an effect it does not have. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the consensus, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "UNVERIFIABLE": you cannot determine from your own knowledge whether what the speaker says is true or false, or the passage makes no checkable factual claim. Use this rather than guessing.\n'
    '\n'
    'Key distinction: a speaker who DENIES a myth ("vaccines do not cause infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these four keys, in this order:\n'
    '  "stance"   - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"  - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"    - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "reason"   - one sentence explaining the verdict\n'
    '\n'
    'Examples:\n'
    '{"stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "reason": "The speaker presents magnetism as real; COVID-19 vaccines contain no metals and cannot make a person magnetic."}\n'
    '{"stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "reason": "The speaker is debunking the magnetism myth, which agrees with the consensus."}\n'
    '{"stance": "neutral", "verdict": "UNVERIFIABLE", "claim": "", "reason": "The speaker describes a local event that cannot be checked against general knowledge."}\n'
    '\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)

PARAMETRIC_GRAMMAR = r"""
root   ::= "{" ws "\"stance\":" ws stance "," ws "\"verdict\":" ws verdict "," ws "\"claim\":" ws string "," ws "\"reason\":" ws string ws "}"
stance  ::= "\"asserting\"" | "\"denying\"" | "\"neutral\""
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"UNVERIFIABLE\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
"""

PARAMETRIC_SCHEMA = {
    "name": "verdict_parametric",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "stance": {"type": "string",
                       "enum": ["asserting", "denying", "neutral"]},
            "verdict": {"type": "string",
                        "enum": ["FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE"]},
            "claim": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["stance", "verdict", "claim", "reason"],
        "additionalProperties": False,
    },
}


def adjudicate_parametric(
    claim: str,
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_grammar: bool = USE_GRAMMAR,
    use_json_mode: bool = USE_JSON_MODE,
    claim_date=None,
) -> dict:
    """Classify one claim window from the model's own knowledge, no evidence."""
    user = f'CLAIM (transcribed speech):\n"{claim}"'
    if _time_check():
        from .temporal import time_note
        when = claim_date.isoformat() if claim_date else "date unknown"
        user = f'CLAIM (transcribed speech, published {when}):\n"{claim}"'
        note = time_note(claim, claim_date, parametric=True)
        if note:
            user += "\n\n" + note
    parsed, err = _chat(
        PARAMETRIC_PROMPT,
        user,
        url, model,
        use_grammar, PARAMETRIC_GRAMMAR if use_grammar else None,
        use_json_mode,
        use_json_schema=USE_JSON_SCHEMA,
        schema=PARAMETRIC_SCHEMA,
    )
    if parsed is None:
        return {"verdict": "ERROR", "addresses": "", "stance": "",
                "claim": "", "evidence_id": "", "reason": err}

    verdict = ""
    for key in ("verdict", "decision", "label", "classification", "result"):
        if parsed.get(key):
            verdict = str(parsed[key]).upper().strip().replace(" ", "_")
            break
    if verdict not in {"FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE"}:
        return {"verdict": "ERROR", "addresses": "", "stance": "", "claim": "",
                "evidence_id": "",
                "reason": f"unrecognised verdict {verdict!r} in {list(parsed)}"}

    stance = str(parsed.get("stance", "")).lower().strip()

    if verdict == "FAKE_NEWS" and stance in {"denying", "neutral"}:
        return {
            "verdict": "NOT_FAKE_NEWS", "addresses": "", "stance": stance,
            "claim": "", "evidence_id": "",
            "reason": (f"Overridden: model said FAKE_NEWS but reported stance "
                       f"{stance!r}. {parsed.get('reason', '')}"),
        }

    return {
        "verdict": verdict, "addresses": "", "stance": stance,
        "claim": str(parsed.get("claim", "")), "evidence_id": "",
        "reason": str(parsed.get("reason", "")),
    }
