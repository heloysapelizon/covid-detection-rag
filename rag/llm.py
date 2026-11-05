"""Local model client: ask a locally served model to adjudicate one claim.

Two servers are supported, both exposing an OpenAI-compatible
/v1/chat/completions endpoint, so this stays a plain HTTP boundary — no SDK, no
cloud. `LLM_SERVER` selects which (see rag/config.py):

    llama-server -m model.gguf -c 8192 --port 8080    # default
    ollama serve                                      # LLM_SERVER=ollama

**The two differ in how the output is constrained, which matters when reading
any measurement.** On llama.cpp the GBNF grammar is enforced during sampling, so
a malformed verdict cannot be generated. On Ollama the grammar is ignored and
`response_format: json_object` is requested instead, which asks rather than
enforces — a field can come back missing or misnamed, and the response is parsed
defensively for that reason.

llama.cpp is the default because it is what the reported experiment runs on.
Ollama is for development only — and every figure recorded in docs/ so far was
measured there, which is why the grammar path is still unexercised. The run
metadata records `USE_GRAMMAR` for each run so the two are never conflated.
"""
from __future__ import annotations

import json
import os
import re

import requests

from .config import (
    FLAG_MIN_SIMILARITY,
    IS_REMOTE,
    LLAMA_MODEL,
    LLAMA_URL,
    PROJECT_ROOT,
    REQUEST_TIMEOUT_S,
    TEMPERATURE,
    USE_GRAMMAR,
    USE_JSON_MODE,
    USE_JSON_SCHEMA,
)

# Constrains output to exactly the fields the pipeline consumes. llama.cpp enforces
# this during sampling, so malformed JSON cannot be produced in the first place.
# `addresses` comes first so relevance is decided before the verdict: the field
# order is the reasoning order for a left-to-right decoder, which is the whole
# mechanism of the gate.
VERDICT_GRAMMAR = r'''
root   ::= "{" ws "\"addresses\":" ws yesno "," ws "\"stance\":" ws stance "," ws "\"verdict\":" ws verdict "," ws "\"claim\":" ws string "," ws "\"evidence_id\":" ws string "," ws "\"reason\":" ws string ws "}"
yesno   ::= "\"yes\"" | "\"no\""
stance  ::= "\"asserting\"" | "\"denying\"" | "\"neutral\""
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"NO_EVIDENCE\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
'''

# The same six-field contract as VERDICT_GRAMMAR, expressed as a JSON schema for
# providers that validate schemas (OpenRouter). `required` lists the fields in
# the same order and `additionalProperties: false` forbids extras, so a response
# that omits or renames a field is rejected rather than parsed into a default.
# Keep this in step with VERDICT_GRAMMAR: they encode one contract in two
# dialects, and a backend-dependent contract would make verdicts incomparable.
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
    "You are a fact-checking classifier. You are given EVIDENCE from public health "
    "authorities and a CLAIM transcribed from a video.\n\n"
    "Decide whether the speaker is asserting something the evidence contradicts.\n\n"
    "Verdicts:\n"
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
    "Key distinction: a speaker who DENIES a myth (\"vaccines do not cause "
    'infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n\n'
    "Before deciding, determine the speaker's STANCE toward the false idea. Look for "
    "negations and denials: \"isn't\", \"aren't\", \"not possible\", \"no reason to "
    'suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker '
    "is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though "
    "the topic matches the evidence. Topic overlap alone is never enough to flag.\n\n"
    "Reply with a JSON object using EXACTLY these six keys, in this order:\n"
    '  "addresses"   - "yes" or "no". Does the EVIDENCE discuss the specific '
    "subject of the CLAIM? Answer this FIRST, before deciding anything else. "
    "Share of a general topic is not enough: evidence about vaccine ingredients "
    "does not address microchips, and evidence about spike-protein immunology "
    'does not address blood clots. If the specific subject is absent, answer "no".\n'
    '  "stance"      - "asserting" if the speaker presents the false idea as true, '
    '"denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". '
    'If "addresses" is "no" you MUST answer "NO_EVIDENCE" — you may only answer '
    '"FAKE_NEWS" or "NOT_FAKE_NEWS" when "addresses" is "yes". '
    'Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are '
    "false, copied verbatim; otherwise an empty string\n"
    '  "evidence_id" - the id of the evidence passage you relied on, e.g. '
    '"CDC-MAGNETIC"\n'
    '  "reason"      - one sentence explaining the verdict\n\n'
    "Examples:\n"
    '{"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", '
    '"claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", '
    '"reason": "The speaker presents magnetism as real; the CDC states vaccines '
    'are free from metals."}\n'
    '{"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", '
    '"claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is '
    'debunking the magnetism myth, which agrees with the evidence."}\n'
    '{"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", '
    '"claim": "", "evidence_id": "", "reason": "The speaker claims the vaccine '
    'contains graphene oxide; the evidence does not discuss graphene oxide, so it '
    'cannot support a verdict either way."}\n\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)


def health(url: str = LLAMA_URL, timeout: float = 5.0) -> tuple[bool, str]:
    """Check whether the configured model server is reachable.

    For a remote provider this also verifies the API key, because a key missing
    at request time would otherwise surface as a per-window ERROR hours into a
    run rather than at preflight.
    """
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
            # /models is public, so reaching it proves nothing about the key.
            # Check credit instead: that endpoint is authenticated, so it fails
            # here rather than per-window hours into a run.
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


def build_prompt(claim: str, passages: list[dict]) -> str:
    """Render retrieved evidence and the claim into one user message."""
    lines = ["EVIDENCE:"]
    for p in passages:
        lines.append(f"[{p['id']}] ({p.get('publisher', '')}, {p.get('updated', '')})")
        lines.append(f"  addresses: {p.get('claim_addressed', '')}")
        lines.append(f"  states: {p.get('text', '')}")
    lines.append("")
    lines.append(f"CLAIM (transcribed speech):\n\"{claim}\"")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a response, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _api_key_from_env(name: str = "OPENROUTER_API_KEY") -> str | None:
    """Read an API key from the environment, falling back to the project .env.

    Only `collect/` calls load_dotenv, so a key placed in .env would otherwise be
    invisible here — and a remote backend that fails for a missing key looks
    identical to one that found no evidence.
    """
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
    """Chat-completions URL for a base that may or may not already end in /v1.

    llama.cpp and Ollama are given a bare host, OpenRouter a base path that
    already includes /v1; appending unconditionally would produce /v1/v1.
    """
    base = url.rstrip("/")
    return (f"{base}/chat/completions" if base.endswith("/v1")
            else f"{base}/v1/chat/completions")


def _chat(
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
        # A provider may return an error object with HTTP 200.
        if "error" in body and "choices" not in body:
            err = body["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return None, f"provider error: {msg[:200]}"
        content = body["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"unexpected response shape: {type(e).__name__}"

    parsed = _extract_json(content)
    if parsed is None:
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
    """Reduce a claim window to a web search query.

    A claim window is ~140 words of unpunctuated speech. Passing it to a search
    engine returns nothing — measured: the full window raises "No results found"
    while a 12-word version of the same claim returns three. Truncation is not a
    reliable substitute (12 words worked, 20 failed, 30 worked on the same
    window), because what matters is whether the text reads as a query, not its
    length.

    Falls back to the first `max_words` words if the model call fails, so a
    search is still attempted rather than silently skipped.
    """
    parsed, _ = _chat(QUERY_PROMPT, f'PASSAGE:\n"{claim}"',
                      url, model, False, None, use_json_mode)
    query = str((parsed or {}).get("query", "")).strip()
    if not query:
        query = " ".join(claim.split()[:max_words])
    # Guard against a model that ignores the limit and echoes the window.
    return " ".join(query.split()[:max_words])


def adjudicate(
    claim: str,
    passages: list[dict],
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_grammar: bool = USE_GRAMMAR,
    use_json_mode: bool = USE_JSON_MODE,
    system_prompt: str | None = None,
) -> dict:
    """Ask the local model to classify one claim against retrieved evidence.

    Returns {verdict, claim, evidence_id, reason}, with verdict one of
    FAKE_NEWS / NOT_FAKE_NEWS / NO_EVIDENCE / ERROR.

    `system_prompt` overrides SYSTEM_PROMPT. It exists so a prompt variant can be
    measured against the current one on identical retrieval without editing this
    module; the pipeline never passes it.
    """
    if not passages:
        return {"verdict": "NO_EVIDENCE", "addresses": "no", "stance": "",
                "claim": "", "evidence_id": "",
                "reason": "No evidence passage passed the similarity threshold."}

    # One HTTP path for every backend (_chat applies the grammar, the schema or
    # json_object, and the auth header for a remote provider), so a change to
    # the transport cannot apply to one caller and not the other.
    parsed, err = _chat(
        system_prompt or SYSTEM_PROMPT,
        build_prompt(claim, passages),
        url, model,
        use_grammar, VERDICT_GRAMMAR if use_grammar else None,
        use_json_mode,
        use_json_schema=USE_JSON_SCHEMA,
        schema=VERDICT_SCHEMA,
    )
    if parsed is None:
        return {"verdict": "ERROR", "addresses": "", "stance": "",
                "claim": "", "evidence_id": "", "reason": err}

    # Models sometimes rename the key despite instructions; accept the common
    # synonyms rather than discarding an otherwise valid judgment.
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

    # Enforce the relevance gate rather than trusting the model to honour it.
    # Measured failure: on 9 of 11 windows of an uncovered narrative the model
    # returned FAKE_NEWS while its own reason read "not mentioned or supported by
    # the evidence provided" — it treated absence of evidence as contradiction.
    # The verdicts happened to be true, but a system whose method claim is
    # evidence-grounded detection cannot assert what its evidence does not
    # license: an unsupported true answer is indistinguishable from an
    # unsupported false one.
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

    # A flag also needs evidence strong enough to ground it. Backstop only — the
    # gate above is the real defence, since similarity does not separate covered
    # from uncovered claims (see FLAG_MIN_SIMILARITY in config).
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

    # Guard the rule rather than trusting the model to apply it: a speaker who is
    # denying or merely describing the myth is not spreading it, whatever verdict
    # the model picked. Topic overlap is the dominant failure mode here.
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

    return {
        "verdict": verdict,
        "addresses": addresses,
        "stance": stance,
        "claim": str(parsed.get("claim", "")),
        "evidence_id": str(parsed.get("evidence_id", "")),
        "reason": str(parsed.get("reason", "")),
    }
