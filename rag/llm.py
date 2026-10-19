"""The llama.cpp client: ask a locally served model to adjudicate one claim.

llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint, so this is
a plain HTTP boundary — no SDK, no cloud. Start a server with:

    llama-server -m model.gguf -c 8192 --port 8080

The model is asked for strict JSON and given a GBNF grammar that makes any other
output unrepresentable, so the verdict can be parsed rather than scraped. Servers
that ignore the grammar still work: the response is parsed defensively.
"""
from __future__ import annotations

import json
import re

import requests

from .config import (
    LLAMA_MODEL,
    LLAMA_URL,
    REQUEST_TIMEOUT_S,
    TEMPERATURE,
    USE_GRAMMAR,
    USE_JSON_MODE,
)

# Constrains output to exactly the fields the pipeline consumes. llama.cpp enforces
# this during sampling, so malformed JSON cannot be produced in the first place.
VERDICT_GRAMMAR = r'''
root   ::= "{" ws "\"verdict\":" ws verdict "," ws "\"claim\":" ws string "," ws "\"evidence_id\":" ws string "," ws "\"reason\":" ws string ws "}"
verdict ::= "\"FAKE_NEWS\"" | "\"NOT_FAKE_NEWS\"" | "\"NO_EVIDENCE\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
'''

# Pass 1. Deliberately sees ONLY the speech, never the evidence: given both, small
# models conflate "this text is about magnetism" with "this text claims magnetism",
# which flags fact-checkers as misinformation. Asked in isolation, the same model
# reads negation far more reliably.
STANCE_PROMPT = (
    "You read a passage of transcribed speech and report the speaker's own position.\n\n"
    "Answer one question: is the speaker presenting a claim as TRUE, or saying it is "
    "FALSE?\n\n"
    'Labels:\n'
    '- "asserting" - the speaker presents the claim as true, or treats it as an '
    "established fact.\n"
    '- "denying"   - the speaker says the claim is false, untrue, impossible, '
    "unsupported, or a myth.\n"
    '- "neutral"   - the speaker is only describing, reporting, or asking about the '
    "claim without taking a side.\n\n"
    "Pay close attention to negation. Phrases like \"there's no\", \"isn't\", "
    '"aren\'t", "not possible", "cannot", "no evidence", "no reason to suspect", '
    '"that\'s a myth", "debunked", "false" mean the speaker is DENYING.\n\n'
    "A speaker who names a false idea in order to reject it is \"denying\", not "
    '"asserting". What the passage is ABOUT does not matter — only what the speaker '
    "says about it.\n\n"
    'Reply with JSON: {"stance": "...", "quote": "...", "why": "..."}\n'
    '  "stance" - asserting, denying or neutral\n'
    '  "quote"  - the words that reveal the stance, copied verbatim\n'
    '  "why"    - one short sentence\n\n'
    "Examples:\n"
    '{"stance": "denying", "quote": "There\'s no chip or anything like that", '
    '"why": "The speaker rejects the microchip claim."}\n'
    '{"stance": "asserting", "quote": "it amplifies electromagnetic fields", '
    '"why": "The speaker presents this as a real property."}\n'
    "JSON only."
)

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
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says.\n\n'
    "Key distinction: a speaker who DENIES a myth (\"vaccines do not cause "
    'infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n\n'
    "Before deciding, determine the speaker's STANCE toward the false idea. Look for "
    "negations and denials: \"isn't\", \"aren't\", \"not possible\", \"no reason to "
    'suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker '
    "is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though "
    "the topic matches the evidence. Topic overlap alone is never enough to flag.\n\n"
    "Reply with a JSON object using EXACTLY these five keys, in this order:\n"
    '  "stance"      - "asserting" if the speaker presents the false idea as true, '
    '"denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". '
    'Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are '
    "false, copied verbatim; otherwise an empty string\n"
    '  "evidence_id" - the id of the evidence passage you relied on, e.g. '
    '"CDC-MAGNETIC"\n'
    '  "reason"      - one sentence explaining the verdict\n\n'
    "Examples:\n"
    '{"stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you '
    'magnetic", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker presents '
    'magnetism as real; the CDC states vaccines are free from metals."}\n'
    '{"stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "evidence_id": '
    '"CDC-MAGNETIC", "reason": "The speaker is debunking the magnetism myth, which '
    'agrees with the evidence."}\n\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)


def health(url: str = LLAMA_URL, timeout: float = 5.0) -> tuple[bool, str]:
    """Check whether a model server is reachable. Returns (ok, detail)."""
    try:
        r = requests.get(f"{url}/v1/models", timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} from {url}/v1/models"
        data = r.json().get("data") or []
        names = [m.get("id", "?") for m in data] if data else []
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


def _chat(
    system: str,
    user: str,
    url: str,
    model: str,
    use_grammar: bool,
    grammar: str | None,
    use_json_mode: bool,
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
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(f"{url}/v1/chat/completions", json=payload,
                          timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"unexpected response shape: {type(e).__name__}"

    parsed = _extract_json(content)
    if parsed is None:
        return None, f"model did not return JSON: {content[:200]}"
    return parsed, ""


def read_stance(
    claim: str,
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_json_mode: bool = USE_JSON_MODE,
) -> dict:
    """Pass 1: report the speaker's stance from the speech alone.

    No evidence is shown, so there is no topic for the model to mistake for a
    claim. Returns {stance, quote, why}; stance is asserting/denying/neutral,
    or "" when the call failed.
    """
    parsed, err = _chat(STANCE_PROMPT, f'PASSAGE:\n"{claim}"',
                        url, model, False, None, use_json_mode)
    if parsed is None:
        return {"stance": "", "quote": "", "why": err}

    stance = str(parsed.get("stance", "")).lower().strip()
    if stance not in {"asserting", "denying", "neutral"}:
        return {"stance": "", "quote": "",
                "why": f"unrecognised stance {stance!r}"}
    return {
        "stance": stance,
        "quote": str(parsed.get("quote", "")),
        "why": str(parsed.get("why", "")),
    }


def adjudicate(
    claim: str,
    passages: list[dict],
    url: str = LLAMA_URL,
    model: str = LLAMA_MODEL,
    use_grammar: bool = USE_GRAMMAR,
    use_json_mode: bool = USE_JSON_MODE,
) -> dict:
    """Ask the local model to classify one claim against retrieved evidence.

    Returns {verdict, claim, evidence_id, reason}, with verdict one of
    FAKE_NEWS / NOT_FAKE_NEWS / NO_EVIDENCE / ERROR.
    """
    if not passages:
        return {"verdict": "NO_EVIDENCE", "stance": "", "claim": "", "evidence_id": "",
                "reason": "No evidence passage passed the similarity threshold."}

    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(claim, passages)},
        ],
    }
    if use_grammar:
        # llama.cpp-specific: enforced during sampling, so invalid output cannot
        # be generated. Ignored by servers that do not support it.
        payload["grammar"] = VERDICT_GRAMMAR
    if use_json_mode:
        # Ollama / OpenAI-style equivalent: guarantees valid JSON, though not the
        # specific shape a grammar pins down.
        payload["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(f"{url}/v1/chat/completions", json=payload,
                          timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return {"verdict": "ERROR", "stance": "", "claim": "", "evidence_id": "",
                "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    except (KeyError, IndexError, ValueError) as e:
        return {"verdict": "ERROR", "stance": "", "claim": "", "evidence_id": "",
                "reason": f"unexpected response shape: {type(e).__name__}"}

    parsed = _extract_json(content)
    if parsed is None:
        return {"verdict": "ERROR", "stance": "", "claim": "", "evidence_id": "",
                "reason": f"model did not return JSON: {content[:200]}"}

    # Models sometimes rename the key despite instructions; accept the common
    # synonyms rather than discarding an otherwise valid judgment.
    verdict = ""
    for key in ("verdict", "decision", "label", "classification", "result"):
        if parsed.get(key):
            verdict = str(parsed[key]).upper().strip().replace(" ", "_")
            break
    if verdict not in {"FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE"}:
        return {"verdict": "ERROR", "stance": "", "claim": "", "evidence_id": "",
                "reason": f"unrecognised verdict {verdict!r} in {list(parsed)}"}

    stance = str(parsed.get("stance", "")).lower().strip()

    # Guard the rule rather than trusting the model to apply it: a speaker who is
    # denying or merely describing the myth is not spreading it, whatever verdict
    # the model picked. Topic overlap is the dominant failure mode here.
    if verdict == "FAKE_NEWS" and stance in {"denying", "neutral"}:
        return {
            "verdict": "NOT_FAKE_NEWS",
            "stance": stance,
            "claim": "",
            "evidence_id": str(parsed.get("evidence_id", "")),
            "reason": (f"Overridden: model said FAKE_NEWS but reported stance "
                       f"{stance!r}. {parsed.get('reason', '')}"),
        }

    return {
        "verdict": verdict,
        "stance": stance,
        "claim": str(parsed.get("claim", "")),
        "evidence_id": str(parsed.get("evidence_id", "")),
        "reason": str(parsed.get("reason", "")),
    }
