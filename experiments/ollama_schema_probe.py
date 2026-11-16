"""Does a server (Ollama or OpenRouter) enforce the verdict JSON schema, or merely get
lucky?"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from rag.config import IS_REMOTE, LLAMA_MODEL, LLAMA_URL, USE_JSON_SCHEMA  # noqa: E402
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import SYSTEM_PROMPT, VERDICT_SCHEMA, _api_key_from_env, _endpoint, build_prompt  # noqa: E402
from rag.pipeline import segment_claims  # noqa: E402
from experiments.run_experiment import load_test_set  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ORDER = list(VERDICT_SCHEMA["schema"]["required"])
ENUMS = {k: v["enum"] for k, v in VERDICT_SCHEMA["schema"]["properties"].items()
         if "enum" in v}


def ask(user: str, schema: dict) -> str:
    payload = {
        "model": LLAMA_MODEL,
        "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    headers = {}
    if IS_REMOTE:
        payload["provider"] = {"require_parameters": True}
        headers["Authorization"] = f"Bearer {_api_key_from_env()}"
    r = requests.post(_endpoint(LLAMA_URL), json=payload, headers=headers, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def check(text: str, order: list[str]) -> str:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return f"not JSON: {text[:120]!r}"
    if not isinstance(obj, dict):
        return "not an object"
    keys = list(obj)
    if keys != order:
        return f"keys {keys}"
    for k, allowed in ENUMS.items():
        if obj[k] not in allowed:
            return f"{k}={obj[k]!r} not in {allowed}"
    return ""


print(f"server {LLAMA_URL}  model {LLAMA_MODEL}  json_schema={USE_JSON_SCHEMA}")
if not USE_JSON_SCHEMA:
    print("WARNING: json_schema is off for this backend (set LLM_CONSTRAINT=json_schema)")

index = EvidenceIndex.load()
candidates = []
for video in load_test_set()[:12]:
    for w in segment_claims(video["transcript"]):
        passages = index.search(w)
        if passages:
            candidates.append((len(build_prompt(w, passages)), w, passages))
candidates.sort(key=lambda c: c[0], reverse=True)
chosen = candidates[:N]
print(f"{len(candidates)} windows with evidence; testing the {len(chosen)} largest "
      f"({chosen[-1][0]}-{chosen[0][0]} chars of prompt)\n")

canary = copy.deepcopy(VERDICT_SCHEMA)
canary["schema"]["properties"]["canary"] = {"type": "string",
                                            "enum": ["schema-enforced"]}
canary["schema"]["required"].append("canary")
_, w0, p0 = chosen[0]
out = ask(build_prompt(w0, p0), canary)
problem = check(out, ORDER + ["canary"])
canary_ok = not problem
print(f"1. canary   : {'PASS' if canary_ok else 'FAIL'}  "
      f"{'schema is enforced' if canary_ok else problem}")

fails = 0
print("2. contract :")
for i, (size, w, passages) in enumerate(chosen, 1):
    problem = check(ask(build_prompt(w, passages), VERDICT_SCHEMA), ORDER)
    fails += bool(problem)
    print(f"   [{i}/{len(chosen)}] {size:5} chars  {'ok  ' if not problem else 'FAIL'} {problem}")

ok = canary_ok and not fails
print(f"\nRESULT: {'USABLE' if ok else 'NOT USABLE'} — "
      f"canary {'pass' if canary_ok else 'fail'}, contract {len(chosen) - fails}/{len(chosen)}")
sys.exit(0 if ok else 1)
