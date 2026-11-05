# Model backends and the four-model comparison

The adjudicator is served over an OpenAI-compatible `/v1/chat/completions`
endpoint, so one client serves three backends. `LLM_SERVER` selects which.

## The comparison

| | model | served by | `LLAMA_MODEL` |
|---|---|---|---|
| local | Gemma 4 12B | llama.cpp / Ollama | the GGUF or Ollama tag you pull |
| local | gpt-oss-20b | llama.cpp / Ollama | the GGUF or Ollama tag you pull |
| remote | Qwen3.6-27B | OpenRouter | `qwen/qwen3.6-27b` |
| remote | gpt-oss-120b | OpenRouter | `openai/gpt-oss-120b` |

Both OpenRouter ids were verified against the live model list on 2026-11-05.
Gemma 4 is published at 26B and 31B there, not 12B, so the 12B is a local-only
model — which is the point of including it: it is the smallest of the four and
tests whether the task survives at a size that runs on a laptop.

## Why the backend is a reported dimension, not a detail

**Transcript text leaves the machine on the remote path.** The local path does
not, and that was an original design property of this project. It is preserved
as a property of the local configuration rather than of the system, and
preflight prints a warning on every remote run.

**The output constraint differs, and the guarantees are not equal:**

```
  GBNF grammar (llama.cpp)   enforced during sampling — a response missing a
                             required field is unrepresentable
  json_schema (OpenRouter)   provider validates against the schema, so the six
                             fields and their order are guaranteed
  json_object (Ollama)       valid JSON of any shape only; a large evidence
                             block was observed dropping required keys
                             (docs/adjudication-experiments.md)
```

`VERDICT_SCHEMA` in `rag/llm.py` encodes **the same contract** as
`VERDICT_GRAMMAR` — the same six fields in the same order, with
`additionalProperties: false`. Field order is part of the contract, not
cosmetic: `addresses` is answered before `verdict` so relevance is decided
before the verdict for a left-to-right decoder, and that is the mechanism of the
relevance gate. The two must be kept in step; a backend-dependent contract would
make verdicts incomparable, which is the one thing a model comparison cannot
tolerate.

Ollama remains the weakest of the three and every figure in `docs/` was measured
there, which is why its own `--limit 3` smoke test is step 1 of any run.

## Cost and runtime

Measured per adjudication: ~2,400 tokens in, ~120 out. 1,588 windows
corpus-only; a Tavily run re-adjudicates ~40% of windows, so ~2,223 calls.

```
  qwen/qwen3.6-27b      corpus-only   $2.23      with Tavily   $3.12
  openai/gpt-oss-120b   corpus-only   $0.05      with Tavily   $0.07
```

Remote removes the runtime constraint entirely: the local CPU estimate is ~9.3 h
corpus-only and ~12.5 h with the fallback, against minutes of wall-clock for a
remote run limited only by rate limits. That is the practical reason the
comparison is affordable at all — four models × two fallback settings is eight
runs, which is ~75 CPU-hours locally.

Prices are per-million tokens from OpenRouter's model list on 2026-11-05 and
will drift; re-check before quoting them.

## Setup

```bash
# remote
export OPENROUTER_API_KEY=sk-or-v1-...      # or put it in .env
LLM_SERVER=openrouter LLAMA_MODEL=qwen/qwen3.6-27b \
    python tests/run_experiment.py --check

# local (llama.cpp is the default, no variable needed)
llama-server -m gemma-4-12b.gguf -c 8192 --port 8080
python tests/run_experiment.py --check

# local via Ollama, for development
LLM_SERVER=ollama LLAMA_MODEL=gpt-oss:20b python tests/run_experiment.py --check
```

Preflight fails before a run starts if the key is rejected or the model id is not
offered — both verified by deliberately breaking them. `/models` is public on
OpenRouter, so reaching it proves nothing about the key; the check calls the
authenticated `/key` endpoint and reports remaining credit.

A remote failure returns `ERROR` with the HTTP cause attached, never a
fabricated verdict.

## Keeping runs apart

`run.settings` in every results file records `LLAMA_MODEL`, `LLAMA_URL`,
`USE_GRAMMAR`, `USE_JSON_SCHEMA` and `IS_REMOTE`, so eight runs across four
models cannot be pooled or confused after the fact. Use a distinct `--out` per
model; the default path keys only on the web backend.
