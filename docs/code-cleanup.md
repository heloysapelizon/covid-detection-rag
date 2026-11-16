# Code cleanup, 2026-11-05

An audit for dead code, duplication and inconsistencies, run before the final
experiment. Every change was verified to leave behaviour identical: dataset
counts (119 test / 42 fake / 77 not, 44 dev, 19 fallback, no leakage) and the
prompt A/B figures (0.714 both prompts) reproduce exactly.

## The server default was backwards

`rag/config.py` defaulted `LLM_SERVER` to `"ollama"`, so the llama.cpp path —
the one the reported experiment runs on — required an environment variable, and
`rag/llm.py` was titled "The llama.cpp client" while actually defaulting to a
server that *ignores* its grammar.

**llama.cpp is now the default**; Ollama is opt-in via `LLM_SERVER=ollama` for
development on a machine without llama-server. This matters beyond tidiness:
`USE_GRAMMAR` follows the server, so the old default silently disabled grammar
enforcement, and a reader of `llm.py` would have believed malformed output was
unrepresentable when it was merely discouraged.

Consequences recorded in the affected files: every figure in `docs/` was
measured on Ollama, so the grammar path is unexercised.

The health-check hint in `rag/main.py` was unconditionally "start llama-server",
which was wrong whenever Ollama was configured. It now names the server the run
is actually pointed at.

## Removed

| what | why it was dead |
|---|---|
| `FALLBACK_ROUNDS` (`rag/config.py`) | Never wired into `pipeline.py` — its own comment said "read by nothing". It was held for the web-search tier, which was instead keyed on the adjudicated verdict. |
| `ROUTE_MIN_SIMILARITY` (`rag/config.py`) | A deprecated alias for `ROUTE_FLOOR` with no remaining callers. |
| `read_stance` + `STANCE_PROMPT` (`rag/llm.py`, 60 lines) | The two-pass stance approach, dead once `stance` was folded into the single adjudication call. Asked in isolation it scored 6/12 against one-pass adjudication's 9/12 — the finding is kept in `experiments/stance_eval.py`, only the code is gone. |
| `_WEB_SUFFIX` (`rag/pipeline.py`) | Superseded by the `_WEB_VERDICTS` mapping. |

`rag/llm.py` went 411 → 350 lines, `rag/config.py` 178 → 170.

## Kept deliberately

**`EvidenceIndex.search(skip=...)`** has no caller now that `FALLBACK_ROUNDS` is
gone. It is retained because it is working, measured code (chunking Exp. 5: round
2 returns four fresh passages at 0.8416–0.8531 with zero overlap) and is the
mechanism any future elimination loop would need. Noted as uncalled in
docs/chunking-experiments.md rather than left to be rediscovered.

**`search()`'s other parameters** (`answered_similarity`, `relative_cutoff`,
`topic_bonus`, `route`) are never passed by a caller either, but all four are
used in the function body and exist so a sweep can vary them. That is
parameterised configuration, not dead code.

## `is_usable` was duplicated four times

Byte-identical copies in `transcribe/config.py`, `rag/main.py`,
`experiments/datasets.py` and `experiments/chunk_eval.py`, each with a docstring noting it
mirrored the others. This function decides **which transcripts enter an
experiment**, so drift between copies would silently change the dataset.

Now one definition, in `transcribe/config.py` — the module that writes the
status strings it interprets. The cost is a new dependency from `rag/` and
`experiments/` onto `transcribe/`; it is acceptable because `transcribe/config.py`
imports only the standard library and `transcribe/__init__.py` loads lazily, so
nothing heavy is pulled in. It was the only cross-package import in `rag/`, and
was taken as a considered trade rather than by accident.

## Stale documentation corrected

Six references to removed identifiers were updated rather than deleted, so the
reasoning survives the code: `FALLBACK_ROUNDS` in
docs/adjudication-experiments.md (×2) and docs/chunking-experiments.md (×2),
`ROUTE_MIN_SIMILARITY` in docs/routing-experiments.md, and `read_stance` in
docs/web-search.md.

## Runtime estimate was optimistic

Preflight reported 17.7 s/window, taken from the 3-video smoke test — which
included a short video. The median over windows from five different test videos
is **21.0 s**, making the corpus-only run ~9.3 h rather than 7.8 h, and the
Tavily run ~12.5 h. Corrected in `experiments/run_experiment.py` and README.

## Not changed

`is_usable`'s four copies were the only true duplication found. The AST scan
flagged 33 unreferenced definitions; 28 were false positives (referenced via
dict literals, `__init__` re-exports, or call sites the walk missed) and were
verified individually with grep before being left alone.
