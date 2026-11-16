# experiments/

The evaluation harness. Nothing here is a unit test: these scripts define the
dev/test split, run the pipeline over it, score the result and record the
sweeps behind every parameter in `rag/config.py`. Run them from the project
root as `python experiments/<name>.py`. Window-level labels live in
`data/labels/windows/`.

## Split and protocol

| script | what it does |
|---|---|
| `datasets.py` | The authority on which videos are test, dev or fallback, with leakage checks. Run it to print the split. |
| `run_experiment.py` | Runs one condition over one split and scores it. `--check` is the preflight, `--score` re-scores a saved file. |
| `score_verifier.py` | Scores both arms of a model together: verifier (both must flag) and evidence veto. |
| `check_run.py` | Sanity-checks a results file: verdict counts and ERROR windows. `--drop-errors` removes them so a resume redoes those videos. |
| `prompt_variants.py` | Candidate prompt pairs for `--prompt NAME` on the dev set. The winner is copied into `rag/llm.py`; the test set always runs the production prompts. |
| `window_metrics.py` | Window- and claim-level precision beside the video-level score. |
| `offclaim_sample.py` | Audits a random sample of off-claim flags to turn the window bounds into an estimate. |
| `baselines.py` | Reference classifiers: all-positive, date and provenance rules, TF-IDF with logistic regression. |

## Publication tools

| script | what it does |
|---|---|
| `redact_text.py` | `--strip` removes the window text from windowed result files, `--restore` recomputes it from the transcripts, `--check` verifies they match. |
| `web_cache_index.py` | Writes the queries, URLs and retrieval times of a web cache or its archive, without the page text. |

## Development-set sweeps

| script | what it does |
|---|---|
| `select_dev.py` | Chooses balanced subsets from a collected candidate pool (`--group mirror` or `uncovered`). |
| `chunk_eval.py` | Chunk-size sweep; `--make-labels` proposes windows to label. |
| `route_eval.py` | Finds the routing confidence threshold below which a claim stays general. |
| `retrieval_compare.py` | Dense versus BM25 versus hybrid retrieval on the labelled phrases. |
| `split.py` | Internal A/B split of the v1 labelled windows, for transfer checks. |
| `select_prompt_cases.py`, `prompt_ab.py`, `score_prompt_ab.py` | The prompt A/B: pick cases, run two prompts over identical retrieval, score. |

## Probes and smoke checks

| script | what it does |
|---|---|
| `gate_probe.py` | The relevance gate on claims the corpus does and does not cover. Needs the model server. |
| `stance_eval.py` | Can the configured model tell asserting from denying? |
| `ollama_schema_probe.py` | Does a server enforce the verdict JSON schema, or merely get lucky? |
| `bench_adjudicate.py` | Times N adjudications on real windows, to compare machines. |
| `search_smoke.py` | Peek at what one web-search query returns. |
| `transcribe_smoke.py` | Transcribe a single video end to end. |

The findings these produced are written up in `docs/`.
