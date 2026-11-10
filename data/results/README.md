# Experiment results

Grouped by protocol version, so figures from different protocols are never
read side by side by accident. New runs go to `RESULTS_VERSION` (default `v2`,
rag/config.py); `--limit` runs go to `smoke/`.

## v1/ — first test round

The 119-video test runs as first measured: the prompts before the 2026-11-09
rewrite, no time check, and in most cases the evidence arm only.

| files | what |
|---|---|
| `test_run_<model>_tavily.*` | evidence arm (gemma-4-12b, gpt-oss-20b, gpt-oss-120b) |
| `test_run_<model>_parametric.*` | parametric arm (gpt-oss-120b; qwen3.6-27b partial) |
| `*.json.bak` | backups written by `tests/check_run.py --drop-errors` |
| `web_cache_after_<model>_tavily.tar.gz` | the web evidence each run used |
| `server_*.log` | llama-server logs from the workstation runs |
| `llm_failures_*.jsonl` | raw requests and responses that carried no JSON |
| `prompt_ab_dev.json` | the earlier prompt A/B (docs/prompt-experiments.md) |

## v2/ — current protocol

`scripts/run_test_model.sh`: production prompts (rules_v2), `--time-check`,
both arms, scored raw and with the overlap veto, plus the verifier
(docs/evidence-as-verifier.md).

| files | what |
|---|---|
| `test_run_<model>_{parametric,tavily}_tc.*` | test-set runs; `_ov_summary` = with the veto |
| `test_run_<model>_verifier_tc.txt` | the verifier, raw and with the veto |
| `protocol_<model>_tc.log` | everything the protocol script printed |
| `dev_run_*` | the development-set runs that chose the v2 protocol: prompt variants (`_base`, `_rules`, `_stance_first`, `_rules_v2`), `_tc` = time check |
| `web_cache_before_v2_test_runs.tar.gz` | the web cache as it stood before the v2 test runs |
| `llm_failures_*.jsonl`, `server_*.log` | as in v1, for v2 runs |

## v3/ — whole-transcript protocol

`WHOLE=1 scripts/run_test_model.sh` (rag/whole.py): one judgement per video over
the whole transcript instead of 140-word windows, with the v1 flagging
instructions and the time check: dates in the prompt and a TIME CHECK note for
time-bound wording (rag/temporal.py). Unlike v2 there is no deterministic time
guard: the model sees the dates and decides. The parametric arm is one call per video; the
evidence arm extracts the claims, retrieves corpus evidence for each, judges
them all in one call, and searches the web for at most 6 claims the corpus
cannot address. Rows are claims, not windows: the overlap veto does not apply,
and the verifier is read at video level (`video-AND`).

| files | what |
|---|---|
| `test_run_<model>_{parametric,tavily}_whole_tc.*` | test-set runs; `model_fake_news` per video = the model's own true/false |
| `test_run_<model>_verifier_whole_tc.txt` | the verifier (read the `video-AND` row) |
| `protocol_<model>_whole_tc.log` | everything the protocol script printed |

## smoke/ — short checks

Runs on a few videos (`--limit`), to check that a backend and the whole chain
work before a full run. Their scores are not results: the smoke videos are all
labelled fake, so nothing there measures false positives.

| files | what |
|---|---|
| `smoke_*.log` | console output of v1 smoke tests |
| `test_run_<model>_none.*` | v1 smoke runs, 3 videos, corpus only |
| `test_run_<model>_*_tc.json` | v2 smoke runs written by `run_test_model.sh` |
