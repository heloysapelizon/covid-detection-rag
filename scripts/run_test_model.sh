#!/usr/bin/env bash
# Full test-set protocol for one model: both arms, then every score.
#
#   export LLM_SERVER=openrouter LLAMA_MODEL=openai/gpt-oss-120b   # remote
#   export LLAMA_MODEL=gpt-oss-20b; unset LLM_SERVER                # local llama.cpp
#   scripts/run_test_model.sh
#
# Steps, in order:
#   1. preflight (server, index, test set, Tavily probe)
#   2. per arm (parametric, then evidence with the Tavily fallback):
#        smoke test on 3 videos, then the full run, both with --time-check
#   3. re-score each arm with the overlap veto (post-hoc, no model calls)
#   4. score the verifier (both arms must flag the same window), raw and vetoed
#
# Prompts: the test set always runs the production prompts in rag/llm.py. Prompt
# variants are compared on dev only (--dataset dev --prompt NAME); the winner is
# then copied into rag/llm.py. Never pass --prompt here.
#
# The overlap veto is applied at scoring time, not during the run, so the
# results file keeps the raw verdicts and both figures can be reported.
#
# Every step resumes: re-running the script after an interruption continues the
# unfinished run and re-scores the rest.
#
# WHOLE=1 runs the whole-transcript protocol instead (rag/whole.py): one
# judgement per video with the v1 flagging instructions plus the time check,
# results in data/results/v3/. The overlap veto does not apply (claims are not
# overlapping windows), and the verifier is read at video level (video-AND).
#
# Where things go (data/results/README.md):
#   data/results/$RESULTS_VERSION/   full runs, scores, verifier, this script's log
#                                    (RESULTS_VERSION defaults to v2)
#   data/results/smoke/              the 3-video smoke tests, kept apart so the
#                                    full run is one complete measurement
set -euo pipefail   # pipefail: a crashed run piped into tee must stop the script
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
: "${LLAMA_MODEL:?set LLAMA_MODEL (and LLM_SERVER for a remote model)}"
SLUG=${LLAMA_MODEL//\//-}
SLUG=${SLUG//:/-}
if [[ ${WHOLE:-0} == 1 ]]; then
  export RESULTS_VERSION=${RESULTS_VERSION:-v3}
  FEATURES=(--whole --time-check); MODE=_whole_tc
else
  export RESULTS_VERSION=${RESULTS_VERSION:-v2}
  FEATURES=(--time-check); MODE=_tc
fi
RESULTS=data/results/$RESULTS_VERSION
SMOKE=data/results/smoke
R=$RESULTS/test_run_${SLUG}
# Same tag the runner adds for LLM_REASONING (rag/config.py reasoning_tag), so a
# non-thinking run never resumes into, or overwrites, a thinking one.
case "${LLM_REASONING:-}" in
  "")  TAG="" ;;
  off) TAG="_nothink" ;;
  *)   TAG="_reasoning-${LLM_REASONING}" ;;
esac
export PYTHONUNBUFFERED=1
mkdir -p "$RESULTS" "$SMOKE"
# Everything this script prints also goes to its protocol log.
exec > >(tee -a "$RESULTS/protocol_${SLUG}${TAG}${MODE}.log") 2>&1

echo "=== $(date '+%F %T') preflight  model=$LLAMA_MODEL  server=${LLM_SERVER:-llamacpp}  reasoning=${LLM_REASONING:-default}"
$PY tests/run_experiment.py --check --web tavily

for arm in parametric tavily; do
  if [[ $arm == parametric ]]; then args=(--condition parametric)
  else args=(--web tavily); fi
  out=${R}_${arm}${TAG}${MODE}.json
  smoke_out=$SMOKE/test_run_${SLUG}_${arm}${TAG}${MODE}.json
  # Each model fetches its own web evidence, as in the earlier test runs: the
  # cache is cleared once, before the evidence arm's first video. Not on a
  # resume, which would refetch (and pay for) evidence the run already used.
  fresh=()
  [[ $arm == tavily && ! -f $out ]] && fresh=(--clear-cache)
  echo "=== $(date '+%F %T') smoke  $arm"
  $PY tests/run_experiment.py "${args[@]}" "${FEATURES[@]}" ${fresh[@]+"${fresh[@]}"} \
    --limit 3 --out "$smoke_out"
  # A window that errors is saved inside a finished video and never retried, so
  # stop here rather than carry a broken backend into a multi-hour run.
  # Captured first, then searched: piping into `grep -q` lets grep exit on the
  # first match, the writer then dies of SIGPIPE, and under pipefail a clean
  # smoke test reads as a failure (that race stopped the first 120b run).
  check=$($PY tests/check_run.py "$smoke_out")
  echo "$check"
  if ! grep -q "^0 video(s) with ERROR" <<< "$check"; then
    echo "=== smoke test of $arm has ERROR windows; fix the cause, then"
    echo "    $PY tests/check_run.py $smoke_out --drop-errors   and re-run this script"
    exit 1
  fi
  echo "=== $(date '+%F %T') full   $arm  -> $out"
  $PY tests/run_experiment.py "${args[@]}" "${FEATURES[@]}" --out "$out" 2>&1 \
    | tee -a "${R}_${arm}${TAG}${MODE}.log"
  if [[ $MODE == _tc ]]; then
    echo "=== $(date '+%F %T') score  $arm with overlap veto"
    $PY tests/run_experiment.py --score --overlap-veto --out "$out"
  fi
done

echo "=== $(date '+%F %T') verifier"
{
  $PY tests/score_verifier.py "${R}_parametric${TAG}${MODE}.json" "${R}_tavily${TAG}${MODE}.json"
  if [[ $MODE == _tc ]]; then
    echo
    $PY tests/score_verifier.py "${R}_parametric${TAG}${MODE}.json" "${R}_tavily${TAG}${MODE}.json" --overlap-veto
  fi
} | tee "${R}_verifier${TAG}${MODE}.txt"
echo "=== $(date '+%F %T') done"
