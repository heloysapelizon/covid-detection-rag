#!/usr/bin/env bash
# Full test-set protocol for one model (set LLAMA_MODEL, optionally LLM_SERVER): both arms,
# smoke tests, overlap-veto scores and the verifier. WHOLE=1 runs the whole-transcript protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
: "${LLAMA_MODEL:?set LLAMA_MODEL (and LLM_SERVER for a remote model)}"
SLUG=${LLAMA_MODEL//\//-}
SLUG=${SLUG//:/-}
if [[ ${WHOLE:-0} == 1 ]]; then
  export RESULTS_VERSION=${RESULTS_VERSION:-v3}
  FEATURES=(--whole --time-check); MODE=_whole_tc
else
  export RESULTS_VERSION=${RESULTS_VERSION:-v1}
  FEATURES=(); MODE=""
fi
RESULTS=data/results/$RESULTS_VERSION
SMOKE=data/results/smoke
R=$RESULTS/test_run_${SLUG}
case "${LLM_REASONING:-}" in
  "")  TAG="" ;;
  off) TAG="_nothink" ;;
  *)   TAG="_reasoning-${LLM_REASONING}" ;;
esac
export PYTHONUNBUFFERED=1
mkdir -p "$RESULTS" "$SMOKE"
exec > >(tee -a "$RESULTS/protocol_${SLUG}${TAG}${MODE}.log") 2>&1

echo "=== $(date '+%F %T') preflight  model=$LLAMA_MODEL  server=${LLM_SERVER:-llamacpp}  reasoning=${LLM_REASONING:-default}"
$PY experiments/run_experiment.py --check --web tavily

for arm in parametric tavily; do
  if [[ $arm == parametric ]]; then args=(--condition parametric)
  else args=(--web tavily); fi
  out=${R}_${arm}${TAG}${MODE}.json
  smoke_out=$SMOKE/test_run_${SLUG}_${arm}${TAG}${MODE}.json
  fresh=()
  [[ $arm == tavily && ! -f $out ]] && fresh=(--clear-cache)
  echo "=== $(date '+%F %T') smoke  $arm"
  $PY experiments/run_experiment.py "${args[@]}" ${FEATURES[@]+"${FEATURES[@]}"} ${fresh[@]+"${fresh[@]}"} \
    --limit 3 --out "$smoke_out"
  check=$($PY experiments/check_run.py "$smoke_out")
  echo "$check"
  if ! grep -q "^0 video(s) with ERROR" <<< "$check"; then
    echo "=== smoke test of $arm has ERROR windows; fix the cause, then"
    echo "    $PY experiments/check_run.py $smoke_out --drop-errors   and re-run this script"
    exit 1
  fi
  echo "=== $(date '+%F %T') full   $arm  -> $out"
  $PY experiments/run_experiment.py "${args[@]}" ${FEATURES[@]+"${FEATURES[@]}"} --out "$out" 2>&1 \
    | tee -a "${R}_${arm}${TAG}${MODE}.log"
  if [[ $MODE != _whole_tc ]]; then
    echo "=== $(date '+%F %T') score  $arm with overlap veto"
    $PY experiments/run_experiment.py --score --overlap-veto --out "$out"
  fi
done

echo "=== $(date '+%F %T') verifier"
{
  $PY experiments/score_verifier.py "${R}_parametric${TAG}${MODE}.json" "${R}_tavily${TAG}${MODE}.json"
  if [[ $MODE != _whole_tc ]]; then
    echo
    $PY experiments/score_verifier.py "${R}_parametric${TAG}${MODE}.json" "${R}_tavily${TAG}${MODE}.json" --overlap-veto
  fi
} | tee "${R}_verifier${TAG}${MODE}.txt"
echo "=== $(date '+%F %T') done"
