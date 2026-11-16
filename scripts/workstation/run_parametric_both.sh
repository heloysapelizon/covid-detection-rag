#!/usr/bin/env bash
# Parametric arm for both local models, one after the other.
cd ~/ANTIGO/Heloysa/covid-detection-rag
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
unset LLM_SERVER
MAIN=data/results/run_parametric_both.log

INHIBIT=""
systemd-inhibit --what=sleep:idle --why=test true 2>/dev/null \
  && INHIBIT="systemd-inhibit --what=sleep:idle --why=Heloysa-TCC-run" \
  || echo "=== WARNING: systemd-inhibit denied, sleep not blocked" | tee -a "$MAIN"

stop_server() {
  tmux kill-window -t heloysa-tcc:server 2>/dev/null
  pkill -f "ANTIGO/Heloysa/run_server"; pkill -f "ANTIGO/Heloysa/models"; sleep 5
}
start_server() {  stop_server
  tmux new-window -d -t heloysa-tcc -n server "$1"
  for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && return 0; sleep 5; done
  return 1
}
run_model() {
  local model=$1 L=data/results/test_run_$1_parametric.log OUT=data/results/test_run_$1_parametric.json
  export LLAMA_MODEL=$model
  if [ -n "$3" ]; then export LLM_CONSTRAINT=$3; else unset LLM_CONSTRAINT; fi
  echo "=== $model: starting server $(date)" | tee -a "$MAIN"
  start_server "$2" || { echo "=== $model: SERVER DID NOT START" | tee -a "$MAIN"; return 1; }
  python -u experiments/run_experiment.py --condition parametric --check 2>&1 | tee -a "$L"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "=== $model: PREFLIGHT FAILED" | tee -a "$MAIN"; return 1; }
  python -u experiments/run_experiment.py --condition parametric --limit 3 2>&1 | tee -a "$L"
  python experiments/check_run.py "$OUT" 2>&1 | tee -a "$L" | grep -q "^0 video(s) with ERROR" \
    || { echo "=== $model: SMOKE TEST HAS ERRORS, skipping full run" | tee -a "$MAIN"; return 1; }
  $INHIBIT python -u experiments/run_experiment.py --condition parametric 2>&1 | tee -a "$L"
  python experiments/check_run.py "$OUT" 2>&1 | tee -a "$L" "$MAIN"
  echo "=== $model: finished $(date)" | tee -a "$MAIN"
}

run_model gemma-4-12b ~/ANTIGO/Heloysa/run_server_gemma.sh ""
run_model gpt-oss-20b ~/ANTIGO/Heloysa/run_server.sh       json_schema
stop_server
echo "=== all done $(date)" | tee -a "$MAIN"

