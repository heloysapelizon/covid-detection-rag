#!/usr/bin/env bash
# Waits for the running v2 protocol, then, from a worktree at 36c481d (v1 code + prompts):
#   1. gpt-oss-20b v1 parametric (resumes)        2. redo its errored videos, if any
#   3. redo gemma-4-12b v1 parametric's errored videos (first attempt kept in .bak)
MAIN=~/ANTIGO/Heloysa/covid-detection-rag
WT=~/ANTIGO/Heloysa/covid-detection-rag-v1
PY=$MAIN/.venv/bin/python
LOG=$MAIN/data/results/queue_v1_parametric.log

echo "=== waiting for the v2 protocol to finish $(date)" | tee -a "$LOG"
while pgrep -f "scripts/run_test_model.sh" >/dev/null; do sleep 60; done
echo "=== v2 protocol finished $(date)" | tee -a "$LOG"

cd "$WT"
export CUDA_VISIBLE_DEVICES=""; unset LLM_SERVER TIME_CHECK OVERLAP_VETO
stop_server() { tmux kill-window -t heloysa-tcc:server 2>/dev/null
  pkill -f "ANTIGO/Heloysa/run_server"; pkill -f "ANTIGO/Heloysa/models"; sleep 5; }
start_server() { stop_server; tmux new-window -d -t heloysa-tcc -n server "$1"
  for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && return 0; sleep 5; done; return 1; }
use_model() {                                # $1 model, $2 server script, $3 constraint or ""
  export LLAMA_MODEL=$1
  if [ -n "$3" ]; then export LLM_CONSTRAINT=$3; else unset LLM_CONSTRAINT; fi
  echo "=== $1: starting server $(date)" | tee -a "$LOG"
  start_server "$2" || { echo "=== $1: SERVER DID NOT START" | tee -a "$LOG"; return 1; }
  $PY tests/run_experiment.py --condition parametric --check >> "$LOG" 2>&1 \
    || { echo "=== $1: PREFLIGHT FAILED" | tee -a "$LOG"; return 1; }
}
run_v1() {                                   # resume the v1 parametric run of $LLAMA_MODEL
  local out=data/results/test_run_${LLAMA_MODEL}_parametric.json
  $PY -u tests/run_experiment.py --condition parametric 2>&1 | tee -a "data/results/test_run_${LLAMA_MODEL}_parametric.log"
  $PY tests/check_run.py "$out" 2>&1 | tee -a "$LOG"
}
redo_errors() {                              # drop videos with ERROR windows, then resume = redo only those
  local out=data/results/test_run_${LLAMA_MODEL}_parametric.json
  if $PY tests/check_run.py "$out" | grep -q "^0 video(s) with ERROR"; then
    echo "=== $LLAMA_MODEL: no ERROR windows, nothing to redo" | tee -a "$LOG"; return 0; fi
  echo "=== $LLAMA_MODEL: redoing errored videos $(date)" | tee -a "$LOG"
  $PY tests/check_run.py "$out" --drop-errors 2>&1 | tee -a "$LOG"
  run_v1
  echo "=== $LLAMA_MODEL: redo finished $(date)" | tee -a "$LOG"
}

use_model gpt-oss-20b ~/ANTIGO/Heloysa/run_server.sh json_schema && { run_v1; redo_errors; }
use_model gemma-4-12b ~/ANTIGO/Heloysa/run_server_gemma.sh ""   && redo_errors
stop_server
echo "=== queue done $(date)" | tee -a "$LOG"
