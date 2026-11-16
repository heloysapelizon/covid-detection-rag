#!/usr/bin/env bash
# v3 whole-transcript protocol for gpt-oss-20b, queued behind the Gemma queue.
MAIN=~/ANTIGO/Heloysa/covid-detection-rag; cd "$MAIN"
M=~/ANTIGO/Heloysa/models
LOG=$MAIN/data/results/queue_v3_whole.log
: "${TAVILY_API_KEY:?TAVILY_API_KEY not set}"

echo "=== gpt-oss-20b queued, waiting for the Gemma queue $(date)" | tee -a "$LOG"
while pgrep -f "queue_v3_whole.sh|scripts/run_test_model.sh|experiments/run_experiment.py" >/dev/null; do sleep 60; done
echo "=== GPU free for gpt-oss-20b $(date)" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="" WHOLE=1 LLM_TIMEOUT_S=1200
unset LLM_SERVER TIME_CHECK OVERLAP_VETO LLM_REASONING RESULTS_VERSION
export LLAMA_MODEL=gpt-oss-20b LLM_CONSTRAINT=json_schema

stop_server() { tmux kill-window -t heloysa-tcc:server 2>/dev/null
  pkill -f "ANTIGO/Heloysa/run_server"; pkill -f "ANTIGO/Heloysa/models"; sleep 5; }
try_server() {  for off in "$@"; do
    stop_server
    tmux new-window -d -t heloysa-tcc -n server -e MODEL_FILE="$M/gpt-oss-20b-MXFP4.gguf" \
      -e EXTRA="-n 12288" -e OFFLOAD="$off" -e SERVER_TAG="gpt-oss-20b" ~/ANTIGO/Heloysa/run_server_whole.sh
    for i in $(seq 36); do
      if curl -sf localhost:8080/health >/dev/null; then
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$mem" -le 7600 ]; then
          echo "=== gpt-oss-20b: server up, 32k ctx, '$off', VRAM ${mem} MiB" | tee -a "$LOG"; return 0; fi
        echo "=== gpt-oss-20b: '$off' uses ${mem} MiB, trying less offload" | tee -a "$LOG"; break
      fi; sleep 5
    done
    echo "=== gpt-oss-20b: '$off' did not fit" | tee -a "$LOG"
  done
  return 1
}

if try_server "-ngl 99 --n-cpu-moe 12" "-ngl 99 --n-cpu-moe 14" "-ngl 99 --n-cpu-moe 16" "-ngl 99 --n-cpu-moe 18"; then
  echo "=== gpt-oss-20b: protocol start $(date)" | tee -a "$LOG"
  scripts/run_test_model.sh; echo "=== gpt-oss-20b: protocol exit $? $(date)" | tee -a "$LOG"
else
  echo "=== gpt-oss-20b: NO SERVER SETTING FITS, skipped" | tee -a "$LOG"
fi
stop_server
echo "=== gpt-oss-20b queue done $(date)" | tee -a "$LOG"
