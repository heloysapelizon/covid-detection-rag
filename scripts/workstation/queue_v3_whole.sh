#!/usr/bin/env bash
# v3 = whole-transcript protocol (WHOLE=1 scripts/run_test_model.sh: parametric + rag/tavily,
# smoke tests, verifier) for gemma-4-12b, then gpt-oss-20b. Waits until the GPU is free.
MAIN=~/ANTIGO/Heloysa/covid-detection-rag; cd "$MAIN"
M=~/ANTIGO/Heloysa/models
LOG=$MAIN/data/results/queue_v3_whole.log
: "${TAVILY_API_KEY:?TAVILY_API_KEY not set}"

echo "=== waiting for other runs $(date)" | tee -a "$LOG"
while pgrep -f "queue_v1_parametric|scripts/run_test_model.sh|tests/run_experiment.py" >/dev/null; do sleep 60; done
echo "=== GPU free $(date)" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="" WHOLE=1 LLM_TIMEOUT_S=1200        # long whole-transcript calls
unset LLM_SERVER TIME_CHECK OVERLAP_VETO LLM_REASONING RESULTS_VERSION

stop_server() { tmux kill-window -t heloysa-tcc:server 2>/dev/null
  pkill -f "ANTIGO/Heloysa/run_server"; pkill -f "ANTIGO/Heloysa/models"; sleep 5; }
# Try offload settings in order until one loads and leaves VRAM headroom (<= 7600 MiB).
try_server() {                     # $1 tag, $2 model file, $3 extra flags, then offload candidates
  local tag=$1 file=$2 extra=$3; shift 3
  for off in "$@"; do
    stop_server
    tmux new-window -d -t heloysa-tcc -n server -e MODEL_FILE="$file" -e EXTRA="$extra" \
      -e OFFLOAD="$off" -e SERVER_TAG="$tag" ~/ANTIGO/Heloysa/run_server_whole.sh
    for i in $(seq 36); do
      if curl -sf localhost:8080/health >/dev/null; then
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$mem" -le 7600 ]; then
          echo "=== $tag: server up, 32k ctx, '$off', VRAM ${mem} MiB" | tee -a "$LOG"; return 0; fi
        echo "=== $tag: '$off' loads but uses ${mem} MiB, trying less offload" | tee -a "$LOG"; break
      fi; sleep 5
    done
    echo "=== $tag: '$off' did not fit" | tee -a "$LOG"
  done
  return 1
}
run_model() {                      # $1 model label, $2 constraint or ""
  export LLAMA_MODEL=$1
  if [ -n "$2" ]; then export LLM_CONSTRAINT=$2; else unset LLM_CONSTRAINT; fi
  echo "=== $1: protocol start $(date)" | tee -a "$LOG"
  scripts/run_test_model.sh; echo "=== $1: protocol exit $? $(date)" | tee -a "$LOG"
}

N=$(grep -oP '(?<=-ngl )\d+' ~/ANTIGO/Heloysa/run_server_gemma.sh | head -1)   # current Gemma split
try_server gemma-4-12b $M/gemma-4-12b-it-Q4_K_M.gguf "--reasoning-budget 0 -n 6144" \
  "-ngl $N" "-ngl $((N-6))" "-ngl $((N-12))" "-ngl $((N-18))" \
  && run_model gemma-4-12b "" \
  || echo "=== gemma-4-12b: NO SERVER SETTING FITS, skipped" | tee -a "$LOG"

try_server gpt-oss-20b $M/gpt-oss-20b-MXFP4.gguf "-n 12288" \
  "-ngl 99 --n-cpu-moe 12" "-ngl 99 --n-cpu-moe 14" "-ngl 99 --n-cpu-moe 16" "-ngl 99 --n-cpu-moe 18" \
  && run_model gpt-oss-20b json_schema \
  || echo "=== gpt-oss-20b: NO SERVER SETTING FITS, skipped" | tee -a "$LOG"

stop_server
echo "=== queue done $(date)" | tee -a "$LOG"
