#!/usr/bin/env bash
# llama-server with a 32k context; the caller sets MODEL_FILE, OFFLOAD, EXTRA and SERVER_TAG.
cd ~/ANTIGO/Heloysa/covid-detection-rag
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$HOME/ANTIGO/Heloysa/llama.cpp/build/bin:/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
mkdir -p data/results/v3; LOG=data/results/v3/server_${SERVER_TAG}.log
while true; do
  ~/ANTIGO/Heloysa/llama.cpp/build/bin/llama-server -m "$MODEL_FILE" \
    -c 32768 -np 1 --port 8080 --host 127.0.0.1 --jinja $EXTRA $OFFLOAD -t 6 2>&1 | tee -a "$LOG"
  echo "=== server exited $(date), restarting in 10 s" | tee -a "$LOG"; sleep 10
done
