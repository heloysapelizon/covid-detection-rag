#!/usr/bin/env bash
cd ~/ANTIGO/Heloysa/covid-detection-rag
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$HOME/ANTIGO/Heloysa/llama.cpp/build/bin:$LD_LIBRARY_PATH"
mkdir -p data/results/v2
LOG=data/results/v2/server_gemma-4-12b.log
while true; do
  ~/ANTIGO/Heloysa/llama.cpp/build/bin/llama-server \
    -m /home/davint-workstation/ANTIGO/Heloysa/models/gemma-4-12b-it-Q4_K_M.gguf \
    -c 8192 -np 1 --port 8080 --host 127.0.0.1 --jinja --reasoning-budget 0 \
    -ngl 48 -t 6 2>&1 | tee -a "$LOG"
  echo "=== server exited $(date), restarting in 10 s" | tee -a "$LOG"
  sleep 10
done
