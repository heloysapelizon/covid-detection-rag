#!/usr/bin/env bash
# CTX sets the context size (default 8192, as the windowed runs used).
# Whole-transcript runs (WHOLE=1) need CTX=32768: the transcript and every
# claim's evidence share one prompt, up to ~22k tokens on the test set.
cd ~/ANTIGO/Heloysa/covid-detection-rag
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$HOME/ANTIGO/Heloysa/llama.cpp/build/bin:$LD_LIBRARY_PATH"
mkdir -p data/results/v2
LOG=data/results/v2/server_gpt-oss-20b.log
while true; do
  ~/ANTIGO/Heloysa/llama.cpp/build/bin/llama-server \
    -m ~/ANTIGO/Heloysa/models/gpt-oss-20b-MXFP4.gguf \
    -c ${CTX:-8192} -np 1 --port 8080 --host 127.0.0.1 --jinja \
    -ngl 99 --n-cpu-moe 12 -t 6 2>&1 | tee -a "$LOG"
  echo "=== server exited $(date), restarting in 10 s" | tee -a "$LOG"
  sleep 10
done
