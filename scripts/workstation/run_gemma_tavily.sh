#!/usr/bin/env bash
# Waits for llama-server, then runs scripts/run_test_model.sh for gemma-4-12b.
cd ~/ANTIGO/Heloysa/covid-detection-rag
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export LLAMA_MODEL=gemma-4-12b CUDA_VISIBLE_DEVICES=""
unset LLM_SERVER LLM_CONSTRAINT
export PY=python

echo "waiting for llama-server..."
for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done

INHIBIT=""
systemd-inhibit --what=sleep:idle --why=test true 2>/dev/null \
  && INHIBIT="systemd-inhibit --what=sleep:idle --why=Heloysa-TCC-run" \
  || echo "=== WARNING: systemd-inhibit denied, sleep not blocked"

$INHIBIT scripts/run_test_model.sh
echo "=== protocol finished $(date) (exit $?)"
exec bash
