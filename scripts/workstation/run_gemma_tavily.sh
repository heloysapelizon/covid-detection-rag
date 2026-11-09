#!/usr/bin/env bash
# Workstation launcher for gemma-4-12b: waits for llama-server, blocks sleep, then
# runs the full test protocol (scripts/run_test_model.sh): both arms with
# --time-check, the overlap-veto scores and the verifier.
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

# The protocol writes its own log: data/results/v2/protocol_<model>_tc.log
$INHIBIT scripts/run_test_model.sh
echo "=== protocol finished $(date) (exit $?)"
exec bash
