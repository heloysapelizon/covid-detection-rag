#!/usr/bin/env bash
# Workstation launcher for gpt-oss-20b: waits for llama-server, blocks sleep, then
# runs the full test protocol (scripts/run_test_model.sh): both arms with
# --time-check, the overlap-veto scores and the verifier.
cd ~/ANTIGO/Heloysa/covid-detection-rag
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export LLAMA_MODEL=gpt-oss-20b LLM_CONSTRAINT=json_schema CUDA_VISIBLE_DEVICES=""
unset LLM_SERVER
export PY=python
LOG=data/results/test_run_gpt-oss-20b_protocol.log

echo "waiting for llama-server..." | tee -a "$LOG"
for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done

INHIBIT=""
systemd-inhibit --what=sleep:idle --why=test true 2>/dev/null \
  && INHIBIT="systemd-inhibit --what=sleep:idle --why=Heloysa-TCC-run" \
  || echo "=== WARNING: systemd-inhibit denied, sleep not blocked" | tee -a "$LOG"

$INHIBIT scripts/run_test_model.sh 2>&1 | tee -a "$LOG"
echo "=== protocol finished $(date) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
exec bash
