#!/usr/bin/env bash
cd ~/ANTIGO/Heloysa/covid-detection-rag
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export LLAMA_MODEL=gemma-4-12b CUDA_VISIBLE_DEVICES=""
unset LLM_SERVER LLM_CONSTRAINT
LOG=data/results/test_run_gemma-4-12b_tavily.log
OUT=data/results/test_run_gemma-4-12b_tavily.json

echo "waiting for llama-server..." | tee -a "$LOG"
for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done

python -u tests/run_experiment.py --check --web tavily 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "=== PREFLIGHT FAILED" | tee -a "$LOG"; exec bash; }

# smoke test: 1 video through the web path; stop if anything errored
python -u tests/run_experiment.py --limit 1 --web tavily 2>&1 | tee -a "$LOG"
python ~/ANTIGO/Heloysa/check_run.py "$OUT" 2>&1 | tee -a "$LOG" | grep -q "^0 video(s) with ERROR" \
  || { echo "=== SMOKE TEST HAS ERRORS, stopping" | tee -a "$LOG"; exec bash; }

INHIBIT=""
systemd-inhibit --what=sleep:idle --why=test true 2>/dev/null \
  && INHIBIT="systemd-inhibit --what=sleep:idle --why=Heloysa-TCC-run" \
  || echo "=== WARNING: systemd-inhibit denied, sleep not blocked" | tee -a "$LOG"

$INHIBIT python -u tests/run_experiment.py --web tavily --clear-cache 2>&1 | tee -a "$LOG"
echo "=== run finished $(date)" | tee -a "$LOG"
exec bash
