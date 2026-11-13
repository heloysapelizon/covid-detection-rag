#!/usr/bin/env bash
# Gemma v3 RAG arm without the smoke gate: full run, one redo of errored videos, verifier.
cd ~/ANTIGO/Heloysa/covid-detection-rag
export CUDA_VISIBLE_DEVICES="" WHOLE=1 LLM_TIMEOUT_S=1200 LLAMA_MODEL=gemma-4-12b PYTHONUNBUFFERED=1
unset LLM_SERVER LLM_CONSTRAINT TIME_CHECK OVERLAP_VETO LLM_REASONING RESULTS_VERSION
: "${TAVILY_API_KEY:?TAVILY_API_KEY not set}"
PY=.venv/bin/python; R=data/results/v3/test_run_gemma-4-12b
OUT=${R}_tavily_whole_tc.json; L=${R}_tavily_whole_tc.log; Q=data/results/queue_v3_whole.log
FEAT=(--web tavily --whole --time-check)

if ! curl -sf localhost:8080/health >/dev/null; then          # same server setting as the parametric arm
  tmux kill-window -t heloysa-tcc:server 2>/dev/null; pkill -f "ANTIGO/Heloysa/run_server"; pkill -f "ANTIGO/Heloysa/models"; sleep 5
  tmux new-window -d -t heloysa-tcc -n server -e MODEL_FILE=$HOME/ANTIGO/Heloysa/models/gemma-4-12b-it-Q4_K_M.gguf \
    -e EXTRA="--reasoning-budget 0 -n 6144" -e OFFLOAD="-ngl 42" -e SERVER_TAG=gemma-4-12b ~/ANTIGO/Heloysa/run_server_whole.sh
  for i in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done
fi
curl -sf localhost:8080/health >/dev/null || { echo "=== gemma RAG: SERVER DID NOT START" | tee -a $Q; exit 1; }

fresh=(); [ -f "$OUT" ] || fresh=(--clear-cache)              # fresh web evidence only on the first start
echo "=== gemma RAG: full run $(date)" | tee -a $Q
$PY tests/run_experiment.py "${FEAT[@]}" ${fresh[@]+"${fresh[@]}"} --out "$OUT" 2>&1 | tee -a "$L"

if ! $PY tests/check_run.py "$OUT" | grep -q "^0 video(s) with ERROR"; then
  echo "=== gemma RAG: redoing errored videos $(date)" | tee -a $Q
  $PY tests/check_run.py "$OUT" --drop-errors 2>&1 | tee -a "$L"      # first attempt kept in .bak
  $PY tests/run_experiment.py "${FEAT[@]}" --out "$OUT" 2>&1 | tee -a "$L"
fi
$PY tests/check_run.py "$OUT" 2>&1 | tee -a $Q

P=${R}_parametric_whole_tc.json
[ -f "$P" ] && $PY tests/score_verifier.py "$P" "$OUT" | tee ${R}_verifier_whole_tc.txt
echo "=== gemma RAG done $(date)" | tee -a $Q
