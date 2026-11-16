# scripts/

`run_test_model.sh` is the full test-set protocol for one model: preflight,
then for each arm (parametric, then evidence with Tavily) a 3-video smoke test
followed by the full run, the overlap-veto re-score, and finally the verifier
over both arms. It resumes an interrupted run and `tee`s everything into
`data/results/<version>/protocol_<model>.log`.

```bash
export LLAMA_MODEL=gemma-4-12b          # always set explicitly; names the output files
scripts/run_test_model.sh               # windowed protocol (v1) -> data/results/v1/
WHOLE=1 scripts/run_test_model.sh       # whole-transcript protocol (v3) -> data/results/v3/
```

`LLM_SERVER`, `LLM_CONSTRAINT`, `LLM_REASONING` and `PY` are passed through;
see the top-level README for the per-model settings.

## workstation/

Drivers for the GPU workstation the reported runs were made on. They hard-code
that machine's paths, tmux session and llama.cpp build, so they will not run
elsewhere unedited. They are kept as the record of how each run was launched:
which GGUF, context size, offload setting and queue order.

| script | what it launched |
|---|---|
| `run_server.sh`, `run_server_gemma.sh` | llama-server for gpt-oss-20b and Gemma 4 12B, 8k context, auto-restart |
| `run_server_whole.sh` | llama-server with a 32k context for the whole-transcript protocol |
| `run_tavily.sh`, `run_gemma_tavily.sh` | wait for the server, then run the v1 protocol for one model |
| `run_parametric_both.sh` | the v1 parametric arm for both local models in sequence |
| `queue_v1_parametric.sh` | the v1 parametric arms from a worktree, queued behind the v2 runs |
| `queue_v3_whole.sh`, `queue_v3_gptoss.sh` | the v3 protocol, trying offload settings until the model fits in VRAM |
| `run_gemma_rag_redo.sh` | the Gemma v3 evidence arm with one redo of errored videos |
