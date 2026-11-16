# COVID-19 vaccine misinformation detection

A pipeline that collects YouTube videos about COVID-19 vaccines, transcribes
their audio, and flags claims that a vetted evidence corpus contradicts. A
locally served language model adjudicates each claim against retrieved
evidence; the result is scored against hand-labelled videos that were never
used to tune anything.

## How it works

```
collect ──► transcribe ──► rag ──► experiments
  │             │           │          │
  YouTube       Whisper     retrieval  held-out test set,
  Data API      audio       + LLM      scored against labels
                            ▲
                 data/evidence (PDFs) ──► data/index (vectors)
```

| stage | package | does |
|---|---|---|
| 1 | `collect/` | Searches the YouTube Data API for the configured queries, completes each hit with full metadata, dedupes and drops videos over an hour. Never scrapes. |
| 2 | `transcribe/` | Downloads each video's audio and transcribes it with faster-whisper. Drops non-English audio after language detection, checkpoints after every video. |
| 3 | `rag/` | Splits a transcript into overlapping claim windows, retrieves corpus passages for each, and asks the model whether the evidence addresses the claim and, only then, whether it contradicts it. Claims the corpus cannot answer can fall back to a web search. |
| 4 | `experiments/` | Defines the test/dev split, runs the pipeline over it, and scores it. Every parameter in `rag/config.py` was chosen on the dev set. |

The pipeline gathers candidates and classifies them afterwards. Collection is
deliberately neutral so that the dataset is assembled before anything is judged.

## Repository layout

```
collect/         stage 1: config  models  filters  data  api  collector  main
transcribe/      stage 2: config  audio  whisper  data  transcriber  main
rag/             stage 3: config  corpus  manifest  router  index  lexical  llm
                          websearch  temporal  pipeline  whole  main
experiments/     evaluation harness: split, runner, scoring, sweeps, probes
scripts/         run_test_model.sh (the protocol); workstation/ (GPU box drivers)
data/            youtube/  evidence/  index/  labels/  results/
docs/            what was measured and what it decided
notebooks/       the original exploratory notebooks, superseded by the packages
LICENSE          MIT, for the code;  LICENSE-DATA.md: terms of every data file
CITATION.cff     citation metadata for the repository and the thesis
```

`experiments/`, `scripts/` and `data/` each have their own README.

The three pipeline packages share one layering, grouped by external dependency
so a failure stays contained and each layer works without the ones above it:

| layer | depends on |
|---|---|
| `config`, `models` | nothing |
| `filters`, `router`, `temporal` | pure logic, no I/O |
| `data`, `corpus`, `manifest` | stdlib and pandas; every write is atomic |
| `api`, `audio`, `whisper`, `index`, `llm`, `websearch` | the external boundary: network, yt-dlp, ffmpeg, Whisper, the model server |
| `collector`, `transcriber`, `pipeline`, `whole` | the layers above; owns the order of work |
| `main` | argument parsing and exit codes |

Dependencies point one way. Public names are re-exported lazily (PEP 562), so
importing one module does not pull in the heavy dependencies of its siblings.
The only cross-package import is `is_usable` from `transcribe.config`, which
decides which transcripts enter an experiment and therefore has a single
definition.

## Setup

Python 3.10 or newer. `ffmpeg` is needed only to transcribe; the experiment
uses the committed transcripts.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # or apt install ffmpeg
```

Keys go in `.env` at the project root (gitignored) or in the environment:

| key | needed for |
|---|---|
| `YOUTUBE_API_KEY` | `collect`. Free: create a project in the Google Cloud Console and enable YouTube Data API v3 |
| `TAVILY_API_KEY` | the `--web tavily` fallback. The free tier covers one full test run |
| `OPENROUTER_API_KEY` | the two remote models |

## Usage

### 1. Collect

```bash
python -m collect                                   # the queries in collect/config.py
python -m collect --links links.txt --no-search     # a curated list instead
python -m collect --name my_dev --exclude data/youtube/exclude_dev2.json
```

`--exclude` keeps already-seen videos out of a new collection, so dev material
cannot overlap the test set. Language is not filtered here: YouTube's language
field is self-reported and often wrong, so `transcribe` decides from the audio.

### 2. Transcribe

```bash
python -m transcribe --list
python -m transcribe covid_vaccine_videos
python -m transcribe covid_vaccine_videos --limit 5
python -m transcribe covid_vaccine_videos --max-transcribe 0    # whole videos
```

Roughly 6x realtime with the `small` model on an M2 Pro. An interrupted run
resumes from its checkpoint. Dev sets are capped at 12 minutes per video;
the test set was transcribed whole.

### 3. Fact-check

The model is served locally through an OpenAI-compatible endpoint. llama.cpp is
the default because it enforces the verdict grammar during sampling, so a
malformed verdict cannot be generated.

```bash
# gpt-oss-20b (needs --jinja: its reasoning preamble otherwise leaks into the output)
llama-server -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 -c 8192 --port 8080 --host 127.0.0.1 --jinja

# Gemma 4 12B
llama-server -hf google/gemma-4-12B-it-qat-q4_0-gguf -c 8192 --port 8080 --host 127.0.0.1
```

`-c 8192` matters: the prompt carries up to eleven passages and a smaller
context silently truncates them. On a GPU add `-ngl 99`.

| variable | default | meaning |
|---|---|---|
| `LLM_SERVER` | `llamacpp` | `ollama` or `openrouter` to switch backend; each sets its own URL and output constraint |
| `LLAMA_URL` | per backend | the server, if not on the default host and port |
| `LLAMA_MODEL` | `qwen2.5:14b` | **set it explicitly**: llama-server ignores it, but it names the results files and is written into run metadata |
| `LLM_CONSTRAINT` | per backend | `grammar`, `json_schema` or `json_object`, to override the backend's default |
| `LLM_REASONING` | model default | `off`, `low`, `medium` or `high`; `off` adds `_nothink` to every output name |
| `LLM_TIMEOUT_S` | 300 (600 remote) | per-request timeout |
| `RESULTS_VERSION` | `v1` | which `data/results/` folder a run writes to |
| `WEB_SEARCH_BACKEND` | `duckduckgo` | fallback backend when `--web` is not given |

Then:

```bash
python -m rag --build-manifest             # resolve DOIs into data/evidence/sources.json
python -m rag --build-index --no-extract   # embed the committed passages -> data/index/
python -m rag --check-server
python -m rag --list                       # videos with usable transcripts
python -m rag Y6_jn0DzXHI                  # fact-check one video
python -m rag Y6_jn0DzXHI --web tavily --json out.json
```

`--build-index` should report 2488 passages. A different number means the
corpus changed and the documented parameter values were measured against a
different one. Which document covers which topic is declared by hand in
`rag/manifest.py`; that is editorial judgement and is not derived.

A verdict is `FAKE_NEWS`, `NOT_FAKE_NEWS`, `NO_EVIDENCE` or `ERROR`. The model
must say whether the evidence addresses the claim before it may give a verdict;
that ordering fixed the dominant failure, where absence of evidence was read as
contradiction. A flag also needs the speaker to be asserting the claim rather
than quoting or denying it. Web-sourced verdicts are suffixed `_WEB` and counted
separately, because that evidence has not been through the corpus's licence and
authorship screening.

### 4. Evaluate

```bash
python experiments/datasets.py                 # split sizes and a leakage check
python experiments/run_experiment.py --check   # preflight: server, index, labels, web, time estimate
```

The protocol for one model is `scripts/run_test_model.sh`: preflight, then for
each arm a 3-video smoke test and the full run, then the scores. The four
reported models:

```bash
unset LLM_SERVER; export LLAMA_MODEL=gpt-oss-20b LLM_CONSTRAINT=json_schema; scripts/run_test_model.sh
unset LLM_SERVER LLM_CONSTRAINT; export LLAMA_MODEL=gemma-4-12b;              scripts/run_test_model.sh
export LLM_SERVER=openrouter LLAMA_MODEL=qwen/qwen3.6-27b LLM_REASONING=off;  scripts/run_test_model.sh
export LLM_SERVER=openrouter LLAMA_MODEL=openai/gpt-oss-120b;                 scripts/run_test_model.sh
```

Local models run one at a time on port 8080. The script resumes an interrupted
run, and a run that is already complete is only re-scored. `WHOLE=1` runs the
whole-transcript protocol (v3) instead of the windowed one (v1).

Each arm is scored on its own, raw and with the overlap veto, and both arms are
scored together as the evidence veto: a parametric flag stands unless the
evidence arm cleared the same window. To re-score by hand:

```bash
python experiments/run_experiment.py --score --out data/results/v1/test_run_<model>_tavily.json
python experiments/run_experiment.py --score --overlap-veto --out data/results/v1/test_run_<model>_tavily.json
python experiments/score_verifier.py data/results/v1/test_run_<model>_parametric.json \
                                     data/results/v1/test_run_<model>_tavily.json --overlap-veto
```

A video is predicted fake when any of its windows is flagged. Results are
grouped by protocol in `data/results/` (see its README); every run records its
full configuration, git commit and timing in a `run` block, so a figure can
always be attributed to the settings that produced it.

The test set always runs the production prompts in `rag/llm.py`. Prompt
variants are compared on the dev set only (`--dataset dev --prompt NAME`).

### Reproducing the tables

Every reported figure is computed from result files that are in the
repository, so no model needs to run. The reported v1 files are, per model,
`data/results/v1/test_run_<model>_parametric.json` and `..._tavily.json`, with
`<model>` one of `gemma-4-12b`, `gpt-oss-20b`, `openai-gpt-oss-120b` and
`qwen-qwen3.6-27b` (the Qwen files carry a `_nothink` suffix). The v3 files are
the `_whole_tc` ones in `data/results/v3/`.

| table | script | input |
|---|---|---|
| per-arm precision, recall, F1 and accuracy, raw and with the overlap veto | `experiments/run_experiment.py --score [--overlap-veto] --out FILE` | one result file; writes `*_summary.txt` / `*_ov_summary.txt` beside it |
| verifier and evidence veto per model | `experiments/score_verifier.py PARAM RAG [--overlap-veto]` | the two arms of one model |
| window- and claim-level precision bounds | `experiments/window_metrics.py FILE...` | any v1 result files |
| off-claim flag audit | `experiments/offclaim_sample.py` with `data/labels/offclaim_key.json` | the v1 result files |
| reference classifiers (all-positive, date and provenance rules, TF-IDF + LR) | `experiments/baselines.py` | the eight v1 files above, read from `data/results/v1/` |
| whole-transcript protocol | `experiments/score_verifier.py` on the `_whole_tc` pair; read the `video-AND` row | `data/results/v3/` |
| dev-set prompt comparison | `experiments/score_prompt_ab.py data/results/v1/prompt_ab_dev.json` | the prompt A/B run |

For example, the four-model verifier table:

```bash
for m in gemma-4-12b gpt-oss-20b openai-gpt-oss-120b; do
  python experiments/score_verifier.py data/results/v1/test_run_${m}_parametric.json \
                                       data/results/v1/test_run_${m}_tavily.json --overlap-veto
done
python experiments/score_verifier.py data/results/v1/test_run_qwen-qwen3.6-27b_parametric_nothink.json \
                                     data/results/v1/test_run_qwen-qwen3.6-27b_tavily_nothink.json --overlap-veto
```

`data/results/README.md` says which files are reported and which are kept only
for the record. The thesis itself: *link to the deposited PDF goes here*.

## Design decisions

- **YouTube Data API, not scraping.** Scraping breaks with the markup, gets IPs
  blocked and violates the Terms of Service. The API is free within quota and
  citable.
- **Whisper on the audio, not captions.** Most of these videos have captions
  disabled. Coverage then depends only on a video being downloadable.
- **An LLM, not keyword matching.** The dataset holds misinformation and
  debunking of the same myths in near-identical vocabulary, and Whisper output
  is unpunctuated, so negation scope cannot be recovered by pattern matching.
- **Stance is the hard part.** Models key on topic and flag debunking content
  as misinformation. The prompt asks for stance explicitly and a `denying` or
  `neutral` stance overrides a flag; `docs/adjudication-experiments.md` records
  how far that holds.
- **Nothing is tuned on the test videos.** The 119 labelled v1 videos are held
  out; every threshold comes from separate dev collections (`docs/dev-sets.md`).
  A parameter gets a value only when the dev set discriminates between values,
  and otherwise the value that makes it inert.
- **Everything runs on the host** with the local backends. Only the two
  OpenRouter models send transcript text off the machine, and preflight warns
  on every remote run.

## Status

| | |
|---|---|
| Test set | 119 videos with usable transcripts, hand-labelled: 42 asserting something false, 77 not |
| Dev sets | 44 transcribed videos mirroring the v1 queries (tuning), 19 on narratives the corpus does not cover (web fallback only) |
| Corpus | 21 licence-cleared, DOI-resolved documents, 2,488 passages |
| Retrieval | exact cosine over MiniLM embeddings, 55-word chunks, topic routing with a confidence gate; BM25 and hybrid were tested and rejected |
| Models | Gemma 4 12B and gpt-oss-20b via llama.cpp; Qwen3.6-27B and gpt-oss-120b via OpenRouter |
| Protocols | v1 windowed (reported), v2 with the time check (tested, scored worse, dropped), v3 whole-transcript |

Open:

- The web fallback has no ground truth. Its verdicts are reported separately
  from the corpus figures.
- Three retrieval parameters are inert for want of evidence, not because they
  are known to be unnecessary; settling them needs windows whose evidence is
  genuinely absent from the corpus.
- The 55-word chunk size wins clearly only on specific claims; on broad ones
  chunk size barely matters.

## Data and ethics

The videos are public YouTube uploads found through the official Data API with
neutral topic queries; no creator was targeted, and the dataset contains
misinformation and debunking of the same myths alike. The transcripts are
machine-generated from the audio and are redistributed here so that the
experiment can be reproduced exactly: the windowed result files repeat that
text in 140-word windows, and every scoring script reads the transcripts. The
spoken content remains the creators' copyright and is not relicensed. Every
record keeps its video and channel id, so a creator can identify their material
and request its removal; `experiments/redact_text.py` strips the window text
from the result files and restores it from re-transcribed audio, so a copy
without transcripts stays scoreable.

The pages retrieved by the web fallback are not redistributed. Only an index of
queries, URLs and retrieval times is kept per run
(`data/results/<version>/web_cache_index_*.json`).

## Licence

Code is MIT (`LICENSE`). Labels are CC BY 4.0. The evidence passages keep the
licence of the document they were extracted from, three of which are
non-commercial, so the corpus as a whole is not relicensed. Transcripts are for
research use only. `LICENSE-DATA.md` lists the terms of every data file,
including the licence of each corpus document.

## Citation and releases

`CITATION.cff` holds the citation metadata GitHub shows under "Cite this
repository"; fill in the Zenodo DOI and the thesis details once they exist.
The commit matching the submitted thesis is tagged `v1.0`, and the thesis cites
that tag (and the Zenodo DOI minted from it). Later fixes become `v1.1` and so
on; `v1.0` is never moved.

```bash
git tag -a v1.0 -m "Version submitted with the thesis" <commit>
git push origin v1.0
```

## Documentation

| doc | covers |
|---|---|
| `docs/corpus-inclusion.md` | what gets into the reference corpus, and why one paper was withdrawn |
| `docs/chunking-experiments.md` | chunk size and stride |
| `docs/retrieval-methods.md` | dense versus BM25 versus hybrid |
| `docs/routing-experiments.md` | topic routing, the confidence gate and the rank bonus |
| `docs/adjudication-experiments.md` | the relevance gate, stance, and the verdict contract |
| `docs/prompt-experiments.md` | the prompt A/B on the dev set |
| `docs/web-search.md` | the web fallback: trigger, backends, caching |
| `docs/dev-sets.md` | how the dev collections were built and kept clean of the test set |
| `docs/model-backends.md` | llama.cpp, Ollama and OpenRouter, and the four-model comparison |
| `docs/evidence-as-verifier.md`, `docs/evidence-veto.md` | combining the parametric and evidence arms |
| `docs/code-cleanup.md` | the dead-code audit before the final experiment |

## Troubleshooting

| symptom | cause |
|---|---|
| `ConnectionError: cannot reach` | no server, or the wrong port: llama.cpp is expected on 8080, Ollama on 11434 |
| every verdict `ERROR` on gpt-oss | the grammar is fighting the reasoning trace; start `llama-server` with `--jinja` |
| `unrecognised verdict ''` | grammar off with a large evidence block; you are on Ollama, unset `LLM_SERVER` |
| every verdict `NO_EVIDENCE` | index empty or wrong corpus; `--build-index` should report 2488 |
| `Index is stale` | passages changed since the index was built; rebuild it |
| results from two models in one file | `LLAMA_MODEL` left at its default; set it per run |
| `OPENROUTER_API_KEY not set` / `HTTP 401` | key missing, wrong, or the account has no credit |
| run much slower than the estimate | no GPU offload; add `-ngl 99`. Reasoning models also spend tokens before answering |
