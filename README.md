# COVID-19 vaccine misinformation detection

A two-stage pipeline that builds a dataset of YouTube videos making claims about
COVID-19 vaccines, and transcribes their spoken audio — the input for a downstream
fact-checking stage.

## Intent

The pipeline **gathers candidates; it does not judge them.** Collection casts a
wide, neutral net over videos that discuss COVID-19 vaccines, and transcription
turns their audio into text. Deciding which claims are false is the job of a later
stage (`rag/`), working from the transcripts this produces.

That separation is deliberate, and it is what makes the method defensible: you
assemble a dataset and then classify it, rather than pre-judging creators or
cherry-picking videos to fit a conclusion.

Two other choices follow from the same reasoning:

- **The official YouTube Data API v3, never HTML scraping.** Scraping violates
  YouTube's Terms of Service, breaks whenever the markup changes, and gets IPs
  blocked. The API is free within a 10,000 unit/day quota, returns structured data,
  and is citable in a methods section.
- **Whisper on the audio, not YouTube captions.** Most videos here have captions
  disabled, and caption scraping is increasingly blocked. Transcribing the audio
  means coverage depends only on a video being downloadable.

## Pipeline

```
collect  ──►  covid_vaccine_videos.json  ──►  transcribe
                                                   │
                       covid_vaccine_videos_transcribed.json
                                                   │
                                                   ▼
                    rag  ◄──  data/index/  ◄──  data/evidence/
                     │         (vectors)        (vetted PDFs)
                     ▼
         verdict + false phrase + debunking evidence
```

### `collect/` — build the dataset

Searches the Data API for the topic queries in `collect/config.py`, optionally adds
a curated list of links, then completes and filters the results.

```bash
python -m collect                                              # search queries
python -m collect --links disinformation_dozen.txt              # search + link list
python -m collect --links disinformation_dozen.txt --no-search  # link list only
```

Every candidate is completed with a `videos.list` pass, which returns untruncated
descriptions plus statistics and duration — `search.list` alone gives none of those.
It costs 1 quota unit per 50 videos, against the 100 each search already spends.

Filters applied before saving:

| Filter | Rule |
|---|---|
| Duplicates | First occurrence wins, preserving original query provenance |
| Over 1 hour | Dropped; unknown durations are kept and counted separately |

Language is **not** filtered here. YouTube's `defaultAudioLanguage` is self-reported
by uploaders and often missing or wrong, so filtering on it would discard real
English videos. `transcribe` decides from the audio instead, which is authoritative.

### `transcribe/` — add the spoken text

Downloads each video's audio, transcribes it with faster-whisper, and writes a copy
of the dataset with transcripts appended.

```bash
python -m transcribe covid_vaccine_videos     # or any dataset name
python -m transcribe --list                   # show available datasets
python -m transcribe covid_vaccine_videos --limit 5
python -m transcribe covid_vaccine_videos --summary-only   # rebuild output, no work
```

Non-English audio is detected and dropped here, before inference — so a Portuguese
video costs a detection pass, not a full transcription.

**This takes hours.** Roughly 6x realtime with the `small` model on an M2 Pro.
Progress is checkpointed after every video, so an interrupted run resumes where it
stopped; audio is deleted immediately after each video, keeping disk use flat.

### `rag/` — flag false claims against evidence

Retrieves evidence from a vectorized corpus and asks a **locally served** model
whether a transcript claim asserts something that evidence contradicts. Each flag
returns the false phrase and the passage that debunks it.

#### Running the model server

Generation runs locally. Either server works — they expose the same
OpenAI-compatible `/v1/chat/completions` endpoint — but the defaults in
`rag/config.py` point at Ollama:

**Ollama** (what this project uses):

```bash
brew install ollama           # once
ollama serve                  # leave running in its own terminal
ollama pull qwen2.5:14b       # once, ~9 GB
```

Keep `ollama serve` in a terminal you can see. Started in the background it
detaches from its parent, and then the only way to stop it is by PID.

```bash
ollama stop qwen2.5:14b   # unload the model, free ~9 GB, leave the server up
pkill ollama              # stop the server entirely
ollama ps                 # what is currently loaded
```

The model also unloads itself after about five minutes idle, so doing nothing is
a valid way to reclaim the memory.

**llama.cpp** (alternative):

```bash
llama-server -m model.gguf -c 8192 --port 8080
```

If you use it, set `LLAMA_URL = "http://localhost:8080"` and `USE_GRAMMAR = True`
in `rag/config.py`. The grammar is worth having: llama.cpp enforces it during
sampling, so a malformed verdict cannot be generated. Ollama ignores `grammar`
and only guarantees valid JSON of any shape, which is why a large evidence block
can make it drop the required fields.

#### Running the pipeline

```bash
python -m rag --build-index        # extract + embed data/evidence/pdf/ -> data/index/
python -m rag --check-server       # confirm the server is reachable
python -m rag --list               # videos with usable transcripts
python -m rag Y6_jn0DzXHI          # fact-check one video
python -m rag Y6_jn0DzXHI --json out.json
```

Expect roughly 15 s per claim window on `qwen2.5:14b`, so a 1,700-word
transcript (~24 windows) takes about 6 minutes.

Evidence is built from the PDFs in `data/evidence/pdf/` (gitignored — only
licence-cleared documents go there). `python -m rag --build-manifest` resolves
each one's DOI and publisher via Crossref into `data/evidence/sources.json`;
`--build-index` then extracts passages and embeds them. Which document covers
which topic is declared by hand in `rag/manifest.py` — that is editorial
judgement and cannot be derived.

Retrieval is exact cosine similarity over a persisted matrix (the corpus is a few
thousand passages, so there is no approximate-NN index to go stale). Claims are
routed to the topics they resemble before ranking, with a confidence gate and a
fallthrough to broader scopes; see `docs/routing-experiments.md`.

A verdict is one of `FAKE_NEWS`, `NOT_FAKE_NEWS`, `NO_EVIDENCE` or `ERROR`, and
the model must answer `addresses` — does the evidence discuss this claim at all —
*before* it may give a verdict. That ordering is the fix for the dominant failure
mode, where the model treated absence of evidence as contradiction
(`docs/adjudication-experiments.md`).

**Why an LLM and not keyword matching:** the dataset contains both misinformation and
debunking of the *same* myths, often in near-identical vocabulary. Keyword retrieval
finds the topic but cannot tell "vaccines cause infertility" from "vaccines do not
cause infertility" — Whisper output is unpunctuated, so negation scope is not
recoverable by pattern matching. Deciding stance is the model's job.

**Stance is the bottleneck, and model size matters.** `tests/stance_eval.py` scores a
model on minimal unambiguous pairs — one sentence each, half asserting a myth and
half denying the same myth:

```bash
python tests/stance_eval.py              # the configured model
python tests/stance_eval.py qwen2.5:14b  # compare another
```

| Model | Clean pairs | Whole videos |
|---|---|---|
| `qwen2.5:7b` | 9/12 (denials 3/6) | — |
| `qwen2.5:14b` | 11/12 (denials 5/6) | **2/5** |
| MNLI cross-encoder (70 MB) | 7/8 (denials 4/4) | — |

The failure is one-directional: models key on *topic*, so **debunking content gets
flagged as misinformation** — the worst possible error here, since it inverts the
result on the fact-checking videos this dataset is full of.

Two findings worth carrying forward:

A 70 MB trained entailment classifier beats a 7B generative model, because
entailment is a classification task with no instruction for safety training to
override. Asked about cats and printers the 7B model reasons about stance correctly;
asked about vaccines it answers "is this true?" instead, and reports the statement
is false. Even reframed as pure grammatical polarity, it labels "The mRNA vaccine
permanently alters your DNA" — which contains no negation at all — as negative.

**Clean-sentence scores overstate real performance.** `qwen2.5:14b` goes from 11/12
on minimal pairs to 2/5 on whole videos, flagging a journalist's question ("Will it
cause females to be sterile?") and a doctor's description ("a belief that they were
going to implant microchips"). Real speech quotes and describes myths constantly;
constructed pairs contain no reported speech, so they miss the dominant failure mode.

Treat retrieval as sound and the verdict as a *candidate* needing review. The
pipeline guards what it can — a reported stance of `denying` or `neutral` overrides
a FAKE_NEWS verdict — but that only helps when the model reports stance honestly,
and `qwen2.5:14b` reports `asserting` on nearly everything.

Everything runs on your machine: embeddings via sentence-transformers, generation via
llama.cpp. No transcript leaves the host.

### `tests/` — evaluation and the dev/test split

Nothing here is tuned on the test videos. `tests/datasets.py` is the authority on
which videos belong to which split:

```bash
python tests/datasets.py          # split sizes and a leakage check
```

Collecting and preparing a development set:

```bash
python -m collect --name covid_vaccine_videos_dev2 \
    --exclude data/youtube/exclude_dev2.json     # keeps test + earlier dev out

# two groups: mirror repeats the v1 queries (tuning), uncovered targets
# narratives the corpus does not cover (web-fallback testing only)
python tests/select_dev.py --group mirror    --n 50 --write
python tests/select_dev.py --group uncovered --n 21 --write

python -m transcribe covid_vaccine_videos_dev2_mirror_selected
python -m transcribe covid_vaccine_videos_dev2_uncovered_selected
```

`--exclude` matters: the dev queries overlap the original ones in subject matter,
so without it YouTube returns the same videos and test material leaks into tuning.
Dev transcription is capped at 12 minutes per video (`TRANSCRIBE_MAX_S`), which
leaves most of the set untouched and cuts inference by about a fifth; pass
`--max-transcribe 0` for whole videos, which is how the test set was transcribed.

The sweeps:

```bash
python tests/chunk_eval.py --make-labels \
    --dataset covid_vaccine_videos_dev2_mirror_selected \
    --labels tests/chunk_eval_labels_dev2.json    # propose windows to label

python tests/chunk_eval.py --labels tests/chunk_eval_labels_dev2.json --limit 100
python tests/route_eval.py  --labels tests/chunk_eval_labels_dev2.json
python tests/retrieval_compare.py --tune          # dense vs BM25 vs hybrid
python tests/gate_probe.py                        # relevance gate, needs the server
```

A proposed window needs two things filled in by hand: `topic` (which corpus topic
a claim in it belongs to, or `general`) and `relevant_evidence` (verbatim phrases
from the passage that answers it). Phrases rather than passage ids, so labels
survive re-chunking. Check that every phrase appears verbatim in the corpus before
scoring — one that does not is a permanent miss and silently depresses recall.

## Running the full experiment

The final run fact-checks the 119 labelled test videos and scores the result
against `data/labels/ground_truth.json`. No parameter was chosen by looking at
these videos — every threshold was derived on separate development collections
(`docs/dev-sets.md`), which is what makes the figure meaningful.

Four models are compared across two deployment kinds
(`docs/model-backends.md`):

| | model | served by | `LLAMA_MODEL` |
|---|---|---|---|
| local | Gemma 4 12B | llama.cpp (default) | your GGUF / Ollama tag |
| local | gpt-oss-20b | llama.cpp (default) | your GGUF / Ollama tag |
| remote | Qwen3.6-27B | OpenRouter | `qwen/qwen3.6-27b` |
| remote | gpt-oss-120b | OpenRouter | `openai/gpt-oss-120b` |

`LLM_SERVER` selects the backend — `llamacpp` (default), `ollama`, or
`openrouter`. Each enforces the six-field verdict contract differently (GBNF
grammar, `json_object`, `json_schema`), so **figures from different backends are
not interchangeable**; the run metadata records which was used, and the default
results filename includes the model name so runs cannot overwrite one another.

On the remote path transcript text leaves the machine, which the local path
avoids. Preflight prints a warning on every remote run.

### 1. Install

```bash
git clone <this repo> && cd TCC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # or apt install ffmpeg
```

`ffmpeg` is only needed if you re-transcribe. The experiment uses the committed
transcripts, so a run-only machine can skip it.

### 2. Build the evidence index

The source PDFs are gitignored (licence terms), but the extracted passages are
committed, so the index rebuilds from those alone:

```bash
python -m rag --build-index --no-extract
```

`--no-extract` skips the PDF reading step. Without it the command still works —
missing PDFs are skipped rather than fatal — but it prints a list of every
missing file, which on a clone is all 21 of them and looks like an error.

Expect `Indexed 2488 evidence passage(s)`. If it reports a different number, the corpus
has changed and the documented parameter values were measured against a
different one.

### 3. Get the model

```bash
# a Q4_K_M GGUF of Qwen2.5-14B-Instruct, ~9 GB
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF \
    qwen2.5-14b-instruct-q4_k_m.gguf --local-dir ./models
```

Any instruction-tuned model served over an OpenAI-compatible endpoint works. The
figures in `docs/` are for `qwen2.5:14b` at Q4_K_M; a different model needs the
stance and gate behaviour re-checked, since those are the failure modes the
prompt is built around (`docs/adjudication-experiments.md`).

### 4. Start llama-server

```bash
llama-server \
    -m ./models/qwen2.5-14b-instruct-q4_k_m.gguf \
    -c 8192 \
    --port 8080 \
    --host 127.0.0.1
```

Keep it in its own terminal. `-c 8192` matters: the prompt carries up to ~11
evidence passages, and a smaller context silently truncates the evidence block.

**On GPU**, add `-ngl 99` to offload all layers. Throughput drives the total run
time. Measured on CPU (M2 Pro, qwen2.5:14b Q4_K_M) over 42-window smoke tests:

| run | s/window | 1,588 windows |
|---|---|---|
| corpus only | 21.0 | ~9.3 h |
| with web fallback | 28.4 | ~12.5 h |

The web fallback is slower because the ~40% of windows that trigger it pay a
search round-trip plus a second adjudication. Preflight prints the estimate for
whichever backend you select. GPU offload is typically an order of magnitude
faster.

### 5. Point the pipeline at it

```bash
python -m rag --check-server
```

llama.cpp is the default, so this already targets `localhost:8080` with the GBNF
grammar enabled and Ollama's JSON mode off. Override individually if needed:

```bash
export LLAMA_URL=http://10.0.0.5:8080     # a server on another host
export LLAMA_MODEL=whatever-you-loaded    # llama-server ignores this
export LLM_TIMEOUT_S=600                  # slower hardware
```

**Why the grammar matters here.** llama.cpp enforces a GBNF grammar during
sampling, so a response missing a required field cannot be generated. Ollama only
guarantees valid JSON of any shape, and at larger evidence blocks it was observed
dropping the required keys entirely (`docs/adjudication-experiments.md`). The
llama.cpp path is the more reliable one, and it is the default.

**Every figure in `docs/` was measured on Ollama**, on a development machine
where llama-server was not set up. The grammar path is therefore unexercised,
which is why step 7 starts with a smoke test rather than the full run.

### 6. Preflight

```bash
python tests/run_experiment.py --check
```

Verifies the server, the index size, the test-set count and the web backend, and
estimates the run time. It exits non-zero and says what is wrong rather than
starting a multi-hour run that will fail.

Expect:

```
  model server   : OK   http://localhost:8080 — models: ...
  grammar mode   : GBNF (llama.cpp)
  evidence index : OK   2488 passages
  test videos    : 119 labelled with usable transcripts
  claim windows  : ~1588
```

### 7. Smoke-test, then run

**The planned final run is llama.cpp + Tavily.** Both differ from the machine
the development measurements were taken on (Ollama, JSON mode, no web), so run
the smoke test here first — it is the only check that the grammar path works.

```bash
# 1. smoke-test the chain, with the backend you will actually use
python tests/run_experiment.py --limit 3 --web tavily

# 2. clear the cache so the real run fetches fresh evidence
#    (the run itself also accepts --clear-cache)
mkdir -p data/results
python tests/run_experiment.py --web tavily --clear-cache 2>&1 \
    | tee data/results/test_run_tavily.log
```

For a corpus-only run (no web evidence, cleaner headline figure) drop `--web`:

```bash
python tests/run_experiment.py 2>&1 | tee data/results/test_run_none.log
```

**Budget.** At the measured trigger rate (35-40% of windows) the 119-video run
costs roughly 560-650 Tavily credits of the free tier's 1,000, so it fits in one
pass with margin. A resumed run costs less: repeated queries are served from the
cache for 0 credits.

**Expect ~9.3 h corpus-only, ~12.5 h with the fallback** at the rates measured
on this project's CPU. A GPU with `-ngl 99` is typically an order of magnitude
faster, so check the preflight estimate on the actual machine rather than
trusting these numbers.

`tee` matters because the per-video progress lines are the only record of the
order videos were processed in and of any transient server error that a resumed
run would otherwise hide.

The smoke test takes 12-17 minutes on CPU (longer with `--web`) and prints a
scored confusion matrix for those 3 videos, so it verifies the whole chain —
adjudication, the flag/label join, and scoring — not just that the server
answers.

Its confusion matrix is **not** a quality signal: all three videos are labelled
fake, so nothing there tests false positives. Treat a 1.000 as "the chain runs",
and read the flagged phrases themselves — the web-fallback smoke test produced a
true positive on an irrelevant phrase, which the video-level metric could not
show (docs/web-search.md).

Checkpointed after every video, so an interrupted run resumes where it stopped
and nothing is recomputed.

Each run leaves three files in `data/results/`, all of them committable:

| file | holds |
|---|---|
| `test_run_<web>.json` | `run` metadata + every window of every video |
| `test_run_<web>_summary.txt` | the scored confusion matrix and error lists |
| `test_run_<web>.log` | the console narrative, if you used `tee` above |

The `run` block records the configuration that produced the results — model,
every parameter in `rag/config.py`, grammar mode, corpus size, web backend, git
commit, timestamp and wall-clock minutes. **Diagnosis later depends on this**: a
confusion matrix with no record of `ROUTE_GAMMA` or `CHUNK_WORDS` cannot be
attributed to a configuration, and the parameters have changed several times
during development. If the configuration changes between an interrupted run and
its resume, the run prints a warning naming each changed value rather than
silently mixing them.

To include the web fallback for claims the corpus cannot address:

```bash
python tests/run_experiment.py --web duckduckgo
python tests/run_experiment.py --web tavily     # needs TAVILY_API_KEY
```

**OpenRouter setup** (for the two remote models). Create a key at
openrouter.ai and put it in the environment or the project `.env` — both are
read:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...

LLM_SERVER=openrouter LLAMA_MODEL=qwen/qwen3.6-27b \
    python tests/run_experiment.py --check
```

Preflight calls the authenticated `/key` endpoint and reports remaining credit,
and fails if the key is rejected or the model id is not offered — `/models` is
public, so reaching it proves nothing about the key. Both failures were verified
deliberately. The whole four-model comparison costs a few dollars
(`docs/model-backends.md`); a remote run is limited by rate limits rather than
the ~9.3 h of local CPU.

**Tavily setup.** Sign up at tavily.com and copy the key from the dashboard; the
free Researcher tier is 1,000 credits a month with no card. Put it in the
environment or in the project `.env` — both are read:

```bash
export TAVILY_API_KEY=tvly-...
```

Nothing to install: the REST endpoint is called with `requests`. At the measured
trigger rate (~35% of windows) a 24-window video costs about 8 credits, so the
free tier covers roughly 125 videos — enough for the 119-video test set in one
pass.

Preflight probes whichever backend is selected and **fails if it returns
nothing**, so a missing key is caught before the run rather than producing a
silently corpus-only result:

```
  web fallback   : duckduckgo
                   probe OK -> https://www.cdc.gov/covid/hcp/vaccine-considerations
```

Web-sourced verdicts are labelled `FAKE_NEWS_WEB` / `NOT_FAKE_NEWS_WEB` and
counted separately, because the evidence behind them has not been through the
corpus's licence, DOI and authorship screening.

### 8. Score

```bash
python tests/run_experiment.py --score                    # corpus-only run
python tests/run_experiment.py --score --web duckduckgo   # the web run
```

Scoring runs automatically at the end of a run; `--score` re-scores saved
results without re-running the model, and rewrites the summary file.

Prints (and writes to `..._summary.txt`) the run's configuration line, the
confusion matrix, precision, recall, F1 and accuracy, plus every false
negative with its narrative and confidence and every false positive with its flag
count — which is what you need to read the result rather than just report it.

**A video is predicted fake news if any of its windows is flagged.** The labels
are per video and the pipeline decides per window, so the join rule is stated in
`tests/run_experiment.py` rather than chosen after seeing the numbers.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `ConnectionError: cannot reach` | server not started, or it is on the other port — the default expects llama.cpp on 8080, `LLM_SERVER=ollama` expects 11434 |
| `unrecognised verdict '' in ['REJECT']` | grammar disabled with a large evidence block — you are on Ollama; unset `LLM_SERVER` to use llama.cpp |
| `Index is stale: N passages no longer exist` | passages changed since the index was built; re-run `--build-index` |
| every verdict `NO_EVIDENCE` | index empty or wrong corpus — check `--build-index` reported 2488 |
| run much slower than estimated | no GPU offload; add `-ngl 99` to `llama-server` |

## Layout

```
collect/        config  models  filters  data  api  collector  main
transcribe/     config  audio  whisper  data  transcriber  main
rag/            config  corpus  manifest  router  index  lexical  llm
                websearch  pipeline  main
tests/          evaluation harnesses and the dev/test split
docs/           what was measured, and what it decided
data/youtube/   datasets, checkpoints, link lists
data/evidence/  pdf/ (gitignored), passages/ (committed), sources.json
data/index/     persisted evidence vectors (rebuildable)
data/labels/    hand-annotated ground truth for the test videos
```

All three packages follow the same layering, grouped by external dependency so a
failure stays contained and each layer works without the ones above it:

| Layer | Depends on |
|---|---|
| `config`, `models` | nothing |
| `filters` | pure logic, no I/O |
| `data` | stdlib + pandas; every write is atomic |
| `api` / `audio`, `whisper` / `index`, `llm` | the external boundary — network, yt-dlp, ffmpeg, Whisper, llama.cpp |
| `collector` / `transcriber` / `pipeline` | the layers above; owns the order of work |
| `main` | argument parsing and exit codes |

Dependencies point one way only. Public names are re-exported lazily (PEP 562), so
importing a submodule does not pull in the heavy dependencies of its siblings.

## Setup

Requires Python 3.10+, `ffmpeg`, and a YouTube Data API key.

```bash
brew install ffmpeg
pip install -r requirements.txt
```

Put the key in `.env` at the project root (free — create one in the
[Google Cloud Console](https://console.cloud.google.com/), enable "YouTube Data
API v3", then create an API key):

```
YOUTUBE_API_KEY=...
```

## Data files

| File | What it is |
|---|---|
| `covid_vaccine_videos.json` / `.csv` | the v1 dataset — 143 videos, metadata only |
| `covid_vaccine_videos_transcribed.json` / `.csv` | the same records plus `transcript`, `transcript_status`, `detected_language`, `audio_duration_s` |
| `covid_vaccine_videos_dev2_mirror_more.json` | mirror candidate pool — 135 videos from the six v1 queries |
| `covid_vaccine_videos_dev2_mirror_selected.json` | 50 of those; the tuning set |
| `covid_vaccine_videos_dev2_uncovered.json` | 158 videos on narratives the corpus does not cover |
| `covid_vaccine_videos_dev2_uncovered_selected.json` | 21 of those; web-fallback testing only |
| `exclude_dev2.json` | 376 ids used by the test set and the retired first dev set |
| `transcripts_checkpoint.json` | per-video progress; what makes a run resumable |
| `data/labels/ground_truth.json` / `.csv` | hand-annotated true/false labels for the 124 v1 videos |
| `tests/chunk_eval_labels.json` | v1 windows — topic + evidence phrases, kept because they are the only labelled windows for the test videos |
| `tests/chunk_eval_labels_dev2.json` | dev windows; what the sweeps tune on |
| `disinformation_dozen.txt` | curated link list, one URL or video id per line |

Collection and transcription each overwrite their own output in place; the
checkpoint is what accumulates, so the transcribed file is always a fresh
projection of it.

## Status

**Data.** 143 videos collected, 124 with usable transcripts, all 124 hand-annotated
for true/false (`data/labels/ground_truth.json`): 43 flagged as asserting something
false, 81 not. Two further collections serve as development sets — see
docs/dev-sets.md.

**Splits.** `tests/datasets.py` reports them; collection and selection are in
docs/dev-sets.md. The 119 usable v1 videos are the held-out test set and have
never been used to choose a setting. Tuning uses a separate collection whose
queries mirror the v1 ones, so figures transfer; a third group targets narratives
the corpus does not cover and is used only to exercise the web fallback. The
separation exists because 45 of the v1 videos had been used to pick seven
hyperparameters, so any figure over them partly measured that tuning.

**Evidence corpus.** 21 documents, 2,488 passages, all licence-cleared and
DOI-resolved. One paper was withdrawn on authorship grounds
(`docs/corpus-inclusion.md`), which is now an explicit inclusion criterion.

**Retrieval.** Every parameter re-derived on development data
(`docs/routing-experiments.md`):

| Parameter | Value | Basis |
|---|---|---|
| `CHUNK_WORDS` / `CHUNK_STRIDE` | 55 / 35 | never beaten; wins clearly on specific claims |
| `ROUTE_GAMMA` | 0.30 | recall collapses by 0.60; general claims unaffected |
| `TOPIC_RANK_BONUS` | 0.03 | mid-plateau; breaks vocabulary ties toward the topic |
| `SCOPE_ANSWERED_SIMILARITY` | = `MIN_SIMILARITY` | inert — no dev basis |
| `RELATIVE_CUTOFF` | 0 | inert — no dev basis |
| `FLAG_MIN_SIMILARITY` | 0 | every floor removed only correct flags |

The rule applied: a parameter gets a value when the development set
discriminates between values, and otherwise the value that makes it inert.
Three did not discriminate and were disabled rather than guessed.

General documents are carried by every routing tier rather than withheld from
narrow scopes, with vocabulary dilution handled by the rank bonus. BM25 and
hybrid fusion were tested and rejected — the queries are 140-word transcript
windows, the shape BM25 handles worst (`docs/retrieval-methods.md`).

**Adjudication.** The relevance gate fixed the dominant failure, where the model
treated absence of evidence as contradiction: a graphene-oxide video went from 9 of
11 windows flagged to 0, with the model reaching NO_EVIDENCE itself rather than
being overridden.

**Web fallback.** Triggers on the adjudicated verdict — NO_EVIDENCE, or
`addresses: "no"` — plus an asserting stance, not on an empty retrieval. The
earlier condition was close to unreachable: over 2,488 passages sharing a vaccine
vocabulary something almost always clears the floor, so only 1.8% of windows
retrieved nothing even in a collection made for uncovered narratives
(`docs/web-search.md`).

### Open

- **No batch run over the full dataset, and no precision figure for the
  verdicts.** The labels exist and the test set is clean; the end-to-end run
  against it has not been made. This is the next step.
- **Three parameters are inert for want of evidence**, not because they are
  known to be unnecessary. `SCOPE_ANSWERED_SIMILARITY` and `RELATIVE_CUTOFF`
  govern what happens when a narrow scope fails, and the development sets rarely
  put the system in that state — 1.8% of windows. Settling them needs windows
  whose evidence is genuinely absent from the corpus.
- **The web fallback has no ground truth.** 13 of the 43 flagged test videos
  concern narratives the corpus cannot answer, so the fallback matters for them,
  but its verdicts are unlabelled and reported separately from the corpus
  figures.
- **Chunk size rests on weaker evidence than it did.** 55w is never beaten, but
  on broad general-claim windows the chunk size barely matters; the clear win is
  only on specific claims (`docs/chunking-experiments.md`).

The two notebooks in the root (`video_collection.ipynb`, `transcribe.ipynb`) are the
original exploratory versions, superseded by these packages.
