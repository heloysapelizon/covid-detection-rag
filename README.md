# COVID-19 vaccine misinformation detection

A two-stage pipeline that builds a dataset of YouTube videos making claims about
COVID-19 vaccines, and transcribes their spoken audio — the input for a downstream
fact-checking stage.

## Intent

The pipeline **gathers candidates; it does not judge them.** Collection casts a
wide, neutral net over videos that discuss COVID-19 vaccines, and transcription
turns their audio into text. Deciding which claims are false is the job of a later
stage (`rag.ipynb`), working from the transcripts this produces.

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
                     │         (vectors)        (CDC passages)
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

Start a model server, then build the index once:

```bash
llama-server -m model.gguf -c 8192 --port 8080   # in another terminal

python -m rag --build-index        # embed data/evidence/ -> data/index/
python -m rag --check-server       # confirm the server is reachable
python -m rag --list               # videos with usable transcripts
python -m rag Y6_jn0DzXHI          # fact-check one video
python -m rag Y6_jn0DzXHI --json out.json
```

Evidence lives in `data/evidence/*.json` as passages carrying the myth they address,
the authority's verdict, and a citable source. Add a file there and re-run
`--build-index` to extend the corpus — nothing else changes.

Retrieval is exact cosine similarity over a persisted matrix (the corpus is small,
so there is no approximate-NN index to go stale). The model is constrained by a GBNF
grammar to emit only `{verdict, claim, evidence_id, reason}`, so the verdict is
parsed rather than scraped, and `verdict` is one of `FAKE_NEWS`, `NOT_FAKE_NEWS`, or
`NO_EVIDENCE`.

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

## Layout

```
collect/        config  models  filters  data  api  collector  main
transcribe/     config  audio  whisper  data  transcriber  main
rag/            config  index  llm  pipeline  main
tests/          exploratory scripts, not part of the experiment
data/youtube/   datasets, checkpoints, link lists
data/evidence/  authority passages to check claims against
data/index/     persisted evidence vectors (rebuildable)
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
| `covid_vaccine_videos.json` / `.csv` | the dataset — 143 videos, metadata only |
| `covid_vaccine_videos_transcribed.json` / `.csv` | the same records plus `transcript`, `transcript_status`, `detected_language`, `audio_duration_s` |
| `transcripts_checkpoint.json` | per-video progress; what makes a run resumable |
| `disinformation_dozen.txt` | curated link list, one URL or video id per line |

Collection and transcription each overwrite their own output in place; the
checkpoint is what accumulates, so the transcribed file is always a fresh
projection of it.

## Status

143 videos collected. Transcription is partway through: 10 videos processed, 9 with
usable transcripts and 1 correctly excluded as non-English.

The two notebooks in the root (`video_collection.ipynb`, `transcribe.ipynb`) are the
original exploratory versions, superseded by these packages.
