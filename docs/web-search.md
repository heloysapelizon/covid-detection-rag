# Web-search fallback

Why the pipeline falls back to the web, which backend it uses and why, and what
was measured. Implementation: `rag/websearch.py`, `rag/llm.build_query`, wired in
`rag/pipeline.check_transcript(web_fetch_fn=...)`.

The trigger and provenance design are in docs/adjudication-experiments.md; this
file covers the backend choice and the query problem.

## Why a fallback at all

The reference corpus is deliberately small and curated
(docs/corpus-inclusion.md): 21 documents, each licence-vetted, DOI-resolved and
authorship-screened. Some narratives have no evidence in it at all — graphene
oxide, microchips, 5G — so retrieval returns nothing and the verdict is
NO_EVIDENCE. Honest, but the system then has no opinion on exactly the claims
most in need of one.

## Backend choice

| option | cost | key | card | reproducibility |
|---|---|---|---|---|
| **DuckDuckGo (`ddgs`)** | free | no | no | snapshot (`retrieved_at`) |
| SearXNG, self-hosted | free | no | no | best — you control the instance |
| Brave | $5/1k, ~1k free/mo | yes | yes | snapshot |
| Serper | ~$0.30-1/1k | yes | yes | snapshot |
| Tavily | $8/1k, 1k free/mo | yes | yes | snapshot |
| Exa | ~$7/1k | yes | yes | snapshot |

**DuckDuckGo, via `ddgs==9.16.1`.** Three reasons:

1. **Citable.** It is the backend used by Tian et al., "Web Retrieval Agents for
   Evidence-Based Misinformation Detection" (arXiv:2409.00009), the one study
   measuring web retrieval for an offline LLM — so the choice follows published
   practice rather than convenience.
2. **No key, no card, no budget.** The trigger fires on roughly 2 windows in 81,
   so the volume is tens of queries. A paid API buys nothing at that scale.
3. **No infrastructure.** SearXNG is better for reproducibility and is what
   FactSearch (ACL 2026 demo) chose, but it needs Docker plus `settings.yml`,
   and a practitioner report found it required disabling the limiter and
   spoofing `X-Forwarded-For` before answering automated requests — with only
   DuckDuckGo responding reliably behind it. That is a more complex route to the
   same backend.

Brave's free tier ended in February 2026 and is now ~1,000 queries/month of
credit; its own pricing page simultaneously advertises "$5 in free" and "up to
$20 in free calls", so the figure is unclear. Third-party comparisons disagreed
with each other on every provider's free tier, so any paid choice should be
verified on the provider's own page.

### What the literature says to expect

Tian et al. measured macro-F1 change from adding web search:

```
  GPT-4-0613     +20.5%
  GPT-4-0125     +12.8%
  Claude 3 Haiku  +7.2%
  Vicuna-13b      -0.9%     <- the only open local model tested
```

**Vicuna-13b got slightly worse.** That is the model closest in class to this
project's `qwen2.5:14b`, so a gain here is not guaranteed and should be measured
rather than assumed. The paper also found retrieval hurt on claims where the
context was already in the model's memory, "sometimes introducing unnecessary
noise" — which argues for a narrow trigger rather than searching every window,
and is consistent with the corpus-empty-plus-asserting gate this pipeline uses.
More encouragingly, calibration improved with retrieval (Brier score and ECE both
fell) and retrieved source quality averaged 0.76-0.80.

### Reproducibility, and how the field handles it

A search API is not a reproducible corpus. Published approaches:

- **self-hosted engine** — FactSearch runs a local SearXNG instance
- **external archiving** — one study saved its top-10 Google results to the
  Wayback Machine
- **date-pinning** — another pinned a snapshot date (October 2024)

This pipeline date-pins: every web row carries `source_url` and `retrieved_at`,
and nothing is cached (see docs/adjudication-experiments.md for why a cache would
make a verdict depend on when a claim first entered the pipeline). Wayback
archiving is the available upgrade and does not reintroduce that problem, since
it is write-only provenance rather than a read cache.

## The query problem

**A claim window cannot be used as a search query.** The first live run returned
zero results for every window. The cause was not rate limiting:

```
  full 140-word window (776 chars)  ->  DDGSException: No results found
  first 12 words of the same window ->  3 results
  first 20 words                    ->  DDGSException: No results found
  first 30 words                    ->  3 results
```

Truncation is not a fix — the length/success relationship is not monotonic,
because what matters is whether the text reads as a query, not how long it is.

`rag/llm.build_query` asks the local model for the query a fact-checker would
type, capped at 12 words, falling back to truncation if the call fails. Measured
on the two windows that trigger a search:

```
  window: "2,000 studies have been done to show the effects of 5G on our..."
  query : "effects of 5G frequency on human health studies"
  -> 4 results, ALL authority:
       pmc.ncbi.nlm.nih.gov       nature.com
       ec.europa.eu               pmc.ncbi.nlm.nih.gov

  window: "is why 95 percent of people lose all their money ... london real"
  query : "95 percent of people lose all their money"
  -> 4 results, NONE authority (day-trading blogs, Reddit)
```

The second is the known wasted search on a sponsor read
(docs/adjudication-experiments.md). Note it is *visibly* wasted: zero authority
results is itself a signal, recorded in the output.

## Authority ranking

General search with authority *preference*, not an allowlist — the narratives
reaching this fallback are precisely the ones WHO and CDC have not published on,
so an allowlist would return nothing exactly when the fallback is needed.

`AUTHORITY_DOMAINS` was **revised against observed results**. The first version
was written from assumption and matched only 1 of the 10 domains DuckDuckGo
actually returns for these narratives — it would have classed `itu.int` and
`news.un.org` as non-authority, which is most of what comes back for a 5G claim.
It now covers health and standards bodies, peer-reviewed publishers, `.edu` and
`.ac.uk`, and dedicated fact-checkers including Brazilian ones.

Matching is by domain suffix on dot boundaries, so `pubmed.ncbi.nlm.nih.gov`
matches `nih.gov` while `notnih.gov.evil.com` does not. Verified on 9 cases
including that spoofing attempt.

**Authority never filters.** `rank_by_authority` is a stable sort that returns
every input — verified: 4 results in, 4 out, and an all-non-authority set passes
through intact. A non-authority page ranks lower but is still used as evidence.
The sponsor-read search returned 4 zero-authority results and all 4 were given to
the adjudicator.

This is sound because the verdict labels carry the provenance: a reader seeing
`FAKE_NEWS_WEB` knows the evidence did not come from the vetted corpus. Dropping
non-authority results would instead mean returning nothing for exactly the
narratives this fallback exists to serve, since fringe claims are discussed on
fringe sites before authorities address them.

For that reasoning to hold, the provenance has to actually be visible, which
required two fixes:

- **Every web-sourced window is now named in the report, not just the flags.** A
  `NOT_FAKE_NEWS_WEB` previously appeared only as a count in the `verdicts:`
  line, so a claim *cleared* on a web page was indistinguishable from one
  cleared on the corpus. The summary now lists each web window with its verdict,
  domain and retrieval time.
- **A cited web passage states whether its domain is an authority**, since
  ranking does not filter and the citation could come from anywhere:

```
          https://www.who.int/news/mythbusters
          [web evidence, recognised authority domain]
          [retrieved 2026-11-01T22:48:00Z]

          https://randomblog.example.com/post
          [web evidence, NOT an authority domain — weigh accordingly]
          [retrieved 2026-11-01T22:48:00Z]
```

Live contrast on the two triggering windows:

```
  5G claim      window 1: NO_EVIDENCE  [pmc.ncbi.nlm.nih.gov]
  sponsor read  window 1: NO_EVIDENCE  [faurit.com]           day-trading blog
```

Both were used as evidence; neither was discarded; the reader can see which is
which.

## End-to-end, live

```
  [1/1] web fallback: 4 result(s) for "effects of 5G frequency on human health studies"
  [1/1] ok   NO_EVIDENCE

  origin      : web
  retrieved_at: 2026-11-01T22:42:35Z
  decision    : corpus empty and speaker is asserting;
                query="effects of 5G frequency on human health studies"
```

The verdict is NO_EVIDENCE because the relevance gate judged that papers on 5G
exposure limits do not address the specific claim that "2,000 studies have been
done". That is the gate working as designed and applying the same standard to web
evidence as to corpus evidence — a web result that merely shares a topic does not
license a verdict either.

Without `web_fetch_fn` the fallback is inert and behaviour is unchanged:
`ajc_ip-IbdQ` still returns NO_EVIDENCE, as before.

## Still open

- **No labelled ground truth for web verdicts**, so there is no precision figure
  for them. This is part of why they are reported separately and excluded from
  the corpus metrics.
- **The Vicuna-13b result above is untested here.** Whether web retrieval helps
  or hurts `qwen2.5:14b` needs measuring on more than the two windows that
  currently trigger, which means labelling windows whose narratives the corpus
  does not cover.
- `ddgs` scrapes rather than using a contracted API, so it can break when
  DuckDuckGo changes its HTML. Failures return `[]` and leave the verdict at
  NO_EVIDENCE rather than crashing a run, but a batch run should check how often
  searches come back empty.


## How often the fallback fires on the dev set: never (2026-11-04)

Measured over the 80 v2 dev windows: **0 trigger the fallback.** The first
condition — the corpus returned no usable evidence — is never met.

```
  set         n    min best-sim   median   empty retrieval
  v1 (test)  81      0.2010       0.6504     4  (4.9%)
  v2 (dev)   80      0.4838       0.7525     0  (0.0%)
```

The minimum best-similarity across every dev window is 0.4838, against a
`MIN_SIMILARITY` floor of 0.25. Nothing comes near empty retrieval, so
`should_search` is never reached and its stance condition is never exercised.

**This is a property of the dev collection, not of the trigger.** The v2 queries
were written to target the four corpus topics (`collect/config.py`), so every dev
window is on-corpus by construction. The lowest-scoring dev windows are the
embalmers'-clots claims at 0.48-0.50 — a misinformation narrative the corpus
addresses only obliquely, which still retrieves clotting passages rather than
nothing.

### Consequences

1. **The trigger cannot be tuned on dev.** There is no dev signal for the stance
   condition, for `max_results`, or for whether authority ranking helps. The
   `should_search` logic in docs/adjudication-experiments.md was measured on v1
   windows, which are now test material, so it remains an untuned component.

2. **The fallback is rare in production, not broken.** On v1 it fired on 4 of 81
   windows (4.9%), of which the stance condition correctly suppressed 2. So the
   expected rate is a few percent of windows, concentrated in videos about
   narratives the corpus does not cover.

3. **13 of the 43 test-set videos labelled as misinformation concern narratives
   with no corpus coverage at all** (pandemic origins, regulatory conduct, excess
   mortality, cancer, neuropsychiatric harm, 5G, shedding, ivermectin —
   data/labels/ground_truth.json). Those are where the fallback matters, and they
   are entirely in the test set.

### What would be needed to tune it

A dev collection aimed at *uncovered* narratives rather than corpus topics —
queries for graphene oxide, microchips, 5G, shedding, pandemic origins. That is
the same gap recorded for `SCOPE_ANSWERED_SIMILARITY` and `RELATIVE_CUTOFF` in
docs/routing-experiments.md: all three parameters govern what happens when the
corpus has nothing, and the current dev set never puts the system in that state.

## The uncovered group: targeting narratives did not raise the fallback rate (2026-11-04)

The second dev collection included 7 queries aimed at narratives the corpus does
not cover — graphene oxide, plandemic and depopulation, excess mortality, cancer,
5G, shedding, suppressed treatments — specifically to exercise the fallback,
after dev1 produced zero eligible windows. 19 videos transcribed, 171 windows.

**It did not work, and the reason matters more than the result.**

```
  set                windows   empty retrieval
  v1 (test)               81     4  (4.9%)
  dev1                    80     0  (0.0%)
  dev2 mirror            456     8  (1.8%)
  dev2 uncovered         171     3  (1.8%)
```

Same rate as the mirror set. Per query, only one narrative produces any empty
retrieval at all:

```
   3/20   covid vaccine graphene oxide
   0/26   covid vaccine depopulation plandemic
   0/20   covid vaccine turbo cancer
   0/16   covid vaccine 5G
   0/ 9   covid vaccine shedding
   0/46   ivermectin covid treatment suppressed
   0/34   covid vaccine excess deaths
```

### Why: the corpus never returns nothing

A window about an uncovered narrative still retrieves corpus passages, just
irrelevant ones:

```
  turbo cancer   -> general mRNA-platform text        0.399
  5G             -> myocarditis passages             0.524
  shedding       -> RCT meta-analysis                0.603
  ivermectin     -> platform efficacy review         0.601
  excess deaths  -> RCT meta-analysis                0.595
  graphene oxide -> infertility papers               0.475
```

2,488 passages over a shared vocabulary ("vaccine", "spike protein", "risk",
"study") means something always clears a 0.25 floor. Empty retrieval requires a
window with almost no vaccine vocabulary at all, which is why the three eligible
graphene windows are the exception rather than the pattern.

### What this means for the trigger

`should_search` requires the corpus to have returned nothing. On this evidence
that condition is close to unreachable: **1.8% of windows, and effectively one
narrative.** The fallback as designed will almost never fire, including on
exactly the 13 test videos whose narratives the corpus cannot answer.

The defect is in the trigger condition, not the corpus. "Retrieval returned no
passage above a floor" is not the same question as "the corpus has nothing that
addresses this claim", and the pipeline already computes the second one — the
relevance gate's `addresses` field, which is what correctly returns NO_EVIDENCE
for microchip and graphene claims that retrieve at 0.57 and 0.39.

**The trigger should be `verdict == NO_EVIDENCE`, not `not passages`.** That is
what was specified originally (docs/adjudication-experiments.md) and what
`should_search` did not implement. Fixed below.


## Trigger rewritten to use the verdict (2026-11-04)

`should_search` now fires on the adjudicated verdict instead of an empty
retrieval. Two conditions, unchanged in spirit:

1. The corpus had nothing that **addresses** the claim — `verdict` is
   NO_EVIDENCE, or the model answered `addresses: "no"`.
2. The window asserts something — `stance` is `"asserting"`.

`check_transcript` adjudicates first and passes both fields in, so the trigger
costs no extra model call. It also replaced a separate stance call that was being
made per candidate window, so the fallback path is now cheaper than before. That
two-pass stance function (`read_stance`/`STANCE_PROMPT` in `rag/llm.py`) became
dead once `stance` was folded into the single adjudication call, and was removed
on 2026-11-05 — asked in isolation it scored 6/12 against one-pass
adjudication's 9/12, a finding recorded in `tests/stance_eval.py`.

When the trigger fires and the web returns results, the claim is
**re-adjudicated against the web evidence** and the verdict relabelled
`FAKE_NEWS_WEB` / `NOT_FAKE_NEWS_WEB`. The corpus verdict is discarded, since it
was NO_EVIDENCE by construction — that is what sent the claim to the web.

### Verified end to end

A graphene-oxide claim, which the old trigger skipped entirely because the corpus
returned four passages at ~0.39:

```
  [1/1] web fallback: 1 result(s) for "graphene oxide in covid vaccine vials"
  [1/1] ok   FAKE_NEWS_WEB
  verdict=FAKE_NEWS_WEB  origin=web  addresses=yes
  why: corpus has no evidence addressing the claim (verdict=NO_EVIDENCE)
  url: https://www.who.int/mythbusters
```

Corpus-covered claims never reach the web — three tested (infertility, clotting,
an accurate myocarditis statement), zero searches. Without a fetch function the
fallback stays inert and the claim remains NO_EVIDENCE, so a run with no web
access behaves exactly as before.

Unit-tested on seven cases including the two that must not fire: a NO_EVIDENCE
verdict with `stance` of `neutral` or `denying`, which is the sponsor-read and
filler case.

### An elimination loop was considered and is not needed

The proposal was to have the model reason over the retrieved chunks and, if they
do not resolve the claim, drop them one at a time until either the claim is
answered or the set empties and the web is called.

The first half already exists — that is the `addresses` field, asked before the
verdict. The second half is unnecessary, measured on an uncovered claim by
dropping the weakest passage each round:

```
  round 1: n=4 top=0.392  addresses=no  NO_EVIDENCE
  round 2: n=3 top=0.392  addresses=no  NO_EVIDENCE
  round 3: n=2 top=0.392  addresses=no  NO_EVIDENCE
  round 4: n=1 top=0.392  addresses=no  NO_EVIDENCE
```

`addresses=no` on the first round, and stable as passages are removed. The model
does not need the weak passages withheld to see that they do not address the
claim. A covered claim is equally stable in the other direction
(`addresses=yes`, FAKE_NEWS, through three rounds), so elimination would add one
model call per round and change nothing.

## Trigger rate after the rewrite (2026-11-04)

Measured on both dev2 groups, two windows per video, with the verdict-based
trigger:

```
  group       windows   fires        verdicts
  mirror           85   30 (35.3%)   49 NO_EVIDENCE, 30 NOT_FAKE_NEWS, 6 FAKE_NEWS
  uncovered        38   20 (52.6%)   31 NO_EVIDENCE,  1 NOT_FAKE_NEWS, 6 FAKE_NEWS
```

Against the old condition's 1.8% on both. The mechanism works exactly as
specified — every firing window is NO_EVIDENCE *and* `addresses: "no"` *and*
`stance: "asserting"`, and the stance condition suppresses a further 29 windows
that are NO_EVIDENCE but `neutral` or `denying`.

### 35% on the mirror set is a corpus-coverage finding, not a trigger bug

The mirror group mirrors the test distribution, so a third of test-like windows
having no corpus evidence that addresses them says something about the corpus.
All 30 firing mirror windows were read individually (22 distinct videos):

```
  correct                                   18
  borderline (no claim in the window)        3
  clearly false                              1
```

The 18 correct ones fall into three groups:

- **Regulatory conduct and trial integrity** — Fauci text messages and diaries,
  HHS injury compensation, the Ventavia whistleblower. The corpus is clinical
  literature and has nothing on how agencies behaved.
- **Accepted coverage gaps** — microchips, Gates and 5G, engineered origin, the
  1965 coronavirus claim. Already recorded in docs/corpus-inclusion.md.
- **Scope boundaries** — the mRNA *flu* vaccine, hepatitis B, flu-and-COVID
  co-administration. These are not failures: the corpus is COVID-specific by
  design, so a flu-vaccine claim correctly has no evidence in it.

The clearest single case for why the trigger had to change is
`4gaG0rWGBd8`, a Strong Medicine video asserting that COVID origins remain
unresolved and that mRNA vaccines prevented roughly one million American deaths.
It retrieves at **0.712-0.734** — high similarity, entirely background
epidemiology about case counts, nothing addressing origins or mortality
modelling. Under the old "retrieval returned nothing" condition it was never
eligible.

The one clearly false trigger is `A22eAcrsjqQ`, an Arizona Humane Society free
pet-vaccination and microchipping clinic, collected because "covid vaccine
microchip" matched animal microchipping. The model's reasoning is correct —
*"discussing pet vaccinations and microchipping, which is unrelated to the
evidence provided about human vaccines"* — but `addresses: "no"` is true of it,
so the trigger fires. It is labelled `off_topic` in the dev labels and excluded
from the sweeps; the trigger has no topic test. A domain check was built for
exactly this and rejected on measurement
(docs/adjudication-experiments.md), because it also rejected the genuine 5G
coverage gap.

The 3 borderline windows contain no claim yet — an introduction, meta-commentary
on how to respond to the myth, and garbled crosstalk. The stance pass called them
`asserting`, which is defensible for a speaker announcing what they will discuss,
but leaves nothing to check.

**An earlier version of this section reported ~60% correct from a sample of five
windows.** That was wrong in both directions: the sample was too small, and one
window judged false (`4gaG0rWGBd8`) was correct — the 150-character excerpt read
at the time cut off before its claims. Reading all 30 gives ~82%.

### What this does not establish

The rate is measured; the **quality** of web verdicts is not. No labelled ground
truth exists for them, which is why they are reported separately from the corpus
figures and never pooled into a precision number (see the provenance section
above).

## Tavily as an alternative backend (2026-11-04)

`rag/websearch.tavily_fetch` is available as a second `fetch_fn`, selected with
`--web tavily`. **DuckDuckGo remains the default**, and the reasoning in the
backend-choice section above still holds: it needs no key and follows published
practice for this task, which makes the choice citable.

### Why have both

Tavily's one concrete advantage is the response shape. It returns cleaned
extracted text per result, so the snippet handed to the relevance gate is article
prose rather than a meta description. The DuckDuckGo adapter's `body` field is
whatever the page put in its description tag, which is sometimes a navigation
blurb.

Against that, it is a paid service outside the literature for this task, so a
verdict produced with it rests on a different footing than one produced with the
backend a published study used. Which backend produced a verdict is therefore
part of its provenance, which is why `get_backend` raises on an unknown name
rather than silently defaulting.

### Setup

1. Create an account at tavily.com and copy the key from the dashboard. The free
   Researcher tier is 1,000 API credits a month with no card required; `basic`
   search depth costs 1 credit per request, `advanced` costs 2.
2. Put the key in the environment or in the project `.env`, either works:

```bash
export TAVILY_API_KEY=tvly-...
```

```
# .env, alongside YOUTUBE_API_KEY
TAVILY_API_KEY=tvly-...
```

`rag/websearch._api_key_from_env` checks the environment first and falls back to
reading `.env` directly. It does that rather than calling `load_dotenv` because
only the collect package loaded `.env`, so a key written there was silently
ignored — the adapter returned no results and the run completed with every
uncovered claim left at NO_EVIDENCE and no indication the web tier never worked.
No package is needed: the REST endpoint is called with `requests`.

3. Run with the backend selected:

```bash
python -m rag <video_id> --web tavily
python -m rag <video_id> --web duckduckgo     # the default backend, explicit
python -m rag <video_id>                      # corpus only, no fallback
```

`WEB_SEARCH_BACKEND` is read when `--web` is not given, so a batch run can switch
backend without a code change.

### Cost at the measured trigger rate

The trigger fires on roughly 35% of windows in a test-distribution set
(measured above). A 1,700-word transcript is about 24 windows, so expect ~8
searches per video, or **~8 credits**. The free tier therefore covers about 125
videos a month, which is more than the 119-video test set.

`basic` depth is used rather than `advanced`: the snippet only has to be long
enough for the gate to judge relevance, and halving the credit cost doubles the
number of videos the free tier covers.

### Failure behaviour

Both adapters return `[]` on any failure — missing key, network error, exhausted
quota, unexpected response body — rather than raising. The web tier is a
fallback, so an outage leaves the verdict at NO_EVIDENCE instead of ending a
transcript run. Verified: with no key set, `--web tavily` produces
`0 result(s)` and the verdict stays NO_EVIDENCE.

### Authority handling

Tavily is called with `include_domains` set to `AUTHORITY_DOMAINS` and
`include_domains_mode: "prefer"`, not `"restrict"`. A hard filter would return
nothing for exactly the narratives that reach this fallback, since they are the
ones health authorities have not published on. The same preference-not-filter
reasoning applies as for the DuckDuckGo path.

## Web-fallback smoke test (2026-11-04)

3 test videos, `--web duckduckgo`, Ollama/qwen2.5:14b, 16.7 min. All three are
labelled fake, so this measures recall and the fallback mechanism only — there
is **no false-positive test here**, and the video-level 1.000 should not be read
as a precision result.

```
  videos                 3/3 predicted fake   (corpus-only was 2/3)
  windows               42
  answered on corpus    25
  answered on web       17  (40.5%)
  verdicts    FAKE_NEWS 11   FAKE_NEWS_WEB 4   NOT_FAKE_NEWS 3
              NOT_FAKE_NEWS_WEB 1   NO_EVIDENCE 23
```

The 40.5% web rate is above the 35.3% measured on the mirror development set,
consistent with these three videos carrying narratives the corpus does not
cover.

### The converted false negative is a true positive on the wrong phrase

`21hkEfN3UmA` (vaccine passports / IBM-Nazi) was the corpus-only smoke test's
single miss — all windows NO_EVIDENCE, a known corpus coverage gap. With the
fallback it flags, so the confusion matrix improves.

Reading the flag, it is not a catch. The flagged phrase is *"The airport company
is largely responsible for putting this together. And you mentioned in private
companies..."* — an unremarkable statement about who built a passport system,
not the video's misinformation. The video is correctly labelled fake, so this
scores as a true positive for the wrong reason, and the underlying corpus gap
is unchanged.

**This is why the video-level join rule needs window-level review alongside it.**
"Any window flagged ⇒ video predicted fake" cannot distinguish a flag on the
narrative from a flag on an aside in the same video.

### Window-level flag quality

Of the 15 flags, read individually:

- **~9-10 substantive** — cardiac arrest causation, the 70% death claim, the
  7,000-to-prevent-one NNT figure, accelerated heart disease, vaccine-induced
  cancer, natural immunity superiority.
- **3 window duplicates** — w15/w16 and w2/w9 flag the same phrase twice. Expected
  from overlapping 140w/70s windows, not an error, but it inflates any flag count.
- **3 not factual assertions** — *"the serious harm is there"* (asserts nothing
  checkable alone) and *"No one has the right to ask you if you've been
  vaccinated. So it's an invasion of privacy"* (a legal/ethical opinion). Both
  corpus-sourced, similarity 0.55-0.62, so similarity does not separate them —
  consistent with the finding that `FLAG_MIN_SIMILARITY` is unusable
  (docs/adjudication-experiments.md).

No window-level precision figure is claimed from n=15 on 3 videos. The point of
recording it is that the video-level metric hides all three problems, so the
full run must be diagnosed at window level.

## Caching, and why the earlier decision was reversed (2026-11-05)

Caching was originally rejected: *"I don't want disk caching because I think
that would add bias."* That reasoning is sound for a measurement run — a cached
result reports what the web said when the entry was written, so a figure quoted
from a warm cache is not a figure about the web today.

It inverts for prompt tuning. Comparing two prompts means both must see the
**same** evidence; if the search engine returns different pages between
variants, the comparison measures the engine, not the prompt. There the cache is
what makes the experiment valid.

So the cache is **on by default and cleared before a measurement run**:

```bash
python tests/run_experiment.py --web duckduckgo --clear-cache   # fresh evidence
python tests/run_experiment.py --web duckduckgo                 # replays cache
```

Implementation notes that keep the bias visible rather than hidden:

- One JSON file per `(backend, query, max_results)` under `data/web_cache/`,
  keyed by SHA-1.
- A cached entry keeps its **original** `retrieved_at`, not the replay time, so
  a result row always says when the evidence was actually fetched. The entry
  also records `cached_at` separately.
- Preflight prints the cache state and entry count, so a run starting from a
  warm cache cannot be mistaken for a fresh one.
- `WEB_CACHE=0` disables it entirely for one run.
- Cache failures never break a run: a read error is a miss, a write error passes.

The cache does not change which claims trigger the fallback, only where the
evidence for a repeated query comes from.
