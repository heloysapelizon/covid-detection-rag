# Adjudication experiments

How the local model is asked to turn a claim window plus retrieved passages into
a verdict, and what has been measured about that step. Retrieval is covered in
docs/chunking-experiments.md and docs/routing-experiments.md; this file is about
what happens after retrieval.

Model: `qwen2.5:14b` (Q4_K_M) served by Ollama at `http://localhost:11434`.
Throughput at the production settings is roughly 15 s per claim window, so a
1,700-word transcript (~24 windows) takes about 6 minutes.

## First end-to-end runs (2026-10-31)

Four videos run through `python -m rag <video_id>`. Recorded for the failure it
exposed, not for its numbers: this run predates the corpus changes in
docs/corpus-inclusion.md (2301 passages then, 2488 now) and `ROUTE_GAMMA = 0.52`
(0.60 now), so the per-video counts are not reproducible. The regression run
further down is the current state.

| Video | Content | Windows | Result |
|---|---|---|---|
| `OqSbJ8TxOx8` | clot-claim podcast | 6 | 2 FAKE_NEWS, 3 NOT_FAKE_NEWS, 1 NO_EVIDENCE |
| `H0F-YvUQEGY` | accurate KSAT fact-check | 4 | **0 flags** |
| `ajc_ip-IbdQ` | microchip implant news | 1 | 1 FAKE_NEWS |
| `Y6_jn0DzXHI` | graphene-oxide narrative | 11 | **9 FAKE_NEWS**, 2 NO_EVIDENCE |

Two results are correct and worth keeping as regression cases:

- `OqSbJ8TxOx8` caught the "burnt spaghetti casserole" clot claim and cited the
  `ijms-22-10791` coagulopathy paper — one of the three `blood_clots` papers
  added the same day, so the new corpus is being used.
- `H0F-YvUQEGY` produced **zero flags across four windows**. This video discusses
  infertility myths in order to debunk them, which is the false-positive case
  that previously failed; stance handling now holds on it.

## The NO_EVIDENCE failure

`Y6_jn0DzXHI` flagged 9 of 11 windows. The model's own stated reasons contradict
its verdicts:

> "graphene oxide in the vials, which is **not mentioned or supported by the
> evidence** provided" — verdict: FAKE_NEWS

> "The evidence **does not address** microchip implants or vaccine status storage
> under the skin; the claim presents a false idea as true" — verdict: FAKE_NEWS

The model is treating **absence of evidence as contradiction**. Both are
NO_EVIDENCE by the prompt's own definition: the corpus holds zero graphene and
zero microchip passages, a known and accepted coverage gap
(docs/corpus-inclusion.md). One flag cited the ICH shedding guidance at
similarity **0.336**, which is noise.

The verdicts are incidentally true — graphene oxide is not in the vaccines — but
they are **unsupported**: the system asserts a conclusion its evidence does not
license. For a study whose method claim is *evidence-grounded* detection, an
unsupported true answer is the more serious error, because it cannot be
distinguished from an unsupported false one.

### It is a prompt problem, not a model limit

`rag/llm.py` already defines NO_EVIDENCE and instructs "Topic overlap alone is
never enough to flag", but it asks for stance and verdict together, which lets
the model answer "is this true?" before "does the evidence speak to this?".

Forcing relevance to be decided **first**, as a separate field, fixes it:

```
EVIDENCE:
{retrieved passages}

CLAIM:
{claim}

Answer in JSON.
1. "addresses": does the EVIDENCE discuss the specific subject of the CLAIM? yes or no.
2. "verdict": if addresses=no you MUST answer "NO_EVIDENCE". Only if addresses=yes
   may you answer "FAKE_NEWS" or "NOT_FAKE_NEWS".
Reply with keys "addresses" and "verdict" only.
```

Measured on five probe claims, with the top retrieval similarity for context:

```
case         top_sim  gated verdict  addresses
microchip      0.570  NO_EVIDENCE    no
graphene       0.408  NO_EVIDENCE    no
magnetism      0.481  FAKE_NEWS      yes
real-clot      0.598  NO_EVIDENCE    no
real-fert      0.729  FAKE_NEWS      yes
```

The two uncovered narratives (microchip, graphene) now return NO_EVIDENCE instead
of an unsupported flag, and a covered claim (`real-fert`, infertility) still
flags. Note `top_sim` does **not** separate these cases: microchip retrieves at
0.570 while magnetism correctly flags at 0.481. A similarity threshold alone
cannot replace the relevance question — the gate has to be asked in words.

### `real-clot`: the routing cause, since fixed

The probe claim *"the spike protein from the vaccine forms clots that cannot be
broken down"* returned NO_EVIDENCE, which looked like a gate failure because the
corpus plainly covers spike-protein coagulopathy. It was not: retrieval had
handed the model four passages about spike-protein immunology and none about
clotting, so the gate answered correctly for what it was given.

The cause was upstream. `blood_clots` scored below γ, so the claim fell back to
the general tier, which excludes the topical passages — while the relevant
coagulopathy passage sat near the top of the flat index. The original trace is
not reproduced here: two of its four quoted passages came from a paper since
withdrawn (docs/corpus-inclusion.md), and γ has since changed, so the specific
similarities no longer hold.

Fixed by the changes described in the next section. The claim now leads with the
`blood_clots` tier and retrieves the `ijms` coagulopathy passages alongside the
general ones.


## Fixes implemented (2026-11-01)

Items 1–3 of the open list above, implemented and verified. Item 4 (re-running
the four videos) is still outstanding — the model server was stopped, so no
end-to-end run has been made since these changes.

### 1. Relevance gate wired into `rag/llm.py`

`addresses` is now the **first** field of the response, before `stance` and
`verdict`, and the GBNF grammar requires it. Field order is the reasoning order
for a left-to-right decoder, which is the whole mechanism — the model has to
commit to "does the evidence discuss this?" before it can emit a verdict.

The prompt rule is enforced in code rather than trusted:

```
addresses == "no" and verdict in {FAKE_NEWS, NOT_FAKE_NEWS}  ->  NO_EVIDENCE
```

The NO_EVIDENCE definition was also strengthened to say that an unsupported true
answer is still an error ("Use this whenever the evidence is silent on the
claim's specific subject, EVEN IF you personally believe the claim is false"),
and a worked `addresses: "no"` example was added.

### 2. Similarity floor for flags — measured, then disabled

`FLAG_MIN_SIMILARITY` downgraded a FAKE_NEWS verdict to NO_EVIDENCE when its best
evidence fell below a floor. It was set to 0.45 from the v1 flag distribution
(0.259-0.723, median 0.471, 5 of 13 flags below 0.45), on the assumption that the
low-similarity flags were the unsupported ones.

**Measured on dev, that assumption is false and the relationship is inverted.**
80 dev windows adjudicated with the floor disabled produced 9 flags; reading each
against the hand annotation:

```
  top_sim   window             correct?
   0.428    s4Z3baKFWu0 w0     yes   embalmers' clots, misinformation
   0.460    s4Z3baKFWu0 w1     yes   spike-protein clot claim
   0.604    tXMna_7NYko w1     yes   Fauci texts framed as miscarriage link
   0.701    6DdIUUPm3ug w0     NO    Doctor Mike, pro-vaccine, misread
   0.789    JQVmrgcEc68 w1     NO    real CDC schedule change, reported
   0.821    JQVmrgcEc68 w0     NO    same, FDA commissioner quote
   0.828    o3Kh99pZnn4 w4     yes   misrepresents the Nordic study
   0.854    o3Kh99pZnn4 w5     yes   overstates cardiac scarring
   0.872    CmNQY0TrLrA w3     NO    accurate menstrual-study reporting
```

All four false positives sit at **0.70-0.87**, above any plausible floor. The two
lowest-similarity flags are both **correct** — misinformation about a narrative
the corpus covers only obliquely still retrieves something, just weakly. So every
floor removes correct flags and not one false positive:

```
  floor 0.45  ->  4 correct + 4 false positives   (lost 1 correct)
  floor 0.50  ->  3 correct + 4 false positives   (lost 2 correct)
  floor 0.65  ->  2 correct + 4 false positives   (lost 3 correct)
```

Disabled (`0.0`). This is consistent with what the original probe already showed —
that `top_sim` does not separate covered from uncovered claims (microchip
retrieves at 0.570 and needs NO_EVIDENCE; magnetism correctly flags at 0.481) —
but the v1 flag distribution was read as though it did.

The four remaining false positives are stance and specificity errors, which no
similarity threshold can reach:

- `6DdIUUPm3ug` and `CmNQY0TrLrA` are stance failures — the speaker is accurate
  or pro-vaccine and the model read assertion of the myth. This is the documented
  dominant failure mode, and the stance guard did not catch them.
- `JQVmrgcEc68` x2 are specificity failures — the video correctly reports a real
  2025 change to the CDC schedule, and the corpus (published earlier) contradicts
  it. The evidence is genuinely out of date, not misread.

The second pair is worth separating out: it is not a model error but a corpus
currency problem, and a threshold would never have addressed it.

### 3. Tier fallthrough — first attempt failed, recorded

The diagnosis in the section above was that `search()` stops at the first tier
clearing `MIN_SIMILARITY` (0.25), so the `all` tier was unreachable. The obvious
fix was a separate, higher floor for the *widening decision*:
`SCOPE_ANSWERED_SIMILARITY = 0.55`, distinct from the floor for returning hits.

**That did not fix the measured case.** With the new corpus the general tier
answers `real-clot` at **0.6649**, far above any plausible widening floor:

```
tiers offered: [('general', 1057), ('all', 2361)]
general  top4: [0.6649, 0.6228, 0.6138, 0.5982]
all      top4: [0.6649, 0.6228, 0.6138, 0.5982]      <- identical
```

Two things this shows, both of which correct the earlier section:

- **Widening to `all` would not have helped even if reached.** The four winning
  passages are general-tier documents that outrank every clot passage on the flat
  index; the best clot passage is rank #9 at 0.5840. The earlier claim that the
  `all` tier was the fix was wrong.
- **The two new general reviews made this case worse**, not better. They are
  platform reviews dense in spike-protein vocabulary, which is exactly what
  outranks the clotting evidence on a spike-protein clotting claim.

So the defect is not in the fallthrough at all. It is that `blood_clots` scored
0.4349 at routing, below γ=0.52, and was therefore dropped from the tier list
entirely — never getting a scope in which its 469 passages are not competing
against 1057 general ones.

#### What actually fixed it

The first attempt was a sub-γ tier carrying the topic's passages **without** the
general documents, so general-tier immunology could not outrank the coagulopathy
passages. It worked on this case and was wrong in principle: a general document
is in scope for every claim by definition, so excluding it from one tier and not
another made the rule depend on which side of γ a claim happened to fall.

Measured on the v2 dev labels, exclusion also optimised the wrong thing — it
scored better on "did retrieval surface the right topic" (55/64 vs 53/64) and
worse on whether the passage that actually answers the window was found
(labelled-phrase recall 0.600 vs 0.667). The first metric rewards topical
passages for being topical; the second is the objective.

The design now in place has **general documents in every tier** and handles the
vocabulary dilution in ranking, via `TOPIC_RANK_BONUS = 0.03` — a small additive
preference for topic-specific passages. Swept on dev, phrase recall is flat at
0.667 from a bonus of 0.00 to 0.04 and falls beyond, and the clot case needs at
least 0.02.

The bonus affects ORDER only. `index.search` keeps the unmodified cosine for
reporting, because a reported similarity is compared against `MIN_SIMILARITY`,
`FLAG_MIN_SIMILARITY` and the relative cutoff — letting a ranking preference
inflate it would have silently shifted all three.

#### The extra tier was then removed

With general documents in every tier, the sub-γ tier earns nothing. Lowering γ so
that no candidate is discarded achieves the same result with one fewer concept:

```
                                   own-topic   phrase recall   clot case
  sub-γ narrow tier, γ=0.60          56/64         0.667          ok
  no narrow tier,     γ=0.30         56/64         0.667          ok
  no narrow tier,     γ=0.60         33/64         0.500          fails
```

The two working configurations fail on **exactly the same 8 windows** — not
merely the same count. So the tier was never protecting against the mis-routes it
was credited with; it was compensating for a γ that discarded correct routes.

γ changed 0.60 -> 0.30 (= `ROUTE_FLOOR`, so every candidate clearing the floor now
leads a tier), and `TopicRouter.scopes` is back to the original three-kind
structure: one tier per confident topic, then general, then all.

#### Probe behaviour after both changes

```
case         tier                      top_sim  retrieved
real-clot    blood_clots                0.5840  ijms coagulopathy        FIXED
real-fert    infertility                0.7895  fendo + pbio             unchanged
magnetism    general                    0.7362  CDC myths sheet          correct
graphene     all                        0.3655  (no coverage)            honest
microchip    natural_immunity           0.5788  irrelevant immunology    gate's job
```

`microchip` is the case the floor cannot catch: it retrieves at 0.5788, above the
0.45 flag floor, on passages that say nothing about microchips. Only the
relevance gate prevents a flag there — which is why item 1 is the real fix and
item 2 is a backstop.

The routing eval is unchanged at every threshold (`route()` was not modified;
only how its output becomes tiers), so γ=0.52 still stands.

### Still open

- **End-to-end regression run.** Not done — server stopped. The four videos in
  the table at the top of this file must be re-run before any batch, and their
  earlier flag counts are no longer reproducible anyway because the corpus
  changed (docs/corpus-inclusion.md).
- `FALLBACK_ROUNDS` was **removed** on 2026-11-05. It was never wired into
  `pipeline.py`, and the web-search tier it was being held for was instead
  keyed on the adjudicated verdict (docs/web-search.md), so nothing attaches
  there any more. `EvidenceIndex.search(skip=...)`, the mechanism it would have
  used, is kept — it is working, measured code (chunking Exp. 5) and is what a
  future elimination loop would need — but it currently has no caller.

### Gate probe, measured against a live server (2026-11-01)

`experiments/gate_probe.py`, `qwen2.5:14b` at temperature 0, corpus 2361 passages,
with the relevance gate, flag floor and sub-γ topic tier all in place.

| case | tier | top_sim | addresses | verdict |
|---|---|---|---|---|
| microchip | `natural_immunity` | 0.5711 | no | NO_EVIDENCE |
| graphene | `all` | 0.3924 | no | NO_EVIDENCE |
| real-clot | `blood_clots` | 0.6137 | yes | FAKE_NEWS |
| magnetism | `all` | 0.5245 | yes | FAKE_NEWS |
| real-fert | `infertility` | 0.7409 | yes | FAKE_NEWS |

**The gate holds on the case that matters.** `microchip` retrieves at 0.5711 —
*above* `FLAG_MIN_SIMILARITY` — with `stance='asserting'`, so neither the
similarity floor nor the stance guard would have stopped it. Only
`addresses='no'` did. This window returned FAKE_NEWS before the change.

**Similarity is not merely a weak gate, it is inverted here.** `magnetism` must
flag and does so at **0.5245**, which is *lower* than `microchip`, which must
not flag at 0.5711. No threshold placed between them classifies both correctly.
This is the sharpest available evidence that relevance has to be asked in words
rather than inferred from a score, and it is a stronger version of the earlier
observation (0.570 vs 0.481) because both numbers moved and the ordering held.

**`real-clot` flags rather than returning NO_EVIDENCE**, confirming the sub-γ
topic tier end to end: it reaches `blood_clots`, retrieves four `ijms`
coagulopathy passages at 0.572–0.614, and the gate correctly reports that the
evidence addresses the claim. Previously this claim was routed to general and
locked out of its own topic's evidence.

`graphene` falls through to `all` and still returns NO_EVIDENCE at 0.3924 — the
correct outcome for an accepted zero-coverage narrative, and reached by two
independent mechanisms (low similarity *and* the gate).

#### One weakness to watch

On `real-clot` the model's `reason` reads "which is **not supported by** the
evidence" for a case where the evidence positively refutes the claimed
mechanism. The verdict and the gate are correct; the explanation prose is
weaker than the decision. That exact phrasing was the tell for the original
absence-as-contradiction bug, so it is worth monitoring across a larger run even
though it is cosmetic here — the `reason` field is reported to readers, so
sloppy wording undercuts a correct verdict.

#### Caveat

5 claims, single run, temperature 0. This demonstrates the mechanism and the
inversion; it is not an accuracy rate. The regression videos are the real test.

## Regression run after the fixes (2026-11-01)

The four videos from the table at the top of this file, re-run with the
relevance gate, flag floor and sub-γ topic tier in place, on the 2361-passage
corpus. Directly comparable to that table.

| Video | Content | Windows | Before | After |
|---|---|---|---|---|
| `OqSbJ8TxOx8` | clot-claim podcast | 6 | 2 FAKE / 3 NOT / 1 NO_EV | **1 FAKE** / 2 NOT / 3 NO_EV |
| `H0F-YvUQEGY` | accurate fact-check | 4 | 0 flags | **0 flags** (4 NOT_FAKE) |
| `ajc_ip-IbdQ` | microchip news | 1 | **1 FAKE_NEWS** | **NO_EVIDENCE** |
| `Y6_jn0DzXHI` | graphene narrative | 11 | **9 FAKE_NEWS** | **0 flags, 11 NO_EVIDENCE** |

The NO_EVIDENCE failure recorded above is resolved. 10 unsupported flags became
NO_EVIDENCE, and the one correct flag survived.

### The graphene video: 9 flags to 0, and not by brute force

All 11 windows return NO_EVIDENCE, and **none of them was overridden in code** —
the model reached that verdict itself through the `addresses` field. The reasons
now agree with the verdicts, which is exactly what failed before:

```
  w1  "asserts that there is graphene oxide in vaccine vials, but the
       evidence does not address this specific claim"          -> NO_EVIDENCE
  w6  "the evidence does not discuss graphene oxide or its interaction with
       electromagnetic fields, so it cannot address the claim"  -> NO_EVIDENCE
```

Compare the same model on the same video before the change, where that identical
observation produced a flag:

```
  "graphene oxide in the vials, which is not mentioned or supported by the
   evidence provided"                                          -> FAKE_NEWS
```

The model always *knew* the evidence was silent. The prompt previously let it
answer "is this true?" before "does the evidence speak to this?". Asking
relevance first, as a field the decoder must emit before the verdict, is the
whole fix. This confirms the earlier conclusion that it was a prompt problem
rather than a model limit — now on 11 real windows instead of 5 probe claims.

`w11` reached NO_EVIDENCE by the pre-existing route (no passage cleared
`MIN_SIMILARITY`), not via the gate.

### `ajc_ip-IbdQ`: the microchip flag is gone

One window, previously FAKE_NEWS, now NO_EVIDENCE with `addresses='no'` and
`stance='asserting'` — the speaker is asserting, and the system still declines to
flag, because nothing retrieved addresses microchips. This is the probe result
reproduced on real transcript text.

### `OqSbJ8TxOx8`: one flag lost, one kept

The surviving flag (w6, similarity 0.5884) is the "burnt spaghetti casserole"
clot claim, cited to the `ijms-22-10791` coagulopathy paper — the same correct
flag as before the change.

The flag that disappeared (w4) was removed by the **similarity floor**, not the
gate:

```
  Overridden: FAKE_NEWS on best evidence at similarity 0.433, below the 0.45
  floor for grounding a flag. The speaker asserts that the spike protein can
  induce fibrin from fibrinogen, but the evidence states there is no
  indication of a causal effect on clotting...
```

This is the one judgement call in the run and it is worth flagging honestly: the
model's reasoning looks substantively *correct* — the evidence does speak to
spike-protein effects on clotting — but its best passage sat at 0.433. The floor
traded a probably-correct flag for the guarantee that no flag rests on weak
evidence. Given that the corpus demonstrably covers this topic, the right
long-term fix is retrieval (this window should have reached
`blood_clots` with a stronger match) rather than relaxing the floor.
Recorded as a known cost, not as a success.

### `H0F-YvUQEGY`: no regression on the hard negative

Still 0 flags across 4 windows, now all NOT_FAKE_NEWS at similarity 0.748–0.849.
This video debunks infertility myths, so it is the false-positive case; the gate
did not push these to NO_EVIDENCE, which would have been a different failure
(treating a well-covered topic as unaddressed).

### Audit fields added to the output

`rag/pipeline.py` was not copying `addresses` or `stance` into its result rows,
so the JSON recorded a NO_EVIDENCE verdict without recording *why* — whether the
evidence was silent or the model declined for another reason. Both fields are now
persisted. Verified: `addresses='no'`, `stance='asserting'` on the microchip
window.

### Status

The blocking defect is cleared: flags are no longer asserted on uncovered
narratives, and the one covered narrative that was unreachable through routing
now reaches its evidence.

Remaining before a full batch run:

1. **No labelled verdict ground truth.** The assessments above are still by
   reading each verdict and its cited passage. 22 windows over 4 videos is enough
   to show the systematic failure is fixed, not enough for a precision figure.
2. **`OqSbJ8TxOx8` w4** — a likely-correct flag lost to the similarity floor
   because retrieval under-performed on a covered topic. Worth investigating as a
   retrieval problem.
3. The `reason` field sometimes says "not supported by the evidence" where the
   evidence actively refutes the claim. Cosmetic, but it is reported to readers
   and was the tell for the original bug.
4. `FALLBACK_ROUNDS` was unwired and has since been removed; the web-search
   tier is keyed on the verdict instead.

## Relative retrieval cutoff and the web-search trigger (2026-11-01)

### Replacing a fixed top_k with a relative cutoff

`TOP_K = 4` returns exactly four passages whatever the evidence looks like, so a
claim with one strong match is padded with three weaker ones. Every padded
passage enters the prompt and gives the model another chance to flag on topical
resemblance — the failure this file opens with.

An **absolute** similarity threshold was measured first and does not work. Over
all 81 labelled windows, counting passages above a fixed similarity:

```
  thresh   median_n    max    empty windows
    0.40        667   1357       11 / 81
    0.55         45    487       17 / 81
    0.65          1    129       40 / 81
```

No value has a usable set size: cosine is not calibrated across windows, so a
verbose window scores high against everything and one threshold means "487
passages" and "nothing" simultaneously. This is why `MIN_SIMILARITY` looked inert
in an earlier sweep — on the 26 labelled windows, which all retrieve at 0.75+,
the floor never binds. It is not dead code (at 0.90 it correctly returns nothing);
it is simply irrelevant to the easy windows.

`RELATIVE_CUTOFF = 0.90` instead keeps passages scoring at least 90% of the best
in-scope hit, so the scale is set per claim. Measured over all 81 windows
(`expand=0`):

```
  mean 3.4   median 4   min 0   max 4
  set-size distribution: {4: 62, 1: 7, 2: 5, 3: 3, 0: 4}
```

62 windows are unaffected, 15 return 1-3 passages where the evidence drops off
sharply, and 4 return nothing.

**Not tuned, and worth being explicit about that:** recall is flat at 0.600 for
every ratio from 0.80 to 0.95, so the label set does not favour a value. 0.90 is
chosen on the argument that a passage below 90% of the best hit is no longer
"the best evidence" but "whatever else was nearby". The gain here is design, not
measured accuracy: bounded, adaptive, and it produces a well-defined empty case.

### The web trigger is not "retrieval was empty"

The 4 empty windows look like coverage gaps and mostly are not:

```
  Y6_jn0DzXHI w10   "95 percent of people lose all their money ... I created
                     the london real investment club"          sponsor read
  rNUwvh4VxlM w15   "I I totally get that a hundred percent um but but"  filler
  6HzRcPGqzvs w62   "we're gonna do a series with you if you don't mind" chatter
  PMDcOTWEEcI w113  "2,000 studies have been done to show the effects of
                     5G on our systems"                        REAL gap
```

Only one is a claim the corpus fails to answer. Searching the web for the other
three would spend network calls establishing that small talk is not
misinformation.

So `rag/websearch.should_search` requires **two** conditions. The second —
that the speaker is asserting something — has held since. The first was
originally "the corpus returned nothing", measured on these four windows:

```
  Y6_jn0DzXHI w10   stance=asserting   search=True
  rNUwvh4VxlM w15   stance=denying     search=False
  6HzRcPGqzvs w62   stance=neutral     search=False
  PMDcOTWEEcI w113  stance=asserting   search=True
  covered claim     (corpus has evidence)  search=False
```

4 candidates down to 2, with the real gap retained and the filler suppressed.

**That first condition was wrong and has been replaced** (2026-11-04). "Retrieval
returned no passage above the floor" is close to unreachable: over 2,488 passages
sharing a vaccine vocabulary something almost always clears 0.25, so only 1.8% of
windows retrieved nothing even in a collection made specifically for narratives
the corpus does not cover. The condition is now the adjudicated verdict —
NO_EVIDENCE, or `addresses: "no"` — which is the question the relevance gate
already answers correctly. Full measurement in docs/web-search.md.

**Known limitation, unchanged:** the sponsor read still triggers. It is genuinely
`asserting` — just not asserting anything medical — and neither the stance field
nor any similarity threshold can tell a financial claim from a health claim. A
domain check was built and rejected on measurement (below), so this stands. It
costs one wasted search rather than producing a wrong verdict.

### Provenance design

Web evidence is kept structurally separate from corpus evidence, because every
corpus passage is licence-vetted, DOI-resolved and authorship-screened — a paper
was withdrawn on authorship grounds alone — and web pages have none of that.

- distinct verdict labels (`FAKE_NEWS_WEB` / `NOT_FAKE_NEWS_WEB`), never merged
  into the corpus verdict counts
- excluded from headline precision and recall
- every row carries `source_url`, `retrieved_at` and `origin: "web"`
- **no caching.** Considered and rejected: a cache fixes the evidence at
  whatever the web said the first time a claim was seen, so later runs are
  adjudicated against stale pages while appearing current, and a verdict comes
  to depend on when a claim first entered the pipeline rather than on the
  evidence. Reproducibility is served by recording `source_url` and
  `retrieved_at` per row instead, which makes a run auditable without freezing
  its inputs. The cost is that re-running a transcript repeats the searches and
  may return different pages — which is a true statement about web evidence, not
  a defect to hide.
- general search with authority-domain *preference*, not an allowlist: the
  narratives that reach this fallback (graphene, microchips, 5G) are precisely
  the ones WHO and CDC have not published on, so an allowlist would return
  nothing exactly when the fallback is needed

The search backend is injected (`fetch_fn`) rather than hardcoded, so the gating,
caching and provenance logic is testable without network access — and so the
pipeline does not depend on which search tool the host provides.

### Status

`rag/websearch.py` is complete and tested as a unit. It is **not yet wired into
`pipeline.py`**: doing so needs a `fetch_fn` bound to a real search backend, and
the `FAKE_NEWS_WEB` verdict labels threaded through `adjudicate` and the report
path. That is the next step.

## Web fallback wired in; topic check built and rejected (2026-11-01)

### The topic check does not work on this data

A domain-relevance gate was built to stop the fallback firing on a crypto sponsor
read. Each window is scored against a description of the study's subject; whole-
window scoring collapses to ~0.00 even for on-topic windows (a mean-pooled
embedding of 140 words of speech is mostly filler), so the score is the maximum
over 14-word fragments. That separates cleanly in general — on-domain labelled
windows score 0.52-0.68.

On the four empty windows it does not:

```
  0.1786  "mask on planes you can tell they're scared"       filler
  0.1121  crypto sponsor read
  0.0997  "2,000 studies ... effects of 5G on our systems"    REAL coverage gap
  0.0886  "SB 276, anything you want to say"                 chatter
```

**The genuine coverage gap scores below the sponsor read**, because that window
discusses 5G rollout without tying it to vaccination. No threshold separates
them, so the gate would have removed the trigger's only real case while keeping
nothing of value. Removed.

There is also a scope argument against it, independent of the measurement: a
system for detecting misinformation should not decline to check a false claim
because the claim is about 5G rather than vaccines. The residual cost is an
occasional wasted search on a sponsor read, which is cheap and visible in the
output (`web_decision` records why each search fired).

### Wiring

`check_transcript` takes `web_fetch_fn`. When it is `None` the fallback is inert
and behaviour is byte-identical to before — verified on the 5G window, which
returns `NO_EVIDENCE` without a fetch function and `NOT_FAKE_NEWS_WEB` with one.

Verdicts reached on web evidence are relabelled at the pipeline level rather than
inside `adjudicate`, so the relevance gate, similarity floor and stance override
are untouched and apply identically to both sources:

```
  FAKE_NEWS      -> FAKE_NEWS_WEB
  NOT_FAKE_NEWS  -> NOT_FAKE_NEWS_WEB
  NO_EVIDENCE, ERROR  unchanged — they assert nothing, so there is no
                      provenance claim to qualify
```

Each result row carries `evidence_origin` ("corpus" or "web"), `retrieved_at` and
`web_decision`. `print_report` prints web flags in a separate block under an
explicit warning that they are leads rather than results, and flags them with
their retrieval timestamp. A web passage has no manifest entry, so citation
resolution is skipped for it and its own publisher and URL are used.

### Verified

```
  authority ranking      WHO promoted above a blog returned first  OK
  no cache file written  data/evidence/web_cache.json absent       OK
  inert without backend  search() returns [] with no fetch_fn      OK
  5G window, no backend  NO_EVIDENCE   origin=corpus               OK
  5G window, with stub   NOT_FAKE_NEWS_WEB  origin=web, 1 search   OK
```

`NOT_FAKE_NEWS_WEB` is the right verdict on that window: it describes 5G rollout
without asserting a COVID link, so the WHO page does not contradict it.

### Still open

- The fallback has never run against a **real** search backend. `fetch_fn` must
  be bound to whatever the host provides, and the result shape
  (`url`/`title`/`text`) adapted to it.
- No labelled ground truth for web verdicts, so there is no precision figure for
  them — which is part of why they are reported separately.

## ROUTE_GAMMA re-derived on the dev set (2026-11-04)

`ROUTE_GAMMA` changed **0.52 -> 0.60**. The full measurement is in
docs/routing-experiments.md; recorded here because it is the setting that decides
which evidence reaches the model, and because it reverses a conclusion this file
relied on.

The earlier analysis found routing precision at 100% for every threshold, and
concluded the gate discarded correct routes without preventing any mis-routes.
That was measured on the v1 labels — 36 topical windows over 24 videos, with only
4 videos behind `myocarditis` and 4 behind `natural_immunity`. On the v2 dev set
(64 topical windows, 38 videos, 10-11 videos per topic) precision is 90.6% at low
thresholds and climbs with γ, so the gate does prevent real mis-routes.

The criterion was not routing precision, which is an intermediate quantity. What
decides the verdict is whether retrieval surfaces evidence from the window's own
topic — and since the sub-γ narrow tier was added, a rejected candidate is
searched in a topic-only scope rather than discarded:

```
  gamma   own-topic evidence reached   confident mis-routes
   0.52              53/64                     3
   0.55              55/64                     2
   0.60              55/64                     0
```

0.60 dominates: more windows reach their own topic, and no wrong topic is ever
accepted. Verified that retrieval behaviour on the probe claims is unchanged:

```
  real-clot   tier=blood_clots               top=0.6649
  real-fert   tier=infertility               top=0.8088
  microchip   tier=natural_immunity          top=0.5653
  magnetism   tier=general                   top=0.7362
```

The verdict probe was not re-run (model server down at the time), so the gate
results above still stand on the γ=0.52 retrieval. Since every probe reaches the
same tier with the same top passage, the verdicts should be unchanged, but that is
an inference rather than a measurement.

### Still tuned on test data

`SCOPE_ANSWERED_SIMILARITY` (0.55), `FLAG_MIN_SIMILARITY` (0.45) and
`RELATIVE_CUTOFF` (0.90) were all set from v1 measurements and have **not** been
re-derived on dev. `FLAG_MIN_SIMILARITY` matters most: it already cost one
probably-correct flag (`OqSbJ8TxOx8` w4, overridden at similarity 0.433), and its
value came from the flag distribution of a 46-window run over test videos.
Re-deriving it needs the model server.

## Prompt wording (2026-11-05)

A shorter, more direct variant of `SYSTEM_PROMPT` (293 words against 519, with
the relevance test placed first) was measured against the current prompt on 35
development windows. **Both scored `addresses` accuracy 0.714; the variant was
not adopted.**

It fixed the two failure modes it targeted — flagging a reported concern, and
flagging while its own reason said the evidence was unrelated — but raised
unsupported *clearances* from 5 to 8 while cutting unsupported flags from 3 to 1.
A wrong NOT_FAKE_NEWS is a silent miss under the video-level join rule, so the
trade moves errors from the visible column to the invisible one.

The finding that matters is the level both prompts share: `addresses` is wrong on
10 of 35 windows regardless of wording, which is evidence the relevance
judgement is not instruction-limited at this model size. Full inputs, outputs
and analysis: docs/prompt-experiments.md.
