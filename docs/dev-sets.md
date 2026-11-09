# Development sets

How the tuning data was collected and selected, and why the first attempt was
replaced. The split itself is reported by `tests/datasets.py`; the selection is
reproduced by `tests/select_dev.py`.

## Why development sets exist here

45 of the 124 v1 videos had been used to choose seven hyperparameters
(`CHUNK_WORDS`/`CHUNK_STRIDE`, `TOP_K`, `EXPAND_NEIGHBOURS`, `ROUTE_GAMMA`,
`SCOPE_ANSWERED_SIMILARITY`, `FLAG_MIN_SIMILARITY`, `RELATIVE_CUTOFF`), so any
figure computed over them partly measured that tuning rather than the system.
The 124 are now the held-out test set (`data/labels/ground_truth.json`), and
every parameter is derived on separately collected videos.

There is no training set: nothing is fitted on data. The embedding and the
language model are pre-trained and frozen, and "tuning" means choosing
thresholds by reading sweep tables. So the split is two-way — a development set
for choosing configuration, a test set reported once.

## First attempt (dev1) and why it was retired

Collected 2026-11-03 with 11 queries aimed at the four corpus topics, because
`myocarditis` and `natural_immunity` rested on 4 videos each in the v1 labels —
too few for the routing sweep to support a rate. 252 candidates, 45 selected, 41
transcribed successfully.

It worked for what it was built for. Measured on its 64 topical windows across 38
videos, the routing sweep overturned a v1 conclusion: precision is 90.6% at low
thresholds and rises with gamma, where v1 had reported 100% everywhere. That v1
result was an artifact of 4 videos per thin topic.

**It was retired because it could not exercise the widening path at all.** Every
one of its 80 windows retrieved corpus evidence above the floor — minimum
best-similarity 0.4838 against a `MIN_SIMILARITY` of 0.25 — so:

```
  set         n    min best-sim   median   empty retrieval
  v1 (test)  81      0.2010       0.6504     4  (4.9%)
  dev1       80      0.4838       0.7525     0  (0.0%)
```

Zero empty-retrieval windows means three parameters had no basis:
`SCOPE_ANSWERED_SIMILARITY` and `RELATIVE_CUTOFF` both govern what happens when
a narrow scope fails, and the web-search trigger fires only when the corpus
returns nothing. All three were set to inert values for lack of evidence
(docs/routing-experiments.md), and the fallback's stance condition was never
reached.

The cause was the collection design: queries aimed at the four corpus topics
produce windows that are on-corpus by construction.

## Second attempt (dev2): two groups, kept apart

Collected 2026-11-04. Two groups with different jobs, which must not be pooled.

**Mirror group** — the six v1 queries verbatim, so the development set has the
same query distribution as the test set and retrieval figures transfer between
them. This is the only group used to choose a parameter.

**Uncovered group** — queries aimed at narratives the evidence corpus does not
cover: graphene oxide, plandemic and depopulation, excess mortality, cancer, 5G,
shedding, suppressed treatments. Deliberately unrepresentative, and used only to
exercise the web-search fallback.

Mirroring alone would not have fixed the dev1 gap. v1's own empty-retrieval rate
is 4.9%, so on ~45 videos a faithful mirror yields 3-4 fallback windows — too few
to measure anything. The uncovered narratives in the test set arrived
incidentally, through "covid vaccine truth" and "dangers" plus a curated link
list, not through any query aimed at them:

```
  which v1 query found the 13 uncovered-narrative test videos?
    8  covid vaccine truth
    3  link_list:disinformation_dozen
    2  covid vaccine dangers
```

The raw collection returned **39 mirror against 158 uncovered**, which is why the
groups are separated in code (`tests/select_dev.py --group`) rather than by
convention: pooling them would let the unrepresentative half dominate every
retrieval figure.

### Reaching 50 mirror videos

The first dev2 pull returned only 39 mirror candidates, because the exclusion
list by then held 376 already-used ids and `MAX_RESULTS_PER_QUERY` was 25.

`MAX_RESULTS_PER_QUERY` was raised to 50 and the six mirror queries re-run,
giving 135 candidates. Deeper paging on the *same* queries keeps the mirror
faithful; adding synonym queries would have changed the distribution the mirror
exists to replicate.

50 rather than 45 because transcription loses videos: dev1 lost 4 of 45 (~9%) to
non-English audio and a download error. At that rate 50 yields ~45 usable.

## Selection criteria

`tests/select_dev.py` balances on the axes that affect what can be measured.
Selection is deterministic — candidates are ordered by SHA-1 of their video id
within each query bucket, so re-running gives the same subset with no stored seed
and no sensitivity to input order.

**Round-robin across queries, not proportional.** The raw pools are uneven (10
candidates for one mirror query against 29 for another), and proportional
allocation would let a productive query crowd out a sparse one.

**At most 2 videos per channel.** Several clips from one broadcaster share a
presenter, format and stock phrasing, so they behave as one observation rather
than several. The dev1 pool had 7 videos from a single broadcaster.

**Duration 1-12 minutes preferred, not required.** A tiebreak only: long enough
to yield several claim windows, short enough to transcribe quickly and to sit
under the transcription cap in `transcribe/config.py`. Nothing is excluded for
length.

**Exclusion is enforced, not assumed.** `data/youtube/exclude_dev2.json` holds
the 376 ids used by the test set (124) and the dev1 pool (252). `select_dev.py`
drops them before selecting and fails if any survives into the output.

Results:

```
  mirror     50 of 135   8-9 per query across all six   46 channels   median 3.5 min
  uncovered  21 of 158   3 per query across all seven   21 channels   median 3.4 min
```

Verified zero overlap between mirror, uncovered, the dev1 pool and the test set.

## Transcription results (2026-11-04)

```
  mirror     44 usable of 50   33,434 words   median 576w   2 hit the 12-min cap
             failures: 3 download errors, 3 non-English (tl, hi, yo)
  uncovered  not yet transcribed
```

Attrition was 12%, against 9% on dev1 — which is why 50 were selected for a
target of ~45. Per-query coverage survived: 5-9 usable videos for each of the six
mirror queries.

**The mirror set reproduces the test set's retrieval profile**, which is what it
was collected for:

```
  set         windows   min best-sim   median   empty retrieval
  v1 (test)        81       0.2010     0.6504     4  (4.9%)
  dev1             80       0.4838     0.7525     0  (0.0%)
  dev2 mirror     456       0.1595     0.5857     8  (1.8%)
```

So the widening path is exercised at last, though 8 windows is thin — enough to
check that `SCOPE_ANSWERED_SIMILARITY` and `RELATIVE_CUTOFF` do something, not
enough to fix a precise value. The uncovered group exists for that.

### A checkpoint bug found during this run

`transcribe/data.py` routed a dataset to its own checkpoint only when its name
did *not* start with `covid_vaccine_videos` — but every dataset in this project
does, so all of them shared `transcripts_checkpoint.json`. The guard its own
docstring described never fired.

The transcription worked; the output did not appear. Transcripts were written to
the shared checkpoint while `build_enriched` looked only for ids in the named
dataset, so the run produced no `_transcribed.json` and `tests/datasets.py`
reported zero. The shared checkpoint had accumulated 143 v1 + 44 dev1 + 27
mirror entries.

Fixed to match on the exact dataset stem. The mirror entries were migrated to
their own checkpoint rather than re-transcribed, and the shared checkpoint was
reduced to the 143 v1 ids — verified still 124 usable, so the test data was
never at risk.

## Reproducing

```bash
# collect (SEARCH_QUERIES in collect/config.py selects which pull)
python -m collect --name covid_vaccine_videos_dev2 \
    --exclude data/youtube/exclude_dev2.json

# split the combined pull into its two groups
python tests/select_dev.py --group mirror    --n 50 --write
python tests/select_dev.py --group uncovered --n 21 --write

# transcribe (capped at TRANSCRIBE_MAX_S; pass --max-transcribe 0 for whole videos)
python -m transcribe covid_vaccine_videos_dev2_mirror_selected
python -m transcribe covid_vaccine_videos_dev2_uncovered_selected
```

The reproducible artifacts are the **pool files**, not the queries: YouTube search
is not reproducible, so re-running collection returns a different candidate set.
Given the committed pools, selection is exact.

## What was deleted

The dev1 pool and its selected, transcribed and labelled files were removed on
2026-11-04 once dev2 replaced them. `exclude_dev2.json` is retained and is what
keeps dev2's non-overlap with dev1 verifiable — it contains the 252 dev1 pool ids
as a flat list, so the guarantee survives the deletion of the pool itself.

The dev1 *findings* are kept: the routing measurement that overturned the v1
100%-precision result is in docs/routing-experiments.md, and the zero-fallback
measurement that motivated dev2 is above and in docs/web-search.md.

## Topic queries reintroduced, deliberately (2026-11-04)

The mirror-only development set closed one gap and opened two others. Both are
visible in its own measurements:

- **Chunk size stopped discriminating.** 55w and 90w tied at 0.590 and 0.594,
  where every earlier set had shown a clear 55w win. Mirroring broad queries
  produces broad windows — 61 of 85 made no single-topic claim — and on a window
  like "are the vaccines safe?" many passages are plausibly relevant, so chunk
  size barely matters.
- **Per-topic coverage collapsed.** `blood_clots` and `natural_immunity` had one
  video each, so no per-topic routing figure was possible.

The first development set had the opposite profile: 64 topical windows, 10 videos
for `blood_clots`, but only 16 general windows and zero empty-retrieval windows.
Neither set alone can do both jobs.

**So the topic queries were reintroduced and the two label sets combined.** This
is deliberate, not an accident of collection: the mirror group supplies the test
set's window distribution, and the topic group supplies the depth per topic that
chunk size and routing need.

The archived first development set was reused rather than re-collected. Its 41
transcripts and its 80 fully labelled windows were preserved when it was retired,
and verified to have zero overlap with the test set, the mirror group and the
uncovered group — so restoring it cost nothing and saved both a collection and
about three hours of transcription. A fresh topic collection (261 candidates) was
run before the archive was checked and has been set aside unused.

### The combined development set

```
  windows            163     (83 mirror + 80 topic)
  videos              84
  labelled phrases    53

  by topic           windows   videos
    general               77       47
    infertility           32       20
    myocarditis           27       15
    blood_clots           16       10
    natural_immunity      11        7
```

`source_group` on each window records which collection it came from, so a figure
can be computed over either group alone when the distinction matters.

### Retrieval profile

```
  set                windows   min best-sim   median   empty retrieval
  v1 (test)               81       0.2010     0.6504     4  (4.9%)
  dev1 (topic)            80       0.4838     0.7525     0  (0.0%)
  dev2 mirror            456       0.1595     0.5857     8  (1.8%)
  dev2 uncovered         171       0.1721     0.5291     3  (1.8%)
```

The 1.8% figure is the important one. It holds for the mirror group *and* for the
uncovered group collected specifically to break it, because the corpus almost
never returns nothing — 2,488 passages over a shared vaccine vocabulary mean
something clears a 0.25 floor even for a turbo-cancer or 5G claim. That is what
made the original web-search trigger ("retrieval returned no passage")
unreachable, and why it now keys on the adjudicated verdict instead
(docs/web-search.md).

### What the combination fixed

```
                          mirror only      combined
  chunk sweep 55w            0.590           0.699
  chunk sweep 140w           0.543           0.638
  55w wins clearly?            no             yes
  topical windows               22              86
  general windows               61              77
  labelled phrases              23              53
```

Both gaps closed: the chunk ordering is decisive again, and `ROUTE_GAMMA = 0.30`
is now measured on 86 topical and 77 general windows rather than 22 and 61
(docs/routing-experiments.md).

## Video-level dev labels (2026-11-09)

Prompt comparison needs a verdict to score against, and the window labels above
only record corpus coverage. `data/labels/dev_ground_truth.json` adds
video-level fake/not-fake labels for the 44 transcribed mirror videos, under
the test set's annotation rules, read from the transcripts as the pipeline
receives them (capped where `ok_capped`).
Six are excluded through a per-item `exclude` field:

```
  content duplicates of test videos (different video id, same words)
    Ti5SpsYxkGo   = test Z6qlcRGzh-Q / -9U3HTacDP8   overlap 1.00
    fnFC7lnHjU8   = test Z6qlcRGzh-Q / -9U3HTacDP8   overlap 0.91
    Z3g_CijXgUI   = test DjMqylDmbP8                  overlap 0.80
    qFR3TlR0SmI   = test giS6CiT1qTE                  overlap 0.49
  off scope (no human-vaccine content)
    A22eAcrsjqQ   pet vaccine and microchip clinic
    Sdr-ZCs1yzU   zoonotic spillover, no vaccine claim
```

The id-based leak check passed all four duplicates: they are the same
syndicated segments re-uploaded by other stations. `tests/datasets.py` now also
reports transcript overlap with test (8-word shingles, threshold 0.4), so a
future duplicate shows up as `DUP` rather than silently entering tuning.

Kept: **38 videos, 8 fake_news / 30 not_fake_news.** The positive class is
small, so recall and F1 move in steps of one video; the false-positive count
over the 30 negatives is the more stable figure for prompt comparison.
