# Chunking experiments

> **Experiments 1-5 used a paper later removed from the corpus** (all rights
> reserved) and a 12-document corpus. Their evidence base is gone. They are kept
> for the method and for the negative result in Experiment 2; the current result
> is the dev-set sweep at the end of this file.
>
> Six intermediate re-runs were removed on 2026-11-04. Each re-scored the sweep
> after a corpus change (12 -> 16 -> 19 -> 20 -> 21 documents) against the v1
> labels, which were drawn from the videos that are now the held-out test set.
> All reached the same conclusion as the dev run, by a route that no longer
> reproduces.

Measurements behind `CHUNK_WORDS`, `CHUNK_STRIDE` and `EXPAND_NEIGHBOURS` in
[`rag/config.py`](../rag/config.py).

All runs on 2026-10-27, conda env `tcc` (Python 3.10.20),
embeddings `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).

**Metric.** `gold@3` — whether the paper's actual finding appears in the top 3
retrieved passages, matched against sentences such as *"evidence favors rejection
of a causal relation"*, *"no effect of COVID-19 vaccines on fertility"*, *"no
detection of such autoantibodies"*. Similarity score alone is not the metric: a
high score against the wrong passage is worse than a lower score against the right
one.

---

## Experiment 1 — chunk size on a research paper

**Why.** The first chunk size (55 words) was tuned on the CDC FAQ, whose Q&A pairs
are already claim-sized. Research papers have argumentative paragraphs, so the
result may not transfer.

**Setup.** `experiments/chunk_eval.py`. One paper indexed alone, so no other document
competes: *Female Infertility and COVID-19 Vaccines* (NCBI Bookshelf, 10 pages,
3,948 words). Queried with 7 items — 5 real transcript windows from the dataset
(3 asserting the myth, 2 debunking it) and 2 mechanism queries.

```
       chunk  passages  mean top1  min top1  gold@3
--------------------------------------------------------
  55w/ 35s        63     0.7422    0.6540    5/7
  90w/ 60s        38     0.7182    0.6237    7/7
 140w/ 90s        25     0.7029    0.5857    7/7
 220w/150s        17     0.6965    0.5824    7/7
 350w/250s        11     0.6941    0.5458    7/7
```

**Result.** Similarity and recall move in opposite directions. 55w has the
highest mean score (0.7422) but misses the finding on 2 of 7 queries; every size
from 90w up reaches 7/7 at lower scores.

**Where 55w fails.** Per-query detail (`GOLD` = finding present in top 3):

```
--- 55w/35s  (63 passages) ---
  assert    JxIkndSVRDI  0.8103 GOLD In evaluating the literature on whether COVID-19 vaccines im
  assert    F-LD9pvPXQE  0.6641 GOLD one of very few studies to include Ad26.COV2.S. The study wa
  assert    generic      0.8199      whether COVID-19 vaccines affected female infertility (Chama
  debunk    gcExM22XkYs  0.7316      In evaluating the literature on whether COVID-19 vaccines im
  debunk    Ts0E-v3zya0  0.6540 GOLD female infertility (Wesselink et al., 2022). None of the stu
  mechanism syncytin     0.8088 GOLD Syncytin-1, a crucial membrane glycoprotein, facilitates the
  mechanism amh          0.7069 GOLD 2023). In a self-controlled study of women without known inf

--- 220w/150s  (17 passages) ---
  assert    JxIkndSVRDI  0.8103 GOLD In evaluating the literature on whether COVID-19 vaccines im
  assert    F-LD9pvPXQE  0.5824 GOLD had a prior history of female infertility (Wesselink et al.,
  assert    generic      0.8216 GOLD had a prior history of female infertility (Wesselink et al.,
  debunk    gcExM22XkYs  0.7316 GOLD In evaluating the literature on whether COVID-19 vaccines im
  debunk    Ts0E-v3zya0  0.6261 GOLD had a prior history of female infertility (Wesselink et al.,
  mechanism syncytin     0.6980 GOLD Syncytin-1, a crucial membrane glycoprotein, facilitates the
  mechanism amh          0.6056 GOLD median AFC (Yang et al., 2023). Results were aggregated. Bas
```

The `generic` query is the clearest case: 55w scores **0.8199** on
*"whether COVID-19 vaccines affected female infertility (Chama..."* — a citation
fragment. The sentence stating the conclusion was split across chunk boundaries.
Small chunks match the topic sharply while cutting findings in half.

---

## Experiment 2 — the same chunk sizes on the CDC FAQ

**Why.** To check whether the original tuning document could have detected this.

**Setup.** `cdc_108736_DS1.pdf` (3 pages, 744 words) indexed alone. 4 queries,
each with the exact answer sentence it should retrieve.

```
      chunk  psg  mean top1  gold@3
  55w/ 35s   19     0.6428  4/4
  90w/ 60s   11     0.6325  4/4
 140w/ 90s    8     0.5930  4/4
 220w/150s    6     0.5488  4/4
```

**Result.** 4/4 at every size — the FAQ is **insensitive** to this parameter,
because each Q&A pair is already one claim. Tuning on it alone was uninformative,
which is why the 55w choice looked safe and was not.

---

## Experiment 3 — neighbour expansion instead of a larger chunk

**Why.** Experiment 1 forces a trade: precision (small chunks) against complete
findings (large chunks). Expansion tests whether both are obtainable — retrieve on
small chunks, then include the adjacent chunks so the surrounding argument travels
with the hit.

**Setup.** Same paper and 7 queries as Experiment 1, chunks fixed at 55w/35s,
comparing top-3 alone against top-3 plus each hit's ±1 neighbours.

```
  JxIkndSVRDI  plain=GOLD  expanded=GOLD  (9 psg)
  F-LD9pvPXQE  plain=GOLD  expanded=GOLD  (7 psg)
  generic      plain=----  expanded=GOLD  (9 psg)
  gcExM22XkYs  plain=----  expanded=GOLD  (6 psg)
  Ts0E-v3zya0  plain=GOLD  expanded=GOLD  (7 psg)
  syncytin     plain=GOLD  expanded=GOLD  (7 psg)
  amh          plain=GOLD  expanded=GOLD  (7 psg)

  gold@3 plain:    5/7
  gold@3 expanded: 7/7
```

**Result.** Expansion recovers both misses — **5/7 → 7/7** — while keeping 55w's
higher scores. Strictly better than enlarging the chunk, which reached 7/7 only by
giving up precision.

---

## Experiment 4 — expansion against the full index

**Why.** Experiments 1–3 index one paper alone. This confirms the behaviour holds
when documents compete, and that neighbours never cross a document boundary.

**Setup.** Live index, 151 passages from 3 PDFs (CDC 16, ICH 74, infertility
paper 61). Query: *"the covid vaccine causes infertility and sterilises women
permanently"*.

```
=== expand=0 (hits only) ===
  0.8722 [nbk-female-infertility ord=54] These studies have been summarized in few systematic reviews.
  0.8699 [nbk-female-infertility ord=23] Studies among females going through in vitro fertilization. Th
  0.8569 [nbk-female-infertility ord=59] measures of infertility and exposure to vaccines were both sel
  0.8547 [nbk-female-infertility ord=55] The studies reviewed reported no effect of COVID-19 vaccines o

=== expand=1 (hits + neighbours) ===
  ctx 0.8722 [ord=53] female infertility (Wesselink et al., 2022). None of the s
  HIT 0.8722 [ord=54] These studies have been summarized in few systematic revie
  ctx 0.8699 [ord=22] In evaluating the literature on whether COVID-19 vaccines
  HIT 0.8699 [ord=23] Studies among females going through in vitro fertilization
  ctx 0.8699 [ord=24] of female infertility. Female infertility was not an outco
  ctx 0.8569 [ord=58] infertility. Very few studies examined Ad26.COV2.S and fem
  HIT 0.8569 [ord=59] measures of infertility and exposure to vaccines were both
  ctx 0.8569 [ord=60] Adverse Effects of COVID-19 Vaccination and Intramuscular
  HIT 0.8547 [ord=55] The studies reviewed reported no effect of COVID-19 vaccin
  ctx 0.8547 [ord=56] lack of an adverse impact on ovarian function further sugg

source files in result: {'nbk-female-infertility-covid-vaccines.pdf'}
```

**Result.** The hit at `ord=54` pulls in `ord=53`, containing *"None of the studies
reported an a[dverse effect]..."* — gold evidence that plain top-4 missed, now
present. All 10 passages come from one document, so expansion respected the
boundary. Scores are higher than in Experiment 1 (0.87 vs 0.81) because the paper
is indexed against competitors rather than alone.

---

## Experiment 5 — fallback queue

**Why.** When a verdict comes back `NO_EVIDENCE`, the next-best passages should be
available rather than the pipeline giving up.

**Setup.** Same query, `skip` set to round 1's passage ids.

```
round 1 (top-4):
  0.8722 ord=54 These studies have been summarized in few systematic rev
  0.8699 ord=23 Studies among females going through in vitro fertilizati
  0.8569 ord=59 measures of infertility and exposure to vaccines were bo
  0.8547 ord=55 The studies reviewed reported no effect of COVID-19 vacc

round 2 (fallback, skipping round 1):
  0.8531 ord=2  Infertility is defined as not being able to conceive aft
  0.8516 ord=53 female infertility (Wesselink et al., 2022). None of the
  0.8458 ord=22 In evaluating the literature on whether COVID-19 vaccine
  0.8416 ord=52 one of very few studies to include Ad26.COV2.S. The stud

overlap between rounds: 0 (want 0)
```

**Result.** Round 2 returns four fresh passages at comparable scores
(0.8416–0.8531), zero overlap. The queue works.

`FALLBACK_ROUNDS` was never wired into `pipeline.py` and was removed on
2026-11-05: the web-search tier it was held for is keyed on the adjudicated
verdict instead (docs/web-search.md). The `skip` parameter this experiment
exercised is retained on `EvidenceIndex.search`, uncalled, because it is the
mechanism any future elimination loop would use.

---

## Settings these produced

| Setting | Value | From |
|---|---|---|
| `CHUNK_WORDS` | 55 | Exp. 1 — highest precision; Exp. 3 fixes its recall gap |
| `CHUNK_STRIDE` | 35 | Exp. 1 — ~36% overlap between adjacent chunks |
| `EXPAND_NEIGHBOURS` | 1 | Exp. 3 — ±1 reaches 7/7; Exp. 4 confirms on the full index |
| ~~`FALLBACK_ROUNDS`~~ | removed | Exp. 5 showed the retry works, but it was never wired up; removed 2026-11-05 |

## Caveats

- One paper (Exp. 1, 3) and one FAQ (Exp. 2). Two document types is enough to show
  the FAQ-only tuning was biased, not enough to claim 55w+expansion is optimal
  across the 21-document corpus. Worth re-running once the remaining 18 PDFs are
  downloaded.
- Gold markers were chosen by reading the paper, so `gold@3` measures retrieval of
  passages *this author judged* to be the finding.
- Only the infertility topic is covered. Myocarditis, blood clots, shedding and
  natural immunity are untested.
- Exp. 1 and 2 are reproducible via `python experiments/chunk_eval.py` (Exp. 1 only; the
  CDC comparison and Exps. 3–5 were run inline and are recorded here as output).

---

## The two 40-word filters: what they discard (2026-10-31)

`extract_pdf` applies `min_words = 40` twice, and the two cases turned out to be
very different. Measured over all 19 documents (125,038 words kept):

| Filter | Discarded | Words | Verdict |
|---|---|---|---|
| A. whole section shorter than 40w | 1,957 sections | 13,286 | kept — mostly table debris |
| B. chunk fragment shorter than 40w | 190 fragments | 5,826 | **was a bug, now fixed** |

### B was a real defect

**All 190 dropped fragments were the last chunk of their section.** Because the
stride (35) is shorter than the window (55), the final window of a section is
usually a short remainder — so the filter was systematically truncating the end
of every section whose length did not divide evenly.

Overlap does not rescue this. Verified on synthetic sections: the dropped tail
contains words that appear in **no other chunk**.

```
  section  100w -> 2 kept chunks, word coverage 90/100   (positions 90..99 lost)
  section  130w -> 3 kept chunks, word coverage 125/130  (positions 125..129 lost)
  section  200w -> 5 kept chunks, word coverage 195/200  (positions 195..199 lost)
```

Some of the lost text was substantive, e.g. from `ijms-22-10791-v2.pdf`:

> additional serious conditions have been reported in association with
> vaccination against SARS-CoV-2, including single cases of capillary leakage
> syndrome and coronary myocarditis after immunization with ChAdOx1 nCoV-19
> ... and BNT162b2 ..., respectively.

**Fix:** `chunk()` now folds a short final remainder into the preceding window
(`tail_min`, default 40) instead of emitting it to be dropped. Word coverage goes
to 100%, the passage count is unchanged at 2301 (text is absorbed into existing
windows, not added as new ones), and the largest window grows to at most 65 words.

Effect on the recovered content — the capillary-leak passage for a direct query:

```
  before:  not in the corpus at all
  after :  rank #1 of 2301  (sim 0.5576)
```

### Effect on the labelled metrics

Measured with a **rank** metric rather than recall@4, because the labelled
phrases were chosen from passages the pre-fix configuration already ranked
highly, which biases any top-k measurement toward the old behaviour:

```
  drop   2301 passages  mean_rank 16.4  median 3  top4 21  top10 23  top20 25
  merge  2301 passages  mean_rank 16.1  median 3  top4 20  top10 23  top20 25
```

Statistically indistinguishable. The headline `--limit 100` recall moved 0.761 →
0.733, and that drop is **label-selection noise, not a regression**: one phrase
("79 reports of thrombosis associated with low platelets") fell out of top-4
because a passage from the newly added TTS paper now outranks it at 0.7197 vs
0.6740 — a better retrieval outcome scored as a miss. The phrase still ranks #10
of 2301.

Kept regardless of the flat metrics: the labelled phrases all come from
mid-section prose, so **none of them exercises the recovered tails**. The fix is
justified by coverage (100% vs 95–99% of section text reaching the index), not by
a metric these labels can move.

### A is doing its job

The 13,286 words look alarming but are mostly not prose. Classifying each
dropped section by the fraction of tokens containing a 3+ letter word:

```
  table-ish (<50% prose)   1294 sections   8,604 words
  mixed     (50-80%)        306 sections   3,134 words
  prose     (>80%)          357 sections   1,548 words
```

57.7% of the dropped sections are 5 words or shorter. These are table cells,
figure captions, author lists and page furniture that `_HEADING_RE` mis-split
into "sections" — numeric rows like `11.00 ± 5.59 11.71 ± 4.77 12.16 ± 5.71`
would embed as noise and compete for rank.

Only **22** are genuine prose of 20+ words, and most of those are mid-sentence
fragments created by the heading regex splitting a paragraph, not standalone
content.

Merging each short section into the preceding one was tested and **rejected**:

```
  section-drop   2301 passages  mean_rank 16.1  top4 20  top20 25
  section-merge  2594 passages  mean_rank 17.5  top4 21  top20 24
```

293 more passages, slightly worse mean rank — the table debris does compete, as
the classification predicted. Filter A stays.

### Residual known loss

~1,548 words of short prose, in 357 fragments, remain discarded. The cause is
`_HEADING_RE` over-splitting prose rather than the word floor itself; fixing it
properly means a better heading detector, not a lower threshold.


## Re-run on the v2 dev set (2026-11-04)

The current result, and the first on labels that never influenced a setting.

The v1 labels came from the videos now held out as the test set
(data/labels/ground_truth.json), and their `relevant_evidence` phrases were
chosen by reading passages the 55w/35s configuration had retrieved — so that
configuration was partly being scored against its own output. The v2 dev labels
are a separate collection (41 videos, 30 labelled phrases) and carry no such
bias.

```
labelled windows : 30
evidence PDFs    : 21
transcript window: 140w / 70s
retrieval        : top_k=4 expand=1 min_sim=0.25

      chunk   psg  recall    prec  hit rate  coverage  mean top1
  55w/ 35s  2488   0.795   0.108     0.700     1.000     0.7757
  90w/ 60s  1585   0.629   0.071     0.567     1.000     0.7764
 140w/ 90s  1136   0.735   0.079     0.700     1.000     0.7673
 220w/150s   815   0.594   0.058     0.567     1.000     0.7515
```

**55w/35s holds**, so `CHUNK_WORDS = 55` and `CHUNK_STRIDE = 35` stand.

The margin is much narrower than the v1 runs reported (0.795 vs 0.735 for 140w
here; the removed runs showed 0.733 vs 0.351). That narrowing is the selection
bias disappearing, and it is the more honest figure. Note also that `mean top1`
no longer declines monotonically with chunk size — 90w edges 55w (0.7764 vs
0.7757) — so similarity alone does not pick the configuration; recall does.

Config-independent rank metric on the dev phrases:

```
  phrases 30   mean 45.8   median 1   top4 21/30   top10 24/30
  (v1 was:     mean 17.2   median 3   top4 19/33)
```

Median and top-4 are better while the mean is worse. That is the signature of
unbiased labels: most phrases rank first, and the few genuinely hard ones sit far
down the list instead of being excluded from the label set for being hard.

### Caveats

- 30 phrases. Enough to confirm the ordering, not enough for small differences —
  the same limit that made the weighted-fusion result in
  docs/retrieval-methods.md untrustworthy.
- The dev labels skew topical by design (the v2 queries targeted the thin
  topics), so this sweep says less about general-tier retrieval than the v1 runs
  did.

## Re-run on the combined development set (2026-11-04)

The mirror-only run above showed 55w and 90w tied (0.590 vs 0.594), which did not
reproduce the clear 55w win of every earlier set. That tie was a property of the
windows: mirroring the broad v1 queries produced 61 general-claim windows out of
85, and on "are the vaccines safe?" many passages are plausibly relevant, so chunk
size barely matters.

Reintroducing the topic queries (docs/dev-sets.md) restores topical windows and
with them the discrimination:

```
labelled windows : 53
evidence PDFs    : 21
transcript window: 140w / 70s
retrieval        : top_k=4 expand=1 min_sim=0.25

      chunk   psg  recall    prec  hit rate  coverage  mean top1
  55w/ 35s  2488   0.699   0.099     0.528     1.000     0.7762
  90w/ 60s  1585   0.612   0.073     0.509     1.000     0.7745
 140w/ 90s  1136   0.638   0.077     0.528     1.000     0.7613
 220w/150s   815   0.580   0.070     0.453     1.000     0.7469
```

**55w/35s wins by 0.06 over the next configuration**, on 53 labelled phrases
across 84 videos — the largest and most balanced development set so far. The
margin is narrower than the v1 runs claimed (0.733 vs 0.361) because those labels
were drawn from passages the 55w configuration had retrieved; it is wider than
the mirror-only run because that set could not discriminate at all.

`CHUNK_WORDS = 55` and `CHUNK_STRIDE = 35` stand, and the earlier caveat that
they rested on weakened evidence is withdrawn.

What survives from the mirror-only run is the narrower finding: **chunk size only
matters for specific claims.** On broad general-claim windows all four
configurations are within 0.05 of each other, so tuning chunk size against such
windows is not informative — which is why a development set mixing both window
types is necessary rather than merely convenient.
