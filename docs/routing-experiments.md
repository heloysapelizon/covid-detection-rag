# Routing experiments

Measurements behind `ROUTE_FLOOR`, `GENERAL_ENTROPY_MIN` and
`GENERAL_TOP_SHARE_MAX` in [`rag/config.py`](../rag/config.py).

(`ROUTE_MIN_SIMILARITY` was a deprecated alias for `ROUTE_FLOOR`; it had no
remaining callers and was removed on 2026-11-05.)

Runs on 2026-10-28, conda env `tcc` (Python 3.10.20),
embeddings `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine),
corpus 1,348 passages from 12 documents across 6 topics.

**Why route at all.** Retrieval over the whole corpus lets a large single-subject
document outrank the right evidence. With 74 of 90 passages from the ICH shedding
guidance, a claim about vaccine magnetism retrieved ICH study-design text instead
of the CDC's answer. Narrowing the scope before ranking removes that competition.

**The risk routing introduces.** A mis-routed claim cannot reach the evidence that
would answer it — the failure is silent and unrecoverable. So below some
confidence the router must decline to commit and leave the scope general.

---

## Experiment 1 — document generality, derived not declared

**Why.** Some documents cover one narrative; others (an authority myth sheet, a
broad review) span many and should stay in scope for any claim. Rather than
hardcoding a `general` flag, generality is measured.

**Setup.** Each passage is matched to the topic description it most resembles;
the spread of those assignments per document is its Shannon entropy (bits).
A document is general when entropy is high **and** no single topic dominates.

```
  GENERAL  pbio.3001506.pdf                 H=2.51 share=0.23 viral_shedding
  GENERAL  s41541-022-00594-7.pdf           H=2.13 share=0.37 natural_immunity
  GENERAL  s41598-023-47043.pdf             H=1.98 share=0.47 natural_immunity
  GENERAL  jcm-11-06272.pdf                 H=1.92 share=0.58 natural_immunity
  GENERAL  cdc_108736_DS1.pdf               H=1.88 share=0.50 myths_and_facts
  GENERAL  cureus-0017-00000098208.pdf      H=1.75 share=0.60 myocarditis
  specific vaccines-11-01693.pdf            H=1.63 share=0.67 natural_immunity
  specific vaccines-10-01316.pdf            H=1.45 share=0.70 myocarditis
  specific clinpract-16-00077.pdf           H=1.35 share=0.72 myocarditis
  specific ich-viral-vector-shedding.pdf    H=1.31 share=0.69 viral_shedding
  specific fendo-15-1356938.pdf             H=1.08 share=0.81 infertility
  specific clc-45-691.pdf                   H=0.84 share=0.87 myocarditis
```

**Result.** The measure orders documents sensibly — the CDC myths sheet (8 distinct
myths) at H=1.88, a single-subject cardiology paper at H=0.84. Thresholds
`GENERAL_ENTROPY_MIN = 1.50` and `GENERAL_TOP_SHARE_MAX = 0.60` were set when the
corpus held 3 documents.

**Open.** At 12 documents those thresholds flag **6 of 12** as general, which is
too many for "general" to mean much — it was meant for the authority sheet and
broad reviews, not half the corpus. Needs re-tuning against labelled routing
outcomes, not by eye.

---

## Experiment 2 — can the collection query serve as routing ground truth?

**Why.** Scoring a threshold needs to know each window's true topic. The obvious
candidate is the search query the video was collected under.

**Result: no.** A video found under `covid vaccine infertility` makes claims about
microchips, DNA and shedding too, and many of its windows make no claim at all
(intros, sponsor reads, filler). The query describes what was *searched for*, not
what a given 140-word window *says*.

Ground truth therefore has to be per-window and hand-assigned. The `topic` field
in [`tests/chunk_eval_labels.json`](../tests/chunk_eval_labels.json) holds it,
alongside the `relevant_evidence` labels the chunking evaluation uses — same
windows, two labels. Valid values: the six topic names, or `general` for a window
that makes no topical claim or spans several.

---

## Experiment 3 — what the router currently does, unlabelled

**Why.** Before labelling, see how the router behaves on 13 real windows and which
confidence signal looks usable.

**Setup.** `tests/chunk_eval_labels.json` windows, routed with no threshold.

```
gcExM22XkYs w1  top=infertility      abs=0.756 margin=0.050
gcExM22XkYs w0  top=infertility      abs=0.763 margin=0.028
hEOO5Qzc8cQ w0  top=infertility      abs=0.871 margin=0.152
hEOO5Qzc8cQ w1  top=infertility      abs=0.843 margin=0.132
JxIkndSVRDI w0  top=infertility      abs=0.640 margin=0.013
JxIkndSVRDI w1  top=infertility      abs=0.239 margin=0.014   <- filler
F-LD9pvPXQE w0  top=infertility      abs=0.475 margin=0.075
MniP4EtW7xI w8  top=infertility      abs=0.779 margin=0.103
MniP4EtW7xI w9  top=infertility      abs=0.716 margin=0.106
zg-Sh7gYXxk w0  top=infertility      abs=0.778 margin=0.124
l2POvDQu96c w11 top=infertility      abs=0.682 margin=0.063
l2POvDQu96c w0  top=myths_and_facts  abs=0.656 margin=0.090   <- show intro
b3-BCUmEYg0 w0  top=infertility      abs=0.747 margin=0.234
```

**Two windows a threshold must catch:**

- `JxIkndSVRDI` w1 — *"theorists were talking about when the VAX was coming out.
  Yeah. Well, it's like..."* — no claim, `abs=0.239`
- `l2POvDQu96c` w0 — *"Hi, and welcome to another edition of Your Health with Dr.
  Christy..."* — a show intro, routed to `myths_and_facts` at `abs=0.656`.
  Confidently wrong, and the more dangerous of the two.

**Margins are uniformly tiny** (0.013–0.234). Topic descriptions overlap — every
COVID-vaccine topic resembles every other — so the gap between first and second
carries little signal. Absolute score separates far better: 0.239 for filler
against 0.871 for a clear claim.

---

## Experiment 4 — where the labels themselves are ambiguous

Three windows sit on the boundary, because the `viral_shedding` description ends
"…how mRNA and viral-vector vaccines work", so any mechanism explainer matches it:

```
  [4]  "the genetic code of a portion of one, either in the form of DNA or
        what's called messenger RNA..."                        abs=0.618
  [25] "Coronavirus mRNA vaccines teach our immune system to make these
        antibodies. How do mRNA vaccines do that?"             abs=0.732
  [27] "with mRNA vaccines, you can produce them much more quickly..."  abs=0.602
```

**Resolved: all three stay `general`.** They describe how the technology works or
how fast it can be manufactured; none asserts something a fact-checker would
adjudicate. The rule this establishes: a neutral mechanism explainer is `general`,
however strongly its vocabulary matches a topic. Only a window making a checkable
claim gets a topic.

That keeps the labels consistent, and it sharpens rather than softens Experiment
4's finding — window [25] scores **0.732**, the highest confidence of any window in
the sample, while asserting nothing at all. The signal is measuring topical
vocabulary, exactly as the false-commit list shows.

---

## Settings and their status

Current values, with where each was last measured.

| Setting | Value | Status |
|---|---|---|
| `ROUTE_GAMMA` | 0.60 | **re-derived on the v2 dev set** — see the final section |
| `ROUTE_FLOOR` | 0.30 | below this a topic is not a candidate; not separately swept |
| `ROUTE_TOP_N` | 3 | not measured |
| `GENERAL_ENTROPY_MIN` | 2.60 | superseded in practice: generality is now *declared* in `rag/manifest.py`, not inferred. `classify_documents` is kept as a diagnostic |
| `GENERAL_TOP_SHARE_MAX` | 0.60 | same |

Three earlier threshold sweeps were removed from this file (2026-11-04). They
scored routing on the v1 labels, which were drawn from the videos that are now
the held-out test set, and on corpora that no longer exist (12, 16 and 19
documents, against 21 now). Their conclusions were superseded by the dev run at
the end of this file, which reverses the central one. The reasoning that
survives them is in Experiments 1-3, 5 and 6.

## Experiment 5 — what the literature says routing is for

Experiments 3–5 scored routing against a per-window `topic` label, counting a
`general` window that routed confidently as an error. That was the wrong frame.
**Routing measures topical similarity, not whether a claim is present** — so window
25 ("How do mRNA vaccines do that?") scoring 0.732 against `viral_shedding` is the
router working, since that text genuinely is about how mRNA vaccines work. The
`general` label encodes "no checkable claim", which is a different question and
belongs to a different component.

### The reference corpus has nothing on this

Searched all 1,348 passages for retrieval/NLP method terms: 18 hits, every one a
false positive — systematic-review screening ("full texts were retrieved and
assessed") and IVF ("transvaginal ultrasound-guided oocyte retrieval"). The corpus
is clinical evidence about vaccines; it cannot ground a retrieval-architecture
decision. That grounding has to come from IR/NLP literature.

### What that literature does

**Confidence-gated routing with a broader fallback is a named pattern.** Sun et al.,
*Asymmetric Dynamic Routing* (arXiv 2609.29282), accept a predicted route when
confidence `c ≥ γ` and otherwise redirect to a middle trajectory:

> `ŷ = y·I(c ≥ γ) + 2·I(c < γ)`, where γ = 0.75 by default. "Moderate is selected
> as the fallback because it provides an intermediate retrieval scope, reducing the
> risk of routing uncertain queries to either extreme."

This is the design asked for here — below a threshold, decline to narrow — and it
confirms the fallback should be a *broader* scope rather than no retrieval.

**Their threshold sensitivity, measured on downstream quality, not routing
accuracy:**

| γ | Effect reported |
|---|---|
| 0.50 | Logicality score drops 1.14 points vs 0.75 — ambiguous queries accepted and mis-routed |
| **0.75** | "favorable balance between routing reliability and computational efficiency" |
| ≥ 0.85 | 68.4% of queries fall back; +58 ms latency, no significant gain (p > 0.05) |

The shape matches Experiment 4 — too low admits noise, too high collapses into
fallback — and their 0.75 sits in the same region as the 0.55–0.60 range where this
corpus started trading coverage for precision. The absolute values are not
comparable (different embedding, different confidence definition), but the
trade-off curve is the same one.

**Routing is evaluated on what it buys, not on label agreement.** Bansal and
Agarwal, *Lightweight Query Routing for Adaptive RAG* (arXiv 2604.03455), do report
classification metrics (macro-F1 0.928, accuracy 93.2% with TF-IDF + SVM over 7,727
queries) — but pair every figure with the downstream consequence, 28.1% token
savings against always using the most expensive paradigm, and explicitly note their
perfect-label reference "is not an upper bound on savings." Routing accuracy is a
proxy; the quantity that matters is the effect on the final answer.

Notably, **lexical TF-IDF features beat MiniLM sentence embeddings by 3.1 macro-F1
points** on their task — "surface keyword patterns are strong predictors". Worth
remembering, since routing here uses MiniLM embeddings of hand-written topic
descriptions.

**Widening scope can actively hurt.** The CHSR-RRF educational-RAG work notes that
when some on-topic results exist but fall below a minimum count, widening the scope
"would replace on-topic results with off-topic results from a larger candidate
pool" — so it declines to widen. Relevant to the general-document fallback here: a
general document staying in scope is cheap, but relaxing the topic filter entirely
is not free.

### What this changes

1. **The threshold cannot be chosen from `topic` labels alone.** Experiment 4's
   "false commits" are mostly windows with no claim, which routing is not
   responsible for detecting. Scoring it that way understates the router.
2. **The right metric is downstream:** does routing at γ retrieve the evidence that
   lets the model reach the correct verdict, versus not routing at all? That needs
   the `relevant_evidence` labels in `chunk_eval_labels.json`, which measure
   whether the right passages were retrieved — not the `topic` labels.
3. **Claim detection is a separate component**, as the literature's separation
   implies. Filler, intros and ads (34 of 58 windows here) should be filtered
   before routing, not by it.

### Sources

- Sun, Zhang, Li, Li, Guo — *Asymmetric Dynamic Routing: Balancing Reasoning Depth
  and Computational Efficiency in Hypergraph RAG*, arXiv:2609.29282
- Bansal, Agarwal — *Lightweight Query Routing for Adaptive RAG: A Baseline Study
  on RAGRouter-Bench*, arXiv:2604.03455
- Wang et al. — *RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG
  Routing*, arXiv:2602.00296
- *CHSR-RRF: A curriculum-gated hybrid retrieval framework…*, arXiv:2609.02913
- Reimers, Gurevych — *Sentence-BERT*, 2019 (the MiniLM family used here)

---

## Re-tuned on the v2 dev set (2026-11-04)

### Why this run supersedes the γ analysis above

Every γ figure in this file was computed on labels drawn from the v1 videos —
the same videos that are now the held-out test set
(data/labels/ground_truth.json). 45 of those 124 videos were used to choose
seven hyperparameters, so any figure over them partly measures that tuning.

The v2 dev set is a separate collection (11 queries aimed at the thin topics,
collected with `--exclude data/labels/ground_truth.json`, 41 videos transcribed
of 45 selected). It has never influenced a threshold.

```
                    v1 labels        v2 dev labels
  topical windows        36                   64
  topical videos         24                   38
  myocarditis videos      4                   11
  natural_immunity        4                   10
```

### The v1 conclusion was an artifact

v1 reported **100% routing precision at every threshold**, from which this file
concluded the gate "prevents no mis-routes; it only discards correct ones". On
dev that does not hold:

```
   thresh  routed  precision  missed  general kept
     0.30      64      90.6%       0      0/16
     0.45      63      90.5%       1      5/16
     0.52      50      94.0%      14      8/16
     0.57      39      97.4%      25     10/16
     0.60      30     100.0%      34     11/16
```

Precision is 90.6% at low thresholds and rises with γ, so the gate does prevent
real mis-routes. Six genuine ones, concentrated in one topic:

```
  mIzuuYB6GrI w0   natural_immunity -> blood_clots       0.593
  BWZYd1YAB7o w5   natural_immunity -> infertility       0.558
  2riTrIQTaaw w6   natural_immunity -> myocarditis       0.525
  WTdorGWyCGs w2   natural_immunity -> infertility       0.479
  WTdorGWyCGs w1   natural_immunity -> infertility       0.453
  tXMna_7NYko w2   infertility      -> natural_immunity  0.476
```

With 4 natural_immunity videos, v1 had too few to expose this. The
100%-precision finding was a sample-size artifact, not a property of the router.

### The right criterion is not routing precision

Routing accuracy is an intermediate quantity. What matters downstream is whether
retrieval surfaces the passage that answers the window. Measured end to end on
the 64 dev topical windows, under the final design — general documents in every
tier, ranking preferring topical passages by `TOPIC_RANK_BONUS`, and no sub-γ
tier (docs/adjudication-experiments.md):

```
  gamma   own-topic reached   labelled-phrase recall   confident mis-routes
   0.30         56/64                 0.667                     6
   0.40         55/64                 0.667                     6
   0.50         51/64                 0.633                     3
   0.60         33/64                 0.500                     0
```

**γ = 0.30.** A high threshold does cut mis-routes to zero, but discards correct
routes far faster: at 0.60 only half the labelled phrases remain reachable. The
six mis-routes at 0.30 do lose their own topic — but the same **8 windows** fail
at 0.30 and at 0.60, not merely the same count, so a higher threshold was not
protecting against them.

0.30 equals `ROUTE_FLOOR`, so in practice every candidate clearing the floor
leads a tier. The gate is retained rather than deleted because it is the
mechanism the design is built on (Sun et al., arXiv:2609.29282) and a corpus with
more topics may need it; at this corpus size it does not earn a value above the
floor.

Note this is the opposite of what plain F-measure would choose, and also the
opposite of the 0.60 this file briefly recommended on 2026-11-04. That earlier
figure was measured when a sub-γ tier existed to catch rejected candidates; once
general documents were carried by every tier, the tier became redundant and the
threshold that depended on it moved with it.

`ROUTE_GAMMA` changed 0.52 -> 0.60 -> 0.30 over two days of measurement.

### Chunk size holds

```
      chunk   psg  recall    prec  hit rate  coverage  mean top1
  55w/ 35s  2488   0.795   0.108     0.700     1.000     0.7757
  90w/ 60s  1585   0.629   0.071     0.567     1.000     0.7764
 140w/ 90s  1136   0.735   0.079     0.700     1.000     0.7673
 220w/150s   815   0.594   0.058     0.567     1.000     0.7515
```

55w/35s wins on independent labels. The margin is much narrower than on v1
(0.795 vs 0.735 for 140w here, against 0.733 vs 0.351 there), which is the
expected consequence of removing the selection bias: the v1 phrases were chosen
by reading passages the 55w configuration had retrieved, so that configuration
was being scored against its own output. The ordering survives; the gap was
inflated.

Config-independent rank metric on the dev phrases: mean 45.8, **median 1**,
top-4 21/30, top-10 24/30. Median and top-4 are better than v1 (median 3, 19/33)
while the mean is worse, which is the signature of unbiased labels — most
phrases rank first, and a few genuinely hard ones sit far down the list rather
than being excluded from the label set for being hard.

### Caveats

- 30 labelled phrases. Enough to confirm the chunk ordering, not enough for
  small differences — the same limit that made the weighted-fusion result in
  docs/retrieval-methods.md untrustworthy.
- 16 general windows is few for measuring how often the gate correctly declines
  to narrow. The general share is lower than v1 (16/80 vs 45/81) because the v2
  queries deliberately targeted topical claims.
- `SCOPE_ANSWERED_SIMILARITY`, `FLAG_MIN_SIMILARITY` and `RELATIVE_CUTOFF` have
  **not** been re-derived on dev. They were set from v1 measurements and remain
  tuned-on-test numbers until that is redone.


## Parameter derivation on the dev set (2026-11-04, final)

Every retrieval parameter, derived on the v2 dev labels only. The rule applied
throughout: **a parameter gets a value when the dev set discriminates between
values, and otherwise gets the value that makes it inert.** Hand-written probe
claims are not evidence for a setting — several of them are test-set narratives,
so tuning against them reintroduces the contamination the split exists to remove.

### Final values

| Parameter | Value | Basis |
|---|---|---|
| `CHUNK_WORDS` / `CHUNK_STRIDE` | 55 / 35 | dev sweep: recall 0.795 vs 0.594-0.735 |
| `TOP_K` | 4 | not re-derived; inherited |
| `EXPAND_NEIGHBOURS` | 1 | not re-derived; inherited |
| `ROUTE_GAMMA` | 0.30 | dev: recall flat 0.30-0.40, collapses by 0.60 |
| `TOPIC_RANK_BONUS` | 0.03 | dev: recall flat 0.00-0.04, falls beyond |
| `SCOPE_ANSWERED_SIMILARITY` | = `MIN_SIMILARITY` | **no dev basis** — inert |
| `RELATIVE_CUTOFF` | 0 | **no dev basis** — inert |
| `FLAG_MIN_SIMILARITY` | 0 | dev: every floor removes only correct flags |

Dev performance at these values: own-topic evidence reached for 56 of 64 topical
windows, labelled-phrase recall 20/30 = 0.667.

### The two parameters with no basis

`SCOPE_ANSWERED_SIMILARITY` decides whether a tier "answered" or the search
should widen. Measured across its whole plausible range:

```
  floor   own-topic   phrase recall
   0.30     55/64        0.667
   0.45     55/64        0.667
   0.55     55/64        0.667
   0.65     55/64        0.667
```

Identical everywhere, and only 5 of 80 windows retrieve anything different
between 0.30 and 0.65. Once general documents are carried by every tier and
ranking prefers topical passages, this floor has nothing left to do. Set equal to
`MIN_SIMILARITY`, which makes it a no-op.

An earlier version of this section set it to 0.55 on the grounds that a lower
value broke four hand-written probe claims (magnetism, shedding, DNA alteration,
microchips). That reasoning is withdrawn: those probes were written by hand, and
magnetism and microchips are test-set narratives, so they are not admissible
evidence for a parameter the dev set is supposed to fix. The dev set says the
parameter does not matter, and that is the finding.

`RELATIVE_CUTOFF` keeps only passages scoring within a ratio of the best hit.
Phrase recall is 0.667 at every ratio from 0.80 to 0.95 **and** with the cutoff
disabled; mean set size falls from 4.00 to 3.10 with no recall effect. It was
briefly justified at 0.90 on a single verdict that differed between runs — which
turned out to expose nondeterminism in the model server rather than an effect of
the setting. Disabled.

Both remain in `rag/config.py` as settings rather than being deleted, so they can
be re-enabled if a larger corpus makes them discriminate.

### Why the dev set cannot settle them

Both parameters govern *widening* — what happens when a narrow scope fails. The
dev labels were collected with queries aimed at four corpus topics, so almost
every dev window has a topic that answers it and the widening path is rarely
taken (2 of 80 windows reach the `all` tier at any floor). Measuring a
widening threshold needs windows whose evidence is NOT in a topic tier, and the
dev set has 16 general windows and no uncovered narratives.

That is a known limitation of this dev set rather than a property of the
parameters. Recorded so that a future collection aimed at general-tier and
uncovered narratives can settle them.

## Re-run on the dev2 mirror set (2026-11-04)

The mirror group repeats the six v1 queries, so this is the first tuning set with
the test set's query distribution. 44 usable videos, 85 labelled windows.

### What this set is good and bad for

```
                       dev1        dev2 mirror
  topical windows        64             22
  general windows        16             61
  blood_clots videos      9              1
  natural_immunity        10             1
  empty retrieval       0.0%           1.8%
```

The two sets are complementary rather than one superseding the other. Mirroring
broad queries produces broad windows: 61 of 85 make no single-topic claim, and
`blood_clots` and `natural_immunity` have one video each. So this set **cannot**
re-derive a per-topic routing rate — but it is the only set that can say what
gamma does to general-claim windows, which dev1 could not with 16 of them.

### Gamma = 0.30 confirmed, from the opposite direction

The concern about a low gamma was that general-claim windows would be pushed into
a topic scope and lose the authority documents. Measured over 61 general windows:

```
  gamma   topical own-topic   general windows reaching general evidence
   0.30         21/22                       59/61
   0.40         20/22                       59/61
   0.50         19/22                       60/61
   0.60         14/22                       60/61
   0.65         12/22                       61/61
```

**The concern does not materialise.** General windows reach general evidence at
every gamma, because every tier now carries the general documents
(`TOPIC_RANK_BONUS`); committing to a topic scope no longer withholds them. The
cost of a high gamma is one-sided: topical windows lose their own topic, falling
from 21/22 to 14/22.

Note the `general kept` column in the routing table reads the opposite way — only
1 of 61 general windows declines to commit to a topic at gamma=0.30, against 60
at 0.60. That column measures the routing *decision*, not what retrieval
returned, and the two diverge precisely because general documents are no longer
withheld from a topic tier. The decision is now largely cosmetic; what reaches
the model is what matters.

`ROUTE_GAMMA = 0.30` stands.

### Chunk size: the dev1 result does not reproduce

```
      chunk   psg  recall    prec  hit rate  coverage  mean top1
  55w/ 35s  2488   0.590   0.089     0.304     1.000     0.7769
  90w/ 60s  1585   0.594   0.075     0.435     1.000     0.7721
 140w/ 90s  1136   0.543   0.075     0.304     1.000     0.7534
 220w/150s   815   0.568   0.087     0.304     1.000     0.7408
```

55w and 90w are **tied** (0.590 vs 0.594), and 220w is close behind at 0.568. The
clear 55w win on dev1 (0.795 vs 0.629) and on v1 (0.733 vs 0.361) is absent.

The config-independent rank metric is also much worse here:

```
  set          phrases   mean   median   top4
  v1 (test)         33   17.2        3   19/33
  dev1              30   45.8        1   21/30
  dev2 mirror       23  114.1       20    6/23
```

This is a property of the windows, not a retrieval regression. A broad window
("are the vaccines safe?", "what are the side effects?") has many plausibly
relevant passages, so the single phrase chosen as the label is often not the
top-ranked one — whereas a topical window ("myocarditis in young men") has a
narrow correct answer. The labels are honest; the metric is just less
discriminating on broad claims.

**`CHUNK_WORDS = 55` is retained, but on weaker evidence than before.** 55w is
not beaten on either set, and it wins decisively where windows make specific
claims. On broad windows the chunk size barely matters, which is itself
informative: chunk tuning is only worth doing against specific claims.

### What is still unsettled

23 labelled phrases over 22 topical windows is too thin for anything finer than
an ordering. `SCOPE_ANSWERED_SIMILARITY` and `RELATIVE_CUTOFF` remain inert: this
set has 8 empty-retrieval windows (1.8%), enough to confirm the widening path is
exercised but not to fix a threshold on it.

## Final γ measurement, combined development set (2026-11-04)

The mirror-only and topic-only sets each measured one side of the γ trade-off.
Combining them (docs/dev-sets.md) gives 86 topical and 77 general windows over 84
videos — enough to measure both at once, end to end:

```
  gamma   topical own-topic   general reaching general ev   phrase recall
   0.30         77/86                   68/77                  0.472
   0.40         75/86                   69/77                  0.472
   0.50         68/86                   71/77                  0.453
   0.55         56/86                   73/77                  0.434
   0.60         44/86                   73/77                  0.358
   0.65         28/86                   75/77                  0.245
```

**γ = 0.30 confirmed.** All three columns now point the same way: topical
own-topic retrieval and labelled-phrase recall both fall monotonically as γ
rises, while general windows gain only 7 of 77 across the entire range. The gain
on general claims is small because every tier carries the general documents, so
committing to a topic scope no longer withholds them.

This supersedes both earlier readings. The v1 figure (γ=0.52) was measured on
labels drawn from the test videos. The first dev reading recommended 0.60, but
that was measured while a sub-γ tier existed to catch rejected candidates; once
general documents were carried by every tier the tier became redundant, and the
threshold that depended on it moved with it.

Routing precision over the same set rises from 93.0% at low thresholds to 100% at
0.60 — so the gate does reject real mis-routes, as the first dev set established.
It is still not the criterion: the six mis-routes it prevents cost 33 correctly
routed windows, and the same windows fail at both settings.

`ROUTE_GAMMA = 0.30` (= `ROUTE_FLOOR`), measured on 163 windows across 84 videos.
