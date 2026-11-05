# Retrieval methods: dense, BM25 and hybrid

Why the pipeline uses dense retrieval alone, measured rather than assumed. The
companion files cover chunk size (docs/chunking-experiments.md), topic routing
(docs/routing-experiments.md) and adjudication (docs/adjudication-experiments.md).

Implementation: `rag/lexical.py` (BM25 + rank fusion),
`tests/retrieval_compare.py` (this comparison).

## The question

The embedding is `all-MiniLM-L6-v2`, a general-purpose model with no clinical
vocabulary. Measured lay-to-technical gaps on terms this corpus actually uses:

```
  cos("can't get pregnant", "fecundability")                       0.225
  cos("side effects", "adverse events following immunisation")     0.353
  cos("blood clots", "thrombosis with thrombocytopenia syndrome")  0.356
  cos("low platelets", "thrombocytopenia")                         0.537
  cos("heart inflammation", "myopericarditis")                     0.579
  cos("miscarriage", "spontaneous abortion")                       0.589
  cos("immunity from catching it", "infection-acquired immunity")   0.695
```

The embedding bridges morphological near-matches but fails where the technical
term is a different word. The concrete failure: `kwac011` (a prospective cohort
measuring fecundability in both partners) is the corpus's strongest evidence for
the male-fertility myth and ranks **#9** for "men who get the vaccine have lower
sperm counts", while ranking #1 when queried in its own vocabulary.

Since the papers also contain the lay phrases verbatim, lexical matching is the
obvious candidate: BM25 would match "blood clots" directly, with no bridging
needed.

## Method

Metric is the **rank of the best passage containing a labelled phrase**, over the
flat index with no routing. Config-independent: it does not depend on `top_k`, on
a similarity floor, or on which passages a given configuration returned, so it is
comparable across retrieval methods in a way recall@k is not.

33 phrases from 26 labelled windows, corpus at 2488 passages.

**Known bias, stated rather than corrected for:** the phrases were chosen by
reading passages the *dense* retriever surfaced
(docs/chunking-experiments.md), so they favour dense. This matters for small
differences and not for the order-of-magnitude gap below.

## Result

```
method            n     mean  median   top4  top10  top20
dense            33     17.2       3     19     22     23
bm25             33    209.5      50      6      8     10
hybrid (RRF)     33     24.2       8      8     18     21
```

BM25 is an order of magnitude worse, and RRF fusion lands *between* the two
rather than above both.

### It is not a tuning artifact

32 BM25 configurations — k1 in {0.9, 1.2, 1.5, 2.0}, b in {0.3, 0.5, 0.75, 0.9},
with and without stopword removal:

```
  BM25 mean rank across all 32 configs:   204.9 - 211.2
  best hybrid (RRF) mean rank:            21.6   (vs dense 17.2)
  best hybrid (RRF) top4:                 10/33  (vs dense 19/33)
```

A 6-point spread on a mean of ~207. Parameters are irrelevant here; query shape
dominates. Stopword removal made BM25 slightly *worse* (209.5 vs 204.9), which
is consistent with that reading — removing filler does not help when the problem
is that the query is 140 words long.

### Why: the query shape, not the corpus

The hybrid-search literature attributes fusion gains to BM25 recovering rare
exact terms that dense retrievers miss — drug codes, gene names, identifiers —
and reports RRF as hard to beat without per-collection tuning. On BEIR NFCorpus,
the closest biomedical analogue, hybrid beats dense on nDCG@10 (p=0.027).

That mechanism does not transfer, because **this corpus is queried with
transcribed speech**:

```
  labelled windows containing an identifier-like token:  7 / 26
  those tokens: CDC (6), VAERS (2), CBS, EMA, SLC, FDA, ECMO
```

Organisation names, not `BNT162b2`, `ChAdOx1` or `VITT`. Speakers say "the
Pfizer one". There are no rare exact terms for BM25 to recover.

The same literature notes that long conversational queries dilute BM25 across
many low-signal terms while favouring dense retrieval. A labelled query is a
140-word window like:

> "infertility? So every day. Jay Huber is a fertility doctor in New Orleans. He
> says there's no evidence that COVID vaccine causes infertility. [...] There's a
> lot of emotion because I've gone th"

BM25 scores term overlap, so this matches on filler. Dense retrieval compresses
the window to one meaning vector. Direct confirmation — the same claim, short and
padded:

```
  BM25, 8-word query  "men who get the vaccine have lower sperm counts"
      -> kwac011 at rank #2
  BM25, same claim padded to ~50 words of natural speech
      -> kwac011 at rank #31
```

BM25's advantage is real but only for short keyword queries, which this pipeline
never produces.

## Weighted fusion: a near-miss, rejected on held-out data

RRF discards score magnitudes, which lets a mostly-noise ranking contribute
equally. Weighted score fusion over min-max normalised scores
(`alpha * dense + (1 - alpha) * bm25`, best BM25 config) does better:

```
 alpha     mean  median   top4
   1.0     17.2       3     19      <- dense alone
   0.9     15.8       3     19
   0.8     15.6       5     16
   0.5     40.5      13      6
   0.0    206.7      55      4      <- BM25 alone
```

On the full label set α=0.9 improves mean rank from 17.2 to **15.8 with top-4
unchanged at 19/33**. That looks like a free gain, and reporting it as one would
have been a mistake.

Against the dev/test split (tests/split.py):

```
            dev                       test
 alpha   mean  median  top4      mean  median  top4
   1.0   19.3       3     9      15.4       4    10
   0.9   17.2       3    10      14.7       6     9
   0.8   14.8       3     8      16.2       7     8
```

**α=0.8 is best on dev and worse than dense on test. α=0.9 gains 0.7 mean rank on
test while losing a top-4 hit.** With 15 and 18 phrases per side these
differences are noise, and the dev-optimal value does not transfer — which is
exactly the overfitting the split exists to detect. The full-set α=0.9 result was
selection on the evaluation set.

## Decision

**Dense retrieval alone.** BM25 is not wired into the pipeline; `rag/lexical.py`
exists so this comparison is reproducible.

For the TCC this is a tested-and-rejected alternative with an identified cause,
which is a stronger claim than not having tried: the literature's pro-hybrid
consensus rests on a mechanism (rare exact-term recovery, short queries) that
transcribed-speech queries do not exhibit.

## What this does not resolve

The vocabulary gap that motivated the comparison is **still open**. BM25 was the
cheap candidate fix and it failed; the gap is in the embedding, so the remaining
options act on the embedding:

1. **A biomedical embedding** (PubMedBERT, BioLORD, SPECTER). Most direct. Costs
   a re-embed of 2488 passages and a re-run of every sweep in these docs.
2. **LLM query expansion** — restate the lay claim in clinical terms before
   retrieval. Measured to move `kwac011` from rank #9 to #3 on the sperm claim,
   but it makes retrieval nondeterministic unless cached, which matters for
   reproducible thesis numbers.
3. **Per-document lay-language `claim_addressed` text** in the manifest, embedded
   alongside passages. The field already exists in `build_prompt`.

Option 1 is the one with a measurable answer: swap `EMBED_MODEL`, rebuild, re-run
this table and the chunk sweep. Untested so far.

## Caveats

- 33 phrases, 26 windows. Adequate for the order-of-magnitude BM25 finding; not
  adequate for small differences, as the weighted-fusion result demonstrates.
- The labels favour dense by construction (see Method).
- BM25 here is a from-scratch implementation, not Anserini/Pyserini. It is a
  textbook Robertson-Sparck-Jones formulation with an inverted index and was
  swept over 32 parameter settings, but a production implementation with
  stemming and a tuned analyser might differ. The 190-point gap makes that
  unlikely to change the decision.
