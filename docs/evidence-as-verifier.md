# Evidence as a verifier

How retrieved evidence earns its place in this pipeline. Measured on the
development set; the test-set figures quoted below come from runs that already
existed when this framing was proposed, and are reported as the observation that
motivated it, not as its confirmation.

## The question

The thesis compares the same model judging each claim window two ways:

- **parametric**: from its own knowledge, no evidence (`--condition parametric`);
- **evidence arm**: against retrieved passages, corpus first and web as
  fallback (`--web tavily`).

The expectation was that the evidence arm would score better. At the video level
it barely does:

```
  dev, gpt-oss-120b, base prompt, 38 videos (8 fake / 30 not)
                       precision  recall   F1    false positives
  parametric              0.28     1.00   0.43         21
  evidence arm            0.31     1.00   0.47         18
```

This document explains why, and where the evidence does pay off.

## Why the two arms look alike

### 1. The video metric hides what evidence does per window

A video is predicted fake news when **any** of its windows is flagged
(tests/run_experiment.py). Counted per window instead, the arms differ sharply:

```
  flagged windows          in fake videos   in not-fake videos
  parametric                    106                 58
  evidence arm                   49                 38
```

The evidence arm withdraws 40 of the parametric arm's flags in not-fake videos,
a third fewer false flags. But one surviving flag still makes a video a false
positive, and a 20-window explainer with one nitpick counts the same as one with
six. The window-level gain mostly disappears in the video count.

### 2. The evidence arm makes errors of its own

Window by window, across all 383 windows:

```
                         evidence arm
  parametric        FLAG   CLEAR   ABSTAIN
  FLAG                63      35        66
  CLEAR               22      76        56
  ABSTAIN              2      14        49
```

Of the evidence arm's flags in not-fake videos, 20 are its own (the parametric
arm did not flag that window): 11 on corpus evidence, 9 on web evidence. Read
individually, they are of two kinds (the dev videos behind each are listed in
the `rules_v2` section of tests/prompt_variants.py):

- **evidence about a different thing** passes the relevance gate. A Pfizer
  efficacy figure is "contradicted" by Moderna's; an mRNA flu vaccine's result
  by a cell-based vaccine's; "vaccines can cause kidney failure" by "flu shots
  are safe for dialysis patients".
- **immaterial details** become contradictions once a precise figure is in
  front of the model: an anaphylaxis rate of "1 in 100,000" against a source
  giving 0.5 per 100,000, or an automatic-transcription spelling of a product
  name.

So the evidence arm removes one set of false positives and introduces a smaller,
different set.

### 3. Evidence cannot add recall here

The parametric arm already finds all 8 fake videos on dev. A 120B model knows
these narratives; with nothing left to find, evidence can only move precision.
The test set shows the same pattern (gpt-oss-120b, 119 videos): recall 0.976
parametric against 0.881 with evidence.

Recall from retrieval would show on claims the model does not know: recent,
local or niche narratives. The development mirror set, built to match the test
set's query distribution, contains almost none.

## Where the evidence pays off: as a verifier

Instead of letting either arm flag on its own, a window is flagged only when
**both** arms flag it. The model proposes; the evidence must confirm. This is
the "generate, then verify against retrieval" pattern (RARR, Gao et al. 2023:
verify the citation before relying on it).

In this role the evidence arm's strength (withdrawing parametric flags) counts,
and its weakness (flags of its own) does not, because a flag the parametric arm
never raised cannot pass.

```
  dev, gpt-oss-120b, base prompt, 38 videos (8 fake / 30 not)
                                  precision  recall   F1    false positives
  parametric                         0.28     1.00   0.43         21
  evidence arm                       0.31     1.00   0.47         18
  verifier                           0.38     1.00   0.55         13
  verifier + overlap veto            0.53     1.00   0.70          7
```

The observation that prompted the framing, on the pre-existing test runs
(gpt-oss-120b, 119 videos, earlier prompt):

```
                       precision  recall   F1    false positives
  parametric              0.500    0.976  0.661        41
  evidence arm            0.481    0.881  0.622        40
  verifier                0.627    0.881  0.733        22
```

### What the verifier is, precisely

It is an agreement rule over two runs of the same model, prompt and dataset
(tests/score_verifier.py): a video is predicted fake when at least one window
is flagged by both. The evidence arm never sees the parametric verdict. That is
weaker than RARR, where the generated claim itself is sent to retrieval, and the
difference should be stated wherever the reference is cited.

Two consequences follow from the design:

- **Recall is capped by the weaker arm on each window.** On dev it costs
  nothing because both arms find all 8 fake videos. On the test set the
  verifier inherits the evidence arm's 0.881.
- **It needs both runs.** Cost and latency double relative to either arm.

### The overlap veto

Claim windows are 140 words with a 70-word stride, so a flagged phrase normally
appears in a second window with different surrounding context. The veto
(`overlap_veto` in rag/pipeline.py, `--overlap-veto`) withdraws a flag when
another window containing the same words judged them NOT_FAKE_NEWS: the typical
case is a myth quoted at the edge of one window and rebutted in the next.
Abstentions do not veto. Letting them veto cost recall in the evidence arm,
where an abstention usually means the neighbour retrieved different evidence,
not that it read the words differently.

It is deterministic and post-hoc, so any finished run can be re-scored with it
(`--score --overlap-veto`, and `tests/score_verifier.py --overlap-veto`).

## How to report it

- Report all three predictors (parametric, evidence arm, verifier) for every
  model, not only the best. The verifier's value is the difference between
  them.
- Report flagged-window counts in fake and in not-fake videos alongside video
  F1. The video join hides most of what retrieval changes.
- State the recall cap and the doubled cost with the verifier figures.
- Keep the web tier. Counting only corpus-grounded flags drops false positives
  from 18 to 10 but loses a fake video (recall 7/8).

## Limits

- **Small dev set.** 8 fake videos: recall moves in steps of 0.125, and one
  false positive moves precision by 0.03 to 0.05.
- **Draft labels.** The dev labels (data/labels/dev_ground_truth.json) were
  drafted under the test set's annotation rules and are pending author review.
- **One model.** All dev figures are gpt-oss-120b. The framing must be checked
  on the other three models before it is claimed in general.
- **Selected on the data it is reported on.** The verifier rule was proposed
  after seeing the 120b test runs. Dev confirms it on separate videos; the
  remaining test-set runs (Gemma, gpt-oss-20b, Qwen) should be scored with the
  rule fixed in advance.
- **Prompts in flux.** These figures use the base prompt (frozen in
  tests/prompt_variants.py, `--prompt base`). The production prompt in
  rag/llm.py is now rules_v2, which targets the evidence arm's own errors, and
  test runs also use the time check (rag/temporal.py). rules_v2 was promoted
  before its dev round finished; if it works, the evidence arm alone should
  begin to beat the parametric arm, and this document should be updated with
  its figures.

## Reproducing

```bash
export LLM_SERVER=openrouter LLAMA_MODEL=openai/gpt-oss-120b
PY=.venv/bin/python
$PY tests/run_experiment.py --dataset dev --prompt base --condition parametric
$PY tests/run_experiment.py --dataset dev --prompt base --web tavily

R=data/results/v2/dev_run_openai-gpt-oss-120b
$PY tests/score_verifier.py ${R}_parametric_base.json ${R}_tavily_base.json
$PY tests/score_verifier.py ${R}_parametric_base.json ${R}_tavily_base.json --overlap-veto
```

"Verifier" in this document is the `window-AND` row of that output. Its
`video-AND` row (both arms flag the video, possibly on different windows) is a
weaker variant and is not used here.
