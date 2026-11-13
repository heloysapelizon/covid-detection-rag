# The evidence veto

How retrieved evidence is combined with the model's own judgement, and why
this is the combination reported. It supersedes the "verifier" framing in
docs/evidence-as-verifier.md, which remains as the record of how the question
was first approached.

## The rule

Both arms judge the same 140-word claim windows: the **parametric** arm from the
model's own knowledge, the **evidence** arm against retrieved passages (corpus
first, the web for windows the corpus cannot address).

> A video is predicted fake news when at least one window is flagged by the
> parametric arm **and** that same window was not explicitly cleared by the
> evidence arm.

"Cleared" means the evidence arm returned `NOT_FAKE_NEWS` or `NOT_FAKE_NEWS_WEB`
for the window. A window the evidence arm abstained on (`NO_EVIDENCE`) keeps its
parametric flag.

It is applied after the runs, from the stored verdicts. It needs no model call,
and has no parameter to tune.

## Why this role, and not the others

The evidence arm was tried in three roles. Each fails or succeeds for a reason
that is visible in the window verdicts.

| role | rule | what goes wrong |
|---|---|---|
| detector | the evidence arm's own flags | it raises flags of its own the parametric arm did not: precise figures in a passage become "contradictions" (an anaphylaxis rate of 1 in 100,000 against a source giving 0.5), and evidence about a different product, group or measure passes the relevance gate |
| verifier | both arms must flag the same window | any fake video whose narratives the corpus does not cover gets `NO_EVIDENCE` and is lost: on the test set gpt-oss-120b drops from 41 to 33 caught fakes |
| **veto** | parametric flag unless the evidence clears it | — |

The veto gives each part the job it is good at. The parametric arm finds the
misinformation (recall about 0.98 for both models, which the evidence arm
cannot match because the corpus does not cover every narrative). The evidence
removes the parametric arm's mistakes where it can actually speak to them: a
passage showing the speaker was right ("myocarditis after vaccination is real",
"fever lowers sperm count temporarily") clears the flag. Where retrieval finds
nothing relevant, the evidence stays out of the decision instead of costing a
fake.

## Results

All figures with the overlap veto applied to both arms first (a flag an
overlapping window cleared is withdrawn; docs/evidence-as-verifier.md). F1,
then false positives and missed fakes.

| | gpt-oss-120b, dev | gpt-oss-120b, test v1 | Qwen3.6-27B (no thinking), test v1 |
|---|---|---|---|
| videos (fake / not) | 38 (8 / 30) | 119 (42 / 77) | 119 (42 / 77) |
| parametric | 0.516 (15, 0) | 0.745 (27, 1) | 0.828 (16, 1) |
| evidence arm alone | 0.533 (14, 0) | 0.630 (32, 8) | 0.796 (20, 1) |
| verifier (both flag the window) | 0.696 (7, 0) | 0.725 (16, 9) | 0.872 (11, 1) |
| **evidence veto** | **0.615** (10, 0) | **0.763** (18, 5) | **0.863** (12, 1) |

The evidence veto is the only combination that is above the parametric arm in
every column: +0.10 on dev, +0.02 for gpt-oss-120b on test, +0.035 for Qwen. The
verifier is higher in two of the three columns but falls below the parametric
arm for gpt-oss-120b on test, where it loses 8 fakes; the veto loses 4.

Without the overlap veto the order is the same (gpt-oss-120b test: parametric
0.661, verifier 0.733, evidence veto 0.748).

### How it was arrived at

The rule was written down before it was scored, and it has no free parameter.
It was, however, formulated after the test-set results of the other roles had
been read, so it is not a pre-registered hypothesis. The development set
confirms the direction independently (+0.10 F1, no fake lost), on videos that
share no transcript with the test set (docs/dev-sets.md).

## What it does not fix

The remaining false positives for gpt-oss-120b on test are the parametric arm's
flags that the evidence arm did not clear because it abstained: mostly small
medical details in otherwise accurate explainers, simplifications ("spike
proteins can't do anything by themselves"), and, in the whole-transcript runs,
recent events the model does not know ("HHS Secretary Kennedy"). The veto can
only clear a flag when retrieval finds a passage that addresses it.

Larger gains would need new runs: retrieving evidence for each flagged claim
specifically (the claim and the model's stated reason, as in RARR, Gao et al.
2023), instead of judging every window independently.

## Whole-transcript runs

The rule needs both arms to judge the same unit. In the whole-transcript
protocol (rag/whole.py, data/results/v3/) the two arms extract their own claims,
so their verdicts cannot be paired and the veto does not apply; only video-level
agreement (`video-AND`) is defined there.

## v2: tested, not reported

The v2 protocol (the `rules_v2` prompt plus the time check, `--time-check`) was
chosen on the development set and then run on the test set for gpt-oss-120b. It
was worse than v1 in six of the seven predictors below, so it is not reported
as a result. It is recorded here as a tested alternative.

gpt-oss-120b, test set, 119 videos:

| predictor | v1 F1 | v2 F1 | change | v1 false positives | v2 false positives |
|---|---|---|---|---|---|
| parametric | 0.661 | 0.621 | −0.040 | 41 | 49 |
| evidence arm | 0.622 | 0.592 | −0.030 | 40 | 46 |
| verifier | 0.733 | 0.667 | −0.066 | 22 | 32 |
| parametric + overlap veto | 0.745 | 0.695 | −0.050 | 27 | 35 |
| evidence arm + overlap veto | 0.630 | 0.667 | +0.037 | 32 | 32 |
| verifier + overlap veto | 0.725 | 0.716 | −0.009 | 16 | 19 |
| evidence veto + overlap veto | 0.763 | 0.722 | −0.041 | 18 | 27 |

Recall was unchanged in every predictor except the evidence arm with the
overlap veto, where v2 kept 37 fakes against v1's 34; that is the one row it
won.

The likely cause, from reading the new false positives: the v2 rewrite removed
the v1 prompt's worked examples (a fake case, a debunk, an abstention) and its
"key distinction" paragraph, and the added rules did not make up for that. The
development set compared `rules_v2` against the rewritten prompt, never against
the v1 prompt, so it could not show this.

Only gpt-oss-120b completed v2. Qwen3.6-27B was stopped partway (43 of 119
videos, parametric arm only) and is not compared.

The v2 files stay in data/results/v2/ for the record. The production prompts in
rag/llm.py are the v1 prompts again (`v0` in tests/prompt_variants.py), and the
windowed protocol in scripts/run_test_model.sh is the v1 protocol.

## Reproducing

```bash
PY=.venv/bin/python
R=data/results/v1/test_run_openai-gpt-oss-120b
$PY tests/score_verifier.py ${R}_parametric.json ${R}_tavily.json                  # raw
$PY tests/score_verifier.py ${R}_parametric.json ${R}_tavily.json --overlap-veto   # as reported
```

The `evidence-veto` row is the rule above. `window-AND` is the verifier.
