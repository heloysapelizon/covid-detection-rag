# Reference corpus: what gets in, and why

The corpus is the evidence the pipeline consults. The claims are in the videos.
Admitting a weak source does not merely add noise: the model would cite it as
ground truth against a video making the very same claim.

## Inclusion criteria

A document is admitted when it is **both** licence-clear and authority-clear.

### Licence

Extracted passages are committed to this repository, so a licence must permit
non-commercial research reuse **and** redistribution of that text:

| Admitted | Excluded |
|---|---|
| CC BY, CC BY-NC, CC BY-SA, CC0 | CC BY-ND / CC BY-NC-ND — NoDerivatives forbids chunking |
| Public-domain government works | All-rights-reserved |
| Standards bodies permitting redistribution (ICH) | Mining-only (Springer Nature TDM: no redistribution) |

PDFs themselves are never committed; `data/evidence/pdf/` is gitignored.

### Authority

Open access and peer review are **not sufficient**. Admitted:

- Public-health authorities and regulators — CDC, WHO, EMA, FDA, ICH
- Systematic reviews and meta-analyses
- Primary studies in established journals with meaningful citation counts

Excluded, regardless of licence:

- Predatory or pay-to-publish venues
- Single case reports offered as general evidence
- Authors and outlets associated with spreading the misinformation under study

**Why this criterion exists.** Searching for spike-persistence sources surfaced
*"Persistence of Vaccine mRNA, Plasmid DNA, Spike Protein… Over 3.5 Years"*
(Medical Research Archives, CC BY, **2 citations**), which argues for a three-year
"washout period". It passes the licence gate cleanly. A comparison review in the
same search area carries **378 citations**. Citation count is not a quality
measure on its own, but a 2-citation paper in a pay-to-publish venue advancing a
claim the dataset's videos also make is the exact failure mode this criterion
prevents: the pipeline would retrieve it and "confirm" the misinformation.

FactCheck.org makes the same point generally — open access and peer review do not
guarantee quality, and a review in an open-access journal may be written by known
misinformation spreaders.

## What counts as evidence for a claim

**A document does not have to refute a claim directly.** An explanation of how the
vaccines work, what they contain, or how they are manufactured is legitimate
justification.

Requiring a point-by-point rebuttal would bias the corpus toward whatever happens
to have been specifically debunked in the literature, and would leave the most
absurd claims — precisely the ones most in need of an answer — uncovered. The
microchip narrative appears in 20 collected videos and no paper tests it, because
it is too physically implausible to study. But a published ingredient list showing
the vaccines contain no electronics, and the CDC statement that they are "free
from metals" and contain no ingredient producing an electromagnetic field, answer
it adequately.

So the corpus holds two kinds of document, and the distinction is about *scope*,
not about strength:

1. **Topic documents** — direct findings on one narrative: fertility outcomes,
   myocarditis incidence, hybrid-immunity comparisons. Routed to by claim.
2. **General documents** — mechanism, composition and manufacture: how mRNA and
   viral-vector vaccines work, what is in them, what becomes of it in the body;
   plus authority myth-sheets. Searched for **every** claim.

A general explanation belongs in `GENERAL` *because* it can justify an answer to
several different claims at once. How mRNA works bears on DNA alteration, on
shedding, on persistence and on magnetism; a composition document bears on
microchips and on metals alike. Filing it under one topic would hide it from the
others.

This is also why a mechanism explainer is never "not evidence". It is evidence
with a wide scope, which is exactly what the general tier is for.

## General-purpose documents

A document mapped to `GENERAL` in `rag/manifest.py` is searched for **every**
claim, whatever that claim routes to. Reserved for authority sources that answer
many narratives by construction — the CDC myth sheet covers magnetism, shedding,
fertility, DNA alteration, test positivity, mandates and catching COVID in three
pages.

Declared, not inferred. Generality was previously derived from the entropy of a
document's passage-topic distribution; that measure flagged 6 of 12 documents at
one threshold and none at another, because it tracks vocabulary spread rather than
editorial role. `classify_documents()` is retained as a diagnostic for spotting a
document whose content has drifted from its declared scope.

## Current coverage gaps

Measured against the claims actually present in the dataset
(docs/chunking-experiments.md Exp. 6): of 24 transcript windows carrying a
checkable claim, the corpus can address 8.

| Gap | Videos affected | What would close it |
|---|---|---|
| Microchips / tracking | 20 | Published ingredient lists; EMA/FDA composition documents |
| Spike-protein persistence | 3 | Biodistribution studies from established journals, not fringe persistence papers |
| Menstrual-cycle effects | 2 | Authority statement or cohort study on menstrual changes |
| Syncytin-1 cross-reactivity | 1 | Immunology review addressing the placental-protein argument |

## Balance pass (2026-10-31)

Target: general ~20%, remaining topics divided evenly. Two steps were taken.

**1. `jcm-11-06272` moved to general.** At 367 passages the narrative review
dominated `natural_immunity` (708 of 1049 passages, 67%). Its content is a broad
survey of infection-acquired and hybrid immunity, durability and reinfection
risk, so general is also the more accurate scope — not only the balancing move.

**2. `blood_clots` filled.** The topic was declared but empty. Three systematic
reviews, each covering a distinct clotting syndrome so they are complementary
rather than redundant:

| File | Publisher | Cites | Licence | Covers |
|---|---|---|---|---|
| `cath-27-10760296211048815.pdf` | Clinical and Applied Thrombosis/Hemostasis | 38 | CC BY-NC | TTS, post hoc analysis |
| `vaccines-10-01444.pdf` | Vaccines (MDPI) | 24 | CC BY | Immune thrombocytopenia |
| `ijms-22-10791-v2.pdf` | Int. J. Molecular Sciences (MDPI) | 23 | CC BY | Coagulopathy mechanism |

All three were downloaded by hand: MDPI and Sage block scripted access, and
spoofing a browser user-agent to defeat that check is not an acceptable method
here. The Sage filename was shortened to `cath-27-<doi-suffix>.pdf` from its
118-character original, which would otherwise dominate every passage id.

### Resulting distribution

```
  total 2301 passages

  general             997   43.3%
  blood_clots         469   20.4%
  natural_immunity    341   14.8%
  myocarditis         301   13.1%
  infertility         193    8.4%
```

General fell from 54.4% to 43.3%, still above the ~20% target. Topic spread
widened rather than narrowed (1.77x → 2.43x) because `ijms-22-10791-v2` alone
contributes 327 passages — a 25-page mechanistic review, far longer than the
other clot papers at 70 and 72.

**The two goals conflict given this document mix.** Evening the topics requires
moving long broad reviews out of topic scopes, and general is the only other
place they can go; so each correction to topic balance raises the general share.
Reaching 20% general would need roughly 4,000 topical passages — on the order of
30 more papers meeting the citation bar. Document length, not document count, is
what drives the imbalance: per-paper yield ranges from 16 passages (CDC sheet) to
367 (`jcm-11-06272`).

Recorded as a measured limit rather than resolved. The 20% figure was set before
the document mix was known, and passage share is not retrieval share: a topic
tier always includes the general documents, so general passages are reachable for
every claim regardless of their proportion.

### Still open

- `infertility` is the smallest topic at 193 passages. Searches found no
  additional papers meeting the citation threshold.
- The three clot papers are unmeasured: no labelled eval window routes to
  `blood_clots`, so their retrieval quality is untested
  (docs/routing-experiments.md).

## Withdrawal on provenance grounds (2026-11-01)

`jcm-11-06272.pdf` — "SARS-CoV-2 — The Role of Natural Immunity: A Narrative
Review" (Diani et al., 2022, doi:10.3390/jcm11216272) — was **removed from the
corpus**. Its authors have since been associated with vaccine-hesitancy
advocacy.

The licence was valid and the citation count met the bar, so this is a criterion
the inclusion rules above did not previously state: **a corpus used to adjudicate
misinformation has to be defensible on authorship, not only on licence and
citation count.** A flag that debunks a hesitancy claim by citing a
hesitancy-aligned source is self-undermining whatever the cited sentence happens
to say — and because this document sat in the general tier, it was in scope for
*every* claim, so any window could have drawn its supporting quote from it.

Added as an explicit criterion:

> **Authorship.** Exclude documents whose authors are publicly identified with
> advocacy against the intervention under study, even where the document itself
> is methodologically sound and openly licensed. The reference corpus is the
> study's standard of truth; its provenance is part of the method, not a
> presentational detail.

This is a conservative rule and it is worth being honest about its cost: it
excludes work on grounds other than content, and applied carelessly it would
shade into excluding inconvenient findings. The narrow justification here is the
document's *role* — it is not being assessed as a scientific contribution but
used as an authority to contradict speakers, which is a function that depends on
the source being unimpeachable to a sceptical reader.

### Replacement

The withdrawn paper was the corpus's largest document at **367 passages (16% of
2301)**. Its role had become general-tier breadth rather than natural-immunity
coverage (it was moved to general in the balance pass above), so it was replaced
with two platform-level reviews rather than another natural-immunity paper:

| File | DOI | Publisher | Cites | Licence | Passages |
|---|---|---|---|---|---|
| `s41392-022-00950-y.pdf` | 10.1038/s41392-022-00950-y | Signal Transduction and Targeted Therapy (Nature) | high | CC BY 4.0 | 293 |
| `IDR-S315727.pdf` | 10.2147/IDR.S315727 | Infection and Drug Resistance (Dove) | 69 | CC BY-NC 3.0 | 134 |

Both are broad mechanism/efficacy/safety reviews spanning vaccine platforms and
adverse-event categories, so they qualify as general by the same reasoning
already applied to `vaccines-09-00467-v2` and `vaccines-11-01118`: they bear on
clotting, myocarditis and efficacy claims alike rather than on one narrative.

The Dove filename was shortened to `IDR-S315727.pdf` from the 94-character
article title it downloaded as, for the same reason the Sage file was shortened
above — the filename becomes the passage-id prefix.

### Resulting distribution

```
  total 2361 passages        (was 2301)

  general            1057   44.8%     (was 997,  43.3%)
  blood_clots         469   19.9%     (unchanged)
  natural_immunity    341   14.4%     (unchanged)
  myocarditis         301   12.7%     (unchanged)
  infertility         193    8.2%     (unchanged)
```

The swap was entirely within the general tier, so **every topic tier is
byte-identical to the previous run** and topic spread stays at 2.43x. General
rose 1.5 points because the replacements yield 427 passages against the 367
withdrawn. The conflict recorded above is unchanged: general remains well above
the ~20% target for the same structural reason.

Verified after the rebuild: 20 PDFs, 20 declarations, 20 passage files, no
duplicate DOIs or filenames, and no content-identical PDFs. `EvidenceIndex.load`
carries a staleness guard that raises if any indexed passage id no longer exists
on disk; it passes, which confirms the withdrawn passages are gone from the
vectors rather than lingering as unreachable rows.

## kwac011 added to infertility (2026-11-01)

| File | DOI | Publisher | Licence | Passages |
|---|---|---|---|---|
| `kwac011.pdf` | 10.1093/aje/kwac011 | American Journal of Epidemiology (Oxford UP / Johns Hopkins) | CC BY-NC 4.0 | 127 |

Wesselink et al., "A Prospective Cohort Study of COVID-19 Vaccination,
SARS-CoV-2 Infection, and Fertility" (2022).

This is the strongest study design in the corpus for the infertility narrative.
The existing `infertility` papers measure proxies — IVF outcomes
(`fendo-15-1356938`) and miscarriage risk (`dead036`) — whereas this is a
**prospective cohort measuring fecundability directly in both partners**, which
is the design the claim actually calls for. It also answers the myth with a
finding rather than a denial: reduced fecundability was observed after recent
*SARS-CoV-2 infection in men*, not after vaccination, so the paper distinguishes
the two rather than simply asserting vaccine safety.

### Resulting distribution

```
  total 2488 passages        (was 2361)

  general            1057   42.5%     (was 44.8%)
  blood_clots         469   18.9%
  natural_immunity    341   13.7%
  infertility         320   12.9%     (was 193,  8.2%)
  myocarditis         301   12.1%
```

**Topic spread improved from 2.43x to 1.56x** — the first real progress against
the balance goal recorded above as a measured limit. `infertility` was the
smallest topic at 193 passages and is now third of four at 320, and the four
topics now span 301–469 rather than 193–469. General fell to 42.5% without any
general document changing, simply by growing the topical share.

This also closes the "still open" item above: `infertility` is no longer the
smallest topic, and it was closed with a better paper rather than merely a
longer one.

### Known limitation, recorded in docs/chunking-experiments.md

This paper is **not retrieved** for the male-fertility claim it best answers:
`cos("lower sperm counts", "fecundability ratio") = 0.178`. The embedding
(`all-MiniLM-L6-v2`) has no clinical vocabulary, so the better a paper's
methodology, the more technical its language and the harder it is to reach from
lay misinformation. Adding stronger papers therefore does not automatically
improve retrieval — a caveat that applies to every future addition, not just
this one.
