# Data licences

The code is MIT-licensed (`LICENSE`). The data in this repository is not: each
kind keeps the terms of its source, listed here.

## Labels: CC BY 4.0

The annotations in `data/labels/` (`ground_truth.json`, `dev_ground_truth.json`,
`offclaim_key.json`, `offclaim_review.csv` and everything under `windows/`) are
released under the
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
licence. Attribute them to Heloysa Pelizon and cite the repository
(`CITATION.cff`).

The licence covers the annotations: the labels, narratives, confidence scores,
notes, evidence phrases and topic assignments. The transcript excerpts quoted
inside the label files (`window_text`, `flagged_phrase`, `context_*`) are
machine transcriptions of third-party videos, included so the labels can be
read and checked. They are not relicensed; see "Transcripts" below.

## Evidence passages: the licence of each source document

`data/evidence/passages/` holds passages extracted from the 21 documents of the
reference corpus. Each passage file keeps the licence of the document it came
from, recorded in the `license` field of `data/evidence/sources.json`. The
collection as a whole therefore cannot be relicensed: three documents are
non-commercial, and two are not Creative Commons at all.

| passages file | document | licence |
|---|---|---|
| `IDR-S315727` | Overview of the Main Anti-SARS-CoV-2 Vaccines (Infection and Drug Resistance) | [CC BY-NC 3.0](http://creativecommons.org/licenses/by-nc/3.0/) |
| `cath-27-10760296211048815` | COVID-19 Vaccine-Associated Thrombosis With Thrombocytopenia Syndrome (Clinical and Applied Thrombosis/Hemostasis) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `kwac011` | A Prospective Cohort Study of COVID-19 Vaccination, SARS-CoV-2 Infection, and Fertility (American Journal of Epidemiology) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `cdc_108736_DS1` | Myths and Facts about COVID-19 Vaccines (CDC) | US federal government work, public domain |
| `ich-viral-vector-shedding` | ICH Considerations: General Principles to Address Virus and Vector Shedding (ICH) | © ICH; see the [source](https://database.ich.org/sites/default/files/ICH_Considerations_Viral-Vector_Shedding_2009.pdf) for terms of use |
| `cureus-0017-00000098208` | Incidence of Myocarditis and Guillain-Barré Syndrome in Adolescents Receiving COVID-19 mRNA Vaccine (Cureus) | CC BY 4.0 by journal policy; no licence recorded in Crossref, verify on the article page |
| `clc-45-691` | Myocarditis after COVID-19 mRNA vaccination (Clinical Cardiology) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `clinpract-16-00077` | The Impact of COVID-19 Vaccination on Myocarditis Risk and Recovery (Clinics and Practice) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `dead036` | The risk of miscarriage following COVID-19 vaccination (Human Reproduction) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `fendo-15-1356938` | The type of COVID-19 vaccination does not affect reproductive function (Frontiers in Endocrinology) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `ijms-22-10791-v2` | (International Journal of Molecular Sciences) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `pbio.3001506` | (PLOS Biology) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `s41392-022-00950-y` | mRNA vaccine platform review (Signal Transduction and Targeted Therapy) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `s41541-022-00594-7` | (npj Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `s41598-023-47043` | (Scientific Reports) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-09-00467-v2` | (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-09-00989-v2` | (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-10-01316` | Myocarditis or Pericarditis Following the COVID-19 Vaccination in Adolescents (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-10-01444` | (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-11-01118` | (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `vaccines-11-01693` | (Vaccines) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Full titles, DOIs and publishers are in `data/evidence/sources.json`; the
licence URLs come from Crossref. The source PDFs themselves are not in the
repository (`data/evidence/pdf/` is gitignored).

## Transcripts and result files: research use, not relicensed

`data/youtube/*_transcribed.json` and the transcription checkpoints contain
machine transcripts (faster-whisper) of publicly available YouTube videos,
collected through the official Data API. The windowed result files in
`data/results/` repeat that text in 140-word windows (`claim_text`), together
with the model's verdicts. The whole-transcript result files quote the claims
the model extracted.

The transcripts are redistributed for research reproducibility only. Copyright
in the spoken content stays with the videos' creators, and the transcripts are
not covered by the MIT or CC BY licences above. The video metadata files keep
every video id and channel, so any creator can identify their material and ask
for it to be removed. `experiments/redact_text.py --strip` removes the window
text from the result files and `--restore` recomputes it from the transcripts,
so a copy without transcripts stays scoreable once the videos are
re-transcribed with `python -m transcribe`.

## Web-search evidence

The pages retrieved by the web fallback are not redistributed. The cache
(`data/web_cache*/`) is gitignored, and what the repository keeps per run is an
index of the queries made, the URLs returned and the retrieval times
(`data/results/<version>/web_cache_index_*.json`, produced by
`experiments/web_cache_index.py`). The result files quote the passage the
model was shown for each web-sourced verdict (`debunk_text`), under the
source's own terms, as the evidence for that verdict.
