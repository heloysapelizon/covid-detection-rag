# Adjudication prompt experiments

Measured comparison of the adjudication system prompt against a shorter, more
direct variant, on development-set windows only. Every input and output is
recorded below so the comparison can be re-read without re-running it.

## Why this test was run

The web-fallback smoke test (docs/web-search.md) produced flags whose cited
evidence did not support the stated reasoning. Three flags on one video cited a
paper titled *"The type of COVID-19 vaccination does not affect reproductive
function and pregnancy"* to refute claims about **death rates**, an **adverse
event rate** and a **risk-benefit ratio in the over-90s**. The model wrote
plausible reasons describing content those passages do not contain — one said
the evidence "states the vaccine is effective and well-tolerated in the elderly
population", which a reproductive-function paper does not say.

Two further flags landed on text that asserts no checkable fact at all
(*"the serious harm is there"*; an invasion-of-privacy opinion), and one flagged
a claim the model's own reason called **true**.

Those failures share a shape: the relevance gate passed evidence that does not
address the claim. The hypothesis under test is that the prompt itself invites
this, because it is long and it defines the flagging verdict before it defines
relevance.

## The two prompts

### Current (519 words)

The `addresses` field — the relevance gate — is first defined **48% of the way
through**, after the full FAKE_NEWS definition including the clause *"It counts
even if the speaker is vague, hedging, or embedding the claim in a longer
story."*

```
  You are a fact-checking classifier. You are given EVIDENCE from public health authorities and a CLAIM transcribed from a video.

  Decide whether the speaker is asserting something the evidence contradicts.

  Verdicts:
  - "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the evidence contradicts. This includes stating that a vaccine contains an ingredient the evidence says it does not contain, or that it has an effect the evidence says it does not have. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.
  - "NOT_FAKE_NEWS": the speaker agrees with the evidence, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.
  - "NO_EVIDENCE": the evidence provided does not address what the speaker says. Use this whenever the evidence is silent on the claim's specific subject, EVEN IF you personally believe the claim is false. Saying something is false without evidence that addresses it is not fact-checking, and an unsupported true answer is still an error here.

  Key distinction: a speaker who DENIES a myth ("vaccines do not cause infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.

  Before deciding, determine the speaker's STANCE toward the false idea. Look for negations and denials: "isn't", "aren't", "not possible", "no reason to suspect", "there's no evidence", "that's a myth", "doesn't". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches the evidence. Topic overlap alone is never enough to flag.

  Reply with a JSON object using EXACTLY these six keys, in this order:
    "addresses"   - "yes" or "no". Does the EVIDENCE discuss the specific subject of the CLAIM? Answer this FIRST, before deciding anything else. Share of a general topic is not enough: evidence about vaccine ingredients does not address microchips, and evidence about spike-protein immunology does not address blood clots. If the specific subject is absent, answer "no".
    "stance"      - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear
    "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". If "addresses" is "no" you MUST answer "NO_EVIDENCE" — you may only answer "FAKE_NEWS" or "NOT_FAKE_NEWS" when "addresses" is "yes". Use FAKE_NEWS only when stance is "asserting".
    "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string
    "evidence_id" - the id of the evidence passage you relied on, e.g. "CDC-MAGNETIC"
    "reason"      - one sentence explaining the verdict

  Examples:
  {"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker presents magnetism as real; the CDC states vaccines are free from metals."}
  {"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is debunking the magnetism myth, which agrees with the evidence."}
  {"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", "claim": "", "evidence_id": "", "reason": "The speaker claims the vaccine contains graphene oxide; the evidence does not discuss graphene oxide, so it cannot support a verdict either way."}

  Use the key "verdict". Do not use "decision" or any other name. JSON only.
```

### Variant `v2_direct` (293 words)

Four deliberate changes:

1. **The relevance test comes first**, at 20% through rather than 48%. A
   left-to-right decoder reads the flagging rule before the relevance rule in
   the current prompt, which primes flagging before relevance has been
   considered.
2. **Relevance is posed as a concrete test** — "does the evidence mention the
   specific thing?" with three worked contrasts (ingredients/microchips,
   pregnancy/death rates, myocarditis/heart attacks) — rather than a rule with
   four qualifying clauses.
3. **The hedging encouragement is dropped**, and replaced with an explicit
   instruction not to flag opinions, questions, value judgements or text too
   vague to check. The original clause has no counterweight and is a plausible
   cause of flags on non-assertions.
4. **One worked example per verdict**, down from three dense ones.

```
  You check whether a speaker says something the EVIDENCE contradicts.

  STEP 1. Does the evidence mention the specific thing the speaker is talking about?
  Not the general topic — the specific thing. Evidence about vaccine ingredients does not mention microchips. Evidence about pregnancy does not mention death rates. Evidence about myocarditis does not mention heart attacks.
  If the specific thing is missing, answer addresses="no" and verdict="NO_EVIDENCE". Stop there. Do not use what you already know.

  STEP 2. Only if the evidence does mention it: is the speaker saying it is TRUE or FALSE?
  - Saying it is true  -> stance="asserting"
  - Saying it is false -> stance="denying"  (this is NOT fake news)
  - Just describing or asking -> stance="neutral"

  STEP 3. The verdict.
  - "FAKE_NEWS"     - the speaker says it is true, the evidence says otherwise.
  - "NOT_FAKE_NEWS" - the speaker agrees with the evidence, or is debunking the false idea.
  - "NO_EVIDENCE"   - anything else, including every case from STEP 1.

  Flag a statement of FACT. Do not flag an opinion, a question, a value judgement, or a sentence too vague to be checked.

  Reply with JSON, exactly these six keys in this order:
  {"addresses": "yes"|"no", "stance": "asserting"|"denying"|"neutral", "verdict": "FAKE_NEWS"|"NOT_FAKE_NEWS"|"NO_EVIDENCE", "claim": "the false words copied from the CLAIM, or empty", "evidence_id": "the id you used, or empty", "reason": "one sentence"}

  Examples:
  {"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker says magnetism is real; the evidence says vaccines contain no metals."}
  {"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is rejecting the magnetism claim."}
  {"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", "claim": "", "evidence_id": "", "reason": "The speaker talks about graphene oxide; the evidence never mentions graphene oxide."}

  JSON only.
```

## Method

**Dev-set cases only.** The failing cases that motivated this test are test-set
videos, so they are used as the hypothesis and never as the decision: the
standing rule is that a parameter gets its value from development data
(docs/dev-sets.md). The 15 smoke-test flags appear nowhere in the scoring below.

**No new labelling was needed.** The dev chunk-evaluation labels carry
`relevant_evidence`, a hand-made judgement of whether the corpus genuinely
covers each window — which is the same question `addresses` asks. So:

```
  relevant_evidence non-empty  ->  addresses should be "yes"
  relevant_evidence empty      ->  addresses should be "no"
```

**Stratified, so a prompt cannot win by flagging less.** Half the cases are
windows the corpus does cover, where withholding a verdict is a miss; half are
windows it does not, where a flag is unsupported. A prompt that simply answers
"no" to everything scores badly on the first half.

Cases are drawn from both dev groups (mirror + topic, 165 windows), ordered
deterministically by SHA-1 of `video_id#window` and spread round-robin across
topics, matching the selection discipline in `tests/select_dev.py`.

**Identical retrieval.** Retrieval runs once per case and both prompts receive
the same passages, so every difference is attributable to the prompt. The
override exists as the `system_prompt` parameter on `adjudicate()`; the pipeline
never passes it.

Configuration:

```
  model              qwen2.5:14b
  temperature        0.0
  server             Ollama (USE_GRAMMAR=False, USE_JSON_MODE=True)
  TOP_K              4  (+ neighbour expansion, so ~10 passages reach the prompt)
  MIN_SIMILARITY     0.25
  CLAIM_WINDOW_WORDS 140
  cost               ~33 s per adjudication on this CPU
```

**On the final server this changes.** These runs used Ollama with JSON mode. The
full experiment runs on llama.cpp with GBNF grammar enforcement, where malformed
output is unrepresentable rather than merely requested. A prompt chosen here
should be re-checked there before it is trusted.

## Results

```
              addresses  correct  correct  false  false  FAKE
prompt         accuracy      yes       no    yes     no  NEWS
current           0.714       18        7      8      2     6
v2_direct         0.714       19        6      9      1     4
```

n = 35 cases (20 covered, 15 uncovered).

- **false yes** = the corpus does NOT cover the window but the model said it
  does. This is the unsupported-flag risk: it is the precondition for every
  flag issued on evidence that does not address the claim.
- **false no** = the corpus DOES cover it but the model said it does not, so a
  real claim can go unchecked.

Verdict distribution:

```
  current      {'NOT_FAKE_NEWS': 20, 'NO_EVIDENCE': 9, 'FAKE_NEWS': 6}
  v2_direct    {'NOT_FAKE_NEWS': 23, 'FAKE_NEWS': 4, 'NO_EVIDENCE': 8}
```

## The 6 disagreements

Both prompts saw identical retrieval on every case, so each difference below
is attributable to the prompt alone.

### `Wtd0Hj00g_E#2` — topic `general`, corpus coverage: **yes**

Window:

```
  all the rest of the virus families make what's called an acute infection, and then they'
  re gone. That's because they aren't messing with our DNA. RNA does not make DNA, and it 
  gets degraded pretty quickly, usually within a day, I would say for the most part. I don
  't know the precise numbers, but it's a short-lived molecule, and that's how it's design
  ed to be. And really, this makes RNA vaccines much safer, in my view.
```

Retrieved (same for both prompts):

```
  [cureus-0017-00000098208#Abstract:6] sim=0.6644
      risk of myocarditis, particularly in males, after the second dose, with a higher incidence following mRNA-1273. No consistent evidence of increased GB
  [cureus-0017-00000098208#Introduction_And_Background:0] sim=0.6644
      The rapid development and deployment of COVID-19 vaccines marked a turning point in the global response to the pandemic [1]. Among these, messenger RN
  [cureus-0017-00000098208#Introduction_And_Background:1] sim=0.6644
      and favorable safety profile in preventing severe disease and hospitalization [2]. Adolescents, a population initially considered at lower risk for se
  [s41392-022-00950-y#After_over_30_years_of_research,_mRNA_vaccines_have_become_a:2] sim=0.6816
      a key role in limiting the spread of the current pandemic. mRNA vaccines, unlike traditional vaccines, may enable adjustment of antigen design and eve
  ... 8 more
```

Hand-labelled relevant evidence:

```
  inserts its mRNA into the cytoplasm (not in the nucleus)
```

Outputs:

```
  current
    verdict     NO_EVIDENCE
    addresses   no
    stance      asserting
    evidence_id 
    reason      The speaker discusses the nature of RNA and its safety, but the evidence does not address these specific claims about RNA or RNA vaccines.
  v2_direct
    verdict     NOT_FAKE_NEWS
    addresses   yes
    stance      asserting
    evidence_id cureus-0017-00000098208#Abstract:6
    reason      The speaker discusses the nature of RNA and its short-lived nature, which aligns with the evidence's discussion of mRNA vaccines' safety and low risk of adverse events.
```

### `jpx2rhuIieQ#5` — topic `natural_immunity`, corpus coverage: **yes**

Window:

```
  antibody level tested. But again, that's another long history thing is people who don't 
  get sick while everyone else has been accused of witchcraft and sorcerers in the past an
  d sometimes hunted down and killed. You know, in the times of smallpox, the groups of pe
  ople that were into cleanliness, that was a real problem for them. I did do nasal swabs 
  to see if I had any antibodies. I did do that and I didn't. Well, that won't tell you an
  tibodies. That just, that's a PCR, that would be your PCR test or your rapid, which one 
  did you do? The one that goes? The rapid antigen test. But that's only gonna tell you if
```

Retrieved (same for both prompts):

```
  [cdc_108736_DS1#COVID-19:4] sim=0.6431
      vaccines. Will getting a COVID-19 vaccine cause me to test positive for COVID-19 on a viral test? No. None of the authorized and recommended COVID-19 
  [cdc_108736_DS1#COVID-19:5] sim=0.6431
      are used to see if you have a current infection.​ If your body develops an immune response to vaccination, which is the goal, you may test positive on
  [cdc_108736_DS1#Other_Myths_and_Facts:0] sim=0.6431
      No. The federal government does not mandate (require) vaccination for people. Additionally, CDC does not maintain or monitor a person’s vaccination re
  [vaccines-11-01693#Demographic_Characteristics_of_Study_Participants:3] sim=0.6091
      44 (5.7%) participants reported having been infected twice since the beginning of the COVID-19 outbreak. Among the 451 infected participants, 8 (2.0%)
  ... 8 more
```

Hand-labelled relevant evidence:

```
  the lower risk of infection remained among individuals with a previous SARS-CoV-2 infection
```

Outputs:

```
  current
    verdict     NOT_FAKE_NEWS
    addresses   yes
    stance      neutral
    evidence_id cdc_108736_DS1
    reason      The speaker is discussing the difference between PCR tests and antibody tests, which aligns with the CDC's explanation that viral tests do not indicate vaccination status, while antibody tes
  v2_direct
    verdict     NO_EVIDENCE
    addresses   yes
    stance      neutral
    evidence_id 
    reason      The speaker is discussing the difference between PCR tests and antibody tests but does not make a claim that contradicts or aligns with the evidence.
```

### `Z3g_CijXgUI#0` — topic `general`, corpus coverage: **no**

Window:

```
  Welcome back to Takeout, coming to you from New York. The Department of Health and Human
   Services is preparing a list of injuries it believes it can assert were presumptively c
  aused by COVID vaccinations. Now, this list would allow people to be compensated for the
  se injuries without, and this is interesting, to prove that the shot, meaning the vaccin
  ation, was the cause of said injuries. HHS Secretary Robert F. Kennedy, as you know, is 
  not only behind this proposal, he's a longtime skeptic slash critic of vaccinations. COV
  ID and otherwise, Dr. Celine Gounder joins us to discuss it all. What is this list, and 
```

Retrieved (same for both prompts):

```
  [vaccines-10-01444#No_comment:1] sim=0.6365
      reduced ejection fraction; HLD: hyperlipidemia; HTN: hypertension; HBV: hepatitis B virus; ITP: immune thrombocytopenic purpura; IVIG: intravenous imm
  [vaccines-10-01444#Discussion:0] sim=0.6365
      As of July 2022, over 500 million doses of the COVID-19 vaccine have been delivered across the United States [50]. While there have been around 300,00
  [vaccines-10-01444#Discussion:1] sim=0.6365
      90% of those were non-serious [51]. Some major adverse events, such as myopericarditis, Guillain-Barre syndrome, and coagulopathy, including ITP, have
  [vaccines-09-00467-v2#Conclusions:2] sim=0.6624
      all the vaccines are missing longer-term assessments. This meta-analysis allows us to incorporate relevant new evidence for summarizing and analyzing 
  ... 7 more
```

Outputs:

```
  current
    verdict     NO_EVIDENCE
    addresses   no
    stance      asserting
    evidence_id 
    reason      The evidence does not discuss the specific subject of the claim, which is the HHS list of injuries presumed to be caused by COVID vaccinations.
  v2_direct
    verdict     NOT_FAKE_NEWS
    addresses   yes
    stance      asserting
    evidence_id vaccines-10-01444#Discussion:0
    reason      The speaker discusses the difficulty of proving causation for vaccine injuries, which aligns with the evidence mentioning that more than 90% of adverse event reports were non-serious and the
```

### `DQxMIJCvanc#7` — topic `infertility`, corpus coverage: **no**

Window:

```
  the world, in this country who have received the vaccine. So in terms of waiting to see,
   you know, we can see right now, talk to your family, your friends, your coworkers, your
   loved ones, your doctor, healthcare providers, because so many people have gotten the v
  accine, get the facts, and then of course, what I want, what we all want, of course, is 
  for people to get the vaccine, one of the most important things we can do here. There ar
  e some hesitancy concerns that I keep hearing repeatedly. For example, I hear from young
   women that they worried about the mRNA vaccines, which are brand new, affecting their f
```

Retrieved (same for both prompts):

```
  [pbio.3001506#Introduction:5] sim=0.7501
      CDC are notably consistent with the conclusions from other studies supporting the safety of COVID-19 mRNA vaccination during pregnancy [11,12]. Even a
  [pbio.3001506#Introduction:6] sim=0.7501
      and nonpregnant individuals alike [13,14], public perception of the risk surrounding vaccination has been impacted by the rapid proliferation of numer
  [pbio.3001506#Introduction:7] sim=0.7326
      likely than men to be vaccine hesitant [15], and a number of these unconfirmed speculations target women’s health issues specifically, spreading fear 
  [pbio.3001506#May_24,_2022.4:2] sim=0.7212
      receiving the COVID-19 mRNA vaccines to date and add to the mounting evidence that a syncytin-1-based mechanism of infertility by mRNA vaccination aga
  ... 4 more
```

Outputs:

```
  current
    verdict     FAKE_NEWS
    addresses   yes
    stance      asserting
    claim       "they worried about the mRNA vaccines, which are brand new, affecting their fertility"
    evidence_id pbio.3001506#May_24,_2022.4:2
    reason      The speaker asserts that women are worried about mRNA vaccines affecting fertility, which the evidence contradicts by stating that a syncytin-1-based mechanism of infertility by mRNA vaccina
  v2_direct
    verdict     NOT_FAKE_NEWS
    addresses   yes
    stance      asserting
    evidence_id pbio.3001506#Introduction:7
    reason      The speaker acknowledges concerns about mRNA vaccines affecting fertility, which the evidence confirms as a major source of vaccine hesitancy.
```

### `tF4DdSWsP5I#1` — topic `myocarditis`, corpus coverage: **no**

Window:

```
  for developing myocarditis. That's an inflammation of the heart muscle, typically associ
  ated with viral infections. Scientists say the COVID vaccine activates immune cells, tri
  ggering inflammation that damages heart muscle cells. While the incidence is rare, docto
  rs say knowing the action of damage gives them a target to make the vaccine safer. Fibro
  ids in the uterus may signal problems with the heart. According to a study by the Americ
  an Heart Association, heart disease risk is much greater among women with fibroids compa
  red to those without. The study of nearly 3 million women over a decade shows fibroids a
```

Retrieved (same for both prompts):

```
  [clinpract-16-00077#Future_Research_Directions:0] sim=0.6628
      Future research should focus on clarifying the long-term cardiac outcomes and underlying mechanisms of myocarditis following mRNA COVID-19 vaccination
  [clinpract-16-00077#Future_Research_Directions:1] sim=0.6628
      and biomarkers is needed to assess potential subclinical fibrosis, diastolic dysfunction, or arrhythmogenic remodeling. Mechanistic studies comparing 
  [clinpract-16-00077#Future_Research_Directions:2] sim=0.6628
      susceptibilities such as HLA or ACE2 variants that predispose individuals to adverse cardiac inflammation. Developing predictive risk models integrati
  [clinpract-16-00077#Interpretation_of_Findings:0] sim=0.6471
      Across multinational studies, mRNA vaccine-associated myocarditis is rare, generally mild, and self-limiting, with highest risk in the young male demo
  ... 8 more
```

Outputs:

```
  current
    verdict     FAKE_NEWS
    addresses   yes
    stance      asserting
    claim       "Fibroids in the uterus may signal problems with the heart."
    evidence_id clinpract-16-00077#Introduction:10
    reason      The speaker asserts a false claim about fibroids and heart health, which is unrelated to the evidence discussing myocarditis risks and benefits of mRNA vaccines.
  v2_direct
    verdict     NO_EVIDENCE
    addresses   no
    stance      neutral
    evidence_id 
    reason      The speaker discusses myocarditis and fibroids, but the evidence focuses on myocarditis following mRNA vaccines without mentioning fibroids or heart disease risk factors related to fibroids.
```

### `gum4DvKEU9M#4` — topic `general`, corpus coverage: **no**

Window:

```
  find it confusing that the vaccination program is saving tens of thousands of lives but 
  the pandemic remains dangerous. Let me explain it in a single word, time, time. Even mov
  ing at the record speed we're moving at, we're not even halfway through vaccinating over
   300 million Americans. This is going to take time. We aren't at the finish line. We sti
  ll have a lot of work to do. We're still in a life and death race against this virus. An
  alysis of state and county data shows new COVID-19 cases in the U.S. rose 5 percent to o
  ver 450,000 last week, the third straight week that infections have increased. The avera
```

Retrieved (same for both prompts):

```
  [s41392-022-00950-y#Coronavirus_disease_2019_(COVID-19)_is_an_emerging_disease:1] sim=0.7448
      of 29,881 nucleotides (GenBank accession number: MN908947) with a methylated 5′-cap and a 3′- poly(A) tail, consisting of 9860 amino acids encoding 16
  [s41392-022-00950-y#February_9,_2022,_COVID-19_has_spread_to_227_countries_and:0] sim=0.7448
      regions worldwide,6 causing over 399 million conﬁrmed cases and 5.75 million deaths.7 The COVID-19 pandemic is an unprecedented event which has caused
  [s41392-022-00950-y#To_control_the_spread_of_the_epidemic,_governments_worldwide:0] sim=0.7448
      have mobilized a considerable amount of manpower and material resources into research and development (R&D) efforts linked to the COVID-19 vaccine. Se
  [s41392-022-00950-y#Advances_in_COVID-19_mRNA_vaccine_development:0] sim=0.7353
      Enyue Fang1,2, Xiaohui Liu1, Miao Li1, Zelun Zhang1, Lifang Song1, Baiyu Zhu3, Xiaohong Wu1, Jingjing Liu1, Danhua Zhao1 and Yuhua Li1✉ To date, the c
  ... 8 more
```

Outputs:

```
  current
    verdict     NO_EVIDENCE
    addresses   no
    stance      neutral
    evidence_id 
    reason      The speaker discusses the ongoing nature of the pandemic despite vaccination efforts, but the evidence does not address this specific claim.
  v2_direct
    verdict     NOT_FAKE_NEWS
    addresses   yes
    stance      asserting
    evidence_id s41392-022-00950-y
    reason      The speaker discusses the ongoing nature of the pandemic despite vaccination efforts, which aligns with the evidence showing the need for continued vaccination and the ongoing spread of the 
```

## Every case

`A` = addresses, `V` = verdict. Expected addresses from the hand-made
`relevant_evidence` label.

```
case                   topic             exp  current                    v2_direct                 
2riTrIQTaaw#6          natural_immunity  no   A=no NO_EVIDENCE           A=no NO_EVIDENCE          
A22eAcrsjqQ#2          off_topic         no   A=no NO_EVIDENCE           A=no NO_EVIDENCE          
A22eAcrsjqQ#6          off_topic         no   A=no NO_EVIDENCE           A=no NO_EVIDENCE          
DQxMIJCvanc#7          infertility       no   A=yes FAKE_NEWS            A=yes NOT_FAKE_NEWS         <- differs
GtqyTBPhbMA#1          blood_clots       no   A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
RaLxhFiOBYk#3          general           no   A=no NO_EVIDENCE           A=no NO_EVIDENCE          
WTdorGWyCGs#2          natural_immunity  no   A=no NO_EVIDENCE           A=no NO_EVIDENCE          
Z3g_CijXgUI#0          general           no   A=no NO_EVIDENCE           A=yes NOT_FAKE_NEWS         <- differs
cMuQxt2k4gk#0          infertility       no   A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
dUlCqxeGFGw#3          infertility       no   A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
gum4DvKEU9M#1          blood_clots       no   A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
gum4DvKEU9M#4          general           no   A=no NO_EVIDENCE           A=yes NOT_FAKE_NEWS         <- differs
s4Z3baKFWu0#1          blood_clots       no   A=yes FAKE_NEWS            A=yes FAKE_NEWS           
tF4DdSWsP5I#0          myocarditis       no   A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
tF4DdSWsP5I#1          myocarditis       no   A=yes FAKE_NEWS            A=no NO_EVIDENCE            <- differs
0RixG3Bus0Q#4          infertility       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
6DdIUUPm3ug#0          natural_immunity  yes  A=yes FAKE_NEWS            A=yes FAKE_NEWS           
ArtNg2LD1Ds#1          myocarditis       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
DQxMIJCvanc#10         infertility       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
GtqyTBPhbMA#2          blood_clots       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
Ozysm76UNaY#1          blood_clots       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
Ozysm76UNaY#2          blood_clots       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
UijQ-6jhVuc#14         blood_clots       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
VZ3Lbhgyeyw#3          infertility       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
Wtd0Hj00g_E#2          general           yes  A=no NO_EVIDENCE           A=yes NOT_FAKE_NEWS         <- differs
_PrUE69QZvA#0          general           yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
jpx2rhuIieQ#5          natural_immunity  yes  A=yes NOT_FAKE_NEWS        A=yes NO_EVIDENCE           <- differs
mIzuuYB6GrI#0          natural_immunity  yes  A=no NO_EVIDENCE           A=no NO_EVIDENCE          
o3Kh99pZnn4#5          myocarditis       yes  A=yes FAKE_NEWS            A=yes FAKE_NEWS           
qFR3TlR0SmI#4          myocarditis       yes  A=yes FAKE_NEWS            A=yes FAKE_NEWS           
qPvC6hd1KBg#0          natural_immunity  yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
umHRAaS4Xlo#1          myocarditis       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
v-NEr3KCug8#7          general           yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
vOvl3vZZsYY#0          infertility       yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
xGHs0u0VzoA#1          general           yes  A=yes NOT_FAKE_NEWS        A=yes NOT_FAKE_NEWS       
```

## Analysis

**Neither prompt is better. Both score `addresses` accuracy 0.714.**

An earlier reading of this experiment reported 0.800 for the current prompt
against 0.840 for `v2_direct`. That was computed on the first 25 cases, which
the case file happens to order covered-first, so the subset was 20 covered
against 5 uncovered. The uncovered half is where the two prompts diverge, and
once it was complete (15 uncovered) the advantage vanished. **The intermediate
figure was an artifact of the sampling order, not a result.**

### What `v2_direct` actually changed

It did fix both of the failures it was designed for:

```
  DQxMIJCvanc#7   current  FAKE_NEWS      v2  NOT_FAKE_NEWS
    "women are worried about mRNA vaccines affecting fertility"
    Reporting a concern, not asserting a falsehood. The current prompt flags it
    anyway — the same shape as the smoke test's "the serious harm is there".

  tF4DdSWsP5I#1   current  FAKE_NEWS      v2  NO_EVIDENCE
    The current prompt's own reason says the claim is "unrelated to the
    evidence" — and it flags regardless. This is the exact contradiction behind
    the fertility-paper citations: a verdict asserted while acknowledging the
    evidence does not apply.
```

So the hypothesis was right about the mechanism. Putting the relevance test
first does stop the prompt from flagging text it has just described as
unaddressed.

**But it broke three other cases**, all in the same direction:

```
  Z3g_CijXgUI#0   current  NO_EVIDENCE    v2  NOT_FAKE_NEWS   (uncovered)
  gum4DvKEU9M#4   current  NO_EVIDENCE    v2  NOT_FAKE_NEWS   (uncovered)
  Wtd0Hj00g_E#2   current  NO_EVIDENCE    v2  NOT_FAKE_NEWS   (covered — v2 right here)
```

### The trade is flags for clearances, which is the wrong direction

Measured on the 15 windows a human judged the corpus does **not** cover:

```
  prompt       addresses=yes   FAKE_NEWS   NOT_FAKE_NEWS   verdict asserted
                                                           on uncovered evidence
  current            8             3             5                 8
  v2_direct          9             1             8                 9
```

`v2_direct` cuts unsupported flags from 3 to 1. It raises unsupported
clearances from 5 to 8. The total number of verdicts asserted on evidence a
human judged irrelevant goes **up**, 8 to 9.

That is not an improvement, and for this system it is arguably worse. A wrong
`FAKE_NEWS` is visible — it appears in the report with its evidence, and a
reader can see the citation does not fit, which is how the fertility-paper
failures were found at all. A wrong `NOT_FAKE_NEWS` is silent: the window is
cleared, nothing is printed, and under the video-level join rule ("any window
flagged ⇒ video predicted fake news") it simply disappears. The shorter prompt
buys its apparent calm by moving errors from the visible column to the invisible
one.

The likely cause is change 3. Dropping *"it counts even if the speaker is vague,
hedging, or embedding the claim"* and adding *"do not flag an opinion, a
question, a value judgement, or a sentence too vague to be checked"* removed the
only pressure toward flagging and replaced it with pressure away. On
`gum4DvKEU9M#4` v2 even reads a window the current prompt calls `neutral` as
`asserting` and then clears it as NOT_FAKE_NEWS.

### Decision

**Keep the current prompt.** `v2_direct` is not adopted. The two failure modes
it fixes are real, but it does not fix them by making relevance judgements
better — `addresses` accuracy is identical — it fixes them by being more
reluctant to flag, and that reluctance costs more in silent misses than it saves
in visible false flags.

No parameter or prompt changes as a result of this experiment.

### What this says about the real problem

The result that matters is not the comparison, it is the level both prompts sit
at: **`addresses` is wrong on 10 of 35 dev windows, and 8-9 verdicts are
asserted on evidence a human judged irrelevant.** Rewording the instruction
moved which cases fail without moving the rate, which is evidence that the
relevance judgement is not instruction-limited at this model size.

That points away from prompt engineering and toward the two things this
experiment holds fixed:

1. **Retrieval.** Up to ~10 passages reach the prompt (`TOP_K = 4` plus
   neighbour expansion), and on an uncovered window all of them are topically
   plausible and none are relevant. The model is being asked to reject its
   entire context. The smoke-run failures support this: one irrelevant paper was
   cited by three separate windows, at similarity 0.60-0.68 — higher than
   several correct flags.
2. **Grammar enforcement.** These runs used Ollama JSON mode. On llama.cpp the
   GBNF grammar constrains `addresses` during sampling, which is a different
   mechanism from asking for it in text, and the gate's enforcement
   (`addresses == "no"` + flagged verdict → NO_EVIDENCE override) has never been
   measured there.

Both are worth testing before any further prompt variant. A third variant of the
wording is unlikely to move a rate that two substantially different phrasings
produced identically.

## Reproducing

```bash
# regenerate the case set from the dev labels (deterministic)
python tests/select_prompt_cases.py 20 tests/prompt_cases.json

# run both prompts over identical retrieval, checkpointed per case
python tests/prompt_ab.py tests/prompt_cases.json data/results/v1/prompt_ab_dev.json

# score
python tests/score_prompt_ab.py data/results/v1/prompt_ab_dev.json
```

Files:

| path | holds |
|---|---|
| `tests/prompt_variants.py` | the candidate prompt(s) under test |
| `tests/select_prompt_cases.py` | stratified, deterministic case selection |
| `tests/prompt_ab.py` | runs each prompt over identical retrieval |
| `tests/score_prompt_ab.py` | metrics + the disagreement listing |
| `tests/prompt_cases.json` | the 40-case set (35 were completed) |
| `data/results/v1/prompt_ab_dev.json` | every input and output from this run |

`adjudicate()` takes an optional `system_prompt` argument for this purpose. The
pipeline never passes it, so the production path is unaffected.

Cases are selected deterministically, so the case set is reproducible from the
dev label files. The model's outputs are not: `temperature = 0.0` makes them
near-deterministic but llama.cpp and Ollama do not guarantee identical sampling
across versions, which is why every output is recorded above rather than
regenerated on demand.
