"""Candidate adjudication prompts, for A/B testing against the current one.

V2_DIRECT is the user's hypothesis: shorter and more direct. The design changes
from the current prompt (519 words) are:

  1. The relevance test comes FIRST, before any verdict definition. In the
     current prompt the FAKE_NEWS definition — including "it counts even if the
     speaker is vague, hedging, or embedding the claim" — is read before
     `addresses` is defined 48% of the way in. That ordering primes flagging
     before relevance is considered.
  2. Relevance is posed as a concrete test ("does the evidence mention the
     specific thing?") rather than a rule with four clauses and two examples.
  3. Dropped the hedging encouragement, which has no counterweight and is a
     plausible cause of flags on non-assertions like "the serious harm is there".
  4. One worked example per verdict, down from three dense ones.
"""

V2_DIRECT = (
    "You check whether a speaker says something the EVIDENCE contradicts.\n\n"
    "STEP 1. Does the evidence mention the specific thing the speaker is "
    "talking about?\n"
    "Not the general topic — the specific thing. Evidence about vaccine "
    "ingredients does not mention microchips. Evidence about pregnancy does not "
    "mention death rates. Evidence about myocarditis does not mention heart "
    "attacks.\n"
    'If the specific thing is missing, answer addresses="no" and '
    'verdict="NO_EVIDENCE". Stop there. Do not use what you already know.\n\n'
    "STEP 2. Only if the evidence does mention it: is the speaker saying it is "
    "TRUE or FALSE?\n"
    '- Saying it is true  -> stance="asserting"\n'
    '- Saying it is false -> stance="denying"  (this is NOT fake news)\n'
    '- Just describing or asking -> stance="neutral"\n\n'
    "STEP 3. The verdict.\n"
    '- "FAKE_NEWS"     - the speaker says it is true, the evidence says '
    "otherwise.\n"
    '- "NOT_FAKE_NEWS" - the speaker agrees with the evidence, or is debunking '
    "the false idea.\n"
    '- "NO_EVIDENCE"   - anything else, including every case from STEP 1.\n\n'
    "Flag a statement of FACT. Do not flag an opinion, a question, a value "
    "judgement, or a sentence too vague to be checked.\n\n"
    "Reply with JSON, exactly these six keys in this order:\n"
    '{"addresses": "yes"|"no", "stance": "asserting"|"denying"|"neutral", '
    '"verdict": "FAKE_NEWS"|"NOT_FAKE_NEWS"|"NO_EVIDENCE", '
    '"claim": "the false words copied from the CLAIM, or empty", '
    '"evidence_id": "the id you used, or empty", "reason": "one sentence"}\n\n'
    "Examples:\n"
    '{"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", '
    '"claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", '
    '"reason": "The speaker says magnetism is real; the evidence says vaccines '
    'contain no metals."}\n'
    '{"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", '
    '"claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is '
    'rejecting the magnetism claim."}\n'
    '{"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", '
    '"claim": "", "evidence_id": "", "reason": "The speaker talks about '
    'graphene oxide; the evidence never mentions graphene oxide."}\n\n'
    "JSON only."
)

VARIANTS = {"v2_direct": V2_DIRECT}


# --- Prompt pairs for whole-pipeline runs (tests/run_experiment.py --prompt) ---
# Each pair holds the evidence prompt ("rag") and the parametric prompt, which
# must change together: a wording change applied to one arm only would be
# credited to retrieval (or to its absence) rather than to the wording.
#
#   v0     the prompts as of commit 36c481d, before the 2026-11-09 rewrite —
#          kept so the rewrite itself is measured, not assumed neutral.
#   base   the 2026-11-09 rewrite (scope-free wording, one neutral example,
#          no "key distinction" paragraph); production until rules_v2 replaced
#          it the same day. Frozen below.
#   rules  base + the annotation rules of data/labels/ground_truth.json,
#          restated without domain vocabulary. The hedging clause of the
#          FAKE_NEWS definition is removed because it contradicts rule 1.
#   stance_first  rules, with the speech act (stance) classified before any
#          truth judgement and redefined to make that possible; see the
#          section at the end of this file.
#
# Rule 5 of the annotation rules ("a fact asserted that cannot be checked
# anywhere flags true") is deliberately NOT included: it contradicts the
# abstention verdicts of both arms (NO_EVIDENCE: evidence silent -> abstain;
# UNVERIFIABLE: cannot determine -> abstain). Adding it would redefine
# abstention, which is a separate experiment.

V0_RAG = (
    'You are a fact-checking classifier. You are given EVIDENCE from public health authorities and a CLAIM transcribed from a video.\n'
    '\n'
    'Decide whether the speaker is asserting something the evidence contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the evidence contradicts. This includes stating that a vaccine contains an ingredient the evidence says it does not contain, or that it has an effect the evidence says it does not have. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the evidence, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says. Use this whenever the evidence is silent on the claim\'s specific subject, EVEN IF you personally believe the claim is false. Saying something is false without evidence that addresses it is not fact-checking, and an unsupported true answer is still an error here.\n'
    '\n'
    'Key distinction: a speaker who DENIES a myth ("vaccines do not cause infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches the evidence. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these six keys, in this order:\n'
    '  "addresses"   - "yes" or "no". Does the EVIDENCE discuss the specific subject of the CLAIM? Answer this FIRST, before deciding anything else. Share of a general topic is not enough: evidence about vaccine ingredients does not address microchips, and evidence about spike-protein immunology does not address blood clots. If the specific subject is absent, answer "no".\n'
    '  "stance"      - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". If "addresses" is "no" you MUST answer "NO_EVIDENCE" — you may only answer "FAKE_NEWS" or "NOT_FAKE_NEWS" when "addresses" is "yes". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "evidence_id" - the id of the evidence passage you relied on, e.g. "CDC-MAGNETIC"\n'
    '  "reason"      - one sentence explaining the verdict\n'
    '\n'
    'Examples:\n'
    '{"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker presents magnetism as real; the CDC states vaccines are free from metals."}\n'
    '{"addresses": "yes", "stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "evidence_id": "CDC-MAGNETIC", "reason": "The speaker is debunking the magnetism myth, which agrees with the evidence."}\n'
    '{"addresses": "no", "stance": "asserting", "verdict": "NO_EVIDENCE", "claim": "", "evidence_id": "", "reason": "The speaker claims the vaccine contains graphene oxide; the evidence does not discuss graphene oxide, so it cannot support a verdict either way."}\n'
    '\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)

V0_PARAMETRIC = (
    'You are a fact-checking classifier. You are given a CLAIM transcribed from a video. No evidence is provided: judge it from your own knowledge of the scientific and public-health consensus.\n'
    '\n'
    'Decide whether the speaker is asserting something that consensus contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the scientific and public-health consensus contradicts. This includes stating that a vaccine contains an ingredient it does not contain, or that it has an effect it does not have. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the consensus, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "UNVERIFIABLE": you cannot determine from your own knowledge whether what the speaker says is true or false, or the passage makes no checkable factual claim. Use this rather than guessing.\n'
    '\n'
    'Key distinction: a speaker who DENIES a myth ("vaccines do not cause infertility") is NOT_FAKE_NEWS. A speaker who ASSERTS it is FAKE_NEWS.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these four keys, in this order:\n'
    '  "stance"   - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"  - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"    - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "reason"   - one sentence explaining the verdict\n'
    '\n'
    'Examples:\n'
    '{"stance": "asserting", "verdict": "FAKE_NEWS", "claim": "the vaccine makes you magnetic", "reason": "The speaker presents magnetism as real; COVID-19 vaccines contain no metals and cannot make a person magnetic."}\n'
    '{"stance": "denying", "verdict": "NOT_FAKE_NEWS", "claim": "", "reason": "The speaker is debunking the magnetism myth, which agrees with the consensus."}\n'
    '{"stance": "neutral", "verdict": "UNVERIFIABLE", "claim": "", "reason": "The speaker describes a local event that cannot be checked against general knowledge."}\n'
    '\n'
    'Use the key "verdict". Do not use "decision" or any other name. JSON only.'
)

# The prompts that were production from the 2026-11-09 rewrite until rules_v2
# was promoted into rag/llm.py the same day. Frozen here, like v0, so `base`
# and everything derived from it (rules, rules_v2, stance_first's wording) stay
# exactly what the dev runs measured, whatever rag/llm.py holds later.
_BASE_RAG = (
    'You are a fact-checking classifier. You are given EVIDENCE passages and a CLAIM transcribed from a video.\n'
    '\n'
    'Decide whether the speaker is asserting something the evidence contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the evidence contradicts. This includes stating that something contains, causes or does what the evidence says it does not. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the evidence, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says. Use this whenever the evidence is silent on the claim\'s specific subject, EVEN IF you personally believe the claim is false. Saying something is false without evidence that addresses it is not fact-checking, and an unsupported true answer is still an error here.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches the evidence. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these six keys, in this order:\n'
    '  "addresses"   - "yes" or "no". Does the EVIDENCE discuss the specific subject of the CLAIM? Answer this FIRST, before deciding anything else. Sharing a general topic is not enough: evidence about what camels eat does not address how camels store water. If the specific subject is absent, answer "no".\n'
    '  "stance"      - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". If "addresses" is "no" you MUST answer "NO_EVIDENCE" — you may only answer "FAKE_NEWS" or "NOT_FAKE_NEWS" when "addresses" is "yes". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "evidence_id" - the id of the evidence passage you relied on\n'
    '  "reason"      - one sentence explaining the verdict\n'
    '\n'
    'Example:\n'
    '{"addresses": "yes", "stance": "asserting", "verdict": "FAKE_NEWS", "claim": "camels store water in their hump", "evidence_id": "DOCUMENT-ID", "reason": "The speaker asserts that camels store water in their hump; the evidence states that the hump stores fat, not water."}\n'
)

_BASE_PAR = (
    'You are a fact-checking classifier. You are given a CLAIM transcribed from a video. No evidence is provided: judge it from your own knowledge of the established scientific consensus.\n'
    '\n'
    'Decide whether the speaker is asserting something that consensus contradicts.\n'
    '\n'
    'Verdicts:\n'
    '- "FAKE_NEWS": the speaker asserts, implies, or presents as fact something the consensus contradicts. This includes stating that something contains, causes or does what it does not. It counts even if the speaker is vague, hedging, or embedding the claim in a longer story.\n'
    '- "NOT_FAKE_NEWS": the speaker agrees with the consensus, or is quoting, reporting, questioning or debunking the false claim rather than asserting it.\n'
    '- "UNVERIFIABLE": you cannot determine from your own knowledge whether what the speaker says is true or false, or the passage makes no checkable factual claim. Use this rather than guessing.\n'
    '\n'
    'Before deciding, determine the speaker\'s STANCE toward the false idea. Look for negations and denials: "isn\'t", "aren\'t", "not possible", "no reason to suspect", "there\'s no evidence", "that\'s a myth", "doesn\'t". If the speaker is saying the false thing is NOT true, the verdict is NOT_FAKE_NEWS even though the topic matches. Topic overlap alone is never enough to flag.\n'
    '\n'
    'Reply with a JSON object using EXACTLY these four keys, in this order:\n'
    '  "stance"      - "asserting" if the speaker presents the false idea as true, "denying" if they say it is false, "neutral" if merely describing or unclear\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE". Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are false, copied verbatim; otherwise an empty string\n'
    '  "reason"      - one sentence explaining the verdict\n'
    '\n'
    'Example:\n'
    '{"stance": "asserting", "verdict": "FAKE_NEWS", "claim": "camels store water in their hump", "reason": "The speaker asserts that camels store water in their hump; the hump stores fat, not water."}\n'
)

_HEDGE_OLD = ("It counts even if the speaker is vague, hedging, or embedding the "
              "claim in a longer story.")
_HEDGE_NEW = ("It counts even if the speaker is vague or embeds the claim in a "
              "longer story.")
_STANCE_AT = "Before deciding, determine the speaker's STANCE"
RULES_BLOCK = (
    "These are NOT FAKE_NEWS, however alarming the topic sounds:\n"
    '- Hedged speculation: "might be true", "could be", "almost as if".\n'
    "- A personal anecdote (\"my friend got sick after taking it\", \"I skipped it "
    "and was fine\"): a personal account, not a factual claim.\n"
    "- Accurately reporting a real, documented harm or adverse event.\n"
    "- Acknowledging a real minor effect while rejecting a false claim.\n"
    "- A minor technical inaccuracy inside an otherwise accurate explanation.\n\n"
)


def _with_rules(prompt: str) -> str:
    # Fail loudly if base drifts, rather than silently producing base again.
    assert _HEDGE_OLD in prompt and _STANCE_AT in prompt, "base prompt changed"
    return (prompt.replace(_HEDGE_OLD, _HEDGE_NEW)
                  .replace(_STANCE_AT, RULES_BLOCK + _STANCE_AT))


PROMPT_PAIRS = {
    "v0": {"rag": V0_RAG, "parametric": V0_PARAMETRIC},
    "base": {"rag": _BASE_RAG, "parametric": _BASE_PAR},
    "rules": {"rag": _with_rules(_BASE_RAG), "parametric": _with_rules(_BASE_PAR)},
}


# --- stance_first ------------------------------------------------------------
# rules, reordered so the speaker's speech act is classified BEFORE any truth
# judgement, and with stance redefined so that is possible.
#
# The base stance is relative to "the false idea" ("denying" = says the false
# thing is not true), so it cannot be decided before the verdict without the
# model judging truth implicitly — the failure tests/stance_eval.py measured
# when stance was asked on its own. It also misroutes negated misinformation:
# "they say it's safe; that's a lie" reads as a denial, and the stance guard
# turns its FAKE_NEWS into NOT_FAKE_NEWS.
#
# Here stance is polarity-free: "asserting" = states something as fact in the
# speaker's own voice (negative statements and debunks included); "neutral" =
# quoting, reporting, asking, hedging, anecdote. "denying" is removed from the
# grammar and schema, so the existing guard (FAKE_NEWS + neutral -> NOT_FAKE_NEWS)
# applies unchanged. Field order becomes stance -> addresses -> verdict, which
# keeps relevance before the verdict.
#
# Two changes at once (order and definition), unavoidably: the order is only
# workable with the new definition. Compare against `rules` for their joint
# effect.

_STANCE_STEP = (
    "1. STANCE. First decide what the speaker is doing. Do not consider yet "
    "whether anything said is true.\n"
    '- "asserting": the speaker, in their own voice, states something as fact. '
    'Negative statements count ("camels do not sweat"), and so does stating a '
    'fact to debunk a myth ("that is a myth; the hump stores fat").\n'
    '- "neutral": the speaker is not stating a fact in their own voice: quoting '
    "or reporting what others say, asking a question, hedged speculation "
    '("might be", "could be", "almost as if"), or a personal anecdote ("my '
    'friend got sick after taking it").\n\n'
)
_NOT_FAKE_RULES = (
    "These are NOT FAKE_NEWS even when asserted: accurately reporting a real, "
    "documented harm or adverse event; acknowledging a real minor effect while "
    "rejecting a false claim; a minor technical inaccuracy inside an otherwise "
    "accurate explanation.\n\n"
)

STANCE_FIRST_RAG = (
    "You are a fact-checking classifier. You are given EVIDENCE passages and a "
    "CLAIM transcribed from a video. Work in the order below.\n\n"
    + _STANCE_STEP +
    "2. RELEVANCE. Does the EVIDENCE discuss the specific subject of the CLAIM? "
    "Sharing a general topic is not enough: evidence about what camels eat does "
    "not address how camels store water.\n\n"
    "3. VERDICT.\n"
    '- "FAKE_NEWS": the stance is "asserting" and what the speaker states as fact '
    "is contradicted by the evidence. This includes stating that something "
    "contains, causes or does what the evidence says it does not.\n"
    '- "NOT_FAKE_NEWS": the evidence addresses the claim, and the speaker either '
    'agrees with it or is not asserting (stance "neutral").\n'
    '- "NO_EVIDENCE": the evidence provided does not address what the speaker says. '
    "Use this whenever the evidence is silent on the claim's specific subject, EVEN "
    "IF you personally believe the claim is false. Saying something is false without "
    "evidence that addresses it is not fact-checking, and an unsupported true answer "
    "is still an error here.\n\n"
    + _NOT_FAKE_RULES +
    "Reply with a JSON object using EXACTLY these six keys, in this order:\n"
    '  "stance"      - "asserting" or "neutral", from step 1\n'
    '  "addresses"   - "yes" or "no", from step 2\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "NO_EVIDENCE". '
    'If "addresses" is "no" you MUST answer "NO_EVIDENCE". '
    'Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are '
    "false, copied verbatim; otherwise an empty string\n"
    '  "evidence_id" - the id of the evidence passage you relied on\n'
    '  "reason"      - one sentence explaining the verdict\n\n'
    "Example:\n"
    '{"stance": "asserting", "addresses": "yes", "verdict": "FAKE_NEWS", '
    '"claim": "camels store water in their hump", "evidence_id": "DOCUMENT-ID", '
    '"reason": "The speaker asserts that camels store water in their hump; the '
    'evidence states that the hump stores fat, not water."}\n'
)

STANCE_FIRST_PARAMETRIC = (
    "You are a fact-checking classifier. You are given a CLAIM transcribed from a "
    "video. No evidence is provided: judge it from your own knowledge of the "
    "established scientific consensus. Work in the order below.\n\n"
    + _STANCE_STEP +
    "2. VERDICT.\n"
    '- "FAKE_NEWS": the stance is "asserting" and what the speaker states as fact '
    "is contradicted by the consensus. This includes stating that something "
    "contains, causes or does what it does not.\n"
    '- "NOT_FAKE_NEWS": the speaker agrees with the consensus, or is not asserting '
    '(stance "neutral").\n'
    '- "UNVERIFIABLE": you cannot determine from your own knowledge whether what the '
    "speaker says is true or false, or the passage makes no checkable factual claim. "
    "Use this rather than guessing.\n\n"
    + _NOT_FAKE_RULES +
    "Reply with a JSON object using EXACTLY these four keys, in this order:\n"
    '  "stance"      - "asserting" or "neutral", from step 1\n'
    '  "verdict"     - one of "FAKE_NEWS", "NOT_FAKE_NEWS", "UNVERIFIABLE". '
    'Use FAKE_NEWS only when stance is "asserting".\n'
    '  "claim"       - for FAKE_NEWS, the exact words from the CLAIM text that are '
    "false, copied verbatim; otherwise an empty string\n"
    '  "reason"      - one sentence explaining the verdict\n\n'
    "Example:\n"
    '{"stance": "asserting", "verdict": "FAKE_NEWS", '
    '"claim": "camels store water in their hump", '
    '"reason": "The speaker asserts that camels store water in their hump; the '
    'hump stores fat, not water."}\n'
)

# Output contracts for the two-value stance, in the new field order. Derived from
# the production ones so every other field stays identical.
import copy as _copy  # noqa: E402

from rag import llm as _llm  # noqa: E402

_STANCE2 = 'stance  ::= "\\"asserting\\"" | "\\"neutral\\""'
STANCE_FIRST_RAG_GRAMMAR = (
    _llm.VERDICT_GRAMMAR
    .replace('"\\"addresses\\":" ws yesno "," ws "\\"stance\\":" ws stance',
             '"\\"stance\\":" ws stance "," ws "\\"addresses\\":" ws yesno')
    .replace('stance  ::= "\\"asserting\\"" | "\\"denying\\"" | "\\"neutral\\""',
             _STANCE2)
)
STANCE_FIRST_PAR_GRAMMAR = _llm.PARAMETRIC_GRAMMAR.replace(
    'stance  ::= "\\"asserting\\"" | "\\"denying\\"" | "\\"neutral\\""', _STANCE2)
assert "denying" not in STANCE_FIRST_RAG_GRAMMAR + STANCE_FIRST_PAR_GRAMMAR
assert STANCE_FIRST_RAG_GRAMMAR.index('"\\"stance\\":"') < \
    STANCE_FIRST_RAG_GRAMMAR.index('"\\"addresses\\":"')


def _schema_two_stance(schema: dict, order: list[str]) -> dict:
    s = _copy.deepcopy(schema)
    s["name"] += "_stance_first"
    s["schema"]["properties"]["stance"]["enum"] = ["asserting", "neutral"]
    s["schema"]["properties"] = {k: s["schema"]["properties"][k] for k in order}
    s["schema"]["required"] = order
    return s


STANCE_FIRST_RAG_SCHEMA = _schema_two_stance(
    _llm.VERDICT_SCHEMA,
    ["stance", "addresses", "verdict", "claim", "evidence_id", "reason"])
STANCE_FIRST_PAR_SCHEMA = _schema_two_stance(
    _llm.PARAMETRIC_SCHEMA, ["stance", "verdict", "claim", "reason"])

PROMPT_PAIRS["stance_first"] = {
    "rag": STANCE_FIRST_RAG, "parametric": STANCE_FIRST_PARAMETRIC,
    "rag_grammar": STANCE_FIRST_RAG_GRAMMAR, "rag_schema": STANCE_FIRST_RAG_SCHEMA,
    "parametric_grammar": STANCE_FIRST_PAR_GRAMMAR,
    "parametric_schema": STANCE_FIRST_PAR_SCHEMA,
}


# --- rules_v2 ------------------------------------------------------------------
# rules + one fix per false-positive cause found on the dev set (2026-11-09,
# gpt-oss-120b, base prompt, overlap veto applied). Each cause, with the dev
# videos it accounted for:
#
#   minor figures / names    _PrUE69QZvA anaphylaxis "1 in 100,000"; UijQ-6jhVuc
#                            "about 40 cases"; LWgmluRbIag "mFluceva" (an ASR
#                            spelling of mFlusiva); Wtd0Hj00g_E "RNA does not
#                            make DNA"
#   different-thing evidence E-TuA4ZfN10 Pfizer efficacy refuted with Moderna's;
#                            Te38Qkxx4c8 mRNA flu figure vs a cell-based one;
#                            KdD76ta4NfA "can cause kidney failure" vs "safe for
#                            dialysis patients"
#   unconfirmed = refuted    RaLxhFiOBYk "the evidence does not confirm ...
#                            contradicting the evidence"
#   hedged forecasts         4gaG0rWGBd8 "likely did more harm than good";
#                            s7cpfJgYS5c "it is unlikely that there are going to
#                            be any serious long-term complications"
#   advertising              E-TuA4ZfN10 "Rise is the only app that ..."
#
# Time-bound statements, the remaining cause, are handled by --time-check
# (rag/temporal.py), not by wording here.

_V2_HEDGE_OLD = '- Hedged speculation: "might be true", "could be", "almost as if".\n'
_V2_HEDGE_NEW = ('- Hedged speculation or forecasts: "might be true", "could be", '
                 '"almost as if", "likely", "probably", "it is unlikely that".\n')
_V2_EXTRA_NOT_FAKE = (
    "- A figure, date, count or name that is approximately right, or whose error "
    "would not change what a listener concludes.\n"
    "- An odd spelling of a name: the CLAIM is an automatic transcription, so "
    "treat misspellings as transcription errors.\n"
    "- Advertising and sponsor messages.\n\n"
)
_V2_RAG_CONTRADICT = (
    "FAKE_NEWS needs evidence that states the opposite of what the speaker says. "
    "Evidence that does not mention it, or does not confirm it, is NO_EVIDENCE, "
    "not a contradiction.\n\n"
)
_V2_PAR_CONTRADICT = (
    "FAKE_NEWS needs a consensus that holds the opposite of what the speaker "
    "says. If what they say is merely unconfirmed, the verdict is UNVERIFIABLE, "
    "not FAKE_NEWS.\n\n"
)
_V2_ADDR_OLD = ("evidence about what camels eat does "
                'not address how camels store water. If the specific subject is absent, answer "no".')
_V2_ADDR_NEW = ("evidence about what camels eat does "
                "not address how camels store water. The evidence must be about the "
                "same thing, the same group and the same measure: a figure for one "
                "product, population or outcome does not address a claim about "
                'another. If the specific subject is absent, answer "no".')


def _v2(prompt: str, contradict: str, rag: bool) -> str:
    s = prompt
    for old in (_V2_HEDGE_OLD, _STANCE_AT):
        assert old in s, "rules prompt changed: " + old[:40]
    s = s.replace(_V2_HEDGE_OLD, _V2_HEDGE_NEW)
    # The rules block ends with a blank line right before the STANCE paragraph;
    # the new not-fake items extend that list, then the contradiction rule.
    s = s.replace("\n\n" + _STANCE_AT,
                  "\n" + _V2_EXTRA_NOT_FAKE + contradict + _STANCE_AT)
    if rag:
        assert _V2_ADDR_OLD in s, "addresses wording changed"
        s = s.replace(_V2_ADDR_OLD, _V2_ADDR_NEW)
    return s


PROMPT_PAIRS["rules_v2"] = {
    "rag": _v2(PROMPT_PAIRS["rules"]["rag"], _V2_RAG_CONTRADICT, rag=True),
    "parametric": _v2(PROMPT_PAIRS["rules"]["parametric"], _V2_PAR_CONTRADICT,
                      rag=False),
}


# rules_v2 is the production prompt (rag/llm.py) since 2026-11-09. Checked here
# so the variant and production cannot drift apart unnoticed: if rag/llm.py is
# edited, this fails until the variant is updated to match (or renamed).
from rag.llm import (  # noqa: E402
    PARAMETRIC_PROMPT as _PROD_PAR,
    SYSTEM_PROMPT as _PROD_RAG,
)

assert (PROMPT_PAIRS["rules_v2"]["rag"], PROMPT_PAIRS["rules_v2"]["parametric"]) \
    == (_PROD_RAG, _PROD_PAR), "rag/llm.py prompts differ from rules_v2"
