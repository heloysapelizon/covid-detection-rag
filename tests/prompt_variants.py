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
