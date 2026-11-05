"""Pick a stratified dev subset for prompt A/B testing.

`relevant_evidence` on a dev window is a hand-made judgement of whether the
corpus genuinely covers that window, which is the same question `addresses`
asks. So it serves as ground truth for the relevance gate with no new
labelling: non-empty => addresses should be "yes", empty => "no".

Stratified so a prompt cannot win by flagging less: half the cases are windows
the corpus DOES cover (where NO_EVIDENCE would be a miss) and half are windows
it does NOT (where a flag is unsupported). Deterministic by SHA-1 of video id +
window, matching tests/select_dev.py.
"""
import json, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
items = json.load(open(ROOT/'tests/chunk_eval_labels_dev2.json'))['items']
items += json.load(open(ROOT/'tests/chunk_eval_labels_dev1_topic.json'))['items']

def key(i): return hashlib.sha1(f"{i['video_id']}#{i['window']}".encode()).hexdigest()

covered   = sorted([i for i in items if i.get('relevant_evidence')], key=key)
uncovered = sorted([i for i in items if not i.get('relevant_evidence')], key=key)

# spread across topics rather than taking the first N of each list
def spread(pool, n):
    by = {}
    for i in pool: by.setdefault(i.get('topic'), []).append(i)
    out, t = [], sorted(by)
    while len(out) < n and any(by.values()):
        for topic in t:
            if by[topic] and len(out) < n: out.append(by[topic].pop(0))
    return out

cases = []
for i in spread(covered, N_PER):
    cases.append({**i, 'expect_addresses': 'yes'})
for i in spread(uncovered, N_PER):
    cases.append({**i, 'expect_addresses': 'no'})

out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('prompt_cases.json')
out.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
import collections
print(f'{len(cases)} cases -> {out}')
print('expect yes:', sum(1 for c in cases if c['expect_addresses']=='yes'))
print('expect no :', sum(1 for c in cases if c['expect_addresses']=='no'))
print('topics:', dict(collections.Counter(c.get('topic') for c in cases)))
