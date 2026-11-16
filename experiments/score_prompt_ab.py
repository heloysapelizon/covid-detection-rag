"""Score the prompt A/B run against the hand-made coverage judgement."""
import json, sys, collections
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
names = sorted({p for r in d.values() for p in r['by_prompt']})
complete = {k: r for k, r in d.items() if all(p in r['by_prompt'] for p in names)}
print(f'{len(complete)} cases scored on {len(names)} prompts: {", ".join(names)}\n')

for p in names:
    tp=fp=fn=tn=0; flags=0; verd=collections.Counter()
    for r in complete.values():
        got = (r['by_prompt'][p].get('addresses') or '').lower()
        exp = r['expect_addresses']
        v = r['by_prompt'][p].get('verdict')
        verd[v]+=1
        if v=='FAKE_NEWS': flags+=1
        if exp=='yes' and got=='yes': tp+=1
        elif exp=='yes' and got=='no': fn+=1
        elif exp=='no' and got=='yes': fp+=1
        else: tn+=1
    n=tp+fp+fn+tn
    acc=(tp+tn)/n if n else 0
    print(f'--- {p}')
    print(f'  addresses accuracy {acc:.3f}   ({tp} correct-yes, {tn} correct-no,')
    print(f'                                  {fp} false-yes, {fn} false-no)')
    print(f'  false-yes = evidence does NOT cover it but model said it does'
          f'  -> unsupported-flag risk')
    print(f'  verdicts: {dict(verd)}   FAKE_NEWS={flags}')
    print()

if len(names)==2:
    a,b=names
    diff=[(k,r) for k,r in complete.items()
          if r['by_prompt'][a].get('verdict')!=r['by_prompt'][b].get('verdict')]
    print(f'{len(diff)} case(s) where the verdict differs:\n')
    for k,r in diff:
        print(f'  {k}  topic={r["topic"]}  expect_addresses={r["expect_addresses"]}')
        for p in names:
            x=r['by_prompt'][p]
            print(f'    {p:10} {x.get("verdict"):14} addresses={x.get("addresses")} '
                  f'stance={x.get("stance")}')
            print(f'               reason: {(x.get("reason") or "")[:95]}')
        print(f'    window: {r["window_text"][:140]}...')
        print()
