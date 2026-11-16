"""Run two adjudication prompts over identical retrieval and compare."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag.index import EvidenceIndex
from rag.llm import adjudicate, health, SYSTEM_PROMPT
from rag.config import TOP_K, MIN_SIMILARITY
from prompt_variants import V2_DIRECT

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: python experiments/prompt_ab.py CASES.json OUT.json")
    raise SystemExit(2)
cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
PROMPTS = {'current': SYSTEM_PROMPT, 'v2_direct': V2_DIRECT}

ok, detail = health(); print('server:', detail, flush=True)
idx = EvidenceIndex.load()
cases = json.loads(cases_path.read_text())
done = json.loads(out_path.read_text()) if out_path.exists() else {}
print(f'{len(cases)} cases x {len(PROMPTS)} prompts, {len(done)} done', flush=True)

t0 = time.time()
for n, c in enumerate(cases, 1):
    key = f"{c['video_id']}#{c['window']}"
    if key in done and all(p in done[key].get('by_prompt', {}) for p in PROMPTS):
        continue
    txt = c['window_text']
    ps = idx.search(txt, top_k=TOP_K, min_similarity=MIN_SIMILARITY)
    rec = done.get(key, {
        'topic': c.get('topic'), 'expect_addresses': c['expect_addresses'],
        'window_text': txt, 'relevant_evidence': c.get('relevant_evidence'),
        'retrieved': [{'id': p['id'], 'sim': round(p.get('similarity') or 0, 4),
                       'text': p.get('text', '')[:300]} for p in ps],
        'by_prompt': {}})
    for name, sp in PROMPTS.items():
        if name in rec['by_prompt']: continue
        v = adjudicate(txt, ps, system_prompt=sp)
        rec['by_prompt'][name] = {k: v.get(k) for k in
            ('verdict','addresses','stance','claim','evidence_id','reason')}
    done[key] = rec
    out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    if n % 5 == 0:
        print(f'  {n}/{len(cases)}  {(time.time()-t0)/60:.1f}min', flush=True)
print(f'done in {(time.time()-t0)/60:.1f} min', flush=True)
