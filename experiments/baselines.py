"""Reference classifiers for the test set (Section 5.4.2): all-positive, date and provenance
rules, and TF-IDF + logistic regression with repeated stratified 10-fold CV."""
import json, sys, numpy as np, random
sys.path.insert(0, '.')
from rag.pipeline import overlap_veto
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

FLAG={'FAKE_NEWS','FAKE_NEWS_WEB'}; CLR={'NOT_FAKE_NEWS','NOT_FAKE_NEWS_WEB'}
files={'Gemma 4 12B':('test_run_gemma-4-12b_parametric.json','test_run_gemma-4-12b_tavily.json'),
 'gpt-oss-20b':('test_run_gpt-oss-20b_parametric.json','test_run_gpt-oss-20b_tavily.json'),
 'Qwen3.6-27B':('test_run_qwen-qwen3.6-27b_parametric_nothink.json','test_run_qwen-qwen3.6-27b_tavily_nothink.json'),
 'gpt-oss-120b':('test_run_openai-gpt-oss-120b_parametric.json','test_run_openai-gpt-oss-120b_tavily.json')}
gt={x['video_id']:x for x in json.load(open('data/labels/ground_truth.json'))['items']}
meta={x['video_id']:x for x in json.load(open('data/youtube/covid_vaccine_videos_transcribed.json'))}
def load(f): return json.load(open('data/results/v1/'+f))['videos']
preds={}
ids=None
for m,(pf,rf) in files.items():
    P,R=load(pf),load(rf)
    ids=sorted(P) if ids is None else ids
    assert set(P)==set(ids)==set(R)
    pv={v:overlap_veto(P[v]['results']) for v in ids}; rv={v:overlap_veto(R[v]['results']) for v in ids}
    fl=lambda res:{r['window'] for r in res if r['verdict'] in FLAG}
    cl=lambda res:{r['window'] for r in res if r['verdict'] in CLR}
    preds[m]={'param':{v:bool(fl(pv[v])) for v in ids},'veto':{v:bool(fl(pv[v])-cl(rv[v])) for v in ids}}
y={v:bool(gt[v]['label_fake_news']) for v in ids}
year={v:int(meta[v]['published_at'][:4]) for v in ids}
prov={v:meta[v]['query'] for v in ids}
text={v:meta[v]['transcript'] for v in ids}
print('n',len(ids),'pos',sum(y.values()), 'provenances',sorted(set(prov.values())))
def sc(p,sub=None):
    vs=ids if sub is None else sub
    tp=sum(p[v] and y[v] for v in vs); fp=sum(p[v] and not y[v] for v in vs); fn=sum(not p[v] and y[v] for v in vs); tn=len(vs)-tp-fp-fn
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F=2*tp/(2*tp+fp+fn) if tp else 0
    return dict(TP=tp,FP=fp,FN=fn,P=round(P,3),R=round(R,3),F1=round(F,3),Acc=round((tp+tn)/len(vs),3))
def f1(p,vs):
    tp=sum(p[v] and y[v] for v in vs); fp=sum(p[v] and not y[v] for v in vs); fn=sum(not p[v] and y[v] for v in vs)
    return 2*tp/(2*tp+fp+fn) if tp else 0
def mcn(a,b):
    n01=sum(a[v]==y[v] and b[v]!=y[v] for v in ids); n10=sum(a[v]!=y[v] and b[v]==y[v] for v in ids)
    p=binomtest(n01,n01+n10,0.5).pvalue if n01+n10 else 1
    return n01,n10,round(p,3)
B={}
B['all-positive']={v:True for v in ids}
B['date>=2023 (in-sample)']={v:year[v]>=2023 for v in ids}
yrs=sorted(set(year.values())); loo={}; chosen=[]
for v in ids:
    rest=[u for u in ids if u!=v]
    best=max(yrs,key=lambda t:(f1({u:year[u]>=t for u in rest},rest),-t))
    chosen.append(best); loo[v]=year[v]>=best
B['date (LOO threshold)']=loo
print('LOO thresholds',sorted(set(chosen)))
B['DD list only']={v:prov[v].lower().startswith('disinformation') or 'dozen' in prov[v].lower() for v in ids}
loo={}
for v in ids:
    rest=[u for u in ids if u!=v and prov[u]==prov[v]]
    loo[v]=(sum(y[u] for u in rest)/len(rest)>0.5) if rest else False
B['provenance majority (LOO)']=loo
for k,p in B.items(): print(f'{k:30s}',sc(p))
X=[text[v] for v in ids]; Y=np.array([y[v] for v in ids])
def cv(feat='text',reps=10):
    res=[];agg=[]
    for s in range(reps):
        skf=StratifiedGroupKFold(10,shuffle=True,random_state=s) if feat.endswith('grouped') else StratifiedKFold(10,shuffle=True,random_state=s); pr=np.zeros(len(ids),bool)
        groups=[meta[v]['channel_id'] for v in ids]
        for tr,te in (skf.split(X,Y,groups) if feat.endswith('grouped') else skf.split(X,Y)):
            clf=make_pipeline(TfidfVectorizer(sublinear_tf=True,ngram_range=(1,2),min_df=2,stop_words='english'),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
            if feat.startswith('text') and 'year' not in feat:
                clf.fit([X[i] for i in tr],Y[tr]); pr[te]=clf.predict([X[i] for i in te])
            else:
                aug=[f'YEAR{year[ids[i]]} '*20+X[i] for i in range(len(ids))]
                clf.fit([aug[i] for i in tr],Y[tr]); pr[te]=clf.predict([aug[i] for i in te])
        p={v:bool(pr[i]) for i,v in enumerate(ids)}; res.append(sc(p)); agg.append(p)
    F=[r['F1'] for r in res]; A=[r['Acc'] for r in res]; Pp=[r['P'] for r in res]; Rr=[r['R'] for r in res]
    return dict(F1=round(np.mean(F),3),F1sd=round(np.std(F),3),F1min=min(F),F1max=max(F),P=round(np.mean(Pp),3),R=round(np.mean(Rr),3),Acc=round(np.mean(A),3)),agg
for C in (1,10):
    pass
r,agg=cv('text'); print('TF-IDF+LR (10x10-fold)',r)
B['tfidf']=agg[0]
r2,_=cv('text grouped'); print('TF-IDF+LR grouped by channel',r2)
print()
for m in preds:
    for c in ('param','veto'):
        print(f'{m:14s}{c:6s}',sc(preds[m][c]),'McNemar vs date:',mcn(preds[m][c],B['date>=2023 (in-sample)']),'vs tfidf seed0:',mcn(preds[m][c],B['tfidf']))
post=[v for v in ids if year[v]>=2023]; pre=[v for v in ids if year[v]<2023]
print('\nstrata: post',len(post),sum(y[v] for v in post),'pre',len(pre),sum(y[v] for v in pre))
print('date rule post',sc(B['date>=2023 (in-sample)'],post),'pre',sc(B['date>=2023 (in-sample)'],pre))
for m in preds:
    for c in ('param','veto'):
        print(f'{m:14s}{c:6s} post',sc(preds[m][c],post),' pre',sc(preds[m][c],pre))
rng=np.random.default_rng(0); bs=[]
for _ in range(2000):
    s=list(rng.choice(ids,len(ids)))
    p=B['date>=2023 (in-sample)']; tp=sum(p[v] and y[v] for v in s); fp=sum(p[v] and not y[v] for v in s); fn=sum(not p[v] and y[v] for v in s)
    bs.append(2*tp/(2*tp+fp+fn))
print('date CI',np.percentile(bs,[2.5,97.5]).round(3))
