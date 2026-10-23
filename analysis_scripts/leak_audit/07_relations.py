"""Per-test-protein relation to a training protein set (capped train libraries by default)."""
import os, sys, pickle, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
p=pd.read_pickle(os.path.join(D,'proteins.pkl'))
edges=pd.read_pickle(os.path.join(D,'protein_homologs.pkl'))
split=pickle.load(open(os.path.join(D,'capped_split_sets.pkl'),'rb'))
train_libs=set(split['train'])

# family -> train library members
fam2libs=collections.defaultdict(set)
for pid,fam in zip(p.pid,p.family):
    if pid.startswith('lib:') and pid[4:] in train_libs:
        fam2libs[fam].add(pid)
# edges indexed
h_by_pid=collections.defaultdict(list)
for a,b,f,e,cv,h,w in zip(edges.a,edges.b,edges.fident,edges.evalue,edges['cov'],edges.homolog,edges.weak):
    h_by_pid[a].append((b,f,e,cv,h,w)); h_by_pid[b].append((a,f,e,cv,h,w))

def relation(pid):
    fam=p.loc[p.pid==pid,'family'].iloc[0]
    if fam in fam2libs: return 'family', 1.0, 0.0, sorted(fam2libs[fam])[0]
    best=None
    for q,f,e,cv,h,w in h_by_pid.get(pid,[]):
        if q.startswith('lib:') and q[4:] in train_libs:
            if best is None or e<best[1]: best=(q,f,e,cv,h,w)
    if best is None: return 'none', 0.0, 1.0, None
    q,f,e,cv,h,w=best
    if h: return 'homolog', f, e, q
    if w: return 'weak', f, e, q
    return 'none', f, e, q

rows=[]
for f in sorted(os.listdir(os.path.join(D,'testsets'))):
    ds=f[:-4]
    if ds.startswith('tsu_hyperopt'): continue
    d=pd.read_pickle(os.path.join(D,'testsets',f))
    rel={}
    for pid in d.pid.unique():
        rel[pid]=relation(pid)
    d['rel']=d.pid.map(lambda x: rel[x][0])
    d['bestID']=d.pid.map(lambda x: rel[x][1])
    g=d.groupby('rel').size().to_dict()
    rows.append(dict(dataset=ds, n=len(d), **g))
    print(f'{ds:22s} rows={len(d):8d} ' + ' '.join(f'{k}={v}' for k,v in sorted(g.items())))
pd.DataFrame(rows).fillna(0).to_csv(os.path.join(ROOT,'leak_audit','out','relation_to_capped_train.csv'), index=False)
