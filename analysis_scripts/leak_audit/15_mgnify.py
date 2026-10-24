"""ESM3dG (MGnify Stability fine-tune) vs the benchmark: exact-sequence and homology overlap."""
import os, sys, pickle, collections, hashlib
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data'); OUT=os.path.join(ROOT,'leak_audit','out')
h=lambda s: hashlib.sha1(s.encode()).hexdigest()
m8=pd.read_csv(os.path.join(D,'raw','test_vs_mgnify.m8'), sep='\t', header=None,
               names=['q','t','fident','alnlen','qlen','tlen','evalue','bits'])
print('hits',len(m8))
m8['cov']=m8.alnlen/m8.qlen
best=m8.sort_values(['evalue','fident'],ascending=[True,False]).drop_duplicates('q')
# exact sequence match
train=set()
for line in open(os.path.join(D,'raw','mgnify_train.fa')):
    if not line.startswith('>'): train.add(h(line.strip()))
p=pd.read_pickle(os.path.join(D,'proteins.pkl'))
tq=p[p.source.str.startswith('test:')].copy(); tq['h']=tq.seq.map(h)
exact=set(tq.loc[tq.h.isin(train),'pid'])
print('test proteins with an exact MGnify sequence:',len(exact), sorted(exact)[:10])
cls={}
for q,f,c,e in zip(best['q'],best['fident'],best['cov'],best['evalue']):
    if q in exact: cls[q]='P'
    elif c>=0.5 and e<=1e-3: cls[q]='H'
    elif c>=0.3 and (e<=1e-2 or f>=0.30): cls[q]='h'
    else: cls[q]='low'
rows=[]
for f in os.listdir(os.path.join(D,'testsets')):
    ds=f[:-4]
    d=pd.read_pickle(os.path.join(D,'testsets',f))
    d['cls']=d.pid.map(lambda x: cls.get(x,'none'))
    rc=d['cls'].value_counts().to_dict()
    rows.append(dict(inventory='esm3dg_mgnify', dataset=ds, n=len(d),
                     n_sameprot=rc.get('P',0), n_H=rc.get('H',0), n_h=rc.get('h',0),
                     n_low=rc.get('low',0), n_none=rc.get('none',0)))
    print(f'{ds:22s} n={len(d):8d} P={rc.get("P",0):6d} H={rc.get("H",0):6d} h={rc.get("h",0):6d} low={rc.get("low",0):6d} none={rc.get("none",0):6d}')
df=pd.DataFrame(rows)
old=pd.read_csv(os.path.join(OUT,'homology_classes.csv'))
df=pd.concat([old,df],ignore_index=True)
df.to_csv(os.path.join(OUT,'homology_classes.csv'),index=False)
# exact coverage for Tsuboyama test/val specifically
for ds in ['tsu_capped_test','tsu_capped_val']:
    d=pd.read_pickle(os.path.join(D,'testsets',ds+'.pkl'))
    print(ds, 'rows with exact-seq protein in MGnify:', int(d.pid.isin(exact).sum()))
