"""Exact sequence-level overlap between ESM3dG's MGnify training corpus and our test rows.

A test row's mutant sequence is 'in training' iff the exact mutant sequence appears among the
960,216 training sequences. This is stronger than a mutation-string match: it catches reverse
mutations (a training mutant equal to the test wild type is not a leak; a training sequence equal
to the test mutant is) and constructs/numbering differences.
"""
import os, sys, hashlib, collections
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
OUT=os.path.join(ROOT,'leak_audit','out')
h=lambda s: hashlib.sha1(s.encode()).hexdigest()
idx=pd.read_csv(os.path.join(D,'raw','mgnify_index.csv'), usecols=['name','split','seq','PDB_name'])
train=idx[idx.split=='train']
tset=set(train.seq.map(h)); wtset=set(idx.loc[idx.split!='train','seq'].map(h))  # held-out seqs, for curiosity
print('train seqs',len(tset),'held-out seqs',len(wtset))

def apply_muts(seq, mut):
    s=list(seq)
    for part in str(mut).split(';'):
        if not part: continue
        m=re.match(r'^([A-Z])(\d+)([A-Z])$',part) if (re:=__import__('re')) else None
        if not m: return None
        wt,p,mt=m.group(1),int(m.group(2)),m.group(3)
        if p<1 or p>len(s) or s[p-1]!=wt: return None
        s[p-1]=mt
    return ''.join(s)

def check(df, label, seqcol=None, mutcol='mutation'):
    n=len(df); wt=0; mut=0; mrows=[]; per=collections.Counter()
    for i,(sq,mu) in enumerate(zip(df[seqcol] if seqcol else df['seq'], df[mutcol])):
        if h(sq) in tset: wt+=1
        ms=apply_muts(sq,mu) if seqcol is None else sq
        if ms is None: continue
        if h(ms) in tset:
            mut+=1
            if len(mrows)<20: mrows.append((i,mu,ms))
    print(f'{label:24s} rows={n:8d} WT_in_train={wt:7d} mutant_in_train={mut:7d}')
    return dict(dataset=label,n=n,wt_in_train=wt,mutant_in_train=mut), mrows

res=[]
# capped Tsuboyama
for split in ['val','test']:
    d=pd.read_csv(os.path.join(ROOT,'remote_data',f'splits_oct06_capped-{split}.csv.gz'), low_memory=False)
    r,_=check(d,f'tsu_capped_{split}',seqcol='aa_seq',mutcol='mut_type'); res.append(r)
# ProteinGym: all 217 DMS, and the 64 Tsuboyama ones
pgm=pd.read_csv(os.path.join(ROOT,'ProteinGym','DMS_substitutions.csv'))
pgdir=os.path.join(ROOT,'ProteinGym','DMS_ProteinGym_substitutions','DMS_ProteinGym_substitutions')
rows=[]; tsu_rows=[]
for _,r in pgm.iterrows():
    f=os.path.join(pgdir,r['DMS_filename'])
    if not os.path.exists(f): continue
    d=pd.read_csv(f, usecols=['mutant','mutated_sequence'])
    n=len(d); hits=sum(1 for x in d['mutated_sequence'] if h(str(x)) in tset)
    is_tsu='Tsuboyama' in str(r['DMS_id'])
    rows.append((r['DMS_id'],n,hits,is_tsu))
pg=pd.DataFrame(rows,columns=['DMS','n','hits','tsuboyama'])
print(f'proteingym all: {int(pg.hits.sum())} rows / {len(pg)} DMS; Tsuboyama: {int(pg[pg.tsuboyama].hits.sum())}/{int(pg[pg.tsuboyama].n.sum())} rows in {int((pg[pg.tsuboyama].hits>0).sum())} DMS')
res.append(dict(dataset='proteingym_all',n=int(pg.n.sum()),wt_in_train=np.nan,mutant_in_train=int(pg.hits.sum())))
res.append(dict(dataset='proteingym_tsuboyama',n=int(pg[pg.tsuboyama].n.sum()),wt_in_train=np.nan,mutant_in_train=int(pg[pg.tsuboyama].hits.sum())))
# external + DMS + Domainome from the canonical testsets
for ds in ['s571','s783','s2648','s8754','s669','s461','ssym','q3421','k3822','k2369','ptmul','ptmuld',
           'DLG4 abund.','DLG4 bind.','GRB2 abund.','GRB2 bind.','MYO','ESTA','GB1','domainome']:
    d=pd.read_pickle(os.path.join(D,'testsets',ds+'.pkl'))
    r,_=check(d,ds); res.append(r)
df=pd.DataFrame(res); df.to_csv(os.path.join(OUT,'esm3dg_exact.csv'),index=False)
pg.to_csv(os.path.join(OUT,'esm3dg_pg_exact_per_dms.csv'),index=False)
print('saved')
