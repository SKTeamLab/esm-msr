"""Per-DMS ProteinGym leak coverage and the effect of removing contaminated DMS."""
import os, sys, pickle, collections, json
import pandas as pd, numpy as np
sys.path.insert(0,os.path.dirname(__file__))
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data'); OUT=os.path.join(ROOT,'leak_audit','out')
ov=pd.read_csv(os.path.join(OUT,'inventory_overlaps.csv'))
# re-use the canonical keys from the engine to get per-DMS coverage: recompute with same logic
fm=pickle.load(open(os.path.join(D,'family_maps.pkl'),'rb')); maps=fm['maps']
pid_fam={p:(v[0]) for p,v in maps.items()}
pg=pd.read_pickle(os.path.join(D,'testsets','proteingym.pkl'))
pg['DMS']=pg['pid'].str.replace('pg:','',regex=False)
excl={'BRCA1_HUMAN_Findlay_2018','BRCA2_HUMAN_Erwood_2022_HEK293T','POLG_CXB3N_Mattenberger_2021','SCN5A_HUMAN_Glazer_2019'}
pg=pg[~pg.DMS.isin(excl)]
# We already have exact hit flags for the capped inventory computed in 04; recompute for all inventories
from lib import parse_mut_str
_pos={}
def canon_one(pid,wt,pos,mt):
    k=(pid,pos)
    if k in _pos: rp=_pos[k]
    else:
        fam,mp,rs=maps[pid]; rp=mp.get(pos)
        if rp is not None and (rp<1 or rp>len(rs) or rs[rp-1]!=wt):
            alt=None
            for d in (1,-1,2,-2,3,-3):
                q=rp+d
                if 1<=q<=len(rs) and rs[q-1]==wt: alt=q; break
            rp=alt
        _pos[k]=rp
    return None if rp is None else (maps[pid][0],rp,wt,mt)
# build inventory variant/single maps
msv=pd.read_pickle(os.path.join(D,'megascale_variants.pkl')); msv=msv[msv.mutation.notna()]
msv['n']=msv.mutation.str.count(';')+1
sp=pickle.load(open(os.path.join(D,'capped_split_sets.pkl'),'rb'))
lib=pd.read_pickle(os.path.join(D,'megascale_libraries.pkl'))
INV={}
def add_ms(iid,codes,keep=None):
    v=collections.defaultdict(set); s=collections.defaultdict(set); fams=set()
    sub=msv[msv.code.isin(codes)]
    if keep is not None: sub=sub[keep(sub)]
    for code,mutstr in zip(sub.code,sub.mutation):
        pid='lib:'+code
        if pid not in maps: continue
        keys=set(); ok=True
        for wt,pos,mt in parse_mut_str(mutstr) or []:
            c=canon_one(pid,wt,pos,mt)
            if c is None: ok=False; break
            keys.add(c)
        if not ok: continue
        f=pid_fam[pid]; v[f].add(frozenset(keys)); s[f].update(keys); fams.add(f)
    INV[iid]=(v,s,fams)
add_ms('esm_msr_capped', sp['train'])
add_ms('me_stage1', sp['train'], lambda d:d.n==1)
add_ms('me_stage2', sp['train'])
add_ms('tmpnn', sp['train'], lambda d:d.n==1)
add_ms('tmpnnd', sp['train'], lambda d:d.n>=2)
tm=pickle.load(open('/home/sareeves/software/ThermoMPNN/dataset_splits/mega_splits.pkl','rb'))
tmtest=set(x.replace('.pdb','') for x in tm['test'])
add_ms('esm3dg_released',[c for c in lib.index if c not in tmtest])
spm=pd.read_csv(os.path.join(D,'spurs_train_mutations.csv.gz'))
for iid in ['spurs_single','spurs_multi']:
    v=collections.defaultdict(set); s=collections.defaultdict(set); fams=set()
    for code,mutstr in zip(spm[spm.model==iid].code_wt, spm[spm.model==iid].mutation):
        pid='lib:'+str(code)
        if pid not in maps: continue
        keys=set(); ok=True
        for wt,pos,mt in parse_mut_str(mutstr) or []:
            c=canon_one(pid,wt,pos,mt)
            if c is None: ok=False; break
            keys.add(c)
        if not ok: continue
        f=pid_fam[pid]; v[f].add(frozenset(keys)); s[f].update(keys); fams.add(f)
    INV[iid]=(v,s,fams)
# per-DMS
rows=[]
pgkey=[]
for pid,mut in zip(pg.pid,pg.mutation):
    if pid not in maps: pgkey.append(None); continue
    keys=set(); ok=True
    for wt,pos,mt in parse_mut_str(mut) or []:
        c=canon_one(pid,wt,pos,mt)
        if c is None: ok=False; break
        keys.add(c)
    pgkey.append(frozenset(keys) if ok else None)
pg['keys']=pgkey
pg['family']=pg.pid.map(pid_fam)
for iid,(v,s,fams) in INV.items():
    exact=[0]*len(pg); const=[0]*len(pg)
    for i,(fam,k) in enumerate(zip(pg['family'], pg['keys'])):
        if fam in fams and k is not None and k:
            if k in v.get(fam,()): exact[i]=1
            elif k <= s.get(fam,set()): const[i]=1
    pg['exact']=exact; pg['const']=const
    g=pg.groupby('DMS').agg(n=('exact','size'),exact=('exact','sum'),const=('const','sum'))
    g['frac']=(g.exact+g.const)/g.n
    g['inventory']=iid
    rows.append(g.reset_index())
    leak=g[g.frac>=0.5]
    print(f'{iid:20s} DMS with >=50% rows trained: {len(leak):3d}/213; rows exact+const {int(g.exact.sum()+g.const.sum()):8d}/{int(g.n.sum())}')
pd.concat(rows).to_csv(os.path.join(OUT,'pg_leak_per_dms.csv'),index=False)

# effect on ESM-MSR devel aggregates (local per-DMS stats)
m=pd.read_csv(os.path.join(ROOT,'ProteinGym','DMS_substitutions.csv'))[['DMS_id','coarse_selection_type']]
summ={}
for s in [1,2,3]:
    d=pd.read_csv(f'{ROOT}/pgym_results/unmasked-esm-msr-seed{s}/sigma1.0/summary.csv')
    col=[c for c in d.columns if 'spearman' in c][0]
    summ[s]=d.set_index('DMS_id')[col]
df=pd.concat(summ.values(),axis=1).mean(axis=1).rename('rho').reset_index().rename(columns={'index':'DMS_id'})
df=df[~df.DMS_id.isin(excl)].merge(m,on='DMS_id')
leaked=set(pd.read_csv(os.path.join(OUT,'pg_leak_per_dms.csv')).query('inventory=="esm_msr_capped" and frac>=0.5').DMS)
df['leak']=df.DMS_id.isin(leaked)
def agg(x):
    cats=x.groupby('coarse_selection_type')['rho'].mean()
    return dict(PGavg=cats.mean(), DMSmean=x.rho.mean(), n=len(x), **{c:cats.get(c,np.nan) for c in ['Activity','Binding','Expression','OrganismalFitness','Stability']})
out={'all':agg(df),'clean':agg(df[~df.leak]),'leaked':agg(df[df.leak]),'n_leaked':len(leaked)}
json.dump(out, open(os.path.join(OUT,'esm_msr_pg_clean.json'),'w'), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
