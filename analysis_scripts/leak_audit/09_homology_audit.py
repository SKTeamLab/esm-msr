"""Validate the repo's homology split: does the rule hold, and is it sensitive/conservative?

Uses the pair table written by preprocessing/split_tsuboyama.py (saved under
analysis/salvaged_2026-10-16/structure-homology-splits-e5ca04/data/tsuboyama/homology/)
and the library assignment of the Oct06 capped split.
"""
import os, sys, json, pickle, collections
import pandas as pd, numpy as np, networkx as nx
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
PAIRS=os.path.join(ROOT,'analysis/salvaged_2026-10-16/structure-homology-splits-e5ca04/data/tsuboyama/homology/homology_pairs.csv')
pairs=pd.read_csv(PAIRS)
la=pd.read_csv(os.path.join(ROOT,'repo/data/splits_oct06_capped_library_assignment.csv'), index_col=0)
split=pickle.load(open(os.path.join(D,'capped_split_sets.pkl'),'rb'))
assign=la['split'].to_dict()
for x in ['GRB2','DLG4','EstA','Myo','GB1']: assign[x]='external'
extids=sorted({x for x in set(pairs.a)|set(pairs.b) if str(x).startswith('test_')})
print('external proteins in pair table', len(extids), 'functional 5; libraries', sum(1 for x in assign if assign[x] in ('train','val','test','excluded')))
res={}

def edges(df, seq_e=1e-3, st_e=1e-3, ident=0.40, tm=None, tm_norm='max'):
    m=(df['seq_evalue']<=seq_e)|(df['struct_evalue']<=st_e)
    if ident is not None: m|=df['identity']>ident
    if tm is not None: m|=df[f'struct_tm_{tm_norm}']>tm
    return m.fillna(False)

def cross(df, mask):
    d=df.assign(sa=df.a.map(assign), sb=df.b.map(assign))
    d=d[mask]
    return d[d.sa.notna() & d.sb.notna() & (d.sa!=d.sb)]

cfg=dict(seq_e=1e-3, st_e=1e-3, ident=0.40, tm=0.70)
m=edges(pairs, **cfg)
cross_d=cross(pairs, m)
real=cross_d[~cross_d.sa.isin(['excluded']) & ~cross_d.sb.isin(['excluded'])]
res['default_edges_total']=int(m.sum())
res['default_cross_edges_involving_excluded']=int(len(cross_d))
res['default_cross_edges_real']=int(len(real))
print('default rule: edges', int(m.sum()), 'cross(any)', len(cross_d), 'cross(non-excluded)', len(real))

# sensitivity sweep
sweep=[]
for name,kw in [
  ('split rule (E1e-3, id>0.40)', dict(seq_e=1e-3, st_e=1e-3, ident=0.40)),
  ('E<=1e-3 only', dict(seq_e=1e-3, st_e=1e-3, ident=None)),
  ('identity>0.30', dict(seq_e=1e-9, st_e=1e-9, ident=0.30)),
  ('identity>0.25', dict(seq_e=1e-9, st_e=1e-9, ident=0.25)),
  ('E<=1e-2', dict(seq_e=1e-2, st_e=1e-2, ident=None)),
  ('E<=1e-1', dict(seq_e=1e-1, st_e=1e-1, ident=None)),
  ('TM_max>=0.5', dict(seq_e=1e-9, st_e=1e-9, ident=None, tm=0.5)),
  ('TM_max>=0.4', dict(seq_e=1e-9, st_e=1e-9, ident=None, tm=0.4)),
  ('split rule + TM>=0.7', dict(seq_e=1e-3, st_e=1e-3, ident=0.40, tm=0.7)),
]:
    mm=edges(pairs, **kw); cc=cross(pairs, mm); rr=cc[~cc.sa.isin(['excluded']) & ~cc.sb.isin(['excluded'])]
    sweep.append(dict(rule=name, edges=int(mm.sum()), cross_any=len(cc), cross_real=len(rr),
                      pairs=len(rr)))
    print(f'{name:32s} edges={int(mm.sum()):6d} cross_any={len(cc):5d} cross_non_excluded={len(rr):5d}')
res['sweep']=sweep

# structure-only value
seqm=pairs['seq_evalue']<=1e-3; stm=pairs['struct_evalue']<=1e-3; idm=pairs['identity']>0.40
st_only=pairs[stm & ~seqm.fillna(False) & ~idm.fillna(False)]
res['pairs_struct_only']=int(len(st_only))
res['struct_only_cross_real']=int(len(cross(st_only, pd.Series(True,index=st_only.index)).pipe(lambda d: d[~d.sa.isin(['excluded']) & ~d.sb.isin(['excluded'])]) ))
print('structure-only pairs', len(st_only), 'of which cross-set (non-excluded):', res['struct_only_cross_real'])

# which of these structure-only cross pairs involve train/val/test vs external
so=cross(st_only, pd.Series(True,index=st_only.index))
so_ext=so[(so.sa.isin(['train','val','test']) & (so.sb=='external')) | (so.sb.isin(['train','val','test']) & (so.sa=='external'))]
print('structure-only train/val/test<->external pairs:', len(so_ext))
res['struct_only_external']=len(so_ext)

# transitive gap: components
g=nx.Graph(); g.add_edges_from(zip(pairs.loc[m,'a'],pairs.loc[m,'b']))
comp={}
for i,c in enumerate(nx.connected_components(g)):
    for n in c: comp[n]=i
extall=set(extids)|{'GRB2','DLG4','EstA','Myo','GB1'}
comp_ext={comp[x] for x in extall if x in comp}
gap={s:[] for s in ['train','val','test']}
for lib,s in assign.items():
    if s in gap and lib in comp and comp[lib] in comp_ext:
        gap[s].append(lib)
res['transitive_connected']={k:len(v) for k,v in gap.items()}
print('libraries in a component with an external protein (transitive):', {k:len(v) for k,v in gap.items()})
# for those, best direct similarity to any external
best=[]
for lib in gap['train']:
    sub=pairs[((pairs.a==lib)&(pairs.b.isin(extids))) | ((pairs.b==lib)&(pairs.a.isin(extids)))]
    if len(sub):
        best.append((lib, sub['identity'].max(), sub['struct_tm_max'].max(), sub['seq_evalue'].min(), sub['struct_evalue'].min()))
    else:
        best.append((lib, np.nan, np.nan, np.nan, np.nan))
bt=pd.DataFrame(best, columns=['lib','max_identity','max_tm','min_seq_e','min_struct_e'])
res['transitive_direct_stats']={c: (float(bt[c].max()) if bt[c].notna().any() else None) for c in ['max_identity','max_tm']}
res['transitive_has_no_direct_pair']=int(bt['max_identity'].isna().sum())
print('transitive train libs: no direct pair to any external:', res['transitive_has_no_direct_pair'], 'of', len(bt))
print('their max identity to an external: max', bt.max_identity.max(), 'median', bt.max_identity.median())
print('their max TM to an external: max', bt.max_tm.max())

# value of the exclusion: how many excluded libs have direct hits to external, and how strong
ex=la[la.split=='excluded']
exl=set(ex.index)
ex_pairs=pairs[((pairs.a.isin(exl))&(pairs.b.isin(extids))) | ((pairs.b.isin(exl))&(pairs.a.isin(extids)))]
res['excluded_direct_pairs']=int(len(ex_pairs))
allext=set(extids); res['excluded_libs_with_direct']=len((set(ex_pairs.a)|set(ex_pairs.b)) - allext)
print('excluded libs', len(exl), 'direct pairs to external', len(ex_pairs), 'libs with a direct pair', res['excluded_libs_with_direct'])
# how many would be strong (identity>0.9)
res['excluded_direct_identity_gt90']=int(((ex_pairs.identity>0.9)).sum())
print('of those pairs, identity>0.9:', res['excluded_direct_identity_gt90'], '; TM>0.9:', float((ex_pairs.struct_tm_max>0.9).sum()))
json.dump(res, open(os.path.join(ROOT,'leak_audit','out','homology_audit.json'),'w'), indent=2)
print('saved')
