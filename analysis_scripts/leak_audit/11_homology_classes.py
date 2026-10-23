"""Refined homolog classification at the level of whole test proteins.

H  (strong) : alignment covers >=50% of the TEST protein and (E<=1e-3 or identity>=40%)
h  (weak)   : alignment covers >=30% of the TEST protein and (E<=1e-2 or identity>=25%)
low         : any other hit (not counted as a leak)
"""
import os, sys, pickle, collections, json
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
OUT=os.path.join(ROOT,'leak_audit','out')
fm=pickle.load(open(os.path.join(D,'family_maps.pkl'),'rb')); maps=fm['maps']
p=pd.read_pickle(os.path.join(D,'proteins.pkl')); seqs=dict(zip(p.pid,p.seq)); fam=dict(zip(p.pid,p.family))
pairs=pd.read_csv(os.path.join(D,'mmseqs_pairs.csv'))
# symmetrise lookup: (a,b) -> best evidence
pk={}
for a,b,f,e,cv,al in zip(pairs.p,pairs.q,pairs.fident,pairs.evalue,pairs['coverage'],pairs.alnlen):
    pk[(a,b)]=(f,e,al)
def get(a,b):
    return pk.get((a,b)) or pk.get((b,a))

sp=pickle.load(open(os.path.join(D,'capped_split_sets.pkl'),'rb'))
msv=pd.read_pickle(os.path.join(D,'megascale_variants.pkl'))
lib=pd.read_pickle(os.path.join(D,'megascale_libraries.pkl'))
INV={}
INV['esm_msr_capped']={'lib:'+c for c in sp['train']}
INV['spurs_single']={'lib:'+c for c in pd.read_csv(os.path.join(D,'spurs_single_train_proteins.csv')).query('retained==True')['code_wt'].astype(str)}
INV['spurs_multi']={'lib:'+c for c in pd.read_csv(os.path.join(D,'spurs_multi_train_proteins.csv')).query('retained==True')['code_wt'].astype(str)}
et=pd.read_csv(os.path.join(D,'esmtherm_train_proteins.csv'))
INV['esmtherm']={'lib:'+str(c).replace('.pdb','').replace('|','_') for c in et.loc[et.split.isin(['train','val']),'code']}
if os.path.exists(os.path.join(D,'ifum_train_proteins.csv')):
    fi=pd.read_csv(os.path.join(D,'ifum_train_proteins.csv'))
    INV['ifum']={'lib:'+str(c).replace('.pdb','').replace('|','_') for c in fi.loc[fi.split.isin(['train','val']),'code']}
# ESM3dG: all libraries except the ThermoMPNN megascale test set (its documented held-out set)
import pickle as pk2
tm=pk2.load(open('/home/sareeves/software/ThermoMPNN/dataset_splits/mega_splits.pkl','rb'))
print('thermompnn split', {k:len(v) for k,v in tm.items()})
tmtest=set(x.replace('.pdb','') for x in tm['test'])
INV['esm3dg_released']={'lib:'+c for c in lib.index if c not in tmtest}
INV['ddgemb_s2450']={x for x in p.pid if x.startswith('inv:ddgemb2450:')}
INV['ddmut_s9028']={x for x in p.pid if x.startswith('inv:ddmut9028:')}
INV['ddmut_sm1242']={x for x in p.pid if x.startswith('inv:ddmutsm1242:')}
INV['geoddg_s8754']={x for x in p.pid if x.startswith('inv:geoddg8754:')}
for k in INV: print(k, len(INV[k]))

# per test protein cached best hit per inventory
tests={}
for f in os.listdir(os.path.join(D,'testsets')):
    ds=f[:-4]
    if ds.startswith('tsu_hyperopt'): continue
    tests[ds]=pd.read_pickle(os.path.join(D,'testsets',f))

rows=[]
for iid, inv in INV.items():
    # index inventory proteins by family and by pid for quick lookup (small sets)
    invlist=list(inv)
    inv_fams={fam[x] for x in invlist if x in fam}
    for ds,d in tests.items():
        counts=collections.Counter()
        seen={}
        for pid in d.pid.unique():
            if fam.get(pid) in inv_fams:
                seen[pid]=(None,'sameprot'); continue
            best=None
            for q in invlist:
                r=get(pid,q)
                if r is None: continue
                f,e,al=r
                cov=al/len(seqs[pid])
                score=(e, -f)
                if best is None or score<best[0]: best=(score,f,e,cov,q)
            if best is None:
                counts['none']+=1
            else:
                _,f,e,cov,q=best
                if cov>=0.5 and (e<=1e-3 or f>=0.40): counts['H']+=1
                elif cov>=0.3 and (e<=1e-2 or f>=0.25): counts['h']+=1
                else: counts['low']+=1
            seen[pid]=(best,fam.get(pid))
        # row-weighted
        rel={pid:('sameprot' if v[1]=='sameprot' else
                  'sameprot' if v[0] and v[0][1]>=0.90 and v[0][3]>=0.5 else
                  'H' if v[0] and v[0][3]>=0.5 and v[0][2]<=1e-3 else
                  'h' if v[0] and v[0][3]>=0.3 and (v[0][2]<=1e-2 or v[0][1]>=0.30) else 'low' if v[0] else 'none') for pid,v in seen.items()}
        drel=d.pid.map(rel)
        rc=drel.value_counts().to_dict()
        rows.append(dict(inventory=iid, dataset=ds, n=len(d), **{'n_'+k:rc.get(k,0) for k in ['sameprot','H','h','low','none']}))
        print(f'{iid:20s} {ds:20s} P={rc.get("sameprot",0):7d} H={rc.get("H",0):7d} h={rc.get("h",0):7d} low={rc.get("low",0):7d} none={rc.get("none",0):7d}')
df=pd.DataFrame(rows).fillna(0)
df.to_csv(os.path.join(OUT,'homology_classes.csv'), index=False)
print('saved')
