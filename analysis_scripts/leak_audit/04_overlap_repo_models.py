"""Exact-mutation and same-protein overlap for the models trained on the Oct06 capped split:
ESM-MSR (released/devel), the retrained MutateEverything stages and ThermoMPNN(-D).

Training inventory = every substitution row of every capped-train library in the Megascale CSV.
That is a superset of what each baseline actually consumed (ME stage1 only singles, TMPNN-D only
doubles, both apply filters), so an overlap found here is a lower bound on the leak.
"""
import os, sys, pickle, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import hash_seq

ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
msv=pd.read_pickle(os.path.join(D,'megascale_variants.pkl'))
lib=pd.read_pickle(os.path.join(D,'megascale_libraries.pkl'))
sp=pickle.load(open(os.path.join(ROOT,'repo','data','splits_oct06_capped.pkl'),'rb'))
code=lambda x: str(x).replace('.pdb','').replace('|','_')
train=set(map(code,sp['train'])); val=set(map(code,sp['val'])); test=set(map(code,sp['test']))
print('train/val/test libs', len(train), len(val), len(test))
msv=msv[msv['mutation'].notna()].copy()
msv['in_train']=msv['code'].isin(train)
trn=msv[msv['in_train']]
print('train variant rows (substitutions)', len(trn), 'libraries with rows', trn.code.nunique())
# exact mutation keys per protein sequence
trn_key=set(zip(trn['code'], trn['mutation']))
trn_seq={c: hash_seq(lib.loc[c,'wt_seq']) for c in train if c in lib.index}
seq_to_libs=collections.defaultdict(set)
for c,h in trn_seq.items(): seq_to_libs[h].add(c)

out={}
for f in sorted(os.listdir(os.path.join(D,'testsets'))):
    ds=f[:-4]
    d=pd.read_pickle(os.path.join(D,'testsets',f))
    if ds in ('tsu_hyperopt_val','tsu_hyperopt_test'): continue
    d['pseq']=d['seq'].map(hash_seq)
    n=len(d)
    # exact: same protein sequence AND same mutation string
    exact=0; sameprot=0; per_lib=collections.Counter()
    for pseq, mut, pid in zip(d['pseq'], d['mutation'], d['pid']):
        libs=seq_to_libs.get(pseq)
        if not libs: continue
        sameprot+=1
        if any((c,mut) in trn_key for c in libs):
            exact+=1; per_lib[pid]+=1
    out[ds]=(n, exact, sameprot, sameprot-exact)
    print(f'{ds:22s} n={n:8d} exact={exact:8d} same_protein={sameprot:8d} sameprot_other_mut={sameprot-exact:8d}')
pd.DataFrame([(k,)+v for k,v in out.items()], columns=['dataset','n_rows','n_exact','n_same_protein','n_same_protein_other']).to_csv(
    os.path.join(ROOT,'leak_audit','out','repo_models_exact_overlap.csv'), index=False)
# save the train-key inventory for reuse (protein seq hash + mutation)
trn[['code','mutation']].assign(seq_hash=trn['code'].map(trn_seq)).to_pickle(os.path.join(D,'capped_train_variants_exact.pkl'))
# also save per-library lists
with open(os.path.join(D,'capped_split_sets.pkl'),'wb') as f:
    pickle.dump({'train':sorted(train),'val':sorted(val),'test':sorted(test)}, f)
print('saved')
