"""Exact-mutation and same-protein overlap for the models trained on the Oct06 capped split:
ESM-MSR (released/devel), the retrained MutateEverything stages, ThermoMPNN(-D), and retrained SPURS.

Training inventory = every substitution row of every capped-train scaffold (``code_wt``) in the Megascale CSV,
including mutant-background libraries (e.g. ``2M7O_I21A``, ``5GU9_L66K``, ``5GU9_V60D``). That is a superset
of what each individual model consumed, so exact overlaps reported here are conservative upper bounds on leak.
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR, hash_seq, norm_wt_name

os.makedirs(OUT_DIR, exist_ok=True)
msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
sp = pickle.load(open(os.path.join(REPO_ROOT, 'data', 'splits_oct06_capped.pkl'), 'rb'))
train = {norm_wt_name(x)[1] for x in sp['train']}
val = {norm_wt_name(x)[1] for x in sp['val']}
test = {norm_wt_name(x)[1] for x in sp['test']}
print('train/val/test code_wt scaffolds:', len(train), len(val), len(test))

msv = msv[msv['mutation'].notna()].copy()
msv['in_train'] = msv['code_wt'].isin(train)
trn = msv[msv['in_train']]
train_codes = set(trn['code'].unique()) | (train & set(lib.index))
print('train variant rows (substitutions):', len(trn), 'distinct library codes:', len(train_codes), 'scaffolds:', trn['code_wt'].nunique())

trn_key = set(zip(trn['code'], trn['mutation']))
trn_seq = {c: hash_seq(lib.loc[c, 'wt_seq']) for c in train_codes if c in lib.index}
seq_to_libs = collections.defaultdict(set)
for c, h in trn_seq.items():
    seq_to_libs[h].add(c)

out = {}
for f in sorted(os.listdir(os.path.join(WORK_DATA_DIR, 'testsets'))):
    ds = f[:-4]
    if ds in ('tsu_hyperopt_val', 'tsu_hyperopt_test'):
        continue
    d = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', f))
    d['pseq'] = d['seq'].map(hash_seq)
    n = len(d)
    exact = sameprot = 0
    for pseq, mut in zip(d['pseq'], d['mutation']):
        libs_hit = seq_to_libs.get(pseq)
        if not libs_hit:
            continue
        sameprot += 1
        if any((c, mut) in trn_key for c in libs_hit):
            exact += 1
    out[ds] = (n, exact, sameprot, sameprot - exact)
    print(f'{ds:22s} n={n:8d} exact={exact:8d} same_protein={sameprot:8d} sameprot_other_mut={sameprot - exact:8d}')

pd.DataFrame(
    [(k,) + v for k, v in out.items()],
    columns=['dataset', 'n_rows', 'n_exact', 'n_same_protein', 'n_same_protein_other'],
).to_csv(os.path.join(OUT_DIR, 'repo_models_exact_overlap.csv'), index=False)

trn[['code', 'code_wt', 'mutation']].assign(seq_hash=trn['code'].map(trn_seq)).to_pickle(
    os.path.join(WORK_DATA_DIR, 'capped_train_variants_exact.pkl')
)
with open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'wb') as f:
    pickle.dump({'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}, f)
print('saved capped_split_sets.pkl')
