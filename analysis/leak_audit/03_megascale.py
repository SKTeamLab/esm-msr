"""Build the compact Megascale/Tsuboyama library and variant tables used by Megascale inventories, and the
``capped_split_sets.pkl`` scaffold lists (train / val / test ``code_wt``) of ``data/splits_oct06_capped.pkl``.

Normalizes both ``code`` (preserving mutant-background suffixes such as ``2MKX_R27A``) and ``code_wt``
(base scaffold such as ``2MKX``), and synthesizes base ``code_wt`` library entries when a scaffold exists
solely as mutant-background libraries so protein-level inventories (IFUM, EsmTherm) never miss scaffolds.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import REPO_ROOT, SOFTWARE_DIR, WORK_DATA_DIR, check_against_seq, hash_seq, norm_wt_name, parse_mut_str

os.makedirs(WORK_DATA_DIR, exist_ok=True)
CSV = os.environ.get(
    'TSUBOYAMA_CSV',
    os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'tsuboyama', 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
)
AF_DIR = os.environ.get(
    'TSUBOYAMA_AF_DIR',
    os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'tsuboyama', 'AlphaFold_model_PDBs'),
)
AA3 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M',
}


def af_pdb_seq(code_wt: str) -> str:
    for stem in (code_wt, code_wt.replace('_', ':')):
        p = os.path.join(AF_DIR, f'{stem}.pdb')
        if os.path.exists(p):
            seen, seq = set(), []
            for line in open(p, errors='ignore'):
                if line.startswith('ATOM  '):
                    k = (line[21], line[22:27])
                    if k not in seen:
                        seen.add(k)
                        seq.append(AA3.get(line[17:20].strip(), 'X'))
            if seq:
                return ''.join(seq)
    return ''


cols = ['name', 'aa_seq', 'mut_type', 'WT_name', 'WT_cluster', 'ddG_ML', 'dG_ML', 'pair_name']
df = pd.read_csv(CSV, usecols=cols, low_memory=False)
print('rows', len(df))

normed = [norm_wt_name(x) for x in df['WT_name'].astype(str)]
df['code'] = [t[0] for t in normed]
df['code_wt'] = [t[1] for t in normed]
df['bg_mut'] = [t[2] for t in normed]
df['is_wt'] = df['mut_type'].astype(str) == 'wt'
print('unique code:', df['code'].nunique(), 'unique code_wt:', df['code_wt'].nunique(), 'wt rows:', int(df['is_wt'].sum()))

# Starting sequence per library code (WT sequence for WT libraries; background sequence for .pdb_<mut> libraries)
wt = df.loc[df['is_wt']].groupby('code')['aa_seq'].agg(lambda s: s.iloc[0])
missing = set(df['code'].unique()) - set(wt.index)
print('libs with wt row:', len(wt), 'libs missing wt row:', len(missing))
for c in sorted(missing):
    c_wt = df.loc[df['code'] == c, 'code_wt'].iloc[0]
    af_s = af_pdb_seq(c_wt)
    if af_s and c == c_wt:
        wt[c] = af_s
    else:
        wt[c] = df.loc[df['code'] == c, 'aa_seq'].iloc[0]
wt = wt.sort_index()

df['n_mut'] = df['mut_type'].astype(str).str.count(':').add(1).where(~df['is_wt'], 0)
code_to_cwt = df.groupby('code')['code_wt'].first()
lib = pd.DataFrame({
    'code_wt': code_to_cwt,
    'n_single': df.loc[df['n_mut'] == 1].groupby('code').size(),
    'n_double': df.loc[df['n_mut'] == 2].groupby('code').size(),
    'n_multi': df.loc[df['n_mut'] >= 3].groupby('code').size(),
    'n_wt': df.loc[df['is_wt']].groupby('code').size(),
    'n_rows': df.groupby('code').size(),
}).fillna({'n_single': 0, 'n_double': 0, 'n_multi': 0, 'n_wt': 0, 'n_rows': 0})
for col in ['n_single', 'n_double', 'n_multi', 'n_wt', 'n_rows']:
    lib[col] = lib[col].astype(int)
lib['wt_seq'] = wt

# Synthesize base code_wt entries for scaffolds that only appear as .pdb_<mut> mutant-background libraries
bg_only_scaffolds = sorted(set(df['code_wt'].unique()) - set(lib.index))
print('mutant-background-only base scaffolds synthesized:', len(bg_only_scaffolds), bg_only_scaffolds)
synth_rows = []
for cwt in bg_only_scaffolds:
    seq = af_pdb_seq(cwt)
    if not seq:
        # Revert the background substitution on the first background library's wt_seq
        sub_c = df.loc[df['code_wt'] == cwt].iloc[0]
        bg_seq = str(wt[sub_c['code']])
        bg_muts = parse_mut_str(sub_c['bg_mut']) or []
        arr = list(bg_seq)
        for orig_aa, pos, mut_aa in bg_muts:
            if 1 <= pos <= len(arr) and arr[pos - 1] == mut_aa:
                arr[pos - 1] = orig_aa
        seq = ''.join(arr)
    synth_rows.append({
        'code': cwt,
        'code_wt': cwt,
        'n_single': 0,
        'n_double': 0,
        'n_multi': 0,
        'n_wt': 0,
        'n_rows': 0,
        'wt_seq': seq,
    })
if synth_rows:
    synth_df = pd.DataFrame(synth_rows).set_index('code')
    lib = pd.concat([lib, synth_df]).sort_index()

lib['seq_hash'] = lib['wt_seq'].map(hash_seq)
lib.index.name = 'code'
lib.to_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
print(lib[['n_single', 'n_double', 'n_multi']].sum().to_string())

# Variant table (mutants only), with parsed mutations validated against their library's starting sequence
v = df.loc[~df['is_wt']].copy()
v['lib_seq'] = v['code'].map(lib['wt_seq'])
parsed = v['mut_type'].map(parse_mut_str)
v['muts'] = parsed
v['mutation'] = parsed.map(lambda m: None if m is None else ';'.join(f'{w}{p}{mt}' for w, p, mt in sorted(m, key=lambda x: x[1])))
ok = parsed.map(lambda m: m is not None)
print('unparsed rows', int((~ok).sum()))
sub = v.loc[ok].copy()
bad = [check_against_seq(s, m) for s, m in zip(sub['lib_seq'], sub['muts'])]
print('rows with wt mismatch against library seq:', sum(1 for b in bad if not b[0]))
v[['code', 'code_wt', 'mut_type', 'mutation', 'n_mut', 'ddG_ML', 'dG_ML', 'pair_name']].to_pickle(
    os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl')
)
print('written', len(v))

# Scaffold (code_wt) sets of the Oct06 capped split, shared by every downstream step
sp = pickle.load(open(os.path.join(REPO_ROOT, 'data', 'splits_oct06_capped.pkl'), 'rb'))
sets = {k: sorted({norm_wt_name(x)[1] for x in sp[k]}) for k in ('train', 'val', 'test')}
print('capped train/val/test scaffolds:', {k: len(v) for k, v in sets.items()})
with open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'wb') as f:
    pickle.dump(sets, f)
