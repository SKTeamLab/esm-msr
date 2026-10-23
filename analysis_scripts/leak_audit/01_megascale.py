"""Build the compact Megascale/Tsuboyama library table used by every trained-on-Megascale inventory."""
import os, sys, pickle, re
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import parse_mut_str, check_against_seq, hash_seq

ROOT = '/home/sareeves/playground/esm-msr-devel'
OUT = os.path.join(ROOT, 'leak_audit', 'data')
os.makedirs(OUT, exist_ok=True)
CSV = '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'

cols = ['name', 'aa_seq', 'mut_type', 'WT_name', 'WT_cluster', 'ddG_ML', 'dG_ML', 'pair_name']
df = pd.read_csv(CSV, usecols=cols, low_memory=False)
print('rows', len(df))
df['lib_raw'] = df['WT_name'].astype(str)
df['code'] = df['lib_raw'].str.replace(r'\.pdb$', '', regex=True).str.replace('|', '_', regex=False)
df['is_wt'] = df['mut_type'].astype(str) == 'wt'
print('libraries', df['code'].nunique(), 'wt rows', df['is_wt'].sum())

# wild-type sequence per library
wt = df.loc[df['is_wt']].groupby('code')['aa_seq'].agg(lambda s: s.iloc[0])
print('libs with wt row', len(wt))
# some libraries may only have mutant rows; fall back to the first aa_seq of that library
missing = set(df['code'].unique()) - set(wt.index)
print('libs missing wt row', len(missing))
for c in sorted(missing):
    wt[c] = df.loc[df['code'] == c, 'aa_seq'].iloc[0]
wt = wt.sort_index()

# mutation counts
df['n_mut'] = df['mut_type'].astype(str).str.count(':').add(1).where(~df['is_wt'], 0)
lib = pd.DataFrame({
    'n_single': df.loc[df['n_mut'] == 1].groupby('code').size(),
    'n_double': df.loc[df['n_mut'] == 2].groupby('code').size(),
    'n_multi': df.loc[df['n_mut'] >= 3].groupby('code').size(),
    'n_wt': df.loc[df['is_wt']].groupby('code').size(),
    'n_rows': df.groupby('code').size(),
}).fillna(0).astype(int)
lib['wt_seq'] = wt
lib['natural'] = [bool(re.match(r'^[0-9][A-Za-z0-9]{3}$', c)) or c.startswith('v2_') for c in lib.index]
lib['seq_hash'] = lib['wt_seq'].map(hash_seq)
lib.index.name = 'code'
lib.to_pickle(os.path.join(OUT, 'megascale_libraries.pkl'))
print(lib[['n_single', 'n_double', 'n_multi', 'natural']].sum().to_string())

# variant table (mutants only), with parsed mutations
v = df.loc[~df['is_wt']].copy()
v['lib_seq'] = v['code'].map(wt)
parsed = v['mut_type'].map(parse_mut_str)
v['muts'] = parsed
fam = parsed.map(lambda m: None if m is None else ';'.join(f'{w}{p}{mt}' for w, p, mt in sorted(m, key=lambda x: x[1])))
v['mutation'] = fam
ok = parsed.map(lambda m: m is not None)
print('unparsed rows', (~ok).sum())
sub = v.loc[ok].copy()
bad = [check_against_seq(s, m) for s, m in zip(sub['lib_seq'], sub['muts'])]
print('rows with wt mismatch against library seq:', sum(1 for b in bad if not b[0]))
v[['code', 'mut_type', 'mutation', 'n_mut', 'ddG_ML', 'dG_ML', 'pair_name']].to_pickle(
    os.path.join(OUT, 'megascale_variants.pkl'))
print('written', len(v))
