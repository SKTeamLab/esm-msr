"""Exact sequence-level overlap between ESM3dG's MGnify training corpus and all benchmark test rows.

A test row's mutant sequence is 'in training' iff the exact mutant sequence appears among the
960,216 MGnify training sequences.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, OUT_DIR, WORK_DATA_DIR, apply_mutations, check_against_seq, data_file, parse_mut_str

os.makedirs(OUT_DIR, exist_ok=True)
h = lambda s: hashlib.sha1(s.encode()).hexdigest()

idx = pd.read_csv(data_file('mgnify_index.csv'), usecols=['name', 'split', 'seq', 'PDB_name'])
train = idx[idx['split'] == 'train']
tset = set(train['seq'].astype(str).map(h))
wtset = set(idx.loc[idx['split'] != 'train', 'seq'].astype(str).map(h))
print('MGnify train seqs:', len(tset), 'held-out seqs:', len(wtset))


def check(df, label, seqcol=None):
    n = len(df)
    wt = mut = 0
    if seqcol is not None:
        for sq in df[seqcol].astype(str):
            if h(sq) in tset:
                mut += 1
    else:
        for sq, mu in zip(df['seq'].astype(str), df['mutation'].astype(str)):
            if h(sq) in tset:
                wt += 1
            muts = parse_mut_str(mu)
            if not muts:
                continue
            ok, _ = check_against_seq(sq, muts)
            if not ok:
                continue
            ms = apply_mutations(sq, muts)
            if h(ms) in tset:
                mut += 1
    print(f'{label:24s} rows={n:8d} WT_in_train={wt:7d} mutant_in_train={mut:7d}')
    return dict(dataset=label, n=n, wt_in_train=wt, mutant_in_train=mut)


res = []
for split in ('val', 'test'):
    d = pd.read_csv(os.path.join(LEAK_ROOT, 'remote_data', f'splits_oct06_capped-{split}.csv.gz'), low_memory=False)
    wt_hits = sum(1 for sq in d['ref_seq'].astype(str) if h(sq) in tset)
    r = check(d, f'tsu_capped_{split}', seqcol='aa_seq')
    r['wt_in_train'] = wt_hits
    res.append(r)

pg_root = os.environ.get('PROTEINGYM_DIR', os.path.join(LEAK_ROOT, 'ProteinGym'))
pgm = pd.read_csv(os.path.join(pg_root, 'DMS_substitutions.csv'))
pgdir = os.path.join(pg_root, 'DMS_ProteinGym_substitutions', 'DMS_ProteinGym_substitutions')
rows = []
for _, r in pgm.iterrows():
    f = os.path.join(pgdir, r['DMS_filename'])
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f, usecols=['mutant', 'mutated_sequence'])
    n = len(d)
    hits = sum(1 for x in d['mutated_sequence'] if h(str(x)) in tset)
    is_tsu = 'Tsuboyama' in str(r['DMS_id'])
    rows.append((r['DMS_id'], n, hits, is_tsu))

pg = pd.DataFrame(rows, columns=['DMS', 'n', 'hits', 'tsuboyama'])
print(
    f'proteingym all: {int(pg.hits.sum())} rows / {len(pg)} DMS; '
    f'Tsuboyama: {int(pg[pg.tsuboyama].hits.sum())}/{int(pg[pg.tsuboyama].n.sum())} rows'
)
res.append(dict(dataset='proteingym_all', n=int(pg['n'].sum()), wt_in_train=np.nan, mutant_in_train=int(pg['hits'].sum())))
res.append(dict(dataset='proteingym_tsuboyama', n=int(pg[pg['tsuboyama']]['n'].sum()), wt_in_train=np.nan, mutant_in_train=int(pg[pg['tsuboyama']]['hits'].sum())))

for ds in [
    's571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld',
    'DLG4 abund.', 'DLG4 bind.', 'GRB2 abund.', 'GRB2 bind.', 'MYO', 'ESTA', 'GB1', 'domainome',
]:
    d = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', f'{ds}.pkl'))
    res.append(check(d, ds))

pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, 'esm3dg_exact.csv'), index=False)
pg.to_csv(os.path.join(OUT_DIR, 'esm3dg_pg_exact_per_dms.csv'), index=False)
print('saved esm3dg_exact.csv and esm3dg_pg_exact_per_dms.csv')
