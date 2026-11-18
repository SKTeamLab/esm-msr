"""The MGnify Stability + Rocklin-design rows the baselines score (data/mgnify/mgnify_mapped.csv, built by preprocessing/preprocess_mgnify.py).

The table is large (450 MB with labels), and a baseline only needs to know which structure and mutation to score, so the
cluster gets two small files instead (`build` writes them from the full table; labels stay on the workstation and are merged on `uid`):

  codes.csv.gz   one row per scaffold: code, source, pdb_file (relative to the structure root), chain, wt_seq, bb (the structure
                 carries a different residue than the library's wild type, `backbone_mutation`, e.g. A19D)
  rows.csv.gz    one row per variant: uid, code, mut_type (structure numbering, `A19D` or `A19D:E59C`; 1 to 3 mutations)

`load(dir, sources)` joins them into the frame the baselines score: id, code, chain, pdb_file (absolute, under dir), mut_type,
overwrite (the wild-type sequence to put on the structure's backbone for a backbone-mutated scaffold, else None).
A backbone-mutated library is a wild-type backbone carrying a background substitution; scoring it means reading the background
sequence on that backbone, as every baseline already does for the Tsuboyama mutant-background libraries.
"""
import os

import pandas as pd

SOURCES = ('mgnify', 'rocklin_denovo')


def build(table, out_dir):
    cols = ['uid', 'source', 'code', 'chain', 'pdb_file', 'backbone_mutation', 'mut_type', 'wt_seq']
    d = pd.read_csv(table, usecols=cols, low_memory=False)
    d['bb'] = d['backbone_mutation'].notna() & (d['backbone_mutation'].astype(str) != '')
    codes = d.drop_duplicates('code')[['code', 'source', 'pdb_file', 'chain', 'wt_seq', 'bb']]
    assert d.groupby('code')[['pdb_file', 'wt_seq']].nunique().max().max() == 1, 'a code with two structures or sequences'
    assert d['uid'].is_unique
    os.makedirs(out_dir, exist_ok=True)
    codes.to_csv(os.path.join(out_dir, 'codes.csv.gz'), index=False)
    d[['uid', 'code', 'mut_type']].to_csv(os.path.join(out_dir, 'rows.csv.gz'), index=False)
    print(f'{len(d)} rows, {len(codes)} scaffolds ({int(codes.bb.sum())} backbone-mutated)')


def load(data_dir, sources=SOURCES):
    codes = pd.read_csv(os.path.join(data_dir, 'codes.csv.gz'), low_memory=False)
    codes = codes[codes['source'].isin(sources)]
    rows = pd.read_csv(os.path.join(data_dir, 'rows.csv.gz'), low_memory=False)
    df = rows.merge(codes, on='code', how='inner', validate='many_to_one')
    df['pdb_file'] = [os.path.normpath(os.path.join(data_dir, p)) for p in df['pdb_file']]
    df['overwrite'] = df['wt_seq'].where(df['bb'])
    df = df.rename(columns={'uid': 'id'})
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'overwrite', 'source']].reset_index(drop=True)


def chunks(df, n):
    """(index, frame) pieces of about `n` rows each, in first-appearance order of the scaffolds; a scaffold is never split."""
    per = df.groupby('code', sort=False).size()
    k = (per.cumsum() - per) // n
    for i, g in df.groupby(df['code'].map(k), sort=True):
        yield int(i), g


if __name__ == '__main__':
    import sys
    build(sys.argv[1], sys.argv[2])
