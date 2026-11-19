"""Task tables of the stability-baselines harness (~/stability-baselines/harness/run_model.py, adapters ESMtherm / IFUM / GeoDDG / ESM3dG ...)
for the MGnify scaffolds and Rocklin designs of data/mgnify (the same rows mgnify_rows.py / esm_msr_testing.py --mgnify score).

One table per ~--rows rows, whole scaffolds only, named mg_<source>_NNN (tasks/<name>.csv.gz): run_model.py skips a finished table, so a killed
run resumes at the next one, and the tables are the unit of parallelism (job k takes tables k, k+n, ...).  Columns are those build_tasks.py
writes (id, row_id, dataset, group, code, chain, pdb_file, backbone_mutation, ref_seq, mut_renum, mut_pdb, truth, n_mut, mut_type, code_wt).
Conventions as in build_tasks.build_tsuboyama: ref_seq = the library wild type (the backbone-mutated scaffolds' structure differs from it at
exactly the backbone_mutation positions), mut_renum numbers 1..N on ref_seq, truth = ddG_ML (positive = stabilising), group = code.

  python build_mgnify_harness_tasks.py --mgnify_dir ~/esm-msr-devel/data/mgnify --out ~/stability-baselines/tasks [--rows 25000]
"""
import argparse
import os

import pandas as pd

COLS = ['uid', 'source', 'code_wt', 'code', 'chain', 'pdb_file', 'backbone_mutation', 'mut_type', 'mut_type_renumbered', 'mut_type_pdb', 'n_mut',
        'ddG_ML', 'dddG_ML', 'wt_seq']
SHORT = {'mgnify': 'mgnify', 'rocklin_denovo': 'rocklin'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mgnify_dir', required=True)
    ap.add_argument('--table', default=None, help='default: <mgnify_dir>/mgnify_mapped.csv.gz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', type=int, default=25000)
    a = ap.parse_args()
    d = pd.read_csv(a.table or os.path.join(a.mgnify_dir, 'mgnify_mapped.csv.gz'), usecols=lambda c: c in COLS, low_memory=False)
    d['pdb_file'] = [os.path.join(a.mgnify_dir, p) if not os.path.isabs(p) else p for p in d['pdb_file']]
    d['backbone_mutation'] = d['backbone_mutation'].fillna('').astype(str).replace({'nan': '', '-': ''})
    d['ref_seq'] = d['wt_seq']
    for m, s in zip(d['mut_type_renumbered'], d['ref_seq']):          # the harness' own indexing guard, early and cheap
        for x in m.split(':'):
            assert s[int(x[1:-1]) - 1] == x[0], (m, s)
    d['mut_renum'], d['mut_pdb'], d['truth'], d['group'] = d['mut_type_renumbered'], d['mut_type_pdb'], d['ddG_ML'], d['code']
    d = d.rename(columns={'uid': 'id'})
    os.makedirs(a.out, exist_ok=True)
    n_tab = 0
    for src, g in d.groupby('source', sort=False):
        codes = list(dict.fromkeys(g['code']))
        by = {c: x for c, x in g.groupby('code', sort=False)}
        cur, n, k = [], 0, 0
        for i, c in enumerate(codes):
            cur.append(by[c]); n += len(by[c])
            if n >= a.rows or i == len(codes) - 1:
                name = f'mg_{SHORT[src]}_{k:03d}'
                t = pd.concat(cur)
                t['dataset'] = name
                t['row_id'] = range(len(t))
                t.to_csv(os.path.join(a.out, f'{name}.csv.gz'), index=False)
                cur, n, k = [], 0, k + 1
                n_tab += 1
        print(f'{src}: {len(g)} rows, {len(codes)} scaffolds -> {k} tables', flush=True)
    print(f'{n_tab} task tables in {a.out}')


if __name__ == '__main__':
    main()
