"""Rosetta Cartesian ddG on ProteinGym: per-DMS Spearman, SINGLE MUTANTS ONLY.

The predictions (proteingym_cartesian_preds_09_11.csv, September 2024) cover single substitutions only, keyed
'P<i>_<pos><aa>': <i> is the row of ProteinGym's reference table DMS_substitutions.csv (217 rows; index 25, BRCA2, is the
one absent, and it is excluded from every comparison anyway), <pos> the position in the target sequence and <aa> the
substituted residue. A DMS's number is therefore the Spearman over its single mutants that Rosetta scored, which is
NOT comparable to a model scored on all variants including multi-mutants, and the tables mark it as such.

The sign is checked rather than assumed. A raw Rosetta ddG is positive for destabilizing substitutions and would need
negating, but this file does not follow that convention: `cartesian_ddg_dir` correlates +0.64 with stability on the
Stability DMS, i.e. it is already direction-corrected (higher = more stable), so it is used as is. The script measures
this on the Stability DMS, where the right sign is unambiguous, and refuses to run if it ever stops holding.

  python analysis_scripts/pgym_rosetta_singles.py --preds proteingym_cartesian_preds_09_11.csv --out pgym_rosetta_singles_per_dms.csv
"""
import argparse

import numpy as np
import pandas as pd

REF = '/home/sreeves/ProteinGym/DMS_substitutions.csv'
INP = '/home/sreeves/esm-msr-devel/pgym_inputs'
CATEGORY = '/home/sreeves/esm-msr-devel/outputs/pgym/pgym_all_models_per_dms.csv'


def rho(a, b):
    v = pd.DataFrame({'a': a, 'b': b}).dropna()
    return float(v.corr('spearman').iloc[0, 1]) if len(v) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out', default='pgym_rosetta_singles_per_dms.csv')
    a = ap.parse_args()
    d = pd.read_csv(a.preds).rename(columns={'Unnamed: 0': 'key', 'cartesian_ddg_dir': 'cart'})
    d['i'] = d.key.str.extract(r'^P(\d+)_')[0].astype(int)
    d['pm'] = d.key.str.split('_', n=1).str[1]                       # '<pos><aa>', no wild-type letter
    ref = pd.read_csv(REF)
    cat = pd.read_csv(CATEGORY).set_index('DMS_id').category
    rows = []
    for i, dms in enumerate(ref.DMS_id):
        if dms not in cat.index:
            continue
        sub = d[d.i == i]
        mine = pd.read_csv(f'{INP}/{dms}.csv', usecols=['mutant', 'DMS_score'])
        mine = mine[~mine.mutant.str.contains(':')].copy()           # singles, on the variants every model saw
        mine['pm'] = mine.mutant.str[1:]                              # drop the wild-type letter: 'A100C' -> '100C'
        j = mine.merge(sub[['pm', 'cart']].drop_duplicates('pm'), on='pm', how='left')
        rows.append(dict(DMS_id=dms, category=cat[dms], n_singles=len(mine), n_scored=int(j.cart.notna().sum()),
                         rho_raw=rho(j.cart, j.DMS_score)))
    D = pd.DataFrame(rows)
    stab = D[D.category == 'Stability'].rho_raw.mean()
    print(f'mean rho of cartesian_ddg_dir over {int((D.category == "Stability").sum())} Stability DMS: {stab:+.3f} '
          '(positive: the file is already stabilizing-positive and is used as is)')
    if not stab > 0.2:
        raise SystemExit('sign check failed: cartesian_ddg_dir no longer correlates positively with stability')
    D['rho'] = D.rho_raw
    D.drop(columns='rho_raw').to_csv(a.out, index=False)
    print(f'{len(D)} DMS -> {a.out}; singles scored {D.n_scored.sum()} of {D.n_singles.sum()} '
          f'({100 * D.n_scored.sum() / D.n_singles.sum():.1f}%); mean rho {D.rho.mean():.4f}; '
          f'DMS with <50% of singles scored: {int((D.n_scored < 0.5 * D.n_singles).sum())}')


if __name__ == '__main__':
    main()
