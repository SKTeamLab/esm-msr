"""Determine which capped-train scaffolds ThermoMPNN-D (the epistatic network) actually trained on.

ThermoMPNN-D trains on the measured doubles of the prepared items table plus upstream's ``double-aug``
doubles, which are built from WT-backbone singles that have a Rosetta model
(``<rosetta_dir>/<library>/pdb_models/A[<mutation>].pdb``). A scaffold with neither a measured double nor
a Rosetta-modelled single (most de novo designed miniproteins) is never seen by ThermoMPNN-D, leaving
134 of the 172 capped-train scaffolds.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, REPO_DATA_DIR, SOFTWARE_DIR, WORK_DATA_DIR, norm_wt_name


def main():
    default_items = os.path.join(LEAK_ROOT, 'baselines', 'retrained', 'thermompnn', 'data', 'allbl', 'data', 'items_train.csv')
    default_rosetta = os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'tsuboyama', 'FINAL_results')
    default_out = os.path.join(REPO_DATA_DIR, 'tmpnnd_train_proteins.csv')

    ap = argparse.ArgumentParser()
    ap.add_argument('--items', default=default_items, help='items_train.csv of the prepared ThermoMPNN-D tables')
    ap.add_argument('--rosetta_dir', default=default_rosetta)
    ap.add_argument('--out', default=default_out)
    a = ap.parse_args()

    if not os.path.exists(a.items):
        if os.path.exists(a.out):
            print(f'--items ({a.items}) not found; keeping existing {a.out}')
            os.makedirs(WORK_DATA_DIR, exist_ok=True)
            shutil.copy(a.out, os.path.join(WORK_DATA_DIR, 'tmpnnd_train_proteins.csv'))
            return
        raise FileNotFoundError(f'Missing items_train.csv at {a.items}')

    it = pd.read_csv(a.items, usecols=['code', 'code_wt', 'mut_type', 'kind'], low_memory=False)
    it['code_wt_norm'] = [norm_wt_name(x)[1] for x in it['code_wt']]
    has_double = set(it.loc[it['kind'] == 'double', 'code_wt_norm'])
    singles = it.loc[it['kind'] == 'single', ['code_wt', 'code_wt_norm', 'mut_type']].drop_duplicates()
    aug = set()
    if os.path.isdir(a.rosetta_dir):
        for (raw_cwt, norm_cwt), g in singles.groupby(['code_wt', 'code_wt_norm']):
            for cand in (str(raw_cwt), str(norm_cwt)):
                d = os.path.join(a.rosetta_dir, cand, 'pdb_models')
                if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f'A[{m}].pdb')) for m in g['mut_type']):
                    aug.add(norm_cwt)
                    break
    else:
        # When the multi-GB cluster Rosetta archive (FINAL_results) is not mounted locally,
        # determine Rosetta model coverage from ThermoMPNN's Zenodo library index
        # (mega_splits.pkl / thermompnn_upstream_train_proteins.csv), which indexes the exact
        # 298 MegaScale libraries for which Dieckhaus et al. computed Rosetta single-mutant models.
        import pickle
        rosetta_scaffolds = set()
        mega_pkl = os.path.join(SOFTWARE_DIR, 'ThermoMPNN', 'dataset_splits', 'mega_splits.pkl')
        up_csv = os.path.join(REPO_DATA_DIR, 'thermompnn_upstream_train_proteins.csv')
        if os.path.exists(mega_pkl):
            ms = pickle.load(open(mega_pkl, 'rb'))
            rosetta_scaffolds = {norm_wt_name(x)[1] for v in ms.values() for x in v}
        elif os.path.exists(up_csv):
            rosetta_scaffolds = set(pd.read_csv(up_csv)['code'].astype(str))
        if not rosetta_scaffolds and os.path.exists(a.out):
            print(f'--rosetta_dir ({a.rosetta_dir}) not found; keeping existing {a.out}')
            os.makedirs(WORK_DATA_DIR, exist_ok=True)
            shutil.copy(a.out, os.path.join(WORK_DATA_DIR, 'tmpnnd_train_proteins.csv'))
            return
        aug = set(singles['code_wt_norm']) & rosetta_scaffolds
    rows = [
        dict(code_wt=c, measured_doubles=c in has_double, rosetta_augmentation=c in aug)
        for c in sorted(set(it['code_wt_norm']))
    ]
    df = pd.DataFrame(rows)
    df['retained'] = df['measured_doubles'] | df['rosetta_augmentation']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    os.makedirs(WORK_DATA_DIR, exist_ok=True)
    work_copy = os.path.join(WORK_DATA_DIR, 'tmpnnd_train_proteins.csv')
    if os.path.abspath(a.out) != os.path.abspath(work_copy):
        shutil.copy(a.out, work_copy)
    print(
        f'{len(df)} libraries: {int(df.measured_doubles.sum())} with measured doubles, '
        f'{int(df.rosetta_augmentation.sum())} with a Rosetta-modelled single, '
        f'{int(df.retained.sum())} seen by ThermoMPNN-D -> {a.out}'
    )


if __name__ == '__main__':
    main()
