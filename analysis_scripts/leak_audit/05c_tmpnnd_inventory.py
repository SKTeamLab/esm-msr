"""Which capped-train libraries ThermoMPNN-D (the epistatic network) actually trained on, for the protein-level classes of 11_homology_classes.py.

It trains on the measured doubles of the prepared items table plus upstream's `double-aug` doubles, which are built from the WT-backbone
singles that have a Rosetta model (`<rosetta_dir>/<library>/pdb_models/A[<mutation>].pdb`). A library with neither a measured double nor a
Rosetta-modelled single (most designed libraries) is never seen, unlike ThermoMPNN's single network and every other retrained model, which
train on all 172 capped-train libraries (their P/H/h columns are identical by construction; ThermoMPNN-D's can only be a subset).

  python 05c_tmpnnd_inventory.py --items outputs/baselines/oct06_capped_allbl/data/items_train.csv \\
      --rosetta_dir ~/software/esm-msr/data/tsuboyama/FINAL_results --out tmpnnd_train_proteins.csv
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True, help='items_train.csv of the prepared tables the ThermoMPNN-D run trained on')
    ap.add_argument('--rosetta_dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    it = pd.read_csv(a.items, usecols=['code', 'code_wt', 'mut_type', 'kind'], low_memory=False)
    has_double = set(it.loc[it['kind'] == 'double', 'code_wt'])
    singles = it.loc[it['kind'] == 'single', ['code_wt', 'mut_type']].drop_duplicates()
    aug = set()
    for code, g in singles.groupby('code_wt'):
        d = os.path.join(a.rosetta_dir, code, 'pdb_models')
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f'A[{m}].pdb')) for m in g['mut_type']):
            aug.add(code)
    rows = [dict(code_wt=c, measured_doubles=c in has_double, rosetta_augmentation=c in aug) for c in sorted(set(it['code_wt']))]
    df = pd.DataFrame(rows)
    df['retained'] = df.measured_doubles | df.rosetta_augmentation
    df.to_csv(a.out, index=False)
    print(f'{len(df)} libraries: {int(df.measured_doubles.sum())} with measured doubles, {int(df.rosetta_augmentation.sum())} with a Rosetta-modelled single, '
          f'{int(df.retained.sum())} seen by ThermoMPNN-D -> {a.out}')


if __name__ == '__main__':
    main()
