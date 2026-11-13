#!/usr/bin/env python
"""EsmTherm training inventory: which Megascale wild types EsmTherm trained on.

Reads EsmTherm's ``datasets/split/wildtype_split.csv`` (local clone, or the tracked copy
``data/esmtherm_split.csv``) and writes ``data/esmtherm_train_proteins.csv`` (one row per wild type x split, with
expected row counts from the Dataset2 proxy and capped-split membership). ``.pdb_<mut>`` mutant-background names
are normalised to their parent scaffold.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import REPO_DATA_DIR, REPO_ROOT, SOFTWARE_DIR, WORK_DATA_DIR, data_file, norm_wt_name

SMALL_OUT = Path(REPO_DATA_DIR)
WORK_OUT = Path(WORK_DATA_DIR)
SMALL_OUT.mkdir(parents=True, exist_ok=True)
WORK_OUT.mkdir(parents=True, exist_ok=True)

ESMTHERM = Path(os.environ.get('ESMTHERM_DIR', os.path.join(SOFTWARE_DIR, 'EsmTherm')))
ESMMSR = Path(os.environ.get('ESMMSR_DATA_DIR', os.path.join(SOFTWARE_DIR, 'esm-msr')))
TSUBO = Path(os.environ.get('TSUBOYAMA_CSV', str(ESMMSR / 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')))

CAP = pickle.load(open(Path(REPO_ROOT) / 'data' / 'splits_oct06_capped.pkl', 'rb'))
CAP_N = {k: set(norm_wt_name(x)[1] for x in v) for k, v in CAP.items() if k in ('train', 'val', 'test')}


def write_small(df: pd.DataFrame, name: str):
    for dst in (SMALL_OUT / name, WORK_OUT / name):
        df.to_csv(dst, index=False)
    print(f'wrote {SMALL_OUT / name} rows={len(df)}')


def main():
    # EsmTherm split membership (from esmtherm_split.csv or local EsmTherm clone)
    esplit_path = ESMTHERM / 'datasets/split/wildtype_split.csv'
    if not esplit_path.exists():
        esplit_path = Path(data_file('esmtherm_split.csv'))
    if esplit_path.exists() and TSUBO.exists():
        esplit = pd.read_csv(esplit_path)
        if esplit_path != SMALL_OUT / 'esmtherm_split.csv':
            esplit.to_csv(SMALL_OUT / 'esmtherm_split.csv', index=False)
        esplit['code'] = esplit.WT_name.map(lambda x: norm_wt_name(x)[1])
        membership = esplit.groupby('code')['split'].apply(lambda s: tuple(sorted(set(s))))
        all3 = set(membership[membership.apply(len) == 3].index)
        test_only = set(membership[membership.apply(lambda x: x == ('test',))].index)

        proxy = pd.read_csv(
            TSUBO,
            usecols=['WT_name', 'mut_type', 'aa_seq_full', 'deltaG_t', 'log10_K50_t', 'deltaG_c', 'log10_K50_c', 'deltaG'],
            low_memory=False,
        )
        proxy = proxy.loc[
            ~proxy.mut_type.str.contains('ins') & ~proxy.mut_type.str.contains('del') & ~proxy.mut_type.str.contains(':')
        ].copy()
        proxy = proxy.loc[(proxy.log10_K50_t < 0.5) & (proxy.log10_K50_c < 0.5)]
        proxy = proxy.dropna(subset=['aa_seq_full', 'deltaG_t', 'log10_K50_t', 'deltaG_c', 'log10_K50_c', 'deltaG'])
        proxy_n = proxy.groupby(proxy.WT_name.map(lambda x: norm_wt_name(x)[1])).size().to_dict()

        rows = []
        for code, mem in membership.items():
            N = proxy_n.get(code)
            for split in mem:
                if N is None:
                    n = np.nan
                    basis = 'unknown: K50_dG_Dataset1_Dataset2.csv not cloned; code absent from Dataset2 proxy'
                elif code in all3:
                    n = round(0.1 * N) if split in ('val', 'test') else (N - 2 * round(0.1 * N))
                    basis = 'expected from 80/10/10 per-row random assignment (build_dataset.py) applied to Dataset2 proxy N'
                else:
                    n = N if split == 'test' else 0
                    basis = 'test-only wildtype: all rows forced to test; Dataset2 proxy N'
                rows.append({
                    'code': code,
                    'split': split,
                    'n_rows': n,
                    'n_rows_basis': basis,
                    'dataset2_proxy_total': N,
                    'in_all_three_splits': code in all3,
                    'test_only_wildtype': code in test_only,
                    'in_capped_train': code in CAP_N['train'],
                    'in_capped_val': code in CAP_N['val'],
                    'in_capped_test': code in CAP_N['test'],
                })
        write_small(pd.DataFrame(rows), 'esmtherm_train_proteins.csv')

    else:
        print('EsmTherm split or Tsuboyama CSV not found; keeping the committed esmtherm_train_proteins.csv.')


if __name__ == '__main__':
    main()
