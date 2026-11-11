#!/usr/bin/env python
"""Regenerate third-party training-data inventories (upstream SPURS, EsmTherm, upstream ThermoMPNN).

Ported from the original upstream ``leak_audit/data/build_inventories.py`` and updated to:
  1. Resolve repository and software paths via ``lib.py`` environment defaults rather than hardcoded paths.
  2. Normalize mutant-background ``WT_name`` entries (``<PDB>.pdb_<mut>``) cleanly into ``(code, code_wt)``.
  3. Write small protein-level CSVs into ``analysis_scripts/leak_audit/data/`` and large intermediate mutation
     tables (``spurs_train_mutations.csv.gz``) into ``$LEAK_DATA_DIR`` (gitignored).
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, REPO_DATA_DIR, REPO_ROOT, SOFTWARE_DIR, WORK_DATA_DIR, data_file, norm_wt_name

SMALL_OUT = Path(REPO_DATA_DIR)
WORK_OUT = Path(WORK_DATA_DIR)
SMALL_OUT.mkdir(parents=True, exist_ok=True)
WORK_OUT.mkdir(parents=True, exist_ok=True)

SPURS = Path(os.environ.get('SPURS_DIR', os.path.join(SOFTWARE_DIR, 'SPURS_mod')))
THERMO = Path(os.environ.get('THERMOMPNN_DIR', os.path.join(SOFTWARE_DIR, 'ThermoMPNN')))
ESMTHERM = Path(os.environ.get('ESMTHERM_DIR', os.path.join(SOFTWARE_DIR, 'EsmTherm')))
ESMMSR = Path(os.environ.get('ESMMSR_DATA_DIR', os.path.join(SOFTWARE_DIR, 'esm-msr')))
TSUBO = Path(os.environ.get('TSUBOYAMA_CSV', str(ESMMSR / 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')))

CAP = pickle.load(open(Path(REPO_ROOT) / 'data' / 'splits_oct06_capped.pkl', 'rb'))
CAP_N = {k: set(norm_wt_name(x)[1] for x in v) for k, v in CAP.items() if k in ('train', 'val', 'test')}
CAP_ASSIGN = pd.read_csv(Path(REPO_ROOT) / 'data' / 'splits_oct06_capped_library_assignment.csv')


def write_small(df: pd.DataFrame, name: str):
    for dst in (SMALL_OUT / name, WORK_OUT / name):
        df.to_csv(dst, index=False)
    print(f'wrote {SMALL_OUT / name} rows={len(df)}')


def main():
    summary = {}

    # 1. Upstream SPURS single + multi (historical reference; main tables use retrained SPURS)
    spurs_split_path = SPURS / 'data/dataset/megascale/mega_splits.pkl'
    spurs_m8_path = SPURS / 'data/dataset/megascale/mmseq_mut_search_0.25.m8'
    if spurs_split_path.exists() and spurs_m8_path.exists() and TSUBO.exists():
        splits = pickle.load(open(spurs_split_path, 'rb'))
        m8_idx = {int(line.split('\t')[1]) for line in open(spurs_m8_path)}
        base = pd.read_csv(TSUBO, usecols=['ddG_ML', 'mut_type', 'WT_name', 'aa_seq', 'dG_ML'], low_memory=False)
        base = base.loc[base.ddG_ML != '-', :].reset_index(drop=True)
        base_single = base.loc[
            ~base.mut_type.str.contains('ins') & ~base.mut_type.str.contains('del') & ~base.mut_type.str.contains(':'), :
        ].reset_index(drop=True)
        base_multi = base.loc[
            (~base.mut_type.str.contains('ins') & ~base.mut_type.str.contains('del') & base.mut_type.str.contains(':'))
            | (base.mut_type == 'wt'),
            :,
        ].reset_index(drop=True)
        single_f = base_single.loc[~base_single.index.isin(m8_idx), :].reset_index(drop=True)
        multi_f = base_multi.loc[~base_multi.index.isin(m8_idx), :].reset_index(drop=True)

        protein_rows, mut_frames = [], []
        for model, raw_df, filt_df in [('spurs_single', base_single, single_f), ('spurs_multi', base_multi, multi_f)]:
            for split in ['train', 'val', 'test']:
                df = raw_df if split == 'test' else filt_df
                for name in list(splits[split]):
                    lib = df[df.WT_name == name]
                    n_wt = int((lib.mut_type == 'wt').sum())
                    n_mut = int((lib.mut_type != 'wt').sum())
                    retained = (n_wt > 0 and n_mut > 0) if model == 'spurs_multi' else (n_wt > 0)
                    code, code_wt, _ = norm_wt_name(name)
                    protein_rows.append({
                        'model': model,
                        'code': code,
                        'code_wt': code_wt,
                        'split': split,
                        'n_wt_rows': n_wt,
                        'n_mut_rows': n_mut,
                        'n_rows_total': n_wt + n_mut,
                        'retained': bool(retained),
                        'in_capped_train': code_wt in CAP_N['train'],
                        'in_capped_val': code_wt in CAP_N['val'],
                        'in_capped_test': code_wt in CAP_N['test'],
                    })
                    if split in ('train', 'val') and retained:
                        m = lib[lib.mut_type != 'wt'][['mut_type']].copy()
                        m['model'] = model
                        m['code'] = code
                        m['code_wt'] = code_wt
                        m['split'] = split
                        mut_frames.append(m.rename(columns={'mut_type': 'mutation'})[['model', 'code', 'code_wt', 'mutation', 'split']])

        proteins = pd.DataFrame(protein_rows)
        write_small(proteins[proteins.model == 'spurs_single'], 'spurs_single_train_proteins.csv')
        write_small(proteins[proteins.model == 'spurs_multi'], 'spurs_multi_train_proteins.csv')
        mut = pd.concat(mut_frames, ignore_index=True)
        mut.to_csv(WORK_OUT / 'spurs_train_mutations.csv.gz', index=False, compression='gzip')
        print(f'wrote {WORK_OUT / "spurs_train_mutations.csv.gz"} rows={len(mut)}')
    else:
        print('Skipping upstream SPURS rebuild (local SPURS_mod clone not found); keeping committed CSVs.')

    # 2. EsmTherm split membership (from esmtherm_split.csv or local EsmTherm clone)
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

    # 3. Upstream ThermoMPNN reference split (for provenance comparison; main tables use retrained ThermoMPNN)
    if (THERMO / 'dataset_splits/megascale_clusters.tsv').exists():
        mega_clusters = pd.read_csv(THERMO / 'dataset_splits/megascale_clusters.tsv', sep='\t', header=None, names=['rep', 'member'])
        mega_cl = dict(zip(mega_clusters.member.map(lambda x: norm_wt_name(x)[1]), mega_clusters.rep.map(lambda x: norm_wt_name(x)[1])))
        fp_clusters = pd.read_csv(THERMO / 'dataset_splits/fireprot_clusters.tsv', sep='\t', header=None, names=['rep', 'member'])
        fp_cl = dict(zip(fp_clusters.member, fp_clusters.rep))
        th_rows = []
        for split, fname in [('train', 'data_all/training/mega_train.csv'),
                             ('val', 'data_all/training/mega_val.csv'),
                             ('test', 'data_all/testing/mega_test.csv')]:
            d = pd.read_csv(THERMO / fname)
            for code_raw, g in d.groupby('WT_name'):
                c = norm_wt_name(code_raw)[1]
                th_rows.append({
                    'code': c, 'source_dataset': 'megascale', 'split': split,
                    'n_rows': int((g.mut_type != 'wt').sum()), 'n_wt_rows': int((g.mut_type == 'wt').sum()),
                    'cluster': mega_cl.get(c, ''), 'in_capped_val': c in CAP_N['val'], 'in_capped_test': c in CAP_N['test'],
                })
        for split, fname in [('train', 'data_all/training/fireprot_train.csv'),
                             ('val', 'data_all/training/fireprot_val.csv'),
                             ('test', 'data_all/testing/fireprot_test.csv')]:
            d = pd.read_csv(THERMO / fname)
            for code_raw, g in d.groupby('pdb_id_corrected'):
                c = str(code_raw)
                th_rows.append({
                    'code': c, 'source_dataset': 'fireprot', 'split': split,
                    'n_rows': int(g.ddG.notna().sum()), 'n_wt_rows': 0,
                    'cluster': fp_cl.get(c, ''), 'in_capped_val': c in CAP_N['val'], 'in_capped_test': c in CAP_N['test'],
                })
        write_small(pd.DataFrame(th_rows), 'thermompnn_upstream_train_proteins.csv')

    summary['capped_counts'] = {k: len(v) for k, v in CAP_N.items()}
    json.dump(summary, open(WORK_OUT / 'inventory_summary.json', 'w'), indent=2)
    print('Upstream inventories ready.')


if __name__ == '__main__':
    main()
