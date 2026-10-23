"""Epistasis hierarchy for Rosetta cartesian_ddg on the capped Tsuboyama split.

Rosetta scores a double directly (the cartesian_ddg of the two-residue mutant), so it has the two things the epistasis
suite needs from any model: a direct double prediction and an additive baseline made of its own singles. This feeds
those to baselines/epi_baselines.py's OWN assemble() and score(), i.e. the code that scored MutateEverything,
ThermoMPNN and ESM-MSR, so the four are evaluated identically on identical rows. Nothing is reimplemented here.

Both heads are scored (`add`: the sum of Rosetta's singles; `mt`: its direct double). `combined` is the direct
prediction for a model with one head, as for the other baselines.

Run on the cluster under msr_venv; CPU, seconds.

  python analysis_scripts/epi_rosetta.py
      -> outputs/baselines_capped/epi_rosetta_test.csv      (test, destabilized backbones dropped: epi_test.csv's rows)
      -> outputs/baselines_capped/epi_rosetta_valfull_allbb.csv   (the full validation population)
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = '/home/sreeves/esm-msr-devel'
SB = '/home/sreeves/stability-baselines'
sys.path.insert(0, os.path.join(REPO, 'baselines'))
import epi_baselines as E   # noqa: E402  (runs the AST loader for esm_msr_testing's metric functions on import)

DG = os.path.join(REPO, 'outputs/baselines_capped/data/dG_wt.json')


def rosetta_frame(truth, half):
    """Rosetta's predictions keyed the way epi_baselines keys a model: (code, mut_type)."""
    task = pd.read_csv(f'{SB}/tasks/splits_oct06_capped-{half}.csv.gz', usecols=['row_id', 'code', 'mut_type'],
                       keep_default_na=False, na_values=[''], low_memory=False)
    pred = pd.read_csv(f'{SB}/results/rosetta/predictions/splits_oct06_capped-{half}.csv', usecols=['row_id', 'pred'])
    j = task.merge(pred, on='row_id', how='left')
    assert len(j) == len(task) and j.pred.notna().all(), 'row_id misaligned or NaN predictions'
    j['code'], j['mut_type'] = j['code'].astype(str), j['mut_type'].astype(str)
    j = j.drop_duplicates(['code', 'mut_type'])
    direct = dict(zip(zip(j.code, j.mut_type), j.pred))
    single = {k: v for k, v in direct.items() if ':' not in k[1]}
    return E.assemble(truth, direct, single, 'Rosetta')


def main():
    for split, half, backbones, out in [('test', 'test', 'wt_only', 'epi_rosetta_test.csv'),
                                        ('valfull', 'val', 'all', 'epi_rosetta_valfull_allbb.csv')]:
        truth, dG_wt = E.truth_table(DG, backbones, split)
        print(f'[{split}] truth rows {len(truth)}, doubles {int(truth.mut_type.astype(str).str.contains(":").sum())}')
        rows = {}
        for h, v in E.score(rosetta_frame(truth, half), dG_wt, ['add', 'mt'], 'Rosetta').items():
            rows[f'Rosetta/{h}'] = v
        df = pd.DataFrame(rows).T
        df.index.name = 'run/head'
        path = os.path.join(REPO, 'outputs/baselines_capped', out)
        df.to_csv(path, na_rep='', float_format='%.6f')
        cols = [c for c in ['epi_n_doubles', 'epi_naive_rho', 'epi_beyond_rho', 'epi_pair_rho', 'epi_line_rho', 'epi_cell_mag',
                            'epi_cell_rank', 'epi_cell_flipacc', 'epi_global_err', 'spearman_all', 'spearman_doubles']
                if c in df.columns]
        print(df[cols].to_string(float_format=lambda x: f'{x:.3f}'), f'\n-> {path}\n')


if __name__ == '__main__':
    main()
