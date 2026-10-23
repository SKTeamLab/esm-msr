"""Epistasis hierarchy for every third-party model that makes a real double-mutant prediction on the capped Tsuboyama
split (valfull and test).

"Real" means the model scores the double itself (the mutant sequence, structure or ensemble), so it has both things the
epistasis suite needs: a direct double prediction and an additive baseline made of its own singles. Rows are fed to
baselines/epi_baselines.py's OWN assemble() and score(), the code that scored MutateEverything, ThermoMPNN, Rosetta and
ESM-MSR, so every model is evaluated identically on identical rows (see epi_rosetta.py, of which this is the general
form). Nothing is reimplemented here.

Not included: ProSST (its pred equals the sum of its singles on every double, so it has no interaction to score) and
DDMut / UniMutStab (no predictions on this split).

Run on the cluster under msr_venv; CPU, seconds per model.

  python analysis_scripts/epi_harness.py
      -> outputs/baselines_capped/epi_harness_test.csv           (test, destabilized backbones dropped: epi_test.csv's rows)
      -> outputs/baselines_capped/epi_harness_valfull_allbb.csv  (the full validation population)
  rows are '<model>/mt' (direct double head) and '<model>/add' (sum of the model's own singles; 0 by construction on
  every residual and rank metric, the suite's validity check).
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

# (row label, harness model dir, prediction tag). SPURS is its multi-mutant model: the hybrid run ('' tag) scores a
# double with that model too, but its additive baseline would then be the single model's.
MODELS = [('ESMtherm', 'esmtherm', ''),
          ('IFUM (WT backbone)', 'ifum', ''),
          ('IFUM (native, ESMFold)', 'ifum', '_esmfold'),
          ('GeoDDG-3D (FASPR)', 'geoddg', '_faspr'),
          ('ESM3dG', 'esm3dg', ''),
          ('SPURS multi', 'spurs', '_multi'),
          ('ProteinMPNN', 'proteinmpnn', ''),
          ('DDGemb', 'ddgemb', '')]


def model_frame(truth, half, model, tag):
    """The model's predictions keyed the way epi_baselines keys one: (code, mut_type)."""
    task = pd.read_csv(f'{SB}/tasks/splits_oct06_capped-{half}.csv.gz', usecols=['row_id', 'code', 'mut_type'],
                       keep_default_na=False, na_values=[''], low_memory=False)
    pred = pd.read_csv(f'{SB}/results/{model}/predictions{tag}/splits_oct06_capped-{half}.csv', usecols=['row_id', 'pred'])
    j = task.merge(pred, on='row_id', how='left')
    assert len(j) == len(task), 'row_id misaligned'
    j['code'], j['mut_type'] = j['code'].astype(str), j['mut_type'].astype(str)
    j = j.drop_duplicates(['code', 'mut_type'])
    direct = {k: v for k, v in zip(zip(j.code, j.mut_type), j.pred)}
    single = {k: v for k, v in direct.items() if ':' not in k[1]}
    return E.assemble(truth, direct, single, model + tag)


def main():
    for split, half, backbones, out in [('test', 'test', 'wt_only', 'epi_harness_test.csv'),
                                        ('valfull', 'val', 'all', 'epi_harness_valfull_allbb.csv')]:
        truth, dG_wt = E.truth_table(DG, backbones, split)
        print(f'[{split}] truth rows {len(truth)}, doubles {int(truth.mut_type.astype(str).str.contains(":").sum())}')
        rows = {}
        for label, model, tag in MODELS:
            for h, v in E.score(model_frame(truth, half, model, tag), dG_wt, ['add', 'mt'], label).items():
                rows[f'{label}/{h}'] = v
        df = pd.DataFrame(rows).T
        df.index.name = 'run/head'
        path = os.path.join(REPO, 'outputs/baselines_capped', out)
        df.to_csv(path, na_rep='', float_format='%.6f')
        cols = [c for c in ['epi_n_doubles', 'epi_naive_rho', 'epi_beyond_rho', 'epi_pair_rho', 'epi_line_rho', 'epi_cell_mag',
                            'epi_cell_rank', 'epi_cell_flipacc', 'epi_global_err'] if c in df.columns]
        print(df[cols].to_string(float_format=lambda x: f'{x:.3f}'), f'\n-> {path}\n')


if __name__ == '__main__':
    main()
