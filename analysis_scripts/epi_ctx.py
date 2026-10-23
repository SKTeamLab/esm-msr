"""Epistasis hierarchy for the WT adapter read in context, ctx = (WT + ~WT)/2, of the released and the three devel
checkpoints, on valfull and test.

The predictions come from inference_scripts/esm_msr_ctx_testing.py, which runs esm_msr_testing.py with the MT pass
replaced by the WT adapter's reverse leg, so in each file
    wt_lora_pred = WT,   mt_lora_pred = ~WT (the reverse leg alone),   combined_pred = ctx.
They are scored exactly as epi_baselines.main() scores an ESM-MSR checkpoint (same truth rows, same observed-scale link
recovered from the checkpoint, same per-library floors), so the rows are comparable with MSR/comb and MSR_dc8_s*/comb:
    <label>_ctx/add   the WT head          (must equal <label>/add of the normal run: the validity check)
    <label>_ctx/mt    ~WT alone
    <label>_ctx/comb  ctx

Run on the cluster under msr_venv:  python analysis_scripts/epi_ctx.py
    -> outputs/baselines_capped/epi_ctx_test.csv, epi_ctx_valfull_allbb.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = '/home/sreeves/esm-msr-devel'
sys.path.insert(0, os.path.join(REPO, 'baselines'))
import epi_baselines as E   # noqa: E402

CTX_PRED = os.path.join(REPO, 'outputs', 'ctx_run', 'predictions')
DG = os.path.join(REPO, 'outputs/baselines_capped/data/dG_wt.json')
Y_FLOOR = os.path.join(REPO, 'outputs/baselines_capped/data/y_floor.json')


def ctx_frame(truth, label):
    """epi_baselines.msr_frame, reading the ctx run's predictions instead of the normal ones."""
    p = os.path.join(CTX_PRED, truth.attrs['where'], f'{E.MSR_CKPTS[label]}_predictions.csv')
    assert os.path.exists(p), f'no ctx predictions for {label}: {p}'
    d = pd.read_csv(p, low_memory=False)
    cols = ['wt_lora_pred', 'mt_lora_pred', 'combined_pred']
    d = d[['code', 'mut_type'] + cols].copy()
    d['code'], d['mut_type'] = d['code'].astype(str), d['mut_type'].astype(str)
    key = truth[['code', 'mut_type']].copy()
    key['code'], key['mut_type'] = key['code'].astype(str), key['mut_type'].astype(str)
    src = key.merge(d.drop_duplicates(['code', 'mut_type']), on=['code', 'mut_type'], how='left')
    n_miss = int(src[cols].isna().all(axis=1).sum())
    print(f'  [{label} ctx] joined {len(src) - n_miss} of {len(truth)} truth rows' + (f' -- MISSING {n_miss}' if n_miss else ''))
    assert n_miss == 0, 'ctx predictions do not cover the truth rows'
    for c in ('mut_type', 'code', 'ddG_ML', 'dddG_ML'):
        src[c] = truth[c].to_numpy()
    f = E.frame(src['mut_type'].to_numpy(), src['code'].astype(str).to_numpy(), src['ddG_ML'].to_numpy(float),
                src['dddG_ML'].to_numpy(float), src['wt_lora_pred'].to_numpy(float), src['mt_lora_pred'].to_numpy(float))
    f['combined_pred'] = src['combined_pred'].to_numpy(float)
    return f


def main():
    y_floor = json.load(open(Y_FLOOR))
    for split, backbones, out in [('test', 'wt_only', 'epi_ctx_test.csv'), ('valfull', 'all', 'epi_ctx_valfull_allbb.csv')]:
        truth, dG_wt = E.truth_table(DG, backbones, split)
        print(f'[{split}] truth rows {len(truth)}, doubles {int(truth.mut_type.astype(str).str.contains(":").sum())}')
        rows = {}
        for lab in E.MSR_CKPTS:
            name = f'{"MSR" if lab == E.MSR_TRUTH else "MSR_" + lab}_ctx'
            link = E.msr_link(lab)
            for h, v in E.score(ctx_frame(truth, lab), dG_wt, ['add', 'mt', 'comb'], name, link, y_floor).items():
                rows[f'{name}/{h}'] = v
        df = pd.DataFrame(rows).T
        df.index.name = 'run/head'
        path = os.path.join(REPO, 'outputs/baselines_capped', out)
        df.to_csv(path, na_rep='', float_format='%.6f')
        cols = [c for c in ['epi_naive_rho', 'epi_beyond_rho', 'epi_pair_rho', 'epi_line_rho', 'epi_cell_mag', 'epi_cell_rank',
                            'epi_cell_flipacc', 'epi_global_err'] if c in df.columns]
        print(df[cols].to_string(float_format=lambda x: f'{x:.3f}'), f'\n-> {path}\n')


if __name__ == '__main__':
    main()
