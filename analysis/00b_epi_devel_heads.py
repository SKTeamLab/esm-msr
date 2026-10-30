#!/usr/bin/env python
"""The epistasis hierarchy of the three devel ESM-MSR checkpoints for two heads, per seed: the epistasis head ``epi`` = MT(AB) - WT(AB) and the context control ``ctx`` = ~WT(AB) - WT(AB)
(the WT adapter read on the mutated sequence, asked about the reverse mutation: what the backbone and the single-mutant training already know about epistasis).

The suite is ``esm_msr.epi_metrics`` (the one that scored BENCHMARKS tables 2.0-2.2), fed exactly as ``analysis_scripts/rescore_epistasis.py`` feeds it - the saved latent heads, the
checkpoint's own link and per-library floors, ``cens != 0`` rows dropped - plus ``wt_rev``, the reverse leg of ``outputs/ctx_run`` on the cluster (``esm_msr_ctx_testing.py``: in those files
``mt_lora_pred`` is ~WT).  The ``epi`` columns reproduce ``outputs/epi_new/rescore_{test,val}.csv`` (checked below), so the two heads are on the same footing.

    python 00b_epi_devel_heads.py --ctx DIR      # DIR/<test|val>/s<k>/*.csv = outputs/ctx_run/predictions/splits_oct06_capped-<split>/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s<k>/

Reads ``msr_predictions/devel_dc8_floorlink_w2a4_m16a16/seed<k>/sigma1`` for WT and MT, ``reference/{y_floor,dG_wt}.json`` (the cluster's per-library floors and wild-type stabilities) and the
checkpoints' ``link_head`` under ``<root>/dc_runs/logs``.  Writes ``reference/epi_devel_heads.csv``: one row per (split, seed), every ``epi_*`` metric of both heads.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from esm_msr import epi_metrics as E  # noqa: E402
from esm_msr.link import numpy_link_from_state_dict  # noqa: E402

from msr_analysis.registry import MSR, REFERENCE, ROOT  # noqa: E402

CKPT = {1: 'epoch=04-val_rho_combined_avg=0.840.ckpt', 2: 'epoch=07-val_rho_combined_avg=0.839.ckpt', 3: 'epoch=05-val_rho_combined_avg=0.841.ckpt'}
_MUT = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')
EXPECT = {('test', 1): dict(naive=0.471, beyond=0.494, pair=0.603, line=0.399, cell_rank=0.216)}       # outputs/epi_new/rescore_test.csv, dc8_s1


def parse_mut(mt, code):
    out = []
    for part in str(mt).split(':'):
        m = _MUT.match(part.strip())
        if not m:
            return None
        out.append((code, int(m.group(2)), m.group(3)))
    return tuple(out)


def link_of(seed):
    p = glob.glob(str(ROOT / f'dc_runs/logs/dc8_floorlink_canon_w2a4_m16a16_s{seed}/0/{CKPT[seed]}'))[0]
    sd = torch.load(p, map_location='cpu', weights_only=False)
    return numpy_link_from_state_dict(sd.get('state_dict', sd))


def table(split, seed, ctx_dir, dg_wt, y_floor):
    d = pd.read_csv(MSR / 'devel_dc8_floorlink_w2a4_m16a16' / f'seed{seed}' / 'sigma1' / f'tsuboyama_capped_{split}.csv', low_memory=False)
    c = pd.read_csv(glob.glob(f'{ctx_dir}/{split}/s{seed}/*.csv')[0], usecols=['code', 'mut_type', 'wt_lora_pred', 'mt_lora_pred'], low_memory=False)
    d['code'], d['mut_type'], c['code'], c['mut_type'] = d['code'].astype(str), d['mut_type'].astype(str), c['code'].astype(str), c['mut_type'].astype(str)
    c = c.drop_duplicates(['code', 'mut_type']).rename(columns={'wt_lora_pred': 'wt_ctx_run', 'mt_lora_pred': 'wt_rev_pred'})
    d = d.merge(c, on=['code', 'mut_type'], how='left')
    both = d['wt_ctx_run'].notna()
    print(f'  {split} s{seed}: {int(both.sum())} of {len(d)} rows have a ctx-run row; WT of the two runs differ by at most '
          f'{np.nanmax(np.abs(d.loc[both, "wt_lora_pred"] - d.loc[both, "wt_ctx_run"])):.2e}')
    d = d[d['cens'] == 0].reset_index(drop=True)
    keys = [parse_mut(m, k) for m, k in zip(d['mut_type'], d['code'])]
    st = ['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys]
    items = dict(mut_key=keys, subset_type=np.array(st, dtype=object), ddG=d['ddG_ML'].to_numpy(float), dddG=d['dddG_ML'].to_numpy(float),
                 dG_wt=np.array([float(dg_wt.get(k[0][0], np.nan)) if k else np.nan for k in keys]),
                 wt=d['wt_lora_pred'].to_numpy(float), mt=d['mt_lora_pred'].to_numpy(float), comb=d['combined_pred'].to_numpy(float), wt_rev=d['wt_rev_pred'].to_numpy(float))
    arr = {k: (v if k in ('mut_key', 'subset_type') else np.asarray(v, float)) for k, v in items.items()}
    arr['y_floor'] = np.array([float(y_floor.get(k[0][0], np.nan)) if k else np.nan for k in keys])
    return E.Table(arr, link_of(seed), heads=('epi', 'comb', 'add', 'mt', 'ctx'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctx', required=True, help='directory with <test|val>/s<k>/*.csv, the ctx run\'s predictions')
    a = ap.parse_args()
    dg_wt = json.load(open(REFERENCE / 'dG_wt.json'))
    y_floor = json.load(open(REFERENCE / 'y_floor.json'))
    rows = []
    for split in ('test', 'val'):
        for seed in (1, 2, 3):
            r = E.compute(table(split, seed, a.ctx, dg_wt, y_floor))
            r.update(split=split, seed=seed)
            rows.append(r)
            print(f'  {split} s{seed}: n_doubles {r["epi_n_doubles"]:.0f}  ' + '  '.join(
                f'{h}: naive {r.get(f"epi_naive_rho_{h}", np.nan):.3f} beyond {r.get(f"epi_beyond_rho_{h}", np.nan):.3f} pair {r.get(f"epi_pair_rho_{h}", np.nan):.3f} '
                f'line {r.get(f"epi_line_rho_{h}", np.nan):.3f} cell_rank {r.get(f"epi_cell_rank_{h}", np.nan):.3f}' for h in ('epi', 'ctx')))
            if (split, seed) in EXPECT:
                e = EXPECT[(split, seed)]
                got = dict(naive=r['epi_naive_rho_epi'], beyond=r['epi_beyond_rho_epi'], pair=r['epi_pair_rho_epi'], line=r['epi_line_rho_epi'], cell_rank=r['epi_cell_rank_epi'])
                print('    rescore_test.csv (dc8_s1):', e, '| max |diff| =', round(max(abs(got[k] - e[k]) for k in e), 4))
    pd.DataFrame(rows).to_csv(REFERENCE / 'epi_devel_heads.csv', index=False, float_format='%.5f')


if __name__ == '__main__':
    main()
