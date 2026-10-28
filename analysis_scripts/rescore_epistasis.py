"""Rescore saved ESM-MSR prediction CSVs with the 2026-10-24 authoritative epistasis metrics.

Nothing is re-run: the authoritative epistasis head is ~MT(AB) - WT(AB), which is exactly
`mt_lora_pred - wt_lora_pred` in the saved CSVs, so every checkpoint can be re-scored from disk.

`epi_metrics.Table` is fed the absolute MT latent as its `epi` head (the same route as
`scripts/epi_from_dump.py`) and applies each checkpoint's own link, so the numbers are on the observed scale.
Rows with `cens != 0` are dropped first: their ddG_ML is the assay's bound, not a measurement.

Usage:
    python analysis_scripts/rescore_epistasis.py --split test --split val [LABEL ...]

Writes outputs/epi_new/rescore_<split>.csv next to the printed table.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

from esm_msr import epi_metrics as E
from esm_msr.link import numpy_link_from_state_dict

REPO = os.environ.get('ESM_MSR_REPO', '/home/sreeves/esm-msr-devel')
PRED = os.path.join(REPO, 'analysis_notebooks', 'predictions')
MODELS = os.path.join(REPO, 'LoRA_models')
DATA = os.path.join(REPO, 'outputs', 'baselines_capped', 'data')
OUT = os.path.join(REPO, 'outputs', 'epi_new')

# label -> checkpoint path relative to LoRA_models. The prediction tag is resolved from disk, so the epsilon ->
# sigma rename (commit 7a5b283) and the mask-strategy tag both work without being spelled out here.
LABELS = {
    'released': 'esm-msr-baseline-w2a4-m16a16-capped/epoch=07-val_rho_combined_avg=0.839.ckpt',
    'bl_s1': 'release-baseline/s1/epoch=04-val_rho_combined_avg=0.855.ckpt',
    'bl_s2': 'release-baseline/s2/epoch=05-val_rho_combined_avg=0.857.ckpt',
    'bl_s3': 'release-baseline/s3/epoch=04-val_rho_combined_avg=0.858.ckpt',
    'dc8_s1': 'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s1/epoch=04-val_rho_combined_avg=0.840.ckpt',
    'dc8_s2': 'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s2/epoch=07-val_rho_combined_avg=0.839.ckpt',
    'dc8_s3': 'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s3/epoch=05-val_rho_combined_avg=0.841.ckpt',
    'trim_s1': 'trim_mask_mt_ind_fast_s1/last.ckpt',
    'trim_s2': 'trim_mask_mt_ind_fast_s2/last.ckpt',
    'trim_s3': 'trim_mask_mt_ind_fast_s3/last.ckpt',
}

_MUT = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')


def parse_mut(mt, code):
    """'A11C' / 'A11C:B22D' -> ((code, pos, mut), ...) -- the mut_key the Table matches singles on."""
    out = []
    for part in str(mt).split(':'):
        m = _MUT.match(part.strip())
        if not m:
            return None
        out.append((code, int(m.group(2)), m.group(3)))
    return tuple(out)


def link_of(ckpt_rel):
    sd = torch.load(os.path.join(MODELS, ckpt_rel), map_location='cpu', weights_only=False)
    return numpy_link_from_state_dict(sd.get('state_dict', sd))


def prediction_path(ckpt_rel, split):
    """The stored prediction CSV for this checkpoint, whatever naming generation wrote it."""
    d = os.path.join(PRED, 'splits_oct06_capped-%s' % split, os.path.dirname(ckpt_rel))
    stem = os.path.basename(ckpt_rel)
    # Token order matters and must not be left to sort(): '_sigma1.0_' sorts before '_sigma1_' ('0x2e' < '0x5f'),
    # and the two spellings hold different data (139 of 156 pairs differ), so the exact '1' spelling is tried first.
    # Generation preference: '_sigma1.0_' is the 2026-10-23 per-mutation-bias regeneration of '_sigma1_'
    # (2026-10-08, per-item bias), so it is tried first. _pre_pmb/ holds the superseded copies.
    for pre in ('sigma', 'epsilon'):
        for tok in ('1.0', '1'):
            hits = glob.glob(os.path.join(d, f'{stem}_{pre}{tok}_*_predictions.csv'))
            if hits:
                unmasked = [f for f in hits if '_unmasked_' in os.path.basename(f)]
                return sorted(unmasked or hits)[0]
    return None


def build(csv_path, dg_wt, y_floor, link):
    d = pd.read_csv(csv_path, low_memory=False)
    if 'cens' in d.columns:
        d = d[d['cens'] == 0].reset_index(drop=True)
    keys, st = [], []
    for mt, cd in zip(d['mut_type'], d['code'].astype(str)):
        k = parse_mut(mt, cd)
        keys.append(k)
        st.append('single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other'))
    items = {
        'mut_key': keys,
        'subset_type': np.array(st, dtype=object),
        'ddG': d['ddG_ML'].to_numpy(float),
        'dddG': d['dddG_ML'].to_numpy(float),
        'dG_wt': np.array([float(dg_wt.get(k[0][0], np.nan)) if k else np.nan for k in keys]),
        'wt': d['wt_lora_pred'].to_numpy(float),
        'mt': d['mt_lora_pred'].to_numpy(float),      # Table feeds its `epi` head from this
        'comb': d['combined_pred'].to_numpy(float),
    }
    if 'wt_rev_pred' in d.columns:
        items['wt_rev'] = d['wt_rev_pred'].to_numpy(float)
    arr = {k: (v if k in ('mut_key', 'subset_type') else np.asarray(v, float)) for k, v in items.items()}
    arr['y_floor'] = np.array([float(y_floor.get(k[0][0], np.nan)) if k else np.nan for k in keys])
    return E.Table(arr, link, heads=('epi', 'comb', 'add', 'mt', 'ctx'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', action='append', choices=['test', 'val'], default=None)
    ap.add_argument('labels', nargs='*')
    a = ap.parse_args()
    splits = a.split or ['test', 'val']
    labels = a.labels or list(LABELS)
    dg_wt = json.load(open(os.path.join(DATA, 'dG_wt.json')))
    y_floor = json.load(open(os.path.join(DATA, 'y_floor.json')))
    os.makedirs(OUT, exist_ok=True)
    for split in splits:
        rows = []
        print(f'=== {split}')
        for lab in labels:
            ck = LABELS[lab]
            f = prediction_path(ck, split)
            if f is None:
                print('MISSING', lab, ck)
                continue
            r = E.compute(build(f, dg_wt, y_floor, link_of(ck)))
            r.update(label=lab, split=split)
            rows.append(r)
            print('%-9s n_dbl=%5d n_pairs=%3d  naive=%6.3f beyond=%6.3f lib=%6.3f pairavg=%6.3f pair=%6.3f '
                  'line=%6.3f cmag=%6.3f crank=%6.3f flip=%6.3f gerr=%6.3f c3=%6.3f' % (
                      lab, r['epi_n_doubles'], r['epi_n_pairs'],
                      r.get('epi_naive_rho_epi', np.nan), r.get('epi_beyond_rho_epi', np.nan),
                      r.get('epi_beyond_lib_epi', np.nan), r.get('epi_beyond_rho_pairavg_epi', np.nan),
                      r.get('epi_pair_rho_epi', np.nan), r.get('epi_line_rho_epi', np.nan),
                      r.get('epi_cell_mag_epi', np.nan), r.get('epi_cell_rank_epi', np.nan),
                      r.get('epi_cell_flipacc_epi', np.nan), r.get('epi_global_err_epi', np.nan),
                      r.get('epi_class3_acc_epi', np.nan)))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rescore_%s.csv' % split), index=False)


if __name__ == '__main__':
    sys.exit(main())
