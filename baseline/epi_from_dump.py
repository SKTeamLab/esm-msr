"""
Rescores validation dumps of this branch (training_logs/<run>/0/val_dump_<tag>.npz) with the current metrics: the symmetric epistasis
metrics of esm_msr.epi_metrics (every metric for comb, mt, ctx; naive / global also for add) and the ddG guards (library-equal means of
rho_combined, rmse_combined, rho_wt_valid). Works on dumps written before the link transplant too: a dump without dG_wt gets it from the raw
Tsuboyama file, one without link_summary is scored without a link (its predictions are taken as observed, as that model was trained).

  PYTHONPATH=src python baseline/epi_from_dump.py DUMP [DUMP ...] [--csv out.csv] [--raw RAW_CSV]

One row per dump. rmse_combined is on the observed scale when the dump has a link (benchmarks, which have no dG_wt, stay latent), as in training.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from esm_msr import epi_metrics                                   # noqa: E402
from esm_msr.link import numpy_link                               # noqa: E402

RAW = '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
BACKGROUND = re.compile(r'_[A-Z][0-9]+[A-Z]$')


def wt_dG(raw):
    df = pd.read_csv(raw, usecols=['WT_name', 'mut_type', 'dG_ML'], low_memory=False)
    wt = df.loc[df['mut_type'] == 'wt'].copy()
    wt['dG_ML'] = pd.to_numeric(wt['dG_ML'], errors='coerce')
    wt['code'] = wt['WT_name'].str.replace('.pdb_', '_', regex=False).str.replace('.pdb', '', regex=False).str.replace('|', '_', regex=False)
    return wt.dropna(subset=['dG_ML']).groupby('code')['dG_ML'].mean().to_dict()


def _rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def score(path, dgw_map):
    d = np.load(path, allow_pickle=False)
    loader = d['loader'].astype(str)
    n_mut = d['n_mut']
    gt = d['ground_truth'].astype(float)
    if 'dG_wt' in d.files:
        dG_wt = d['dG_wt'].astype(float)
    else:
        m = dgw_map()
        dG_wt = np.array([m.get(l, np.nan) for l in loader])
    st = d['subset_type'].astype(str) if 'subset_type' in d.files else np.where(n_mut == 1, 'single', 'double')
    link = None
    if 'link_summary' in d.files:
        ls = json.loads(str(d['link_summary']))
        link = numpy_link(ls['lo'], ls['hi'], ls['tau_lo'], ls['tau_hi'])
    keys = [tuple(tuple(m) for m in json.loads(k)) for k in d['mut_key']]
    # --link_floor_file: the per-library floor offset the run used (absent in dumps written before it, where the floor was shared)
    foff = d['dG_floor_offset'].astype(float) if 'dG_floor_offset' in d.files else np.zeros(len(keys))
    items = {'mut_key': keys, 'subset_type': list(st), 'cens': None, 'ddG': gt, 'dddG': d['dddG'].astype(float), 'dG_wt': dG_wt,
             'wt': d['wt'].astype(float), 'mt': d['mt'].astype(float), 'comb': d['comb'].astype(float), 'wt_rev': d['wt_rev'].astype(float),
             'dG_floor_offset': foff}
    row = epi_metrics.compute(epi_metrics.Table(items, link))
    comb = items['comb']
    obs = comb if link is None else np.where(np.isfinite(dG_wt), link(np.nan_to_num(dG_wt) + comb, foff) - np.nan_to_num(dG_wt), comb)
    g = {'rho_combined': [], 'rmse_combined': [], 'rho_wt_valid': []}
    for name in dict.fromkeys(loader):
        s = loader == name
        ok = s & np.isfinite(gt)
        g['rho_combined'].append(_rho(comb[ok], gt[ok]))
        g['rmse_combined'].append(float(np.sqrt(np.mean((obs[ok] - gt[ok]) ** 2))) if ok.any() else np.nan)
        sg = ok & (st == 'single')
        g['rho_wt_valid'].append(np.nan if BACKGROUND.search(name) else _rho(items['wt'][sg], gt[sg]))
    for k, v in g.items():
        row[f'{k}_avg'] = float(np.nanmean(v))
    if link is not None:
        row.update({f'link_{k}': v for k, v in ls.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--csv')
    ap.add_argument('--raw', default=RAW)
    a = ap.parse_args()
    cache = {}

    def dgw_map():                                                  # read the raw file once, and only if some dump needs it
        if 'm' not in cache:
            cache['m'] = wt_dG(a.raw)
        return cache['m']
    rows = []
    for p in a.dumps:
        r = score(p, dgw_map)
        r['dump'] = p
        rows.append(r)
    df = pd.DataFrame(rows).set_index('dump')
    if a.csv:
        df.to_csv(a.csv)
    with pd.option_context('display.width', 250, 'display.max_columns', 200, 'display.max_rows', 200):
        print(df.T.round(4))


if __name__ == '__main__':
    main()
