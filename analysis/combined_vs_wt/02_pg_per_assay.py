#!/usr/bin/env python
"""ProteinGym: per-assay rho of the WT head, the MT head and the combined prediction, on all variants, the single mutants and the multi-mutants of each assay.

The prediction files of the testing queue (``final_pgym_comb/<family>/s<seed>/<assay>.csv``) hold ``wt_lora_pred``, ``mt_lora_pred`` and ``combined_pred`` for every variant of the assay, so the
three heads are scored on the same rows.  FINAL_p0 has seeds 1-3, FINAL_p1 seed 1 (its seed 2 is incomplete); FINAL_p0_pw0 has no combined ProteinGym run.  The figures use seed 1 for both families
(the ESM-MSR convention); ``rho_*_p0`` columns of ``seed_mean`` average the three p0 seeds.  The 23 stability assays whose variants are in the training data (``pgsigma.train_overlap``) are flagged
(``train_overlap``) and left out by every figure.

For each assay: n variants, n multi-mutants, and per family (p0, p1) and head the Spearman rho on ``all``, ``single`` and ``multi`` rows (multi only when the assay has >= 20 multi-mutants and
>= 20 single mutants for the single/multi split), plus the Spearman correlation between the WT and the MT head over all rows (``r_wt_mt``).  Writes ``results/pg_per_assay.csv``.

    python analysis/combined_vs_wt/02_pg_per_assay.py
"""
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cvw as C
import pgsigma as P
from msr_analysis.final_runs import FINAL_PGYM_COMB

RUNS = {('FINAL_p0', 1): 'p0_s1', ('FINAL_p0', 2): 'p0_s2', ('FINAL_p0', 3): 'p0_s3', ('FINAL_p1', 1): 'p1_s1'}
COLS = {'wt': 'wt_lora_pred', 'mt': 'mt_lora_pred', 'comb': 'combined_pred'}
MIN_SPLIT = 20


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() >= 10 else np.nan


def one(assay):
    out = dict(assay=assay)
    for (fam, seed), tag in RUNS.items():
        f = FINAL_PGYM_COMB / fam / f's{seed}' / f'{assay}.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=['mutant', 'DMS_score'] + list(COLS.values()))
        y = d['DMS_score'].to_numpy(float)
        multi = d['mutant'].str.contains(':').to_numpy()
        out['n'] = len(d)
        out['n_multi'] = int(multi.sum())
        split = multi.sum() >= MIN_SPLIT and (~multi).sum() >= MIN_SPLIT
        for h, c in COLS.items():
            x = d[c].to_numpy(float)
            out[f'{tag}_all_{h}'] = rho(x, y)
            if split:
                out[f'{tag}_single_{h}'] = rho(x[~multi], y[~multi])
                out[f'{tag}_multi_{h}'] = rho(x[multi], y[multi])
        out[f'{tag}_r_wt_mt'] = rho(d[COLS['wt']].to_numpy(float), d[COLS['mt']].to_numpy(float))
    return out


def main():
    leak = pd.read_csv(P.OUT / 'leak_per_assay.csv').set_index('assay')
    assays = list(leak.index)
    with mp.Pool(6) as pool:
        rows = pool.map(one, assays, chunksize=4)
    t = pd.DataFrame(rows).set_index('assay')
    t.insert(0, 'category', leak.category)
    t.insert(1, 'train_overlap', [a in P.train_overlap() for a in t.index])
    t['leaked'] = leak.leaked
    t.to_csv(C.OUT / 'pg_per_assay.csv', float_format='%.4f')
    u = t[~t.train_overlap]
    print(len(t), 'assays;', len(u), 'without training overlap;', int((u.n_multi >= MIN_SPLIT).sum()), 'with >= 20 multi-mutants')
    for fam in ('p0_s1', 'p1_s1'):
        d = u[f'{fam}_all_comb'] - u[f'{fam}_all_wt']
        print(fam, d.groupby(u.category).agg(['mean', 'count']).round(4).to_string())


if __name__ == '__main__':
    sys.exit(main())
