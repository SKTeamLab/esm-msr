#!/usr/bin/env python
"""Where could the Tobit constants (sigma_y, mu_f, sigma_f) come from? Estimates them from the raw Tsuboyama table, by split, with library-level bootstrap.

Model (losses.tobit_regression_loss): a reading is Y = max(X, F), X ~ N(z, sigma_y^2) the variant's own stability measured with noise, F ~ N(mu_f, sigma_f^2) the
unfolded floor. The curated dG_ML column reports numbers on [-1, 5] and the tags '<-1' / '>5' outside it, so a dead variant's reading is a CENSORED draw of F.

* sigma_y: replicate SD of the label itself (variants measured more than once: same library, same mutation, different DNA), pooled over bins of dG_ML, and the
  paired-protease SD, SD(dG_trypsin - dG_chymotrypsin)/sqrt(2), on variants the combined fit puts in the alive range.
* mu_f, sigma_f: censored-normal MLE (readings below -1 enter as P(F < -1), numeric readings as densities) on doubles that are dead by evidence independent of their own
  reading. Dead sets: (A) at least one single of the double is tagged '<-1'; (B) both singles are tagged or read <= 0; (C) both singles tagged; (D) additive expectation
  dG_wt + ddG_A + ddG_B <= -2 / -3 (needs both singles numeric). The NAIVE numeric-only mean / SD (what the Nov 3 check reported: -0.14 / 0.5) is also printed: it is the
  mean / SD of the upper part of F that survives the -1 cut, not of F.

Splits are the capped split (data/splits_oct06_capped.pkl): the constants should come from TRAIN libraries only (val / test shown for consistency).

    python analysis/tobit_constants/estimate_constants.py [--raw CSV] [--B 500]
Outputs in this directory: sigma_y.csv, floor_mle.csv, floor_by_library.csv, README.md (written by the script).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = Path.home() / 'software' / 'esm-msr' / 'data' / 'tsuboyama' / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
COLS = ['WT_name', 'mut_type', 'deltaG_t', 'deltaG_c', 'dG_ML', 'ddG_ML']
SUB = r'[A-Z]\d+[A-Z](:[A-Z]\d+[A-Z])?'


def load(raw):
    d = pd.read_csv(raw, usecols=COLS, low_memory=False, dtype={'dG_ML': object, 'ddG_ML': object})
    sp = pickle.load(open(REPO / 'data' / 'splits_oct06_capped.pkl', 'rb'))
    m = {n: s for s in ('train', 'val', 'test') for n in sp[s]}
    d['split'] = d.WT_name.map(m)
    d = d[d.split.notna() & (d.mut_type.str.fullmatch(SUB) | (d.mut_type == 'wt'))].copy()
    d['dG'] = pd.to_numeric(d.dG_ML, errors='coerce')
    d['ddG'] = pd.to_numeric(d.ddG_ML, errors='coerce')
    d['tag'] = np.select([d.dG_ML == '<-1', d.dG_ML == '>5', d.dG_ML == '-', d.dG.notna()], ['lo', 'hi', 'none', 'num'], 'other')
    d['n_mut'] = np.where(d.mut_type == 'wt', 0, d.mut_type.str.count(':') + 1)
    return d


def cens_mle(y, n_lo, lo=-1.0):
    """N(mu, sd): numeric readings y observed above `lo`, n_lo readings censored below it."""
    f = lambda p: -(norm.logpdf(y, p[0], np.exp(p[1])).sum() + n_lo * norm.logcdf((lo - p[0]) / np.exp(p[1])))
    r = minimize(f, [y.mean() - 0.5, np.log(y.std() + 0.2)], method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-8, maxiter=2000))
    return float(r.x[0]), float(np.exp(r.x[1]))


def doubles_table(d):
    sing = d[d.n_mut == 1].copy()
    sing['key'] = sing.WT_name + '|' + sing.mut_type
    st = sing.groupby('key').agg(tag=('tag', lambda s: 'lo' if (s == 'lo').any() else ('num' if (s == 'num').any() else 'other')), dG=('dG', 'mean'), ddG=('ddG', 'mean'))
    dbl = d[d.n_mut == 2].copy()
    p = dbl.mut_type.str.split(':', expand=True)
    for c, i in (('a', 0), ('b', 1)):
        k = dbl.WT_name + '|' + p[i]
        dbl['t' + c], dbl['g' + c], dbl['dd' + c] = k.map(st.tag), k.map(st.dG), k.map(st.ddG)
    wt = d[d.mut_type == 'wt'].groupby('WT_name').dG.mean()
    dbl['dG_add'] = dbl.WT_name.map(wt) + dbl.dda + dbl.ddb
    return dbl


DEAD = {
    'A: >=1 single tagged <-1': lambda x: (x.ta == 'lo') | (x.tb == 'lo'),
    'B: both singles tagged or <= 0': lambda x: ((x.ta == 'lo') | (x.ga <= 0)) & ((x.tb == 'lo') | (x.gb <= 0)),
    'C: both singles tagged <-1': lambda x: (x.ta == 'lo') & (x.tb == 'lo'),
    'D2: additive expectation <= -2': lambda x: x.dG_add <= -2,
    'D3: additive expectation <= -3': lambda x: x.dG_add <= -3,
}


def fit(x):
    y = x.dG[x.tag == 'num'].to_numpy(float)
    n_lo = int((x.tag == 'lo').sum())
    if len(y) < 40:
        return None
    mu, sd = cens_mle(y, n_lo)
    return dict(n=len(x), frac_tagged=n_lo / (len(y) + n_lo), naive_mean=y.mean(), naive_sd=y.std(), mu_f=mu, sigma_f=sd, P_below_minus1=float(norm.cdf((-1 - mu) / sd)))


def sigma_y_tables(d):
    rows = []
    g = d[d.tag == 'num'].copy()
    g['n'] = g.groupby(['WT_name', 'mut_type']).dG.transform('size')
    rep = g[g.n >= 2].copy()
    rep['m'] = rep.groupby(['WT_name', 'mut_type']).dG.transform('mean')

    def pooled(x):
        k = x.groupby(['WT_name', 'mut_type']).dG.agg(['var', 'size'])
        return float(np.sqrt(((k['size'] - 1) * k['var']).sum() / (k['size'] - 1).sum())), int((k['size'] - 1).sum())
    for lo, hi in [(-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)]:
        s, dof = pooled(rep[(rep.m > lo) & (rep.m <= hi)])
        rows.append(dict(estimator='replicate SD of dG_ML (same variant, different DNA)', bin=f'({lo},{hi}]', split='all', sigma=s, dof=dof))
    for sp in ('train', 'val', 'test'):
        s, dof = pooled(rep[(rep.split == sp) & (rep.m > 1) & (rep.m <= 4)])
        rows.append(dict(estimator='replicate SD of dG_ML (same variant, different DNA)', bin='(1,4]', split=sp, sigma=s, dof=dof))
    a = d[(d.tag == 'num') & d.deltaG_t.between(-1, 5) & d.deltaG_c.between(-1, 5)]
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)]:
        b = a[(a.dG > lo) & (a.dG <= hi)]
        rows.append(dict(estimator='paired proteases SD(t - c)/sqrt2', bin=f'({lo},{hi}]', split='all', sigma=float((b.deltaG_t - b.deltaG_c).std() / np.sqrt(2)), dof=len(b)))
    return pd.DataFrame(rows)


def main(raw, B):
    d = load(raw)
    sy = sigma_y_tables(d)
    sy.round(4).to_csv(HERE / 'sigma_y.csv', index=False)
    dbl = doubles_table(d)
    rows = []
    for name, sel in DEAD.items():
        for sp in ('train', 'val', 'test'):
            x = dbl[sel(dbl) & (dbl.split == sp)]
            r = fit(x)
            if r:
                rows.append(dict(dead_set=name, split=sp, **r))
    fl = pd.DataFrame(rows)
    # library-level bootstrap on the train libraries for each dead set
    rng = np.random.default_rng(0)
    ci = []
    for name, sel in DEAD.items():
        x = dbl[sel(dbl) & (dbl.split == 'train')]
        libs = x.WT_name.unique()
        by = {l: g for l, g in x.groupby('WT_name')}
        bs = []
        for _ in range(B):
            pick = rng.choice(libs, len(libs))
            r = fit(pd.concat([by[l] for l in pick]))
            if r:
                bs.append((r['mu_f'], r['sigma_f']))
        bs = np.array(bs)
        ci.append(dict(dead_set=name, split='train', n_libs=len(libs), mu_f_lo=np.percentile(bs[:, 0], 2.5), mu_f_hi=np.percentile(bs[:, 0], 97.5),
                       sigma_f_lo=np.percentile(bs[:, 1], 2.5), sigma_f_hi=np.percentile(bs[:, 1], 97.5)))
    fl = fl.merge(pd.DataFrame(ci), on=['dead_set', 'split'], how='left')
    fl.round(3).to_csv(HERE / 'floor_mle.csv', index=False)
    # per-library spread of the floor (dead set A, train libraries with enough readings)
    x = dbl[DEAD['A: >=1 single tagged <-1'](dbl) & (dbl.split == 'train')]
    per = []
    for l, g in x.groupby('WT_name'):
        r = fit(g)
        if r:
            per.append(dict(library=l, **r))
    per = pd.DataFrame(per)
    per.round(3).to_csv(HERE / 'floor_by_library.csv', index=False)
    cur = (-0.14, 0.5)
    lines = ['# Tobit constants: what the data say', '', f'Generated by `estimate_constants.py` (B={B} library bootstrap, capped split, train libraries unless stated). Current FINAL_p0 constants: sigma_y 0.28, mu_f -0.14, sigma_f 0.5; '
             f'under them a dead variant would be tagged <-1 with probability {norm.cdf((-1 - cur[0]) / cur[1]):.3f}.', '', '## sigma_y', '', sy.round(3).to_markdown(index=False), '',
             '## Floor (mu_f, sigma_f): censored MLE vs the naive numeric-only statistics', '', fl.round(3).to_markdown(index=False), '',
             f'Per-library floor (dead set A, train libraries with >= 40 numeric readings): {len(per)} libraries; mu_f median {per.mu_f.median():.2f} (IQR {per.mu_f.quantile(.25):.2f} to {per.mu_f.quantile(.75):.2f}), '
             f'sigma_f median {per.sigma_f.median():.2f} (IQR {per.sigma_f.quantile(.25):.2f} to {per.sigma_f.quantile(.75):.2f}).', '']
    (HERE / 'README.md').write_text('\n'.join(lines))
    print((HERE / 'README.md').read_text())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(RAW))
    ap.add_argument('--B', type=int, default=300)
    a = ap.parse_args()
    main(a.raw, a.B)
