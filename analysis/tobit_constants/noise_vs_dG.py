#!/usr/bin/env python
"""Measurement noise of the Tsuboyama dG as a function of dG, and where it starts to grow towards the floor.

Two independent noise estimators on the curated readings (numeric dG_ML in [-1, 5]):
* REPLICATES: variants measured more than once (same library, same mutation, different DNA). The residual sum of squares about the variant's mean has n - 1 degrees of
  freedom and expectation (n - 1) sigma^2; binned (and modelled) on the variant's mean reading.
* PROTEASES: SD(dG_trypsin - dG_chymotrypsin)/sqrt(2), per bin of the combined dG_ML. It is per-protease noise plus any protease-specific systematic disagreement, so it is an
  upper bound on the reproducibility noise of one protease (and of the combined reading by sqrt(2)).

Change-point model of the heteroscedastic noise: sigma(g) = sigma0 + k * max(0, g0 - g) on g in [-1, 3.5] (constant above g0, growing linearly towards the floor below it);
g0 by profile likelihood (95% = within 1.92 log-likelihood units), library-level bootstrap for the protease estimator.

    python analysis/tobit_constants/noise_vs_dG.py [--raw CSV] [--B 200]
Writes noise_vs_dG.csv, noise_changepoint.csv, noise_vs_dG.png in this directory.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimate_constants import RAW, load  # noqa: E402

GRID = np.round(np.arange(-1.0, 2.01, 0.1), 2)
LO, HI = -1.0, 3.5


def sigma_fn(g, s0, k, g0):
    return np.maximum(s0 + k * np.maximum(0.0, g0 - g), 1e-3)


def nll_rep(p, g0, g, ss, dof):
    s = sigma_fn(g, p[0], p[1], g0)
    return float((dof * np.log(s) + ss / (2 * s ** 2)).sum())


def nll_pair(p, g0, g, d):
    s = np.sqrt(2) * sigma_fn(g, p[0], p[1], g0)
    return float((np.log(s) + d ** 2 / (2 * s ** 2)).sum())


def profile(nll, args):
    out = []
    for g0 in GRID:
        r = minimize(lambda p: nll(p, g0, *args) if p[1] >= 0 and p[0] > 0 else 1e12, [0.25, 0.05], method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-8))
        out.append((g0, r.fun, r.x[0], r.x[1]))
    return pd.DataFrame(out, columns=['g0', 'nll', 'sigma0', 'k'])


def replicate_data(d):
    g = d[d.tag == 'num'].groupby(['WT_name', 'mut_type']).dG
    r = g.agg(['mean', 'size', lambda s: float(((s - s.mean()) ** 2).sum())]).reset_index()
    r.columns = ['WT_name', 'mut_type', 'g', 'n', 'ss']
    r = r[(r.n >= 2) & r.g.between(LO, HI)]
    r['dof'] = r.n - 1
    return r


def protease_data(d):
    a = d[(d.tag == 'num') & d.deltaG_t.between(-1, 5) & d.deltaG_c.between(-1, 5) & d.dG.between(LO, HI)].copy()
    a['d'] = a.deltaG_t - a.deltaG_c
    a['d'] = a.d - a.d.mean()
    return a


def binned(g, num, den, edges, boot=None):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g > lo) & (g <= hi)
        if m.sum() < 8:
            continue
        rows.append(dict(lo=lo, hi=hi, mid=(lo + hi) / 2, n=int(m.sum()), sigma=float(np.sqrt(num[m].sum() / den[m].sum()))))
    return pd.DataFrame(rows)


def main(raw, B):
    d = load(raw)
    # the load() frame has no per-protease columns beyond deltaG_t / deltaG_c: fine, both are numeric everywhere
    rep, pro = replicate_data(d), protease_data(d)
    edges = np.round(np.arange(-1.0, 4.01, 0.25), 2)
    br = binned(rep.g.to_numpy(), rep.ss.to_numpy(), rep.dof.to_numpy(), edges)
    br['estimator'] = 'replicates'
    bp = binned(pro.dG.to_numpy(), (pro.d ** 2 / 2).to_numpy(), np.ones(len(pro)), edges)
    bp['estimator'] = 'protease pairs'
    # bootstrap CI of each binned SD (variants for replicates, libraries for protease pairs)
    rng = np.random.default_rng(0)
    for df, src, key, num, den in ((br, rep, 'WT_name', 'ss', 'dof'), (bp, pro, 'WT_name', None, None)):
        libs = src[key].unique(); by = {l: g for l, g in src.groupby(key)}
        draws = []
        for _ in range(B):
            s = pd.concat([by[l] for l in rng.choice(libs, len(libs))])
            if num:
                draws.append(binned(s.g.to_numpy(), s.ss.to_numpy(), s.dof.to_numpy(), edges).set_index('lo').sigma)
            else:
                draws.append(binned(s.dG.to_numpy(), (s.d ** 2 / 2).to_numpy(), np.ones(len(s)), edges).set_index('lo').sigma)
        dd = pd.concat(draws, axis=1)
        df['lo_ci'] = df.lo.map(dd.quantile(0.025, axis=1)); df['hi_ci'] = df.lo.map(dd.quantile(0.975, axis=1))
    tab = pd.concat([br, bp])
    tab.round(4).to_csv(HERE / 'noise_vs_dG.csv', index=False)
    # change points
    cps = []
    pr = profile(nll_rep, (rep.g.to_numpy(), rep.ss.to_numpy(), rep.dof.to_numpy()))
    pp = profile(nll_pair, (pro.dG.to_numpy(), pro.d.to_numpy()))
    for name, p, n in (('replicates', pr, len(rep)), ('protease pairs', pp, len(pro))):
        best = p.loc[p.nll.idxmin()]
        ok = p[p.nll <= best.nll + 1.92]
        flat = float(p[p.g0 <= -0.99].nll.min() - best.nll)       # no growth at all: g0 = -1 (hinge outside the range)
        cps.append(dict(estimator=name, n=n, g0_hat=best.g0, g0_lo=ok.g0.min(), g0_hi=ok.g0.max(), sigma0=best.sigma0, k=best.k, sigma_at_floor=best.sigma0 + best.k * (best.g0 - LO),
                        dnll_vs_constant_noise=flat))
    # library bootstrap for the protease change point (fewer draws: each is a profile)
    libs = pro.WT_name.unique(); by = {l: g for l, g in pro.groupby('WT_name')}
    bs = []
    for _ in range(min(B, 60)):
        s = pd.concat([by[l] for l in rng.choice(libs, len(libs))])
        q = profile(nll_pair, (s.dG.to_numpy(), s.d.to_numpy()))
        bs.append(q.loc[q.nll.idxmin(), 'g0'])
    cps.append(dict(estimator='protease pairs, library bootstrap (60 draws)', n=len(pro), g0_hat=float(np.median(bs)), g0_lo=float(np.percentile(bs, 2.5)), g0_hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(cps).round(3).to_csv(HERE / 'noise_changepoint.csv', index=False)
    print(tab.round(3).to_string(index=False)); print(pd.DataFrame(cps).round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, (nm, df, best) in zip(ax, (('replicates (same variant, different DNA)', br, cps[0]), ('trypsin vs chymotrypsin, SD/sqrt2', bp, cps[1]))):
        a.errorbar(df.mid, df.sigma, yerr=[df.sigma - df.lo_ci, df.hi_ci - df.sigma], fmt='o', color='C0', capsize=2)
        gg = np.linspace(LO, HI, 200); a.plot(gg, sigma_fn(gg, best['sigma0'], best['k'], best['g0_hat']), color='C3', label=f"hinge fit: flat above g0 = {best['g0_hat']:.1f} ({best['g0_lo']:.1f} to {best['g0_hi']:.1f})")
        for x, c, t in ((-0.14, 'k', 'mu_f -0.14'), (-1.0, 'C2', 'tag bound -1')):
            a.axvline(x, color=c, ls=':', lw=1)
        a.set_title(nm, fontsize=10); a.set_xlabel('dG (kcal/mol)'); a.legend(fontsize=8, loc='upper center'); a.spines[['top', 'right']].set_visible(False)
    ax[0].set_ylabel('noise SD (kcal/mol)')
    fig.tight_layout(); fig.savefig(HERE / 'noise_vs_dG.png', dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(RAW))
    ap.add_argument('--B', type=int, default=200)
    a = ap.parse_args()
    main(a.raw, a.B)
