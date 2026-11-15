#!/usr/bin/env python
"""An equation for the rectified gain per variant of the Homology-Restricted MegaScale test set.

``16_screening_overview.py`` shows the rectified gain per variant: the sum of max(ddG, 0) over the K best-scored variants of a protein divided by K (the mean stabilization of the K picks, in kcal/mol),
averaged over proteins.  On a log-K axis the curves are straight lines, so the fit is

    gain per variant  g(K) = a - b ln K            (a: the gain of the single best pick; b: the loss per e-fold more variants screened, b ln 10 per decade)

for every model, for the oracle (the K largest labels) and, as a constant, for random picks.  Other shapes are fitted for comparison (a power law, a power law with a floor, an exponential with a floor, the
logarithm shifted by k0); the table reports the root-mean-square error and the largest error of each over the seven K (1, 3, 5, 10, 20, 50, 100).  The parameters of the line get a 95% interval from a
bootstrap over proteins (the proteins are resampled jointly for all K; the same draws for every model).  The data are the per-protein tables of ``08_screening.py`` (single mutants, Cys excluded; the "multi"
version scores the doubles with the multi-mutant model of each family, as in ``16_screening_overview.py``).

Writes ``results/17_gain_fit/``: ``gain_fit_params.csv`` (a, b with intervals, rmse, max error, R^2 per model), ``gain_fit_shapes.csv`` (the other shapes), ``gain_fit_{single,multi}.png/pdf``.

    python analysis/17_gain_per_variant_fit.py [--B 200]
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from msr_analysis import screening as scr_core  # noqa: E402
from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

scr = importlib.import_module('08_screening')
OUT = out_dir('17_gain_fit')
DS = 'tsuboyama_capped_test'
KS = [1, 3, 5, 10, 20, 50, 100]
LK = np.log(KS)
EXTRA = ['esm3', 'pmpnn', 'prosst', 'rosetta']
VERSIONS = {'single': ('singles', ['msrp0', 'me', 'tm', 'spurs'] + EXTRA), 'multi': ('composite', ['msr', 'me', 'tm', 'spurs'] + EXTRA)}
MAIN = {'single': ['msrp0', 'me', 'tm', 'spurs'], 'multi': ['msr', 'me', 'tm', 'spurs']}
SHAPES = {'a - b ln K': (lambda k, a, b: a - b * np.log(k), [0.8, 0.1]),
          'a K^-b': (lambda k, a, b: a * k ** -b, [0.8, 0.2]),
          'a K^-b + c': (lambda k, a, b, c: a * k ** -b + c, [0.8, 0.5, 0.0]),
          'a exp(-K/t) + c': (lambda k, a, t, c: a * np.exp(-k / t) + c, [0.5, 30, 0.2]),
          'a - b ln(K + k0)': (lambda k, a, b, k0: a - b * np.log(k + k0), [0.8, 0.1, 0.5])}


def tables(fam, mode):
    """{K: per-protein table of gain, oracle_gain, rand_gain} for one family (the rows of ``08_screening.per_protein``: Cys excluded, usable rows, finite predictions)."""
    fr = load_frame(DS)
    sp = fr.spine
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
    keep = ~to_cys & ((sp['n_mut'].to_numpy() == 1) if mode == 'singles' else True)
    k = scr.fam_key(fam, fr, mode=mode)
    ok = sp['use'].to_numpy() & keep & np.isfinite(fr.preds[k]).all(axis=0)
    y, g = sp['y'].to_numpy()[ok], sp['group'].to_numpy()[ok]
    return {K: scr_core.per_protein_table(g, fr.preds[k][0][ok], y, K, 0.5) for K in KS}


def matrix(tabs, col):
    """proteins x K matrix of ``col`` / K (NaN where a protein has too few variants for that K)."""
    prots = sorted(set().union(*[set(t['group']) for t in tabs.values()]))
    M = np.full((len(prots), len(KS)), np.nan)
    for j, K in enumerate(KS):
        s = tabs[K].set_index('group')[col] / K
        M[:, j] = s.reindex(prots).to_numpy()
    return M


def line(g):
    b, a = np.polyfit(LK, g, 1)
    return a, -b


def boot_line(M, B, seed=0):
    """(a, b) of every bootstrap replicate: proteins resampled jointly across K, the mean at each K over the proteins present."""
    rng = np.random.default_rng(seed)
    n = len(M)
    ab = []
    for _ in range(B):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        m = np.isfinite(M)
        g = (np.where(m, M, 0) * w[:, None]).sum(0) / (m * w[:, None]).sum(0)
        ab.append(line(g))
    return np.array(ab)


def shapes(g):
    out = {}
    for name, (f, p0) in SHAPES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p, _ = curve_fit(f, np.array(KS, float), g, p0=p0, maxfev=20000)
            r = g - f(np.array(KS, float), *p)
            out[name] = (float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()), p)
        except Exception:
            out[name] = (np.nan, np.nan, None)
    return out


def r2(g, pred):
    return float(1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200)
    B = ap.parse_args().B
    rows, srows, curves = [], [], {}
    for ver, (mode, fams) in VERSIONS.items():
        cur, nmed = {}, {}
        for fam in fams:
            tabs = tables(fam, mode)
            M = matrix(tabs, 'gain')
            m = np.isfinite(M)
            g = np.array([M[m[:, j], j].mean() for j in range(len(KS))])
            cur[fam] = g
            if fam == fams[0]:                                                       # the oracle and random lines come from the first family's tables (they do not depend on the model)
                for nm, col in (('oracle', 'oracle_gain'), ('random', 'rand_gain')):
                    Mo = matrix(tabs, col)
                    mo = np.isfinite(Mo)
                    cur[nm] = np.array([Mo[mo[:, j], j].mean() for j in range(len(KS))])
                    cur[nm + '_M'] = Mo
                nmed['v'] = float(np.median(tabs[1]['n']))                          # variants per protein (median)
            cur[fam + '_M'] = M
        for name, g in [(n, v) for n, v in cur.items() if not n.endswith('_M')]:
            M = cur[name + '_M']
            if name == 'random':
                rows.append(dict(version=ver, model=name, a=float(g.mean()), a_lo=np.nan, a_hi=np.nan, b=0.0, b_lo=np.nan, b_hi=np.nan, rmse=float(g.std()), max_err=float(np.abs(g - g.mean()).max()), r2=np.nan,
                                 n_proteins=int(len(cur['oracle_M'])), k0=np.nan, n_variants_median=nmed['v']))
                continue
            ab = boot_line(M, B)
            a, b = line(g)
            pred = a - b * LK
            rows.append(dict(version=ver, model=name, a=a, a_lo=np.quantile(ab[:, 0], .025), a_hi=np.quantile(ab[:, 0], .975), b=b, b_lo=np.quantile(ab[:, 1], .025), b_hi=np.quantile(ab[:, 1], .975),
                             rmse=float(np.sqrt(((g - pred) ** 2).mean())), max_err=float(np.abs(g - pred).max()), r2=r2(g, pred), n_proteins=int(len(M)),
                             k0=float(np.exp((a - float(cur['random'].mean())) / b)), n_variants_median=nmed['v']))
            for sh, (rm, mx, p) in shapes(g).items():
                srows.append(dict(version=ver, model=name, shape=sh, rmse=rm, max_err=mx, params=' '.join(f'{v:.4g}' for v in p) if p is not None else ''))
        curves[ver] = cur
    P = pd.DataFrame(rows)
    P.to_csv(OUT / 'gain_fit_params.csv', index=False, float_format='%.4f')
    S = pd.DataFrame(srows)
    S.to_csv(OUT / 'gain_fit_shapes.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    print(P.round(3).to_string())
    print(S.pivot_table(index=['version', 'model'], columns='shape', values='rmse').round(4).to_string())
    for ver in VERSIONS:
        draw(ver, P[P.version == ver].set_index('model'), curves[ver])


LABEL = {'msrp0': 'ESM-MSR', 'msr': 'ESM-MSR', 'me': 'MutateEverything', 'tm': 'ThermoMPNN', 'spurs': 'SPURS', 'oracle': 'oracle (K largest labels)', 'random': 'random picks'}
COLOR = {'msrp0': '#000000', 'msr': '#000000', 'me': '#2ca02c', 'tm': '#1f77b4', 'spurs': '#ff7f0e', 'oracle': '#2c3e50', 'random': '#c0392b'}


def draw(ver, P, cur):
    main = MAIN[ver]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax, rx = axes
    xx = np.geomspace(1, 100, 100)
    for name in main + ['oracle']:
        r = P.loc[name]
        col = COLOR[name]
        ax.plot(KS, cur[name], 'o', color=col, ms=5.5, zorder=4, mec='white', mew=0.8)
        ax.plot(xx, r.a - r.b * np.log(xx), color=col, lw=1.8 if name != 'oracle' else 1.4, ls='-' if name != 'oracle' else ':', zorder=3,
                label=f'{LABEL[name]}:  {r.a:.2f} − {r.b:.3f} ln K   (rmse {r.rmse:.3f})')
        rx.plot(KS, cur[name] - (r.a - r.b * np.log(KS)), 'o-', color=col, ms=4.5, lw=1.1, label=LABEL[name])
    ax.plot(KS, cur['random'], 's', color=COLOR['random'], ms=4)
    ax.axhline(P.loc['random', 'a'], color=COLOR['random'], ls=(0, (3, 2)), lw=1.4, label=f'random picks:  {P.loc["random", "a"]:.3f}')
    ax.set_xscale('log')
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel('variants screened per protein (K)')
    ax.set_ylabel('rectified gain per variant (kcal/mol)')
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.95)
    rx.axhline(0, color='#888888', lw=0.8)
    rx.set_xscale('log')
    rx.set_xticks(KS)
    rx.set_xticklabels(KS)
    rx.set_xlabel('variants screened per protein (K)')
    rx.set_ylabel('data − fit (kcal/mol)')
    rx.set_ylim(-0.06, 0.06)
    rx.legend(fontsize=8, loc='upper right', framealpha=0.95)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=8, color='#333333')
    t = fig.text(0.012, 0.012, style.wrap_to_px(fig, 'Points: rectified gain per variant of the Homology-Restricted MegaScale test set (single mutants, Cys excluded; ' + ('singles scored by the single-mutant model, doubles by the multi-mutant model' if ver == 'multi' else 'single-mutant models') +
                                                   f'), the mean over proteins of (sum of max(ddG, 0) over the K best-scored variants) / K.  Lines: the least-squares line a − b ln K through the seven K.  Left panel: the fitted equation of each model with the '
                                                   f'root-mean-square error of the fit; the oracle takes the K largest labels.  Right: the same fits as residuals.  ESM-MSR\'s line reaches the random level at K = {P.loc[main[0], "k0"]:.0f}, '
                                                   f'close to the number of variants of a protein (median {P.loc[main[0], "n_variants_median"]:.0f}), where the picks are all variants and the gain per variant must equal the random level.', 0.97 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    fig.suptitle('Rectified gain per variant falls linearly in ln K: g(K) = a − b ln K', x=0.012, ha='left', fontsize=12, fontweight='bold')
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', ver, bad)
    style.save(fig, OUT / f'gain_fit_{ver}')


if __name__ == '__main__':
    main()
