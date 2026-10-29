#!/usr/bin/env python
"""Tsuboyama (Megascale) capped validation and test libraries: BENCHMARKS tables 1.2 and 1.6 (Tsuboyama part), and the
singles / doubles split of table 2.3.

``tsuboyama_overview``   mean per-library rho of every model on the validation (23) and test (33) libraries, 95% intervals from resampling
                         libraries (thick) and mutations within libraries (thin); red rings: documented training overlap with the libraries
                         (SPURS' published checkpoint is not here: the retrained one is; ESMtherm, IFUM, DDGemb, GeoDDG see the libraries).
``tsuboyama_per_library``  raw per-library rho: ESM-MSR (combined) against each retrained baseline, one dot per test library.
``tsuboyama_singles_doubles``  per-library rho on singles and on doubles separately (the WT head is additive and cannot rank within a double's
                         pair; the combined prediction can).
``tsuboyama_scatter``    predicted vs measured ddG of the test variants, singles and doubles.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import leak_table, load_rho_df, out_dir, subframe  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import MODELS, TSU  # noqa: E402

OUT = out_dir('03_tsuboyama')
KEYS = ['msr:comb', 'msr:wt', 'msrdev:comb', 'msrdev:wt', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm', 'esm3dg', 'rosetta', 'ifum_esmfold', 'ifum_wt', 'esmtherm', 'ddgemb',
        'geoddg', 'prosst', 'pmpnn', 'msrrel@0:comb']
RETR = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']


def overview():
    df = load_rho_df()
    df = df[df.dataset.isin(TSU)]
    P = {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in df[df.kind == 'protein'].itertuples()}
    Mu = {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in df[df.kind == 'mutation'].itertuples()}
    leak = leak_table()
    keys = [k for k in KEYS if (TSU[1], k) in P]
    keys = sorted(keys, key=lambda k: P[(TSU[1], k)][0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.33 * len(keys) + 1.4), sharey=True)
    for ax, ds, ttl in zip(axes, TSU, ('validation (23 libraries)', 'test (33 libraries)')):
        for i, k in enumerate(keys):
            if (ds, k) not in P:
                continue
            v, lo, hi = P[(ds, k)]
            kw = style.marker_kwargs(k, size=6.5)
            if k.startswith('msr:'):
                kw['color'] = kw['mec'] = 'k'
            ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.4, alpha=.45, solid_capstyle='butt')
            m = Mu.get((ds, k))
            if m:
                ax.plot([m[1], m[2]], [i, i], color=kw['color'], lw=0.8)
            ax.plot(v, i, **kw)
            if leak.get((k, ds), False):
                ax.plot(v, i, marker='o', ms=11, mfc='none', mec='#c0392b', mew=1.0, ls='none')
            ax.text(max(hi, m[2] if m else hi) + .008, i, f'{v:.3f}', va='center', fontsize=7.5, color='#444')
        ax.set_title(ttl, loc='left'); ax.grid(axis='x', color=style.GRID, lw=.6)
        ax.set_xlabel('mean per-library Spearman ρ'); ax.set_xlim(0.40, 0.93)
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels([MODELS[k]['label'] for k in keys], fontsize=8.5)
    for t, k in zip(axes[0].get_yticklabels(), keys):
        t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555'}[MODELS[k]['group']])
        if MODELS[k]['group'] == 'msr':
            t.set_fontweight('bold')
    fig.suptitle('thick bar: resampling libraries; thin line: resampling mutations; red ring: documented training overlap', fontsize=8.5, y=1.0)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_overview')
    return P


def per_library():
    fr = load_frame(TSU[1])
    sp = fr.spine
    ok = sp['use'].to_numpy()
    y = sp['y'].to_numpy()
    libs = sorted(sp['group'][ok].unique())

    def lib_rho(k):
        out = {}
        for g in libs:
            m = ok & (sp['group'] == g).to_numpy() & np.isfinite(fr.preds[k]).all(0)
            out[g] = np.mean([spearmanr(p[m], y[m])[0] for p in fr.preds[k]])
        return pd.Series(out)
    ref = lib_rho('msr:comb')
    R = pd.DataFrame({'msr:comb': ref, **{k: lib_rho(k) for k in RETR}})
    R.to_csv(OUT / 'per_library_rho_test.csv', float_format='%.4f')
    fig, axes = plt.subplots(1, len(RETR), figsize=(2.6 * len(RETR), 3.0), sharex=True, sharey=True)
    for ax, k in zip(axes, RETR):
        ax.scatter(R[k], R['msr:comb'], s=22, color=MODELS[k]['color'], alpha=.8, lw=0)
        ax.plot([.3, 1], [.3, 1], color='#999', lw=.8, ls='--')
        ax.set_title(MODELS[k]['label'].replace(' (retrained)', ''), fontsize=8.5)
        ax.text(.04, .95, f'ESM-MSR higher in {(R["msr:comb"] > R[k]).sum()} / {len(R)}', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('ρ of the baseline')
    axes[0].set_ylabel('ρ of ESM-MSR combined')
    fig.suptitle('Each dot is one test library', fontsize=9, y=1.0)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_per_library')


def singles_doubles():
    rows = []
    for ds in TSU:
        fr = load_frame(ds)
        sp = fr.spine
        for nm, m in (('singles', (sp['n_mut'] == 1).to_numpy() & sp['use'].to_numpy()), ('doubles', (sp['n_mut'] == 2).to_numpy() & sp['use'].to_numpy())):
            sub = subframe(fr, m, keys=KEYS)
            point, reps = bootstrap_rho({ds: sub}, KEYS, kind='protein', B=1000, seed=6, per_protein=True)
            for k, (rho, ps, n, ng) in point[ds].items():
                lo, hi = ci(reps[ds][k])
                rows.append(dict(dataset=ds, subset=nm, model=k, rho=rho, lo=lo, hi=hi, n=n, n_libraries=ng))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'tsuboyama_singles_doubles.csv', index=False, float_format='%.4f')
    groups = [f'{d}|{s}' for d in TSU for s in ('singles', 'doubles')]
    glabel = [f'{"val" if d == TSU[0] else "test"}\n{s}' for d in TSU for s in ('singles', 'doubles')]
    models = ['msr:wt', 'msr:comb', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
    table = {(f'{r.dataset}|{r.subset}', r.model): r.rho for r in R.itertuples()}
    cis = {(f'{r.dataset}|{r.subset}', r.model): (r.lo, r.hi) for r in R.itertuples()}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    style.dot_panel(ax, groups, models, table, cis, highlight=['msr:wt', 'msr:comb'], ylabel='mean per-library ρ', xlabels=glabel, rot=0, size=6.5, ylim=(0.5, 0.95))
    style.model_legend(ax, models, loc='lower left', ncol=4, fontsize=7.5)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_singles_doubles')


def scatter():
    fr = load_frame(TSU[1])
    sp = fr.spine
    y = sp['y'].to_numpy()
    ok = sp['use'].to_numpy()
    panels = [('msr:wt', 'singles', 'ESM-MSR WT head'), ('msr:wt', 'doubles', 'ESM-MSR WT head'), ('msr:comb', 'doubles', 'ESM-MSR combined'), ('me2', 'doubles', 'MutateEverything st2'),
              ('tmpnnd', 'doubles', 'ThermoMPNN + D'), ('spursm', 'doubles', 'SPURS multi (retrained)')]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.2), sharey=True)
    for ax, (k, subset, t) in zip(axes, panels):
        m = ok & ((sp['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy()) & np.isfinite(fr.preds[k][0])
        style.density_scatter(ax, fr.preds[k][0][m], y[m], s=3, line=True, max_points=40000)
        ax.set_title(f'{t}\n{subset}', fontsize=8.5); ax.set_xlabel('predicted ddG')
    axes[0].set_ylabel('measured ddG (kcal/mol)')
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_scatter')


if __name__ == '__main__':
    P = overview()
    per_library()
    singles_doubles()
    scatter()
    for k in ('msr:comb', 'msr:wt', 'me1', 'tmpnn', 'spurs'):
        print(k, [round(P[(d, k)][0], 3) for d in TSU])
