#!/usr/bin/env python
"""Thermostability: the 12 external sets (BENCHMARKS tables 1.1 and the external-set part of 1.6).

Three figures, all with protein-level bootstrap intervals (whole proteins resampled, the same proteins for every model, so
differences are paired):

``thermo_all_models``      ESM-MSR against every model of the benchmark, one small panel per set, models sorted by their mean over the
                           12 sets (the column on the right, with its own interval).  Red rings mark cells whose model has documented training
                           overlap with the set (leak audit, BENCHMARKS section 6); the retrained baselines and ESM-MSR have none.
``thermo_vs_retrains``     the central comparison: ESM-MSR (WT head and combined) against the models retrained on the same split, set by set, and the
                           paired difference (ESM-MSR minus each baseline) with its interval.
``thermo_interval_types``  how much wider the interval is when proteins, not mutations, are resampled.

ESM-MSR is shown by its WT head on the ten single-mutant sets and by its combined prediction on PTMUL and PTMUL-D.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import ci  # noqa: E402
from msr_analysis.common import leak_table, load_rho_df, out_dir, primary_ci  # noqa: E402
from msr_analysis.registry import CACHE, EXT12, LABEL, MEGASCALE_NOTE, MEGASCALE_TRAINED, MODELS, RETRAINS  # noqa: E402

OUT = out_dir('02_thermostability')
MULTI = {'ptmul', 'ptmuld'}
ALL_KEYS = ['msr:wt', 'msr:comb', 'msrdev:wt', 'msrdev:comb', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm', 'esmtherm', 'ddgemb', 'ddmut', 'ifum_wt',
            'ifum_esmfold', 'geoddg', 'esm3dg', 'prosst', 'pmpnn', 'esm3s', 'esm3m', 'esm3l', 'msrrel@0:comb']


def msr_primary(ds, fam='msr'):
    return f'{fam}:comb' if ds in MULTI else f'{fam}:wt'


def main():
    df = load_rho_df()
    P = primary_ci(df[df.dataset.isin(EXT12)])
    reps = pickle.load(open(CACHE / 'boot_protein.pkl', 'rb'))
    leak = leak_table()
    # ---- mean over the 12 sets, with interval, for models scored on all twelve -------------------------------
    mean12 = {}
    for k in ALL_KEYS:
        if all(k in reps[d] for d in EXT12):
            r = np.mean([reps[d][k] for d in EXT12], axis=0)
            pt = np.mean([P[(d, k)][0] for d in EXT12])
            mean12[k] = (pt,) + ci(r)
    order = sorted(mean12, key=lambda k: mean12[k][0])
    partial = [k for k in ALL_KEYS if k not in mean12]
    rows = []
    for k in ALL_KEYS:
        for d in EXT12:
            if (d, k) in P:
                rows.append(dict(model=k, dataset=d, rho=P[(d, k)][0], lo=P[(d, k)][1], hi=P[(d, k)][2], leak_flag=leak.get((k, d), False)))
        if k in mean12:
            rows.append(dict(model=k, dataset='mean of 12', rho=mean12[k][0], lo=mean12[k][1], hi=mean12[k][2], leak_flag=any(leak.get((k, d), False) for d in EXT12)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'thermostability_rho.csv', index=False, float_format='%.4f')

    # ---- figure 1: every model ---------------------------------------------------------------------------------
    mega = [k for k in order if k in MEGASCALE_TRAINED]                   # trained on Megascale: last (bottom), greyed, starred
    models = mega + [k for k in partial if any((d, k) in P for d in EXT12)] + [k for k in order if k not in MEGASCALE_TRAINED]
    fig, axes = plt.subplots(1, 13, figsize=(17, 0.34 * len(models) + 1.6), sharey=True, gridspec_kw=dict(width_ratios=[1] * 12 + [1.15]))
    ypos = {k: i for i, k in enumerate(models)}
    for ax, d in zip(axes, EXT12 + ['mean of 12']):
        for k, i in ypos.items():
            v = (mean12[k] if d == 'mean of 12' else P.get((d, k))) if (d != 'mean of 12' or k in mean12) else None
            if v is None:
                continue
            kw = style.marker_kwargs(k, size=5.5 if not k.startswith('msr:') else 7)
            if k.startswith('msr') and not k.startswith('msrdev') and not k.startswith('msrrel'):
                kw['color'] = kw['mec'] = 'k'
            ax.plot([v[1], v[2]], [i, i], color=kw['color'], lw=1.0, alpha=.8, zorder=2)
            ax.plot(v[0], i, zorder=4, **kw)
            if d != 'mean of 12' and leak.get((k, d), False):
                ax.plot(v[0], i, marker='o', ms=10, mfc='none', mec='#c0392b', mew=1.0, ls='none', zorder=5)
        for k in ('msr:wt', 'msr:comb'):
            ax.axhspan(ypos[k] - .5, ypos[k] + .5, color='#f2f2f2', zorder=0)
        ax.axhspan(-.5, len(mega) - .5, color='#bdbdbd', alpha=.35, zorder=0)
        ax.set_title(LABEL.get(d, d), fontsize=9)
        ax.grid(axis='x', color=style.GRID, lw=.6)
        ax.set_ylim(-.7, len(models) - .3)
        ax.tick_params(axis='x', labelsize=7.5)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([MODELS[k]['label'] for k in models], fontsize=8.5)
    for t, k in zip(axes[0].get_yticklabels(), models):
        t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555555'}[MODELS[k]['group']])
        if MODELS[k]['group'] == 'msr':
            t.set_fontweight('bold')
    fig.supxlabel('Spearman ρ (95% protein-level bootstrap); red ring = documented training overlap with the set.   ' + MEGASCALE_NOTE + ' (grey rows)', fontsize=9.5, y=-0.01)
    fig.tight_layout(w_pad=0.3)
    style.save(fig, OUT / 'thermo_all_models')

    # ---- figure 2: ESM-MSR vs the retrained baselines ------------------------------------------------------------------
    cols = ['msr:wt', 'msr:comb'] + RETRAINS
    table = {(d, k): P[(d, k)][0] for d in EXT12 for k in cols if (d, k) in P}
    cis = {(d, k): P[(d, k)][1:] for d in EXT12 for k in cols if (d, k) in P}
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[4.3, 1], hspace=0.38, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    style.dot_panel(ax, EXT12, cols, table, cis, highlight=['msr:wt', 'msr:comb'], ylabel='Spearman ρ', ylim=(0.2, 0.97), spread=0.78, size=6.2)
    style.model_legend(ax, cols, loc='upper center', ncol=8, fontsize=7.5, columnspacing=0.9, handletextpad=0.2)
    ax.set_title('A  ESM-MSR against the baselines retrained on the same split (seed means; whiskers: 95% protein-level bootstrap)', loc='left')
    # paired differences against the primary head
    ax2 = fig.add_subplot(gs[1, 0])
    n = len(EXT12)
    style.band_axis(ax2, n)
    offs = np.linspace(-.34, .34, len(RETRAINS))
    summary = []
    for j, k in enumerate(RETRAINS):
        xs, ys, lo, hi = [], [], [], []
        dm = []
        for i, d in enumerate(EXT12):
            if k not in reps[d]:
                continue
            a = reps[d][msr_primary(d)]
            diff = a - reps[d][k]
            pt = P[(d, msr_primary(d))][0] - P[(d, k)][0]
            l, h = ci(diff)
            xs.append(i + offs[j]); ys.append(pt); lo.append(pt - l); hi.append(h - pt)
            dm.append(diff)
            summary.append(dict(dataset=d, baseline=k, delta=pt, lo=l, hi=h, p_two_sided=float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
        kw = style.marker_kwargs(k, size=5.8)
        ax2.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=kw['color'], elinewidth=1, capsize=1.4, zorder=2)
        ax2.plot(xs, ys, zorder=3, **kw)
        if len(dm) == n:
            m = np.mean(dm, axis=0)
            pt = np.mean([P[(d, msr_primary(d))][0] - P[(d, k)][0] for d in EXT12])
            l, h = ci(m)
            summary.append(dict(dataset='mean of 12', baseline=k, delta=pt, lo=l, hi=h, p_two_sided=float(2 * min((m <= 0).mean(), (m >= 0).mean()))))
    ax2.axhline(0, color='k', lw=.8)
    ax2.set_xticks(range(n)); ax2.set_xticklabels([LABEL[d] for d in EXT12], rotation=30, ha='right')
    ax2.set_ylabel('Δρ = ESM-MSR − baseline'); ax2.set_title('B  Paired difference per set (positive: ESM-MSR ahead)', loc='left')
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    S2 = pd.DataFrame(summary)
    S2.to_csv(OUT / 'thermo_paired_vs_retrains.csv', index=False, float_format='%.4f')
    m12 = S2[S2.dataset == 'mean of 12']
    for j, r in enumerate(m12.itertuples()):
        kw = style.marker_kwargs(r.baseline, size=7)
        ax3.errorbar([j], [r.delta], yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt='none', ecolor=kw['color'], capsize=2)
        ax3.plot([j], [r.delta], **kw)
    ax3.axhline(0, color='k', lw=.8); ax3.set_xticks(range(len(m12))); ax3.set_xticklabels([MODELS[b]['label'].replace(' (retrained)', '').replace('MutateEverything', 'ME').replace('ThermoMPNN + ThermoMPNN-D', 'TM+D') for b in m12.baseline], rotation=60, ha='right', fontsize=8)
    ax3.set_title('C  Mean of 12', loc='left'); plt.setp(ax3.get_yticklabels(), visible=False)
    style.save(fig, OUT / 'thermo_vs_retrains')
    print(S2[S2.dataset == 'mean of 12'].round(4).to_string(index=False))

    # ---- figure 3: what the interval depends on -------------------------------------------------------------------------
    M = df[df.kind == 'mutation'].set_index(['dataset', 'model'])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(n)
    wp = [(P[(d, msr_primary(d))][2] - P[(d, msr_primary(d))][1]) / 2 for d in EXT12]
    wm = [(M.loc[(d, msr_primary(d)), 'hi'] - M.loc[(d, msr_primary(d)), 'lo']) / 2 for d in EXT12]
    ax.bar(x - .2, wm, .4, color='#9ecae1', label='resampling mutations')
    ax.bar(x + .2, wp, .4, color='#08519c', label='resampling proteins')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in EXT12], rotation=30, ha='right'); ax.set_ylabel('95% interval half-width of ρ')
    ax.legend(); ax.set_title('Intervals for ESM-MSR (WT head / combined): mutations in one protein are not independent', loc='left')
    fig.tight_layout()
    style.save(fig, OUT / 'thermo_interval_types')


def scatter():
    """raw predictions against measurements for the 12 sets (ESM-MSR, seed 1, the head the set calls for)"""
    from msr_analysis.data import load_frame
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.4))
    for ax, d in zip(axes.ravel(), EXT12):
        fr = load_frame(d)
        k = msr_primary(d)
        ok = fr.spine['use'].to_numpy() & np.isfinite(fr.preds[k][0])
        style.density_scatter(ax, fr.preds[k][0][ok], fr.spine['y'].to_numpy()[ok], s=6, line=False, max_points=9000)
        ax.set_title(f'{LABEL[d]}  ({"combined" if d in MULTI else "WT head"})', fontsize=9)
        if d == 's571':
            ax.set_ylabel('measured (S571 labels are not in kcal/mol)')
    for ax in axes[:, 0]:
        ax.set_ylabel('measured ddG (kcal/mol)')
    for ax in axes[-1]:
        ax.set_xlabel('ESM-MSR prediction')
    fig.suptitle('ESM-MSR prediction against measurement on the external stability sets (seed 1, all variants)', y=1.0, fontsize=10)
    fig.tight_layout()
    style.save(fig, OUT / 'thermo_scatter_external')


if __name__ == '__main__':
    main()
    scatter()
