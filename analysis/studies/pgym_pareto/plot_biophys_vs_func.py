#!/usr/bin/env python3
"""Render standalone publication-ready Biophys-2 vs Functional-3 Pareto figure (176 unseen ProteinGym assays)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis import style
from msr_analysis.paths import results_dir


def main() -> int:
    out = results_dir('99_misc/pgym_pareto', create=True)
    tab = pd.read_csv(out / 'pgym_176_all105_summary.csv', index_col='model')
    cis_df = pd.read_csv(out / 'pgym_176_bootstrap_cis.csv')
    ci_map = {(r.model, r.metric): (float(r.lo), float(r.hi)) for r in cis_df.itertuples()}

    # Exclude leaked stability baselines from the main-text panel
    plot_tab = tab[tab['modality'] != 'Stability baseline (leaked*)'].copy()
    xcol, ycol = 'Biophys2_macro', 'Func3_macro'

    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    mod_style = {
        'Sequence-only': dict(color='#b8c0c8', marker='o', s=28, alpha=0.55, zorder=2, label='Public: Sequence-only (46)'),
        'Structure + Sequence (no MSA)': dict(color='#4d90bd', marker='s', s=34, alpha=0.75, zorder=3, label='Public: Structure + Sequence, no MSA (26)'),
        'MSA / Retrieval': dict(color='#e08c3a', marker='^', s=36, alpha=0.75, zorder=3, label='Public: MSA / Retrieval (23)'),
        'Stability baseline (clean)': dict(color='#2d8a4e', marker='D', s=44, alpha=0.90, zorder=4, label='Stability baselines (SPURS, ThermoMPNN, ME, GeoDDG, Rosetta)'),
    }
    for mod, kw in mod_style.items():
        sub = plot_tab[plot_tab['modality'] == mod]
        if len(sub):
            ax.scatter(sub[xcol], sub[ycol], edgecolors='none', **kw)

    # Compute exact 2D Pareto frontier (non-dominated)
    A = plot_tab[[xcol, ycol]].to_numpy(float)
    names = plot_tab.index.tolist()
    nd = [
        names[i]
        for i in range(len(names))
        if not any(np.all(A[j] >= A[i]) and np.any(A[j] > A[i]) for j in range(len(names)) if j != i)
    ]
    pf = plot_tab.loc[nd].sort_values(xcol, ascending=False)

    # Draw crisp Pareto frontier staircase + subtle right/top boundary shading
    px = pf[xcol].to_numpy(float)
    py = pf[ycol].to_numpy(float)
    sx, sy = [px[0]], [py[0]]
    for k in range(len(px) - 1):
        sx.extend([px[k + 1], px[k + 1]])
        sy.extend([py[k], py[k + 1]])
    ax.plot(sx, sy, color='#222222', lw=1.6, ls='-', alpha=0.78, zorder=5, label='Global Pareto frontier (3 of 105 models)')

    # Ring non-MSR frontier point(s)
    for nm in pf.index:
        if not nm.startswith('ESM-MSR'):
            m_st = mod_style[plot_tab.loc[nm, 'modality']]
            ax.scatter(
                plot_tab.loc[nm, xcol], plot_tab.loc[nm, ycol],
                s=68, marker=m_st['marker'], facecolor=m_st['color'], edgecolor='#111111', linewidth=1.2, zorder=7,
            )

    # Also lightly ring ProSST-2048 as top alignment-free public baseline
    for nm in ('ProSST-2048',):
        m_st = mod_style[plot_tab.loc[nm, 'modality']]
        ax.scatter(
            plot_tab.loc[nm, xcol], plot_tab.loc[nm, ycol],
            s=52, marker=m_st['marker'], facecolor=m_st['color'], edgecolor='#1b4f72', linewidth=0.9, zorder=6,
        )

    # ESM-MSR sigma trajectory: 0 -> 0.5 -> 1.0
    traj_keys = ('ESM-MSR (σ=0.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=1.0)')
    xs = [float(plot_tab.loc[k, xcol]) for k in traj_keys]
    ys = [float(plot_tab.loc[k, ycol]) for k in traj_keys]
    ax.plot(xs, ys, color='#6c2e9c', lw=1.9, ls=':', alpha=0.9, zorder=6)

    msr_styles = {
        'ESM-MSR (σ=0.0)': dict(color='#555555', marker='o', s=68, ec='#111111', lw=1.1, label='ESM3-small-open (σ = 0)'),
        'ESM-MSR (σ=0.5)': dict(color='#7a3eb1', marker='P', s=120, ec='#111111', lw=1.1, label='ESM-MSR σ = 0.5  [Pareto: 99.9%]'),
        'ESM-MSR (σ=1.0)': dict(color='#c0262d', marker='*', s=210, ec='#111111', lw=0.9, label='ESM-MSR σ = 1.0  [Pareto: 100%]'),
    }
    for nm, kw in msr_styles.items():
        xv, yv = float(plot_tab.loc[nm, xcol]), float(plot_tab.loc[nm, ycol])
        if (nm, xcol) in ci_map and (nm, ycol) in ci_map:
            xlo, xhi = ci_map[(nm, xcol)]
            ylo, yhi = ci_map[(nm, ycol)]
            ax.errorbar(
                xv, yv,
                xerr=[[xv - xlo], [xhi - xv]],
                yerr=[[yv - ylo], [yhi - yv]],
                fmt='none', ecolor=kw['color'], elinewidth=1.25, alpha=0.42, capsize=0, zorder=6,
            )
        ax.scatter(xv, yv, s=kw['s'], marker=kw['marker'], facecolor=kw['color'], edgecolor=kw['ec'], linewidth=kw['lw'], zorder=9, label=kw['label'])

    # Clean non-overlapping annotations
    label_specs = {
        'ESM-MSR (σ=1.0)': ('ESM-MSR σ = 1.0', (8, -14), 'bold', '#7a0c10'),
        'ESM-MSR (σ=0.5)': ('ESM-MSR σ = 0.5', (10, 6), 'bold', '#551a8b'),
        'ESM-MSR (σ=0.0)': ('σ = 0 (ESM3)', (-16, -16), 'bold', '#333333'),
        'VenusREM': ('VenusREM (MSA)', (8, 7), 'bold', '#8a4b00'),
        'S3F_MSA': ('S3F_MSA', (-12, 8), 'normal', '#444444'),
        'ProSST-2048': ('ProSST-2048 (top no-MSA)', (10, 2), 'normal', '#1b4f72'),
        'SPURS': ('SPURS', (-10, 6), 'normal', '#1b5e34'),
        'MutateEverything': ('MutateEverything', (12, -13), 'normal', '#1b5e34'),
        'ThermoMPNN': ('ThermoMPNN', (-10, -14), 'normal', '#1b5e34'),
    }
    for nm, (txt, (dx, dy), fw, col_t) in label_specs.items():
        if nm not in plot_tab.index:
            continue
        xv, yv = float(plot_tab.loc[nm, xcol]), float(plot_tab.loc[nm, ycol])
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(
            txt, (xv, yv), xytext=(dx, dy), textcoords='offset points',
            fontsize=8.2, fontweight=fw, color=col_t, ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.65, shrinkA=1, shrinkB=4) if (abs(dx) > 10 or abs(dy) > 12) else None,
            zorder=10,
        )

    ax.set_xlabel('Biophysical properties: mean Spearman ρ\n(Stability [n = 37] & Expression [n = 18])', fontsize=10.2)
    ax.set_ylabel('Functional properties: mean Spearman ρ\n(Activity [38], Binding [13], Org. fitness [70])', fontsize=10.2)
    ax.set_xlim(0.27, 0.765)
    ax.set_ylim(0.14, 0.515)
    ax.grid(color=style.GRID, lw=0.6, zorder=0)
    ax.legend(fontsize=7.6, loc='lower left', frameon=True, facecolor='white', edgecolor='#dddddd')

    style.finish_figure(
        fig,
        out / 'pgym_pareto_biophys_vs_func',
        title='ProteinGym multi-property Pareto frontier (176 unseen assays)',
        note=(
            'All 176 unseen ProteinGym assays under the four-part sequence + structure homology filter (41/217 excluded). '
            'Across all 95 public zero-shot models and all stability baselines, only three models are Pareto non-dominated: '
            'ESM-MSR σ = 1.0, ESM-MSR σ = 0.5, and VenusREM (MSA-based). Whiskers = 95% joint [assay × triplicate-seed] bootstrap (B = 10,000).'
        ),
        note_fontsize=7.5,
        top=0.92,
    )
    print(f'Saved {out / "pgym_pareto_biophys_vs_func.png"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
