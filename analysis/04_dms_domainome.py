#!/usr/bin/env python
"""Deep-mutational scans and Domainome: unified main models, seed 1, mutation-level bootstrap.

Evaluates the unified model roster across the 7 external DMS datasets, their mean, and Domainome
using seed 1 only and 95% mutation-level bootstrap intervals (within-domain mutation resampling
for Domainome):
- Singles-only assays (ESTA dTm, Domainome): singles models (ESM-MSR p0 WT head, MutateEverything st1,
  ThermoMPNN, SPURS, ESM3-small-open WT head) plus ProteinMPNN, ProSST, GeoDDG-3D.
- Multi-mutant assays (DLG4 x2, GRB2 x2, MYO, GB1): multi models (ESM-MSR p0 combined,
  MutateEverything st2, ThermoMPNN-D on 2-mutant sets / additive ThermoMPNN on GB1 with 3-4-mutants,
  SPURS multi, ESM3-small-open combined) plus ProteinMPNN, ProSST, GeoDDG-3D.
- Excludes sigma in {0.75, 0.5, 0.25} sweep points (not computed for p0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import ci  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.external_s1 import EXT_ORDER, load_s1_mutation_boot, register_ext_styles  # noqa: E402
from msr_analysis.registry import DMS7, LABEL, MODELS  # noqa: E402

OUT = out_dir('04_dms_domainome')
DS = DMS7 + ['domainome']
COLS = DMS7 + ['mean of 7 DMS', 'domainome']
NOTE = (
    'Seed 1 only; 95% mutation-level bootstrap (within-domain mutations for Domainome). '
    'Singles models on ESTA dTm and Domainome; multi models on DLG4/GRB2/MYO/GB1 '
    '(ThermoMPNN-D on 2-mutant sets, additive ThermoMPNN on GB1).'
)


def build_table(models, P, reps, keys_used):
    """Build table[(col, model)], cis[(col, model)], and DataFrame rows across COLS."""
    t, c, rows = {}, {}, []
    for m in models:
        for d in DS:
            if (d, m) not in P:
                continue
            v = P[(d, m)]
            t[(d, m)] = v[0]
            c[(d, m)] = (v[1], v[2])
            rows.append(dict(column=d, model=m, key=keys_used.get((d, m), ''), rho=v[0], lo=v[1], hi=v[2]))
        if all((d, m) in P for d in DMS7):
            pt = float(np.mean([P[(d, m)][0] for d in DMS7]))
            b_min = min(len(reps[d][m]) for d in DMS7)
            r = np.mean([reps[d][m][:b_min] for d in DMS7], axis=0)
            lo, hi = ci(r)
            t[('mean of 7 DMS', m)] = pt
            c[('mean of 7 DMS', m)] = (lo, hi)
            rows.append(dict(column='mean of 7 DMS', model=m, key='wt|comb', rho=pt, lo=lo, hi=hi))
    return t, c, pd.DataFrame(rows)


def _draw_top_legend(fig, models, bbox=(0.5, 0.94), ncol=4):
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in models]
    return fig.legend(
        handles=hs, loc='upper center', bbox_to_anchor=bbox, ncol=ncol, fontsize=8.0,
        frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.2, handletextpad=0.35,
    )


def main(refresh=False):
    register_ext_styles()
    boot = load_s1_mutation_boot(refresh=refresh)
    P, reps, keys_used = boot['point'], boot['reps'], boot['keys']

    active_models = [m for m in EXT_ORDER if all((d, m) in P for d in DS)]

    # ---- figure 1: unified all-model comparison across DMS7 + mean + Domainome -------------------------
    t_all, c_all, df_all = build_table(active_models, P, reps, keys_used)
    df_all.to_csv(OUT / 'dms_vs_retrains.csv', index=False, float_format='%.4f')
    df_all.to_csv(OUT / 'dms_external_rho.csv', index=False, float_format='%.4f')

    for stem in ('dms_vs_retrains', 'dms_external_rho'):
        fig, ax = plt.subplots(figsize=(13.5, 5.8))
        style.dot_panel(
            ax, COLS, active_models, t_all, c_all,
            highlight=['ESM-MSR'], ylabel='Spearman ρ',
            xlabels=[LABEL.get(x, x) for x in COLS],
            size=6.2, spread=0.78, ylim=(0.0, 0.90),
        )
        _draw_top_legend(fig, active_models, bbox=(0.5, 0.942), ncol=4)
        fig.suptitle(
            'External DMS & Domainome: Spearman ρ across all models (p0, seed 1; 95% mutation-level bootstrap)',
            x=0.012, ha='left', y=0.985, fontsize=10.2, fontweight='bold',
        )
        fig.text(0.012, 0.01, NOTE, fontsize=8.0, va='bottom', ha='left', color='#444444')
        fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.84))
        style.save(fig, OUT / stem)

    # ---- figure 2: ESM-MSR (p0, sigma = 1) vs zero-shot models (no sigma in {0.75, 0.5, 0.25}) ----------
    zs_models = ['ESM-MSR', 'ESM3-small-open', 'ProteinMPNN', 'ProSST']
    t_zs, c_zs, df_zs = build_table(zs_models, P, reps, keys_used)
    df_zs.to_csv(OUT / 'dms_sigma_vs_zero_shot.csv', index=False, float_format='%.4f')

    fig3, ax3 = plt.subplots(figsize=(12.5, 5.2))
    style.dot_panel(
        ax3, COLS, zs_models, t_zs, c_zs,
        highlight=['ESM-MSR'], ylabel='Spearman ρ',
        xlabels=[LABEL.get(x, x) for x in COLS],
        size=6.4, spread=0.68, ylim=(0.0, 0.90),
    )
    _draw_top_legend(fig3, zs_models, bbox=(0.5, 0.942), ncol=4)
    fig3.suptitle(
        'ESM-MSR (p0, seed 1, σ = 1) vs zero-shot models (σ = 0 and baselines; 95% mutation-level bootstrap)',
        x=0.012, ha='left', y=0.985, fontsize=10.0, fontweight='bold',
    )
    fig3.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    style.save(fig3, OUT / 'dms_sigma_vs_zero_shot')

    # ---- figure 3: sigma = 0 -> 1 comparison (no intermediate sigma values for p0) ----------------------
    fig4, axes4 = plt.subplots(1, 8, figsize=(15.5, 3.0), sharey=False)
    for ax, col in zip(axes4, DS):
        v0 = P[(col, 'ESM3-small-open')]
        v1 = P[(col, 'ESM-MSR')]
        xs = [0.0, 1.0]
        ys = [v0[0], v1[0]]
        los = [v0[1], v1[1]]
        his = [v0[2], v1[2]]
        ax.plot(xs, ys, '-o', color='k', ms=5.0, lw=1.5)
        ax.errorbar(xs, ys, yerr=[[ys[0] - los[0], ys[1] - los[1]], [his[0] - ys[0], his[1] - ys[1]]],
                    fmt='none', ecolor='k', elinewidth=1.1, capsize=2.5)
        ax.set_title(LABEL[col], fontsize=9)
        ax.set_xlabel('LoRA scale σ')
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(['0', '1'])
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis='y', color=style.GRID, lw=0.6)
    axes4[0].set_ylabel('Spearman ρ')
    fig4.suptitle('ESM-MSR (p0, seed 1) vs σ = 0 zero-shot control (95% mutation-level bootstrap)', fontsize=10, y=1.02)
    fig4.tight_layout()
    style.save(fig4, OUT / 'dms_sigma_sweep')

    print(df_all.pivot(index='model', columns='column', values='rho').reindex(active_models)[COLS].round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='Recompute seed-1 mutation bootstrap cache')
    args = ap.parse_args()
    main(refresh=args.refresh)
