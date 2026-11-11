#!/usr/bin/env python
"""Publication figure 5: the epistasis hierarchy, level by level, with domain-level bootstrap intervals (numbers: ``11d_epistasis_levels_bootstrap.py``).

``fig5a_epistasis_levels_bars_{test,val}``: five small square subplots (one per level: naive / raw rho, saturation-subtracted rho, pair rho, line rho, cell rank) for side-by-side
comparison of bars across models, with vertical x-axis labels and 95% intervals from resampling domains.

Models, in the three categories of the other publication figures: ESM-MSR (combined_AB - WT_AB, no linking) and ESM-MSR ctx ctrl (~WT - WT: the WT adapter read on the mutated sequence,
asked about the reverse mutation: what the backbone and the single-mutant training already know about epistasis) | ME stage 2, ThermoMPNN-D, SPURS multi (retrained, three seeds) |
ESM3-small-open (unmasked combined head - WT head), ProteinMPNN | GeoDDG-3D, Rosetta.  Outputs: ``results/06_epistasis``.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402

pub = importlib.import_module('11_publication_figures')
OUT = out_dir('06_epistasis')

# (model as named in the bootstrap csv, category, colour, axis label)
MODELS = [
    ('ESM-MSR (baseline) · combined_AB − WT_AB', 'retrained', '#7f8c8d', 'ESM-MSR (baseline)'),
    ('ESM-MSR · combined_AB − WT_AB', 'retrained', '#000000', 'ESM-MSR'),
    ('ESM-MSR (p0) · combined_AB − WT_AB', 'retrained', '#6c3483', 'ESM-MSR (p0)'),
    ('ESM-MSR · ~WT − WT', 'retrained', '#b2babb', 'ESM-MSR ctx ctrl'),
    ('ME stage 2', 'retrained', '#1b8a4b', 'ME stage 2'),
    ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'),
    ('SPURS multi', 'retrained', '#d35400', 'SPURS multi'),
    ('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open'),
    ('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN'),
    ('GeoDDG-3D', 'supervised', '#117a65', 'GeoDDG-3D'),
    ('Rosetta', 'supervised', '#1abc9c', 'Rosetta'),
]
LEVELS = [
    ('naive_rho', 'naive / raw ρ'),
    ('beyond_rho', 'saturation-subtracted ρ'),
    ('pair_rho', 'pair ρ (coarsest)'),
    ('line_rho', 'line ρ'),
    ('cell_rank', 'cell rank (finest)'),
]
LEVELS_CELL_MAG = [
    ('naive_rho', 'naive / raw ρ'),
    ('beyond_rho', 'saturation-subtracted ρ'),
    ('pair_rho', 'pair ρ (coarsest)'),
    ('line_rho', 'line ρ'),
    ('cell_mag_pooled', 'cell ρ (residual, finest)'),
]
LEVELS_CELL_COMPARE = [
    ('cell_rank', 'cell rank\n(col ranks of ddG_AB)'),
    ('cell_mag', 'cell ρ (block avg)\n(double-centred r)'),
    ('cell_mag_pooled', 'cell ρ (pooled)\n(double-centred r)'),
    ('cell_mag_gt0', 'cell ρ (dG_add > 0)\n(unsaturated cells)'),
]
SHORT_CATS = {
    'retrained': 'Retrained',
    'zeroshot': 'ZS',
    'supervised': 'Supervised',
}
BAR_W = 0.68
TITLE = {'test': 'Test', 'val': 'Validation'}
FOOT = [
    'Epistasis = double-mutant prediction minus additive singles: ESM-MSR = combined_AB − WT_AB (no linking), ESM-MSR ctx ctrl = ~WT − WT; ME stage 2, ThermoMPNN-D, SPURS multi minus',
    'the same seed\'s singles (3-seed means); ESM3-small-open unmasked combined − WT; ProteinMPNN, GeoDDG-3D, Rosetta minus sum of singles. Intervals: 95% bootstrap over domains.',
    'Levels (residuals r = dddG − E[dddG | x] and r\' = dddG\' − E[dddG\' | x\'] via 25-quantile-bin smoothers on additive dG expectations x, x\'): (1) naive / raw ρ: single Spearman ρ pooled over all doubles on raw dddG;',
    '(2) saturation-subtracted ρ: single Spearman ρ pooled over all doubles on residuals r\', r; (3) pair ρ: single Spearman ρ across position-pair matrices (≥20 doubles) of each pair\'s mean residual;',
    '(4) line ρ: single Spearman ρ pooled over all row/column lines (≥3 cells) of line-mean residuals minus their pair mean; (5) cell rank: computed per position-pair complete block (≥3×3) and averaged',
    'across blocks — within each block, column ranks of ddG_AB are double-centred (row, column and block means removed) and correlated by Spearman ρ (averaged over both row/column orientations).',
]
FOOT_CELL_MAG = [
    'Epistasis = double-mutant prediction minus additive singles: ESM-MSR = combined_AB − WT_AB (no linking), ESM-MSR ctx ctrl = ~WT − WT; ME stage 2, ThermoMPNN-D, SPURS multi minus',
    'the same seed\'s singles (3-seed means); ESM3-small-open unmasked combined − WT; ProteinMPNN, GeoDDG-3D, Rosetta minus sum of singles. Intervals: 95% bootstrap over domains.',
    'Horizontal line behind each bar = the same level metric restricted to doubles with additive expectation dG_additive = dG_wt + ddG_A + ddG_B > 0 kcal/mol (7,405 test, 5,500 val doubles).',
    'Sequential residual hierarchy (all 5 levels pooled; r = dddG − E[dddG | x] and r\' = dddG\' − E[dddG\' | x\'] via 25-quantile-bin smoothers): (1) naive / raw ρ: pooled over all doubles on raw dddG;',
    '(2) saturation-subtracted ρ: pooled over all doubles on residuals r\', r; (3) pair ρ: across position-pair matrices (≥20 doubles) of each pair\'s mean residual;',
    '(4) line ρ: pooled over all row/column lines (≥3 cells) of pair-centred line means; (5) cell ρ (residual): pooled across all position-pair complete blocks (≥3×3) of global+pair+line-corrected',
    'residuals (c = r − r_row − r_col + r_pair) correlated by Spearman ρ.',
]
FOOT_CELL_COMPARE = [
    'Comparison of cell-level (identity-specific) epistasis metrics on the same models and 95% domain-bootstrap resamples:',
    '(1) cell rank: double-centred within-column ranks of raw ddG_AB per complete block, averaged across blocks (saturation-invariant, but sensitive to single-mutant main-effect scale/spacing);',
    '(2) cell ρ (block avg, cell_mag): Spearman ρ of global+pair+line-corrected residuals (c = r − r_row − r_col + r_pair) per complete block, averaged across blocks;',
    '(3) cell ρ (pooled): single Spearman ρ pooled across all global+pair+line-corrected cells of all complete blocks;',
    '(4) cell ρ (dG_add > 0): per-pair Spearman ρ of double-centred residuals restricted to unsaturated doubles (measured dG_additive > 0 kcal/mol).',
]


def _layout_compact(cols, gap=1.25):
    xs, spans, x, prev = [], {}, 0.0, None
    for c in cols:
        if prev is not None and c['cat'] != prev:
            x += gap
        xs.append(x)
        lo, hi = spans.get(c['cat'], (x, x))
        spans[c['cat']] = (min(lo, x), max(hi, x))
        prev = c['cat']
        x += 1.0
    return xs, spans


def draw_version(T, split, levels=LEVELS, foot=FOOT, suffix='', title_desc='epistasis predicted level by level', overlay_sub0=False):
    nd, nn = int(T.n_domains.iloc[0]), int(T.n_doubles.iloc[0])
    cols = [
        dict(name=n, cat=c, color=col, label=lab)
        for n, c, col, lab in MODELS
        if (T.model == n).any()
    ]
    xs, spans = _layout_compact(cols)

    fig_w = 17.6 if len(levels) == 5 else 14.5
    fig, axes = plt.subplots(1, len(levels), figsize=(fig_w, 6.0), sharey=True)
    metric_names = [m for m, _ in levels]
    if overlay_sub0:
        metric_names += [f'{m}_sub0' for m, _ in levels]
    sub_T = T[T.metric.isin(metric_names) & T.model.isin([c['name'] for c in cols])]
    lo_, hi_ = float(np.nanmin(sub_T[['lo', 'value']].to_numpy())), float(np.nanmax(sub_T[['hi', 'value']].to_numpy()))
    ylim = (min(-0.12, lo_ - 0.04), max(0.52 if len(levels) == 4 else 0.90, hi_ + 0.14))
    top = 0.016 * (ylim[1] - ylim[0])
    dash_hw = 0.45

    for idx, (ax, (metric, level_title)) in enumerate(zip(axes, levels)):
        ax.set_box_aspect(1)
        if metric == 'naive_rho':
            r_ctrl = T[(T.model == '−dG_add') & (T.metric == 'naive_rho')]
            if len(r_ctrl) and np.isfinite(r_ctrl.iloc[0].value):
                ax.axhline(float(r_ctrl.iloc[0].value), color='#c0392b', ls=':', lw=1.8, zorder=2.6)
        for c, x in zip(cols, xs):
            r = T[(T.model == c['name']) & (T.metric == metric)].iloc[0]
            val_sub0 = np.nan
            if overlay_sub0:
                r_sub0 = T[(T.model == c['name']) & (T.metric == f'{metric}_sub0')]
                if len(r_sub0) and np.isfinite(r_sub0.iloc[0].value):
                    val_sub0 = float(r_sub0.iloc[0].value)
                    ax.plot([x - dash_hw, x + dash_hw], [val_sub0, val_sub0], color='#111111', lw=2.2, zorder=2.2, solid_capstyle='butt')
            ax.bar(x, r.value, BAR_W, color=c['color'], alpha=0.88 if overlay_sub0 else 0.92, edgecolor='black', lw=0.75, zorder=3)
            ax.errorbar(
                x,
                r.value,
                yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]],
                fmt='none',
                ecolor='#c0392b',
                elinewidth=1.25,
                capsize=2.3,
                zorder=5,
            )
            y_txt = max(
                r.hi,
                r.value,
                val_sub0 if np.isfinite(val_sub0) else 0.0,
                0.0,
            ) + top
            ax.text(
                x,
                y_txt,
                f'{r.value:.2f}',
                ha='center',
                va='bottom',
                fontsize=8.3,
                rotation=90,
                zorder=8,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.3),
            )
        if metric == 'naive_rho':
            handles = []
            if overlay_sub0:
                handles.append(plt.Line2D([0], [0], color='#111111', lw=2.2, label='dG_add > 0'))
            r_ctrl = T[(T.model == '−dG_add') & (T.metric == 'naive_rho')]
            if len(r_ctrl) and np.isfinite(r_ctrl.iloc[0].value):
                handles.append(plt.Line2D([0], [0], color='#c0392b', ls=':', lw=1.8, label='−dG_add'))
            if handles:
                ax.legend(
                    handles=handles,
                    loc='upper right',
                    fontsize=9.5,
                    framealpha=0.90,
                    handlelength=1.5,
                    borderpad=0.35,
                )

        ax.set_xticks(xs)
        ax.set_xticklabels([c['label'] for c in cols], fontsize=11.0, rotation=90, ha='center', va='top')
        ax.tick_params(axis='y', labelsize=11.5)
        if idx == 0:
            ax.set_ylabel('level metric (higher is better)', fontsize=13.0)
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(0, 0.45, 0.1) if len(levels) == 4 else np.arange(0, 0.81, 0.2))
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(xs[0] - 0.68, xs[-1] + 0.68)

        for i, (cat, (x0, x1)) in enumerate(spans.items()):
            if i % 2:
                ax.axvspan(x0 - 0.5 - 0.35, x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.38, x1 + 0.38], [1.015, 1.015], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text(
                (x0 + x1) / 2,
                1.028,
                SHORT_CATS[cat],
                transform=tr,
                ha='center',
                va='bottom',
                fontsize=8.6,
                fontweight='bold',
                color='#333333',
            )
        ax.set_title(level_title, fontsize=11.5 if '\n' in level_title else 13.0, fontweight='bold', pad=26)

    fig.suptitle(
        f'{TITLE[split]}: {title_desc}, {nn:,} doubles with both singles measured in {nd} domains',
        x=0.012,
        ha='left',
        fontsize=13.8,
        y=0.955 if len(levels) == 4 else 0.92,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    fig.text(0.012, -0.04, '\n'.join(foot), fontsize=9.2 if len(levels) == 4 else 9.5, va='top', ha='left', color='#444444')
    style.save(fig, OUT / f'fig5a_epistasis_levels_bars_{split}{suffix}')


def main():
    for split in ('test', 'val'):
        T = pd.read_csv(OUT / f'fig5_epistasis_levels_bootstrap_{split}.csv')
        draw_version(T, split, levels=LEVELS, foot=FOOT, suffix='')
        if (T.metric == 'cell_mag_pooled').any():
            draw_version(T, split, levels=LEVELS_CELL_MAG, foot=FOOT_CELL_MAG, suffix='_cell_mag', overlay_sub0=True)
            draw_version(T, split, levels=LEVELS_CELL_COMPARE, foot=FOOT_CELL_COMPARE, suffix='_cell_compare', title_desc='cell-level epistasis formulations compared')
        order = [m[0] for m in MODELS if (T.model == m[0]).any()]
        cols_order = [m for m in ('naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank', 'cell_mag', 'cell_mag_pooled', 'cell_mag_gt0') if m in T.metric.values]
        print(split, T.pivot(index='model', columns='metric', values='value').reindex(index=order, columns=cols_order).round(3).to_string())


if __name__ == '__main__':
    main()
