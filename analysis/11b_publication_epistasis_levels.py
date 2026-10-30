#!/usr/bin/env python
"""Publication figure 5: the epistasis hierarchy, level by level (BENCHMARKS tables 2.1 / 2.2, the document's numbers: the metric suite lives with the training code).

Two designs of the same numbers, for the validation and the test doubles:

  A  ``fig5a_epistasis_levels_bars_*``     three bars per model, the hierarchy's levels from coarse to fine (pair ρ, line ρ, cell rank; light to dark), and two horizontal
                                           dashes across the model's bars: naive ρ (raw dddG, saturation included) and saturation-subtracted ρ (the "beyond ρ" of the tables,
                                           each side minus E[dddG | its own additive expectation]).
  B  ``fig5b_epistasis_levels_panels_*``   one dot panel per metric (naive, saturation-subtracted, pair, line, cell rank), models on the rows in the same three categories.

Models are the ones of figures 1-4 that the tables hold: ESM-MSR (baseline triplicate and devel), ME stage 2, ThermoMPNN-D, SPURS multi | ProteinMPNN | GeoDDG-3D, Rosetta.
ESM3-small-open and ProSST are not in the tables; ESM3ΔG is left out (overlap with its MGnify fine-tune); the Megascale-trained ESMtherm and IFUM stay in the heat maps of
``06_epistasis``.  Error bars: standard deviation over the three seeds where the table has them (the ESM-MSR rows and SPURS multi); the other rows are single checkpoints or
public models and carry none, and a protein-level bootstrap of these metrics needs the suite itself.  Outputs: ``results/11_publication`` (also ``fig5_epistasis_levels_*.csv``).
"""
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.registry import REFERENCE  # noqa: E402
from msr_analysis.tables import clean_label, numeric, read_table  # noqa: E402

pub = importlib.import_module('11_publication_figures')       # CATS, layout(), the footnotes
OUT = out_dir('11_publication')

# (table row, category, colour, axis label)
MODELS = [('ESM-MSR (baseline)', 'retrained', '#000000', 'ESM-MSR\nbaseline'),
          ('ESM-MSR (devel) (3 seeds)', 'retrained', '#666666', 'ESM-MSR\ndevel'),
          ('MutateEverything stage 2', 'retrained', '#1b8a4b', 'ME\nstage 2'),
          ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'),
          ('SPURS multi (retrained, 3 seeds)', 'retrained', '#d35400', 'SPURS\nmulti'),
          ('ProteinMPNN (zero-shot)', 'zeroshot', '#34495e', 'ProteinMPNN'),
          ('GeoDDG-3D (FASPR)', 'supervised', '#117a65', 'GeoDDG-3D†'),
          ('Rosetta Cartesian ΔΔG', 'supervised', '#1abc9c', 'Rosetta')]
METRICS = [('naive ρ', 'naive ρ'), ('beyond ρ', 'saturation-subtracted ρ'), ('pair ρ', 'pair ρ'), ('line ρ', 'line ρ'), ('cell_rank', 'cell rank')]
BARS = ['pair ρ', 'line ρ', 'cell_rank']
SHADE = [0.38, 0.68, 1.0]
C_NAIVE, C_SAT = '#e67e22', '#2e86c1'
SETS = {'test': ('2.2.1', '2.2 ', 'Test', '12,455 doubles with both singles measured, 43 position pairs'),
        'val': ('2.1.1', '2.1 ', 'Validation', '8,494 doubles with both singles measured, 24 position pairs')}
FOOT = ['† trained on S8754 (ProThermDB + ThermoMutDB), not on Megascale. ESM3-small-open and ProSST are not in the epistasis tables; ESM3ΔG is left out (overlap with its MGnify fine-tune).',
        'Error bars: sd over 3 seeds where the table has seeds (ESM-MSR rows, SPURS multi); other rows are single checkpoints or public models. Numbers are the benchmark document\'s (tables 2.1 / 2.2).']


def load(split):
    tid, prefix, _, _ = SETS[split]
    d = read_table(tid)
    d.iloc[:, 0] = [clean_label(x) for x in d.iloc[:, 0]]
    V, _ = numeric(d)
    sd = pd.read_csv(REFERENCE / 'seed_sd_by_table_cell.csv')
    sd = sd[sd.table.str.startswith(prefix)].copy()
    sd['row'] = [re.sub(r'[ᴹᵐᴾᴴʰᴰᶜᵁ‡#*†\s]+$', '', clean_label(r)) for r in sd['row']]
    S = sd.pivot_table(index='row', columns='column', values='sd_n_minus_1', aggfunc='first')
    rows = []
    for name, cat, color, label in MODELS:
        for m, _ in METRICS:
            rows.append(dict(model=name, category=cat, metric=m, value=V.loc[name, m], seed_sd=S.loc[name, m] if name in S.index and m in S.columns else np.nan))
    return pd.DataFrame(rows)


def cols_of():
    return [dict(name=n, cat=c, color=col, label=lab) for n, c, col, lab in MODELS]


def design_a(T, split):
    cols = cols_of()
    xs, spans = pub.layout(cols)
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    w = 0.26
    for c, x in zip(cols, xs):
        t = T[T.model == c['name']].set_index('metric')
        for j, (m, a) in enumerate(zip(BARS, SHADE)):
            xx = x + (j - 1) * w * 1.08
            v, s = t.loc[m, 'value'], t.loc[m, 'seed_sd']
            ax.bar(xx, v, w, color=c['color'], alpha=a, edgecolor='black', lw=0.7, zorder=3)
            if np.isfinite(s):
                ax.errorbar(xx, v, yerr=s, fmt='none', ecolor='#c0392b', elinewidth=1.2, capsize=2.2, zorder=5)
            top = max(v, 0) + (s if np.isfinite(s) else 0)
            ax.text(xx, top + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.6))
        for m, colr in (('naive ρ', C_NAIVE), ('beyond ρ', C_SAT)):
            ax.plot([x - 1.5 * w * 1.08, x + 1.5 * w * 1.08], [t.loc[m, 'value']] * 2, color=colr, lw=2.8, zorder=6, solid_capstyle='butt')
    ax.set_xticks(xs)
    ax.set_xticklabels([c['label'] for c in cols], fontsize=8.3)
    ax.set_ylabel('level metric (higher is better)')
    ax.set_ylim(-0.1, 0.92)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.axhline(0, color='black', lw=0.8, zorder=2.5)
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2:
            ax.axvspan(x0 - 0.85, x1 + 0.85, color='#000000', alpha=0.045, lw=0, zorder=0)
        tr = ax.get_xaxis_transform()
        ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
        ax.text((x0 + x1) / 2, 1.02, pub.CATS[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
    handles = [Patch(fc='#7f7f7f', alpha=a, ec='black', lw=0.7, label=lab) for a, lab in zip(SHADE, ('pair ρ (coarsest)', 'line ρ', 'cell rank (finest)'))]
    handles += [Line2D([0], [0], color=C_NAIVE, lw=2.8, label='naive ρ (raw dddG, saturation included)'), Line2D([0], [0], color=C_SAT, lw=2.8, label='saturation-subtracted ρ')]
    ax.legend(handles=handles, loc='upper right', fontsize=8.2, ncol=5, framealpha=0.92, columnspacing=1.1, handlelength=1.5)
    pub.finish(fig, f'{SETS[split][2]}: epistasis predicted level by level, {SETS[split][3]}', FOOT)
    style.save(fig, OUT / f'fig5a_epistasis_levels_bars_{split}')


def design_b(T, split):
    cols = cols_of()
    n = len(cols)
    ys = []
    y, prev = 0.0, None
    for c in cols:
        if prev is not None and c['cat'] != prev:
            y += 0.6
        ys.append(y)
        prev = c['cat']
        y += 1.0
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.0, 4.6), sharey=True)
    for ax, (m, title) in zip(axes, METRICS):
        for c, yy in zip(cols, ys):
            r = T[(T.model == c['name']) & (T.metric == m)].iloc[0]
            ax.plot([0, r.value], [yy, yy], color=c['color'], lw=1.0, alpha=0.55, zorder=2)
            ax.scatter([r.value], [yy], s=46, color=c['color'], zorder=4, edgecolor='black', linewidth=0.5)
            if np.isfinite(r.seed_sd):
                ax.errorbar(r.value, yy, xerr=r.seed_sd, fmt='none', ecolor='#c0392b', elinewidth=1.2, capsize=2.2, zorder=5)
            ax.text(max(r.value, 0) + (r.seed_sd if np.isfinite(r.seed_sd) else 0) + 0.02, yy, f'{r.value:.2f}', va='center', fontsize=7.2, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
        for cat in dict.fromkeys(c['cat'] for c in cols):
            yy = [y_ for c, y_ in zip(cols, ys) if c['cat'] == cat]
            if list(pub.CATS).index(cat) % 2:
                ax.axhspan(min(yy) - 0.5, max(yy) + 0.5, color='#000000', alpha=0.045, lw=0, zorder=0)
        ax.axvline(0, color='black', lw=0.8, zorder=1)
        ax.set_xlim(-0.12, 0.95)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, fontsize=9.5, color=C_NAIVE if m == 'naive ρ' else (C_SAT if m == 'beyond ρ' else 'black'))
        ax.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c['label'].replace('\n', ' ') for c in cols], fontsize=8.5)
    axes[0].set_ylim(ys[-1] + 0.7, ys[0] - 1.0)
    for cat in dict.fromkeys(c['cat'] for c in cols):
        yy = [y_ for c, y_ in zip(cols, ys) if c['cat'] == cat]
        axes[0].text(-0.115, min(yy) - 0.58, {'retrained': 'Retrained', 'zeroshot': 'Zero-shot', 'supervised': 'Supervised, not Megascale'}[cat], ha='left', va='center', fontsize=8.2, fontweight='bold', color='#333333')
    pub.finish(fig, f'{SETS[split][2]}: epistasis predicted level by level, {SETS[split][3]}', FOOT, top=0.93)
    style.save(fig, OUT / f'fig5b_epistasis_levels_panels_{split}')


def main():
    for split in SETS:
        T = load(split)
        T.assign(split=split).to_csv(OUT / f'fig5_epistasis_levels_{split}.csv', index=False, float_format='%.4f')
        design_a(T, split)
        design_b(T, split)
        print(split, '\n', T.pivot(index='model', columns='metric', values='value').reindex([m[0] for m in MODELS]).round(3).to_string())


if __name__ == '__main__':
    main()
