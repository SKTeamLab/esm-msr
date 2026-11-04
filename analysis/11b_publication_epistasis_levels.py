#!/usr/bin/env python
"""Publication figure 5: the epistasis hierarchy, level by level, with domain-level bootstrap intervals (numbers: ``11d_epistasis_levels_bootstrap.py``).

``fig5a_epistasis_levels_bars_{test,val}``: three bars per model for the levels of the hierarchy from coarse to fine (pair rho, line rho, cell rank; light to dark), and a black horizontal
dash behind them for saturation-subtracted rho (the "beyond rho" of BENCHMARKS tables 2.0-2.2: each side minus E[dddG | its own additive expectation]).  Bars carry 95% intervals from
resampling domains; the dash sticks out to the right of the bars, with its interval on one side of the protruding end and its value on top (the style of the overall dash in figure 2).
Naive rho (raw dddG, saturation included) is in ``fig5_epistasis_levels_bootstrap_*.csv``.

Models, in the three categories of the other publication figures: ESM-MSR as its epistasis head MT - WT (latent) and as its context control ~WT - WT (the WT adapter read on the mutated
sequence, asked about the reverse mutation: what the backbone and the single-mutant training already know about epistasis) | ME stage 2, ThermoMPNN-D, SPURS multi (retrained, three seeds) |
ESM3-small-open (unmasked combined head - WT head), ProteinMPNN | GeoDDG-3D, Rosetta.  Outputs: ``results/06_epistasis``.
"""
import importlib
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

pub = importlib.import_module('11_publication_figures')       # CATS, layout(), finish()
OUT = out_dir('06_epistasis')

# (model as named in the bootstrap csv, category, colour, axis label)
MODELS = [('ESM-MSR · MT − WT', 'retrained', '#000000', 'ESM-MSR\nMT − WT'),
          ('ESM-MSR · ~WT − WT', 'retrained', '#7f7f7f', 'ESM-MSR\n~WT − WT\n(context control)'),
          ('ME stage 2', 'retrained', '#1b8a4b', 'ME\nstage 2'),
          ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'),
          ('SPURS multi', 'retrained', '#d35400', 'SPURS\nmulti'),
          ('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open'),
          ('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN'),
          ('GeoDDG-3D', 'supervised', '#117a65', 'GeoDDG-3D'),
          ('Rosetta', 'supervised', '#1abc9c', 'Rosetta')]
BARS = [('pair_rho', 'pair ρ (coarsest)'), ('line_rho', 'line ρ'), ('cell_rank', 'cell rank (finest)')]
SHADE = [0.38, 0.68, 1.0]
W3 = 0.20                                                      # three bars in the extent of figure 2's pair (-0.42 to 0.22): the same group width, dash and interval position
CENTRES = (-0.32, -0.10, 0.12)
TITLE = {'test': 'Test', 'val': 'Validation'}
FOOT = ['GeoDDG-3D is trained on S8754 (ProThermDB + ThermoMutDB), not on Megascale.',
        'Epistasis = the double-mutant prediction of one model minus the additive reading of another or of another head: ESM-MSR MT − WT and ~WT − WT; ME stage 2, ThermoMPNN-D, SPURS multi minus the same seed\'s singles;',
        'ESM3-small-open unmasked combined head minus its WT head; ProteinMPNN, GeoDDG-3D, Rosetta minus the sum of their singles. Seed means over the three seeds of the retrained models.',
        'Intervals: 95% bootstrap over domains (the same resample for every model; the epistasis suite recomputed on each); the black dash is the saturation-subtracted ρ, with its interval on the protruding end.']


def main():
    for split in ('test', 'val'):
        T = pd.read_csv(OUT / f'fig5_epistasis_levels_bootstrap_{split}.csv')
        nd, nn = int(T.n_domains.iloc[0]), int(T.n_doubles.iloc[0])
        cols = [dict(name=n, cat=c, color=col, label=lab) for n, c, col, lab in MODELS if (T.model == n).any()]
        xs, spans = pub.layout(cols)
        fig, ax = plt.subplots(figsize=(pub.fig_width(cols), 4.9))
        w = W3
        lo_, hi_ = float(np.nanmin(T.lo)), float(np.nanmax(T.hi))
        ylim = (min(-0.12, lo_ - 0.03), max(0.95, hi_ + 0.28))
        top = 0.012 * (ylim[1] - ylim[0])                                                    # the same label offset as figure 2
        for c, x in zip(cols, xs):
            t = T[T.model == c['name']].set_index('metric')
            for k, ((m, _), a) in enumerate(zip(BARS, SHADE)):
                xx = x + CENTRES[k]
                r = t.loc[m]
                ax.bar(xx, r.value, w, color=c['color'], alpha=a, edgecolor='black', lw=0.8, zorder=3)
                ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.5, zorder=5)
                ax.text(xx, max(r.hi, r.value, 0) + top, f'{r.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
            r = t.loc['beyond_rho']                                                         # the dash runs behind the bars and sticks out on the right
            ax.plot([x + pub.DASH[0], x + pub.DASH[1]], [r.value] * 2, color='black', lw=1.8, zorder=2.2, solid_capstyle='butt')
            ax.errorbar(x + pub.X_ERR, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor='black', elinewidth=1.3, capsize=2.5, zorder=5)
            ax.text(x + pub.X_ERR, max(r.hi, r.value, 0) + top, f'{r.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, fontweight='bold',
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
        ax.set_xticks(xs)
        ax.set_xticklabels([c['label'] for c in cols], fontsize=pub.TICK_FS)
        ax.set_ylabel('level metric (higher is better)')
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(0, 0.91, 0.2))
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(xs[0] - pub.XPAD[0], xs[-1] + pub.XPAD[1])
        for i, (cat, (x0, x1)) in enumerate(spans.items()):
            if i % 2:
                ax.axvspan(x0 - 0.5 - 0.35, x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, pub.CATS[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
        handles = [Patch(fc='#7f7f7f', alpha=a, ec='black', lw=0.7, label=lab) for a, (_, lab) in zip(SHADE, BARS)]
        handles += [Line2D([0], [0], color='black', lw=1.8, label='saturation-subtracted ρ')]
        ax.legend(handles=handles, ncol=len(handles), **pub.LEGEND_KW)
        pub.finish(fig, f'{TITLE[split]}: epistasis predicted level by level, {nn:,} doubles with both singles measured in {nd} domains', FOOT)
        style.save(fig, OUT / f'fig5a_epistasis_levels_bars_{split}')
        print(split, T.pivot(index='model', columns='metric', values='value').reindex([m[0] for m in MODELS]).round(3).to_string())


if __name__ == '__main__':
    main()
