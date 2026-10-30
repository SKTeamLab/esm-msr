#!/usr/bin/env python
"""Publication figure 5: the epistasis hierarchy, level by level, with domain-level bootstrap intervals (numbers: ``11d_epistasis_levels_bootstrap.py``).

``fig5a_epistasis_levels_bars_{test,val}``: three bars per model for the levels of the hierarchy from coarse to fine (pair rho, line rho, cell rank; light to dark), and two horizontal
dashes across the model's bars: naive rho (raw dddG, saturation included) and saturation-subtracted rho (the "beyond rho" of BENCHMARKS tables 2.0-2.2: each side minus E[dddG | its own
additive expectation]).  Bars carry 95% intervals from resampling domains; a dash has its interval as the whisker at its left (naive) or right (saturation-subtracted) end.

Models, in the three categories of the other publication figures: ESM-MSR devel as its epistasis head MT - WT and as its context control ~WT - WT (the WT adapter read on the mutated
sequence, asked about the reverse mutation: what the backbone and the single-mutant training already know about epistasis) | ME stage 2, ThermoMPNN-D, SPURS multi (retrained, three seeds) |
ESM3-small-open (independent-masking mutant pass - masked-marginal WT pass), ProteinMPNN | GeoDDG-3D, Rosetta.  ProSST has no interaction term; ESM3ΔG is left out (overlap with its
MGnify fine-tune); the Megascale-trained ESMtherm and IFUM stay in the heat maps of ``06_epistasis``.  Outputs: ``results/11_publication``.
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
OUT = out_dir('11_publication')

# (model as named in the bootstrap csv, category, colour, axis label)
MODELS = [('ESM-MSR devel · MT − WT', 'retrained', '#000000', 'ESM-MSR devel\nMT − WT'),
          ('ESM-MSR devel · ~WT − WT', 'retrained', '#7f7f7f', 'ESM-MSR devel\n~WT − WT\n(context control)'),
          ('ME stage 2', 'retrained', '#1b8a4b', 'ME\nstage 2'),
          ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'),
          ('SPURS multi', 'retrained', '#d35400', 'SPURS\nmulti'),
          ('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open'),
          ('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN'),
          ('GeoDDG-3D', 'supervised', '#117a65', 'GeoDDG-3D†'),
          ('Rosetta', 'supervised', '#1abc9c', 'Rosetta')]
BARS = [('pair_rho', 'pair ρ (coarsest)'), ('line_rho', 'line ρ'), ('cell_rank', 'cell rank (finest)')]
SHADE = [0.38, 0.68, 1.0]
C_NAIVE, C_SAT = '#e67e22', '#2e86c1'
TITLE = {'test': 'Test', 'val': 'Validation'}
FOOT = ['† trained on S8754 (ProThermDB + ThermoMutDB), not on Megascale. ProSST has no interaction term; ESM3ΔG is left out (overlap with its MGnify fine-tune).',
        'Epistasis = the double-mutant prediction of one model minus the additive reading of another or of another head: ESM-MSR MT − WT and ~WT − WT; ME stage 2, ThermoMPNN-D, SPURS multi minus the same seed\'s singles;',
        'ESM3-small-open independent-masking mutant pass minus masked-marginal WT pass; ProteinMPNN, GeoDDG-3D, Rosetta minus the sum of their singles. Seed means over the three seeds of the retrained models.',
        'Intervals: 95% bootstrap over domains (the same resample for every model; the epistasis suite recomputed on each); the whiskers on the dashes (orange left, blue right) are their intervals.']


def main():
    for split in ('test', 'val'):
        T = pd.read_csv(OUT / f'fig5_epistasis_levels_bootstrap_{split}.csv')
        nd, nn = int(T.n_domains.iloc[0]), int(T.n_doubles.iloc[0])
        cols = [dict(name=n, cat=c, color=col, label=lab) for n, c, col, lab in MODELS if (T.model == n).any()]
        xs, spans = pub.layout(cols)
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        w = 0.26
        for c, x in zip(cols, xs):
            t = T[T.model == c['name']].set_index('metric')
            for k, ((m, _), a) in enumerate(zip(BARS, SHADE)):
                xx = x + (k - 1) * w * 1.08
                r = t.loc[m]
                ax.bar(xx, r.value, w, color=c['color'], alpha=a, edgecolor='black', lw=0.7, zorder=3)
                ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor='#c0392b', elinewidth=1.2, capsize=2.2, zorder=5)
                ax.text(xx, max(r.hi, r.value, 0) + 0.012, f'{r.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
            x0, x1 = x - 1.5 * w * 1.08, x + 1.5 * w * 1.08
            for m, colr, xe in (('naive_rho', C_NAIVE, x0), ('beyond_rho', C_SAT, x1)):       # the interval of a dash is the whisker at its left (naive) or right (saturation-subtracted) end
                r = t.loc[m]
                ax.plot([x0, x1], [r.value] * 2, color=colr, lw=2.8, zorder=6, solid_capstyle='butt')
                ax.errorbar(xe, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor=colr, elinewidth=1.5, capsize=3, zorder=6.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([c['label'] for c in cols], fontsize=8.3)
        ax.set_ylabel('level metric (higher is better)')
        lo_, hi_ = float(np.nanmin(T.lo)), float(np.nanmax(T.hi))
        ax.set_ylim(min(-0.12, lo_ - 0.03), max(0.95, hi_ + 0.28))
        ax.set_yticks(np.arange(0, 0.91, 0.2))
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        for i, (cat, (x0, x1)) in enumerate(spans.items()):
            if i % 2:
                ax.axvspan(x0 - 0.85, x1 + 0.85, color='#000000', alpha=0.045, lw=0, zorder=0)
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, pub.CATS[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
        handles = [Patch(fc='#7f7f7f', alpha=a, ec='black', lw=0.7, label=lab) for a, (_, lab) in zip(SHADE, BARS)]
        handles += [Line2D([0], [0], color=C_NAIVE, lw=2.8, label='naive ρ (raw dddG, saturation included)'), Line2D([0], [0], color=C_SAT, lw=2.8, label='saturation-subtracted ρ')]
        ax.legend(handles=handles, loc='upper right', fontsize=8.2, ncol=5, framealpha=0.92, columnspacing=1.1, handlelength=1.5)
        pub.finish(fig, f'{TITLE[split]}: epistasis predicted level by level, {nn:,} doubles with both singles measured in {nd} domains', FOOT)
        style.save(fig, OUT / f'fig5a_epistasis_levels_bars_{split}')
        print(split, T.pivot(index='model', columns='metric', values='value').reindex([m[0] for m in MODELS]).round(3).to_string())


if __name__ == '__main__':
    main()
