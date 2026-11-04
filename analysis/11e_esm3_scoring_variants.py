#!/usr/bin/env python
"""Figures 3 and 2 with ESM3-small-open drawn three ways side by side (a one-off comparison: the other columns are those of the figure, the ESM3 columns need not line up with it).

  unmasked              WT head of the sigma = 0 checkpoint (additive) -> combined head (epistatic), no masking
  independent masking   WT pass, each mutated position masked on its own (additive) -> mean of that and the mutant pass (epistatic); the version figures 1-4 draw
  masked marginal       sum of the singles' marginals (additive) -> the double scored with all its mutated positions masked at once (epistatic)

Figure 2's version scores every variant (singles and doubles) with one readout per ESM3 column: the combined head unmasked, ``esm3avg`` (independent masking), ``esm3mm`` (masked
marginal); the three agree on singles up to the readout.  Domain-level bootstrap (mean over domains), draft sizes, as in ``11_publication_figures.py``.
Outputs: ``results/06_epistasis/fig3_esm3_scoring_variants`` and ``results/03_tsuboyama/fig2_esm3_scoring_variants``.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import data as D  # noqa: E402
from msr_analysis import style  # noqa: E402

pub = importlib.import_module('11_publication_figures')


def add_marginal_additive(fr):
    """``esm3mm+``: the masked-marginal run's own additive reading (sum of its singles) for every double."""
    p = D._esm3_file('marginal', pub.DS)
    d = pd.read_csv(p, usecols=['id', 'wt_lora_pred_additive'])
    fr.preds['esm3mm+'] = D._align_id(fr.spine['id'].to_numpy(dtype=object), d['id'].astype(str).to_numpy(), d['wt_lora_pred_additive'].to_numpy())[None, :]


def main():
    fr, _ = pub.load()
    add_marginal_additive(fr)
    sp = fr.spine
    use, n_mut = sp['use'].to_numpy(), sp['n_mut'].to_numpy()
    double = use & (n_mut == 2)
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=pub.B).astype(float)
    best, _ = pub.best_other(fr)
    cols = pub.pair_columns(fr, best)
    grey = ['#aab7b8', '#7f8c8d', '#4d5656']
    three = [pub.col('ESM3 unmasked', 'zeroshot', grey[0], 'ESM3-small-open\nunmasked\nWT → combined', 'msrrel@0:wt', 'msrrel@0:comb'),
             pub.col('ESM3 independent masking', 'zeroshot', grey[1], 'ESM3-small-open\nind. masking\nWT → combined', 'esm3ind+', 'esm3avg'),
             pub.col('ESM3 masked marginal', 'zeroshot', grey[2], 'ESM3-small-open\nmasked marginal\nsingles → double', 'esm3mm+', 'esm3mm')]
    i = [k for k, c in enumerate(cols) if c['name'] == 'ESM3-small-open'][0]
    cols = cols[:i] + three + cols[i + 1:]
    cells = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    T = pub.table_grouped(fr, cols, cells, pub.rho, pub.MIN_N, counts, libs)
    pub.save_tab(T, 'fig3_esm3_scoring_variants')
    fig, ax = plt.subplots(figsize=(1.2 * pub.fig_width(cols), 4.9))                  # wider than the other figures: room for the three ESM3 labels
    pub.bars(ax, cols, T, ['additive', 'epistatic'], [pub.count_text(T, 'additive', 'additive: singles model summed'), pub.count_text(T, 'epistatic', 'epistatic: model scores the double')],
             "Spearman's ρ, double mutants", (0, 1.12), tick_fs=7.8)
    pub.finish(fig, f'{pub.SPLIT} test doubles, additive version against epistatic version, ESM3-small-open scored three ways; 95% intervals from resampling domains',
               pub.foots(best=best))
    style.save(fig, pub.OUT / 'fig3_esm3_scoring_variants')
    print(T[T.column.str.startswith('ESM3')].round(3).to_string())

    # figure 2: stabilizing / destabilizing / overall over all variants
    cols2 = pub.main_columns(fr, best, 'average')
    three2 = [pub.col('ESM3 unmasked', 'zeroshot', grey[0], 'ESM3-small-open\nunmasked\n(combined)', 'msrrel@0:comb'),
              pub.col('ESM3 independent masking', 'zeroshot', grey[1], 'ESM3-small-open\nind. masking\n(WT, MT average)', 'esm3avg'),
              pub.col('ESM3 masked marginal', 'zeroshot', grey[2], 'ESM3-small-open\nmasked marginal', 'esm3mm')]
    i = [k for k, c in enumerate(cols2) if c['name'] == 'ESM3-small-open'][0]
    cols2 = cols2[:i] + three2 + cols2[i + 1:]
    y = sp['y'].to_numpy()
    cells2 = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}
    T2 = pub.table_grouped(fr, cols2, cells2, pub.rho, pub.MIN_N, counts, libs)
    ov2 = pub.table_grouped(fr, cols2, {'overall': (use, 'all')}, pub.rho, 3, counts, libs)
    pub.save_tab(pd.concat([T2, ov2]), 'fig2_esm3_scoring_variants')
    fig, ax = plt.subplots(figsize=(1.2 * pub.fig_width(cols2), 4.9))
    pub.bars(ax, cols2, T2, ['stabilizing', 'destabilizing'], [pub.count_text(T2, 'stabilizing', 'stabilizing, ddG > 0'), pub.count_text(T2, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ (mean over test domains)", (-0.2, 1.12), overall=ov2, overall_label=f"overall (n = {int(ov2.n_variants.max()):,})", zero_line=True, tick_fs=7.8)
    pub.finish(fig, f'{pub.SPLIT} test, all variants (singles and doubles): Spearman ρ of stabilizing against destabilizing variants, ESM3-small-open scored three ways; '
                    f'95% intervals from resampling domains', pub.foots(composite=True, best=best))
    style.save(fig, pub.OUT / 'fig2_esm3_scoring_variants')
    t = pd.concat([T2, ov2])
    print(t[t.column.str.startswith('ESM3')].round(3).to_string())


if __name__ == '__main__':
    main()
