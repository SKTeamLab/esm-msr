"""Figures 2 and 3 of 11_publication_figures stacked in one square (1:1) figure.

Top: Spearman rho of stabilizing against destabilizing variants with the overall rho as the black dash (fig 2, singles and doubles together).
Bottom: Spearman rho on double mutants, the additive version of each model against its epistatic version (fig 3).
Both panels are drawn from the tables 11_publication_figures wrote (``results/03_tsuboyama/fig2_spearman_stab_destab.csv`` and
``results/06_epistasis/fig3_spearman_doubles_additive_vs_epistatic.csv``; 95% intervals from resampling domains), so run that script first.
Panel c carries a bracket over the additive / epistatic pair of every model with the one-sided p that the epistatic version beats the additive one (paired domain bootstrap, ``results/06_epistasis/epistatic_vs_additive_pvalues.csv``
from ``11l_epistatic_vs_additive_pvalues.py``, run it first).
Output: ``results/00_main_text_figs/fig2_fig3_stacked.{png,pdf}``, 12.5 x 12.5 in; the panels are lettered b and c (a is drawn elsewhere).
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.common import out_dir

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pub', HERE / '11_publication_figures.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

OUT = out_dir('00_main_text_figs')
EPI_P = out_dir('06_epistasis') / 'epistatic_vs_additive_pvalues.csv'    # 11l_epistatic_vs_additive_pvalues.py: one-sided p that the epistatic version beats the additive one
P_NAME = {'best other ddG': 'GeoDDG-3D'}                         # a column's name in that table when it differs from the column name
SIDE = 12.5                                                      # inches; the width of figure 1, so the bars keep its scale


def fmt_p(p):
    return 'p < 1e-5' if p <= 1.5e-5 else f'p = {p:.1e}' if p < 1e-3 else f'p = {p:.3f}'


def pair_brackets(ax, cols, T, subsets=('additive', 'epistatic')):
    """A bracket over the two bars of every model that has both, labelled with the one-sided p that the epistatic version beats the additive one (``EPI_P``); the bracket spans the two bars."""
    if not EPI_P.exists():
        print(f'no {EPI_P}: brackets not drawn (run 11l_epistatic_vs_additive_pvalues.py)')
        return
    P = pd.read_csv(EPI_P).set_index('model')['p_one']
    cols = [c for c in cols if c.get('draw', True)]
    xs, _ = pub.layout(cols)
    top = 0.0
    for c, x in zip(cols, xs):
        r = [T[(T.column == c['name']) & (T.subset == sub)] for sub in subsets]
        name = P_NAME.get(c['name'], c['name'])
        if name not in P.index or not all(len(q) for q in r):
            continue
        hi = max(float(q.iloc[0].hi) for q in r)
        y = hi + 0.115                                               # above the rotated value labels of the bars
        x0, x1 = x + pub.PAIR_CENTRED[0], x + pub.PAIR_CENTRED[1]
        h = 0.014
        ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color='#333333', lw=1.0, solid_capstyle='butt', clip_on=False, zorder=9)
        ax.text((x0 + x1) / 2, y + 0.008, fmt_p(float(P[name])), ha='center', va='bottom', fontsize=8, color='#333333', zorder=9)
        top = max(top, y + 0.075)
    return top


def main():
    fr, _ = pub.load()
    best, _ = pub.best_other(fr)
    pub.NAMES['best other ddG'] = pub.BEST_LABEL[best]
    # the columns of figures 2 and 3 of 11_publication_figures (``msr_analysis.models``: ESM-MSR scored by its WT-view on singles and its Dual-view on multi-mutants, the same colours and order as every figure)
    cols2 = pub.main_columns(fr, best, 'unmasked')
    cols3 = pub.pair_columns(fr, best)
    t2 = pd.read_csv(pub.OUT_TSU / 'fig2_spearman_stab_destab.csv')
    t3 = pd.read_csv(pub.OUT_EPI / 'fig3_spearman_doubles_additive_vs_epistatic.csv')
    T2, ov2 = t2[t2.subset != 'overall'], t2[t2.subset == 'overall']

    fig, (a, b) = plt.subplots(2, 1, figsize=(SIDE, SIDE))
    pub.bars(a, cols2, T2, ['stabilizing', 'destabilizing'], [pub.count_text(T2, 'stabilizing', 'stabilizing, ddG > 0'), pub.count_text(T2, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ (mean over test domains)", (-0.2, 1.12), overall=ov2, overall_label=f'overall (n = {int(ov2.n_variants.max()):,})', zero_line=True, cats=pub.CATS)
    pub.bars(b, cols3, t3, ['additive', 'epistatic'], [pub.count_text(t3, 'additive', 'additive: singles model summed'), pub.count_text(t3, 'epistatic', 'epistatic: model scores the double')],
             "Spearman's ρ, double mutants", (0, 1.12))
    top = pair_brackets(b, cols3, t3)
    if top and top > 1.12:                                           # make room for the bracket labels; the category headers sit above the axes
        b.set_ylim(0, top)
    for ax, s in ((a, 'b'), (b, 'c')):                           # subplot a of the main-text figure is drawn elsewhere
        labs = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticklabels(labs, rotation=30, ha='right', rotation_mode='anchor', fontsize=8.5)         # angled so neighbouring names never touch
        ax.text(-0.045, 1.0, s, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')
    for ax in (a, b):                                            # raise the smallest text (the value labels are 6.8 pt in bars()) to 8 pt
        for t in ax.texts:
            if t.get_fontsize() < 8:
                t.set_fontsize(8)
    fig.tight_layout(pad=0.5, h_pad=1.2)                         # no title or footnotes: the panels fill the square
    plt.rcParams['savefig.bbox'] = 'standard'                    # no tight cropping: the page stays exactly SIDE x SIDE
    fig.savefig(OUT / 'fig2_fig3_stacked.png', dpi=300)
    fig.savefig(OUT / 'fig2_fig3_stacked.pdf')
    print('wrote', OUT / 'fig2_fig3_stacked.png', fig.get_size_inches())


if __name__ == '__main__':
    main()
