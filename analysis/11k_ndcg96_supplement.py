"""Supplementary figure: NDCG@96 of the Tsuboyama capped-test domains, singles against doubles, in the conventions of the main-text stacked figure (``11h_stacked_stab_epistatic``).

NDCG@96 here has relevance ddG - (the lowest ddG among the ranked variants of the domain) (``11l_epistatic_vs_additive_pvalues.ndcg96_shifted``): non-negative for every variant, so it does not depend on any variant
exceeding ddG = 0 and is defined for every domain (figure 1 of ``11_publication_figures`` uses the rectified relevance max(ddG, 0), which needs a stabilizing variant).  Singles and doubles are ranked separately, each
shifted by its own minimum; the overall dash ranks all variants of the domain together.  Every model is recomputed here (domains with at least ``MIN_N`` variants of the subset).

ESM-MSR is p0 only, drawn in black: the WT head on singles and the combined head on doubles (WT-view / Dual-view).  SPURS is the retrain with the trained adapter.  ESM3-small-open is the unmasked sigma-0 readout.  Names split
over two lines, angled tick labels, no title or footnotes.  NDCG@96 is the rectified-relevance NDCG (``11_publication_figures.ndcg96``) per domain, then averaged; intervals resample domains (the same draws as
``11h``).  The rows of the other models are read from the same domain draws as ``11h``.
The overall black dash (singles and doubles ranked together) is drawn only when it differs from the singles bar, as in figure 1.

Output ``results/00_supplementary_figs/ndcg96_singles_doubles.{png,pdf}``, 12.5 x 6.4 in.

    python analysis/11k_ndcg96_supplement.py
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.common import out_dir
from msr_analysis.data import _align_id

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pub', HERE / '11_publication_figures.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
spec = importlib.util.spec_from_file_location('l11', HERE / '11l_epistatic_vs_additive_pvalues.py')
l11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l11)
spec = importlib.util.spec_from_file_location('h11', HERE / '11h_stacked_stab_epistatic.py')
h11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h11)

OUT = out_dir('00_supplementary_figs')
H, W = 6.4, h11.SIDE


def main():
    fr, _ = pub.load()
    best, _ = pub.best_other(fr)
    pub.NAMES['best other ddG'] = pub.BEST_LABEL[best]
    sp = fr.spine
    ids = sp['id'].to_numpy(dtype=object)
    S, M = [], []
    for sd in (1, 2, 3):
        d = pd.read_csv(h11.UNF / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_single_additive', 'SPURS_multi'], low_memory=False)
        S.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_single_additive'].to_numpy()))
        M.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_multi'].to_numpy()))
    fr.preds['spursun'], fr.preds['spursunm'] = np.vstack(S), np.vstack(M)
    h11.p0_three_seeds(fr)

    keep = {'ESM-MSR (p0) combined': 'ESM-MSR\nWT-view (& Dual-view)'}
    cols = []
    for c in pub.main_columns(fr, best, 'unmasked'):
        if c['name'].startswith('ESM-MSR') and c['name'] not in keep:
            continue
        if c['name'] in keep:
            c = dict(c, color='#000000', single='msrp0:wt', double='msrp0:comb')
        if c['name'] == 'SPURS':
            c = dict(c, single='spursun', double='spursunm')
        if c['name'] == 'ESM3-small-open':
            c = dict(c, single='msrrel@0:wt')
        cols.append(dict(c, label=h11.LAB_A.get(c['name'], c['name'])))
    for c in cols:                                                      # the composite 'all' key of the models with two versions, from the keys set above
        if c['single'] != c['double']:
            n1 = (sp['n_mut'] == 1).to_numpy()
            c['all'] = f"{c['single']}|{c['double']}"
            fr.preds[c['all']] = np.where(n1[None, :], fr.preds[c['single']], fr.preds[c['double']])

    use, n_mut = sp['use'].to_numpy(), sp['n_mut'].to_numpy()
    single, double = use & (n_mut == 1), use & (n_mut == 2)
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=pub.B).astype(float)
    cells = {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}
    T = pub.table_grouped(fr, cols, cells, l11.ndcg96_shifted, pub.MIN_N, counts, libs)
    ov = pub.table_grouped(fr, cols, {'overall': (use, 'all')}, l11.ndcg96_shifted, pub.MIN_N, counts, libs)
    d1 = (ov.set_index('column').value - T[T.subset == 'singles'].set_index('column').value).dropna()
    d1 = d1[[c['name'] for c in cols if c['name'] in d1.index]]
    draw_overall = int((d1.abs() <= 0.01).sum()) < 0.9 * len(d1)
    print(f'overall NDCG@96 within 0.01 of the singles bar: {int((d1.abs() <= 0.01).sum())} of {len(d1)} columns -> {"drawn" if draw_overall else "not drawn"}')

    fig, ax = plt.subplots(figsize=(W, H))
    pub.bars(ax, cols, T, ['singles', 'doubles'], [pub.count_text(T, 'singles', 'singles'), pub.count_text(T, 'doubles', 'doubles')], f'NDCG@{pub.K} (mean over test domains)', (0, 1.12),
             overall=ov if draw_overall else None, overall_label=f'overall (n = {int(ov.n_variants.max()):,})')
    labs = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels(labs, rotation=30, ha='right', rotation_mode='anchor', fontsize=8.5)
    for t in ax.texts:
        if t.get_fontsize() < 8:
            t.set_fontsize(8)
    fig.tight_layout(pad=0.5)
    plt.rcParams['savefig.bbox'] = 'standard'
    fig.savefig(OUT / 'ndcg96_singles_doubles.png', dpi=300)
    fig.savefig(OUT / 'ndcg96_singles_doubles.pdf')
    pd.concat([T, ov]).pivot(index='column', columns='subset', values='value').round(3).to_csv(OUT / 'ndcg96_singles_doubles_values.csv')
    print('wrote', OUT / 'ndcg96_singles_doubles.png', fig.get_size_inches())


if __name__ == '__main__':
    main()
