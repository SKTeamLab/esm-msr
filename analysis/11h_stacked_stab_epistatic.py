"""Figures 2 and 3 of 11_publication_figures stacked in one square (1:1) figure.

Top: Spearman rho of stabilizing against destabilizing variants with the overall rho as the black dash (fig 2, singles and doubles together).
Bottom: Spearman rho on double mutants, the additive version of each model against its epistatic version (fig 3).
Both panels are drawn from the tables 11_publication_figures wrote (``results/03_tsuboyama/fig2_spearman_stab_destab.csv`` and
``results/06_epistasis/fig3_spearman_doubles_additive_vs_epistatic.csv``; 95% intervals from resampling domains), so run that script first.
Output: ``results/00_main_text_figs/fig2_fig3_stacked.{png,pdf}``, 12.5 x 12.5 in; the panels are lettered b and c (a is drawn elsewhere).
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.data import _align_id
from msr_analysis.registry import BASE

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pub', HERE / '11_publication_figures.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

UNF = BASE / 'retrained' / 'spurs_unfrozen' / 'predictions'      # SPURS retrained with the trained (unfrozen) adapter on the capped split, seeds 1-3

LAB_A = {'ESM-MSR (p0) combined': 'ESM-MSR\nWT-view (& Dual-view)', 'MutateEverything': 'MutateEverything\n(Stage 1 & Stage 2)', 'ThermoMPNN': 'ThermoMPNN\n(& ThermoMPNN-D)', 'SPURS': 'SPURS\n(& SPURS-multi)',
         'ESM3-small-open': 'ESM3-small-open\nWT-view (& Dual-view)', 'best other ddG': 'GeoDDG-3D'}
LAB_B = {'ESM-MSR (p0)': 'ESM-MSR\nWT-view → Dual-view', 'MutateEverything': 'MutateEverything\nStage 1 → Stage 2', 'ThermoMPNN': 'ThermoMPNN\n→ ThermoMPNN-D', 'SPURS': 'SPURS\n→ SPURS-multi',
         'ESM3-small-open': 'ESM3-small-open\nWT-view → Dual-view', 'ProteinMPNN': 'ProteinMPNN\nOne-shot singles → AR doubles', 'ProSST': 'ProSST\n(additive only)',
         'best other ddG': 'GeoDDG-3D\nAdd. singles → sim. double', 'Rosetta': 'Rosetta\nAdd. singles → sim. double'}

from msr_analysis.common import out_dir

OUT = out_dir('00_main_text_figs')
SIDE = 12.5                                                      # inches; the width of figure 1, so the bars keep its scale


def main():
    fr, _ = pub.load()
    best, _ = pub.best_other(fr)
    pub.NAMES['best other ddG'] = pub.BEST_LABEL[best]
    sp = fr.spine
    ids = sp['id'].to_numpy(dtype=object)
    S, M = [], []
    for sd in (1, 2, 3):
        d = pd.read_csv(UNF / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_single_additive', 'SPURS_multi'], low_memory=False)
        S.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_single_additive'].to_numpy()))
        M.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_multi'].to_numpy()))
    fr.preds['spursun'], fr.preds['spursunm'] = np.vstack(S), np.vstack(M)
    use = sp['use'].to_numpy()
    print(f'unfrozen SPURS: {np.isfinite(fr.preds["spursun"][:, use]).mean():.4f} of the scored rows covered (frozen: {np.isfinite(fr.preds["spurs"][:, use]).mean():.4f})')
    # ESM-MSR is the p0 model only, drawn in black: the combined head on all variants (a), WT_A+B -> comb_AB on doubles (b); the other ESM-MSR versions stay in the tables
    keep = {'ESM-MSR (p0) combined': 'ESM-MSR\nWT-view (& Dual-view)', 'ESM-MSR (p0)': 'ESM-MSR'}
    pick = lambda cols: [dict(c, color='#000000', label=keep[c['name']] if c['name'] == 'ESM-MSR (p0) combined' else c['label'].replace('(p0)\n', '')) if c['name'] in keep else c
                         for c in cols if not c['name'].startswith('ESM-MSR') or c['name'] in keep]
    def relabel(cols, lab, top=False):
        out = []
        for c in pick(cols):
            if c['name'] == 'SPURS':
                c = dict(c, single='spursun', double='spursunm')                                  # the unfrozen retrain, not the frozen-adapter one
            if top and c['name'] == 'ESM3-small-open':
                c = dict(c, single='msrrel@0:wt')                                                 # sigma 0, unmasked: WT head on singles, combined head on doubles
            if c['name'] == 'ESM-MSR (p0) combined':
                c = dict(c, single='msrp0:wt', double='msrp0:comb')                               # WT head on singles, combined head on multi-mutants (deployed readout)
            out.append(dict(c, label=lab.get(c['name'], c['name'])))
        return out
    cols2 = relabel(pub.main_columns(fr, best, 'unmasked'), LAB_A, top=True)
    cols3 = relabel(pub.pair_columns(fr, best), LAB_B)
    t2 = pd.read_csv(pub.OUT_TSU / 'fig2_spearman_stab_destab.csv')
    t3 = pd.read_csv(pub.OUT_EPI / 'fig3_spearman_doubles_additive_vs_epistatic.csv')
    # the stored tables hold the frozen-adapter SPURS: redraw its rows from the unfrozen retrain with the same bootstrap draws as 11_publication_figures
    n_mut, y = sp['n_mut'].to_numpy(), sp['y'].to_numpy()
    single, double = use & (n_mut == 1), use & (n_mut == 2)
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=pub.B).astype(float)
    sc2 = [c for c in cols2 if c['name'] in ('SPURS', 'ESM-MSR (p0) combined', 'ESM3-small-open')]
    sc3 = [c for c in cols3 if c['name'] == 'SPURS']
    new2 = pd.concat([pub.table_grouped(fr, sc2, {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}, pub.rho, pub.MIN_N, counts, libs),
                      pub.table_grouped(fr, sc2, {'overall': (use, 'all')}, pub.rho, 3, counts, libs)])
    new3 = pub.table_grouped(fr, sc3, {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}, pub.rho, pub.MIN_N, counts, libs)
    t2 = pd.concat([t2[~t2.column.isin([c['name'] for c in sc2])], new2])
    t3 = pd.concat([t3[t3.column != 'SPURS'], new3])
    T2, ov2 = t2[t2.subset != 'overall'], t2[t2.subset == 'overall']

    fig, (a, b) = plt.subplots(2, 1, figsize=(SIDE, SIDE))
    pub.bars(a, cols2, T2, ['stabilizing', 'destabilizing'], [pub.count_text(T2, 'stabilizing', 'stabilizing, ddG > 0'), pub.count_text(T2, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ (mean over test domains)", (-0.2, 1.12), overall=ov2, overall_label=f'overall (n = {int(ov2.n_variants.max()):,})', zero_line=True, cats=pub.CATS)
    pub.bars(b, cols3, t3, ['additive', 'epistatic'], [pub.count_text(t3, 'additive', 'additive: singles model summed'), pub.count_text(t3, 'epistatic', 'epistatic: model scores the double')],
             "Spearman's ρ, double mutants", (0, 1.12))
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
