#!/usr/bin/env python
"""Summary figures: one panel that puts the benchmarks with many assays side by side.

``pg_categories``   ProteinGym by category (187 of 213 assays), PG avg, DMS mean, the 458 low-homology Domainome domains and the MGnify scaffolds: mean Spearman rho of the nine unified models (seed 1) per
                    column.  ProteinGym and Domainome come from ``results/05_proteingym/pg_summary.csv`` (``05_proteingym.py``); the MGnify column is ESM-MSR p0 alone (the only model run on MGnify), the mean per-scaffold rho of
                    the combined prediction over the scaffolds with at least 10 measured variants, three seeds, scaffolds with a strong homolog or the same protein in the capped training set removed, from
                    ``results/18_mgnify/mgnify_summary.csv`` (``18_mgnify.py``).

Needs ``05_proteingym.py`` and ``18_mgnify.py`` first; without the MGnify table the column is left out.

    python analysis/01_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.external_s1 import EXT_ORDER, register_ext_styles  # noqa: E402
from msr_analysis.registry import MODELS, RESULTS  # noqa: E402

OUT = out_dir('01_summary')
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']
CAT_LABEL = {
    'Activity': 'Activity\n(43)',
    'Binding': 'Binding\n(13)',
    'Expression': 'Expression\n(18)',
    'OrganismalFitness': 'Org. fitness\n(73)',
    'Stability': 'Stability*\n(40/66)',
    'PG avg': 'PG avg*',
    'DMS mean': 'DMS mean*\n(187/213)',
    'Domainome': 'Domainome*\n(458/522)',
}
PG_COLS = CATS + ['PG avg', 'DMS mean']


def mgnify_column():
    """(value, lo, hi, n scaffolds) of ESM-MSR p0 on MGnify, or None when ``18_mgnify.py`` has not run."""
    f = RESULTS / '18_mgnify' / 'mgnify_summary.csv'
    if not f.exists():
        return None
    S = pd.read_csv(f)
    t = S[(S.dataset == 'MGnify') & (S.stratum == 'lowid') & (S['subset'] == 'overall') & (S.unit == 'per scaffold') & (S['head'] == 'comb') & (S.form == 'latent')]
    if not len(t):
        return None
    r = t.iloc[0]
    return float(r.rho), float(r.lo), float(r.hi), int(r.n_scaffolds)


def main():
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    R = pd.read_csv(RESULTS / '05_proteingym' / 'pg_summary.csv')
    note = (RESULTS / '05_proteingym' / 'pg_categories_note.txt').read_text().strip()
    cat_cols = PG_COLS + ['Domainome']
    table = {(r.column, r.model): r.value for r in R.itertuples()}
    cis = {(r.column, r.model): (r.lo, r.hi) for r in R.itertuples()}
    mg = mgnify_column()
    groups = cat_cols + (['MGnify'] if mg else [])
    labels = [CAT_LABEL[c] for c in cat_cols] + ([f'MGnify*\n({mg[3]:,})'] if mg else [])

    fig, ax = plt.subplots(figsize=(15.6, 5.9))
    style.dot_panel(ax, groups, EXT_ORDER, table, cis, highlight=['ESM-MSR'], ylabel='Spearman ρ (mean over assays / domains / scaffolds)', xlabels=labels, rot=0, size=6.2, spread=0.76, ylim=(0.12, 0.96))
    ax.axvline(6.5, color='#999999', lw=0.95, ls='--', zorder=2)
    if mg:
        ax.axvline(7.5, color='#999999', lw=0.95, ls='--', zorder=2)
        kw = style.marker_kwargs('ESM-MSR', size=6.2 * 1.35)
        x = len(groups) - 1
        ax.errorbar([x], [mg[0]], yerr=[[mg[0] - mg[1]], [mg[2] - mg[0]]], fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=4)
        ax.plot([x], [mg[0]], zorder=5, **kw)
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in EXT_ORDER]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.942), ncol=5, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.35)
    fig.suptitle('ProteinGym (187 DMS assays), Domainome (458 low-homology domains) & MGnify (%s scaffolds): Spearman ρ across models (p0, seed 1; whiskers: 95%% assay / domain / scaffold bootstrap)'
                 % (f'{mg[3]:,}' if mg else '–'), x=0.012, ha='left', y=0.985, fontsize=10.0, fontweight='bold')
    if mg:
        note += ('\nMGnify*: ESM-MSR p0 only (the other models were not run on it); combined prediction, mean per-scaffold ρ over the %s scaffolds with at least 10 measured variants, mean of seeds 1-3; '
                 'scaffolds with a strong homolog (H) or the same protein in the capped training set removed (MMseqs2); whiskers resample scaffolds and seeds.' % f'{mg[3]:,}')
    fig.text(0.012, 0.008, note, fontsize=7.6, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0.0, 0.085 if mg else 0.065, 1.0, 0.84))
    style.save(fig, OUT / 'pg_categories')
    print('wrote', OUT / 'pg_categories', '| MGnify column:', mg)


if __name__ == '__main__':
    main()
