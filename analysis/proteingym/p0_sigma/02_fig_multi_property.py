#!/usr/bin/env python
"""Proteins with more than one distinct property: rho vs sigma per assay (the look of the earlier sigma analysis, F12), re-rendered with FINAL_p0 and sigma = 0, 0.5, 1.

Index (``results/multi_property_index.csv``): the ProteinGym proteins whose assays measure different properties (group A of the hand annotation in ``../sigma_tradeoff/scripts/common.py``:
abundance, activity, binding, growth, drug resistance), plus DLG4 and GRB2, whose ProteinGym assay is the abundance-type Faure set (its scores equal those of the abundance sets of the earlier DMS benchmark)
and whose binding set comes from the DMS benchmark.  One panel per protein, one line per assay coloured by property; sigma = 1, 0.5: FINAL_p0 WT head, mean over seeds 1-3; sigma = 0: ESM3-small-open unmasked;
band = 95% bootstrap over variants (rows resampled jointly across the sigmas and seeds, 1000 replicates).

    python analysis/proteingym/p0_sigma/02_fig_multi_property.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata

import pgsigma as P
sys.path.insert(0, str(P.HERE.parent / 'sigma_tradeoff' / 'scripts'))
import common as earlier  # noqa: E402
from msr_analysis import style  # noqa: E402

B = 1000
EXTRA = [('DLG4_HUMAN', 'DLG4_HUMAN_Faure_2021', 'ABUND', 'abundance PCA', 'DLG4_HUMAN_Faure_2021_binding_domain'),
         ('GRB2_HUMAN', 'GRB2_HUMAN_Faure_2021', 'ABUND', 'abundance PCA', 'GRB2_HUMAN_Faure_2021_binding_domain')]
ORDER = {'ABUND': 0, 'BIND': 1, 'ACT': 2, 'FIT': 3, 'SPEC': 4}


def tag(dms):
    parts = dms.split('_')
    i = next(k for k, p in enumerate(parts) if p.isdigit() and len(p) == 4)
    return f'{parts[i - 1]} {parts[i]}'


def index():
    rows = [dict(protein=prot, assay=dms, source='ProteinGym', axis=axis, note=note) for dms, (prot, grp, axis, note) in earlier.ANNOT.items() if grp == 'A']
    for prot, pg, axis, note, ds in EXTRA:
        rows.append(dict(protein=prot, assay=pg, source='ProteinGym', axis=axis, note=note))
        rows.append(dict(protein=prot, assay=ds, source='DMS benchmark', axis='BIND', note='binding PCA'))
    t = pd.DataFrame(rows)
    t['label'] = [f"{a.axis.lower()[:5]}: {tag(a.assay.replace('_binding_domain', '_x')) if a.source != 'ProteinGym' else tag(a.assay)}" + (' (DMS benchmark)' if a.source != 'ProteinGym' else '') for a in t.itertuples()]
    t['order'] = t['axis'].map(ORDER)
    return t.sort_values(['protein', 'order', 'assay']).drop(columns='order').reset_index(drop=True)


def bootstrap(name, rng):
    """Bootstrap over the variants of one assay (rows resampled jointly across the sigmas and the FINAL_p0 seeds); the statistic is the mean over the seeds of the rho (sigma 0 is one run)."""
    ds = [d for d in (P.load_assay(name, k) for k in P.SEEDS) if d is not None]
    ds = [d for d in ds if len(d) == len(ds[0])]                       # every seed scores the same variants in the same order; a seed that does not is left out
    y = ds[0]['y'].to_numpy(float)
    Xs = [np.column_stack([d[f's{s}'].to_numpy(float) for s in P.SIGMAS]) for d in ds]
    n = len(y)
    out = np.empty((B, len(P.SIGMAS)))
    for b in range(B):
        i = rng.integers(0, n, n)
        ry = rankdata(y[i])
        out[b] = np.mean([[np.corrcoef(rankdata(X[i, k]), ry)[0, 1] for k in range(len(P.SIGMAS))] for X in Xs], axis=0)
    return out


def main():
    t = index()
    t = t[~t.assay.isin(P.train_overlap())].reset_index(drop=True)          # assays that overlap the training data are always left out
    rho = pd.read_csv(P.OUT / 'per_assay_rho.csv').set_index('assay')
    for s in P.SIGMAS:
        t[f'rho_{s}'] = t.assay.map(rho[f'rho_{s}'])
    t['n'] = t.assay.map(rho['n'])
    t.to_csv(P.OUT / 'multi_property_index.csv', index=False, float_format='%.4f')
    print(f'{t.protein.nunique()} proteins, {len(t)} assays')
    rng = np.random.default_rng(1)
    boots = {a: bootstrap(a, rng) for a in t.assay}
    prots = sorted(t.protein.unique())
    nc = 4
    nr = int(np.ceil(len(prots) / nc))
    with plt.rc_context(P.RC):
        fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.0 * nr), squeeze=False)
        for ax, prot in zip(axs.ravel(), prots):
            sub = t[t.protein == prot].reset_index(drop=True)
            ys = sub['rho_1.0'].to_numpy(float)
            gap = 0.07 * (sub[[f'rho_{s}' for s in P.SIGMAS]].to_numpy().max() - sub[[f'rho_{s}' for s in P.SIGMAS]].to_numpy().min())
            ypos = {}
            for i in np.argsort(ys):                                    # push the end labels apart
                y = ys[i]
                for j in sorted(ypos.values()):
                    if abs(y - j) < gap:
                        y = j + gap
                ypos[i] = y
            for k, r in sub.iterrows():
                col = P.AXIS_COLOR[r.axis]
                lo, hi = np.percentile(boots[r.assay], [2.5, 97.5], axis=0)
                ax.fill_between(P.SIGMAS, lo, hi, color=col, alpha=0.13, lw=0)
                ax.plot(P.SIGMAS, [r[f'rho_{s}'] for s in P.SIGMAS], marker='o', color=col, mec=P.SURFACE, mew=1.2, ms=5.5)
                ax.text(1.03, ypos[k], r.label, fontsize=6.5, color=P.INK2, va='center')
            ax.set_title(prot.replace('_', ' '), fontsize=9.5)
            ax.set_xticks(P.SIGMAS)
            ax.set_xticklabels(['0', '0.5', '1'])
            ax.set_xlim(-0.05, 1.02)
        for ax in axs.ravel()[len(prots):]:
            ax.axis('off')
        for ax in axs[:, 0]:
            ax.set_ylabel('Spearman rho')
        for ax in axs.ravel():
            if ax.axison and ax.get_subplotspec().is_last_row() or (ax.axison and not axs[-1, ax.get_subplotspec().colspan.start].axison):
                ax.set_xlabel('sigma')
        h = [Line2D([], [], color=P.AXIS_COLOR[k], lw=2, marker='o', label=P.AXIS_NAME[k]) for k in ('ABUND', 'ACT', 'BIND', 'FIT', 'SPEC') if k in set(t.axis)]
        fig.legend(handles=h, loc='lower center', ncol=len(h), fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle('Proteins with assays of different properties: rho vs sigma per assay (p0: sigma 1, 0.5; sigma 0: ESM3-small-open, unmasked; band = 95% bootstrap over variants)',
                     x=0.01, ha='left', fontsize=11.5, fontweight='bold')
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        bad = style.text_outside(fig)
        if bad:
            print('WARNING: text outside the figure box:', bad)
        style.save(fig, P.OUT / 'multi_property_rho_vs_sigma')


if __name__ == '__main__':
    main()
