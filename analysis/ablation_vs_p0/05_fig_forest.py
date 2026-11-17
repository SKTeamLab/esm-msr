#!/usr/bin/env python
"""Every ablation against FINAL_p0 on every set, with the intervals: small multiples.

  ablation_forest_wt, ablation_forest_comb   one panel per ablation (columns), one row per set (the rows of the heatmap), delta rho = rho(family) - rho(p0) with its 95% bootstrap interval, for the WT head
                                             and for the combined prediction; filled marker when the interval excludes 0.

Inputs: ``results/paired_vs_p0.csv``, ``results/paired_vs_p0_pg.csv``.

    python analysis/ablation_vs_p0/05_fig_forest.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import abl as A

XLIM = 0.06
TITLE = {'FINAL_p1': 'p1\n(masked)', 'FINAL_p1_marg': 'p1 + MT\nmarginal', 'FINAL_p0_pw0': 'pw0\n(no pair loss)', 'FINAL_p0_norank': 'no rank\nloss', 'FINAL_p0_rcthr0': 'rank censor\nthreshold 0',
         'FINAL_p0_noseqhwm': 'no sequence\nhead', 'FINAL_p0_r2a4': 'MT LoRA\nrank 2', 'FINAL_p0_mega': 'full MegaScale\n(not homology-\nrestricted)'}


def one(D, head):
    groups = A.column_spec()
    rows, y, bands = [], 0.0, []
    for gname, specs in groups:
        y0 = y
        rows.append((y, None, gname))                                          # header row of the group
        y += 1
        for sp in specs:
            if head == 'wt' and sp[4] == 'epi':
                continue
            rows.append((y, sp, gname))
            y += 1
        bands.append((gname.replace('\n', ' '), y0, y))
        y += 0.7
    with plt.rc_context(A.C.RC):
        fig, axes = plt.subplots(1, len(A.ABL), figsize=(15.5, 0.235 * len(rows) + 2.6), sharey=True)
        for j, (ax, fam) in enumerate(zip(axes, A.ABL)):
            ax.grid(axis='y', visible=False)
            for k, (gname, a, b) in enumerate(bands):
                if k % 2 == 0:
                    ax.axhspan(a - 0.5, b - 0.5, color='#000000', alpha=0.04, lw=0, zorder=0)
            for yy, sp, gname in rows:
                if sp is None:
                    continue
                r = A.cell(D, sp, fam, head)
                if r is None:
                    continue
                sig = r.lo > 0 or r.hi < 0
                big = sp[3] in ('composite', 'average')
                ax.plot([r.lo, r.hi], [yy, yy], color=A.COLOR[fam], lw=1.5 if big else 1.0, zorder=3, solid_capstyle='butt')
                ax.plot(r.delta, yy, A.MARKER[fam], ms=5.6 if big else 3.9, color=A.COLOR[fam], mfc=A.COLOR[fam] if sig else 'white', mew=1.2, zorder=4)
            ax.axvline(0, color=A.C.MUTED, lw=1, zorder=1)
            ax.set_xlim(-XLIM, XLIM)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.set_title(TITLE[fam], fontsize=8.4, color=A.COLOR[fam], loc='left', fontweight='bold')
            ax.tick_params(axis='x', labelsize=8)
            if j == 0:
                ax.set_yticks([r[0] for r in rows])
                ax.set_yticklabels([r[2].replace('\n', ' ') if r[1] is None else r[1][3] for r in rows], fontsize=7.8)
                for tk, r in zip(ax.get_yticklabels(), rows):
                    if r[1] is None:
                        tk.set_fontweight('bold')
                        tk.set_fontsize(8.2)
        axes[0].set_ylim(y - 0.7, -0.8)
        axes[len(axes) // 2].set_xlabel('delta rho vs p0')
        A.C.finish(fig, f'{"WT head" if head == "wt" else "Combined prediction, 1/2 (MT + WT)"}: delta rho = rho(family) - rho(FINAL_p0) on the same variants, with the 95% bootstrap interval of the paired difference '
                   '(100 resamples; ProteinGym: 20000 resamples over assays, seed 1); filled marker: the interval excludes 0; the axis is cut at +-0.06. Rows as in the heatmap: composite = mean of the rho of the sets of '
                   'the group over one shared draw of proteins; ProteinGym = mean over the assays of the category; epistasis = rho of comb(AB) - WT(A) - WT(B) with the measured dddG (combined only). Seeds 1-3 (full '
                   'MegaScale: 1-2; ProteinGym: 1). Blank: no prediction for that family and set.', f'ablation_forest_{head}', out=A.OUT, pad=0.045)


def main():
    D = A.load_all()
    for head in A.HEADS:
        one(D, head)


if __name__ == '__main__':
    sys.exit(main())
