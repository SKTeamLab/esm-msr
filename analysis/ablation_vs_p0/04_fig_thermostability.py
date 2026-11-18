#!/usr/bin/env python
"""Does any ablation beat FINAL_p0 on the thermostability benchmarks?

  thermostability_vs_p0   one panel per benchmark group (the 10 external single-mutant sets, PTMUL and PTMUL-D, and for reference Homology-Restricted MegaScale test), top: the WT head, bottom: the combined
                          prediction.  Large marker and bar = the group's composite (mean of the rho of its sets over one shared draw of proteins) delta rho = rho(family) - rho(p0) with its 95% bootstrap
                          interval and p; small grey dots = the individual sets.  A family beats p0 where its whole interval is above 0.

Also prints, over the set x family x head cells of the two thermostability groups, how many have an interval above 0 (and below).  Inputs: ``results/paired_vs_p0.csv``.

    python analysis/ablation_vs_p0/04_fig_thermostability.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import abl as A

PANELS = [('External single-mutant sets (10)', 'ext_singles', 'ext_singles_composite', 'all', 'composite of the 10 sets'),
          ('PTMUL and PTMUL-D', 'ext_multi', 'ext_multi_composite', 'all', 'composite of the 2 sets'),
          ('Homology-Restricted MegaScale', 'tsuboyama', A.C.LABEL['tsuboyama_capped_test'], 'all', 'test set (big), validation set (small)')]


def main():
    D = A.load_all()
    t = D[D.group.isin(['ext_singles', 'ext_multi']) & ~D.set.str.contains('composite')]
    nc, up, dn = len(t), int((t.lo > 0).sum()), int((t.hi < 0).sum())
    print(f'thermostability cells (set x family x head): {nc}; interval above 0: {up}; below 0: {dn}')
    print(t[t.lo > 0][['set', 'family', 'head', 'delta', 'lo', 'hi', 'p']].round(3).to_string(index=False))
    comp = D[D.set.isin(['ext_singles_composite', 'ext_multi_composite'])]
    print('composites with an interval above 0:', int((comp.lo > 0).sum()), 'of', len(comp))
    gap = 0.5
    ys, bands, y = {}, [], 0.0
    for gname in ['masking', 'loss', 'architecture', 'data']:
        fs = [f for f in A.ABL if A.GROUP[f] == gname]
        y0 = y
        for f in fs:
            ys[f] = y
            y += 1
        bands.append((gname, y0, y))
        y += gap
    with plt.rc_context(A.C.RC):
        fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), sharey=True)
        for j, (title, grp, name, sub, what) in enumerate(PANELS):
            lo_, hi_ = 0.0, 0.0
            for i, head in enumerate(A.HEADS):
                ax = axes[i, j]
                ax.grid(axis='y', visible=False)
                sel = D[(D.group == grp) & (D['head'] == head) & (D.subset == sub)]
                for k, (gname, a, b) in enumerate(bands):
                    if k % 2 == 0:
                        ax.axhspan(a - 0.5, b - 0.5, color='#000000', alpha=0.04, lw=0, zorder=0)
                others = sel[sel.set != name] if grp != 'tsuboyama' else D[(D.group == 'tsuboyama') & (D['head'] == head) & (D.subset == sub) & (D.set == A.C.LABEL['tsuboyama_capped_val'])]
                for r in others.itertuples():
                    ax.plot(r.delta, ys[r.family], 'o', ms=3.2, color='#9a9a9a', alpha=0.8, mew=0, zorder=2)
                for r in sel[sel.set == name].itertuples():
                    yy = ys[r.family]
                    ax.plot([r.lo, r.hi], [yy, yy], color=A.COLOR[r.family], lw=1.8, zorder=3, solid_capstyle='butt')
                    win = r.lo > 0
                    ax.plot(r.delta, yy, A.MARKER[r.family], ms=7, color=A.COLOR[r.family], mfc=A.COLOR[r.family] if (r.lo > 0 or r.hi < 0) else 'white', mew=1.4, zorder=4)
                    ax.text(1.0, yy, f'p={r.p:.2f}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7.4, color='#555555', fontweight='bold' if win else 'normal')
                    lo_, hi_ = min(lo_, r.lo), max(hi_, r.hi)
                ax.axvline(0, color=A.C.MUTED, lw=1, zorder=1)
                ax.set_ylim(y - gap - 0.5, -0.6)
                if j == 0:
                    ax.set_yticks([ys[f] for f in A.ABL])
                    ax.set_yticklabels([A.LABEL[f] for f in A.ABL], fontsize=8.6)
                    for tk, f in zip(ax.get_yticklabels(), A.ABL):
                        tk.set_color(A.COLOR[f])
                if i == 0:
                    ax.set_title(f'{title}\n', loc='left', fontsize=10)
                    ax.text(0, 1.02, what, transform=ax.transAxes, fontsize=7.8, color='#555555', va='bottom')
                ax.xaxis.set_major_locator(MaxNLocator(5))
                if i == 1:
                    ax.set_xlabel('delta rho vs p0')
            pad = 0.15 * (hi_ - lo_)
            for i in range(2):
                axes[i, j].set_xlim(lo_ - pad, hi_ + 0.5 * (hi_ - lo_) + 0.01)
        for i, head in enumerate(A.HEADS):
            axes[i, 0].set_ylabel('WT head' if head == 'wt' else 'combined prediction', fontsize=10, fontweight='bold', labelpad=130)
        A.C.finish(fig, 'Delta rho = rho(family) - rho(FINAL_p0), the same readout on the same variants. Big marker and bar: composite of the group (the mean of the rho of its sets over one shared draw of proteins) and its '
                   '95% bootstrap interval (100 resamples; seeds 1-3, full MegaScale 1-2); filled: the interval excludes 0; p: two-sided bootstrap p, not adjusted for multiple comparisons (smallest possible 0.01), bold when '
                   f'the interval is above 0. Small grey dots: the individual sets (Homology-Restricted MegaScale panel: validation set). Over the {nc} set x family x head cells of the 12 thermostability sets, {up} has '
                   f'an interval above 0 and {dn} have one below 0.', 'thermostability_vs_p0', out=A.OUT, pad=0.04)


if __name__ == '__main__':
    sys.exit(main())
