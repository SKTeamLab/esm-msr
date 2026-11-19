#!/usr/bin/env python
"""Every FINAL_* ablation against FINAL_p0 on every benchmark: heatmaps of the paired delta rho = rho(family) - rho(p0).

  ablation_heatmap   rows = the eight ablations (grouped by what they change), columns = the sets in six groups (thermostability single- and multi-mutant sets with their composites, Homology-Restricted
                     MegaScale, Domainome, the DMS benchmark, ProteinGym categories); top: the WT head, bottom: the combined prediction (the epistasis column is the combined readout only).  The number
                     in a cell is delta rho x 1000; bold with an outline when the 95% interval excludes 0; hatched when the family has no prediction for it (no combined run, or no run on the set).

Inputs: ``results/paired_vs_p0.csv`` (``01_paired_vs_p0.py``) and ``results/paired_vs_p0_pg.csv`` (``02_pg_vs_p0.py``).

    python analysis/ablation_vs_p0/03_fig_heatmap.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import abl as A

VLIM = 0.04
CMAP = plt.get_cmap('RdBu')


def draw(ax, D, spec_groups, head, show_header):
    norm = Normalize(-VLIM, VLIM)
    gap, x = 0.6, 0.0
    ticks, labs, spans = [], [], []
    for gname, specs in spec_groups:
        x0 = x
        for sp in specs:
            for i, fam in enumerate(A.ABL):
                r = A.cell(D, sp, fam, head)
                y = i
                if r is None or (head == 'wt' and sp[4] == 'epi'):
                    ax.add_patch(Rectangle((x, y), 1, 1, fc='#f4f4f2', ec='white', lw=1.2, hatch='////', zorder=1))
                    ax.add_patch(Rectangle((x, y), 1, 1, fc='none', ec='#cfcfcb', lw=0, hatch='////', zorder=1))
                    continue
                sig = (r.lo > 0) or (r.hi < 0)
                ax.add_patch(Rectangle((x, y), 1, 1, fc=CMAP(norm(float(np.clip(r.delta, -VLIM, VLIM)))), ec='white', lw=1.2, zorder=1))
                if sig:
                    ax.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, fc='none', ec='#111111', lw=1.4, zorder=3))
                ax.text(x + 0.5, y + 0.52, f'{r.delta * 1000:+.0f}', ha='center', va='center', fontsize=7.6, fontweight='bold' if sig else 'normal', color='white' if abs(r.delta) > 0.027 else ('#111111' if sig else '#555555'), zorder=4)
            ticks.append(x + 0.5)
            labs.append(sp[3])
            x += 1
        spans.append((gname, x0, x))
        x += gap
    ax.set_xlim(-0.2, x - gap + 0.2)
    ax.set_ylim(len(A.ABL), -0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=62, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_yticks(np.arange(len(A.ABL)) + 0.5)
    ax.set_yticklabels([A.LABEL[f] for f in A.ABL], fontsize=8.6)
    for t, f in zip(ax.get_yticklabels(), A.ABL):
        t.set_color(A.COLOR[f])
    ax.grid(False)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    if show_header:
        for gname, x0, x1 in spans:
            ax.text((x0 + x1) / 2, -0.4, gname, ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#333333', linespacing=1.05)
            ax.plot([x0, x1], [-0.2, -0.2], color='#444444', lw=1.2, clip_on=False)
    ax.set_title('WT head' if head == 'wt' else 'Combined prediction, 1/2 (MT + WT)', loc='left', fontsize=10, fontweight='bold', pad=36 if show_header else 6)
    return spans


def main():
    D = A.load_all()
    groups = A.column_spec()
    ncol = sum(len(s) for _, s in groups)
    with plt.rc_context(A.C.RC):
        fig = plt.figure(figsize=(0.5 * ncol + 3.2, 10.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.018], wspace=0.03, left=0.175, right=0.955, top=0.905, bottom=0.19, hspace=0.42)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        for ax, head in zip(axes, A.HEADS):
            draw(ax, D, groups, head, head == 'wt')
        sm = cm.ScalarMappable(norm=Normalize(-VLIM * 1000, VLIM * 1000), cmap=CMAP)
        cb = fig.colorbar(sm, cax=fig.add_subplot(gs[0, 1]), orientation='vertical')
        cb.ax.set_title('delta rho\nx 1000', fontsize=8.2, pad=6)
        cb.ax.tick_params(labelsize=8)
        A.C.finish(fig, 'Cell = delta rho x 1000 = rho(family) - rho(FINAL_p0) under the same readout on the same variants (+12 = +0.012; blue: better than p0, red: worse). Bold with an outline: the 95% bootstrap interval '
                   'of the paired difference excludes 0 (100 resamples; proteins for the external sets, PTMUL and the per-protein means of Homology-Restricted MegaScale and Domainome, sites for the DMS sets; '
                   'ProteinGym: 20000 resamples over assays, seeds 1-3 averaged per assay, mean of the assays of the category). composite = mean of the rho of the sets of the group over one shared draw of proteins. Seeds 1-3 '
                   '(full MegaScale: 1-2; ProteinGym: 1). Epistasis = rho of comb(AB) - WT(A) - WT(B) with the measured dddG. Hatched: no prediction (no combined run, or the family is not scored on the set).',
                   'ablation_heatmap', out=A.OUT, tight=False)


if __name__ == '__main__':
    sys.exit(main())
