#!/usr/bin/env python
"""Combined versus WT head across the benchmarks: the overview and the dependence on the number of mutations.

  overview_forest     delta rho = rho(combined) - rho(WT head) per benchmark set, for p0, p1 and pw0 (circle, diamond, square), with the 95% bootstrap interval; a filled marker when the interval excludes
                      0.  Groups: stability / abundance single mutants, stability multi-mutants, the DMS benchmark sets (all variants), ProteinGym by category (mean over assays).  At the right the
                      p0 rho of the WT head and of the combined prediction.
  by_mutation_count   delta rho against the number of mutations of the variant, one panel per set with multi-mutants (p0, p1, pw0), and the rho of the predicted epistasis comb(AB) - WT(A) - WT(B) with
                      the measured dddG on the Homology-Restricted MegaScale doubles (the WT head has none: its double is the sum of its singles).

Inputs: ``results/benchmark_delta.csv`` and ``results/epistasis_rho.csv`` (``01_benchmark_delta.py``), ``results/pg_delta_pvalues.csv`` (``04_fig_proteingym.py``).

    python analysis/combined_vs_wt/05_fig_overview.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvw as C

SHORT = {'tsuboyama_capped_val': 'Homology-Restricted MegaScale, val', 'tsuboyama_capped_test': 'Homology-Restricted MegaScale, test'}
SHORT2 = {'tsuboyama_capped_val': 'Homology-Restricted\nMegaScale, val', 'tsuboyama_capped_test': 'Homology-Restricted\nMegaScale, test'}


def rows(D, PG):
    """[(group title, [(label, {family: (delta, lo, hi)}, (p0 rho wt, p0 rho comb))])]"""
    def cell(ds, cid):
        out = {}
        for fam in C.FAMS:
            r = D[(D.dataset == ds) & (D.cell == cid) & (D.family == fam)]
            if len(r):
                r = r.iloc[0]
                out[fam] = (r.d_comb, r.d_comb_lo, r.d_comb_hi)
                if fam == 'FINAL_p0':
                    out['rho'] = (r.rho_wt, r.rho_comb)
        return out

    def srt(items):
        return sorted(items, key=lambda it: -it[1]['FINAL_p0'][0])

    g1 = [(SHORT['tsuboyama_capped_test'], cell('tsuboyama_capped_test', 'singles')), (SHORT['tsuboyama_capped_val'], cell('tsuboyama_capped_val', 'singles')), ('Domainome', cell('domainome', 'all'))]
    g1 += srt([(C.LABEL[d], cell(d, 'all')) for d in C.EXT_SINGLE])
    g2 = [(SHORT['tsuboyama_capped_test'], cell('tsuboyama_capped_test', 'multi')), (SHORT['tsuboyama_capped_val'], cell('tsuboyama_capped_val', 'multi')),
          ('PTMUL', cell('ptmul', 'all')), ('PTMUL-D', cell('ptmuld', 'all'))]
    g3 = [(C.LABEL[d], cell(d, 'all')) for d in C.DMS]
    g4 = []
    for cat in C.CATS:
        r = PG[(PG.row == 'top') & (PG.group == cat)].set_index('family')
        c = {fam: (r.loc[fam, 'mean_delta'], r.loc[fam, 'ci_lo'], r.loc[fam, 'ci_hi']) for fam in ('FINAL_p0', 'FINAL_p1')}
        g4.append((f'{C.CAT_LABEL[cat]} ({int(r.loc["FINAL_p0", "n_assays"])})', c))
    return [('Stability and abundance, single mutants', g1), ('Stability, multi-mutants', g2), ('DMS benchmark sets (singles and multi-mutants)', g3), ('ProteinGym assays, by category (mean over assays)', g4)]


def forest(D, PG):
    G = rows(D, PG)
    n = sum(len(g) + 1 for _, g in G)
    off = {'FINAL_p0': -0.26, 'FINAL_p1': 0.0, 'FINAL_p0_pw0': 0.26}
    with plt.rc_context(C.RC):
        fig, ax = plt.subplots(figsize=(11.5, 0.27 * n + 1.9))
        y = 0.0
        ticks, labs, bands = [], [], []
        for gi, (title, items) in enumerate(G):
            ax.text(0.008, y, title, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=9.5, fontweight='bold', color='#222222')
            y += 1
            start = y
            for lab, c in items:
                for fam in C.FAMS:
                    if fam not in c:
                        continue
                    d, lo, hi = c[fam]
                    sig = lo > 0 or hi < 0
                    ax.plot([lo, hi], [y + off[fam]] * 2, color=C.FAM_COLOR[fam], lw=1.4, solid_capstyle='butt', zorder=3)
                    ax.plot(d, y + off[fam], C.FAM_MARKER[fam], ms=5.2, color=C.FAM_COLOR[fam], mfc=C.FAM_COLOR[fam] if sig else 'white', mew=1.3, zorder=4)
                ticks.append(y)
                labs.append(lab)
                if 'rho' in c:
                    ax.text(1.015, y, f'{c["rho"][0]:.3f} → {c["rho"][1]:.3f}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8, color='#444444')
                y += 1
            bands.append((start - 0.5, y - 0.5, gi))
            y += 0.35
        for a, b, gi in bands:
            if gi % 2 == 0:
                ax.axhspan(a, b, color='#000000', alpha=0.04, lw=0, zorder=0)
        ax.axvline(0, color=C.MUTED, lw=1, zorder=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=8.6)
        ax.invert_yaxis()
        ax.set_ylim(y, -0.7)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('delta rho: combined - WT head (> 0: combined is better)')
        ax.text(1.015, 1.0, 'ESM-MSR rho: WT-view → Dual-view', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.6, color='#444444')
        h = [plt.Line2D([], [], marker=C.FAM_MARKER[f], color=C.FAM_COLOR[f], ls='-', lw=1.4, ms=5.2, label=C.FAM_LABEL[f]) for f in C.FAMS]
        ax.legend(handles=h, loc='lower right', fontsize=8.6, handlelength=2.2, ncol=1)
        C.finish(fig, 'Delta rho = Spearman rho of the combined prediction, 1/2 (MT + WT), minus rho of the WT head, on the same variants of one checkpoint, sigma = 1, seeds 1-3 (ProteinGym: seed 1). Bars: 95% bootstrap '
                 'interval (100 resamples; proteins resampled for the external sets, PTMUL and ProteinGym assays, sites for DLG4 binding, GRB2 binding and MYO; for Homology-Restricted MegaScale and Domainome the '
                 'per-protein rho is averaged over proteins); filled marker: the interval excludes 0. Right: rho of the WT head and of the combined prediction for p0. Multi-mutants are scored with the WT head\'s '
                 'own prediction of the variant. p0: unmasked passes (ESM-MSR); p1: independent masking; pw0: no pair loss (no combined ProteinGym or Domainome run).', 'overview_forest')


def by_count(D, E):
    sets = ['tsuboyama_capped_val', 'tsuboyama_capped_test', 'ptmul'] + C.DMS
    off = {'FINAL_p0': -0.18, 'FINAL_p1': 0.0, 'FINAL_p0_pw0': 0.18}
    with plt.rc_context(C.RC):
        fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.2))
        axes = axes.ravel()
        for ax, ds in zip(axes, sets):
            d = D[(D.dataset == ds) & D.cell.str.startswith('n')]
            cells = sorted(d.cell.unique(), key=lambda c: int(c[1]))
            for fam in C.FAMS:
                r = d[d.family == fam].set_index('cell').reindex(cells)
                x = np.arange(len(cells)) + off[fam]
                ax.plot(x, r.d_comb, '-', color=C.FAM_COLOR[fam], lw=0.9, alpha=0.5, zorder=2)
                for xi, (dd, lo, hi) in zip(x, zip(r.d_comb, r.d_comb_lo, r.d_comb_hi)):
                    if not np.isfinite(dd):
                        continue
                    ax.plot([xi, xi], [lo, hi], color=C.FAM_COLOR[fam], lw=1.3, zorder=3)
                    ax.plot(xi, dd, C.FAM_MARKER[fam], ms=4.8, color=C.FAM_COLOR[fam], mfc=C.FAM_COLOR[fam] if (lo > 0 or hi < 0) else 'white', mew=1.2, zorder=4)
            nn = d[d.family == 'FINAL_p0'].set_index('cell').reindex(cells).n
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([f'{c[1:]}\n(n={int(v):,})' if v < 10000 else f'{c[1:]}\n(n={v / 1000:.0f}k)' for c, v in zip(cells, nn)], fontsize=8)
            ax.axhline(0, color=C.MUTED, lw=1, zorder=1)
            ax.set_xlim(-0.6, len(cells) - 0.4)
            ax.set_title(SHORT2.get(ds, C.LABEL[ds]), fontsize=9.5, loc='left')
            ax.set_xlabel('number of mutations', fontsize=8.6)
            ax.set_ylim(-0.17, 0.12)
        for ax in axes[[0, 4]]:
            ax.set_ylabel('delta rho: combined - WT head')
        ax = axes[6]
        for fam in C.FAMS:
            for j, ds in enumerate(C.TSU):
                r = E[(E.dataset == ds) & (E.family == fam)].iloc[0]
                x = j + off[fam] * 1.6
                ax.plot([x, x], [r.lo, r.hi], color=C.FAM_COLOR[fam], lw=1.3, zorder=3)
                ax.plot(x, r.rho_epi, C.FAM_MARKER[fam], ms=5.2, color=C.FAM_COLOR[fam], zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['val', 'test'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.3, 0.9)
        ax.set_yticks(np.arange(0, 1.0, 0.2))
        ax.set_ylabel('rho with measured dddG', fontsize=8.6)
        ax.set_title('Predicted epistasis\n(combined only)', fontsize=9.5, loc='left')
        h = [plt.Line2D([], [], marker=C.FAM_MARKER[f], color=C.FAM_COLOR[f], ls='-', lw=1.3, ms=5, label=C.FAM_LABEL[f]) for f in C.FAMS]
        axes[7].axis('off')
        axes[7].legend(handles=h, loc='center left', fontsize=9, handlelength=2.2)
        C.finish(fig, 'Delta rho = Spearman rho of the combined prediction, 1/2 (MT + WT), minus rho of the WT head, on the variants with that many mutations (sigma = 1, seeds 1-3); bars: 95% bootstrap '
                 'interval (100 resamples; proteins for PTMUL and the per-protein mean of Homology-Restricted MegaScale, sites for DLG4 binding, GRB2 binding and MYO); filled marker: the interval excludes 0; '
                 'PTMUL: 5+ = five or more mutations. Last panel: Spearman rho between the predicted epistasis comb(AB) - WT(A) - WT(B) and the measured dddG, Homology-Restricted MegaScale doubles with both '
                 'singles measured (the WT head\'s double is the sum of its singles: no epistasis); bars: 95% interval over proteins.', 'by_mutation_count')


def main():
    D = pd.read_csv(C.OUT / 'benchmark_delta.csv')
    E = pd.read_csv(C.OUT / 'epistasis_rho.csv')
    PG = pd.read_csv(C.OUT / 'pg_delta_pvalues.csv')
    forest(D, PG)
    by_count(D, E)


if __name__ == '__main__':
    sys.exit(main())
