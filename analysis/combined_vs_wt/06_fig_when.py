#!/usr/bin/env python
"""When does averaging the MT head into the WT head pay?  The gain of the combined prediction against how much worse (or better) the MT head is than the WT head.

The combined prediction is 1/2 (MT + WT).  For one protein, assay or benchmark set: delta_comb = rho(combined) - rho(WT head) against delta_mt = rho(MT head) - rho(WT head).

  when_rule   panels p0, p1: one dot per ProteinGym assay (colour = category; seed mean), per Homology-Restricted MegaScale library (val and test, all variants; seeds 1-3) and per Domainome domain;
              line = least squares of delta_comb on delta_mt over all dots of the panel, with the delta_mt at which the line crosses 0 (the break-even MT deficit).  Last panel: the benchmark sets
              of ``01_benchmark_delta.py`` (all / single mutants / multi-mutants cells) for p0, p1 and pw0.

``results/when_rule.csv``: the fit per panel (intercept, slope, R^2, n, break-even delta_mt) and the same fit with the correlation between the WT and the MT predictions as a second predictor.

    python analysis/combined_vs_wt/06_fig_when.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvw as C

UNIT_COLOR = {'MegaScale': '#222222', 'Domainome': '#9a9a9a'}


def units():
    """Per-unit table: family, kind, category, d_comb, d_mt, r_wt_mt."""
    out = []
    pg = pd.read_csv(C.OUT / 'pg_per_assay.csv', index_col='assay')
    pg = pg[~pg.train_overlap]
    for fam, tag in (('FINAL_p0', 'p0'), ('FINAL_p1', 'p1')):
        t = pd.DataFrame({'d_comb': pg[f'{tag}_all_comb'] - pg[f'{tag}_all_wt'], 'd_mt': pg[f'{tag}_all_mt'] - pg[f'{tag}_all_wt'], 'r_wt_mt': pg[f'{tag}_r_wt_mt']})
        t['family'], t['kind'], t['category'] = fam, 'ProteinGym', pg.category
        out.append(t.reset_index(drop=True))
    u = pd.read_csv(C.OUT / 'unit_rho.csv')
    u = u[u['subset'] == 'all']
    u['kind'] = np.where(u.dataset == 'domainome', 'Domainome', 'MegaScale')
    u['category'] = u.kind
    out.append(u[['family', 'kind', 'category', 'd_comb', 'd_mt', 'r_wt_mt']])
    return pd.concat(out, ignore_index=True).dropna()


def fit(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2


def main():
    U = units()
    rows = []
    for fam in C.FAMS:
        u = U[U.family == fam]
        if not len(u):
            continue
        b, r2 = fit(u.d_mt.to_numpy(), u.d_comb.to_numpy())
        A = np.column_stack([np.ones(len(u)), u.d_mt, 1 - u.r_wt_mt])
        b2 = np.linalg.lstsq(A, u.d_comb.to_numpy(), rcond=None)[0]
        r22 = 1 - ((u.d_comb - A @ b2) ** 2).sum() / ((u.d_comb - u.d_comb.mean()) ** 2).sum()
        rows.append(dict(family=fam, n_units=len(u), intercept=b[0], slope=b[1], r2=r2, breakeven_d_mt=-b[0] / b[1], intercept2=b2[0], slope_d_mt=b2[1], slope_one_minus_r=b2[2], r2_two_predictors=r22))
    F = pd.DataFrame(rows)
    F.to_csv(C.OUT / 'when_rule.csv', index=False, float_format='%.4f')
    print(F.round(3).to_string())
    D = pd.read_csv(C.OUT / 'benchmark_delta.csv')
    with plt.rc_context(C.RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
        for ax, fam in zip(axes[:2], C.FAMS[:2]):
            u = U[U.family == fam]
            for kind, col in (('Domainome', UNIT_COLOR['Domainome']), ('MegaScale', UNIT_COLOR['MegaScale'])):
                k = u[u.kind == kind]
                ax.plot(k.d_mt, k.d_comb, 'o', ms=3.6 if kind == 'Domainome' else 4.6, color=col, alpha=0.35 if kind == 'Domainome' else 0.85, mew=0, zorder=2 if kind == 'Domainome' else 3,
                        label=f'{"Domainome domains" if kind == "Domainome" else "Homology-Restricted MegaScale libraries"} ({len(k)})')
            for cat in C.CATS:
                k = u[u.category == cat]
                ax.plot(k.d_mt, k.d_comb, 'D', ms=4.6, color=C.CAT_COLOR[cat], alpha=0.85, mew=0, zorder=4, label=f'ProteinGym {C.CAT_LABEL[cat]} ({len(k)})')
            r = F[F.family == fam].iloc[0]
            xs = np.array([-0.45, 0.25])
            ax.plot(xs, r.intercept + r.slope * xs, color='#c0392b', lw=1.3, zorder=5)
            ax.axhline(0, color=C.MUTED, lw=1, zorder=1)
            ax.axvline(0, color=C.MUTED, lw=1, zorder=1)
            ax.text(0.03, 0.04, f'delta_comb = {r.intercept:+.3f} {r.slope:+.2f} delta_mt\nR² = {r.r2:.2f}; crosses 0 at delta_mt = {r.breakeven_d_mt:.3f}', transform=ax.transAxes, fontsize=8.4, va='bottom', ha='left',
                    bbox=dict(fc=C.SURFACE, ec='none', alpha=0.85, pad=2))
            ax.set_xlim(-0.4, 0.21)
            ax.set_ylim(-0.17, 0.2)
            ax.set_title(f'{C.FAM_LABEL[fam]}: proteins and assays', fontsize=10, loc='left')
            ax.set_xlabel('delta_mt = rho(MT head) - rho(WT head)')
            ax.set_ylabel('delta_comb = rho(combined) - rho(WT head)')
        axes[0].legend(loc='upper left', fontsize=7.4, handlelength=1.2, borderaxespad=0.2)
        ax = axes[2]
        grp = {'Stability / abundance': '#6250d6', 'DMS benchmark sets': '#eb6834'}
        sets = D[D.cell.isin(['all', 'singles', 'multi'])]
        for fam in C.FAMS:
            s = sets[sets.family == fam]
            col = np.where(s.dataset.isin(C.DMS), grp['DMS benchmark sets'], grp['Stability / abundance'])
            ax.scatter(s.d_mt, s.d_comb, marker=C.FAM_MARKER[fam], s=24, c=col, alpha=0.85, edgecolors=C.FAM_COLOR[fam], linewidths=0.8, zorder=3)
        ax.axhline(0, color=C.MUTED, lw=1, zorder=1)
        ax.axvline(0, color=C.MUTED, lw=1, zorder=1)
        marks = [('DLG4_HUMAN_Faure_2021_binding_domain', 'all', 'DLG4 binding', (-4, 12)),
                 ('GRB2_HUMAN_Faure_2021_binding_domain', 'all', 'GRB2 binding', (6, -12)), ('tsuboyama_capped_test', 'multi', 'MegaScale test, doubles', (5, 3)),
                 ('tsuboyama_capped_val', 'multi', 'MegaScale val, doubles', (5, 3))]
        for ds, cid, nm, off in marks:
            r = sets[(sets.family == 'FINAL_p0') & (sets.dataset == ds) & (sets.cell == cid)].iloc[0]
            ax.annotate(nm, (r.d_mt, r.d_comb), xytext=off, textcoords='offset points', fontsize=7.4, color='#333333', ha='right' if off[0] < 0 else 'left')
        ax.set_xlim(-0.16, 0.06)
        ax.set_ylim(-0.065, 0.075)
        ax.set_title('Benchmark sets and variant classes', fontsize=10, loc='left')
        ax.set_xlabel('delta_mt = rho(MT head) - rho(WT head)')
        ax.set_ylabel('delta_comb = rho(combined) - rho(WT head)')
        h = [plt.Line2D([], [], marker=C.FAM_MARKER[f], color='white', mfc='#bbbbbb', mec=C.FAM_COLOR[f], ms=6, mew=1.2, label=C.FAM_LABEL[f]) for f in C.FAMS]
        h += [plt.Line2D([], [], marker='o', color='white', mfc=c, mec=c, ms=6, label=n) for n, c in grp.items()]
        ax.legend(handles=h, loc='upper left', fontsize=7.8, borderaxespad=0.2)
        C.finish(fig, 'delta_comb = Spearman rho of the combined prediction, 1/2 (MT + WT), minus rho of the WT head; delta_mt = rho of the MT head minus rho of the WT head; sigma = 1, same variants and checkpoint. '
                 'Dots of the first two panels: ProteinGym assays (seed mean, all variants), Homology-Restricted MegaScale libraries (val and test, all variants, seeds 1-3) and Domainome domains (seeds 1-3); red line: '
                 'least squares over the dots of the panel. Last panel: the sets of the overview (all variants, single mutants and multi-mutants of each set), one marker per family.', 'when_rule')


if __name__ == '__main__':
    sys.exit(main())
