#!/usr/bin/env python
"""ProteinGym: does the combined prediction beat the WT head, by assay category and variant class (p0 and p1).

  pg_combined_vs_wt   top row: per category (Stability, Expression, Activity, Binding, Organismal fitness) one box per family (p0, p1) of delta rho = rho(combined) - rho(WT head) over the assays,
                      all variants of the assay (the ProteinGym metric).  Bottom row: the assays with at least 20 single and at least 20 multi-mutants, delta rho on their single mutants and on
                      their multi-mutants, for the Stability assays and for the assays of the other categories together.  IQR box, median, jittered dots (circles p0, diamonds p1), faint lines
                      joining the same assay in p0 and p1, dark line = mean (value printed); under the ticks the p-value of the mean being 0.

The 23 stability assays whose variants are in the training data are left out (``pgsigma.train_overlap``), which leaves 194 assays.  Seed 1 of both families.  p-values: two-sided bootstrap over assays
(20000 resamples, p = 2 min(P(mean <= 0), P(mean >= 0)), as ``msr_analysis.boot.paired_diff``), Holm-adjusted over the group x family comparisons of a row (groups with at least 10 assays); the
Wilcoxon signed-rank p is in ``results/pg_delta_pvalues.csv`` (also the mean delta rho, its 95% interval, the share of assays that gain and, for p0, the mean over seeds 1-3).

    python analysis/combined_vs_wt/04_fig_proteingym.py
"""
import colorsys
import sys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import cvw as C

FAMS = [('FINAL_p0', 'p0'), ('FINAL_p1', 'p1')]
MIN_P = 10                       # a group needs this many assays to get a p-value
OTHER = '#6b6b6b'


def data():
    t = pd.read_csv(C.OUT / 'pg_per_assay.csv', index_col='assay')
    t = t[~t.train_overlap].copy()
    for sub in ('all', 'single', 'multi'):
        for fam, tag in FAMS:
            t[f'd_{sub}_{tag}'] = t[f'{tag}_s1_{sub}_comb'] - t[f'{tag}_s1_{sub}_wt']
    t['d_all_p0_3seed'] = np.mean([t[f'p0_s{s}_all_comb'] - t[f'p0_s{s}_all_wt'] for s in (1, 2, 3)], axis=0)
    return t


def groups(t):
    """[(row, group id, label, colour, variant class, boolean mask over the assays)]"""
    g = [('top', cat, C.CAT_LABEL[cat], C.CAT_COLOR[cat], 'all', (t.category == cat).to_numpy()) for cat in C.CATS]
    both = t['d_single_p0'].notna() & t['d_multi_p0'].notna() & t['d_single_p1'].notna() & t['d_multi_p1'].notna()
    stab = (t.category == 'Stability').to_numpy()
    for kind, lab in (('single', 'single mutants'), ('multi', 'multi-mutants')):
        g.append(('bottom', f'Stability_{kind}', f'Stability, {lab}', C.CAT_COLOR['Stability'], kind, (both.to_numpy() & stab)))
    for kind, lab in (('single', 'single mutants'), ('multi', 'multi-mutants')):
        g.append(('bottom', f'Other_{kind}', f'Other categories, {lab}', OTHER, kind, (both.to_numpy() & ~stab)))
    return g


def pvalues(t, G, B=20000, seed=3):
    """Per group and family: mean delta rho, bootstrap interval and p, Holm over the row, Wilcoxon cross-check."""
    rng = np.random.default_rng(seed)
    rows = []
    for row, gid, lab, col, sub, m in G:
        for fam, tag in FAMS:
            d = t.loc[m, f'd_{sub}_{tag}'].dropna().to_numpy(float)
            if len(d) < 3:
                continue
            b = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
            p = max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / (B + 1))
            r = dict(row=row, group=gid, variants=sub, family=fam, n_assays=len(d), mean_delta=d.mean(), median_delta=np.median(d), ci_lo=np.quantile(b, .025), ci_hi=np.quantile(b, .975),
                     share_gain=float((d > 0).mean()), p_boot=p, p_wilcoxon=float(wilcoxon(d).pvalue))
            if row == 'top' and tag == 'p0':
                r['mean_delta_p0_3seed'] = float(t.loc[m, 'd_all_p0_3seed'].mean())
            rows.append(r)
    R = pd.DataFrame(rows)
    R['p_holm'] = np.nan
    R['p_wilcoxon_holm'] = np.nan
    for row in ('top', 'bottom'):
        i = (R.row == row) & (R.n_assays >= MIN_P)
        R.loc[i, 'p_holm'] = C.holm(R.loc[i, 'p_boot'])
        R.loc[i, 'p_wilcoxon_holm'] = C.holm(R.loc[i, 'p_wilcoxon'])
    R.to_csv(C.OUT / 'pg_delta_pvalues.csv', index=False, float_format='%.4g')
    return R


def darker(c):
    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return mc.to_hex(colorsys.hls_to_rgb(h, max(0.22, min(0.42, l * 0.56)), min(1.0, max(s, 0.72))))


def draw_row(ax, t, R, G, row, title, rng, gap=0.9, w=0.8):
    ax.grid(axis='x', visible=False)
    x, ticks, labs = 0.0, [], []
    tr = ax.get_xaxis_transform()
    for i, (rw, gid, lab, col, sub, m) in enumerate(g_ for g_ in G if g_[0] == row):
        g = t.loc[m, [f'd_{sub}_p0', f'd_{sub}_p1']].dropna()
        if i % 2 == 0:
            ax.axvspan(x - 0.5 - gap / 2, x + 1.5 + gap / 2, color='#000000', alpha=0.045, lw=0, zorder=0)
        jit = (rng.random(len(g)) - 0.5) * (w * 0.55)
        for r_ in range(len(g)):
            ax.plot([x + jit[r_], x + 1 + jit[r_]], g.iloc[r_].to_numpy(), color=col, alpha=0.16, lw=0.6, zorder=2)
        for k, (fam, tag) in enumerate(FAMS):
            c = x + k
            y = g[f'd_{sub}_{tag}'].to_numpy()
            q1, med, q3 = np.quantile(y, [0.25, 0.5, 0.75])
            ax.add_patch(plt.Rectangle((c - w / 2, q1), w, q3 - q1, fc=col, alpha=0.22, ec='none', zorder=1))
            ax.plot([c - w / 2, c + w / 2], [med, med], color=darker(col), lw=1.1, zorder=4, solid_capstyle='butt')
            ax.plot(c + jit, y, C.FAM_MARKER[fam], ms=2.8, color=col, alpha=0.8, mew=0, zorder=3)
            mean = float(y.mean())
            ax.plot([c - w / 2, c + w / 2], [mean, mean], color='#333333', lw=1.8, zorder=5)
            ax.text(c, max(q3, mean), f'{mean:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='semibold', zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.1))
            pr = R[(R.row == row) & (R.group == gid) & (R.family == fam)].p_holm.iloc[0]
            ticks.append(c)
            labs.append(f'{tag}\n{C.fmt_p(pr)}' if np.isfinite(pr) else tag)
        ax.plot([x - 0.44, x + 1.44], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.3)
        ax.text(x + 0.5, 1.022, f'{lab} ({len(g)})', transform=tr, ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#333333', wrap=False)
        x += 2 + gap
    ax.axhline(0, color=C.MUTED, lw=1, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    ax.set_xlim(-0.8, x - gap - 0.2)
    ax.set_ylabel('delta rho: combined - WT head')
    ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold', pad=30)


def figure(t, R, G):
    rng = np.random.default_rng(1)
    with plt.rc_context(C.RC):
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.6))
        draw_row(axes[0], t, R, G, 'top', 'All variants of the assay, by category', rng)
        draw_row(axes[1], t, R, G, 'bottom', 'Single mutants and multi-mutants of the assays that have both', rng)
        C.finish(fig, 'One dot per ProteinGym assay (194 assays); faint lines join the same assay for p0 (circles) and p1 (diamonds). '
                 'Delta rho = Spearman rho of the combined prediction, 1/2 (MT + WT), minus rho of the WT head, on the same variants, sigma = 1, seed 1. Box = interquartile range, thin darker line = median, dark line = mean '
                 '(value printed). Bottom row: the assays with at least 20 single mutants and at least 20 multi-mutants, delta rho over their single mutants and over their multi-mutants. p under the ticks: two-sided '
                 'bootstrap test over assays that the mean delta rho is 0 (20000 resamples), Holm-adjusted over the group x family comparisons of the row (groups with at least 10 assays).', 'pg_combined_vs_wt')


def main():
    t = data()
    G = groups(t)
    R = pvalues(t, G)
    pd.set_option('display.width', 250)
    print(R[['row', 'group', 'family', 'n_assays', 'mean_delta', 'share_gain', 'p_boot', 'p_holm', 'p_wilcoxon_holm']].round(4).to_string())
    figure(t, R, G)


if __name__ == '__main__':
    sys.exit(main())
