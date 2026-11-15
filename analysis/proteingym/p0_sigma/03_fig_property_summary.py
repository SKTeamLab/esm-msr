#!/usr/bin/env python
"""How the WT adapter's sigma changes ProteinGym performance, by property (assay category): paired distributions.

  property_delta_paired_boxes             one box per property and sigma (IQR box, median, jittered dots, mean line with its value): sigma 0 (the anchor, delta = 0), sigma 0.5 and sigma 1 side by side,
                                         faint thin lines connect the same assay across the sigmas (the style of the property-scaffold plots of ``14_property_category_gain.py``); every assay is drawn and
                                         counted, nothing is marked
  property_delta_paired_boxes_leaks_black  the same, every assay drawn, the assays whose protein has a training-set homolog (``04_leak_per_assay.py``: TM-score > 0.7 or sequence identity > 0.4) in black and left
                                         out of the statistics: the boxes, medians, means, counts and p-values are those of the other assays

Delta rho = rho of the WT-adapter prediction at sigma minus rho at sigma 0; sigma = 1, 0.5: FINAL_p0 seed 1 WT head; sigma = 0: ESM3-small-open, unmasked.  The ProteinGym assays with predictions at all three
sigmas (``results/per_assay_rho.csv``) except the 23 stability assays whose variants are in the training data (``pgsigma.train_overlap``).  The p-value under each sigma tick: is the mean delta rho of the category
different from 0?  Two-sided bootstrap over assays (20000 resamples, ``msr_analysis.boot.paired_diff``'s convention), Holm-adjusted over the ten category x sigma comparisons of the figure; the Wilcoxon signed-rank
p is in the csv (``results/property_delta_pvalues.csv`` for all assays, ``property_delta_pvalues_nonleaked.csv`` for the figure with the black assays left out of the statistics).

    python analysis/proteingym/p0_sigma/03_fig_property_summary.py
"""
import sys
import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import pgsigma as P
sys.path.insert(0, str(P.HERE.parent.parent))
from msr_analysis import style  # noqa: E402


def darker(c):
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, max(0.22, min(0.42, l * 0.56)), min(1.0, max(s, 0.72))))


def data():
    t = P.per_assay()                                                      # ProteinGym assays that do not overlap the training data, with the per-assay leak flag
    for s in (0.5, 1.0):
        t[f'd_{s}'] = t[f'rho_{s}'] - t['rho_0.0']
    return t


def holm(p):
    """Holm-adjusted p-values of an array of raw p-values (one family)."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[o])
    out = np.empty(len(p))
    out[o] = np.maximum.accumulate(adj)
    return out


def pvalues(t, csv, B=20000, seed=3):
    """Per category and sigma (0.5, 1): is the mean delta rho versus sigma 0 different from 0?  Two-sided bootstrap p over assays (``B`` resamples, the repo's convention of ``msr_analysis.boot.paired_diff``:
    p = 2 min(P(mean <= 0), P(mean >= 0)), at least 1 / (B + 1)), Holm-adjusted over the 10 category x sigma comparisons; the Wilcoxon signed-rank p over assays is given as a cross-check.
    Assays are treated as independent (as in the figures).  Written to ``results/<csv>``."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat in P.CATS:
        g = t[t.category == cat]
        for s_ in (0.5, 1.0):
            d = g[f'd_{s_}'].to_numpy(float)
            m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
            p = max(2 * min((m <= 0).mean(), (m >= 0).mean()), 1.0 / (B + 1))
            rows.append(dict(category=cat, sigma=s_, n_assays=len(d), mean_delta=d.mean(), median_delta=np.median(d), ci_lo=np.quantile(m, .025), ci_hi=np.quantile(m, .975),
                             share_improved=float((d > 0).mean()), p_boot=p, p_wilcoxon=float(wilcoxon(d).pvalue)))
    R = pd.DataFrame(rows)
    R['p_holm'] = holm(R.p_boot)
    R['p_wilcoxon_holm'] = holm(R.p_wilcoxon)
    R.to_csv(P.OUT / csv, index=False, float_format='%.4g')
    return R


def fmt_p(p):
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}' if p < 0.1 else f'p={p:.2f}'


def tick_labels(R, cat):
    """x tick labels of one category: sigma, and under 0.5 and 1 the Holm-adjusted p of the mean delta rho."""
    r = R[R.category == cat].set_index('sigma')
    return ['0', f'0.5\n{fmt_p(r.loc[0.5, "p_holm"])}', f'1\n{fmt_p(r.loc[1.0, "p_holm"])}']


def finish(fig, ax_titles, note, stem):
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=7.6, color='#333333')
    tt = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = tt.get_window_extent(r).height / fig.bbox.height
    for _ in range(3):
        fig.tight_layout(rect=(0, foot + 0.02, 1, 0.99))
        for ax, raw in ax_titles:
            ax.set_title(style.wrap_to_px(fig, raw, 0.97 * ax.get_window_extent(r).width, fontsize=10, fontweight='bold'), loc='left', fontsize=10)
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, P.OUT / stem)


def paired_boxes(t, R, stem, note, mark_leaks=False, stats_without_leaks=False):
    """All assays of ``t`` are drawn; ``mark_leaks`` draws the leaked ones black, ``stats_without_leaks`` leaves them out of the boxes, medians, means and counts (``R`` must then be the p-values of the others)."""
    rng = np.random.default_rng(1)
    gap, w = 0.9, 0.82
    pos, spans, x = {}, {}, 0.0
    for cat in P.CATS:
        for k in range(3):
            pos[(cat, k)] = x
            x += 1.0
        spans[cat] = (pos[(cat, 0)], pos[(cat, 2)])
        x += gap
    with plt.rc_context(P.RC):
        fig, ax = plt.subplots(figsize=(13.5, 5.6))
        ax.grid(axis='x', visible=False)
        for i, cat in enumerate(P.CATS):
            x0, x1 = spans[cat]
            if i % 2 == 0:
                ax.axvspan(x0 - 0.5 - gap / 2, x1 + 0.5 + gap / 2, color='#000000', alpha=0.045, lw=0, zorder=0)
            g = t[t.category == cat]
            col = P.CAT_COLOR[cat]
            Y = np.column_stack([np.zeros(len(g)), g['d_0.5'], g['d_1.0']])
            jit = (rng.random(len(g)) - 0.5) * (w * 0.56)                      # one offset per assay, so the connecting lines stay tidy
            leaked = g['leaked'].to_numpy()
            lk = leaked if mark_leaks else np.zeros(len(g), bool)               # drawn black
            use = ~leaked if stats_without_leaks else np.ones(len(g), bool)      # enter the statistics
            for r, y in enumerate(Y):
                ax.plot([pos[(cat, k)] + jit[r] for k in range(3)], y, color='#000000' if lk[r] else col, alpha=0.4 if lk[r] else 0.16, lw=0.6, zorder=2.5 if lk[r] else 2)
            for k, lab in enumerate(['0', '0.5', '1']):
                c = pos[(cat, k)]
                y = Y[:, k]
                if k > 0:
                    q1, med, q3 = np.quantile(y[use], [0.25, 0.5, 0.75])
                    ax.add_patch(plt.Rectangle((c - w / 2, q1), w, q3 - q1, fc=col, alpha=0.22, ec='none', zorder=1))
                    ax.plot([c - w / 2, c + w / 2], [med, med], color=darker(col), lw=1.1, zorder=4, solid_capstyle='butt')
                ax.plot((c + jit)[~lk], y[~lk], 'o', ms=2.6, color=col, alpha=0.8, mew=0, zorder=3)
                ax.plot((c + jit)[lk], y[lk], 'o', ms=3.0, color='#000000', alpha=0.95, mew=0, zorder=3.5)
                mean = float(y[use].mean())
                ax.plot([c - w / 2, c + w / 2], [mean, mean], color='#333333', lw=1.8, zorder=5)
                if k > 0:
                    ax.text(c, mean + 0.012, f'{mean:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='semibold', zorder=8, bbox=dict(fc='white', ec='none', alpha=0.68, pad=0.1))
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.44, x1 + 0.44], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.3)
            ax.text((x0 + x1) / 2, 1.022, f'{P.CAT_LABEL[cat]} ({int(use.sum())}' + (f' of {len(g)})' if use.sum() < len(g) else ')'), transform=tr, ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#333333')
        ax.axhline(0, color=P.MUTED, lw=1, zorder=1)
        ax.set_xticks([pos[(c, k)] for c in P.CATS for k in range(3)])
        ax.set_xticklabels([l for c in P.CATS for l in tick_labels(R, c)])
        ax.set_xlim(-0.8, x - gap - 0.2)
        ax.set_xlabel('sigma')
        ax.set_ylabel('delta rho versus sigma = 0')
        finish(fig, [], 'One dot per ProteinGym assay; faint thin lines connect the same assay at sigma 0 (delta = 0 by definition: the anchor), 0.5 and 1. Box = interquartile range, thin darker line = median, '
               'dark line = mean (value printed); p under the ticks: two-sided bootstrap test over assays that the mean delta rho is 0 (20000 resamples), Holm-adjusted over the ten category x sigma comparisons. '
               f'Delta rho = Spearman rho of the WT-adapter prediction at sigma minus rho at sigma = 0 (ESM3-small-open, unmasked); sigma = 0.5 and 1: FINAL_p0 seed 1; {len(t)} assays. ' + note, stem)


def main():
    t = data()
    print(t.groupby('category')[['d_0.5', 'd_1.0']].agg(['mean', 'median']).round(3).assign(n=t.groupby('category').size()).to_string())
    R = pvalues(t, 'property_delta_pvalues.csv')
    print(R[['category', 'sigma', 'n_assays', 'mean_delta', 'p_boot', 'p_holm', 'p_wilcoxon', 'p_wilcoxon_holm']].round(4).to_string())
    paired_boxes(t, R, 'property_delta_paired_boxes', 'Every assay enters the statistics.')
    clean = t[~t.leaked].reset_index(drop=True)
    Rc = pvalues(clean, 'property_delta_pvalues_nonleaked.csv')
    print(f'--- statistics of the {len(clean)} assays without a training-set homolog (the {int(t.leaked.sum())} others are drawn black) ---')
    print(Rc[['category', 'sigma', 'n_assays', 'mean_delta', 'p_boot', 'p_holm', 'p_wilcoxon', 'p_wilcoxon_holm']].round(4).to_string())
    paired_boxes(t, Rc, 'property_delta_paired_boxes_leaks_black',
                 f'Black dots and lines: assays whose protein is a training-set homolog (TM-score > 0.7 or sequence identity > 0.4 to a training protein; {int(t.leaked.sum())} of {len(t)}); they are not used in the boxes, '
                 f'medians, means, counts in the headers (n used of n drawn) and p-values.', mark_leaks=True, stats_without_leaks=True)


if __name__ == '__main__':
    main()
