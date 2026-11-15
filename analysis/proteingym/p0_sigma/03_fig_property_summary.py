#!/usr/bin/env python
"""How the WT adapter's sigma changes ProteinGym performance, by property (assay category): two drafts.

  property_delta_trajectories   one panel per property; x = sigma (0, 0.5, 1), y = delta rho versus sigma 0, one thin line per assay (assays are treated as independent: no protein structure), the bold
                                line is the mean over the assays with a 95% bootstrap interval over assays
  property_delta_paired_boxes   one box per property and sigma (IQR box, median, jittered dots, mean line with its value): sigma 0 (the anchor, delta = 0), sigma 0.5 and sigma 1 side by side,
                                faint thin lines connect the same assay across the sigmas (the style of the property-scaffold plots of ``14_property_category_gain.py``)

sigma = 1, 0.5: FINAL_p0 seed 1 WT head; sigma = 0: ESM3-small-open, unmasked.  The ProteinGym assays with predictions at all three sigmas (``results/per_assay_rho.csv``) except the 23 stability assays that overlap the training data (``pgsigma.train_overlap``).

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

B = 2000


def darker(c):
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, max(0.22, min(0.42, l * 0.56)), min(1.0, max(s, 0.72))))


def boot_mean(v, rng):
    i = rng.integers(0, len(v), (B, len(v)))
    m = v[i].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def data():
    t = P.per_assay()                                                      # ProteinGym assays that do not overlap the training data
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


def pvalues(t, B=20000, seed=3):
    """Per category and sigma (0.5, 1): is the mean delta rho versus sigma 0 different from 0?  Two-sided bootstrap p over assays (``B`` resamples, the repo's convention of ``msr_analysis.boot.paired_diff``:
    p = 2 min(P(mean <= 0), P(mean >= 0)), at least 1 / (B + 1)), Holm-adjusted over the 10 category x sigma comparisons; the Wilcoxon signed-rank p over assays is given as a cross-check.
    Assays are treated as independent (as in the figures)."""
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
    R.to_csv(P.OUT / 'property_delta_pvalues.csv', index=False, float_format='%.4g')
    return R


def fmt_p(p):
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}' if p < 0.1 else f'p={p:.2f}'


def tick_labels(R, cat):
    """x tick labels of one category: sigma, and under 0.5 and 1 the Holm-adjusted p of the mean delta rho."""
    r = R[R.category == cat].set_index('sigma')
    return ['0', f'0.5\n{fmt_p(r.loc[0.5, "p_holm"])}', f'1\n{fmt_p(r.loc[1.0, "p_holm"])}']


def finish(fig, ax_titles, note, stem, legend=None):
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


def trajectories(t, R):
    rng = np.random.default_rng(0)
    xs = [0.0, 0.5, 1.0]
    with plt.rc_context(P.RC):
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.3), sharey=True)
        ax_titles = []
        for ax, cat in zip(axes, P.CATS):
            g = t[t.category == cat]
            col = P.CAT_COLOR[cat]
            Y = np.column_stack([np.zeros(len(g)), g['d_0.5'], g['d_1.0']])
            for y in Y:
                ax.plot(xs, y, color=col, alpha=0.28, lw=0.9, zorder=2)
            m = [boot_mean(Y[:, k], rng) for k in range(3)]
            ax.fill_between(xs, [v[1] for v in m], [v[2] for v in m], color=col, alpha=0.25, lw=0, zorder=3)
            ax.plot(xs, [v[0] for v in m], color=P.INK, lw=2.4, marker='o', ms=6, mec=P.SURFACE, mew=1.2, zorder=5)
            for x, v in zip(xs[1:], m[1:]):
                ax.annotate(f'{v[0]:+.3f}', (x, v[0]), xytext=(-6, 7), textcoords='offset points', ha='right', fontsize=8, fontweight='semibold', color=P.INK, zorder=6)
            ax.axhline(0, color=P.MUTED, lw=1, zorder=1)
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels(R, cat))
            ax.set_xlim(-0.04, 1.04)
            ax.set_xlabel('sigma')
            ax_titles.append((ax, f'{P.CAT_LABEL[cat]} ({len(g)} assays)'))
        axes[0].set_ylabel('delta rho versus sigma = 0')
        finish(fig, ax_titles, 'One thin line per ProteinGym assay (assays independent; no protein grouping); bold line = mean over the assays with its 95% bootstrap interval over assays (2000 replicates). p under the ticks: two-sided bootstrap test over assays that the mean delta rho is 0, Holm-adjusted over the ten category x sigma comparisons. '
               f'Delta rho = Spearman rho of the WT-adapter prediction at sigma minus rho at sigma = 0 (ESM3-small-open, unmasked); sigma = 0.5 and 1: FINAL_p0 seed 1; {len(t)} assays.', 'property_delta_trajectories')


def paired_boxes(t, R):
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
            for r, y in enumerate(Y):
                ax.plot([pos[(cat, k)] + jit[r] for k in range(3)], y, color=col, alpha=0.16, lw=0.6, zorder=2)
            for k, lab in enumerate(['0', '0.5', '1']):
                c = pos[(cat, k)]
                y = Y[:, k]
                if k > 0:
                    q1, med, q3 = np.quantile(y, [0.25, 0.5, 0.75])
                    ax.add_patch(plt.Rectangle((c - w / 2, q1), w, q3 - q1, fc=col, alpha=0.22, ec='none', zorder=1))
                    ax.plot([c - w / 2, c + w / 2], [med, med], color=darker(col), lw=1.1, zorder=4, solid_capstyle='butt')
                ax.plot(c + jit, y, 'o', ms=2.6, color=col, alpha=0.8, mew=0, zorder=3)
                mean = float(y.mean())
                ax.plot([c - w / 2, c + w / 2], [mean, mean], color='#333333', lw=1.8, zorder=5)
                if k > 0:
                    ax.text(c, mean + 0.012, f'{mean:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='semibold', zorder=8, bbox=dict(fc='white', ec='none', alpha=0.68, pad=0.1))
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.44, x1 + 0.44], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.3)
            ax.text((x0 + x1) / 2, 1.022, f'{P.CAT_LABEL[cat]} ({len(g)})', transform=tr, ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#333333')
        ax.axhline(0, color=P.MUTED, lw=1, zorder=1)
        ax.set_xticks([pos[(c, k)] for c in P.CATS for k in range(3)])
        ax.set_xticklabels([l for c in P.CATS for l in tick_labels(R, c)])
        ax.set_xlim(-0.8, x - gap - 0.2)
        ax.set_xlabel('sigma')
        ax.set_ylabel('delta rho versus sigma = 0')
        finish(fig, [], 'One dot per ProteinGym assay; faint thin lines connect the same assay at sigma 0 (delta = 0 by definition: the anchor), 0.5 and 1. Box = interquartile range, thin darker line = median, '
               f'dark line = mean (value printed); p under the ticks: two-sided bootstrap test over assays that the mean delta rho is 0 (20000 resamples), Holm-adjusted over the ten category x sigma comparisons. Delta rho = Spearman rho of the WT-adapter prediction at sigma minus rho at sigma = 0 (ESM3-small-open, unmasked); sigma = 0.5 and 1: FINAL_p0 seed 1; {len(t)} assays.', 'property_delta_paired_boxes')


def main():
    t = data()
    print(t.groupby('category')[['d_0.5', 'd_1.0']].agg(['mean', 'median']).round(3).assign(n=t.groupby('category').size()).to_string())
    R = pvalues(t)
    print(R[['category', 'sigma', 'n_assays', 'mean_delta', 'p_boot', 'p_holm', 'p_wilcoxon', 'p_wilcoxon_holm']].round(4).to_string())
    trajectories(t, R)
    paired_boxes(t, R)


if __name__ == '__main__':
    main()
