#!/usr/bin/env python
"""Single-mutant screening figures on the Homology-Restricted MegaScale test set (33 domains).

Every figure, CSV, and p-value table is produced in two versions:
  - all single mutants (n = 35,421)
  - excluding mutations to cysteine (suffix ``_nocys``, n = 33,854; 1,567 X->C substitutions removed)

Three views of a virtual screen on single mutants only (measured rows, all backbones):

1. ``fig_suggested_hit_rate_with_counts[_nocys]`` (+ ``fig_suggested_hit_rate[_nocys]``, ``_pooled[_nocys]``)
   Variants suggested by predicted ddG > 0.5 kcal/mol:
   - Top panel: Precision among suggested variants (fraction with measured ddG > 0 stabilizing [hatched] and
     measured ddG > 0.5 kcal/mol [solid]).
   - Middle panel: Recall @ 0 / 0.5 (fraction of all true measured ddG > 0 [hatched] and measured ddG > 0.5 kcal/mol
     [solid] single mutants in each domain that are suggested at predicted ddG > 0.5 kcal/mol).
   - Bottom panel: Single bar per model showing the mean number of suggested variants (predicted ddG > 0.5 kcal/mol)
     per domain, against the true number of ddG > 0.5 kcal/mol variants per domain.

2. ``fig_top6_recovery_k6_k24_k96[_nocys]`` (+ ``_atleast1[_nocys]``, ``_both[_nocys]``)
   Probability of recovering the top-N (N = 6) most stabilizing single mutants per protein under fixed screening
   budgets K = 6, 24, 96 (three bars per model, light to dark):
   - per-mutant recovery probability (fraction |TopK_pred & TopN_true| / N recovered per protein, averaged over the
     33 domains), and
   - at-least-one recovery probability P(|TopK_pred & TopN_true| >= 1 per protein),
   each compared against the exact hypergeometric random-pick expectation.

3. ``fig_best_ddg_vs_k_panels[_nocys]`` (+ ``fig_best_ddg_vs_k[_nocys]``)
   Best measured ddG recovered among the top K predicted single mutants per protein as a function of every discrete
   screening budget K = 1 .. 96 (one line per model, no markers), separated by model type:
   - Panel A: Models retrained on Homology-Restricted MegaScale (ESM-MSR WT head, ESM-MSR combined, MutateEverything st1,
     ThermoMPNN, SPURS).
   - Panel B: ESM-MSR vs Zero-shot & Supervised (not Megascale-trained) models.

Conventions: ESM-MSR WT head and ESM-MSR combined (no baseline); "Homology-Restricted MegaScale"; ESM3-small-open read
unmasked; domain-level bootstrap (draft default --B 200); no GeoDDG-3D annotation; no exclusion text.
Outputs: ``results/08_screening/``.
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import ci  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

pub = importlib.import_module('11_publication_figures')

OUT = out_dir('08_screening')
DS = 'tsuboyama_capped_test'
SPLIT = pub.SPLIT
B_DEFAULT = 200
KS_TICKS = [1, 2, 3, 6, 12, 24, 48, 96]


# ----------------------------------------------------------------------------------------------
# model columns (single mutants only: ESM-MSR WT head and ESM-MSR combined)
# ----------------------------------------------------------------------------------------------
def single_columns(fr, best):
    """Columns in the three categories of 11_publication_figures.py, showing baseline, ESM-MSR (devel), and p0 (WT head and combined)."""
    cols = [
        dict(name='ESM-MSR (baseline) WT head', cat='retrained', color='#7f8c8d', ls='-',
             label='ESM-MSR\n(baseline)\nWT head', key='msr:wt', draw=True),
        dict(name='ESM-MSR (baseline) combined', cat='retrained', color='#95a5a6', ls='--',
             label='ESM-MSR\n(baseline)\ncombined', key='msr:comb', draw=True),
        dict(name='ESM-MSR WT head', cat='retrained', color='#000000', ls='-',
             label='ESM-MSR\nWT head', key='msrdev:wt', draw=True),
        dict(name='ESM-MSR combined', cat='retrained', color='#555555', ls='--',
             label='ESM-MSR\ncombined', key='msrdev:comb', draw=True),
        dict(name='ESM-MSR (p0) WT head', cat='retrained', color='#6c3483', ls='-',
             label='ESM-MSR\n(p0)\nWT head', key='msrp0:wt', draw=True),
        dict(name='ESM-MSR (p0) combined', cat='retrained', color='#9b59b6', ls='--',
             label='ESM-MSR\n(p0)\ncombined', key='msrp0:comb', draw=True),
        dict(name='MutateEverything st1', cat='retrained', color='#1b8a4b', ls='-',
             label='MutateEverything\nst1', key='me1', draw=True),
        dict(name='ThermoMPNN', cat='retrained', color='#1f5f8b', ls='-',
             label='ThermoMPNN', key='tmpnn', draw=True),
        dict(name='SPURS', cat='retrained', color='#d35400', ls='-',
             label='SPURS', key='spurs', draw=True),
        dict(name='ESM3-small-open', cat='zeroshot', color='#7f8c8d', ls='-',
             label='ESM3-small-open\nunmasked', key='msrrel@0:comb', draw=True),
        dict(name='ProteinMPNN', cat='zeroshot', color='#34495e', ls='-',
             label='ProteinMPNN', key='pmpnn', draw=True),
        dict(name='ProSST', cat='zeroshot', color='#8e44ad', ls='-',
             label='ProSST', key='prosst', draw=True),
        dict(name=pub.BEST_LABEL[best], cat='supervised', color=pub.BEST_COLOR[best], ls='-',
             label=pub.BEST_LABEL[best], key=best, draw=True),
        dict(name='Rosetta', cat='supervised', color='#1abc9c', ls='-',
             label='Rosetta', key='rosetta', draw=True),
    ]
    return [c for c in cols if c['key'] in fr.preds]


# ----------------------------------------------------------------------------------------------
# hypergeometric & order-statistic exact random expectations per domain
# ----------------------------------------------------------------------------------------------
def _logc(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def hypergeom_p_atleast1(n, n_hits, k):
    """Exact P(at least 1 hit in k random draws without replacement from n items containing n_hits)."""
    k = min(k, n)
    if n_hits <= 0 or k <= 0:
        return 0.0
    if n - n_hits < k:
        return 1.0
    return float(1.0 - np.exp(_logc(n - n_hits, k) - _logc(n, k)))


def exact_random_best(y_sorted, k):
    """Exact E[max of k random picks without replacement] from sorted array y_sorted."""
    n = len(y_sorted)
    k = min(k, n)
    j = np.arange(1, n + 1)
    mask = j >= k
    logp = np.full(n, -np.inf)
    logp[mask] = _logc(j[mask] - 1, k - 1) - _logc(n, k)
    return float(np.sum(np.exp(logp) * y_sorted))


# ----------------------------------------------------------------------------------------------
# drawing helpers (1-bar, 2-bar, and 3-bar panels in the publication layout)
# ----------------------------------------------------------------------------------------------
def _draw_category_spans(ax, xs, spans, headers=True, cats=pub.CATS):
    ax.set_xlim(xs[0] - pub.XPAD[0], xs[-1] + pub.XPAD[1])
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2:
            ax.axvspan(x0 - 0.5 - (0.35 if i else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers:
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, cats[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')


def _bar_and_ref_legend(ax, sub_labels, ref_lines, ncol=2):
    handles = [
        Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=sub_labels[0]),
        Patch(fc='#999999', ec='black', label=sub_labels[1]),
    ] + list(ref_lines)
    ax.legend(handles=handles, ncol=ncol, **pub.LEGEND_KW)


def bars_single(ax, cols, T, subset, bar_label, ylabel, ylim, ref_lines=(), headers=False, fmt='{:.1f}'):
    """One bar per model column with 95% domain-bootstrap interval and value label."""
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = pub.layout(cols)
    w = 0.42
    top = 0.012 * (ylim[1] - ylim[0])
    for c, x in zip(cols, xs):
        r = T[(T.column == c['name']) & (T.subset == subset)]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.bar(x, r.value, w, color=c['color'], alpha=0.90, edgecolor='black', lw=0.8, zorder=3)
        ax.errorbar(x, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]],
                    fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.5, zorder=5)
        ax.text(x, max(r.hi, r.value, 0) + top, fmt.format(r.value), ha='center', va='bottom',
                fontsize=6.8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
    ax.set_xticks(xs)
    ax.set_xticklabels([c['label'] for c in cols], fontsize=pub.TICK_FS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    _draw_category_spans(ax, xs, spans, headers=headers)
    handles = [Patch(fc='#999999', ec='black', alpha=0.90, label=bar_label)] + list(ref_lines)
    ax.legend(handles=handles, ncol=len(handles), **pub.LEGEND_KW)


def bars_triple(ax, cols, T, subsets, sub_labels, ylabel, ylim, ref_lines=(), headers=True):
    """Three bars per model column (light to dark shade, as in fig5a) with 95% intervals."""
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = pub.layout(cols)
    w = 0.22
    centres = (-0.24, 0.0, 0.24)
    shades = (0.38, 0.68, 1.0)
    hatches = ('////', None, None)
    top = 0.012 * (ylim[1] - ylim[0])
    for c, x in zip(cols, xs):
        for j, (sub, off, sh, ht) in enumerate(zip(subsets, centres, shades, hatches)):
            r = T[(T.column == c['name']) & (T.subset == sub)]
            if not len(r):
                continue
            r = r.iloc[0]
            xx = x + off
            ax.bar(xx, r.value, w, color=c['color'], alpha=sh, edgecolor=c['color'] if j == 0 else 'black',
                   hatch=ht, lw=0.8, zorder=3)
            ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]],
                        fmt='none', ecolor='#c0392b', elinewidth=1.2, capsize=2.2, zorder=5)
            ax.text(xx, max(r.hi, r.value, 0) + top, f'{r.value:.2f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([c['label'] for c in cols], fontsize=pub.TICK_FS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in np.arange(0.0, 1.0001, 0.2)])
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    _draw_category_spans(ax, xs, spans, headers=headers)
    handles = [
        Patch(fc='#888888', ec='#666666' if j == 0 else 'black', alpha=sh, hatch=ht, label=lab)
        for j, (sh, ht, lab) in enumerate(zip(shades, hatches, sub_labels))
    ] + list(ref_lines)
    ax.legend(handles=handles, ncol=3, **pub.LEGEND_KW)


# ----------------------------------------------------------------------------------------------
# Figure 1: Precision, Recall @ 0 / 0.5, and Suggested Counts at predicted ddG > 0.5 kcal/mol
# ----------------------------------------------------------------------------------------------
def compute_suggested_threshold(fr, cols, pool_mask, libs, counts, tau_pred=0.5, tau_hit=0.5):
    """Per-domain and pooled precision, recall @ 0 / 0.5, and suggested counts for `pred > tau_pred`."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()
    n_libs = len(libs)

    dom_rows, pool_rows = [], []
    dom_vals, pool_reps = {}, {}

    base_hit_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > tau_hit).mean()) for lib in libs})
    base_stab_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > 0.0).mean()) for lib in libs})
    true_hits_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > tau_hit).sum()) for lib in libs})
    true_stab_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > 0.0).sum()) for lib in libs})

    for c in cols:
        arr = fr.preds[c['key']]
        n_seeds = arr.shape[0]
        n_sug_s = np.zeros((n_seeds, n_libs))
        n_hit_s = np.zeros((n_seeds, n_libs))
        n_stab_s = np.zeros((n_seeds, n_libs))
        n_true_hit_s = np.zeros((n_seeds, n_libs))
        n_true_stab_s = np.zeros((n_seeds, n_libs))

        for s in range(n_seeds):
            p = arr[s]
            for j, lib in enumerate(libs):
                m = pool_mask & (g == lib) & np.isfinite(p) & np.isfinite(y)
                sug = m & (p > tau_pred)
                n_sug_s[s, j] = sug.sum()
                n_hit_s[s, j] = (sug & (y > tau_hit)).sum()
                n_stab_s[s, j] = (sug & (y > 0.0)).sum()
                n_true_hit_s[s, j] = (m & (y > tau_hit)).sum()
                n_true_stab_s[s, j] = (m & (y > 0.0)).sum()

        # 1. Per-domain precision & recall (evaluated per seed, integrated across seeds and domains)
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            prec_hit_s = np.where(n_sug_s > 0, n_hit_s / n_sug_s, np.nan)
            prec_stab_s = np.where(n_sug_s > 0, n_stab_s / n_sug_s, np.nan)
            rec_hit_s = np.where(n_true_hit_s > 0, n_hit_s / n_true_hit_s, np.nan)
            rec_stab_s = np.where(n_true_stab_s > 0, n_stab_s / n_true_stab_s, np.nan)

            prec_hit_dom = pd.DataFrame(prec_hit_s.T, index=libs)
            prec_stab_dom = pd.DataFrame(prec_stab_s.T, index=libs)
            rec_hit_dom = pd.DataFrame(rec_hit_s.T, index=libs)
            rec_stab_dom = pd.DataFrame(rec_stab_s.T, index=libs)

        sug_count_dom = pd.DataFrame(n_sug_s.T, index=libs)
        hit_count_dom = pd.DataFrame(n_hit_s.T, index=libs)
        sug_total = float(sug_count_dom.mean(axis=1).sum())
        hit_total = float(hit_count_dom.mean(axis=1).sum())

        for sub, ser in (
            (f'precision_ddG_gt_{tau_hit:g}', prec_hit_dom),
            ('precision_ddG_gt_0', prec_stab_dom),
            (f'recall_ddG_gt_{tau_hit:g}', rec_hit_dom),
            ('recall_ddG_gt_0', rec_stab_dom),
            ('n_suggested_per_domain', sug_count_dom),
            ('n_hits_recovered_per_domain', hit_count_dom),
        ):
            pt, lo, hi, n_dom = pub.boot_mean(ser, libs, counts)
            dom_vals[(c['name'], sub)] = (c['key'], ser, pool_mask, y)
            dom_rows.append(dict(
                column=c['name'], key=c['key'], subset=sub, value=pt, lo=lo, hi=hi,
                n_libraries=n_dom, n_suggested_total=sug_total,
                n_hits_total=hit_total,
            ))

        # 2. Pooled precision & recall (ratio of domain sums per seed, bootstrapped over domains and seeds)
        sug_w = counts @ n_sug_s.T
        hit_w = counts @ n_hit_s.T
        stab_w = counts @ n_stab_s.T
        thit_w = counts @ n_true_hit_s.T
        tstab_w = counts @ n_true_stab_s.T
        seed_w = (np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
                  if n_seeds > 1 else np.ones((len(counts), 1), float))
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            rep_prec_hit = np.nansum(np.where(sug_w > 0, hit_w / sug_w, np.nan) * seed_w, axis=1)
            rep_prec_stab = np.nansum(np.where(sug_w > 0, stab_w / sug_w, np.nan) * seed_w, axis=1)
            rep_rec_hit = np.nansum(np.where(thit_w > 0, hit_w / thit_w, np.nan) * seed_w, axis=1)
            rep_rec_stab = np.nansum(np.where(tstab_w > 0, stab_w / tstab_w, np.nan) * seed_w, axis=1)

        pt_prec_hit = float(np.nanmean(n_hit_s.sum(axis=1) / np.maximum(n_sug_s.sum(axis=1), 1)))
        pt_prec_stab = float(np.nanmean(n_stab_s.sum(axis=1) / np.maximum(n_sug_s.sum(axis=1), 1)))
        pt_rec_hit = float(np.nanmean(n_hit_s.sum(axis=1) / np.maximum(n_true_hit_s.sum(axis=1), 1)))
        pt_rec_stab = float(np.nanmean(n_stab_s.sum(axis=1) / np.maximum(n_true_stab_s.sum(axis=1), 1)))

        for sub, pt, rep in (
            (f'precision_ddG_gt_{tau_hit:g}', pt_prec_hit, rep_prec_hit),
            ('precision_ddG_gt_0', pt_prec_stab, rep_prec_stab),
            (f'recall_ddG_gt_{tau_hit:g}', pt_rec_hit, rep_rec_hit),
            ('recall_ddG_gt_0', pt_rec_stab, rep_rec_stab),
        ):
            lo, hi = ci(rep)
            pool_reps[(sub, c['name'])] = rep
            pool_rows.append(dict(
                column=c['name'], key=c['key'], subset=sub, value=pt, lo=lo, hi=hi,
                n_libraries=int((sug_count_dom.mean(axis=1) > 0).sum() if sub.startswith('precision') else n_libs),
                n_suggested_total=sug_total,
                n_hits_total=hit_total,
            ))

    return (pd.DataFrame(dom_rows), pd.DataFrame(pool_rows), dom_vals, pool_reps,
            base_hit_dom, base_stab_dom, true_hits_dom, true_stab_dom)


def draw_suggested_figures(cols, T_dom, T_pool, base_hit_dom, base_stab_dom, true_hits_dom, true_stab_dom,
                           n_pool_total, tau_pred, tau_hit, sfx='', pool_title='single mutants'):
    drawn = [c for c in cols if c.get('draw', True)]
    p_hit = f'precision_ddG_gt_{tau_hit:g}'
    p_stab = 'precision_ddG_gt_0'
    r_hit = f'recall_ddG_gt_{tau_hit:g}'
    r_stab = 'recall_ddG_gt_0'

    bh_dom = float(base_hit_dom.mean())
    bs_dom = float(base_stab_dom.mean())
    tot_hits = int(true_hits_dom.sum())
    tot_stab = int(true_stab_dom.sum())
    n_dom_hits = int((true_hits_dom > 0).sum())
    bh_pool = float(tot_hits / n_pool_total)
    bs_pool = float(tot_stab / n_pool_total)

    foot = [
        f'Suggested = {pool_title} with predicted ddG > {tau_pred:g} kcal/mol (n = {n_pool_total:,} across 33 test domains; '
        f'{tot_hits:,} have measured ddG > {tau_hit:g} kcal/mol in {n_dom_hits} domains, {tot_stab:,} have ddG > 0 in 33 domains).'
    ]

    # --- 1A: Single-panel precision bar charts (per-domain mean & pooled) ---
    for T, tag, yl, bh, bs, desc in (
        (T_dom[T_dom.subset.isin([p_stab, p_hit])], '',
         'Precision among suggested\n(mean over domains)', bh_dom, bs_dom,
         'mean over domains with ≥ 1 suggested variant'),
        (T_pool[T_pool.subset.isin([p_stab, p_hit])], '_pooled',
         'Precision among suggested\n(pooled over domains)', bh_pool, bs_pool,
         'pooled over domains'),
    ):
        fig, ax = plt.subplots(figsize=(pub.fig_width(drawn), 4.9))
        T_plot = T.assign(n_variants=T['n_suggested_total'].round().astype(int))
        sub_labs = ['stabilizing: measured ddG > 0', f'exceeds threshold: measured ddG > {tau_hit:g} kcal/mol']
        pub.bars(ax, drawn, T_plot, [p_stab, p_hit], sub_labs, yl, (0, 1.15))
        ax.axhline(bs, color='#555555', ls=':', lw=1.1, zorder=2)
        ax.axhline(bh, color='#c0392b', ls='--', lw=1.1, zorder=2)
        _bar_and_ref_legend(ax, sub_labs, [
            Line2D([0], [0], color='#555555', ls=':', lw=1.1, label=f'base rate ddG > 0 ({bs:.2f})'),
            Line2D([0], [0], color='#c0392b', ls='--', lw=1.1, label=f'base rate ddG > {tau_hit:g} ({bh:.3f})'),
        ], ncol=2)
        pub.finish(
            fig,
            f'{SPLIT} test, {pool_title}: fraction of suggested variants (predicted > {tau_pred:g} kcal/mol) exceeding threshold ({desc}); '
            f'95% intervals from resampling domains',
            foot,
        )
        style.save(fig, OUT / f'fig_suggested_hit_rate{tag}{sfx}')

    # --- 1B: Three-panel figure (Top: Precision; Middle: Recall @ 0 / 0.5; Bottom: Single bar of suggested counts) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(pub.fig_width(drawn), 11.2), sharex=True)

    # Top panel: Precision among suggested (pred > 0.5)
    T1 = T_dom[T_dom.subset.isin([p_stab, p_hit])].assign(n_variants=T_dom['n_suggested_total'].round().astype(int))
    labs_p = ['precision @ ddG > 0 (stabilizing)', f'precision @ ddG > {tau_hit:g} kcal/mol']
    pub.bars(ax1, drawn, T1, [p_stab, p_hit], labs_p,
             'Precision among suggested\n(mean over domains)', (0, 1.15), headers=True)
    ax1.axhline(bs_dom, color='#555555', ls=':', lw=1.1, zorder=2)
    ax1.axhline(bh_dom, color='#c0392b', ls='--', lw=1.1, zorder=2)
    _bar_and_ref_legend(ax1, labs_p, [
        Line2D([0], [0], color='#555555', ls=':', lw=1.1, label=f'base rate ddG > 0 ({bs_dom:.2f})'),
        Line2D([0], [0], color='#c0392b', ls='--', lw=1.1, label=f'base rate ddG > {tau_hit:g} ({bh_dom:.3f})'),
    ], ncol=2)

    # Middle panel: Recall @ 0 / 0.5 among suggested (pred > 0.5)
    T2 = T_dom[T_dom.subset.isin([r_stab, r_hit])].assign(n_variants=T_dom['n_suggested_total'].round().astype(int))
    labs_r = [
        f'recall of true ddG > 0 variants (suggested at pred > {tau_pred:g})',
        f'recall of true ddG > {tau_hit:g} kcal/mol variants (suggested at pred > {tau_pred:g})',
    ]
    pub.bars(ax2, drawn, T2, [r_stab, r_hit], labs_r,
             'Recall at pred > 0.5\n(mean over domains)', (0, 0.78), headers=False)
    ax2.set_yticks(np.arange(0.0, 0.71, 0.2))
    _bar_and_ref_legend(ax2, labs_r, [], ncol=2)

    # Bottom panel: Single bar per model of suggested counts per domain (pred > 0.5)
    T3 = T_dom[T_dom.subset == 'n_suggested_per_domain']
    ymax3 = float(T3[T3.column.isin([c['name'] for c in drawn])]['hi'].max()) * 1.25
    oracle_hits = float(true_hits_dom.mean())
    bars_single(
        ax3, drawn, T3, 'n_suggested_per_domain',
        f'suggested variants per domain (predicted ddG > {tau_pred:g} kcal/mol)',
        'Suggested variants\nper domain (mean)', (0, ymax3),
        ref_lines=[
            Line2D([0], [0], color='#2c3e50', ls='--', lw=1.2,
                   label=f'true ddG > {tau_hit:g} kcal/mol variants per domain ({oracle_hits:.1f}, total {tot_hits:,})'),
        ],
        headers=False, fmt='{:.1f}',
    )
    ax3.set_yticks(np.arange(0, ymax3 + 1, 50))
    ax3.axhline(oracle_hits, color='#2c3e50', ls='--', lw=1.2, zorder=2)

    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: precision, recall, and count of variants suggested at predicted ddG > {tau_pred:g} kcal/mol; '
        f'95% intervals from resampling domains',
        foot,
    )
    style.save(fig, OUT / f'fig_suggested_hit_rate_with_counts{sfx}')


# ----------------------------------------------------------------------------------------------
# Figure 2: Probability of recovering top-N (N=6) most stabilizing mutants under budgets K=6, 24, 96
# ----------------------------------------------------------------------------------------------
def compute_topn_recovery(fr, cols, pool_mask, libs, counts, top_n=6, budgets=(6, 24, 96)):
    """Per-domain recovery of the top_n most stabilizing single mutants under fixed screening budgets."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()

    dom_info = {}
    for lib in libs:
        idx = np.flatnonzero(pool_mask & (g == lib) & np.isfinite(y))
        yd = y[idx]
        topn_local = set(np.argsort(-yd, kind='stable')[:top_n])
        dom_info[lib] = (idx, yd, topn_local, len(yd))

    frac_rows, any1_rows = [], []
    frac_vals, any1_vals = {}, {}
    rand_frac, rand_any1 = {}, {}

    for K in budgets:
        b_label = f'K = {K}'
        rf = pd.Series({lib: min(K, dom_info[lib][3]) / float(dom_info[lib][3]) for lib in libs})
        ra = pd.Series({lib: hypergeom_p_atleast1(dom_info[lib][3], top_n, K) for lib in libs})
        rand_frac[b_label] = float(rf.mean())
        rand_any1[b_label] = float(ra.mean())

        for c in cols:
            arr = fr.preds[c['key']]
            f_dom, a_dom = {}, {}
            for lib in libs:
                idx, yd, topn_local, n_d = dom_info[lib]
                k_eff = min(K, n_d)
                f_seeds, a_seeds = [], []
                for p in arr:
                    pd_ = p[idx]
                    if not np.isfinite(pd_).all():
                        continue
                    picks = set(np.argsort(-pd_, kind='stable')[:k_eff])
                    ov = len(picks & topn_local)
                    f_seeds.append(ov / float(top_n))
                    a_seeds.append(float(ov >= 1))
                if f_seeds:
                    f_dom[lib] = f_seeds
                    a_dom[lib] = a_seeds

            ser_f = pd.DataFrame.from_dict(f_dom, orient='index', dtype=float)
            ser_a = pd.DataFrame.from_dict(a_dom, orient='index', dtype=float)
            pt_f, lo_f, hi_f, n_f = pub.boot_mean(ser_f, libs, counts)
            pt_a, lo_a, hi_a, n_a = pub.boot_mean(ser_a, libs, counts)

            frac_vals[(c['name'], b_label)] = (c['key'], ser_f, pool_mask, y)
            any1_vals[(c['name'], b_label)] = (c['key'], ser_a, pool_mask, y)

            frac_rows.append(dict(
                column=c['name'], key=c['key'], subset=b_label, budget=K, top_n=top_n,
                metric='fraction_of_topN_recovered', value=pt_f, lo=lo_f, hi=hi_f,
                random_expectation=rand_frac[b_label], n_libraries=n_f, n_variants=int(pool_mask.sum()),
            ))
            any1_rows.append(dict(
                column=c['name'], key=c['key'], subset=b_label, budget=K, top_n=top_n,
                metric='p_recover_at_least_1_of_topN', value=pt_a, lo=lo_a, hi=hi_a,
                random_expectation=rand_any1[b_label], n_libraries=n_a, n_variants=int(pool_mask.sum()),
            ))

    nth_vals = [np.sort(dom_info[lib][1])[-top_n] for lib in libs]
    return pd.DataFrame(frac_rows), pd.DataFrame(any1_rows), frac_vals, any1_vals, rand_frac, rand_any1, nth_vals


def draw_topn_figures(cols, T_frac, T_any1, rand_frac, rand_any1, nth_vals, top_n, budgets,
                      sfx='', pool_title='single mutants'):
    drawn = [c for c in cols if c.get('draw', True)]
    b_subs = [f'K = {K}' for K in budgets]
    sub_labs = [f'budget K = {K}' for K in budgets]
    ref_styles = [('#777777', ':'), ('#2c3e50', '-.'), ('#c0392b', '--')]
    b_tag = '_'.join(f'k{K}' for K in budgets)

    foot = [
        f'Top-{top_n} most stabilizing {pool_title} per domain across the 33 test domains '
        f'({top_n}th-best ddG per domain averages {np.mean(nth_vals):+.2f} kcal/mol, min {np.min(nth_vals):+.2f} kcal/mol).'
    ]

    def _add_rand(ax, rdict):
        lines = []
        for K, b_s, (col_h, ls_h) in zip(budgets, b_subs, ref_styles):
            ax.axhline(rdict[b_s], color=col_h, ls=ls_h, lw=1.1, zorder=2)
            lines.append(Line2D([0], [0], color=col_h, ls=ls_h, lw=1.1, label=f'random K = {K} ({rdict[b_s]:.3f})'))
        return lines

    # --- 2A: Per-mutant recovery probability (fraction of top-6 recovered per protein) ---
    fig, ax = plt.subplots(figsize=(pub.fig_width(drawn), 5.0))
    ref_f = [Line2D([0], [0], color=col_h, ls=ls_h, lw=1.1, label=f'random K = {K} ({rand_frac[b_s]:.3f})')
             for K, b_s, (col_h, ls_h) in zip(budgets, b_subs, ref_styles)]
    bars_triple(
        ax, drawn, T_frac, b_subs, sub_labs,
        f'P(top-{top_n} mutant recovered)\n= fraction of top-{top_n} found per domain',
        (0, 1.15), ref_lines=ref_f, headers=True,
    )
    _add_rand(ax, rand_frac)
    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: probability of recovering a top-{top_n} most stabilizing mutant per protein under fixed screening budgets '
        f'(K = {", ".join(str(k) for k in budgets)}); 95% intervals from resampling domains',
        foot,
    )
    style.save(fig, OUT / f'fig_top{top_n}_recovery_{b_tag}{sfx}')

    # --- 2B: At-least-one recovery probability P(recover >= 1 of top-6) ---
    fig, ax = plt.subplots(figsize=(pub.fig_width(drawn), 5.0))
    ref_a = [Line2D([0], [0], color=col_h, ls=ls_h, lw=1.1, label=f'random K = {K} ({rand_any1[b_s]:.3f})')
             for K, b_s, (col_h, ls_h) in zip(budgets, b_subs, ref_styles)]
    bars_triple(
        ax, drawn, T_any1, b_subs, sub_labs,
        f'P(recover ≥ 1 of top-{top_n} per domain)',
        (0, 1.24), ref_lines=ref_a, headers=True,
    )
    _add_rand(ax, rand_any1)
    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: probability of recovering at least one of the top-{top_n} most stabilizing mutants per protein '
        f'(budgets K = {", ".join(str(k) for k in budgets)}); 95% intervals from resampling domains',
        foot,
    )
    style.save(fig, OUT / f'fig_top{top_n}_recovery_atleast1_{b_tag}{sfx}')

    # --- 2C: Two-panel combined figure (Top: fraction of top-6 recovered; Bottom: P(recover >= 1 of top-6)) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(pub.fig_width(drawn), 8.6), sharex=True)
    bars_triple(
        ax1, drawn, T_frac, b_subs, sub_labs,
        f'Fraction of top-{top_n} recovered\nper domain',
        (0, 1.15), ref_lines=ref_f, headers=True,
    )
    _add_rand(ax1, rand_frac)
    bars_triple(
        ax2, drawn, T_any1, b_subs, sub_labs,
        f'P(recover ≥ 1 of top-{top_n})\nper domain',
        (0, 1.25), ref_lines=ref_a, headers=False,
    )
    _add_rand(ax2, rand_any1)
    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: top-{top_n} recovery per protein under screening budgets K = {", ".join(str(k) for k in budgets)}; '
        f'95% intervals from resampling domains',
        foot,
    )
    style.save(fig, OUT / f'fig_top{top_n}_recovery_both_{b_tag}{sfx}')


# ----------------------------------------------------------------------------------------------
# Figure 3: Best ddG label recovered among top K per-protein (K = 1 .. 96 on x axis, no markers)
# ----------------------------------------------------------------------------------------------
def compute_best_ddg_curve(fr, cols, pool_mask, libs, counts, ks):
    """Per-domain best measured ddG among top K predicted single mutants for each K in ks."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()

    dom_y = {}
    for lib in libs:
        idx = np.flatnonzero(pool_mask & (g == lib) & np.isfinite(y))
        dom_y[lib] = (idx, y[idx], np.sort(y[idx]))

    rows = []
    vals = {}

    for K in ks:
        orc_ser = pd.Series({lib: float(dom_y[lib][2][-1]) for lib in libs})
        rnd_ser = pd.Series({lib: exact_random_best(dom_y[lib][2], K) for lib in libs})
        for name, key, ser in (('Oracle (best in domain)', 'oracle', orc_ser),
                               ('Random picks', 'random', rnd_ser)):
            pt, lo, hi, n_d = pub.boot_mean(ser, libs, counts)
            rows.append(dict(column=name, key=key, K=K, value=pt, lo=lo, hi=hi, n_libraries=n_d))

    for c in cols:
        arr = fr.preds[c['key']]
        seed_sorted_y = {}
        for lib in libs:
            idx, yd, _ = dom_y[lib]
            s_list = []
            for p in arr:
                pd_ = p[idx]
                if np.isfinite(pd_).all():
                    order = np.argsort(-pd_, kind='stable')
                    s_list.append(np.maximum.accumulate(yd[order]))
            seed_sorted_y[lib] = np.array(s_list, dtype=float) if s_list else None

        for K in ks:
            b_dom = {}
            for lib in libs:
                curves = seed_sorted_y[lib]
                if curves is not None:
                    b_dom[lib] = curves[:, min(K, curves.shape[1]) - 1]
            ser = pd.DataFrame.from_dict(b_dom, orient='index', dtype=float)
            pt, lo, hi, n_d = pub.boot_mean(ser, libs, counts)
            vals[(c['name'], f'K = {K}')] = (c['key'], ser, pool_mask, y)
            rows.append(dict(column=c['name'], key=c['key'], K=K, value=pt, lo=lo, hi=hi, n_libraries=n_d))

    return pd.DataFrame(rows), vals


def _plot_best_ddg_ax(ax, D, models_to_draw, ks_ticks, ribbon_names, title=None):
    ks = sorted(D['K'].unique())
    orc = D[D.column == 'Oracle (best in domain)'].sort_values('K')
    rnd = D[D.column == 'Random picks'].sort_values('K')
    ax.plot(orc.K, orc.value, color='#2c3e50', ls=':', lw=1.6, label='Oracle (best in domain)', zorder=2)
    ax.fill_between(orc.K, orc.lo, orc.hi, color='#2c3e50', alpha=0.07, lw=0, zorder=1)
    ax.plot(rnd.K, rnd.value, color='#c0392b', ls='--', lw=1.4, label='Random picks (exact expectation)', zorder=2)
    ax.fill_between(rnd.K, rnd.lo, rnd.hi, color='#c0392b', alpha=0.08, lw=0, zorder=1)

    ax.axhline(0.0, color='#888888', lw=0.7, ls=':', zorder=1)
    for b_k in (6, 24, 96):
        if b_k in ks:
            ax.axvline(b_k, color='#cccccc', lw=0.8, ls='--', zorder=1)

    for c in models_to_draw:
        sub = D[D.column == c['name']].sort_values('K')
        is_m = c['name'].startswith('ESM-MSR')
        lw = 2.2 if is_m else 1.6
        z = 6 if is_m else 4
        # Single continuous curve over every discrete budget K = 1 .. 96, no markers
        ax.plot(sub.K, sub.value, color=c['color'], ls=c.get('ls', '-'), lw=lw, label=c['name'], zorder=z)
        if c['name'] in ribbon_names:
            ax.fill_between(sub.K, sub.lo, sub.hi, color=c['color'], alpha=0.11, lw=0, zorder=z - 1)

    ax.set_xscale('log')
    ax.set_xticks(ks_ticks)
    ax.set_xticklabels([str(k) for k in ks_ticks])
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Variants screened per protein (K)')
    ax.set_ylabel('Best measured ddG in top K (kcal/mol)')
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    if title:
        ax.set_title(title, loc='left', fontsize=9.8)


def draw_best_ddg_figures(cols, D, ks_ticks, sfx='', pool_title='single mutants'):
    drawn = [c for c in cols if c.get('draw', True)]

    # --- 3A: Two-panel figure separating Retrained on MegaScale vs Zero-shot & Supervised ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    retrained = [c for c in drawn if c['cat'] == 'retrained']
    others = [c for c in drawn if c['name'].startswith('ESM-MSR') or c['cat'] in ('zeroshot', 'supervised')]
    _plot_best_ddg_ax(
        ax1, D, retrained, ks_ticks,
        ribbon_names={c['name'] for c in retrained},
        title=f'A. Retrained on {SPLIT}',
    )
    _plot_best_ddg_ax(
        ax2, D, others, ks_ticks,
        ribbon_names={c['name'] for c in others if c['name'].startswith('ESM-MSR') or c['cat'] == 'supervised'},
        title='B. ESM-MSR vs Zero-shot & Supervised (not Megascale-trained)',
    )
    ax2.set_ylabel('')
    ax1.legend(loc='lower right', ncol=2, fontsize=7.6, framealpha=0.92)
    ax2.legend(loc='lower right', ncol=2, fontsize=7.6, framealpha=0.92)
    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: best measured ddG recovered among top K per protein (K = 1 .. 96; 95% domain-bootstrap bands)',
        [],
    )
    style.save(fig, OUT / f'fig_best_ddg_vs_k_panels{sfx}')

    # --- 3B: Single-panel figure with all models ---
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    _plot_best_ddg_ax(
        ax, D, drawn, ks_ticks,
        ribbon_names={c['name'] for c in drawn if c['name'].startswith('ESM-MSR')},
    )
    ax.legend(loc='lower right', ncol=2, fontsize=7.4, framealpha=0.92)
    pub.finish(
        fig,
        f'{SPLIT} test, {pool_title}: best measured ddG recovered among top K per protein (K = 1 .. 96)',
        [],
    )
    style.save(fig, OUT / f'fig_best_ddg_vs_k{sfx}')


# ----------------------------------------------------------------------------------------------
# run one variant pool (all single mutants, or excluding mutations to cysteine)
# ----------------------------------------------------------------------------------------------
def run_pool(fr, cols, pool_mask, libs, counts, a, sfx, pool_title):
    n_pool_total = int(pool_mask.sum())
    print(f'\n================ {pool_title} ({sfx or "default"}): {n_pool_total:,} variants in {len(libs)} domains ================')

    # --- 1. Precision, recall @ 0 / 0.5, and suggested counts at predicted ddG > 0.5 ---
    T_dom, T_pool, dom_vals, pool_reps, bh_dom, bs_dom, th_dom, ts_dom = compute_suggested_threshold(
        fr, cols, pool_mask, libs, counts, tau_pred=a.tau, tau_hit=a.tau
    )
    T_dom.to_csv(OUT / f'fig_suggested_hit_rate{sfx}.csv', index=False, float_format='%.4f')
    T_pool.to_csv(OUT / f'fig_suggested_hit_rate_pooled{sfx}.csv', index=False, float_format='%.4f')
    p_subs = [f'precision_ddG_gt_{a.tau:g}', 'precision_ddG_gt_0', f'recall_ddG_gt_{a.tau:g}', 'recall_ddG_gt_0']
    p_dom = pub.compare_grouped(fr, dom_vals, p_subs + ['n_suggested_per_domain'], counts, libs)
    pub.add_holm(pd.DataFrame(p_dom)).to_csv(OUT / f'fig_suggested_hit_rate{sfx}_pvalues.csv', index=False, float_format='%.4g')
    p_pool = pub.compare_pooled(T_pool, pool_reps, p_subs)
    pub.add_holm(pd.DataFrame(p_pool)).to_csv(OUT / f'fig_suggested_hit_rate_pooled{sfx}_pvalues.csv', index=False, float_format='%.4g')
    draw_suggested_figures(
        cols, T_dom, T_pool, bh_dom, bs_dom, th_dom, ts_dom,
        n_pool_total, a.tau, a.tau, sfx=sfx, pool_title=pool_title,
    )

    # --- 2. Probability of recovering top-N (N=6) most stabilizing mutants under budgets K=6, 24, 96 ---
    T_frac, T_any1, frac_vals, any1_vals, rand_frac, rand_any1, nth_vals = compute_topn_recovery(
        fr, cols, pool_mask, libs, counts, top_n=a.top_n, budgets=a.budgets
    )
    b_subs = [f'K = {K}' for K in a.budgets]
    b_tag = '_'.join(f'k{K}' for K in a.budgets)
    f_prefix = f'fig_top{a.top_n}_recovery_{b_tag}{sfx}'
    T_frac.to_csv(OUT / f'{f_prefix}.csv', index=False, float_format='%.4f')
    T_any1.to_csv(OUT / f'fig_top{a.top_n}_recovery_atleast1_{b_tag}{sfx}.csv', index=False, float_format='%.4f')
    p_frac = pub.compare_grouped(fr, frac_vals, b_subs, counts, libs)
    pub.add_holm(pd.DataFrame(p_frac)).to_csv(OUT / f'{f_prefix}_pvalues.csv', index=False, float_format='%.4g')
    p_any1 = pub.compare_grouped(fr, any1_vals, b_subs, counts, libs)
    pub.add_holm(pd.DataFrame(p_any1)).to_csv(
        OUT / f'fig_top{a.top_n}_recovery_atleast1_{b_tag}{sfx}_pvalues.csv', index=False, float_format='%.4g'
    )
    draw_topn_figures(cols, T_frac, T_any1, rand_frac, rand_any1, nth_vals, a.top_n, a.budgets, sfx=sfx, pool_title=pool_title)

    # --- 3. Best ddG label recovered among top K per-protein (K = 1 .. 96) ---
    ks_all = list(range(1, max(KS_TICKS) + 1))
    D_all, best_vals = compute_best_ddg_curve(fr, cols, pool_mask, libs, counts, ks=ks_all)
    D_mark = D_all[D_all.K.isin(KS_TICKS)].copy()
    D_mark.to_csv(OUT / f'fig_best_ddg_vs_k{sfx}.csv', index=False, float_format='%.4f')
    D_all.to_csv(OUT / f'fig_best_ddg_vs_k_full_1to96{sfx}.csv', index=False, float_format='%.4f')
    p_best = pub.compare_grouped(fr, best_vals, [f'K = {K}' for K in KS_TICKS], counts, libs)
    pub.add_holm(pd.DataFrame(p_best)).to_csv(OUT / f'fig_best_ddg_vs_k{sfx}_pvalues.csv', index=False, float_format='%.4g')
    draw_best_ddg_figures(cols, D_all, KS_TICKS, sfx=sfx, pool_title=pool_title)

    order = [c['name'] for c in cols if c.get('draw', True)]
    print('\n--- 1. Suggested variants (pred > 0.5): per-domain mean precision, recall & counts ---')
    print(T_dom.pivot(index='column', columns='subset', values='value').reindex(order).round(3).to_string())
    print(f'\n--- 2. Top-{a.top_n} recovery probability (fraction of top-{a.top_n} recovered per domain) ---')
    print(T_frac.pivot(index='column', columns='subset', values='value').reindex(order).round(3).to_string())
    print(f'\n--- 2b. P(recover >= 1 of top-{a.top_n} per domain) ---')
    print(T_any1.pivot(index='column', columns='subset', values='value').reindex(order).round(3).to_string())
    print('\n--- 3. Best ddG recovered among top K per protein (kcal/mol) ---')
    print(D_mark.pivot(index='column', columns='K', values='value')
          .reindex(order + ['Oracle (best in domain)', 'Random picks']).round(3).to_string())


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=B_DEFAULT, help='domain bootstrap resamples (draft default: 200)')
    ap.add_argument('--tau', type=float, default=0.5, help='ddG threshold in kcal/mol (default: 0.5)')
    ap.add_argument('--top_n', type=int, default=6, help='N most stabilizing variants per protein to recover (default: 6)')
    ap.add_argument('--budgets', nargs='+', type=int, default=[6, 24, 96], help='fixed screening budgets K (default: 6 24 96)')
    a = ap.parse_args()

    fr = load_frame(DS)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    to_cys = (sp['mut'].astype(str).str[-1] == 'C').to_numpy()
    single_nocys = single & ~to_cys

    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    rng = np.random.default_rng(5)
    counts = rng.multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=a.B).astype(float)

    best, _ = pub.best_other(fr)
    cols = single_columns(fr, best)
    pub.NAMES.update({c['name']: c['name'] for c in cols})

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 20)

    for mask, sfx, title in (
        (single, '', 'single mutants'),
        (single_nocys, '_nocys', 'single mutants (excluding mutations to Cys)'),
    ):
        run_pool(fr, cols, mask, libs, counts, a, sfx, title)

    print('\ndone ->', OUT)


if __name__ == '__main__':
    main()
