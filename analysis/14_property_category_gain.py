#!/usr/bin/env python
"""Rectified gain of a top-k screen by property category: the notebooks' ``analyze_features_precomputed`` figures on the Homology-Restricted MegaScale test set.

Two families, both on the single mutants of the 33 test domains (measured rows, all backbones), every model ranked within each domain and its top k taken (k = 24 and 96):

* ``gain_aa_stacked_k<k>``  stacked bars (notebook cells 15-16, ``plot_aa_preference_and_gain``): the rectified total gain max(ddG, 0), summed over the picks of all domains, split by the amino acid
  the pick mutates to (grouped and shaded by property: hydrophobic, polar, positive, negative; cysteine on its own), next to the count of picks per amino acid.  Whiskers on the bar totals: 95% intervals
  from resampling domains.
* ``gain_<split>_k<k>``     distribution plots (notebook cells 18-23, ``compare_performance_per_group`` with ``net_stabilization``): per model and property category, one dot per domain with a half
  violin and the interquartile box; the coloured dash is the mean over domains (value written, 95% domain-bootstrap interval as the black whisker), the dark dash the median.  Top row: rectified gain
  of the picks the model calls stabilizing (predicted ddG > 0), summed per domain; bottom row: picks per domain.  The leftmost panel is every variant of the domain ("Total"); dashed lines are the mean of
  the oracle ("Top k per Domain": the k most stabilizing measured variants).  Splits: ``asa_hydrophobicity`` (relative ASA 0.5 x change in Kyte-Doolittle hydrophobicity 0; k = 24 and 96), ``volume``
  (change in side-chain volume, cut points -1 and 1) and ``conservation`` (% conservation of the position in the MSA, 25 and 75; positions with more than 2 sequences), k = 96.

Features are the notebooks' (``tsuboyama_all_subs_corrected_feats.csv``, joined on library + mutation; the labels of the joined rows equal the spine's exactly).  A multi-seed model is scored per seed and the
per-domain value averaged over seeds, so every interval is the domain interval.  Columns: ESM-MSR (baseline) and ESM-MSR WT heads (``--heads wt comb`` adds the combined heads), the retrained baselines'
singles models (ME stage 1, ThermoMPNN, SPURS), ESM3-small-open unmasked, ProteinMPNN, ProSST, the better of GeoDDG-3D / DDGemb and Rosetta.  Every distribution plot is made under two rules for what counts as gain: the notebook's (only picks the model predicts stabilizing, predicted ddG > 0; ProSST left out, ``--prosst`` keeps it) and every top-k pick
(files with the suffix ``_allpicks``, every model including ProSST); ``gain_rule_comparison`` sets the two side by side.  Defaults are draft bootstrap sizes (``--B 200``).
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

pub = importlib.import_module('11_publication_figures')

OUT = out_dir('08_screening') / 'property_gain'
OUT.mkdir(parents=True, exist_ok=True)
DS = 'tsuboyama_capped_test'
FEATURES = Path('~/PSLMs/data/features/tsuboyama_all_subs_corrected_feats.csv').expanduser()
SPLIT = pub.SPLIT
MIN_N = 10
CATS = {'oracle': 'Oracle', **pub.CATS}

PALETTE = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']        # the notebook's category colours
AA_GROUPS = {'Hydrophobic': (['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G'], plt.cm.Blues), 'Polar': (['S', 'T', 'Y', 'N', 'Q'], plt.cm.Greens),
             'Positive': (['R', 'H', 'K'], plt.cm.Reds), 'Negative': (['D', 'E'], plt.cm.Purples)}
AA_ORDER = ['C'] + [a for g, _ in AA_GROUPS.values() for a in g]  # cysteine first, so it stacks at the bottom
AA_COLOR = {'C': '#FFD700'}
for _aas, _cmap in AA_GROUPS.values():
    for _aa, _c in zip(_aas, _cmap(np.linspace(0.35, 0.9, len(_aas)))):
        AA_COLOR[_aa] = _c


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_features(ids):
    """The notebook's per-substitution features laid out on the spine's ids (NaN where the join misses)."""
    cols = ['code', 'mut_type', 'rel_ASA', 'kdh_wt', 'kdh_mut', 'vol_wt', 'vol_mut', 'conservation', 'n_seqs']
    f = pd.read_csv(FEATURES, usecols=cols, low_memory=False)
    f['uid'] = (f['code'].astype(str) + '_' + f['mut_type'].astype(str)).str.replace('|', '_', regex=False)
    f = f.drop_duplicates('uid').set_index('uid')
    f['d_hydrophobicity'] = f['kdh_mut'] - f['kdh_wt']
    f['d_volume'] = f['vol_mut'] - f['vol_wt']
    return f.reindex(ids)[['rel_ASA', 'd_hydrophobicity', 'd_volume', 'conservation', 'n_seqs']].reset_index(drop=True)


def best_other(fr, single):
    g = fr.spine['group'].to_numpy()
    sc = {}
    for k in ('geoddg', 'ddgemb'):
        if k not in fr.preds:
            continue
        v = []
        for lib in np.unique(g[single]):
            m = single & (g == lib) & np.isfinite(fr.preds[k]).all(axis=0)
            if m.sum() >= MIN_N:
                v.append(np.nanmean([spearmanr(p[m], fr.spine['y'].to_numpy()[m])[0] for p in fr.preds[k]]))
        sc[k] = float(np.nanmean(v))
    return max(sc, key=sc.get)


def columns(fr, single, msr='p0', heads=('wt', 'comb')):
    """Model columns in the publication bar-chart order and styling, defaulting to only the p0 MSR model (WT head and combined)."""
    best = best_other(fr, single)
    msr_specs = [
        ('msr', 'ESM-MSR (baseline)', 'ESM-MSR\n(baseline)', {'wt': '#7f8c8d', 'comb': '#95a5a6'}),
        ('msrdev', 'ESM-MSR', 'ESM-MSR', {'wt': '#000000', 'comb': '#555555'}),
        ('msrp0', 'ESM-MSR (p0)', 'ESM-MSR\n(p0)', {'wt': '#6c3483', 'comb': '#9b59b6'}),
    ]
    want_fams = {'p0': {'msrp0'}, 'devel': {'msrdev'}, 'baseline': {'msr'}, 'all': {'msr', 'msrdev', 'msrp0'}}[msr]
    cols = []
    for fam, name_pref, lab_pref, h_colors in msr_specs:
        if fam not in want_fams:
            continue
        for h in heads:
            h_name = 'WT head' if h == 'wt' else 'combined'
            cols.append(dict(
                name=f'{name_pref} {h_name}',
                cat='retrained',
                color=h_colors[h],
                label=f'{lab_pref}\n{h_name}',
                key=f'{fam}:{h}',
            ))
    cols += [
        dict(name='MutateEverything st1', cat='retrained', color='#1b8a4b', label='MutateEverything\nst1', key='me1'),
        dict(name='ThermoMPNN', cat='retrained', color='#1f5f8b', label='ThermoMPNN', key='tmpnn'),
        dict(name='SPURS', cat='retrained', color='#d35400', label='SPURS', key='spurs'),
        dict(name='ESM3-small-open', cat='zeroshot', color='#7f8c8d', label='ESM3-small-open\nunmasked', key='msrrel@0:comb'),
        dict(name='ProteinMPNN', cat='zeroshot', color='#34495e', label='ProteinMPNN', key='pmpnn'),
        dict(name='ProSST', cat='zeroshot', color='#8e44ad', label='ProSST', key='prosst'),
        dict(name=pub.BEST_LABEL[best], cat='supervised', color=pub.BEST_COLOR[best], label=pub.BEST_LABEL[best], key=best),
        dict(name='Rosetta', cat='supervised', color='#1abc9c', label='Rosetta', key='rosetta'),
    ]
    return [c for c in cols if c['key'] in fr.preds], best


def with_oracle_col(cols, k):
    """Prepend 'Top k per Domain' at the far left of the model columns."""
    orc = dict(name=f'Top {k} per Domain', cat='oracle', color='#2c3e50', label=f'Top {k}\nper domain', key='oracle')
    return [orc] + list(cols)


def _draw_category_spans(ax, xs, spans, headers=True, cats=CATS, xpad=pub.XPAD):
    ax.set_xlim(xs[0] - xpad[0], xs[-1] + xpad[1])
    for cat, (x0, x1) in spans.items():
        if cat in ('oracle', 'zeroshot'):
            left_pad = 0.5 if cat == 'oracle' else 0.85
            right_pad = 0.85
            ax.axvspan(x0 - left_pad, x1 + right_pad, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers and cat in cats:
            tr_ax = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr_ax, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, cats[cat], transform=tr_ax, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def selection(score, y, cat, ncat, doms, k, positive_only):
    """Per domain and category: the rectified gain and the number of the top-k picks.  ``score`` is (n,); NaN rows cannot be picked."""
    G = np.zeros((len(doms), ncat))
    N = np.zeros((len(doms), ncat))
    for d, idx in enumerate(doms):
        ii = idx[np.isfinite(score[idx])]
        o = ii[np.argsort(-score[ii], kind='stable')[:k]]
        rect = np.maximum(y[o], 0.0)
        keep = score[o] > 0 if positive_only else np.ones(len(o), bool)
        G[d] = np.bincount(cat[o][keep], weights=rect[keep], minlength=ncat)
        N[d] = np.bincount(cat[o], minlength=ncat)
    return G, N


def model_stats(fr, cols, y, cat, ncat, doms, k, positive_only):
    """{column name: (gain_seeds, picks_seeds)} of shape (n_seeds, domains, categories) evaluated per seed without ensemble averaging; 'Top k per Domain' is the oracle."""
    g_orc, n_orc = selection(y, y, cat, ncat, doms, k, positive_only)
    out = {f'Top {k} per Domain': (g_orc[None, :, :], n_orc[None, :, :])}
    for c in cols:
        g = [selection(p, y, cat, ncat, doms, k, positive_only) for p in fr.preds[c['key']]]
        out[c['name']] = (np.array([a for a, _ in g], dtype=float), np.array([b for _, b in g], dtype=float))
    return out


def total_stats(y, cat, ncat, doms):
    G = np.zeros((len(doms), ncat))
    N = np.zeros((len(doms), ncat))
    for d, idx in enumerate(doms):
        G[d] = np.bincount(cat[idx], weights=np.maximum(y[idx], 0.0), minlength=ncat)
        N[d] = np.bincount(cat[idx], minlength=ncat)
    return G[None, :, :], N[None, :, :]


def boot_mean_ci(V, counts):
    """V (n_seeds x domains x cells) or (domains x cells): mean over domains and seeds with 95% interval integrating shared domain resamples ``counts`` and seed resamples."""
    V3 = V[None, :, :] if V.ndim == 2 else V
    n_seeds = V3.shape[0]
    w_dom = counts / counts.sum(axis=1, keepdims=True)
    dom_reps = np.einsum('bd,sdc->bsc', w_dom, V3)
    if n_seeds == 1:
        reps = dom_reps[:, 0, :]
    else:
        seed_w = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
        reps = np.einsum('bs,bsc->bc', seed_w, dom_reps)
    lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
    return V3.mean(axis=(0, 1)), lo, hi


# ----------------------------------------------------------------------------------------------
# distribution plots
# ----------------------------------------------------------------------------------------------
def tr(y):
    """Symmetric-log-like axis: linear up to 1, decades above (the notebooks' symlog with linthresh 1)."""
    y = np.asarray(y, float)
    return np.where(y <= 1, y, 1 + np.log10(np.maximum(y, 1.0)))


def ticks(top):
    v = [0, 1] + [10 ** i for i in range(1, 5) if 10 ** i <= top * 1.0001]
    return tr(v), [f'{x:g}' for x in v]


def rain(ax, x0, V_seeds, color, mean, lo, hi, width=0.82, rng=None, text=True, fs=5.8):
    """One distribution: full interquartile box, jittered domain dots, and median line with horizontal annotation text right over it.
    ``V_seeds`` is (n_seeds, n_domains) or (n_domains,); statistics are computed per seed and averaged across seeds without ensemble averaging."""
    V2 = V_seeds[None, :] if V_seeds.ndim == 1 else V_seeds
    t_dom = tr(V2.mean(axis=0))
    c = x0
    box_w = width * 0.78
    q1, med, q3 = np.mean(np.quantile(tr(V2), [0.25, 0.5, 0.75], axis=1), axis=1)
    ax.add_patch(plt.Rectangle((c - box_w / 2, q1), box_w, q3 - q1, fc=color, alpha=0.22, ec='none', zorder=1))
    jit = (rng.random(len(t_dom)) - 0.5) * (box_w * 0.70) + c
    ax.plot(jit, t_dom, 'o', ms=1.8, color=color, alpha=0.8, mew=0, zorder=3)
    ax.plot([c - box_w / 2, c + box_w / 2], [med, med], color='#444444', lw=1.6, zorder=5)
    if text:
        ax.text(
            c, med + 0.035,
            f'{mean:.2f}' if mean < 100 else f'{mean:.0f}',
            rotation=0, ha='center', va='bottom', fontsize=fs, zorder=8,
            bbox=dict(fc='white', ec='none', alpha=0.65, pad=0.3),
        )


def distribution_figure(path, dcols, stats_k, total, counts, cat_labels, k, gain_label_total='Rectified total gain (kcal/mol)', note='', include_picks=True):
    """stats_k: {model_name: (gain_seeds, picks_seeds)}; total: (gain_seeds, picks_seeds) of every variant.
    When ``include_picks=True``: two rows (gain, picks), two columns (Total, models).
    When ``include_picks=False``: single row (gain only, no lower panel), two columns (Total, models)."""
    ncat = len(cat_labels)
    mcols = with_oracle_col(dcols, k)
    xs, spans = pub.layout(mcols)
    rng = np.random.default_rng(3)
    colors = PALETTE[:ncat]
    slot = 1.0
    cw = slot / (ncat + 0.5)
    x_units = xs[-1] - xs[0] + sum(pub.XPAD)
    fig_w = 2.2 + 1.25 * x_units
    row_specs = [('gain', gain_label_total, f'Rectified gain @ k={k} (kcal/mol)')]
    if include_picks:
        row_specs.append(('picks', 'Total # substitutions per domain', 'Predicted most stabilizing per domain'))
    n_rows = len(row_specs)
    fig_h = 9.2 if include_picks else 5.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    if include_picks:
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1.15, x_units],
            height_ratios=[1, 0.82],
            hspace=0.07, wspace=0.11,
            left=0.055, right=0.992, top=0.935, bottom=0.19,
        )
    else:
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[1.15, x_units],
            wspace=0.11,
            left=0.055, right=0.992, top=0.895, bottom=0.29,
        )
    axs = {(r, c): fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(2)}
    table = []
    for r, (what, ylab_t, ylab_m) in enumerate(row_specs):
        idx = 0 if what == 'gain' else 1
        # the oracle's means, as the dashed lines
        omean, _, _ = boot_mean_ci(stats_k[mcols[0]['name']][idx], counts)
        tmean, tlo, thi = boot_mean_ci(total[idx], counts)
        top = 0.0

        # Left column: Total
        fs_txt = 5.6 if ncat >= 4 else 6.0
        ax_t = axs[r, 0]
        V_tot = total[idx]
        for c_idx in range(ncat):
            x0 = (c_idx - (ncat - 1) / 2) * cw
            rain(ax_t, x0, V_tot[:, :, c_idx], colors[c_idx], tmean[c_idx], tlo[c_idx], thi[c_idx], width=cw * 0.95, rng=rng, text=True, fs=fs_txt)
            top = max(top, float(tr(thi[c_idx])), float(tr(V_tot[:, :, c_idx]).max()))
            med_tot = float(np.mean(np.median(V_tot[:, :, c_idx], axis=1)))
            table.append(dict(statistic=what, model='Total', category=cat_labels[c_idx], mean=tmean[c_idx], lo=tlo[c_idx], hi=thi[c_idx], median=med_tot, n_domains=V_tot.shape[1]))
        ax_t.set_xlim(-0.55, 0.55)

        # Right column: Top k per domain + models in publication bar-chart order and layout
        ax_m = axs[r, 1]
        for mc, x_col in zip(mcols, xs):
            m = mc['name']
            V_m = stats_k[m][idx]
            mean, lo, hi = boot_mean_ci(V_m, counts)
            for c_idx in range(ncat):
                x0 = x_col + (c_idx - (ncat - 1) / 2) * cw
                rain(ax_m, x0, V_m[:, :, c_idx], colors[c_idx], mean[c_idx], lo[c_idx], hi[c_idx], width=cw * 0.95, rng=rng, text=True, fs=fs_txt)
                top = max(top, float(tr(hi[c_idx])), float(tr(V_m[:, :, c_idx]).max()))
                med_m = float(np.mean(np.median(V_m[:, :, c_idx], axis=1)))
                table.append(dict(statistic=what, model=m, category=cat_labels[c_idx], mean=mean[c_idx], lo=lo[c_idx], hi=hi[c_idx], median=med_m, n_domains=V_m.shape[1]))
        _draw_category_spans(ax_m, xs, spans, headers=(r == 0))

        for ax in (ax_t, ax_m):
            tk, tl = ticks(10 ** (top - 1) if top > 1 else 1)
            ax.set_yticks(tk)
            ax.set_yticklabels(tl)
            ax.set_ylim(-0.28, top + 0.65)
            ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
            for c_idx in range(ncat):
                ax.axhline(tr(omean[c_idx]), color=colors[c_idx], ls='--', lw=0.9, alpha=0.9, zorder=1)
        ax_t.set_ylabel(ylab_t)
        ax_m.set_ylabel(ylab_m)
    if include_picks:
        for c_col in (0, 1):
            axs[0, c_col].set_xticks([])
    bot_r = n_rows - 1
    axs[bot_r, 0].set_xticks([0])
    axs[bot_r, 0].set_xticklabels(['Total'], fontsize=pub.TICK_FS + 0.5)
    axs[bot_r, 1].set_xticks(xs)
    axs[bot_r, 1].set_xticklabels([mc['label'] for mc in mcols], fontsize=pub.TICK_FS)
    fig.legend(
        handles=[Patch(fc=colors[c_idx], alpha=0.8, label=cat_labels[c_idx]) for c_idx in range(ncat)],
        loc='lower center', ncol=2 if ncat == 4 else ncat,
        bbox_to_anchor=(0.5, 0.045 if include_picks else 0.06), fontsize=9.5, frameon=False,
    )
    if note:
        fig.text(0.055, 0.008 if include_picks else 0.012, note, fontsize=7.5, va='bottom', ha='left', color='#444444')
    style.save(fig, path)
    return pd.DataFrame(table)


# ----------------------------------------------------------------------------------------------
# stacked bars
# ----------------------------------------------------------------------------------------------
def stacked_figure(path, dcols, G, N, counts, k, note=''):
    """G, N: {model: (n_seeds x domains x 20 amino acids)} of rectified gain / picks (columns in AA_ORDER). Sum over domains, whiskers from the domain & seed bootstrap of the total."""
    mcols = with_oracle_col(dcols, k)
    xs, spans = pub.layout(mcols)
    x = np.array(xs, dtype=float)
    x_units = xs[-1] - xs[0] + sum(pub.XPAD)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.05 * x_units, 6.0))
    rows = []
    for ax, D, ylab in ((ax1, G, 'Rectified total gain (kcal/mol)'), (ax2, N, 'Mutation count')):
        # Per-model mean across seeds of the domain-summed per-AA values
        tot = np.array([D[mc['name']].mean(axis=0).sum() for mc in mcols])
        ci_bounds = {}
        for mc in mcols:
            m = mc['name']
            # D[m] is (n_seeds, n_domains, 20); sum over AAs -> (n_seeds, n_domains)
            dom_tot = D[m].sum(axis=2)
            n_s = dom_tot.shape[0]
            dom_reps = counts @ dom_tot.T  # (B, n_seeds)
            if n_s == 1:
                reps = dom_reps[:, 0]
            else:
                seed_w = np.random.default_rng(17).multinomial(n_s, np.full(n_s, 1.0 / n_s), size=len(counts)).astype(float) / n_s
                reps = np.sum(dom_reps * seed_w, axis=1)
            ci_bounds[m] = np.quantile(reps, [0.025, 0.975])
        ymax = max(tot.max(), max(hi for _, hi in ci_bounds.values()))
        thr = 0.03 * tot.max()
        bottom = np.zeros(len(mcols))
        for j, aa in enumerate(AA_ORDER):
            v = np.array([D[mc['name']][:, :, j].mean(axis=0).sum() for mc in mcols])
            ax.bar(x, v, bottom=bottom, color=AA_COLOR[aa], edgecolor='white', linewidth=0.5, width=0.68, label=aa, zorder=3)
            for i in np.flatnonzero(v > thr):
                ax.text(x[i], bottom[i] + v[i] / 2, aa, ha='center', va='center', fontsize=7.2, fontweight='bold', zorder=4)
            bottom += v
        for i, mc in enumerate(mcols):
            m = mc['name']
            lo, hi = ci_bounds[m]
            ax.errorbar(x[i], tot[i], yerr=[[max(tot[i] - lo, 0)], [max(hi - tot[i], 0)]],
                        fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.8, zorder=6)
            ax.text(x[i], max(hi, tot[i]) + 0.012 * ymax, f'{tot[i]:.0f}', ha='center', va='bottom', fontsize=7.5,
                    zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
            rows.append(dict(
                panel='gain' if D is G else 'picks', model=m, total=tot[i], lo=lo, hi=hi,
                **{aa: D[m][:, :, j].mean(axis=0).sum() for j, aa in enumerate(AA_ORDER)},
            ))
        ax.set_ylabel(ylab)
        ax.set_xticks(xs)
        ax.set_xticklabels([mc['label'] for mc in mcols], fontsize=pub.TICK_FS)
        ax.set_ylim(0, ymax * 1.12)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        _draw_category_spans(ax, xs, spans, headers=True)
    handles = [Patch(fc=AA_COLOR[aa], label=aa) for aa in AA_ORDER[::-1]]
    ax1.legend(handles=handles, loc='upper right', title='Amino acids', ncol=4, fontsize=7.8, handlelength=1.2, frameon=True, framealpha=0.9)
    if note:
        fig.text(0.012, 0.008, note, fontsize=7.5, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 0.94))
    style.save(fig, path)
    return pd.DataFrame(rows)


def comparison_figure(path, d, dcols):
    """Mean rectified gain per domain (summed over the categories of the ASA x hydrophobicity split), top 24 and 96, under the notebook's rule (picks predicted > 0) and with every pick."""
    d.to_csv(str(path) + '.csv', index=False, float_format='%.4f')
    ks = sorted(d['k'].unique())
    mcols_ref = with_oracle_col(dcols, ks[0])
    xs, _ = pub.layout(mcols_ref)
    x_units = xs[-1] - xs[0] + sum(pub.XPAD)
    fig, axs = plt.subplots(1, len(ks), figsize=(1.02 * x_units * len(ks), 5.0), sharey=False)
    for ax, k in zip(np.atleast_1d(axs), ks):
        mcols = with_oracle_col(dcols, k)
        xs_k, spans_k = pub.layout(mcols)
        x = np.array(xs_k, dtype=float)
        models = [mc['name'] for mc in mcols]
        sub = d[d['k'] == k]
        for off, rule, col, mk in ((-0.14, 'notebook', '#555555', 'o'), (0.14, 'all picks', '#c0392b', 'D')):
            r = sub[sub['rule'] == rule].set_index('model').reindex(models)
            ax.errorbar(
                x + off, r['mean_gain'],
                yerr=[r['mean_gain'] - r['lo'], r['hi'] - r['mean_gain']],
                fmt=mk, ms=5, color=col, capsize=2.2, lw=1.2, zorder=4,
                label={'notebook': 'picks predicted stabilizing (notebook rule)', 'all picks': 'every pick'}[rule],
            )
        ax.set_xticks(xs_k)
        ax.set_xticklabels([mc['label'] for mc in mcols], fontsize=pub.TICK_FS)
        ax.set_ylabel(f'Rectified gain @ k={k}, per domain (kcal/mol)')
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_ylim(0, float(sub['hi'].max()) * 1.15)
        _draw_category_spans(ax, xs_k, spans_k, headers=True)
    np.atleast_1d(axs)[0].legend(loc='upper right', fontsize=8.2, framealpha=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save(fig, path)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200, help='domain resamples (draft size)')
    ap.add_argument('--msr', choices=['p0', 'devel', 'baseline', 'all'], default='p0', help='which ESM-MSR model(s) to draw (default: p0 only)')
    ap.add_argument('--heads', nargs='+', choices=['wt', 'comb'], default=['wt', 'comb'], help='ESM-MSR heads drawn (default: wt comb)')
    ap.add_argument('--no-prosst', action='store_true', help='omit ProSST from the notebook-rule distribution plots')
    a = ap.parse_args()

    fr = load_frame(DS)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    y = sp['y'].to_numpy()
    feat = load_features(sp['id'].tolist())
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    group = sp['group'].to_numpy()
    cols, best = columns(fr, single, a.msr, a.heads)
    print(f'{len(libs)} test domains, {single.sum():,} singles; features present for {feat["rel_ASA"].notna()[single].mean():.1%}; columns: {[c["name"] for c in cols]} (best other ddG model: {best})')
    rng = np.random.default_rng(5)
    counts = rng.multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=a.B).astype(float)

    def domains(pool):
        return [np.flatnonzero(pool & (group == lib)) for lib in libs]

    # ---- stacked bars by the amino acid mutated to --------------------------------------------------
    aa = sp['mut'].astype(str).str[-1].to_numpy()
    aa_idx = np.array([AA_ORDER.index(c) if c in AA_ORDER else -1 for c in aa])
    pool_aa = single & (aa_idx >= 0)
    dom_aa = domains(pool_aa)
    for k in (24, 96):
        st = model_stats(fr, cols, y, np.where(aa_idx >= 0, aa_idx, 0), len(AA_ORDER), dom_aa, k, positive_only=False)
        models = list(st)
        t = stacked_figure(
            OUT / f'gain_aa_stacked_k{k}', cols,
            {m: st[m][0] for m in models}, {m: st[m][1] for m in models},
            counts, k,
            note=f'Top {k} of each of the {len(libs)} {SPLIT} test domains (single mutants); gain = sum of max(ddG, 0) over the picks, whiskers: 95% interval over domains (B = {a.B}).',
        )
        t.to_csv(OUT / f'gain_aa_stacked_k{k}.csv', index=False, float_format='%.4f')

    # ---- distribution plots by property category --------------------------------------------------
    f = feat
    asa, dh, dv, cons, nseq = (f[c].to_numpy() for c in ('rel_ASA', 'd_hydrophobicity', 'd_volume', 'conservation', 'n_seqs'))
    splits = [
        dict(name='asa_hydrophobicity', ks=(24, 96), pool=single & np.isfinite(asa) & np.isfinite(dh), cat=2 * (asa > 0.5) + (dh > 0),
             labels=['relative ASA ≤ 0.5 & Δ hydrophobicity ≤ 0', 'relative ASA ≤ 0.5 & Δ hydrophobicity > 0', 'relative ASA > 0.5 & Δ hydrophobicity ≤ 0', 'relative ASA > 0.5 & Δ hydrophobicity > 0']),
        dict(name='volume', ks=(96,), pool=single & np.isfinite(dv), cat=(dv > -1).astype(int) + (dv > 1), labels=['Δ volume ≤ −1', '−1 < Δ volume ≤ 1', 'Δ volume > 1']),
        dict(name='conservation', ks=(96,), pool=single & np.isfinite(cons) & np.isfinite(nseq) & (nseq > 2), cat=(cons > 25).astype(int) + (cons > 75),
             labels=['% conservation ≤ 25', '25 < % conservation ≤ 75', '% conservation > 75']),
    ]
    rules = [
        ('', True, [c for c in cols if not (a.no_prosst and c['key'] == 'prosst')], 'the picks predicted stabilizing (> 0)'),
        ('_allpicks', False, cols, 'every pick, whatever its predicted sign'),
    ]
    cmp_rows = []
    for s in splits:
        cat = np.nan_to_num(np.asarray(s['cat'], float)).astype(int)
        dom = [d for d in domains(s['pool']) if len(d)]            # domains with at least one variant that has the features (the designed domains have no MSA: no conservation)
        cnt = np.random.default_rng(5).multinomial(len(dom), np.full(len(dom), 1.0 / len(dom)), size=a.B).astype(float)
        tot = total_stats(y, cat, len(s['labels']), dom)
        for k in s['ks']:
            for suffix, positive_only, dcols, rule in rules:
                st = model_stats(fr, dcols, y, cat, len(s['labels']), dom, k, positive_only=positive_only)
                note = (f'Top {k} per {SPLIT} test domain ({len(dom)} domains with the feature, {int(s["pool"].sum()):,} single mutants); gain = sum of max(ddG, 0) over {rule}; '
                        f'boxes: interquartile range with median line; dots: domains; annotations: mean over domains; dashed lines: oracle mean.')
                t = distribution_figure(OUT / f'gain_{s["name"]}_k{k}{suffix}', dcols, st, tot, cnt, s['labels'], k, note=note, include_picks=True)
                t.to_csv(OUT / f'gain_{s["name"]}_k{k}{suffix}.csv', index=False, float_format='%.4f')
                distribution_figure(OUT / f'gain_{s["name"]}_k{k}{suffix}_gainonly', dcols, st, tot, cnt, s['labels'], k, note=note, include_picks=False)
                if s['name'] == 'asa_hydrophobicity':                 # the split that keeps all 33 domains: the two rules are compared on its category-summed gain, every model (ProSST too) under both
                    st_all = st if len(dcols) == len(cols) else model_stats(fr, cols, y, cat, len(s['labels']), dom, k, positive_only=True)
                    for m, (G, N) in st_all.items():
                        mean, lo, hi = boot_mean_ci(G.sum(axis=2, keepdims=True), cnt)
                        cmp_rows.append(dict(k=k, rule='notebook' if positive_only else 'all picks', model=m, mean_gain=mean[0], lo=lo[0], hi=hi[0], picks_counted=float(N.sum(axis=2).mean())))
            print(f'{s["name"]} k={k}: {len(dom)} domains')
    comparison_figure(OUT / 'gain_rule_comparison', pd.DataFrame(cmp_rows), cols)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
