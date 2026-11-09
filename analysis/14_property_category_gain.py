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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import ci  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

OUT = out_dir('08_screening') / 'property_gain'
OUT.mkdir(parents=True, exist_ok=True)
DS = 'tsuboyama_capped_test'
FEATURES = Path('~/PSLMs/data/features/tsuboyama_all_subs_corrected_feats.csv').expanduser()
SPLIT = 'Homology-Restricted MegaScale'
MIN_N = 10

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


def columns(fr, single, msr, heads):
    best = best_other(fr, single)
    names = {'geoddg': 'GeoDDG-3D', 'ddgemb': 'DDGemb'}
    cols = []
    for fam, label in (('msr', 'ESM-MSR (baseline)'), ('msrdev', 'ESM-MSR')):
        if (fam == 'msr' and msr == 'devel') or (fam == 'msrdev' and msr == 'baseline'):
            continue
        for h in heads:
            cols.append((f'{label}{" WT head" if h == "wt" else " combined"}', f'{fam}:{h}'))
    cols += [('MutateEverything st1', 'me1'), ('ThermoMPNN', 'tmpnn'), ('SPURS', 'spurs'), ('ESM3-small-open', 'msrrel@0:comb'),
             ('ProteinMPNN', 'pmpnn'), ('ProSST', 'prosst'), (names[best], best), ('Rosetta', 'rosetta')]
    return [(n, k) for n, k in cols if k in fr.preds], best


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
    """{column name: (gain, picks)} (domains x categories), seed means; 'Top k per Domain' is the oracle (the labels as scores)."""
    out = {f'Top {k} per Domain': selection(y, y, cat, ncat, doms, k, positive_only)}
    for name, key in cols:
        g = [selection(p, y, cat, ncat, doms, k, positive_only) for p in fr.preds[key]]
        out[name] = (np.mean([a for a, _ in g], axis=0), np.mean([b for _, b in g], axis=0))
    return out


def total_stats(y, cat, ncat, doms):
    G = np.zeros((len(doms), ncat))
    N = np.zeros((len(doms), ncat))
    for d, idx in enumerate(doms):
        G[d] = np.bincount(cat[idx], weights=np.maximum(y[idx], 0.0), minlength=ncat)
        N[d] = np.bincount(cat[idx], minlength=ncat)
    return G, N


def boot_mean_ci(V, counts):
    """V (domains x cells): mean over domains with the 95% interval under the shared domain resamples ``counts`` (B x domains)."""
    reps = counts @ V / counts.sum(axis=1, keepdims=True)
    lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
    return V.mean(axis=0), lo, hi


# ----------------------------------------------------------------------------------------------
# distribution plots
# ----------------------------------------------------------------------------------------------
def disp(name):
    """Tick label: the head on its own line."""
    return name.replace(' WT head', '\nWT head').replace(' combined', '\ncombined').replace(' (baseline)', '\n(baseline)')


def tr(y):
    """Symmetric-log-like axis: linear up to 1, decades above (the notebooks' symlog with linthresh 1)."""
    y = np.asarray(y, float)
    return np.where(y <= 1, y, 1 + np.log10(np.maximum(y, 1.0)))


def ticks(top):
    v = [0, 1] + [10 ** i for i in range(1, 5) if 10 ** i <= top * 1.0001]
    return tr(v), [f'{x:g}' for x in v]


def rain(ax, x0, V, color, mean, lo, hi, width=0.82, rng=None, text=True, fs=7):
    """One distribution: half violin on the left, interquartile box and dots on the right, mean dash (coloured) with its interval, median dash."""
    t = tr(V)
    c = x0
    if len(t) >= 3 and np.ptp(t) > 1e-9:
        try:
            kde = gaussian_kde(t)
            ys = np.linspace(t.min(), t.max(), 120)
            dens = kde(ys)
            dens = dens / dens.max() * width * 0.5
            ax.fill_betweenx(ys, c, c - dens, color=color, alpha=0.8, lw=0, zorder=2)
        except np.linalg.LinAlgError:
            pass
    q1, med, q3 = np.quantile(t, [0.25, 0.5, 0.75])
    ax.add_patch(plt.Rectangle((c + 0.02, q1), width * 0.24, q3 - q1, fc=color, alpha=0.22, ec='none', zorder=1))
    jit = (rng.random(len(t)) - 0.5) * width * 0.2 + c + width * 0.14
    ax.plot(jit, t, 'o', ms=1.8, color=color, alpha=0.8, mew=0, zorder=3)
    ax.plot([c + 0.01, c + width * 0.26], [med, med], color='#444444', lw=1.5, zorder=5)
    ax.plot([c - width * 0.22, c + 0.01], [tr(mean)] * 2, color=color, lw=2.4, zorder=6, solid_capstyle='butt')
    ax.plot([c + width * 0.13] * 2, [tr(lo), tr(hi)], color='black', lw=0.9, zorder=7)
    if text:
        ax.text(c + width * 0.02, tr(hi) + 0.05, f'{mean:.2f}' if mean < 100 else f'{mean:.0f}', rotation=90, ha='center', va='bottom', fontsize=fs, zorder=8)


def distribution_figure(path, fr_names, stats_k, total, counts, cat_labels, k, gain_label_total='Rectified total gain (kcal/mol)', note=''):
    """stats_k: {model: (gain, picks)}; total: (gain, picks) of every variant.  Two rows (gain, picks), two columns (Total, models)."""
    ncat = len(cat_labels)
    models = list(stats_k)
    rng = np.random.default_rng(3)
    colors = PALETTE[:ncat]
    slot = 1.0
    cw = slot / (ncat + 0.5)
    fig_w = 2.2 + 1.45 * len(models)
    fig = plt.figure(figsize=(fig_w, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, len(models) * 1.0], height_ratios=[1, 0.8], hspace=0.06, wspace=0.13, left=0.065, right=0.995, top=0.985, bottom=0.25)
    axs = {(r, c): fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)}
    table = []
    for r, (what, ylab_t, ylab_m) in enumerate((('gain', gain_label_total, f'Rectified gain @ k={k} (kcal/mol)'),
                                                 ('picks', 'Total # substitutions per domain', 'Predicted most stabilizing per domain'))):
        idx = 0 if what == 'gain' else 1
        # the oracle's means, as the dashed lines
        omean, _, _ = boot_mean_ci(stats_k[models[0]][idx], counts)
        tmean, tlo, thi = boot_mean_ci(total[idx], counts)
        top = 0.0
        panels = [(axs[r, 0], [('Total', total[idx], tmean, tlo, thi)]),
                  (axs[r, 1], [(m, stats_k[m][idx], *boot_mean_ci(stats_k[m][idx], counts)) for m in models])]
        for ax, items in panels:
            for i, (m, V, mean, lo, hi) in enumerate(items):
                for c in range(ncat):
                    x0 = i + (c - (ncat - 1) / 2) * cw
                    rain(ax, x0, V[:, c], colors[c], mean[c], lo[c], hi[c], width=cw * 0.95, rng=rng, text=True, fs=6.3)
                    top = max(top, tr(hi[c]), tr(V[:, c]).max())
                    table.append(dict(statistic=what, model=m, category=cat_labels[c], mean=mean[c], lo=lo[c], hi=hi[c], median=float(np.median(V[:, c])), n_domains=len(V)))
            ax.set_xlim(-0.55, len(items) - 0.45)
        for ax in (axs[r, 0], axs[r, 1]):
            tk, tl = ticks(10 ** (top - 1) if top > 1 else 1)
            ax.set_yticks(tk)
            ax.set_yticklabels(tl)
            ax.set_ylim(-0.28, top + 0.6)
            ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
            for c in range(ncat):
                ax.axhline(tr(omean[c]), color=colors[c], ls='--', lw=0.9, alpha=0.9, zorder=1)
        axs[r, 0].set_ylabel(ylab_t)
        axs[r, 1].set_ylabel(ylab_m)
    for c in (0, 1):
        axs[0, c].set_xticks([])
    axs[1, 0].set_xticks([0])
    axs[1, 0].set_xticklabels(['Total'], rotation=45, ha='right', fontsize=11)
    axs[1, 1].set_xticks(range(len(models)))
    axs[1, 1].set_xticklabels([disp(m) for m in models], rotation=45, ha='right', fontsize=10)
    fig.legend(handles=[Patch(fc=colors[c], alpha=0.8, label=cat_labels[c]) for c in range(ncat)], loc='lower center', ncol=2 if ncat == 4 else ncat, bbox_to_anchor=(0.5, 0.03),
               fontsize=10, frameon=False)
    if note:
        fig.text(0.065, 0.008, note, fontsize=7.5, va='bottom', ha='left', color='#444444')
    style.save(fig, path)
    return pd.DataFrame(table)


# ----------------------------------------------------------------------------------------------
# stacked bars
# ----------------------------------------------------------------------------------------------
def stacked_figure(path, models, G, N, counts, k, note=''):
    """G, N: {model: (domains x 20 amino acids)} of rectified gain / picks (columns in AA_ORDER).  Sum over domains, whiskers from the domain bootstrap of the total."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.2 + 1.0 * len(models) * 1.9, 6.2))
    x = np.arange(len(models))
    rows = []
    for ax, D, ylab in ((ax1, G, 'Rectified total gain (kcal/mol)'), (ax2, N, 'Mutation count')):
        tot = np.array([D[m].sum() for m in models])
        ymax = max(tot.max(), max(np.quantile(counts @ D[m].sum(axis=1), 0.975) for m in models))
        thr = 0.03 * tot.max()
        bottom = np.zeros(len(models))
        for j, aa in enumerate(AA_ORDER):
            v = np.array([D[m][:, j].sum() for m in models])
            ax.bar(x, v, bottom=bottom, color=AA_COLOR[aa], edgecolor='white', linewidth=0.5, width=0.8, label=aa)
            for i in np.flatnonzero(v > thr):
                ax.text(x[i], bottom[i] + v[i] / 2, aa, ha='center', va='center', fontsize=7.5, fontweight='bold')
            bottom += v
        for i, m in enumerate(models):
            reps = counts @ D[m].sum(axis=1)
            lo, hi = np.quantile(reps, [0.025, 0.975])
            ax.plot([x[i]] * 2, [lo, hi], color='black', lw=1.0, zorder=6)
            ax.plot([x[i] - 0.12, x[i] + 0.12], [lo, lo], color='black', lw=1.0, zorder=6)
            ax.plot([x[i] - 0.12, x[i] + 0.12], [hi, hi], color='black', lw=1.0, zorder=6)
            ax.text(x[i], hi + 0.01 * ymax, f'{tot[i]:.0f}', ha='center', va='bottom', fontsize=8)
            rows.append(dict(panel='gain' if D is G else 'picks', model=m, total=tot[i], lo=lo, hi=hi, **{aa: D[m][:, j].sum() for j, aa in enumerate(AA_ORDER)}))
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([disp(m) for m in models], rotation=45, ha='right')
        ax.set_ylim(0, ymax * 1.08)
    handles = [Patch(fc=AA_COLOR[aa], label=aa) for aa in AA_ORDER[::-1]]
    ax1.legend(handles=handles, loc='upper right', title='Amino acids', ncol=4, fontsize=8, handlelength=1.2, frameon=True, framealpha=0.9)
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, va='top', ha='left', color='#444444')
    fig.tight_layout()
    style.save(fig, path)
    return pd.DataFrame(rows)


def comparison_figure(path, d):
    """Mean rectified gain per domain (summed over the categories of the ASA x hydrophobicity split), top 24 and 96, under the notebook's rule (picks predicted > 0) and with every pick."""
    d.to_csv(str(path) + '.csv', index=False, float_format='%.4f')
    ks = sorted(d['k'].unique())
    fig, axs = plt.subplots(1, len(ks), figsize=(6.6 * len(ks), 4.6), sharey=False)
    for ax, k in zip(np.atleast_1d(axs), ks):
        sub = d[d['k'] == k]
        models = list(dict.fromkeys(sub['model']))
        x = np.arange(len(models))
        for off, rule, col, mk in ((-0.12, 'notebook', '#888888', 'o'), (0.12, 'all picks', '#c0392b', 'D')):
            r = sub[sub['rule'] == rule].set_index('model').reindex(models)
            ax.errorbar(x + off, r['mean_gain'], yerr=[r['mean_gain'] - r['lo'], r['hi'] - r['mean_gain']], fmt=mk, ms=5, color=col, capsize=2, lw=1, label={'notebook': 'picks predicted stabilizing (notebook rule)', 'all picks': 'every pick'}[rule])
        ax.set_xticks(x)
        ax.set_xticklabels([disp(m) for m in models], rotation=45, ha='right')
        ax.set_ylabel(f'Rectified gain @ k={k}, per domain (kcal/mol)')
        ax.grid(axis='y', color=style.GRID, lw=0.6)
        ax.set_ylim(bottom=0)
    np.atleast_1d(axs)[0].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    style.save(fig, path)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200, help='domain resamples (draft size)')
    ap.add_argument('--msr', choices=['both', 'baseline', 'devel'], default='both', help='which ESM-MSR triplicates to draw (baseline = release-baseline, devel = dc8-floorlink)')
    ap.add_argument('--heads', nargs='+', choices=['wt', 'comb'], default=['wt'], help='ESM-MSR heads drawn')
    ap.add_argument('--prosst', action='store_true', help='keep ProSST in the notebook-rule distribution plots (the _allpicks set always has it; its scores are mostly negative, so the predicted-stabilizing rule leaves it almost nothing)')
    a = ap.parse_args()

    fr = load_frame(DS)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    y = sp['y'].to_numpy()
    feat = load_features(sp['id'].tolist())
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    group = sp['group'].to_numpy()
    cols, best = columns(fr, single, a.msr, a.heads)
    print(f'{len(libs)} test domains, {single.sum():,} singles; features present for {feat["rel_ASA"].notna()[single].mean():.1%}; columns: {[n for n, _ in cols]} (best other ddG model: {best})')
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
        t = stacked_figure(OUT / f'gain_aa_stacked_k{k}', models, {m: st[m][0] for m in models}, {m: st[m][1] for m in models}, counts, k,
                           note=f'Top {k} of each of the {len(libs)} {SPLIT} test domains (single mutants); gain = sum of max(ddG, 0) over the picks, whiskers: 95% interval over domains (B = {a.B}).')
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
    # two rules for what counts as gain: the notebook's (only picks the model predicts stabilizing, predicted > 0; ProSST left out unless --prosst) and every top-k pick (suffix _allpicks; all models)
    rules = [('', True, [(n, k) for n, k in cols if a.prosst or k != 'prosst'], 'the picks predicted stabilizing (> 0)'),
             ('_allpicks', False, cols, 'every pick, whatever its predicted sign')]
    cmp_rows = []
    for s in splits:
        cat = np.nan_to_num(np.asarray(s['cat'], float)).astype(int)
        dom = [d for d in domains(s['pool']) if len(d)]            # domains with at least one variant that has the features (the designed domains have no MSA: no conservation)
        cnt = np.random.default_rng(5).multinomial(len(dom), np.full(len(dom), 1.0 / len(dom)), size=a.B).astype(float)
        tot = total_stats(y, cat, len(s['labels']), dom)
        for k in s['ks']:
            for suffix, positive_only, dcols, rule in rules:
                st = model_stats(fr, dcols, y, cat, len(s['labels']), dom, k, positive_only=positive_only)
                note = (f'Top {k} per {SPLIT} test domain ({len(dom)} domains with the feature, {int(s["pool"].sum()):,} single mutants); gain = sum of max(ddG, 0) over {rule}; dots: domains; '
                        f'coloured dash: mean with 95% interval over domains (B = {a.B}); dark dash: median; dashed lines: oracle mean.')
                t = distribution_figure(OUT / f'gain_{s["name"]}_k{k}{suffix}', dcols, st, tot, cnt, s['labels'], k, note=note)
                t.to_csv(OUT / f'gain_{s["name"]}_k{k}{suffix}.csv', index=False, float_format='%.4f')
                if s['name'] == 'asa_hydrophobicity':                 # the split that keeps all 33 domains: the two rules are compared on its category-summed gain, every model (ProSST too) under both
                    st_all = st if suffix else model_stats(fr, cols, y, cat, len(s['labels']), dom, k, positive_only=True)
                    for m, (G, N) in st_all.items():
                        mean, lo, hi = boot_mean_ci(G.sum(axis=1, keepdims=True), cnt)
                        cmp_rows.append(dict(k=k, rule='notebook' if positive_only else 'all picks', model=m, mean_gain=mean[0], lo=lo[0], hi=hi[0], picks_counted=float(N.sum(axis=1).mean())))
            print(f'{s["name"]} k={k}: {len(dom)} domains')
    comparison_figure(OUT / 'gain_rule_comparison', pd.DataFrame(cmp_rows))
    print('done ->', OUT)


if __name__ == '__main__':
    main()
