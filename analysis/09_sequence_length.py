#!/usr/bin/env python
"""How big are the proteins in the test sets, and does the size of a protein change how well a model does on it?

* ``length_summary.csv``  per dataset: number of proteins and variants, median / min / max wild-type sequence length
  (one length per protein; also the variant-weighted median).
* ``sequence_length_distribution``  the lengths themselves, one dot per protein, per dataset.
* ``length_vs_performance``  per-protein Spearman rho, and per-protein RMSE after the one-dataset recalibration of
  ``07_rmse_calibration.py``, against sequence length, for ESM-MSR (WT head for singles, combined for multi-mutant sets) and
  the retrained baselines, with the Spearman correlation between length and the per-protein metric and its protein-level
  bootstrap interval.  Pooled over the external stability sets (a protein in two sets counts twice); the calibration set
  S461 and its superset S669 are left out of the RMSE panels.

The full-sized proteins are the external stability sets (20-780 residues) and the single-protein DMS; the Tsuboyama libraries
(40-72) and Domainome domains (19-97) are small domains and are shown for contrast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import calib, style  # noqa: E402
from msr_analysis.common import has_multi, out_dir, primary_key  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import ALL_DATASETS, EXT12, LABEL, MODELS, TSU  # noqa: E402

OUT = out_dir('09_sequence_length')
CALIB_SET = 's461'
MODELS_LEN = ['msr', 'me1', 'me2', 'tmpnn', 'spurs']          # 'msr' resolves to the head the set calls for
MIN_N = 5


def protein_lengths(fr):
    sp = fr.spine[fr.spine['use']]
    g = sp.groupby('cluster')
    return g['seq_len'].mean(), g.size()


def summary():
    rows, per_prot = [], {}
    for ds in ALL_DATASETS:
        fr = load_frame(ds)
        L, n = protein_lengths(fr)
        sp = fr.spine[fr.spine['use']]
        per_prot[ds] = L
        rows.append(dict(dataset=ds, label=LABEL[ds], n_proteins=len(L), n_variants=int(len(sp)), median_len=float(L.median()),
                         min_len=float(L.min()), max_len=float(L.max()), q25=float(L.quantile(.25)), q75=float(L.quantile(.75)),
                         variant_weighted_median_len=float(sp['seq_len'].median()), n_multi=int((sp['n_mut'] > 1).sum())))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'length_summary.csv', index=False, float_format='%.1f')
    return d, per_prot


def fig_distribution(d, per_prot):
    order = EXT12 + ['DLG4_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'MYO_HUMAN_Kung_2025_display',
                     'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain', 'domainome', TSU[1]]
    names = {**LABEL, 'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4', 'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2'}
    fig, ax = plt.subplots(figsize=(11, 3.9))
    style.band_axis(ax, len(order))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(order):
        L = per_prot[ds].to_numpy()
        col = '#2c7fb8' if ds in EXT12 else ('#7f7f7f' if ds in ('domainome', TSU[1]) else '#d95f0e')
        ax.scatter(i + rng.uniform(-.28, .28, len(L)), L, s=7, color=col, alpha=.55, lw=0, zorder=3, rasterized=True)
        q = np.percentile(L, [25, 50, 75])
        ax.plot([i - .36, i + .36], [q[1]] * 2, color='k', lw=1.8, zorder=4)
        ax.plot([i, i], [q[0], q[2]], color='k', lw=0.8, zorder=4)
        ax.text(i, 900, f'{int(np.median(L))}\n({int(L.min())}–{int(L.max())})', ha='center', va='top', fontsize=7)
    ax.set_yscale('log'); ax.set_ylim(15, 1000)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([names[o] for o in order], rotation=30, ha='right')
    ax.set_ylabel('wild-type sequence length (residues)')
    ax.set_title('Protein sizes in the test sets (median and range above each column; black bar = median, line = IQR)', fontsize=10)
    style.save(fig, OUT / 'sequence_length_distribution')


def per_protein_metrics(ds, key, ab=None):
    """DataFrame (cluster, length, n, rho, rmse, rmse_raw) for one model on one dataset; predictions seed-averaged per metric."""
    fr = load_frame(ds)
    sp = fr.spine
    k = key if key in fr.preds else None
    if k is None:
        return None
    ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
    y = sp['y'].to_numpy()
    rows = []
    idx = np.flatnonzero(ok)
    cl = sp['cluster'].to_numpy()[idx]
    for c in pd.unique(cl):
        m = idx[cl == c]
        if len(m) < MIN_N:
            continue
        rho, rm, rr = [], [], []
        for p in fr.preds[k]:
            rho.append(spearmanr(p[m], y[m])[0])
            rr.append(calib.rmse(p[m], y[m]))
            if ab is not None:
                rm.append(calib.rmse(calib.apply(p[m], ab), y[m]))
        rows.append(dict(dataset=ds, cluster=c, length=float(sp['seq_len'].to_numpy()[m].mean()), n=len(m), rho=np.mean(rho),
                         rmse_raw=np.mean(rr), rmse=np.mean(rm) if rm else np.nan))
    return pd.DataFrame(rows)


def calibration_for(key_family):
    """(scale, bias) fitted on the sacrificial set for the head the family reads there (seed-averaged prediction)."""
    fr = load_frame(CALIB_SET)
    k = primary_key(fr) if key_family == 'msr' else key_family
    sp = fr.spine
    ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
    return calib.fit_scale_bias(fr.preds[k].mean(axis=0)[ok], sp['y'].to_numpy()[ok])


def boot_len_corr(df, col, B=2000, seed=0):
    """Spearman between length and a per-protein metric; bootstrap over (protein, dataset) rows grouped by protein cluster."""
    d = df.dropna(subset=[col])
    r0 = spearmanr(d['length'], d[col])[0]
    cl = d['cluster'].astype(str).to_numpy()
    u, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(seed)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(u))]
    reps = []
    L, V = d['length'].to_numpy(), d[col].to_numpy()
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([rows_of[p] for p in pick])
        reps.append(spearmanr(L[ii], V[ii])[0])
    lo, hi = np.nanquantile(reps, [.025, .975])
    return r0, lo, hi, len(d)


def fig_performance(sets):
    ab = {m: calibration_for(m) for m in MODELS_LEN}
    allrows = []
    for key in MODELS_LEN:
        for ds in sets:
            fr = load_frame(ds)
            k = primary_key(fr) if key == 'msr' else key
            t = per_protein_metrics(ds, k, ab[key])
            if t is not None and len(t):
                t['model'] = key
                allrows.append(t)
    D = pd.concat(allrows, ignore_index=True)
    D.to_csv(OUT / 'per_protein_metrics.csv', index=False, float_format='%.4f')
    # figure: rows = models, cols = (rho, rmse)
    fig, axes = plt.subplots(len(MODELS_LEN), 2, figsize=(9.2, 2.15 * len(MODELS_LEN)), sharex=True)
    stats = []
    for r, key in enumerate(MODELS_LEN):
        d = D[D.model == key]
        lab = 'ESM-MSR (WT head / combined)' if key == 'msr' else MODELS[key]['label']
        col = MODELS['msr:wt']['color'] if key == 'msr' else MODELS[key]['color']
        for c, (metric, yl) in enumerate((('rho', 'per-protein Spearman ρ'), ('rmse', 'per-protein RMSE (kcal/mol), recalibrated'))):
            ax = axes[r, c]
            dd = d if metric == 'rho' else d[~d.dataset.isin(['s461', 's669'])]
            ax.scatter(dd['length'], dd[metric], s=11, color=col, alpha=.5, lw=0, rasterized=True)
            ax.set_xscale('log')
            r0, lo, hi, n = boot_len_corr(dd, metric, B=1000)
            stats.append(dict(model=key, metric=metric, spearman_length=r0, lo=lo, hi=hi, n_pairs=n))
            clip = ''
            if metric == 'rmse':
                clip = f'; {int((dd[metric] > 8).sum())} above 8 not shown'
                ax.set_ylim(0, 8)
            ax.text(.99, 1.01, f'ρ(length, {metric}) = {r0:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {n}{clip}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
            if r == 0:
                ax.set_title(yl, pad=16)
    for ax in axes[-1]:
        ax.set_xlabel('wild-type sequence length (residues)')
    fig.tight_layout()
    style.save(fig, OUT / 'length_vs_performance')
    pd.DataFrame(stats).to_csv(OUT / 'length_vs_performance_stats.csv', index=False, float_format='%.4f')
    # summary dots
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    S = pd.DataFrame(stats)
    for c, metric in enumerate(('rho', 'rmse')):
        ax = axes[c]
        s = S[S.metric == metric].reset_index(drop=True)
        for i, row in s.iterrows():
            col = MODELS['msr:wt']['color'] if row.model == 'msr' else MODELS[row.model]['color']
            ax.errorbar(row.spearman_length, i, xerr=[[row.spearman_length - row.lo], [row.hi - row.spearman_length]], fmt='o', color=col, capsize=2)
        ax.axvline(0, color='#888', lw=.8, ls=':')
        ax.set_xlabel(f'Spearman(length, per-protein {metric})')
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(['ESM-MSR' if m == 'msr' else MODELS[m]['label'] for m in s.model])
    fig.tight_layout()
    style.save(fig, OUT / 'length_correlation_summary')


def main():
    d, per_prot = summary()
    print(d[['dataset', 'n_proteins', 'n_variants', 'median_len', 'min_len', 'max_len']].to_string(index=False))
    fig_distribution(d, per_prot)
    fig_performance(EXT12)


if __name__ == '__main__':
    main()
