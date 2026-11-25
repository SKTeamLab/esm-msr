#!/usr/bin/env python3
"""Pipeline p08: Affine RMSE calibration transfer and sequence-length performance analysis.

Consolidates:
  - 07_rmse_calibration.py -> results/02_thermostability/rmse/{calibration_candidates{,_ssym}.csv, rmse_by_dataset{,_ssym}.csv,
                              calibration_transfer{,_ssym}.csv, rmse_recalibrated{,_ssym}.{png,pdf}}
  - 09_sequence_length.py  -> results/08_sequence_length/{length_summary.csv, sequence_length_distribution,
                              per_protein_metrics.csv, length_vs_performance, length_vs_performance_stats.csv,
                              length_correlation_summary}
                           + results/99_misc/dms3/{length_summary.csv, length_per_protein_metrics.csv}

Key methodological fixes vs legacy scripts:
  - Uses msr_analysis.calib.{calibration_key, fit_frame_scale_bias, calibrated_preds} so PTMUL and PTMUL-D fit and evaluate
    their deployed multi-mutant head (msrp0:comb) rather than hardcoding s461's single-mutant head (msrp0:wt) in calibration_transfer.
  - Resamples triplicate seeds jointly with protein clusters in the RMSE bootstrap.
  - Zero in-place mutation of registry.MODELS (uses style_overrides).
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from msr_analysis import calib, style
from msr_analysis.boot import ci, global_cluster_index
from msr_analysis.data import Frame, load_frame
from msr_analysis.paths import atomic_to_csv, results_dir
from msr_analysis.registry import DMS3, EXT12, LABEL, MAIN_BENCHMARK_DATASETS, MODELS, TSU_CAPPED

from . import base_parser, resolve_b

STAGE = 'calibration_length'
DESCRIPTION = 'Affine RMSE calibration transfer and sequence-length performance analysis'

RMSE_FAMILIES = ['msr', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
RMSE_SETS = [d for d in EXT12 if d != 's571']
MSR_STYLE_OV = {'msr': {'label': 'ESM-MSR', 'color': '#000000', 'marker': 'D', 'fill': True, 'group': 'msr', 'model': 'ESM-MSR', 'lw': 2.6}}

LEN_MODELS = ['msr', 'me1', 'tmpnn', 'spursm']
LEN_SETS = [d for d in EXT12 if d not in ('ptmul', 'ptmuld')]
LEN_LABEL = {'msr': 'ESM-MSR WT-view', 'me1': 'MutateEverything Stage 1', 'tmpnn': 'ThermoMPNN', 'spursm': 'SPURS-multi'}
MIN_PROT_N = 5


def _valid(fr: Frame, k: str) -> np.ndarray:
    return fr.spine['use'].to_numpy() & np.isfinite(fr.spine['y'].to_numpy()) & np.isfinite(fr.preds[k]).all(axis=0)


def _boot_sd(v: np.ndarray, cluster: np.ndarray, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    v, cl = np.asarray(v, float), np.asarray(cluster)
    u, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s1 = np.bincount(inv, v)
    s2 = np.bincount(inv, v * v)
    c = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1.0 / len(u)), size=B).astype(float)
    N, S1, S2 = c @ n, c @ s1, c @ s2
    sd = np.sqrt(np.maximum(S2 / N - (S1 / N) ** 2, 0.0))
    return tuple(float(x) for x in np.percentile(sd, [2.5, 97.5]))


def run_rmse_calibration_for_donor(donor: str, star_sets: set[str], B: int) -> None:
    out = results_dir('rmse')
    sfx = '' if donor == 's461' else f'_{donor}'
    all_frames = {d: load_frame(d) for d in EXT12}

    # 1. Calibration candidates table
    cand = []
    for d, fr in all_frames.items():
        k = calib.calibration_key('msr', fr)
        m = _valid(fr, k)
        y = fr.spine['y'].to_numpy()[m]
        p_mean = fr.preds[k].mean(0)[m]
        ab = calib.fit_scale_bias(p_mean, y)
        cand.append(dict(
            dataset=d, n=len(y), sd_y=float(y.std()),
            p05=float(np.percentile(y, 5)), p95=float(np.percentile(y, 95)),
            range_5_95=float(np.ptp(np.percentile(y, [5, 95]))),
            fit_scale=ab[0], fit_bias=ab[1],
            rho_in_set=float(spearmanr(p_mean, y)[0]),
        ))
    atomic_to_csv(pd.DataFrame(cand), out / f'calibration_candidates{sfx}.csv', index=False, float_format='%.3f')

    frames = {d: all_frames[d] for d in RMSE_SETS}
    names = {d: fr.spine['cluster'].to_numpy() for d, fr in frames.items()}
    gidx, ncl = global_cluster_index(names)
    rng = np.random.default_rng(3)
    counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=B).astype(float)
    seed_w = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0

    rows = []
    donor_fr = all_frames[donor]
    for fam in RMSE_FAMILIES:
        ab = calib.fit_frame_scale_bias(donor_fr, fam)
        if ab is None:
            continue
        for d, fr in frames.items():
            k = calib.calibration_key(fam, fr)
            if k is None or k not in fr.preds:
                continue
            m = _valid(fr, k)
            if m.sum() < 20:
                continue
            y = fr.spine['y'].to_numpy()[m]
            p_cal = calib.calibrated_preds(fr, fam, ab)[:, m]
            p_raw = np.atleast_2d(fr.preds[k])[:, m]
            e2_cal = (p_cal - y[None, :]) ** 2
            e2_raw = (p_raw - y[None, :]) ** 2
            g = gidx[d][m]
            res = {}
            ns = p_raw.shape[0]
            for tag, e2 in (('cal', e2_cal), ('raw', e2_raw)):
                point = float(np.mean(np.sqrt(e2.mean(1))))
                reps = np.empty(B, dtype=float)
                for b_i in range(B):
                    w = counts[b_i][g]
                    per_seed_rm = np.sqrt((e2 * w).sum(1) / w.sum())
                    reps[b_i] = float(np.dot(seed_w[b_i], per_seed_rm)) if ns == 3 else float(np.mean(per_seed_rm))
                res[tag] = (point, *ci(reps))
            rows.append(dict(
                model=fam, key=k, dataset=d,
                rmse=res['cal'][0], lo=res['cal'][1], hi=res['cal'][2],
                rmse_raw=res['raw'][0], lo_raw=res['raw'][1], hi_raw=res['raw'][2],
                n=int(m.sum()), sacrificial=(d == donor), contains_sacrificial=(d in star_sets),
            ))
    R = pd.DataFrame(rows)
    atomic_to_csv(R, out / f'rmse_by_dataset{sfx}.csv', index=False, float_format='%.4f')

    # Transfer matrix for ESM-MSR: fits dataset-appropriate head on donor row dfit, evaluates on target col dev!
    T = np.full((len(RMSE_SETS), len(RMSE_SETS)), np.nan)
    for i, dfit in enumerate(RMSE_SETS):
        ab_i = calib.fit_frame_scale_bias(frames[dfit], 'msr')
        for j, dev in enumerate(RMSE_SETS):
            fr_j = frames[dev]
            k_j = calib.calibration_key('msr', fr_j)
            m_j = _valid(fr_j, k_j)
            y_j = fr_j.spine['y'].to_numpy()[m_j]
            p_cal_j = calib.calibrated_preds(fr_j, 'msr', ab_i)[:, m_j]
            T[i, j] = float(np.mean([calib.rmse(p_s, y_j) for p_s in p_cal_j]))
    T_df = pd.DataFrame(T, index=[LABEL[d] for d in RMSE_SETS], columns=[LABEL[d] for d in RMSE_SETS])
    atomic_to_csv(T_df, out / f'calibration_transfer{sfx}.csv', float_format='%.3f')

    # Composite 4-panel calibration figure
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[1, 1.05], hspace=0.5, wspace=0.28)
    k_don = calib.calibration_key('msr', donor_fr)
    m_don = _valid(donor_fr, k_don)
    p_don = donor_fr.preds[k_don].mean(0)[m_don]
    y_don = donor_fr.spine['y'].to_numpy()[m_don]
    ab_don = calib.fit_scale_bias(p_don, y_don)

    axA = fig.add_subplot(gs[0, 0])
    style.density_scatter(axA, p_don, y_don, s=9, cmap='viridis', line=False)
    xs = np.array(axA.get_xlim())
    axA.plot(xs, xs, color='#999', lw=0.9, ls=':', label='y = x (as trained)')
    axA.plot(xs, ab_don[0] * xs + ab_don[1], color='#c0392b', lw=1.4, label=f'recalibrated: {ab_don[0]:.2f}·pred {ab_don[1]:+.2f}')
    axA.set_xlabel('ESM-MSR prediction, protease scale')
    axA.set_ylabel(f'measured ddG, {LABEL[donor]} (kcal/mol)')
    axA.legend(loc='lower right', fontsize=7.5)
    axA.set_title(f'A  Calibration set: {LABEL[donor]}*')

    axB = fig.add_subplot(gs[0, 1])
    sd_pred, sd_y, ci_pred, ci_y = [], [], [], []
    for d in RMSE_SETS:
        fd = frames[d]
        kd = calib.calibration_key('msr', fd)
        md = _valid(fd, kd)
        vp, vy, cl = fd.preds[kd].mean(0)[md], fd.spine['y'].to_numpy()[md], fd.spine['cluster'].to_numpy()[md]
        sd_pred.append(float(vp.std()))
        sd_y.append(float(vy.std()))
        ci_pred.append(_boot_sd(vp, cl, B=B))
        ci_y.append(_boot_sd(vy, cl, B=B))
    x = np.arange(len(RMSE_SETS))
    err_fn = lambda sds, cis: [[max(v - c[0], 0) for v, c in zip(sds, cis)], [max(c[1] - v, 0) for v, c in zip(sds, cis)]]
    axB.bar(x - 0.2, sd_y, 0.4, color='#999999', label='measured ddG', yerr=err_fn(sd_y, ci_y), error_kw=dict(ecolor=style.RING, elinewidth=1.1, capsize=2))
    axB.bar(x + 0.2, sd_pred, 0.4, color=MODELS['msrp0:wt']['color'], label='ESM-MSR (raw)', yerr=err_fn(sd_pred, ci_pred), error_kw=dict(ecolor=style.RING, elinewidth=1.1, capsize=2))
    axB.set_xticks(x)
    axB.set_xticklabels([LABEL[d] for d in RMSE_SETS], rotation=40, ha='right')
    axB.set_ylabel('standard deviation (kcal/mol)\n(whiskers: 95% bootstrap over proteins)')
    axB.legend()
    axB.set_title('B  Spread of raw vs measured ddG')

    axC = fig.add_subplot(gs[0, 2])
    im = axC.imshow(T, cmap='viridis_r', aspect='auto')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            axC.text(j, i, f'{T[i, j]:.1f}', ha='center', va='center', fontsize=6.5, color='white' if T[i, j] > np.nanmedian(T) else 'black')
    axC.set_xticks(range(len(RMSE_SETS)))
    axC.set_xticklabels([LABEL[d] for d in RMSE_SETS], rotation=60, ha='right', fontsize=7.5)
    axC.set_yticks(range(len(RMSE_SETS)))
    axC.set_yticklabels([LABEL[d] + ('*' if d == donor else '') for d in RMSE_SETS], fontsize=7.5)
    axC.set_xlabel('scored on')
    axC.set_ylabel('calibrated on')
    axC.set_title('C  RMSE hardly depends on the calibration set')
    fig.colorbar(im, ax=axC, fraction=0.04)

    axD = fig.add_subplot(gs[1, :])
    models = [f for f in RMSE_FAMILIES if f in set(R['model'])]
    plot_models = ['ESM-MSR' if f == 'msr' else f for f in models]
    style.dot_panel(
        axD, RMSE_SETS, plot_models,
        {(r.dataset, 'ESM-MSR' if r.model == 'msr' else r.model): r.rmse for r in R.itertuples()},
        {(r.dataset, 'ESM-MSR' if r.model == 'msr' else r.model): (r.lo, r.hi) for r in R.itertuples()},
        highlight=['ESM-MSR'], ylabel='RMSE (kcal/mol), recalibrated',
        xlabels=[LABEL[d] + ('*' if d in star_sets else '') for d in RMSE_SETS],
        ylim=(0.6, None),
    )
    for i, d in enumerate(RMSE_SETS):
        r = R[(R['dataset'] == d) & (R['model'] == 'msr')]
        if len(r):
            axD.plot(i - 0.36, r['rmse_raw'].iloc[0], marker='x', color='#d62728', ms=9, mew=2.2, ls='none', zorder=12)
    hs = [Line2D([0], [0], label='ESM-MSR', **style.marker_kwargs('ESM-MSR', 7))]
    hs += [Line2D([0], [0], label=MODELS[f]['label'], **style.marker_kwargs(f, 7)) for f in models if f != 'msr']
    hs += [Line2D([0], [0], marker='x', color='#d62728', ms=8, mew=2, ls='none', label='ESM-MSR before recalibration')]
    axD.legend(handles=hs, ncol=4, loc='upper left', fontsize=8)
    axD.set_title(f'D  * calibrated on {LABEL[donor]} (and sets containing it, starred): not out-of-sample.  Whiskers: 95% protein-level bootstrap (B = {B:,})')
    style.save(fig, out / f'rmse_recalibrated{sfx}')


# ----------------------------------------------------------------------------------------------
# 2. Sequence-length distributions & length vs performance (09_sequence_length.py)
# ----------------------------------------------------------------------------------------------
def _boot_len_corr(df: pd.DataFrame, col: str, B: int = 1000, seed: int = 0) -> tuple[float, float, float, int]:
    d = df.dropna(subset=[col])
    r0 = float(spearmanr(d['length'], d[col])[0])
    cl = d['cluster'].astype(str).to_numpy()
    u, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(seed)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(u))]
    L, V = d['length'].to_numpy(), d[col].to_numpy()
    reps = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([rows_of[p] for p in pick])
        reps.append(spearmanr(L[ii], V[ii])[0])
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return r0, float(lo), float(hi), len(d)


def run_sequence_length(B: int) -> None:
    out = results_dir('sequence_length')
    out_dms = results_dir('dms3')

    def _summarize_lengths(datasets: list[str]) -> tuple[list[dict], dict[str, pd.Series]]:
        rows_out, prot_map = [], {}
        for ds in datasets:
            fr = load_frame(ds)
            sp = fr.spine[fr.spine['use']]
            L = sp.groupby('cluster')['seq_len'].mean()
            prot_map[ds] = L
            rows_out.append(dict(
                dataset=ds, label=LABEL[ds], n_proteins=len(L), n_variants=int(len(sp)),
                median_len=float(L.median()), min_len=float(L.min()), max_len=float(L.max()),
                q25=float(L.quantile(0.25)), q75=float(L.quantile(0.75)),
                variant_weighted_median_len=float(sp['seq_len'].median()),
                n_multi=int((sp['n_mut'] > 1).sum()),
            ))
        return rows_out, prot_map

    rows_main, per_prot = _summarize_lengths(MAIN_BENCHMARK_DATASETS)
    atomic_to_csv(pd.DataFrame(rows_main), out / 'length_summary.csv', index=False, float_format='%.1f')
    rows_dms, _ = _summarize_lengths(DMS3)
    atomic_to_csv(pd.DataFrame(rows_dms), out_dms / 'length_summary.csv', index=False, float_format='%.1f')

    # Sequence-length distribution figure (main non-DMS3 benchmarks only)
    order = EXT12 + ['domainome', TSU_CAPPED[1]]
    fig_d, ax_d = plt.subplots(figsize=(10, 4.1))
    style.band_axis(ax_d, len(order))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(order):
        L = per_prot[ds].to_numpy()
        col = '#2c7fb8' if ds in EXT12 else '#7f7f7f'
        ax_d.scatter(i + rng.uniform(-0.28, 0.28, len(L)), L, s=7, color=col, alpha=0.55, lw=0, zorder=3, rasterized=True)
        q = np.percentile(L, [25, 50, 75])
        ax_d.plot([i - 0.36, i + 0.36], [q[1]] * 2, color='k', lw=1.8, zorder=4)
        ax_d.plot([i, i], [q[0], q[2]], color='k', lw=0.8, zorder=4)
        ax_d.text(i, 900, f'{int(np.median(L))}\n({int(L.min())}–{int(L.max())})', ha='center', va='top', fontsize=7)
    ax_d.set_yscale('log')
    ax_d.set_ylim(15, 1000)
    ax_d.set_xticks(range(len(order)))
    ax_d.set_xticklabels([LABEL[o] for o in order], rotation=30, ha='right')
    ax_d.set_ylabel('wild-type sequence length (residues)')
    ax_d.set_title('Protein sizes in the test sets (median and range above each column; black bar = median, line = IQR)', fontsize=10)
    fig_d.text(0.01, -0.02, 'Blue: external thermostability sets; grey: small-domain benchmarks (Domainome, MegaScale test).', fontsize=8, color='#444444')
    style.save(fig_d, out / 'sequence_length_distribution')

    # Per-protein Spearman & recalibrated RMSE vs sequence length
    s461_fr = load_frame('s461')
    ab_map = {}
    for m_fam in LEN_MODELS:
        k_c = calib.calibration_key(m_fam, s461_fr)
        ok_c = _valid(s461_fr, k_c)
        ab_map[m_fam] = calib.fit_scale_bias(s461_fr.preds[k_c].mean(0)[ok_c], s461_fr.spine['y'].to_numpy()[ok_c])

    allrows = []
    for fam in LEN_MODELS:
        for ds in LEN_SETS:
            fr = load_frame(ds)
            k = 'msrp0:wt' if fam == 'msr' else fam
            if k not in fr.preds:
                continue
            sp = fr.spine
            ok = _valid(fr, k)
            y = sp['y'].to_numpy()
            idx = np.flatnonzero(ok)
            cl = sp['cluster'].to_numpy()[idx]
            for c_nm in pd.unique(cl):
                m_idx = idx[cl == c_nm]
                if len(m_idx) < MIN_PROT_N:
                    continue
                rhos = [spearmanr(p[m_idx], y[m_idx])[0] for p in fr.preds[k]]
                rrs = [calib.rmse(p[m_idx], y[m_idx]) for p in fr.preds[k]]
                rms = [calib.rmse(calib.apply(p[m_idx], ab_map[fam]), y[m_idx]) for p in fr.preds[k]]
                allrows.append(dict(
                    dataset=ds, cluster=c_nm, length=float(sp['seq_len'].to_numpy()[m_idx].mean()),
                    n=len(m_idx), rho=float(np.mean(rhos)), rmse_raw=float(np.mean(rrs)), rmse=float(np.mean(rms)),
                    model=fam,
                ))
    D = pd.DataFrame(allrows)
    atomic_to_csv(D, out / 'per_protein_metrics.csv', index=False, float_format='%.4f')

    # 3-DMS sequence-length metrics saved separately under 99_misc/dms3/
    dms_metric_rows = []
    dms_models = [('msr', 'ESM-MSR'), ('me2', 'MutateEverything'), ('tmpnnd', 'ThermoMPNN'), ('spursm', 'SPURS')]
    for fam_tag, roster_name in dms_models:
        for ds in DMS3:
            fr = load_frame(ds)
            k = calib.calibration_key(roster_name, fr)
            if k is None or k not in fr.preds:
                continue
            sp = fr.spine
            ok = _valid(fr, k)
            y = sp['y'].to_numpy()
            idx = np.flatnonzero(ok)
            cl = sp['cluster'].to_numpy()[idx]
            for c_nm in pd.unique(cl):
                m_idx = idx[cl == c_nm]
                if len(m_idx) < MIN_PROT_N:
                    continue
                rhos = [spearmanr(p[m_idx], y[m_idx])[0] for p in fr.preds[k]]
                dms_metric_rows.append(dict(
                    dataset=ds, cluster=c_nm, length=float(sp['seq_len'].to_numpy()[m_idx].mean()),
                    n=len(m_idx), rho=float(np.mean(rhos)), rmse_raw=np.nan, rmse=np.nan,
                    model=fam_tag,
                ))
    atomic_to_csv(pd.DataFrame(dms_metric_rows), out_dms / 'length_per_protein_metrics.csv', index=False, float_format='%.4f')

    fig_p, axes_p = plt.subplots(len(LEN_MODELS), 2, figsize=(9.2, 2.3 * len(LEN_MODELS)), sharex=True)
    stats = []
    for r_i, fam in enumerate(LEN_MODELS):
        d_m = D[D['model'] == fam]
        col = MODELS['msrp0:wt']['color'] if fam == 'msr' else MODELS[fam]['color']
        for c_i, (metric, yl) in enumerate((('rho', 'per-protein Spearman ρ'), ('rmse', 'per-protein RMSE (kcal/mol), recalibrated'))):
            ax = axes_p[r_i, c_i]
            dd = d_m if metric == 'rho' else d_m[~d_m['dataset'].isin(['s461', 's669', 's571'])]
            ax.scatter(dd['length'], dd[metric], s=11, color=col, alpha=0.5, lw=0, rasterized=True)
            ax.set_xscale('log')
            r0, lo, hi, n_p = _boot_len_corr(dd, metric, B=B)
            stats.append(dict(model=fam, metric=metric, spearman_length=r0, lo=lo, hi=hi, n_pairs=n_p))
            clip = ''
            if metric == 'rmse':
                clip = f'; {int((dd[metric] > 8).sum())} above 8 not shown'
                ax.set_ylim(0, 8)
            ax.text(0.99, 1.01, f'ρ(length, {metric}) = {r0:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {n_p}{clip}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5)
            if c_i == 0:
                ax.set_ylabel(LEN_LABEL[fam], fontsize=9)
            if r_i == 0:
                ax.set_title(yl, pad=16)
    for ax in axes_p[-1]:
        ax.set_xlabel('wild-type sequence length (residues)')
    fig_p.tight_layout()
    style.save(fig_p, out / 'length_vs_performance')
    S_len = pd.DataFrame(stats)
    atomic_to_csv(S_len, out / 'length_vs_performance_stats.csv', index=False, float_format='%.4f')

    fig_cs, axes_cs = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for c_i, metric in enumerate(('rho', 'rmse')):
        ax = axes_cs[c_i]
        s_m = S_len[S_len['metric'] == metric].reset_index(drop=True)
        for i, row in s_m.iterrows():
            col = MODELS['msrp0:wt']['color'] if row.model == 'msr' else MODELS[row.model]['color']
            ax.errorbar(row.spearman_length, i, xerr=[[row.spearman_length - row.lo], [row.hi - row.spearman_length]], fmt='o', color=col, capsize=2)
        ax.axvline(0, color='#888', lw=0.8, ls=':')
        ax.set_xlabel(f'Spearman(length, per-protein {metric})')
        ax.set_yticks(range(len(s_m)))
        ax.set_yticklabels([LEN_LABEL[m] for m in s_m['model']])
    fig_cs.tight_layout()
    style.save(fig_cs, out / 'length_correlation_summary')
    build_supp_figure6()


def build_supp_figure6() -> Path:
    """Build Supplementary Figure 6 (`00_supplementary_figs/supp_figure6.{png,pdf}`).

    Actual structure C-alpha sequence-length distributions across all 16 datasets from
    Supplementary Figure 1:
      a. Thermostability assays (9 sets: Ssym, PTMUL, S461, K2369, Q3421, S571, S783, S2648, S8754).
      b. MegaScale splits (Train, Validation, Test) and held-out benchmarks
         (ProteinGym*, Domainome*, MGnify*, Rocklin designs*).
    """
    from audits.homology_and_splits.active_builders import set_overlap_matrix as som
    from msr_analysis.proteingym import PG_DROP

    out_supp = results_dir('supplementary_figs')
    out_len = results_dir('sequence_length')
    sets, seqs, _pdbs = som.members()

    # Keep canonical 173 scored held-out ProteinGym assays (excluding >1,800-aa PG_DROP)
    sets['ProteinGym'] = [k for k in sets['ProteinGym'] if k.split('::', 1)[1] not in PG_DROP]

    thermo_order = [
        ('ssym', 'Ssym'),
        ('ptmul', 'PTMUL'),
        ('s461', 'S461'),
        ('k2369', 'K2369'),
        ('q3421', 'Q3421'),
        ('s571', 'S571'),
        ('s783', 'S783'),
        ('s2648', 'S2648'),
        ('s8754', 'S8754'),
    ]
    other_order = [
        ('train', 'Train'),
        ('val', 'Validation'),
        ('test', 'Test'),
        ('ProteinGym', 'ProteinGym*'),
        ('Domainome', 'Domainome*'),
        ('MGnify', 'MGnify*'),
        ('Rocklin designs', 'Rocklin\ndesigns*'),
    ]

    lengths_by_key: dict[str, np.ndarray] = {}
    summary_rows = []
    for panel_tag, group_list in (('a_thermostability', thermo_order), ('b_splits_and_benchmarks', other_order)):
        for ds_key, disp in group_list:
            arr = np.array([len(seqs[k]) for k in sets[ds_key]], dtype=float)
            lengths_by_key[ds_key] = arr
            q25, med, q75 = np.percentile(arr, [25, 50, 75])
            summary_rows.append(dict(
                panel=panel_tag, dataset=ds_key, label=disp.replace('\n', ' '),
                n_structures=int(len(arr)), median=float(med), q25=float(q25), q75=float(q75),
                min=int(arr.min()), max=int(arr.max()),
            ))
    atomic_to_csv(pd.DataFrame(summary_rows), out_len / 'structure_length_summary.csv', index=False, float_format='%.1f')

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(15.4, 6.2), sharey=True,
        gridspec_kw=dict(width_ratios=[len(thermo_order), len(other_order)], wspace=0.06),
        facecolor='white',
    )
    rng = np.random.default_rng(7)
    y_min_all = min(float(v.min()) for v in lengths_by_key.values())
    y_max_all = max(float(v.max()) for v in lengths_by_key.values())
    ylo = 10 ** (np.log10(y_min_all) - 0.10)
    yhi = 10 ** (np.log10(y_max_all) + 0.34)

    def _plot_group(ax, items: list[tuple[str, str]], color_fn, title_txt: str) -> None:
        style.band_axis(ax, len(items))
        for i, (ds_key, _disp) in enumerate(items):
            arr = lengths_by_key[ds_key]
            col = color_fn(ds_key)
            # Subsample only huge sets (e.g. MGnify 152k) for crisp rendering; exact stats use all points
            pts = rng.choice(arr, 1800, replace=False) if len(arr) > 2500 else arr
            jit = rng.uniform(-0.28, 0.28, size=len(pts))
            ax.scatter(i + jit, pts, s=6.5 if len(pts) < 600 else 3.5, color=col,
                       alpha=0.52 if len(pts) < 600 else 0.22, lw=0, zorder=3, rasterized=True)
            q25, med, q75 = np.percentile(arr, [25, 50, 75])
            ax.plot([i, i], [q25, q75], color='#111111', lw=1.6, zorder=5)
            ax.plot([i - 0.32, i + 0.32], [med, med], color='#111111', lw=2.3, zorder=6)
            n_str = f'{len(arr):,}' if len(arr) < 10000 else f'{len(arr) / 1000:.1f}k'
            ax.text(
                i, 0.982, f'{med:.0f}\n({int(arr.min())}–{int(arr.max())})\nn={n_str}',
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=7.3, color='#222222', linespacing=1.18, zorder=7,
            )
        ax.set_yscale('log')
        ax.set_xlim(-0.55, len(items) - 0.45)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([disp for _, disp in items], rotation=28, ha='right', fontsize=9.0)
        ax.set_title(title_txt, fontsize=10.6, fontweight='bold', loc='left', pad=8)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _plot_group(ax_a, thermo_order, lambda _k: '#4c86b8', 'Thermostability benchmarks')
    _plot_group(
        ax_b, other_order,
        lambda k: '#555555' if k in ('train', 'val', 'test') else '#1b8a6b',
        'MegaScale splits & held-out benchmarks',
    )
    ax_a.set_ylabel('Structure sequence length (residues, log scale)', fontsize=10.4)
    ax_b.tick_params(axis='y', labelleft=False)

    fig.subplots_adjust(left=0.060, right=0.988, top=0.900, bottom=0.225)
    fig.canvas.draw()
    pb = ax_b.get_position()
    fig.text(0.010, 0.975, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pb.x0 - 0.022, 0.975, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    note = (
        'Actual PDB / AlphaFold structure Cα sequence lengths across all 16 datasets from Supplementary Figure 1.  '
        'Each column shows individual structures (MGnify* displays a 1,800-structure random subsample for legibility; summary statistics use all 152,680 scaffolds), '
        'horizontal black bar = median, vertical black line = interquartile range (IQR), and top annotations = median, (min–max), and structure count n.  '
        'a. External thermostability benchmarks (Ssym includes both wild-type and reverse-mutation structures).  '
        'b. Homology-Restricted MegaScale splits (grey: Train, Validation, Test) and reduced, held-out external benchmarks (teal): '
        'ProteinGym* (173/213 scored assays), Domainome* (436/522 domains), MGnify* (152,680/180,367 scaffolds), and Rocklin designs* (742/1,000 designs).'
    )
    kw = dict(fontsize=7.8, color='#333333', linespacing=1.25)
    fig.text(0.014, 0.010, style.wrap_to_px(fig, note, 0.972 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'supp_figure6'
    style.save(fig, out_path)
    return out_path


COMBINED_LEN_MODELS = [
    ('msrp0:comb', 'ESM-MSR', '#000000', 'Retrained'),
    ('me2', 'MutateEverything Stage 2', '#1b8a4b', 'Retrained'),
    ('tmpnnd', 'ThermoMPNN-D', '#1f5f8b', 'Retrained'),
    ('spursm', 'SPURS-multi', '#d35400', 'Retrained'),
    ('pmpnn', 'ProteinMPNN', '#34495e', 'Zero-shot'),
    ('msrrel@0:comb', 'ESM3-small-open', '#7f8c8d', 'Zero-shot'),
    ('geoddg', 'GeoDDG-3D', '#117a65', 'Supervised'),
    ('rosetta', 'Rosetta', '#8c564b', 'Supervised'),
]


def build_supp_sequence_length_combined():
    """Build combined SI sequence-length figure (`00_supplementary_figs/_sequence_length.{png,pdf}`):
    a: sequence length distribution across EXT12 + Domainome + Tsuboyama test.
    b: 2x4 per-protein Spearman vs sequence length across all 8 main models (combined/multi-mutant heads,
       including PTMUL/PTMUL-D multimutant proteins highlighted with a thin red outline).
    """
    out_supp = results_dir('supplementary_figs')
    out_len = results_dir('sequence_length')

    # --- Data for panel a ---
    order_a = EXT12 + ['domainome', TSU_CAPPED[1]]
    frames_ext = {ds: load_frame(ds) for ds in EXT12}
    per_prot = {}
    for ds in order_a:
        fr = frames_ext[ds] if ds in frames_ext else load_frame(ds)
        sp = fr.spine[fr.spine['use']]
        per_prot[ds] = sp.groupby('cluster')['seq_len'].mean()

    # --- Data for panel b (all EXT12 including PTMUL & PTMUL-D) ---
    rows_b = []
    for key, label, _col, _grp in COMBINED_LEN_MODELS:
        for ds in EXT12:
            fr = frames_ext[ds]
            if key not in fr.preds:
                continue
            sp = fr.spine
            ok = _valid(fr, key)
            y = sp['y'].to_numpy(float)
            n_mut = sp['n_mut'].to_numpy()
            seq_len = sp['seq_len'].to_numpy(float)
            idx = np.flatnonzero(ok)
            cl = sp['cluster'].to_numpy()[idx]
            for c_nm in pd.unique(cl):
                m_idx = idx[cl == c_nm]
                if len(m_idx) < MIN_PROT_N:
                    continue
                rhos = [spearmanr(p[m_idx], y[m_idx])[0] for p in fr.preds[key]]
                rho_val = float(np.nanmean(rhos))
                if not np.isfinite(rho_val):
                    continue
                rows_b.append(dict(
                    model=label, key=key, dataset=ds, cluster=c_nm,
                    length=float(seq_len[m_idx].mean()), n=len(m_idx),
                    rho=rho_val, has_multi=bool((n_mut[m_idx] > 1).any()),
                ))
    D_b = pd.DataFrame(rows_b)
    atomic_to_csv(D_b, out_len / 'per_protein_metrics_8models.csv', index=False, float_format='%.4f')

    # --- Layout: panel a on top, panel b (2x4) below, no overall figure title ---
    fig = plt.figure(figsize=(14.4, 9.8), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.85], hspace=0.34)
    ax_a = fig.add_subplot(outer[0])
    gs_b = outer[1].subgridspec(2, 4, hspace=0.30, wspace=0.14)

    # Panel a: sequence length distribution
    style.band_axis(ax_a, len(order_a))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(order_a):
        L = per_prot[ds].to_numpy()
        col = '#2c7fb8' if ds in EXT12 else '#7f7f7f'
        ax_a.scatter(i + rng.uniform(-0.28, 0.28, len(L)), L, s=7, color=col, alpha=0.55, lw=0, zorder=3, rasterized=True)
        q = np.percentile(L, [25, 50, 75])
        ax_a.plot([i - 0.36, i + 0.36], [q[1]] * 2, color='k', lw=1.8, zorder=4)
        ax_a.plot([i, i], [q[0], q[2]], color='k', lw=0.9, zorder=4)
        ax_a.text(i, 920, f'{int(np.median(L))}\n({int(L.min())}–{int(L.max())})', ha='center', va='top', fontsize=7.6)
    ax_a.set_yscale('log')
    ax_a.set_ylim(15, 1050)
    ax_a.set_xlim(-0.55, len(order_a) - 0.45)
    ax_a.set_xticks(range(len(order_a)))
    ax_a.set_xticklabels([LABEL[o] for o in order_a], rotation=26, ha='right', fontsize=9.2)
    ax_a.set_ylabel('wild-type sequence length\n(residues)', fontsize=10.2)
    ax_a.tick_params(axis='y', labelsize=9.2)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker='o', ls='', color='#2c7fb8', ms=5, alpha=0.75, label='External thermostability sets'),
            Line2D([0], [0], marker='o', ls='', color='#7f7f7f', ms=5, alpha=0.75, label='Small-domain benchmarks (Domainome, MegaScale test)'),
            Line2D([0], [0], color='k', lw=1.8, label='Median & IQR'),
        ],
        loc='lower left', ncol=3, fontsize=8.2, frameon=True, framealpha=0.9, edgecolor='#dddddd',
    )

    # Panel b: 2x4 per-protein Spearman vs sequence length
    axes_b = np.empty((2, 4), dtype=object)
    for idx, (key, label, col, grp) in enumerate(COMBINED_LEN_MODELS):
        r_i, c_i = divmod(idx, 4)
        ax = fig.add_subplot(
            gs_b[r_i, c_i],
            sharex=axes_b[0, 0] if idx > 0 else None,
            sharey=axes_b[0, 0] if idx > 0 else None,
        )
        axes_b[r_i, c_i] = ax
        dm = D_b[D_b['model'] == label]
        d_s = dm[~dm['has_multi']]
        d_m = dm[dm['has_multi']]

        ax.axhline(0.0, color='#cccccc', lw=0.8, ls=':', zorder=1)
        ax.scatter(d_s['length'], d_s['rho'], s=13, color=col, alpha=0.45, edgecolors='none', zorder=3, rasterized=True)
        ax.scatter(
            d_m['length'], d_m['rho'], s=24, color=col, alpha=0.82,
            edgecolors='#e31a1c', linewidths=0.9, zorder=4, rasterized=True,
        )
        ax.set_xscale('log')
        ax.set_ylim(-1.05, 1.08)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        r0 = float(spearmanr(dm['length'], dm['rho'])[0])
        ax.text(
            0.04, 0.05, f'ρ = {r0:+.2f}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12.0, fontweight='bold', color='#111111', zorder=10,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.92),
        )
        ax.text(
            0.96, 0.05, grp,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color='#555555', zorder=10,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.75),
        )
        ax.set_title(label, fontsize=10.5, fontweight='bold', pad=5)
        if c_i == 0:
            ax.set_ylabel('per-protein Spearman ρ', fontsize=9.8)
        else:
            ax.tick_params(axis='y', labelleft=False)
        if r_i == 1:
            ax.set_xlabel('wild-type sequence length (residues)', fontsize=9.6)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='both', labelsize=8.8)

    # Legend for multimutant outline in Panel b (aligned right above the 2x4 grid)
    fig.subplots_adjust(left=0.068, right=0.985, top=0.975, bottom=0.105)
    fig.canvas.draw()
    pa = ax_a.get_position()
    pb0 = axes_b[0, 0].get_position()
    fig.text(0.012, pa.y1 + 0.012, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(0.012, pb0.y1 + 0.032, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    b_handles = [
        Line2D([0], [0], marker='o', ls='', color='#444444', ms=4.5, alpha=0.55, label='Single-mutant protein (n = 664)'),
        Line2D([0], [0], marker='o', ls='', color='#444444', markeredgecolor='#e31a1c', markeredgewidth=0.95, ms=5.8, alpha=0.85, label='Contains multi-mutants: PTMUL / PTMUL-D (n = 59)'),
    ]
    fig.legend(
        handles=b_handles, loc='lower right',
        bbox_to_anchor=(0.985, pb0.y1 + 0.018),
        ncol=2, fontsize=8.4, frameon=True, framealpha=0.92, edgecolor='#dddddd',
    )

    n_tot = len(D_b[D_b['model'] == COMBINED_LEN_MODELS[0][1]])
    note = (
        f'a. Wild-type sequence length per protein cluster across external thermostability sets and small-domain benchmarks (top annotations: median and min–max).  '
        f'b. Per-protein Spearman ρ vs wild-type sequence length across all 12 external thermostability sets (proteins with ≥ {MIN_PROT_N} mutations, n = {n_tot}); '
        f'retrained models use their combined/multi-mutant head (3-seed mean ρ); red-outlined points contain multi-mutants.'
    )
    kw_note = dict(fontsize=8.0, color='#444444', linespacing=1.25)
    fig.text(0.012, 0.010, style.wrap_to_px(fig, note, 0.970 * fig.bbox.width, **kw_note), ha='left', va='bottom', **kw_note)

    out_path = out_supp / '_sequence_length'
    style.save(fig, out_path)
    return out_path


def build_supp_ssym_antisymmetry_bias(B: int = 10_000) -> Path:
    """Build 2-panel supplementary Ssym antisymmetry (a) and bias (b) figure across all 10 main methods.

    Uses all 342 exact direct-reverse pairs in Ssym (index_rev = index_dir + 342; 684 variants across 15 WT proteins).
    - Panel a: Antisymmetry Pearson r(y_dir, y_rev) (ideal = -1.0).
    - Panel b: Bias <delta> = mean(y_dir + y_rev) / 2 (ideal = 0.0; Pucci et al. 2018 convention).
    Shows WT-view (hatched) alongside Dual-view (solid) for ESM-MSR and ESM3-small-open, and canonical single-mutant
    readout (solid) for the other 8 methods. No overall figure title.
    """
    from matplotlib.patches import Patch
    from scipy.stats import pearsonr
    from msr_analysis import models as M
    from msr_analysis.boot import ci as boot_ci
    from msr_analysis.common import leak_table
    from msr_analysis.paths import TASKS, atomic_write_text

    fr = load_frame('ssym')
    sp = fr.spine
    t = pd.read_csv(TASKS / 'ssym.csv.gz')
    d_df = t[t['index'] < 342].set_index('index').sort_index()
    r_df = t[t['index'] >= 342].assign(pair_idx=lambda x: x['index'] - 342).set_index('pair_idx').sort_index()
    assert len(d_df) == 342 and len(r_df) == 342
    assert bool((d_df['wt_code'] == r_df['wt_code']).all())
    assert bool(((d_df['wild_type'] == r_df['mutation']) & (d_df['mutation'] == r_df['wild_type'])).all())

    id2i = {int(rid): i for i, rid in enumerate(sp['row_id'].to_numpy())}
    i_dir = np.array([id2i[int(rid)] for rid in d_df['row_id']], dtype=int)
    i_rev = np.array([id2i[int(rid)] for rid in r_df['row_id']], dtype=int)
    y_dir = sp['y'].to_numpy(float)[i_dir]
    y_rev = sp['y'].to_numpy(float)[i_rev]
    y_all = np.concatenate([y_dir, y_rev])
    n_pairs = len(i_dir)
    n_prots = int(d_df['wt_code'].nunique())
    lk = leak_table()

    # All 10 canonical main methods in manuscript order
    # Each entry: (name, cat, color, label, [(view_tag, key), ...])
    specs = []
    for name in M.UNIFIED:
        m = M.BY_NAME[name]
        is_leak = bool(lk.get((m.single, 'ssym'), False))
        lbl = f'{name}*' if is_leak else name
        if name in ('ESM-MSR', 'ESM3-small-open'):
            views = [('wt', m.single), ('dual', m.multi)]
            lbl = f'{name}\n(WT / Dual-view)'
        else:
            views = [('dual', m.single)]
        specs.append(dict(name=name, cat=m.group, color=m.color, label=lbl, views=views, leak=is_leak))

    # Shared paired bootstrap draws: B x n_pairs multinomial weights + B x 3 seed weights
    rng = np.random.default_rng(20260)
    pair_w = rng.multinomial(n_pairs, np.full(n_pairs, 1.0 / n_pairs), size=B).astype(float)  # (B, 342)
    seed_w3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0

    def _w_pearson(x: np.ndarray, yv: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Vectorized weighted Pearson r for 1D arrays x, yv and (B, N) weight matrix w."""
        ws = w.sum(axis=1, keepdims=True)
        mx = (w @ x)[:, None] / ws
        my = (w @ yv)[:, None] / ws
        dx = x[None, :] - mx
        dy = yv[None, :] - my
        cov = np.sum(w * dx * dy, axis=1)
        vx = np.sum(w * dx * dx, axis=1)
        vy = np.sum(w * dy * dy, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return cov / np.sqrt(vx * vy)

    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    sum_rows = []
    pair_rows = []
    r_reps_map: dict[tuple[str, str], np.ndarray] = {}
    b_reps_map: dict[tuple[str, str], np.ndarray] = {}
    pt_map: dict[tuple[str, str], dict] = {}

    for spc in specs:
        name = spc['name']
        for vtag, key in spc['views']:
            arr = fr.preds[key]
            ns = arr.shape[0]
            seed_r_draws = np.empty((B, ns), dtype=float)
            seed_b_draws = np.empty((B, ns), dtype=float)
            s_r, s_rho_dr, s_bias, s_std_b = [], [], [], []
            s_rho_d, s_rho_r, s_rho_all, s_pr_all, s_rmse_all = [], [], [], [], []

            for s in range(ns):
                pd_ = arr[s, i_dir].astype(float)
                pr_ = arr[s, i_rev].astype(float)
                ok = np.isfinite(pd_) & np.isfinite(pr_)
                w_eff = pair_w[:, ok] if not ok.all() else pair_w
                seed_r_draws[:, s] = _w_pearson(pd_[ok], pr_[ok], w_eff)
                delta = 0.5 * (pd_[ok] + pr_[ok])
                seed_b_draws[:, s] = (w_eff @ delta) / w_eff.sum(axis=1)

                s_r.append(float(pearsonr(pd_[ok], pr_[ok])[0]))
                s_rho_dr.append(float(spearmanr(pd_[ok], pr_[ok])[0]))
                s_bias.append(float(np.mean(delta)))
                p_all = np.concatenate([pd_[ok], pr_[ok]])
                y_ok = np.concatenate([y_dir[ok], y_rev[ok]])
                s_std_b.append(float(np.mean(delta) / max(np.std(p_all), 1e-12)))
                s_rho_d.append(float(spearmanr(pd_[ok], y_dir[ok])[0]))
                s_rho_r.append(float(spearmanr(pr_[ok], y_rev[ok])[0]))
                s_rho_all.append(float(spearmanr(p_all, y_ok)[0]))
                s_pr_all.append(float(pearsonr(p_all, y_ok)[0]))
                s_rmse_all.append(float(np.sqrt(np.mean((p_all - y_ok) ** 2))))

            r_reps = np.sum(seed_r_draws * seed_w3, axis=1) if ns == 3 else seed_r_draws.mean(axis=1)
            b_reps = np.sum(seed_b_draws * seed_w3, axis=1) if ns == 3 else seed_b_draws.mean(axis=1)
            r_pt = float(np.mean(s_r))
            b_pt = float(np.mean(s_bias))
            r_lo, r_hi = boot_ci(r_reps)
            b_lo, b_hi = boot_ci(b_reps)

            r_reps_map[(name, vtag)] = r_reps
            b_reps_map[(name, vtag)] = b_reps
            pt_map[(name, vtag)] = dict(
                r=r_pt, bias=b_pt, pearson_r=float(np.mean(s_pr_all)), rmse=float(np.mean(s_rmse_all)),
                leak=spc['leak'],
            )
            sum_rows.append(dict(
                model=name, view=vtag, key=key, group=spc['cat'], leak_flag=spc['leak'],
                r_dir_rev=r_pt, r_lo=r_lo, r_hi=r_hi,
                rho_dir_rev=float(np.mean(s_rho_dr)),
                bias_delta=b_pt, bias_lo=b_lo, bias_hi=b_hi,
                abs_bias=abs(b_pt),
                std_bias=float(np.mean(s_std_b)),
                spearman_dir=float(np.mean(s_rho_d)),
                spearman_rev=float(np.mean(s_rho_r)),
                spearman_all=float(np.mean(s_rho_all)),
                pearson_r=float(np.mean(s_pr_all)),
                rmse=float(np.mean(s_rmse_all)),
                n_pairs=n_pairs, n_variants=2 * n_pairs, n_proteins=n_prots, n_seeds=ns,
            ))

            # Per-pair seed-mean predictions for full reproducibility
            pd_mean = arr[:, i_dir].mean(axis=0)
            pr_mean = arr[:, i_rev].mean(axis=0)
            for pidx in range(n_pairs):
                pair_rows.append(dict(
                    pair_idx=pidx, wt_code=str(d_df['wt_code'].iloc[pidx]),
                    id_dir=str(d_df['id'].iloc[pidx]), id_rev=str(r_df['id'].iloc[pidx]),
                    y_dir=float(y_dir[pidx]), y_rev=float(y_rev[pidx]),
                    model=name, view=vtag, key=key,
                    pred_dir=float(pd_mean[pidx]), pred_rev=float(pr_mean[pidx]),
                    delta_half=0.5 * float(pd_mean[pidx] + pr_mean[pidx]),
                ))

    df_sum = pd.DataFrame(sum_rows)

    # Render 2-panel stacked figure (a: r_dir_rev; b: bias <delta>), global grouped styling, no overall title
    xs, spans = style.layout_grouped_columns(specs, gap=0.7)
    w_fig = max(12.4, (12.5 / 11.85) * (xs[-1] - xs[0] + 1.45))
    fig = plt.figure(figsize=(w_fig, 10.2), facecolor='white')
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.985, top=0.945, bottom=0.125, hspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    def _style_grouped_ax(ax, ylabel: str, ylim: tuple[float, float], show_legend: bool = False):
        ax.set_xticks(xs)
        ax.set_xticklabels([s['label'] for s in specs], fontsize=8.5, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.75)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.axhline(0.0, color='black', lw=0.85, zorder=2.5)
        for i_s, (cat, (x0, x1)) in enumerate(spans.items()):
            if i_s % 2:
                ax.axvspan(x0 - 0.5 - (0.35 if i_s else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
            if cat in M.GROUP_TITLE:
                tr = ax.get_xaxis_transform()
                ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
                ax.text((x0 + x1) / 2, 1.02, M.GROUP_TITLE[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
        if show_legend:
            handles = [
                Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label='WT-view (single pass)'),
                Patch(fc='#999999', ec='black', label='Dual-view / standard readout'),
                Line2D([0], [0], color='#555555', ls='--', lw=1.2, label='ideal antisymmetry (r = −1.0)'),
            ]
            ax.legend(handles=handles, ncol=3, loc='lower left', fontsize=8.2, framealpha=0.92, borderpad=0.35, columnspacing=1.1)

    # Panel a: Antisymmetry r_dir_rev (bars hang downward from 0 toward -1.0)
    ylim_a = (-1.20, 0.08)
    _style_grouped_ax(ax_a, 'Antisymmetry, Pearson r(ΔΔG_dir, ΔΔG_rev)', ylim_a, show_legend=True)
    ax_a.axhline(-1.0, color='#555555', ls='--', lw=1.15, zorder=2.2)

    for s, x in zip(specs, xs):
        sub_df = df_sum[df_sum['model'] == s['name']]
        two = len(sub_df) == 2
        bar_w = 0.32 if two else 0.50
        offs = (-0.17, 0.17) if two else (0.0,)
        for j, (_, row) in enumerate(sub_df.iterrows()):
            xx = x + offs[j]
            hatched = two and (row['view'] == 'wt')
            v, lo, hi = float(row['r_dir_rev']), float(row['r_lo']), float(row['r_hi'])
            ax_a.bar(
                xx, v, bar_w, color=s['color'],
                alpha=0.55 if hatched else 0.95,
                edgecolor=s['color'] if hatched else 'black',
                hatch='////' if hatched else None, lw=0.8, zorder=3,
            )
            ax_a.errorbar(
                xx, v, yerr=[[max(v - lo, 0.0)], [max(hi - v, 0.0)]],
                fmt='none', ecolor=style.RING, elinewidth=1.3, capsize=2.5, zorder=5,
            )
            ax_a.text(
                xx, min(lo, v) - 0.018, f'{v:.2f}',
                ha='center', va='top', fontsize=8.0, rotation=90, zorder=8,
                bbox=dict(fc='white', ec='none', alpha=0.72, pad=0.5),
            )

    # Panel b: Bias <delta> = mean(y_dir + y_rev) / 2 (ideal = 0.0)
    ylim_b = (-0.98, 0.09)
    _style_grouped_ax(ax_b, 'Bias ⟨δ⟩ = mean(ΔΔG_dir + ΔΔG_rev) / 2 (kcal/mol)', ylim_b, show_legend=False)
    handles_b = [
        Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label='WT-view (single pass)'),
        Patch(fc='#999999', ec='black', label='Dual-view / standard readout'),
    ]
    ax_b.legend(handles=handles_b, ncol=2, loc='lower right', fontsize=8.2, framealpha=0.92, borderpad=0.35, columnspacing=1.1)

    trunc_y = -0.84
    for s, x in zip(specs, xs):
        sub_df = df_sum[df_sum['model'] == s['name']]
        two = len(sub_df) == 2
        bar_w = 0.32 if two else 0.50
        offs = (-0.17, 0.17) if two else (0.0,)
        for j, (_, row) in enumerate(sub_df.iterrows()):
            xx = x + offs[j]
            hatched = two and (row['view'] == 'wt')
            v, lo, hi = float(row['bias_delta']), float(row['bias_lo']), float(row['bias_hi'])
            if v < trunc_y:
                # Off-scale uncalibrated log-odds bar (ProSST = -3.12): truncate cleanly with // break mark
                ax_b.bar(xx, trunc_y, bar_w, color=s['color'], alpha=0.95, edgecolor='black', lw=0.8, zorder=3)
                # Draw white diagonal break slashes across the bar near bottom
                by = trunc_y * 0.80
                for dy_s in (-0.016, 0.016):
                    ax_b.plot([xx - bar_w * 0.54, xx + bar_w * 0.54], [by + dy_s - 0.018, by + dy_s + 0.018], color='white', lw=3.2, zorder=6)
                    ax_b.plot([xx - bar_w * 0.54, xx + bar_w * 0.54], [by + dy_s - 0.018, by + dy_s + 0.018], color='#333333', lw=0.9, zorder=7)
                ax_b.text(
                    xx, trunc_y - 0.016, f'{v:.2f}\n(log-odds)',
                    ha='center', va='top', fontsize=7.6, color='#222222', zorder=8,
                    bbox=dict(fc='white', ec='none', alpha=0.80, pad=0.4),
                )
            else:
                ax_b.bar(
                    xx, v, bar_w, color=s['color'],
                    alpha=0.55 if hatched else 0.95,
                    edgecolor=s['color'] if hatched else 'black',
                    hatch='////' if hatched else None, lw=0.8, zorder=3,
                )
                ax_b.errorbar(
                    xx, v, yerr=[[max(v - lo, 0.0)], [max(hi - v, 0.0)]],
                    fmt='none', ecolor=style.RING, elinewidth=1.3, capsize=2.5, zorder=5,
                )
                lbl_val = f'{v:+.2f}' if abs(v) >= 0.005 else '0.00'
                ax_b.text(
                    xx, min(lo, v) - 0.018, lbl_val,
                    ha='center', va='top', fontsize=8.0, rotation=90, zorder=8,
                    bbox=dict(fc='white', ec='none', alpha=0.72, pad=0.5),
                )

    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.012, pa.y1 + 0.030, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(0.012, pb.y1 + 0.030, 'b', ha='left', va='top', fontsize=16, fontweight='bold')

    supp_dir = results_dir('supplementary_figs')
    out_path = supp_dir / '_ssym_antisymmetry_bias'
    style.save(fig, out_path)

    # Export source data + paired one-sided bootstrap p-value table & annotation
    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(df_sum, src_dir / '_ssym_antisymmetry_bias_summary_sourcedata.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(pair_rows), src_dir / '_ssym_antisymmetry_bias_per_pair_sourcedata.csv', index=False, float_format='%.5f')

    # Paired one-sided bootstrap p-values comparing ESM-MSR Dual-view against every other method
    msr_key = ('ESM-MSR', 'dual')
    msr_info = pt_map[msr_key]
    comp_keys = [(s['name'], 'dual') for s in specs if s['name'] != 'ESM-MSR'] + [('ESM-MSR', 'wt')]
    p_rows = []
    for metric_name, is_bias in (('antisymmetry_r_dir_rev', False), ('abs_bias_delta', True)):
        # Clean family = all external models without Ssym direct training overlap (excludes GeoDDG-3D and internal ESM-MSR WT-view)
        clean_comps = [(n, vt) for (n, vt) in comp_keys if n != 'ESM-MSR' and not pt_map[(n, vt)]['leak']]
        m_clean = len(clean_comps)
        clean_ps, clean_deltas = {}, []
        for cname, vtag in comp_keys:
            is_internal = (cname == 'ESM-MSR')
            is_leak = bool(pt_map[(cname, vtag)]['leak'])
            is_clean = (not is_internal) and (not is_leak)
            disp_c = f'{cname} (WT-view)' if is_internal else cname
            if not is_bias:
                # Better antisymmetry = more negative r_dir_rev -> (-r_msr) > (-r_comp) => r_comp - r_msr > 0
                d_arr = r_reps_map[(cname, vtag)] - r_reps_map[msr_key]
                v_msr = msr_info['r']
                v_cmp = pt_map[(cname, vtag)]['r']
                delta_pt = v_cmp - v_msr
                direction = 'ESM-MSR more negative r_dir_rev'
            else:
                # Smaller |bias| -> |b_comp| - |b_msr| > 0
                d_arr = np.abs(b_reps_map[(cname, vtag)]) - np.abs(b_reps_map[msr_key])
                v_msr = abs(msr_info['bias'])
                v_cmp = abs(pt_map[(cname, vtag)]['bias'])
                delta_pt = v_cmp - v_msr
                direction = 'ESM-MSR smaller |bias|'
            p_one, nf, b_eff = _one_sided_p(d_arr)
            lo_d, hi_d = boot_ci(d_arr)
            if is_clean:
                clean_ps[disp_c] = p_one
                clean_deltas.append(d_arr)
            p_rows.append(dict(
                benchmark='Ssym_342_pairs',
                subset_or_metric=metric_name,
                resampling_unit='mutation_pair_x_seed',
                test_type='pairwise',
                msr_model='ESM-MSR (Dual-view)',
                comparator=disp_c,
                excluded_training_overlap=is_leak,
                internal_control_or_variant=is_internal,
                direction=direction,
                value_msr=v_msr,
                value_comparator=v_cmp,
                pearson_r_msr=msr_info['pearson_r'],
                pearson_r_comparator=pt_map[(cname, vtag)]['pearson_r'],
                rmse_msr=msr_info['rmse'],
                rmse_comparator=pt_map[(cname, vtag)]['rmse'],
                delta_favoring_msr=delta_pt,
                ci95_lo=lo_d, ci95_hi=hi_d,
                n_units=n_pairs, n_variants=2 * n_pairs, B=b_eff, n_boot_fail=nf,
                p_one_sided_uncorrected=p_one,
                bonferroni_m=m_clean if is_clean else np.nan,
                p_one_sided_bonferroni=min(1.0, m_clean * p_one) if is_clean else np.nan,
            ))
        # Summary rows over clean comparators
        if clean_comps:
            if not is_bias:
                best_k = min(clean_comps, key=lambda k: pt_map[k]['r'])
                d_nb = r_reps_map[best_k] - r_reps_map[msr_key]
                v_msr, v_cmp = msr_info['r'], pt_map[best_k]['r']
                delta_nb = v_cmp - v_msr
                direction = 'ESM-MSR more negative r_dir_rev'
            else:
                best_k = min(clean_comps, key=lambda k: abs(pt_map[k]['bias']))
                d_nb = np.abs(b_reps_map[best_k]) - np.abs(b_reps_map[msr_key])
                v_msr, v_cmp = abs(msr_info['bias']), abs(pt_map[best_k]['bias'])
                delta_nb = v_cmp - v_msr
                direction = 'ESM-MSR smaller |bias|'
            p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
            lo_nb, hi_nb = boot_ci(d_nb)
            p_rows.append(dict(
                benchmark='Ssym_342_pairs', subset_or_metric=metric_name, resampling_unit='mutation_pair_x_seed',
                test_type='next_best_clean', msr_model='ESM-MSR (Dual-view)', comparator=best_k[0],
                excluded_training_overlap=False, internal_control_or_variant=False, direction=direction,
                value_msr=v_msr, value_comparator=v_cmp,
                pearson_r_msr=msr_info['pearson_r'], pearson_r_comparator=pt_map[best_k]['pearson_r'],
                rmse_msr=msr_info['rmse'], rmse_comparator=pt_map[best_k]['rmse'],
                delta_favoring_msr=delta_nb, ci95_lo=lo_nb, ci95_hi=hi_nb,
                n_units=n_pairs, n_variants=2 * n_pairs, B=b_nb, n_boot_fail=nf_nb,
                p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
            ))
            worst_name = max(clean_ps, key=lambda k: clean_ps[k])
            worst_k = next(k for k in clean_comps if k[0] == worst_name)
            d_w = (r_reps_map[worst_k] - r_reps_map[msr_key]) if not is_bias else (np.abs(b_reps_map[worst_k]) - np.abs(b_reps_map[msr_key]))
            p_iut, nf_w, b_w = _one_sided_p(d_w)
            lo_w, hi_w = boot_ci(d_w)
            p_rows.append(dict(
                benchmark='Ssym_342_pairs', subset_or_metric=metric_name, resampling_unit='mutation_pair_x_seed',
                test_type='all_clean_IUT', msr_model='ESM-MSR (Dual-view)', comparator=f'ALL_CLEAN_IUT (limiting: {worst_name})',
                excluded_training_overlap=False, internal_control_or_variant=False, direction=direction,
                value_msr=v_msr, value_comparator=pt_map[worst_k]['r'] if not is_bias else abs(pt_map[worst_k]['bias']),
                pearson_r_msr=msr_info['pearson_r'], pearson_r_comparator=pt_map[worst_k]['pearson_r'],
                rmse_msr=msr_info['rmse'], rmse_comparator=pt_map[worst_k]['rmse'],
                delta_favoring_msr=(pt_map[worst_k]['r'] - v_msr) if not is_bias else (abs(pt_map[worst_k]['bias']) - v_msr),
                ci95_lo=lo_w, ci95_hi=hi_w,
                n_units=n_pairs, n_variants=2 * n_pairs, B=b_w, n_boot_fail=nf_w,
                p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
            ))

    atomic_to_csv(pd.DataFrame(p_rows), pval_dir / '_ssym_antisymmetry_bias_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_ssym_antisymmetry_bias_pvalues_annotation.txt',
        f"""Statistical Annotation: _ssym_antisymmetry_bias_pvalues.csv
===========================================================
Figure / Panel: Supplementary Figure (_ssym_antisymmetry_bias) on Ssym (342 exact direct-reverse mutation pairs; 684 variants across 15 WT protein structures).
  - Panel a: Antisymmetry Pearson r(ddG_dir, ddG_rev) (ideal = -1.0; more negative is better).
  - Panel b: Bias <delta> = mean(ddG_dir + ddG_rev) / 2 (Pucci et al., 2018; ideal = 0.0; smaller |<delta>| is better). Note: ProSST (-3.12) and ProteinMPNN (-0.66) output uncalibrated log-odds rather than kcal/mol; scale-free standardized bias (<delta> / SD) is also provided in the summary source data table.

Statistical Test & Resampling Procedure:
  Paired nonparametric mutation-pair x seed bootstrap (B = {B:,}, deterministic seed).
  In each replicate, the 342 direct-reverse pairs are resampled with replacement via one shared Multinomial(342, 1/342) weight vector across all models, and 3-seed retrained models are resampled jointly over training seeds.
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  GeoDDG-3D (trained on S2648 with 50% direct mutation overlap with Ssym, and antisymmetric by construction) is flagged `excluded_training_overlap=True` and excluded from the m = 8 clean comparator family. ESM3dG predicts absolute dG(mut) - dG(wt) (structurally antisymmetric up to crystal structure differences) and is retained in the clean family (no direct mutation leak).
""",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    ap.add_argument('--only', choices=('all', 'supp_figure6', 'ssym_antisym'), default='all', help='Optional substage selector')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    if args.only == 'ssym_antisym':
        out = build_supp_ssym_antisymmetry_bias(B=max(B, 10_000))
        print(f'[p08:calibration_length] Saved {out}.{{png,pdf}}')
        return 0

    if args.only == 'supp_figure6':
        out = build_supp_figure6()
        print(f'[p08:calibration_length] Saved {out}.{{png,pdf}}')
        return 0

    run_rmse_calibration_for_donor('s461', {'s461', 's669'}, B=B)
    run_rmse_calibration_for_donor('ssym', {'ssym'}, B=B)
    run_sequence_length(B=B)
    return 0


if __name__ == '__main__':
    sys.exit(main())
