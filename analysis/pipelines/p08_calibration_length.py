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


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    args = ap.parse_args(argv)
    B = resolve_b(args)

    run_rmse_calibration_for_donor('s461', {'s461', 's669'}, B=B)
    run_rmse_calibration_for_donor('ssym', {'ssym'}, B=B)
    run_sequence_length(B=B)
    return 0


if __name__ == '__main__':
    sys.exit(main())
