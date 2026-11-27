#!/usr/bin/env python3
"""Pipeline p06: Epistasis hierarchy (val/test), multi-vs-additive ranking, masking ablations, and main composite.

Consolidates:
  - 06_epistasis.py                          -> results/01_megascale/epistasis/{epistasis_naive_recomputed, epistasis_scatter_dddG,
                                                doubles_pooled_vs_per_library.csv, epistasis_scatter_ddG_doubles, ddg_ranking, lambda_scan}
                                                + results/99_misc/dms3/epistasis_{naive_recomputed,scatter_dddG,lambda_scan}.{csv,pdf,png}
  - 11_publication_figures.py (fig3, fig4)   -> results/01_megascale/epistasis/{spearman_doubles_additive_vs_epistatic*, naive_epistasis_pos_neg_*}
  - 11e_esm3_scoring_variants.py (fig3 part) -> results/01_megascale/epistasis/esm3_scoring_variants
  - 11l_epistatic_vs_additive_pvalues.py     -> results/01_megascale/epistasis/epistatic_vs_additive{,_ndcg96,_ndcg96_rectified}_pvalues.{csv,md,png,pdf}
  - 11h_stacked_stab_epistatic.py            -> results/00_main_text_figs/fig2_fig3_stacked.{png,pdf}
  - 11d + 11b + 11n                          -> results/01_megascale/epistasis/epistasis_levels_bootstrap_{test,val}.csv,
                                                results/01_megascale/epistasis/epistasis_levels_bars_{test,val}{,_cell_mag,_cell_compare}.{png,pdf},
                                                results/00_supplementary_figs/fig5_epistasis_levels_{p0,masking}_{bootstrap,paired}_test.csv,
                                                results/00_supplementary_figs/fig5_epistasis_levels_{p0,masking}_reps_test.npz,
                                                results/00_supplementary_figs/fig5a_epistasis_levels_bars_test_{p0,masking}.{png,pdf},
                                                results/00_main_text_figs/epistasis_levels_e.{png,pdf}
  - 11m_masking_schemes.py                   -> results/00_supplementary_figs/masking_schemes{,_values,_pvalues}.*
  - NEW composite figure                     -> results/00_main_text_figs/epistasis_main_composite.{png,pdf}

Key methodological fixes vs legacy scripts:
  - Runs ONE unified boot_epistasis_models pass on test covering all roster models, SPURS unfrozen, ctx control, and p1/p1_marg masking
    ablations — eliminating the 3x redundant multiprocessing bootstrap across 11d and 11n!
  - Fixes 11n paired_tests delta to use exact point[i0] - point[i] rather than bootstrap replicate mean difference.
  - Runs all 3 metrics of 11l (spearman, ndcg96, ndcg96_rectified) and generates fig2_fig3_stacked automatically.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata, spearmanr

from esm_msr import epi_metrics as E

from msr_analysis import epistasis as EPI_LIB
from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import SpearmanEngine, bootstrap_rho, ci
from msr_analysis.common import epi_rows, out_dir, subframe
from msr_analysis.data import _align_id, load_frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.paths import BASE, FINAL_CTX, REFERENCE, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.registry import MODELS, PER_PROTEIN, tsu_datasets
from msr_analysis.stats import boot_pvalue, fmt_p, holm, seed_sd

from . import base_parser, resolve_b
from ._tsu_common import (
    BEST_LABEL,
    MIN_N,
    add_esm3_marginal_additive,
    add_holm_by_group,
    best_other,
    boot_mean,
    boot_reps,
    col_spec,
    compare_grouped,
    compare_pooled,
    count_text,
    diff_df,
    epi_columns,
    fig_width,
    finish_bar_fig,
    load_tsu_test_frame,
    main_columns,
    make_domain_counts,
    mean_val,
    ndcg96_rectified,
    ndcg96_shifted,
    pair_columns,
    per_library,
    rho_fn,
    row_of,
    split_title,
    table_grouped,
    table_pooled,
)

STAGE = 'epistasis_hierarchy'
DESCRIPTION = 'Epistasis hierarchy (val/test), multi vs additive ranking, masking ablations, and main composite'

DMS_DOUBLES = ['GRB2_HUMAN_Faure_2021_binding_domain', 'DLG4_HUMAN_Faure_2021_binding_domain']
LAMBDAS = np.round(np.arange(0, 1.0001, 0.05), 2)

EPI_DESC = '1/2(MT_AB+WT_AB) − WT_A − WT_B'
CTX_DESC = '1/2(~WT_AB+WT_AB) − WT_A − WT_B'
P0_EPI_NAME = f'ESM-MSR p0 · {EPI_DESC}'
P0_CTX_NAME = f'ESM-MSR p0 · {CTX_DESC}'
P1_EPI_NAME = f'ESM-MSR p1 (independent masking) · {EPI_DESC}'
P1M_EPI_NAME = f'ESM-MSR p1 (masked marginal MT) · {EPI_DESC}'
STD_EPI_NAME = f'ESM-MSR · {EPI_DESC}'
STD_CTX_NAME = f'ESM-MSR · {CTX_DESC}'

LEVELS_MAIN = [
    ('naive_rho', 'naive / raw ρ'),
    ('beyond_rho', 'saturation-subtracted ρ'),
    ('pair_rho', 'pair ρ (coarsest)'),
    ('line_rho', 'line ρ'),
    ('cell_rank', 'cell rank (finest)'),
]
LEVELS_CELL_MAG = [
    ('naive_rho', 'naive / raw ρ'),
    ('beyond_rho', 'saturation-subtracted ρ'),
    ('pair_rho', 'pair ρ (coarsest)'),
    ('line_rho', 'line ρ'),
    ('cell_mag_pooled', 'cell ρ (residual, finest)'),
]
LEVELS_CELL_COMPARE = [
    ('cell_rank', 'cell rank\n(col ranks of ddG_AB)'),
    ('cell_mag', 'cell ρ (block avg)\n(double-centred r)'),
    ('cell_mag_pooled', 'cell ρ (pooled)\n(double-centred r)'),
    ('cell_mag_gt0', 'cell ρ (dG_add > 0)\n(unsaturated cells)'),
]
SHORT_CATS = {'zeroshot': 'Zero-shot', 'transfer': 'Retrained', 'supervised': 'Supervised'}

FOOT_HIER = [
    'Epistasis = double-mutant prediction minus additive singles: ESM-MSR = 1/2(MT_AB+WT_AB) − WT_A − WT_B (no linking); MutateEverything Stage 2, ThermoMPNN-D, SPURS-multi minus',
    "the same seed's singles (3-seed means); ESM3-small-open unmasked 1/2(MT_AB+WT_AB) − WT_A − WT_B; ProSST Dual-view, ProteinMPNN, GeoDDG-3D, Rosetta, ESM3ΔG minus sum of singles. Intervals: 95% bootstrap over domains & seeds.",
    "Levels (residuals r = dddG − E[dddG | x] and r' = dddG' − E[dddG' | x'] via 25-quantile-bin smoothers on additive dG expectations x, x'): (1) naive / raw ρ: pooled over all doubles on raw dddG;",
    "(2) saturation-subtracted ρ: pooled over all doubles on residuals r', r; (3) pair ρ: across position-pair matrices (≥20 doubles) of each pair's mean residual;",
    '(4) line ρ: pooled over all row/column lines (≥3 cells) of pair-centred line means; (5) cell rank: per position-pair complete block (≥3×3), double-centred within-column ranks averaged across blocks.',
]


# ----------------------------------------------------------------------------------------------
# 1. Core epistasis metrics, scatters, ddG ranking, and lambda scan (06_epistasis.py)
# ----------------------------------------------------------------------------------------------
def _rmse_boot(fr, mask: np.ndarray, key: str, B: int = 1000, seed: int = 0) -> tuple[float, tuple[float, float], float]:
    sp = fr.spine
    y = sp['dddG_ML'].to_numpy()
    cl = sp['cluster'].astype(str).to_numpy()
    ok = mask & np.isfinite(fr.preds[key]).all(axis=0)
    u, inv = np.unique(cl[ok], return_inverse=True)
    rng = np.random.default_rng(seed)
    e = np.stack([(p[ok] - y[ok]) for p in fr.preds[key]])
    pt = float(np.mean(np.sqrt((e ** 2).mean(1))))
    bias = float(e.mean())
    ns = e.shape[0]
    reps = []
    for _ in range(B):
        c = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u))).astype(float)[inv]
        sw = rng.multinomial(ns, np.full(ns, 1.0 / ns)).astype(float) / ns if ns > 1 else np.ones(1)
        per_seed_rm = np.sqrt(((e ** 2) * c).sum(1) / c.sum())
        reps.append(float(np.dot(sw, per_seed_rm)))
    return pt, ci(reps), bias


def _eval_naive_epistasis_sets(eval_sets: list[str], tsu_pair: list[str], epi_keys: list[str], B: int) -> pd.DataFrame:
    from .p01_benchmarks_external import _fast_bootstrap_rho

    rows = []
    for ds in eval_sets:
        fr = load_frame(ds)
        m = epi_rows(fr)
        keys_ds = [k for k in epi_keys if k in fr.preds]
        sub = subframe(fr, m, y=fr.spine['dddG_ML'].to_numpy(), keys=keys_ds)
        kind = 'protein' if ds in tsu_pair else 'mutation'
        point, reps = _fast_bootstrap_rho({ds: sub}, keys_ds, kind=kind, B=B, seed=3, per_protein=False)
        for k, (r_val, per_seed, n, ng) in point[ds].items():
            lo, hi = ci(reps[ds][k])
            r = dict(
                dataset=ds, key=k, kind=kind, naive_rho=r_val, lo=lo, hi=hi,
                n_doubles=n, n_libraries=ng,
                seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                doc_naive_rho=np.nan, doc_global_err=np.nan,
                global_err=np.nan, global_err_lo=np.nan, global_err_hi=np.nan, global_bias=np.nan,
            )
            if ds in tsu_pair:
                rm, (rlo, rhi), bias = _rmse_boot(fr, m, k, B=B)
                r.update(global_err=rm, global_err_lo=rlo, global_err_hi=rhi, global_bias=bias)
            rows.append(r)
    return pd.DataFrame(rows)


def _render_dddg_scatter_grid(sc_sets: list[tuple[str, str]], epi_keys: list[str], style_ov: dict, out_stem, suptitle: str, unit_lbl: str = 'kcal/mol') -> None:
    from scipy.stats import pearsonr

    fig_sc, axes_sc = plt.subplots(len(sc_sets), len(epi_keys), figsize=(3.0 * len(epi_keys), 3.1 * len(sc_sets)), squeeze=False)
    stat_rows = []
    for r_i, (ds, lab) in enumerate(sc_sets):
        fr = load_frame(ds)
        m = epi_rows(fr)
        y_dd = fr.spine['dddG_ML'].to_numpy()
        for c_i, k in enumerate(epi_keys):
            ax = axes_sc[r_i, c_i]
            if k not in fr.preds:
                ax.axis('off')
                continue
            p = fr.preds[k].mean(axis=0)
            ok = m & np.isfinite(p) & np.isfinite(y_dd)
            style.density_scatter(ax, p[ok], y_dd[ok], s=4, cmap='viridis', line=(unit_lbl == 'kcal/mol'), max_points=40000)
            ax.axhline(0, color='#bbb', lw=0.6)
            ax.axvline(0, color='#bbb', lw=0.6)
            if r_i == 0:
                ax.set_title(style.style_of(k, overrides=style_ov.get(k)).label, fontsize=8.5)
            if c_i == 0:
                ax.set_ylabel(f'{lab}\nmeasured dddG ({unit_lbl})' if unit_lbl else f'{lab}\nmeasured epistasis ε')
            if r_i == len(sc_sets) - 1:
                ax.set_xlabel(f'predicted epistasis ({unit_lbl})' if unit_lbl else 'predicted epistasis (kcal/mol)')
            stat_rows.append(dict(
                dataset=ds, key=k, n_doubles=int(ok.sum()),
                spearman_rho=float(spearmanr(p[ok], y_dd[ok])[0]),
                pearson_r=float(pearsonr(p[ok], y_dd[ok])[0]),
            ))
    fig_sc.suptitle(suptitle, y=1.0, fontsize=10)
    fig_sc.tight_layout()
    style.save(fig_sc, out_stem)
    atomic_to_csv(pd.DataFrame(stat_rows), out_stem.parent / f'{out_stem.name}.csv', index=False, float_format='%.4f')


def _run_lambda_scan_sets(scan_sets: list[tuple[str, str]], tag_lam: str, B: int, out_stem) -> None:
    lam_rows, curves = [], {}
    for ds, _lab in scan_sets:
        fr = load_frame(ds)
        sp_l = fr.spine
        m_l = (sp_l['n_mut'] >= 2).to_numpy() & sp_l['use'].to_numpy()
        A_p, M_p = fr.preds['msrp0:wt'][:, m_l], fr.preds['msrp0:mt'][:, m_l]
        ok = np.isfinite(A_p).all(0) & np.isfinite(M_p).all(0)
        A_p, M_p, y_l = A_p[:, ok], M_p[:, ok], sp_l['y'].to_numpy()[m_l][ok]
        n_l = len(y_l)
        g_l = pd.factorize(sp_l['group'].to_numpy()[m_l][ok])[0] if ds in PER_PROTEIN else np.zeros(n_l, np.int64)
        G_l = int(g_l.max()) + 1
        red = (lambda r: np.nanmean(r)) if G_l > 1 else (lambda r: r[0])
        eng = SpearmanEngine(y_l, g_l, G_l)
        ranks = [[eng.prepare((1 - l_v) * a + l_v * mm) for l_v in LAMBDAS] for a, mm in zip(A_p, M_p)]
        w1 = np.ones(n_l)
        pt = np.array([np.mean([red(eng.rho(ranks[s][i], w1)) for s in range(len(A_p))]) for i in range(len(LAMBDAS))])
        b_lam = min(B, 200 if n_l < 20000 else 80)
        rng = np.random.default_rng(1)
        reps_l = np.empty((b_lam, len(LAMBDAS)))
        for b_i in range(b_lam):
            w = rng.multinomial(n_l, np.full(n_l, 1.0 / n_l)).astype(float)
            sw = rng.multinomial(3, np.full(3, 1.0 / 3.0)).astype(float) / 3.0
            ry = eng.ry.ranks(w)
            reps_l[b_i] = [float(np.dot(sw, [red(eng.rho(ranks[s][i], w, ry)) for s in range(len(A_p))])) for i in range(len(LAMBDAS))]
        curves[ds] = (pt, reps_l)
        i0, i5, i1, ib = 0, int(np.where(LAMBDAS == 0.5)[0][0]), len(LAMBDAS) - 1, int(pt.argmax())
        lam_rows.append(dict(
            checkpoint=tag_lam, dataset=ds, n=n_l,
            rho_lam0=pt[i0], rho_lam05=pt[i5], rho_lam1=pt[i1],
            best_lambda=LAMBDAS[ib], rho_best=pt[ib],
            gap_05_to_best=pt[ib] - pt[i5],
            gap_lo=float(np.quantile(reps_l[:, ib] - reps_l[:, i5], 0.025)),
            gap_hi=float(np.quantile(reps_l[:, ib] - reps_l[:, i5], 0.975)),
        ))
    atomic_to_csv(pd.DataFrame(lam_rows), out_stem.parent / f'{out_stem.name}.csv', index=False, float_format='%.4f')
    fig_l, axes_l = plt.subplots(1, len(scan_sets), figsize=(max(3.3 * len(scan_sets), 4.4), 3.3), sharey=False, squeeze=False)
    for ax, (ds, lab) in zip(axes_l[0], scan_sets):
        pt, reps_l = curves[ds]
        lo, hi = np.quantile(reps_l, [0.025, 0.975], axis=0)
        ax.fill_between(LAMBDAS, lo, hi, color='k', alpha=0.15, lw=0)
        ax.plot(LAMBDAS, pt, color='k', lw=1.6, label=tag_lam)
        ax.plot(LAMBDAS[pt.argmax()], pt.max(), 'o', color='k', ms=4)
        ax.axvline(0.5, color='#c0392b', lw=0.8, ls=':')
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel('λ  (fraction of MT epistasis kept)')
    axes_l[0, 0].set_ylabel('ρ on multi-mutants')
    axes_l[0, 0].legend(fontsize=7.5)
    fig_l.tight_layout()
    style.save(fig_l, out_stem)


def run_core_epistasis(B: int, split: str = 'capped') -> None:
    out = out_dir('epistasis', split=split)
    mega = (split == 'mega')
    tsu_pair = tsu_datasets(split)
    epi_keys = ['msrp0:epi', 'me2:epi', 'tmpnnd:epi', 'spursm:epi']
    style_ov = {}
    if mega:
        epi_keys += ['spursum:epi', 'tmpnndpub:epi']
        style_ov['spursm:epi'] = {'label': 'SPURS multi, frozen adapter (vs singles)'}

    # 1a. Naive rho and RMSE recomputed — MegaScale only in 01_megascale/epistasis/, 3-DMS in 99_misc/dms3/
    R_tsu = _eval_naive_epistasis_sets(list(tsu_pair), tsu_pair, epi_keys, B=B)
    atomic_to_csv(R_tsu, out / 'epistasis_naive_recomputed.csv', index=False, float_format='%.4f')

    tsu_xlabels = ['MegaScale val (mega_splits)', 'MegaScale test (mega_splits)'] if mega else ['MegaScale val', 'MegaScale test']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0 if mega else 4.8), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    style.dot_panel(
        axes[0], list(tsu_pair), epi_keys,
        {(r.dataset, r.key): r.naive_rho for r in R_tsu.itertuples()},
        {(r.dataset, r.key): (r.lo, r.hi) for r in R_tsu.itertuples()},
        highlight=['msrp0:epi'], ylabel='naive ρ: predicted vs measured dddG (all doubles)',
        xlabels=tsu_xlabels, size=6.5, style_overrides=style_ov,
    )
    axes[0].set_title(f'A  naive ρ; whiskers: 95% library-level bootstrap (B = {B:,})', loc='left', fontsize=9)
    style.dot_panel(
        axes[1], list(tsu_pair), epi_keys,
        {(r.dataset, r.key): r.global_err for r in R_tsu.itertuples()},
        {(r.dataset, r.key): (r.global_err_lo, r.global_err_hi) for r in R_tsu.itertuples()},
        highlight=['msrp0:epi'], ylabel='RMSE of predicted epistasis (kcal/mol)',
        xlabels=tsu_xlabels, size=6.5, style_overrides=style_ov,
    )
    axes[1].set_title('B  global error (lower is better)', loc='left', fontsize=9)
    hs_naive = [
        Line2D([0], [0], label=style.style_of(k, overrides=style_ov.get(k)).label, **style.marker_kwargs(k, size=6.5, overrides=style_ov.get(k)))
        for k in epi_keys
    ]
    fig.legend(
        handles=hs_naive, loc='upper center', bbox_to_anchor=(0.5, 0.995),
        ncol=4, fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86 if mega else 0.90))
    style.save(fig, out / 'epistasis_naive_recomputed')

    if not mega:
        out_dms = out_dir('dms3')
        R_dms = _eval_naive_epistasis_sets(list(DMS_DOUBLES), tsu_pair, epi_keys, B=B)
        atomic_to_csv(R_dms, out_dms / 'epistasis_naive_recomputed.csv', index=False, float_format='%.4f')
        fig_d, ax_d = plt.subplots(figsize=(7.2, 4.6))
        style.dot_panel(
            ax_d, list(DMS_DOUBLES), epi_keys,
            {(r.dataset, r.key): r.naive_rho for r in R_dms.itertuples()},
            {(r.dataset, r.key): (r.lo, r.hi) for r in R_dms.itertuples()},
            highlight=['msrp0:epi'], ylabel='naive ρ: predicted vs measured epistasis (all doubles)',
            xlabels=['GRB2 bind.', 'DLG4 bind.'], size=6.8, style_overrides=style_ov,
        )
        style.model_legend(ax_d, epi_keys, loc='upper left', fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', style_overrides=style_ov)
        ax_d.set_title(f'3-DMS combinatorial binding sets: naive epistasis ρ (95% mutation-level bootstrap, B = {B:,})', loc='left', fontsize=9.2)
        fig_d.tight_layout()
        style.save(fig_d, out_dms / 'epistasis_naive_recomputed')

    # 1b. Raw dddG scatters (MegaScale only in 01_megascale/epistasis/; GRB2 & DLG4 in 99_misc/dms3/)
    _render_dddg_scatter_grid(
        [(tsu_pair[1], 'MegaScale test (mega_splits)' if mega else 'MegaScale test')],
        epi_keys, style_ov, out / 'epistasis_scatter_dddG',
        suptitle='MegaScale test doubles: predicted vs measured epistasis (seed mean 1–3; dashed = y = x)',
        unit_lbl='kcal/mol',
    )
    if not mega:
        _render_dddg_scatter_grid(
            [('GRB2_HUMAN_Faure_2021_binding_domain', 'GRB2 bind.'), ('DLG4_HUMAN_Faure_2021_binding_domain', 'DLG4 bind.')],
            epi_keys, style_ov, out_dir('dms3') / 'epistasis_scatter_dddG',
            suptitle='3-DMS combinatorial binding doubles (Faure et al.): predicted vs measured epistasis (seed mean 1–3)',
            unit_lbl='',
        )

    fr_test = load_frame(tsu_pair[1])
    sp_t = fr_test.spine
    y_t = sp_t['y'].to_numpy()
    m_dbl = ((sp_t['n_mut'] == 2) & sp_t['use']).to_numpy()
    groups_t = sp_t['group'].to_numpy()
    items_ddg = [
        ('msrp0:comb', 'ESM-MSR Dual-view'), ('msrp0:wt', 'ESM-MSR WT-view\n(additive)'),
        ('me1', 'MutateEverything Stage 1'), ('me2', 'MutateEverything Stage 2'),
        ('tmpnn', 'ThermoMPNN'), ('tmpnnd', 'ThermoMPNN-D'),
        ('spurs', 'SPURS'), ('spursm', 'SPURS-multi'),
    ]
    ddg_rows = []
    for k, _ in items_ddg:
        pooled_l, per_l = [], []
        for p in fr_test.preds[k]:
            ok = m_dbl & np.isfinite(p)
            pooled_l.append(spearmanr(p[ok], y_t[ok])[0])
            r_g = [spearmanr(p[ok & (groups_t == g)], y_t[ok & (groups_t == g)])[0] for g in np.unique(groups_t[ok]) if (ok & (groups_t == g)).sum() >= 3]
            per_l.append(np.nanmean(r_g))
        ddg_rows.append(dict(model=k, pooled_rho=float(np.mean(pooled_l)), per_library_mean_rho=float(np.mean(per_l)), n_doubles=int(m_dbl.sum())))
    T_ddg = pd.DataFrame(ddg_rows)
    atomic_to_csv(T_ddg, out / 'doubles_pooled_vs_per_library.csv', index=False, float_format='%.4f')

    ncol = (len(items_ddg) + 1) // 2
    fig_ddg, axes_ddg = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.6), sharey=True)
    for ax, (k, ttl) in zip(axes_ddg.ravel(), items_ddg):
        p = fr_test.preds[k].mean(axis=0)
        ok = m_dbl & np.isfinite(p)
        style.density_scatter(ax, p[ok], y_t[ok], s=3, line=True, max_points=30000, label_stats=False)
        r_row = T_ddg[T_ddg['model'] == k].iloc[0]
        ax.set_title(ttl, fontsize=8.8)
        ax.text(
            0.04, 0.96, f'pooled ρ = {r_row.pooled_rho:.3f}\nper-library ρ = {r_row.per_library_mean_rho:.3f}',
            transform=ax.transAxes, va='top', fontsize=8.2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9),
        )
    for ax in axes_ddg[1]:
        ax.set_xlabel('predicted ddG of the double')
    for ax in axes_ddg[:, 0]:
        ax.set_ylabel('measured ddG (kcal/mol)')
    fig_ddg.suptitle(f'MegaScale test doubles (n = {int(m_dbl.sum()):,}; prediction averaged over seeds 1-3, latent predictions)', y=1.01, fontsize=9)
    fig_ddg.tight_layout()
    style.save(fig_ddg, out / 'epistasis_scatter_ddG_doubles')

    # 1c. Table 2.3: ddG ranking
    ms = ['msrp0:wt', 'msrp0:comb', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
    if mega:
        ms += [k for k in ('spursu', 'spursum', 'tmpnnpub', 'tmpnndpub') if k in fr_test.preds]
    rk_rows = []
    for ds, sp_lbl in zip(tsu_pair, ('val', 'test')):
        fr = load_frame(ds)
        sp_d = fr.spine
        dbl = (sp_d['n_mut'] == 2).to_numpy() & sp_d['use'].to_numpy()
        e_m = dbl & np.isfinite(sp_d['dddG_ML'].to_numpy())
        epi_src = ['msrp0:epi', 'me2:epi', 'tmpnnd:epi', 'spursm:epi'] + (['spursum:epi', 'tmpnndpub:epi'] if mega else [])
        epi_src = [k for k in epi_src if k in fr.preds]
        ren = {
            'msrp0:epi': 'msrp0:comb', 'me2:epi': 'me2', 'tmpnnd:epi': 'tmpnnd', 'spursm:epi': 'spursm',
            'spursum:epi': 'spursum', 'tmpnndpub:epi': 'tmpnndpub',
        }
        sets_rk = {
            'all variants': (fr, ms),
            'doubles': (subframe(fr, dbl, keys=ms), ms),
            'dddG of doubles': (subframe(fr, e_m, y=sp_d['dddG_ML'].to_numpy(), keys=epi_src, rename=ren), [ren[k] for k in epi_src]),
        }
        for nm, (f_sub, ks_sub) in sets_rk.items():
            pt_rk, rp_rk = bootstrap_rho({ds: f_sub}, ks_sub, kind='protein', B=B, seed=4, per_protein=True)
            for k, (r_v, per_s, n_v, ng_v) in pt_rk[ds].items():
                lo, hi = ci(rp_rk[ds][k])
                rk_rows.append(dict(
                    split=sp_lbl, metric=nm, model=k, rho=r_v, lo=lo, hi=hi,
                    seed_sd=float(np.std(per_s, ddof=1)) if len(per_s) > 1 else np.nan,
                    n=n_v, n_proteins=ng_v,
                ))
    R_rk = pd.DataFrame(rk_rows)
    atomic_to_csv(R_rk, out / 'ddg_ranking_recomputed.csv', index=False, float_format='%.4f')
    grp_rk = [(s, m) for s in ('val', 'test') for m in ('all variants', 'doubles', 'dddG of doubles')]
    fig_rk, ax_rk = plt.subplots(figsize=(11, 4.4))
    style.dot_panel(
        ax_rk, [f'{s} {m}' for s, m in grp_rk], ms,
        {(f'{r.split} {r.metric}', r.model): r.rho for r in R_rk.itertuples()},
        {(f'{r.split} {r.metric}', r.model): (r.lo, r.hi) for r in R_rk.itertuples()},
        highlight=['msrp0:comb'], ylabel='mean per-protein Spearman ρ',
        xlabels=[f'{s}\n{m}' for s, m in grp_rk], rot=0, size=6.5,
    )
    style.model_legend(ax_rk, ms, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=8, fontsize=7.5, columnspacing=1.0, handletextpad=0.3)
    ax_rk.set_title(f'Table 2.3: ddG ranking; whiskers 95% protein-level bootstrap (B = {B:,})', loc='left', fontsize=9)
    fig_rk.tight_layout()
    style.save(fig_rk, out / 'ddg_ranking')

    # 1d. Table 2.6: lambda scan — MegaScale in 01_megascale/epistasis/lambda_scan, 3-DMS in 99_misc/dms3/epistasis_lambda_scan
    scan_sets_main = (
        [(tsu_pair[1], 'MegaScale test (mega_splits)')] if mega else
        [(tsu_pair[1], 'MegaScale test'), ('ptmul', 'PTMUL'), ('ptmuld', 'PTMUL-D')]
    )
    tag_lam = 'ESM-MSR (FINAL_p0_mega, 3 seeds)' if mega else 'ESM-MSR (FINAL_p0, 3 seeds)'
    _run_lambda_scan_sets(scan_sets_main, tag_lam, B=B, out_stem=out / 'lambda_scan')
    if not mega:
        scan_sets_dms = [
            ('GRB2_HUMAN_Faure_2021_binding_domain', 'GRB2 bind.'),
            ('DLG4_HUMAN_Faure_2021_binding_domain', 'DLG4 bind.'),
        ]
        _run_lambda_scan_sets(scan_sets_dms, tag_lam, B=B, out_stem=out_dir('dms3') / 'epistasis_lambda_scan')


# ----------------------------------------------------------------------------------------------
# 2. Publication epistasis bars (fig3, fig4, fig3_esm3, 11l p-values, 11h stacked, 11m masking)
# ----------------------------------------------------------------------------------------------
def _render_epi_vs_add_table(res: pd.DataFrame, metric: str, out_dir_path) -> None:
    label_map = {
        'msrp0:wt+': 'WT_A + WT_B', 'msrp0:comb': 'Dual-view',
        'me1': 'Stage 1', 'me2': 'Stage 2',
        'tmpnn': 'ThermoMPNN', 'tmpnnd': 'ThermoMPNN-D',
        'spurs': 'SPURS single', 'spursm': 'SPURS-multi',
        'msrrel@0:wt+': 'WT_A + WT_B', 'msrrel@0:comb': 'Dual-view',
        'pmpnn+': 'sum of singles', 'pmpnn': 'direct double',
        'geoddg+': 'sum of singles', 'geoddg': 'direct double',
        'rosetta+': 'sum of singles', 'rosetta': 'direct double',
        'esm3dg+': 'sum of singles', 'esm3dg': 'direct double',
        'ddgemb+': 'sum of singles', 'ddgemb': 'direct double',
    }
    res_l = res.assign(additive_label=res['additive'].map(label_map), epistatic_label=res['epistatic'].map(label_map))
    nd = (metric != 'spearman')
    head0 = ['Model', 'Additive →\nepistatic', 'ρ\nadditive', 'ρ\nepistatic', 'Δρ [95% CI]', 'Domains\nepistatic higher', 'p\none-sided', 'p one-sided\n(Holm)', 'p\ntwo-sided']
    headers = [h.replace('ρ', 'NDCG@96') if nd else h for h in head0]
    rows = [
        [
            r.model, f'{r.additive_label} → {r.epistatic_label}', f'{r.rho_additive:.3f}', f'{r.rho_epistatic:.3f}',
            f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]', f'{r.domains_higher} / {r.n_domains}',
            fmt_p(r.p_one, B=r.B, prefix=''), fmt_p(r.p_one_holm, B=r.B, prefix=''), fmt_p(r.p_two, B=r.B, prefix=''),
        ]
        for r in res_l.itertuples()
    ]
    sfx = '' if metric == 'spearman' else f'_{metric}'
    stem = out_dir_path / f'epistatic_vs_additive{sfx}_pvalues'
    widths = [2.5, 3.2, 1.25, 1.25, 2.8, 1.5, 1.05, 1.25, 1.05] if nd else [2.5, 3.2, 0.9, 0.9, 2.8, 1.5, 1.05, 1.25, 1.05]
    fig = plt.figure(figsize=(sum(widths) + 0.4, 0.42 * (len(rows) + 1.6) + 0.85))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 0.62 / fig.get_figheight(), sum(widths) / fig.get_figwidth(), 1 - 0.7 / fig.get_figheight()])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=headers, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(widths) for w in widths], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#cccccc')
        c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
            continue
        r = res_l.iloc[i - 1]
        if j == 0:
            c.set_text_props(ha='left', fontweight='bold')
            c._loc = 'left'
        if j == 7 and r.p_one_holm < 0.05:
            c.set_facecolor('#d6ecd9')
        elif j == 7 and r.delta < 0 and r.p_two < 0.05:
            c.set_facecolor('#f6d5d0')
    b_val = int(res_l['B'].iloc[0])
    n_dom = int(res_l['n_domains'].iloc[0])
    desc = (
        f'Tsuboyama capped test set, double mutants. NDCG@96 with shifted relevance ΔΔG − min(ΔΔG) over {n_dom} domains (≥ {MIN_N} doubles); Δ = epistatic − additive. '
        if metric == 'ndcg96' else
        f'Tsuboyama capped test set, double mutants. Rectified NDCG@96 max(ΔΔG, 0) over {n_dom} domains with a stabilizing double; Δ = epistatic − additive. '
        if metric == 'ndcg96_rectified' else
        f'Tsuboyama capped test set, double mutants. ρ: mean over {n_dom} test domains (≥ {MIN_N} doubles) of per-domain Spearman ρ, seeds averaged; Δρ = epistatic − additive. '
    )
    note = desc + f'Paired bootstrap over domains & seeds ({b_val:,} draws shared by all models). One-sided p = P(Δ ≤ 0); Holm over models.'
    fig.text(0.2 / fig.get_figwidth(), 0.5 / fig.get_figheight(), note, fontsize=8.5, color='#444444', va='top', ha='left')
    style.save(fig, stem)
    md_lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in headers) + ' |', '|' + '---|' * len(headers)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    atomic_write_text(out_dir_path / f'epistatic_vs_additive{sfx}_pvalues.md', '\n'.join(md_lines) + '\n')


def _draw_pair_brackets(ax, cols: list[dict], T: pd.DataFrame, p_series: pd.Series, subsets=('additive', 'epistatic'), p_name_map: dict | None = None, B_p: int | None = None) -> float:
    nm = p_name_map or {}
    drawn = [c for c in cols if c.get('draw', True)]
    xs, _ = style.layout_grouped_columns(drawn, gap=0.7)
    top = 0.0
    for c, x in zip(drawn, xs):
        r = [T[(T['column'] == c['name']) & (T['subset'] == sub)] for sub in subsets]
        name = nm.get(c['name'], c['name'])
        if name not in p_series.index or not all(len(q) for q in r):
            continue
        hi = max(float(q.iloc[0]['hi']) for q in r)
        y = hi + 0.135
        x0, x1 = x - 0.16, x + 0.16
        h = 0.014
        ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color='#333333', lw=1.0, solid_capstyle='butt', clip_on=False, zorder=9)
        ax.text((x0 + x1) / 2, y + 0.008, fmt_p(float(p_series[name]), B=B_p, prefix='p = '), ha='center', va='bottom', fontsize=8, color='#333333', zorder=9)
        top = max(top, y + 0.075)
    return top


def run_publication_epistasis_bars(B: int, split: str = 'capped') -> tuple[pd.DataFrame, pd.Series | None]:
    out = out_dir('epistasis', split=split)
    fr, _ = load_tsu_test_frame(split=split)
    add_esm3_marginal_additive(fr, split=split)
    sp = fr.spine
    use = sp['use'].to_numpy()
    n_mut = sp['n_mut'].to_numpy()
    double = use & (n_mut == 2)
    dd = sp['dddG_ML'].to_numpy()
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = make_domain_counts(libs, B=B, seed=5)
    b_pool = min(B, 200)
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    s_title = f'{split_title(split)} test'

    # Doubles additive vs epistatic
    cols3 = pair_columns(fr, best)
    v3: dict = {}
    T3 = table_grouped(fr, cols3, {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}, rho_fn, MIN_N, counts, libs, vals=v3)
    atomic_to_csv(T3, out / 'spearman_doubles_additive_vs_epistatic.csv', index=False, float_format='%.4f')
    d_rows = []
    for c in cols3:
        if c['double'] is None or c['single'] not in fr.preds or c['double'] not in fr.preds:
            continue
        va, ve = per_library(fr, c['single'], double, rho_fn, MIN_N), per_library(fr, c['double'], double, rho_fn, MIN_N)
        both = va.index.intersection(ve.index)
        pt, lo, hi, n = boot_mean(diff_df(ve, va, both), libs, counts)
        d_rows.append(dict(column=c['name'], delta=pt, lo=lo, hi=hi, n_libraries=n))
    atomic_to_csv(pd.DataFrame(d_rows), out / 'spearman_doubles_additive_vs_epistatic_paired_difference.csv', index=False, float_format='%.4f')

    gain = 'gain (epistatic - additive)'
    p3 = []
    for c in cols3:
        if (c['name'], 'additive') in v3 and (c['name'], 'epistatic') in v3:
            _ka, va, _ma, _ya = v3[(c['name'], 'additive')]
            _ke, ve, _me, _ye = v3[(c['name'], 'epistatic')]
            both = va.index.intersection(ve.index)
            d = diff_df(ve, va, both)
            v3[(c['name'], gain)] = (None, d, None, None)
            p3.append(row_of('own gain', f"{name_map.get(c['name'], c['name'])} (epistatic)", f"{name_map.get(c['name'], c['name'])} (additive)", mean_val(ve.loc[both]), mean_val(va.loc[both]), mean_val(d), boot_reps(d, libs, counts), len(both), 'domains', 'mean over domains', name_map))
    p3_all = compare_grouped(fr, v3, ['additive', 'epistatic', gain], counts, libs, name_map=name_map) + p3
    atomic_to_csv(add_holm_by_group(pd.DataFrame(p3_all)), out / 'spearman_doubles_additive_vs_epistatic_pvalues.csv', index=False, float_format='%.4g')

    fig3, ax3 = plt.subplots(figsize=(fig_width(cols3), 4.9))
    style.bar_panel(
        ax3, cols3, T3, ('additive', 'epistatic'),
        (count_text(T3, 'additive', 'additive: singles model summed'), count_text(T3, 'epistatic', 'epistatic: model scores the double')),
        "Spearman's ρ, double mutants", (0.0, 1.12),
    )
    finish_bar_fig(fig3, f'{s_title} doubles, additive version against epistatic version; 95% intervals from resampling domains (B = {B:,})', [])
    style.save(fig3, out / 'spearman_doubles_additive_vs_epistatic')

    # Naive epistasis positive vs negative (grouped and pooled)
    erows = epi_rows(fr)
    cells4 = {'positive': (erows & (dd > 0), 'doubles'), 'negative': (erows & (dd < 0), 'doubles')}
    cols4 = epi_columns(fr, best)
    v4: dict = {}
    Tg = table_grouped(fr, cols4, cells4, rho_fn, MIN_N, counts, libs, y=dd, vals=v4)
    Og = table_grouped(fr, cols4, {'overall': (erows, 'doubles')}, rho_fn, MIN_N, counts, libs, y=dd, vals=v4)
    atomic_to_csv(pd.concat([Tg, Og]), out / 'naive_epistasis_pos_neg_grouped.csv', index=False, float_format='%.4f')
    atomic_to_csv(add_holm_by_group(pd.DataFrame(compare_grouped(fr, v4, ['positive', 'negative', 'overall'], counts, libs, name_map=name_map))), out / 'naive_epistasis_pos_neg_grouped_pvalues.csv', index=False, float_format='%.4g')

    rp4: dict = {}
    Tp = table_pooled(fr, cols4, {**cells4, 'overall': (erows, 'doubles')}, dd, B_=b_pool, reps_out=rp4)
    atomic_to_csv(Tp, out / 'naive_epistasis_pos_neg_pooled.csv', index=False, float_format='%.4f')
    atomic_to_csv(add_holm_by_group(pd.DataFrame(compare_pooled(Tp, rp4, ['positive', 'negative', 'overall'], name_map=name_map))), out / 'naive_epistasis_pos_neg_pooled_pvalues.csv', index=False, float_format='%.4g')

    for tab, ov, stem, yl in (
        (Tg, Og, 'naive_epistasis_pos_neg_grouped', 'naive epistasis ρ (mean over domains)'),
        (Tp[Tp['subset'] != 'overall'], Tp[Tp['subset'] == 'overall'], 'naive_epistasis_pos_neg_pooled', 'naive epistasis ρ, pooled over domains'),
    ):
        drawn = [c['name'] for c in cols4 if c.get('draw', True)]
        lo_min = min(tab[tab['column'].isin(drawn)]['lo'].min(), ov[ov['column'].isin(drawn)]['lo'].min())
        ylo = float(np.floor((lo_min - 0.02) / 0.05) * 0.05)
        fig4, ax4 = plt.subplots(figsize=(fig_width(cols4), 5.15 - 2.0 * (1.6 - (1.0 - ylo))))
        style.bar_panel(
            ax4, cols4, tab, ('positive', 'negative'),
            (count_text(tab, 'positive', 'positive epistasis, dddG > 0'), count_text(tab, 'negative', 'negative epistasis, dddG < 0')),
            yl, (ylo, 1.0), overall_df=ov, overall_label=f'overall (n = {int(ov.n_variants.max()):,})', zero_line=True,
        )
        ax4.set_xlabel(f'predicted epistasis ({EPI_DESC} for ESM-MSR / ESM3; direct − sum of singles for others)', fontsize=8.3)
        finish_bar_fig(fig4, f'{s_title} doubles with both singles measured: predicted vs measured dddG; 95% intervals from resampling domains', [], top=0.95)
        style.save(fig4, out / stem)

    if split != 'capped':
        return T3, None

    # ESM3 scoring variants (11e, epistasis part)
    if 'esm3avg' in fr.preds and 'esm3mm+' in fr.preds:
        cols3_e = pair_columns(fr, best)
        grey = [M.color('ESM3-small-open'), '#aab7b8', '#4d5656']
        three3 = [
            col_spec('ESM3 unmasked', 'zeroshot', grey[0], 'ESM3-small-open\nunmasked\nWT → combined', 'msrrel@0:wt', 'msrrel@0:comb'),
            col_spec('ESM3 independent masking', 'zeroshot', grey[1], 'ESM3-small-open\nind. masking\nWT → combined', 'esm3ind+', 'esm3avg'),
            col_spec('ESM3 masked marginal', 'zeroshot', grey[2], 'ESM3-small-open\nmasked marginal\nsingles → double', 'esm3mm+', 'esm3mm'),
        ]
        idx_e = next(i for i, c in enumerate(cols3_e) if c['name'] == 'ESM3-small-open')
        cols3_e = cols3_e[:idx_e] + three3 + cols3_e[idx_e + 1:]
        T3e = table_grouped(fr, cols3_e, {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}, rho_fn, MIN_N, counts, libs)
        atomic_to_csv(T3e, out / 'esm3_scoring_variants.csv', index=False, float_format='%.4f')
        fig3e, ax3e = plt.subplots(figsize=(1.2 * fig_width(cols3_e), 4.9))
        style.bar_panel(
            ax3e, cols3_e, T3e, ('additive', 'epistatic'),
            (count_text(T3e, 'additive', 'additive: singles model summed'), count_text(T3e, 'epistatic', 'epistatic: model scores the double')),
            "Spearman's ρ, double mutants", (0.0, 1.12), tick_fs=7.8,
        )
        finish_bar_fig(fig3e, f'{s_title} doubles, additive vs epistatic version, ESM3-small-open scored three ways (B = {B:,})', [])
        style.save(fig3e, out / 'esm3_scoring_variants')

    # 11l: Epistatic vs additive p-values across all 3 metrics (spearman, ndcg96, ndcg96_rectified)
    keep_11l = ['ProSST', 'ProteinMPNN', 'ESM3-small-open', 'ESM-MSR', 'MutateEverything', 'ThermoMPNN', 'SPURS', 'best other ddG', 'Rosetta', 'ESM3ΔG']
    b_11l = B if B <= 200 else max(B, 10000)
    libs_dbl = sorted(np.unique(sp['group'].to_numpy()[double]))
    counts_11l = make_domain_counts(libs_dbl, B=b_11l, seed=5)
    p_spearman_ser = None
    for m_key, (sfx, fn_m, mn_m) in (
        ('spearman', ('', rho_fn, MIN_N)),
        ('ndcg96', ('_ndcg96', ndcg96_shifted, MIN_N)),
        ('ndcg96_rectified', ('_ndcg96_rectified', ndcg96_rectified, 1)),
    ):
        l_rows = []
        for c in cols3:
            if c['name'] not in keep_11l or c['double'] is None:
                continue
            ka, ke = c['single'], c['double']
            va, ve = per_library(fr, ka, double, fn_m, mn_m), per_library(fr, ke, double, fn_m, mn_m)
            both = va.index.intersection(ve.index)
            d = diff_df(ve, va, both)
            reps = boot_reps(d, libs_dbl, counts_11l)
            reps = reps[np.isfinite(reps)]
            dom = d.mean(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            lbl = name_map.get(c['name'], c['name'])
            l_rows.append(dict(
                model=lbl, additive=ka, epistatic=ke, n_domains=len(both), n_seeds=va.shape[1],
                rho_additive=float(va.loc[both].mean(axis=1).mean()),
                rho_epistatic=float(ve.loc[both].mean(axis=1).mean()),
                delta=float(dom.mean()), lo=float(lo), hi=float(hi),
                domains_higher=int((dom > 0).sum()),
                p_one=float(max((reps <= 0).mean(), 1.0 / (len(reps) + 1))),
                p_two=float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1.0 / (len(reps) + 1))),
                B=b_11l,
            ))
        df_11l = pd.DataFrame(l_rows)
        df_11l['p_one_holm'] = holm(df_11l['p_one'].to_numpy())
        atomic_to_csv(df_11l, out / f'epistatic_vs_additive{sfx}_pvalues.csv', index=False, float_format='%.5g')
        _render_epi_vs_add_table(df_11l, m_key, out)
        if m_key == 'spearman':
            p_spearman_ser = df_11l.set_index('model')['p_one']

    # Main-text Figure 1 (vertical architecture schematic 'a' + stacked b/c panels)
    f2_csv = results_dir('megascale', split=split, create=False) / 'spearman_stab_destab.csv'
    cols2 = main_columns(fr, best, 'unmasked')
    if f2_csv.exists():
        t2 = pd.read_csv(f2_csv)
    else:
        y = sp['y'].to_numpy()
        T2_tmp = table_grouped(fr, cols2, {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}, rho_fn, MIN_N, counts, libs)
        ov2_tmp = table_grouped(fr, cols2, {'overall': (use, 'all')}, rho_fn, 3, counts, libs)
        t2 = pd.concat([T2_tmp, ov2_tmp], ignore_index=True)
    try:
        from analysis.main_figures.figure1 import build_figure1
        build_figure1(split=split, t2=t2, t3=T3, p_spearman_ser=p_spearman_ser, B_p=b_11l)
    except Exception as exc:
        print(f'[p06] Skipping figure1 composite: {exc}')

    # 11m: Masking schemes supplement (ESM3-small-open & ESM-MSR across unmasked, independent, marginal)
    out_supp = out_dir('supplementary_figs')
    f_mask = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
    if f_mask is not None and 'esm3avg' in fr.preds and 'esm3mm' in fr.preds:
        fr.preds['ESM3-small-open:unm'] = fr.preds['msrrel@0:comb']
        fr.preds['ESM3-small-open:ind'] = fr.preds['esm3avg']
        fr.preds['ESM3-small-open:mar'] = fr.preds['esm3mm']
        for tag, fam in (('unm', 'FINAL_p0'), ('ind', 'FINAL_p1'), ('mar', 'FINAL_p1_marg')):
            fr.preds[f'ESM-MSR:{tag}'] = f_mask.preds[f'{fam}:comb']
        counts_m = make_domain_counts(libs, B=b_11l, seed=5)
        vals_m, bars_m, pairs_m = {}, [], []
        for model_nm in ('ESM3-small-open', 'ESM-MSR'):
            for sch in ('unm', 'ind', 'mar'):
                v = per_library(fr, f'{model_nm}:{sch}', double, rho_fn, MIN_N)
                vals_m[(model_nm, sch)] = v
                mn_v, lo_v, hi_v, n_v = boot_mean(v, libs, counts_m)
                bars_m.append(dict(model=model_nm, subset='doubles', scheme=sch, rho=mn_v, lo=lo_v, hi=hi_v, n_domains=n_v))
            for sa, sb in (('unm', 'mar'), ('ind', 'mar'), ('unm', 'ind')):
                va, vb = vals_m[(model_nm, sa)], vals_m[(model_nm, sb)]
                both = va.index.intersection(vb.index)
                d = diff_df(va, vb, both)
                reps = boot_reps(d, libs, counts_m)
                reps = reps[np.isfinite(reps)]
                lo_p, hi_p = np.percentile(reps, [2.5, 97.5])
                pairs_m.append(dict(
                    model=model_nm, subset='doubles', scheme_a=sa, scheme_b=sb,
                    delta=float(d.mean(axis=1).mean()), lo=float(lo_p), hi=float(hi_p), n_domains=len(both),
                    p_two=boot_pvalue(reps, null=0.0, alternative='two-sided'),
                    p_one_a_beats_b=boot_pvalue(reps, null=0.0, alternative='greater'),
                    B=b_11l,
                ))
        df_bars_m, df_pairs_m = pd.DataFrame(bars_m), pd.DataFrame(pairs_m)
        atomic_to_csv(df_bars_m, out_supp / 'masking_schemes_values.csv', index=False, float_format='%.5g')
        atomic_to_csv(df_pairs_m, out_supp / 'masking_schemes_pvalues.csv', index=False, float_format='%.5g')
        md_m = ['| model | subset | A vs B | Δρ (A − B) [95% CI] | p two-sided | p one-sided (A beats B) |', '|---|---|---|---|---|---|']
        for r in df_pairs_m.itertuples():
            md_m.append(f'| {r.model} | {r.subset} | {r.scheme_a} vs {r.scheme_b} | {r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {fmt_p(r.p_two, B=r.B, prefix="")} | {fmt_p(r.p_one_a_beats_b, B=r.B, prefix="")} |')
        atomic_write_text(out_supp / 'masking_schemes_pvalues.md', '\n'.join(md_m) + '\n')

        fig_ms, ax_ms = plt.subplots(figsize=(7.6, 5.4))
        off_m = {'unm': -0.28, 'ind': 0.0, 'mar': 0.28}
        ax_ms.set_axisbelow(True)
        ax_ms.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        for i_m, model_nm in enumerate(('ESM3-small-open', 'ESM-MSR')):
            color = M.color(model_nm)
            top_h = {}
            for sch in ('unm', 'ind', 'mar'):
                r = df_bars_m[(df_bars_m['model'] == model_nm) & (df_bars_m['scheme'] == sch)].iloc[0]
                x = i_m + off_m[sch]
                kw_b = (
                    dict(color=color, alpha=1.0, edgecolor='black', lw=0.8) if sch == 'unm' else
                    dict(color=color, alpha=0.62, edgecolor='black', lw=0.8) if sch == 'ind' else
                    dict(color=color, alpha=0.3, edgecolor=color, hatch='////', lw=0.8)
                )
                ax_ms.bar(x, r.rho, 0.26, zorder=3, **kw_b)
                ax_ms.errorbar(x, r.rho, yerr=[[max(r.rho - r.lo, 0)], [max(r.hi - r.rho, 0)]], fmt='none', ecolor=style.RING, elinewidth=1.3, capsize=2.5, zorder=5)
                ax_ms.text(x, r.hi + 0.012, f'{r.rho:.2f}', ha='center', va='bottom', fontsize=8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
                top_h[sch] = r.hi
            q_fn = lambda sa, sb: df_pairs_m[(df_pairs_m['model'] == model_nm) & (df_pairs_m['scheme_a'] == sa) & (df_pairs_m['scheme_b'] == sb)].iloc[0]
            for sa, sb, y_br in (('ind', 'mar', max(top_h['ind'], top_h['mar']) + 0.115), ('unm', 'mar', max(max(top_h['ind'], top_h['mar']) + 0.20, top_h['unm'] + 0.115))):
                x0, x1 = i_m + off_m[sa], i_m + off_m[sb]
                ax_ms.plot([x0, x0, x1, x1], [y_br - 0.012, y_br, y_br, y_br - 0.012], color='#333333', lw=0.9, solid_capstyle='butt', clip_on=False, zorder=9)
                ax_ms.text((x0 + x1) / 2, y_br + 0.006, fmt_p(q_fn(sa, sb).p_two, B=b_11l, prefix='p = '), ha='center', va='bottom', fontsize=7.6, color='#333333', zorder=9)
        ax_ms.set_xticks([0, 1])
        ax_ms.set_xticklabels(['ESM3-small-open', 'ESM-MSR'], fontsize=10.5)
        ax_ms.set_xlim(-0.55, 1.55)
        ax_ms.set_ylim(0.0, 1.18)
        ax_ms.set_yticks(np.arange(0, 1.01, 0.2))
        ax_ms.set_ylabel(f"Spearman's ρ on double mutants (mean over {int(df_bars_m['n_domains'].max())} test domains)")
        h_ms = [
            Patch(fc='#555555', ec='black', label='Unmasked'),
            Patch(fc='#555555', alpha=0.62, ec='black', label='One-at-a-time masking'),
            Patch(fc='#555555', alpha=0.3, ec='#555555', hatch='////', label='Masked marginal'),
        ]
        ax_ms.legend(handles=h_ms, ncol=1, loc='upper left', fontsize=8.5, frameon=True, facecolor='white', edgecolor='#cccccc')
        fig_ms.tight_layout()
        style.save(fig_ms, out_supp / 'masking_schemes')
        build_supp_figure4(df_bars_m=df_bars_m, df_pairs_m=df_pairs_m)

    return T3, p_spearman_ser


def build_supp_figure4(
    df_bars_m: pd.DataFrame | None = None,
    df_pairs_m: pd.DataFrame | None = None,
) -> Path:
    """Build Supplementary Figure 4 (`00_supplementary_figs/supp_figure4.{png,pdf}`).

    Comparison of masking strategies (Unmasked, One-at-a-time masking, Masked marginal)
    on Homology-Restricted MegaScale test double mutants for:
      a. ESM3-small-open (zero-shot)
      b. ESM-MSR (fine-tuned)
    """
    out_supp = out_dir('supplementary_figs')
    if df_bars_m is None:
        df_bars_m = pd.read_csv(out_supp / 'masking_schemes_values.csv')
    if df_pairs_m is None:
        df_pairs_m = pd.read_csv(out_supp / 'masking_schemes_pvalues.csv')

    n_dom = int(df_bars_m['n_domains'].max())
    B_val = int(df_pairs_m['B'].iloc[0]) if 'B' in df_pairs_m.columns and len(df_pairs_m) else 10000

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True, facecolor='white')
    schemes = [
        ('unm', 'Unmasked'),
        ('ind', 'One-at-a-time\nmasking'),
        ('mar', 'Masked\nmarginal'),
    ]
    xs = np.arange(len(schemes), dtype=float)

    for ax, model_nm, title_lbl in (
        (ax_a, 'ESM3-small-open', 'ESM3-small-open (zero-shot)'),
        (ax_b, 'ESM-MSR', 'ESM-MSR (fine-tuned)'),
    ):
        color = M.color(model_nm)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        top_h = {}
        for idx_s, (sch, _lbl) in enumerate(schemes):
            r = df_bars_m[(df_bars_m['model'] == model_nm) & (df_bars_m['scheme'] == sch)].iloc[0]
            kw_b = (
                dict(color=color, alpha=1.0, edgecolor='black', lw=0.9) if sch == 'unm' else
                dict(color=color, alpha=0.62, edgecolor='black', lw=0.9) if sch == 'ind' else
                dict(color=color, alpha=0.30, edgecolor=color, hatch='////', lw=0.9)
            )
            ax.bar(xs[idx_s], r.rho, width=0.56, zorder=3, **kw_b)
            ax.errorbar(
                xs[idx_s], r.rho,
                yerr=[[max(r.rho - r.lo, 0)], [max(r.hi - r.rho, 0)]],
                fmt='none', ecolor=style.RING, elinewidth=1.4, capsize=3.2, zorder=5,
            )
            ax.text(
                xs[idx_s], r.hi + 0.014, f'{r.rho:.2f}',
                ha='center', va='bottom', fontsize=9.0, fontweight='bold' if sch == 'unm' else 'normal',
                color='#111111', zorder=8,
            )
            top_h[sch] = float(r.hi)

        idx_map = {'unm': 0.0, 'ind': 1.0, 'mar': 2.0}
        def _q(sa, sb):
            return df_pairs_m[(df_pairs_m['model'] == model_nm) & (df_pairs_m['scheme_a'] == sa) & (df_pairs_m['scheme_b'] == sb)].iloc[0]

        y1 = max(top_h['ind'], top_h['mar']) + 0.085
        y2 = max(y1 + 0.095, top_h['unm'] + 0.085)
        for sa, sb, y_br in (('ind', 'mar', y1), ('unm', 'mar', y2)):
            x0, x1 = idx_map[sa], idx_map[sb]
            ax.plot([x0, x0, x1, x1], [y_br - 0.014, y_br, y_br, y_br - 0.014],
                    color='#333333', lw=1.0, solid_capstyle='butt', clip_on=False, zorder=9)
            p_txt = fmt_p(float(_q(sa, sb).p_two), B=B_val, prefix='p = ').replace('= <', '<')
            ax.text((x0 + x1) / 2, y_br + 0.008, p_txt, ha='center', va='bottom', fontsize=8.4,
                    fontweight='bold' if float(_q(sa, sb).p_two) < 0.05 else 'normal', color='#222222', zorder=9)

        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in schemes], fontsize=9.4)
        ax.set_xlim(-0.55, 2.55)
        ax.set_title(title_lbl, fontsize=10.6, fontweight='bold', loc='left', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_a.set_ylim(0.0, 1.12)
    ax_a.set_yticks(np.arange(0, 1.01, 0.2))
    ax_a.set_ylabel(f"Spearman's ρ on double mutants\n(mean over {n_dom} test domains)", fontsize=10.2)
    ax_b.tick_params(axis='y', labelleft=False)

    fig.subplots_adjust(left=0.095, right=0.980, top=0.895, bottom=0.220, wspace=0.14)
    fig.canvas.draw()
    pb = ax_b.get_position()
    fig.text(0.014, 0.975, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pb.x0 - 0.042, 0.975, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    note = (
        f'Homology-Restricted MegaScale test double mutants (mean per-domain Spearman ρ across {n_dom} test domains; 3 seeds averaged for ESM-MSR).  '
        f'Whiskers show 95% joint domain × seed bootstrap CIs (B = {B_val:,}); brackets show two-sided paired bootstrap p-values.  '
        f'a. Zero-shot ESM3-small-open shows little difference across masking strategies (unmasked 0.41, one-at-a-time masking 0.41, masked marginal 0.40; all p > 0.2).  '
        f'b. Fine-tuned ESM-MSR benefits significantly from unmasked scoring (0.78) over masked marginal (0.72, p < 0.001), with one-at-a-time masking intermediate (0.75).'
    )
    kw = dict(fontsize=7.8, color='#333333', linespacing=1.25)
    fig.text(0.016, 0.012, style.wrap_to_px(fig, note, 0.968 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'supp_figure4'
    style.save(fig, out_path)
    return out_path


# ----------------------------------------------------------------------------------------------
# 3. Unified 5-level epistasis hierarchy bootstrap & figures (11d + 11b + 11n) + Main Composite
# ----------------------------------------------------------------------------------------------
def _ctx_rev_for_seed(sp: pd.DataFrame, split_key: str, seed: int) -> np.ndarray | None:
    folder = 'splits_oct06_capped-test' if split_key == 'test' else 'splits_oct06_capped-val'
    f = FINAL_CTX / 'predictions' / folder / 'final_checkpoints' / 'FINAL_p0' / f's{seed}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv'
    if not f.exists():
        f = FINAL_CTX / folder / 'final_checkpoints' / 'FINAL_p0' / f's{seed}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv'
    if not f.exists():
        return None
    c = pd.read_csv(f, usecols=['code', 'mut_type', 'mt_lora_pred'], low_memory=False)
    c['code'], c['mut_type'] = c['code'].astype(str), c['mut_type'].astype(str)
    c = c.drop_duplicates(['code', 'mut_type'])
    arr = pd.DataFrame({'code': sp['code'].astype(str), 'mut_type': sp['mut']}).merge(c, on=['code', 'mut_type'], how='left')['mt_lora_pred'].to_numpy(float)
    return arr if np.isfinite(arr).any() else None


def _hier_entry(csv_name: str, model: str, label: str | None = None, color: str | None = None) -> tuple[str, str, str, str]:
    m = M.BY_NAME[model]
    disp = label or (M.version_label(model, 1) if m.two_versions and model != 'ESM-MSR' else m.name)
    return (csv_name, m.group, color or m.color, disp)


def _draw_hierarchy_axes(axes, T: pd.DataFrame, model_entries: list[tuple[str, str, str, str]], levels: list[tuple[str, str]], overlay_sub0: bool = False, compact_fonts: bool = False) -> None:
    cols = [dict(name=n, cat=c, color=col, label=lab) for n, c, col, lab in model_entries if (T['model'] == n).any()]
    xs, spans = style.layout_grouped_columns(cols, gap=1.25)
    metric_names = [m for m, _ in levels] + ([f'{m}_sub0' for m, _ in levels] if overlay_sub0 else [])
    sub_T = T[T['metric'].isin(metric_names) & T['model'].isin([c['name'] for c in cols])]
    lo_ = float(np.nanmin(sub_T[['lo', 'value']].to_numpy()))
    hi_ = float(np.nanmax(sub_T[['hi', 'value']].to_numpy()))
    ylim = (min(-0.12, lo_ - 0.04), max(0.52 if len(levels) == 4 else 0.90, hi_ + 0.14))
    top = 0.016 * (ylim[1] - ylim[0])
    dash_hw = 0.45
    fs_val = 7.4 if compact_fonts else 8.3
    fs_xt = 9.2 if compact_fonts else 11.0
    fs_ttl = 10.5 if compact_fonts else (11.5 if any('\n' in t for _, t in levels) else 13.0)

    for idx, (ax, (metric, level_title)) in enumerate(zip(axes, levels)):
        ax.set_box_aspect(1)
        if metric == 'naive_rho':
            r_ctrl = T[(T['model'] == '−dG_add') & (T['metric'] == 'naive_rho')]
            if len(r_ctrl) and np.isfinite(r_ctrl.iloc[0]['value']):
                ax.axhline(float(r_ctrl.iloc[0]['value']), color='#c0392b', ls=':', lw=1.8, zorder=2.6)
        for c, x in zip(cols, xs):
            r = T[(T['model'] == c['name']) & (T['metric'] == metric)].iloc[0]
            val_sub0 = np.nan
            if overlay_sub0:
                r_sub0 = T[(T['model'] == c['name']) & (T['metric'] == f'{metric}_sub0')]
                if len(r_sub0) and np.isfinite(r_sub0.iloc[0]['value']):
                    val_sub0 = float(r_sub0.iloc[0]['value'])
                    ax.plot([x - dash_hw, x + dash_hw], [val_sub0, val_sub0], color='#111111', lw=2.2, zorder=2.2, solid_capstyle='butt')
            ax.bar(x, r.value, 0.68, color=c['color'], alpha=0.88 if overlay_sub0 else 0.92, edgecolor='black', lw=0.75, zorder=3)
            ax.errorbar(x, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor=style.RING, elinewidth=1.25, capsize=2.3, zorder=5)
            y_txt = max(r.hi, r.value, val_sub0 if np.isfinite(val_sub0) else 0.0, 0.0) + top
            ax.text(x, y_txt, f'{r.value:.2f}', ha='center', va='bottom', fontsize=fs_val, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.3))
        if metric == 'naive_rho':
            handles = []
            if overlay_sub0:
                handles.append(Line2D([0], [0], color='#111111', lw=2.2, label='dG_add > 0'))
            r_ctrl = T[(T['model'] == '−dG_add') & (T['metric'] == 'naive_rho')]
            if len(r_ctrl) and np.isfinite(r_ctrl.iloc[0]['value']):
                handles.append(Line2D([0], [0], color='#c0392b', ls=':', lw=1.8, label='−dG_add'))
            if handles:
                ax.legend(handles=handles, loc='upper right', fontsize=8.4 if compact_fonts else 9.5, framealpha=0.90, handlelength=1.5, borderpad=0.35)
        ax.set_xticks(xs)
        ax.set_xticklabels([c['label'] for c in cols], fontsize=fs_xt, rotation=90, ha='center', va='top')
        ax.tick_params(axis='y', labelsize=9.8 if compact_fonts else 11.5)
        if idx == 0:
            ax.set_ylabel('level metric (higher is better)', fontsize=10.5 if compact_fonts else 13.0)
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(0, 0.45, 0.1) if len(levels) == 4 else np.arange(0, 0.81, 0.2))
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(xs[0] - 0.68, xs[-1] + 0.68)
        for i, (cat, (x0, x1)) in enumerate(spans.items()):
            if i % 2:
                ax.axvspan(x0 - 0.85, x1 + 0.85, color='#000000', alpha=0.045, lw=0, zorder=0)
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.38, x1 + 0.38], [1.015, 1.015], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.028, SHORT_CATS[cat], transform=tr, ha='center', va='bottom', fontsize=8.0 if compact_fonts else 8.6, fontweight='bold', color='#333333')
        ax.set_title(level_title, fontsize=fs_ttl, fontweight='bold', pad=22 if compact_fonts else 26)


def _draw_hierarchy_figure(T: pd.DataFrame, split_lbl: str, out_stem, model_entries: list, levels: list, foot: list[str], title_desc: str = 'epistasis predicted level by level', overlay_sub0: bool = False, panel_letter: str | None = None) -> None:
    nd, nn = int(T['n_domains'].iloc[0]), int(T['n_doubles'].iloc[0])
    fig_w = 17.6 if len(levels) == 5 else 14.5
    fig, axes = plt.subplots(1, len(levels), figsize=(fig_w, 6.0), sharey=True)
    _draw_hierarchy_axes(axes, T, model_entries, levels, overlay_sub0=overlay_sub0)
    if panel_letter is None:
        t_word = 'Test' if split_lbl == 'test' else 'Validation'
        fig.suptitle(
            f'{t_word}: {title_desc}, {nn:,} doubles with both singles measured in {nd} domains (B = {int(T["B"].iloc[0]):,})',
            x=0.012, ha='left', fontsize=13.4, y=0.955 if len(levels) == 4 else 0.92,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.90))
        fig.text(0.012, -0.04, '\n'.join(foot), fontsize=9.0, va='top', ha='left', color='#444444')
    else:
        fig.tight_layout(rect=(0.012, 0, 1, 0.985))
        fig.text(0.004, 0.995, panel_letter, fontsize=15, fontweight='bold', va='top', ha='left')
    style.save(fig, out_stem)


def _slice_paired_tests(T_df: pd.DataFrame, reps: np.ndarray, models_list: list[str], metric_names: list[str], ref_model: str) -> pd.DataFrame:
    """Compute paired domain-bootstrap tests using exact point[i0] - point[i] delta (fixing legacy 11n bug!)."""
    i0 = models_list.index(ref_model)
    pt_lookup = {(r.model, r.metric): float(r.value) for r in T_df.itertuples()}
    rows = []
    for m in ('cell_rank', 'cell_mag_pooled', 'line_rho', 'pair_rho', 'beyond_rho', 'naive_rho'):
        j = metric_names.index(m)
        grp = []
        for i, name in enumerate(models_list):
            if i == i0 or name == '−dG_add':
                continue
            d = reps[:, i0, j] - reps[:, i, j]
            d = d[np.isfinite(d)]
            exact_delta = pt_lookup[(ref_model, m)] - pt_lookup[(name, m)]
            grp.append(dict(
                metric=m, other=name, delta=exact_delta,
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_one=boot_pvalue(d, null=0.0, alternative='greater'),
                p_two=boot_pvalue(d, null=0.0, alternative='two-sided'),
                B=len(d),
            ))
        p_adj = holm([g['p_one'] for g in grp])
        for g, pa in zip(grp, p_adj):
            g['p_one_holm'] = float(pa)
        rows.extend(grp)
    return pd.DataFrame(rows)


_FAST_EPI_STATE: dict = {}


def _fast_rho(a, b, min_n: int = 3) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < min_n:
        return E.NAN
    ao, bo = a[ok], b[ok]
    if np.ptp(bo) < 1e-12:
        return E.NAN
    if np.std(ao) < E.FLAT:
        return 0.0
    ra = rankdata(ao)
    rb = rankdata(bo)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / den) if den > 0.0 else E.NAN


def _smooth_eval_at(x_fit: np.ndarray, v_fit: np.ndarray, x_eval: np.ndarray, nbins: int = E.SMOOTH_BINS) -> np.ndarray:
    out = np.full(len(x_eval), np.nan)
    ok = np.isfinite(x_fit) & np.isfinite(v_fit)
    n_ok = int(ok.sum())
    if n_ok < 2 * nbins:
        nbins = max(1, n_ok // 2)
    if n_ok < 2:
        return out
    xo, vo = x_fit[ok], v_fit[ok]
    qs = np.quantile(xo, np.linspace(0, 1, nbins + 1))
    bidx = np.clip(np.searchsorted(qs, xo, side='right') - 1, 0, nbins - 1)
    cnts = np.bincount(bidx, minlength=nbins)
    ne = cnts > 0
    cx = np.bincount(bidx, weights=xo, minlength=nbins)[ne] / cnts[ne]
    cv = np.bincount(bidx, weights=vo, minlength=nbins)[ne] / cnts[ne]
    o = np.argsort(cx)
    cx, cv = cx[o], cv[o]
    fin = np.isfinite(x_eval)
    xe = x_eval[fin]
    ye = np.interp(xe, cx, cv)
    if len(cx) >= 2:
        lo, hi = xe < cx[0], xe > cx[-1]
        ye[lo] = cv[0] + (xe[lo] - cx[0]) * (cv[1] - cv[0]) / max(cx[1] - cx[0], 1e-12)
        ye[hi] = cv[-1] + (xe[hi] - cx[-1]) * (cv[-1] - cv[-2]) / max(cx[-1] - cx[-2], 1e-12)
    out[fin] = ye
    return out


def _dc_pruned(X: np.ndarray, mask: np.ndarray, n_iter: int = 8) -> np.ndarray:
    Y = np.where(mask, X, np.nan)
    for _ in range(n_iter):
        Y = Y - np.nanmean(Y, axis=1, keepdims=True)
        Y = Y - np.nanmean(Y, axis=0, keepdims=True)
    return Y[mask]


def _precompute_model_regime(t_reg: E.Table, head: str, libs: np.ndarray, lib_to_i: dict[int, int], with_gt0: bool) -> dict:
    add_singles = (t_reg.PA['add'] + t_reg.PB['add']) if ('add' in t_reg.PA and 'add' in t_reg.PB) else (t_reg.PA.get(head, 0.0) + t_reg.PB.get(head, 0.0))
    if head in E.WT_REFERENCED:
        ref = E.WT_REFERENCED[head]
        d_obs = (t_reg.P[ref] if ref in t_reg.P else t_reg.P[head]) - add_singles
    elif head == 'add':
        d_obs = t_reg.P['add'] - add_singles
    else:
        d_obs = t_reg.P[head] - add_singles

    d_lat, xh = E.predicted_dddg(t_reg, head)
    fin_row = np.isfinite(t_reg.dddG) & np.isfinite(t_reg.x) & np.isfinite(d_lat) & np.isfinite(xh)

    mats = t_reg.matrices()
    cr_libs, cr_vals = [], []
    mat_specs = []
    for M_mat in mats:
        li = lib_to_i[int(t_reg.lib[M_mat[M_mat >= 0][0]])]
        # Rank level (cell_rank)
        Y_r, L_r = E._lay(M_mat, t_reg.y), E._lay(M_mat, t_reg.L[head])
        blk_r = E._complete_block(np.where(np.isfinite(L_r), Y_r, np.nan))
        if blk_r is not None:
            Yb, Lb = Y_r[np.ix_(*blk_r)], L_r[np.ix_(*blk_r)]
            rv = E._mean([_fast_rho(E._dc(E._colrank(B_)).ravel(), E._dc(E._colrank(A_)).ravel()) for A_, B_ in ((Yb, Lb), (Yb.T, Lb.T))])
            if np.isfinite(rv):
                cr_libs.append(li)
                cr_vals.append(rv)

        # Residual levels
        ok = M_mat >= 0
        ok_idx = M_mat[ok]
        ok_sub = ok.copy()
        ok_sub[ok] = fin_row[ok_idx]
        n_ok = int(ok_sub.sum())
        if n_ok < E.MIN_PAIR_CELLS:
            continue
        M_safe = np.where(ok_sub, M_mat, 0)
        c_cnt = ok_sub.sum(axis=0)
        r_cnt = ok_sub.sum(axis=1)
        c_keep = c_cnt >= E.MIN_LINE
        r_keep = r_cnt >= E.MIN_LINE
        blk_c = E._complete_block(np.where(ok_sub, 0.0, np.nan))
        blk_M = M_mat[np.ix_(*blk_c)] if blk_c is not None else None

        gt0_spec = None
        if with_gt0:
            X_mat = E._lay(M_mat, t_reg.x)
            mask_u = ok_sub & (X_mat > 0.0)
            while True:
                ru, cu = mask_u.sum(1), mask_u.sum(0)
                bad_r = (ru > 0) & (ru < 3)
                bad_c = (cu > 0) & (cu < 3)
                if not bad_r.any() and not bad_c.any():
                    break
                mask_u[bad_r, :] = False
                mask_u[:, bad_c] = False
            if int(mask_u.sum()) >= 9:
                r_nz = mask_u.sum(1) > 0
                c_nz = mask_u.sum(0) > 0
                sub_mask = mask_u[np.ix_(r_nz, c_nz)]
                sub_M = np.where(sub_mask, M_mat[np.ix_(r_nz, c_nz)], 0)
                gt0_spec = (sub_M, sub_mask)

        mat_specs.append((li, M_safe, ok_sub, n_ok, c_cnt[c_keep], c_keep, r_cnt[r_keep], r_keep, blk_M, gt0_spec))

    return dict(
        d_obs=d_obs,
        d_lat=d_lat,
        xh=xh,
        cr_libs=np.asarray(cr_libs, dtype=int),
        cr_vals=np.asarray(cr_vals, dtype=float),
        mats=mat_specs,
    )


def _eval_model_regime_draw(
    spec: dict,
    counts: np.ndarray,
    idx_reg: np.ndarray,
    x_reg: np.ndarray,
    dddG_reg: np.ndarray,
    rm_reg: np.ndarray,
    with_gt0: bool,
) -> dict[str, float]:
    if len(idx_reg) == 0:
        keys = ('naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank', 'cell_mag', 'cell_mag_pooled') + (('cell_mag_gt0',) if with_gt0 else ())
        return {k: E.NAN for k in keys}

    d_obs, d_lat, xh = spec['d_obs'], spec['d_lat'], spec['xh']
    n_rho = _fast_rho(d_obs[idx_reg], dddG_reg[idx_reg])

    d_cloned = d_lat[idx_reg]
    fin_d = np.isfinite(d_cloned)
    if (not fin_d.any()) or np.nanstd(d_cloned) < E.FLAT:
        rp_reg = np.where(np.isfinite(d_lat), 0.0, np.nan)
    else:
        rp_reg = d_lat - _smooth_eval_at(xh[idx_reg], d_cloned, xh)

    b_rho = _fast_rho(rp_reg[idx_reg], rm_reg[idx_reg])

    # cell_rank weighted over cloned domains
    cr_libs, cr_vals = spec['cr_libs'], spec['cr_vals']
    if len(cr_vals):
        w_cr = counts[cr_libs]
        tot_cr = int(w_cr.sum())
        c_rank = float(np.dot(w_cr, cr_vals) / tot_cr) if tot_cr > 0 else E.NAN
    else:
        c_rank = E.NAN

    pm, pp, lm, lp, blk_m, blk_p = [], [], [], [], [], []
    cm_num, cm_den = 0.0, 0
    gt0_num, gt0_den = 0.0, 0
    both_ok = np.isfinite(rm_reg).any() and np.isfinite(rp_reg).any()

    if both_ok:
        for li, M_safe, ok_sub, n_ok, c_cnt_k, c_keep, r_cnt_k, r_keep, blk_M, gt0_spec in spec['mats']:
            c = int(counts[li])
            if c <= 0:
                continue
            Rm_z = np.where(ok_sub, rm_reg[M_safe], 0.0)
            Rp_z = np.where(ok_sub, rp_reg[M_safe], 0.0)
            mm = float(Rm_z.sum() / n_ok)
            mp = float(Rp_z.sum() / n_ok)
            pm.extend([mm] * c)
            pp.extend([mp] * c)

            if c_keep.any():
                lm0 = Rm_z.sum(axis=0)[c_keep] / c_cnt_k - mm
                lp0 = Rp_z.sum(axis=0)[c_keep] / c_cnt_k - mp
            else:
                lm0 = lp0 = None
            if r_keep.any():
                lm1 = Rm_z.sum(axis=1)[r_keep] / r_cnt_k - mm
                lp1 = Rp_z.sum(axis=1)[r_keep] / r_cnt_k - mp
            else:
                lm1 = lp1 = None
            if lm0 is not None and lm1 is not None:
                lm_one = np.concatenate([lm0, lm1])
                lp_one = np.concatenate([lp0, lp1])
            elif lm0 is not None:
                lm_one, lp_one = lm0, lp0
            elif lm1 is not None:
                lm_one, lp_one = lm1, lp1
            else:
                lm_one = lp_one = None
            if lm_one is not None and len(lm_one):
                lm.append(np.tile(lm_one, c))
                lp.append(np.tile(lp_one, c))

            if blk_M is not None:
                dc_rm = E._dc(rm_reg[blk_M]).ravel()
                dc_rp = E._dc(rp_reg[blk_M]).ravel()
                rv_c = _fast_rho(dc_rp, dc_rm)
                if np.isfinite(rv_c):
                    cm_num += c * rv_c
                    cm_den += c
                blk_m.append(np.tile(dc_rm, c))
                blk_p.append(np.tile(dc_rp, c))

            if with_gt0 and gt0_spec is not None:
                sub_M, sub_mask = gt0_spec
                cm_u = _dc_pruned(rm_reg[sub_M], sub_mask)
                cp_u = _dc_pruned(rp_reg[sub_M], sub_mask)
                rv_u = _fast_rho(cp_u, cm_u)
                if np.isfinite(rv_u):
                    gt0_num += c * rv_u
                    gt0_den += c

    p_rho = _fast_rho(pp, pm) if len(pm) >= E.MIN_PAIRS else E.NAN
    if lm:
        lm_all = np.concatenate(lm)
        lp_all = np.concatenate(lp)
        l_rho = _fast_rho(lp_all, lm_all) if len(lm_all) >= 2 * E.MIN_PAIRS else E.NAN
    else:
        l_rho = E.NAN
    c_mag = (cm_num / cm_den) if cm_den > 0 else E.NAN
    if blk_m:
        bm_all = np.concatenate(blk_m)
        bp_all = np.concatenate(blk_p)
        c_mag_pool = _fast_rho(bp_all, bm_all) if len(bm_all) >= 9 else E.NAN
    else:
        c_mag_pool = E.NAN

    res = dict(
        naive_rho=n_rho,
        beyond_rho=b_rho,
        pair_rho=p_rho,
        line_rho=l_rho,
        cell_rank=c_rank,
        cell_mag=c_mag,
        cell_mag_pooled=c_mag_pool,
    )
    if with_gt0:
        res['cell_mag_gt0'] = (gt0_num / gt0_den) if gt0_den > 0 else E.NAN
    return res


def _fast_epi_worker_draw(task) -> np.ndarray:
    counts, seed_weights_by_ns = task
    st = _FAST_EPI_STATE
    models = st['models']
    metric_names = st['metric_names']
    regimes = st['regimes']
    prep = st['prep']
    seeds_by_key = st['seeds_by_key']

    # Shared measured residuals rm_reg per regime for this domain draw
    draw_reg = {}
    for reg_name, (by_lib, x_reg, dddG_reg) in regimes.items():
        idx_l = [by_lib[li] for li, c in enumerate(counts) if c > 0 and len(by_lib[li]) for _ in range(int(c))]
        idx_reg = np.concatenate(idx_l) if idx_l else np.zeros(0, dtype=int)
        rm_reg = (dddG_reg - _smooth_eval_at(x_reg[idx_reg], dddG_reg[idx_reg], x_reg)) if len(idx_reg) >= 2 else np.full(len(x_reg), np.nan)
        draw_reg[reg_name] = (idx_reg, x_reg, dddG_reg, rm_reg)

    eval_cache: dict[tuple[str, int, str], list[float]] = {}
    out_row = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    for mi, (_disp, key, head) in enumerate(models):
        s_list = seeds_by_key[key]
        ns = len(s_list)
        sw = seed_weights_by_ns.get(ns, np.ones(ns, dtype=float))
        vals, weights = [], []
        for s, w_s in zip(s_list, sw):
            if w_s <= 0:
                continue
            ck = (key, s, head)
            if ck not in eval_cache:
                r_all = _eval_model_regime_draw(prep[(ck, 'all')], counts, *draw_reg['all'], with_gt0=True)
                r_gt1 = _eval_model_regime_draw(prep[(ck, 'gt1')], counts, *draw_reg['gt1'], with_gt0=False)
                r_sub0 = _eval_model_regime_draw(prep[(ck, 'sub0')], counts, *draw_reg['sub0'], with_gt0=False)
                md = dict(r_all)
                for m_k, v_k in r_gt1.items():
                    md[f'{m_k}_gt1'] = v_k
                for m_k, v_k in r_sub0.items():
                    md[f'{m_k}_sub0'] = v_k
                eval_cache[ck] = [md[m] for m in metric_names]
            vals.append(eval_cache[ck])
            weights.append(w_s)
        if not vals:
            continue
        V = np.asarray(vals, dtype=float)
        W = np.asarray(weights, dtype=float)[:, None]
        fin = np.isfinite(V)
        with np.errstate(divide='ignore', invalid='ignore'):
            den = np.sum(np.where(fin, W, 0.0), axis=0)
            num = np.sum(np.where(fin, W * V, 0.0), axis=0)
            out_row[mi] = np.where(den > 0, num / den, np.nan)
    return out_row


def _prep_epi_spine(spine: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Parse ``Frame.spine`` mutation keys once per split for fast multi-model ``E.Table`` construction."""
    idx = np.flatnonzero(spine['use'].to_numpy(dtype=bool) if 'use' in spine else np.ones(len(spine), dtype=bool))
    code_col = 'code' if 'code' in spine else 'code_wt'
    mut_col = 'mut' if 'mut' in spine else 'mut_type'
    ddg_col = 'y' if 'y' in spine else 'ddG_ML'
    code = spine[code_col].astype(str).to_numpy()[idx]
    mut = spine[mut_col].astype(str).to_numpy()[idx]
    keys = [EPI_LIB.parse_mut_key(m, c) for m, c in zip(mut, code)]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    base = dict(
        mut_key=keys,
        subset_type=st,
        ddG=spine[ddg_col].to_numpy(dtype=float)[idx],
        dddG=spine['dddG_ML'].to_numpy(dtype=float)[idx],
        dG_wt=spine['dG_wt'].to_numpy(dtype=float)[idx],
    )
    return idx, base


def _make_epi_table_fast(idx_base: tuple[np.ndarray, dict], wt, mt, heads: tuple[str, ...] = ('epi', 'add'), wt_rev=None) -> E.Table:
    idx, base = idx_base
    wt_arr = np.asarray(wt, dtype=float)[idx]
    mt_arr = np.asarray(mt, dtype=float)[idx]
    items = dict(base, wt=wt_arr, mt=mt_arr, comb=mt_arr)
    if wt_rev is not None:
        items['wt_rev'] = np.asarray(wt_rev, dtype=float)[idx]
    return E.Table(items, None, heads=heads)


def _prep_ksh_worker(ksh: tuple[str, int, str]) -> tuple[tuple[str, int, str], list[float], dict[str, dict]]:
    key, s, head = ksh
    st = _FAST_EPI_STATE
    t_cur = st['tables'][(key, s)]
    metric_names = st['metric_names']
    libs = st['libs']
    lib_to_i = st['lib_to_i']
    reg_masks = st['reg_masks']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        pt_d = EPI_LIB.compute_epi_metrics(t_cur, head=head, extended=True)
        pt_vec = [pt_d[m] for m in metric_names]
        reg_prep = {}
        for reg_name, rmask in reg_masks.items():
            t_r = t_cur if reg_name == 'all' else EPI_LIB.subset_epi_table(t_cur, rmask)
            reg_prep[reg_name] = _precompute_model_regime(t_r, head, libs, lib_to_i, with_gt0=(reg_name == 'all'))
    return ksh, pt_vec, reg_prep


def _fast_boot_epistasis_models(
    tables_by_model_seed: dict[tuple[str, int], E.Table],
    models: list[tuple[str, str, str]],
    B: int = 1000,
    seed: int = 11,
    jobs: int = 8,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Exact (<1e-13) precomputed-block epistasis hierarchy bootstrap matching EPI_LIB.boot_epistasis_models(extended=True)."""
    metric_names = list(EPI_LIB.EPI_EXTENDED_METRICS)
    first_key = models[0][1]
    ref_table = next(t for (k, _s), t in sorted(tables_by_model_seed.items()) if k == first_key)
    libs = np.unique(ref_table.lib)
    lib_to_i = {int(lib_id): i for i, lib_id in enumerate(libs)}
    n_libs = len(libs)
    seeds_by_key = {
        key: sorted([s for (k_, s) in tables_by_model_seed if k_ == key])
        for _disp, key, _head in models
    }
    distinct_ns = sorted({len(v) for v in seeds_by_key.values()})

    reg_masks = {
        'all': np.ones(ref_table.n, dtype=bool),
        'gt1': ref_table.x > 1.0,
        'sub0': ref_table.x > 0.0,
    }
    regimes = {}
    for reg_name, rmask in reg_masks.items():
        t0_r = EPI_LIB.subset_epi_table(ref_table, rmask)
        by_lib = [np.flatnonzero(t0_r.lib == lib_id) for lib_id in libs]
        regimes[reg_name] = (by_lib, t0_r.x, t0_r.dddG)

    unique_ksh = sorted({(key, s, head) for _disp, key, head in models for s in seeds_by_key[key]})
    _FAST_EPI_STATE.clear()
    _FAST_EPI_STATE.update(
        tables=tables_by_model_seed, metric_names=metric_names, libs=libs, lib_to_i=lib_to_i, reg_masks=reg_masks,
    )
    try:
        n_pw = min(max(1, jobs), len(unique_ksh))
        if n_pw > 1:
            with mp.get_context('fork').Pool(n_pw) as pool:
                prep_list = pool.map(_prep_ksh_worker, unique_ksh, chunksize=1)
        else:
            prep_list = [_prep_ksh_worker(ksh) for ksh in unique_ksh]
    finally:
        _FAST_EPI_STATE.clear()

    pt_by_ksh: dict[tuple[str, int, str], list[float]] = {}
    prep: dict[tuple[tuple[str, int, str], str], dict] = {}
    for ksh, pt_vec, reg_prep in prep_list:
        pt_by_ksh[ksh] = pt_vec
        for reg_name, sp_d in reg_prep.items():
            prep[(ksh, reg_name)] = sp_d

    point_mat = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    sd_mat = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    n_seeds_list = []
    for mi, (_disp, key, head) in enumerate(models):
        s_list = seeds_by_key[key]
        n_seeds_list.append(len(s_list))
        seed_vals = np.asarray([pt_by_ksh[(key, s, head)] for s in s_list], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            point_mat[mi] = np.nanmean(seed_vals, axis=0)
        for j in range(len(metric_names)):
            sd_mat[mi, j] = seed_sd(seed_vals[:, j])

    rng = np.random.default_rng(seed)
    tasks = []
    ordered_ns = [3] + [ns for ns in distinct_ns if ns not in (0, 1, 3)]
    for _ in range(B):
        cnt = rng.multinomial(n_libs, np.full(n_libs, 1.0 / n_libs))
        sw_map = {1: np.ones(1, dtype=float)}
        for ns in ordered_ns:
            sw_map[ns] = rng.multinomial(ns, np.full(ns, 1.0 / ns)).astype(float)
        tasks.append((cnt, sw_map))

    _FAST_EPI_STATE.clear()
    _FAST_EPI_STATE.update(models=models, metric_names=metric_names, regimes=regimes, prep=prep, seeds_by_key=seeds_by_key)
    try:
        if jobs > 1 and B > 1:
            with mp.get_context('fork').Pool(jobs) as pool:
                draws = pool.map(_fast_epi_worker_draw, tasks, chunksize=max(1, B // (jobs * 4)))
        else:
            draws = [_fast_epi_worker_draw(tk) for tk in tasks]
    finally:
        _FAST_EPI_STATE.clear()

    reps = np.asarray(draws, dtype=float)
    rows = []
    for mi, (disp, key, head) in enumerate(models):
        for ji, mname in enumerate(metric_names):
            lo, hi = ci(reps[:, mi, ji]) if B > 0 else (np.nan, np.nan)
            se = float(np.nanstd(reps[:, mi, ji], ddof=1)) if np.isfinite(reps[:, mi, ji]).sum() > 1 else np.nan
            rows.append(dict(
                model=disp,
                key=key,
                head=head,
                metric=mname,
                value=float(point_mat[mi, ji]),
                lo=lo,
                hi=hi,
                se=se,
                seed_sd=float(sd_mat[mi, ji]),
                n_seeds=n_seeds_list[mi],
                n_doubles=int(ref_table.n),
                n_domains=int(n_libs),
                B=int(B),
            ))
    return pd.DataFrame(rows), reps, metric_names


def run_hierarchy_and_composite(B: int, split: str = 'capped', procs: int = 6, T3_df: pd.DataFrame | None = None, p_spearman_ser: pd.Series | None = None) -> None:
    out_epi = out_dir('epistasis', split=split)
    mega = (split == 'mega')
    tsu_pair = tsu_datasets(split)
    ds_map = {'val': tsu_pair[0], 'test': tsu_pair[1]}
    dg_json = json.loads((REFERENCE / 'dG_wt.json').read_text())

    std_entries = [
        _hier_entry('ProSST', 'ProSST'),
        _hier_entry('ProteinMPNN', 'ProteinMPNN'),
        _hier_entry('ESM3-small-open', 'ESM3-small-open'),
        _hier_entry(STD_EPI_NAME, 'ESM-MSR'),
        _hier_entry(STD_CTX_NAME, 'ESM-MSR', 'ESM-MSR ctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
        _hier_entry('ME stage 2', 'MutateEverything'),
        _hier_entry('ThermoMPNN-D', 'ThermoMPNN'),
        _hier_entry('SPURS multi', 'SPURS'),
        _hier_entry('GeoDDG-3D', 'GeoDDG-3D'),
        _hier_entry('Rosetta', 'Rosetta'),
        _hier_entry('ESM3ΔG', 'ESM3ΔG'),
    ]

    for sp_key in ('val', 'test'):
        fr = load_frame(ds_map[sp_key])
        sp, p = fr.spine, fr.preds
        ids = sp['id'].to_numpy(dtype=object)
        epi_sp = _prep_epi_spine(sp)
        tables_map: dict[tuple[str, int], E.Table] = {}
        models_spec: list[tuple[str, str, str]] = []

        # ESM-MSR p0 (+ ctx control on capped test if available, + masking ablations on capped test)
        have_ctx = False
        if not mega and sp_key == 'test':
            f_fin = load_final_frame(ds_map['test'], families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
            revs = [_ctx_rev_for_seed(sp, 'test', s) for s in (1, 2, 3)]
            have_ctx = all(rv is not None for rv in revs)
            for s in (1, 2, 3):
                heads = ('epi', 'ctx', 'add') if have_ctx else ('epi', 'add')
                tables_map[('p0', s)] = _make_epi_table_fast(epi_sp, f_fin.preds['FINAL_p0:wt'][s - 1], f_fin.preds['FINAL_p0:comb'][s - 1], heads=heads, wt_rev=revs[s - 1] if have_ctx else None)
                tables_map[('p1', s)] = _make_epi_table_fast(epi_sp, f_fin.preds['FINAL_p1:wt'][s - 1], f_fin.preds['FINAL_p1:comb'][s - 1], heads=('epi', 'add'))
                tables_map[('p1m', s)] = _make_epi_table_fast(epi_sp, f_fin.preds['FINAL_p1_marg:wt'][s - 1], f_fin.preds['FINAL_p1_marg:comb'][s - 1], heads=('epi', 'add'))
            models_spec.append((STD_EPI_NAME, 'p0', 'epi'))
            if have_ctx:
                models_spec.append((STD_CTX_NAME, 'p0', 'ctx'))
            models_spec.extend([(P1_EPI_NAME, 'p1', 'epi'), (P1M_EPI_NAME, 'p1m', 'epi')])
        else:
            for s in (1, 2, 3):
                tables_map[('p0', s)] = _make_epi_table_fast(epi_sp, p['msrp0:wt'][s - 1], p['msrp0:comb'][s - 1], heads=('epi', 'add'))
            models_spec.append((STD_EPI_NAME, 'p0', 'epi'))

        # Plain baselines
        plain = [
            ('ME stage 2', 'me2', 'me1'), ('ThermoMPNN-D', 'tmpnnd', 'tmpnn'), ('SPURS multi', 'spursm', 'spurs'),
            ('ProSST', 'prosst:comb', 'prosst:comb+'),
            ('ESM3-small-open', 'msrrel@0:comb', 'msrrel@0:wt'), ('ProteinMPNN', 'pmpnn', 'pmpnn+'),
            ('GeoDDG-3D', 'geoddg', 'geoddg+'), ('Rosetta', 'rosetta', 'rosetta+'), ('ESM3ΔG', 'esm3dg', 'esm3dg+'),
        ]
        if mega:
            plain = plain[:3] + [
                ('SPURS multi (trained adapter)', 'spursum', 'spursu'),
                ('ThermoMPNN-D (published)', 'tmpnndpub', 'tmpnnpub'),
            ] + plain[3:]
        for name, direct, add in plain:
            if direct not in p or add not in p:
                continue
            ns = p[direct].shape[0]
            for s in range(ns):
                tables_map[(name, s + 1)] = _make_epi_table_fast(epi_sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], heads=('epi', 'add'))
            models_spec.append((name, name, 'epi'))

        # SPURS unfrozen on capped test (for 11n slice!)
        unf_dir = BASE / 'retrained' / 'spurs_unfrozen' / 'predictions'
        have_unf = (not mega and sp_key == 'test' and all((unf_dir / f'seed{sd}' / 'tsuboyama_test.csv').exists() for sd in (1, 2, 3)))
        if have_unf:
            for sd in (1, 2, 3):
                d_u = pd.read_csv(unf_dir / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_single_additive', 'SPURS_multi'], low_memory=False)
                s_u = _align_id(ids, d_u['uid'].astype(str).to_numpy(), d_u['SPURS_single_additive'].to_numpy())
                m_u = _align_id(ids, d_u['uid'].astype(str).to_numpy(), d_u['SPURS_multi'].to_numpy())
                tables_map[('spurs_unf', sd)] = _make_epi_table_fast(epi_sp, s_u, m_u, heads=('epi', 'add'))
            models_spec.append(('SPURS multi (unfrozen)', 'spurs_unf', 'epi'))

        # -dG_add saturation floor control
        c_exact = np.array([float(dg_json.get(k, np.nan)) for k in sp['code'].astype(str)])
        if mega:
            lib_c = sp.groupby(sp['code'].astype(str))['dG_wt'].median()
            c_exact = np.where(np.isfinite(c_exact), c_exact, sp['code'].astype(str).map(lib_c).to_numpy(float))
        wt_true = np.where((sp['n_mut'] == 1).to_numpy(), sp['y'].to_numpy(float), sp['ddG_additive_ML'].to_numpy(float))
        tables_map[('neg_dg_add', 1)] = _make_epi_table_fast(epi_sp, wt_true, -c_exact, heads=('epi', 'add'))
        models_spec.append(('−dG_add', 'neg_dg_add', 'epi'))

        # ONE shared domain-cloning + 3-seed bootstrap for this split!
        T_all, reps_all, metric_names = _fast_boot_epistasis_models(tables_map, models_spec, B=B, seed=11, jobs=procs)
        T_all = T_all.assign(split=sp_key)

        # Slice 1: 11d / 11b standard table & figures
        std_keep = {m[0] for m in std_entries} | {'−dG_add', 'SPURS multi (trained adapter)', 'ThermoMPNN-D (published)'}
        T_11d = T_all[T_all['model'].isin(std_keep)].copy()
        atomic_to_csv(T_11d, out_epi / f'epistasis_levels_bootstrap_{sp_key}.csv', index=False, float_format='%.5f')
        _draw_hierarchy_figure(T_11d, sp_key, out_epi / f'epistasis_levels_bars_{sp_key}', std_entries, LEVELS_MAIN, FOOT_HIER)
        _draw_hierarchy_figure(T_11d, sp_key, out_epi / f'epistasis_levels_bars_{sp_key}_cell_mag', std_entries, LEVELS_CELL_MAG, FOOT_HIER, overlay_sub0=True)
        _draw_hierarchy_figure(T_11d, sp_key, out_epi / f'epistasis_levels_bars_{sp_key}_cell_compare', std_entries, LEVELS_CELL_COMPARE, FOOT_HIER, title_desc='cell-level epistasis formulations compared')

        if mega or sp_key != 'test':
            continue

        # Slice 2 & 3 (capped test only): 11n p0 + masking supplementary figures, paired tests, main-text panel e, and NEW main composite!
        out_supp = out_dir('supplementary_figs')
        out_main = out_dir('main_text_figs')
        all_names = [m[0] for m in models_spec]

        # Build p0 slice (renaming STD_EPI_NAME -> P0_EPI_NAME and SPURS multi (unfrozen) -> SPURS multi to match 11n schema)
        spurs_src = 'SPURS multi (unfrozen)' if have_unf else 'SPURS multi'
        p0_src_names = (
            [STD_EPI_NAME]
            + ([STD_CTX_NAME] if have_ctx else [])
            + ['ME stage 2', 'ThermoMPNN-D', spurs_src, 'ProSST', 'ESM3-small-open', 'ProteinMPNN', 'GeoDDG-3D', 'Rosetta', 'ESM3ΔG', '−dG_add']
        )
        p0_src_names = [n for n in p0_src_names if n in all_names]
        p0_rename = {STD_EPI_NAME: P0_EPI_NAME, STD_CTX_NAME: P0_CTX_NAME, 'SPURS multi (unfrozen)': 'SPURS multi'}
        p0_idx = [all_names.index(n) for n in p0_src_names]
        p0_disp = [p0_rename.get(n, n) for n in p0_src_names]
        T_p0 = T_all[T_all['model'].isin(p0_src_names)].copy()
        T_p0['model'] = T_p0['model'].map(lambda x: p0_rename.get(x, x))
        reps_p0 = reps_all[:, p0_idx, :]
        atomic_to_csv(T_p0, out_supp / 'fig5_epistasis_levels_p0_bootstrap_test.csv', index=False, float_format='%.5f')
        np.savez(out_supp / 'fig5_epistasis_levels_p0_reps_test.npz', reps=reps_p0, models=np.array(p0_disp), metrics=np.array(metric_names), B=B)
        atomic_to_csv(_slice_paired_tests(T_p0, reps_p0, p0_disp, metric_names, P0_EPI_NAME), out_supp / 'fig5_epistasis_levels_p0_paired_test.csv', index=False, float_format='%.5f')

        p0_entries = [
            _hier_entry('ProSST', 'ProSST'),
            _hier_entry('ProteinMPNN', 'ProteinMPNN'),
            _hier_entry('ESM3-small-open', 'ESM3-small-open'),
            _hier_entry(P0_EPI_NAME, 'ESM-MSR'),
            _hier_entry(P0_CTX_NAME, 'ESM-MSR', 'ESM-MSR ctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
            _hier_entry('ME stage 2', 'MutateEverything'),
            _hier_entry('ThermoMPNN-D', 'ThermoMPNN'),
            _hier_entry('SPURS multi', 'SPURS'),
            _hier_entry('GeoDDG-3D', 'GeoDDG-3D'),
            _hier_entry('Rosetta', 'Rosetta'),
            _hier_entry('ESM3ΔG', 'ESM3ΔG'),
        ]
        _draw_hierarchy_figure(T_p0, 'test', out_supp / 'fig5a_epistasis_levels_bars_test_p0', p0_entries, LEVELS_MAIN, FOOT_HIER)

        # Build masking slice
        mk_src = [STD_EPI_NAME] + ([STD_CTX_NAME] if have_ctx else []) + [P1_EPI_NAME, P1M_EPI_NAME, '−dG_add']
        mk_idx = [all_names.index(n) for n in mk_src]
        mk_disp = [p0_rename.get(n, n) for n in mk_src]
        T_mk = T_all[T_all['model'].isin(mk_src)].copy()
        T_mk['model'] = T_mk['model'].map(lambda x: p0_rename.get(x, x))
        reps_mk = reps_all[:, mk_idx, :]
        atomic_to_csv(T_mk, out_supp / 'fig5_epistasis_levels_masking_bootstrap_test.csv', index=False, float_format='%.5f')
        np.savez(out_supp / 'fig5_epistasis_levels_masking_reps_test.npz', reps=reps_mk, models=np.array(mk_disp), metrics=np.array(metric_names), B=B)
        atomic_to_csv(_slice_paired_tests(T_mk, reps_mk, mk_disp, metric_names, P0_EPI_NAME), out_supp / 'fig5_epistasis_levels_masking_paired_test.csv', index=False, float_format='%.5f')
        mk_entries = [
            _hier_entry(P0_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(unmasked)'),
            _hier_entry(P0_CTX_NAME, 'ESM-MSR', 'ESM-MSR\nctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
            _hier_entry(P1_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(independent masking)', M.VARIANT_COLOR['ESM-MSR independent masking']),
            _hier_entry(P1M_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(masked marginal)', M.VARIANT_COLOR['ESM-MSR masked marginal']),
        ]
        _draw_hierarchy_figure(T_mk, 'test', out_supp / 'fig5a_epistasis_levels_bars_test_masking', mk_entries, LEVELS_MAIN, FOOT_HIER, title_desc='ESM-MSR masking ablations, epistasis level by level')
        build_supp_figure5(T_mk=T_mk)


def _build_epistasis_main_composite(out_main, fr_test, T3_df, p_spearman_ser, T_p0, p0_entries, tables_map, have_unf: bool, B: int) -> None:
    """Create unified 3-row publication composite epistasis_main_composite.{png,pdf}."""
    best, _ = best_other(fr_test)
    cols3 = pair_columns(fr_test, best)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}

    fig = plt.figure(figsize=(17.2, 15.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.95, 1.25, 0.12, 0.92], hspace=0.58)

    # Row a: Double-mutant additive vs epistatic ranking + significance brackets
    ax_a = fig.add_subplot(gs[0])
    if T3_df is not None:
        style.bar_panel(
            ax_a, cols3, T3_df, ('additive', 'epistatic'),
            (count_text(T3_df, 'additive', 'additive: singles model summed'), count_text(T3_df, 'epistatic', 'epistatic: model scores the double')),
            "Spearman's ρ, double mutants", (0.0, 1.16), tick_fs=8.8,
        )
        if p_spearman_ser is not None:
            b_11l_c = B if B <= 200 else max(B, 10000)
            top_a = _draw_pair_brackets(ax_a, cols3, T3_df, p_spearman_ser, p_name_map=name_map, B_p=b_11l_c)
            if top_a > 1.16:
                ax_a.set_ylim(0.0, top_a)
    ax_a.text(-0.035, 1.04, 'a', transform=ax_a.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

    # Row b: 5-level epistasis hierarchy with dG_add > 0 overlay
    gs_b = gs[1].subgridspec(1, 5, wspace=0.14)
    axes_b = [fig.add_subplot(gs_b[0, k]) for k in range(5)]
    for k in range(1, 5):
        axes_b[k].sharey(axes_b[0])
        plt.setp(axes_b[k].get_yticklabels(), visible=False)
    _draw_hierarchy_axes(axes_b, T_p0, p0_entries, LEVELS_MAIN, overlay_sub0=True, compact_fonts=True)
    axes_b[0].text(-0.22, 1.14, 'b', transform=axes_b[0].transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

    # Row c: Beyond-global saturation-subtracted residual scatters (ESM-MSR vs 3 transfer baselines)
    gs_c = gs[3].subgridspec(1, 4, wspace=0.18)
    axes_c = [fig.add_subplot(gs_c[0, k]) for k in range(4)]
    spurs_key = 'spurs_unf' if have_unf else 'SPURS multi'
    scatter_specs = [
        ('p0', 'ESM-MSR Dual-view', P0_EPI_NAME),
        (spurs_key, 'SPURS-multi (trained adapter)', 'SPURS multi'),
        ('ME stage 2', 'MutateEverything Stage 2', 'ME stage 2'),
        ('ThermoMPNN-D', 'ThermoMPNN-D', 'ThermoMPNN-D'),
    ]
    rm_base = E.beyond_global(tables_map[('p0', 1)], 'epi')[0]
    for idx_c, (ax_c, (t_key, title_c, csv_nm)) in enumerate(zip(axes_c, scatter_specs)):
        rp_seeds = [E.beyond_global(tables_map[(t_key, s)], 'epi')[1] for s in (1, 2, 3) if (t_key, s) in tables_map]
        rp_mean = np.mean(rp_seeds, axis=0)
        ok = np.isfinite(rm_base) & np.isfinite(rp_mean)
        style.density_scatter(ax_c, rp_mean[ok], rm_base[ok], s=3, cmap='viridis', line=True, max_points=15000, label_stats=False)
        ax_c.axhline(0, color='#bbbbbb', lw=0.7)
        ax_c.axvline(0, color='#bbbbbb', lw=0.7)
        b_row = T_p0[(T_p0['model'] == csv_nm) & (T_p0['metric'] == 'beyond_rho')].iloc[0]
        n_row = T_p0[(T_p0['model'] == csv_nm) & (T_p0['metric'] == 'naive_rho')].iloc[0]
        ax_c.set_title(title_c, fontsize=9.8, fontweight='bold' if t_key == 'p0' else 'normal')
        ax_c.text(
            0.04, 0.96, f'beyond ρ = {b_row.value:.3f}\nnaive ρ = {n_row.value:.3f}',
            transform=ax_c.transAxes, va='top', fontsize=8.4,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9),
        )
        ax_c.set_xlabel("predicted residual r' (kcal/mol)", fontsize=9.0)
        if idx_c == 0:
            ax_c.set_ylabel('measured residual r (kcal/mol)', fontsize=9.2)
    axes_c[0].text(-0.18, 1.12, 'c', transform=axes_c[0].transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

    foot_c = (
        f'Homology-Restricted MegaScale test set (12,455 measured doubles across 13 domains with pairs; 95% intervals from resampling domains & triplicate seeds, B = {B:,}). '
        'a, Per-domain Spearman ρ on double mutants comparing additive single-mutant sums (hatched) against direct double-mutant models (solid; brackets: one-sided paired bootstrap p-values). '
        'b, Five-level epistasis hierarchy from raw dddG to double-centred cell ranks; black dashes show performance on unsaturated doubles (dG_add > 0 kcal/mol); red dotted line in panel 1 shows the −dG_add floor-saturation ceiling. '
        'c, Saturation-subtracted epistasis residuals (r = dddG − E[dddG | dG_add], seed-mean r\') for ESM-MSR vs retrained transfer baselines.'
    )
    kw_f = dict(fontsize=8.0, color='#444444')
    fig.text(0.012, 0.006, style.wrap_to_px(fig, foot_c, 0.975 * fig.bbox.width, **kw_f), va='bottom', ha='left', **kw_f)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.955, bottom=0.075)
    style.save(fig, out_main / 'epistasis_main_composite')


def build_supp_figure5(T_mk: pd.DataFrame | None = None) -> Path:
    """Build Supplementary Figure 5 (`00_supplementary_figs/supp_figure5.{png,pdf}`).

    ESM-MSR masking ablations across all five pooled epistasis levels (`LEVELS_CELL_MAG`,
    with `dG_add > 0` overlays and the `-dG_add` floor ceiling).
    """
    out_supp = out_dir('supplementary_figs')
    if T_mk is None:
        T_mk = pd.read_csv(out_supp / 'fig5_epistasis_levels_masking_bootstrap_test.csv')

    mk_entries = [
        _hier_entry(P0_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(unmasked)'),
        _hier_entry(P0_CTX_NAME, 'ESM-MSR', 'ESM-MSR\nctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
        _hier_entry(P1_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(one-at-a-time\nmasking)', M.VARIANT_COLOR['ESM-MSR independent masking']),
        _hier_entry(P1M_EPI_NAME, 'ESM-MSR', 'ESM-MSR\n(masked\nmarginal)', M.VARIANT_COLOR['ESM-MSR masked marginal']),
    ]

    nd = int(T_mk['n_domains'].iloc[0])
    nn = int(T_mk['n_doubles'].iloc[0])
    B_val = int(T_mk['B'].iloc[0])

    fig, axes = plt.subplots(1, len(LEVELS_CELL_MAG), figsize=(16.0, 6.2), sharey=True, facecolor='white')
    _draw_hierarchy_axes(axes, T_mk, mk_entries, LEVELS_CELL_MAG, overlay_sub0=True)
    fig.suptitle(
        f'ESM-MSR masking ablations across the five pooled epistasis levels ({nn:,} test doubles across {nd} domains; B = {B_val:,})',
        x=0.015, ha='left', y=0.985, fontsize=11.6, fontweight='bold',
    )

    note = (
        'Homology-Restricted MegaScale test doubles with both singles measured (3 seeds averaged per model; whiskers: 95% joint domain × seed bootstrap; '
        'black horizontal dashes: unsaturated subset ΔG_add > 0 kcal/mol; red dotted line: −ΔG_add floor-saturation ceiling).  '
        'All five levels are evaluated in pooled form: (1) naive / raw ρ on raw δΔΔG; (2) saturation-subtracted ρ on residuals r = δΔΔG − E[δΔΔG | ΔG_add]; '
        '(3) pair ρ across position-pair matrices (≥ 20 doubles); (4) line ρ across pair-centred row/column lines (≥ 3 cells); '
        '(5) cell ρ pooled across all double-centred residual cells. Masked marginal scoring collapses cell-level epistasis to zero because its additive factorization cannot represent pairwise interactions.'
    )
    kw = dict(fontsize=7.8, color='#333333', linespacing=1.25)
    fig.subplots_adjust(left=0.055, right=0.990, top=0.835, bottom=0.295, wspace=0.14)
    fig.text(0.015, 0.012, style.wrap_to_px(fig, note, 0.970 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'supp_figure5'
    style.save(fig, out_path)
    return out_path


def build_fig5a_epistasis_definitions_compare(B: int = 1000, procs: int = 4) -> Path:
    """Build 2-row a/b supplementary Figure 5a (`00_supplementary_figs/fig5a_epistasis_levels_bars_test_p0.{png,pdf}`):
      - Panel a (top): Identical to Main Figure 2e (deployed-singles epistasis: comb_AB − WT_A − WT_B for ESM-MSR & ESM3-small-open,
        Stage 2 − Stage 1 for MutateEverything, SPURS-multi − SPURS-single for SPURS).
      - Panel b (bottom): Self-consistent multi-mutant model epistasis everywhere:
        comb_AB − (comb_A + comb_B) for ProSST, ESM3-small-open, ESM-MSR, and ESM-MSR ctx ctrl;
        Stage 2_AB − (Stage 2_A + Stage 2_B) for MutateEverything; native SPURS_multi_epi (= SPURS_multi − SPURS_multi_additive)
        for SPURS-multi; native epistatic_pred for ThermoMPNN-D; direct − sum of singles for ProteinMPNN, GeoDDG-3D, Rosetta.
    """
    out_supp = out_dir('supplementary_figs')
    T_a = pd.read_csv(out_supp / 'fig5_epistasis_levels_p0_bootstrap_test.csv')

    # Build Panel b bootstrap table (B draws, same domain x seed scheme)
    fr = load_frame('tsuboyama_capped_test')
    sp, p = fr.spine, fr.preds
    ids = sp['id'].to_numpy(dtype=object)
    epi_sp = _prep_epi_spine(sp)
    dg_json = json.loads((REFERENCE / 'dG_wt.json').read_text())

    f_fin = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0'])
    revs = [_ctx_rev_for_seed(sp, 'test', s) for s in (1, 2, 3)]
    have_ctx = all(rv is not None for rv in revs)

    tables_b: dict[tuple[str, int], E.Table] = {}
    specs_b: list[tuple[str, str, str]] = []

    # ESM-MSR: comb_AB - (comb_A + comb_B)
    for s in (1, 2, 3):
        comb_s = f_fin.preds['FINAL_p0:comb'][s - 1]
        tables_b[('p0_sc', s)] = _make_epi_table_fast(epi_sp, comb_s, comb_s, heads=('epi', 'add'))
    specs_b.append((P0_EPI_NAME, 'p0_sc', 'epi'))

    # ESM-MSR ctx ctrl: ctx_AB - (ctx_A + ctx_B) where ctx = 0.5 * (wt + wt_rev)
    if have_ctx:
        for s in (1, 2, 3):
            ctx_s = 0.5 * (f_fin.preds['FINAL_p0:wt'][s - 1] + revs[s - 1])
            tables_b[('ctx_sc', s)] = _make_epi_table_fast(epi_sp, ctx_s, ctx_s, heads=('epi', 'add'))
        specs_b.append((P0_CTX_NAME, 'ctx_sc', 'epi'))

    # Self-consistent baselines: f_multi(AB) - (f_multi(A) + f_multi(B))
    sc_plain = [
        ('ProSST', 'prosst:comb', 'prosst:comb'),
        ('ProteinMPNN', 'pmpnn', 'pmpnn+'),
        ('ESM3-small-open', 'msrrel@0:comb', 'msrrel@0:comb'),
        ('ME stage 2', 'me2', 'me2'),
        ('ThermoMPNN-D', 'tmpnnd', 'tmpnn'),
        ('GeoDDG-3D', 'geoddg', 'geoddg+'),
        ('Rosetta', 'rosetta', 'rosetta+'),
    ]
    for name, direct, add in sc_plain:
        if direct not in p or add not in p:
            continue
        ns = p[direct].shape[0]
        for s in range(ns):
            tables_b[(name, s + 1)] = _make_epi_table_fast(epi_sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], heads=('epi', 'add'))
        specs_b.append((name, name, 'epi'))

    # SPURS-multi: native SPURS_multi_epi = SPURS_multi - SPURS_multi_additive
    unf_dir = BASE / 'retrained' / 'spurs_unfrozen' / 'predictions'
    have_unf = all((unf_dir / f'seed{sd}' / 'tsuboyama_test.csv').exists() for sd in (1, 2, 3))
    if have_unf:
        for sd in (1, 2, 3):
            d_u = pd.read_csv(unf_dir / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_multi_additive', 'SPURS_multi'], low_memory=False)
            a_u = _align_id(ids, d_u['uid'].astype(str).to_numpy(), d_u['SPURS_multi_additive'].to_numpy())
            m_u = _align_id(ids, d_u['uid'].astype(str).to_numpy(), d_u['SPURS_multi'].to_numpy())
            tables_b[('spurs_sc', sd)] = _make_epi_table_fast(epi_sp, a_u, m_u, heads=('epi', 'add'))
        specs_b.append(('SPURS multi', 'spurs_sc', 'epi'))
    elif 'spursm' in p:
        for sd in range(p['spursm'].shape[0]):
            tables_b[('spurs_sc', sd + 1)] = _make_epi_table_fast(epi_sp, p['spursm'][sd], p['spursm'][sd], heads=('epi', 'add'))
        specs_b.append(('SPURS multi', 'spurs_sc', 'epi'))

    # -dG_add saturation floor ceiling
    c_exact = np.array([float(dg_json.get(k, np.nan)) for k in sp['code'].astype(str)])
    wt_true = np.where((sp['n_mut'] == 1).to_numpy(), sp['y'].to_numpy(float), sp['ddG_additive_ML'].to_numpy(float))
    tables_b[('neg_dg_add', 1)] = _make_epi_table_fast(epi_sp, wt_true, -c_exact, heads=('epi', 'add'))
    specs_b.append(('−dG_add', 'neg_dg_add', 'epi'))

    T_b, reps_b, metric_names = _fast_boot_epistasis_models(tables_b, specs_b, B=B, seed=11, jobs=procs)
    T_b = T_b.assign(split='test')
    atomic_to_csv(T_b, out_supp / 'fig5_epistasis_levels_self_consistent_bootstrap_test.csv', index=False, float_format='%.5f')
    all_disp_b = [m[0] for m in specs_b]
    atomic_to_csv(_slice_paired_tests(T_b, reps_b, all_disp_b, metric_names, P0_EPI_NAME), out_supp / 'fig5_epistasis_levels_self_consistent_paired_test.csv', index=False, float_format='%.5f')

    entries = [
        _hier_entry('ProSST', 'ProSST'),
        _hier_entry('ProteinMPNN', 'ProteinMPNN'),
        _hier_entry('ESM3-small-open', 'ESM3-small-open'),
        _hier_entry(P0_EPI_NAME, 'ESM-MSR'),
        _hier_entry(P0_CTX_NAME, 'ESM-MSR', 'ESM-MSR ctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
        _hier_entry('ME stage 2', 'MutateEverything'),
        _hier_entry('ThermoMPNN-D', 'ThermoMPNN'),
        _hier_entry('SPURS multi', 'SPURS'),
        _hier_entry('GeoDDG-3D', 'GeoDDG-3D'),
        _hier_entry('Rosetta', 'Rosetta'),
    ]

    nd = int(T_a['n_domains'].iloc[0])
    nn = int(T_a['n_doubles'].iloc[0])
    B_a = int(T_a['B'].iloc[0])

    fig = plt.figure(figsize=(18.0, 13.8), facecolor='white')
    gs_a = fig.add_gridspec(1, 5, left=0.052, right=0.990, top=0.920, bottom=0.675, wspace=0.14)
    gs_b = fig.add_gridspec(1, 5, left=0.052, right=0.990, top=0.465, bottom=0.220, wspace=0.14)
    axes_a = [fig.add_subplot(gs_a[0, k]) for k in range(5)]
    axes_b = [fig.add_subplot(gs_b[0, k]) for k in range(5)]
    for k in range(1, 5):
        axes_a[k].sharey(axes_a[0])
        axes_a[k].tick_params(axis='y', labelleft=False)
        axes_b[k].sharey(axes_b[0])
        axes_b[k].tick_params(axis='y', labelleft=False)

    _draw_hierarchy_axes(axes_a, T_a, entries, LEVELS_CELL_MAG, overlay_sub0=True, compact_fonts=True)
    _draw_hierarchy_axes(axes_b, T_b, entries, LEVELS_CELL_MAG, overlay_sub0=True, compact_fonts=True)

    fig.canvas.draw()
    pa = axes_a[0].get_position()
    pb = axes_b[0].get_position()
    fig.text(0.010, pa.y1 + 0.062, 'a', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(
        0.032, pa.y1 + 0.060,
        'Deployed-singles epistasis (Main Fig. 2e): ESM-MSR & ESM3-small-open comb_AB − (WT_A + WT_B); ME Stage 2 − Stage 1; SPURS-multi − SPURS-single',
        ha='left', va='top', fontsize=10.6, fontweight='bold', color='#222222',
    )
    fig.text(0.010, pb.y1 + 0.062, 'b', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(
        0.032, pb.y1 + 0.060,
        'Self-consistent multi-model epistasis: comb_AB − (comb_A + comb_B) for ProSST / ESM3 / ESM-MSR / ctx ctrl; ME Stage 2_AB − (Stage 2_A + Stage 2_B); native SPURS_multi_epi',
        ha='left', va='top', fontsize=10.6, fontweight='bold', color='#222222',
    )

    note = (
        f'Homology-Restricted MegaScale test doubles with both singles measured (n = {nn:,} doubles across {nd} domains; 3-seed means for retrained models; '
        f'whiskers: 95% domain × seed bootstrap, B = {B_a:,} in a, B = {B:,} in b; black horizontal dashes: unsaturated subset ΔG_add > 0 kcal/mol; '
        f'red dotted line: −ΔG_add floor ceiling).  '
        'Panel a reproduces Main Fig. 2e using each model\'s deployed single-mutant head as the additive baseline (requiring only 1 WT pass/protein for ESM-MSR & ESM3-small-open; '
        'ProSST uses comb_AB − (comb_A + comb_B) in both panels to cancel unmasked self-bias; ThermoMPNN-D, ProteinMPNN, GeoDDG-3D, and Rosetta are identical in a and b).  '
        'Panel b subtracts each multi-mutant model\'s own single-mutant predictions (or native epistatic head SPURS_multi_epi), removing cross-head offsets at the cost of scoring every single mutant with the multi-mutant model.'
    )
    kw = dict(fontsize=8.0, color='#444444', linespacing=1.28)
    fig.text(0.012, 0.010, style.wrap_to_px(fig, note, 0.976 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'fig5a_epistasis_levels_bars_test_p0'
    style.save(fig, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION, with_split=True)
    ap.add_argument('--jobs', '--procs', dest='procs', type=int, default=8, help='Worker processes for epistasis hierarchy bootstrap')
    ap.add_argument('--only', choices=('all', 'core', 'bars', 'hierarchy', 'supp_figure4', 'supp_figure5', 'fig5a_compare'), default='all', help='Optional substage selector')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    if args.only == 'fig5a_compare':
        out = build_fig5a_epistasis_definitions_compare(B=B, procs=args.procs)
        print(f'[p06:epistasis_hierarchy] Saved {out}.{{png,pdf}}')
        return 0
    if args.only == 'supp_figure4':
        out = build_supp_figure4()
        print(f'[p06:epistasis_hierarchy] Saved {out}.{{png,pdf}}')
        return 0
    if args.only == 'supp_figure5':
        out = build_supp_figure5()
        print(f'[p06:epistasis_hierarchy] Saved {out}.{{png,pdf}}')
        return 0

    import time
    if args.only in ('all', 'core'):
        t0 = time.perf_counter()
        run_core_epistasis(B=B, split=args.split)
        print(f'  [p06:{args.split}] core_epistasis             B={B:4d}  {time.perf_counter() - t0:5.1f}s', flush=True)
    T3_df, p_ser = None, None
    if args.only in ('all', 'bars'):
        t1 = time.perf_counter()
        T3_df, p_ser = run_publication_epistasis_bars(B=B, split=args.split)
        print(f'  [p06:{args.split}] publication_epistasis_bars B={B:4d}  {time.perf_counter() - t1:5.1f}s', flush=True)
    if args.only in ('all', 'hierarchy'):
        t2 = time.perf_counter()
        run_hierarchy_and_composite(B=B, split=args.split, procs=args.procs, T3_df=T3_df, p_spearman_ser=p_ser)
        print(f'  [p06:{args.split}] hierarchy_and_composite    B={B:4d}  {time.perf_counter() - t2:5.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
