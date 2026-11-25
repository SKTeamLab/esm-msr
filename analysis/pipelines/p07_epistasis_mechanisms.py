#!/usr/bin/env python3
"""Pipeline p07: Assay floor saturation, Tobit link, 3D proximity mechanisms, pair heatmaps, and Rosetta multi-vs-additive.

Consolidates:
  - 11c_publication_epistasis_scatters.py -> results/01_megascale/epistasis/{dG_additive_vs_dddG, esm_msr_dddG_pred_vs_true*,
                                             esm_msr_models_dddG_pred_vs_true*, ddG_additive_vs_dddG_*, epistasis_scatters.csv, ddG_additive_kinds.csv}
  - 11f_epistasis_types.py                -> results/01_megascale/epistasis/{epistasis_types_by_additive_dG, alt_observed_additive_dG}.{csv,png,pdf}
  - 11g_link_figure.py                    -> results/01_megascale/epistasis/link_binned.csv (+ link_option_* under --exploratory)
  - 12_pair_epistasis_heatmaps.py         -> results/01_megascale/epistasis/pair_heatmaps/{pairs.csv, agreement.csv} (+ 30 per-pair PNGs under --pair-pngs)
  - 13_dddG_correlation_scatters.py       -> results/01_megascale/epistasis/dddG_scatters/{stats.csv, rosetta_unit_check.csv} and
                                             results/01_megascale/epistasis/dG_additive_vs_pred_dddG/{stats.csv, esm_msr_models_dG_additive_vs_pred_dddG}
  - 19_rosetta_multi_vs_additive.py       -> results/99_misc/rosetta_multi_vs_additive/rosetta_multi_vs_additive.{csv,md,png,pdf}
  - NEW mechanistic composite             -> results/00_supplementary_figs/floor_saturation_mechanism.{png,pdf}

Key methodological fixes vs legacy scripts:
  - Uses msr_analysis.paths.TSUBOYAMA_PDB_DIR + msr_analysis.structures + msr_analysis.tobit + msr_analysis.epistasis.smooth_on.
  - Fixes 19_rosetta_multi_vs_additive footnote to interpolate the actual B used in the run (removing hardcoded 200,000 / 100,000).
  - Replaces 8 redundant option_* / link_option_* drafts with the 4-panel floor_saturation_mechanism composite (gating drafts behind --exploratory).
"""
from __future__ import annotations

import multiprocessing as mp
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from msr_analysis import style
from msr_analysis.boot import bootstrap_rho
from msr_analysis.common import epi_rows, out_dir, singles_of_doubles
from msr_analysis.data import Frame, load_frame
from msr_analysis.epistasis import smooth_on
from msr_analysis.paths import TSUBOYAMA_PDB_DIR, atomic_to_csv, atomic_write_text
from msr_analysis.registry import tsu_datasets
from msr_analysis.stats import boot_pvalue, fmt_p
from msr_analysis.structures import load_pdb_residues, pair_cb_distance, parse_mutations
from msr_analysis.tobit import TOBIT, inv_leaky_link, inv_softplus_link, leaky_link, softplus_link
from esm_msr.val_stats import tobit_link

from . import base_parser, resolve_b

STAGE = 'epistasis_mechanisms'
DESCRIPTION = 'Assay floor saturation, Tobit link, mutant-backbone & 3D proximity mechanisms, and pair heatmaps'

Y_DDD_LABEL = 'measured δΔΔG = dddG (kcal/mol)'
X_DG_ADD_LABEL = 'ΔG additive = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)'
SCATTER_FONT = dict(
    label_fontsize=13.5,
    tick_fontsize=12.0,
    stats_fontsize=11.5,
    legend_fontsize=10.5,
    foot_fontsize=9.5,
    cbar_fontsize=12.5,
    cbar_tick_fontsize=10.5,
)

# Epistasis ladder constants (11f)
LO_F, TAU_F, EPS_F, CLIP_F = 0.0, 0.59, 0.05, -1.0
D_THR, S_THR, U_THR = 0.5, 1.0, 2.0
EPI_TYPES = ['implausible epistasis', 'compensation / positive sign', 'weak positive', 'near-neutral', 'weak negative', 'clash / negative sign']
EPI_RANGES = [f'ε > {U_THR:g}', f'{S_THR:g} < ε ≤ {U_THR:g}', f'{D_THR:g} < ε ≤ {S_THR:g}', f'|ε| ≤ {D_THR:g}', f'−{S_THR:g} ≤ ε < −{D_THR:g}', f'ε < −{S_THR:g}']
EPI_COLOURS = ['#000000', '#2166ac', '#92c5de', '#d9d9d9', '#f4a582', '#b2182b']
HATCH_EC = ['white', 'black', 'black', 'black', 'black', 'black']
DISTANT_A = 8.0

PAIR_AA = list('KRHDESTNQYCAVILMFWPG')
PAIR_IDX = {a: i for i, a in enumerate(PAIR_AA)}
HEATMAP_MODELS = [
    ('ProteinMPNN', ('diff', 'pmpnn', 'pmpnn+'), 'pmpnn+', 'pmpnn'),
    ('ESM3-small-open', ('key', 'msrrel@0:epi'), 'msrrel@0:wt+', 'msrrel@0:wt'),
    ('ESM-MSR', ('key', 'msrp0:epi'), 'msrp0:wt+', 'msrp0:wt'),
    ('MutateEverything Stage 2', ('key', 'me2:epi'), 'me1', 'me1'),
    ('ThermoMPNN-D', ('key', 'tmpnnd:epi'), 'tmpnn', 'tmpnn'),
    ('SPURS-multi', ('key', 'spursm:epi'), 'spurs', 'spurs'),
    ('GeoDDG-3D', ('diff', 'geoddg', 'geoddg+'), 'geoddg+', 'geoddg'),
    ('Rosetta', ('diff', 'rosetta', 'rosetta+'), 'rosetta+', 'rosetta'),
    ('ESM3ΔG', ('diff', 'esm3dg', 'esm3dg+'), 'esm3dg+', 'esm3dg'),
    ('ESMtherm*', ('diff', 'esmtherm', 'esmtherm+'), 'esmtherm+', 'esmtherm'),
    ('IFUM*', ('diff', 'ifum_wt', 'ifum_wt+'), 'ifum_wt+', 'ifum_wt'),
    ('DDGemb', ('diff', 'ddgemb', 'ddgemb+'), 'ddgemb+', 'ddgemb'),
]


# ----------------------------------------------------------------------------------------------
# 1. Fig 6 scatters & Fig 7 epistasis types (11c + 11f)
# ----------------------------------------------------------------------------------------------
MAIN_DDD_GRID_MODELS = [
    ('ESM-MSR', 'Retrained'),
    ('MutateEverything Stage 2', 'Retrained'),
    ('ThermoMPNN-D', 'Retrained'),
    ('SPURS-multi', 'Retrained'),
    ('ProteinMPNN', 'Zero-shot'),
    ('ESM3-small-open', 'Zero-shot'),
    ('GeoDDG-3D', 'Supervised'),
    ('Rosetta', 'Supervised'),
]


def _main_models_dddg_grid(out_dddg, split: str, fit: np.ndarray, y_raw: np.ndarray, dG_add: np.ndarray, model: dict) -> None:
    """2x4 subplots of predicted vs measured dddG across all 8 main epistasis models (all doubles and dG_add > 0)."""
    entries = [(lbl, grp, model[lbl]['raw_seeds']) for lbl, grp in MAIN_DDD_GRID_MODELS if lbl in model]
    if not entries:
        return
    sp_lbl = 'MegaScale mega_splits test doubles' if split == 'mega' else 'Homology-Restricted MegaScale test doubles'
    cm = plt.get_cmap('plasma')
    ncols = 4
    nrows = int(np.ceil(len(entries) / ncols))

    for fname, subset_title, cut in (
        ('main_models_dddG_pred_vs_true_all', 'all doubles', None),
        ('main_models_dddG_pred_vs_true_dG_gt0', 'ΔG_add > 0 (unsaturated doubles)', 0.0),
    ):
        ok = fit & np.isfinite(y_raw) & (True if cut is None else (np.isfinite(dG_add) & (dG_add > cut)))
        for _, _, p in entries:
            ok &= np.isfinite(p).all(axis=0)
        yv_1d = y_raw[ok]
        n_var = len(yv_1d)
        # Compute shared square axis limits from measured y and pooled model predictions (0.3–99.7 pct)
        pool_vals = np.concatenate([yv_1d] + [p[:, ok].ravel() for _, _, p in entries])
        lo, hi = np.percentile(pool_vals, [0.3, 99.7])
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 5.1 * nrows),
            sharex=True, sharey=True, squeeze=False,
        )
        flat_axes = axes.ravel()
        for idx, (ax, (m_label, grp, p)) in enumerate(zip(flat_axes, entries)):
            p_ok = p[:, ok]
            n_seeds = p_ok.shape[0]
            xv = p_ok.ravel()
            yv = np.tile(yv_1d, n_seeds)
            n_pts = len(xv)
            sub = np.random.default_rng(0).choice(n_pts, min(n_pts, 15000), replace=False)
            z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
            o = np.argsort(z)
            col = cm(Normalize(z.min(), z.max())(z))
            col[:, 3] = 0.3 / np.sqrt(n_seeds)

            ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=1)
            ax.axvline(0, color='#bbbbbb', lw=0.8, zorder=1)
            ax.scatter(xv[o], yv[o], c=col[o], s=7, edgecolors='none', rasterized=True, zorder=2)
            ax.plot(lim, lim, color='#c0392b', ls='--', lw=1.5, zorder=4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            row, col_i = divmod(idx, ncols)
            if col_i == 0:
                ax.set_ylabel(Y_DDD_LABEL, fontsize=12.0)
            if row == nrows - 1:
                ax.set_xlabel('predicted δΔΔG = dddG (kcal/mol)', fontsize=12.0)
            ax.tick_params(axis='both', labelsize=10.5)

            rho_v = float(np.mean([spearmanr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((p_ok[s] - yv_1d) ** 2)) for s in range(n_seeds)]))
            n_str = f'n = {n_var:,} (×{n_seeds})' if n_seeds > 1 else f'n = {n_var:,}'
            ax.text(
                0.04, 0.96, f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nRMSE = {rmse_v:.2f}\n{n_str}',
                transform=ax.transAxes, va='top', fontsize=10.2, zorder=10,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#cccccc', alpha=0.92),
            )
            ax.text(
                0.96, 0.04, grp,
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5, color='#555555', zorder=10,
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.75),
            )
            if idx == 0:
                ax.legend(handles=[Line2D([0], [0], color='#c0392b', ls='--', lw=1.5, label='x = y')], loc='lower left', fontsize=9.5, framealpha=0.9)
            ax.set_title(m_label, fontsize=13.0, fontweight='bold', pad=7)

        for ax in flat_axes[len(entries):]:
            ax.set_visible(False)

        fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.995])
        fig.text(
            0.01, 0.005,
            f'{sp_lbl} with both singles measured ({subset_title}, n = {n_var:,}). Retrained models show 3 triplicate seeds integrated without ensemble averaging; stats are seed means.',
            fontsize=9.5, va='bottom', ha='left', color='#444444',
        )
        also_supp = (split == 'capped' and cut == 0.0)
        style.save(fig, out_dddg / fname, close=not also_supp)
        if also_supp:
            style.save(fig, out_dir('supplementary_figs') / '_epi_rmse')


def _main_models_dddg_rmse_bars(out_dddg, split: str, fr: Frame, fit: np.ndarray, dG_add: np.ndarray, B: int = 1000) -> None:
    """Grouped bar chart of epistasis RMSE (all doubles vs dG_add > 0) across all main models in global bar_panel style."""
    from msr_analysis.boot import ci as boot_ci
    from ._tsu_common import best_other, epi_columns, fig_width

    sp = fr.spine
    y = sp['dddG_ML'].to_numpy(float)
    cl = sp['cluster'].astype(str).to_numpy()
    grp = sp['group'].astype(str).to_numpy()
    best, _ = best_other(fr)
    cols = epi_columns(fr, best)
    subsets = (
        ('all', fit & np.isfinite(y), 'all doubles'),
        ('dG_gt0', fit & np.isfinite(y) & np.isfinite(dG_add) & (dG_add > 0.0), 'ΔG_add > 0 (unsaturated)'),
    )

    rows = []
    for c in cols:
        if not c.get('draw', True):
            continue
        k = c['double']
        if k not in fr.preds:
            continue
        preds = fr.preds[k]
        for sub_key, m_sub, _ in subsets:
            ok = m_sub & np.isfinite(preds).all(axis=0)
            u, inv = np.unique(cl[ok], return_inverse=True)
            e = np.stack([p[ok] - y[ok] for p in preds])
            pt = float(np.mean(np.sqrt((e ** 2).mean(axis=1))))
            ns = e.shape[0]
            rng = np.random.default_rng(0)
            reps = np.empty(B, dtype=float)
            for b in range(B):
                w = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u))).astype(float)[inv]
                sw = rng.multinomial(ns, np.full(ns, 1.0 / ns)).astype(float) / ns if ns > 1 else np.ones(1)
                per_seed_rm = np.sqrt(((e ** 2) * w).sum(axis=1) / w.sum())
                reps[b] = float(np.dot(sw, per_seed_rm))
            lo, hi = boot_ci(reps)
            rows.append(dict(
                column=c['name'], key=k, subset=sub_key, value=pt, lo=lo, hi=hi,
                n_variants=int(ok.sum()), n_libraries=int(len(np.unique(grp[ok]))), n_clusters=int(len(u)), n_seeds=ns,
            ))

    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dddg / 'main_models_dddG_rmse_bars.csv', index=False, float_format='%.4f')

    sub_labels = tuple(
        f"{desc} (n = {int(df.loc[df['subset'] == sk, 'n_variants'].max()):,}, {int(df.loc[df['subset'] == sk, 'n_libraries'].max())} domains)"
        for sk, _, desc in subsets
    )
    ymax = float(np.ceil((df['hi'].max() + 0.34) / 0.2) * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width(cols), 4.8))
    style.bar_panel(
        ax, cols, df, ('all', 'dG_gt0'), sub_labels,
        ylabel='epistasis RMSE (kcal/mol; lower is better)',
        ylim=(0.0, ymax),
        zero_line=False,
    )
    ax.set_xlabel('predicted epistasis (1/2(MT_AB+WT_AB) − WT_A − WT_B for ESM-MSR / ESM3; direct − sum of singles for others)', fontsize=8.3)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    style.save(fig, out_dddg / 'main_models_dddG_rmse_bars', close=(split != 'capped'))
    if split == 'capped':
        style.save(fig, out_dir('supplementary_figs') / '_epi_rmse_bars')


def _compare_models_scatters(out_path, title_str: str, m: np.ndarray, y: np.ndarray, dG_add: np.ndarray, model_preds: list) -> None:
    ok = m & np.isfinite(y)
    for _, p in model_preds:
        ok &= np.isfinite(p).all(axis=0)
    yv_1d = y[ok]
    hl_1d = (dG_add < 0)[ok]
    n_var = len(yv_1d)
    all_vals = np.concatenate([yv_1d] + [p[:, ok].ravel() for _, p in model_preds])
    lo, hi = np.percentile(all_vals, [0.3, 99.7])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    cm = plt.get_cmap('plasma')
    formula = "ESM-MSR 1/2(MT_AB+WT_AB) − WT_A − WT_B; baselines: direct double − the same seed's singles"

    for fname, use_hl in (
        ('esm_msr_models_dddG_pred_vs_true', False),
        ('esm_msr_models_dddG_pred_vs_true_dG_additive_lt0', True),
    ):
        fig, axes = plt.subplots(1, len(model_preds), figsize=(6.0 * len(model_preds), 5.8), sharex=True, sharey=True)
        for idx, (ax, (m_label, p)) in enumerate(zip(axes, model_preds)):
            p_ok = p[:, ok]
            n_seeds = p_ok.shape[0]
            xv = p_ok.ravel()
            yv = np.tile(yv_1d, n_seeds)
            n_pts = len(xv)
            sub = np.random.default_rng(0).choice(n_pts, min(n_pts, 15000), replace=False)
            z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
            o = np.argsort(z)
            col = cm(Normalize(z.min(), z.max())(z))
            col[:, 3] = 0.3 / np.sqrt(n_seeds)
            hl = np.tile(hl_1d, n_seeds) if use_hl else np.zeros(n_pts, bool)
            keep = o[~hl[o]]
            ax.scatter(xv[keep], yv[keep], c=col[keep], s=8, edgecolors='none', rasterized=True)
            handles = []
            if hl.any():
                ax.scatter(xv[hl], yv[hl], c='#1a9850', s=8, alpha=0.55 / np.sqrt(n_seeds), edgecolors='none', rasterized=True, zorder=3)
                handles.append(Line2D([0], [0], marker='o', ls='', color='#1a9850', alpha=0.8, label=f'ΔG additive < 0 (n = {int(hl_1d.sum()):,})'))
            ax.plot(lim, lim, color='#c0392b', ls='--', lw=1.6, zorder=4)
            handles.append(Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='x = y'))
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f'{m_label} predicted δΔΔG = dddG (kcal/mol)', fontsize=SCATTER_FONT['label_fontsize'])
            if idx == 0:
                ax.set_ylabel(Y_DDD_LABEL, fontsize=SCATTER_FONT['label_fontsize'])
            ax.tick_params(axis='both', labelsize=SCATTER_FONT['tick_fontsize'])
            rho_v = float(np.mean([spearmanr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((p_ok[s] - yv_1d) ** 2)) for s in range(n_seeds)]))
            ax.text(
                0.04, 0.96, f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {n_var:,} (×{n_seeds} seeds)\nRMSE = {rmse_v:.2f}',
                transform=ax.transAxes, va='top', fontsize=SCATTER_FONT['stats_fontsize'], zorder=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94),
            )
            ax.legend(handles=handles, loc='lower right', fontsize=SCATTER_FONT['legend_fontsize'], framealpha=0.92)
            ax.set_title(m_label, fontsize=13.5, fontweight='bold', pad=8)
        fig.tight_layout()
        fig.text(
            0.012, -0.01, f'{title_str} (n = {n_var:,}, 3 triplicate seeds integrated without ensemble averaging); predicted epistasis = {formula}',
            fontsize=SCATTER_FONT['foot_fontsize'], va='top', ha='left', color='#444444',
        )
        style.save(fig, out_path / fname)


def run_fig6_scatters(split: str = 'capped', exploratory: bool = False) -> None:
    out = out_dir('epistasis', split=split)
    ds = tsu_datasets(split)[1]
    title_str = 'MegaScale mega_splits test doubles with both singles measured' if split == 'mega' else 'Homology-Restricted MegaScale test doubles with both singles measured'
    fr = load_frame(ds)
    sp = fr.spine
    m = epi_rows(fr)
    y = sp['dddG_ML'].to_numpy()
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    pred_msr = fr.preds['msrp0:epi']

    rows = []
    jobs = [
        ('dG_additive_vs_dddG', dG_add, X_DG_ADD_LABEL, dict(ref='neg', ref_label='y = −x: dG_AB = 0,\nwhere the unfolded doubles pile up')),
        ('esm_msr_dddG_pred_vs_true', pred_msr, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True)),
        ('esm_msr_dddG_pred_vs_true_dG_additive_lt0', pred_msr, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True, highlight=dG_add < 0, highlight_label='ΔG additive < 0')),
    ]
    for name, x, xl, kw in jobs:
        if x.ndim == 2:
            ok = m & np.isfinite(x).all(axis=0) & np.isfinite(y)
            xs_in = x[:, ok]
            n_seeds = xs_in.shape[0]
            rho_v = float(np.mean([spearmanr(xs_in[s], y[ok])[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(xs_in[s], y[ok])[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((xs_in[s] - y[ok]) ** 2)) for s in range(n_seeds)])) if kw.get('ref') == 'xy' else np.nan
            rho_pool = float(spearmanr(xs_in.ravel(), np.tile(y[ok], n_seeds))[0])
            r_pool = float(pearsonr(xs_in.ravel(), np.tile(y[ok], n_seeds))[0])
        else:
            ok = m & np.isfinite(x) & np.isfinite(y)
            xs_in = x[ok]
            n_seeds = 1
            rho_v = rho_pool = float(spearmanr(xs_in, y[ok])[0])
            r_v = r_pool = float(pearsonr(xs_in, y[ok])[0])
            rmse_v = float(np.sqrt(np.mean((xs_in - y[ok]) ** 2))) if kw.get('ref') == 'xy' else np.nan
        kw_use = {**kw, 'highlight': kw['highlight'][ok]} if 'highlight' in kw else kw
        fig = style.density_scatter_plain(xs_in, y[ok], xl, Y_DDD_LABEL, title=None, foot=f'{title_str} (n = {int(ok.sum()):,})', **SCATTER_FONT, **kw_use)
        if name == 'dG_additive_vs_dddG':
            ax = fig.axes[0]
            xl_, yl_ = ax.get_xlim(), ax.get_ylim()
            ax.plot(xl_, [-v - 1.0 for v in xl_], color='#c0392b', ls=':', lw=1.6, zorder=4)
            ax.set_xlim(xl_)
            ax.set_ylim(yl_)
            ax.legend(
                handles=ax.get_legend().legend_handles + [Line2D([0], [0], color='#c0392b', ls=':', lw=1.6, label='y = −x − 1: dG_AB = −1,\nthe lowest value the assay reports')],
                loc='lower left', fontsize=10.2, frameon=False, labelspacing=0.45,
            )
        style.save(fig, out / name)
        rows.append(dict(
            figure=name, n=int(ok.sum()), n_seeds=n_seeds,
            spearman=rho_v, pearson=r_v, spearman_stacked=rho_pool, pearson_stacked=r_pool,
            share_dG_additive_lt0=float(np.mean(dG_add[ok] < 0)), rmse_to_xy=rmse_v,
        ))
    atomic_to_csv(pd.DataFrame(rows), out / 'epistasis_scatters.csv', index=False, float_format='%.4f')

    _compare_models_scatters(
        out, title_str, m, y, dG_add,
        [('ESM-MSR', pred_msr), ('MutateEverything Stage 2', fr.preds['me2:epi']), ('ThermoMPNN-D', fr.preds['tmpnnd:epi']), ('SPURS-multi', fr.preds['spursm:epi'])],
    )

    # Unsaturated doubles (dG_add > 0)
    ddg = sp['y'].to_numpy(float)
    x_add = sp['ddG_additive_ML'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok_d = m & (dG_add > 0) & np.isfinite(x_add) & np.isfinite(y) & (ia >= 0) & (ib >= 0)
    xs, ys, dab = x_add[ok_d], y[ok_d], ddg[ok_d]
    sa, sb = ddg[ia[ok_d]], ddg[ib[ok_d]]
    ttl_d = f'{title_str}, dG additive > 0 (n = {int(ok_d.sum()):,})'
    xl_d = 'ΔΔG additive = ΔΔG_A + ΔΔG_B (kcal/mol)'
    fig_d = style.density_scatter_plain(xs, ys, xl_d, Y_DDD_LABEL, title=None, foot=ttl_d, ref='neg', ref_label='y = −x: ΔΔG_AB = 0', **SCATTER_FONT)
    ax_d = fig_d.axes[0]
    ax_d.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
    ax_d.legend(handles=ax_d.get_legend().legend_handles + [Line2D([0], [0], color='#555555', lw=1.2, ls=':', label='y = 0: no epistasis')], loc='lower left', fontsize=10.5, frameon=False)
    style.save(fig_d, out / 'ddG_additive_vs_dddG_dG_additive_gt0')

    kind = np.where((sa > 0) & (sb > 0), 'both stabilizing', np.where((sa > 0) | (sb > 0), 'one stabilizing', 'both destabilizing'))
    colours = {'both destabilizing': '#2c3e50', 'one stabilizing': '#e67e22', 'both stabilizing': '#1a9850'}
    limx, limy = np.percentile(xs, [0.3, 99.7]), np.percentile(ys, [0.3, 99.7])
    padx, pady = 0.05 * (limx[1] - limx[0]), 0.05 * (limy[1] - limy[0])
    fig_e, ax_e = plt.subplots(figsize=(6.6, 5.8))
    handles_e, k_rows = [], []
    for name in ('both destabilizing', 'one stabilizing', 'both stabilizing'):
        k = kind == name
        if not k.any():
            continue
        ax_e.scatter(xs[k], ys[k], s=8, c=colours[name], alpha=0.3, edgecolors='none', rasterized=True)
        handles_e.append(Line2D([0], [0], marker='o', ls='', color=colours[name], alpha=0.8, label=f'{name} (n = {int(k.sum()):,})'))
        k_rows.append(dict(
            singles=name, n=int(k.sum()), mean_ddG_additive=float(xs[k].mean()), mean_dddG=float(ys[k].mean()),
            share_positive=float(np.mean(ys[k] > 0.5)), share_none=float(np.mean(np.abs(ys[k]) <= 0.5)),
            share_negative=float(np.mean(ys[k] < -0.5)), share_double_stabilizing=float(np.mean(dab[k] > 0)),
            spearman=float(spearmanr(xs[k], ys[k])[0]) if k.sum() > 2 else np.nan,
        ))
    ax_e.plot([limx[0] - padx, limx[1] + padx], [-(limx[0] - padx), -(limx[1] + padx)], color='#c0392b', ls='--', lw=1.6, zorder=4)
    ax_e.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
    handles_e += [Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='y = −x: ΔΔG_AB = 0'), Line2D([0], [0], color='#555555', lw=1.2, ls=':', label='y = 0: no epistasis')]
    ax_e.set_xlim(limx[0] - padx, limx[1] + padx)
    ax_e.set_ylim(limy[0] - pady, limy[1] + pady)
    ax_e.set_xlabel(xl_d, fontsize=13.5)
    ax_e.set_ylabel(Y_DDD_LABEL, fontsize=13.5)
    ax_e.tick_params(axis='both', labelsize=12.0)
    ax_e.legend(handles=handles_e, loc='upper right', fontsize=10.0, framealpha=0.92)
    fig_e.tight_layout()
    fig_e.text(0.015, -0.01, ttl_d, fontsize=9.5, va='top', ha='left', color='#444444')
    style.save(fig_e, out / 'ddG_additive_vs_dddG_by_singles')
    atomic_to_csv(pd.DataFrame(k_rows), out / 'ddG_additive_kinds.csv', index=False, float_format='%.4f')

    if exploratory:
        _draw_exploratory_fig6f(out, title_str, sp, m, y, dG_add, ia, ib)


def _draw_exploratory_fig6f(out, title_str: str, sp: pd.DataFrame, m: np.ndarray, y: np.ndarray, dG_add: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> None:
    ddg = sp['y'].to_numpy(float)
    ok = m & np.isfinite(y) & np.isfinite(dG_add) & (ia >= 0) & (ib >= 0)
    a, b = ddg[ia[ok]], ddg[ib[ok]]
    xa, yb, z, g = np.minimum(a, b), np.maximum(a, b), y[ok], dG_add[ok]
    sat = g <= 0.0
    vmax = float(np.ceil(np.percentile(np.abs(z[~sat]), 98) * 4) / 4)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    limx, limy = np.percentile(xa, [0.3, 99.7]), np.percentile(yb, [0.3, 99.7])
    padx, pady = 0.05 * (limx[1] - limx[0]), 0.05 * (limy[1] - limy[0])
    ext = (limx[0] - padx, limx[1] + padx, limy[0] - pady, limy[1] + pady)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    for ax, ttl in zip(axes, (f'dG additive > 0: clear of floor (n = {int((~sat).sum()):,})', f'with saturation-dominated doubles in black (n = {len(z):,})')):
        ax.plot([ext[0], ext[3]], [ext[0], ext[3]], color='#888888', ls=':', lw=1.0, zorder=0)
        ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
        ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel('ΔΔG_A: the more destabilizing single (kcal/mol)')
        ax.set_title(ttl, fontsize=9, fontweight='bold', loc='left')
    axes[0].set_ylabel('ΔΔG_B: the other single (kcal/mol)')
    o_clear = np.flatnonzero(~sat)
    o_clear = o_clear[np.argsort(np.abs(z[o_clear]))]
    sc = axes[0].scatter(xa[o_clear], yb[o_clear], c=z[o_clear], cmap='RdBu', norm=norm, s=8, alpha=0.8, edgecolors='none', rasterized=True)
    axes[1].scatter(xa[sat], yb[sat], c='black', s=7, alpha=0.4, edgecolors='none', rasterized=True)
    axes[1].scatter(xa[o_clear], yb[o_clear], c=z[o_clear], cmap='RdBu', norm=norm, s=8, alpha=0.8, edgecolors='none', rasterized=True)
    fig.suptitle(f'{title_str}: measured epistasis over the plane of the two singles', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    cax = fig.add_axes([0.945, 0.12, 0.014, 0.74])
    fig.colorbar(sc, cax=cax).set_label('measured dddG (kcal/mol); blue positive, red negative')
    style.save(fig, out / 'option_A_scatter_black_added')


def _compute_epistasis_types(fr: Frame, x_kind: str, B: int) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    c = sp['dG_wt'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = epi_rows(fr) & (ia >= 0) & (ib >= 0)
    W = np.maximum(c[ok], CLIP_F)
    A = np.maximum(c[ok] + ddg[ia[ok]], CLIP_F)
    Bm = np.maximum(c[ok] + ddg[ib[ok]], CLIP_F)
    Y = np.maximum(c[ok] + ddg[ok], CLIP_F)
    dom = sp['group'].to_numpy()[ok]
    z = inv_softplus_link(A, floor=LO_F, tau=TAU_F, eps=EPS_F) + inv_softplus_link(Bm, floor=LO_F, tau=TAU_F, eps=EPS_F) - inv_softplus_link(W, floor=LO_F, tau=TAU_F, eps=EPS_F)
    x = A + Bm - W
    eps = Y - x
    net = (A - W) + (Bm - W)
    if x_kind == 'observed':
        z = x
        edges = np.arange(-5, 6)
    else:
        edges = np.arange(-6, 6)
    labels_bin = [f'{a}..{b}' for a, b in zip(edges[:-1], edges[1:])]
    flag = np.abs(softplus_link(z, floor=LO_F, tau=TAU_F) - x) > D_THR
    bi = np.clip(np.floor(z).astype(int) - int(edges[0]), 0, len(labels_bin) - 1)
    half = (net > 0).astype(int)
    cat = np.where(eps > U_THR, 0, np.where(eps > S_THR, 1, np.where(eps > D_THR, 2, np.where(eps >= -D_THR, 3, np.where(eps >= -S_THR, 4, 5)))))
    doms = np.unique(dom)
    counts = np.random.default_rng(5).multinomial(len(doms), np.full(len(doms), 1.0 / len(doms)), size=B).astype(float)

    nb, nc = len(labels_bin), len(EPI_TYPES)
    N = np.zeros((len(doms), nb, nc))
    F = np.zeros((len(doms), nb, nc))
    H = np.zeros((len(doms), nb, 2))
    for k, d in enumerate(doms):
        mk = dom == d
        np.add.at(N[k], (bi[mk], cat[mk]), 1)
        np.add.at(F[k], (bi[mk & flag], cat[mk & flag]), 1)
        np.add.at(H[k], (bi[mk], half[mk]), 1)
    tot, fl = N.sum(0), F.sum(0)
    nbin = np.maximum(tot.sum(1, keepdims=True), 1)
    rn, rf = np.einsum('rk,kbc->rbc', counts, N), np.einsum('rk,kbc->rbc', counts, F)
    rb = np.maximum(rn.sum(2, keepdims=True), 1)
    ci_fn = lambda arr: (np.nanpercentile(arr / rb, 2.5, axis=0), np.nanpercentile(arr / rb, 97.5, axis=0))
    r = dict(share=tot / nbin, floor=fl / nbin, n=tot, nbin=tot.sum(1), half=H.sum(0), ci=ci_fn(rn), ci_floor=ci_fn(rf))
    keep = r['nbin'] > 0
    if not keep.all():
        labels_bin = [lab for lab, k in zip(labels_bin, keep) if k]
        r = dict(
            share=r['share'][keep], floor=r['floor'][keep], n=r['n'][keep],
            nbin=r['nbin'][keep], half=r['half'][keep],
            ci=(r['ci'][0][keep], r['ci'][1][keep]),
            ci_floor=(r['ci_floor'][0][keep], r['ci_floor'][1][keep]),
        )
    return r, labels_bin, cat, flag


def run_fig7_epistasis_types(B: int, split: str = 'capped') -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    out = out_dir('epistasis', split=split)
    fr = load_frame(tsu_datasets(split)[1])
    latent_result = None
    for x_kind in ('latent', 'observed'):
        r, labels_bin, cat, flag = _compute_epistasis_types(fr, x_kind, B=B)
        if x_kind == 'latent':
            latent_result = (r, labels_bin, cat, flag)
        rows = []
        for b in range(len(labels_bin)):
            for t in range(len(EPI_TYPES)):
                rows.append(dict(
                    z_bin=labels_bin[b], type=EPI_TYPES[t], n_bin=int(r['nbin'][b]),
                    n_net_ddG_additive_negative=int(r['half'][b, 0]), n_net_ddG_additive_positive=int(r['half'][b, 1]),
                    n=int(r['n'][b, t]), share=r['share'][b, t],
                    lo=r['ci'][0][b, t], hi=r['ci'][1][b, t],
                    share_floor_decides=r['floor'][b, t], lo_floor=r['ci_floor'][0][b, t], hi_floor=r['ci_floor'][1][b, t],
                ))
        stem = 'epistasis_types_by_additive_dG' if x_kind == 'latent' else 'alt_observed_additive_dG'
        atomic_to_csv(pd.DataFrame(rows), out / f'{stem}.csv', index=False, float_format='%.4f')

        fig, (ax, an) = plt.subplots(2, 1, figsize=(8.2, 8.24), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        nb, nc = len(labels_bin), len(EPI_TYPES)
        xs = np.arange(nb)
        bottom = np.zeros(nb)
        for t in range(nc):
            rest = r['share'][:, t] - r['floor'][:, t]
            ax.bar(xs, r['floor'][:, t], bottom=bottom, color=EPI_COLOURS[t], width=0.9, edgecolor=HATCH_EC[t], lw=0, hatch='////')
            bottom += r['floor'][:, t]
            ax.bar(xs, rest, bottom=bottom, color=EPI_COLOURS[t], width=0.9, edgecolor='white', lw=0.4)
            bottom += rest
        ax.set_ylim(0, 1)
        ax.set_ylabel('fraction of doubles', fontsize=13.0)
        ax.tick_params(axis='y', labelsize=11.5)
        ov = np.bincount(cat, minlength=nc) / len(cat)
        handles = [Patch(fc=EPI_COLOURS[t], ec='#999999', lw=0.4) for t in range(nc)] + [Patch(fc='white', ec='black', lw=0, hatch='////')]
        labels = [f'{EPI_TYPES[t]} ({EPI_RANGES[t]}; {100 * ov[t]:.0f}%)' for t in range(nc)] + [f'hatched: floor decides ({100 * flag.mean():.0f}%)']
        ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.008, 0.02), fontsize=10.2, handlelength=1.4, frameon=True, facecolor='white', framealpha=0.86, edgecolor='#999999').set_zorder(10)
        wd = 0.45
        an.bar(xs - wd / 2, r['half'][:, 0], color='#888888', width=wd, edgecolor='white', lw=0.4, label="singles' summed ΔΔG < 0")
        an.bar(xs + wd / 2, r['half'][:, 1], color='#222222', width=wd, edgecolor='white', lw=0.4, label="singles' summed ΔΔG > 0")
        an.set_yscale('log')
        an.set_ylabel('n', fontsize=13.0)
        an.legend(loc='upper right', fontsize=10.0, frameon=False, handlelength=1.3)
        an.set_xticks(range(nb))
        an.set_xticklabels(labels_bin, rotation=45, fontsize=11.0)
        an.set_xlabel('additive expected dG: latent z (kcal/mol)' if x_kind == 'latent' else 'additive dG = dG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=12.0)
        fig.suptitle(
            f'MegaScale test doubles with both singles measured (n = {len(cat):,}):\n'
            'kinds of epistasis as the additive expectation falls into the assay floor (hatched: floor decides)',
            fontsize=10.2, x=0.015, ha='left', y=0.985,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        style.save(fig, out / stem)
    return latent_result


# ----------------------------------------------------------------------------------------------
# 2. Link curves, 3D distant pairs, and NEW floor_saturation_mechanism composite (11g + composite)
# ----------------------------------------------------------------------------------------------
def _cb_distances_for_doubles(mut_arr: np.ndarray, code_arr: np.ndarray) -> np.ndarray:
    cache: dict[str, dict] = {}
    dist = np.full(len(mut_arr), np.nan, dtype=float)
    if not TSUBOYAMA_PDB_DIR.is_dir():
        return dist
    for k, (mm, cd) in enumerate(zip(mut_arr, code_arr)):
        f = TSUBOYAMA_PDB_DIR / f'{cd}.pdb'
        if not f.exists():
            continue
        residues = cache.setdefault(cd, load_pdb_residues(f))
        toks = parse_mutations(str(mm), sep=':')
        if toks and len(toks) == 2:
            (wa, pa, _ma), (wb, pb, _mb) = toks
            if pa in residues and pb in residues and residues[pa]['aa'] == wa and residues[pb]['aa'] == wb:
                dist[k] = pair_cb_distance(residues, pa, pb)
    return dist


def _binned_quantiles(x: np.ndarray, y: np.ndarray, edges: np.ndarray, nmin: int = 15) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x > lo) & (x <= hi)
        if m.sum() >= nmin:
            rows.append((float(x[m].mean()), int(m.sum()), float(np.median(y[m])), *np.quantile(y[m], [0.25, 0.75])))
    return pd.DataFrame(rows, columns=['x', 'n', 'median', 'q25', 'q75'])


def run_link_and_floor_composite(split: str, latent_types: tuple, exploratory: bool = False) -> None:
    out_epi = out_dir('epistasis', split=split)
    fr = load_frame(tsu_datasets(split)[1])
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    c_all = sp['dG_wt'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = epi_rows(fr) & (ia >= 0) & (ib >= 0)
    idx = np.flatnonzero(ok)
    W = c_all[ok]
    A, B_s = W + ddg[ia[ok]], W + ddg[ib[ok]]
    Y = W + ddg[ok]
    x = A + B_s - W
    dist = _cb_distances_for_doubles(sp['mut'].to_numpy(dtype=object)[idx], sp['code'].astype(str).to_numpy()[idx])
    far = np.isfinite(dist) & (dist > DISTANT_A)
    edges = np.arange(-4.5, 4.51, 0.5)
    b_all = _binned_quantiles(x, Y, edges, nmin=15)
    b_far = _binned_quantiles(x[far], Y[far], edges, nmin=10) if far.any() else pd.DataFrame(columns=['x', 'n', 'median', 'q25', 'q75'])
    binned_out = pd.concat({'all': b_all, 'distant': b_far}, names=['set']).reset_index(level=0)
    atomic_to_csv(binned_out, out_epi / 'link_binned.csv', index=False, float_format='%.4f')

    if split != 'capped':
        return

    # NEW High-Impact Composite Figure #2: results/00_supplementary_figs/floor_saturation_mechanism.{png,pdf}
    out_supp = out_dir('supplementary_figs')
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 11.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    xs_curve = np.linspace(-4.8, 4.8, 400)
    xlim, ylim_a = (-4.8, 4.8), (-1.25, 5.0)
    c_soft, c_leaky = '#000000', '#e07b00'

    # Panel a: Observed dG_AB vs additive dG on distant pairs (CB-CB > 8 A) with softplus & leaky link curves
    x_plot = x[far] if far.sum() >= 50 else x
    y_plot = Y[far] if far.sum() >= 50 else Y
    b_plot = b_far if far.sum() >= 50 else b_all
    far_lbl = f'distant doubles (Cβ–Cβ > {DISTANT_A:g} Å, n = {len(y_plot):,})' if far.sum() >= 50 else f'all doubles (n = {len(y_plot):,})'
    ax_a.hexbin(x_plot, y_plot, gridsize=40, extent=(*xlim, *ylim_a), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
    ax_a.plot(xlim, xlim, color='#999999', ls=':', lw=1.0, zorder=3)
    if len(b_plot):
        ax_a.fill_between(b_plot['x'], b_plot['q25'], b_plot['q75'], color='#7f0000', alpha=0.18, lw=0, zorder=4)
        ax_a.plot(b_plot['x'], b_plot['median'], color='#7f0000', lw=1.4, marker='o', ms=2.6, zorder=5)
    ax_a.plot(xs_curve, softplus_link(xs_curve), color=c_soft, lw=1.9, zorder=6)
    ax_a.plot(xs_curve, leaky_link(xs_curve), color=c_leaky, lw=1.9, ls='--', zorder=6)
    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(*ylim_a)
    ax_a.set_xlabel('additive ΔG = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=10.2)
    ax_a.set_ylabel('observed ΔG_AB of double (kcal/mol)', fontsize=10.2)
    ax_a.set_title(f'Obs. ΔG_AB vs additive ΔG on {far_lbl}', fontsize=10.5, fontweight='bold', loc='left')
    ax_a.legend(
        handles=[
            Line2D([0], [0], color=c_soft, lw=1.9, label='softplus floor link (f = 0, τ = 0.59)'),
            Line2D([0], [0], color=c_leaky, lw=1.9, ls='--', label='leaky Tobit link (α = 0.10, clamp −1)'),
            Line2D([0], [0], color='#7f0000', marker='o', ms=3, label='empirical median & IQR'),
            Line2D([0], [0], color='#999999', ls=':', label='y = x (unsaturated)'),
        ],
        loc='upper left', fontsize=8.3, framealpha=0.94,
    )

    # Panel b: Induced saturation epistasis dddG vs additive dG across all doubles
    ylim_b = (-1.6, 4.9)
    ax_b.hexbin(x, Y - x, gridsize=40, extent=(*xlim, *ylim_b), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
    ax_b.axhline(0, color='#999999', ls=':', lw=1.0, zorder=3)
    if len(b_all):
        ax_b.fill_between(b_all['x'], b_all['q25'] - b_all['x'], b_all['q75'] - b_all['x'], color='#7f0000', alpha=0.18, lw=0, zorder=4)
        ax_b.plot(b_all['x'], b_all['median'] - b_all['x'], color='#7f0000', lw=1.4, marker='o', ms=2.6, zorder=5)
    ax_b.plot(xs_curve, softplus_link(xs_curve) - xs_curve, color=c_soft, lw=1.9, zorder=6)
    ax_b.plot(xs_curve, leaky_link(xs_curve) - xs_curve, color=c_leaky, lw=1.9, ls='--', zorder=6)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(*ylim_b)
    ax_b.set_xlabel('additive ΔG = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=10.2)
    ax_b.set_ylabel('measured epistasis δΔΔG = ΔG_AB − ΔG_add (kcal/mol)', fontsize=10.2)
    ax_b.set_title(f'Apparent saturation epistasis across all doubles (n = {len(Y):,})', fontsize=10.5, fontweight='bold', loc='left')
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=c_soft, lw=1.9, label='softplus: h(x) − x'),
            Line2D([0], [0], color=c_leaky, lw=1.9, ls='--', label='leaky: h(x) − x'),
            Line2D([0], [0], color='#7f0000', marker='o', ms=3, label='empirical median & IQR'),
        ],
        loc='upper right', fontsize=8.3, framealpha=0.94,
    )

    # Panel c: Stacked epistasis-type shares across latent additive dG bins
    r_lat, labels_bin, cat_lat, flag_lat = latent_types
    nb, nc = len(labels_bin), len(EPI_TYPES)
    xs_b = np.arange(nb)
    bottom = np.zeros(nb)
    for t in range(nc):
        rest = r_lat['share'][:, t] - r_lat['floor'][:, t]
        ax_c.bar(xs_b, r_lat['floor'][:, t], bottom=bottom, color=EPI_COLOURS[t], width=0.88, edgecolor=HATCH_EC[t], lw=0, hatch='////')
        bottom += r_lat['floor'][:, t]
        ax_c.bar(xs_b, rest, bottom=bottom, color=EPI_COLOURS[t], width=0.88, edgecolor='white', lw=0.4)
        bottom += rest
    ax_c.set_ylim(0, 1.0)
    ax_c.set_xticks(xs_b)
    ax_c.set_xticklabels(labels_bin, rotation=40, ha='right', fontsize=8.8)
    ax_c.set_xlabel('latent additive expected ΔG z (kcal/mol)', fontsize=10.2)
    ax_c.set_ylabel('fraction of doubles', fontsize=10.2)
    ax_c.set_title('Epistasis-type shares across latent ΔG_add (hatched: floor decides)', fontsize=10.5, fontweight='bold', loc='left')
    ov = np.bincount(cat_lat, minlength=nc) / len(cat_lat)
    h_c = [Patch(fc=EPI_COLOURS[t], ec='#999999', lw=0.4) for t in range(nc)] + [Patch(fc='white', ec='black', lw=0, hatch='////')]
    l_c = [f'{EPI_TYPES[t]} ({100 * ov[t]:.0f}%)' for t in range(nc)] + [f'floor decides ({100 * flag_lat.mean():.0f}%)']
    ax_c.legend(h_c, l_c, loc='lower left', fontsize=7.6, frameon=True, facecolor='white', framealpha=0.90, edgecolor='#cccccc')

    # Panel d: (ddG_A, ddG_B) singles-plane epistasis map (unsaturated coloured vs floor-dominated black)
    sa_all, sb_all = ddg[ia[ok]], ddg[ib[ok]]
    xa, yb, z_epi = np.minimum(sa_all, sb_all), np.maximum(sa_all, sb_all), sp['dddG_ML'].to_numpy(float)[ok]
    sat = x <= 0.0
    vmax = float(np.ceil(np.percentile(np.abs(z_epi[~sat]), 98) * 4) / 4)
    norm_d = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    limx, limy = np.percentile(xa, [0.3, 99.7]), np.percentile(yb, [0.3, 99.7])
    padx, pady = 0.05 * (limx[1] - limx[0]), 0.05 * (limy[1] - limy[0])
    ax_d.plot([limx[0] - padx, limy[1] + pady], [limx[0] - padx, limy[1] + pady], color='#888888', ls=':', lw=1.0, zorder=0)
    ax_d.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax_d.axvline(0, color='#cccccc', lw=0.8, zorder=0)
    ax_d.scatter(xa[sat], yb[sat], c='#1a1a1a', s=6, alpha=0.38, edgecolors='none', rasterized=True)
    o_cl = np.flatnonzero(~sat)
    o_cl = o_cl[np.argsort(np.abs(z_epi[o_cl]))]
    sc_d = ax_d.scatter(xa[o_cl], yb[o_cl], c=z_epi[o_cl], cmap='RdBu', norm=norm_d, s=8, alpha=0.82, edgecolors='none', rasterized=True)
    ax_d.set_xlim(limx[0] - padx, limx[1] + padx)
    ax_d.set_ylim(limy[0] - pady, limy[1] + pady)
    ax_d.set_xlabel('ΔΔG_A: more destabilizing single (kcal/mol)', fontsize=10.2)
    ax_d.set_ylabel('ΔΔG_B: second single (kcal/mol)', fontsize=10.2)
    ax_d.set_title(f'Singles-plane map (coloured: ΔG_add > 0, n = {int((~sat).sum()):,}; black: ≤ 0)', fontsize=10.5, fontweight='bold', loc='left')
    ax_d.legend(handles=[Line2D([0], [0], marker='o', ls='', color='#1a1a1a', alpha=0.6, label=f'ΔG_add ≤ 0: floor-dominated (n = {int(sat.sum()):,})')], loc='lower right', fontsize=8.0, framealpha=0.92)
    cb_d = fig.colorbar(sc_d, ax=ax_d, fraction=0.046, pad=0.03)
    cb_d.set_label('measured δΔΔG (kcal/mol)', fontsize=9.0)

    for ax_i, ltr in zip((ax_a, ax_b, ax_c, ax_d), ('a', 'b', 'c', 'd')):
        ax_i.text(-0.08, 1.07, ltr, transform=ax_i.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

    fig.suptitle(
        'Unfolded-state floor saturation in MegaScale cDNA proteolysis generates apparent positive epistasis unless accounted for via Tobit / residual modelling',
        fontsize=11.0, fontweight='bold', x=0.012, ha='left', y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    style.save(fig, out_supp / 'floor_saturation_mechanism')

    # Standalone 2-panel Tobit floor-link figure (a/b split out with default ESM-MSR Tobit mapping + explanatory caption)
    build_tobit_floor_link_figure(
        x=x, Y=Y, x_plot=x_plot, y_plot=y_plot,
        b_all=b_all, b_plot=b_plot, far_lbl=far_lbl,
        out_path=out_supp / 'tobit_floor_link',
    )

    if exploratory:
        fig_oa, axes_oa = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)
        for ax_o, (xx, yy, ttl_o, b_o) in zip(axes_oa, ((x_plot, y_plot, far_lbl, b_plot), (x, Y, f'all doubles (n = {len(Y):,})', b_all))):
            ax_o.hexbin(xx, yy, gridsize=40, extent=(*xlim, *ylim_a), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
            ax_o.plot(xlim, xlim, color='#999999', ls=':', lw=1.0)
            if len(b_o):
                ax_o.plot(b_o['x'], b_o['median'], color='#7f0000', lw=1.4, marker='o', ms=2.5)
            ax_o.plot(xs_curve, softplus_link(xs_curve), color=c_soft, lw=1.8)
            ax_o.plot(xs_curve, leaky_link(xs_curve), color=c_leaky, lw=1.8, ls='--')
            ax_o.set_xlim(*xlim)
            ax_o.set_ylim(*ylim_a)
            ax_o.set_title(ttl_o, loc='left', fontsize=9)
        fig_oa.tight_layout()
        style.save(fig_oa, out_epi / 'link_option_A_observed_additive')


# ----------------------------------------------------------------------------------------------
# 3. Pair-epistasis heatmaps & per-model dddG correlation tables/scatters (12 + 13)
# ----------------------------------------------------------------------------------------------
def _build_pair_models(fr: Frame) -> tuple[pd.DataFrame, np.ndarray, dict, dict, dict]:
    sp, p = fr.spine, fr.preds
    fit = epi_rows(fr)
    c = sp['dG_wt'].to_numpy(float)
    meas = {'raw': sp['dddG_ML'].to_numpy(float)}
    x_meas = c + sp['ddG_additive_ML'].to_numpy(float)
    raw_fit = np.where(fit, meas['raw'], np.nan)
    meas['saturation_corrected'] = (
        np.where(np.isfinite(meas['raw']), 0.0, np.nan)
        if (not np.isfinite(meas['raw'][fit]).any() or np.nanstd(meas['raw'][fit]) < 0.02)
        else meas['raw'] - smooth_on(x_meas, raw_fit)
    )
    model = {}
    for label, build_, addk, sk in HEATMAP_MODELS:
        need = [build_[1]] + ([build_[2]] if build_[0] == 'diff' else []) + [addk, sk]
        if any(k not in p for k in need):
            continue
        e_seeds = p[build_[1]] - (p[build_[2]] if build_[0] == 'diff' else 0.0)
        e_mean = np.nanmean(e_seeds, axis=0)
        model[label] = dict(raw=e_mean, raw_seeds=e_seeds, saturation_corrected=e_mean, single=np.nanmean(p[sk], axis=0))

    mut = sp['mut'].to_numpy(dtype=object)
    nm = sp['n_mut'].to_numpy()
    d = pd.DataFrame({'i': np.flatnonzero(nm == 2)})
    pp = [parse_mutations(str(mut[i]), sep=':') for i in d['i']]
    d['code_wt'] = sp['code_wt'].to_numpy()[d['i']]
    d['wt1'] = [t[0][0] if t and len(t) == 2 else None for t in pp]
    d['pos1'] = [t[0][1] if t and len(t) == 2 else None for t in pp]
    d['mt1'] = [t[0][2] if t and len(t) == 2 else None for t in pp]
    d['wt2'] = [t[1][0] if t and len(t) == 2 else None for t in pp]
    d['pos2'] = [t[1][1] if t and len(t) == 2 else None for t in pp]
    d['mt2'] = [t[1][2] if t and len(t) == 2 else None for t in pp]
    singles = {(cw, m): i for i, (cw, m) in enumerate(zip(sp['code_wt'].to_numpy(), mut)) if nm[i] == 1}
    return d, fit, meas, model, singles


def _pair_matrix(sub: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    M_mat = np.full((20, 20), np.nan)
    for r in sub.itertuples():
        if r.mt1 in PAIR_IDX and r.mt2 in PAIR_IDX:
            M_mat[PAIR_IDX[r.mt1], PAIR_IDX[r.mt2]] = values[r.i]
    return M_mat


def _pair_margin(code: str, wt: str, pos: int, singles: dict, vals: np.ndarray) -> np.ndarray:
    out = np.full(20, np.nan)
    for a in PAIR_AA:
        if a != wt and (code, f'{wt}{pos}{a}') in singles:
            out[PAIR_IDX[a]] = vals[singles[(code, f'{wt}{pos}{a}')]]
    return out


def _draw_one_pair_png(out_dir_hm, split: str, mode: str, pair: tuple, sub: pd.DataFrame, Mm: np.ndarray, top_m: np.ndarray, right_m: np.ndarray, panels: list, vmax: float) -> None:
    code, pos1, pos2 = pair
    wt1, wt2 = sub['wt1'].iloc[0], sub['wt2'].iloc[0]
    cells_ok = np.isfinite(Mm)
    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    for a_ in axes.ravel():
        a_.axis('off')
    im = None
    for k, (label, M_p, top_p, right_p, rho_p) in enumerate(panels):
        ax = axes.ravel()[k]
        ax.axis('on')
        ttl = ('Measured dddG' + (' − E[dddG | additive]' if mode == 'saturation_corrected' else '')) if k == 0 else f'{label}  (ρ = {rho_p:.2f})'
        cm = plt.get_cmap('RdBu').copy()
        cm.set_bad('white')
        im = ax.imshow(np.ma.masked_invalid(M_p), origin='lower', interpolation='nearest', cmap=cm, vmin=-vmax, vmax=vmax, aspect='equal')
        yy, xx = np.where(~np.isfinite(M_p))
        ax.scatter(xx, yy, marker='x', s=6, color='#8c8c8c', linewidths=0.55, zorder=3)
        ax.set_xticks(range(20))
        ax.set_xticklabels(PAIR_AA, fontsize=5.2)
        ax.set_yticks(range(20))
        ax.set_yticklabels(PAIR_AA, fontsize=5.2)
        ax.tick_params(length=1.5, pad=1)
        if k % 4 == 0:
            ax.set_xlabel(f'residue at {wt2}{pos2}', fontsize=7)
            ax.set_ylabel(f'residue at {wt1}{pos1}', fontsize=7)
        div = make_axes_locatable(ax)
        at = div.append_axes('top', size='16%', pad=0.06, sharex=ax)
        ar = div.append_axes('right', size='16%', pad=0.06, sharey=ax)
        at.set_title(ttl, fontsize=7.8, pad=5)
        at.bar(range(20), top_p, color='#4682b4', width=0.8)
        ar.barh(range(20), right_p, color='#4682b4', height=0.8)
        for a_m, o_m in ((at, 'x'), (ar, 'y')):
            a_m.tick_params(labelbottom=False, labelleft=False, labelsize=5, length=1.5)
            a_m.axhline(0, color='k', lw=0.4) if o_m == 'x' else a_m.axvline(0, color='k', lw=0.4)
        at.tick_params(axis='y', labelleft=True, labelsize=4.8)
        ar.tick_params(axis='x', labelbottom=True, labelsize=4.8)
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(-0.5, 19.5)
    info = axes.ravel()[len(panels)]
    info.axis('on')
    info.set_xticks([])
    info.set_yticks([])
    for s in info.spines.values():
        s.set_visible(False)
    info.text(
        0.0, 1.0,
        f'{code}  {wt1}{pos1} × {wt2}{pos2}\n\n{len(sub)} of 361 identity pairs, {int(cells_ok.sum())} usable',
        va='top', ha='left', fontsize=9, transform=info.transAxes,
    )
    if im is not None:
        cax = info.inset_axes([0.04, 0.10, 0.92, 0.045])
        fig.colorbar(im, cax=cax, orientation='horizontal').set_label(f'dddG (kcal/mol); clipped at ±{vmax:.2f}', fontsize=7.6)
    sp_lbl = 'mega_splits test' if split == 'mega' else 'Homology-Restricted MegaScale test'
    fig.suptitle(f'{code} {wt1}{pos1} × {wt2}{pos2}: pair epistasis dddG ({mode}, {sp_lbl})', fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.42, wspace=0.42, left=0.05, right=0.985, top=0.95, bottom=0.03)
    folder = out_dir_hm / mode / code
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{wt1}{pos1}_{wt2}{pos2}.png', dpi=85)
    plt.close(fig)


def run_pair_heatmaps_and_dddg_tables(split: str = 'capped', pair_pngs: bool = False, per_model_pngs: bool = False) -> None:
    out_hm = out_dir('pair_heatmaps', split=split)
    out_epi = out_dir('epistasis', split=split)
    fr = load_frame(tsu_datasets(split)[1])
    sp = fr.spine
    d, fit, meas, model, singles = _build_pair_models(fr)

    # 12: Pair epistasis agreement tables (+ optional 30 full-grid PNGs)
    g = d.drop_duplicates(['code_wt', 'pos1', 'pos2', 'mt1', 'mt2']).groupby(['code_wt', 'pos1', 'pos2']).size()
    pairs = [k for k, n in g.items() if n >= 361]
    agree, info = [], {}
    pair_rows = np.zeros(len(sp), bool)
    for pair in pairs:
        pair_rows[d.loc[(d['code_wt'] == pair[0]) & (d['pos1'] == pair[1]) & (d['pos2'] == pair[2]), 'i'].to_numpy()] = True
    y_s = np.where(sp['use'].to_numpy(), sp['y'].to_numpy(float), np.nan)

    for mode in ('raw', 'saturation_corrected'):
        v = meas[mode][pair_rows & fit]
        vmax = float(np.nanpercentile(np.abs(v[np.isfinite(v)]), 98)) if np.isfinite(v).any() else 2.0
        for pair in pairs:
            code, pos1, pos2 = pair
            sub = d[(d['code_wt'] == code) & (d['pos1'] == pos1) & (d['pos2'] == pos2)].drop_duplicates(['mt1', 'mt2'])
            wt1, wt2 = sub['wt1'].iloc[0], sub['wt2'].iloc[0]
            Mm = _pair_matrix(sub, np.where(fit, meas[mode], np.nan))
            cells_ok = np.isfinite(Mm)
            top_m = _pair_margin(code, wt2, pos2, singles, y_s)
            right_m = _pair_margin(code, wt1, pos1, singles, y_s)
            panels = [('Measured', Mm, top_m, right_m, np.nan)]
            for label, mv in model.items():
                Mp = _pair_matrix(sub, mv[mode])
                ok = cells_ok & np.isfinite(Mp)
                rho = float(spearmanr(Mm[ok], Mp[ok])[0]) if ok.sum() > 10 and np.ptp(Mp[ok]) > 0 else np.nan
                agree.append(dict(mode=mode, domain=code, pos1=pos1, pos2=pos2, model=label, spearman=rho, n_cells=int(ok.sum())))
                if pair_pngs:
                    ts = _pair_margin(code, wt2, pos2, singles, mv['single'])
                    rs = _pair_margin(code, wt1, pos1, singles, mv['single'])
                    panels.append((label, Mp, ts, rs, rho))
            info[pair] = dict(domain=code, pos1=pos1, pos2=pos2, wt1=wt1, wt2=wt2, identity_pairs=len(sub), usable_cells=int(cells_ok.sum()))
            if pair_pngs:
                _draw_one_pair_png(out_hm, split, mode, pair, sub, Mm, top_m, right_m, panels, vmax)

    atomic_to_csv(pd.DataFrame(list(info.values())), out_hm / 'pairs.csv', index=False)
    atomic_to_csv(pd.DataFrame(agree), out_hm / 'agreement.csv', index=False, float_format='%.4f')

    # 13: dddG_scatters/stats.csv, rosetta_unit_check.csv, and dG_additive_vs_pred_dddG/
    out_dddg = out_epi / 'dddG_scatters'
    out_dgadd = out_epi / 'dG_additive_vs_pred_dddG'
    out_dddg.mkdir(parents=True, exist_ok=True)
    out_dgadd.mkdir(parents=True, exist_ok=True)
    y_raw = meas['raw']
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)

    if split != 'mega' and 'rosetta' in fr.preds and 'rosetta+' in fr.preds:
        nm_u, use_u = sp['n_mut'].to_numpy(), sp['use'].to_numpy()
        y_u, d3_u = sp['y'].to_numpy(float), sp['dddG_ML'].to_numpy(float)
        ros_kcal, add_kcal = fr.preds['rosetta'][0], fr.preds['rosetta+'][0]
        ros_reu, add_reu = ros_kcal * 2.94, add_kcal * 2.94
        u_rows = []
        for what, a_reu, a_kcal, b_t, m_u in (
            ('dddG, doubles', ros_reu - add_reu, ros_kcal - add_kcal, d3_u, fit),
            ('ddG, singles', ros_reu, ros_kcal, y_u, use_u & (nm_u == 1)),
            ('ddG, doubles', ros_reu, ros_kcal, y_u, use_u & (nm_u == 2)),
        ):
            ok_u = m_u & np.isfinite(a_reu) & np.isfinite(b_t)
            k_opt = float((a_reu[ok_u] * b_t[ok_u]).sum() / (a_reu[ok_u] ** 2).sum())
            u_rows.append(dict(
                quantity=what, n=int(ok_u.sum()),
                rmse_reu=float(np.sqrt(np.mean((a_reu[ok_u] - b_t[ok_u]) ** 2))),
                rmse_div_2_94=float(np.sqrt(np.mean((a_kcal[ok_u] - b_t[ok_u]) ** 2))),
                best_scale_through_origin=k_opt,
                rmse_best_scale=float(np.sqrt(np.mean((k_opt * a_reu[ok_u] - b_t[ok_u]) ** 2))),
            ))
        atomic_to_csv(pd.DataFrame(u_rows), out_dddg / 'rosetta_unit_check.csv', index=False, float_format='%.4f')

    s_rows, dg_rows = [], []
    slug_fn = lambda lbl: re.sub(r'[^A-Za-z0-9-]+', '_', lbl.replace('*', '').replace('Δ', 'D')).strip('_')
    for label, mv in model.items():
        x_seeds = mv['raw_seeds']
        n_seeds = x_seeds.shape[0]
        for ver, (desc, cut) in (
            ('all_dG_additive', ('all doubles', None)),
            ('dG_additive_positive', ('doubles with ΔG additive > 0', 0.0)),
            ('dG_additive_gt1', ('doubles with ΔG additive > 1', 1.0)),
        ):
            ok = fit & np.isfinite(x_seeds).all(axis=0) & np.isfinite(y_raw) & (True if cut is None else (dG_add > cut))
            rho_v = float(np.mean([spearmanr(x_seeds[s, ok], y_raw[ok])[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(x_seeds[s, ok], y_raw[ok])[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((x_seeds[s, ok] - y_raw[ok]) ** 2)) for s in range(n_seeds)]))
            s_rows.append(dict(model=label, version=ver, prediction='latent' if label.startswith('ESM-MSR') else 'as is', n=int(ok.sum()), n_seeds=n_seeds, spearman=rho_v, pearson=r_v, rmse=rmse_v))
            if per_model_pngs:
                folder = out_dddg / ver
                folder.mkdir(parents=True, exist_ok=True)
                xs_in = x_seeds[:, ok] if n_seeds > 1 else x_seeds[0, ok]
                fig_pm = style.density_scatter_plain(xs_in, y_raw[ok], f'{label} predicted dddG (kcal/mol)', Y_DDD_LABEL, ref='xy', rmse=True, title=None, foot=f'{label}: {desc} (n = {int(ok.sum()):,})', **SCATTER_FONT)
                style.save(fig_pm, folder / slug_fn(label))

        ok_dg = fit & np.isfinite(dG_add) & np.isfinite(x_seeds).all(axis=0)
        ok_gt0 = ok_dg & (dG_add > 0)
        dg_rows.append(dict(
            model=label, n=int(ok_dg.sum()), n_seeds=n_seeds,
            spearman=float(np.mean([spearmanr(dG_add[ok_dg], x_seeds[s, ok_dg])[0] for s in range(n_seeds)])),
            pearson=float(np.mean([pearsonr(dG_add[ok_dg], x_seeds[s, ok_dg])[0] for s in range(n_seeds)])),
            spearman_dG_add_gt0=float(np.mean([spearmanr(dG_add[ok_gt0], x_seeds[s, ok_gt0])[0] for s in range(n_seeds)])),
            pearson_dG_add_gt0=float(np.mean([pearsonr(dG_add[ok_gt0], x_seeds[s, ok_gt0])[0] for s in range(n_seeds)])),
        ))
    atomic_to_csv(pd.DataFrame(s_rows), out_dddg / 'stats.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(dg_rows), out_dgadd / 'stats.csv', index=False, float_format='%.4f')

    # 2x4 subplots of all 8 main models (all doubles + dG_add > 0) and paired RMSE bar chart
    _main_models_dddg_grid(out_dddg, split, fit, y_raw, dG_add, model)
    _main_models_dddg_rmse_bars(out_dddg, split, fr, fit, dG_add)

    # 4-panel dG_additive vs predicted dddG side-by-side figure
    msr_names = [m for m in ('ESM-MSR', 'MutateEverything Stage 2', 'ThermoMPNN-D', 'SPURS-multi') if m in model]
    if len(msr_names) == 4:
        ok_c = fit & np.isfinite(dG_add)
        for mn in msr_names:
            ok_c &= np.isfinite(model[mn]['raw_seeds']).all(axis=0)
        xv_1d = dG_add[ok_c]
        limx = np.percentile(xv_1d, [0.3, 99.7])
        padx = 0.05 * (limx[1] - limx[0])
        xlim_c = (limx[0] - padx, limx[1] + padx)
        all_y = np.concatenate([model[mn]['raw_seeds'][:, ok_c].ravel() for mn in msr_names])
        limy = np.percentile(all_y, [0.3, 99.7])
        pady = 0.05 * (limy[1] - limy[0])
        ylim_c = (limy[0] - pady, limy[1] + pady)
        cm = plt.get_cmap('plasma')
        fig_c, axes_c = plt.subplots(1, 4, figsize=(24.0, 5.8), sharex=True, sharey=True)
        for idx, (ax, mn) in enumerate(zip(axes_c, msr_names)):
            y_ok = model[mn]['raw_seeds'][:, ok_c]
            ns = y_ok.shape[0]
            xv = np.tile(xv_1d, ns)
            yv = y_ok.ravel()
            sub = np.random.default_rng(0).choice(len(xv), min(len(xv), 15000), replace=False)
            z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
            o = np.argsort(z)
            col = cm(Normalize(z.min(), z.max())(z))
            col[:, 3] = 0.3 / np.sqrt(ns)
            ax.scatter(xv[o], yv[o], c=col[o], s=8, edgecolors='none', rasterized=True)
            ax.plot(xlim_c, [-xlim_c[0], -xlim_c[1]], color='#c0392b', ls='--', lw=1.6, zorder=4)
            ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
            ax.set_xlim(xlim_c)
            ax.set_ylim(ylim_c)
            ax.set_xlabel(X_DG_ADD_LABEL, fontsize=SCATTER_FONT['label_fontsize'])
            if idx == 0:
                ax.set_ylabel('predicted δΔΔG = dddG (kcal/mol)', fontsize=SCATTER_FONT['label_fontsize'])
            rho_v = float(np.mean([spearmanr(xv_1d, y_ok[s])[0] for s in range(ns)]))
            r_v = float(np.mean([pearsonr(xv_1d, y_ok[s])[0] for s in range(ns)]))
            ax.text(0.04, 0.96, f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {len(xv_1d):,} (×{ns} seeds)', transform=ax.transAxes, va='top', fontsize=SCATTER_FONT['stats_fontsize'], zorder=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94))
            ax.legend(handles=[Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='y = −x: dG_AB = 0'), Line2D([0], [0], color='#555555', ls=':', lw=1.2, label='y = 0: no epistasis')], loc='lower left', fontsize=SCATTER_FONT['legend_fontsize'], frameon=False)
            ax.set_title(mn, fontsize=13.5, fontweight='bold', pad=8)
        fig_c.tight_layout()
        style.save(fig_c, out_dgadd / 'esm_msr_models_dG_additive_vs_pred_dddG')


# ----------------------------------------------------------------------------------------------
# 4. Rosetta multi vs additive paired bootstrap (19_rosetta_multi_vs_additive.py, dynamic B fix)
# ----------------------------------------------------------------------------------------------
_ROS_FR: dict = {}
_ROS_SUB: dict = {}


def _ros_worker(job):
    key, unit, B_c, seed, ns = job
    fr, mask = _ROS_FR[(key[0], ns)], _ROS_SUB[key]
    kind = 'mutation' if unit == 'mutation' else 'protein'
    pt, rp = bootstrap_rho({key[0]: fr}, ['multi', 'additive'], kind=kind, B=B_c, seed=seed, per_protein=(unit == 'library mean'), common=True, subsets={key[0]: mask})
    return key, unit, ns, dict(rp[key[0]]), {m: (v[0], v[2], v[3]) for m, v in pt[key[0]].items()}


def run_rosetta_multi_vs_additive(B: int, workers: int = 4) -> None:
    out = out_dir('rosetta_multi_vs_additive')
    specs = [
        ('ptmul', 'all', lambda n: n >= 2, ('mutation', 'protein')),
        ('ptmul', 'doubles', lambda n: n == 2, ('mutation', 'protein')),
        ('ptmul', '3+ mutations', lambda n: n >= 3, ('mutation', 'protein')),
        ('ptmuld', 'doubles', lambda n: n == 2, ('mutation', 'protein')),
        ('tsuboyama_capped_test', 'doubles', lambda n: n == 2, ('mutation', 'protein', 'library mean')),
        ('tsuboyama_capped_val', 'doubles', lambda n: n == 2, ('mutation', 'protein', 'library mean')),
    ]
    lbl_map = {'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D', 'tsuboyama_capped_test': 'Tsuboyama test', 'tsuboyama_capped_val': 'Tsuboyama val'}

    _ROS_FR.clear()
    _ROS_SUB.clear()
    for ds in {s[0] for s in specs}:
        fr = load_frame(ds)
        nrep = fr.preds['rosetta'].shape[0]
        for ns in sorted({1, nrep}):
            _ROS_FR[(ds, ns)] = Frame(ds, fr.spine, {'multi': fr.preds['rosetta'][:ns], 'additive': fr.preds['rosetta+'][:ns]})

    jobs = []
    chunk_sz = 5000
    for ds, lab, fn_m, units in specs:
        _ROS_SUB[(ds, lab)] = fn_m(_ROS_FR[(ds, 1)].spine['n_mut'].to_numpy())
        for ns in sorted({n for (d, n) in _ROS_FR if d == ds}):
            for unit in units:
                jobs += [
                    ((ds, lab), unit, min(chunk_sz, B - i * chunk_sz), 31 * i + 7 * ns + len(unit), ns)
                    for i in range(int(np.ceil(B / chunk_sz)))
                ]

    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_ros_worker, jobs, chunksize=1)
    reps_acc: dict = {}
    pts: dict = {}
    for key, unit, ns, rp, pt in res:
        for m, v in rp.items():
            reps_acc.setdefault((key, unit, ns, m), []).append(v)
        pts[(key, unit, ns)] = pt

    rows = []
    for ds, lab, _fn_m, units in specs:
        for ns in sorted({n for (d, n) in _ROS_FR if d == ds}):
            for unit in units:
                key = (ds, lab)
                mu, ad = (np.concatenate(reps_acc[(key, unit, ns, m)]) for m in ('multi', 'additive'))
                ok = np.isfinite(mu) & np.isfinite(ad)
                mu, ad = mu[ok], ad[ok]
                d = mu - ad
                pt = pts[(key, unit, ns)]
                rows.append(dict(
                    dataset=ds, subset=lab, unit=unit, replicate='iteration 1' if ns == 1 else f'{ns} iterations',
                    n_rows=int(pt['multi'][1]), n_groups=int(pt['multi'][2]),
                    rho_multi=pt['multi'][0], rho_additive=pt['additive'][0],
                    delta=pt['multi'][0] - pt['additive'][0],
                    lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)),
                    multi_lo=float(np.quantile(mu, 0.025)), multi_hi=float(np.quantile(mu, 0.975)),
                    add_lo=float(np.quantile(ad, 0.025)), add_hi=float(np.quantile(ad, 0.975)),
                    p_one_sided=boot_pvalue(d, null=0.0, alternative='greater'),
                    p_two_sided=boot_pvalue(d, null=0.0, alternative='two-sided'),
                    B=len(d),
                ))
    R = pd.DataFrame(rows)
    atomic_to_csv(R, out / 'rosetta_multi_vs_additive.csv', index=False, float_format='%.6g')

    md_lines = [
        '| set | subset | unit | replicate | rows | rho multi | rho additive | Δρ (multi − additive) | 95% interval | p, multi better (one-sided) | p two-sided |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in R.itertuples():
        md_lines.append(
            f'| {lbl_map[r.dataset]} | {r.subset} | {r.unit} | {r.replicate} | {r.n_rows:,} | {r.rho_multi:.3f} | {r.rho_additive:.3f} | '
            f'{r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {fmt_p(r.p_one_sided, B=r.B, prefix="")} | {fmt_p(r.p_two_sided, B=r.B, prefix="")} |'
        )
    atomic_write_text(out / 'rosetta_multi_vs_additive.md', '\n'.join(md_lines) + '\n')

    use_plot = [
        ('ptmul', 'all', 'mutation', 'PTMUL, all'),
        ('ptmul', 'doubles', 'mutation', 'PTMUL, doubles'),
        ('ptmul', '3+ mutations', 'mutation', 'PTMUL, 3+ mutations'),
        ('ptmuld', 'doubles', 'mutation', 'PTMUL-D, doubles'),
        ('tsuboyama_capped_test', 'doubles', 'library mean', 'Tsuboyama test, doubles, per library'),
        ('tsuboyama_capped_val', 'doubles', 'library mean', 'Tsuboyama val, doubles, per library'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1.0]), sharey=True)
    ys = np.arange(len(use_plot))[::-1]
    b_mut_used = int(R[(R['dataset'] == 'ptmul') & (R['subset'] == 'all') & (R['unit'] == 'mutation') & (R['replicate'] == 'iteration 1')]['B'].iloc[0])
    b_lib_used = int(R[(R['dataset'] == 'tsuboyama_capped_test') & (R['subset'] == 'doubles') & (R['unit'] == 'library mean') & (R['replicate'] == 'iteration 1')]['B'].iloc[0])

    for y_i, (ds, sub, unit, _lab) in zip(ys, use_plot):
        r = R[(R['dataset'] == ds) & (R['subset'] == sub) & (R['unit'] == unit) & (R['replicate'] == 'iteration 1')].iloc[0]
        for off, v, lo, hi, filled in ((0.13, r.rho_multi, r.multi_lo, r.multi_hi, True), (-0.13, r.rho_additive, r.add_lo, r.add_hi, False)):
            axes[0].plot([lo, hi], [y_i + off] * 2, color='#8c564b', lw=1.5, alpha=0.6)
            axes[0].plot(v, y_i + off, marker='P', ms=7.5 * (1.0 if filled else 1.2), mfc='#8c564b' if filled else 'none', mec='#8c564b', mew=1.2 if filled else 2.0, ls='none')
        axes[1].plot([r.lo, r.hi], [y_i, y_i], color='#444444', lw=1.6)
        axes[1].plot(r.delta, y_i, marker='o', ms=6.5, color='#444444')
        axes[1].text(0.172, y_i, f'{r.delta:+.3f}   {fmt_p(r.p_one_sided, B=int(r.B), prefix="p = ")}', va='center', fontsize=8.2)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f'{u[3]} (n = {int(R[(R["dataset"] == u[0]) & (R["subset"] == u[1]) & (R["unit"] == u[2])]["n_rows"].iloc[0]):,})' for u in use_plot])
    axes[0].set_xlabel(r'Spearman $\rho$')
    axes[0].set_title('A  filled: multi-mutant scored directly; hollow: singles summed', loc='left', fontsize=9.5)
    axes[1].axvline(0, color='#999999', lw=0.8)
    axes[1].set_xlim(-0.06, 0.31)
    axes[1].set_xlabel(r'$\Delta\rho$ (multi $-$ additive)')
    axes[1].set_title('B  Difference (95% interval), one-sided p', loc='left', fontsize=9.5)
    for ax in axes:
        ax.grid(axis='x', color=style.GRID, lw=0.6)
    kw = dict(fontsize=7.8, color='#333333')
    note = (
        f'Rosetta Cartesian ddG, iteration 1. Both versions evaluated on identical rows and paired bootstrap draws. '
        f'PTMUL / PTMUL-D: pooled ρ, mutations resampled (B = {b_mut_used:,}); '
        f'Tsuboyama: mean of per-library ρ, libraries resampled (B = {b_lib_used:,}). One-sided p = P(multi − additive ≤ 0).'
    )
    t = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.96 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, foot + 0.05, 1, 1))
    style.save(fig, out / 'rosetta_multi_vs_additive')


def build_tobit_floor_link_figure(
    x: np.ndarray,
    Y: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    b_all: pd.DataFrame,
    b_plot: pd.DataFrame,
    far_lbl: str,
    out_path: Path,
    split_desc: str = 'Homology-Restricted MegaScale test',
) -> Path:
    """Standalone supplementary 2-panel figure splitting floor_saturation_mechanism (a/b)
    and highlighting ESM-MSR's default chosen Tobit observation link (sigma_y=0.28, mu_f=-0.14, sigma_f=0.50).
    """
    sigma_y = float(TOBIT['sigma_y'])
    mu_f = float(TOBIT['mu_f'])
    sigma_f = float(TOBIT['sigma_f'])
    theta = float(np.hypot(sigma_y, sigma_f))
    h_tobit = tobit_link(sigma_y=sigma_y, mu_f=mu_f, sigma_f=sigma_f)
    h_u0s1 = tobit_link(sigma_y=0.28, mu_f=0.0, sigma_f=1.0)
    h_deep = tobit_link(sigma_y=0.28, mu_f=-1.0, sigma_f=1.0)

    xs_curve = np.linspace(-4.8, 4.8, 400)
    y_tobit = h_tobit(xs_curve)
    y_u0s1 = h_u0s1(xs_curve)
    y_deep = h_deep(xs_curve)
    y_soft = softplus_link(xs_curve)

    xlim, ylim_a, ylim_b = (-4.8, 4.8), (-1.25, 5.0), (-1.6, 4.9)
    c_tobit, c_u0s1, c_deep, c_soft, c_emp = '#000000', '#009e73', '#e07b00', '#1f77b4', '#9b1c1c'

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.8, 6.4), facecolor='white')

    # --- Panel a: Distant pairs (Cbeta-Cbeta > 8 A) ---
    ax_a.hexbin(x_plot, y_plot, gridsize=40, extent=(*xlim, *ylim_a), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
    ax_a.plot(xlim, xlim, color='#888888', ls=':', lw=1.1, zorder=3)
    ax_a.axhline(mu_f, color='#666666', ls=(0, (2, 3)), lw=0.9, zorder=3)
    if len(b_plot):
        ax_a.fill_between(b_plot['x'], b_plot['q25'], b_plot['q75'], color=c_emp, alpha=0.18, lw=0, zorder=4)
        ax_a.plot(b_plot['x'], b_plot['median'], color=c_emp, lw=1.4, marker='o', ms=2.8, zorder=5)
    ax_a.plot(xs_curve, y_soft, color=c_soft, lw=1.4, ls='-.', zorder=6)
    ax_a.plot(xs_curve, y_u0s1, color=c_u0s1, lw=1.8, ls=(0, (4, 1.5, 1.5, 1.5)), zorder=6)
    ax_a.plot(xs_curve, y_deep, color=c_deep, lw=1.8, ls='--', zorder=6)
    ax_a.plot(xs_curve, y_tobit, color=c_tobit, lw=2.3, zorder=7)
    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(*ylim_a)
    ax_a.set_xlabel('additive ΔG_add = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=10.2)
    ax_a.set_ylabel('observed ΔG_AB of double (kcal/mol)', fontsize=10.2)
    ax_a.set_title(f'Observed ΔG_AB vs additive ΔG on {far_lbl}', fontsize=10.4, fontweight='bold', loc='left', pad=6)
    ax_a.legend(
        handles=[
            Line2D([0], [0], color=c_tobit, lw=2.3, label=f'default Tobit link h(z) (σ_y={sigma_y:.2f}, μ_f={mu_f:.2f}, σ_f={sigma_f:.2f})'),
            Line2D([0], [0], color=c_u0s1, lw=1.8, ls=(0, (4, 1.5, 1.5, 1.5)), label='wide-floor Tobit link (σ_y=0.28, μ_f=0.00, σ_f=1.00)'),
            Line2D([0], [0], color=c_deep, lw=1.8, ls='--', label='deep-floor Tobit link (σ_y=0.28, μ_f=−1.00, σ_f=1.00)'),
            Line2D([0], [0], color=c_soft, lw=1.4, ls='-.', label='softplus floor link (f = 0, τ = 0.59)'),
            Line2D([0], [0], color=c_emp, marker='o', ms=3.2, lw=1.4, label='empirical median & IQR'),
            Line2D([0], [0], color='#888888', ls=':', lw=1.1, label='y = x (unsaturated)'),
            Line2D([0], [0], color='#666666', ls=(0, (2, 3)), lw=0.9, label=f'default floor mean μ_f = {mu_f:.2f} kcal/mol'),
        ],
        loc='upper left', fontsize=7.8, framealpha=0.94, edgecolor='#cccccc',
    )
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # --- Panel b: Induced apparent saturation epistasis across all doubles ---
    ax_b.hexbin(x, Y - x, gridsize=40, extent=(*xlim, *ylim_b), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
    ax_b.axhline(0, color='#888888', ls=':', lw=1.1, zorder=3)
    if len(b_all):
        ax_b.fill_between(b_all['x'], b_all['q25'] - b_all['x'], b_all['q75'] - b_all['x'], color=c_emp, alpha=0.18, lw=0, zorder=4)
        ax_b.plot(b_all['x'], b_all['median'] - b_all['x'], color=c_emp, lw=1.4, marker='o', ms=2.8, zorder=5)
    ax_b.plot(xs_curve, y_soft - xs_curve, color=c_soft, lw=1.4, ls='-.', zorder=6)
    ax_b.plot(xs_curve, y_u0s1 - xs_curve, color=c_u0s1, lw=1.8, ls=(0, (4, 1.5, 1.5, 1.5)), zorder=6)
    ax_b.plot(xs_curve, y_deep - xs_curve, color=c_deep, lw=1.8, ls='--', zorder=6)
    ax_b.plot(xs_curve, y_tobit - xs_curve, color=c_tobit, lw=2.3, zorder=7)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(*ylim_b)
    ax_b.set_xlabel('additive ΔG_add = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=10.2)
    ax_b.set_ylabel('measured epistasis δΔΔG = ΔG_AB − ΔG_add (kcal/mol)', fontsize=10.2)
    ax_b.set_title(f'Apparent saturation epistasis across all doubles (n = {len(Y):,})', fontsize=10.4, fontweight='bold', loc='left', pad=6)
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=c_tobit, lw=2.3, label='default Tobit: h_Tobit(x) − x'),
            Line2D([0], [0], color=c_u0s1, lw=1.8, ls=(0, (4, 1.5, 1.5, 1.5)), label='wide-floor Tobit (μ_f=0, σ_f=1): h(x) − x'),
            Line2D([0], [0], color=c_deep, lw=1.8, ls='--', label='deep-floor Tobit (μ_f=−1, σ_f=1): h(x) − x'),
            Line2D([0], [0], color=c_soft, lw=1.4, ls='-.', label='softplus: h(x) − x'),
            Line2D([0], [0], color=c_emp, marker='o', ms=3.2, lw=1.4, label='empirical median & IQR'),
            Line2D([0], [0], color='#888888', ls=':', lw=1.1, label='δΔΔG = 0 (additive)'),
        ],
        loc='upper right', fontsize=7.8, framealpha=0.94, edgecolor='#cccccc',
    )
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Nature-style sub-figure annotations 'a' and 'b'
    fig.subplots_adjust(left=0.062, right=0.985, top=0.915, bottom=0.255, wspace=0.19)
    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.010, 0.975, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pb.x0 - 0.052, 0.975, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    # Self-contained explanatory figure text
    far_desc = far_lbl.replace('distant doubles ', '')
    note = (
        f'Unfolded-state floor saturation in MegaScale cDNA proteolysis and the default ESM-MSR Tobit observation link.  '
        f'a. Structurally distant double mutants {far_desc}, where direct residue–residue physical coupling is minimal, isolate the assay measurement floor: '
        f'observed ΔG_AB follows y = x for folded variants (ΔG_add > 1 kcal/mol) but flattens onto a noisy unfolded floor as ΔG_add drops below 0. '
        f'Under ESM-MSR\'s default noisy-floor Tobit likelihood Y = max(X, F) with X ~ N(z, σ_y²), F ~ N(μ_f, σ_f²) and chosen parameters '
        f'σ_y = {sigma_y:.2f}, μ_f = {mu_f:.2f}, σ_f = {sigma_f:.2f} kcal/mol (θ = √(σ_y² + σ_f²) = {theta:.3f}), the closed-form expected-reading map '
        f'h_Tobit(z) = z Φ(a) + μ_f Φ(−a) + θ φ(a) [a = (z − μ_f) / θ] (solid black) reproduces the empirical median (dark red) across the entire transition with no extra fitted parameters '
        f'(alternative Tobit floors μ_f = 0, σ_f = 1 in teal and μ_f = −1, σ_f = 1 in orange, plus analytical softplus in blue shown for comparison).  '
        f'b. Across all {len(Y):,} {split_desc} doubles, subtracting ΔG_add converts this measurement floor into large apparent positive epistasis '
        f'(δΔΔG ≈ −ΔG_add for ΔG_add < 0), closely tracked by h_Tobit(x) − x. Supervising ESM-MSR with the Tobit NLL trains the model in unsaturated latent space without conflating floor saturation with true biophysical coupling.'
    )
    kw = dict(fontsize=8.0, color='#333333', linespacing=1.28)
    fig.text(0.014, 0.014, style.wrap_to_px(fig, note, 0.972 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    style.save(fig, out_path)
    return out_path


def _render_tobit_link_for_arrays(
    x: np.ndarray,
    Y: np.ndarray,
    muts: np.ndarray,
    codes: np.ndarray,
    out_path: Path,
    split_desc: str,
) -> Path:
    dist = _cb_distances_for_doubles(muts, codes)
    far = np.isfinite(dist) & (dist > DISTANT_A)
    edges = np.arange(-4.5, 4.51, 0.5)
    b_all = _binned_quantiles(x, Y, edges, nmin=15)
    b_far = _binned_quantiles(x[far], Y[far], edges, nmin=10) if far.any() else pd.DataFrame(columns=['x', 'n', 'median', 'q25', 'q75'])
    x_plot = x[far] if far.sum() >= 50 else x
    y_plot = Y[far] if far.sum() >= 50 else Y
    b_plot = b_far if far.sum() >= 50 else b_all
    far_lbl = f'distant doubles (Cβ–Cβ > {DISTANT_A:g} Å, n = {len(y_plot):,})' if far.sum() >= 50 else f'all doubles (n = {len(y_plot):,})'
    return build_tobit_floor_link_figure(
        x=x, Y=Y, x_plot=x_plot, y_plot=y_plot,
        b_all=b_all, b_plot=b_plot, far_lbl=far_lbl,
        out_path=out_path, split_desc=split_desc,
    )


def _extract_train_doubles_capped() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract (x=dG_add, Y=dG_AB, muts, codes) for Homology-Restricted MegaScale training doubles with both singles measured."""
    import json
    import pickle
    from msr_analysis.paths import DATA_DIR, REFERENCE

    raw_csv = TSUBOYAMA_PDB_DIR.parent / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
    pkl_path = DATA_DIR / 'splits_oct06_capped.pkl'
    if not raw_csv.exists() or not pkl_path.exists():
        print(f'[p07] Skipping train tobit_floor_link: missing {raw_csv} or {pkl_path}')
        return None

    with open(pkl_path, 'rb') as fh:
        sp_pkl = pickle.load(fh)
    train_pdbs = set(sp_pkl['train'])
    train_wt_names = {p.replace('.pdb', '') for p in train_pdbs}

    dg_wt_map: dict[str, float] = {}
    dg_json_path = REFERENCE / 'dG_wt.json'
    if dg_json_path.exists():
        dg_wt_map = json.loads(dg_json_path.read_text(encoding='utf-8'))

    df = pd.read_csv(
        raw_csv,
        usecols=['WT_name', 'mut_type', 'dG_ML', 'ddG_ML'],
        low_memory=False,
    )
    wt_clean = df['WT_name'].astype(str).str.replace('.pdb', '', regex=False).str.replace('|', '_', regex=False)
    df = df[wt_clean.isin(train_wt_names)].copy()
    df['code'] = wt_clean[wt_clean.isin(train_wt_names)].to_numpy()
    df['ddG_ML'] = pd.to_numeric(df['ddG_ML'], errors='coerce')
    df['dG_ML'] = pd.to_numeric(df['dG_ML'], errors='coerce')
    df['mut_type'] = df['mut_type'].astype(str)
    # Keep substitutions only (exclude wt, ins, del)
    df = df[~df['mut_type'].str.contains('wt|ins|del', case=False, na=True)]
    df = df[np.isfinite(df['ddG_ML']) & np.isfinite(df['dG_ML'])].copy()
    # Deduplicate (code, mut_type) by mean
    df = df.groupby(['code', 'mut_type'], as_index=False)[['dG_ML', 'ddG_ML']].mean()
    # Wild-type dG per code
    wt_dg = df['code'].map(dg_wt_map)
    # Fallback: dG_ML - ddG_ML median per code
    fb_dg = (df['dG_ML'] - df['ddG_ML']).groupby(df['code']).median()
    df['dG_wt'] = np.where(np.isfinite(wt_dg), wt_dg, df['code'].map(fb_dg))

    n_mut = df['mut_type'].str.count(':') + 1
    singles = df[n_mut == 1]
    single_ddg = dict(zip(zip(singles['code'], singles['mut_type']), singles['ddG_ML']))

    doubles = df[n_mut == 2]
    xs, ys, muts, codes = [], [], [], []
    for r in doubles.itertuples(index=False):
        parts = r.mut_type.split(':')
        if len(parts) != 2:
            continue
        da = single_ddg.get((r.code, parts[0]))
        db = single_ddg.get((r.code, parts[1]))
        if da is None or db is None or not (np.isfinite(da) and np.isfinite(db) and np.isfinite(r.dG_wt)):
            continue
        xs.append(float(r.dG_wt + da + db))
        ys.append(float(r.dG_wt + r.ddG_ML))
        muts.append(r.mut_type)
        codes.append(r.code)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(muts, dtype=object),
        np.asarray(codes, dtype=object),
    )


def run_tobit_floor_link_only(split: str = 'capped') -> list[Path]:
    """Standalone runner to build 00_supplementary_figs/tobit_floor_link{,_val,_train}.{png,pdf}."""
    out_supp = out_dir('supplementary_figs')
    saved: list[Path] = []

    for ds_name, stem, desc in (
        (tsu_datasets(split)[1], 'tobit_floor_link', 'Homology-Restricted MegaScale test'),
        (tsu_datasets(split)[0], 'tobit_floor_link_val', 'Homology-Restricted MegaScale validation'),
    ):
        fr = load_frame(ds_name)
        sp = fr.spine
        ddg = sp['y'].to_numpy(float)
        c_all = sp['dG_wt'].to_numpy(float)
        ia, ib = singles_of_doubles(sp)
        ok = epi_rows(fr) & (ia >= 0) & (ib >= 0)
        idx = np.flatnonzero(ok)
        W = c_all[ok]
        A, B_s = W + ddg[ia[ok]], W + ddg[ib[ok]]
        Y = W + ddg[ok]
        x = A + B_s - W
        saved.append(_render_tobit_link_for_arrays(
            x=x, Y=Y,
            muts=sp['mut'].to_numpy(dtype=object)[idx],
            codes=sp['code'].astype(str).to_numpy()[idx],
            out_path=out_supp / stem,
            split_desc=desc,
        ))

    if split == 'capped':
        tr = _extract_train_doubles_capped()
        if tr is not None:
            x_tr, Y_tr, m_tr, c_tr = tr
            saved.append(_render_tobit_link_for_arrays(
                x=x_tr, Y=Y_tr, muts=m_tr, codes=c_tr,
                out_path=out_supp / 'tobit_floor_link_train',
                split_desc='Homology-Restricted MegaScale training',
            ))
    return saved


def build_supp_epistasis_mcc_figure(split: str = 'capped', B: int = 10_000) -> Path:
    """Build 2-panel supplementary epistasis classification MCC figure (a: raw dddG > 0; b: saturation-corrected r > 0).

    Uses canonical manuscript model order (epi_columns), threshold 0 on both measured and predicted values,
    pooled MCC over all usable doubles with both measured singles (E.Table, matching Figure 2e), and paired
    domain-cluster x seed bootstrap (B=10,000) for 95% CIs and one-sided ESM-MSR > X p-values.
    """
    from scipy.stats import pearsonr
    from esm_msr import epi_metrics as E
    from msr_analysis.boot import ci as boot_ci
    from msr_analysis.epistasis import make_epi_table
    from msr_analysis.stats import make_domain_seed_draws
    from ._tsu_common import BEST_LABEL, best_other, epi_columns, fig_width

    ds = tsu_datasets(split)[1]
    fr = load_frame(ds)
    sp = fr.spine
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols = [c for c in epi_columns(fr, best) if c.get('draw', True)]

    # Exact (singles_key, doubles_key) pairs matching Figure 2e / p06_epistasis_hierarchy.py
    key_pairs = {
        'ProteinMPNN': ('pmpnn+', 'pmpnn'),
        'ESM3-small-open': ('msrrel@0:wt', 'msrrel@0:comb'),
        'ESM-MSR': ('msrp0:wt', 'msrp0:comb'),
        'MutateEverything': ('me1', 'me2'),
        'ThermoMPNN': ('tmpnn', 'tmpnnd'),
        'SPURS': ('spurs', 'spursm'),
        'best other ddG': ('geoddg+', 'geoddg'),
        'Rosetta': ('rosetta+', 'rosetta'),
    }

    def _mcc_from_counts(tp, tn, fp, fn):
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(den > 0, num / den, 0.0)

    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    # Build E.Table for each model x seed, verify common domain set
    ref_tab = make_epi_table(sp, fr.preds['msrp0:wt'][0], fr.preds['msrp0:comb'][0], comb=fr.preds['msrp0:comb'][0], heads=('comb', 'add'))
    libs = np.unique(ref_tab.lib)
    n_libs = len(libs)
    lib_idx = {l: i for i, l in enumerate(libs)}
    dom_counts, seed_w = make_domain_seed_draws(n_libs, B=B, seed=2026, seed_counts=(3,))

    # Map E.Table lib ids back to domain names for per-domain source data
    use_idx = np.flatnonzero(sp['use'].to_numpy(dtype=bool))
    code_col = 'code' if 'code' in sp else 'code_wt'
    lib_names_map: dict[int, str] = {}
    sp_grp = sp['group'].astype(str).to_numpy()[use_idx]
    sp_code = sp[code_col].astype(str).to_numpy()[use_idx]
    # Reconstruct exact lib_id -> code_wt mapping in E.Table order
    code_to_grp = dict(zip(sp_code, sp_grp))
    # Use ref_tab.lib indices alongside spine codes
    sp_mut = sp['mut'].astype(str).to_numpy()[use_idx]
    from msr_analysis.epistasis import parse_mut_key
    keys_all = [parse_mut_key(m, c) for m, c in zip(sp_mut, sp_code)]
    st_all = ['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys_all]
    gt_all = sp['y'].to_numpy(float)[use_idx]
    dd_all = sp['dddG_ML'].to_numpy(float)[use_idx]
    c_all = sp['dG_wt'].to_numpy(float)[use_idx]
    singles_set = {tuple(k[0]) for i, k in enumerate(keys_all) if st_all[i] == 'single' and k is not None and len(k) == 1 and np.isfinite(gt_all[i])}
    seen_libs: dict[tuple, int] = {}
    for i, k in enumerate(keys_all):
        if st_all[i] != 'double' or k is None or len(k) != 2:
            continue
        if not (np.isfinite(dd_all[i]) and np.isfinite(gt_all[i]) and np.isfinite(c_all[i])):
            continue
        m1, m2 = tuple(k[0]), tuple(k[1])
        if m1[:-1] > m2[:-1]:
            m1, m2 = m2, m1
        if m1 not in singles_set or m2 not in singles_set:
            continue
        lk = tuple(k[0][:-2])
        if lk not in seen_libs:
            lid = len(seen_libs)
            seen_libs[lk] = lid
            lib_names_map[lid] = code_to_grp.get(str(lk[0]), str(lk[0]))

    summary_rows = []
    per_dom_rows = []
    reps_store: dict[tuple[str, str], np.ndarray] = {}
    pt_store: dict[tuple[str, str], float] = {}
    extra_stats: dict[tuple[str, str], tuple[float, float]] = {}  # (pearson_r, rmse)
    n_var_by_panel: dict[str, int] = {}

    for c in cols:
        cname = c['name']
        disp = name_map.get(cname, cname)
        kw, kd = key_pairs[cname]
        wt_arr = fr.preds[kw]
        db_arr = fr.preds[kd]
        ns = min(wt_arr.shape[0], db_arr.shape[0])

        for panel_key in ('raw', 'sat_corrected'):
            # Per-seed, per-domain 2x2 counts (ns, n_libs)
            tp_mat = np.zeros((ns, n_libs), dtype=float)
            tn_mat = np.zeros((ns, n_libs), dtype=float)
            fp_mat = np.zeros((ns, n_libs), dtype=float)
            fn_mat = np.zeros((ns, n_libs), dtype=float)
            seed_mccs, seed_accs, seed_f1s, seed_precs, seed_recs = [], [], [], [], []
            seed_rhos, seed_rs, seed_rmses = [], [], []
            n_ok_last = 0

            for s in range(ns):
                tab = make_epi_table(sp, wt_arr[s], db_arr[s], comb=db_arr[s], heads=('comb', 'add'))
                if panel_key == 'raw':
                    ym = tab.dddG
                    yp, _ = E.predicted_dddg(tab, 'comb')
                else:
                    ym, yp = E.beyond_global(tab, 'comb')
                ok = np.isfinite(ym) & np.isfinite(yp)
                n_ok_last = int(ok.sum())
                y_b = ym[ok] > 0.0
                p_b = yp[ok] > 0.0
                lib_ok = tab.lib[ok]

                for l_val in libs:
                    ml = (lib_ok == l_val)
                    li = lib_idx[l_val]
                    tp = float(np.sum(y_b[ml] & p_b[ml]))
                    tn = float(np.sum((~y_b[ml]) & (~p_b[ml])))
                    fp = float(np.sum((~y_b[ml]) & p_b[ml]))
                    fn = float(np.sum(y_b[ml] & (~p_b[ml])))
                    tp_mat[s, li], tn_mat[s, li], fp_mat[s, li], fn_mat[s, li] = tp, tn, fp, fn
                    per_dom_rows.append(dict(
                        panel=panel_key, model=disp, column=cname, domain=lib_names_map.get(int(l_val), str(l_val)),
                        seed=s + 1, n_variants=int(ml.sum()), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                        mcc=float(_mcc_from_counts(tp, tn, fp, fn)),
                    ))

                tp_t, tn_t, fp_t, fn_t = tp_mat[s].sum(), tn_mat[s].sum(), fp_mat[s].sum(), fn_mat[s].sum()
                seed_mccs.append(float(_mcc_from_counts(tp_t, tn_t, fp_t, fn_t)))
                seed_accs.append(float((tp_t + tn_t) / max(tp_t + tn_t + fp_t + fn_t, 1.0)))
                prec = float(tp_t / max(tp_t + fp_t, 1.0))
                rec = float(tp_t / max(tp_t + fn_t, 1.0))
                seed_precs.append(prec)
                seed_recs.append(rec)
                seed_f1s.append(float(2 * prec * rec / max(prec + rec, 1e-12)))
                seed_rhos.append(float(spearmanr(yp[ok], ym[ok])[0]) if np.ptp(yp[ok]) > 0 else np.nan)
                seed_rs.append(float(pearsonr(yp[ok], ym[ok])[0]) if np.ptp(yp[ok]) > 0 else np.nan)
                seed_rmses.append(float(np.sqrt(np.mean((yp[ok] - ym[ok]) ** 2))))

            # Exact vectorized domain-cluster x seed bootstrap: (B, ns)
            tp_b = dom_counts @ tp_mat.T
            tn_b = dom_counts @ tn_mat.T
            fp_b = dom_counts @ fp_mat.T
            fn_b = dom_counts @ fn_mat.T
            mcc_b_seed = _mcc_from_counts(tp_b, tn_b, fp_b, fn_b)
            if ns > 1:
                sw = seed_w[ns]
                reps = np.sum(mcc_b_seed * sw, axis=1)
            else:
                reps = mcc_b_seed[:, 0]

            pt = float(np.mean(seed_mccs))
            lo, hi = boot_ci(reps)
            reps_store[(panel_key, disp)] = reps
            pt_store[(panel_key, disp)] = pt
            extra_stats[(panel_key, disp)] = (float(np.nanmean(seed_rs)), float(np.nanmean(seed_rmses)))
            n_var_by_panel[panel_key] = n_ok_last

            summary_rows.append(dict(
                panel=panel_key,
                column=cname,
                model=disp,
                subset=panel_key,
                value=pt,
                lo=lo,
                hi=hi,
                accuracy=float(np.mean(seed_accs)),
                f1=float(np.mean(seed_f1s)),
                precision=float(np.mean(seed_precs)),
                recall=float(np.mean(seed_recs)),
                tp_mean=float(tp_mat.sum(axis=1).mean()),
                tn_mean=float(tn_mat.sum(axis=1).mean()),
                fp_mean=float(fp_mat.sum(axis=1).mean()),
                fn_mean=float(fn_mat.sum(axis=1).mean()),
                spearman_rho=float(np.nanmean(seed_rhos)),
                pearson_r=float(np.nanmean(seed_rs)),
                rmse=float(np.nanmean(seed_rmses)),
                n_variants=n_ok_last,
                n_libraries=n_libs,
                n_seeds=ns,
            ))

    df_sum = pd.DataFrame(summary_rows)
    df_raw = df_sum[df_sum['panel'] == 'raw'].copy()
    df_sat = df_sum[df_sum['panel'] == 'sat_corrected'].copy()

    # Two-panel stacked bar figure (a: raw dddG > 0; b: saturation-corrected r > 0), no overall title
    w_fig = max(11.4, fig_width(cols))
    fig = plt.figure(figsize=(w_fig, 9.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, left=0.080, right=0.985, top=0.950, bottom=0.125, hspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    lo_a = min(-0.08, float(np.floor((df_raw['lo'].min() - 0.04) * 20) / 20))
    hi_a = max(0.35, float(np.ceil((df_raw['hi'].max() + 0.11) * 20) / 20))
    lo_b = min(-0.08, float(np.floor((df_sat['lo'].min() - 0.04) * 20) / 20))
    hi_b = max(0.55, float(np.ceil((df_sat['hi'].max() + 0.11) * 20) / 20))

    style.bar_panel(
        ax_a, cols, df_raw, ('raw',), ('raw epistasis (δΔΔG > 0)',),
        ylabel='Epistasis sign MCC\n(raw δΔΔG > 0)',
        ylim=(lo_a, hi_a),
        zero_line=True, tick_fs=8.5,
    )
    style.bar_panel(
        ax_b, cols, df_sat, ('sat_corrected',), ('saturation-corrected (r > 0)',),
        ylabel='Epistasis sign MCC\n(saturation-corrected r > 0)',
        ylim=(lo_b, hi_b),
        zero_line=True, tick_fs=8.5,
    )
    for ax_s in (ax_a, ax_b):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.012, pa.y1 + 0.032, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(0.012, pb.y1 + 0.032, 'b', ha='left', va='top', fontsize=16, fontweight='bold')

    supp_dir = out_dir('supplementary_figs')
    out_path = supp_dir / '_epi_mcc'
    style.save(fig, out_path)

    # Export source data (gitignored *_sourcedata.csv) + paired one-sided bootstrap p-value table & annotation
    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(df_sum, src_dir / '_epi_mcc_summary_sourcedata.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(per_dom_rows), src_dir / '_epi_mcc_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    models_disp = [name_map.get(c['name'], c['name']) for c in cols]
    comps = [m for m in models_disp if m != 'ESM-MSR']
    m_clean = len(comps)
    p_rows = []
    for panel_key, metric_lbl in (
        ('raw', 'epistasis_sign_MCC_raw_dddG_gt0'),
        ('sat_corrected', 'epistasis_sign_MCC_saturation_corrected_r_gt0'),
    ):
        msr_pt = pt_store[(panel_key, 'ESM-MSR')]
        msr_rp = reps_store[(panel_key, 'ESM-MSR')]
        msr_r, msr_e = extra_stats[(panel_key, 'ESM-MSR')]
        n_v = n_var_by_panel[panel_key]
        p_by_c = {}
        d_list = []
        for comp in comps:
            d_arr = msr_rp - reps_store[(panel_key, comp)]
            p_one, nf, b_eff = _one_sided_p(d_arr)
            p_by_c[comp] = p_one
            d_list.append(d_arr)
            lo_d, hi_d = boot_ci(d_arr)
            cr, ce = extra_stats[(panel_key, comp)]
            p_rows.append(dict(
                benchmark='MegaScale_capped_test_epistasis',
                subset_or_metric=metric_lbl,
                resampling_unit='domain_cluster_x_seed',
                test_type='pairwise',
                msr_model='ESM-MSR',
                comparator=comp,
                direction='ESM-MSR > comparator',
                value_msr=msr_pt,
                value_comparator=pt_store[(panel_key, comp)],
                pearson_r_msr=msr_r,
                pearson_r_comparator=cr,
                rmse_msr=msr_e,
                rmse_comparator=ce,
                delta_favoring_msr=msr_pt - pt_store[(panel_key, comp)],
                ci95_lo=lo_d,
                ci95_hi=hi_d,
                n_units=n_libs,
                n_variants=n_v,
                B=b_eff,
                n_boot_fail=nf,
                p_one_sided_uncorrected=p_one,
                bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_one),
            ))
        best_c = max(comps, key=lambda x: pt_store[(panel_key, x)])
        d_nb = msr_rp - reps_store[(panel_key, best_c)]
        p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
        lo_nb, hi_nb = boot_ci(d_nb)
        cr, ce = extra_stats[(panel_key, best_c)]
        p_rows.append(dict(
            benchmark='MegaScale_capped_test_epistasis',
            subset_or_metric=metric_lbl,
            resampling_unit='domain_cluster_x_seed',
            test_type='next_best_clean',
            msr_model='ESM-MSR',
            comparator=best_c,
            direction='ESM-MSR > comparator',
            value_msr=msr_pt,
            value_comparator=pt_store[(panel_key, best_c)],
            pearson_r_msr=msr_r,
            pearson_r_comparator=cr,
            rmse_msr=msr_e,
            rmse_comparator=ce,
            delta_favoring_msr=msr_pt - pt_store[(panel_key, best_c)],
            ci95_lo=lo_nb,
            ci95_hi=hi_nb,
            n_units=n_libs,
            n_variants=n_v,
            B=b_nb,
            n_boot_fail=nf_nb,
            p_one_sided_uncorrected=p_nb,
            bonferroni_m=m_clean,
            p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
        ))
        worst_c = max(comps, key=lambda x: p_by_c[x])
        d_w = msr_rp - reps_store[(panel_key, worst_c)]
        p_iut, nf_w, b_w = _one_sided_p(d_w)
        lo_w, hi_w = boot_ci(d_w)
        cr, ce = extra_stats[(panel_key, worst_c)]
        p_rows.append(dict(
            benchmark='MegaScale_capped_test_epistasis',
            subset_or_metric=metric_lbl,
            resampling_unit='domain_cluster_x_seed',
            test_type='all_clean_IUT',
            msr_model='ESM-MSR',
            comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})',
            direction='ESM-MSR > comparator',
            value_msr=msr_pt,
            value_comparator=pt_store[(panel_key, worst_c)],
            pearson_r_msr=msr_r,
            pearson_r_comparator=cr,
            rmse_msr=msr_e,
            rmse_comparator=ce,
            delta_favoring_msr=msr_pt - pt_store[(panel_key, worst_c)],
            ci95_lo=lo_w,
            ci95_hi=hi_w,
            n_units=n_libs,
            n_variants=n_v,
            B=b_w,
            n_boot_fail=nf_w,
            p_one_sided_uncorrected=p_iut,
            bonferroni_m=1,
            p_one_sided_bonferroni=p_iut,
        ))
        d_min = np.nanmin(np.vstack(d_list), axis=0)
        p_jm, nf_jm, b_jm = _one_sided_p(d_min)
        lo_jm, hi_jm = boot_ci(d_min)
        p_rows.append(dict(
            benchmark='MegaScale_capped_test_epistasis',
            subset_or_metric=metric_lbl,
            resampling_unit='domain_cluster_x_seed',
            test_type='all_clean_joint_max',
            msr_model='ESM-MSR',
            comparator=f'ALL_CLEAN_JOINT_MAX (max of {m_clean} clean models per draw)',
            direction='ESM-MSR > comparator',
            value_msr=msr_pt,
            value_comparator=np.nan,
            pearson_r_msr=msr_r,
            pearson_r_comparator=np.nan,
            rmse_msr=msr_e,
            rmse_comparator=np.nan,
            delta_favoring_msr=msr_pt - max(pt_store[(panel_key, x)] for x in comps),
            ci95_lo=lo_jm,
            ci95_hi=hi_jm,
            n_units=n_libs,
            n_variants=n_v,
            B=b_jm,
            n_boot_fail=nf_jm,
            p_one_sided_uncorrected=p_jm,
            bonferroni_m=1,
            p_one_sided_bonferroni=p_jm,
        ))

    atomic_to_csv(pd.DataFrame(p_rows), pval_dir / '_epi_mcc_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_epi_mcc_pvalues_annotation.txt',
        f"""Statistical Annotation: _epi_mcc_pvalues.csv
============================================
Figure / Panel: Supplementary Figure (_epi_mcc), Panel a (raw epistasis sign classification, threshold = 0 on both measured dddG and predicted dddG) and Panel b (saturation-corrected epistasis sign classification, threshold = 0 on both measured residual r = dddG - E[dddG | dG_add] and predicted residual r' = dddG' - E[dddG' | dG_add']).

Metric Tested:
  Pooled Matthews Correlation Coefficient (MCC) across all {n_var_by_panel['raw']:,} usable Homology-Restricted MegaScale test doubles with both constituent singles measured ({n_libs} test domains; 3-seed average for retrained transfer models). ProSST is omitted because it is strictly additive (predicted dddG = 0 identically).
  Mean Pearson's r and RMSE (kcal/mol) on the same raw / saturation-corrected quantities are included alongside MCC.

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B:,}, deterministic seed).
  In each replicate, the {n_libs} test domains are resampled with replacement via a single shared Multinomial({n_libs}, 1/{n_libs}) weight vector applied to every model's per-domain 2x2 contingency counts (TP, TN, FP, FN), and 3-seed retrained models are resampled jointly across training seeds.
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction adjusts within each panel over m = {m_clean} clean comparator models.
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j); `all_clean_joint_max` evaluates min_j(MCC_ESM-MSR^(b) - MCC_j^(b)) per draw.
""",
    )
    return out_path


def _kde_density_2d(xv: np.ndarray, yv: np.ndarray, max_fit: int = 12_000, grid_n: int = 180) -> np.ndarray:
    """Fast, smooth 2D Gaussian KDE density via grid evaluation + bilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    n_pts = len(xv)
    rng = np.random.default_rng(0)
    sub = rng.choice(n_pts, min(n_pts, max_fit), replace=False)
    kde = gaussian_kde(np.vstack([xv[sub], yv[sub]]))
    if n_pts <= 20_000:
        return kde(np.vstack([xv, yv]))
    x_min, x_max = float(xv.min()), float(xv.max())
    y_min, y_max = float(yv.min()), float(yv.max())
    pad_x = 1e-6 * max(1.0, x_max - x_min)
    pad_y = 1e-6 * max(1.0, y_max - y_min)
    gx = np.linspace(x_min - pad_x, x_max + pad_x, grid_n)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, grid_n)
    XX, YY = np.meshgrid(gx, gy, indexing='ij')
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(grid_n, grid_n)
    interp = RegularGridInterpolator((gx, gy), Z, bounds_error=False, fill_value=float(Z.min()))
    return interp(np.column_stack([xv, yv]))


_SCATTER_BOOT_STATE: dict = {}


def _scatter_spearman_chunk(b_slice: tuple[int, int]) -> np.ndarray:
    b0, b1 = b_slice
    st = _SCATTER_BOOT_STATE
    d_cnt, dom_idx, eng, xr_flat = st['d_cnt'], st['dom_idx'], st['eng'], st['xr_flat']
    out_c = np.empty((b1 - b0, len(xr_flat)), dtype=float)
    for bi, b in enumerate(range(b0, b1)):
        wb = d_cnt[b, dom_idx].astype(float)
        ry_b = eng.ry.ranks(wb)
        sw_b = float(wb.sum())
        my_b = float(np.dot(wb, ry_b) / sw_b)
        vy_b = float(np.dot(wb, ry_b * ry_b) / sw_b - my_b * my_b)
        for j, xr_j in enumerate(xr_flat):
            rx_b = xr_j.ranks(wb)
            mx_b = float(np.dot(wb, rx_b) / sw_b)
            vx_b = float(np.dot(wb, rx_b * rx_b) / sw_b - mx_b * mx_b)
            cxy_b = float(np.dot(wb, rx_b * ry_b) / sw_b - mx_b * my_b)
            den_b = np.sqrt(max(vx_b * vy_b, 0.0))
            out_c[bi, j] = (cxy_b / den_b) if den_b > 0.0 else np.nan
    return out_c


def _pooled_domain_boot_r_rmse(
    p_seeds: np.ndarray,
    y_1d: np.ndarray,
    dom_idx: np.ndarray,
    n_dom: int,
    d_cnt: np.ndarray,
    s_w: np.ndarray | None,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Exact domain-cluster x seed bootstrap for pooled Pearson r and pooled RMSE."""
    ns = p_seeds.shape[0]
    B = d_cnt.shape[0]
    nd = np.bincount(dom_idx, minlength=n_dom).astype(float)
    sy = np.bincount(dom_idx, weights=y_1d, minlength=n_dom)
    sy2 = np.bincount(dom_idx, weights=y_1d ** 2, minlength=n_dom)

    r_seed_reps = np.empty((B, ns), dtype=float)
    e_seed_reps = np.empty((B, ns), dtype=float)
    r_pts, e_pts = [], []
    W_n = d_cnt @ nd
    W_sy = d_cnt @ sy
    W_sy2 = d_cnt @ sy2
    for s in range(ns):
        ps = p_seeds[s]
        r_pts.append(float(pearsonr(ps, y_1d)[0]))
        e_pts.append(float(np.sqrt(np.mean((ps - y_1d) ** 2))))
        sx = np.bincount(dom_idx, weights=ps, minlength=n_dom)
        sx2 = np.bincount(dom_idx, weights=ps ** 2, minlength=n_dom)
        sxy = np.bincount(dom_idx, weights=ps * y_1d, minlength=n_dom)
        sse = np.bincount(dom_idx, weights=(ps - y_1d) ** 2, minlength=n_dom)
        W_sx = d_cnt @ sx
        W_sx2 = d_cnt @ sx2
        W_sxy = d_cnt @ sxy
        W_sse = d_cnt @ sse
        cov = W_sxy - (W_sx * W_sy) / W_n
        vx = np.maximum(W_sx2 - (W_sx ** 2) / W_n, 0.0)
        vy = np.maximum(W_sy2 - (W_sy ** 2) / W_n, 0.0)
        denom = np.sqrt(vx * vy)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_seed_reps[:, s] = np.where(denom > 0, cov / denom, np.nan)
            e_seed_reps[:, s] = np.sqrt(W_sse / W_n)
    if ns == 1 or s_w is None:
        return float(r_pts[0]), r_seed_reps[:, 0], float(e_pts[0]), e_seed_reps[:, 0]
    return (
        float(np.mean(r_pts)),
        np.sum(r_seed_reps * s_w, axis=1),
        float(np.mean(e_pts)),
        np.sum(e_seed_reps * s_w, axis=1),
    )


def build_supp_megascale_scatter_trio(split: str = 'capped', B: int = 10_000, compute_pvals: bool = True) -> list[Path]:
    """Build 3 style-matched supplementary scatter figures on MegaScale test:
      1. _megascale_singles_scatters.{png,pdf}   (9 main singles models, 3x3, shared y-axis)
      2. _megascale_doubles_scatters.{png,pdf}   (9 main doubles models, 3x3, shared y-axis)
      3. _megascale_epistasis_scatters.{png,pdf} (8 main epistasis models, 2x4, shared y-axis)
    Matches _epi_rmse styling: plasma KDE density coloring, x=y line, 3-seed integration without
    ensemble averaging, and seed-mean rho / r / RMSE / n summary boxes.
    """
    from msr_analysis.boot import SpearmanEngine, ci as boot_ci
    from msr_analysis.stats import make_domain_seed_draws

    fr = load_frame(tsu_datasets(split)[1])
    sp = fr.spine
    use = sp['use'].to_numpy()
    y_ddg = sp['y'].to_numpy(float)
    y_epi = sp['dddG_ML'].to_numpy(float)
    n_mut = sp['n_mut'].to_numpy()
    groups_all = sp['group'].astype(str).to_numpy()

    configs = [
        dict(
            stem='_megascale_singles_scatters',
            benchmark='MegaScale_capped_test_singles',
            subset_desc='Homology-Restricted MegaScale test single mutants' if split == 'capped' else 'MegaScale mega_splits test single mutants',
            mask=use & (n_mut == 1) & np.isfinite(y_ddg),
            y_arr=y_ddg,
            ylabel='measured ΔΔG (kcal/mol)',
            default_xlabel='predicted ΔΔG (kcal/mol)',
            ncols=3,
            pt_size=5.0,
            alpha_base=0.22,
            leg_loc='lower right',
            leg_bbox=(0.985, 0.13),
            msr_name='ESM-MSR WT-view',
            models=[
                ('ProSST', 'prosst', 'Zero-shot', 'predicted score (log-likelihood)'),
                ('ProteinMPNN', 'pmpnn', 'Zero-shot', 'predicted score (log-odds)'),
                ('ESM3-small-open WT-view', 'msrrel@0:wt', 'Zero-shot', 'predicted ΔΔG (kcal/mol)'),
                ('ESM-MSR WT-view', 'msrp0:wt', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('MutateEverything Stage 1', 'me1', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('ThermoMPNN', 'tmpnn', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('SPURS', 'spurs', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('GeoDDG-3D', 'geoddg', 'Supervised', 'predicted ΔΔG (kcal/mol)'),
                ('Rosetta', 'rosetta', 'Supervised', 'predicted ΔΔG (kcal/mol)'),
            ],
            extra_note='ProSST and ProteinMPNN output uncalibrated log-likelihood/log-odds scores rather than kcal/mol; all panels share the measured y-axis while x-axes span each model\'s prediction range.',
        ),
        dict(
            stem='_megascale_doubles_scatters',
            benchmark='MegaScale_capped_test_doubles',
            subset_desc='Homology-Restricted MegaScale test double mutants' if split == 'capped' else 'MegaScale mega_splits test double mutants',
            mask=use & (n_mut == 2) & np.isfinite(y_ddg),
            y_arr=y_ddg,
            ylabel='measured ΔΔG (kcal/mol)',
            default_xlabel='predicted ΔΔG (kcal/mol)',
            ncols=3,
            pt_size=6.5,
            alpha_base=0.28,
            leg_loc='lower right',
            leg_bbox=(0.985, 0.13),
            msr_name='ESM-MSR Dual-view',
            models=[
                ('ProSST (additive)', 'prosst', 'Zero-shot', 'predicted score (log-likelihood)'),
                ('ProteinMPNN', 'pmpnn', 'Zero-shot', 'predicted score (log-odds)'),
                ('ESM3-small-open Dual-view', 'msrrel@0:comb', 'Zero-shot', 'predicted ΔΔG (kcal/mol)'),
                ('ESM-MSR Dual-view', 'msrp0:comb', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('MutateEverything Stage 2', 'me2', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('ThermoMPNN-D', 'tmpnnd', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('SPURS-multi', 'spursm', 'Retrained', 'predicted ΔΔG (kcal/mol)'),
                ('GeoDDG-3D', 'geoddg', 'Supervised', 'predicted ΔΔG (kcal/mol)'),
                ('Rosetta', 'rosetta', 'Supervised', 'predicted ΔΔG (kcal/mol)'),
            ],
            extra_note='ProSST (additive sum of singles) and ProteinMPNN output uncalibrated log-likelihood/log-odds scores rather than kcal/mol; all panels share the measured y-axis while x-axes span each model\'s prediction range.',
        ),
        dict(
            stem='_megascale_epistasis_scatters',
            benchmark='MegaScale_capped_test_epistasis',
            subset_desc='Homology-Restricted MegaScale test doubles with both singles measured (all epistasis)' if split == 'capped' else 'MegaScale mega_splits test doubles with both singles measured (all epistasis)',
            mask=epi_rows(fr) & np.isfinite(y_epi),
            y_arr=y_epi,
            ylabel=Y_DDD_LABEL,
            default_xlabel='predicted δΔΔG = dddG (kcal/mol)',
            ncols=4,
            pt_size=6.5,
            alpha_base=0.30,
            leg_loc='lower left',
            leg_bbox=None,
            msr_name='ESM-MSR',
            models=[
                ('ProteinMPNN', 'pmpnn:epi', 'Zero-shot', 'predicted δΔΔG (log-odds)'),
                ('ESM3-small-open', 'msrrel@0:epi', 'Zero-shot', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('ESM-MSR', 'msrp0:epi', 'Retrained', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('MutateEverything Stage 2', 'me2:epi', 'Retrained', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('ThermoMPNN-D', 'tmpnnd:epi', 'Retrained', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('SPURS-multi', 'spursm:epi', 'Retrained', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('GeoDDG-3D', 'geoddg:epi', 'Supervised', 'predicted δΔΔG = dddG (kcal/mol)'),
                ('Rosetta', 'rosetta:epi', 'Supervised', 'predicted δΔΔG = dddG (kcal/mol)'),
            ],
            extra_note='ProSST is omitted because it is strictly additive (predicted δΔΔG = 0 identically). All panels share the measured y-axis while x-axes span each model\'s prediction range.',
        ),
    ]

    supp_dir = out_dir('supplementary_figs')
    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)
    cm = plt.get_cmap('plasma')
    saved_paths: list[Path] = []

    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    for cfg_i, cfg in enumerate(configs):
        entries = [
            (lbl, k, grp, xlab, fr.preds[k])
            for lbl, k, grp, xlab in cfg['models']
            if k in fr.preds
        ]
        ok = cfg['mask'].copy()
        for _, _, _, _, p_arr in entries:
            ok &= np.isfinite(p_arr).all(axis=0)
        yv_1d = cfg['y_arr'][ok]
        dom_arr = groups_all[ok]
        uniq_doms, dom_idx = np.unique(dom_arr, return_inverse=True)
        n_dom = len(uniq_doms)
        n_var = len(yv_1d)

        # Shared y-axis limits from measured y (0.2–99.8 pct + 5% padding)
        y_lo, y_hi = np.percentile(yv_1d, [0.2, 99.8])
        y_pad = 0.055 * (y_hi - y_lo)
        ylim = (float(y_lo - y_pad), float(y_hi + y_pad))

        ncols = cfg['ncols']
        nrows = int(np.ceil(len(entries) / ncols))
        fig_w = 19.2 if ncols == 4 else 14.8
        fig_h = 11.4 if nrows == 2 else 15.8
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            sharex=False, sharey=True, squeeze=False,
            facecolor='white',
        )
        flat_axes = axes.ravel()

        sum_rows = []
        seed_rows = []
        preds_ok_map: dict[str, np.ndarray] = {}

        for idx, (ax, (m_label, k, grp, xlab, p_full)) in enumerate(zip(flat_axes, entries)):
            p_ok = p_full[:, ok]
            preds_ok_map[m_label] = p_ok
            n_seeds = p_ok.shape[0]
            xv = p_ok.ravel()
            yv = np.tile(yv_1d, n_seeds)

            # Per-panel x-limits covering at least [ylim[0], ylim[1]] and [0.3, 99.7] pct of predictions
            x_q0, x_q1 = np.percentile(xv, [0.3, 99.7])
            x_pad = 0.055 * max(x_q1 - x_q0, 1e-3)
            xlim = (float(min(x_q0 - x_pad, ylim[0])), float(max(x_q1 + x_pad, ylim[1])))

            z = _kde_density_2d(xv, yv)
            o = np.argsort(z)
            col = cm(Normalize(float(z.min()), float(z.max()))(z))
            col[:, 3] = cfg['alpha_base'] / np.sqrt(n_seeds)

            ax.axhline(0.0, color='#bbbbbb', lw=0.8, zorder=1)
            ax.axvline(0.0, color='#bbbbbb', lw=0.8, zorder=1)
            ax.scatter(xv[o], yv[o], c=col[o], s=cfg['pt_size'], edgecolors='none', rasterized=True, zorder=2)
            diag_lo, diag_hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
            ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color='#c0392b', ls='--', lw=1.5, zorder=4, clip_on=True)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_box_aspect(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            row_i, col_i = divmod(idx, ncols)
            if col_i == 0:
                ax.set_ylabel(cfg['ylabel'], fontsize=11.5)
            ax.set_xlabel(xlab, fontsize=10.2)
            ax.tick_params(axis='both', labelsize=9.8)

            rhos = [float(spearmanr(p_ok[s], yv_1d)[0]) for s in range(n_seeds)]
            rs = [float(pearsonr(p_ok[s], yv_1d)[0]) for s in range(n_seeds)]
            rmses = [float(np.sqrt(np.mean((p_ok[s] - yv_1d) ** 2))) for s in range(n_seeds)]
            rho_m = float(np.mean(rhos))
            r_m = float(np.mean(rs))
            rmse_m = float(np.mean(rmses))
            rho_sd = float(np.std(rhos, ddof=1)) if n_seeds > 1 else np.nan
            r_sd = float(np.std(rs, ddof=1)) if n_seeds > 1 else np.nan
            rmse_sd = float(np.std(rmses, ddof=1)) if n_seeds > 1 else np.nan

            sum_rows.append(dict(
                model=m_label, group=grp, key=k, n_seeds=n_seeds, n_variants=n_var, n_domains=n_dom,
                spearman_rho=rho_m, spearman_rho_sd=rho_sd,
                pearson_r=r_m, pearson_r_sd=r_sd,
                rmse=rmse_m, rmse_sd=rmse_sd,
            ))
            for s in range(n_seeds):
                seed_rows.append(dict(
                    model=m_label, group=grp, key=k, seed=s + 1, n_variants=n_var, n_domains=n_dom,
                    spearman_rho=rhos[s], pearson_r=rs[s], rmse=rmses[s],
                ))

            n_str = f'n = {n_var:,} (×{n_seeds})' if n_seeds > 1 else f'n = {n_var:,}'
            ax.text(
                0.04, 0.96, f'ρ = {rho_m:.2f}\nr = {r_m:.2f}\nRMSE = {rmse_m:.2f}\n{n_str}',
                transform=ax.transAxes, va='top', ha='left', fontsize=9.8, zorder=10,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#cccccc', alpha=0.92),
            )
            ax.text(
                0.96, 0.04, grp,
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9.2, color='#555555', zorder=10,
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.78),
            )
            if idx == 0:
                leg_kw = dict(handles=[Line2D([0], [0], color='#c0392b', ls='--', lw=1.5, label='x = y')],
                              loc=cfg['leg_loc'], fontsize=9.2, framealpha=0.92, edgecolor='#cccccc')
                if cfg['leg_bbox'] is not None:
                    leg_kw['bbox_to_anchor'] = cfg['leg_bbox']
                ax.legend(**leg_kw)
            ax.set_title(m_label, fontsize=12.2, fontweight='bold', pad=7)

        for ax in flat_axes[len(entries):]:
            ax.set_visible(False)

        if nrows == 2:
            fig.subplots_adjust(left=0.048, right=0.992, top=0.962, bottom=0.108, hspace=0.28, wspace=0.11)
        else:
            fig.subplots_adjust(left=0.058, right=0.992, top=0.974, bottom=0.074, hspace=0.26, wspace=0.13)
        foot_txt = (
            f'{cfg["subset_desc"]} (n = {n_var:,}, {n_dom} domains). '
            f'Retrained models show 3 triplicate seeds integrated without ensemble averaging; stats are seed means. '
            f'{cfg["extra_note"]}'
        )
        kw_f = dict(fontsize=9.0, color='#444444')
        fig.text(0.012, 0.008, style.wrap_to_px(fig, foot_txt, 0.975 * fig.bbox.width, **kw_f), va='bottom', ha='left', **kw_f)

        out_path = supp_dir / cfg['stem']
        style.save(fig, out_path)
        saved_paths.append(out_path)

        # Save source data (summary + per-seed)
        atomic_to_csv(pd.DataFrame(sum_rows), src_dir / f'{cfg["stem"]}_summary_sourcedata.csv', index=False, float_format='%.6g')
        atomic_to_csv(pd.DataFrame(seed_rows), src_dir / f'{cfg["stem"]}_per_seed_sourcedata.csv', index=False, float_format='%.6g')

        if not compute_pvals:
            continue

        # Paired one-sided domain-cluster x seed bootstrap p-values (ESM-MSR vs all comparators for pooled rho, r, RMSE)
        d_cnt, s_w = make_domain_seed_draws(n_dom, B=B, seed=701 + cfg_i, seed_counts=(3,))
        # Exact count-weighted pooled Spearman bootstrap via SpearmanEngine (shared domain weights per draw)
        g_zero = np.zeros(n_var, dtype=np.int64)
        eng = SpearmanEngine(yv_1d, g_zero, 1)
        m_list = list(preds_ok_map.keys())
        xr_flat = []
        m_seed_slices = {}
        cur_idx = 0
        for m_label in m_list:
            p_ok = preds_ok_map[m_label]
            ns = p_ok.shape[0]
            m_seed_slices[m_label] = (cur_idx, cur_idx + ns)
            for s in range(ns):
                xr_flat.append(eng.prepare(p_ok[s]))
            cur_idx += ns

        n_workers = min(8, max(1, mp.cpu_count() or 1))
        chunk_sz = max(1, (B + n_workers * 2 - 1) // (n_workers * 2))
        chunks = [(i, min(B, i + chunk_sz)) for i in range(0, B, chunk_sz)]
        _SCATTER_BOOT_STATE.clear()
        _SCATTER_BOOT_STATE.update(d_cnt=d_cnt, dom_idx=dom_idx, eng=eng, xr_flat=xr_flat)
        try:
            if n_workers > 1 and B > 100:
                with mp.get_context('fork').Pool(n_workers) as pool:
                    rho_mat = np.vstack(pool.map(_scatter_spearman_chunk, chunks))
            else:
                rho_mat = np.vstack([_scatter_spearman_chunk(ch) for ch in chunks])
        finally:
            _SCATTER_BOOT_STATE.clear()

        rho_pt_map, rho_rep_map = {}, {}
        r_pt_map, r_rep_map = {}, {}
        e_pt_map, e_rep_map = {}, {}
        for m_label in m_list:
            p_ok = preds_ok_map[m_label]
            ns = p_ok.shape[0]
            sw_ns = s_w[ns] if (isinstance(s_w, dict) and ns in s_w) else (s_w if not isinstance(s_w, dict) else None)
            j0, j1 = m_seed_slices[m_label]
            s_rhos = rho_mat[:, j0:j1]
            s_pts = [float(spearmanr(p_ok[s], yv_1d)[0]) for s in range(ns)]
            rho_pt_map[m_label] = float(np.mean(s_pts))
            rho_rep_map[m_label] = s_rhos[:, 0] if (ns == 1 or sw_ns is None) else np.sum(s_rhos * sw_ns, axis=1)
            r_pt, r_rp, e_pt, e_rp = _pooled_domain_boot_r_rmse(p_ok, yv_1d, dom_idx, n_dom, d_cnt, sw_ns if ns > 1 else None)
            r_pt_map[m_label], r_rep_map[m_label] = r_pt, r_rp
            e_pt_map[m_label], e_rep_map[m_label] = e_pt, e_rp

        msr_nm = cfg['msr_name']
        comps = [m for m in preds_ok_map if m != msr_nm]
        m_clean = len(comps)
        p_rows = []
        for met_name, pt_m, rp_m, higher_better in (
            ('pooled_spearman_rho', rho_pt_map, rho_rep_map, True),
            ('pooled_pearson_r', r_pt_map, r_rep_map, True),
            ('pooled_rmse', e_pt_map, e_rep_map, False),
        ):
            sign = 1.0 if higher_better else -1.0
            dir_str = 'ESM-MSR > comparator' if higher_better else 'ESM-MSR < comparator (lower RMSE better)'
            p_by_c = {}
            for comp in comps:
                d_arr = sign * (rp_m[msr_nm] - rp_m[comp])
                p_one, nf, b_eff = _one_sided_p(d_arr)
                p_by_c[comp] = p_one
                lo_d, hi_d = boot_ci(d_arr)
                p_rows.append(dict(
                    benchmark=cfg['benchmark'], metric=met_name,
                    resampling_unit='domain_cluster_x_seed', test_type='pairwise',
                    msr_model=msr_nm, comparator=comp, direction=dir_str,
                    value_msr=pt_m[msr_nm], value_comparator=pt_m[comp],
                    delta_favoring_msr=sign * (pt_m[msr_nm] - pt_m[comp]),
                    ci95_lo=lo_d, ci95_hi=hi_d,
                    n_units=n_dom, n_variants=n_var, B=b_eff, n_boot_fail=nf,
                    p_one_sided_uncorrected=p_one, bonferroni_m=m_clean,
                    p_one_sided_bonferroni=min(1.0, m_clean * p_one),
                ))
            best_c = (max(comps, key=lambda c: pt_m[c]) if higher_better else min(comps, key=lambda c: pt_m[c]))
            d_nb = sign * (rp_m[msr_nm] - rp_m[best_c])
            p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
            lo_nb, hi_nb = boot_ci(d_nb)
            p_rows.append(dict(
                benchmark=cfg['benchmark'], metric=met_name,
                resampling_unit='domain_cluster_x_seed', test_type='next_best_clean',
                msr_model=msr_nm, comparator=best_c, direction=dir_str,
                value_msr=pt_m[msr_nm], value_comparator=pt_m[best_c],
                delta_favoring_msr=sign * (pt_m[msr_nm] - pt_m[best_c]),
                ci95_lo=lo_nb, ci95_hi=hi_nb,
                n_units=n_dom, n_variants=n_var, B=b_nb, n_boot_fail=nf_nb,
                p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
            ))
            worst_c = max(comps, key=lambda c: p_by_c[c])
            d_w = sign * (rp_m[msr_nm] - rp_m[worst_c])
            p_iut, nf_w, b_w = _one_sided_p(d_w)
            lo_w, hi_w = boot_ci(d_w)
            p_rows.append(dict(
                benchmark=cfg['benchmark'], metric=met_name,
                resampling_unit='domain_cluster_x_seed', test_type='all_clean_IUT',
                msr_model=msr_nm, comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})', direction=dir_str,
                value_msr=pt_m[msr_nm], value_comparator=pt_m[worst_c],
                delta_favoring_msr=sign * (pt_m[msr_nm] - pt_m[worst_c]),
                ci95_lo=lo_w, ci95_hi=hi_w,
                n_units=n_dom, n_variants=n_var, B=b_w, n_boot_fail=nf_w,
                p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
            ))

        atomic_to_csv(pd.DataFrame(p_rows), pval_dir / f'{cfg["stem"]}_pvalues.csv', index=False, float_format='%.6g')
        atomic_write_text(
            pval_dir / f'{cfg["stem"]}_pvalues_annotation.txt',
            f"""Statistical Annotation: {cfg['stem']}_pvalues.csv
==================================================
Figure: Supplementary Figure ({cfg['stem']}) — {cfg['subset_desc']} (n = {n_var:,}, {n_dom} domains).

Metrics Tested:
  Pooled Spearman's rho, pooled Pearson's r, and pooled RMSE (kcal/mol) across all {n_var:,} variants, matching the seed-mean summary statistics shown in each scatter panel.

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B:,}, deterministic seed).
  All {n_dom} test domains are resampled identically across models via shared Multinomial({n_dom}, 1/{n_dom}) cluster weights, jointly with Multinomial(3, 1/3) seed weights for 3-seed retrained models.
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta_favoring_msr^(b) <= 0}}) / (B + 1).
  Bonferroni correction adjusts within each metric over all m = {m_clean} clean comparator models.
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j).
""",
        )
    return saved_paths


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION, with_split=True)
    ap.add_argument('--exploratory', action='store_true', help='Also render exploratory multi-option drafts (option_*, link_option_*)')
    ap.add_argument('--pair-pngs', action='store_true', help='Also render all 30 individual 20x20 pair-epistasis PNG grids')
    ap.add_argument('--per-model-pngs', action='store_true', help='Also render all 36 individual per-model dddG scatter PNGs')
    ap.add_argument('--workers', type=int, default=4, help='Worker processes for Rosetta bootstrap')
    ap.add_argument('--only', choices=('all', 'fig6', 'fig7', 'link', 'tobit_floor_link', 'heatmaps', 'rosetta', 'epi_mcc', 'scatter_trio'), default='all', help='Optional substage selector')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    if args.only == 'scatter_trio':
        outs = build_supp_megascale_scatter_trio(split=args.split, B=max(B, 10_000))
        print('[p07:epistasis_mechanisms] Saved scatter trio:', ', '.join(str(o) for o in outs))
        return 0

    if args.only == 'epi_mcc':
        out = build_supp_epistasis_mcc_figure(split=args.split, B=max(B, 10_000))
        print(f'[p07:epistasis_mechanisms] Saved {out}.{{png,pdf}}')
        return 0

    if args.only == 'tobit_floor_link':
        out = run_tobit_floor_link_only(split=args.split)
        print(f'[p07:epistasis_mechanisms] Saved {out}.{{png,pdf}}')
        return 0

    if args.only in ('all', 'fig6'):
        run_fig6_scatters(split=args.split, exploratory=args.exploratory)
    if args.only in ('all', 'fig7', 'link'):
        latent_types = run_fig7_epistasis_types(B=B, split=args.split)
        if args.only in ('all', 'link'):
            run_link_and_floor_composite(split=args.split, latent_types=latent_types, exploratory=args.exploratory)
    if args.only in ('all', 'heatmaps'):
        run_pair_heatmaps_and_dddg_tables(split=args.split, pair_pngs=args.pair_pngs, per_model_pngs=args.per_model_pngs)
    if args.only in ('all', 'rosetta') and args.split == 'capped':
        run_rosetta_multi_vs_additive(B=B, workers=args.workers)
    return 0


if __name__ == '__main__':
    sys.exit(main())
