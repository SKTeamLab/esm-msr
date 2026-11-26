#!/usr/bin/env python
"""Unified ProteinGym benchmark, deployed σ ∈ {0, 0.5, 1} property summary, and ours-vs-published zero-shot pipeline.

Replaces ``05_proteingym.py``, ``23_pg_ours_vs_published.py``, and ``proteingym/p0_sigma/{01..05}`` on top of
``msr_analysis.proteingym`` with:
  - Canonical **173 held-out assays** (``Activity=38, Binding=13, Expression=18, OrganismalFitness=67, Stability=37``)
    from the 213 scored assays (excluding the 4 ``PG_DROP`` ultra-large assays and 40 training-homologous assays under
    the four-part split rule), eliminating the 173-vs-176 mismatch between Figure 3b and Figure 3c.
  - Joint ``[category-stratified assay × triplicate-seed]`` bootstrap across all multi-seed models (including our 3
    ``ProteinMPNN`` runs).
  - Dynamic held-out and unfiltered assay counts on every axis, legend, title, and footnote (zero stale ``187/213`` or
    ``40/66`` strings).
  - Single canonical ``pg_summary.csv``, ``pg_sigma_summary.csv``, and ``property_delta_pvalues.csv`` in
    ``results/03_proteingym/`` (eliminating byte-duplicate ``*_tsu.csv`` and ``*_nonleaked.csv`` tables), plus
    ``pg_ours_vs_published.{csv,pdf,png}`` in ``results/00_supplementary_figs/``.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, rankdata, spearmanr

from msr_analysis import models as M
from msr_analysis import proteingym as PG
from msr_analysis import screening as SCR
from msr_analysis import style
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import EXT_ORDER, register_ext_styles, seed_frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.heldout import excluded
from msr_analysis.paths import (
    CACHE,
    ESM3_S0_PGYM_COMB,
    FINAL_BENCH,
    FINAL_PGYM_WT,
    FINAL_PGYM_WT_S05,
    MSR,
    PGYM,
    PGYM_REF_CSV,
    PUBLIC,
    REFERENCE,
    atomic_pickle_dump,
    atomic_to_csv,
    atomic_write_text,
    results_dir,
)

STAGE = 'proteingym'
DESCRIPTION = 'ProteinGym 5-category benchmarks, σ ∈ {0, 0.5, 1} property summaries, and ours-vs-published zero-shot validation (173 held-out assays).'

MULTI_PROP_ANNOT = {
    'CP2C9_HUMAN_Amorosi_2021_abundance': ('CP2C9_HUMAN', 'ABUND', 'VAMP-seq abundance'),
    'CP2C9_HUMAN_Amorosi_2021_activity': ('CP2C9_HUMAN', 'ACT', 'probe-binding activity'),
    'HXK4_HUMAN_Gersing_2023_abundance': ('HXK4_HUMAN', 'ABUND', 'DHFR-PCA abundance'),
    'HXK4_HUMAN_Gersing_2022_activity': ('HXK4_HUMAN', 'ACT', 'complementation activity'),
    'KCNE1_HUMAN_Muhammad_2023_expression': ('KCNE1_HUMAN', 'ABUND', 'surface expression'),
    'KCNE1_HUMAN_Muhammad_2023_function': ('KCNE1_HUMAN', 'ACT', 'channel function'),
    'KCNJ2_MOUSE_Coyote-Maestas_2022_surface': ('KCNJ2_MOUSE', 'ABUND', 'surface trafficking'),
    'KCNJ2_MOUSE_Coyote-Maestas_2022_function': ('KCNJ2_MOUSE', 'ACT', 'ion conduction'),
    'OXDA_RHOTO_Vanella_2023_expression': ('OXDA_RHOTO', 'ABUND', 'surface expression'),
    'OXDA_RHOTO_Vanella_2023_activity': ('OXDA_RHOTO', 'ACT', 'enzyme activity'),
    'PTEN_HUMAN_Matreyek_2021': ('PTEN_HUMAN', 'ABUND', 'VAMP-seq abundance'),
    'PTEN_HUMAN_Mighell_2018': ('PTEN_HUMAN', 'ACT', 'lipid phosphatase (yeast growth)'),
    'Q53Z42_HUMAN_McShan_2019_expression': ('Q53Z42_HUMAN', 'ABUND', 'surface expression'),
    'Q53Z42_HUMAN_McShan_2019_binding-TAPBPR': ('Q53Z42_HUMAN', 'BIND', 'TAPBPR binding'),
    'RASK_HUMAN_Weng_2022_abundance': ('RASK_HUMAN', 'ABUND', 'AbundancePCA'),
    'RASK_HUMAN_Weng_2022_binding-DARPin_K55': ('RASK_HUMAN', 'BIND', 'DARPin K55 binding'),
    'S22A1_HUMAN_Yee_2023_abundance': ('S22A1_HUMAN', 'ABUND', 'split-GFP abundance'),
    'S22A1_HUMAN_Yee_2023_activity': ('S22A1_HUMAN', 'ACT', 'substrate uptake (survival)'),
    'SPIKE_SARS2_Starr_2020_expression': ('SPIKE_SARS2', 'ABUND', 'RBD expression'),
    'SPIKE_SARS2_Starr_2020_binding': ('SPIKE_SARS2', 'BIND', 'ACE2 binding'),
    'VKOR1_HUMAN_Chiasson_2020_abundance': ('VKOR1_HUMAN', 'ABUND', 'eGFP abundance'),
    'VKOR1_HUMAN_Chiasson_2020_activity': ('VKOR1_HUMAN', 'ACT', 'carboxylation activity'),
    'RL40A_YEAST_Roscoe_2013': ('RL40A_YEAST', 'FIT', 'yeast growth'),
    'RL40A_YEAST_Mavor_2016': ('RL40A_YEAST', 'FIT', 'yeast growth (stress DMSO)'),
    'RL40A_YEAST_Roscoe_2014': ('RL40A_YEAST', 'ACT', 'E1 reactivity'),
    'SRC_HUMAN_Ahler_2019': ('SRC_HUMAN', 'ACT', 'kinase activity (yeast toxicity)'),
    'SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM': ('SRC_HUMAN', 'SPEC', 'dasatinib resistance'),
    'SRC_HUMAN_Nguyen_2022': ('SRC_HUMAN', 'SPEC', 'Hsp90 dependence'),
}
MULTI_PROP_EXTRA = (
    ('DLG4_HUMAN', 'DLG4_HUMAN_Faure_2021', 'ABUND', 'abundance PCA', 'DLG4_HUMAN_Faure_2021_binding_domain'),
    ('GRB2_HUMAN', 'GRB2_HUMAN_Faure_2021', 'ABUND', 'abundance PCA', 'GRB2_HUMAN_Faure_2021_binding_domain'),
)
AXIS_ORDER = {'ABUND': 0, 'BIND': 1, 'ACT': 2, 'FIT': 3, 'SPEC': 4}
AXIS_NAME = {
    'ABUND': 'abundance / expression',
    'ACT': 'activity / function',
    'BIND': 'binding',
    'FIT': 'growth / fitness',
    'SPEC': 'drug resistance / Hsp90 dependence',
}
AXIS_COLOR = {'ABUND': '#1b6ca8', 'ACT': '#d1495b', 'BIND': '#edae49', 'FIT': '#66a182', 'SPEC': '#8d6fb5'}

WORST_ASSAYS = [
    'KCNE1_HUMAN_Muhammad_2023_expression', 'KCNE1_HUMAN_Muhammad_2023_function',
    'RPC1_LAMBD_Li_2019_low-expression', 'RPC1_LAMBD_Li_2019_high-expression',
    'B2L11_HUMAN_Dutta_2010_binding-Mcl-1', 'TADBP_HUMAN_Bolognesi_2019',
    'IF1_ECOLI_Kelsic_2016', 'RNC_ECOLI_Weeks_2023', 'DLG4_RAT_McLaughlin_2012', 'Q837P4_ENTFA_Meier_2023',
]
KD_HYDROPATHY = dict(
    A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
    L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2,
)
KS_ACT = [1, 3, 5, 10, 20, 50, 100]


# ----------------------------------------------------------------------------------------------
# Domainome held-out summary (paired into pg_summary.csv & pg_sigma_summary.csv for Figure 3b)
# ----------------------------------------------------------------------------------------------
def _domainome_summary(B: int = 1000, refresh: bool = False) -> dict[str, tuple[float, float, float, int]]:
    cache = CACHE / 'pg_domainome_tsu_boot.pkl'
    dom_hom = excluded('domainome')
    n_keep = 522 - len(dom_hom)
    if cache.exists() and not refresh:
        with open(cache, 'rb') as fh:
            cached = pickle.load(fh)
        if (
            isinstance(cached, dict)
            and cached.get('B') == B
            and cached.get('n_domains') == n_keep
            and set(cached.get('summary', {})) >= (set(EXT_ORDER) - {'Rosetta'}) | {'ESM-MSR σ=0.5'}
        ):
            return cached['summary']

    raw_dom = load_frame('domainome', heldout=False)
    keep = ~raw_dom.spine['group'].astype(str).isin(dom_hom).to_numpy()
    dom_held = Frame(
        'domainome',
        raw_dom.spine.loc[keep].reset_index(drop=True),
        {k: v[:, keep] for k, v in raw_dom.preds.items()},
    )
    dom_fr, _ = seed_frame('domainome', dom_held)
    s05 = load_final_frame('domainome', families=['FINAL_p0'], sigma='0.5', seeds=(1, 2, 3), heldout=False)
    assert (s05.spine['id'].to_numpy() == raw_dom.spine['id'].to_numpy()).all()
    dom_fr.preds['ESM-MSR σ=0.5'] = s05.preds['FINAL_p0:wt'][:, keep]
    pt, rp = bootstrap_rho({'domainome': dom_fr}, list(dom_fr.preds), kind='protein', B=B, seed=30, per_protein=True)
    out = {}
    for m, (rho_v, _ps, _n, ng) in pt['domainome'].items():
        lo, hi = ci(rp['domainome'][m])
        out[m] = (float(rho_v), float(lo), float(hi), int(ng))
    atomic_pickle_dump(dict(B=B, n_domains=n_keep, summary=out), cache)
    return out


# ----------------------------------------------------------------------------------------------
# Main ProteinGym 5-category figures (03_proteingym) with dynamic 173/213 counts
# ----------------------------------------------------------------------------------------------
def _draw_main_pgym_figures(
    out: Path,
    T_213: pd.DataFrame,
    Tf_173: pd.DataFrame,
    R_full_173: pd.DataFrame,
    full_models: list[str],
    B: int,
) -> None:
    pg_cols = list(PG.PG_CATEGORIES_ALPHA) + ['PG avg', 'DMS mean']
    counts_173 = {c: int((Tf_173['category'] == c).sum()) for c in PG.PG_CATEGORIES_ALPHA}
    counts_213 = {c: int((T_213['category'] == c).sum()) for c in PG.PG_CATEGORIES_ALPHA}
    n_173, n_213 = len(Tf_173), len(T_213)
    n_removed = n_213 - n_173
    ros_ov = {'Rosetta': {'label': 'Rosetta (singles)'}}

    cat_hdr = {}
    ov_titles = {}
    cat_leg = {}
    for c in PG.PG_CATEGORIES_ALPHA:
        short = PG.PG_CATEGORY_SHORT[c]
        c_k, c_u = counts_173[c], counts_213[c]
        mark = '*' if c_k < c_u else ''
        cnt = f'({c_k}/{c_u})' if c_k < c_u else f'({c_k})'
        cat_hdr[c] = f'{short}{mark} {cnt}'
        ov_titles[c] = f'{short}{mark} ({c_k}/{c_u} assays)' if c_k < c_u else f'{short} ({c_k} assays)'
        cat_leg[c] = f'{short}{mark} {cnt}'
    cat_hdr['PG avg'] = 'PG avg'
    cat_hdr['DMS mean'] = f'DMS mean* ({n_173}/{n_213})'

    # 1. Ranked table heatmap across all models
    Z = R_full_173.pivot(index='model', columns='column', values='value')[pg_cols]
    Z = Z.loc[M.ordered(Z.index)]
    fig, ax = plt.subplots(figsize=(10.2, 0.33 * len(Z) + 2.1))
    rk = Z.rank(pct=True)
    ax.imshow(rk.to_numpy(), cmap='YlGnBu', vmin=0, vmax=1.15, aspect='auto')
    for i in range(len(Z)):
        for j in range(len(pg_cols)):
            val = Z.iloc[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.6,
                    color='white' if rk.iloc[i, j] > 0.75 else 'black',
                )
    ax.set_xticks(range(len(pg_cols)))
    ax.set_xticklabels([cat_hdr[c] for c in pg_cols], rotation=28, ha='right', fontsize=8.7)
    ax.set_yticks(range(len(Z)))
    ax.set_yticklabels([style.style_of(m, overrides=ros_ov.get(m)).label for m in Z.index], fontsize=8.4)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    style.finish_figure(
        fig,
        out / 'pg_table_heatmap',
        title=f'ProteinGym held-out summary ({n_173} of {n_213} assays; seeds 1–3 averaged; colour = column percentile rank)',
        note=(
            f'* Four-part training-homology filter applied across all categories ({n_removed} homologous assays excluded; '
            f'4 large BRCA1/BRCA2/POLG/SCN5A assays omitted for uniform baseline coverage). '
            'Singles version on singles assays, Dual-view / Stage 2 / -D / -multi on multi-mutant assays. '
            'Models marked * (ESMtherm*) trained on unsplit MegaScale libraries.'
        ),
        note_fontsize=7.3,
    )

    # 2. Horizontal 5-category forest overview
    P_full = {(r.column, r.model): (r.value, r.lo, r.hi) for r in R_full_173.itertuples()}
    ov_keys = M.ordered(full_models)
    mega_pg = set(PG.SUPPLEMENT_PG)
    fig, axes = plt.subplots(1, len(PG.PG_CATEGORIES_ALPHA), figsize=(17.5, 0.32 * len(ov_keys) + 2.1), sharey=True)
    for ax, cat in zip(axes, PG.PG_CATEGORIES_ALPHA):
        panel_vals = [P_full[(cat, k)] for k in ov_keys if (cat, k) in P_full and np.isfinite(P_full[(cat, k)][0])]
        min_lo = min(lo for _v, lo, _hi in panel_vals)
        max_hi = max(hi for _v, _lo, hi in panel_vals)
        x_lo = float(np.floor((min_lo - 0.03) * 20) / 20)
        span = max(0.25, max_hi - x_lo)
        x_hi = max_hi + 0.24 * span
        for i, k in enumerate(ov_keys):
            if k in mega_pg:
                ax.axhspan(i - 0.5, i + 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
            v, lo, hi = P_full[(cat, k)]
            if not np.isfinite(v):
                continue
            kw = style.marker_kwargs(k, size=6.0, overrides=ros_ov.get(k))
            ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.4, alpha=0.45, solid_capstyle='butt')
            ax.plot(v, i, **kw)
            if cat == 'Stability' and k in PG.PG_STAB_LEAK:
                ax.plot(v, i, marker='o', ms=10.5, mfc='none', mec=style.RING, mew=1.0, ls='none')
            ax.text(hi + 0.022 * span, i, f'{v:.3f}', va='center', fontsize=7.3, color='#444444')
        ax.set_title(ov_titles[cat], loc='left', fontsize=9.5)
        ax.grid(axis='x', color=style.GRID, lw=0.6)
        ax.set_xlabel('mean per-assay ρ', fontsize=8.8)
        ax.set_xlim(x_lo, x_hi)

    axes[0].set_yticks(range(len(ov_keys)))
    specs = [style.style_of(k, overrides=ros_ov.get(k)) for k in ov_keys]
    axes[0].set_yticklabels([s.label for s in specs], fontsize=8.3)
    axes[0].set_ylim(len(ov_keys) - 0.4, -0.6)
    group_ink = {'msr': 'black', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555555'}
    for tick_lbl, s in zip(axes[0].get_yticklabels(), specs):
        tick_lbl.set_color(group_ink.get(s.group, '#333333'))
        if s.group == 'msr':
            tick_lbl.set_fontweight('bold')
    style.finish_figure(
        fig,
        out / 'pg_overview',
        title=f'ProteinGym per-category overview ({n_173}/{n_213} held-out assays; B = {B:,} joint assay × seed bootstrap)',
        note=(
            f'* Four-part training-homology exclusions remove {n_removed} assays across categories '
            f'(Stability keeps {counts_173["Stability"]}/{counts_213["Stability"]}, '
            f'Activity {counts_173["Activity"]}/{counts_213["Activity"]}, '
            f'Org. fitness {counts_173["OrganismalFitness"]}/{counts_213["OrganismalFitness"]}). '
            'Red ring = Stability training/homolog overlap; grey rows = checkpoints trained on unsplit MegaScale libraries.'
        ),
        note_fontsize=7.4,
    )

    # 3. Per-assay scatter (ESM-MSR vs ProSST & ESM3-small-open)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    for ax, (xm, xl) in zip(axes, [('ProSST', 'ProSST (zero-shot)'), ('ESM3-small-open', 'ESM3-small-open (σ = 0)')]):
        for c in PG.PG_CATEGORIES_ALPHA:
            s = Tf_173[Tf_173['category'] == c]
            ax.scatter(s[xm], s['ESM-MSR'], s=18, color=M.CATEGORY_COLOR[c], alpha=0.80, lw=0, label=cat_leg[c])
        lim = [-0.2, 1.0]
        ax.plot(lim, lim, color='#999999', lw=0.8, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'ρ, {xl}')
        ax.set_ylabel('ρ, ESM-MSR (seeds 1–3)')
        wins = int((Tf_173['ESM-MSR'] > Tf_173[xm]).sum())
        ax.text(0.03, 0.97, f'ESM-MSR higher on {wins} of {n_173} assays', transform=ax.transAxes, va='top', fontsize=8.5)
    axes[0].legend(fontsize=7.8, loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
    style.finish_figure(
        fig,
        out / 'pg_per_dms_scatter',
        title=f'ProteinGym per-assay Spearman ρ ({n_173} of {n_213} held-out assays under four-part split rule)',
    )


# ----------------------------------------------------------------------------------------------
# Sigma ∈ {0, 0.5, 1} property summaries, multi-property trajectories & worst-assay diagnostics
# ----------------------------------------------------------------------------------------------
def _run_sigma_property_outputs(out: Path, B: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dms = results_dir('dms3')
    sig_all213, seeds_213 = PG.load_sigma_pgym(heldout=False, drop_large=True, include_fallback_small=False, include_dms3=False, refresh=refresh)
    sig_173, seeds_173 = PG.load_sigma_pgym(heldout=True, drop_large=True, include_fallback_small=False, include_dms3=False, refresh=False)
    sig_pg217, _ = PG.load_sigma_pgym(heldout=False, drop_large=False, include_fallback_small=True, include_dms3=False, refresh=False)
    sig_with_dms, _ = PG.load_sigma_pgym(heldout=False, drop_large=False, include_fallback_small=True, include_dms3=True, refresh=False)

    atomic_to_csv(sig_pg217, out / 'per_assay_sigma_rho.csv', index=False, float_format='%.4f')
    sig_dms_only = sig_with_dms[sig_with_dms['source'] != 'ProteinGym'].reset_index(drop=True)
    if len(sig_dms_only):
        atomic_to_csv(sig_dms_only, out_dms / 'per_assay_sigma_rho.csv', index=False, float_format='%.4f')
    pvals_173 = PG.sigma_delta_pvalues(sig_173, per_seed_dep=seeds_173, B=max(B, 10000), seed=3)
    atomic_to_csv(pvals_173, out / 'property_delta_pvalues.csv', index=False, float_format='%.4g')

    n_mm = int(sig_173['has_multi'].astype(bool).sum())
    k05 = int((sig_173.loc[sig_173['has_multi'].astype(bool), 'dep_head_0.5'] == 'combined').sum())
    note_common = (
        f'ESM-MSR deployed readout across {len(sig_173)} held-out ProteinGym assays (of {len(sig_all213)} non-dropped assays; '
        f'4 ultra-large BRCA1/BRCA2/POLG/SCN5A assays excluded uniformly across Figure 3b and 3c): Dual-view combined '
        f'prediction on the {n_mm} multi-mutant assays ({k05}/{n_mm} available at σ = 0.5), WT-view on single-mutant assays. '
        f'Box = IQR, thick dark bar = median (value labelled), thin darker bar = mean; p-values under ticks are two-sided '
        f'joint [assay × triplicate-seed] bootstrap tests (B = {max(B, 10000):,}), Holm-adjusted across the 10 category × σ comparisons.'
    )

    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    PG.draw_property_sigma_panel(ax, sig_173, pvals_173, mark_leaks=False, stats_without_leaks=False, compact=False)
    ax.set_title(f'ProteinGym Δρ vs LoRA scale σ by coarse property ({len(sig_173)} held-out assays)', loc='left', fontsize=10.5, fontweight='bold', pad=26)
    style.finish_figure(
        fig,
        out / 'pg_property_sigma_summary',
        note=note_common,
        top=0.92,
    )

    fig2, ax2 = plt.subplots(figsize=(13.6, 5.7))
    PG.draw_property_sigma_panel(ax2, sig_all213, pvals_173, mark_leaks=True, stats_without_leaks=True, compact=False)
    ax2.set_title(f'ProteinGym Δρ vs LoRA scale σ ({len(sig_173)} held-out dots; {len(sig_all213) - len(sig_173)} homologous assays marked ×)', loc='left', fontsize=10.5, fontweight='bold', pad=26)
    style.finish_figure(
        fig2,
        out / 'pg_property_sigma_summary_with_homologs',
        note=note_common + ' Grey × markers show training-homologous assays excluded from boxes, medians, means, and p-values.',
        top=0.92,
    )

    def _short_tag(dms_name: str, src: str) -> str:
        clean = dms_name.replace('_binding_domain', '_x') if src != 'ProteinGym' else dms_name
        parts = clean.split('_')
        i = next(k for k, p in enumerate(parts) if p.isdigit() and len(p) == 4)
        return f'{parts[i - 1]} {parts[i]}' + (' (DMS)' if src != 'ProteinGym' else '')

    def _load_wt_assay_seeds(name: str, src: str) -> list[pd.DataFrame]:
        out_dfs = []
        for sd in PG.SEEDS:
            if src == 'ProteinGym':
                files = {
                    1.0: FINAL_PGYM_WT / 'FINAL_p0' / f's{sd}' / f'{name}.csv',
                    0.5: FINAL_PGYM_WT_S05 / 'FINAL_p0' / f's{sd}' / f'{name}.csv',
                    0.0: (
                        MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only' / f'{name}.csv'
                        if (MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only' / f'{name}.csv').exists()
                        else ESM3_S0_PGYM_COMB / f'{name}.csv'
                    ),
                }
                id_col, y_col = 'mutant', 'DMS_score'
            else:
                files = {
                    1.0: FINAL_BENCH / name / 'final_checkpoints' / 'FINAL_p0' / f's{sd}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv',
                    0.5: FINAL_BENCH / name / 'final_checkpoints' / 'FINAL_p0' / f's{sd}' / 'best.ckpt_sigma0.5_unmasked_predictions.csv',
                    0.0: MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / f'{name}.csv',
                }
                id_col, y_col = 'id', 'ddG_ML'
            if not all(f.exists() for f in files.values()):
                continue
            joined = None
            for sg, f in files.items():
                d = pd.read_csv(f, usecols=[id_col, y_col, 'wt_lora_pred']).drop_duplicates(id_col)
                d = d.rename(columns={id_col: 'vid', y_col: 'y', 'wt_lora_pred': f's{sg}'}).set_index('vid')
                joined = d if joined is None else joined.join(d[[f's{sg}']], how='inner')
            out_dfs.append(joined.reset_index())
        return [d for d in out_dfs if len(d) == len(out_dfs[0])] if out_dfs else []

    def _render_multi_property(rows_in: list[dict], lookup_df: pd.DataFrame, csv_path: Path, fig_stem: Path, title_prefix: str, nc_max: int = 4) -> None:
        mp = pd.DataFrame(rows_in)
        mp = mp[~mp['assay'].isin(excluded('proteingym'))].reset_index(drop=True)
        mp['label'] = [f"{r.axis.lower()[:5]}: {_short_tag(r.assay, r.source)}" for r in mp.itertuples()]
        mp['order'] = mp['axis'].map(AXIS_ORDER)
        mp = mp.sort_values(['protein', 'order', 'assay']).drop(columns='order').reset_index(drop=True)
        rho_lookup = lookup_df.set_index('assay')
        for s in PG.SIGMAS:
            mp[f'rho_{s}'] = mp['assay'].map(rho_lookup[f'rho_{s}'])
        mp['n'] = mp['assay'].map(rho_lookup['n'])
        mp = mp[mp['rho_1.0'].notna()].reset_index(drop=True)
        atomic_to_csv(mp, csv_path, index=False, float_format='%.4f')
        if mp.empty:
            return

        rng_mp = np.random.default_rng(1)
        boots_mp = {}
        for r in mp.itertuples():
            ds_list = _load_wt_assay_seeds(r.assay, r.source)
            y_v = ds_list[0]['y'].to_numpy(float)
            Xs = [np.column_stack([d[f's{sg}'].to_numpy(float) for sg in PG.SIGMAS]) for d in ds_list]
            n_v = len(y_v)
            draws = np.empty((B, len(PG.SIGMAS)), dtype=float)
            sw = rng_mp.multinomial(len(Xs), np.full(len(Xs), 1.0 / len(Xs)), size=B).astype(float) / len(Xs)
            for b in range(B):
                idx_v = rng_mp.integers(0, n_v, n_v)
                ry = rankdata(y_v[idx_v])
                seed_r = np.array([[np.corrcoef(rankdata(X[idx_v, k]), ry)[0, 1] for k in range(len(PG.SIGMAS))] for X in Xs])
                draws[b] = (sw[b, :, None] * seed_r).sum(axis=0)
            boots_mp[r.assay] = draws

        prots = sorted(mp['protein'].unique())
        nc = min(nc_max, max(1, len(prots)))
        nr = int(np.ceil(len(prots) / nc))
        fig_w = (5.4 * nc) if nr == 1 else (4.4 * nc)
        fig3, axs = plt.subplots(nr, nc, figsize=(fig_w, 3.0 * nr + (0.55 if nr == 1 else 0.0)), squeeze=False)
        for ax, prot in zip(axs.ravel(), prots):
            sub = mp[mp['protein'] == prot].reset_index(drop=True)
            ys = sub['rho_1.0'].to_numpy(float)
            span_y = max(0.08, float(np.ptp(sub[[f'rho_{s}' for s in PG.SIGMAS]].to_numpy())))
            gap = 0.08 * span_y
            ypos = {}
            for i in np.argsort(ys):
                y_cur = ys[i]
                for j in sorted(ypos.values()):
                    if abs(y_cur - j) < gap:
                        y_cur = j + gap
                ypos[i] = y_cur
            for k, r in sub.iterrows():
                col = AXIS_COLOR[r.axis]
                lo, hi = np.percentile(boots_mp[r.assay], [2.5, 97.5], axis=0)
                ax.fill_between(PG.SIGMAS, lo, hi, color=col, alpha=0.14, lw=0)
                ax.plot(PG.SIGMAS, [r[f'rho_{s}'] for s in PG.SIGMAS], marker='o', color=col, mec='white', mew=1.0, ms=5.5)
                ax.text(1.03, ypos[k], r.label, fontsize=6.5, color='#333333', va='center')
            ax.set_title(prot.replace('_', ' '), fontsize=9.5, loc='left')
            ax.set_xticks(PG.SIGMAS)
            ax.set_xticklabels(['0', '0.5', '1'])
            ax.set_xlim(-0.05, 1.55)
            ax.grid(axis='y', color=style.GRID, lw=0.6)
        for ax in axs.ravel()[len(prots):]:
            ax.axis('off')
        for ax in axs[:, 0]:
            if ax.axison:
                ax.set_ylabel('Spearman ρ')
        for ax in axs[-1, :]:
            if ax.axison:
                ax.set_xlabel('σ (LoRA scale)')
        h_leg = [Line2D([], [], color=AXIS_COLOR[k], lw=2.0, marker='o', label=AXIS_NAME[k]) for k in ('ABUND', 'ACT', 'BIND', 'FIT', 'SPEC') if k in set(mp['axis'])]
        fig3.legend(handles=h_leg, loc='upper right', ncol=len(h_leg), fontsize=8.2, bbox_to_anchor=(0.99, 0.995))
        style.finish_figure(
            fig3,
            fig_stem,
            title=f'{title_prefix} ({len(prots)} proteins, {len(mp)} held-out assays): ρ vs LoRA scale σ',
            note=f'FINAL_p0 WT head across seeds 1–3 at σ = 0.5, 1.0 vs ESM3-small-open at σ = 0; shaded band = 95% joint [variant × seed] bootstrap (B = {B:,}).',
            top=0.90 if nr == 1 else 0.93,
        )

    # 1. Pure ProteinGym multi-property per-protein trajectories -> 03_proteingym/ (zero 3-DMS)
    mp_pg_rows = [
        dict(protein=prot, assay=dms, source='ProteinGym', axis=axis, note=note)
        for dms, (prot, axis, note) in MULTI_PROP_ANNOT.items()
    ]
    _render_multi_property(
        mp_pg_rows, sig_pg217,
        out / 'multi_property_index.csv', out / 'pg_multi_property_trajectories',
        title_prefix='ProteinGym multi-property proteins', nc_max=4,
    )

    # 2. 3-DMS DLG4/GRB2 binding vs abundance trajectories -> 99_misc/dms3/
    mp_dms_rows = []
    for prot, pg_id, axis, note, ds_bind in MULTI_PROP_EXTRA:
        mp_dms_rows.append(dict(protein=prot, assay=pg_id, source='ProteinGym', axis=axis, note=note))
        mp_dms_rows.append(dict(protein=prot, assay=ds_bind, source='DMS benchmark', axis='BIND', note='binding PCA'))
    _render_multi_property(
        mp_dms_rows, sig_with_dms,
        out_dms / 'dms_multi_property_index.csv', out_dms / 'dms_multi_property_trajectories',
        title_prefix='DLG4 & GRB2', nc_max=2,
    )

    # Worst-degraded held-out assays at sigma=1 (integrating p0_sigma/05_worst_assays.py, previously omitted from run_all.sh)
    w_rows = []
    held_set = set(sig_173['assay'])
    for a_name in [w for w in WORST_ASSAYS if w in held_set]:
        ds_list = _load_wt_assay_seeds(a_name, 'ProteinGym')
        if not ds_list:
            continue
        per_sd = []
        for d in ds_list:
            sing = d[~d['vid'].astype(str).str.contains(':')].copy()
            if len(sing) < 10:
                continue
            pos = sing['vid'].astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()
            y_v, s0, s1 = sing['y'].to_numpy(), sing['s0.0'].to_numpy(), sing['s1.0'].to_numpy()
            dl = (s1 - s1.mean()) / s1.std() - (s0 - s0.mean()) / s0.std()
            g_pos = pd.DataFrame(dict(pos=pos, y=y_v, s0=s0, s1=s1)).groupby('pos').mean()
            within0, within1 = [], []
            for _, h_df in pd.DataFrame(dict(pos=pos, y=y_v, s0=s0, s1=s1)).groupby('pos'):
                if len(h_df) >= 5 and h_df['y'].nunique() > 2:
                    within0.append(float(spearmanr(h_df['s0'], h_df['y'])[0]))
                    within1.append(float(spearmanr(h_df['s1'], h_df['y'])[0]))
            hyd = sing['vid'].astype(str).str[-1].map(KD_HYDROPATHY).to_numpy(float)
            per_sd.append(dict(
                n=len(sing), n_positions=len(g_pos),
                rho0=float(spearmanr(s0, y_v)[0]), rho1=float(spearmanr(s1, y_v)[0]),
                delta_rho=float(spearmanr(s1, y_v)[0] - spearmanr(s0, y_v)[0]),
                rho_added=float(spearmanr(dl, y_v)[0]),
                between0=float(spearmanr(g_pos['s0'], g_pos['y'])[0]),
                between1=float(spearmanr(g_pos['s1'], g_pos['y'])[0]),
                within0=float(np.nanmean(within0)), within1=float(np.nanmean(within1)),
                hydropathy_y=float(spearmanr(hyd, y_v)[0]),
                hydropathy_added=float(spearmanr(hyd, dl)[0]),
            ))
        if per_sd:
            cat_a = str(sig_173.loc[sig_173['assay'] == a_name, 'category'].iloc[0])
            w_rows.append(dict(assay=a_name, category=cat_a, **pd.DataFrame(per_sd).mean().to_dict(), n_seeds=len(per_sd)))
    w_df = pd.DataFrame(w_rows).sort_values('delta_rho').reset_index(drop=True)
    atomic_to_csv(w_df, out / 'worst_assays_sigma1.csv', index=False, float_format='%.4f')

    if len(w_df):
        fig4, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
        y_pos = np.arange(len(w_df))

        def _worst_label(assay_id: str) -> str:
            parts = assay_id.split('_')
            yi = next((k for k, p in enumerate(parts) if p.isdigit() and len(p) == 4), 3)
            author = parts[yi - 1]
            tail = '_'.join(parts[yi + 1:]).replace('expression', 'expr.').replace('binding-', '')
            return f"{parts[0]}_{parts[1]} ({author}, {tail})" if tail else f"{parts[0]}_{parts[1]} ({author})"

        short_lbls = [_worst_label(r.assay) for r in w_df.itertuples()]
        ax_l.axvline(0, color='#666666', lw=0.8)
        ax_l.barh(y_pos - 0.18, w_df['between1'] - w_df['between0'], 0.34, color='#2a78d6', label='Between-site Δρ (σ=1 − σ=0)')
        ax_l.barh(y_pos + 0.18, w_df['within1'] - w_df['within0'], 0.34, color='#eb6834', label='Within-site Δρ (σ=1 − σ=0)')
        ax_l.set_yticks(y_pos)
        ax_l.set_yticklabels(short_lbls, fontsize=8.3)
        ax_l.set_ylim(-1.75, len(w_df) - 0.35)
        ax_l.set_xlabel('Δ Spearman ρ (σ = 1 minus σ = 0)')
        ax_l.set_title('a   Site vs substitution Δρ on worst-degraded assays', loc='left', fontsize=9.6)
        ax_l.legend(fontsize=7.9, loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
        ax_l.grid(axis='x', color=style.GRID, lw=0.6)

        ax_r.axvline(0, color='#666666', lw=0.8)
        ax_r.barh(y_pos - 0.18, w_df['hydropathy_y'], 0.34, color='#1baf7a', label='ρ(hydropathy, measured y)')
        ax_r.barh(y_pos + 0.18, w_df['hydropathy_added'], 0.34, color='#6250d6', label='ρ(hydropathy, z(s1) − z(s0))')
        ax_r.set_xlabel('Spearman ρ with Kyte–Doolittle hydropathy')
        ax_r.set_title('b   Hydropathy preference vs LoRA stability shift', loc='left', fontsize=9.6)
        ax_r.legend(fontsize=7.9, loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
        ax_r.grid(axis='x', color=style.GRID, lw=0.6)
        style.finish_figure(
            fig4,
            out / 'pg_worst_assays_sigma1',
            title=f'Mechanistic decomposition of held-out assays where σ = 1 reduces ρ ({len(w_df)} assays; FINAL_p0 seeds 1–3)',
            note='Single mutants only. Adapter stability signal z(s1) − z(s0) promotes hydrophobic substitutions; losses occur where membrane expression or polar active-site assays penalize hydrophobicity.',
        )
    return sig_173, pvals_173


# ----------------------------------------------------------------------------------------------
# Ours vs Published zero-shot comparison (00_supplementary_figs/pg_ours_vs_published) — zero importlib!
# ----------------------------------------------------------------------------------------------
def _activity_ours_vs_published_curves(B: int) -> dict[tuple[str, str], tuple[pd.DataFrame, int]]:
    """Compute Activity hurt-rate screening curves (0.5 assay SD drop) for ESM3-small-open and ProteinMPNN (ours vs published)."""
    ref = pd.read_csv(PGYM_REF_CSV)
    ref = ref[ref['coarse_selection_type'] == 'Activity'].set_index('DMS_id')
    ref = ref[~ref.index.isin(excluded('proteingym')) & ~ref.index.isin(PG.PG_DROP)]
    wtref = pd.read_csv(REFERENCE / 'pg_activity' / 'wt_reference.csv').set_index('DMS_id')
    aicda = pd.read_csv(REFERENCE / 'pg_activity' / 'aicda_wt_by_position.csv').set_index('pos')

    pmpnn_dirs = {1: 'fast', 2: 'fast_seed1', 3: 'fast_seed2'}
    groups_all, y_all, use_all = [], [], []
    p_e3_ours, p_e3_pub, p_pm_pub = [], [], []
    p_pm_ours = {sd: [] for sd in PG.SEEDS}

    for dms, r in ref.iterrows():
        base = pd.read_csv(FINAL_PGYM_WT / 'FINAL_p0' / 's1' / f'{dms}.csv', usecols=['mutant', 'DMS_score'])
        base = base[~base['mutant'].astype(str).str.contains(':')].reset_index(drop=True)
        x = base['DMS_score'].to_numpy(dtype=float)
        cut = float(r['DMS_binarization_cutoff'])
        if (cut < x.min() or cut > x.max()) and x.min() <= -cut <= x.max():
            cut = -cut
        frac_up = (x > cut).mean()
        cut_eff = float(np.median(x)) if (frac_up < 0.05 or frac_up > 0.95) else cut
        up = x[x > cut_eff]
        xs = np.linspace(up.min(), up.max(), 400)
        w_pl = float(xs[np.argmax(gaussian_kde(up)(xs))])

        t = wtref.loc[dms]
        use = np.ones(len(x), dtype=bool)
        if t['tier'] == 'C-position':
            pos = base['mutant'].astype(str).str.extract(r'(\d+)')[0].astype(int)
            wt = pos.map(aicda['wt_score']).to_numpy(dtype=float)
            use = (pos.map(aicda['wt_count_selected']) >= 100).to_numpy()
        elif t['tier'] == 'C-open':
            wt = np.full(len(x), w_pl)
        else:
            wt = np.full(len(x), float(t['wt_value']))
        assay_sd = float(np.sqrt(x[use].var()))
        # Positive y = drop in units of assay SD; hurt iff y > 0.5
        y_drop = -(x - wt) / assay_sd
        groups_all.extend([dms] * len(x))
        y_all.append(y_drop)
        use_all.append(use)

        f0 = MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only' / f'{dms}.csv'
        if not f0.exists():
            f0 = ESM3_S0_PGYM_COMB / f'{dms}.csv'
        s0 = pd.read_csv(f0, usecols=['mutant', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant')['wt_lora_pred']
        p_e3_ours.append(s0.reindex(base['mutant']).to_numpy(dtype=float))

        for sd in PG.SEEDS:
            pm_f = PUBLIC / 'proteinmpnn' / 'proteingym' / pmpnn_dirs[sd] / f'{dms}.csv'
            spm = pd.read_csv(pm_f, usecols=['mutant', 'mpnn_score']).drop_duplicates('mutant').set_index('mutant')['mpnn_score']
            p_pm_ours[sd].append(spm.reindex(base['mutant']).to_numpy(dtype=float))

        pub = pd.read_csv(PUBLIC / 'proteingym_published' / f'{dms}.csv', usecols=['mutant', 'ESM3', 'ProteinMPNN']).drop_duplicates('mutant').set_index('mutant')
        p_e3_pub.append(pub['ESM3'].reindex(base['mutant']).to_numpy(dtype=float))
        p_pm_pub.append(pub['ProteinMPNN'].reindex(base['mutant']).to_numpy(dtype=float))

    use_mask = np.concatenate(use_all)
    grp_arr = np.asarray(groups_all, dtype=object)[use_mask]
    y_arr = np.concatenate(y_all)[use_mask]
    n_act = int(len(np.unique(grp_arr)))

    series = {
        ('ESM3-small-open', 'ours'): np.concatenate(p_e3_ours)[None, use_mask],
        ('ESM3-small-open', 'published'): np.concatenate(p_e3_pub)[None, use_mask],
        ('ProteinMPNN', 'ours'): np.stack([np.concatenate(p_pm_ours[sd])[use_mask] for sd in PG.SEEDS]),
        ('ProteinMPNN', 'published'): np.concatenate(p_pm_pub)[None, use_mask],
    }
    out = {}
    for key, sc_mat in series.items():
        curve = SCR.per_protein_curve(grp_arr, sc_mat, y_arr, KS_ACT, tau=0.5, B=B, seed=0, only_recoverable=False)
        out[key] = (curve[curve['metric'] == 'hit_rate'].sort_values('K').reset_index(drop=True), n_act)
    return out


def _run_ours_vs_published(
    out_23: Path,
    Tf_173: pd.DataFrame,
    seeds_173: dict[str, pd.DataFrame],
    prosst_src: str,
    B: int,
) -> None:
    pub = pd.read_csv(PGYM / 'pgym_public_per_dms.csv').set_index('DMS_id')
    pairs = [('ESM3-small-open', 'ESM3 rho_full'), ('ProteinMPNN', 'ProteinMPNN rho_full')]
    if prosst_src == 'ours':
        pairs.append(('ProSST', 'ProSST-2048 rho_full'))

    per_seed_cmp = {}
    D = Tf_173[['category']].copy()
    for ours_name, pub_col in pairs:
        D[f'{ours_name} ours'] = Tf_173[ours_name]
        D[f'{ours_name} published'] = pub[pub_col].reindex(Tf_173.index)
        per_seed_cmp[f'{ours_name} ours'] = seeds_173[ours_name]
        per_seed_cmp[f'{ours_name} published'] = pd.DataFrame({'s1': D[f'{ours_name} published']}, index=Tf_173.index)

    cats_order = list(PG.PG_CATEGORIES)
    cols = cats_order + ['PG avg']
    summ_df, reps_map = PG.boot_pgym_categories(per_seed_cmp, Tf_173['category'], B=B, seed=2, columns=(*PG.PG_CATEGORIES_ALPHA, 'PG avg'))
    summ_idx = summ_df.set_index(['model', 'column'])

    rows = []
    for name, _ in pairs:
        for c in cols:
            r_o = summ_idx.loc[(f'{name} ours', c)]
            r_p = summ_idx.loc[(f'{name} published', c)]
            d_reps = reps_map[f'{name} ours'][c] - reps_map[f'{name} published'][c]
            d_lo, d_hi = ci(d_reps)
            for ver, r_v in (('ours', r_o), ('published', r_p)):
                rows.append(dict(
                    model=name, version=ver, column=c,
                    value=float(r_v['value']), lo=float(r_v['lo']), hi=float(r_v['hi']),
                    delta_ours_minus_published=float(r_o['value'] - r_p['value']),
                    delta_lo=float(d_lo), delta_hi=float(d_hi),
                    n_assays=int(r_v['n_dms']), n_seeds=int(r_v['n_seeds']),
                ))
    C = pd.DataFrame(rows)
    atomic_to_csv(C, out_23 / 'pg_ours_vs_published.csv', index=False, float_format='%.4f')
    act = _activity_ours_vs_published_curves(B=B)

    fig = plt.figure(figsize=(15.2 + 3.6 * (len(pairs) - 2), 9.8))
    gs = fig.add_gridspec(2, 1 + len(pairs), height_ratios=[1, 1], width_ratios=[1.5] + [1] * len(pairs), hspace=0.40, wspace=0.28)
    axa = fig.add_subplot(gs[0, :])
    w = 0.19
    for j, (name, _) in enumerate(pairs):
        col = M.color(name)
        for i, ver in enumerate(('ours', 'published')):
            x = np.arange(len(cols)) + (j * 2 + i - (len(pairs) - 0.5)) * w
            r = C[(C['model'] == name) & (C['version'] == ver)].set_index('column').loc[cols]
            axa.bar(
                x, r['value'], w * 0.92,
                color=col if ver == 'ours' else M.tint(col, 0.55),
                edgecolor='black' if ver == 'ours' else col,
                hatch=None if ver == 'ours' else '////', lw=0.8, zorder=3,
            )
            axa.errorbar(x, r['value'], yerr=[r['value'] - r['lo'], r['hi'] - r['value']], fmt='none', ecolor=style.RING, elinewidth=1.2, capsize=2.2, zorder=5)
            for xx, v, hi in zip(x, r['value'], r['hi']):
                axa.text(xx, hi + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=7.0, rotation=90)
    axa.set_xticks(range(len(cols)))
    n_map = C[C['model'] == pairs[0][0]].drop_duplicates('column').set_index('column')['n_assays']
    cat_lbl_map = {**PG.PG_CATEGORY_SHORT, 'PG avg': 'ProteinGym\naverage'}
    axa.set_xticklabels([f'{cat_lbl_map[c]}\n({int(n_map[c])} assays)' for c in cols], fontsize=9.0)
    axa.set_ylabel('mean per-assay Spearman ρ')
    axa.set_ylim(0, 0.88)
    axa.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    leg_patches = []
    for name, _ in pairs:
        col = M.color(name)
        leg_patches.append(Patch(fc=col, ec='black', label=f'{name}, ours'))
        leg_patches.append(Patch(fc=M.tint(col, 0.55), ec=col, hatch='////', label=f'{name}, ProteinGym published'))
    axa.legend(handles=leg_patches, ncol=2, loc='upper right', fontsize=8.3, frameon=True, facecolor='white', edgecolor='#cccccc')
    axa.set_title(f'a   Figure 3b columns: mean ρ per ProteinGym category ({len(Tf_173)} held-out assays; B = {B:,} joint assay × seed bootstrap)', loc='left', fontsize=10, fontweight='bold')

    for k, (name, _) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, 1 + k])
        for c in cats_order:
            s = D[D['category'] == c]
            ax.scatter(s[f'{name} published'], s[f'{name} ours'], s=16, color=M.CATEGORY_COLOR[c], alpha=0.80, lw=0, label=PG.PG_CATEGORY_SHORT[c])
        ax.plot([-0.2, 1], [-0.2, 1], color='#999999', lw=0.8, ls='--')
        ax.set_xlim(-0.2, 1)
        ax.set_ylim(-0.2, 1)
        ok = D[[f'{name} ours', f'{name} published']].dropna()
        rho_pair = float(spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0])
        wins = int((ok.iloc[:, 0] > ok.iloc[:, 1]).sum())
        ax.text(0.03, 0.97, f'ρ between runs = {rho_pair:.2f}\nours higher on {wins} of {len(ok)} assays', transform=ax.transAxes, va='top', fontsize=8.4)
        ax.set_xlabel(f'ρ, {name} published')
        ax.set_ylabel(f'ρ, {name} ours')
        ax.set_title(f'{"cde"[k]}   {name}: per assay', loc='left', fontsize=10, fontweight='bold')
        if k == 0:
            ax.legend(fontsize=7.6, loc='lower right')

    axb = fig.add_subplot(gs[1, 0])
    n_act_assays = next(iter(act.values()))[1]
    for (name, ver), (t_c, _) in act.items():
        col = M.color(name)
        axb.plot(t_c['K'], t_c['value'], color=col if ver == 'ours' else M.tint(col, 0.35), ls='-' if ver == 'ours' else '--', lw=2.2, marker='o', ms=4, label=f'{name}, {ver}')
        axb.fill_between(t_c['K'], t_c['lo'], t_c['hi'], color=col, alpha=0.10 if ver == 'ours' else 0.05, lw=0)
    axb.set_xscale('log')
    axb.set_xlabel('variants screened per assay (K)')
    axb.set_ylabel('fraction of K picks that hurt (lower is better)')
    axb.set_xticks(KS_ACT)
    axb.set_xticklabels(KS_ACT)
    axb.grid(axis='y', color=style.GRID, lw=0.6)
    axb.legend(fontsize=8.0)
    axb.set_title(f'b   Activity screening ({n_act_assays} held-out assays, single mutants)', loc='left', fontsize=10, fontweight='bold')

    style.finish_figure(
        fig,
        out_23 / 'pg_ours_vs_published',
        note=(
            f'Ours: ESM3-small-open = σ = 0 read of the ESM-MSR checkpoint (WT-view on singles assays, Dual-view on multi-mutant assays); '
            f'ProteinMPNN = our 3 per-variant runs (seeds resampled jointly with assays, B = {B:,}). '
            f'Published: scores shipped by ProteinGym. ProSST source: {prosst_src}. Held-out = {len(Tf_173)} of 213 assays.'
        ),
        note_fontsize=7.5,
    )


# ----------------------------------------------------------------------------------------------
# Main Entrypoint
# ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--B', type=int, default=1000, help='Bootstrap replicates (default: 1000)')
    ap.add_argument('--fast', action='store_true', help='Fast smoke mode (B = 100)')
    ap.add_argument('--refresh', action='store_true', help='Recompute cached ProteinGym & Domainome summaries')
    ap.add_argument('--split', default='capped', choices=('capped', 'mega'), help='Accepted for runner uniformity')
    args = ap.parse_args(argv)
    B = 100 if args.fast else int(args.B)

    register_ext_styles()
    out_pg = results_dir('proteingym')
    out_supp = results_dir('supplementary_figs')

    T_213, seeds_213 = PG.load_all_models_pgym(heldout=False, drop_large=True, refresh=args.refresh)
    Tf_173, seeds_173 = PG.load_all_models_pgym(heldout=True, drop_large=True, refresh=False)

    prosst_dir = PUBLIC / 'prosst' / 'proteingym'
    prosst_src = 'ours' if all((prosst_dir / f'{d}.csv').exists() for d in T_213.index) else 'published'
    atomic_write_text(
        out_pg / 'pg_zero_shot_sources.txt',
        f'ProSST: {prosst_src} (baselines/public/prosst/proteingym when complete; else ProteinGym published)\n'
        'ProteinMPNN: our 3 per-variant runs (baselines/public/proteinmpnn/proteingym, seeds resampled)\n'
        'ESM3-small-open: our sigma-0 read (WT-view on singles, Dual-view on multi-mutant assays)\n'
        'ESM3ΔG: scored on all assays; Stability training-overlap assays marked with red ring\n',
    )

    full_models = list(EXT_ORDER) + list(PG.SUPPLEMENT_PG) + ['ESM-MSR σ=0.5']
    t_uni = T_213[['category'] + list(EXT_ORDER)].copy()
    atomic_to_csv(t_uni, out_pg / 'pg_per_dms.csv', float_format='%.4f')
    t_full = T_213[['category'] + full_models].copy()
    atomic_to_csv(t_full, out_pg / 'pg_all_models_per_dms.csv', float_format='%.4f')

    pg_cols = (*PG.PG_CATEGORIES_ALPHA, 'PG avg', 'DMS mean')
    dom = _domainome_summary(B=B, refresh=args.refresh)

    # All 213 assays summaries
    R_uni_213, _ = PG.boot_pgym_categories({m: seeds_213[m] for m in EXT_ORDER}, T_213['category'], B=B, seed=2, columns=pg_cols)
    atomic_to_csv(R_uni_213, out_pg / 'pg_summary_all213.csv', index=False, float_format='%.4f')
    R_full_213, _ = PG.boot_pgym_categories({m: seeds_213[m] for m in full_models}, T_213['category'], B=B, seed=2, columns=pg_cols)
    atomic_to_csv(R_full_213, out_pg / 'pg_all_models_summary_all213.csv', index=False, float_format='%.4f')

    # Canonical 173 held-out assays summaries (+ 436 held-out Domainome domains)
    R_uni_173, _ = PG.boot_pgym_categories({m: seeds_173[m] for m in EXT_ORDER}, Tf_173['category'], B=B, seed=2, columns=pg_cols)
    dom_rows = pd.DataFrame([
        dict(model=m, column='Domainome', value=v[0], lo=v[1], hi=v[2], n_dms=v[3], n_seeds=seeds_173[m].shape[1], seed_sd_pg_avg=np.nan)
        for m, v in dom.items() if m in EXT_ORDER
    ])
    R_uni = pd.concat([R_uni_173, dom_rows], ignore_index=True)
    atomic_to_csv(R_uni, out_pg / 'pg_summary.csv', index=False, float_format='%.4f')

    R_sig_173, _ = PG.boot_pgym_categories(
        {'ESM-MSR σ=1': seeds_173['ESM-MSR'], 'ESM-MSR σ=0.5': seeds_173['ESM-MSR σ=0.5']},
        Tf_173['category'], B=B, seed=2, columns=pg_cols,
    )
    dom_sig = pd.DataFrame([
        dict(model='ESM-MSR σ=1', column='Domainome', value=dom['ESM-MSR'][0], lo=dom['ESM-MSR'][1], hi=dom['ESM-MSR'][2], n_dms=dom['ESM-MSR'][3], n_seeds=3, seed_sd_pg_avg=np.nan),
        dict(model='ESM-MSR σ=0.5', column='Domainome', value=dom['ESM-MSR σ=0.5'][0], lo=dom['ESM-MSR σ=0.5'][1], hi=dom['ESM-MSR σ=0.5'][2], n_dms=dom['ESM-MSR σ=0.5'][3], n_seeds=3, seed_sd_pg_avg=np.nan),
    ])
    atomic_to_csv(pd.concat([R_sig_173, dom_sig], ignore_index=True), out_pg / 'pg_sigma_summary.csv', index=False, float_format='%.4f')

    R_full_173, _ = PG.boot_pgym_categories({m: seeds_173[m] for m in full_models}, Tf_173['category'], B=B, seed=2, columns=pg_cols)
    atomic_to_csv(R_full_173, out_pg / 'pg_all_models_summary.csv', index=False, float_format='%.4f')

    # Sigma property tables & figures (standardized on 173 held-out assays)
    sig_173, _pvals_173 = _run_sigma_property_outputs(out_pg, B=B, refresh=args.refresh)

    # Notes consumed by p05_summary_main (Figure 3)
    mm_173 = sig_173[sig_173['has_multi'].astype(bool)]
    k05 = int((mm_173['dep_head_0.5'] == 'combined').sum())
    atomic_write_text(out_pg / 'pg_sigma_note.txt', f'{k05} of {len(mm_173)}\n')

    removed_ids = sorted(set(T_213.index) - set(Tf_173.index))
    cnt_removed = T_213.loc[removed_ids, 'category'].value_counts().to_dict()
    n_dom_excl = 522 - dom['ESM-MSR'][3]
    tsu_note = (
        f'{len(removed_ids)} of {len(T_213)} ProteinGym assays '
        f'({", ".join(f"{k} {v}" for k, v in cnt_removed.items())}) and {n_dom_excl} of 522 Domainome domains\n'
    )
    atomic_write_text(out_pg / 'pg_tsu_note.txt', tsu_note)
    atomic_write_text(
        out_pg / 'pg_categories_note.txt',
        f'* Four-part training-homology filter leaves {len(Tf_173)} of {len(T_213)} ProteinGym assays and '
        f'{dom["ESM-MSR"][3]} of 522 Domainome domains (4 large BRCA1/BRCA2/POLG/SCN5A assays omitted uniformly).\n'
        f'Joint [assay × triplicate-seed] bootstrap (B = {B:,}).\n',
    )

    _draw_main_pgym_figures(out_pg, T_213, Tf_173, R_full_173, full_models, B=B)
    _run_ours_vs_published(out_supp, Tf_173, seeds_173, prosst_src=prosst_src, B=B)

    piv = (
        R_uni[R_uni['column'].isin(['Stability', 'PG avg', 'DMS mean', 'Domainome'])]
        .pivot(index='model', columns='column', values='value')[['Stability', 'PG avg', 'DMS mean', 'Domainome']]
        .reindex(EXT_ORDER)
    )
    print(f'[p03_proteingym] Held-out assays: {len(Tf_173)}/{len(T_213)} (B={B}):\n{piv.round(4).to_string()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
