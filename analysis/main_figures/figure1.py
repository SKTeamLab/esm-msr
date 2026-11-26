#!/usr/bin/env python3
"""Build Main-Text Figure 1 (`analysis/results/00_main_text_figs/figure1.{png,pdf}`).

Layout:
  - Panel a (full left side): Vertical ESM-MSR architecture schematic (`_assets/esm_msr_vertical.svg`,
    rendered without its standalone label so 'a' is placed consistently in the combined figure).
  - Panel b (top right): Stabilizing vs destabilizing MegaScale test-set Spearman's rho (`spearman_stab_destab.csv`).
  - Panel c (bottom right): Additive vs epistatic double-mutant Spearman's rho (`spearman_doubles_additive_vs_epistatic.csv`)
    with paired one-sided bootstrap p-value brackets (`epistatic_vs_additive_pvalues.csv`).

Usage:
    PYTHONPATH=src:analysis python -m analysis.main_figures.figure1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import ci
from msr_analysis.paths import REPO_ROOT, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.stats import boot_domain_seeds, boot_pvalue, fmt_p, holm, make_domain_seed_draws
from pipelines._tsu_common import (
    BEST_LABEL,
    MIN_N,
    best_other,
    boot_reps,
    count_text,
    diff_df,
    fig_width,
    key_for,
    load_tsu_test_frame,
    main_columns,
    make_domain_counts,
    pair_columns,
    per_library,
    rho_fn,
    table_grouped,
)


def _load_schematic_module():
    mod_path = REPO_ROOT / '_assets' / 'generate_esm_msr_schematics.py'
    spec = importlib.util.spec_from_file_location('_esm_msr_schematics', mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load schematic generator from {mod_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_vertical_schematic_image(scale: float = 4.0) -> np.ndarray:
    """Render `_assets/esm_msr_vertical.svg` (no standalone 'a' label) to a high-DPI RGB array."""
    mod = _load_schematic_module()
    svg_str = mod.build_vertical_svg(include_label=False)
    with tempfile.TemporaryDirectory(prefix='esm_msr_fig1_') as tmpdir:
        tmp = Path(tmpdir)
        svg_path = tmp / 'esm_msr_vertical.svg'
        png_path = tmp / 'esm_msr_vertical.png'
        svg_path.write_text(svg_str, encoding='utf-8')
        mod.render_svg_to_png(svg_path, png_path, scale=scale)
        with Image.open(png_path) as im:
            return np.asarray(im.convert('RGB'))


def _draw_pair_brackets(
    ax,
    cols: list[dict],
    T: pd.DataFrame,
    p_series: pd.Series,
    subsets: tuple[str, str] = ('additive', 'epistatic'),
    p_name_map: dict[str, str] | None = None,
    B_p: int | None = None,
) -> float:
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
        ax.text(
            (x0 + x1) / 2, y + 0.008, fmt_p(float(p_series[name]), B=B_p, prefix='p = '),
            ha='center', va='bottom', fontsize=8.0, color='#333333', zorder=9,
        )
        top = max(top, y + 0.075)
    return top


def build_figure1(
    *,
    split: str = 'capped',
    t2: pd.DataFrame | None = None,
    t3: pd.DataFrame | None = None,
    p_spearman_ser: pd.Series | None = None,
    B_p: int | None = None,
    schematic_scale: float = 4.0,
) -> Path:
    """Build and save Main-Text Figure 1 (`00_main_text_figs/figure1.{png,pdf}`)."""
    fr, _ = load_tsu_test_frame(split=split)
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols2 = main_columns(fr, best, 'unmasked')
    cols3 = pair_columns(fr, best)

    mega_dir = results_dir('megascale', split=split, create=False)
    epi_dir = results_dir('epistasis', split=split, create=False)

    if t2 is None:
        t2 = pd.read_csv(mega_dir / 'spearman_stab_destab.csv')
    if t3 is None:
        t3 = pd.read_csv(epi_dir / 'spearman_doubles_additive_vs_epistatic.csv')
    if p_spearman_ser is None:
        p_csv = epi_dir / 'epistatic_vs_additive_pvalues.csv'
        if p_csv.exists():
            p_df = pd.read_csv(p_csv)
            p_spearman_ser = p_df.set_index('model')['p_one']
            if B_p is None and 'B' in p_df.columns and len(p_df):
                B_p = int(p_df['B'].iloc[0])

    T2_st = t2[t2['subset'] != 'overall']
    ov2_st = t2[t2['subset'] == 'overall']

    # Render vertical architecture schematic without its standalone 'a' label
    img_a = _render_vertical_schematic_image(scale=schematic_scale)

    fig = plt.figure(figsize=(22.4, 12.6), facecolor='white')

    # Left panel (a): full-height vertical architecture schematic
    gs_left = fig.add_gridspec(1, 1, left=0.018, right=0.454, top=0.985, bottom=0.015)
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_a.imshow(img_a, interpolation='lanczos')
    ax_a.set_axis_off()

    # Right panels (b stacked above c): MegaScale stabilizing/destabilizing + additive/epistatic doubles
    gs_right = fig.add_gridspec(2, 1, left=0.508, right=0.992, top=0.955, bottom=0.105, hspace=0.42)
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_c = fig.add_subplot(gs_right[1, 0])

    style.bar_panel(
        ax_b, cols2, T2_st, ('stabilizing', 'destabilizing'),
        (
            count_text(T2_st, 'stabilizing', 'stabilizing, ddG > 0'),
            count_text(T2_st, 'destabilizing', 'destabilizing, ddG < 0'),
        ),
        "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
        overall_df=ov2_st, overall_label=f'overall (n = {int(ov2_st.n_variants.max()):,})',
        zero_line=True, tick_fs=8.5,
    )
    style.bar_panel(
        ax_c, cols3, t3, ('additive', 'epistatic'),
        (
            count_text(t3, 'additive', 'additive: singles model summed'),
            count_text(t3, 'epistatic', 'epistatic: model scores the double'),
        ),
        "Spearman's ρ, double mutants", (0.0, 1.12),
        tick_fs=8.5,
    )
    if p_spearman_ser is not None:
        top_y = _draw_pair_brackets(ax_c, cols3, t3, p_spearman_ser, p_name_map=name_map, B_p=B_p)
        if top_y > 1.12:
            ax_c.set_ylim(0.0, top_y)

    for ax_s in (ax_b, ax_c):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    # Nature-style bold lowercase sans-serif sub-figure annotations: a (left), b (top right), c (bottom right)
    fig.canvas.draw()
    pa = ax_a.get_position()
    pb = ax_b.get_position()
    pc = ax_c.get_position()
    y_top_lbl = max(pa.y1, pb.y1 + 0.032)
    fig.text(0.008, y_top_lbl, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(pb.x0 - 0.042, y_top_lbl, 'b', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(pc.x0 - 0.042, pc.y1 + 0.032, 'c', ha='left', va='top', fontsize=16, fontweight='bold')

    out_path = results_dir('main_text_figs') / 'figure1'
    style.save(fig, out_path)
    return out_path


def _pearson_fn(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr
    return float(pearsonr(p, y)[0]) if len(y) >= 3 and np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan


def _rmse_fn(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) >= 1 else np.nan


def _enrich_summary_table(fr, T: pd.DataFrame, cells: dict, min_n_map: dict[str, int], name_map: dict[str, str]) -> pd.DataFrame:
    """Add mean per-domain Pearson's r and RMSE (kcal/mol) alongside Spearman's rho and sample sizes."""
    out = T.copy()
    r_vals, rmse_vals = [], []
    for row in out.itertuples(index=False):
        m_sub = cells[row.subset][0]
        mn = min_n_map[row.subset]
        df_r = per_library(fr, row.key, m_sub, _pearson_fn, mn)
        df_e = per_library(fr, row.key, m_sub, _rmse_fn, mn)
        r_vals.append(float(df_r.mean(axis=1).mean()) if len(df_r) else np.nan)
        rmse_vals.append(float(df_e.mean(axis=1).mean()) if len(df_e) else np.nan)
    out['model'] = out['column'].map(lambda x: name_map.get(x, x))
    out['pearson_r'] = r_vals
    out['rmse'] = rmse_vals
    return out


def _compute_epi_vs_add_pvalues(fr, cols_pair: list[dict], double_mask: np.ndarray, B: int = 10_000) -> pd.DataFrame:
    """Paired one-sided domain x seed bootstrap test for H1: epistatic > additive Spearman rho on doubles."""
    name_map = {c['name']: c['name'] for c in cols_pair}
    rows = []
    # Collect per-domain tables for models with distinct additive & epistatic keys
    valid = []
    for c in cols_pair:
        ka, ke = c.get('single'), c.get('double')
        if not ka or not ke or ka == ke or ka not in fr.preds or ke not in fr.preds:
            continue
        va = per_library(fr, ka, double_mask, rho_fn, MIN_N)
        ve = per_library(fr, ke, double_mask, rho_fn, MIN_N)
        common = sorted(set(va.index) & set(ve.index))
        if len(common) < 2:
            continue
        valid.append((c['name'], ka, ke, va.reindex(common), ve.reindex(common), common))
    if not valid:
        return pd.DataFrame()
    all_libs = sorted(set().union(*(set(x[5]) for x in valid)))
    counts = make_domain_counts(all_libs, B=B, seed=29)
    for m_name, ka, ke, va, ve, common in valid:
        ra = boot_reps(va, all_libs, counts)
        re_ = boot_reps(ve, all_libs, counts)
        d_reps = re_ - ra
        pt_a = float(va.mean(axis=1).mean())
        pt_e = float(ve.mean(axis=1).mean())
        delta = pt_e - pt_a
        lo, hi = ci(d_reps)
        p_one = boot_pvalue(d_reps, null=0.0, alternative='greater')
        p_two = boot_pvalue(d_reps, null=0.0, alternative='two-sided')
        dom_higher = int((ve.mean(axis=1) > va.mean(axis=1)).sum())
        rows.append(dict(
            model=name_map[m_name],
            additive=ka,
            epistatic=ke,
            rho_additive=pt_a,
            rho_epistatic=pt_e,
            delta=delta,
            lo=lo,
            hi=hi,
            domains_higher=dom_higher,
            n_domains=len(common),
            p_one=p_one,
            p_two=p_two,
            B=B,
        ))
    res = pd.DataFrame(rows)
    if len(res):
        res['p_one_holm'] = holm(res['p_one'].to_numpy(float))
    return res


def build_supp_mega_splits_figure1(*, B: int = 10_000, refresh: bool = True) -> Path:
    """Build Supplementary Figure 1 reproduction on full MegaScale (`mega_splits`), without Panel a or suptitle."""
    from msr_analysis import data as D

    if refresh:
        D.load_frame('tsuboyama_mega_test', refresh=True)
    fr, _ = load_tsu_test_frame(split='mega')
    sp = fr.spine
    use = sp['use'].to_numpy()
    y = sp['y'].to_numpy(float)
    n_mut = sp['n_mut'].to_numpy()
    double = use & (n_mut == 2)
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = make_domain_counts(libs, B=B, seed=5)

    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols_a = main_columns(fr, best, 'unmasked')
    cols_b = pair_columns(fr, best)
    drawn_a = [c for c in cols_a if c.get('draw', True)]
    drawn_b = [c for c in cols_b if c.get('draw', True)]

    cells_a = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}
    t_a_st = table_grouped(fr, drawn_a, cells_a, rho_fn, MIN_N, counts, libs)
    ov_a_st = table_grouped(fr, drawn_a, {'overall': (use, 'all')}, rho_fn, 3, counts, libs)
    t_a_all = pd.concat([t_a_st, ov_a_st], ignore_index=True)

    cells_b = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    t_b = table_grouped(fr, drawn_b, cells_b, rho_fn, MIN_N, counts, libs)
    p_epi_df = _compute_epi_vs_add_pvalues(fr, drawn_b, double, B=B)
    p_epi_ser = p_epi_df.set_index('model')['p_one'] if len(p_epi_df) else pd.Series(dtype=float)

    cats_mega = dict(M.GROUP_TITLE, transfer='Retrained on Full MegaScale (mega_splits)')
    w_fig = max(11.6, fig_width(drawn_a))
    fig = plt.figure(figsize=(w_fig, 10.8), facecolor='white')
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.985, top=0.955, bottom=0.115, hspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    style.bar_panel(
        ax_a, drawn_a, t_a_st, ('stabilizing', 'destabilizing'),
        (
            count_text(t_a_st, 'stabilizing', 'stabilizing, ddG > 0'),
            count_text(t_a_st, 'destabilizing', 'destabilizing, ddG < 0'),
        ),
        "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
        overall_df=ov_a_st, overall_label=f'overall (n = {int(ov_a_st.n_variants.max()):,})',
        zero_line=True, tick_fs=8.5, cats=cats_mega,
    )
    style.bar_panel(
        ax_b, drawn_b, t_b, ('additive', 'epistatic'),
        (
            count_text(t_b, 'additive', 'additive: singles model summed'),
            count_text(t_b, 'epistatic', 'epistatic: model scores the double'),
        ),
        "Spearman's ρ, double mutants", (0.0, 1.12),
        tick_fs=8.5, cats=cats_mega,
    )
    if len(p_epi_ser):
        top_y = _draw_pair_brackets(ax_b, drawn_b, t_b, p_epi_ser, p_name_map={}, B_p=B)
        if top_y > 1.12:
            ax_b.set_ylim(0.0, top_y)

    for ax_s in (ax_a, ax_b):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    fig.canvas.draw()
    pa = ax_a.get_position()
    pb = ax_b.get_position()
    fig.text(0.012, pa.y1 + 0.030, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(0.012, pb.y1 + 0.030, 'b', ha='left', va='top', fontsize=16, fontweight='bold')

    supp_dir = results_dir('supplementary_figs')
    out_path = supp_dir / '_mega_splits_fig1'
    style.save(fig, out_path)

    # Export source data (with Pearson's r, RMSE, and sample sizes) + paired one-sided bootstrap p-value tables
    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)

    cells_a_full = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all'), 'overall': (use, 'all')}
    min_n_a = {'stabilizing': MIN_N, 'destabilizing': MIN_N, 'overall': 3}
    t_a_enriched = _enrich_summary_table(fr, t_a_all, cells_a_full, min_n_a, name_map)
    t_b_enriched = _enrich_summary_table(fr, t_b, cells_b, {'additive': MIN_N, 'epistatic': MIN_N}, name_map)
    atomic_to_csv(t_a_enriched, src_dir / '_mega_splits_fig1_panel_a_summary_sourcedata.csv', index=False, float_format='%.4f')
    atomic_to_csv(t_b_enriched, src_dir / '_mega_splits_fig1_panel_b_summary_sourcedata.csv', index=False, float_format='%.4f')
    if len(p_epi_df):
        p_epi_out = p_epi_df.copy()
        p_epi_out['model'] = p_epi_out['model'].map(lambda x: name_map.get(x, x))
        atomic_to_csv(p_epi_out, src_dir / '_mega_splits_fig1_panel_b_brackets_sourcedata.csv', index=False, float_format='%.6g')

    # Per-domain source data + ESM-MSR vs comparator paired one-sided bootstrap p-values
    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    # Panel a per-domain + p-values
    per_dom_a = []
    vals_a: dict[tuple[str, str], pd.DataFrame] = {}
    r_a: dict[tuple[str, str], float] = {}
    e_a: dict[tuple[str, str], float] = {}
    n_var_a = {}
    for sub_name, (m_sub, which) in cells_a_full.items():
        mn = min_n_a[sub_name]
        n_var_a[sub_name] = int((m_sub & np.isfinite(y)).sum())
        for c in drawn_a:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            disp = name_map.get(c['name'], c['name'])
            df_v = per_library(fr, k, m_sub, rho_fn, mn)
            df_r = per_library(fr, k, m_sub, _pearson_fn, mn)
            df_e = per_library(fr, k, m_sub, _rmse_fn, mn)
            vals_a[(sub_name, disp)] = df_v
            r_a[(sub_name, disp)] = float(df_r.mean(axis=1).mean()) if len(df_r) else np.nan
            e_a[(sub_name, disp)] = float(df_e.mean(axis=1).mean()) if len(df_e) else np.nan
            for dom, row in df_v.iterrows():
                for s_i, val in enumerate(row.values):
                    per_dom_a.append(dict(
                        subset=sub_name, model=disp, key=k, domain=dom, seed=s_i + 1,
                        spearman_rho=float(val),
                        pearson_r=float(df_r.loc[dom].iloc[s_i]) if dom in df_r.index else np.nan,
                        rmse=float(df_e.loc[dom].iloc[s_i]) if dom in df_e.index else np.nan,
                    ))
    atomic_to_csv(pd.DataFrame(per_dom_a), src_dir / '_mega_splits_fig1_panel_a_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    p_rows_a = []
    for sub_name in ('overall', 'stabilizing', 'destabilizing'):
        models_here = [name_map.get(c['name'], c['name']) for c in drawn_a if (sub_name, name_map.get(c['name'], c['name'])) in vals_a]
        if 'ESM-MSR' not in models_here or len(models_here) < 2:
            continue
        common_doms = sorted(set.intersection(*(set(vals_a[(sub_name, m)].index) for m in models_here)))
        d_cnt, s_w = make_domain_seed_draws(len(common_doms), B=B, seed=301 + len(p_rows_a), seed_counts=(3,))
        pt_map, rep_map = {}, {}
        for m in models_here:
            arr = vals_a[(sub_name, m)].reindex(common_doms).to_numpy(float)
            pt_map[m], rep_map[m] = boot_domain_seeds(arr, B=B, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
        comps = [m for m in models_here if m != 'ESM-MSR']
        m_clean = len(comps)
        p_by_m = {}
        deltas = []
        for comp in comps:
            d_arr = rep_map['ESM-MSR'] - rep_map[comp]
            p_one, nf, b_eff = _one_sided_p(d_arr)
            p_by_m[comp] = p_one
            deltas.append(d_arr)
            lo_d, hi_d = ci(d_arr)
            p_rows_a.append(dict(
                benchmark='MegaScale_mega_splits_test', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
                resampling_unit='domain_cluster_x_seed', test_type='pairwise',
                msr_model='ESM-MSR', comparator=comp, direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[comp],
                pearson_r_msr=r_a[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_a[(sub_name, comp)],
                rmse_msr=e_a[(sub_name, 'ESM-MSR')], rmse_comparator=e_a[(sub_name, comp)],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[comp], ci95_lo=lo_d, ci95_hi=hi_d,
                n_units=len(common_doms), n_variants=n_var_a[sub_name], B=b_eff, n_boot_fail=nf,
                p_one_sided_uncorrected=p_one, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_one),
            ))
        best_c = max(comps, key=lambda x: pt_map[x])
        d_nb = rep_map['ESM-MSR'] - rep_map[best_c]
        p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
        lo_nb, hi_nb = ci(d_nb)
        p_rows_a.append(dict(
            benchmark='MegaScale_mega_splits_test', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed', test_type='next_best_clean',
            msr_model='ESM-MSR', comparator=best_c, direction='ESM-MSR > comparator',
            value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[best_c],
            pearson_r_msr=r_a[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_a[(sub_name, best_c)],
            rmse_msr=e_a[(sub_name, 'ESM-MSR')], rmse_comparator=e_a[(sub_name, best_c)],
            delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[best_c], ci95_lo=lo_nb, ci95_hi=hi_nb,
            n_units=len(common_doms), n_variants=n_var_a[sub_name], B=b_nb, n_boot_fail=nf_nb,
            p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
            p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
        ))
        worst_c = max(comps, key=lambda x: p_by_m[x])
        p_iut = p_by_m[worst_c]
        lo_w, hi_w = ci(rep_map['ESM-MSR'] - rep_map[worst_c])
        p_rows_a.append(dict(
            benchmark='MegaScale_mega_splits_test', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed', test_type='all_clean_IUT',
            msr_model='ESM-MSR', comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})', direction='ESM-MSR > comparator',
            value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[worst_c],
            pearson_r_msr=r_a[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_a[(sub_name, worst_c)],
            rmse_msr=e_a[(sub_name, 'ESM-MSR')], rmse_comparator=e_a[(sub_name, worst_c)],
            delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[worst_c], ci95_lo=lo_w, ci95_hi=hi_w,
            n_units=len(common_doms), n_variants=n_var_a[sub_name], B=B, n_boot_fail=int(np.sum((rep_map['ESM-MSR'] - rep_map[worst_c]) <= 0)),
            p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
        ))
    atomic_to_csv(pd.DataFrame(p_rows_a), pval_dir / '_mega_splits_fig1a_stab_destab_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_mega_splits_fig1a_stab_destab_pvalues_annotation.txt',
        f"""Statistical Annotation: _mega_splits_fig1a_stab_destab_pvalues.csv
=====================================================================
Figure / Panel: Supplementary Figure (_mega_splits_fig1), Panel a (Full MegaScale `mega_splits` test set: overall, stabilizing ddG > 0, and destabilizing ddG < 0 variants).

Metric Tested:
  Mean per-domain Spearman's rho across the {len(libs)} `mega_splits` test domains (with mean per-domain Pearson's r and RMSE in kcal/mol also reported).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B:,}, shared Multinomial domain weights across all models; 3-seed retrained transfer models resampled jointly over training seeds).
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction adjusts within each subset over all m = {len(drawn_a) - 1} comparator models present.
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j) for beating all comparators simultaneously.
""",
    )

    # Panel b per-domain + p-values
    per_dom_b = []
    vals_b: dict[tuple[str, str], pd.DataFrame] = {}
    r_b: dict[tuple[str, str], float] = {}
    e_b: dict[tuple[str, str], float] = {}
    n_var_b = int((double & np.isfinite(y)).sum())
    for sub_name, (m_sub, which) in cells_b.items():
        for c in drawn_b:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            disp = name_map.get(c['name'], c['name'])
            df_v = per_library(fr, k, m_sub, rho_fn, MIN_N)
            df_r = per_library(fr, k, m_sub, _pearson_fn, MIN_N)
            df_e = per_library(fr, k, m_sub, _rmse_fn, MIN_N)
            vals_b[(sub_name, disp)] = df_v
            r_b[(sub_name, disp)] = float(df_r.mean(axis=1).mean()) if len(df_r) else np.nan
            e_b[(sub_name, disp)] = float(df_e.mean(axis=1).mean()) if len(df_e) else np.nan
            for dom, row in df_v.iterrows():
                for s_i, val in enumerate(row.values):
                    per_dom_b.append(dict(
                        subset=sub_name, model=disp, key=k, domain=dom, seed=s_i + 1,
                        spearman_rho=float(val),
                        pearson_r=float(df_r.loc[dom].iloc[s_i]) if dom in df_r.index else np.nan,
                        rmse=float(df_e.loc[dom].iloc[s_i]) if dom in df_e.index else np.nan,
                    ))
    atomic_to_csv(pd.DataFrame(per_dom_b), src_dir / '_mega_splits_fig1_panel_b_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    p_rows_b = []
    for sub_name in ('epistatic', 'additive'):
        models_here = [name_map.get(c['name'], c['name']) for c in drawn_b if (sub_name, name_map.get(c['name'], c['name'])) in vals_b]
        if 'ESM-MSR' not in models_here or len(models_here) < 2:
            continue
        common_doms = sorted(set.intersection(*(set(vals_b[(sub_name, m)].index) for m in models_here)))
        d_cnt, s_w = make_domain_seed_draws(len(common_doms), B=B, seed=401 + len(p_rows_b), seed_counts=(3,))
        pt_map, rep_map = {}, {}
        for m in models_here:
            arr = vals_b[(sub_name, m)].reindex(common_doms).to_numpy(float)
            pt_map[m], rep_map[m] = boot_domain_seeds(arr, B=B, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
        comps = [m for m in models_here if m != 'ESM-MSR']
        m_clean = len(comps)
        p_by_m = {}
        for comp in comps:
            d_arr = rep_map['ESM-MSR'] - rep_map[comp]
            p_one, nf, b_eff = _one_sided_p(d_arr)
            p_by_m[comp] = p_one
            lo_d, hi_d = ci(d_arr)
            p_rows_b.append(dict(
                benchmark='MegaScale_mega_splits_test_doubles', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
                resampling_unit='domain_cluster_x_seed', test_type='pairwise',
                msr_model='ESM-MSR', comparator=comp, direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[comp],
                pearson_r_msr=r_b[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_b[(sub_name, comp)],
                rmse_msr=e_b[(sub_name, 'ESM-MSR')], rmse_comparator=e_b[(sub_name, comp)],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[comp], ci95_lo=lo_d, ci95_hi=hi_d,
                n_units=len(common_doms), n_variants=n_var_b, B=b_eff, n_boot_fail=nf,
                p_one_sided_uncorrected=p_one, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_one),
            ))
        best_c = max(comps, key=lambda x: pt_map[x])
        d_nb = rep_map['ESM-MSR'] - rep_map[best_c]
        p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
        lo_nb, hi_nb = ci(d_nb)
        p_rows_b.append(dict(
            benchmark='MegaScale_mega_splits_test_doubles', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed', test_type='next_best_clean',
            msr_model='ESM-MSR', comparator=best_c, direction='ESM-MSR > comparator',
            value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[best_c],
            pearson_r_msr=r_b[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_b[(sub_name, best_c)],
            rmse_msr=e_b[(sub_name, 'ESM-MSR')], rmse_comparator=e_b[(sub_name, best_c)],
            delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[best_c], ci95_lo=lo_nb, ci95_hi=hi_nb,
            n_units=len(common_doms), n_variants=n_var_b, B=b_nb, n_boot_fail=nf_nb,
            p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
            p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
        ))
        worst_c = max(comps, key=lambda x: p_by_m[x])
        p_iut = p_by_m[worst_c]
        lo_w, hi_w = ci(rep_map['ESM-MSR'] - rep_map[worst_c])
        p_rows_b.append(dict(
            benchmark='MegaScale_mega_splits_test_doubles', subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed', test_type='all_clean_IUT',
            msr_model='ESM-MSR', comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})', direction='ESM-MSR > comparator',
            value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[worst_c],
            pearson_r_msr=r_b[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_b[(sub_name, worst_c)],
            rmse_msr=e_b[(sub_name, 'ESM-MSR')], rmse_comparator=e_b[(sub_name, worst_c)],
            delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[worst_c], ci95_lo=lo_w, ci95_hi=hi_w,
            n_units=len(common_doms), n_variants=n_var_b, B=B, n_boot_fail=int(np.sum((rep_map['ESM-MSR'] - rep_map[worst_c]) <= 0)),
            p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
        ))
    atomic_to_csv(pd.DataFrame(p_rows_b), pval_dir / '_mega_splits_fig1b_doubles_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_mega_splits_fig1b_doubles_pvalues_annotation.txt',
        f"""Statistical Annotation: _mega_splits_fig1b_doubles_pvalues.csv
==================================================================
Figure / Panel: Supplementary Figure (_mega_splits_fig1), Panel b (Full MegaScale `mega_splits` test set double mutants: additive vs epistatic scoring).

Metric Tested:
  Mean per-domain Spearman's rho across double-mutant test domains (>= {MIN_N} doubles), comparing ESM-MSR against every comparator under both epistatic (direct double) and additive (summed singles) scoring (with mean per-domain Pearson's r and RMSE in kcal/mol also reported).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B:,}, shared Multinomial domain weights across all models; 3-seed retrained transfer models resampled jointly over training seeds).
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction adjusts within each subset over all m = {len(drawn_b) - 1} comparator models present.
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j) for beating all comparators simultaneously.
""",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Build Main-Text Figure 1 (architecture schematic + MegaScale bars).')
    ap.add_argument('--split', choices=('capped', 'mega'), default='capped')
    ap.add_argument('--scale', type=float, default=4.0, help='Rasterization scale for the SVG schematic (default: 4.0).')
    ap.add_argument('--supp-mega', action='store_true', help='Build supplementary mega_splits Figure 1 (no Panel a, no suptitle) + tables.')
    ap.add_argument('--B', type=int, default=10_000, help='Bootstrap replicates for --supp-mega (default: 10000).')
    args = ap.parse_args(argv)
    if args.supp_mega:
        out = build_supp_mega_splits_figure1(B=args.B, refresh=True)
        print(f'[main_figures:figure1] Saved supplementary mega_splits Figure 1 -> {out}.{{png,pdf}}')
        return 0
    out = build_figure1(split=args.split, schematic_scale=args.scale)
    print(f'[main_figures:figure1] Saved {out}.{{png,pdf}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
