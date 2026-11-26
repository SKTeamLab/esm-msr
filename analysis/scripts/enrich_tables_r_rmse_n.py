#!/usr/bin/env python3
"""Enrich all major summary source-data tables and p-value tables with Pearson's r,
RMSE, and exact sample sizes (n_variants / n_units / n_items), without additional
significance testing.

Usage:
    PYTHONPATH=src:analysis python analysis/scripts/enrich_tables_r_rmse_n.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm_msr import epi_metrics as E
from msr_analysis import models as M
from msr_analysis import proteingym as pg_mod
from msr_analysis.common import epi_rows
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import EXT_ORDER, fresh_frames, seed_frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.heldout import excluded as heldout_excluded
from msr_analysis.paths import PGYM, PUBLIC, REFERENCE, RETRAINED, ROOT, atomic_to_csv, results_dir
from msr_analysis.registry import EXT12, LABEL
from pipelines._tsu_common import (
    BEST_LABEL,
    MIN_N,
    best_other,
    key_for,
    load_tsu_test_frame,
    main_columns,
    pair_columns,
    per_library,
)
from pipelines import p04_mgnify_rocklin as p04
from pipelines import p05_summary_main as p05
from pipelines import p06_epistasis_hierarchy as p06
from pipelines import p07_epistasis_mechanisms as p07
from pipelines import p09_screening as p09

MAIN_SRC = results_dir('main_text_figs') / 'source_data'
MAIN_PVAL = results_dir('main_text_figs') / 'pvalues'
SUPP_SRC = results_dir('supplementary_figs') / 'source_data'
SUPP_PVAL = results_dir('supplementary_figs') / 'pvalues'


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    return float(pearsonr(x[ok], y[ok])[0])


def safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    return float(spearmanr(x[ok], y[ok])[0])


def pooled_r_rmse_n(preds_2d: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, int]:
    p = np.atleast_2d(np.asarray(preds_2d, dtype=float))
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(yv) & np.isfinite(p).all(axis=0)
    if mask is not None:
        ok = ok & np.asarray(mask, dtype=bool)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    rs = [safe_pearson(p[s, ok], yv[ok]) for s in range(p.shape[0])]
    rms = [safe_rmse(p[s, ok], yv[ok]) for s in range(p.shape[0])]
    return float(np.nanmean(rs)), float(np.nanmean(rms)), n


def per_domain_r_rmse_n(fr: Frame, key: str, mask: np.ndarray, min_n: int = MIN_N) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    sp = fr.spine
    y = sp['y'].to_numpy(float)
    g = sp['group'].to_numpy()
    p = np.atleast_2d(fr.preds[key])
    r_map, rm_map, n_map = {}, {}, {}
    for lib in np.unique(g[mask]):
        m = mask & (g == lib) & np.isfinite(y) & np.isfinite(p).all(axis=0)
        n_m = int(m.sum())
        if n_m < min_n:
            continue
        r_map[lib] = [safe_pearson(p[s, m], y[m]) for s in range(p.shape[0])]
        rm_map[lib] = [safe_rmse(p[s, m], y[m]) for s in range(p.shape[0])]
        n_map[lib] = n_m
    df_r = pd.DataFrame.from_dict(r_map, orient='index', dtype=float) if r_map else pd.DataFrame(dtype=float)
    df_rm = pd.DataFrame.from_dict(rm_map, orient='index', dtype=float) if rm_map else pd.DataFrame(dtype=float)
    s_n = pd.Series(n_map, dtype=int)
    return df_r, df_rm, s_n


def _attach_companion_cols_to_pval(
    pval_csv: Path,
    lookup_fn,
    n_var_fn=None,
) -> None:
    """Attach pearson_r_msr, pearson_r_comparator, rmse_msr, rmse_comparator (and fill missing n_variants)."""
    df = pd.read_csv(pval_csv)
    r_m, r_c, rm_m, rm_c, nv_list = [], [], [], [], []
    for row in df.itertuples():
        rm_val, rc_val, rms_m, rms_c = lookup_fn(row)
        r_m.append(rm_val)
        r_c.append(rc_val)
        rm_m.append(rms_m)
        rm_c.append(rms_c)
        if n_var_fn is not None:
            nv = n_var_fn(row)
            nv_list.append(nv if (nv is not None and np.isfinite(nv)) else row.n_variants)
        else:
            nv_list.append(row.n_variants)
    df['pearson_r_msr'] = r_m
    df['pearson_r_comparator'] = r_c
    df['rmse_msr'] = rm_m
    df['rmse_comparator'] = rm_c
    df['n_variants'] = nv_list
    atomic_to_csv(df, pval_csv, index=False, float_format='%.6g')


def _append_note_once(ann_path: Path, extra_note: str) -> None:
    if not ann_path.exists():
        return
    txt = ann_path.read_text(encoding='utf-8')
    marker = 'Companion Point Metrics (No Additional Significance Testing):'
    if marker in txt:
        return
    txt = txt.rstrip() + '\n\n' + marker + '\n' + extra_note.strip() + '\n'
    ann_path.write_text(txt, encoding='utf-8')


# --------------------------------------------------------------------------------------
# 1. Figure 1b, 1c + Supp Fig 2, 3, 4 (all on MegaScale capped test)
# --------------------------------------------------------------------------------------
def enrich_megascale_tables() -> None:
    print('[1/6] Enriching Figure 1b, 1c and Supp Figs 2, 3, 4 (MegaScale capped test)...')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    inv_name_map = {v: k for k, v in name_map.items()}
    use = sp['use'].to_numpy()
    y = sp['y'].to_numpy(float)
    single = use & (sp['n_mut'] == 1).to_numpy()
    double = use & (sp['n_mut'] == 2).to_numpy()

    cols_b = main_columns(fr, best, 'unmasked')
    cols_c = pair_columns(fr, best)

    # --- Fig 1b ---
    sub_b = {
        'overall': (use, 'all', 3),
        'stabilizing': (use & (y > 0), 'all', MIN_N),
        'destabilizing': (use & (y < 0), 'all', MIN_N),
    }
    stats_1b: dict[tuple[str, str], tuple[float, float, int, int]] = {}  # (subset, col_or_disp) -> (r, rmse, n_var, n_dom)
    dom_1b: dict[tuple[str, str, str, int], tuple[float, float, int]] = {}
    for sub_name, (m_sub, which, mn) in sub_b.items():
        for c in cols_b:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            df_r, df_rm, s_n = per_domain_r_rmse_n(fr, k, m_sub, min_n=mn)
            r_mean = float(df_r.mean(axis=1).mean()) if not df_r.empty else np.nan
            rm_mean = float(df_rm.mean(axis=1).mean()) if not df_rm.empty else np.nan
            n_v = int(s_n.sum())
            n_d = int(len(s_n))
            disp = name_map.get(c['name'], c['name'])
            stats_1b[(sub_name, c['name'])] = (r_mean, rm_mean, n_v, n_d)
            stats_1b[(sub_name, disp)] = (r_mean, rm_mean, n_v, n_d)
            for dom in df_r.index:
                for si in range(df_r.shape[1]):
                    dom_1b[(sub_name, disp, dom, si + 1)] = (float(df_r.loc[dom].iloc[si]), float(df_rm.loc[dom].iloc[si]), int(s_n.loc[dom]))

    # Update Fig 1b summary & per-domain source data
    p_s1b = MAIN_SRC / 'figure1_panel_b_summary_sourcedata.csv'
    df_s1b = pd.read_csv(p_s1b)
    df_s1b['pearson_r'] = [stats_1b.get((r.subset, r.column), (np.nan,) * 4)[0] for r in df_s1b.itertuples()]
    df_s1b['rmse'] = [stats_1b.get((r.subset, r.column), (np.nan,) * 4)[1] for r in df_s1b.itertuples()]
    atomic_to_csv(df_s1b, p_s1b, index=False, float_format='%.6g')

    p_d1b = MAIN_SRC / 'figure1_panel_b_per_domain_sourcedata.csv'
    df_d1b = pd.read_csv(p_d1b)
    df_d1b['pearson_r'] = [dom_1b.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[0] for r in df_d1b.itertuples()]
    df_d1b['rmse'] = [dom_1b.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[1] for r in df_d1b.itertuples()]
    df_d1b['n_variants'] = [dom_1b.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[2] for r in df_d1b.itertuples()]
    atomic_to_csv(df_d1b, p_d1b, index=False, float_format='%.6g')

    def _lk_1b(row):
        sub = str(row.subset_or_metric).replace('per_domain_mean_spearman_', '')
        rm_m, rms_m, _, _ = stats_1b.get((sub, 'ESM-MSR'), (np.nan,) * 4)
        comp = str(row.comparator)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _, _ = stats_1b.get((sub, comp), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure1b_megascale_stab_destab_pvalues.csv', _lk_1b)
    _append_note_once(
        MAIN_PVAL / 'figure1b_megascale_stab_destab_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, and `rmse_comparator` report the\n'
        '  corresponding per-domain mean Pearson r and per-domain mean RMSE (kcal/mol, seed-averaged) on the\n'
        '  exact same domains and variants; `n_units` = domains, `n_variants` = total variants. No additional\n'
        '  significance testing was performed on Pearson r or RMSE.',
    )

    # --- Fig 1c ---
    sub_c = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    stats_1c: dict[tuple[str, str], tuple[float, float, int, int]] = {}
    dom_1c: dict[tuple[str, str, str, int], tuple[float, float, int]] = {}
    for mode, (m_sub, which) in sub_c.items():
        for c in cols_c:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            df_r, df_rm, s_n = per_domain_r_rmse_n(fr, k, m_sub, min_n=MIN_N)
            r_mean = float(df_r.mean(axis=1).mean()) if not df_r.empty else np.nan
            rm_mean = float(df_rm.mean(axis=1).mean()) if not df_rm.empty else np.nan
            n_v, n_d = int(s_n.sum()), int(len(s_n))
            disp = name_map.get(c['name'], c['name'])
            stats_1c[(mode, c['name'])] = (r_mean, rm_mean, n_v, n_d)
            stats_1c[(mode, disp)] = (r_mean, rm_mean, n_v, n_d)
            for dom in df_r.index:
                for si in range(df_r.shape[1]):
                    dom_1c[(mode, disp, dom, si + 1)] = (float(df_r.loc[dom].iloc[si]), float(df_rm.loc[dom].iloc[si]), int(s_n.loc[dom]))

    p_s1c = MAIN_SRC / 'figure1_panel_c_summary_sourcedata.csv'
    df_s1c = pd.read_csv(p_s1c)
    df_s1c['pearson_r'] = [stats_1c.get((r.subset, r.column), (np.nan,) * 4)[0] for r in df_s1c.itertuples()]
    df_s1c['rmse'] = [stats_1c.get((r.subset, r.column), (np.nan,) * 4)[1] for r in df_s1c.itertuples()]
    atomic_to_csv(df_s1c, p_s1c, index=False, float_format='%.6g')

    p_d1c = MAIN_SRC / 'figure1_panel_c_per_domain_sourcedata.csv'
    df_d1c = pd.read_csv(p_d1c)
    df_d1c['pearson_r'] = [dom_1c.get((r.mode, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[0] for r in df_d1c.itertuples()]
    df_d1c['rmse'] = [dom_1c.get((r.mode, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[1] for r in df_d1c.itertuples()]
    df_d1c['n_variants'] = [dom_1c.get((r.mode, r.model, r.domain, int(r.seed)), (np.nan, np.nan, np.nan))[2] for r in df_d1c.itertuples()]
    atomic_to_csv(df_d1c, p_d1c, index=False, float_format='%.6g')

    def _lk_1c(row):
        sm = str(row.subset_or_metric)
        if sm == 'within_model_epistatic_minus_additive':
            m_base = str(row.msr_model).replace(' (epistatic)', '')
            rm_e, rms_e, _, _ = stats_1c.get(('epistatic', m_base), (np.nan,) * 4)
            rm_a, rms_a, _, _ = stats_1c.get(('additive', m_base), (np.nan,) * 4)
            return rm_e, rm_a, rms_e, rms_a
        mode = sm.replace('per_domain_mean_spearman_', '')
        rm_m, rms_m, _, _ = stats_1c.get((mode, 'ESM-MSR'), (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _, _ = stats_1c.get((mode, str(row.comparator)), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure1c_doubles_add_vs_epi_pvalues.csv', _lk_1c)
    _append_note_once(
        MAIN_PVAL / 'figure1c_doubles_add_vs_epi_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, and `rmse_comparator` report mean\n'
        '  per-domain Pearson r and RMSE (kcal/mol) across the 13 doubles domains (n_variants = 13,319).',
    )

    # --- Supp Fig 2 (Shifted NDCG@96 singles, doubles, overall) ---
    sub_s2 = {'singles': (single, 'singles'), 'doubles': (double, 'doubles'), 'overall': (use, 'all')}
    stats_s2, dom_s2 = {}, {}
    for sub_name, (m_sub, which) in sub_s2.items():
        for c in cols_b:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            df_r, df_rm, s_n = per_domain_r_rmse_n(fr, k, m_sub, min_n=MIN_N)
            r_mean = float(df_r.mean(axis=1).mean()) if not df_r.empty else np.nan
            rm_mean = float(df_rm.mean(axis=1).mean()) if not df_rm.empty else np.nan
            disp = name_map.get(c['name'], c['name'])
            stats_s2[(sub_name, c['name'])] = (r_mean, rm_mean, int(s_n.sum()), int(len(s_n)))
            stats_s2[(sub_name, disp)] = (r_mean, rm_mean, int(s_n.sum()), int(len(s_n)))
            for dom in df_r.index:
                for si in range(df_r.shape[1]):
                    dom_s2[(sub_name, disp, dom, si + 1)] = (float(df_r.loc[dom].iloc[si]), float(df_rm.loc[dom].iloc[si]), int(s_n.loc[dom]))

    p_ss2 = SUPP_SRC / 'supp_figure2_ndcg96_summary_sourcedata.csv'
    df_ss2 = pd.read_csv(p_ss2)
    df_ss2['pearson_r'] = [stats_s2.get((r.subset, r.column), (np.nan,) * 4)[0] for r in df_ss2.itertuples()]
    df_ss2['rmse'] = [stats_s2.get((r.subset, r.column), (np.nan,) * 4)[1] for r in df_ss2.itertuples()]
    atomic_to_csv(df_ss2, p_ss2, index=False, float_format='%.6g')

    p_ds2 = SUPP_SRC / 'supp_figure2_ndcg96_per_domain_sourcedata.csv'
    df_ds2 = pd.read_csv(p_ds2)
    df_ds2['pearson_r'] = [dom_s2.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan,) * 3)[0] for r in df_ds2.itertuples()]
    df_ds2['rmse'] = [dom_s2.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan,) * 3)[1] for r in df_ds2.itertuples()]
    df_ds2['n_variants'] = [dom_s2.get((r.subset, r.model, r.domain, int(r.seed)), (np.nan,) * 3)[2] for r in df_ds2.itertuples()]
    atomic_to_csv(df_ds2, p_ds2, index=False, float_format='%.6g')

    def _lk_s2(row):
        sub = str(row.subset_or_metric).replace('shifted_NDCG96_', '')
        rm_m, rms_m, _, _ = stats_s2.get((sub, 'ESM-MSR'), (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _, _ = stats_s2.get((sub, str(row.comparator)), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(SUPP_PVAL / 'supp_figure2_ndcg96_pvalues.csv', _lk_s2)

    # --- Supp Fig 3 (doubles-trained vs singles-only on singles) ---
    pairs_s3 = [
        ('ESM-MSR', 'msrp0:wt', 'msrp0:comb'),
        ('MutateEverything', 'me1', 'me2'),
        ('ThermoMPNN', 'tmpnn', 'tmpnnd'),
        ('SPURS', 'spurs', 'spursm'),
    ]
    stats_s3, dom_s3 = {}, {}
    for m_nm, k_s, k_d in pairs_s3:
        for tag, k_use in (('singles_only', k_s), ('doubles_trained', k_d)):
            df_r, df_rm, s_n = per_domain_r_rmse_n(fr, k_use, single, min_n=MIN_N)
            stats_s3[(m_nm, tag)] = (float(df_r.mean(axis=1).mean()), float(df_rm.mean(axis=1).mean()), int(s_n.sum()), int(len(s_n)))
            for dom in df_r.index:
                for si in range(df_r.shape[1]):
                    dom_s3[(m_nm, tag, dom, si + 1)] = (float(df_r.loc[dom].iloc[si]), float(df_rm.loc[dom].iloc[si]), int(s_n.loc[dom]))

    p_ss3 = SUPP_SRC / 'supp_figure3_doubles_vs_singles_summary_sourcedata.csv'
    df_ss3 = pd.read_csv(p_ss3)
    df_ss3['pearson_r_singles_only'] = [stats_s3.get((r.model, 'singles_only'), (np.nan,) * 4)[0] for r in df_ss3.itertuples()]
    df_ss3['pearson_r_doubles_trained'] = [stats_s3.get((r.model, 'doubles_trained'), (np.nan,) * 4)[0] for r in df_ss3.itertuples()]
    df_ss3['rmse_singles_only'] = [stats_s3.get((r.model, 'singles_only'), (np.nan,) * 4)[1] for r in df_ss3.itertuples()]
    df_ss3['rmse_doubles_trained'] = [stats_s3.get((r.model, 'doubles_trained'), (np.nan,) * 4)[1] for r in df_ss3.itertuples()]
    df_ss3['n_variants'] = [stats_s3.get((r.model, 'singles_only'), (np.nan,) * 4)[2] for r in df_ss3.itertuples()]
    atomic_to_csv(df_ss3, p_ss3, index=False, float_format='%.6g')

    p_ds3 = SUPP_SRC / 'supp_figure3_doubles_vs_singles_per_domain_sourcedata.csv'
    df_ds3 = pd.read_csv(p_ds3)
    df_ds3['pearson_r_singles_only'] = [dom_s3.get((r.model, 'singles_only', r.domain, int(r.seed)), (np.nan,) * 3)[0] for r in df_ds3.itertuples()]
    df_ds3['pearson_r_doubles_trained'] = [dom_s3.get((r.model, 'doubles_trained', r.domain, int(r.seed)), (np.nan,) * 3)[0] for r in df_ds3.itertuples()]
    df_ds3['rmse_singles_only'] = [dom_s3.get((r.model, 'singles_only', r.domain, int(r.seed)), (np.nan,) * 3)[1] for r in df_ds3.itertuples()]
    df_ds3['rmse_doubles_trained'] = [dom_s3.get((r.model, 'doubles_trained', r.domain, int(r.seed)), (np.nan,) * 3)[1] for r in df_ds3.itertuples()]
    df_ds3['n_variants'] = [dom_s3.get((r.model, 'singles_only', r.domain, int(r.seed)), (np.nan,) * 3)[2] for r in df_ds3.itertuples()]
    atomic_to_csv(df_ds3, p_ds3, index=False, float_format='%.6g')

    def _lk_s3(row):
        if row.test_type == 'within_model_degradation':
            mb = str(row.msr_model).replace(' (singles-only)', '')
            rs, rms, _, _ = stats_s3.get((mb, 'singles_only'), (np.nan,) * 4)
            rd, rmd, _, _ = stats_s3.get((mb, 'doubles_trained'), (np.nan,) * 4)
            return rs, rd, rms, rmd
        rm_m, rms_m, _, _ = stats_s3.get(('ESM-MSR', 'doubles_trained'), (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            cb = str(row.comparator).replace(' (doubles-trained)', '')
            rm_c, rms_c, _, _ = stats_s3.get((cb, 'doubles_trained'), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(SUPP_PVAL / 'supp_figure3_doubles_vs_singles_pvalues.csv', _lk_s3)

    # --- Supp Fig 4 (Masking schemes on doubles) ---
    f_mask = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
    d_fm = (f_mask.spine['use'].to_numpy() & (f_mask.spine['n_mut'] == 2).to_numpy()) if f_mask is not None else None
    mask_specs = [
        ('ESM-MSR', 'unm', 'ESM-MSR (Unmasked)', fr, 'msrp0:comb', double),
        ('ESM-MSR', 'ind', 'ESM-MSR (One-at-a-time masking)', f_mask, 'FINAL_p1:comb', d_fm),
        ('ESM-MSR', 'mar', 'ESM-MSR (Masked marginal)', f_mask, 'FINAL_p1_marg:comb', d_fm),
        ('ESM3-small-open', 'unm', 'ESM3-small-open (Unmasked)', fr, 'msrrel@0:comb', double),
        ('ESM3-small-open', 'ind', 'ESM3-small-open (One-at-a-time)', fr, 'esm3avg', double),
        ('ESM3-small-open', 'mar', 'ESM3-small-open (Masked marginal)', fr, 'esm3mm', double),
    ]
    stats_s4, stats_s4_by_lbl = {}, {}
    for m_nm, sch, disp_lbl, fr_u, k_u, m_u in mask_specs:
        if fr_u is None or k_u not in fr_u.preds:
            continue
        df_r, df_rm, s_n = per_domain_r_rmse_n(fr_u, k_u, m_u, min_n=MIN_N)
        tup = (float(df_r.mean(axis=1).mean()), float(df_rm.mean(axis=1).mean()), int(s_n.sum()), int(len(s_n)))
        stats_s4[(m_nm, sch)] = tup
        stats_s4_by_lbl[disp_lbl] = tup

    p_ss4 = SUPP_SRC / 'supp_figure4_masking_schemes_sourcedata.csv'
    df_ss4 = pd.read_csv(p_ss4)
    df_ss4['pearson_r'] = [stats_s4.get((r.model, r.scheme), (np.nan,) * 4)[0] for r in df_ss4.itertuples()]
    df_ss4['rmse'] = [stats_s4.get((r.model, r.scheme), (np.nan,) * 4)[1] for r in df_ss4.itertuples()]
    df_ss4['n_variants'] = [stats_s4.get((r.model, r.scheme), (np.nan,) * 4)[2] for r in df_ss4.itertuples()]
    atomic_to_csv(df_ss4, p_ss4, index=False, float_format='%.6g')

    def _lk_s4(row):
        rm_m, rms_m, _, _ = stats_s4_by_lbl.get('ESM-MSR (Unmasked)', (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _, _ = stats_s4_by_lbl.get(str(row.comparator), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(SUPP_PVAL / 'supp_figure4_masking_doubles_pvalues.csv', _lk_s4)


# --------------------------------------------------------------------------------------
# 2. Figure 2e & Supp Fig 5 (5-level epistasis hierarchy: exact Pearson r, RMSE, n_items)
# --------------------------------------------------------------------------------------
def _epistasis_level_r_rmse_n(tab: E.Table, head: str = 'epi') -> dict[str, tuple[float, float, int]]:
    """Compute exact Pearson r, RMSE, and item count for all 5 pooled epistasis levels (all & dG_add > 0)."""
    out: dict[str, tuple[float, float, int]] = {}
    pred_dddg, x_pred = E.predicted_dddg(tab, head)
    meas_dddg = tab.dddG
    x_meas = tab.x
    unique_pairs = np.unique(tab.pair)
    pair_idx_list = [np.flatnonzero(tab.pair == pr) for pr in unique_pairs]

    for sfx, mask in (('', np.ones(len(meas_dddg), dtype=bool)), ('_sub0', x_meas > 0.0), ('_gt1', x_meas > 1.0)):
        ok = mask & np.isfinite(pred_dddg) & np.isfinite(meas_dddg)
        # Level 1: naive
        out['naive_rho' + sfx] = (safe_pearson(pred_dddg[ok], meas_dddg[ok]), safe_rmse(pred_dddg[ok], meas_dddg[ok]), int(ok.sum()))
        # Level 2: beyond (saturation-subtracted)
        r_meas = meas_dddg - E.smooth_on(x_meas, meas_dddg)
        r_pred = pred_dddg - E.smooth_on(x_pred, pred_dddg)
        ok_b = mask & np.isfinite(r_meas) & np.isfinite(r_pred)
        out['beyond_rho' + sfx] = (safe_pearson(r_pred[ok_b], r_meas[ok_b]), safe_rmse(r_pred[ok_b], r_meas[ok_b]), int(ok_b.sum()))

        # Pair, Line, Cell levels (matching epi_metrics.py decomposition over pair matrices >= MIN_PAIR_CELLS)
        pair_m, pair_p = [], []
        line_m, line_p = [], []
        cell_m, cell_p = [], []
        for idx in pair_idx_list:
            ii = idx[mask[idx]] if sfx else idx
            if len(ii) < E.MIN_PAIR_CELLS:
                continue
            rm_i, rp_i = r_meas[ii], r_pred[ii]
            ok_i = np.isfinite(rm_i) & np.isfinite(rp_i)
            if ok_i.sum() < E.MIN_PAIR_CELLS:
                continue
            ii = ii[ok_i]
            rm_i, rp_i = r_meas[ii], r_pred[ii]
            pm_mean, pp_mean = float(rm_i.mean()), float(rp_i.mean())
            pair_m.append(pm_mean)
            pair_p.append(pp_mean)

            # Build (row, col) matrix for line & cell residuals
            ra, rb = tab.row[ii], tab.col[ii]
            ua, ia = np.unique(ra, return_inverse=True)
            ub, ib = np.unique(rb, return_inverse=True)
            M_m = np.full((len(ua), len(ub)), np.nan)
            M_p = np.full((len(ua), len(ub)), np.nan)
            M_m[ia, ib] = rm_i - pm_mean
            M_p[ia, ib] = rp_i - pp_mean
            # Lines: rows and columns with >= MIN_LINE cells
            for axis_i in (1, 0):
                cnt = np.isfinite(M_m).sum(axis=axis_i)
                keep_l = cnt >= E.MIN_LINE
                if keep_l.any():
                    with np.errstate(invalid='ignore'):
                        lm = np.nanmean(M_m, axis=axis_i)[keep_l]
                        lp = np.nanmean(M_p, axis=axis_i)[keep_l]
                    line_m.append(lm)
                    line_p.append(lp)
            # Double-centered cells on complete block
            blk = E._complete_block(M_m)
            if blk is not None:
                bm = E._dc(M_m[np.ix_(blk[0], blk[1])])
                bp = E._dc(M_p[np.ix_(blk[0], blk[1])])
                cell_m.append(bm.ravel())
                cell_p.append(bp.ravel())

        out['pair_rho' + sfx] = (safe_pearson(pair_p, pair_m), safe_rmse(pair_p, pair_m), len(pair_m))
        lm_all = np.concatenate(line_m) if line_m else np.array([])
        lp_all = np.concatenate(line_p) if line_p else np.array([])
        out['line_rho' + sfx] = (safe_pearson(lp_all, lm_all), safe_rmse(lp_all, lm_all), len(lm_all))
        cm_all = np.concatenate(cell_m) if cell_m else np.array([])
        cp_all = np.concatenate(cell_p) if cell_p else np.array([])
        out['cell_mag_pooled' + sfx] = (safe_pearson(cp_all, cm_all), safe_rmse(cp_all, cm_all), len(cm_all))
        out['cell_mag' + sfx] = out['cell_mag_pooled' + sfx]
        if sfx == '_sub0':
            out['cell_mag_gt0'] = out['cell_mag_pooled_sub0']
            out['naive_rho_gt0'] = out['naive_rho_sub0']
            out['beyond_rho_gt0'] = out['beyond_rho_sub0']
            out['pair_rho_gt0'] = out['pair_rho_sub0']
            out['line_rho_gt0'] = out['line_rho_sub0']
    return out


def enrich_epistasis_hierarchy() -> None:
    print('[2/6] Enriching Figure 2e and Supp Fig 5 (5-level epistasis hierarchy)...')
    fr, _ = load_tsu_test_frame('capped')
    sp, p = fr.spine, fr.preds
    epi_sp = p06._prep_epi_spine(sp)
    dg_json = json.loads((REFERENCE / 'dG_wt.json').read_text())

    # Build tables for all models in Fig 2e & Supp Fig 5 (exact same construction as p06)
    f_fin = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
    tables_by_model: dict[str, list[E.Table]] = {}

    # ESM-MSR p0 & masking variants
    for disp_m, fam_k in (
        (p06.P0_EPI_NAME, 'FINAL_p0'),
        (p06.P1_EPI_NAME, 'FINAL_p1'),
        (p06.P1M_EPI_NAME, 'FINAL_p1_marg'),
    ):
        if f_fin is not None and f'{fam_k}:wt' in f_fin.preds and f'{fam_k}:comb' in f_fin.preds:
            wt_a, cb_a = f_fin.preds[f'{fam_k}:wt'], f_fin.preds[f'{fam_k}:comb']
            tables_by_model[disp_m] = [p06._make_epi_table_fast(epi_sp, wt_a[s], cb_a[s], heads=('epi', 'add')) for s in range(wt_a.shape[0])]

    # ESM-MSR ctx ctrl (using p06._ctx_rev_for_seed)
    ctx_revs = [p06._ctx_rev_for_seed(sp, 'test', sd) for sd in (1, 2, 3)]
    if f_fin is not None and 'FINAL_p0:wt' in f_fin.preds and all(r is not None for r in ctx_revs):
        wt_p0 = f_fin.preds['FINAL_p0:wt']
        tables_by_model[p06.P0_CTX_NAME] = [
            p06._make_epi_table_fast(epi_sp, wt_p0[sd - 1], 0.5 * (wt_p0[sd - 1] + ctx_revs[sd - 1]), heads=('epi', 'add'))
            for sd in (1, 2, 3)
        ]

    # Plain baselines
    plain = [
        ('ME stage 2', 'me2', 'me1'),
        ('ThermoMPNN-D', 'tmpnnd', 'tmpnn'),
        ('SPURS multi', 'spursm', 'spurs'),
        ('ESM3-small-open', 'msrrel@0:comb', 'msrrel@0:wt'),
        ('ProteinMPNN', 'pmpnn', 'pmpnn+'),
        ('GeoDDG-3D', 'geoddg', 'geoddg+'),
        ('Rosetta', 'rosetta', 'rosetta+'),
    ]
    for name, direct, add in plain:
        if direct in p and add in p:
            ns = p[direct].shape[0]
            tables_by_model[name] = [
                p06._make_epi_table_fast(epi_sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], heads=('epi', 'add'))
                for s in range(ns)
            ]

    # SPURS unfrozen (used for 'SPURS multi' in T_p0 if available)
    unf_dir = p06.BASE / 'retrained' / 'spurs_unfrozen' / 'predictions'
    if all((unf_dir / f'seed{sd}' / 'tsuboyama_test.csv').exists() for sd in (1, 2, 3)):
        ids = sp['id'].astype(str).to_numpy()
        u_tabs = []
        for sd in (1, 2, 3):
            du = pd.read_csv(unf_dir / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_single_additive', 'SPURS_multi'], low_memory=False)
            su = p06._align_id(ids, du['uid'].astype(str).to_numpy(), du['SPURS_single_additive'].to_numpy())
            mu = p06._align_id(ids, du['uid'].astype(str).to_numpy(), du['SPURS_multi'].to_numpy())
            u_tabs.append(p06._make_epi_table_fast(epi_sp, su, mu, heads=('epi', 'add')))
        tables_by_model['SPURS multi'] = u_tabs

    # -dG_add floor control
    c_exact = np.array([float(dg_json.get(k, np.nan)) for k in sp['code'].astype(str)])
    wt_true = np.where((sp['n_mut'] == 1).to_numpy(), sp['y'].to_numpy(float), sp['ddG_additive_ML'].to_numpy(float))
    tables_by_model['−dG_add'] = [p06._make_epi_table_fast(epi_sp, wt_true, -c_exact, heads=('epi', 'add'))]

    # Compute seed-averaged (pearson_r, rmse, n_items) for each (model, metric)
    epi_stats: dict[tuple[str, str], tuple[float, float, int]] = {}
    for m_nm, tlist in tables_by_model.items():
        per_s = [_epistasis_level_r_rmse_n(t, 'epi') for t in tlist]
        for met in per_s[0]:
            rs = [d[met][0] for d in per_s]
            rms = [d[met][1] for d in per_s]
            n_it = int(per_s[0][met][2])
            epi_stats[(m_nm, met)] = (float(np.nanmean(rs)), float(np.nanmean(rms)), n_it)

    disp_to_raw = {
        'ESM-MSR': p06.P0_EPI_NAME,
        'ESM-MSR p0 (unmasked)': p06.P0_EPI_NAME,
        'ESM-MSR ctx ctrl': p06.P0_CTX_NAME,
        'MutateEverything Stage 2': 'ME stage 2',
        'ThermoMPNN-D': 'ThermoMPNN-D',
        'SPURS-multi': 'SPURS multi',
        'ESM3-small-open': 'ESM3-small-open',
        'ProteinMPNN': 'ProteinMPNN',
        'GeoDDG-3D': 'GeoDDG-3D',
        'Rosetta': 'Rosetta',
        '−dG_add (floor ceiling)': '−dG_add',
    }

    # Enrich Fig 2e & Supp Fig 5 source data
    for src_csv in (MAIN_SRC / 'figure2_panel_e_epistasis_hierarchy_sourcedata.csv', SUPP_SRC / 'supp_figure5_epistasis_masking_sourcedata.csv'):
        df = pd.read_csv(src_csv)
        df['pearson_r'] = [epi_stats.get((r.model, r.metric), (np.nan, np.nan, np.nan))[0] for r in df.itertuples()]
        df['rmse'] = [epi_stats.get((r.model, r.metric), (np.nan, np.nan, np.nan))[1] for r in df.itertuples()]
        df['n_items'] = [epi_stats.get((r.model, r.metric), (np.nan, np.nan, np.nan))[2] for r in df.itertuples()]
        atomic_to_csv(df, src_csv, index=False, float_format='%.6g')

    # Enrich Fig 2e & Supp Fig 5 p-value tables
    for pval_csv in (MAIN_PVAL / 'figure2e_epistasis_hierarchy_pvalues.csv', SUPP_PVAL / 'supp_figure5_epistasis_masking_pvalues.csv'):
        def _lk_epi(row):
            met = str(row.subset_or_metric).split(' ')[0]
            rm_m, rms_m, _ = epi_stats.get((p06.P0_EPI_NAME, met), (np.nan,) * 3)
            if row.test_type in ('pairwise', 'next_best_clean'):
                raw_c = disp_to_raw.get(str(row.comparator), str(row.comparator))
                rm_c, rms_c, _ = epi_stats.get((raw_c, met), (np.nan,) * 3)
                return rm_m, rm_c, rms_m, rms_c
            return rm_m, np.nan, rms_m, np.nan

        def _nv_epi(row):
            met = str(row.subset_or_metric).split(' ')[0]
            return epi_stats.get((p06.P0_EPI_NAME, met), (np.nan, np.nan, np.nan))[2]

        _attach_companion_cols_to_pval(pval_csv, _lk_epi, n_var_fn=_nv_epi)
        _append_note_once(
            pval_csv.with_name(pval_csv.stem + '_annotation.txt'),
            '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, and `rmse_comparator` report the\n'
            '  seed-averaged Pearson r and RMSE (kcal/mol) computed on the exact same pooled residual vectors at each\n'
            '  hierarchy level; `n_variants` records the exact item count (doubles, pairs, lines, or double-centered cells).',
        )


# --------------------------------------------------------------------------------------
# 3. Figure 3a (8 external thermostability sets)
# --------------------------------------------------------------------------------------
def enrich_figure3a() -> None:
    print('[3/6] Enriching Figure 3a (8 external thermostability sets)...')
    p_src = MAIN_SRC / 'figure3_panel_a_thermostability_sourcedata.csv'
    df_src = pd.read_csv(p_src)
    sets = list(df_src['dataset'].unique())
    frames = fresh_frames(df_src)

    stats_3a: dict[tuple[str, str], tuple[float, float, int, int]] = {}  # (ds, model) -> (r, rmse, n_var, n_prot)
    for ds in sets:
        fr_s, _ = seed_frame(ds, frames[ds])
        sp = fr_s.spine
        y = sp['y'].to_numpy(float)
        use = sp['use'].to_numpy()
        cl = sp['cluster'].to_numpy()
        for m, arr in fr_s.preds.items():
            ok = use & np.isfinite(y) & np.isfinite(arr).all(axis=0)
            r_v, rm_v, n_v = pooled_r_rmse_n(arr, y, ok)
            n_p = int(len(np.unique(cl[ok])))
            stats_3a[(ds, m)] = (r_v, rm_v, n_v, n_p)
            stats_3a[(LABEL.get(ds, ds), m)] = (r_v, rm_v, n_v, n_p)

    df_src['pearson_r'] = [stats_3a.get((r.dataset, r.model), (np.nan,) * 4)[0] for r in df_src.itertuples()]
    df_src['rmse'] = [stats_3a.get((r.dataset, r.model), (np.nan,) * 4)[1] for r in df_src.itertuples()]
    df_src['n_variants'] = [stats_3a.get((r.dataset, r.model), (np.nan,) * 4)[2] for r in df_src.itertuples()]
    df_src['n_proteins'] = [stats_3a.get((r.dataset, r.model), (np.nan,) * 4)[3] for r in df_src.itertuples()]
    atomic_to_csv(df_src, p_src, index=False, float_format='%.6g')

    def _lk_3a(row):
        rm_m, rms_m, _, _ = stats_3a.get((str(row.benchmark), 'ESM-MSR'), (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _, _ = stats_3a.get((str(row.benchmark), str(row.comparator)), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure3a_thermostability_pvalues.csv', _lk_3a)
    _append_note_once(
        MAIN_PVAL / 'figure3a_thermostability_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, and `rmse_comparator` report the\n'
        '  raw (unrecalibrated) pooled Pearson r and RMSE (kcal/mol, seed-averaged) on each benchmark set;\n'
        '  `n_variants` = total variants, `n_units` = mutations (for mutation_x_seed) or protein clusters (for protein_cluster_x_seed).',
    )


# --------------------------------------------------------------------------------------
# 4. Figure 3b & 3c (ProteinGym 173, Domainome 436, MGnify Pooled 612k, Rocklin Pooled 110k)
# --------------------------------------------------------------------------------------
def _pgym_all_models_pearson_and_n() -> tuple[dict[str, float], int]:
    """Compute category-weighted mean per-assay Pearson r (matching Figure 3b PG avg) and total n_variants across the 173 held-out assays."""
    ref = pg_mod.pgym_reference(heldout=True, drop_large=True, include_fallback_small=False)
    assays = list(ref.index)
    cats = ref['category']
    seeds = (1, 2, 3)

    def _sd(*parts):
        return {k: RETRAINED.joinpath(*parts, f'seed{k}', 'proteingym') for k in seeds}

    tmp_sa_dir = _sd('thermompnn', 'predictions', 'single_additive')
    tmp_d_dir = _sd('thermompnn', 'predictions', 'epistatic_corrected')
    me1_dir = _sd('mutate_everything', 'predictions', 'stage1')
    me2_dir = _sd('mutate_everything', 'predictions', 'stage2')
    sp_dir = _sd('spurs', 'predictions')
    prosst_dir = PUBLIC / 'prosst' / 'proteingym'
    esm3dg_dir = PUBLIC / 'esm3dg' / 'proteingym'
    geo_dir = PUBLIC / 'geoddg' / 'faspr' / 'proteingym'
    pm_dirs = {k: PUBLIC / 'proteinmpnn' / 'proteingym' / {1: 'fast', 2: 'fast_seed1', 3: 'fast_seed2'}[k] for k in seeds}
    ros_csv = PUBLIC / 'rosetta' / 'raw_narval' / 'proteingym_cartesian_preds_09_11.csv'
    ros_df = pd.read_csv(ros_csv).rename(columns={'Unnamed: 0': 'key', 'cartesian_ddg_dir': 'cart'})
    ros_df['i'] = ros_df['key'].str.extract(r'^P(\d+)_')[0].astype(int)
    ros_df['pm'] = ros_df['key'].str.split('_', n=1).str[1]
    ros_map = {i: g.drop_duplicates('pm').set_index('pm')['cart'] for i, g in ros_df.groupby('i')}
    full_ref = pd.read_csv(pg_mod.PGYM_REF_CSV)
    row_of = {d: i for i, d in enumerate(full_ref['DMS_id'])}

    per_m: dict[str, list[float]] = {m: [] for m in p05.MODEL_ORDER}
    total_variants = 0

    for dms in assays:
        dt = pd.read_csv(tmp_sa_dir[1] / f'{dms}.csv', usecols=['mut_type_renumbered', 'mutant', 'n_mut', 'pred_single_additive', 'DMS_score'])
        y_full = dt['DMS_score'].to_numpy(float)
        total_variants += int(np.isfinite(y_full).sum())
        has_mm = bool((dt['n_mut'] > 1).any())
        has_dbl = bool((dt['n_mut'] == 2).any())

        # ESM-MSR sigma=1.0 & sigma=0.5 deployed
        for m_nm, sg in (('ESM-MSR', 1.0), ('ESM-MSR σ=0.5', 0.5)):
            rs = []
            for k in seeds:
                p_comb = pg_mod._comb_csv_path('FINAL_p0', k, dms, sg) if has_mm else None
                if p_comb is not None and p_comb.exists():
                    d = pd.read_csv(p_comb)
                    col = 'combined_pred' if 'combined_pred' in d.columns else 'wt_lora_pred'
                    rs.append(safe_pearson(d[col].to_numpy(float), d['DMS_score'].to_numpy(float)))
                else:
                    p_wt = pg_mod._wt_csv_path('FINAL_p0', k, dms, sg)
                    if p_wt is not None and p_wt.exists():
                        d = pd.read_csv(p_wt, usecols=['wt_lora_pred', 'DMS_score'])
                        rs.append(safe_pearson(d['wt_lora_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            per_m[m_nm].append(float(np.nanmean(rs)) if rs else np.nan)

        # ESM3-small-open (sigma=0)
        e3_f = pg_mod.ESM3_S0_PGYM_COMB / f'{dms}.csv'
        if has_mm and e3_f.exists():
            d = pd.read_csv(e3_f, usecols=['combined_pred', 'DMS_score'])
            per_m['ESM3-small-open'].append(safe_pearson(d['combined_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
        else:
            p0 = pg_mod._wt_csv_path('FINAL_p0', 1, dms, 0.0)
            if p0 is not None and p0.exists():
                d = pd.read_csv(p0, usecols=['wt_lora_pred', 'DMS_score'])
                per_m['ESM3-small-open'].append(safe_pearson(d['wt_lora_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            else:
                per_m['ESM3-small-open'].append(np.nan)

        # ThermoMPNN / -D
        if has_dbl:
            rs = []
            for k in seeds:
                f = tmp_d_dir[k] / f'{dms}.csv'
                if f.exists():
                    dtd = pd.read_csv(f, usecols=['n_mut', 'pred_single_additive', 'pred_combined', 'DMS_score'])
                    comb_t = np.where(dtd['n_mut'] == 2, dtd['pred_combined'], dtd['pred_single_additive'])
                    rs.append(safe_pearson(comb_t, dtd['DMS_score'].to_numpy(float)))
            per_m['ThermoMPNN'].append(float(np.nanmean(rs)) if rs else np.nan)
        else:
            rs = []
            for k in seeds:
                f = tmp_sa_dir[k] / f'{dms}.csv'
                if f.exists():
                    d = pd.read_csv(f, usecols=['pred_single_additive', 'DMS_score'])
                    rs.append(safe_pearson(d['pred_single_additive'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            per_m['ThermoMPNN'].append(float(np.nanmean(rs)) if rs else np.nan)

        # MutateEverything
        if has_mm:
            rs = []
            for k in seeds:
                f = me2_dir[k] / f'{dms}.csv'
                if f.exists():
                    dm2 = pd.read_csv(f, usecols=['mut_info', 'ddg_pred'])
                    m_ren = dm2['mut_info'].isin(set(dt['mut_type_renumbered'])).mean()
                    m_mut = dm2['mut_info'].isin(set(dt['mutant'])).mean()
                    kcol = 'mut_type_renumbered' if m_ren >= m_mut else 'mutant'
                    s_map = dm2.drop_duplicates('mut_info').set_index('mut_info')['ddg_pred']
                    v2 = -dt[kcol].map(s_map).to_numpy(float)
                    rs.append(safe_pearson(v2, y_full))
            per_m['MutateEverything'].append(float(np.nanmean(rs)) if rs else np.nan)
        else:
            rs = []
            for k in seeds:
                f = me1_dir[k] / f'{dms}.csv'
                if f.exists():
                    d = pd.read_csv(f, usecols=['pred_combined', 'DMS_score'])
                    rs.append(safe_pearson(d['pred_combined'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            per_m['MutateEverything'].append(float(np.nanmean(rs)) if rs else np.nan)

        # SPURS
        col_sp = 'SPURS_multi' if has_mm else 'SPURS_single_additive'
        rs = []
        for k in seeds:
            f = sp_dir[k] / f'{dms}.csv'
            if f.exists():
                d = pd.read_csv(f, usecols=[col_sp, 'DMS_score'])
                rs.append(safe_pearson(d[col_sp].to_numpy(float), d['DMS_score'].to_numpy(float)))
        per_m['SPURS'].append(float(np.nanmean(rs)) if rs else np.nan)

        # ProteinMPNN
        rs_pm = []
        for k in seeds:
            f = pm_dirs[k] / f'{dms}.csv'
            if f.exists():
                d = pd.read_csv(f, usecols=['mutant', 'mpnn_score']).drop_duplicates('mutant').set_index('mutant')['mpnn_score']
                rs_pm.append(safe_pearson(dt['mutant'].map(d).to_numpy(float), y_full))
        per_m['ProteinMPNN'].append(float(np.nanmean(rs_pm)) if rs_pm else np.nan)

        # Row-id models: ProSST (fallback to proteingym_published ProSST-2048), GeoDDG-3D, ESM3dG
        for m_nm, folder in (('ProSST', prosst_dir), ('GeoDDG-3D', geo_dir), ('ESM3ΔG', esm3dg_dir)):
            f = folder / f'{dms}.csv'
            if f.exists():
                dp = pd.read_csv(f, usecols=['row_id', 'pred'])
                rid = dp['row_id'].to_numpy(int)
                per_m[m_nm].append(safe_pearson(dp['pred'].to_numpy(float), y_full[rid]))
            elif m_nm == 'ProSST' and (PUBLIC / 'proteingym_published' / f'{dms}.csv').exists():
                pub_f = pd.read_csv(PUBLIC / 'proteingym_published' / f'{dms}.csv', usecols=['mutant', 'ProSST-2048']).drop_duplicates('mutant').set_index('mutant')['ProSST-2048']
                per_m[m_nm].append(safe_pearson(dt['mutant'].map(pub_f).to_numpy(float), y_full))
            else:
                per_m[m_nm].append(np.nan)

        # Rosetta (singles only)
        if not has_mm and row_of.get(dms) in ros_map:
            s_ros = ros_map[row_of[dms]]
            v_ros = dt['mutant'].str[1:].map(s_ros).to_numpy(float)
            per_m['Rosetta'].append(safe_pearson(v_ros, y_full))
        else:
            per_m['Rosetta'].append(np.nan)

    # Category-weighted macro average (matching PG avg in Figure 3b!)
    df_r = pd.DataFrame(per_m, index=assays)
    cat_avg: dict[str, float] = {}
    for m in p05.MODEL_ORDER:
        cat_means = [float(np.nanmean(df_r.loc[cats == c, m])) for c in pg_mod.PG_CATEGORIES if np.isfinite(df_r.loc[cats == c, m]).any()]
        cat_avg[m] = float(np.mean(cat_means)) if cat_means else np.nan
    return cat_avg, total_variants


def enrich_figure3bc() -> None:
    print('[4/6] Enriching Figure 3b & 3c (ProteinGym, Domainome, MGnify, Rocklin)...')
    # (a) ProteinGym 173 held-out assays: Pearson r + n_variants (RMSE is NaN: non-kcal/mol DMS scales)
    pg_r_map, pg_n_var = _pgym_all_models_pearson_and_n()

    # (b) Domainome 436 held-out domains
    dom_fr = load_frame('domainome')
    dom_s, _ = seed_frame('domainome', dom_fr, models=p05.MODEL_ORDER)
    hom_dom = heldout_excluded('domainome')
    sp_d = dom_s.spine.copy()
    keep_d = sp_d['use'].to_numpy() & ~sp_d['group'].astype(str).isin(hom_dom).to_numpy()
    dom_stats: dict[str, tuple[float, float, int, int]] = {}
    for m in p05.MODEL_ORDER:
        if m not in dom_s.preds:
            continue
        df_r, df_rm, s_n = per_domain_r_rmse_n(dom_s, m, keep_d, min_n=MIN_N)
        dom_stats[m] = (float(df_r.mean(axis=1).mean()), float(df_rm.mean(axis=1).mean()), int(s_n.sum()), int(len(s_n)))

    # (c) MGnify Pooled (612k) & Rocklin Pooled (110k)
    mg_stats: dict[tuple[str, str], tuple[float, float, int, int]] = {}  # (x0/x1 or ds, model) -> (r, rmse, n_var, n_scaf)
    for g_key, ds_name, cfg in (
        ('x0', 'MGnify', dict(run='all', src='mgnify')),
        ('x1', 'Rocklin designs', dict(run='rdenovo', src='rocklin_denovo')),
    ):
        d_df, preds_mg = p04._prepare_heldout_models(cfg)
        y_mg = d_df['y'].to_numpy(float)
        n_scaf = int(d_df['code_wt'].nunique())
        for mn, (_, _, P_pool) in preds_mg.items():
            if mn not in p05.INFERRED_MODEL:
                continue
            disp = p05.INFERRED_MODEL[mn]
            arr = np.vstack(P_pool)
            r_v, rm_v, n_v = pooled_r_rmse_n(arr, y_mg)
            mg_stats[(g_key, disp)] = (r_v, rm_v, n_v, n_scaf)
            mg_stats[(f'{ds_name.replace(" ", "_")}_heldout_pooled', disp)] = (r_v, rm_v, n_v, n_scaf)

    # Enrich Figure 3b source data
    p_s3b = MAIN_SRC / 'figure3_panel_b_heldout_benchmarks_sourcedata.csv'
    df_s3b = pd.read_csv(p_s3b)
    r_col, rm_col, nv_col, nu_col = [], [], [], []
    for row in df_s3b.itertuples():
        gk, m = str(row.benchmark_key), str(row.model)
        if gk == 'PG avg':
            r_col.append(pg_r_map.get(m, np.nan))
            rm_col.append(np.nan)
            nv_col.append(pg_n_var)
            nu_col.append(173)
        elif gk == 'Domainome':
            rv, rmv, nv, nu = dom_stats.get(m, (np.nan, np.nan, np.nan, np.nan))
            r_col.append(rv); rm_col.append(rmv); nv_col.append(nv); nu_col.append(nu)
        else:
            rv, rmv, nv, nu = mg_stats.get((gk, m), (np.nan, np.nan, np.nan, np.nan))
            r_col.append(rv); rm_col.append(rmv); nv_col.append(nv); nu_col.append(nu)
    df_s3b['pearson_r'] = r_col
    df_s3b['rmse'] = rm_col
    df_s3b['n_variants'] = nv_col
    df_s3b['n_units'] = nu_col
    atomic_to_csv(df_s3b, p_s3b, index=False, float_format='%.6g')

    # Enrich Figure 3b p-value table
    def _lk_3b(row):
        bm = str(row.benchmark)
        comp = str(row.comparator)
        is_pair = row.test_type in ('pairwise', 'next_best_clean')
        if bm == 'ProteinGym_heldout_173_assays':
            return pg_r_map.get('ESM-MSR', np.nan), (pg_r_map.get(comp, np.nan) if is_pair else np.nan), np.nan, np.nan
        if bm == 'Domainome_heldout_436_domains':
            rm_m, rms_m, _, _ = dom_stats.get('ESM-MSR', (np.nan,) * 4)
            rm_c, rms_c, _, _ = dom_stats.get(comp, (np.nan,) * 4) if is_pair else (np.nan,) * 4
            return rm_m, rm_c, rms_m, rms_c
        rm_m, rms_m, _, _ = mg_stats.get((bm, 'ESM-MSR'), (np.nan,) * 4)
        rm_c, rms_c, _, _ = mg_stats.get((bm, comp), (np.nan,) * 4) if is_pair else (np.nan,) * 4
        return rm_m, rm_c, rms_m, rms_c

    def _nv_3b(row):
        if str(row.benchmark) == 'ProteinGym_heldout_173_assays':
            return pg_n_var
        return row.n_variants

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure3b_heldout_benchmarks_pvalues.csv', _lk_3b, n_var_fn=_nv_3b)
    _append_note_once(
        MAIN_PVAL / 'figure3b_heldout_benchmarks_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, `rmse_comparator`, and `n_variants` report\n'
        '  companion point metrics without additional significance testing. ProteinGym RMSE is NaN because ProteinGym\n'
        '  DMS assay scores are on heterogeneous, non-kcal/mol scales (whereas Pearson r is scale-invariant).',
    )

    # --- Figure 3c: ProteinGym per-assay Pearson r across sigma = 0, 0.5, 1.0 ---
    p_s3c = MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv'
    df_s3c = pd.read_csv(p_s3c)
    seeds = (1, 2, 3)
    pr0, pr05, pr1 = [], [], []
    for row in df_s3c.itertuples():
        dms = str(row.assay)
        has_mm = bool(row.has_multi)
        # sigma = 0
        e3_f = pg_mod.ESM3_S0_PGYM_COMB / f'{dms}.csv'
        if has_mm and e3_f.exists():
            d = pd.read_csv(e3_f, usecols=['combined_pred', 'DMS_score'])
            pr0.append(safe_pearson(d['combined_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
        else:
            p0 = pg_mod._wt_csv_path('FINAL_p0', 1, dms, 0.0)
            if p0 is not None and p0.exists():
                d = pd.read_csv(p0, usecols=['wt_lora_pred', 'DMS_score'])
                pr0.append(safe_pearson(d['wt_lora_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            else:
                pr0.append(np.nan)
        for sg, dest in ((0.5, pr05), (1.0, pr1)):
            rs = []
            for k in seeds:
                p_c = pg_mod._comb_csv_path('FINAL_p0', k, dms, sg) if has_mm else None
                if p_c is not None and p_c.exists():
                    d = pd.read_csv(p_c)
                    col = 'combined_pred' if 'combined_pred' in d.columns else 'wt_lora_pred'
                    rs.append(safe_pearson(d[col].to_numpy(float), d['DMS_score'].to_numpy(float)))
                else:
                    p_w = pg_mod._wt_csv_path('FINAL_p0', k, dms, sg)
                    if p_w is not None and p_w.exists():
                        d = pd.read_csv(p_w, usecols=['wt_lora_pred', 'DMS_score'])
                        rs.append(safe_pearson(d['wt_lora_pred'].to_numpy(float), d['DMS_score'].to_numpy(float)))
            dest.append(float(np.nanmean(rs)) if rs else np.nan)
    df_s3c['pearson_r_dep_0.0'] = pr0
    df_s3c['pearson_r_dep_0.5'] = pr05
    df_s3c['pearson_r_dep_1.0'] = pr1
    df_s3c['rmse_note'] = 'NA_non_kcal_mol_DMS_scale'
    atomic_to_csv(df_s3c, p_s3c, index=False, float_format='%.6g')

    held_3c = df_s3c[~df_s3c['leaked'].astype(bool)]
    cat_3c = {}
    for cat in pg_mod.PG_CATEGORIES:
        g = held_3c[held_3c['category'] == cat]
        cat_3c[cat] = dict(
            r0=float(np.nanmean(g['pearson_r_dep_0.0'])),
            r05=float(np.nanmean(g['pearson_r_dep_0.5'])),
            r1=float(np.nanmean(g['pearson_r_dep_1.0'])),
            n_var=int(g['n'].sum()),
        )

    def _lk_3c(row):
        cat = str(row.subset_or_metric).split(':')[0]
        c = cat_3c.get(cat, {})
        m_map = {'ESM3-small-open (σ=0)': 'r0', 'ESM-MSR (σ=0.5)': 'r05', 'ESM-MSR (σ=1.0)': 'r1'}
        return c.get(m_map.get(str(row.msr_model), ''), np.nan), c.get(m_map.get(str(row.comparator), ''), np.nan), np.nan, np.nan

    def _nv_3c(row):
        cat = str(row.subset_or_metric).split(':')[0]
        return cat_3c.get(cat, {}).get('n_var', np.nan)

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure3c_proteingym_sigma_pvalues.csv', _lk_3c, n_var_fn=_nv_3c)
    _append_note_once(
        MAIN_PVAL / 'figure3c_proteingym_sigma_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, and `n_variants` report category-mean per-assay Pearson r\n'
        '  and total variants across the held-out assays in each category. RMSE is NaN (non-kcal/mol DMS scales).',
    )


# --------------------------------------------------------------------------------------
# 5. Figure 4a–d & Supp Fig 7 (screening pools: r, RMSE, n_variants)
# --------------------------------------------------------------------------------------
def enrich_figure4_and_supp7() -> None:
    print('[5/6] Enriching Figure 4a–d and Supp Fig 7...')
    specs = p09.build_family_specs('capped')
    flab = specs['fam_label_singles']
    fams_all = list(specs['fam_all'])

    # MegaScale capped test singles (excl. Cys for 4b & Supp Fig 7; all feature-annotated singles for 4a)
    fr_m, _ = load_tsu_test_frame('capped')
    sp_m = fr_m.spine
    single_m = sp_m['use'].to_numpy() & (sp_m['n_mut'].to_numpy() == 1)
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp_m['mut']], dtype=bool)
    keep_m_nocys = single_m & ~to_cys
    n_var_m_nocys = int((keep_m_nocys & np.isfinite(sp_m['y'].to_numpy())).sum())
    n_var_m_all = int((single_m & np.isfinite(sp_m['y'].to_numpy())).sum())

    # Model-level per-domain mean r & RMSE on MegaScale singles (all & excl. Cys)
    mega_all_r_rm, mega_nocys_r_rm = {}, {}
    for fam in fams_all:
        k = p09.fam_key(fam, fr_m, specs, mode='singles')
        if k is None or k not in fr_m.preds:
            continue
        disp = flab.get(fam, fam)
        for mask_u, dest in ((single_m, mega_all_r_rm), (keep_m_nocys, mega_nocys_r_rm)):
            df_r, df_rm, s_n = per_domain_r_rmse_n(fr_m, k, mask_u, min_n=MIN_N)
            tup = (float(df_r.mean(axis=1).mean()), float(df_rm.mean(axis=1).mean()), int(s_n.sum()))
            dest[fam] = tup
            dest[disp] = tup
            # Short roster name mapping for Figure 4a
            short = disp.split(' ')[0] if disp.startswith('ESM-MSR') else disp
            dest[short] = tup
    # Explicit aliases for Figure 4a model names
    for alias, src_fam in (
        ('ESM-MSR', 'msr'), ('ESM-MSR WT-view', 'msr'), ('MutateEverything', 'me'), ('ThermoMPNN', 'tm'),
        ('SPURS', 'spurs'), ('ProteinMPNN', 'pmpnn'), ('ProSST', 'prosst'), ('ESM3-small-open', 'esm3'),
        ('best other ddG', 'geoddg'), ('GeoDDG-3D', 'geoddg'), ('Rosetta', 'rosetta'),
    ):
        if src_fam in mega_all_r_rm:
            mega_all_r_rm[alias] = mega_all_r_rm[src_fam]

    # Figure 4a summary source data + p-values
    p_4a_src = MAIN_SRC / 'figure4_panel_a_summary_sourcedata.csv'
    df_4a_src = pd.read_csv(p_4a_src)
    df_4a_src['n_variants_pool'] = n_var_m_all
    df_4a_src['pool_pearson_r'] = [mega_all_r_rm.get(str(r.model), (np.nan,) * 3)[0] for r in df_4a_src.itertuples()]
    df_4a_src['pool_rmse'] = [mega_all_r_rm.get(str(r.model), (np.nan,) * 3)[1] for r in df_4a_src.itertuples()]
    atomic_to_csv(df_4a_src, p_4a_src, index=False, float_format='%.6g')

    def _lk_4a(row):
        rm_m, rms_m, _ = mega_all_r_rm.get('ESM-MSR', (np.nan,) * 3)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _ = mega_all_r_rm.get(str(row.comparator), (np.nan,) * 3)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure4a_property_gain_pvalues.csv', _lk_4a)

    # Domainome held-out singles
    fr_d = load_frame('domainome')
    hom_d = heldout_excluded('domainome')
    keep_d = fr_d.spine['use'].to_numpy() & ~fr_d.spine['group'].astype(str).isin(hom_d).to_numpy() & (fr_d.spine['n_mut'].to_numpy() == 1)
    dom_r_rm = {}
    for fam in fams_all:
        k = p09.fam_key(fam, fr_d, specs, mode='singles')
        if k is None or k not in fr_d.preds:
            continue
        df_r, df_rm, s_n = per_domain_r_rmse_n(fr_d, k, keep_d, min_n=MIN_N)
        tup = (float(df_r.mean(axis=1).mean()), float(df_rm.mean(axis=1).mean()), int(s_n.sum()))
        dom_r_rm[fam] = tup
        dom_r_rm[flab.get(fam, fam)] = tup
    n_var_dom = int((keep_d & np.isfinite(fr_d.spine['y'].to_numpy())).sum())

    # ProteinGym Activity held-out (all 38 & 16 enzymes): per-assay mean Pearson r (RMSE is NaN on DMS scales) + n_variants
    pg_fr, pg_info = p09.load_pg_activity_assays()
    pg_act_r: dict[tuple[str, str], tuple[float, float, int]] = {}
    for tag, gs in (('Figure4d_PGym_Activity_all_38', set(pg_info.index)), ('Figure4d_PGym_Activity_enzymes_16', p09.PG_ENZYME)):
        g_list = [d for d in pg_info.index if d in gs]
        lf = p09._label_frame(pg_fr, g_list, 'z_sd', +1, 'use_wt')
        keep_p = lf.spine['use'].to_numpy()
        n_vp = int((keep_p & np.isfinite(lf.spine['y'].to_numpy())).sum())
        for fam in fams_all:
            k = p09.fam_key(fam, lf, specs, mode='singles')
            if k is None or k not in lf.preds:
                continue
            df_r, _, s_n = per_domain_r_rmse_n(lf, k, keep_p, min_n=MIN_N)
            tup = (float(df_r.mean(axis=1).mean()), np.nan, n_vp)
            pg_act_r[(tag, fam)] = tup
            pg_act_r[(tag, flab.get(fam, fam))] = tup

    # Enrich Fig 4b, 4c, 4d & Supp Fig 7 source data
    for csv_p, lookup_d, n_v_def in (
        (MAIN_SRC / 'figure4_panel_b_megascale_screening_sourcedata.csv', mega_nocys_r_rm, n_var_m_nocys),
        (SUPP_SRC / 'supp_figure7_gain_law_sourcedata.csv', mega_nocys_r_rm, n_var_m_nocys),
        (MAIN_SRC / 'figure4_panel_c_domainome_screening_sourcedata.csv', dom_r_rm, n_var_dom),
        (MAIN_SRC / 'figure4_panel_d_pgym_all_screening_sourcedata.csv', {k[1]: v for k, v in pg_act_r.items() if k[0] == 'Figure4d_PGym_Activity_all_38'}, pg_act_r[('Figure4d_PGym_Activity_all_38', 'msr')][2]),
        (MAIN_SRC / 'figure4_panel_d_pgym_enzymes_screening_sourcedata.csv', {k[1]: v for k, v in pg_act_r.items() if k[0] == 'Figure4d_PGym_Activity_enzymes_16'}, pg_act_r[('Figure4d_PGym_Activity_enzymes_16', 'msr')][2]),
    ):
        if not csv_p.exists():
            continue
        df = pd.read_csv(csv_p)
        df['n_variants_pool'] = n_v_def
        df['pool_pearson_r'] = [lookup_d.get(str(r.family), (np.nan,) * 3)[0] for r in df.itertuples()]
        df['pool_rmse'] = [lookup_d.get(str(r.family), (np.nan,) * 3)[1] for r in df.itertuples()]
        atomic_to_csv(df, csv_p, index=False, float_format='%.6g')

    # Enrich Fig 4bcd & Supp Fig 7 p-value tables
    def _lk_4bcd(row):
        bm = str(row.benchmark)
        comp = str(row.comparator)
        is_pair = row.test_type in ('pairwise', 'next_best_clean')
        if bm == 'Figure4b_MegaScale_singles_nocys':
            rm_m, rms_m, _ = mega_nocys_r_rm.get('msr', (np.nan,) * 3)
            rm_c, rms_c, _ = mega_nocys_r_rm.get(comp, (np.nan,) * 3) if is_pair else (np.nan,) * 3
            return rm_m, rm_c, rms_m, rms_c
        if bm == 'Figure4c_Domainome_heldout_singles':
            rm_m, rms_m, _ = dom_r_rm.get('msr', (np.nan,) * 3)
            rm_c, rms_c, _ = dom_r_rm.get(comp, (np.nan,) * 3) if is_pair else (np.nan,) * 3
            return rm_m, rm_c, rms_m, rms_c
        rm_m, rms_m, _ = pg_act_r.get((bm, 'msr'), (np.nan,) * 3)
        rm_c, rms_c, _ = pg_act_r.get((bm, comp), (np.nan,) * 3) if is_pair else (np.nan,) * 3
        return rm_m, rm_c, rms_m, rms_c

    def _nv_4bcd(row):
        bm = str(row.benchmark)
        if bm == 'Figure4b_MegaScale_singles_nocys':
            return n_var_m_nocys
        if bm == 'Figure4c_Domainome_heldout_singles':
            return n_var_dom
        return pg_act_r.get((bm, 'msr'), (np.nan, np.nan, np.nan))[2]

    _attach_companion_cols_to_pval(MAIN_PVAL / 'figure4bcd_screening_pvalues.csv', _lk_4bcd, n_var_fn=_nv_4bcd)
    _append_note_once(
        MAIN_PVAL / 'figure4bcd_screening_pvalues_annotation.txt',
        '  Columns `pearson_r_msr`, `pearson_r_comparator`, `rmse_msr`, `rmse_comparator`, and `n_variants` report\n'
        '  the continuous per-protein/per-assay mean Pearson r, RMSE (NaN for ProteinGym Activity non-kcal/mol scales),\n'
        '  and total variant pool size underlying each screening curve.',
    )

    def _lk_s7(row):
        rm_m, rms_m, _ = mega_nocys_r_rm.get('msr', (np.nan,) * 3)
        if row.test_type in ('pairwise', 'next_best_clean'):
            rm_c, rms_c, _ = mega_nocys_r_rm.get(str(row.comparator), (np.nan,) * 3)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(SUPP_PVAL / 'supp_figure7_gain_law_pvalues.csv', _lk_s7)


# --------------------------------------------------------------------------------------
# 6. _epi_rmse_bars & _sequence_length (per-protein r & RMSE)
# --------------------------------------------------------------------------------------
def enrich_epi_rmse_and_seqlen() -> None:
    print('[6/6] Enriching _epi_rmse_bars, _epi_rmse_pvalues, and _sequence_length...')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    _, fit, meas, model_dict, _ = p07._build_pair_models(fr)
    y_epi = meas['raw']
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    main8 = [lbl for lbl, _ in p07.MAIN_DDD_GRID_MODELS if lbl in model_dict]
    ok_all = fit & np.isfinite(y_epi)
    for lbl in main8:
        ok_all &= np.isfinite(model_dict[lbl]['raw_seeds']).all(axis=0)

    epi_metrics = {}
    alias_to_main = {
        'ProteinMPNN': 'ProteinMPNN', 'ESM3-small-open': 'ESM3-small-open',
        'ESM-MSR': 'ESM-MSR', 'MutateEverything': 'MutateEverything Stage 2',
        'MutateEverything Stage 2': 'MutateEverything Stage 2',
        'ThermoMPNN': 'ThermoMPNN-D', 'ThermoMPNN-D': 'ThermoMPNN-D',
        'SPURS': 'SPURS-multi', 'SPURS-multi': 'SPURS-multi',
        'best other ddG': 'GeoDDG-3D', 'GeoDDG-3D': 'GeoDDG-3D', 'Rosetta': 'Rosetta',
    }
    for sub_k, mask_u in (('all', ok_all), ('all_doubles', ok_all), ('dG_gt0', ok_all & (dG_add > 0)), ('dG_add_gt0', ok_all & (dG_add > 0))):
        for lbl in main8:
            arr = model_dict[lbl]['raw_seeds'][:, mask_u]
            yv = y_epi[mask_u]
            rho_v = float(np.mean([safe_spearman(arr[s], yv) for s in range(arr.shape[0])]))
            r_v = float(np.mean([safe_pearson(arr[s], yv) for s in range(arr.shape[0])]))
            rm_v = float(np.mean([safe_rmse(arr[s], yv) for s in range(arr.shape[0])]))
            epi_metrics[(sub_k, lbl)] = (rho_v, r_v, rm_v, int(mask_u.sum()))

    for csv_p in (
        SUPP_SRC / '_epi_rmse_bars_sourcedata.csv',
        results_dir('epistasis', split='capped', create=False) / 'dddG_scatters' / 'main_models_dddG_rmse_bars.csv',
    ):
        df = pd.read_csv(csv_p)
        df['spearman_rho'] = [epi_metrics.get((r.subset, alias_to_main.get(str(r.column), str(r.column))), (np.nan,) * 4)[0] for r in df.itertuples()]
        df['pearson_r'] = [epi_metrics.get((r.subset, alias_to_main.get(str(r.column), str(r.column))), (np.nan,) * 4)[1] for r in df.itertuples()]
        df['rmse'] = df['value']
        atomic_to_csv(df, csv_p, index=False, float_format='%.6g')

    def _lk_rmse(row):
        sub = str(row.subset_or_metric)
        _, rm_m, rms_m, _ = epi_metrics.get((sub, 'ESM-MSR'), (np.nan,) * 4)
        if row.test_type in ('pairwise', 'next_best_clean'):
            _, rm_c, rms_c, _ = epi_metrics.get((sub, alias_to_main.get(str(row.comparator), str(row.comparator))), (np.nan,) * 4)
            return rm_m, rm_c, rms_m, rms_c
        return rm_m, np.nan, rms_m, np.nan

    _attach_companion_cols_to_pval(SUPP_PVAL / '_epi_rmse_pvalues.csv', _lk_rmse)

    # _sequence_length panel b: add per-protein pearson_r and rmse
    from pipelines import p08_calibration_length as p08
    frames_ext = {ds: load_frame(ds) for ds in EXT12}
    sl_lookup = {}
    for key, label, _col, _grp in p08.COMBINED_LEN_MODELS:
        for ds in EXT12:
            fr_e = frames_ext[ds]
            if key not in fr_e.preds:
                continue
            sp_e = fr_e.spine
            ok = sp_e['use'].to_numpy() & np.isfinite(sp_e['y'].to_numpy()) & np.isfinite(fr_e.preds[key]).all(axis=0)
            y_e = sp_e['y'].to_numpy(float)
            idx = np.flatnonzero(ok)
            cl = sp_e['cluster'].to_numpy()[idx]
            for c_nm in pd.unique(cl):
                m_idx = idx[cl == c_nm]
                if len(m_idx) < p08.MIN_PROT_N:
                    continue
                rs = [safe_pearson(p[m_idx], y_e[m_idx]) for p in fr_e.preds[key]]
                rms = [safe_rmse(p[m_idx], y_e[m_idx]) for p in fr_e.preds[key]]
                sl_lookup[(label, ds, str(c_nm))] = (float(np.nanmean(rs)), float(np.nanmean(rms)))

    for csv_p in (
        SUPP_SRC / '_sequence_length_panel_b_sourcedata.csv',
        results_dir('sequence_length', create=False) / 'per_protein_metrics_8models.csv',
    ):
        df = pd.read_csv(csv_p)
        df['pearson_r'] = [sl_lookup.get((str(r.model), str(r.dataset), str(r.cluster)), (np.nan, np.nan))[0] for r in df.itertuples()]
        df['rmse'] = [sl_lookup.get((str(r.model), str(r.dataset), str(r.cluster)), (np.nan, np.nan))[1] for r in df.itertuples()]
        atomic_to_csv(df, csv_p, index=False, float_format='%.6g')


def main() -> int:
    enrich_megascale_tables()
    enrich_epistasis_hierarchy()
    enrich_figure3a()
    enrich_figure3bc()
    enrich_figure4_and_supp7()
    enrich_epi_rmse_and_seqlen()
    print('All major summary source-data and p-value tables enriched with Pearson r, RMSE, and sample size!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
