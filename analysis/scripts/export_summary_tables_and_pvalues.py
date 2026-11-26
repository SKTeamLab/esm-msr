#!/usr/bin/env python3
"""Export complete reproduction source-data tables (*_sourcedata.csv, gitignored),
one-sided paired bootstrap p-value tables (*_pvalues.csv), and detailed statistical
annotations (*_pvalues_annotation.txt) for all major main-text and supplementary figures.

Usage:
    PYTHONPATH=src:analysis python analysis/scripts/export_summary_tables_and_pvalues.py
"""
from __future__ import annotations

import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis import models as M
from msr_analysis import screening as scr_core
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import epi_rows
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import EXT_ORDER, fresh_frames, seed_frame
from msr_analysis.heldout import excluded as heldout_excluded, flags as heldout_flags
from msr_analysis.paths import REFERENCE, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.proteingym import (
    PG_CATEGORIES,
    boot_pgym_categories,
    load_all_models_pgym,
    load_sigma_pgym,
)
from msr_analysis.registry import EXT12, LABEL, TSU_CAPPED
from msr_analysis.stats import boot_domain_seeds, make_domain_seed_draws
from pipelines._tsu_common import (
    BEST_LABEL,
    MIN_N,
    best_other,
    boot_reps,
    diff_df,
    key_for,
    load_tsu_test_frame,
    main_columns,
    make_domain_counts,
    mean_val,
    ndcg96_shifted,
    pair_columns,
    per_library,
    rho_fn,
)
from pipelines import p04_mgnify_rocklin as p04
from pipelines import p05_summary_main as p05
from pipelines import p06_epistasis_hierarchy as p06
from pipelines import p07_epistasis_mechanisms as p07
from pipelines import p09_screening as p09

B_FAST = 10000
B_MGNIFY_POOLED = 1000
WORKERS = 8

MAIN_SRC = results_dir('main_text_figs') / 'source_data'
MAIN_PVAL = results_dir('main_text_figs') / 'pvalues'
SUPP_SRC = results_dir('supplementary_figs') / 'source_data'
SUPP_PVAL = results_dir('supplementary_figs') / 'pvalues'
for _d in (MAIN_SRC, MAIN_PVAL, SUPP_SRC, SUPP_PVAL):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Shared statistical helpers
# --------------------------------------------------------------------------------------
def one_sided_boot_p(delta_draws: np.ndarray) -> tuple[float, int, int]:
    """Phipson-Smyth exact Monte Carlo one-sided p-value for H1: delta > 0 (H0: delta <= 0).

    Returns (p_one_sided, n_fail, B_eff) where n_fail = #{b: delta_b <= 0}.
    """
    d = np.asarray(delta_draws, dtype=float)
    d = d[np.isfinite(d)]
    B_eff = int(len(d))
    if B_eff == 0:
        return np.nan, 0, 0
    n_fail = int(np.sum(d <= 0.0))
    return float((n_fail + 1) / (B_eff + 1)), n_fail, B_eff


def build_comparison_rows(
    *,
    benchmark: str,
    subset_or_metric: str,
    resampling_unit: str,
    msr_label: str,
    msr_point: float,
    msr_reps: np.ndarray,
    comparators: list[dict],
    higher_is_better: bool = True,
    n_units: int,
    n_variants: int | None = None,
    bonferroni_family: str,
) -> list[dict]:
    """Build standardized pairwise + next_best_clean + all_clean_IUT + all_clean_joint_max rows.

    Each entry in `comparators` has:
      name: str
      point: float
      reps: np.ndarray (aligned paired bootstrap draws of same length as msr_reps)
      excluded_overlap: bool (default False)
      internal_variant: bool (default False; e.g. ESM-MSR sigma=0.5, ctx ctrl, oracle)
      notes: str (optional)
    """
    sign = 1.0 if higher_is_better else -1.0
    direction = 'higher_is_better' if higher_is_better else 'lower_is_better'

    clean_entries = [
        c for c in comparators
        if not c.get('excluded_overlap', False) and not c.get('internal_variant', False)
        and np.isfinite(c['point']) and np.isfinite(c['reps']).any()
    ]
    m_clean = len(clean_entries)

    rows = []
    clean_ps: dict[str, float] = {}
    clean_deltas_draws: list[np.ndarray] = []

    for c in comparators:
        d_draws = sign * (np.asarray(msr_reps, float) - np.asarray(c['reps'], float))
        fin = d_draws[np.isfinite(d_draws)]
        p_one, n_fail, B_eff = one_sided_boot_p(fin)
        lo, hi = (float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))) if B_eff > 0 else (np.nan, np.nan)
        se = float(np.std(fin, ddof=1)) if B_eff > 1 else np.nan
        delta_pt = float(sign * (msr_point - c['point']))
        is_clean = (not c.get('excluded_overlap', False)) and (not c.get('internal_variant', False))
        p_bonf = float(min(1.0, m_clean * p_one)) if (is_clean and m_clean > 0 and np.isfinite(p_one)) else np.nan

        if is_clean and np.isfinite(p_one):
            clean_ps[c['name']] = p_one
            clean_deltas_draws.append(d_draws)

        rows.append(dict(
            benchmark=benchmark,
            subset_or_metric=subset_or_metric,
            resampling_unit=resampling_unit,
            test_type='pairwise',
            msr_model=msr_label,
            comparator=c['name'],
            excluded_training_overlap=bool(c.get('excluded_overlap', False)),
            internal_control_or_variant=bool(c.get('internal_variant', False)),
            direction=direction,
            value_msr=float(msr_point),
            value_comparator=float(c['point']),
            delta_favoring_msr=delta_pt,
            ci95_lo=lo,
            ci95_hi=hi,
            se_boot=se,
            n_units=int(n_units),
            n_variants=int(n_variants) if n_variants is not None else np.nan,
            B=B_eff,
            n_boot_fail=n_fail,
            p_one_sided_uncorrected=p_one,
            bonferroni_m=int(m_clean) if is_clean else np.nan,
            p_one_sided_bonferroni=p_bonf,
            bonferroni_family=bonferroni_family if is_clean else 'excluded_from_clean_family',
            notes=c.get('notes', ''),
        ))

    if clean_entries:
        # 1. Next-best clean comparator (by point estimate)
        best_c = max(clean_entries, key=lambda c: c['point'] if higher_is_better else -c['point'])
        d_nb = sign * (np.asarray(msr_reps, float) - np.asarray(best_c['reps'], float))
        fin_nb = d_nb[np.isfinite(d_nb)]
        p_nb, nf_nb, B_nb = one_sided_boot_p(fin_nb)
        lo_nb, hi_nb = float(np.quantile(fin_nb, 0.025)), float(np.quantile(fin_nb, 0.975))
        se_nb = float(np.std(fin_nb, ddof=1)) if B_nb > 1 else np.nan
        rows.append(dict(
            benchmark=benchmark,
            subset_or_metric=subset_or_metric,
            resampling_unit=resampling_unit,
            test_type='next_best_clean',
            msr_model=msr_label,
            comparator=best_c['name'],
            excluded_training_overlap=False,
            internal_control_or_variant=False,
            direction=direction,
            value_msr=float(msr_point),
            value_comparator=float(best_c['point']),
            delta_favoring_msr=float(sign * (msr_point - best_c['point'])),
            ci95_lo=lo_nb,
            ci95_hi=hi_nb,
            se_boot=se_nb,
            n_units=int(n_units),
            n_variants=int(n_variants) if n_variants is not None else np.nan,
            B=B_nb,
            n_boot_fail=nf_nb,
            p_one_sided_uncorrected=p_nb,
            bonferroni_m=int(m_clean),
            p_one_sided_bonferroni=float(min(1.0, m_clean * p_nb)),
            bonferroni_family=bonferroni_family,
            notes=f'Next-best clean comparator by point estimate ({best_c["name"]}); uncorrected p treats comparator as fixed; Bonferroni adjusts over all m={m_clean} clean comparators.',
        ))

        # 2. Intersection-Union Test (Berger 1982): H1 = ESM-MSR beats ALL clean comparators simultaneously -> p_IUT = max_j p_j
        worst_name = max(clean_ps, key=lambda k: clean_ps[k])
        worst_c = next(c for c in clean_entries if c['name'] == worst_name)
        d_w = sign * (np.asarray(msr_reps, float) - np.asarray(worst_c['reps'], float))
        fin_w = d_w[np.isfinite(d_w)]
        p_iut, nf_w, B_w = one_sided_boot_p(fin_w)
        lo_w, hi_w = float(np.quantile(fin_w, 0.025)), float(np.quantile(fin_w, 0.975))
        se_w = float(np.std(fin_w, ddof=1)) if B_w > 1 else np.nan
        rows.append(dict(
            benchmark=benchmark,
            subset_or_metric=subset_or_metric,
            resampling_unit=resampling_unit,
            test_type='all_clean_IUT',
            msr_model=msr_label,
            comparator=f'ALL_CLEAN_IUT (limiting: {worst_name})',
            excluded_training_overlap=False,
            internal_control_or_variant=False,
            direction=direction,
            value_msr=float(msr_point),
            value_comparator=float(worst_c['point']),
            delta_favoring_msr=float(sign * (msr_point - worst_c['point'])),
            ci95_lo=lo_w,
            ci95_hi=hi_w,
            se_boot=se_w,
            n_units=int(n_units),
            n_variants=int(n_variants) if n_variants is not None else np.nan,
            B=B_w,
            n_boot_fail=nf_w,
            p_one_sided_uncorrected=p_iut,
            bonferroni_m=1,
            p_one_sided_bonferroni=p_iut,
            bonferroni_family='IUT_no_multiplicity_penalty',
            notes=f'Berger (1982) Intersection-Union Test for H1: ESM-MSR beats ALL m={m_clean} clean models simultaneously; exact level-alpha without Bonferroni (p_IUT = max_j p_j, attained by {worst_name}).',
        ))

        # 3. Joint simultaneous bootstrap: min_j (delta_j^(b)) across all clean comparators in each resample
        mat = np.vstack(clean_deltas_draws)  # (m_clean, B)
        d_min = np.nanmin(mat, axis=0)
        fin_min = d_min[np.isfinite(d_min)]
        p_joint, nf_min, B_min = one_sided_boot_p(fin_min)
        lo_min, hi_min = float(np.quantile(fin_min, 0.025)), float(np.quantile(fin_min, 0.975))
        se_min = float(np.std(fin_min, ddof=1)) if B_min > 1 else np.nan
        rows.append(dict(
            benchmark=benchmark,
            subset_or_metric=subset_or_metric,
            resampling_unit=resampling_unit,
            test_type='all_clean_joint_max',
            msr_model=msr_label,
            comparator=f'ALL_CLEAN_JOINT_MAX (max of {m_clean} clean models per draw)',
            excluded_training_overlap=False,
            internal_control_or_variant=False,
            direction=direction,
            value_msr=float(msr_point),
            value_comparator=float(best_c['point']),
            delta_favoring_msr=float(sign * (msr_point - best_c['point'])),
            ci95_lo=lo_min,
            ci95_hi=hi_min,
            se_boot=se_min,
            n_units=int(n_units),
            n_variants=int(n_variants) if n_variants is not None else np.nan,
            B=B_min,
            n_boot_fail=nf_min,
            p_one_sided_uncorrected=p_joint,
            bonferroni_m=1,
            p_one_sided_bonferroni=p_joint,
            bonferroni_family='joint_simultaneous_draw',
            notes=f'Simultaneous paired bootstrap margin min_j(delta_j^(b)) over all m={m_clean} clean models in the same resample; P_boot(any clean model >= ESM-MSR) satisfies max_j p_j <= p_joint <= min(1, sum_j p_j).',
        ))

    return rows


def save_table_and_annotation(df: pd.DataFrame, out_csv: Path, annotation_text: str) -> None:
    atomic_to_csv(df, out_csv, index=False, float_format='%.6g')
    ann_path = out_csv.with_name(out_csv.stem + '_annotation.txt')
    atomic_write_text(ann_path, annotation_text.strip() + '\n')


def _copy_as_sourcedata(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# --------------------------------------------------------------------------------------
# FIGURE 1: MegaScale stabilizing/destabilizing (b) & additive vs epistatic doubles (c)
# --------------------------------------------------------------------------------------
def export_figure1() -> None:
    print('[1/10] Exporting Figure 1 source data & p-values...')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols_b = [c for c in main_columns(fr, best, 'unmasked') if c.get('draw', True)]
    cols_c = [c for c in pair_columns(fr, best) if c.get('draw', True)]

    mega_dir = results_dir('megascale', split='capped', create=False)
    epi_dir = results_dir('epistasis', split='capped', create=False)

    # Source data: summary tables directly used to draw Fig 1b & 1c
    _copy_as_sourcedata(mega_dir / 'spearman_stab_destab.csv', MAIN_SRC / 'figure1_panel_b_summary_sourcedata.csv')
    _copy_as_sourcedata(epi_dir / 'spearman_doubles_additive_vs_epistatic.csv', MAIN_SRC / 'figure1_panel_c_summary_sourcedata.csv')
    if (epi_dir / 'epistatic_vs_additive_pvalues.csv').exists():
        _copy_as_sourcedata(epi_dir / 'epistatic_vs_additive_pvalues.csv', MAIN_SRC / 'figure1_panel_c_brackets_sourcedata.csv')

    use = sp['use'].to_numpy()
    y = sp['y'].to_numpy(float)
    double = use & (sp['n_mut'] == 2).to_numpy()

    # --- Panel b per-domain source data + B=10,000 paired domain x seed bootstrap ---
    subsets_b = {
        'overall': (use, 'all', 3),
        'stabilizing': (use & (y > 0), 'all', MIN_N),
        'destabilizing': (use & (y < 0), 'all', MIN_N),
    }
    per_dom_b_rows = []
    vals_b: dict[tuple[str, str], pd.DataFrame] = {}
    n_var_b: dict[str, int] = {}
    for sub_name, (m_sub, which, mn) in subsets_b.items():
        n_var_b[sub_name] = int((m_sub & np.isfinite(y)).sum())
        for c in cols_b:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            disp = name_map.get(c['name'], c['name'])
            df_v = per_library(fr, k, m_sub, rho_fn, mn)
            vals_b[(sub_name, disp)] = df_v
            for dom, row in df_v.iterrows():
                for s_i, val in enumerate(row.values):
                    per_dom_b_rows.append(dict(subset=sub_name, model=disp, key=k, domain=dom, seed=s_i + 1, spearman_rho=float(val)))
    atomic_to_csv(pd.DataFrame(per_dom_b_rows), MAIN_SRC / 'figure1_panel_b_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    libs_b = sorted(vals_b[('overall', 'ESM-MSR')].index.tolist())
    dom_counts_b, seed_w_b = make_domain_seed_draws(len(libs_b), B=B_FAST, seed=101, seed_counts=(3,))
    p_rows_b = []
    for sub_name in ('overall', 'stabilizing', 'destabilizing'):
        models_here = [name_map.get(c['name'], c['name']) for c in cols_b if (sub_name, name_map.get(c['name'], c['name'])) in vals_b]
        # Align on common domain set across models for this subset
        common_doms = sorted(set.intersection(*(set(vals_b[(sub_name, m)].index) for m in models_here)))
        d_cnt, s_w = make_domain_seed_draws(len(common_doms), B=B_FAST, seed=101 + len(p_rows_b), seed_counts=(3,))
        pt_map, rep_map = {}, {}
        for m in models_here:
            arr = vals_b[(sub_name, m)].reindex(common_doms).to_numpy(float)
            pt, rp = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
            pt_map[m], rep_map[m] = pt, rp
        comps = [dict(name=m, point=pt_map[m], reps=rep_map[m]) for m in models_here if m != 'ESM-MSR']
        p_rows_b += build_comparison_rows(
            benchmark='MegaScale_capped_test',
            subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=pt_map['ESM-MSR'],
            msr_reps=rep_map['ESM-MSR'],
            comparators=comps,
            higher_is_better=True,
            n_units=len(common_doms),
            n_variants=n_var_b[sub_name],
            bonferroni_family=f'Figure1b_{sub_name}_8_comparators',
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows_b),
        MAIN_PVAL / 'figure1b_megascale_stab_destab_pvalues.csv',
        f"""Statistical Annotation: figure1b_megascale_stab_destab_pvalues.csv
====================================================================
Figure / Panel: Main-Text Figure 1, Panel b (Homology-Restricted MegaScale test set: overall, stabilizing ddG > 0, and destabilizing ddG < 0 variants).

Metric Tested:
  Mean per-domain Spearman's rank correlation (rho) across the 33 held-out MegaScale test domains
  (each retrained transfer model averaged over 3 independent training seeds).

Statistical Test & Resampling Procedure:
  Paired nonparametric cluster x seed bootstrap (B = {B_FAST:,}, deterministic seed).
  In each bootstrap replicate b = 1, ..., B:
    1. All N_domains = 33 test domains are resampled with replacement using a single shared
       Multinomial(33, [1/33, ..., 1/33]) weight vector applied identically to every model (exact pairing).
    2. For 3-seed retrained models (ESM-MSR, MutateEverything, ThermoMPNN, SPURS), training seeds are
       resampled jointly with Multinomial(3, [1/3, 1/3, 1/3]) weights so both domain-sampling and
       training-seed variance are propagated.
    3. Paired difference delta^(b) = rho_ESM-MSR^(b) - rho_X^(b) is recorded for each comparator X.

Hypotheses & One-Sided P-Value Formula:
  H0: E[rho_ESM-MSR - rho_X] <= 0
  H1: E[rho_ESM-MSR - rho_X] > 0  (ESM-MSR strictly outperforms comparator X).
  Exact Monte Carlo one-sided p-value (Phipson & Smyth, 2010):
    p_one_sided_uncorrected = (1 + #{{b : delta^(b) <= 0}}) / (B + 1),
  giving minimum achievable resolution 1 / ({B_FAST + 1:,}) = {1.0 / (B_FAST + 1):.4g}.

Multiple-Testing Correction (Bonferroni):
  Within each subset (overall, stabilizing, destabilizing), p_one_sided_bonferroni = min(1, m * p_one_sided_uncorrected)
  over the m = 8 clean comparator models (ProSST, ProteinMPNN, ESM3-small-open, MutateEverything,
  ThermoMPNN, SPURS, GeoDDG-3D, Rosetta; none have MegaScale capped-test homology overlap).

Summary Test Types Included:
  - pairwise: Individual one-sided test of ESM-MSR vs each model X.
  - next_best_clean: ESM-MSR vs the single highest-rho comparator j* (by point estimate).
  - all_clean_IUT: Berger (1982) Intersection-Union Test for H1: ESM-MSR beats ALL m=8 comparators
    simultaneously; p_IUT = max_j p_j controls Type I error at level alpha without Bonferroni penalty.
  - all_clean_joint_max: Simultaneous paired bootstrap margin min_j(rho_ESM-MSR^(b) - rho_j^(b)),
    measuring the bootstrap probability that ANY comparator ties or exceeds ESM-MSR in the same draw.

Key Assumptions & Limitations:
  - Test domains (33 homology-restricted clusters) are treated as independent, exchangeable biological units.
  - Within-domain mutation correlation is fully accounted for by resampling at the domain level.
  - Resampling 3 training seeds via Multinomial(3, 1/3) is discrete (10 distinct weight triples) and captures
    between-seed spread without parametric normality assumptions.
""",
    )

    # --- Panel c per-domain source data + B=10,000 paired domain x seed bootstrap ---
    subsets_c = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    per_dom_c_rows = []
    vals_c: dict[tuple[str, str], pd.DataFrame] = {}
    n_var_c = int((double & np.isfinite(y)).sum())
    for sub_name, (m_sub, which) in subsets_c.items():
        for c in cols_c:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            disp = name_map.get(c['name'], c['name'])
            df_v = per_library(fr, k, m_sub, rho_fn, MIN_N)
            vals_c[(sub_name, disp)] = df_v
            for dom, row in df_v.iterrows():
                for s_i, val in enumerate(row.values):
                    per_dom_c_rows.append(dict(mode=sub_name, model=disp, key=k, domain=dom, seed=s_i + 1, spearman_rho=float(val)))
    atomic_to_csv(pd.DataFrame(per_dom_c_rows), MAIN_SRC / 'figure1_panel_c_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    models_c = [name_map.get(c['name'], c['name']) for c in cols_c if ('epistatic', name_map.get(c['name'], c['name'])) in vals_c]
    common_doms_c = sorted(set.intersection(*(set(vals_c[('epistatic', m)].index) for m in models_c)))
    d_cnt_c, s_w_c = make_domain_seed_draws(len(common_doms_c), B=B_FAST, seed=202, seed_counts=(3,))
    p_rows_c = []
    pt_c, rep_c = {}, {}
    for sub_name in ('epistatic', 'additive'):
        for m in models_c:
            if (sub_name, m) not in vals_c:
                continue
            arr = vals_c[(sub_name, m)].reindex(common_doms_c).to_numpy(float)
            pt, rp = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt_c, seed_weights=s_w_c if arr.shape[1] > 1 else None)
            pt_c[(sub_name, m)], rep_c[(sub_name, m)] = pt, rp
        comps = [dict(name=m, point=pt_c[(sub_name, m)], reps=rep_c[(sub_name, m)]) for m in models_c if m != 'ESM-MSR' and (sub_name, m) in pt_c]
        p_rows_c += build_comparison_rows(
            benchmark='MegaScale_capped_test_doubles',
            subset_or_metric=f'per_domain_mean_spearman_{sub_name}',
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=pt_c[(sub_name, 'ESM-MSR')],
            msr_reps=rep_c[(sub_name, 'ESM-MSR')],
            comparators=comps,
            higher_is_better=True,
            n_units=len(common_doms_c),
            n_variants=n_var_c,
            bonferroni_family=f'Figure1c_{sub_name}_comparators',
        )

    # Also within-model epistatic > additive one-sided tests (matching Figure 1c brackets)
    non_add_models = [m for m in models_c if ('additive', m) in pt_c and ('epistatic', m) in pt_c and m != 'ProSST']
    m_within = len(non_add_models)
    for m in non_add_models:
        d_w = rep_c[('epistatic', m)] - rep_c[('additive', m)]
        p_one, n_fail, B_eff = one_sided_boot_p(d_w)
        lo, hi = float(np.quantile(d_w, 0.025)), float(np.quantile(d_w, 0.975))
        se = float(np.std(d_w, ddof=1))
        p_rows_c.append(dict(
            benchmark='MegaScale_capped_test_doubles',
            subset_or_metric='within_model_epistatic_minus_additive',
            resampling_unit='domain_cluster_x_seed',
            test_type='within_model_epistatic_vs_additive',
            msr_model=f'{m} (epistatic)',
            comparator=f'{m} (additive)',
            excluded_training_overlap=False,
            internal_control_or_variant=False,
            direction='higher_is_better',
            value_msr=float(pt_c[('epistatic', m)]),
            value_comparator=float(pt_c[('additive', m)]),
            delta_favoring_msr=float(pt_c[('epistatic', m)] - pt_c[('additive', m)]),
            ci95_lo=lo,
            ci95_hi=hi,
            se_boot=se,
            n_units=len(common_doms_c),
            n_variants=n_var_c,
            B=B_eff,
            n_boot_fail=n_fail,
            p_one_sided_uncorrected=p_one,
            bonferroni_m=m_within,
            p_one_sided_bonferroni=float(min(1.0, m_within * p_one)),
            bonferroni_family=f'Figure1c_within_model_{m_within}_models',
            notes='One-sided paired bootstrap test that the epistatic double-mutant score exceeds the additive singles sum within the same model.',
        ))

    save_table_and_annotation(
        pd.DataFrame(p_rows_c),
        MAIN_PVAL / 'figure1c_doubles_add_vs_epi_pvalues.csv',
        f"""Statistical Annotation: figure1c_doubles_add_vs_epi_pvalues.csv
=================================================================
Figure / Panel: Main-Text Figure 1, Panel c (Homology-Restricted MegaScale test double mutants: additive vs epistatic ranking).

Metrics Tested:
  1. Between-model comparisons (per_domain_mean_spearman_epistatic and per_domain_mean_spearman_additive):
     Mean per-domain Spearman's rho across the 13 test domains with >= {MIN_N} double mutants ({n_var_c:,} doubles).
  2. Within-model comparisons (within_model_epistatic_minus_additive):
     Whether scoring the double mutant directly (epistatic) improves over summing singles predictions (additive)
     for each non-additive model (reproducing the bracket tests in Figure 1c).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B_FAST:,}).
  All 13 doubles domains are resampled identically across models and modes via shared Multinomial(13, 1/13)
  weights; 3-seed retrained models jointly resample training seeds via shared Multinomial(3, 1/3) weights.

Hypotheses & One-Sided P-Value Formula:
  - Between-model: H0: rho_ESM-MSR <= rho_X vs H1: rho_ESM-MSR > rho_X.
  - Within-model:  H0: rho_epistatic <= rho_additive vs H1: rho_epistatic > rho_additive.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).

Multiple-Testing Correction (Bonferroni):
  - Between-model epistatic: Bonferroni over m = 8 clean comparators (ProSST is additive-only and included in additive & epistatic panels).
  - Between-model additive: Bonferroni over m = 8 clean comparators.
  - Within-model epistatic vs additive: Bonferroni over m = {m_within} non-additive models.

Key Assumptions & Limitations:
  - Only N = 13 independent test domains contain double mutants, so domain-level bootstrap CIs are naturally
    wider than mutation-level CIs, making this test appropriately conservative against domain-specific effects.
""",
    )


# --------------------------------------------------------------------------------------
# FIGURE 2: Floor saturation scatter (b), epistasis types (c), 5-level hierarchy (e)
# --------------------------------------------------------------------------------------
def export_figure2() -> None:
    print('[2/10] Exporting Figure 2 source data & p-values...')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    m = epi_rows(fr)
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    dddG = sp['dddG_ML'].to_numpy(float)
    ok = m & np.isfinite(dG_add) & np.isfinite(dddG)
    df_b = pd.DataFrame(dict(
        id=sp['id'].to_numpy()[ok],
        code_wt=sp['code_wt'].to_numpy()[ok],
        group=sp['group'].to_numpy()[ok],
        mut=sp['mut'].to_numpy()[ok],
        dG_wt=sp['dG_wt'].to_numpy(float)[ok],
        ddG_additive_ML=sp['ddG_additive_ML'].to_numpy(float)[ok],
        dG_add=dG_add[ok],
        dddG_ML=dddG[ok],
    ))
    atomic_to_csv(df_b, MAIN_SRC / 'figure2_panel_b_floor_scatter_sourcedata.csv', index=False, float_format='%.5f')

    epi_dir = results_dir('epistasis', split='capped', create=False)
    supp_dir = results_dir('supplementary_figs', create=False)
    _copy_as_sourcedata(epi_dir / 'alt_observed_additive_dG.csv', MAIN_SRC / 'figure2_panel_c_epistasis_types_sourcedata.csv')
    _copy_as_sourcedata(supp_dir / 'fig5_epistasis_levels_p0_bootstrap_test.csv', MAIN_SRC / 'figure2_panel_e_epistasis_hierarchy_sourcedata.csv')

    # Paired p-values from exact saved domain x seed bootstrap replicates in fig5_epistasis_levels_p0_reps_test.npz
    T_p0 = pd.read_csv(supp_dir / 'fig5_epistasis_levels_p0_bootstrap_test.csv')
    z = np.load(supp_dir / 'fig5_epistasis_levels_p0_reps_test.npz', allow_pickle=True)
    reps_all = z['reps']        # (B, n_models, n_metrics)
    models_z = [str(x) for x in z['models']]
    metrics_z = [str(x) for x in z['metrics']]
    B_z = int(z['B'])
    n_dom = int(T_p0['n_domains'].iloc[0])

    disp_map = {
        p06.P0_EPI_NAME: 'ESM-MSR',
        p06.P0_CTX_NAME: 'ESM-MSR ctx ctrl',
        'ME stage 2': 'MutateEverything Stage 2',
        'ThermoMPNN-D': 'ThermoMPNN-D',
        'SPURS multi': 'SPURS-multi',
        'ESM3-small-open': 'ESM3-small-open',
        'ProteinMPNN': 'ProteinMPNN',
        'GeoDDG-3D': 'GeoDDG-3D',
        'Rosetta': 'Rosetta',
        '−dG_add': '−dG_add (floor ceiling)',
    }
    i_msr = models_z.index(p06.P0_EPI_NAME)
    levels = [
        ('naive_rho', 'all_doubles'), ('naive_rho_gt0', 'dG_add_gt0'),
        ('beyond_rho', 'all_doubles'), ('beyond_rho_gt0', 'dG_add_gt0'),
        ('pair_rho', 'all_doubles'), ('pair_rho_gt0', 'dG_add_gt0'),
        ('line_rho', 'all_doubles'), ('line_rho_gt0', 'dG_add_gt0'),
        ('cell_mag_pooled', 'all_doubles'), ('cell_mag_gt0', 'dG_add_gt0'),
    ]
    p_rows = []
    for met, regime in levels:
        if met not in metrics_z:
            continue
        j_m = metrics_z.index(met)
        pt_sub = T_p0[T_p0['metric'] == met].set_index('model')['value']
        msr_pt = float(pt_sub.get(p06.P0_EPI_NAME, np.nan))
        msr_rp = reps_all[:, i_msr, j_m]
        n_v = int(df_b.shape[0]) if regime == 'all_doubles' else int((df_b['dG_add'] > 0).sum())
        comps = []
        for idx_c, raw_m in enumerate(models_z):
            if idx_c == i_msr or raw_m not in pt_sub.index:
                continue
            is_internal = raw_m in (p06.P0_CTX_NAME, '−dG_add')
            comps.append(dict(
                name=disp_map.get(raw_m, raw_m),
                point=float(pt_sub[raw_m]),
                reps=reps_all[:, idx_c, j_m],
                internal_variant=is_internal,
                notes='Internal control / floor ceiling (excluded from clean external comparator Bonferroni family).' if is_internal else '',
            ))
        p_rows += build_comparison_rows(
            benchmark='MegaScale_epistasis_5level_hierarchy',
            subset_or_metric=f'{met} ({regime})',
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=msr_pt,
            msr_reps=msr_rp,
            comparators=comps,
            higher_is_better=True,
            n_units=n_dom,
            n_variants=n_v,
            bonferroni_family=f'Figure2e_{met}_7_external_models',
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows),
        MAIN_PVAL / 'figure2e_epistasis_hierarchy_pvalues.csv',
        f"""Statistical Annotation: figure2e_epistasis_hierarchy_pvalues.csv
==================================================================
Figure / Panel: Main-Text Figure 2, Panel e (Five-level pooled epistasis residual hierarchy on Homology-Restricted MegaScale test doubles).

Metrics Tested:
  All 5 pooled epistasis levels in both regimes (all doubles, n = 12,455; unsaturated doubles dG_add > 0, n = 7,405):
    1. naive_rho / naive_rho_gt0: raw Spearman rho between predicted and measured dddG.
    2. beyond_rho / beyond_rho_gt0: saturation-subtracted residual rho (after removing global smooth E[dddG | dG_add]).
    3. pair_rho / pair_rho_gt0: Spearman rho of mean epistasis across position-pair matrices (>= 20 doubles).
    4. line_rho / line_rho_gt0: Spearman rho of row/column main effects within pair-centered matrices (>= 3 cells).
    5. cell_mag_pooled / cell_mag_gt0: Spearman rho pooled over double-centered cell interaction residuals.

Statistical Test & Resampling Procedure:
  Exact paired domain-cloning x training-seed bootstrap replicates saved by pipeline p06
  (`fig5_epistasis_levels_p0_reps_test.npz`, B = {B_z:,}, N_domains = {n_dom} test domains with doubles).
  In each bootstrap draw b = 1, ..., B, domains are resampled with replacement (re-fitting the saturation
  smooth and ANOVA decompositions inside each bootstrap replicate to propagate estimation uncertainty at every level),
  and 3-seed retrained models jointly resample training seeds.

Hypotheses & One-Sided P-Value Formula:
  H0: rho_ESM-MSR <= rho_X  vs  H1: rho_ESM-MSR > rho_X.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1),
  with minimum resolution 1 / ({B_z + 1:,}) = {1.0 / (B_z + 1):.4g}.

Multiple-Testing Correction (Bonferroni):
  Within each (level, regime), Bonferroni correction multiplies p_one_sided_uncorrected by m = 7 clean external
  comparator models (ProteinMPNN, ESM3-small-open, MutateEverything Stage 2, ThermoMPNN-D, SPURS-multi,
  GeoDDG-3D, Rosetta). Internal controls (`ESM-MSR ctx ctrl` and `−dG_add (floor ceiling)`) are reported with
  their uncorrected one-sided p-values and marked `internal_control_or_variant = True`.

Key Assumptions & Limitations:
  - Re-computing the saturation curve and ANOVA decomposition inside every domain-resampled draw avoids
    underestimating variance from fixed-smooth conditioning.
  - B = {B_z:,} matches the canonical p06 pipeline replicates (resolution ~1e-3); with N = {n_dom} domains,
    domain-cluster uncertainty dominates Monte Carlo variance.
""",
    )


# --------------------------------------------------------------------------------------
# FIGURE 3: External thermostability (a), held-out benchmarks (b), PGym sigma sweep (c)
# --------------------------------------------------------------------------------------
_F3A_FR: dict = {}


def _f3a_worker(job):
    ds, kind, B_chunk, seed = job
    fr = _F3A_FR[ds]
    pt, rp = bootstrap_rho({ds: fr}, list(fr.preds), kind=kind, B=B_chunk, seed=seed, per_protein=False)
    return ds, kind, dict(rp[ds]), {m: (v[0], v[2], v[3]) for m, v in pt[ds].items()}


def export_figure3a() -> None:
    print('[3/10] Exporting Figure 3a (thermostability, B=10,000 mutation & protein bootstrap)...')
    out_sum = results_dir('summary', create=False)
    th = p05._thermo_data()
    ov_labels = p05._overlap_short_labels()
    order_map = {d: i + 1 for i, d in enumerate(th['sets'])}
    D_th = th['S'].assign(
        order=th['S']['dataset'].map(order_map),
        msr_p0_seed_mean=th['S']['dataset'].map(th['val']),
        overlap_label=[ov_labels.get((r.dataset, r.model), '') for r in th['S'].itertuples()],
    ).sort_values(['order', 'model'])
    atomic_to_csv(D_th, MAIN_SRC / 'figure3_panel_a_thermostability_sourcedata.csv', index=False, float_format='%.6f')

    sets = list(D_th.sort_values('order').drop_duplicates('dataset')['dataset'])
    leak = {(r.dataset, r.model): bool(r.leak_flag) for r in D_th.itertuples()}
    _F3A_FR.clear()
    for ds, fr in fresh_frames(D_th[D_th['dataset'].isin(sets)]).items():
        _F3A_FR[ds] = seed_frame(ds, fr)[0]

    chunk = 2500
    jobs = [
        (ds, kind, min(chunk, B_FAST - i * chunk), 31000 + 1000 * int(kind == 'protein') + 50 * sets.index(ds) + i)
        for ds in sets for kind in ('mutation', 'protein') for i in range(int(np.ceil(B_FAST / chunk)))
    ]
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(_f3a_worker, jobs, chunksize=1)

    reps_acc: dict = {}
    pts_a: dict = {}
    for ds, kind, rp, pt in res:
        for m, v in rp.items():
            reps_acc.setdefault((ds, kind, m), []).append(v)
        pts_a[ds] = pt
    reps_a = {k: np.concatenate(v) for k, v in reps_acc.items()}

    p_rows_3a = []
    for ds in sets:
        others = [m for m in EXT_ORDER if m != 'ESM-MSR' and m in pts_a[ds]]
        msr_pt, msr_n, msr_ng = pts_a[ds]['ESM-MSR']
        for kind in ('mutation', 'protein'):
            msr_rp = reps_a[(ds, kind, 'ESM-MSR')]
            comps = []
            for m in others:
                ov = leak.get((ds, m), False)
                ov_lbl = ov_labels.get((ds, m), '')
                comps.append(dict(
                    name=m,
                    point=pts_a[ds][m][0],
                    reps=reps_a[(ds, kind, m)],
                    excluded_overlap=ov,
                    notes=(f'Documented training overlap ({ov_lbl}); excluded from clean Bonferroni family.' if ov else ''),
                ))
            p_rows_3a += build_comparison_rows(
                benchmark=LABEL.get(ds, ds),
                subset_or_metric='pooled_spearman_rho',
                resampling_unit=('mutation_x_seed' if kind == 'mutation' else 'protein_cluster_x_seed'),
                msr_label='ESM-MSR',
                msr_point=msr_pt,
                msr_reps=msr_rp,
                comparators=comps,
                higher_is_better=True,
                n_units=(msr_n if kind == 'mutation' else msr_ng),
                n_variants=msr_n,
                bonferroni_family=f'Figure3a_{LABEL.get(ds, ds)}_{kind}_clean_models',
            )

    save_table_and_annotation(
        pd.DataFrame(p_rows_3a),
        MAIN_PVAL / 'figure3a_thermostability_pvalues.csv',
        f"""Statistical Annotation: figure3a_thermostability_pvalues.csv
==============================================================
Figure / Panel: Main-Text Figure 3, Panel a (8 external thermostability benchmarks: Ssym, S461, K2369, S783, S2648, Q3421, S8754, PTMUL).

Metric Tested:
  Pooled Spearman's rank correlation (rho) over all measured variants in each benchmark dataset
  (singles-only heads on single-mutant benchmarks; multi-mutant/doubles heads on PTMUL; 3-seed mean for retrained models).

Statistical Tests & Resampling Units (BOTH reported):
  Paired nonparametric bootstrap with joint data-unit x training-seed resampling (B = {B_FAST:,}):
  1. `mutation_x_seed` (PRIMARY — matches Figure 3a's plotted 95% error bars and standard ddG benchmark literature):
     Resamples individual mutations with replacement identically across all models, jointly resampling
     the 3 training seeds via Multinomial(3, 1/3) for multi-seed transfer models. Tests generalization
     to new mutations drawn from this panel of proteins (especially informative on small-protein-count sets
     like Ssym with only 15 proteins).
  2. `protein_cluster_x_seed` (CONSERVATIVE SENSITIVITY CHECK):
     Resamples whole protein homology clusters with replacement identically across models (jointly with seeds).
     Fully accounts for within-protein correlation of mutations and tests generalization to unseen proteins.
     On PTMUL (90 protein clusters, 914 mutations), protein-cluster resampling has substantially wider CIs
     than mutation resampling because mutations on the same protein share structural/experimental context.

Hypotheses & One-Sided P-Value Formula:
  H0: rho_ESM-MSR <= rho_X  vs  H1: rho_ESM-MSR > rho_X.
  Exact Monte Carlo one-sided p-value (Phipson & Smyth, 2010):
    p_one_sided_uncorrected = (1 + #{{b : rho_ESM-MSR^(b) - rho_X^(b) <= 0}}) / (B + 1),
  minimum resolution 1 / ({B_FAST + 1:,}) = {1.0 / (B_FAST + 1):.4g}.

Training Overlap & Multiple-Testing Correction (Bonferroni):
  Models with documented training overlap from `leak_audit_matrix.csv` (GeoDDG-3D on all 8 sets; ESM3dG on
  K2369, S2648, S8754, PTMUL) are flagged `excluded_training_overlap = True` and excluded from the clean
  comparator family (their uncorrected p-values are still reported for transparency).
  Bonferroni correction is applied within each (dataset, resampling_unit) across the m_clean clean external
  comparators (m_clean = 7 on K2369, S2648, S8754, PTMUL; m_clean = 8 on Ssym, S461, S783, Q3421).

CRITICAL DISTINCTION — Next-Best Clean Model vs Exceeding Every Clean Model:
  For every (dataset, unit), three distinct summary rows are included:
  1. `next_best_clean`: One-sided test of ESM-MSR vs the single runner-up j* = argmax_{{j in clean}} rho_hat_j
     (e.g., Rosetta on PTMUL). Reports both uncorrected p_j* and Bonferroni-adjusted min(1, m_clean * p_j*).
  2. `all_clean_IUT`: Berger's (1982) Intersection-Union Test for the composite claim
     H1: (forall j in clean, rho_ESM-MSR > rho_j) against H0: (exists j in clean, rho_ESM-MSR <= rho_j).
     By the IUT theorem, rejecting iff all m_clean one-sided tests reject at level alpha strictly controls
     Type I error <= alpha WITHOUT Bonferroni correction; its p-value is p_IUT = max_{{j in clean}} p_j.
     Note: p_IUT equals p(next_best_clean) ONLY when the point-estimate runner-up also has the largest
     pairwise p-value; if another comparator has higher paired SE, p_IUT can strictly exceed p(next_best_clean).
  3. `all_clean_joint_max`: Simultaneous paired bootstrap margin Delta_min^(b) = rho_ESM-MSR^(b) - max_{{j in clean}} rho_j^(b).
     Its p-value P_boot(exists j in clean: rho_j^(b) >= rho_ESM-MSR^(b)) = P_boot(Union_j E_j) directly accounts
     for multiple runners-up beating ESM-MSR on DIFFERENT bootstrap draws (e.g., Rosetta on some draws and
     MutateEverything on other draws in PTMUL). By probability union bounds:
       max_j p_j (IUT) <= p_joint <= min(1, sum_j p_j) <= min(1, m_clean * max_j p_j) (Bonferroni).
     Thus `next_best_clean`, `all_clean_IUT`, and `all_clean_joint_max` answer related but mathematically distinct
     questions and are NOT identical in general.
""",
    )

def export_figure3bc() -> None:
    # --- Figure 3b: Held-out benchmarks (ProteinGym 173, Domainome 436, MGnify Pooled 612k, Rocklin Pooled 110k) ---
    print('[4/10] Exporting Figure 3b (ProteinGym, Domainome, MGnify Pooled, Rocklin Pooled)...')
    pg_prep = p05._pg_prepare(B=1000, pooled=True)
    # Source data for Figure 3b
    b_src_rows = []
    ov_b = p05._overlap_short_labels()
    for g_idx, (g_key, g_lbl) in enumerate(zip(pg_prep['groups'], pg_prep['labels'])):
        clean_lbl = g_lbl.replace('\n', ' ')
        for m in p05.MODEL_ORDER:
            if (g_key, m) not in pg_prep['table']:
                continue
            lo_v, hi_v = pg_prep['cis'].get((g_key, m), (np.nan, np.nan))
            b_src_rows.append(dict(
                benchmark_key=g_key, benchmark_label=clean_lbl, model=m,
                rho=float(pg_prep['table'][(g_key, m)]), lo=float(lo_v), hi=float(hi_v),
                hollow_singles=(g_key, m) in pg_prep['hollow'],
                training_overlap=(g_key, m) in pg_prep['rings'],
                overlap_label=ov_b.get((g_key, m), 'H 2%' if (g_key == 'PG avg' and m == 'ESM3ΔG') else ''),
            ))
    atomic_to_csv(pd.DataFrame(b_src_rows), MAIN_SRC / 'figure3_panel_b_heldout_benchmarks_sourcedata.csv', index=False, float_format='%.6f')

    p_rows_3b = []
    # (i) ProteinGym avg (173 held-out assays) — paired category-stratified assay x seed bootstrap B=10,000
    mean_pg, seeds_pg = load_all_models_pgym(heldout=True)
    sum_pg, reps_pg = boot_pgym_categories(seeds_pg, mean_pg['category'], B=B_FAST, seed=41)
    pt_pg = {(r.model, r.column): float(r.value) for r in sum_pg.itertuples()}
    n_pg_assays = int(len(mean_pg))
    comps_pg = []
    for m in p05.MODEL_ORDER:
        if m == 'ESM-MSR' or m not in reps_pg:
            continue
        is_int = (m == 'ESM-MSR σ=0.5')
        is_ov = (m == 'ESM3ΔG')
        comps_pg.append(dict(
            name=m, point=pt_pg[(m, 'PG avg')], reps=reps_pg[m]['PG avg'],
            excluded_overlap=is_ov, internal_variant=is_int,
            notes=('Documented training overlap (H 2%); excluded from clean family.' if is_ov
                   else ('Internal LoRA-scale variant of ESM-MSR.' if is_int else '')),
        ))
    p_rows_3b += build_comparison_rows(
        benchmark='ProteinGym_heldout_173_assays',
        subset_or_metric='category_weighted_mean_spearman_rho',
        resampling_unit='category_stratified_assay_x_seed',
        msr_label='ESM-MSR (σ=1)',
        msr_point=pt_pg[('ESM-MSR', 'PG avg')],
        msr_reps=reps_pg['ESM-MSR']['PG avg'],
        comparators=comps_pg,
        higher_is_better=True,
        n_units=n_pg_assays,
        n_variants=None,
        bonferroni_family='Figure3b_ProteinGym_clean_models',
    )

    # (ii) Domainome avg (436 held-out domains) — paired domain x seed bootstrap B=10,000
    dom_fr = load_frame('domainome')
    dom_s, _ = seed_frame('domainome', dom_fr, models=p05.MODEL_ORDER)
    hom_dom = heldout_excluded('domainome')
    sp_d = dom_s.spine.copy()
    sp_d['use'] = sp_d['use'].to_numpy() & ~sp_d['group'].astype(str).isin(hom_dom).to_numpy()
    dom_held = Frame('domainome', sp_d, dom_s.preds)
    pt_d, rp_d = bootstrap_rho({'domainome': dom_held}, list(dom_held.preds), kind='protein', B=B_FAST, seed=42, per_protein=True)
    n_dom_units = int(pt_d['domainome']['ESM-MSR'][3])
    n_dom_vars = int(pt_d['domainome']['ESM-MSR'][2])
    comps_dom = []
    for m in p05.MODEL_ORDER:
        if m == 'ESM-MSR' or m not in pt_d['domainome']:
            continue
        is_int = (m == 'ESM-MSR σ=0.5')
        comps_dom.append(dict(
            name=m, point=pt_d['domainome'][m][0], reps=rp_d['domainome'][m],
            internal_variant=is_int,
            notes=('Internal LoRA-scale variant of ESM-MSR.' if is_int else ''),
        ))
    p_rows_3b += build_comparison_rows(
        benchmark='Domainome_heldout_436_domains',
        subset_or_metric='per_domain_mean_spearman_rho',
        resampling_unit='domain_cluster_x_seed',
        msr_label='ESM-MSR',
        msr_point=pt_d['domainome']['ESM-MSR'][0],
        msr_reps=rp_d['domainome']['ESM-MSR'],
        comparators=comps_dom,
        higher_is_better=True,
        n_units=n_dom_units,
        n_variants=n_dom_vars,
        bonferroni_family='Figure3b_Domainome_clean_models',
    )

    # (iii) MGnify Pooled (612k variants) & Rocklin Pooled (110k variants) — paired scaffold x seed bootstrap B=1,000
    for ds_name, cfg in (('MGnify', dict(run='all', src='mgnify')), ('Rocklin designs', dict(run='rdenovo', src='rocklin_denovo'))):
        d_df, preds_mg = p04._prepare_heldout_models(cfg)
        code, uniq = pd.factorize(d_df['code_wt'])
        code = code.astype(np.int64)
        n_uniq = len(uniq)
        y_mg = d_df['y'].to_numpy(float)
        rng_p = np.random.default_rng(77 if ds_name == 'MGnify' else 78)
        scaf_counts = rng_p.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq), size=B_MGNIFY_POOLED).astype(float)
        seed_w = rng_p.multinomial(3, np.full(3, 1.0 / 3.0), size=B_MGNIFY_POOLED).astype(float) / 3.0

        # Filter to the Figure 3b INFERRED_MODEL roster
        xr_dict = {}
        for mn, (_, _, P_pool) in preds_mg.items():
            if mn not in p05.INFERRED_MODEL:
                continue
            if np.all([np.isfinite(p) for p in P_pool]):
                xr_dict[p05.INFERRED_MODEL[mn]] = [p04._FastPooledRank(p) for p in P_pool]
        pt_mg, rep_mg = p04._eval_pooled_fast(
            xr_dict, p04._FastPooledRank(y_mg, need_inv=True), len(y_mg), code, scaf_counts, seed_w, workers=WORKERS,
        )
        comps_mg = []
        for m in p05.MODEL_ORDER:
            if m == 'ESM-MSR' or m not in pt_mg:
                continue
            is_int = (m == 'ESM-MSR σ=0.5')
            is_ov = (m == 'ESM3ΔG')
            comps_mg.append(dict(
                name=m, point=float(pt_mg[m][0]), reps=rep_mg[m],
                excluded_overlap=is_ov, internal_variant=is_int,
                notes=('ESM3dG fine-tuned on MGnify stability data (train src); excluded from clean family.' if is_ov
                       else ('Internal LoRA-scale variant of ESM-MSR.' if is_int else '')),
            ))
        p_rows_3b += build_comparison_rows(
            benchmark=f'{ds_name.replace(" ", "_")}_heldout_pooled',
            subset_or_metric='pooled_spearman_rho',
            resampling_unit='scaffold_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=float(pt_mg['ESM-MSR'][0]),
            msr_reps=rep_mg['ESM-MSR'],
            comparators=comps_mg,
            higher_is_better=True,
            n_units=n_uniq,
            n_variants=len(y_mg),
            bonferroni_family=f'Figure3b_{ds_name.replace(" ", "_")}_clean_models',
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows_3b),
        MAIN_PVAL / 'figure3b_heldout_benchmarks_pvalues.csv',
        f"""Statistical Annotation: figure3b_heldout_benchmarks_pvalues.csv
=================================================================
Figure / Panel: Main-Text Figure 3, Panel b (Held-out external benchmarks: ProteinGym avg., Domainome avg., MGnify Pooled, Rocklin Designs Pooled).

Metrics & Resampling Units (matching Figure 3b exactly):
  1. ProteinGym avg. (173 held-out assays): Equal-category-weighted mean of per-assay Spearman's rho
     (macro average across the 5 functional categories: Stability, Expression, Activity, Binding, Org. fitness).
     Resampling unit: paired category-stratified assay x training-seed bootstrap (B = {B_FAST:,}).
     ESM3dG is excluded from the clean family (H 2% training overlap); ESM-MSR sigma=0.5 is marked internal variant.
  2. Domainome avg. (436 held-out domains): Mean per-domain Spearman's rho.
     Resampling unit: paired domain-cluster x training-seed bootstrap (B = {B_FAST:,}).
     (Rosetta is not scored on Domainome; ESM-MSR sigma=0.5 is marked internal variant.)
  3. MGnify Pooled (612,406 variants across 152,680 held-out scaffolds) and Rocklin Designs Pooled
     (109,581 variants across 742 held-out scaffolds): One pooled Spearman's rho over all variants.
     Resampling unit: paired scaffold-cluster x training-seed bootstrap (B = {B_MGNIFY_POOLED:,}) using shared
     scaffold multinomial weights and seed weights across all models (exact O(n) rank engine).
     ESM3dG is excluded from the clean family (fine-tuned directly on MGnify stability data).

Hypotheses & One-Sided P-Value Formula:
  H0: rho_ESM-MSR <= rho_X  vs  H1: rho_ESM-MSR > rho_X.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction is applied within each benchmark over the m_clean clean external comparators.

Key Assumptions & Limitations:
  - Assays (ProteinGym), domains (Domainome), and wild-type scaffolds (MGnify, Rocklin) are treated as
    independent biological units; all mutations within a scaffold/domain/assay are resampled together as a block.
  - All benchmarks strictly apply the 4-part homology holdout filter against Homology-Restricted MegaScale train.
""",
    )

    # --- Figure 3c: ProteinGym delta-rho vs sigma=0 across 5 functional categories ---
    print('[5/10] Exporting Figure 3c (ProteinGym LoRA scale sigma sweep)...')
    t_all, _ = load_sigma_pgym(heldout=False, drop_large=True, include_fallback_small=False)
    t_held, per_seed_held = load_sigma_pgym(heldout=True, drop_large=True, include_fallback_small=False)
    atomic_to_csv(t_all.reset_index(), MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv', index=False, float_format='%.6f')

    p_rows_3c = []
    m_sig_family = len(PG_CATEGORIES) * 2  # 5 categories x 2 sigma levels (0.5 and 1.0)
    for cat in PG_CATEGORIES:
        sub_h = t_held[t_held['category'] == cat]
        n_a = int(len(sub_h))
        assays = list(sub_h['assay'])
        # Build (n_assays, n_seeds) arrays for sigma=0, 0.5, 1.0 using deployed readout (matching Figure 3c)
        a0 = per_seed_held[('deployed', 0.0)].reindex(assays).to_numpy(float)[:, :1]
        a05 = per_seed_held[('deployed', 0.5)].reindex(assays).to_numpy(float)
        a1 = per_seed_held[('deployed', 1.0)].reindex(assays).to_numpy(float)
        d_cnt, s_w = make_domain_seed_draws(n_a, B=B_FAST, seed=330 + PG_CATEGORIES.index(cat), seed_counts=(3,))
        pt0, rp0 = boot_domain_seeds(a0, B=B_FAST, counts=d_cnt, seed_weights=None)
        pt05, rp05 = boot_domain_seeds(a05, B=B_FAST, counts=d_cnt, seed_weights=s_w if a05.shape[1] > 1 else None)
        pt1, rp1 = boot_domain_seeds(a1, B=B_FAST, counts=d_cnt, seed_weights=s_w if a1.shape[1] > 1 else None)

        for sig_lbl, pt_s, rp_s, comp_lbl, pt_c0, rp_c0, fam_m, fam_name in (
            ('ESM-MSR (σ=0.5)', pt05, rp05, 'ESM3-small-open (σ=0)', pt0, rp0, m_sig_family, 'Figure3c_10_category_x_sigma_vs_sigma0'),
            ('ESM-MSR (σ=1.0)', pt1, rp1, 'ESM3-small-open (σ=0)', pt0, rp0, m_sig_family, 'Figure3c_10_category_x_sigma_vs_sigma0'),
            ('ESM-MSR (σ=1.0)', pt1, rp1, 'ESM-MSR (σ=0.5)', pt05, rp05, len(PG_CATEGORIES), 'Figure3c_5_categories_sigma1_vs_sigma0.5'),
        ):
            d = rp_s - rp_c0
            p_one, nf, Be = one_sided_boot_p(d)
            p_rows_3c.append(dict(
                benchmark='ProteinGym_heldout_by_category',
                subset_or_metric=f'{cat}: mean_delta_rho',
                resampling_unit='assay_x_seed',
                test_type='pairwise',
                msr_model=sig_lbl,
                comparator=comp_lbl,
                excluded_training_overlap=False,
                internal_control_or_variant=False,
                direction='higher_is_better',
                value_msr=float(pt_s),
                value_comparator=float(pt_c0),
                delta_favoring_msr=float(pt_s - pt_c0),
                ci95_lo=float(np.quantile(d, 0.025)),
                ci95_hi=float(np.quantile(d, 0.975)),
                se_boot=float(np.std(d, ddof=1)),
                n_units=n_a,
                n_variants=np.nan,
                B=Be,
                n_boot_fail=nf,
                p_one_sided_uncorrected=p_one,
                bonferroni_m=fam_m,
                p_one_sided_bonferroni=float(min(1.0, fam_m * p_one)),
                bonferroni_family=fam_name,
                notes='Held-out assays only (40 homologous assays excluded).',
            ))

    save_table_and_annotation(
        pd.DataFrame(p_rows_3c),
        MAIN_PVAL / 'figure3c_proteingym_sigma_pvalues.csv',
        f"""Statistical Annotation: figure3c_proteingym_sigma_pvalues.csv
===============================================================
Figure / Panel: Main-Text Figure 3, Panel c (Per-category ProteinGym delta-rho vs zero-shot sigma = 0 for ESM-MSR sigma = 0.5 and sigma = 1.0).

Metric Tested:
  Mean per-assay paired gain Delta_rho = rho(sigma) - rho(sigma = 0) across held-out ProteinGym assays within each
  of the 5 functional categories (Stability: 37, Expression: 18, Activity: 38, Binding: 13, Organismal fitness: 67),
  plus sigma = 1.0 vs sigma = 0.5 within each category.

Statistical Test & Resampling Procedure:
  Paired nonparametric assay x training-seed bootstrap (B = {B_FAST:,}) within each functional category.
  Assays are resampled with replacement identically across sigma = 0, 0.5, 1.0, jointly with Multinomial(3, 1/3)
  seed weights for ESM-MSR seeds 1–3.

Hypotheses & One-Sided P-Value Formula:
  H0: E[rho(sigma_high) - rho(sigma_low)] <= 0  vs  H1: E[rho(sigma_high) - rho(sigma_low)] > 0.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).
  Note: Figure 3c itself annotates two-sided Holm-adjusted p-values across the 10 (category x sigma) tests;
  this table reports the directional one-sided uncorrected and Bonferroni-corrected p-values (m = 10 for vs sigma=0;
  m = 5 for sigma=1.0 vs sigma=0.5).

Key Assumptions & Limitations:
  - Held-out assays within a functional category are treated as independent experimental units.
  - The 40 ProteinGym assays homologous to MegaScale train are excluded from all statistics (matching Figure 3c).
""",
    )


# --------------------------------------------------------------------------------------
# FIGURE 4: Rectified gain @ k=96 (a) and screening curves (b, c, d)
# --------------------------------------------------------------------------------------
def export_supp_property_gain_k24_pvalues() -> None:
    print('[6b/10] Exporting Supplementary Figure _property_gain_k24 (ASA x hydrophobicity rectified gain @ k=24)...')
    prop_bundle = p09._compute_asa_k96_allpicks_bundle(split='capped', B=1000, k=24)
    assert prop_bundle is not None
    stats_k = prop_bundle['stats_k']
    cat_labels = prop_bundle['cat_labels']
    n_dom = prop_bundle['n_domains']
    n_muts = prop_bundle['n_muts']

    d_cnt, s_w = make_domain_seed_draws(n_dom, B=B_FAST, seed=401, seed_counts=(3,))
    model_names = [m for m in stats_k if not m.startswith('Top ') and m != 'oracle']
    p_rows = []
    for c_idx, c_lbl in list(enumerate(cat_labels)) + [(-1, 'Total rectified gain @ k=24 (sum over 4 quadrants)')]:
        pts, rps = {}, {}
        for m in model_names:
            G_m = stats_k[m][0]
            arr = (G_m.sum(axis=2) if c_idx == -1 else G_m[:, :, c_idx]).T
            pt, rp = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
            pts[m], rps[m] = pt, rp
        comps = [dict(name=m, point=pts[m], reps=rps[m]) for m in model_names if m != 'ESM-MSR']
        p_rows += build_comparison_rows(
            benchmark='MegaScale_capped_test_singles_k24',
            subset_or_metric=c_lbl,
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR WT-view',
            msr_point=pts['ESM-MSR'],
            msr_reps=rps['ESM-MSR'],
            comparators=comps,
            higher_is_better=True,
            n_units=n_dom,
            n_variants=n_muts,
            bonferroni_family=f'SuppPropertyGainK24_quadrant_{c_idx}_8_comparators',
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows),
        SUPP_PVAL / '_property_gain_k24_pvalues.csv',
        f"""Statistical Annotation: _property_gain_k24_pvalues.csv
============================================================
Figure / Panel: Supplementary Figure _property_gain_k24 (Rectified gain @ k = 24 across the 4 ASA x hydrophobicity quadrants on Homology-Restricted MegaScale test single mutants; k = 24 companion to Main-Text Figure 4a).

Metric Tested:
  Mean per-domain rectified stability gain sum(max(ddG_true, 0)) among each model's top k = 24 single-mutant picks
  per domain (33 domains with ASA/hydrophobicity annotations, {n_muts:,} single mutants), evaluated per quadrant
  and summed across all 4 quadrants (Total rectified gain @ k = 24).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}).
  All 33 test domains are resampled identically across models via shared Multinomial(33, 1/33) weights,
  jointly with Multinomial(3, 1/3) seed weights for 3-seed retrained models. Oracle is excluded from statistical
  comparisons (upper bound, not a predictive model).

Hypotheses & One-Sided P-Value Formula:
  H0: E[gain_ESM-MSR - gain_X] <= 0  vs  H1: E[gain_ESM-MSR - gain_X] > 0.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction multiplies p_one_sided_uncorrected by m = 8 clean comparator models within each quadrant
  (and within total gain).

Key Assumptions & Limitations:
  - Test domains (N = 33) are treated as independent biological units.
  - Because picks are fixed to k = 24 total per domain, quadrant-level gains depend on both pick allocation
    across quadrants (shown in the Picks % strip) and ranking quality within each quadrant.
""",
    )


def export_figure4() -> None:
    print('[6/10] Exporting Figure 4a (ASA x hydrophobicity rectified gain @ k=96)...')
    prop_bundle = p09._compute_asa_k96_allpicks_bundle(split='capped', B=1000)
    assert prop_bundle is not None
    dcols = prop_bundle['dcols']
    stats_k = prop_bundle['stats_k']
    cat_labels = prop_bundle['cat_labels']
    n_dom = prop_bundle['n_domains']
    n_muts = prop_bundle['n_muts']

    # Source data: summary + full per-domain (33 domains x models x seeds x 4 quadrants)
    pg_dir = results_dir('megascale_screening', split='capped', create=False) / 'property_gain'
    _copy_as_sourcedata(pg_dir / 'gain_asa_hydrophobicity_k96_allpicks.csv', MAIN_SRC / 'figure4_panel_a_summary_sourcedata.csv')

    dom_rows = []
    for m_name, (G_m, N_m) in stats_k.items():
        # G_m shape: (n_seeds, n_domains, n_cats)
        for s_i in range(G_m.shape[0]):
            for d_i in range(G_m.shape[1]):
                for c_i, c_lbl in enumerate(cat_labels):
                    dom_rows.append(dict(
                        model=m_name, domain_idx=d_i, seed=s_i + 1, quadrant=c_lbl,
                        rectified_gain_k96=float(G_m[s_i, d_i, c_i]),
                        n_picks=float(N_m[s_i, d_i, c_i]),
                    ))
    atomic_to_csv(pd.DataFrame(dom_rows), MAIN_SRC / 'figure4_panel_a_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    # Paired domain x seed bootstrap (B = 10,000) for each of the 4 quadrants + Total gain @ k=96
    d_cnt, s_w = make_domain_seed_draws(n_dom, B=B_FAST, seed=401, seed_counts=(3,))
    model_names = [m for m in stats_k if m != 'oracle']
    p_rows_4a = []
    for c_idx, c_lbl in list(enumerate(cat_labels)) + [(-1, 'Total rectified gain @ k=96 (sum over 4 quadrants)')]:
        pts, rps = {}, {}
        for m in model_names:
            G_m = stats_k[m][0]
            arr = (G_m.sum(axis=2) if c_idx == -1 else G_m[:, :, c_idx]).T  # (n_dom, n_seeds)
            pt, rp = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
            pts[m], rps[m] = pt, rp
        comps = [dict(name=m, point=pts[m], reps=rps[m]) for m in model_names if m != 'ESM-MSR']
        p_rows_4a += build_comparison_rows(
            benchmark='MegaScale_capped_test_singles_k96',
            subset_or_metric=c_lbl,
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR WT-view',
            msr_point=pts['ESM-MSR'],
            msr_reps=rps['ESM-MSR'],
            comparators=comps,
            higher_is_better=True,
            n_units=n_dom,
            n_variants=n_muts,
            bonferroni_family=f'Figure4a_quadrant_{c_idx}_8_comparators',
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows_4a),
        MAIN_PVAL / 'figure4a_property_gain_pvalues.csv',
        f"""Statistical Annotation: figure4a_property_gain_pvalues.csv
============================================================
Figure / Panel: Main-Text Figure 4, Panel a (Rectified gain @ k = 96 across the 4 ASA x hydrophobicity quadrants on Homology-Restricted MegaScale test single mutants).

Metric Tested:
  Mean per-domain rectified stability gain sum(max(ddG_true, 0)) among each model's top k = 96 single-mutant picks
  per domain (33 domains with ASA/hydrophobicity annotations, {n_muts:,} single mutants), evaluated per quadrant
  and summed across all 4 quadrants (Total rectified gain @ k = 96).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}).
  All 33 test domains are resampled identically across models via shared Multinomial(33, 1/33) weights,
  jointly with Multinomial(3, 1/3) seed weights for 3-seed retrained models. Oracle is excluded from statistical
  comparisons (upper bound, not a predictive model).

Hypotheses & One-Sided P-Value Formula:
  H0: E[gain_ESM-MSR - gain_X] <= 0  vs  H1: E[gain_ESM-MSR - gain_X] > 0.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction multiplies p_one_sided_uncorrected by m = 8 clean comparator models within each quadrant
  (and within total gain).

Key Assumptions & Limitations:
  - Test domains (N = 33) are treated as independent biological units.
  - Because picks are fixed to k = 96 total per domain, quadrant-level gains depend on both pick allocation
    across quadrants (shown in the Picks % strip) and ranking quality within each quadrant.
""",
    )

    export_supp_property_gain_k24_pvalues()

    # --- Figure 4b, 4c, 4d: Screening curves (MegaScale, Domainome, ProteinGym Activity) ---
    print('[7/10] Exporting Figure 4b/c/d (MegaScale, Domainome, ProteinGym Activity screening curves)...')
    mega_scr_csv = results_dir('megascale_screening', split='capped', create=False) / 'screening_per_protein_singles_all_models.csv'
    dom_scr_csv = results_dir('domainome_screening', create=False) / 'screening_per_protein_all_models.csv'
    pg_scr_dir = results_dir('proteingym_screening', create=False)
    _copy_as_sourcedata(mega_scr_csv, MAIN_SRC / 'figure4_panel_b_megascale_screening_sourcedata.csv')
    _copy_as_sourcedata(dom_scr_csv, MAIN_SRC / 'figure4_panel_c_domainome_screening_sourcedata.csv')
    _copy_as_sourcedata(pg_scr_dir / 'screening_all_all_models.csv', MAIN_SRC / 'figure4_panel_d_pgym_all_screening_sourcedata.csv')
    _copy_as_sourcedata(pg_scr_dir / 'screening_enzymes_all_models.csv', MAIN_SRC / 'figure4_panel_d_pgym_enzymes_screening_sourcedata.csv')

    specs = p09.build_family_specs('capped')
    fams_all = list(specs['fam_all'])
    flab = specs['fam_label_singles']
    ks_curve = list(p09.KS_CURVE)  # [1, 3, 5, 10, 20, 50, 100]
    p_rows_4bcd = []

    def _run_screening_pvals(
        bench_name: str, fr: Frame, keep_mask: np.ndarray, tau: float,
        metrics_spec: list[tuple[str, str, bool]], unit_name: str, fams_use: list[str],
        overlap_fams: set[str] | None = None,
    ) -> None:
        overlap_fams = overlap_fams or set()
        sp = fr.spine
        for K in ks_curve:
            # Build per-protein tables for each model & seed
            per_fam: dict[str, tuple[list[str], dict[str, np.ndarray]]] = {}
            for fam in fams_use:
                k = p09.fam_key(fam, fr, specs, mode='singles')
                if k is None or k not in fr.preds:
                    continue
                ok = sp['use'].to_numpy() & keep_mask & np.isfinite(fr.preds[k]).all(axis=0)
                if not ok.any():
                    continue
                y_ok = sp['y'].to_numpy()[ok]
                g_ok = sp['group'].to_numpy()[ok]
                tabs = [scr_core.per_protein_table(g_ok, pr[ok], y_ok, K, tau).set_index('group') for pr in fr.preds[k]]
                groups_common = sorted(set.intersection(*(set(t.index) for t in tabs)))
                mat_by_col = {}
                for col_key, _, _ in metrics_spec:
                    mat_by_col[col_key] = np.column_stack([t.reindex(groups_common)[col_key].to_numpy(float) for t in tabs])
                per_fam[fam] = (groups_common, mat_by_col)

            if 'msr' not in per_fam:
                continue
            avail_fams = [f for f in fams_use if f in per_fam]
            for col_key, met_label, higher_better in metrics_spec:
                # Common proteins/assays where ESM-MSR and comparators are finite (Note: for p_recover, restrict to proteins with >=1 hit)
                # For exact consistency with p09 curves, use all proteins in per_fam['msr'] and mask non-finite per model
                common_g = sorted(set.intersection(*(set(per_fam[f][0]) for f in avail_fams)))
                # For p_recover, keep proteins where ESM-MSR is finite (has >=1 true hit)
                idx_msr = [per_fam['msr'][0].index(g) for g in common_g]
                arr_msr_full = per_fam['msr'][1][col_key][idx_msr]
                valid_g_mask = np.isfinite(arr_msr_full).all(axis=1)
                use_g = [g for g, v in zip(common_g, valid_g_mask) if v]
                n_u = len(use_g)
                if n_u < 2:
                    continue
                d_cnt, s_w = make_domain_seed_draws(n_u, B=B_FAST, seed=450 + K, seed_counts=(3,))
                pts, rps = {}, {}
                for f in avail_fams:
                    idx_f = [per_fam[f][0].index(g) for g in use_g]
                    arr = per_fam[f][1][col_key][idx_f]
                    pt, rp = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
                    pts[f], rps[f] = pt, rp
                comps = [
                    dict(
                        name=flab.get(f, f), point=pts[f], reps=rps[f],
                        excluded_overlap=(f in overlap_fams),
                        notes=('Documented training overlap (ESM3dG); excluded from clean family.' if f in overlap_fams else ''),
                    )
                    for f in avail_fams if f != 'msr'
                ]
                p_rows_4bcd.extend(build_comparison_rows(
                    benchmark=bench_name,
                    subset_or_metric=f'{met_label}@K={K}',
                    resampling_unit=unit_name,
                    msr_label=flab.get('msr', 'ESM-MSR WT-view'),
                    msr_point=pts['msr'],
                    msr_reps=rps['msr'],
                    comparators=comps,
                    higher_is_better=higher_better,
                    n_units=n_u,
                    n_variants=None,
                    bonferroni_family=f'Figure4_{bench_name}_{met_label}_K{K}_clean_models',
                ))

    # Panel b: MegaScale capped test singles (excl. Cys), tau = 0.5 kcal/mol
    fr_mega, _ = load_tsu_test_frame('capped')
    sp_m = fr_mega.spine
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp_m['mut']], dtype=bool)
    keep_m = (sp_m['n_mut'].to_numpy() == 1) & ~to_cys
    _run_screening_pvals(
        'Figure4b_MegaScale_singles_nocys', fr_mega, keep_m, 0.5,
        [('hit', 'hit_rate_gt_0.5', True), ('rec', 'P_at_least_one_hit', True)],
        'domain_cluster_x_seed', fams_all,
    )

    # Panel c: Domainome held-out singles, tau = 0.0 (abundance > WT)
    fr_dom = load_frame('domainome')
    hom_dom = heldout_excluded('domainome')
    keep_d = ~fr_dom.spine['group'].astype(str).isin(hom_dom).to_numpy() & (fr_dom.spine['n_mut'].to_numpy() == 1)
    _run_screening_pvals(
        'Figure4c_Domainome_heldout_singles', fr_dom, keep_d, 0.0,
        [('hit', 'hit_rate_gt_WT', True), ('rec', 'P_at_least_one_hit', True)],
        'domain_cluster_x_seed', fams_all,
    )

    # Panel d: ProteinGym Activity held-out singles (all 38 assays & 16 enzymes), lower harm rate is better!
    pg_fr, pg_info = p09.load_pg_activity_assays()
    pg_fams = [f for f in fams_all if f not in ('geoddg',)]
    for set_tag, gs in (('Figure4d_PGym_Activity_all_38', set(pg_info.index)), ('Figure4d_PGym_Activity_enzymes_16', p09.PG_ENZYME)):
        g_list = [d for d in pg_info.index if d in gs]
        lf = p09._label_frame(pg_fr, g_list, 'z_sd', -1, 'use_wt')
        keep_p = np.ones(len(lf.spine), dtype=bool)
        _run_screening_pvals(
            set_tag, lf, keep_p, 0.5,
            [('hit', 'harm_rate_lt_WT_minus_0.5SD', False)],
            'assay_x_seed', pg_fams, overlap_fams={'esm3dg'},
        )

    save_table_and_annotation(
        pd.DataFrame(p_rows_4bcd),
        MAIN_PVAL / 'figure4bcd_screening_pvalues.csv',
        f"""Statistical Annotation: figure4bcd_screening_pvalues.csv
========================================================
Figure / Panel: Main-Text Figure 4, Panels b, c, and d (Single-mutant screening performance across budgets K in {{1, 3, 5, 10, 20, 50, 100}}).

Metrics Tested:
  - Panel b (MegaScale capped test, 33 domains, singles excl. Cys):
      * hit_rate_gt_0.5: mean over domains of fraction of top-K picks with measured ddG > 0.5 kcal/mol (higher is better).
      * P_at_least_one_hit: fraction of hit-containing domains with >= 1 true hit in top K (higher is better).
  - Panel c (Held-out Domainome, 436 domains, singles):
      * hit_rate_gt_WT: mean over domains of fraction of top-K picks with abundance > wild type (higher is better).
      * P_at_least_one_hit: fraction of domains with >= 1 stabilizing hit in top K (higher is better).
  - Panel d (Held-out ProteinGym Activity, 38 all assays and 16 enzyme assays, singles):
      * harm_rate_lt_WT_minus_0.5SD: mean over assays of fraction of top-K picks whose measured activity falls
        below WT - 0.5 assay SD (LOWER is better; delta_favoring_msr = harm_X - harm_ESM-MSR).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain/assay x training-seed bootstrap (B = {B_FAST:,}).
  At each budget K, domains/assays are resampled identically across all models using shared multinomial
  weights, jointly with Multinomial(3, 1/3) seed weights for 3-seed retrained models.

Hypotheses & One-Sided P-Value Formula:
  - Panels b & c (higher is better): H0: metric_ESM-MSR <= metric_X vs H1: metric_ESM-MSR > metric_X.
  - Panel d (lower harm is better):  H0: harm_ESM-MSR >= harm_X     vs H1: harm_ESM-MSR < harm_X.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: delta_favoring_msr^(b) <= 0}}) / (B + 1).
  Bonferroni correction is applied within each (panel, metric, K) family over all m_clean clean comparators.
  (On Panel d, ESM3dG is flagged for PGym training overlap and excluded from the clean Bonferroni family.)

Key Assumptions & Limitations:
  - Domains/assays are treated as independent biological units.
  - At very small K (e.g., K = 1), per-protein hit/harm indicators are binary (0 or 1), so paired differences
    are discrete; nonparametric bootstrap handles this naturally without Gaussianity assumptions.
""",
    )


# --------------------------------------------------------------------------------------
# SUPPLEMENTARY FIGURES: source data + paired one-sided p-values where applicable
# --------------------------------------------------------------------------------------
def export_supplementary() -> None:
    print('[8/10] Exporting Supplementary Figures 1–7, _epi_rmse, _epi_rmse_bars, _sequence_length...')
    supp_dir = results_dir('supplementary_figs', create=False)
    mega_dir = results_dir('megascale', split='capped', create=False)
    epi_dir = results_dir('epistasis', split='capped', create=False)
    len_dir = results_dir('sequence_length', create=False)

    # Supp Fig 1 (split construction): source data only (descriptive)
    sc_dir = supp_dir / 'split_construction'
    if (sc_dir / 'split_best_partner_capped.csv').exists():
        _copy_as_sourcedata(sc_dir / 'split_best_partner_capped.csv', SUPP_SRC / 'supp_figure1_best_partner_sourcedata.csv')
    if (sc_dir / 'mega_clean_vs_homologous_test.csv').exists():
        _copy_as_sourcedata(sc_dir / 'mega_clean_vs_homologous_test.csv', SUPP_SRC / 'supp_figure1_split_overlap_sourcedata.csv')

    # Supp Fig 2 (Shifted NDCG@96 singles & doubles): source data + paired domain x seed p-values
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols_s2 = [c for c in main_columns(fr, best, 'unmasked') if c.get('draw', True)]
    use = sp['use'].to_numpy()
    single = use & (sp['n_mut'] == 1).to_numpy()
    double = use & (sp['n_mut'] == 2).to_numpy()
    _copy_as_sourcedata(mega_dir / 'ndcg96_singles_doubles.csv', SUPP_SRC / 'supp_figure2_ndcg96_summary_sourcedata.csv')

    p_rows_s2 = []
    s2_dom_rows = []
    for sub_name, m_sub, which in (('singles', single, 'singles'), ('doubles', double, 'doubles'), ('overall', use, 'all')):
        vmap = {}
        for c in cols_s2:
            k = key_for(c, which)
            if k is None or k not in fr.preds:
                continue
            disp = name_map.get(c['name'], c['name'])
            df_v = per_library(fr, k, m_sub, ndcg96_shifted, MIN_N)
            vmap[disp] = df_v
            for dom, row in df_v.iterrows():
                for si, val in enumerate(row.values):
                    s2_dom_rows.append(dict(subset=sub_name, model=disp, key=k, domain=dom, seed=si + 1, shifted_ndcg96=float(val)))
        models_here = [name_map.get(c['name'], c['name']) for c in cols_s2 if name_map.get(c['name'], c['name']) in vmap]
        common_d = sorted(set.intersection(*(set(vmap[m].index) for m in models_here)))
        d_cnt, s_w = make_domain_seed_draws(len(common_d), B=B_FAST, seed=502 + len(p_rows_s2), seed_counts=(3,))
        pts, rps = {}, {}
        for m in models_here:
            arr = vmap[m].reindex(common_d).to_numpy(float)
            pts[m], rps[m] = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
        comps = [dict(name=m, point=pts[m], reps=rps[m]) for m in models_here if m != 'ESM-MSR']
        p_rows_s2 += build_comparison_rows(
            benchmark='MegaScale_capped_test',
            subset_or_metric=f'shifted_NDCG96_{sub_name}',
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=pts['ESM-MSR'],
            msr_reps=rps['ESM-MSR'],
            comparators=comps,
            higher_is_better=True,
            n_units=len(common_d),
            n_variants=int((m_sub & np.isfinite(sp['y'].to_numpy())).sum()),
            bonferroni_family=f'SuppFig2_NDCG96_{sub_name}_8_comparators',
        )
    atomic_to_csv(pd.DataFrame(s2_dom_rows), SUPP_SRC / 'supp_figure2_ndcg96_per_domain_sourcedata.csv', index=False, float_format='%.6f')
    save_table_and_annotation(
        pd.DataFrame(p_rows_s2),
        SUPP_PVAL / 'supp_figure2_ndcg96_pvalues.csv',
        f"""Statistical Annotation: supp_figure2_ndcg96_pvalues.csv
=======================================================
Figure: Supplementary Figure 2 (Shifted NDCG@96 on Homology-Restricted MegaScale test singles, doubles, and overall).

Metric Tested:
  Mean per-domain shifted NDCG@96 (relevance = ddG - min(ddG) per domain and subset) across all 33 domains
  (singles and overall) and 13 doubles domains.

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}), resampling domains and 3-seed
  training replicates jointly with shared multinomial weights.

Hypotheses & Multiple-Testing Correction:
  H0: NDCG_ESM-MSR <= NDCG_X vs H1: NDCG_ESM-MSR > NDCG_X; p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction over m = 8 clean comparators within each subset (singles, doubles, overall).
""",
    )

    # Supp Fig 3 (doubles-trained vs singles-only on singles): source data + paired p-values
    _copy_as_sourcedata(mega_dir / 'doubles_vs_singles_on_singles.csv', SUPP_SRC / 'supp_figure3_doubles_vs_singles_summary_sourcedata.csv')
    pairs_s3 = [
        ('ESM-MSR', 'msrp0:wt', 'msrp0:comb'),
        ('MutateEverything', 'me1', 'me2'),
        ('ThermoMPNN', 'tmpnn', 'tmpnnd'),
        ('SPURS', 'spurs', 'spursm'),
    ]
    s3_dom_rows, v_s3 = [], {}
    for m_nm, k_s, k_d in pairs_s3:
        ds_df = per_library(fr, k_s, single, rho_fn, MIN_N)
        dd_df = per_library(fr, k_d, single, rho_fn, MIN_N)
        v_s3[(m_nm, 'singles_only')] = ds_df
        v_s3[(m_nm, 'doubles_trained')] = dd_df
        for dom in ds_df.index:
            for si in range(ds_df.shape[1]):
                s3_dom_rows.append(dict(model=m_nm, domain=dom, seed=si + 1, rho_singles_only=float(ds_df.loc[dom].iloc[si]), rho_doubles_trained=float(dd_df.loc[dom].iloc[si])))
    atomic_to_csv(pd.DataFrame(s3_dom_rows), SUPP_SRC / 'supp_figure3_doubles_vs_singles_per_domain_sourcedata.csv', index=False, float_format='%.6f')

    common_s3 = sorted(set.intersection(*(set(v_s3[(m, 'singles_only')].index) for m, _, _ in pairs_s3)))
    d_cnt_s3, s_w_s3 = make_domain_seed_draws(len(common_s3), B=B_FAST, seed=503, seed_counts=(3,))
    pt_s3, rp_s3 = {}, {}
    for m_nm, _, _ in pairs_s3:
        for tag in ('singles_only', 'doubles_trained'):
            arr = v_s3[(m_nm, tag)].reindex(common_s3).to_numpy(float)
            pt_s3[(m_nm, tag)], rp_s3[(m_nm, tag)] = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt_s3, seed_weights=s_w_s3)
    p_rows_s3 = []
    # ESM-MSR Dual-view vs other doubles-trained models on singles
    p_rows_s3 += build_comparison_rows(
        benchmark='MegaScale_capped_test_singles',
        subset_or_metric='doubles_trained_model_on_singles_spearman_rho',
        resampling_unit='domain_cluster_x_seed',
        msr_label='ESM-MSR Dual-view',
        msr_point=pt_s3[('ESM-MSR', 'doubles_trained')],
        msr_reps=rp_s3[('ESM-MSR', 'doubles_trained')],
        comparators=[dict(name=f'{m} (doubles-trained)', point=pt_s3[(m, 'doubles_trained')], reps=rp_s3[(m, 'doubles_trained')]) for m, _, _ in pairs_s3 if m != 'ESM-MSR'],
        higher_is_better=True,
        n_units=len(common_s3),
        n_variants=int((single & np.isfinite(sp['y'].to_numpy())).sum()),
        bonferroni_family='SuppFig3_doubles_trained_3_comparators',
    )
    # Within-model retention: doubles-trained >= singles-only (or test for degradation singles-only > doubles-trained)
    for m_nm, _, _ in pairs_s3:
        d_deg = rp_s3[(m_nm, 'singles_only')] - rp_s3[(m_nm, 'doubles_trained')]
        p_one, nf, Be = one_sided_boot_p(d_deg)
        p_rows_s3.append(dict(
            benchmark='MegaScale_capped_test_singles',
            subset_or_metric='within_model_degradation_singles_only_minus_doubles_trained',
            resampling_unit='domain_cluster_x_seed',
            test_type='within_model_degradation',
            msr_model=f'{m_nm} (singles-only)',
            comparator=f'{m_nm} (doubles-trained)',
            excluded_training_overlap=False,
            internal_control_or_variant=False,
            direction='higher_is_better',
            value_msr=float(pt_s3[(m_nm, 'singles_only')]),
            value_comparator=float(pt_s3[(m_nm, 'doubles_trained')]),
            delta_favoring_msr=float(pt_s3[(m_nm, 'singles_only')] - pt_s3[(m_nm, 'doubles_trained')]),
            ci95_lo=float(np.quantile(d_deg, 0.025)),
            ci95_hi=float(np.quantile(d_deg, 0.975)),
            se_boot=float(np.std(d_deg, ddof=1)),
            n_units=len(common_s3),
            n_variants=int((single & np.isfinite(sp['y'].to_numpy())).sum()),
            B=Be,
            n_boot_fail=nf,
            p_one_sided_uncorrected=p_one,
            bonferroni_m=len(pairs_s3),
            p_one_sided_bonferroni=float(min(1.0, len(pairs_s3) * p_one)),
            bonferroni_family='SuppFig3_within_model_degradation_4_models',
            notes='One-sided test that training with doubles degrades single-mutant Spearman rho relative to the singles-only counterpart (delta = singles_only - doubles_trained > 0).',
        ))
    save_table_and_annotation(
        pd.DataFrame(p_rows_s3),
        SUPP_PVAL / 'supp_figure3_doubles_vs_singles_pvalues.csv',
        f"""Statistical Annotation: supp_figure3_doubles_vs_singles_pvalues.csv
===================================================================
Figure: Supplementary Figure 3 (Doubles-trained vs singles-only versions evaluated on Homology-Restricted MegaScale test single mutants).

Metrics Tested:
  1. Between-model: ESM-MSR Dual-view vs MutateEverything Stage 2, ThermoMPNN-D, and SPURS-multi on single mutants (mean per-domain rho, 33 domains).
  2. Within-model degradation: One-sided test that singles-only > doubles-trained (i.e., whether training on doubles hurts single-mutant accuracy).

Statistical Test:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}), exact Monte Carlo one-sided p-value
  p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1), Bonferroni-corrected within each test family (m = 3 between-model; m = 4 within-model).
""",
    )

    # Supp Fig 4 (masking strategies on doubles): source data + paired p-values
    _copy_as_sourcedata(supp_dir / 'masking_schemes_values.csv', SUPP_SRC / 'supp_figure4_masking_schemes_sourcedata.csv')
    if (supp_dir / 'masking_schemes_pvalues.csv').exists():
        _copy_as_sourcedata(supp_dir / 'masking_schemes_pvalues.csv', SUPP_SRC / 'supp_figure4_masking_schemes_existing_pvalues_sourcedata.csv')
    # Recompute exact one-sided B=10,000 paired domain x seed p-values for ESM-MSR unmasked vs ind/mar & ESM3 zero-shot
    if 'msrp0:comb' in fr.preds:
        from msr_analysis.final_runs import load_final_frame
        f_mask = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
        d_fm = (f_mask.spine['use'].to_numpy() & (f_mask.spine['n_mut'] == 2).to_numpy()) if f_mask is not None else None
        mask_map = {
            'ESM-MSR (Unmasked)': per_library(fr, 'msrp0:comb', double, rho_fn, MIN_N),
            'ESM-MSR (One-at-a-time masking)': per_library(f_mask, 'FINAL_p1:comb', d_fm, rho_fn, MIN_N) if f_mask is not None and 'FINAL_p1:comb' in f_mask.preds else None,
            'ESM-MSR (Masked marginal)': per_library(f_mask, 'FINAL_p1_marg:comb', d_fm, rho_fn, MIN_N) if f_mask is not None and 'FINAL_p1_marg:comb' in f_mask.preds else None,
            'ESM3-small-open (Unmasked)': per_library(fr, 'msrrel@0:comb', double, rho_fn, MIN_N),
            'ESM3-small-open (One-at-a-time)': per_library(fr, 'esm3avg', double, rho_fn, MIN_N) if 'esm3avg' in fr.preds else None,
            'ESM3-small-open (Masked marginal)': per_library(fr, 'esm3mm', double, rho_fn, MIN_N) if 'esm3mm' in fr.preds else None,
        }
        mask_map = {k: v for k, v in mask_map.items() if v is not None and not v.empty}
        if 'ESM-MSR (Unmasked)' in mask_map and len(mask_map) >= 2:
            common_m4 = sorted(set.intersection(*(set(v.index) for v in mask_map.values())))
            d_cnt4, s_w4 = make_domain_seed_draws(len(common_m4), B=B_FAST, seed=504, seed_counts=(3,))
            pt4, rp4 = {}, {}
            for k, df_v in mask_map.items():
                arr = df_v.reindex(common_m4).to_numpy(float)
                pt4[k], rp4[k] = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt4, seed_weights=s_w4 if arr.shape[1] > 1 else None)
            comps4 = [dict(name=k, point=pt4[k], reps=rp4[k]) for k in mask_map if k != 'ESM-MSR (Unmasked)']
            p_rows_s4 = build_comparison_rows(
                benchmark='MegaScale_capped_test_doubles',
                subset_or_metric='masking_strategy_per_domain_spearman_rho',
                resampling_unit='domain_cluster_x_seed',
                msr_label='ESM-MSR (Unmasked)',
                msr_point=pt4['ESM-MSR (Unmasked)'],
                msr_reps=rp4['ESM-MSR (Unmasked)'],
                comparators=comps4,
                higher_is_better=True,
                n_units=len(common_m4),
                n_variants=int((double & np.isfinite(sp['y'].to_numpy())).sum()),
                bonferroni_family=f'SuppFig4_masking_{len(comps4)}_comparators',
            )
            save_table_and_annotation(
                pd.DataFrame(p_rows_s4),
                SUPP_PVAL / 'supp_figure4_masking_doubles_pvalues.csv',
                f"""Statistical Annotation: supp_figure4_masking_doubles_pvalues.csv
================================================================
Figure: Supplementary Figure 4 (Masking strategies on Homology-Restricted MegaScale test double mutants).

Metric Tested:
  Mean per-domain Spearman's rho on double mutants (13 test domains) comparing ESM-MSR (Unmasked) against
  ESM-MSR masking ablations (One-at-a-time masking, Masked marginal) and zero-shot ESM3-small-open variants.

Statistical Test:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B_FAST:,}); one-sided H1: ESM-MSR (Unmasked) > comparator;
  p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1); Bonferroni-corrected over all m = {len(comps4)} comparators.
""",
            )

    # Supp Fig 5 (ESM-MSR masking ablations across 5 epistasis levels): source data + paired p-values
    _copy_as_sourcedata(supp_dir / 'fig5_epistasis_levels_masking_bootstrap_test.csv', SUPP_SRC / 'supp_figure5_epistasis_masking_sourcedata.csv')
    T_mk = pd.read_csv(supp_dir / 'fig5_epistasis_levels_masking_bootstrap_test.csv')
    zm = np.load(supp_dir / 'fig5_epistasis_levels_masking_reps_test.npz', allow_pickle=True)
    reps_mk = zm['reps']
    models_mk = [str(x) for x in zm['models']]
    metrics_mk = [str(x) for x in zm['metrics']]
    B_mk = int(zm['B'])
    i_msr_mk = models_mk.index(p06.P0_EPI_NAME)
    p_rows_s5 = []
    for met, regime in (
        ('naive_rho', 'all_doubles'), ('naive_rho_gt0', 'dG_add_gt0'),
        ('beyond_rho', 'all_doubles'), ('beyond_rho_gt0', 'dG_add_gt0'),
        ('pair_rho', 'all_doubles'), ('pair_rho_gt0', 'dG_add_gt0'),
        ('line_rho', 'all_doubles'), ('line_rho_gt0', 'dG_add_gt0'),
        ('cell_mag_pooled', 'all_doubles'), ('cell_mag_gt0', 'dG_add_gt0'),
    ):
        if met not in metrics_mk:
            continue
        j_m = metrics_mk.index(met)
        pt_sub = T_mk[T_mk['metric'] == met].set_index('model')['value']
        comps = []
        for idx_c, raw_m in enumerate(models_mk):
            if idx_c == i_msr_mk or raw_m not in pt_sub.index:
                continue
            is_int = (raw_m == '−dG_add')
            comps.append(dict(name=raw_m, point=float(pt_sub[raw_m]), reps=reps_mk[:, idx_c, j_m], internal_variant=is_int))
        p_rows_s5 += build_comparison_rows(
            benchmark='MegaScale_epistasis_5level_masking_ablation',
            subset_or_metric=f'{met} ({regime})',
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR p0 (unmasked)',
            msr_point=float(pt_sub[p06.P0_EPI_NAME]),
            msr_reps=reps_mk[:, i_msr_mk, j_m],
            comparators=comps,
            higher_is_better=True,
            n_units=int(T_mk['n_domains'].iloc[0]),
            n_variants=None,
            bonferroni_family=f'SuppFig5_{met}_3_masking_ablations',
        )
    save_table_and_annotation(
        pd.DataFrame(p_rows_s5),
        SUPP_PVAL / 'supp_figure5_epistasis_masking_pvalues.csv',
        f"""Statistical Annotation: supp_figure5_epistasis_masking_pvalues.csv
==================================================================
Figure: Supplementary Figure 5 (ESM-MSR masking ablations across the 5-level pooled epistasis hierarchy).

Statistical Test:
  Exact paired domain-cloning x training-seed bootstrap replicates (`fig5_epistasis_levels_masking_reps_test.npz`,
  B = {B_mk:,}, N = 13 domains); one-sided H1: ESM-MSR p0 (unmasked) > comparator;
  p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1); Bonferroni-corrected across the m = 3 masking/control variants
  within each (level, regime) (`−dG_add` floor ceiling excluded from the masking family).
""",
    )

    # Supp Fig 6 (structure C-alpha sequence length distributions across 16 datasets): source data only
    if (len_dir / 'structure_length_summary.csv').exists():
        _copy_as_sourcedata(len_dir / 'structure_length_summary.csv', SUPP_SRC / 'supp_figure6_structure_lengths_summary_sourcedata.csv')

    # Supp Fig 7 (rectified gain per variant g(K) = a - b ln K): source data + paired domain x seed p-values
    print('[9/10] Exporting Supplementary Figure 7, _epi_rmse, _epi_rmse_bars, and _sequence_length...')
    mega_scr_csv = results_dir('megascale_screening', split='capped', create=False) / 'screening_per_protein_singles_all_models.csv'
    D_s7 = pd.read_csv(mega_scr_csv)
    D_s7_gain = D_s7[D_s7['metric'].isin(['gain', 'oracle_gain', 'rand_gain'])].copy()
    D_s7_gain['gain_per_variant'] = D_s7_gain['value'] / D_s7_gain['K']
    D_s7_gain['lo_per_variant'] = D_s7_gain['lo'] / D_s7_gain['K']
    D_s7_gain['hi_per_variant'] = D_s7_gain['hi'] / D_s7_gain['K']
    atomic_to_csv(D_s7_gain, SUPP_SRC / 'supp_figure7_gain_law_sourcedata.csv', index=False, float_format='%.6f')

    specs = p09.build_family_specs('capped')
    fams_all = list(specs['fam_all'])
    flab = specs['fam_label_singles']
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
    keep_s7 = (sp['n_mut'].to_numpy() == 1) & ~to_cys
    p_rows_s7 = []
    for K in p09.KS_CURVE:
        per_fam = {}
        for fam in fams_all:
            k = p09.fam_key(fam, fr, specs, mode='singles')
            if k is None or k not in fr.preds:
                continue
            ok = sp['use'].to_numpy() & keep_s7 & np.isfinite(fr.preds[k]).all(axis=0)
            tabs = [scr_core.per_protein_table(sp['group'].to_numpy()[ok], pr[ok], sp['y'].to_numpy()[ok], K, 0.5).set_index('group') for pr in fr.preds[k]]
            g_c = sorted(set.intersection(*(set(t.index) for t in tabs)))
            per_fam[fam] = (g_c, np.column_stack([t.reindex(g_c)['gain'].to_numpy(float) / K for t in tabs]))
        avail = [f for f in fams_all if f in per_fam]
        common_g = sorted(set.intersection(*(set(per_fam[f][0]) for f in avail)))
        d_cnt, s_w = make_domain_seed_draws(len(common_g), B=B_FAST, seed=507 + K, seed_counts=(3,))
        pts, rps = {}, {}
        for f in avail:
            idx_f = [per_fam[f][0].index(g) for g in common_g]
            arr = per_fam[f][1][idx_f]
            pts[f], rps[f] = boot_domain_seeds(arr, B=B_FAST, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
        p_rows_s7 += build_comparison_rows(
            benchmark='MegaScale_capped_test_singles_nocys',
            subset_or_metric=f'rectified_gain_per_variant@K={K}',
            resampling_unit='domain_cluster_x_seed',
            msr_label=flab.get('msr', 'ESM-MSR WT-view'),
            msr_point=pts['msr'],
            msr_reps=rps['msr'],
            comparators=[dict(name=flab.get(f, f), point=pts[f], reps=rps[f]) for f in avail if f != 'msr'],
            higher_is_better=True,
            n_units=len(common_g),
            n_variants=int((sp['use'].to_numpy() & keep_s7).sum()),
            bonferroni_family=f'SuppFig7_gain_per_variant_K{K}_clean_models',
        )
    save_table_and_annotation(
        pd.DataFrame(p_rows_s7),
        SUPP_PVAL / 'supp_figure7_gain_law_pvalues.csv',
        f"""Statistical Annotation: supp_figure7_gain_law_pvalues.csv
=========================================================
Figure: Supplementary Figure 7 (Rectified stability gain per screened variant g(K) on Homology-Restricted MegaScale test singles excl. Cys, 33 domains).

Metric Tested:
  Mean over 33 domains of rectified gain per screened variant (sum_{{i=1..K}} max(ddG_i, 0)) / K across budgets
  K in {{1, 3, 5, 10, 20, 50, 100}}.

Statistical Test:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}); one-sided H1: g_ESM-MSR(K) > g_X(K);
  p = (1 + #{{b: delta^(b) <= 0}}) / (B + 1); Bonferroni-corrected across all m = 9 clean comparators within each K.
""",
    )

    # _epi_rmse & _epi_rmse_bars: source data + paired cluster x seed bootstrap one-sided p-values (lower RMSE is better!)
    _copy_as_sourcedata(epi_dir / 'dddG_scatters' / 'main_models_dddG_rmse_bars.csv', SUPP_SRC / '_epi_rmse_bars_sourcedata.csv')
    d_pair, fit, meas, model_dict, _ = p07._build_pair_models(fr)
    y_epi = meas['raw']
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    cl_epi = sp['cluster'].astype(str).to_numpy()

    # Export full per-variant predicted vs true dddG source data for all 8 main models
    main8 = [lbl for lbl, _ in p07.MAIN_DDD_GRID_MODELS if lbl in model_dict]
    ok_all = fit & np.isfinite(y_epi)
    for lbl in main8:
        ok_all &= np.isfinite(model_dict[lbl]['raw_seeds']).all(axis=0)
    epi_src = dict(
        id=sp['id'].to_numpy()[ok_all],
        code_wt=sp['code_wt'].to_numpy()[ok_all],
        cluster=cl_epi[ok_all],
        mut=sp['mut'].to_numpy()[ok_all],
        dG_add=dG_add[ok_all],
        dddG_true=y_epi[ok_all],
    )
    for lbl in main8:
        arr = model_dict[lbl]['raw_seeds'][:, ok_all]
        slug = lbl.replace(' ', '_').replace('-', '_')
        if arr.shape[0] == 1:
            epi_src[f'pred_{slug}'] = arr[0]
        else:
            for si in range(arr.shape[0]):
                epi_src[f'pred_{slug}_seed{si + 1}'] = arr[si]
    atomic_to_csv(pd.DataFrame(epi_src), SUPP_SRC / '_epi_rmse_scatters_sourcedata.csv', index=False, float_format='%.5f')

    # Paired cluster x seed bootstrap B = 10,000 for epistasis RMSE (all doubles and dG_add > 0; LOWER is better!)
    p_rows_rmse = []
    for sub_name, m_sub in (('all_doubles', ok_all), ('dG_add_gt0', ok_all & (dG_add > 0.0))):
        u_cl, inv_cl = np.unique(cl_epi[m_sub], return_inverse=True)
        n_cl = len(u_cl)
        rng = np.random.default_rng(601 if sub_name == 'all_doubles' else 602)
        cl_w = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl), size=B_FAST).astype(float)  # (B, n_cl)
        sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0    # (B, 3)

        pts_r, rps_r = {}, {}
        for lbl in main8:
            e2 = (model_dict[lbl]['raw_seeds'][:, m_sub] - y_epi[None, m_sub]) ** 2  # (ns, n_var)
            ns = e2.shape[0]
            pts_r[lbl] = float(np.mean(np.sqrt(e2.mean(axis=1))))
            # Aggregate sum(e2) and count per cluster -> O(B * n_cl) exact cluster bootstrap!
            cnt_cl = np.bincount(inv_cl, minlength=n_cl).astype(float)
            s_e2_cl = np.vstack([np.bincount(inv_cl, weights=e2[si], minlength=n_cl) for si in range(ns)])  # (ns, n_cl)
            denom = cl_w @ cnt_cl  # (B,)
            num = cl_w @ s_e2_cl.T  # (B, ns)
            rm_per_seed = np.sqrt(num / denom[:, None])  # (B, ns)
            rps_r[lbl] = np.sum(sw3 * rm_per_seed, axis=1) if ns == 3 else rm_per_seed[:, 0]

        p_rows_rmse += build_comparison_rows(
            benchmark='MegaScale_capped_test_epistasis_RMSE',
            subset_or_metric=sub_name,
            resampling_unit='domain_cluster_x_seed',
            msr_label='ESM-MSR',
            msr_point=pts_r['ESM-MSR'],
            msr_reps=rps_r['ESM-MSR'],
            comparators=[dict(name=lbl, point=pts_r[lbl], reps=rps_r[lbl]) for lbl in main8 if lbl != 'ESM-MSR'],
            higher_is_better=False,
            n_units=n_cl,
            n_variants=int(m_sub.sum()),
            bonferroni_family=f'Supp_epi_rmse_{sub_name}_7_comparators',
        )
    save_table_and_annotation(
        pd.DataFrame(p_rows_rmse),
        SUPP_PVAL / '_epi_rmse_pvalues.csv',
        f"""Statistical Annotation: _epi_rmse_pvalues.csv
=============================================
Figure: Supplementary Figures `_epi_rmse` (2x4 dddG scatter, dG_add > 0) and `_epi_rmse_bars` (paired epistasis RMSE bar chart).

Metric Tested:
  Pooled epistasis RMSE = sqrt(mean((pred_dddG - true_dddG)^2)) across Homology-Restricted MegaScale test doubles
  (averaged over seeds 1–3 for 3-seed retrained models) for:
    - all_doubles (n = 12,455 across 13 homology clusters),
    - dG_add_gt0 (unsaturated doubles with dG_wt + ddG_A + ddG_B > 0; n = 7,405 across 13 clusters).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x training-seed bootstrap (B = {B_FAST:,}).
  In each draw, the 13 domain clusters are resampled identically across all 8 models via Multinomial(13, 1/13)
  weights (using exact per-cluster squared-error sufficient statistics), jointly with Multinomial(3, 1/3)
  seed weights for 3-seed models.

Hypotheses & One-Sided P-Value Formula:
  LOWER RMSE is better, so delta_favoring_msr = RMSE_X - RMSE_ESM-MSR:
    H0: RMSE_ESM-MSR >= RMSE_X  vs  H1: RMSE_ESM-MSR < RMSE_X.
  Exact Monte Carlo one-sided p-value: p = (1 + #{{b: RMSE_X^(b) - RMSE_ESM-MSR^(b) <= 0}}) / (B + 1).
  Bonferroni correction multiplies p_one_sided_uncorrected by m = 7 clean comparators within each subset.
""",
    )

    # _sequence_length: source data (panel a + panel b)
    _copy_as_sourcedata(len_dir / 'length_summary.csv', SUPP_SRC / '_sequence_length_panel_a_summary_sourcedata.csv')
    if (len_dir / 'per_protein_metrics_8models.csv').exists():
        _copy_as_sourcedata(len_dir / 'per_protein_metrics_8models.csv', SUPP_SRC / '_sequence_length_panel_b_sourcedata.csv')
    # Also export per-protein lengths for Panel a exact reproduction
    len_a_rows = []
    for ds in EXT12 + ['domainome', TSU_CAPPED[1]]:
        fr_l = load_frame(ds)
        sp_l = fr_l.spine[fr_l.spine['use']]
        for cl_nm, L_val in sp_l.groupby('cluster')['seq_len'].mean().items():
            len_a_rows.append(dict(dataset=ds, label=LABEL[ds], cluster=cl_nm, seq_len=float(L_val)))
    atomic_to_csv(pd.DataFrame(len_a_rows), SUPP_SRC / '_sequence_length_panel_a_per_protein_sourcedata.csv', index=False, float_format='%.2f')


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=['fig1', 'fig2', 'fig3a', 'fig3bc', 'fig4', 'supp'])
    args = ap.parse_args()
    st = set(args.stages)
    if 'fig1' in st:
        export_figure1()
    if 'fig2' in st:
        export_figure2()
    if 'fig3a' in st:
        export_figure3a()
    if 'fig3bc' in st:
        export_figure3bc()
    if 'fig4' in st:
        export_figure4()
    if 'supp' in st:
        export_supplementary()
    print('[10/10] Requested source data tables, p-value tables, and annotation files exported successfully!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
