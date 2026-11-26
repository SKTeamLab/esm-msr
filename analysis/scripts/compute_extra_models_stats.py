#!/usr/bin/env python3
"""Compute Spearman rho, Pearson r, raw RMSE, and one-sided paired bootstrap p-values
vs ESM-MSR ONLY for additional models (and the 4 additional external thermostability sets)
not already present in wt-dddg-scatters/analysis/results/{00_main_text_figs,00_supplementary_figs}/pvalues/*.csv.

Zero existing numbers from wt-dddg-scatters are overwritten or reproduced; this script fills
in the extra models we have predictions for (ESM-MSR sigma=0.5, DDGemb, DDMut, ESMtherm,
IFUM WT/ESMFold, ESM3-small/medium/large, SPURS published/unfrozen, ThermoMPNN published,
ProSST published) using the exact same resampling units, B, and Phipson-Smyth formula.

Output: analysis/results/si_latex_tables/_extra_models_pvalues.csv
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/sareeves/playground/esm-msr-devel')
WT_DDDG = ROOT / 'wt-dddg-scatters'
WT_SI = Path(__file__).resolve().parents[2]
OUT_CSV = WT_SI / 'analysis/results/si_latex_tables/_extra_models_pvalues.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

for _p in (str(WT_DDDG / 'src'), str(WT_DDDG / 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm_msr import epi_metrics as E  # noqa: E402
from msr_analysis import epistasis as EPI_LIB  # noqa: E402
from msr_analysis.data import Frame, load_frame  # noqa: E402
from msr_analysis.heldout import excluded as heldout_excluded  # noqa: E402
from msr_analysis.proteingym import PG_CATEGORIES, load_all_models_pgym  # noqa: E402
from msr_analysis.registry import LABEL  # noqa: E402
from pipelines import p04_mgnify_rocklin as p04  # noqa: E402
from pipelines._tsu_common import MIN_N, load_tsu_test_frame  # noqa: E402

B_FAST = 10000
B_MGNIFY = 1000


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    return float(pearsonr(x[ok], y[ok])[0])


def safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))


def one_sided_p(delta_draws: np.ndarray) -> float:
    d = np.asarray(delta_draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float((np.sum(d <= 0.0) + 1) / (len(d) + 1))


def _pooled_boot_pair(
    msr_2d: np.ndarray,
    comp_2d: np.ndarray,
    y: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    B: int,
    seed: int,
) -> tuple[float, float, float, float, float, float, float, int]:
    """Paired cluster x seed pooled Spearman bootstrap via O(n) rank engine.
    Returns (rho_msr, rho_comp, r_comp, rmse_comp, p_one_sided, r_msr, rmse_msr, n_valid).
    """
    m2 = np.atleast_2d(np.asarray(msr_2d, dtype=float))
    c2 = np.atleast_2d(np.asarray(comp_2d, dtype=float))
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(yv) & np.isfinite(m2).all(axis=0) & np.isfinite(c2).all(axis=0)
    n_ok = int(ok.sum())
    if n_ok < 5:
        return (np.nan,) * 7 + (n_ok,)
    m2 = m2[:, ok]
    c2 = c2[:, ok]
    yv = yv[ok]
    gc_raw = group_codes[ok]
    gc, uniq = pd.factorize(gc_raw)
    ng = len(uniq)

    rho_msr = float(np.mean([safe_spearman(m2[s], yv) for s in range(m2.shape[0])]))
    r_msr = float(np.mean([safe_pearson(m2[s], yv) for s in range(m2.shape[0])]))
    rmse_msr = float(np.mean([safe_rmse(m2[s], yv) for s in range(m2.shape[0])]))
    rho_comp = float(np.mean([safe_spearman(c2[s], yv) for s in range(c2.shape[0])]))
    r_comp = float(np.mean([safe_pearson(c2[s], yv) for s in range(c2.shape[0])]))
    rmse_comp = float(np.mean([safe_rmse(c2[s], yv) for s in range(c2.shape[0])]))

    rng = np.random.default_rng(seed)
    g_counts = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=B).astype(float)
    seed_w = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0

    xr_dict = {
        'msr': [p04._FastPooledRank(m2[s]) for s in range(m2.shape[0])],
        'comp': [p04._FastPooledRank(c2[s]) for s in range(c2.shape[0])],
    }
    _, reps = p04._eval_pooled_fast(
        xr_dict,
        p04._FastPooledRank(yv, need_inv=True),
        n_ok,
        gc.astype(np.int64),
        g_counts,
        seed_w,
        workers=1,
    )
    p_val = one_sided_p(reps['msr'] - reps['comp'])
    return rho_msr, rho_comp, r_comp, rmse_comp, p_val, r_msr, rmse_msr, n_ok


def _ext_dataset_worker(args: tuple[str, bool, list[tuple[str, str]]]) -> list[dict]:
    ds, is_multi, model_keys = args
    fr = load_frame(ds)
    sp = fr.spine
    y = sp['y'].to_numpy(float)
    n_total = len(y)
    mut_codes = np.arange(n_total, dtype=np.int64)
    prot_codes, prot_uniq = pd.factorize(sp['cluster'].astype(str))
    n_prot = len(prot_uniq)

    msr_key = 'msrp0:comb' if is_multi else 'msrp0:wt'
    msr_arr = np.atleast_2d(fr.preds[msr_key])
    bm_label = LABEL.get(ds, ds)

    rho_msr = float(np.mean([safe_spearman(msr_arr[s], y) for s in range(msr_arr.shape[0])]))
    r_msr = float(np.mean([safe_pearson(msr_arr[s], y) for s in range(msr_arr.shape[0])]))
    rmse_msr = float(np.mean([safe_rmse(msr_arr[s], y) for s in range(msr_arr.shape[0])]))

    # Partition models into full-coverage (batch together in ONE O(n) pass!) vs partial-coverage (e.g. DDMut on S669/S461)
    full_models: list[tuple[str, np.ndarray]] = []
    partial_models: list[tuple[str, np.ndarray]] = []
    y_fin = np.isfinite(y) & np.isfinite(msr_arr).all(axis=0)
    for mname, pkey in model_keys:
        if pkey not in fr.preds:
            continue
        comp_arr = np.atleast_2d(fr.preds[pkey])
        if not np.isfinite(comp_arr).any():
            continue
        if np.isfinite(comp_arr[:, y_fin]).all():
            full_models.append((mname, comp_arr))
        else:
            partial_models.append((mname, comp_arr))

    rows = []
    # 1. Batch all full-coverage models in a single fast call for mutation and protein resampling
    if full_models:
        yv = y[y_fin]
        n_ok = int(y_fin.sum())
        mc_ok = np.arange(n_ok, dtype=np.int64)
        pc_ok, pc_u = pd.factorize(sp['cluster'].astype(str).to_numpy()[y_fin])
        n_pc = len(pc_u)
        m2_ok = msr_arr[:, y_fin]

        xr_dict = {'__MSR__': [p04._FastPooledRank(m2_ok[s]) for s in range(m2_ok.shape[0])]}
        for mname, c_arr in full_models:
            c_ok = c_arr[:, y_fin]
            xr_dict[mname] = [p04._FastPooledRank(c_ok[s]) for s in range(c_ok.shape[0])]

        yr = p04._FastPooledRank(yv, need_inv=True)
        rng_m = np.random.default_rng(1001)
        g_mut = rng_m.multinomial(n_ok, np.full(n_ok, 1.0 / n_ok), size=B_FAST).astype(float)
        sw_m = rng_m.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0
        pt_m, rep_m = p04._eval_pooled_fast(xr_dict, yr, n_ok, mc_ok, g_mut, sw_m, workers=1)

        rng_p = np.random.default_rng(2001)
        g_prot = rng_p.multinomial(n_pc, np.full(n_pc, 1.0 / n_pc), size=B_FAST).astype(float)
        sw_p = rng_p.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0
        _, rep_p = p04._eval_pooled_fast(xr_dict, yr, n_ok, pc_ok.astype(np.int64), g_prot, sw_p, workers=1)

        for mname, c_arr in full_models:
            c_ok = c_arr[:, y_fin]
            rows.append({
                'section': 'thermostability',
                'benchmark': bm_label,
                'dataset_key': ds,
                'subset': 'pooled_spearman_rho',
                'resampling_unit': 'mutation_x_seed',
                'model': mname,
                'value_msr': rho_msr,
                'pearson_r_msr': r_msr,
                'rmse_msr': rmse_msr,
                'value': float(pt_m[mname][0]),
                'pearson_r': float(np.mean([safe_pearson(c_ok[s], yv) for s in range(c_ok.shape[0])])),
                'rmse': float(np.mean([safe_rmse(c_ok[s], yv) for s in range(c_ok.shape[0])])),
                'p_one_sided': one_sided_p(rep_m['__MSR__'] - rep_m[mname]),
                'p_one_sided_prot': one_sided_p(rep_p['__MSR__'] - rep_p[mname]),
                'n_units': n_pc,
                'n_variants': n_ok,
                'B': B_FAST,
            })

    # 2. Partial-coverage models individually on their valid intersection
    for idx_m, (mname, comp_arr) in enumerate(partial_models):
        _, rho_c, r_c, rm_c, p_mut, _, _, n_ok = _pooled_boot_pair(
            msr_arr, comp_arr, y, mut_codes, n_total, B=B_FAST, seed=3000 + idx_m
        )
        _, _, _, _, p_prot, _, _, _ = _pooled_boot_pair(
            msr_arr, comp_arr, y, prot_codes, n_prot, B=B_FAST, seed=4000 + idx_m
        )
        rows.append({
            'section': 'thermostability',
            'benchmark': bm_label,
            'dataset_key': ds,
            'subset': 'pooled_spearman_rho',
            'resampling_unit': 'mutation_x_seed',
            'model': mname,
            'value_msr': rho_msr,
            'pearson_r_msr': r_msr,
            'rmse_msr': rmse_msr,
            'value': rho_c,
            'pearson_r': r_c,
            'rmse': rm_c,
            'p_one_sided': p_mut,
            'p_one_sided_prot': p_prot,
            'n_units': n_prot,
            'n_variants': n_ok,
            'B': B_FAST,
        })
    print(f'  finished ext dataset {ds} ({len(rows)} models)', flush=True)
    return rows


def compute_thermostability_extras() -> list[dict]:
    print('[1/5] Computing extra models for all 12 external thermostability benchmarks...')
    main8 = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754', 'ptmul']
    extra4 = ['s571', 's669', 'k3822', 'ptmuld']

    def model_list(is_multi: bool, include_main: bool) -> list[tuple[str, str]]:
        extras = [] if is_multi else [('ESM-MSR Dual-View (incl. singles)', 'msrp0:comb')]
        extras += [
            ('ESM-MSR σ=0.5', 'msrp0@0.5:comb' if is_multi else 'msrp0@0.5:wt'),
            ('DDGemb', 'ddgemb'),
            ('DDMut', 'ddmut'),
            ('ESMtherm', 'esmtherm'),
            ('IFUM (WT backbone)', 'ifum_wt'),
            ('IFUM (ESMFold)', 'ifum_esmfold'),
            ('ESM3-small', 'esm3s'),
            ('ESM3-medium', 'esm3m'),
            ('ESM3-large', 'esm3l'),
        ]
        if not include_main:
            return extras
        mains = [
            ('SPURS', 'spursm' if is_multi else 'spurs'),
            ('MutateEverything', 'me2' if is_multi else 'me1'),
            ('ThermoMPNN', 'tmpnnd' if is_multi else 'tmpnn'),
            ('GeoDDG-3D', 'geoddg'),
            ('ESM3ΔG', 'esm3dg'),
            ('Rosetta', 'rosetta'),
            ('ESM3-small-open', 'msrrel@0:comb' if is_multi else 'msrrel@0:wt'),
            ('ProteinMPNN', 'pmpnn'),
            ('ProSST', 'prosst'),
        ]
        return mains + extras

    tasks = []
    for ds in main8:
        tasks.append((ds, ds in ('ptmul', 'ptmuld'), model_list(ds in ('ptmul', 'ptmuld'), include_main=False)))
    for ds in extra4:
        tasks.append((ds, ds in ('ptmul', 'ptmuld'), model_list(ds in ('ptmul', 'ptmuld'), include_main=True)))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for chunk in ex.map(_ext_dataset_worker, tasks):
            rows.extend(chunk)
    return rows


def _domain_mean_boot(
    fr: Frame,
    msr_key: str,
    extra_models: list[tuple[str, str]],
    mask: np.ndarray,
    group_col: str,
    section: str,
    benchmark: str,
    subset: str,
    B: int = B_FAST,
    seed: int = 42,
    min_n: int = MIN_N,
) -> list[dict]:
    """Paired cluster x seed bootstrap of per-group mean Spearman rho (+ mean Pearson r & RMSE)."""
    sp = fr.spine
    y = sp['y'].to_numpy(float)
    g = sp[group_col].astype(str).to_numpy()
    msr_2d = np.atleast_2d(fr.preds[msr_key])
    groups = sorted(np.unique(g[mask]))

    def per_group_stats(arr_2d: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
        rho_m, r_m, rm_m = {}, {}, {}
        for grp in groups:
            m = mask & (g == grp) & np.isfinite(y) & np.isfinite(arr_2d).all(axis=0)
            if m.sum() < min_n:
                continue
            rho_m[grp] = np.array([safe_spearman(arr_2d[s, m], y[m]) for s in range(arr_2d.shape[0])], float)
            r_m[grp] = float(np.nanmean([safe_pearson(arr_2d[s, m], y[m]) for s in range(arr_2d.shape[0])]))
            rm_m[grp] = float(np.nanmean([safe_rmse(arr_2d[s, m], y[m]) for s in range(arr_2d.shape[0])]))
        return rho_m, r_m, rm_m

    msr_rho, msr_r, msr_rm = per_group_stats(msr_2d)
    valid_groups = sorted(msr_rho.keys())
    if not valid_groups:
        return []
    n_g = len(valid_groups)
    n_var = int((mask & np.isin(g, valid_groups) & np.isfinite(y)).sum())

    rng = np.random.default_rng(seed)
    g_w = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=B).astype(float) / n_g  # (B, n_g)
    s_w3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0  # (B, 3)

    M_msr = np.vstack([msr_rho[grp] for grp in valid_groups])  # (n_g, S_m)
    if M_msr.shape[1] == 3:
        reps_msr = np.einsum('bg,gs,bs->b', g_w, M_msr, s_w3)
    else:
        reps_msr = g_w @ M_msr.mean(axis=1)
    pt_msr = float(np.nanmean(M_msr))
    pt_r_msr = float(np.nanmean([msr_r[grp] for grp in valid_groups]))
    pt_rm_msr = float(np.nanmean([msr_rm[grp] for grp in valid_groups]))

    rows = []
    for mname, pkey in extra_models:
        if pkey not in fr.preds:
            continue
        arr_2d = np.atleast_2d(fr.preds[pkey])
        if not np.isfinite(arr_2d[:, mask]).any():
            continue
        c_rho, c_r, c_rm = per_group_stats(arr_2d)
        common = [grp for grp in valid_groups if grp in c_rho and np.isfinite(c_rho[grp]).all()]
        if len(common) < max(3, int(0.8 * n_g)):
            continue
        idx_c = [valid_groups.index(grp) for grp in common]
        gw_c = g_w[:, idx_c]
        gw_c = gw_c / np.maximum(gw_c.sum(axis=1, keepdims=True), 1e-12)
        Mc = np.vstack([c_rho[grp] for grp in common])
        Mm = np.vstack([msr_rho[grp] for grp in common])
        reps_m_c = np.einsum('bg,gs,bs->b', gw_c, Mm, s_w3) if Mm.shape[1] == 3 else (gw_c @ Mm.mean(axis=1))
        reps_c = np.einsum('bg,gs,bs->b', gw_c, Mc, s_w3) if Mc.shape[1] == 3 else (gw_c @ Mc.mean(axis=1))
        rows.append({
            'section': section,
            'benchmark': benchmark,
            'dataset_key': benchmark,
            'subset': subset,
            'resampling_unit': f'{group_col}_x_seed',
            'model': mname,
            'value_msr': pt_msr,
            'pearson_r_msr': pt_r_msr,
            'rmse_msr': pt_rm_msr,
            'value': float(np.nanmean(Mc)),
            'pearson_r': float(np.nanmean([c_r[grp] for grp in common])),
            'rmse': float(np.nanmean([c_rm[grp] for grp in common])),
            'p_one_sided': one_sided_p(reps_m_c - reps_c),
            'p_one_sided_prot': np.nan,
            'n_units': len(common),
            'n_variants': n_var,
            'B': B,
        })
    return rows


def compute_megascale_extras() -> list[dict]:
    print('[2/5] Computing extra models for MegaScale (Overall/Stab/Destab, Doubles, MegaSplits, Backbone/Origin)...')
    rows: list[dict] = []
    # Exact frame used by figure1b / figure1c / _backbone_and_origin
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    use = sp['use'].to_numpy(bool)
    y = sp['y'].to_numpy(float)
    n_mut = sp['n_mut'].to_numpy(int)

    # S2a: Overall, Stabilizing, Destabilizing (WT-view for ESM-MSR singles/overall matching Figure 1b!)
    extra_s2a = [
        ('ESM-MSR σ=0.5', 'msrp0@0.5:wt'),
        ('DDGemb', 'ddgemb'),
        ('ESMtherm', 'esmtherm'),
        ('IFUM (WT backbone)', 'ifum_wt'),
        ('IFUM (ESMFold)', 'ifum_esmfold'),
    ]
    for sub_name, m_sub in [
        ('per_domain_mean_spearman_overall', use),
        ('per_domain_mean_spearman_stabilizing', use & (y > 0)),
        ('per_domain_mean_spearman_destabilizing', use & (y <= 0)),
    ]:
        rows.extend(_domain_mean_boot(
            fr, 'msrp0:wt', extra_s2a, m_sub, 'group',
            section='megascale_s2a', benchmark='MegaScale_capped_test', subset=sub_name,
        ))

    # S2b: Doubles Epistatic & Additive
    dbl_mask = use & (n_mut == 2)
    extra_dbl_epi = [
        ('ESM-MSR σ=0.5', 'msrp0@0.5:comb'),
        ('DDGemb', 'ddgemb'),
        ('ESMtherm', 'esmtherm'),
        ('IFUM (WT backbone)', 'ifum_wt'),
        ('IFUM (ESMFold)', 'ifum_esmfold'),
    ]
    extra_dbl_add = [
        ('ESM-MSR σ=0.5', 'msrp0@0.5:wt+'),
        ('DDGemb', 'ddgemb+'),
        ('ESMtherm', 'esmtherm+'),
        ('IFUM (WT backbone)', 'ifum_wt+'),
        ('IFUM (ESMFold)', 'ifum_esmfold+'),
    ]
    rows.extend(_domain_mean_boot(
        fr, 'msrp0:comb', extra_dbl_epi, dbl_mask, 'group',
        section='megascale_s2b', benchmark='MegaScale_capped_test_doubles', subset='per_domain_mean_spearman_epistatic',
    ))
    rows.extend(_domain_mean_boot(
        fr, 'msrp0:wt+', extra_dbl_add, dbl_mask, 'group',
        section='megascale_s2b', benchmark='MegaScale_capped_test_doubles', subset='per_domain_mean_spearman_additive',
    ))

    # S2d: Backbone (native vs destabilized) & Origin (natural vs synthetic)
    if 'code' in sp.columns:
        is_destab = sp['code'].astype(str).str.contains(r'_[A-Z]\d+[A-Z]|^v2', regex=True).to_numpy()
        rows.extend(_domain_mean_boot(
            fr, 'msrp0:wt', extra_s2a, use & (~is_destab), 'code',
            section='megascale_s2d', benchmark='MegaScale_capped_test_backbone', subset='per_code_mean_spearman_native',
        ))
        rows.extend(_domain_mean_boot(
            fr, 'msrp0:wt', extra_s2a, use & is_destab, 'code',
            section='megascale_s2d', benchmark='MegaScale_capped_test_backbone', subset='per_code_mean_spearman_destabilized',
        ))
    # Origin: check natural vs synthetic using group naming (same heuristic as p02)
    is_synth = sp['group'].astype(str).str.contains(r'rd\d|TrROS|EEHEE|HEEH|HHH|EHEE|r\d+_', regex=True).to_numpy()
    rows.extend(_domain_mean_boot(
        fr, 'msrp0:wt', extra_s2a, use & (~is_synth), 'group',
        section='megascale_s2d', benchmark='MegaScale_capped_test_origin', subset='per_group_mean_spearman_natural',
    ))
    rows.extend(_domain_mean_boot(
        fr, 'msrp0:wt', extra_s2a, use & is_synth, 'group',
        section='megascale_s2d', benchmark='MegaScale_capped_test_origin', subset='per_group_mean_spearman_synthetic',
    ))

    # S2c: Full MegaScale (mega_splits) extra published models
    fr_ms = load_frame('tsuboyama_mega_test')
    sp_ms = fr_ms.spine
    use_ms = sp_ms['use'].to_numpy(bool)
    y_ms = sp_ms['y'].to_numpy(float)
    n_mut_ms = sp_ms['n_mut'].to_numpy(int)
    extra_ms_s = [
        ('SPURS published', 'spurspub'),
        ('ThermoMPNN published*', 'tmpnnpub'),
    ]
    for sub_name, m_sub in [
        ('per_domain_mean_spearman_overall', use_ms),
        ('per_domain_mean_spearman_stabilizing', use_ms & (y_ms > 0)),
        ('per_domain_mean_spearman_destabilizing', use_ms & (y_ms <= 0)),
    ]:
        rows.extend(_domain_mean_boot(
            fr_ms, 'msrp0:wt', extra_ms_s, m_sub, 'group',
            section='megascale_s2c', benchmark='MegaScale_mega_splits_test', subset=sub_name,
        ))
    extra_ms_dbl = [
        ('SPURS published', 'spurspubm'),
        ('ThermoMPNN published*', 'tmpnndpub'),
    ]
    rows.extend(_domain_mean_boot(
        fr_ms, 'msrp0:comb', extra_ms_dbl, use_ms & (n_mut_ms == 2), 'group',
        section='megascale_s2c', benchmark='MegaScale_mega_splits_test_doubles', subset='per_domain_mean_spearman_epistatic',
    ))
    rows.extend(_domain_mean_boot(
        fr_ms, 'msrp0:wt', [('SPURS published', 'spurspub'), ('ThermoMPNN published*', 'tmpnnpub')], use_ms & (n_mut_ms == 2), 'group',
        section='megascale_s2c', benchmark='MegaScale_mega_splits_test_doubles', subset='per_domain_mean_spearman_additive',
    ))
    return rows


def compute_epistasis_extras() -> list[dict]:
    print('[3/5] Computing extra models for MegaScale Epistasis Hierarchy & Calibration...')
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    idx = np.flatnonzero(sp['use'].to_numpy(bool))
    code = sp['code'].astype(str).to_numpy()[idx] if 'code' in sp.columns else sp['code_wt'].astype(str).to_numpy()[idx]
    mut = sp['mut'].astype(str).to_numpy()[idx] if 'mut' in sp.columns else sp['mut_type'].astype(str).to_numpy()[idx]
    keys = [EPI_LIB.parse_mut_key(m, c) for m, c in zip(mut, code)]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    base_items = dict(
        code=code,
        mut_key=keys,
        subset_type=st,
        ddG=sp['y'].to_numpy(float)[idx],
        dddG=sp['dddG_ML'].to_numpy(float)[idx],
        dG_wt=sp['dG_wt'].to_numpy(float)[idx],
    )

    ddg_add = sp['ddG_additive_ML'].to_numpy(float)[idx] if 'ddG_additive_ML' in sp.columns else np.full(len(idx), np.nan)
    dg_add_row = sp['dG_wt'].to_numpy(float)[idx] + ddg_add
    keep_gt0 = (st == 'single') | ((st == 'double') & (dg_add_row > 0))
    base_items_gt0 = {k: (v[keep_gt0] if isinstance(v, np.ndarray) else [x for x, b in zip(v, keep_gt0) if b]) for k, v in base_items.items()}

    def eval_epi_model(wt_key: str, comb_key: str) -> dict[str, float]:
        wt_2d = np.atleast_2d(fr.preds[wt_key])[:, idx]
        cb_2d = np.atleast_2d(fr.preds[comb_key])[:, idx]
        n_s = min(wt_2d.shape[0], cb_2d.shape[0])
        metrics_acc: dict[str, list[float]] = {}
        r_all, rm_all, r_gt0, rm_gt0 = [], [], [], []
        for s in range(n_s):
            items = dict(base_items, wt=wt_2d[s], mt=cb_2d[s], comb=cb_2d[s])
            tbl = E.Table(items, None, heads=('comb', 'add'))
            res = E.compute(tbl)
            for mk in ('epi_naive_rho_comb', 'epi_beyond_rho_comb', 'epi_pair_rho_comb', 'epi_line_rho_comb', 'epi_cell_mag_comb'):
                metrics_acc.setdefault(mk, []).append(float(res.get(mk, np.nan)))
            d_pred = tbl.P['comb'] - (tbl.PA['add'] + tbl.PB['add'])
            d_true = tbl.dddG
            dg_add = tbl.x
            r_all.append(safe_pearson(d_pred, d_true))
            rm_all.append(safe_rmse(d_pred, d_true))
            gt0 = dg_add > 0
            r_gt0.append(safe_pearson(d_pred[gt0], d_true[gt0]))
            rm_gt0.append(safe_rmse(d_pred[gt0], d_true[gt0]))
            items_gt0 = dict(base_items_gt0, wt=wt_2d[s][keep_gt0], mt=cb_2d[s][keep_gt0], comb=cb_2d[s][keep_gt0])
            tbl_gt0 = E.Table(items_gt0, None, heads=('comb', 'add'))
            res_gt0 = E.compute(tbl_gt0)
            metrics_acc.setdefault('cell_mag_gt0', []).append(float(res_gt0.get('epi_cell_mag_comb', np.nan)))
        out = {k: float(np.nanmean(v)) for k, v in metrics_acc.items()}
        out['pearson_all'] = float(np.nanmean(r_all))
        out['rmse_all'] = float(np.nanmean(rm_all))
        out['pearson_gt0'] = float(np.nanmean(r_gt0))
        out['rmse_gt0'] = float(np.nanmean(rm_gt0))
        return out

    extra_epi = [
        ('ESM-MSR σ=0.5', 'msrp0@0.5:wt', 'msrp0@0.5:comb'),
        ('DDGemb', 'ddgemb', 'ddgemb'),
        ('ESMtherm', 'esmtherm', 'esmtherm'),
        ('IFUM (WT backbone)', 'ifum_wt', 'ifum_wt'),
        ('IFUM (ESMFold)', 'ifum_esmfold', 'ifum_esmfold'),
    ]
    rows: list[dict] = []
    for mname, wk, ck in extra_epi:
        if wk not in fr.preds or ck not in fr.preds:
            continue
        vals = eval_epi_model(wk, ck)
        for src_k, met_name in [
            ('epi_naive_rho_comb', 'naive_rho (all_doubles)'),
            ('epi_beyond_rho_comb', 'beyond_rho (all_doubles)'),
            ('epi_pair_rho_comb', 'pair_rho (all_doubles)'),
            ('epi_line_rho_comb', 'line_rho (all_doubles)'),
            ('epi_cell_mag_comb', 'cell_mag_pooled (all_doubles)'),
            ('cell_mag_gt0', 'cell_mag_gt0 (dG_add_gt0)'),
        ]:
            rows.append({
                'section': 'epistasis_s3a',
                'benchmark': 'MegaScale_epistasis_5level_hierarchy',
                'dataset_key': 'tsuboyama_capped_test',
                'subset': met_name,
                'resampling_unit': 'domain_cluster_x_seed',
                'model': mname,
                'value_msr': np.nan,
                'pearson_r_msr': np.nan,
                'rmse_msr': np.nan,
                'value': vals.get(src_k, np.nan),
                'pearson_r': vals['pearson_all'],
                'rmse': vals['rmse_all'],
                'p_one_sided': np.nan,
                'p_one_sided_prot': np.nan,
                'n_units': 13,
                'n_variants': 12455,
                'B': B_FAST,
            })
        # Also record epistasis calibration RMSE rows for S3b
        for sub_k, rm_k, r_k, n_v in [
            ('all_doubles', 'rmse_all', 'pearson_all', 12455),
            ('dG_add_gt0', 'rmse_gt0', 'pearson_gt0', 7405),
        ]:
            rows.append({
                'section': 'epistasis_s3b_rmse',
                'benchmark': 'MegaScale_capped_test_epistasis_RMSE',
                'dataset_key': 'tsuboyama_capped_test',
                'subset': sub_k,
                'resampling_unit': 'domain_cluster_x_seed',
                'model': mname,
                'value_msr': np.nan,
                'pearson_r_msr': np.nan,
                'rmse_msr': np.nan,
                'value': vals[rm_k],
                'pearson_r': vals[r_k],
                'rmse': vals[rm_k],
                'p_one_sided': np.nan,
                'p_one_sided_prot': np.nan,
                'n_units': 13,
                'n_variants': n_v,
                'B': B_FAST,
            })
    return rows


def compute_domainome_and_pgym_extras() -> list[dict]:
    print('[4/5] Computing extra models for Held-Out Domainome (436) and ProteinGym (173)...')
    rows: list[dict] = []

    # Domainome (436 held-out domains)
    fr_d = load_frame('domainome')
    sp_d = fr_d.spine
    bad_d = set(heldout_excluded('domainome'))
    m_d = sp_d['use'].to_numpy(bool) & (~sp_d['group'].astype(str).isin(bad_d).to_numpy())
    extra_dom = [
        ('ESM-MSR Dual-View (incl. singles)', 'msrp0:comb'),
        ('DDGemb', 'ddgemb'),
        ('ESMtherm', 'esmtherm'),
        ('IFUM (WT backbone)', 'ifum_wt'),
        ('IFUM (ESMFold)', 'ifum_esmfold'),
        ('ESM3-large', 'esm3l'),
    ]
    rows.extend(_domain_mean_boot(
        fr_d, 'msrp0:wt', extra_dom, m_d, 'group',
        section='ood_s5a', benchmark='Domainome_heldout_436_domains', subset='per_domain_mean_spearman_rho',
    ))

    # ProteinGym (173 held-out assays)
    meta_df, pg_dict = load_all_models_pgym(heldout=True)
    cats = meta_df['category'].astype(str).to_numpy()
    uniq_cats = list(PG_CATEGORIES)
    msr_df = pg_dict['ESM-MSR']  # (173, 3)
    M_msr = msr_df.to_numpy(float)

    # Category-stratified assay x seed bootstrap (matching figure3b)
    rng = np.random.default_rng(123)
    s_w3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0
    cat_weights = np.zeros((B_FAST, len(cats)), dtype=float)
    for c in uniq_cats:
        idx_c = np.flatnonzero(cats == c)
        w_c = rng.multinomial(len(idx_c), np.full(len(idx_c), 1.0 / len(idx_c)), size=B_FAST).astype(float) / len(idx_c)
        cat_weights[:, idx_c] = w_c / len(uniq_cats)

    def macro_point(arr_2d: np.ndarray) -> float:
        per_a = np.nanmean(arr_2d, axis=1)
        return float(np.mean([np.nanmean(per_a[cats == c]) for c in uniq_cats]))

    reps_msr = np.einsum('ba,as,bs->b', cat_weights, M_msr, s_w3)
    pt_msr = macro_point(M_msr)

    for mname in ('ESMtherm', 'ProSST published'):
        if mname not in pg_dict:
            continue
        arr = pg_dict[mname].to_numpy(float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] == 3:
            reps_c = np.einsum('ba,as,bs->b', cat_weights, arr, s_w3)
        else:
            reps_c = cat_weights @ np.nanmean(arr, axis=1)
        rows.append({
            'section': 'ood_s5a',
            'benchmark': 'ProteinGym_heldout_173_assays',
            'dataset_key': 'proteingym',
            'subset': 'category_weighted_mean_spearman_rho',
            'resampling_unit': 'category_stratified_assay_x_seed',
            'model': mname,
            'value_msr': pt_msr,
            'pearson_r_msr': np.nan,
            'rmse_msr': np.nan,
            'value': macro_point(arr),
            'pearson_r': np.nan,
            'rmse': np.nan,
            'p_one_sided': one_sided_p(reps_msr - reps_c),
            'p_one_sided_prot': np.nan,
            'n_units': len(cats),
            'n_variants': 2351490,
            'B': B_FAST,
        })

    # ProteinGym Dual-View (incl. singles): load combined_pred from final_pgym_comb/FINAL_p0/s{1,2,3}
    assays_pg = list(msr_df.index)
    M_cb_rho = np.full((len(assays_pg), 3), np.nan, dtype=float)
    M_cb_r = np.full((len(assays_pg), 3), np.nan, dtype=float)
    for s_idx, s in enumerate((1, 2, 3)):
        pg_dir = ROOT / 'final_pgym_comb/FINAL_p0' / f's{s}'
        if not pg_dir.exists():
            continue
        for a_i, assay in enumerate(assays_pg):
            f_a = pg_dir / f'{assay}.csv'
            if not f_a.exists():
                ms = list(pg_dir.glob(f'{assay}*.csv'))
                f_a = ms[0] if ms else None
            if f_a is not None and f_a.exists():
                da = pd.read_csv(f_a, usecols=['DMS_score', 'combined_pred'])
                yv_a = da['DMS_score'].to_numpy(float)
                pv_a = da['combined_pred'].to_numpy(float)
                M_cb_rho[a_i, s_idx] = safe_spearman(pv_a, yv_a)
                M_cb_r[a_i, s_idx] = safe_pearson(pv_a, yv_a)
    if np.isfinite(M_cb_rho).all():
        reps_cb = np.einsum('ba,as,bs->b', cat_weights, M_cb_rho, s_w3)
        rows.append({
            'section': 'ood_s5a',
            'benchmark': 'ProteinGym_heldout_173_assays',
            'dataset_key': 'proteingym',
            'subset': 'category_weighted_mean_spearman_rho',
            'resampling_unit': 'category_stratified_assay_x_seed',
            'model': 'ESM-MSR Dual-View (incl. singles)',
            'value_msr': pt_msr,
            'pearson_r_msr': np.nan,
            'rmse_msr': np.nan,
            'value': macro_point(M_cb_rho),
            'pearson_r': macro_point(M_cb_r),
            'rmse': np.nan,
            'p_one_sided': one_sided_p(reps_msr - reps_cb),
            'p_one_sided_prot': np.nan,
            'n_units': len(cats),
            'n_variants': 2351490,
            'B': B_FAST,
        })
    return rows


def compute_mgnify_rocklin_extras() -> list[dict]:
    print('[5/5] Computing extra models (ESMtherm, IFUM) for Held-Out MGnify & Rocklin...')
    rows: list[dict] = []
    for ds_name, bm_key, cfg, seed in (
        ('MGnify', 'MGnify_heldout_pooled', dict(run='all', src='mgnify'), 77),
        ('Rocklin designs', 'Rocklin_designs_heldout_pooled', dict(run='rdenovo', src='rocklin_denovo'), 78),
    ):
        d_df, preds_mg = p04._prepare_heldout_models(cfg)
        code, uniq = pd.factorize(d_df['code_wt'])
        code = code.astype(np.int64)
        n_uniq = len(uniq)
        y_mg = d_df['y'].to_numpy(float)

        rng_p = np.random.default_rng(seed)
        scaf_counts = rng_p.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq), size=B_MGNIFY).astype(float)
        seed_w = rng_p.multinomial(3, np.full(3, 1.0 / 3.0), size=B_MGNIFY).astype(float) / 3.0

        target_models = ['ESM-MSR', 'ESMtherm', 'IFUM']
        xr_dict = {}
        raw_p = {}
        for mn in target_models:
            if mn not in preds_mg:
                continue
            P_pool = preds_mg[mn][2]
            if np.all([np.isfinite(p).all() for p in P_pool]):
                xr_dict[mn] = [p04._FastPooledRank(p) for p in P_pool]
                raw_p[mn] = P_pool
        pt_mg, rep_mg = p04._eval_pooled_fast(
            xr_dict, p04._FastPooledRank(y_mg, need_inv=True), len(y_mg), code, scaf_counts, seed_w, workers=8,
        )
        msr_reps = rep_mg['ESM-MSR']
        msr_pt = float(pt_mg['ESM-MSR'][0])
        for mn in ('ESMtherm', 'IFUM'):
            if mn not in pt_mg:
                continue
            P_c = raw_p[mn]
            rows.append({
                'section': 'ood_s5a',
                'benchmark': bm_key,
                'dataset_key': cfg['src'],
                'subset': 'pooled_spearman_rho',
                'resampling_unit': 'scaffold_cluster_x_seed',
                'model': mn,
                'value_msr': msr_pt,
                'pearson_r_msr': np.nan,
                'rmse_msr': np.nan,
                'value': float(pt_mg[mn][0]),
                'pearson_r': float(np.mean([safe_pearson(p, y_mg) for p in P_c])),
                'rmse': float(np.mean([safe_rmse(p, y_mg) for p in P_c])),
                'p_one_sided': one_sided_p(msr_reps - rep_mg[mn]),
                'p_one_sided_prot': np.nan,
                'n_units': n_uniq,
                'n_variants': len(y_mg),
                'B': B_MGNIFY,
            })
    return rows


def compute_thermo_rmse_dualview_extras() -> list[dict]:
    print('[6/7] Computing Dual-View (incl. singles) raw RMSE & p-values for Table S4c...')
    rows: list[dict] = []
    bms = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754', 'ptmul']
    for idx_b, ds in enumerate(bms):
        fr = load_frame(ds)
        sp = fr.spine
        y = sp['y'].to_numpy(float)
        wt = np.atleast_2d(fr.preds['msrp0:wt'])
        cb = np.atleast_2d(fr.preds['msrp0:comb'])
        ok = np.isfinite(y) & np.isfinite(wt).all(axis=0) & np.isfinite(cb).all(axis=0)
        yv = y[ok]
        wt_ok = wt[:, ok]
        cb_ok = cb[:, ok]
        n_ok = int(ok.sum())
        rm_wt = float(np.mean([safe_rmse(wt_ok[s], yv) for s in range(wt_ok.shape[0])]))
        rm_cb = float(np.mean([safe_rmse(cb_ok[s], yv) for s in range(cb_ok.shape[0])]))
        if ds == 'ptmul':
            p_val = np.nan
        else:
            # Paired mutation x seed bootstrap testing H1: RMSE_wt < RMSE_comb (lower is better => rep_comb - rep_wt > 0)
            rng = np.random.default_rng(6000 + idx_b)
            se_wt = (wt_ok - yv[None, :]) ** 2  # (3, n_ok)
            se_cb = (cb_ok - yv[None, :]) ** 2
            sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0
            rep_wt = np.empty(B_FAST, dtype=float)
            rep_cb = np.empty(B_FAST, dtype=float)
            chunk = 1000
            for i0 in range(0, B_FAST, chunk):
                i1 = min(B_FAST, i0 + chunk)
                m_w = rng.multinomial(n_ok, np.full(n_ok, 1.0 / n_ok), size=i1 - i0).astype(float) / n_ok
                r_w = np.sqrt(m_w @ se_wt.T)  # (chunk, 3)
                r_c = np.sqrt(m_w @ se_cb.T)
                rep_wt[i0:i1] = np.sum(r_w * sw3[i0:i1], axis=1)
                rep_cb[i0:i1] = np.sum(r_c * sw3[i0:i1], axis=1)
            p_val = one_sided_p(rep_cb - rep_wt)
        rows.append({
            'section': 'thermo_rmse_s4c_extra',
            'benchmark': ds,
            'dataset_key': ds,
            'subset': 'rmse_raw',
            'resampling_unit': 'mutation_x_seed',
            'model': 'ESM-MSR Dual-View (incl. singles)',
            'value_msr': rm_cb if ds == 'ptmul' else rm_wt,
            'pearson_r_msr': np.nan,
            'rmse_msr': rm_cb if ds == 'ptmul' else rm_wt,
            'value': rm_cb,
            'pearson_r': np.nan,
            'rmse': rm_cb,
            'p_one_sided': p_val,
            'p_one_sided_prot': np.nan,
            'n_units': n_ok,
            'n_variants': n_ok,
            'B': B_FAST,
        })
    return rows


def compute_s6_dualview_extras() -> list[dict]:
    print('[7/7] Computing Dual-View (incl. singles) for Tables S6a (NDCG@96, Property Gain) & S6b (Screening)...')
    from msr_analysis import screening as scr_core
    from pipelines import p09_screening as p09
    from pipelines._tsu_common import ndcg96_rectified, ndcg96_shifted, per_library

    rows: list[dict] = []
    DV_NAME = 'ESM-MSR Dual-View (incl. singles)'

    def _boot_dom_pair(arr_msr: np.ndarray, arr_dv: np.ndarray, higher_is_better: bool, seed: int) -> tuple[float, float, float]:
        n_d = arr_msr.shape[0]
        rng = np.random.default_rng(seed)
        gw = rng.multinomial(n_d, np.full(n_d, 1.0 / n_d), size=B_FAST).astype(float) / n_d
        sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_FAST).astype(float) / 3.0
        rm = np.einsum('bd,ds,bs->b', gw, arr_msr, sw3)
        rd = np.einsum('bd,ds,bs->b', gw, arr_dv, sw3)
        diff = (rm - rd) if higher_is_better else (rd - rm)
        return float(np.mean(arr_msr)), float(np.mean(arr_dv)), one_sided_p(diff)

    # --- 1. Table S6a top: MegaScale NDCG@96 (Shifted & Rectified; Overall / Singles / Doubles) ---
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    use = sp['use'].to_numpy(bool)
    n1 = (sp['n_mut'].to_numpy(int) == 1)
    n2 = (sp['n_mut'].to_numpy(int) == 2)
    fr.preds['msrp0:dep'] = np.where(n1[None, :], fr.preds['msrp0:wt'], fr.preds['msrp0:comb'])

    for mode_tag, fn in (('shifted', ndcg96_shifted), ('rectified', ndcg96_rectified)):
        for idx_s, (sub_tag, mask_s, msr_k) in enumerate([
            ('overall', use, 'msrp0:dep'),
            ('singles', use & n1, 'msrp0:wt'),
            ('doubles', use & n2, 'msrp0:comb'),
        ]):
            df_m = per_library(fr, msr_k, mask_s, fn, MIN_N)
            df_d = per_library(fr, 'msrp0:comb', mask_s, fn, MIN_N)
            common = sorted(set(df_m.index) & set(df_d.index))
            am = df_m.reindex(common).to_numpy(float)
            ad = df_d.reindex(common).to_numpy(float)
            pt_m, pt_d, p_v = _boot_dom_pair(am, ad, True, seed=7100 + idx_s + (10 if mode_tag == 'rectified' else 0))
            if sub_tag == 'doubles':
                p_v = np.nan
            rows.append({
                'section': 'ndcg96_s6a',
                'benchmark': 'MegaScale_capped_test',
                'dataset_key': 'tsu_capped_test',
                'subset': f'ndcg96_{mode_tag}_{sub_tag}',
                'resampling_unit': 'domain_cluster_x_seed',
                'model': DV_NAME,
                'value_msr': pt_m,
                'pearson_r_msr': np.nan,
                'rmse_msr': np.nan,
                'value': pt_d,
                'pearson_r': np.nan,
                'rmse': np.nan,
                'p_one_sided': p_v,
                'p_one_sided_prot': np.nan,
                'n_units': len(common),
                'n_variants': int((mask_s & np.isfinite(sp['y'].to_numpy(float))).sum()),
                'B': B_FAST,
            })

    # --- 2. Table S6a bottom: Property Gain @ k=96 (Total + 4 ASA x hydrophobicity quadrants) ---
    feat = p09._load_tsu_features(sp['id'].tolist())
    asa = feat['rel_ASA'].to_numpy()
    dh = feat['d_hydrophobicity'].to_numpy()
    single = use & n1
    pool = single & np.isfinite(asa) & np.isfinite(dh)
    cat = np.nan_to_num(np.asarray(2 * (dh > 0) + (asa > 0.5), float)).astype(int)
    group_arr = sp['group'].to_numpy()
    libs = sorted(np.unique(group_arr[single]))
    dom_idx = [d for d in (np.flatnonzero(pool & (group_arr == lib)) for lib in libs) if len(d)]
    y_arr = sp['y'].to_numpy(float)
    st_pg = p09._model_cat_stats(
        fr,
        [{'name': 'ESM-MSR', 'key': 'msrp0:wt'}, {'name': DV_NAME, 'key': 'msrp0:comb'}],
        y_arr, cat, len(p09.ASA_HYDRO_LABELS), dom_idx, 96, positive_only=False,
    )
    G_wt = st_pg['ESM-MSR'][0]  # (3, n_dom, 4)
    G_cb = st_pg[DV_NAME][0]
    for c_i, c_lbl in list(enumerate(p09.ASA_HYDRO_LABELS)) + [(-1, 'Total rectified gain @ k=96 (sum over 4 quadrants)')]:
        am = (G_wt.sum(axis=2) if c_i == -1 else G_wt[:, :, c_i]).T
        ad = (G_cb.sum(axis=2) if c_i == -1 else G_cb[:, :, c_i]).T
        pt_m, pt_d, p_v = _boot_dom_pair(am, ad, True, seed=7200 + c_i + 2)
        rows.append({
            'section': 'property_gain_s6a',
            'benchmark': 'MegaScale_capped_test_singles_k96',
            'dataset_key': 'tsu_capped_test',
            'subset': c_lbl,
            'resampling_unit': 'domain_cluster_x_seed',
            'model': DV_NAME,
            'value_msr': pt_m,
            'pearson_r_msr': np.nan,
            'rmse_msr': np.nan,
            'value': pt_d,
            'pearson_r': np.nan,
            'rmse': np.nan,
            'p_one_sided': p_v,
            'p_one_sided_prot': np.nan,
            'n_units': len(dom_idx),
            'n_variants': int(pool.sum()),
            'B': B_FAST,
        })

    # --- 3. Table S6b: All 4 Practical Library Screening Benchmarks ---
    def _screen_pair_rows(
        bm_key: str, ds_key: str, fr_s: Frame, keep_mask: np.ndarray, tau: float,
        ks_list: list[int], metrics_spec: list[tuple[str, str, bool]], unit_name: str,
        has_rmse: bool, seed_base: int,
    ) -> list[dict]:
        sp_s = fr_s.spine
        wt_p = np.atleast_2d(fr_s.preds['msrp0:wt'])
        cb_p = np.atleast_2d(fr_s.preds['msrp0:comb'])
        ok = sp_s['use'].to_numpy(bool) & keep_mask & np.isfinite(wt_p).all(axis=0) & np.isfinite(cb_p).all(axis=0)
        y_ok = sp_s['y'].to_numpy(float)[ok]
        g_ok = sp_s['group'].to_numpy()[ok]
        # Match enrich_tables_r_rmse_n.py: per-domain / per-assay mean Pearson r & RMSE,
        # using +z_sd (un-negated y) for PGym Activity where _label_frame used sign=-1 for harm.
        y_corr = (-y_ok) if bm_key.startswith('Figure4d_PGym') else y_ok
        grp_rs, grp_rms = [], []
        for grp in sorted(np.unique(g_ok)):
            mg = (g_ok == grp)
            if mg.sum() >= MIN_N:
                grp_rs.append(float(np.mean([safe_pearson(cb_p[s, ok][mg], y_corr[mg]) for s in range(cb_p.shape[0])])))
                if has_rmse:
                    grp_rms.append(float(np.mean([safe_rmse(cb_p[s, ok][mg], y_corr[mg]) for s in range(cb_p.shape[0])])))
        r_cb_pool = float(np.nanmean(grp_rs)) if grp_rs else np.nan
        rm_cb_pool = float(np.nanmean(grp_rms)) if (has_rmse and grp_rms) else np.nan
        out_r = []
        for K in ks_list:
            tabs_wt = [scr_core.per_protein_table(g_ok, wt_p[s, ok], y_ok, K, tau).set_index('group') for s in range(wt_p.shape[0])]
            tabs_cb = [scr_core.per_protein_table(g_ok, cb_p[s, ok], y_ok, K, tau).set_index('group') for s in range(cb_p.shape[0])]
            g_comm = sorted(set.intersection(*(set(t.index) for t in tabs_wt + tabs_cb)))
            for col_k, met_lbl, hib in metrics_spec:
                am = np.column_stack([t.reindex(g_comm)[col_k].to_numpy(float) for t in tabs_wt])
                ad = np.column_stack([t.reindex(g_comm)[col_k].to_numpy(float) for t in tabs_cb])
                fin = np.isfinite(am).all(axis=1) & np.isfinite(ad).all(axis=1)
                am, ad = am[fin], ad[fin]
                pt_m, pt_d, p_v = _boot_dom_pair(am, ad, hib, seed=seed_base + K)
                out_r.append({
                    'section': 'screening_s6b_extra',
                    'benchmark': bm_key,
                    'dataset_key': ds_key,
                    'subset': f'{met_lbl}@K={K}',
                    'resampling_unit': unit_name,
                    'model': DV_NAME,
                    'value_msr': pt_m,
                    'pearson_r_msr': np.nan,
                    'rmse_msr': np.nan,
                    'value': pt_d,
                    'pearson_r': r_cb_pool,
                    'rmse': rm_cb_pool,
                    'p_one_sided': p_v,
                    'p_one_sided_prot': np.nan,
                    'n_units': int(fin.sum()),
                    'n_variants': int(ok.sum()),
                    'B': B_FAST,
                })
        return out_r

    # 3a. MegaScale Singles (no-Cys)
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
    rows.extend(_screen_pair_rows(
        'Figure4b_MegaScale_singles_nocys', 'tsu_capped_test', fr, n1 & (~to_cys), 0.5,
        [1, 10, 50], [('hit', 'hit_rate_gt_0.5', True), ('rec', 'P_at_least_one_hit', True)],
        'domain_cluster_x_seed', True, 7300,
    ))

    # 3b. Domainome Held-Out (436 domains)
    fr_dom = load_frame('domainome')
    hom_dom = heldout_excluded('domainome')
    keep_d = (~fr_dom.spine['group'].astype(str).isin(hom_dom).to_numpy()) & (fr_dom.spine['n_mut'].to_numpy(int) == 1)
    rows.extend(_screen_pair_rows(
        'Figure4c_Domainome_heldout_singles', 'domainome_heldout', fr_dom, keep_d, 0.0,
        [1, 10, 50], [('hit', 'hit_rate_gt_WT', True), ('rec', 'P_at_least_one_hit', True)],
        'domain_cluster_x_seed', True, 7400,
    ))

    # 3c/d. ProteinGym Activity (38 assays & 16 enzymes) — attach 3-seed combined_pred from final_pgym_comb/FINAL_p0
    pg_fr, pg_info = p09.load_pg_activity_assays()
    pg_sp = pg_fr.spine
    cb_pg = np.full((3, len(pg_sp)), np.nan, dtype=float)
    grp_series = pg_sp['group'].astype(str).to_numpy()
    mut_series = pg_sp['mut'].astype(str).to_numpy()
    for assay in sorted(pg_info.index):
        idx_a = np.flatnonzero(grp_series == assay)
        if len(idx_a) == 0:
            continue
        m_a = mut_series[idx_a]
        for s_i, s in enumerate((1, 2, 3)):
            pg_dir = ROOT / 'final_pgym_comb/FINAL_p0' / f's{s}'
            f_a = pg_dir / f'{assay}.csv'
            if not f_a.exists():
                ms = list(pg_dir.glob(f'{assay}*.csv'))
                f_a = ms[0] if ms else None
            if f_a is not None and f_a.exists():
                da = pd.read_csv(f_a, usecols=['mutant', 'combined_pred'])
                lut = dict(zip(da['mutant'].astype(str), da['combined_pred'].to_numpy(float)))
                cb_pg[s_i, idx_a] = np.array([lut.get(m, np.nan) for m in m_a], dtype=float)
    pg_fr.preds['msrp0:comb'] = cb_pg
    for set_tag, gs, sb in (
        ('Figure4d_PGym_Activity_all_38', set(pg_info.index), 7500),
        ('Figure4d_PGym_Activity_enzymes_16', p09.PG_ENZYME, 7600),
    ):
        g_list = [d for d in pg_info.index if d in gs]
        lf = p09._label_frame(pg_fr, g_list, 'z_sd', -1, 'use_wt')
        rows.extend(_screen_pair_rows(
            set_tag, 'proteingym_heldout', lf, np.ones(len(lf.spine), dtype=bool), 0.5,
            [1, 10, 20, 50, 100], [('hit', 'harm_rate_lt_WT_minus_0.5SD', False)],
            'assay_x_seed', False, sb,
        ))
    return rows


def main() -> None:
    all_rows: list[dict] = []
    all_rows.extend(compute_thermostability_extras())
    all_rows.extend(compute_megascale_extras())
    all_rows.extend(compute_epistasis_extras())
    all_rows.extend(compute_domainome_and_pgym_extras())
    all_rows.extend(compute_mgnify_rocklin_extras())
    all_rows.extend(compute_thermo_rmse_dualview_extras())
    all_rows.extend(compute_s6_dualview_extras())

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(df)} extra model rows to {OUT_CSV}')
    print(df.groupby(['section', 'benchmark'])['model'].nunique().to_string())


if __name__ == '__main__':
    main()
