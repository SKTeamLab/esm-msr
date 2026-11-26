#!/usr/bin/env python3
"""Compute Tobit-linked ESM-MSR (and ESM-MSR sigma=0.5) metrics, paired bootstrap p-values,
and unlinked latent RMSEs across ALL cDNA-proteolysis benchmarks where dG_wt is available:
  - MegaScale capped test: S2a (Overall/Stab/Destab), S2b (Doubles Epi/Add),
    S2d (Backbone/Origin), S3a (5-level Epistasis Hierarchy), S3b (Epistasis RMSE & MCC),
    S4c (MegaScale raw pooled RMSE), S6b (MegaScale Singles Pool r & RMSE).
  - MegaScale full split (mega_splits): S2c (Overall/Stab/Destab & Doubles Epi/Add).
  - Held-out MGnify & Rocklin (S5a): unlinked latent RMSEs (primary ESM-MSR is already Tobit-linked).

Saves to analysis/results/si_latex_tables/_tobit_linked_si_overrides.csv.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/sareeves/playground/esm-msr-devel')
WT_DDDG = ROOT / 'wt-dddg-scatters'
WT_SI = Path(__file__).resolve().parents[2]
OUT_CSV = WT_SI / 'analysis/results/si_latex_tables/_tobit_linked_si_overrides.csv'

for _p in (str(WT_DDDG / 'src'), str(WT_DDDG / 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm_msr import epi_metrics as E  # noqa: E402
from esm_msr.val_stats import tobit_link  # noqa: E402
from msr_analysis import epistasis as EPI_LIB  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.heldout import excluded as heldout_excluded  # noqa: E402
from msr_analysis.tobit import TOBIT  # noqa: E402

H_LINK = tobit_link(**TOBIT)
B_BOOT = 10000
B_EPI_HIER = 2000


def apply_link(P: np.ndarray, dg_wt: np.ndarray) -> np.ndarray:
    P2 = np.atleast_2d(np.asarray(P, dtype=float))
    return H_LINK(dg_wt[None, :] + P2) - dg_wt[None, :]


def deployed_arr(fr, single_key: str, multi_key: str) -> np.ndarray:
    n1 = (fr.spine['n_mut'].to_numpy(int) == 1)
    ps = np.atleast_2d(np.asarray(fr.preds[single_key], dtype=float))
    pm = np.atleast_2d(np.asarray(fr.preds[multi_key], dtype=float))
    n_s = min(ps.shape[0], pm.shape[0])
    return np.where(n1[None, :], ps[:n_s], pm[:n_s])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(pearsonr(x[ok], y[ok]).statistic)


def safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))


def safe_mcc(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    tp = float(np.sum(pred_pos & true_pos))
    tn = float(np.sum((~pred_pos) & (~true_pos)))
    fp = float(np.sum(pred_pos & (~true_pos)))
    fn = float(np.sum((~pred_pos) & true_pos))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def one_sided_p(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan
    return float((np.sum(diffs <= 0.0) + 1.0) / (len(diffs) + 1.0))


def _row(section, benchmark, subset, model, val=np.nan, r=np.nan, rmse=np.nan, p=np.nan, B=B_BOOT, is_unlinked=False):
    return {
        'section': section,
        'benchmark': benchmark,
        'subset': subset,
        'model': model,
        'value': val,
        'pearson_r': r,
        'rmse': rmse,
        'p_one_sided': p,
        'B': int(B),
        'is_unlinked': bool(is_unlinked),
    }


def eval_per_group_panel(
    fr,
    msr_linked: np.ndarray,
    msr_unlinked: np.ndarray,
    comparators: list[tuple[str, np.ndarray]],
    mask: np.ndarray,
    group_col: str,
    section: str,
    benchmark: str,
    subset: str,
    seed: int = 42,
) -> list[dict]:
    sp = fr.spine
    y = sp['y'].to_numpy(float)
    groups = sp[group_col].astype(str).to_numpy()
    uniq_g = sorted(np.unique(groups[mask]))

    def per_grp_stats(P: np.ndarray):
        P2 = np.atleast_2d(np.asarray(P, dtype=float))
        n_s = P2.shape[0]
        rhos, rs, rms = {}, {}, {}
        for g in uniq_g:
            m = mask & (groups == g)
            if m.sum() < 3:
                continue
            r_s = [safe_spearman(P2[s, m], y[m]) for s in range(n_s)]
            p_s = [safe_pearson(P2[s, m], y[m]) for s in range(n_s)]
            e_s = [safe_rmse(P2[s, m], y[m]) for s in range(n_s)]
            if all(np.isfinite(r_s)):
                rhos[g] = np.array(r_s, dtype=float)
                rs[g] = float(np.nanmean(p_s))
                rms[g] = float(np.nanmean(e_s))
        return rhos, rs, rms

    m_rho, m_r, m_rm = per_grp_stats(msr_linked)
    _, _, u_rm = per_grp_stats(msr_unlinked)
    valid_g = sorted(m_rho.keys())
    n_g = len(valid_g)

    rng = np.random.default_rng(seed)
    g_w = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=B_BOOT).astype(float) / n_g
    s_w3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_BOOT).astype(float) / 3.0
    g_idx = {g: i for i, g in enumerate(valid_g)}

    pt_rho_m = float(np.mean([m_rho[g].mean() for g in valid_g]))
    pt_r_m = float(np.nanmean([m_r[g] for g in valid_g]))
    pt_rm_m = float(np.nanmean([m_rm[g] for g in valid_g]))
    pt_rm_u = float(np.nanmean([u_rm[g] for g in valid_g]))

    out = [
        _row(section, benchmark, subset, 'ESM-MSR', val=pt_rho_m, r=pt_r_m, rmse=pt_rm_m, p=np.nan),
        _row(section, benchmark, subset, 'ESM-MSR (unlinked)', rmse=pt_rm_u, is_unlinked=True),
    ]

    for cname, carr in comparators:
        c_rho, c_r, c_rm = per_grp_stats(carr)
        common = [g for g in valid_g if g in c_rho]
        if not common:
            continue
        idx_c = [g_idx[g] for g in common]
        gw_c = g_w[:, idx_c]
        gw_c = gw_c / np.maximum(gw_c.sum(axis=1, keepdims=True), 1e-12)
        Mm = np.vstack([m_rho[g] for g in common])
        Mc = np.vstack([c_rho[g] for g in common])
        reps_m = np.einsum('bg,gs,bs->b', gw_c, Mm, s_w3) if Mm.shape[1] == 3 else (gw_c @ Mm.mean(axis=1))
        reps_c = np.einsum('bg,gs,bs->b', gw_c, Mc, s_w3) if Mc.shape[1] == 3 else (gw_c @ Mc.mean(axis=1))
        out.append(_row(
            section, benchmark, subset, cname,
            val=float(np.mean(Mc)),
            r=float(np.nanmean([c_r[g] for g in common])),
            rmse=float(np.nanmean([c_rm[g] for g in common])),
            p=one_sided_p(reps_m - reps_c),
        ))
    return out


def compute_s2_megascale() -> list[dict]:
    print('[1/5] Computing Tobit-linked S2a, S2b, S2c, S2d...')
    rows: list[dict] = []

    # ---- Homology-Restricted MegaScale capped test (S2a, S2b, S2d) ----
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    use = sp['use'].to_numpy(bool)
    y = sp['y'].to_numpy(float)
    dg = sp['dG_wt'].to_numpy(float)
    n_mut = sp['n_mut'].to_numpy(int)

    msr_dep_un = deployed_arr(fr, 'msrp0:wt', 'msrp0:comb')
    msr_dep_lk = apply_link(msr_dep_un, dg)
    msr_cb_all_lk = apply_link(fr.preds['msrp0:comb'], dg)
    msr05_dep_lk = apply_link(deployed_arr(fr, 'msrp0@0.5:wt', 'msrp0@0.5:comb'), dg)

    comps_dep = [
        ('ESM-MSR Dual-View (incl. singles)', msr_cb_all_lk),
        ('ESM-MSR σ=0.5', msr05_dep_lk),
        ('SPURS', deployed_arr(fr, 'spurs', 'spursm')),
        ('MutateEverything', deployed_arr(fr, 'me1', 'me2')),
        ('ThermoMPNN', deployed_arr(fr, 'tmpnn', 'tmpnnd')),
        ('GeoDDG-3D', fr.preds['geoddg']),
        ('Rosetta', fr.preds['rosetta']),
        ('DDGemb', fr.preds['ddgemb']),
        ('ESMtherm', fr.preds['esmtherm']),
        ('IFUM (WT backbone)', fr.preds['ifum_wt']),
        ('IFUM (ESMFold)', fr.preds['ifum_esmfold']),
        ('ESM3-small-open', deployed_arr(fr, 'msrrel@0:wt', 'msrrel@0:comb')),
        ('ProteinMPNN', fr.preds['pmpnn']),
        ('ProSST', fr.preds['prosst']),
    ]

    for sub, m_sub in [
        ('per_domain_mean_spearman_overall', use),
        ('per_domain_mean_spearman_stabilizing', use & (y > 0)),
        ('per_domain_mean_spearman_destabilizing', use & (y <= 0)),
    ]:
        rows.extend(eval_per_group_panel(
            fr, msr_dep_lk, msr_dep_un, comps_dep, m_sub, 'group',
            section='megascale_s2a', benchmark='MegaScale_capped_test', subset=sub,
        ))

    # S2b: Doubles Epistatic & Additive
    dbl = use & (n_mut == 2)
    msr_cb_un = np.atleast_2d(fr.preds['msrp0:comb'])
    msr_cb_lk = apply_link(msr_cb_un, dg)
    msr_wt_un = np.atleast_2d(fr.preds['msrp0:wt+'])
    msr_wt_lk = apply_link(msr_wt_un, dg)

    comps_epi = [
        ('ESM-MSR σ=0.5', apply_link(fr.preds['msrp0@0.5:comb'], dg)),
        ('SPURS', fr.preds['spursm']),
        ('MutateEverything', fr.preds['me2']),
        ('ThermoMPNN', fr.preds['tmpnnd']),
        ('GeoDDG-3D', fr.preds['geoddg']),
        ('Rosetta', fr.preds['rosetta']),
        ('DDGemb', fr.preds['ddgemb']),
        ('ESMtherm', fr.preds['esmtherm']),
        ('IFUM (WT backbone)', fr.preds['ifum_wt']),
        ('IFUM (ESMFold)', fr.preds['ifum_esmfold']),
        ('ESM3-small-open', fr.preds['msrrel@0:comb']),
        ('ProteinMPNN', fr.preds['pmpnn']),
    ]
    comps_add = [
        ('ESM-MSR σ=0.5', apply_link(fr.preds['msrp0@0.5:wt+'], dg)),
        ('SPURS', fr.preds['spurs']),
        ('MutateEverything', fr.preds['me1']),
        ('ThermoMPNN', fr.preds['tmpnn']),
        ('GeoDDG-3D', fr.preds['geoddg+']),
        ('Rosetta', fr.preds['rosetta+']),
        ('DDGemb', fr.preds['ddgemb+']),
        ('ESMtherm', fr.preds['esmtherm+']),
        ('IFUM (WT backbone)', fr.preds['ifum_wt+']),
        ('IFUM (ESMFold)', fr.preds['ifum_esmfold+']),
        ('ESM3-small-open', fr.preds['msrrel@0:wt+']),
        ('ProteinMPNN', fr.preds['pmpnn+']),
    ]
    rows.extend(eval_per_group_panel(
        fr, msr_cb_lk, msr_cb_un, comps_epi, dbl, 'group',
        section='megascale_s2b', benchmark='MegaScale_capped_test_doubles', subset='per_domain_mean_spearman_epistatic',
    ))
    rows.extend(eval_per_group_panel(
        fr, msr_wt_lk, msr_wt_un, comps_add, dbl, 'group',
        section='megascale_s2b', benchmark='MegaScale_capped_test_doubles', subset='per_domain_mean_spearman_additive',
    ))

    # S2d: Backbone (code != group => 7 destabilized, 30 native) & Origin (20 natural, 13 synthetic)
    is_destab = (sp['code'].astype(str) != sp['group'].astype(str)).to_numpy()
    rows.extend(eval_per_group_panel(
        fr, msr_dep_lk, msr_dep_un, comps_dep, use & (~is_destab), 'code',
        section='megascale_s2d', benchmark='MegaScale_capped_test_backbone', subset='per_code_mean_spearman_native',
    ))
    rows.extend(eval_per_group_panel(
        fr, msr_dep_lk, msr_dep_un, comps_dep, use & is_destab, 'code',
        section='megascale_s2d', benchmark='MegaScale_capped_test_backbone', subset='per_code_mean_spearman_destabilized',
    ))
    is_synth = sp['group'].astype(str).str.contains(r'rd\d|TrROS|EEHEE|HEEH|HHH|EHEE|r\d+_', regex=True).to_numpy()
    rows.extend(eval_per_group_panel(
        fr, msr_dep_lk, msr_dep_un, comps_dep, use & (~is_synth), 'group',
        section='megascale_s2d', benchmark='MegaScale_capped_test_origin', subset='per_group_mean_spearman_natural',
    ))
    rows.extend(eval_per_group_panel(
        fr, msr_dep_lk, msr_dep_un, comps_dep, use & is_synth, 'group',
        section='megascale_s2d', benchmark='MegaScale_capped_test_origin', subset='per_group_mean_spearman_synthetic',
    ))

    # ---- Full MegaScale split (mega_splits) S2c ----
    fr_m = load_frame('tsuboyama_mega_test')
    sp_m = fr_m.spine
    use_m = sp_m['use'].to_numpy(bool)
    y_m = sp_m['y'].to_numpy(float)
    dg_m = sp_m['dG_wt'].to_numpy(float)
    n_mut_m = sp_m['n_mut'].to_numpy(int)

    m_dep_un = deployed_arr(fr_m, 'msrp0:wt', 'msrp0:comb')
    m_dep_lk = apply_link(m_dep_un, dg_m)
    m_cb_all_lk = apply_link(fr_m.preds['msrp0:comb'], dg_m)
    comps_m_dep = [
        ('ESM-MSR Dual-View (incl. singles)', m_cb_all_lk),
        ('SPURS', deployed_arr(fr_m, 'spurs', 'spursm')),
        ('SPURS published', deployed_arr(fr_m, 'spurspub', 'spurspubm')),
        ('MutateEverything', deployed_arr(fr_m, 'me1', 'me2')),
        ('ThermoMPNN', deployed_arr(fr_m, 'tmpnn', 'tmpnnd')),
        ('ThermoMPNN published*', deployed_arr(fr_m, 'tmpnnpub', 'tmpnndpub')),
        ('Rosetta', fr_m.preds['rosetta']),
        ('ESM3-small-open', deployed_arr(fr_m, 'msrrel@0:wt', 'msrrel@0:comb')),
        ('ProteinMPNN', fr_m.preds['pmpnn']),
        ('ProSST', fr_m.preds['prosst']),
    ]
    for sub, m_sub in [
        ('per_domain_mean_spearman_overall', use_m),
        ('per_domain_mean_spearman_stabilizing', use_m & (y_m > 0)),
        ('per_domain_mean_spearman_destabilizing', use_m & (y_m <= 0)),
    ]:
        rows.extend(eval_per_group_panel(
            fr_m, m_dep_lk, m_dep_un, comps_m_dep, m_sub, 'group',
            section='megascale_s2c', benchmark='MegaScale_mega_splits_test', subset=sub,
        ))

    dbl_m = use_m & (n_mut_m == 2)
    m_cb_un = np.atleast_2d(fr_m.preds['msrp0:comb'])
    m_cb_lk = apply_link(m_cb_un, dg_m)
    m_wt_un = np.atleast_2d(fr_m.preds['msrp0:wt'])
    m_wt_lk = apply_link(m_wt_un, dg_m)
    comps_m_epi = [
        ('SPURS', fr_m.preds['spursm']),
        ('SPURS published', fr_m.preds['spurspubm']),
        ('MutateEverything', fr_m.preds['me2']),
        ('ThermoMPNN', fr_m.preds['tmpnnd']),
        ('ThermoMPNN published*', fr_m.preds['tmpnndpub']),
        ('Rosetta', fr_m.preds['rosetta']),
        ('ESM3-small-open', fr_m.preds['msrrel@0:comb']),
        ('ProteinMPNN', fr_m.preds['pmpnn']),
    ]
    comps_m_add = [
        ('SPURS', fr_m.preds['spurs']),
        ('SPURS published', fr_m.preds['spurspub']),
        ('MutateEverything', fr_m.preds['me1']),
        ('ThermoMPNN', fr_m.preds['tmpnn']),
        ('ThermoMPNN published*', fr_m.preds['tmpnnpub']),
        ('Rosetta', fr_m.preds['rosetta+']),
        ('ESM3-small-open', fr_m.preds['msrrel@0:wt']),
        ('ProteinMPNN', fr_m.preds['pmpnn+']),
        ('ProSST', fr_m.preds['prosst']),
    ]
    rows.extend(eval_per_group_panel(
        fr_m, m_cb_lk, m_cb_un, comps_m_epi, dbl_m, 'group',
        section='megascale_s2c', benchmark='MegaScale_mega_splits_test_doubles', subset='per_domain_mean_spearman_epistatic',
    ))
    rows.extend(eval_per_group_panel(
        fr_m, m_wt_lk, m_wt_un, comps_m_add, dbl_m, 'group',
        section='megascale_s2c', benchmark='MegaScale_mega_splits_test_doubles', subset='per_domain_mean_spearman_additive',
    ))
    return rows


# ---- Epistasis S3a & S3b ----
_EPI_CTX = {}


def _epi_boot_chunk(args):
    seed_rng, n_rep = args
    rng = np.random.default_rng(seed_rng)
    doms = _EPI_CTX['doms']
    n_d = len(doms)
    tables = _EPI_CTX['tables']
    models = _EPI_CTX['models']
    keep_gt0_by_dom = _EPI_CTX['keep_gt0_by_dom']

    out = {m: {k: np.empty(n_rep, dtype=float) for k in ('naive', 'beyond', 'pair', 'line', 'cmag_all', 'cmag_gt0')} for m in models}
    for b in range(n_rep):
        draw_d = rng.integers(0, n_d, size=n_d)
        draw_s = rng.integers(0, 3, size=3)
        # Re-index domain names to keep clones distinct
        for m in models:
            n_s_m = len(tables[m])
            s_list = draw_s if n_s_m == 3 else [0]
            acc = {k: [] for k in out[m]}
            for s_idx in s_list:
                t_s = tables[m][s_idx]
                sub_t = t_s.resample_domains([doms[i] for i in draw_d]) if hasattr(t_s, 'resample_domains') else None
                if sub_t is None:
                    raise RuntimeError('Table has no resample_domains')
                res = E.compute(sub_t)
                acc['naive'].append(res['epi_naive_rho_comb'])
                acc['beyond'].append(res['epi_beyond_rho_comb'])
                acc['pair'].append(res['epi_pair_rho_comb'])
                acc['line'].append(res['epi_line_rho_comb'])
                acc['cmag_all'].append(res['epi_cell_mag_comb'])
                sub_g = tables[m + '__gt0'][s_idx].resample_domains([doms[i] for i in draw_d])
                acc['cmag_gt0'].append(E.compute(sub_g)['epi_cell_mag_comb'])
            for k in out[m]:
                out[m][k][b] = float(np.nanmean(acc[k]))
    return out


def compute_s3_epistasis() -> list[dict]:
    print('[2/5] Computing Tobit-linked S3a (Epistasis Hierarchy) & S3b (Epistasis RMSE & Sign MCC)...')
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    idx = np.flatnonzero(sp['use'].to_numpy(bool))
    y = sp['y'].to_numpy(float)[idx]
    dg = sp['dG_wt'].to_numpy(float)[idx]
    dddg = sp['dddG_ML'].to_numpy(float)[idx]
    ddg_add = sp['ddG_additive_ML'].to_numpy(float)[idx]
    code = sp['code'].astype(str).to_numpy()[idx]
    mut = sp['mut'].astype(str).to_numpy()[idx]
    keys = [EPI_LIB.parse_mut_key(m, c) for m, c in zip(mut, code)]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    keep_gt0 = (st == 'single') | ((st == 'double') & ((dg + ddg_add) > 0))

    base_items = dict(code=code, mut_key=keys, subset_type=st, ddG=y, dddG=dddg, dG_wt=dg)
    base_items_gt0 = {k: (v[keep_gt0] if isinstance(v, np.ndarray) else [x for x, b in zip(v, keep_gt0) if b]) for k, v in base_items.items()}

    wt_un = np.atleast_2d(fr.preds['msrp0:wt'])[:, idx]
    cb_un = np.atleast_2d(fr.preds['msrp0:comb'])[:, idx]
    wt_lk = apply_link(wt_un, dg)
    cb_lk = apply_link(cb_un, dg)

    wt05_lk = apply_link(np.atleast_2d(fr.preds['msrp0@0.5:wt'])[:, idx], dg)
    cb05_lk = apply_link(np.atleast_2d(fr.preds['msrp0@0.5:comb'])[:, idx], dg)

    model_defs = [
        ('ESM-MSR', wt_lk, cb_lk),
        ('ESM-MSR (unlinked)', wt_un, cb_un),
        ('ESM-MSR σ=0.5', wt05_lk, cb05_lk),
        ('SPURS-multi', np.atleast_2d(fr.preds['spurs'])[:, idx], np.atleast_2d(fr.preds['spursm'])[:, idx]),
        ('MutateEverything Stage 2', np.atleast_2d(fr.preds['me1'])[:, idx], np.atleast_2d(fr.preds['me2'])[:, idx]),
        ('ThermoMPNN-D', np.atleast_2d(fr.preds['tmpnn'])[:, idx], np.atleast_2d(fr.preds['tmpnnd'])[:, idx]),
        ('GeoDDG-3D', np.atleast_2d(fr.preds['geoddg+'])[:, idx], np.atleast_2d(fr.preds['geoddg'])[:, idx]),
        ('Rosetta', np.atleast_2d(fr.preds['rosetta+'])[:, idx], np.atleast_2d(fr.preds['rosetta'])[:, idx]),
        ('ESM3-small-open', np.atleast_2d(fr.preds['msrrel@0:wt'])[:, idx], np.atleast_2d(fr.preds['msrrel@0:comb'])[:, idx]),
        ('ProteinMPNN', np.atleast_2d(fr.preds['pmpnn+'])[:, idx], np.atleast_2d(fr.preds['pmpnn'])[:, idx]),
    ]

    # Point metrics for linked ESM-MSR, unlinked ESM-MSR, and linked ESM-MSR σ=0.5
    rows: list[dict] = []
    dpred_by_model: dict[str, np.ndarray] = {}
    tbl_ref = None

    for mname, W, C in model_defs:
        n_s = min(W.shape[0], C.shape[0])
        dps = []
        acc = {k: [] for k in ('naive', 'beyond', 'pair', 'line', 'cmag_all', 'cmag_gt0', 'r_all', 'rm_all', 'r_gt0', 'rm_gt0', 'mcc_raw', 'mcc_sat')}
        for s in range(n_s):
            items = dict(base_items, wt=W[s], mt=C[s], comb=C[s])
            tbl = E.Table(items, None, heads=('comb', 'add'))
            if tbl_ref is None:
                tbl_ref = tbl
            res = E.compute(tbl)
            dp = tbl.P['comb'] - (tbl.PA['add'] + tbl.PB['add'])
            dps.append(dp)
            dt = tbl.dddG
            gt0 = tbl.x > 0
            acc['naive'].append(res['epi_naive_rho_comb'])
            acc['beyond'].append(res['epi_beyond_rho_comb'])
            acc['pair'].append(res['epi_pair_rho_comb'])
            acc['line'].append(res['epi_line_rho_comb'])
            acc['cmag_all'].append(res['epi_cell_mag_comb'])
            acc['r_all'].append(safe_pearson(dp, dt))
            acc['rm_all'].append(safe_rmse(dp, dt))
            acc['r_gt0'].append(safe_pearson(dp[gt0], dt[gt0]))
            acc['rm_gt0'].append(safe_rmse(dp[gt0], dt[gt0]))
            acc['mcc_raw'].append(safe_mcc(dp > 0, dt > 0))
            items_g = dict(base_items_gt0, wt=W[s][keep_gt0], mt=C[s][keep_gt0], comb=C[s][keep_gt0])
            tbl_g = E.Table(items_g, None, heads=('comb', 'add'))
            acc['cmag_gt0'].append(E.compute(tbl_g)['epi_cell_mag_comb'])
        dpred_by_model[mname] = np.vstack(dps)

        if mname == 'ESM-MSR':
            for mk, sub_lbl in [
                ('naive', 'naive_rho (all_doubles)'),
                ('beyond', 'beyond_rho (all_doubles)'),
                ('pair', 'pair_rho (all_doubles)'),
                ('line', 'line_rho (all_doubles)'),
                ('cmag_all', 'cell_mag_pooled (all_doubles)'),
                ('cmag_gt0', 'cell_mag_gt0 (dG_add_gt0)'),
            ]:
                rows.append(_row(
                    'epistasis_s3a', 'MegaScale_epistasis_5level_hierarchy', sub_lbl, 'ESM-MSR',
                    val=float(np.mean(acc[mk])), r=float(np.mean(acc['r_all'])), rmse=float(np.mean(acc['rm_all'])),
                ))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles', 'ESM-MSR',
                             val=float(np.mean(acc['rm_all'])), r=float(np.mean(acc['r_all'])), rmse=float(np.mean(acc['rm_all']))))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0', 'ESM-MSR',
                             val=float(np.mean(acc['rm_gt0'])), r=float(np.mean(acc['r_gt0'])), rmse=float(np.mean(acc['rm_gt0']))))
            rows.append(_row('epistasis_s3b_mcc', 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_raw_dddG_gt0', 'ESM-MSR',
                             val=float(np.mean(acc['mcc_raw']))))
        elif mname == 'ESM-MSR (unlinked)':
            rows.append(_row('epistasis_s3a', 'MegaScale_epistasis_5level_hierarchy', 'naive_rho (all_doubles)',
                             'ESM-MSR (unlinked)', rmse=float(np.mean(acc['rm_all'])), is_unlinked=True))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles',
                             'ESM-MSR (unlinked)', rmse=float(np.mean(acc['rm_all'])), is_unlinked=True))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0',
                             'ESM-MSR (unlinked)', rmse=float(np.mean(acc['rm_gt0'])), is_unlinked=True))
        elif mname == 'ESM-MSR σ=0.5':
            for mk, sub_lbl in [
                ('naive', 'naive_rho (all_doubles)'),
                ('beyond', 'beyond_rho (all_doubles)'),
                ('pair', 'pair_rho (all_doubles)'),
                ('line', 'line_rho (all_doubles)'),
                ('cmag_all', 'cell_mag_pooled (all_doubles)'),
                ('cmag_gt0', 'cell_mag_gt0 (dG_add_gt0)'),
            ]:
                rows.append(_row(
                    'epistasis_s3a', 'MegaScale_epistasis_5level_hierarchy', sub_lbl, 'ESM-MSR σ=0.5',
                    val=float(np.mean(acc[mk])), r=float(np.mean(acc['r_all'])), rmse=float(np.mean(acc['rm_all'])),
                ))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles', 'ESM-MSR σ=0.5',
                             val=float(np.mean(acc['rm_all'])), r=float(np.mean(acc['r_all'])), rmse=float(np.mean(acc['rm_all']))))
            rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0', 'ESM-MSR σ=0.5',
                             val=float(np.mean(acc['rm_gt0'])), r=float(np.mean(acc['r_gt0'])), rmse=float(np.mean(acc['rm_gt0']))))

    # Paired domain-cluster x seed bootstrap for S3b RMSE (all_doubles & dG_add_gt0) & MCC (raw)
    dt_all = tbl_ref.dddG
    gt0_mask = tbl_ref.x > 0
    dg_id = np.asarray(tbl_ref.lib, dtype=int)
    n_dg = int(dg_id.max()) + 1

    rng = np.random.default_rng(77)
    gw = rng.multinomial(n_dg, np.full(n_dg, 1.0 / n_dg), size=B_BOOT).astype(float)
    sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_BOOT).astype(float) / 3.0

    for sub_k, mask_s in (('all_doubles', np.ones(len(dt_all), bool)), ('dG_add_gt0', gt0_mask)):
        dt_s = dt_all[mask_s]
        dg_s = dg_id[mask_s]
        # Per-domain count and sum of squared errors per seed
        cnt_d = np.bincount(dg_s, minlength=n_dg).astype(float)
        tot_n = gw @ cnt_d

        def boot_rmse(dp_arr: np.ndarray) -> np.ndarray:
            dp_s = dp_arr[:, mask_s]
            n_s = dp_s.shape[0]
            sse_ds = np.zeros((n_dg, n_s), dtype=float)
            for s in range(n_s):
                np.add.at(sse_ds[:, s], dg_s, (dp_s[s] - dt_s) ** 2)
            # (B, n_s)
            sse_bs = gw @ sse_ds
            rmse_bs = np.sqrt(sse_bs / np.maximum(tot_n[:, None], 1.0))
            return np.sum(rmse_bs * sw3, axis=1) if n_s == 3 else rmse_bs[:, 0]

        reps_msr = boot_rmse(dpred_by_model['ESM-MSR'])
        for mname in ('ESM-MSR σ=0.5', 'SPURS-multi', 'MutateEverything Stage 2', 'ThermoMPNN-D', 'GeoDDG-3D', 'Rosetta', 'ESM3-small-open', 'ProteinMPNN'):
            reps_c = boot_rmse(dpred_by_model[mname])
            # Lower RMSE is better => H1: RMSE_comp - RMSE_msr > 0
            p_v = one_sided_p(reps_c - reps_msr)
            # Keep existing value/r/rmse for external comparators; only update p_one_sided (except σ=0.5 already added)
            if mname == 'ESM-MSR σ=0.5':
                for rr in rows:
                    if rr['section'] == 'epistasis_s3b_rmse' and rr['subset'] == sub_k and rr['model'] == mname:
                        rr['p_one_sided'] = p_v
            else:
                rows.append(_row('epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', sub_k, mname, p=p_v))

    # Paired bootstrap for raw MCC
    tpos = dt_all > 0
    def boot_mcc(dp_arr: np.ndarray) -> np.ndarray:
        n_s = dp_arr.shape[0]
        tp_ds = np.zeros((n_dg, n_s)); tn_ds = np.zeros((n_dg, n_s))
        fp_ds = np.zeros((n_dg, n_s)); fn_ds = np.zeros((n_dg, n_s))
        for s in range(n_s):
            ppos = dp_arr[s] > 0
            np.add.at(tp_ds[:, s], dg_id, (ppos & tpos).astype(float))
            np.add.at(tn_ds[:, s], dg_id, ((~ppos) & (~tpos)).astype(float))
            np.add.at(fp_ds[:, s], dg_id, (ppos & (~tpos)).astype(float))
            np.add.at(fn_ds[:, s], dg_id, ((~ppos) & tpos).astype(float))
        tp = gw @ tp_ds; tn = gw @ tn_ds; fp = gw @ fp_ds; fn = gw @ fn_ds
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_bs = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
        return np.sum(mcc_bs * sw3, axis=1) if n_s == 3 else mcc_bs[:, 0]

    mcc_msr = boot_mcc(dpred_by_model['ESM-MSR'])
    for mname in ('SPURS-multi', 'MutateEverything Stage 2', 'ThermoMPNN-D', 'GeoDDG-3D', 'Rosetta', 'ESM3-small-open', 'ProteinMPNN'):
        p_m = one_sided_p(mcc_msr - boot_mcc(dpred_by_model[mname]))
        rows.append(_row('epistasis_s3b_mcc', 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_raw_dddG_gt0', mname, p=p_m))

    return rows


def compute_s4c_megascale_rmse() -> list[dict]:
    print('[3/5] Computing Tobit-linked S4c (MegaScale pooled raw RMSE)...')
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    ok = sp['use'].to_numpy(bool)
    y = sp['y'].to_numpy(float)[ok]
    dg = sp['dG_wt'].to_numpy(float)[ok]
    grps = sp['group'].astype(str).to_numpy()[ok]
    uniq_g = sorted(np.unique(grps))
    n_g = len(uniq_g)
    gid = np.array([uniq_g.index(g) for g in grps], dtype=int)

    msr_un = deployed_arr(fr, 'msrp0:wt', 'msrp0:comb')[:, ok]
    msr_lk = apply_link(msr_un, dg)
    msr_cb_lk = apply_link(np.atleast_2d(fr.preds['msrp0:comb'])[:, ok], dg)

    comps = [
        ('ESM-MSR Dual-View (incl. singles)', msr_cb_lk),
        ('SPURS', deployed_arr(fr, 'spurs', 'spursm')[:, ok]),
        ('MutateEverything', deployed_arr(fr, 'me1', 'me2')[:, ok]),
        ('ThermoMPNN', deployed_arr(fr, 'tmpnn', 'tmpnnd')[:, ok]),
        ('GeoDDG-3D', np.atleast_2d(fr.preds['geoddg'])[:, ok]),
        ('Rosetta', np.atleast_2d(fr.preds['rosetta'])[:, ok]),
        ('ESM3-small-open', deployed_arr(fr, 'msrrel@0:wt', 'msrrel@0:comb')[:, ok]),
    ]

    rng = np.random.default_rng(101)
    gw = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=B_BOOT).astype(float)
    sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B_BOOT).astype(float) / 3.0
    cnt_g = np.bincount(gid, minlength=n_g).astype(float)
    tot_n = gw @ cnt_g

    def boot_pooled_rmse(P: np.ndarray):
        n_s = P.shape[0]
        pt = float(np.mean([np.sqrt(np.mean((P[s] - y) ** 2)) for s in range(n_s)]))
        sse_gs = np.zeros((n_g, n_s), dtype=float)
        for s in range(n_s):
            np.add.at(sse_gs[:, s], gid, (P[s] - y) ** 2)
        rmse_bs = np.sqrt((gw @ sse_gs) / np.maximum(tot_n[:, None], 1.0))
        reps = np.sum(rmse_bs * sw3, axis=1) if n_s == 3 else rmse_bs[:, 0]
        return pt, reps

    pt_lk, rep_lk = boot_pooled_rmse(msr_lk)
    pt_un, _ = boot_pooled_rmse(msr_un)

    rows = [
        _row('thermo_rmse_s4c', 'tsuboyama_capped_test', 'rmse_raw', 'ESM-MSR', val=pt_lk, rmse=pt_lk),
        _row('thermo_rmse_s4c', 'tsuboyama_capped_test', 'rmse_raw', 'ESM-MSR (unlinked)', rmse=pt_un, is_unlinked=True),
    ]
    for cname, carr in comps:
        pt_c, rep_c = boot_pooled_rmse(carr)
        rows.append(_row('thermo_rmse_s4c', 'tsuboyama_capped_test', 'rmse_raw', cname, val=pt_c, rmse=pt_c, p=one_sided_p(rep_c - rep_lk)))
    return rows


def compute_s5a_unlinked_mgnify_rocklin() -> list[dict]:
    print('[4/5] Computing unlinked RMSEs for S5a (MGnify & Rocklin)...')
    rows: list[dict] = []
    for run, src, bm in (
        ('all', 'mgnify', 'MGnify_heldout_pooled'),
        ('rdenovo', 'rocklin_denovo', 'Rocklin_designs_heldout_pooled'),
    ):
        excl = set(heldout_excluded(src))
        with open(ROOT / 'repo/analysis/cache' / f'frame_mgnify_FINAL_p0_{run}_s123.pkl', 'rb') as fh:
            fr = pickle.load(fh)
        sp = fr.spine
        m = sp['use'].to_numpy() & (~sp['code_wt'].astype(str).isin(excl).to_numpy())
        y = sp.loc[m, 'y'].to_numpy(float)
        P_un = np.atleast_2d(fr.preds['comb'])[:, m]
        rm_un = float(np.mean([safe_rmse(P_un[s], y) for s in range(P_un.shape[0])]))
        rows.append(_row('ood_s5a_unlinked', bm, 'pooled_spearman_rho', 'ESM-MSR (unlinked)', rmse=rm_un, is_unlinked=True))
    return rows


def compute_s6b_megascale_singles() -> list[dict]:
    print('[5/5] Computing Tobit-linked Pool r & RMSE for S6b (MegaScale Singles no-Cys)...')
    fr = load_frame('tsuboyama_capped_test')
    sp = fr.spine
    ok = sp['use'].to_numpy(bool)
    n1 = (sp['n_mut'].to_numpy(int) == 1)
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
    m = ok & n1 & (~to_cys)
    y = sp['y'].to_numpy(float)[m]
    dg = sp['dG_wt'].to_numpy(float)[m]
    wt_un = np.atleast_2d(fr.preds['msrp0:wt'])[:, m]
    wt_lk = apply_link(wt_un, dg)
    r_lk = float(np.mean([safe_pearson(wt_lk[s], y) for s in range(wt_lk.shape[0])]))
    rm_lk = float(np.mean([safe_rmse(wt_lk[s], y) for s in range(wt_lk.shape[0])]))
    rm_un = float(np.mean([safe_rmse(wt_un[s], y) for s in range(wt_un.shape[0])]))
    cb_un = np.atleast_2d(fr.preds['msrp0:comb'])[:, m]
    cb_lk = apply_link(cb_un, dg)
    r_cb_lk = float(np.mean([safe_pearson(cb_lk[s], y) for s in range(cb_lk.shape[0])]))
    rm_cb_lk = float(np.mean([safe_rmse(cb_lk[s], y) for s in range(cb_lk.shape[0])]))
    return [
        _row('screening_s6b', 'Figure4b_MegaScale_singles_nocys', 'pool', 'ESM-MSR', r=r_lk, rmse=rm_lk),
        _row('screening_s6b', 'Figure4b_MegaScale_singles_nocys', 'pool', 'ESM-MSR (unlinked)', rmse=rm_un, is_unlinked=True),
        _row('screening_s6b', 'Figure4b_MegaScale_singles_nocys', 'pool', 'ESM-MSR Dual-View (incl. singles)', r=r_cb_lk, rmse=rm_cb_lk),
    ]


def main() -> None:
    rows: list[dict] = []
    rows.extend(compute_s2_megascale())
    rows.extend(compute_s3_epistasis())
    rows.extend(compute_s4c_megascale_rmse())
    rows.extend(compute_s5a_unlinked_mgnify_rocklin())
    rows.extend(compute_s6b_megascale_singles())
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(df)} Tobit-linked override/unlinked rows to {OUT_CSV}')
    print(df.groupby(['section', 'benchmark'])['model'].count().to_string())


if __name__ == '__main__':
    main()
