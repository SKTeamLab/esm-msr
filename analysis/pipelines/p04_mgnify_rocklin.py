#!/usr/bin/env python3
"""Pipeline p04: MGnify & Rocklin de novo benchmarks, homology strata, multi-model comparisons, and beyond-global epistasis.

Consolidates:
  - 18_mgnify.py                     -> results/05_mgnify/{mgnify_summary.csv, mgnify_rmse.csv, mgnify_per_scaffold_rho.csv,
                                        table_mgnify{,_ci}.md, table_rocklin{,_ci}.md, table_homology.md,
                                        mgnify_overview, rocklin_overview, mgnify_homology, mgnify_per_scaffold, mgnify_scatter}
  - 18b_mgnify_models.py             -> results/05_mgnify/{mgnify_models_per_scaffold.csv, mgnify_models_pooled.csv}
  - 18c_mgnify_by_mutation_count.py  -> results/05_mgnify/{mgnify_by_mutation_count.csv, mgnify_by_mutation_count}
  - scoring/mgnify_rdenovo_stats.py  -> results/05_mgnify/rocklin_designs/{stats.csv, epistasis.csv}
  - scoring/mgnify_epistasis_beyond.py -> results/05_mgnify/new_data_epistasis/{summary.csv, beyond_rho_epistasis}

Key methodological fixes vs legacy scripts:
  - Enforces joint scaffold + 3-seed resampling across 18b, 18c, rocklin_designs, and new_data_epistasis.
  - Uses O(n) SpearmanEngine for 600k-variant pooled bootstraps (replacing O(n log n) scipy.stats.rankdata loops).
  - Applies consistent 4-part holdout filtering (msr_analysis.heldout) to Rocklin de novo designs and beyond-global epistasis,
    while keeping explicit homology-strata comparisons ('all', 'heldout', 'clean', 'h', 'strong') clearly labelled in 05_mgnify.
  - Uses ddof=1 (msr_analysis.stats.seed_sd) and NaN-safe Spearman/Pearson throughout.
"""
from __future__ import annotations

import copy
import glob
import multiprocessing as mp
import os
import re
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from esm_msr import epi_metrics as E
from esm_msr.val_stats import tobit_link

from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import SpearmanEngine, bootstrap_rho, ci, paired_diff
from msr_analysis.data import Frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.heldout import excluded
from msr_analysis.mgnify import RUNS, STRATA, finished_seeds, legacy_cens, load_mgnify
from msr_analysis.paths import FINAL_MGNIFY, MGNIFY_TABLE, PUBLIC, TASKS, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.stats import boot_domain_seeds, make_domain_seed_draws, seed_sd
from msr_analysis.tobit import TOBIT

from . import base_parser, resolve_b

STAGE = 'mgnify_rocklin'
DESCRIPTION = 'MGnify & Rocklin de novo stability benchmarks, homology strata, and beyond-global epistasis'

MIN_VARIANTS = 10
HEAD_LABEL = {'wt': 'WT head', 'mt': 'MT head', 'comb': 'combined'}
FORMS = [('latent', ''), ('linked', '~')]
SUBSETS = {
    'overall': lambda s: np.ones(len(s), bool),
    'singles': lambda s: (s['n_mut'] == 1).to_numpy(),
    'doubles': lambda s: (s['n_mut'] == 2).to_numpy(),
    '3 mutations': lambda s: (s['n_mut'] == 3).to_numpy(),
    'dG > 0': lambda s: (s['dG_ML'] > 0).to_numpy(),
    'dG <= 0': lambda s: (s['dG_ML'] <= 0).to_numpy(),
    'stabilizing (ddG > 0)': lambda s: (s['y'] > 0).to_numpy(),
    'destabilizing (ddG < 0)': lambda s: (s['y'] < 0).to_numpy(),
}
MAIN_SUBSETS = ['overall', 'singles', 'doubles', 'dG > 0', 'stabilizing (ddG > 0)', 'destabilizing (ddG < 0)']
ROCKLIN_SUBSETS = ['overall', 'singles', 'doubles', '3 mutations', 'dG > 0', 'stabilizing (ddG > 0)', 'destabilizing (ddG < 0)']
COLORS = {'latent': '#2a78d6', 'linked': '#eb6834'}
TSU_COLOR = '#777777'

BASE_DIR = RUNS / 'baselines'
BASELINES_CFG = {
    'SPURS': ('spurs_unfrozen_s{}', 'SPURS_multi', (1, 2, 3), 'SPURS-multi'),
    'ThermoMPNN': ('thermompnn_s{}', 'pred_combined', (1, 2, 3), 'ThermoMPNN-D'),
    'ProteinMPNN': ('proteinmpnn', 'mpnn_score', (None,), 'direct'),
    'ProSST': ('prosst', 'prosst', (None,), 'direct'),
}
HARNESS_CFG = {'GeoDDG-3D': 'geoddg', 'ESM3dG': 'esm3dg', 'ESMtherm': 'esmtherm', 'IFUM': 'ifum'}
OVERLAP_MODELS = {'ESM3dG'}
N_HARNESS_TABLES = 40

_WORK_FR: dict = {}


# ----------------------------------------------------------------------------------------------
# 1. ESM-MSR head/form/homology-strata evaluation (18_mgnify.py)
# ----------------------------------------------------------------------------------------------
_POOLED_STATE: dict = {}


class _FastPooledRank:
    """L3 V-Cache resident 1D O(n) weighted mid-rank evaluator with sparse multi-row tie-run fixup."""

    __slots__ = ('order', 'has_ties', 's_m', 'e_m', 'tied_pos', 'tied_gid', 'inv')

    def __init__(self, v: np.ndarray, need_inv: bool = False):
        order = np.argsort(v, kind='mergesort').astype(np.int32)
        vs = v[order]
        n = len(v)
        same = np.zeros(n, dtype=bool)
        if n > 1:
            same[1:] = vs[1:] == vs[:-1]
        self.order = order
        self.has_ties = bool(same.any())
        if self.has_ties:
            is_start = np.zeros(n, dtype=bool)
            is_start[:-1] = (~same[:-1]) & same[1:]
            is_end = np.zeros(n, dtype=bool)
            is_end[:-1] = same[:-1] & (~same[1:])
            is_end[-1] = same[-1]
            self.s_m = np.flatnonzero(is_start).astype(np.int32)
            self.e_m = np.flatnonzero(is_end).astype(np.int32)
            in_tie = same | is_start
            self.tied_pos = np.flatnonzero(in_tie).astype(np.int32)
            self.tied_gid = (np.cumsum(is_start)[in_tie] - 1).astype(np.int32)
        else:
            self.s_m = self.e_m = self.tied_pos = self.tied_gid = None
        if need_inv:
            inv = np.empty(n, dtype=np.int32)
            inv[order] = np.arange(n, dtype=np.int32)
            self.inv = inv
        else:
            self.inv = None

    def ranks_sorted_1d(self, Wo: np.ndarray) -> np.ndarray:
        cs = np.cumsum(Wo, dtype=np.int64)
        r = cs.astype(np.float64) - 0.5 * Wo + 0.5
        if self.has_ties:
            s_m = self.s_m
            before_m = cs[s_m] - Wo[s_m]
            tie_t_m = cs[self.e_m] - before_m
            mid_m = before_m + (tie_t_m + 1.0) * 0.5
            r[self.tied_pos] = mid_m[self.tied_gid]
        return r

    def ranks_input_1d(self, W: np.ndarray) -> np.ndarray:
        return self.ranks_sorted_1d(W[self.order])[self.inv]

    def corr_1d_with_y(self, sc: np.ndarray, code_x: np.ndarray, W_ry: np.ndarray, sw: float, my: float, vy: float) -> float:
        if sw < 3.0 or vy <= 0.0 or not np.isfinite(vy):
            return np.nan
        Wo = sc[code_x]
        rx_s = self.ranks_sorted_1d(Wo)
        wrx = Wo * rx_s
        mx = float(wrx.sum()) / sw
        vx = float(np.dot(wrx, rx_s)) / sw - mx * mx
        if vx <= 0.0 or not np.isfinite(vx):
            return np.nan
        cxy = float(np.dot(rx_s, W_ry[self.order])) / sw - mx * my
        return float(cxy / np.sqrt(vx * vy))


def _pooled_shard_worker(b_indices: list[int]) -> dict[str, np.ndarray]:
    xr_list_dict = _POOLED_STATE['xr_list_dict']
    code_x_dict = _POOLED_STATE['code_x_dict']
    ry_prep = _POOLED_STATE['ry_prep']
    code_y = _POOLED_STATE['code_y']
    code = _POOLED_STATE['code']
    scaf_i32 = _POOLED_STATE['scaf_i32']
    seed_w = _POOLED_STATE['seed_w']
    out = {k: np.empty(len(b_indices), dtype=float) for k in xr_list_dict}
    for j, bi in enumerate(b_indices):
        sc = scaf_i32[bi]
        W = sc[code]
        sw = float(W.sum(dtype=np.int64))
        ry_s = ry_prep.ranks_sorted_1d(sc[code_y])
        ry = ry_s[ry_prep.inv]
        W_ry = W * ry
        if sw >= 3.0:
            my = float(W_ry.sum()) / sw
            vy = float(np.dot(W_ry, ry)) / sw - my * my
        else:
            my = vy = np.nan
        sw_b = seed_w[bi]
        for k, xrs in xr_list_dict.items():
            cxs = code_x_dict[k]
            rs = np.array([xr.corr_1d_with_y(sc, cx, W_ry, sw, my, vy) for xr, cx in zip(xrs, cxs)], dtype=float)
            out[k][j] = float(np.dot(sw_b, rs)) if len(rs) == 3 else float(np.nanmean(rs))
    return out


def _eval_pooled_fast(xr_list_dict: dict[str, list[_FastPooledRank]], ry_prep: _FastPooledRank, n_rows: int, code: np.ndarray, scaf_counts: np.ndarray, seed_w: np.ndarray, chunk: int = 25, workers: int = 1) -> tuple[dict[str, tuple[float, np.ndarray]], dict[str, np.ndarray]]:
    """Evaluate point and ``B``-draw pooled Spearman replicates using L3-resident 1D AVX-512 gathers + BLAS ddot."""
    code_i32 = np.asarray(code, dtype=np.int32)
    code_y = code_i32[ry_prep.order]
    code_x_dict = {k: [code_i32[xr.order] for xr in xrs] for k, xrs in xr_list_dict.items()}

    sc1 = np.ones(int(code_i32.max()) + 1 if len(code_i32) else 1, dtype=np.int32)
    ry1 = ry_prep.ranks_sorted_1d(np.ones(n_rows, dtype=np.int32))[ry_prep.inv]
    sw1 = float(n_rows)
    my1 = float(ry1.mean())
    vy1 = float(np.dot(ry1, ry1)) / sw1 - my1 * my1
    point_out = {}
    for k, xrs in xr_list_dict.items():
        ps = np.array([xr.corr_1d_with_y(sc1, cx, ry1, sw1, my1, vy1) for xr, cx in zip(xrs, code_x_dict[k])], dtype=float)
        point_out[k] = (float(np.mean(ps)), ps)

    B_tot = len(scaf_counts)
    reps_out = {k: np.empty(B_tot, dtype=float) for k in xr_list_dict}
    if B_tot == 0:
        return point_out, reps_out

    scaf_i32 = np.ascontiguousarray(scaf_counts, dtype=np.int32)
    n_w = min(max(1, workers), B_tot)
    _POOLED_STATE.clear()
    _POOLED_STATE.update(
        xr_list_dict=xr_list_dict, code_x_dict=code_x_dict, ry_prep=ry_prep,
        code_y=code_y, code=code_i32, scaf_i32=scaf_i32, seed_w=seed_w,
    )
    try:
        if n_w > 1 and n_rows >= 20_000 and B_tot >= 16:
            shards = [list(range(i, B_tot, n_w)) for i in range(n_w)]
            with mp.get_context('fork').Pool(n_w) as pool:
                parts = pool.map(_pooled_shard_worker, shards)
            for sh, part in zip(shards, parts):
                for k, arr in part.items():
                    reps_out[k][sh] = arr
        else:
            part = _pooled_shard_worker(list(range(B_tot)))
            for k, arr in part.items():
                reps_out[k][:] = arr
    finally:
        _POOLED_STATE.clear()
    return point_out, reps_out


def _cell_stats(fr: Frame, mask: np.ndarray, label: str, B: int, seed: int, heads=('wt', 'comb')) -> list[dict]:
    rows = []
    sp = fr.spine
    y_all = sp['y'].to_numpy(float)
    m_base = sp['use'].to_numpy() & mask & np.isfinite(y_all)
    for h in heads:
        if h in fr.preds:
            m_base &= np.isfinite(fr.preds[h]).all(axis=0)
    n_rows = int(m_base.sum())
    if n_rows < 30:
        return rows

    # 1. Pooled Spearman rho (fast BLAS O(n) mid-ranks + joint scaffold & seed resampling)
    idx_p = np.flatnonzero(m_base)
    y_p = y_all[idx_p]
    code_p, uniq_p = pd.factorize(sp['cluster'].to_numpy()[idx_p])
    code_p = code_p.astype(np.int32)
    n_scaf_p = len(uniq_p)
    b_pool = B if n_rows <= 100_000 else min(B, 200)
    rng_p = np.random.default_rng(seed)
    scaf_counts_p = rng_p.multinomial(n_scaf_p, np.full(n_scaf_p, 1.0 / n_scaf_p), size=b_pool).astype(float)
    seed_w_p = rng_p.multinomial(3, np.full(3, 1.0 / 3.0), size=b_pool).astype(float) / 3.0

    ry_prep = _FastPooledRank(y_p, need_inv=True)
    keys = [h + f for h in heads for _, f in FORMS if (h + f) in fr.preds]
    xr_map = {k: [_FastPooledRank(p[idx_p]) for p in fr.preds[k]] for k in keys}
    point_p, reps_p = _eval_pooled_fast(xr_map, ry_prep, n_rows, code_p, scaf_counts_p, seed_w_p, chunk=25 if n_rows > 150_000 else 50)

    for h in heads:
        for form, f in FORMS:
            k = h + f
            if k not in point_p:
                continue
            rho_v, per_seed = point_p[k]
            lo, hi = ci(reps_p[k])
            rows.append(dict(
                subset=label, unit='pooled', head=h, form=form,
                rho=rho_v, lo=lo, hi=hi, n_rows=n_rows, n_scaffolds=n_scaf_p,
                seeds='/'.join(f'{x:.3f}' for x in per_seed),
            ))
        if h in reps_p and (h + '~') in reps_p:
            d = paired_diff(reps_p[h + '~'], reps_p[h], point_a=point_p[h + '~'][0], point_b=point_p[h][0])
            rows.append(dict(
                subset=label, unit='pooled', head=h, form='linked - latent',
                rho=d['mean'], lo=d['lo'], hi=d['hi'], n_rows=n_rows, n_scaffolds=n_scaf_p,
                seeds=f"p={d['p']:.3g}",
            ))

    # 2. Per-scaffold mean Spearman rho (exact single-pass per-scaffold rho + B-draw domain×seed bootstrap)
    grp_vals = sp['group'].to_numpy()[m_base]
    vc = pd.Series(grp_vals).value_counts()
    keep_g = set(vc[vc >= MIN_VARIANTS].index)
    m_ps = m_base & sp['group'].isin(keep_g).to_numpy()
    n_rows_ps = int(m_ps.sum())
    if n_rows_ps >= 30 and len(keep_g) >= 2:
        idx_s = np.flatnonzero(m_ps)
        y_s = y_all[idx_s]
        code_s, uniq_s = pd.factorize(sp['group'].to_numpy()[idx_s])
        code_s = code_s.astype(np.int64)
        ng_s = len(uniq_s)
        counts_s, seed_w_s = make_domain_seed_draws(ng_s, B=B, seed=seed)
        for h in heads:
            if h not in fr.preds:
                continue
            V_h = np.column_stack([_scaffold_rho_per_seed(code_s, ng_s, p[idx_s], y_s) for p in fr.preds[h]])
            per_seed_s = np.nanmean(V_h, axis=0)
            pt_s, reps_s = boot_domain_seeds(V_h, B=B, counts=counts_s, seed_weights=seed_w_s)
            lo_s, hi_s = ci(reps_s)
            seeds_str = '/'.join(f'{x:.3f}' for x in per_seed_s)
            # Within any scaffold dG_wt is constant and Tobit link is strictly monotone -> latent == linked
            for form, f in FORMS:
                if (h + f) not in fr.preds:
                    continue
                rows.append(dict(
                    subset=label, unit='per scaffold', head=h, form=form,
                    rho=pt_s, lo=lo_s, hi=hi_s, n_rows=n_rows_ps, n_scaffolds=ng_s,
                    seeds=seeds_str,
                ))
            if (h + '~') in fr.preds:
                rows.append(dict(
                    subset=label, unit='per scaffold', head=h, form='linked - latent',
                    rho=0.0, lo=0.0, hi=0.0, n_rows=n_rows_ps, n_scaffolds=ng_s,
                    seeds='identical (monotone within a scaffold)',
                ))
    return rows


def _tsuboyama_p0_frame() -> Frame:
    fr = load_final_frame('tsuboyama_capped_test', ['FINAL_p0'])
    sp = fr.spine.copy()
    t = pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', usecols=['id', 'dG_ML']).drop_duplicates('id').set_index('id')['dG_ML']
    sp['dG_ML'] = sp['id'].map(t).to_numpy()
    h = tobit_link(**TOBIT)
    dg = sp['dG_wt'].to_numpy(float)
    preds = {}
    for k in ('wt', 'comb', 'mt'):
        a = fr.preds[f'FINAL_p0:{k}']
        preds[k] = a
        preds[k + '~'] = h(dg[None, :] + a) - dg[None, :]
    return Frame('tsuboyama_p0', sp, preds)


def _strata_worker(job_args) -> list[dict]:
    which, stratum, name, B, seed, heads = job_args
    fr = _WORK_FR[which]
    sp = fr.spine
    mask = SUBSETS[name](sp)
    if which == 'mgnify':
        if stratum == 'heldout':
            mask &= ~sp['code_wt'].astype(str).isin(excluded('mgnify')).to_numpy()
        elif STRATA.get(stratum) is not None:
            mask &= sp['cls'].isin(STRATA[stratum]).to_numpy()
    elif which == 'rocklin' and stratum == 'heldout':
        mask &= ~sp['code_wt'].astype(str).isin(excluded('rocklin_denovo')).to_numpy()
    ds_label = {'mgnify': 'MGnify', 'tsuboyama': 'Tsuboyama capped test', 'rocklin': 'Rocklin designs'}[which]
    return [dict(dataset=ds_label, stratum=stratum, **r) for r in _cell_stats(fr, mask, name, B, seed, heads)]


def _finish_foot(fig, text: str, out_stem, pad: float = 0.04, top: float = 1.0) -> None:
    kw = dict(fontsize=7.6, color='#333333')
    t = fig.text(0.012, 0.008, style.wrap_to_px(fig, text, 0.96 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    fh = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, fh + pad, 1, top))
    style.save(fig, out_stem)


def run_esm_msr_strata(B: int, workers: int = 4) -> None:
    out = results_dir('mgnify')
    seeds = finished_seeds()
    fr = load_mgnify(seeds=seeds, heldout=False)
    tf = _tsuboyama_p0_frame()
    rk_seeds = finished_seeds(run='rdenovo')
    rk = load_mgnify(run='rdenovo', seeds=rk_seeds, heldout=False) if rk_seeds else None

    _WORK_FR.clear()
    _WORK_FR.update(mgnify=fr, tsuboyama=tf, **({'rocklin': rk} if rk is not None else {}))

    jobs = []
    for stratum in (*STRATA.keys(), 'heldout'):
        names = (
            MAIN_SUBSETS + (['dG <= 0'] if stratum in ('all', 'heldout') else [])
            if stratum in ('all', 'heldout', 'clean')
            else ['overall', 'singles', 'doubles']
        )
        for name in names:
            jobs.append(('mgnify', stratum, name, B, 11, ('wt', 'comb', 'mt') if stratum in ('all', 'heldout') else ('wt', 'comb')))
    for name in MAIN_SUBSETS:
        jobs.append(('tsuboyama', 'all', name, B, 12, ('wt', 'comb', 'mt')))
    if rk is not None:
        for stratum in ('heldout', 'all'):
            for name in ROCKLIN_SUBSETS:
                jobs.append(('rocklin', stratum, name, B, 13, ('wt', 'comb', 'mt')))

    # Schedule largest MGnify jobs first so worker pool has zero tail stragglers
    sub_prio = {'overall': 0, 'destabilizing (ddG < 0)': 1, 'singles': 2, 'dG > 0': 3, 'dG <= 0': 4, 'doubles': 5}
    st_prio = {'all': 0, 'lowid': 1, 'heldout': 2, 'clean': 3, 'h': 4, 'strong': 5}
    jobs_sorted = sorted(enumerate(jobs), key=lambda t: (0 if t[1][0] == 'mgnify' else 1, sub_prio.get(t[1][2], 6), st_prio.get(t[1][1], 6)))
    with mp.get_context('fork').Pool(workers) as pool:
        parts_sorted = pool.map(_strata_worker, [j for _, j in jobs_sorted], chunksize=1)
    parts = [None] * len(jobs)
    for (orig_i, _), part in zip(jobs_sorted, parts_sorted):
        parts[orig_i] = part
    S = pd.DataFrame([r for part in parts for r in part])
    atomic_to_csv(S, out / 'mgnify_summary.csv', index=False, float_format='%.4f')

    # RMSE table (held-out MGnify & held-out Rocklin, plus full strata noted)
    rmse_rows = []
    for ds_lbl, f_obj, excl_set in (
        ('MGnify', fr, excluded('mgnify')),
        ('Tsuboyama capped test', tf, set()),
        ('Rocklin designs', rk, excluded('rocklin_denovo')),
    ):
        if f_obj is None:
            continue
        sp = f_obj.spine
        keep_ho = ~sp['code_wt'].astype(str).isin(excl_set).to_numpy() if excl_set else np.ones(len(sp), bool)
        for name in (ROCKLIN_SUBSETS if ds_lbl == 'Rocklin designs' else MAIN_SUBSETS):
            m = sp['use'].to_numpy() & keep_ho & SUBSETS[name](sp)
            y = sp['y'].to_numpy()[m]
            for h in ('wt', 'comb'):
                for form, suf in FORMS:
                    p = f_obj.preds[h + suf][:, m]
                    rmse_rows.append(dict(
                        dataset=ds_lbl, subset=name, head=h, form=form, n=int(m.sum()),
                        rmse=float(np.mean([np.sqrt(np.mean((q - y) ** 2)) for q in p])),
                        mae=float(np.mean([np.mean(np.abs(q - y)) for q in p])),
                        pearson=float(np.mean([pearsonr(q, y)[0] for q in p])),
                        mean_pred=float(p.mean()), mean_measured=float(y.mean()),
                    ))
    R = pd.DataFrame(rmse_rows)
    atomic_to_csv(R, out / 'mgnify_rmse.csv', index=False, float_format='%.4f')

    def _get(ds_n: str, st: str, sub: str, unit: str, head: str, form: str):
        t = S[(S['dataset'] == ds_n) & (S['stratum'] == st) & (S['subset'] == sub) & (S['unit'] == unit) & (S['head'] == head) & (S['form'] == form)]
        return (float(t['rho'].iloc[0]), float(t['lo'].iloc[0]), float(t['hi'].iloc[0])) if len(t) else (np.nan, np.nan, np.nan)

    def _fig_overview(ds_main: str, st_main: str, subsets: list[str], stem: str) -> None:
        unit_word = 'scaffolds' if ds_main == 'MGnify' else 'designs'
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.9 * len(subsets) + 0.2), sharey=True)
        ys = np.arange(len(subsets))[::-1]
        for ax, unit, ttl in zip(axes, ('pooled', 'per scaffold'), ('A  Pooled Spearman ρ', f'B  Mean per-{unit_word[:-1]} ρ (≥ {MIN_VARIANTS} variants)')):
            for j, (head, _hl) in enumerate((('wt', 'WT head'), ('comb', 'combined'))):
                for ds_i, st_i, mk in ((ds_main, st_main, 'o'), ('Tsuboyama capped test', 'all', 's')):
                    for fi, (form, _) in enumerate(FORMS):
                        if unit == 'per scaffold' and form == 'linked':
                            continue
                        off = (0.34 if ds_i == ds_main else -0.06) - 0.17 * j - (0.085 if fi else 0)
                        for y_pos, s_nm in zip(ys, subsets):
                            v, lo, hi = _get(ds_i, st_i, s_nm, unit, head, form)
                            if not np.isfinite(v):
                                continue
                            col = COLORS[form] if ds_i == ds_main else TSU_COLOR
                            ax.plot([lo, hi], [y_pos + off] * 2, color=col, lw=1.6, alpha=0.55, solid_capstyle='butt')
                            ax.plot(v, y_pos + off, marker=mk, ms=6.4, mfc=col if head == 'comb' else 'white', mec=col, mew=1.3, ls='none')
            ax.set_yticks(ys)
            ax.set_yticklabels(subsets)
            ax.set_xlabel('Spearman ρ')
            ax.set_title(ttl, loc='left')
            ax.grid(axis='x', color=style.GRID, lw=0.6)
            ax.set_xlim(0.0, 1.0)
        h_leg = [
            Line2D([0], [0], marker='o', color=COLORS['latent'], mfc=COLORS['latent'], ls='none', label=f'{ds_main} (held-out), latent'),
            Line2D([0], [0], marker='o', color=COLORS['linked'], mfc=COLORS['linked'], ls='none', label=f'{ds_main} (held-out), Tobit-linked'),
            Line2D([0], [0], marker='s', color=TSU_COLOR, mfc=TSU_COLOR, ls='none', label='Tsuboyama capped test (33 libraries)'),
            Line2D([0], [0], marker='o', color='#555555', mfc='#555555', ls='none', label='filled: combined'),
            Line2D([0], [0], marker='o', color='#555555', mfc='white', ls='none', label='open: WT head'),
        ]
        fig.legend(handles=h_leg, loc='upper center', ncol=3, fontsize=8, handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
        b_pool_note = f'{min(B, 200):,}' if ds_main == 'MGnify' else f'{B:,}'
        _finish_foot(
            fig,
            f'ESM-MSR, seed mean of FINAL_p0 s1-s3 on held-out {ds_main} (4-part training-homology rule; cens == 0). '
            f'Pooled: one Spearman ρ over all variants of the subset, 95% interval from resampling {unit_word} and seeds, B = {b_pool_note}. '
            f'Per {unit_word[:-1]}: mean over {unit_word} of within-{unit_word[:-1]} Spearman ρ (B = {B:,}; latent and Tobit-linked are identical within a {unit_word[:-1]}).',
            out / stem, top=0.9,
        )

    _fig_overview('MGnify', 'heldout', MAIN_SUBSETS, 'mgnify_overview')
    if rk is not None:
        _fig_overview('Rocklin designs', 'heldout', ROCKLIN_SUBSETS, 'rocklin_overview')

    # Homology strata figure (explicitly compares all, heldout, clean, h, strong)
    strata_plot = ['all', 'heldout', 'clean', 'h', 'strong']
    names_h = ['overall', 'singles', 'doubles']
    fig_h, axes_h = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    w = 0.15
    cols_h = {'all': '#444444', 'heldout': '#2a78d6', 'clean': '#1baf7a', 'h': '#d6a02a', 'strong': '#c0392b'}
    for ax, unit, ttl in zip(axes_h, ('pooled', 'per scaffold'), ('A  Pooled ρ, combined', 'B  Mean per-scaffold ρ, combined')):
        for si, st in enumerate(strata_plot):
            for ni, nm in enumerate(names_h):
                for fi, (form, _) in enumerate(FORMS):
                    if unit == 'per scaffold' and form == 'linked':
                        continue
                    v, lo, hi = _get('MGnify', st, nm, unit, 'comb', form)
                    if not np.isfinite(v):
                        continue
                    x = ni + (si - 2.0) * w + (0.038 if fi else -0.038) * (unit == 'pooled')
                    ax.plot([x, x], [lo, hi], color=cols_h[st], lw=1.5, alpha=0.6)
                    ax.plot(x, v, marker='o' if form == 'latent' else 'D', ms=5.6, mfc=cols_h[st] if form == 'latent' else 'white', mec=cols_h[st], mew=1.3, ls='none')
        ax.set_xticks(range(len(names_h)))
        ax.set_xticklabels(names_h)
        ax.set_title(ttl, loc='left')
        ax.grid(axis='y', color=style.GRID, lw=0.6)
    axes_h[0].set_ylabel('Spearman ρ')
    n_sc = {
        st: int(S[(S['dataset'] == 'MGnify') & (S['stratum'] == st) & (S['subset'] == 'overall') & (S['unit'] == 'pooled') & (S['head'] == 'comb') & (S['form'] == 'latent')]['n_scaffolds'].iloc[0])
        for st in strata_plot
    }
    lab_h = {
        'all': 'all scaffolds (unfiltered)',
        'heldout': '4-part held-out',
        'clean': 'sequence-clean (none + low)',
        'h': 'h (weak homolog)',
        'strong': 'strong (H + same protein)',
    }
    h_items = [Line2D([0], [0], marker='o', color=cols_h[st], mfc=cols_h[st], ls='none', label=f'{lab_h[st]}, {n_sc[st]:,} scaffolds') for st in strata_plot]
    h_items += [
        Line2D([0], [0], marker='o', color='#555555', mfc='#555555', ls='none', label='latent (filled)'),
        Line2D([0], [0], marker='D', color='#555555', mfc='white', ls='none', label='Tobit-linked (open, pooled)'),
    ]
    fig_h.legend(handles=h_items, loc='upper center', ncol=4, fontsize=7.6, handletextpad=0.3, columnspacing=1.3, bbox_to_anchor=(0.5, 1.0))
    _finish_foot(
        fig_h,
        f'ESM-MSR Dual-view prediction on MGnify stratified by homology to the capped training set (all, 4-part held-out, MMseqs2 clean, weak h, strong H + same protein). '
        f'Seed mean of s1-s3, 95% intervals from resampling scaffolds and seeds, B = {B:,}.',
        out / 'mgnify_homology', top=0.87,
    )

    # Per-scaffold scatter & distribution (held-out MGnify)
    sp_mg = fr.spine
    m0 = sp_mg['use'].to_numpy() & (sp_mg['n_mut'] >= 1).to_numpy() & ~sp_mg['code_wt'].astype(str).isin(excluded('mgnify')).to_numpy()
    d_sc = pd.DataFrame({'g': sp_mg['group'].to_numpy()[m0], 'y': sp_mg['y'].to_numpy()[m0]})
    for h in ('wt', 'comb'):
        d_sc[h] = fr.preds[h][:, m0].mean(0)
    sc_map = {
        g: (len(x), spearmanr(x['wt'], x['y'])[0], spearmanr(x['comb'], x['y'])[0])
        for g, x in d_sc.groupby('g') if len(x) >= MIN_VARIANTS
    }
    P_sc = pd.DataFrame(sc_map, index=['n', 'wt', 'comb']).T
    atomic_to_csv(P_sc, out / 'mgnify_per_scaffold_rho.csv', float_format='%.4f')
    fig_ps, axes_ps = plt.subplots(1, 2, figsize=(10.6, 4.6))
    sc_pts = axes_ps[0].scatter(P_sc['wt'], P_sc['comb'], s=7, c=np.log10(P_sc['n']), cmap='viridis', alpha=0.7, lw=0)
    axes_ps[0].plot([-0.2, 1.0], [-0.2, 1.0], color='#999', lw=0.8, ls='--')
    axes_ps[0].set_xlabel('ρ of the WT head')
    axes_ps[0].set_ylabel('ρ of the combined prediction')
    axes_ps[0].text(0.04, 0.95, f'combined higher in {(P_sc["comb"] > P_sc["wt"]).sum():,} / {len(P_sc):,} scaffolds', transform=axes_ps[0].transAxes, va='top', fontsize=8.5)
    axes_ps[0].set_title('A  Each dot is one held-out scaffold', loc='left')
    cb = fig_ps.colorbar(sc_pts, ax=axes_ps[0], pad=0.02)
    cb.set_label('log10 variants in the scaffold')
    bins = np.linspace(-0.3, 1.0, 40)
    for h, c_col in (('wt', '#2a78d6'), ('comb', '#eb6834')):
        axes_ps[1].hist(P_sc[h], bins=bins, histtype='step', color=c_col, lw=1.7, label=f'{HEAD_LABEL[h]} (mean {P_sc[h].mean():.3f}, median {P_sc[h].median():.3f})')
    axes_ps[1].set_xlabel('per-scaffold Spearman ρ')
    axes_ps[1].set_ylabel('scaffolds')
    axes_ps[1].legend(fontsize=8)
    axes_ps[1].set_title('B  Distribution over held-out scaffolds', loc='left')
    _finish_foot(
        fig_ps,
        f'Held-out MGnify scaffolds with ≥ {MIN_VARIANTS} measured variants (n = {len(P_sc):,}), ESM-MSR predictions averaged over seeds 1-3.',
        out / 'mgnify_per_scaffold',
    )

    # Predicted vs measured ddG scatters (held-out MGnify)
    fig_sct, axes_sct = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True)
    for ax, (h, suf, subset) in zip(axes_sct, [('wt', '', 'singles'), ('wt', '~', 'singles'), ('comb', '', 'doubles'), ('comb', '~', 'doubles')]):
        m_sub = m0 & (sp_mg['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy()
        x_v = fr.preds[h + suf][:, m_sub].mean(0)
        style.density_scatter(ax, x_v, sp_mg['y'].to_numpy()[m_sub], s=2, line=True, max_points=40000)
        ax.set_title(f'{HEAD_LABEL[h]}, {subset}\n{"Tobit-linked" if suf else "latent"}', fontsize=9, loc='left')
        ax.set_xlabel('predicted ddG (kcal/mol)')
    axes_sct[0].set_ylabel('measured ddG (kcal/mol)')
    _finish_foot(
        fig_sct,
        'Held-out MGnify measured variants, ESM-MSR seed mean. The Tobit link h(dG_wt + latent) − dG_wt compresses predictions of very destabilized variants toward the unfolded floor.',
        out / 'mgnify_scatter',
    )

    # Markdown tables
    def _cell_fmt(ds_n, st, sub, unit, head, form, with_ci=False):
        t = S[(S['dataset'] == ds_n) & (S['stratum'] == st) & (S['subset'] == sub) & (S['unit'] == unit) & (S['head'] == head) & (S['form'] == form)]
        if not len(t):
            return '–'
        r = t.iloc[0]
        return f'{r.rho:.3f} [{r.lo:.3f}, {r.hi:.3f}]' if with_ci else f'{r.rho:.3f}'

    def _n_of(ds_n, st, sub, unit, head='comb'):
        t = S[(S['dataset'] == ds_n) & (S['stratum'] == st) & (S['subset'] == sub) & (S['unit'] == unit) & (S['head'] == head) & (S['form'] == 'latent')]
        return (int(t['n_rows'].iloc[0]), int(t['n_scaffolds'].iloc[0])) if len(t) else (0, 0)

    def _rmse_fmt(ds_n, sub, head, form):
        t = R[(R['dataset'] == ds_n) & (R['subset'] == sub) & (R['head'] == head) & (R['form'] == form)]
        return f'{t["rmse"].iloc[0]:.3f}' if len(t) else '–'

    for tag, ci_flag in (('table_mgnify', False), ('table_mgnify_ci', True)):
        lines = []
        for head in ('comb', 'wt', 'mt'):
            lines += [f'**{HEAD_LABEL[head]}**', '', '| set | subset | variants | scaffolds | pooled ρ, latent | pooled ρ, Tobit-linked | per-scaffold ρ | scaffolds (per-scaffold) | RMSE latent | RMSE linked |', '|---|---|---|---|---|---|---|---|---|---|']
            for ds_n, st, label in (('MGnify', 'heldout', 'MGnify (held-out)'), ('Tsuboyama capped test', 'all', 'Tsuboyama capped test')):
                for sub in MAIN_SUBSETS:
                    if head != 'comb' and sub not in ('overall', 'singles', 'doubles'):
                        continue
                    n_r, n_s = _n_of(ds_n, st, sub, 'pooled', head)
                    _, n_ps = _n_of(ds_n, st, sub, 'per scaffold', head)
                    if not n_r:
                        continue
                    lines.append(
                        f'| {label} | {sub} | {n_r:,} | {n_s:,} | {_cell_fmt(ds_n, st, sub, "pooled", head, "latent", ci_flag)} | '
                        f'{_cell_fmt(ds_n, st, sub, "pooled", head, "linked", ci_flag)} | {_cell_fmt(ds_n, st, sub, "per scaffold", head, "latent", ci_flag)} | '
                        f'{n_ps:,} | {_rmse_fmt(ds_n, sub, head, "latent") if head != "mt" else "–"} | {_rmse_fmt(ds_n, sub, head, "linked") if head != "mt" else "–"} |'
                    )
            lines.append('')
        atomic_write_text(out / f'{tag}.md', '\n'.join(lines) + '\n')

    if rk is not None:
        for tag, ci_flag in (('table_rocklin', False), ('table_rocklin_ci', True)):
            lines = []
            for head in ('comb', 'wt', 'mt'):
                lines += [f'**{HEAD_LABEL[head]}**', '', '| set | subset | variants | designs | pooled ρ, latent | pooled ρ, Tobit-linked | per-design ρ | designs (per-design) | RMSE latent | RMSE linked |', '|---|---|---|---|---|---|---|---|---|---|']
                for sub in ROCKLIN_SUBSETS:
                    if head != 'comb' and sub not in ('overall', 'singles', 'doubles', '3 mutations'):
                        continue
                    n_r, n_s = _n_of('Rocklin designs', 'heldout', sub, 'pooled', head)
                    _, n_ps = _n_of('Rocklin designs', 'heldout', sub, 'per scaffold', head)
                    if not n_r:
                        continue
                    lines.append(
                        f'| Rocklin designs (held-out) | {sub} | {n_r:,} | {n_s:,} | {_cell_fmt("Rocklin designs", "heldout", sub, "pooled", head, "latent", ci_flag)} | '
                        f'{_cell_fmt("Rocklin designs", "heldout", sub, "pooled", head, "linked", ci_flag)} | {_cell_fmt("Rocklin designs", "heldout", sub, "per scaffold", head, "latent", ci_flag)} | '
                        f'{n_ps:,} | {_rmse_fmt("Rocklin designs", sub, head, "latent") if head != "mt" else "–"} | {_rmse_fmt("Rocklin designs", sub, head, "linked") if head != "mt" else "–"} |'
                    )
                lines.append('')
            atomic_write_text(out / f'{tag}.md', '\n'.join(lines) + '\n')

    hom_lines = ['| stratum | subset | scaffolds | pooled ρ, latent | pooled ρ, Tobit-linked | per-scaffold ρ |', '|---|---|---|---|---|---|']
    for st in ('all', 'heldout', 'lowid', 'clean', 'h', 'strong'):
        for sub in ('overall', 'singles', 'doubles'):
            n_r, n_s = _n_of('MGnify', st, sub, 'pooled')
            if n_r:
                hom_lines.append(
                    f'| {st} | {sub} | {n_s:,} | {_cell_fmt("MGnify", st, sub, "pooled", "comb", "latent", True)} | '
                    f'{_cell_fmt("MGnify", st, sub, "pooled", "comb", "linked", True)} | {_cell_fmt("MGnify", st, sub, "per scaffold", "comb", "latent", True)} |'
                )
    atomic_write_text(out / 'table_homology.md', '\n'.join(hom_lines) + '\n')


# ----------------------------------------------------------------------------------------------
# 2. Multi-model comparison & mutation-count stratification (18b + 18c) with fast SpearmanEngine
# ----------------------------------------------------------------------------------------------
def _read_chunks(tpl: str, seed, src: str, col: str) -> pd.Series | None:
    fs = glob.glob(str(BASE_DIR / (tpl.format(seed) if seed else tpl) / 'mgnify' / src / 'chunk_*.csv.gz'))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, usecols=['id', col], low_memory=False) for f in fs]).drop_duplicates('id')
    return d.set_index('id')[col]


def _read_prosst_multi_chunks(src: str, col: str = 'comb_pred') -> pd.Series | None:
    fs = glob.glob(str(PUBLIC / 'prosst_multi' / 'dual_unmasked' / 'mgnify' / src / 'chunk_*.csv.gz'))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, usecols=['id', col], low_memory=False) for f in fs]).drop_duplicates('id')
    return d.set_index('id')[col]


def _read_harness(model_dir: str, ids: np.ndarray) -> np.ndarray | None:
    fs = sorted(glob.glob(str(PUBLIC / model_dir / 'mgnify' / 'mg_*_[0-9][0-9][0-9].csv')))
    if len(fs) < N_HARNESS_TABLES:
        return None
    d = pd.concat([pd.read_csv(f, usecols=['id', 'pred']) for f in fs]).drop_duplicates('id').set_index('id')['pred']
    return d.reindex(ids).to_numpy(float)


def _read_me_rocklin(seed: int, ids: np.ndarray) -> np.ndarray | None:
    f = BASE_DIR / f'me_st2_s{seed}' / 'results_rocklin_denovo.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=['pdb_id', 'mut_info', 'scores'])
    s = pd.Series(d['scores'].to_numpy(float), index=(d['pdb_id'].astype(str) + '_' + d['mut_info'].astype(str)).to_numpy())
    return s[~s.index.duplicated()].reindex(ids).to_numpy(float)


def _read_esm3_wt(ids: np.ndarray) -> np.ndarray | None:
    f = BASE_DIR / 'esm3_zeroshot' / 'mgnify' / 'predictions.csv.gz'
    if f.exists():
        d = pd.read_csv(f, usecols=['uid', 'wt_lora_pred'], low_memory=False)
    else:
        fs = glob.glob(str(BASE_DIR / 'esm3_zeroshot' / 'mgnify' / 'shards' / 'shard_*.csv'))
        if not fs:
            return None
        d = pd.concat([pd.read_csv(g, usecols=['uid', 'wt_lora_pred'], low_memory=False) for g in fs])
    return d.drop_duplicates('uid').set_index('uid')['wt_lora_pred'].reindex(ids).to_numpy(float)


def _scaffold_rho_per_seed(code: np.ndarray, n_groups: int, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast per-scaffold Spearman rho returning length-``n_groups`` vector (NaN when any row in scaffold is NaN)."""
    bad_g = np.bincount(code, (~np.isfinite(pred)).astype(int), minlength=n_groups) > 0
    pred_clean = np.where(np.isfinite(pred), pred, 0.0)
    eng = SpearmanEngine(y, code, n_groups)
    xr = eng.prepare(pred_clean)
    r = eng.rho(xr, np.ones(len(y), dtype=float))
    r[bad_g] = np.nan
    return r


def _pooled_scaffold_boot(preds_list: list[np.ndarray], y: np.ndarray, code: np.ndarray, n_scaf: int, scaf_counts: np.ndarray, seed_draws: np.ndarray, mask: np.ndarray | None = None, workers: int = 1) -> tuple[float, float, float]:
    """Fast O(n) pooled Spearman bootstrap resampling whole scaffolds and triplicate seeds."""
    if mask is not None:
        idx = np.flatnonzero(mask)
        y_s = y[idx]
        c_s = code[idx]
        P_s = [p[idx] for p in preds_list]
    else:
        y_s = y
        c_s = code
        P_s = preds_list
    ry_prep = _FastPooledRank(y_s, need_inv=True)
    xrs = [_FastPooledRank(p) for p in P_s]
    pt_m, reps_m = _eval_pooled_fast({'m': xrs}, ry_prep, len(y_s), c_s, scaf_counts, seed_draws, workers=workers)
    pt = pt_m['m'][0]
    bs = reps_m['m']
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pt, float(lo), float(hi)


def _prepare_heldout_models(cfg: dict) -> tuple[pd.DataFrame, dict]:
    fr = load_mgnify(run=cfg['run'], heldout=True)
    sp = fr.spine
    m = sp['use'].to_numpy()
    d = sp[m].reset_index(drop=True)
    ids = d['id'].to_numpy()
    preds = {
        'ESM-MSR': (
            'combined (latent)',
            [fr.preds['comb'][i, m] for i in range(fr.preds['comb'].shape[0])],
            [fr.preds['comb~'][i, m] for i in range(fr.preds['comb~'].shape[0])],
        ),
    }
    fr05 = load_mgnify(run=cfg['run'], sigma=0.5, heldout=True)
    preds['ESM-MSR σ=0.5'] = (
        'combined (latent)',
        [fr05.preds['comb'][i, m] for i in range(fr05.preds['comb'].shape[0])],
        [fr05.preds['comb~'][i, m] for i in range(fr05.preds['comb~'].shape[0])],
    )
    for mn, (tpl, col, seeds, readout) in BASELINES_CFG.items():
        P = []
        for s in seeds:
            c = _read_chunks(tpl, s, cfg['src'], col)
            P.append(None if c is None else c.reindex(ids).to_numpy(float))
        if not any(p is None for p in P):
            preds[mn] = (readout, P, P)
    # Use ProSST dual-view unmasked where full coverage exists (Rocklin) and for multi-mutant strata on MGnify.
    pm_s = _read_prosst_multi_chunks(cfg['src'], 'comb_pred')
    if pm_s is not None:
        pm_arr = pm_s.reindex(ids).to_numpy(float)
        if cfg['src'] == 'rocklin_denovo' and np.isfinite(pm_arr).mean() >= 0.99:
            preds['ProSST'] = ('Dual-view', [pm_arr], [pm_arr])
        else:
            preds['_ProSST_multi'] = ('Dual-view (multi)', [pm_arr], [pm_arr])
    if cfg['src'] == 'rocklin_denovo':
        me = [_read_me_rocklin(k, ids) for k in (1, 2, 3)]
        if all(x is not None for x in me):
            preds['MutateEverything'] = ('stage 2, every variant', me, me)
    for mn, md in HARNESS_CFG.items():
        h = _read_harness(md, ids)
        if h is not None and np.isfinite(h).any():
            preds[mn] = ('direct', [h], [h])
    e3 = _read_esm3_wt(ids)
    if e3 is not None and np.isfinite(e3).any():
        preds['ESM3-small-open'] = ('WT head (sigma 0)', [e3], [e3])
    return d, preds


def run_all_models_and_mutcount(B: int, workers: int = 8) -> None:
    """18b & 18c: Evaluate all models on 4-part held-out MGnify and Rocklin designs."""
    out = results_dir('mgnify')
    sets_cfg = {
        'MGnify': dict(run='all', src='mgnify'),
        'Rocklin designs': dict(run='rdenovo', src='rocklin_denovo'),
    }
    b_pool = min(B, 200)
    rows_s, rows_p, rows_mc = [], [], []
    mut_strata = [(1, '1 mutation'), (2, '2 mutations'), (3, '3+ mutations')]
    mc_order = M.ordered(['ESM-MSR', 'ESM3-small-open', 'SPURS', 'ThermoMPNN', 'ProteinMPNN', 'ProSST'])

    for name, cfg in sets_cfg.items():
        d, preds = _prepare_heldout_models(cfg)
        code, uniq = pd.factorize(d['code_wt'])
        code = code.astype(np.int64)
        n_uniq = len(uniq)
        y = d['y'].to_numpy(float)
        cnt = np.bincount(code, minlength=n_uniq)
        big = cnt >= MIN_VARIANTS
        n_big = int(big.sum())

        # Per-scaffold mean rho (joint scaffold + seed resampling via boot_domain_seeds)
        scaf_counts_big, seed_w_big = (
            np.random.default_rng(3).multinomial(n_big, np.full(n_big, 1.0 / n_big), size=B).astype(float),
            np.random.default_rng(17).multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0,
        )
        per_mat = {
            mn: np.column_stack([_scaffold_rho_per_seed(code, n_uniq, p, y)[big] for p in P_raw])
            for mn, (_, P_raw, _) in preds.items() if not mn.startswith('_')
        }
        for mn, (readout, _, _) in preds.items():
            if mn.startswith('_'):
                continue
            V = per_mat[mn]
            ok = np.isfinite(V).all(axis=1)
            if not ok.any():
                continue
            V_use = np.where(ok[:, None], V, np.nan)
            _, reps_s = boot_domain_seeds(V_use, B=B, counts=scaf_counts_big, seed_weights=seed_w_big if V.shape[1] == 3 else None)
            lo, hi = np.nanpercentile(reps_s, [2.5, 97.5])
            rows_s.append(dict(
                dataset=name, model=mn, readout=readout,
                rho=float(np.nanmean(V_use.mean(axis=1))), lo=float(lo), hi=float(hi),
                n_scaffolds=int(ok.sum()), n_set=n_big, coverage=float(ok.mean()),
                overlap=(mn in OVERLAP_MODELS),
            ))
            if ok.mean() < 0.999 and not mn.startswith('ESM-MSR'):
                V_msr = np.where(ok[:, None], per_mat['ESM-MSR'], np.nan)
                _, reps_m = boot_domain_seeds(V_msr, B=B, counts=scaf_counts_big, seed_weights=seed_w_big)
                lo2, hi2 = np.nanpercentile(reps_m, [2.5, 97.5])
                rows_s.append(dict(
                    dataset=name, model=f'ESM-MSR, same scaffolds as {mn}', readout='combined (latent)',
                    rho=float(np.nanmean(V_msr.mean(axis=1))), lo=float(lo2), hi=float(hi2),
                    n_scaffolds=int(ok.sum()), n_set=n_big, coverage=float(ok.mean()),
                    overlap=False,
                ))

        # Pooled rho over all measured variants (O(n) batched L3 V-Cache Spearman + joint scaffold & seed resampling)
        rng_p = np.random.default_rng(7)
        scaf_counts_all = rng_p.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq), size=b_pool).astype(float)
        seed_w_pool = rng_p.multinomial(3, np.full(3, 1.0 / 3.0), size=b_pool).astype(float) / 3.0
        full_xr_p: dict[str, list[_FastPooledRank]] = {}
        ok_by_mn: dict[str, np.ndarray] = {}
        for mn, (_readout, _, P_pool) in preds.items():
            if mn.startswith('_'):
                continue
            ok = np.all([np.isfinite(p) for p in P_pool], axis=0)
            if ok.mean() < 0.02:
                continue
            ok_by_mn[mn] = ok
            if ok.all():
                full_xr_p[mn] = [_FastPooledRank(p) for p in P_pool]
        full_pt_p, full_reps_p = (
            _eval_pooled_fast(full_xr_p, _FastPooledRank(y, need_inv=True), len(y), code, scaf_counts_all, seed_w_pool, workers=workers)
            if full_xr_p else ({}, {})
        )
        for mn, (readout, _, P_pool) in preds.items():
            if mn not in ok_by_mn:
                continue
            ok = ok_by_mn[mn]
            if mn in full_pt_p:
                pt = full_pt_p[mn][0]
                lo, hi = map(float, np.nanpercentile(full_reps_p[mn], [2.5, 97.5]))
                msk = None
            else:
                msk = ok
                pt, lo, hi = _pooled_scaffold_boot(P_pool, y, code, n_uniq, scaf_counts_all, seed_w_pool, mask=msk, workers=workers)
            rows_p.append(dict(
                dataset=name, model=mn,
                readout=readout + (', Tobit-linked' if mn.startswith('ESM-MSR') else ''),
                rho=pt, lo=lo, hi=hi, n_variants=int(ok.sum()), n_set=len(y),
                coverage=float(ok.mean()), overlap=(mn in OVERLAP_MODELS),
            ))
            if not mn.startswith('ESM-MSR') and msk is not None:
                pt2, lo2, hi2 = _pooled_scaffold_boot(preds['ESM-MSR'][2], y, code, n_uniq, scaf_counts_all, seed_w_pool, mask=ok, workers=workers)
                rows_p.append(dict(
                    dataset=name, model=f'ESM-MSR, same rows as {mn}', readout='combined (Tobit-linked)',
                    rho=pt2, lo=lo2, hi=hi2, n_variants=int(ok.sum()), n_set=len(y),
                    coverage=float(ok.mean()), overlap=False,
                ))

        # 18c: Pooled rho by mutation count (1, 2, 3+)
        nm = np.clip(d['n_mut'].to_numpy(int), 1, 3)
        rng_mc = np.random.default_rng(11)
        for k_mc, _lab_mc in mut_strata:
            sel = np.flatnonzero(nm == k_mc)
            if len(sel) < 100:
                rows_mc.append(dict(dataset=name, n_mut=k_mc, metric='n', model='', rho=np.nan, lo=np.nan, hi=np.nan, n_variants=len(sel), n_scaffolds=int(d.iloc[sel]['code_wt'].nunique())))
                continue
            c_mc, u_mc = pd.factorize(d['code_wt'].iloc[sel])
            c_mc = c_mc.astype(np.int64)
            y_mc = y[sel]
            sc_mc = rng_mc.multinomial(len(u_mc), np.full(len(u_mc), 1.0 / len(u_mc)), size=b_pool).astype(float)
            sw_mc = rng_mc.multinomial(3, np.full(3, 1.0 / 3.0), size=b_pool).astype(float) / 3.0
            full_xr_mc: dict[str, list[_FastPooledRank]] = {}
            sub_meta: list[tuple[str, list[np.ndarray], np.ndarray]] = []
            for mn in mc_order:
                if mn not in preds:
                    continue
                src_key = '_ProSST_multi' if (mn == 'ProSST' and k_mc >= 2 and '_ProSST_multi' in preds) else mn
                P_sub = [p[sel] for p in preds[src_key][2]]
                ok_sub = np.all([np.isfinite(p) for p in P_sub], axis=0)
                if ok_sub.mean() < 0.02:
                    continue
                sub_meta.append((mn, P_sub, ok_sub))
                if ok_sub.all():
                    full_xr_mc[mn] = [_FastPooledRank(p) for p in P_sub]
            full_pt_mc, full_reps_mc = (
                _eval_pooled_fast(full_xr_mc, _FastPooledRank(y_mc, need_inv=True), len(y_mc), c_mc, sc_mc, sw_mc, workers=workers)
                if full_xr_mc else ({}, {})
            )
            for mn, P_sub, ok_sub in sub_meta:
                if mn in full_pt_mc:
                    pt_m = full_pt_mc[mn][0]
                    lo_m, hi_m = map(float, np.nanpercentile(full_reps_mc[mn], [2.5, 97.5]))
                else:
                    pt_m, lo_m, hi_m = _pooled_scaffold_boot(P_sub, y_mc, c_mc, len(u_mc), sc_mc, sw_mc, mask=ok_sub, workers=workers)
                rows_mc.append(dict(
                    dataset=name, n_mut=k_mc, metric='pooled', model=mn,
                    rho=pt_m, lo=lo_m, hi=hi_m, n_variants=len(sel), n_scaffolds=len(u_mc),
                ))

    atomic_to_csv(pd.DataFrame(rows_s), out / 'mgnify_models_per_scaffold.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(rows_p), out / 'mgnify_models_pooled.csv', index=False, float_format='%.4f')
    df_mc = pd.DataFrame(rows_mc)
    atomic_to_csv(df_mc, out / 'mgnify_by_mutation_count.csv', index=False, float_format='%.4f')

    # Draw 18c figure
    scored = df_mc[df_mc['metric'] == 'pooled']
    ks_by_ds = {ds: [(k, lab) for k, lab in mut_strata if ((scored['dataset'] == ds) & (scored['n_mut'] == k)).any()] for ds in sets_cfg}
    fig_mc, axes_mc = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True, gridspec_kw=dict(width_ratios=[len(ks_by_ds[d]) + 0.25 for d in sets_cfg]))
    mc_legend = {
        'ESM-MSR': 'ESM-MSR Dual-view',
        'ESM3-small-open': 'ESM3-small-open (σ = 0, WT-view)',
        'SPURS': 'SPURS-multi',
        'ThermoMPNN': 'ThermoMPNN-D',
        'ProteinMPNN': 'ProteinMPNN',
        'ProSST': 'ProSST (WT / Dual-view)',
    }
    for c_i, ds in enumerate(sets_cfg):
        ax = axes_mc[c_i]
        R_ds = scored[scored['dataset'] == ds]
        tab, ci_m, labels = {}, {}, []
        for k_v, lab in ks_by_ds[ds]:
            m_df = R_ds[R_ds['n_mut'] == k_v]
            labels.append(f'{lab}\n({int(m_df["n_variants"].iloc[0]):,} variants)')
            for x in m_df.itertuples():
                tab[(k_v, x.model)] = x.rho
                ci_m[(k_v, x.model)] = (x.lo, x.hi)
        style.dot_panel(
            ax, [k_v for k_v, _ in ks_by_ds[ds]], mc_order, tab, ci_m,
            highlight=['ESM-MSR'], hollow={(k_v, 'ESM3-small-open') for k_v, _ in ks_by_ds[ds]},
            ylabel=r'Spearman $\rho$ (pooled over variants)' if c_i == 0 else None,
            xlabels=labels, rot=0, size=6.8, spread=0.78, ylim=(0.0, 0.88),
        )
        ax.text(-0.02 if c_i else -0.12, 1.02, 'ab'[c_i], transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')
        ax.set_title(f'{ds} (held-out)', fontsize=9.5, loc='right', color='#444444')
        if c_i == 0:
            hs = [Line2D([0], [0], label=mc_legend[m], **style.marker_kwargs(m, size=6.8, hollow=(m == 'ESM3-small-open'))) for m in mc_order]
            ax.legend(handles=hs, loc='upper left', ncol=2, fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.0, handletextpad=0.3, borderpad=0.4)
    fig_mc.tight_layout(w_pad=1.0)
    style.save(fig_mc, out / 'mgnify_by_mutation_count')


# ----------------------------------------------------------------------------------------------
# 3. Rocklin de novo detailed stats & epistasis (scoring/mgnify_rdenovo_stats.py, held-out)
# ----------------------------------------------------------------------------------------------
_V2_LABEL_CACHE: pd.DataFrame | None = None


def _load_v2_label_map() -> pd.DataFrame:
    global _V2_LABEL_CACHE
    if _V2_LABEL_CACHE is not None:
        return _V2_LABEL_CACHE
    want = ['uid', 'dG_wt', 'dG_ML', 'ddG', 'ddG_ML', 'dddG_ML', 'ddG_additive_ML', 'cens', 'in_range_1_5', 'has_all_singles']
    hdr = pd.read_csv(MGNIFY_TABLE, nrows=0).columns
    cols = [c for c in want if c in hdr]
    t = pd.read_csv(MGNIFY_TABLE, usecols=cols, low_memory=False).set_index('uid')
    if 'cens' in t.columns:
        t['cens'] = legacy_cens(t['cens'])
    _V2_LABEL_CACHE = t
    return t


def _apply_v2(df: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keep = out['uid'].isin(v2.index)
    for c in v2.columns:
        mapped = out['uid'].map(v2[c])
        out[c] = mapped.where(keep, out[c]) if c in out.columns else mapped
    return out


def _nan_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(spearmanr(x[ok], y[ok])[0])


def _nan_pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(pearsonr(x[ok], y[ok])[0])


def run_rocklin_rdenovo_stats(B: int, workers: int = 8) -> None:
    """Triplicate Rocklin de novo pooled, within-scaffold, and epistasis statistics (4-part held-out)."""
    out = results_dir('rocklin_designs')
    root = FINAL_MGNIFY / 'FINAL_p0'
    if not all((root / f's{s}' / 'rdenovo' / 'predictions.csv.gz').exists() for s in (1, 2, 3)):
        return

    v2 = _load_v2_label_map()
    bad_rk = excluded('rocklin_denovo')
    link_fn = tobit_link(**TOBIT)
    ds = []
    for s in (1, 2, 3):
        d = pd.read_csv(root / f's{s}' / 'rdenovo' / 'predictions.csv.gz', low_memory=False)
        d = _apply_v2(d, v2).sort_values('uid', ignore_index=True)
        d = d[~d['code_wt'].astype(str).isin(bad_rk)].reset_index(drop=True)
        dg = d['dG_wt'].astype(float)
        for k in ('wt_lora_pred', 'mt_lora_pred', 'combined_pred'):
            d[k + '_linked'] = link_fn(dg + d[k]) - dg
        d['epi_msr'] = d['combined_pred'] - d['wt_lora_pred_additive']
        d['resid_add'] = d['ddG_ML'] - d['wt_lora_pred_additive']
        d['in_range'] = d['in_range_1_5'].astype(bool)
        ds.append(d)

    b0 = ds[0]
    sing = b0[b0['n_mut'] == 1].set_index(['code_wt', 'mut_type'])['cens']
    dbl = b0[b0['n_mut'] == 2]
    ok_unc = np.array([all(sing.get((r.code_wt, x), np.nan) == 0 for x in str(r.mut_type).split(':')) for r in dbl.itertuples()])
    singles_uncens = np.zeros(len(b0), dtype=bool)
    singles_uncens[dbl.index[ok_unc]] = True

    b_fast = min(B, 200)
    ok = b0['ddG_ML'].notna().to_numpy()
    nm = b0['n_mut'].to_numpy()
    subsets = {'all': nm >= 1, 'n=1': nm == 1, 'n=2': nm == 2, 'n>=3': nm >= 3, 'multi(n>=2)': nm >= 2}
    regimes = {'in_range': b0['in_range'].to_numpy(), 'all_rows': np.ones(len(b0), bool)}
    heads = {'wt head': 'wt_lora_pred', 'mt head': 'mt_lora_pred', 'combined': 'combined_pred'}

    rows = []
    for rn, rm in regimes.items():
        for sn, sm in subsets.items():
            m = ok & rm & sm
            if m.sum() < 10:
                continue
            code, uniq = pd.factorize(b0.loc[m, 'code_wt'])
            code = code.astype(np.int64)
            n_u = len(uniq)
            y_m = b0.loc[m, 'ddG_ML'].to_numpy(float)
            cnt_u = np.bincount(code, minlength=n_u)
            big_u = cnt_u >= 10
            n_big = int(big_u.sum())
            rng = np.random.default_rng(0)
            scaf_counts_p = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u), size=b_fast).astype(float)
            seed_w_p = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=b_fast).astype(float) / 3.0
            scaf_counts_w = rng.multinomial(n_big, np.full(n_big, 1.0 / n_big), size=B).astype(float) if n_big >= 2 else None
            seed_w_w = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0 if n_big >= 2 else None

            cols_eval = [(h, 'latent', c) for h, c in heads.items()] + [(h, 'linked', c + '_linked') for h, c in heads.items()]
            if sn in ('n=2', 'n>=3', 'multi(n>=2)'):
                cols_eval.append(('additive composition (WT singles)', 'latent', 'wt_lora_pred_additive'))

            ry_m = _FastPooledRank(y_m, need_inv=True)
            p_arr_map = {col: [d.loc[m, col].to_numpy(float) for d in ds] for _, _, col in cols_eval}
            xr_eval = {col: [_FastPooledRank(np.where(np.isfinite(p), p, 0.0)) for p in P_arr] for col, P_arr in p_arr_map.items()}
            pt_map, bs_map = _eval_pooled_fast(xr_eval, ry_m, len(y_m), code.astype(np.int32), scaf_counts_p, seed_w_p, workers=workers)

            for h, form, col in cols_eval:
                P_arr = p_arr_map[col]
                pt_val, pt_seeds = pt_map[col]
                bs_p = bs_map[col]
                rows.append(dict(
                    kind='pooled', regime=rn, subset=sn, head=h, form=form,
                    value=pt_val, sd_seeds=seed_sd(pt_seeds),
                    lo=float(np.nanpercentile(bs_p, 2.5)), hi=float(np.nanpercentile(bs_p, 97.5)),
                    n=int(m.sum()), n_scaffolds=n_u,
                ))
                # Within-scaffold
                if n_big >= 2:
                    V_w = np.column_stack([_scaffold_rho_per_seed(code, n_u, p, y_m)[big_u] for p in P_arr])
                    pt_w = np.nanmean(V_w, axis=0)
                    _, reps_w = boot_domain_seeds(V_w, B=B, counts=scaf_counts_w, seed_weights=seed_w_w)
                    rows.append(dict(
                        kind='within-scaffold', regime=rn, subset=sn, head=h, form=form,
                        value=float(np.nanmean(pt_w)), sd_seeds=seed_sd(pt_w),
                        lo=float(np.nanpercentile(reps_w, 2.5)), hi=float(np.nanpercentile(reps_w, 97.5)),
                        n=n_big, n_scaffolds=n_u,
                    ))

    atomic_to_csv(pd.DataFrame(rows), out / 'stats.csv', index=False, float_format='%.4f')

    # Epistasis table on held-out Rocklin designs
    er = []
    m_epi = (nm == 2) & b0['has_all_singles'].fillna(False).astype(bool).to_numpy() & b0['dddG_ML'].notna().to_numpy() & singles_uncens

    def _boot_small_cell(mask_c: np.ndarray, fn_seed, rng_seed: int) -> dict | None:
        idx = np.flatnonzero(mask_c)
        if len(idx) < 10:
            return None
        sub = [d.iloc[idx].reset_index(drop=True) for d in ds]
        pt_v = np.array([fn_seed(f) for f in sub], dtype=float)
        u_c, inv_c = np.unique(sub[0]['code_wt'].to_numpy(), return_inverse=True)
        order_c = np.argsort(inv_c, kind='stable')
        bounds_c = np.searchsorted(inv_c[order_c], np.arange(len(u_c) + 1))
        rng_c = np.random.default_rng(rng_seed)
        bs_c = []
        for _ in range(b_fast):
            pick = rng_c.integers(0, len(u_c), len(u_c))
            j_idx = np.concatenate([order_c[bounds_c[i]:bounds_c[i + 1]] for i in pick])
            sw = rng_c.multinomial(3, np.full(3, 1.0 / 3.0)).astype(float) / 3.0
            vals_s = np.array([fn_seed(f.iloc[j_idx]) for f in sub], dtype=float)
            bs_c.append(float(np.dot(sw, vals_s)))
        return dict(
            value=float(np.nanmean(pt_v)), sd_seeds=seed_sd(pt_v),
            lo=float(np.nanpercentile(bs_c, 2.5)), hi=float(np.nanpercentile(bs_c, 97.5)),
            n=len(idx), n_scaffolds=len(u_c),
        )

    for head in ('epi_msr', 'combined_dddg_pred', 'mt_lora_dddg_pred'):
        if head not in b0.columns:
            continue
        mc = m_epi & b0[head].notna().to_numpy()
        for stat, fn in (
            ('spearman', lambda f, h=head: _nan_spearman(f[h].to_numpy(float), f['dddG_ML'].to_numpy(float))),
            ('pearson', lambda f, h=head: _nan_pearson(f[h].to_numpy(float), f['dddG_ML'].to_numpy(float))),
            ('sign agreement', lambda f, h=head: float((np.sign(f[h].to_numpy(float)) == np.sign(f['dddG_ML'].to_numpy(float))).mean())),
            ('OLS slope (measured on predicted)', lambda f, h=head: float(np.polyfit(f[h].to_numpy(float), f['dddG_ML'].to_numpy(float), 1)[0])),
        ):
            r_c = _boot_small_cell(mc, fn, rng_seed=1)
            if r_c:
                er.append(dict(test='measured dddG (doubles, singles assayed)', readout=head, stat=stat, **r_c))

    for sn, sm in (('multi(n>=2)', nm >= 2), ('n=2', nm == 2), ('n>=3', nm >= 3)):
        mm = sm & b0['in_range'].to_numpy() & ok & np.isfinite(b0['resid_add'].to_numpy(float))
        if mm.sum() < 10:
            continue
        code_e, uniq_e = pd.factorize(b0.loc[mm, 'code_wt'])
        code_e = code_e.astype(np.int64)
        n_ue = len(uniq_e)
        cnt_ue = np.bincount(code_e, minlength=n_ue)
        big_ue = cnt_ue >= 10
        rng_e = np.random.default_rng(2)
        sc_ep = rng_e.multinomial(n_ue, np.full(n_ue, 1.0 / n_ue), size=b_fast).astype(float)
        sw_ep = rng_e.multinomial(3, np.full(3, 1.0 / 3.0), size=b_fast).astype(float) / 3.0
        # Pooled model-referenced non-additivity (note resid_add depends on seed!)
        pts_p = np.array([_nan_spearman(d.loc[mm, 'epi_msr'].to_numpy(float), d.loc[mm, 'resid_add'].to_numpy(float)) for d in ds])
        n_mm = int(mm.sum())
        code_e32 = code_e.astype(np.int32)
        seed_reps_ep = np.vstack([
            _eval_pooled_fast(
                {'s': [_FastPooledRank(d.loc[mm, 'epi_msr'].to_numpy(float))]},
                _FastPooledRank(d.loc[mm, 'resid_add'].to_numpy(float), need_inv=True),
                n_mm, code_e32, sc_ep, sw_ep, workers=workers,
            )[1]['s']
            for d in ds
        ])
        bs_ep = (sw_ep.T * seed_reps_ep).sum(axis=0)
        er.append(dict(
            test=f'model-referenced non-additivity, {sn}, in range', readout='epi_msr vs ddG - additive', stat='pooled spearman',
            value=float(np.nanmean(pts_p)), sd_seeds=seed_sd(pts_p),
            lo=float(np.nanpercentile(bs_ep, 2.5)), hi=float(np.nanpercentile(bs_ep, 97.5)),
            n=int(mm.sum()), n_scaffolds=n_ue,
        ))
        if big_ue.sum() >= 2:
            V_we = np.column_stack([
                _scaffold_rho_per_seed(code_e, n_ue, d.loc[mm, 'epi_msr'].to_numpy(float), d.loc[mm, 'resid_add'].to_numpy(float))[big_ue]
                for d in ds
            ])
            pt_we = np.nanmean(V_we, axis=0)
            sc_we = rng_e.multinomial(int(big_ue.sum()), np.full(int(big_ue.sum()), 1.0 / int(big_ue.sum())), size=B).astype(float)
            sw_we = rng_e.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0
            _, reps_we = boot_domain_seeds(V_we, B=B, counts=sc_we, seed_weights=sw_we)
            er.append(dict(
                test=f'model-referenced non-additivity, {sn}, in range', readout='epi_msr vs ddG - additive', stat='within-scaffold spearman',
                value=float(np.nanmean(pt_we)), sd_seeds=seed_sd(pt_we),
                lo=float(np.nanpercentile(reps_we, 2.5)), hi=float(np.nanpercentile(reps_we, 97.5)),
                n=int(big_ue.sum()), n_scaffolds=n_ue,
            ))

    atomic_to_csv(pd.DataFrame(er), out / 'epistasis.csv', index=False, float_format='%.4f')


# ----------------------------------------------------------------------------------------------
# 4. Beyond-global epistasis on held-out MGnify + Rocklin designs (scoring/mgnify_epistasis_beyond.py)
# ----------------------------------------------------------------------------------------------
_MUT_RE = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')
_BEYOND_TABLES: dict = {}
_BEYOND_GROUPS = ('MGnify', 'Rocklin designs', 'Together')


def _parse_mut_key(m_str: str, code: str):
    out = []
    for part in str(m_str).split(':'):
        g = _MUT_RE.match(part.strip())
        if not g:
            return None
        out.append((code, int(g.group(2)), g.group(3)))
    return tuple(out)


def _sub_epi_table(t, idx: np.ndarray):
    r = copy.copy(t)
    n0 = t.n
    for name, v in vars(t).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    r._mats = None
    return r


def _beyond_draw_worker(task) -> tuple:
    g, kind, i = task
    rng = np.random.default_rng([7, _BEYOND_GROUPS.index(g), int(kind == 'scaffold'), i])
    t0 = _BEYOND_TABLES[(g, 1)]
    if kind == 'mutation':
        idx = rng.integers(0, t0.n, t0.n)
    else:
        libs = np.unique(t0.lib)
        order = np.argsort(t0.lib, kind='stable')
        bounds = np.searchsorted(t0.lib[order], np.r_[libs, libs.max() + 1])
        pick = rng.integers(0, len(libs), len(libs))
        idx = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
    sw = rng.multinomial(3, np.full(3, 1.0 / 3.0)).astype(float) / 3.0
    v = np.array([
        (E._rho(*E.beyond_global(_sub_epi_table(_BEYOND_TABLES[(g, s)], idx), 'epi')[::-1]),
         E.naive_rho(_sub_epi_table(_BEYOND_TABLES[(g, s)], idx), 'epi'))
        for s in (1, 2, 3)
    ])
    return (g, kind, i, float(np.dot(sw, v[:, 0])), float(np.dot(sw, v[:, 1])))


def run_beyond_epistasis(B: int, workers: int = 4) -> None:
    """Beyond-global saturation-subtracted epistasis on 4-part held-out MGnify + Rocklin designs."""
    out = results_dir('mgnify_epistasis')
    root = FINAL_MGNIFY / 'FINAL_p0'
    if not all((root / f's{s}' / sub / 'predictions.csv.gz').exists() for s in (1, 2, 3) for sub in ('all', 'rdenovo')):
        return

    v2 = _load_v2_label_map()
    bad_mg = excluded('mgnify')
    bad_rk = excluded('rocklin_denovo')
    want_cols = {'uid', 'source', 'code_wt', 'mut_type', 'n_mut', 'ddG_ML', 'dddG_ML', 'dG_wt', 'cens', 'backbone_mutation', 'wt_lora_pred', 'combined_pred'}

    per = []
    for s in (1, 2, 3):
        parts = []
        for sub, src_default, bad_set in (('all', 'mgnify', bad_mg), ('rdenovo', 'rocklin_denovo', bad_rk)):
            d = pd.read_csv(root / f's{s}' / sub / 'predictions.csv.gz', usecols=lambda c: c in want_cols, low_memory=False)
            d = _apply_v2(d, v2)
            if 'source' not in d.columns:
                d['source'] = src_default
            d = d[~d['code_wt'].astype(str).isin(bad_set)]
            parts.append(d)
        per.append(pd.concat(parts, ignore_index=True).sort_values('uid', ignore_index=True))

    b0 = per[0]
    bb = b0['backbone_mutation'] if 'backbone_mutation' in b0.columns else pd.Series(False, index=b0.index)
    wtbb = ~bb.map(lambda v: str(v).lower() in ('true', '1', '1.0')).to_numpy()
    use = np.isfinite(b0['ddG_ML'].to_numpy(float)) & (b0['cens'].to_numpy() == 0) & wtbb
    is_mg = (b0['source'] == 'mgnify').to_numpy()

    _BEYOND_TABLES.clear()
    meta = {}
    for g, gm in (('MGnify', is_mg), ('Rocklin designs', ~is_mg), ('Together', np.ones(len(b0), bool))):
        idx = np.flatnonzero(use & gm & b0['n_mut'].isin([1, 2]).to_numpy())
        keys = [_parse_mut_key(m, c) for m, c in zip(b0['mut_type'].to_numpy()[idx], b0['code_wt'].astype(str).to_numpy()[idx])]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        base = dict(
            mut_key=keys, subset_type=st,
            ddG=b0['ddG_ML'].to_numpy(float)[idx],
            dddG=b0['dddG_ML'].to_numpy(float)[idx],
            dG_wt=b0['dG_wt'].to_numpy(float)[idx],
        )
        for j, s in enumerate((1, 2, 3)):
            wt = per[j]['wt_lora_pred'].to_numpy(float)[idx]
            comb = per[j]['combined_pred'].to_numpy(float)[idx]
            _BEYOND_TABLES[(g, s)] = E.Table(dict(base, wt=wt, mt=comb, comb=comb), None, heads=('epi', 'add'))
        t1 = _BEYOND_TABLES[(g, 1)]
        meta[g] = dict(n_doubles=int(t1.n), n_scaffolds=int(len(np.unique(t1.lib))))

    pts = {
        g: np.array([(E._rho(*E.beyond_global(_BEYOND_TABLES[(g, s)], 'epi')[::-1]), E.naive_rho(_BEYOND_TABLES[(g, s)], 'epi')) for s in (1, 2, 3)])
        for g in _BEYOND_GROUPS
    }
    b_scaf = min(B, 200)
    tasks = [(g, 'mutation', i) for g in _BEYOND_GROUPS for i in range(B)] + [(g, 'scaffold', i) for g in _BEYOND_GROUPS for i in range(b_scaf)]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_beyond_draw_worker, tasks, chunksize=4)
    bd = pd.DataFrame(res, columns=['group', 'kind', 'draw', 'beyond_rho', 'naive_rho'])

    rows = []
    for g in _BEYOND_GROUPS:
        for mi, metric in enumerate(('beyond_rho', 'naive_rho')):
            r = dict(
                group=g, metric=metric, **meta[g],
                seed_values=' '.join(f'{x:.4f}' for x in pts[g][:, mi]),
                value=float(pts[g][:, mi].mean()),
                sd_seeds=seed_sd(pts[g][:, mi]),
            )
            for kind, tag in (('mutation', 'mut'), ('scaffold', 'scaf')):
                v = bd[(bd['group'] == g) & (bd['kind'] == kind)][metric].to_numpy()
                r.update({
                    f'lo_{tag}': float(np.percentile(v, 2.5)),
                    f'hi_{tag}': float(np.percentile(v, 97.5)),
                    f'B_{tag}': len(v),
                    f'p_gt0_{tag}': float((1 + (v <= 0).sum()) / (len(v) + 1)),
                    f'frac_le0_{tag}': float((v <= 0).mean()),
                })
            rows.append(r)
    S_b = pd.DataFrame(rows)
    atomic_to_csv(S_b, out / 'summary.csv', index=False, float_format='%.5g')

    t_tog = _BEYOND_TABLES[('Together', 1)]
    rm = E.beyond_global(t_tog, 'epi')[0]
    rp = np.mean([E.beyond_global(_BEYOND_TABLES[('Together', s)], 'epi')[1] for s in (1, 2, 3)], axis=0)

    col_g = {'MGnify': '#1baf7a', 'Rocklin designs': '#eb6834', 'Together': '#2a78d6'}
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw=dict(width_ratios=[1.05, 1.0, 1.05]))
    ok_r = np.isfinite(rm) & np.isfinite(rp)
    xl = np.percentile(rm[ok_r], [0.5, 99.5])
    yl = np.percentile(rp[ok_r], [0.5, 99.5])
    hb = ax[0].hexbin(rm[ok_r], rp[ok_r], gridsize=55, extent=(*xl, *yl), bins='log', cmap='Blues', mincnt=1, linewidths=0)
    ax[0].axhline(0, color=style.GRID, lw=0.8, zorder=0)
    ax[0].axvline(0, color=style.GRID, lw=0.8, zorder=0)
    sT = S_b[(S_b['group'] == 'Together') & (S_b['metric'] == 'beyond_rho')].iloc[0]
    ax[0].set_xlabel('measured residual r (kcal/mol)')
    ax[0].set_ylabel('predicted, r′ (seed mean; kcal/mol)')
    ax[0].set_title(f'A  {int(sT.n_doubles):,} held-out doubles, MGnify + Rocklin', loc='left', fontsize=9.5)
    ax[0].text(0.03, 0.96, f'beyond ρ = {sT.value:.3f}', transform=ax[0].transAxes, va='top', fontsize=9.5)
    cb = fig.colorbar(hb, ax=ax[0], fraction=0.045, pad=0.02)
    cb.set_label('doubles per cell (log)')
    cb.outline.set_visible(False)

    ypos = {(g, m): k + (0.17 if m == 'naive_rho' else -0.17) for k, g in enumerate(_BEYOND_GROUPS[::-1]) for m in ('beyond_rho', 'naive_rho')}
    for _, r in S_b.iterrows():
        y_p = ypos[(r.group, r.metric)]
        ax[1].plot([r.lo_scaf, r.hi_scaf], [y_p, y_p], color=col_g[r.group], lw=1.0, alpha=0.55, solid_capstyle='butt')
        ax[1].plot([r.lo_mut, r.hi_mut], [y_p, y_p], color=col_g[r.group], lw=3.2, solid_capstyle='butt')
        ax[1].plot([r.value], [y_p], 'o' if r.metric == 'beyond_rho' else 's', ms=7, mfc=col_g[r.group] if r.metric == 'beyond_rho' else 'white', mec=col_g[r.group], mew=1.6, zorder=3)
        ax[1].text(max(r.hi_scaf, r.hi_mut) + 0.006, y_p, f'{r.value:.3f}', va='center', fontsize=8.5)
    ax[1].axvline(0, color='#555555', lw=0.8)
    ax[1].set_yticks(range(len(_BEYOND_GROUPS)))
    ax[1].set_yticklabels([f'{g}\n(n = {meta[g]["n_doubles"]:,})' for g in _BEYOND_GROUPS[::-1]])
    ax[1].set_xlabel('Spearman ρ against measured epistasis')
    ax[1].set_title('B  beyond ρ (●) and naive ρ (□)', loc='left', fontsize=9.5)
    ax[1].text(0.0, -0.22, 'thick: 95% mutation-level bootstrap; thin: 95% scaffold-level (seeds resampled)', transform=ax[1].transAxes, fontsize=7.8, color='#555555')
    ax[1].grid(axis='x', color=style.GRID, lw=0.6)

    for g in _BEYOND_GROUPS:
        v = bd[(bd['group'] == g) & (bd['kind'] == 'mutation')]['beyond_rho'].to_numpy()
        r = S_b[(S_b['group'] == g) & (S_b['metric'] == 'beyond_rho')].iloc[0]
        lab = f'{g}: p < 0.001' if (v <= 0).sum() == 0 and len(v) >= 999 else (f'{g}: p < {1 / (len(v) + 1):.3g}' if (v <= 0).sum() == 0 else f'{g}: p = {r.p_gt0_mut:.3g}')
        ax[2].hist(v, bins=40, histtype='stepfilled' if g == 'Together' else 'step', color=col_g[g], alpha=0.35 if g == 'Together' else 1.0, lw=1.6, label=lab)
    ax[2].axvline(0, color='#0b0b0b', lw=1.0)
    ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.25)
    ax[2].set_xlabel('beyond ρ of the resample (seed mean)')
    ax[2].set_ylabel('bootstrap draws')
    ax[2].set_title(f'C  mutation-level bootstrap (B = {B:,}), H0: beyond ρ ≤ 0', loc='left', fontsize=9.5)
    ax[2].legend(frameon=False, fontsize=8.0, loc='upper right')
    fig.tight_layout()
    style.save(fig, out / 'beyond_rho_epistasis')


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    ap.add_argument('--workers', type=int, default=8, help='Worker processes for parallel scaffold bootstraps')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    import time
    for lbl, fn in (
        ('esm_msr_strata', lambda: run_esm_msr_strata(B=B, workers=args.workers)),
        ('all_models_and_mutcount', lambda: run_all_models_and_mutcount(B=B, workers=args.workers)),
        ('rocklin_rdenovo_stats', lambda: run_rocklin_rdenovo_stats(B=B, workers=args.workers)),
        ('beyond_epistasis', lambda: run_beyond_epistasis(B=B, workers=args.workers)),
    ):
        t0 = time.perf_counter()
        fn()
        print(f'  [p04] {lbl:25s} B={B:4d}  {time.perf_counter() - t0:5.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
