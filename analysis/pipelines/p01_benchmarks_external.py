#!/usr/bin/env python3
"""Pipeline p01: Core Spearman bootstrap, 10 external thermostability sets, and DMS & Domainome benchmarks.

Consolidates:
  - 01_core_rho.py       -> results/00_core/rho_bootstrap.csv
  - 02_thermostability.py -> results/02_thermostability/{thermostability_rho.csv, thermo_external_rho, thermo_overview, thermo_interval_types}
  - 04_dms_domainome.py  -> results/04_dms_domainome/{dms_vs_retrains, dms_sigma_vs_zero_shot, dms_sigma_sweep}

Key methodological fixes vs legacy scripts:
  - Uniform bootstrap count B across protein, site, and mutation resampling (removes silent B // 2 halving on mutation).
  - Never mutates registry.MODELS in place; uses local label overrides for training-overlap stars.
  - Fixes dms_sigma_sweep from a half-empty 1x8 grid to a clean 1x4 grid over the 4 evaluated datasets.
  - Drops byte-identical duplicate dms_external_rho.{csv,png,pdf} and dead cache/boot_{protein,site,mutation}.pkl dumps.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from msr_analysis import models as R
from msr_analysis import style
from msr_analysis.boot import DatasetBoot, _make_seed_sampler, bootstrap_rho, ci, global_cluster_index
from msr_analysis.common import leak_table, out_dir
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import EXT_ORDER, load_mutation_boot, model_key_for_dataset, seed_frame
from msr_analysis.paths import CACHE, atomic_pickle_dump, atomic_to_csv
from msr_analysis.registry import DMS3, EXT12, LABEL, MODELS, PER_PROTEIN, TSU_CAPPED

from . import base_parser, resolve_b

STAGE = 'benchmarks_external'
DESCRIPTION = 'Task spines, core Spearman bootstrap, 12 external thermostability sets, DMS & Domainome'

EXT10 = ['s571', 's783', 's2648', 's8754', 's461', 'ssym', 'q3421', 'k2369', 'ptmul', 'ptmuld']
OVERVIEW_MODELS = [m.name for m in R.ROSTER if not m.sigma]
CORE_KEYS = tuple(dict.fromkeys(
    [k for m in R.ROSTER for k in (m.single, m.multi) if k] + ['ifum_esmfold']
))
SAW_TEST_DATA = ('GeoDDG-3D', 'DDGemb', 'DDMut')
DMS_DS = DMS3 + ['domainome']
DMS_COLS = DMS3 + ['mean of 3 DMS', 'domainome']

_PP_MUT_STATE: dict = {}


def _keys_for(fr: Frame) -> list[str]:
    return [k for k in CORE_KEYS if k in fr.preds]


def _pp_mut_worker(p_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact within-protein mutation bootstrap Spearman correlations in L2-resident protein blocks."""
    prot_rows = _PP_MUT_STATE['prot_rows']
    y_all = _PP_MUT_STATE['y_all']
    u_preds = _PP_MUT_STATE['u_preds']
    u_valid = _PP_MUT_STATE['u_valid']
    B = _PP_MUT_STATE['B']
    seed = _PP_MUT_STATE['seed']
    n_u = len(u_preds)
    sum_r = np.zeros((n_u, B), dtype=np.float64)
    cnt_r = np.zeros((n_u, B), dtype=np.int32)
    C = 125

    for p in p_list:
        rows = prot_rows[p]
        n_p = len(rows)
        if n_p == 0:
            continue
        yp = y_all[rows]
        rng_p = np.random.default_rng([seed, 991, p])
        W_p = np.ascontiguousarray(rng_p.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=B).T, dtype=np.float64)

        by_m: dict[bytes, tuple[np.ndarray, list[tuple[int, np.ndarray]]]] = {}
        for u_idx in range(n_u):
            ok = u_valid[u_idx][rows]
            if not ok.any():
                continue
            by_m.setdefault(ok.tobytes(), (ok, []))[1].append((u_idx, u_preds[u_idx][rows]))

        for _ok_b, (ok, s_items) in by_m.items():
            all_ok = bool(ok.all())
            ym = yp if all_ok else yp[ok]
            Wm = W_p if all_ok else W_p[ok]
            nm = len(ym)
            oy = np.argsort(ym, kind='mergesort')
            ys = ym[oy]
            nt_y = np.ones(nm, dtype=bool)
            if nm > 1:
                nt_y[1:] = ys[1:] != ys[:-1]
            ht_y = not bool(nt_y.all())
            if ht_y:
                ts_y = np.flatnonzero(nt_y)
                te_y = np.r_[ts_y[1:] - 1, nm - 1]
                tid_y = np.cumsum(nt_y) - 1
            inv_y = np.empty(nm, dtype=np.int32)
            inv_y[oy] = np.arange(nm, dtype=np.int32)

            x_pre = []
            for u_idx, xp in s_items:
                xm = xp if all_ok else xp[ok]
                ox = np.argsort(xm, kind='mergesort')
                xs = xm[ox]
                nt_x = np.ones(nm, dtype=bool)
                if nm > 1:
                    nt_x[1:] = xs[1:] != xs[:-1]
                ht_x = not bool(nt_x.all())
                ts_x = np.flatnonzero(nt_x) if ht_x else None
                te_x = np.r_[ts_x[1:] - 1, nm - 1] if ht_x else None
                tid_x = (np.cumsum(nt_x) - 1) if ht_x else None
                x_pre.append((u_idx, inv_y[ox], ht_x, ts_x, te_x, tid_x))

            for c0 in range(0, B, C):
                c1 = min(B, c0 + C)
                Wy = Wm[oy, c0:c1]
                csy = np.cumsum(Wy, axis=0)
                if not ht_y:
                    ry = csy - Wy * 0.5 + 0.5
                else:
                    by = csy[ts_y] - Wy[ts_y]
                    ry = (by + (csy[te_y] - by + 1.0) * 0.5)[tid_y]
                sw = csy[-1]
                wry = Wy * ry
                with np.errstate(invalid='ignore', divide='ignore'):
                    my = wry.sum(axis=0) / sw
                    vy = (wry * ry).sum(axis=0) / sw - my * my
                bady = (sw < 3.0) | (vy <= 0.0) | ~np.isfinite(vy)
                for u_idx, perm, ht_x, ts_x, te_x, tid_x in x_pre:
                    Wx = Wy[perm]
                    csx = np.cumsum(Wx, axis=0)
                    if not ht_x:
                        rx = csx - Wx * 0.5 + 0.5
                    else:
                        bx = csx[ts_x] - Wx[ts_x]
                        rx = (bx + (csx[te_x] - bx + 1.0) * 0.5)[tid_x]
                    wrx = Wx * rx
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mx = wrx.sum(axis=0) / sw
                        vx = (wrx * rx).sum(axis=0) / sw - mx * mx
                        cxy = (rx * wry[perm]).sum(axis=0) / sw - mx * my
                        r = cxy / np.sqrt(vx * vy)
                    ok_r = ~bady & (vx > 0.0) & np.isfinite(r)
                    sum_r[u_idx, c0:c1] += np.where(ok_r, r, 0.0)
                    cnt_r[u_idx, c0:c1] += ok_r
    return sum_r, cnt_r


def _fast_per_protein_mutation_boot(ds: str, fr: Frame, keys: list[str], B: int, seed: int = 11, procs: int = 8):
    """Exact within-protein mutation + seed bootstrap for a single ``PER_PROTEIN`` dataset."""
    b = DatasetBoot(fr, keys, per_protein=True)
    point = {ds: {k: b.point(k) for k in b.keys}}
    reps_ds = {k: np.full(B, np.nan) for k in b.keys}
    if B <= 0 or not b.keys:
        return point, {ds: reps_ds}

    base = np.flatnonzero(b.base)
    g_b = b.groups_all[base]
    ord_b = np.argsort(g_b, kind='stable')
    gs = g_b[ord_b]
    starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    ends = np.r_[starts[1:], len(gs)]
    base_sorted = base[ord_b]
    prot_rows = [base_sorted[starts[p]:ends[p]] for p in range(len(starts))]

    u_map: dict[tuple[bytes, bytes], int] = {}
    u_preds: list[np.ndarray] = []
    u_valid: list[np.ndarray] = []
    k_seed_u: dict[str, list[int]] = {}
    for k in b.keys:
        vb = b.valid[k].tobytes()
        uids = []
        for si in range(fr.preds[k].shape[0]):
            arr = np.asarray(fr.preds[k][si], dtype=np.float64)
            sig = (vb, arr.tobytes())
            uid = u_map.get(sig)
            if uid is None:
                uid = len(u_preds)
                u_map[sig] = uid
                u_preds.append(arr)
                u_valid.append(b.valid[k])
            uids.append(uid)
        k_seed_u[k] = uids

    n_workers = min(max(1, procs), len(prot_rows))
    _PP_MUT_STATE.clear()
    _PP_MUT_STATE.update(
        prot_rows=prot_rows,
        y_all=fr.spine['y'].to_numpy(np.float64),
        u_preds=u_preds,
        u_valid=u_valid,
        B=B,
        seed=seed,
    )
    try:
        shards = [list(range(i, len(prot_rows), n_workers)) for i in range(n_workers)]
        if n_workers > 1 and B >= 50:
            with mp.get_context('fork').Pool(n_workers) as pool:
                parts = pool.map(_pp_mut_worker, shards)
        else:
            parts = [_pp_mut_worker(sh) for sh in shards]
    finally:
        _PP_MUT_STATE.clear()

    sum_tot = sum(p[0] for p in parts)
    cnt_tot = sum(p[1] for p in parts)
    with np.errstate(invalid='ignore', divide='ignore'):
        u_means = np.where(cnt_tot > 0, sum_tot / cnt_tot, np.nan)

    rng_s = np.random.default_rng(seed)
    sw_all: dict[int, np.ndarray] = {}
    for j in range(B):
        sampler = _make_seed_sampler(rng_s)
        sw_all.setdefault(3, np.zeros((3, B), dtype=np.float64))[:, j] = sampler(3)
        for k in b.keys:
            ns = len(k_seed_u[k])
            if ns > 1 and ns != 3:
                arr_s = sw_all.setdefault(ns, np.zeros((ns, B), dtype=np.float64))
                if arr_s[:, j].sum() == 0:
                    arr_s[:, j] = sampler(ns)

    for k, uids in k_seed_u.items():
        if len(uids) <= 1:
            reps_ds[k] = u_means[uids[0]].copy()
        else:
            seed_r = u_means[uids]
            sw_m = sw_all[len(uids)]
            ok = np.isfinite(seed_r) & (sw_m > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                num = np.where(ok, sw_m * seed_r, 0.0).sum(axis=0)
                den = np.where(ok, sw_m, 0.0).sum(axis=0)
                reps_ds[k] = np.where(den > 0, num / den, np.nan)
    return point, {ds: reps_ds}


def _ranks_matrix_sorted(rk, W_ord: np.ndarray) -> np.ndarray:
    """Exact 2D mid-ranks in ``rk.order`` for ``C`` simultaneous bootstrap draws ``(n, C)``."""
    before = np.cumsum(W_ord, axis=0) - W_ord
    single_grp = (len(rk.base_idx) == 0) or (rk.base_idx[-1] == 0)
    if len(rk.tie_start) == len(rk.order):
        if single_grp:
            return before + (W_ord + 1.0) * 0.5
        return (before - before[rk.base_idx]) + (W_ord + 1.0) * 0.5
    tie_T = np.add.reduceat(W_ord, rk.tie_start, axis=0)
    ts = rk.tie_start
    tie_before = before[ts] if single_grp else (before[ts] - before[rk.base_idx[ts]])
    mid = tie_before + (tie_T + 1.0) * 0.5
    return mid[rk.tie_id]


def _eval_boot_chunk(b: DatasetBoot, W_chunk: np.ndarray, sw_chunk: dict[int, np.ndarray], reps_ds: dict[str, np.ndarray], c0: int, c1: int, mk_meta: dict) -> None:
    """Evaluate all models in ``b`` across ``C = c1 - c0`` draws simultaneously in O(n * C) contiguous array ops."""
    for mk, (mk_keys, grp_starts) in mk_meta.items():
        idx, _g, _G, eng = b._engines[mk]
        W = W_chunk[idx]
        W_ry_ord = W[eng.ry.order]
        ry_ord = _ranks_matrix_sorted(eng.ry, W_ry_ord)
        sw = np.add.reduceat(W_ry_ord, grp_starts, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            my = np.add.reduceat(W_ry_ord * ry_ord, grp_starts, axis=0) / sw
            vy = np.add.reduceat(W_ry_ord * ry_ord * ry_ord, grp_starts, axis=0) / sw - my * my
        ry = ry_ord[eng.ry.inv]
        W_ry = W * ry
        bad_y = (sw < 3.0) | (vy <= 0.0) | ~np.isfinite(vy)

        for k in mk_keys:
            xrs = b._xr[k]
            ns = len(xrs)
            seed_r = np.empty((ns, c1 - c0), dtype=np.float64)
            for si, xr in enumerate(xrs):
                W_ord = W[xr.order]
                rx_ord = _ranks_matrix_sorted(xr, W_ord)
                w_rx = W_ord * rx_ord
                with np.errstate(invalid='ignore', divide='ignore'):
                    mx = np.add.reduceat(w_rx, grp_starts, axis=0) / sw
                    vx = np.add.reduceat(w_rx * rx_ord, grp_starts, axis=0) / sw - mx * mx
                    cxy = np.add.reduceat(rx_ord * W_ry[xr.order], grp_starts, axis=0) / sw - mx * my
                    r_mat = cxy / np.sqrt(vx * vy)
                r_mat[bad_y | (vx <= 0.0) | ~np.isfinite(r_mat)] = np.nan
                if b.per_protein:
                    with np.errstate(invalid='ignore'):
                        seed_r[si] = np.nanmean(r_mat, axis=0)
                else:
                    seed_r[si] = r_mat[0]
            if ns <= 1:
                reps_ds[k][c0:c1] = seed_r[0]
            else:
                sw_m = sw_chunk[ns]
                ok = np.isfinite(seed_r) & (sw_m > 0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    num = np.where(ok, sw_m * seed_r, 0.0).sum(axis=0)
                    den = np.where(ok, sw_m, 0.0).sum(axis=0)
                    reps_ds[k][c0:c1] = np.where(den > 0, num / den, np.nan)


_CHUNK_STATE: dict = {}


def _chunk_worker(task) -> dict[str, dict[str, np.ndarray]]:
    mode, c0, c1, payload, sw_c = task
    boots = _CHUNK_STATE['boots']
    mk_meta_all = _CHUNK_STATE['mk_meta_all']
    gidx = _CHUNK_STATE.get('gidx')
    use_pp = _CHUNK_STATE.get('use_pp')
    C = c1 - c0
    out_c: dict[str, dict[str, np.ndarray]] = {}
    for ds, b in boots.items():
        if mode == 'cluster':
            if use_pp[ds]:
                continue
            W_chunk = payload[gidx[ds]].astype(np.float64) if payload.shape[0] else np.zeros((b.n_rows, C), dtype=np.float64)
        else:
            W_chunk = payload[ds].astype(np.float64)
        reps_c = {k: np.full(C, np.nan, dtype=np.float64) for k in b.keys}
        _eval_boot_chunk(b, W_chunk, sw_c, reps_c, 0, C, mk_meta_all[ds])
        out_c[ds] = reps_c
    return out_c


def _fast_per_protein_protein_boot(ds: str, fr: Frame, keys: list[str], B: int, seed: int = 11):
    """Exact (<1e-13) per-protein cluster bootstrap avoiding full-frame 480k-row lexsort overhead."""
    sp = fr.spine
    y_all = sp['y'].to_numpy(np.float64)
    base = sp['use'].to_numpy() & np.isfinite(y_all)
    valid = {k: base & np.isfinite(fr.preds[k]).all(axis=0) for k in keys if k in fr.preds}
    b_keys = [k for k in keys if k in valid and valid[k].sum() >= 10]
    allp, g_inv = np.unique(sp['group'].to_numpy().astype(str), return_inverse=True)
    npro = len(allp)

    ord_g = np.argsort(g_inv, kind='stable')
    gs = g_inv[ord_g]
    starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    ends = np.r_[starts[1:], len(gs)]
    prot_rows = [ord_g[starts[p]:ends[p]] for p in range(npro)]

    u_map: dict[tuple[bytes, bytes], int] = {}
    u_preds: list[np.ndarray] = []
    u_valid: list[np.ndarray] = []
    k_seed_u: dict[str, list[int]] = {}
    for k in b_keys:
        vb = valid[k].tobytes()
        uids = []
        for si in range(fr.preds[k].shape[0]):
            arr = np.asarray(fr.preds[k][si], dtype=np.float64)
            sig = (vb, arr.tobytes())
            uid = u_map.get(sig)
            if uid is None:
                uid = len(u_preds)
                u_map[sig] = uid
                u_preds.append(arr)
                u_valid.append(valid[k])
            uids.append(uid)
        k_seed_u[k] = uids

    from scipy.stats import rankdata
    r_u = np.full((len(u_preds), npro), np.nan, dtype=np.float64)
    has_p: dict[bytes, np.ndarray] = {}
    for p, rows in enumerate(prot_rows):
        yp = y_all[rows]
        by_m: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        for uid in range(len(u_preds)):
            ok = u_valid[uid][rows]
            if not ok.any():
                continue
            by_m.setdefault(ok.tobytes(), (ok, []))[1].append(uid)
        for _mb, (ok, uids) in by_m.items():
            if ok.sum() < 3:
                continue
            ym = yp[ok]
            ry = rankdata(ym)
            ry -= ry.mean()
            vy = float(np.dot(ry, ry))
            if vy <= 0.0:
                continue
            for uid in uids:
                xm = u_preds[uid][rows][ok]
                rx = rankdata(xm)
                rx -= rx.mean()
                vx = float(np.dot(rx, rx))
                if vx > 0.0:
                    r_u[uid, p] = float(np.dot(rx, ry) / np.sqrt(vx * vy))

    for k in b_keys:
        vb = valid[k].tobytes()
        if vb not in has_p:
            has_p[vb] = np.bincount(g_inv[valid[k]], minlength=npro) > 0

    point_ds = {}
    pp_tables = {}
    for k in b_keys:
        vb = valid[k].tobytes()
        p_idx = np.flatnonzero(has_p[vb])
        r_seeds = r_u[np.ix_(k_seed_u[k], p_idx)]
        per_seed = np.nanmean(r_seeds, axis=1)
        point_ds[k] = (float(np.mean(per_seed)), per_seed, int(valid[k].sum()), int(len(p_idx)))
        pp_tables[k] = (p_idx, r_seeds)

    reps_ds = {k: np.full(B, np.nan, dtype=np.float64) for k in b_keys}
    rng = np.random.default_rng(seed)
    for bi in range(B):
        sampler = _make_seed_sampler(rng)
        cnt = rng.multinomial(npro, np.full(npro, 1.0 / npro)).astype(float)
        for k, (p_idx, r_seeds) in pp_tables.items():
            c_v = cnt[p_idx]
            sw = sampler(r_seeds.shape[0])
            W_p = np.outer(sw, c_v)
            ok = np.isfinite(r_seeds) & (W_p > 0)
            reps_ds[k][bi] = float(np.sum(W_p[ok] * r_seeds[ok]) / np.sum(W_p[ok])) if ok.any() else np.nan
    return {ds: point_ds}, {ds: reps_ds}


def _fast_bootstrap_rho(frames: dict[str, Frame], keys: list[str], kind: str, B: int, seed: int = 11, per_protein: bool | None = None, procs: int = 8):
    """Chunk-vectorized exact equivalent of ``msr_analysis.boot.bootstrap_rho``."""
    if kind == 'protein' and all(((d in PER_PROTEIN) if per_protein is None else per_protein) for d in frames):
        if len(frames) == 1:
            ds0, fr0 = next(iter(frames.items()))
            return _fast_per_protein_protein_boot(ds0, fr0, keys, B=B, seed=seed)
        return bootstrap_rho(frames, keys, kind=kind, B=B, seed=seed, per_protein=per_protein)
    if kind == 'mutation' and len(frames) == 1:
        ds0, fr0 = next(iter(frames.items()))
        if (ds0 in PER_PROTEIN) if per_protein is None else per_protein:
            return _fast_per_protein_mutation_boot(ds0, fr0, keys, B=B, seed=seed, procs=procs)

    rng = np.random.default_rng(seed)
    boots = {ds: DatasetBoot(fr, keys, per_protein=(ds in PER_PROTEIN) if per_protein is None else per_protein) for ds, fr in frames.items()}
    point = {ds: {k: b.point(k) for k in b.keys} for ds, b in boots.items()}
    reps = {ds: {k: np.full(B, np.nan) for k in b.keys} for ds, b in boots.items()}

    mk_meta_all = {}
    mut_prep = {}
    max_rows = max((b.n_rows for b in boots.values()), default=1)
    chunk = 25 if max_rows >= 150_000 else (50 if max_rows >= 20_000 else 64)

    for ds, b in boots.items():
        mk_groups: dict[bytes, list[str]] = {}
        for k in b.keys:
            mk_groups.setdefault(b.valid[k].tobytes(), []).append(k)
        meta_ds = {}
        for mk, mk_keys in mk_groups.items():
            _idx, g, _G, eng = b._engines[mk]
            g_ord = g[eng.ry.order]
            grp_starts = np.r_[0, np.flatnonzero(g_ord[1:] != g_ord[:-1]) + 1] if len(g_ord) else np.array([0], dtype=np.int64)
            meta_ds[mk] = (mk_keys, grp_starts)
        mk_meta_all[ds] = meta_ds
        if kind == 'mutation':
            base = np.flatnonzero(b.base)
            if b.per_protein:
                g_b = b.groups_all[base]
                ord_b = np.argsort(g_b, kind='stable')
                gs = g_b[ord_b]
                starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
                sizes = np.diff(np.r_[starts, len(gs)])
                mut_prep[ds] = (True, base[ord_b], np.repeat(starts, sizes), np.repeat(sizes, sizes))
            else:
                mut_prep[ds] = (False, base, None, None)

    tasks = []
    if kind in ('protein', 'site'):
        use_pp = {ds: (kind == 'protein' and b.per_protein) for ds, b in boots.items()}
        names = {ds: b.cluster_names[kind] for ds, b in boots.items() if not use_pp[ds]}
        gidx, ncl = global_cluster_index(names)
        pp_tables, pp_pos = {}, {}
        for ds, b in boots.items():
            if use_pp[ds]:
                pp_tables[ds] = {k: b.per_protein_rho_seeds(k) for k in b.keys}
                allp, _ = np.unique(b.frame.spine['group'].to_numpy().astype(str), return_inverse=True)
                pp_pos[ds] = {p: i for i, p in enumerate(allp)}
        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            C = c1 - c0
            counts_c = np.zeros((ncl, C), dtype=np.uint16)
            sw_c: dict[int, np.ndarray] = {}
            for j, bi in enumerate(range(c0, c1)):
                if ncl:
                    counts_c[:, j] = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl))
                sampler = _make_seed_sampler(rng)
                sw_c.setdefault(3, np.zeros((3, C), dtype=np.float64))[:, j] = sampler(3)
                for ds, b in boots.items():
                    if use_pp[ds]:
                        npro = len(pp_pos[ds])
                        cnt = rng.multinomial(npro, np.full(npro, 1.0 / npro)).astype(float)
                        for k, (nm, r_seeds) in pp_tables[ds].items():
                            c_v = cnt[[pp_pos[ds][str(x)] for x in nm]]
                            sw = sampler(r_seeds.shape[0])
                            W_p = np.outer(sw, c_v)
                            ok = np.isfinite(r_seeds) & (W_p > 0)
                            reps[ds][k][bi] = float(np.sum(W_p[ok] * r_seeds[ok]) / np.sum(W_p[ok])) if ok.any() else np.nan
                    else:
                        for k in b.keys:
                            ns = len(b._xr[k])
                            if ns > 1 and ns != 3:
                                arr_s = sw_c.setdefault(ns, np.zeros((ns, C), dtype=np.float64))
                                if arr_s[:, j].sum() == 0:
                                    arr_s[:, j] = sampler(ns)
            if any(not use_pp[ds] for ds in boots):
                tasks.append(('cluster', c0, c1, counts_c, sw_c))
        state_extra = dict(gidx=gidx, use_pp=use_pp)
    else:
        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            C = c1 - c0
            W_chunks = {ds: np.zeros((b.n_rows, C), dtype=np.uint16) for ds, b in boots.items()}
            sw_c = {}
            for j in range(C):
                sampler = _make_seed_sampler(rng)
                sw_c.setdefault(3, np.zeros((3, C), dtype=np.float64))[:, j] = sampler(3)
                for ds, b in boots.items():
                    is_pp, base_arr, first, size = mut_prep[ds]
                    if is_pp:
                        pick = first + np.floor(rng.random(len(base_arr)) * size).astype(np.int64)
                        sel = base_arr[pick]
                    else:
                        sel = base_arr[rng.integers(0, len(base_arr), len(base_arr))]
                    W_chunks[ds][:, j] = np.bincount(sel, minlength=b.n_rows)
                    for k in b.keys:
                        ns = len(b._xr[k])
                        if ns > 1 and ns != 3:
                            arr_s = sw_c.setdefault(ns, np.zeros((ns, C), dtype=np.float64))
                            if arr_s[:, j].sum() == 0:
                                arr_s[:, j] = sampler(ns)
            tasks.append(('mutation', c0, c1, W_chunks, sw_c))
        state_extra = {}

    if tasks:
        n_w = min(max(1, procs), len(tasks))
        _CHUNK_STATE.clear()
        _CHUNK_STATE.update(boots=boots, mk_meta_all=mk_meta_all, **state_extra)
        try:
            if n_w > 1 and B >= 100:
                with mp.get_context('fork').Pool(n_w) as pool:
                    outs = pool.map(_chunk_worker, tasks)
            else:
                outs = [_chunk_worker(tk) for tk in tasks]
        finally:
            _CHUNK_STATE.clear()
        for (_mode, c0, c1, _pl, _sw), out_c in zip(tasks, outs):
            for ds, reps_c in out_c.items():
                for k, arr_c in reps_c.items():
                    reps[ds][k][c0:c1] = arr_c
    return point, reps


def run_core_rho(B: int, ext_boot: dict | None = None) -> pd.DataFrame:
    """Compute protein-, site-, and mutation-level Spearman bootstrap intervals across all benchmark datasets."""
    out = out_dir('00_core')
    rows: list[dict] = []

    cached_mut: dict[tuple[str, str], np.ndarray] = {}
    if ext_boot is not None:
        ext_reps = ext_boot.get('reps', {})
        for (ds_c, m_c), raw_k in ext_boot.get('keys', {}).items():
            arr_c = ext_reps.get(ds_c, {}).get(m_c)
            if arr_c is not None and len(arr_c) == B:
                cached_mut[(ds_c, raw_k)] = np.asarray(arr_c, dtype=np.float64)

    def _run_kind(kind: str, groups: dict[str, list[str]]) -> None:
        for name, dss in groups.items():
            t0 = time.perf_counter()
            frames = {d: load_frame(d) for d in dss}
            keys = sorted({k for f in frames.values() for k in _keys_for(f)})
            if kind == 'mutation' and cached_mut:
                # Reuse exact B-draw mutation replicates already computed in boot_ext_mutation.pkl (including single-mutant aliases)
                point: dict[str, dict] = {}
                reps: dict[str, dict[str, np.ndarray]] = {d: {} for d in dss}
                need_frames: dict[str, Frame] = {}
                need_keys_set: set[str] = set()
                for ds, fr in frames.items():
                    ds_keys = _keys_for(fr)
                    if ds in PER_PROTEIN:
                        pt_ds, _ = _fast_per_protein_protein_boot(ds, fr, ds_keys, B=0, seed=11)
                        point[ds] = pt_ds[ds]
                    else:
                        b_tmp = DatasetBoot(fr, ds_keys, per_protein=False)
                        point[ds] = {k: b_tmp.point(k) for k in b_tmp.keys}
                    is_single_only = not bool((fr.spine.loc[fr.spine['use'], 'n_mut'] > 1).any())
                    alias = {'msrp0:comb': 'msrp0:wt', 'msrrel@0:comb': 'msrrel@0:wt'} if is_single_only else {}
                    miss_ds = []
                    for k in point[ds]:
                        src_k = alias.get(k, k)
                        if (ds, src_k) in cached_mut and (src_k == k or np.array_equal(fr.preds[k], fr.preds[src_k])):
                            reps[ds][k] = cached_mut[(ds, src_k)]
                        else:
                            miss_ds.append(k)
                    if miss_ds:
                        need_frames[ds] = fr
                        need_keys_set.update(miss_ds)
                if need_frames:
                    _, rp_miss = _fast_bootstrap_rho(need_frames, sorted(need_keys_set), kind='mutation', B=B, seed=11)
                    for ds_m, kmap_m in rp_miss.items():
                        for k_m, arr_m in kmap_m.items():
                            if k_m not in reps[ds_m]:
                                reps[ds_m][k_m] = arr_m
            else:
                point, reps = _fast_bootstrap_rho(frames, keys, kind=kind, B=B, seed=11)
            for ds in dss:
                for k, (rho, per_seed, n, ng) in point[ds].items():
                    lo, hi = ci(reps[ds][k])
                    rows.append(dict(
                        dataset=ds,
                        model=k,
                        kind=kind,
                        rho=rho,
                        lo=lo,
                        hi=hi,
                        se=float(np.nanstd(reps[ds][k], ddof=1)),
                        n=n,
                        n_proteins=ng,
                        seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                        B=B,
                    ))
            print(f'  [core_rho] {kind:8s} {name:12s} B={B:4d} {len(keys):2d} models  {time.perf_counter() - t0:5.1f}s', flush=True)

    _run_kind('protein', {
        'external': EXT12,
        'tsu_val': [TSU_CAPPED[0]],
        'tsu_test': [TSU_CAPPED[1]],
        'domainome': ['domainome'],
    })
    _run_kind('site', {d: [d] for d in DMS3})
    _run_kind('mutation', {
        'external': EXT12,
        **{d: [d] for d in DMS3 + list(TSU_CAPPED) + ['domainome']},
    })

    df = pd.DataFrame(rows)
    atomic_to_csv(df, out / 'rho_bootstrap.csv', index=False, float_format='%.5f')
    return df


def _get_ext_mutation_boot(B: int, refresh: bool = False) -> dict:
    """Return mutation-level bootstrap payload for EXT12 + DMS3 + domainome (reusing cache when B == 1000)."""
    if B == 1000:
        return load_mutation_boot(refresh=refresh)

    point_out: dict = {}
    reps_out: dict = {}
    keys_out: dict = {}

    def _run_group(frames: dict[str, Frame], keys: list[str], seed: int, per_protein: bool) -> None:
        pt, rp = _fast_bootstrap_rho(frames, keys, kind='mutation', B=B, seed=seed, per_protein=per_protein)
        for ds, kmap in pt.items():
            reps_out[ds] = {}
            for m, (r, _ps, _n, _ng) in kmap.items():
                lo, hi = ci(rp[ds][m])
                point_out[(ds, m)] = (r, lo, hi)
                reps_out[ds][m] = rp[ds][m]

    ext_frames = {}
    for ds in EXT12:
        fr, used = seed_frame(ds, load_frame(ds))
        ext_frames[ds] = fr
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
    ext_keys = sorted({m for fr in ext_frames.values() for m in fr.preds})
    _run_group(ext_frames, ext_keys, seed=11, per_protein=False)

    for i, ds in enumerate(DMS3):
        fr, used = seed_frame(ds, load_frame(ds))
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
        _run_group({ds: fr}, list(fr.preds), seed=12 + i, per_protein=False)

    dom_fr, dom_used = seed_frame('domainome', load_frame('domainome'))
    for m, rk in dom_used.items():
        keys_out[('domainome', m)] = rk
    _run_group({'domainome': dom_fr}, list(dom_fr.preds), seed=30, per_protein=True)

    return dict(point=point_out, reps=reps_out, keys=keys_out)


def _load_thermo_overview_boot(B: int, refresh: bool = False) -> dict:
    cache_path = CACHE / 'thermo_overview_boot.pkl'
    if B == 1000 and cache_path.exists() and not refresh:
        import pickle
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('models') == OVERVIEW_MODELS and cached.get('B') == 1000:
            return cached

    frames, meta, used = {}, {}, {}
    for d in EXT10:
        fr = load_frame(d)
        sp = fr.spine
        u = sp['use'].to_numpy()
        meta[d] = (int(u.sum()), int(sp.loc[u, 'cluster'].nunique()))
        frames[d], kk = seed_frame(d, fr, models=OVERVIEW_MODELS)
        used.update({(d, m): k for m, k in kk.items()})

    pt_p, rep_p = bootstrap_rho(frames, OVERVIEW_MODELS, kind='protein', B=B, seed=11, per_protein=False)
    pt_m, rep_m = bootstrap_rho(frames, OVERVIEW_MODELS, kind='mutation', B=B, seed=11, per_protein=False)

    P_prot, P_mut = {}, {}
    for d in EXT10:
        for k in OVERVIEW_MODELS:
            if k in pt_p[d]:
                lo_p, hi_p = ci(rep_p[d][k])
                P_prot[(d, k)] = (float(pt_p[d][k][0]), float(lo_p), float(hi_p))
            if k in pt_m[d]:
                lo_m, hi_m = ci(rep_m[d][k])
                P_mut[(d, k)] = (float(pt_m[d][k][0]), float(lo_m), float(hi_m))

    payload = dict(models=list(OVERVIEW_MODELS), B=B, meta=meta, keys=used, P_prot=P_prot, P_mut=P_mut)
    if B == 1000:
        atomic_pickle_dump(payload, cache_path)
    return payload


def run_thermostability(boot_payload: dict, B: int, refresh: bool = False) -> pd.DataFrame:
    """Generate external thermostability CSV and three figures under results/02_thermostability/."""
    out = out_dir('02_thermostability')
    P, keys_used = boot_payload['point'], boot_payload['keys']
    leak = leak_table()
    active_models = [m for m in EXT_ORDER if any((d, m) in P for d in EXT10)]

    rows = []
    for m in active_models:
        for d in EXT10:
            if (d, m) not in P:
                continue
            raw_k = keys_used[(d, m)]
            v = P[(d, m)]
            rows.append(dict(
                model=m, key=raw_k, dataset=d,
                rho=v[0], lo=v[1], hi=v[2],
                leak_flag=bool(leak.get((raw_k, d), False)),
            ))
    S = pd.DataFrame(rows)
    atomic_to_csv(S, out / 'thermostability_rho.csv', index=False, float_format='%.4f')

    # Figure 1: vertical dataset bands across EXT10
    table = {(d, m): P[(d, m)][0] for d in EXT10 for m in active_models if (d, m) in P}
    cis = {(d, m): (P[(d, m)][1], P[(d, m)][2]) for d in EXT10 for m in active_models if (d, m) in P}
    min_lo = min(v[0] for v in cis.values() if np.isfinite(v[0]))
    ymin = min(0.10, float(np.floor((min_lo - 0.03) * 20) / 20))

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    spread = 0.78
    style.dot_panel(
        ax, EXT10, active_models, table, cis,
        highlight=['ESM-MSR'], ylabel='Spearman ρ',
        xlabels=[LABEL.get(d, d) for d in EXT10],
        ylim=(ymin, 0.98), spread=spread, size=6.0, hollow=style.hollow_cells(keys_used),
    )
    offs = np.linspace(-spread / 2, spread / 2, len(active_models))
    for i, d in enumerate(EXT10):
        for j, m in enumerate(active_models):
            if (d, m) in P and leak.get((keys_used[(d, m)], d), False):
                ax.plot(i + offs[j], P[(d, m)][0], marker='o', ms=10.5, mfc='none', mec=style.RING, mew=1.1, ls='none', zorder=6)

    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in active_models]
    flagged = [m for m in active_models if any(leak.get((keys_used[(d, m)], d), False) for d in EXT10 if (d, m) in P)]
    hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec=style.RING, mew=1.1, ls='none', label=f'training overlap ({", ".join(flagged)})'))
    hs += style.version_handles()
    fig.legend(
        handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.942), ncol=5, fontsize=8.0,
        frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3,
    )
    fig.suptitle(
        f'External thermostability sets: Spearman ρ across all models (seeds 1-3 averaged; whiskers: 95% mutation-level bootstrap, B = {B:,})',
        x=0.012, ha='left', y=0.985, fontsize=10.5, fontweight='bold',
    )
    note = (
        'Seeds 1-3 averaged for the retrained models (seeds resampled with the draws); 95% mutation-level bootstrap. '
        'Singles version of every model on the 8 single-mutant sets (hollow markers for the models that have two); '
        'doubles version on PTMUL/PTMUL-D (ThermoMPNN additive on PTMUL, -D on PTMUL-D; Rosetta Cartesian ddG throughout). '
        f'Red ring = documented training overlap ({", ".join(flagged)}).'
    )
    fig.text(0.012, 0.01, note, fontsize=8.0, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.84))
    style.save(fig, out / 'thermo_external_rho')

    # Figure 2: mutation-level vs protein-level bootstrap interval half-widths
    p0_frames = {}
    for d in EXT10:
        fr = load_frame(d)
        k = model_key_for_dataset('ESM-MSR', fr)
        p0_frames[d] = Frame(d, fr.spine, {'ESM-MSR': fr.preds[k]})
    _, prot_reps = bootstrap_rho(p0_frames, ['ESM-MSR'], kind='protein', B=B, seed=21, per_protein=False)
    wp, wm = [], []
    for d in EXT10:
        pl, ph = ci(prot_reps[d]['ESM-MSR'])
        ml, mh = P[(d, 'ESM-MSR')][1], P[(d, 'ESM-MSR')][2]
        wp.append((ph - pl) / 2.0)
        wm.append((mh - ml) / 2.0)
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(EXT10))
    ax2.bar(x - 0.2, wm, 0.4, color='#9ecae1', label='resampling mutations')
    ax2.bar(x + 0.2, wp, 0.4, color='#08519c', label='resampling proteins')
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[d] for d in EXT10], rotation=30, ha='right')
    ax2.set_ylabel('95% interval half-width of ρ')
    ax2.legend()
    ax2.set_title(f'Intervals for ESM-MSR (seeds 1-3; WT-view on singles sets / Dual-view on PTMUL(-D); B = {B:,})', loc='left')
    fig2.tight_layout()
    style.save(fig2, out / 'thermo_interval_types')

    # Figure 3: 2x5 horizontal forest plot across EXT10 (zero in-place MODELS mutation)
    ov = _load_thermo_overview_boot(B=B, refresh=refresh)
    P_prot, P_mut, meta, used = ov['P_prot'], ov['P_mut'], ov['meta'], ov['keys']
    names = [m for m in OVERVIEW_MODELS if any((d, m) in P_prot for d in EXT10)]
    star_labels = {'GeoDDG-3D': 'GeoDDG-3D*', 'DDGemb': 'DDGemb*', 'DDMut': 'DDMut*'}
    display_labels = {k: star_labels.get(k, MODELS[k]['label']) for k in names}

    row_sets = [EXT10[:5], EXT10[5:]]
    fig3, axes = plt.subplots(2, 5, figsize=(17.8, 2 * (0.28 * len(names) + 1.2)), sharey=True)
    for r_idx, row_ds in enumerate(row_sets):
        for c_idx, ds in enumerate(row_ds):
            ax_cell = axes[r_idx, c_idx]
            vals = [P_prot[(ds, k)] for k in names if (ds, k) in P_prot]
            mut_vals = [P_mut[(ds, k)] for k in names if (ds, k) in P_mut]
            min_lo_cell = min(min(lo for _v, lo, _hi in vals), min(lo for _v, lo, _hi in mut_vals))
            max_hi_cell = max(max(hi for _v, _lo, hi in vals), max(hi for _v, _lo, hi in mut_vals))
            x_lo = float(np.floor((min_lo_cell - 0.03) * 20) / 20)
            span = max(0.30, max_hi_cell - x_lo)
            x_hi = max_hi_cell + 0.24 * span

            for i, k in enumerate(names):
                if (ds, k) not in P_prot:
                    continue
                v, lo, hi = P_prot[(ds, k)]
                m_ci = P_mut.get((ds, k))
                kw = style.marker_kwargs(k, size=5.6, hollow=bool(R.BY_NAME[k].two_versions and used[(ds, k)] == R.BY_NAME[k].single))
                ax_cell.plot([lo, hi], [i, i], color=kw['color'], lw=2.3, alpha=0.45, solid_capstyle='butt')
                if m_ci:
                    ax_cell.plot([m_ci[1], m_ci[2]], [i, i], color=kw['color'], lw=0.8)
                ax_cell.plot(v, i, **kw)
                if leak.get((used[(ds, k)], ds), False):
                    ax_cell.plot(v, i, marker='o', ms=9.5, mfc='none', mec=style.RING, mew=0.95, ls='none')
                r_edge = max(hi, m_ci[2] if m_ci else hi)
                ax_cell.text(r_edge + 0.022 * span, i, f'{v:.3f}', va='center', fontsize=6.9, color='#444')
            for i, k in enumerate(names):
                if k in SAW_TEST_DATA:
                    ax_cell.axhspan(i - 0.5, i + 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
            n_use, n_prot = meta[ds]
            ax_cell.set_title(f'{LABEL[ds]} (n = {n_use:,}, {n_prot} prot.)', loc='left', fontsize=9.3)
            ax_cell.grid(axis='x', color=style.GRID, lw=0.6)
            ax_cell.set_xlim(x_lo, x_hi)
            if r_idx == 1:
                ax_cell.set_xlabel('Spearman ρ', fontsize=9.0)

        ax_l = axes[r_idx, 0]
        ax_l.set_yticks(range(len(names)))
        ax_l.set_yticklabels([display_labels[k] for k in names], fontsize=8.0)
        ax_l.set_ylim(len(names) - 0.4, -0.6)
        for t, k in zip(ax_l.get_yticklabels(), names):
            t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555'}[MODELS[k]['group']])
            if MODELS[k]['group'] == 'msr':
                t.set_fontweight('bold')

    fig3.suptitle(
        'External thermostability sets (seeds 1-3 averaged): thick bar = resampling proteins; thin line = resampling mutations; red ring = documented training overlap\n'
        'Singles version on the singles sets, doubles version on PTMUL(-D); * trained on ProThermDB / ThermoMutDB / S2648 / S8754: not held out (grey rows); ESMtherm* and IFUM* were trained on Megascale',
        fontsize=9.2, y=1.01,
    )
    fig3.tight_layout()
    style.save(fig3, out / 'thermo_overview')
    return S


def _build_dms_table(models: list[str], P: dict, reps: dict, keys_used: dict) -> tuple[dict, dict, pd.DataFrame]:
    t, c, rows = {}, {}, []
    for m in models:
        for d in DMS_DS:
            if (d, m) not in P:
                continue
            v = P[(d, m)]
            t[(d, m)] = v[0]
            c[(d, m)] = (v[1], v[2])
            rows.append(dict(column=d, model=m, key=keys_used.get((d, m), ''), rho=v[0], lo=v[1], hi=v[2]))
        if all((d, m) in P for d in DMS3):
            pt = float(np.mean([P[(d, m)][0] for d in DMS3]))
            b_min = min(len(reps[d][m]) for d in DMS3)
            r = np.mean([reps[d][m][:b_min] for d in DMS3], axis=0)
            lo, hi = ci(r)
            t[('mean of 3 DMS', m)] = pt
            c[('mean of 3 DMS', m)] = (lo, hi)
            rows.append(dict(column='mean of 3 DMS', model=m, key='wt|comb', rho=pt, lo=lo, hi=hi))
    return t, c, pd.DataFrame(rows)


def run_dms_domainome(boot_payload: dict, B: int) -> pd.DataFrame:
    """Generate DMS & held-out Domainome CSVs and figures under results/04_dms_domainome/."""
    out = out_dir('04_dms_domainome')
    P, reps, keys_used = boot_payload['point'], boot_payload['reps'], boot_payload['keys']
    active_models = [m for m in EXT_ORDER if all((d, m) in P for d in DMS_DS)]
    hollow = style.hollow_cells(keys_used)

    def _draw_top_legend(fig, models, bbox=(0.5, 0.94), ncol=4):
        hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8, hollow=False)) for m in models] + style.version_handles()
        return fig.legend(
            handles=hs, loc='upper center', bbox_to_anchor=bbox, ncol=ncol, fontsize=8.0,
            frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.2, handletextpad=0.35,
        )

    # Figure 1: unified all-model comparison across DMS3 + mean + held-out Domainome
    t_all, c_all, df_all = _build_dms_table(active_models, P, reps, keys_used)
    atomic_to_csv(df_all, out / 'dms_vs_retrains.csv', index=False, float_format='%.4f')

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    style.dot_panel(
        ax, DMS_COLS, active_models, t_all, c_all,
        highlight=['ESM-MSR'], ylabel='Spearman ρ',
        xlabels=[LABEL.get(x, x) for x in DMS_COLS],
        size=6.2, spread=0.78, ylim=(0.0, 0.90), hollow=hollow,
    )
    _draw_top_legend(fig, active_models, bbox=(0.5, 0.942), ncol=4)
    fig.suptitle(
        f'External DMS & Domainome: Spearman ρ across all models (seeds 1-3 averaged; 95% mutation-level bootstrap, B = {B:,})',
        x=0.012, ha='left', y=0.985, fontsize=10.2, fontweight='bold',
    )
    note = (
        'Seeds 1-3 averaged for the retrained models; 95% mutation-level bootstrap (within-domain mutations for held-out Domainome). '
        'Singles version of every model on Domainome (hollow markers for the models that have two); doubles version on DLG4 binding / GRB2 binding / MYO '
        '(ThermoMPNN-D on the 2-mutant sets).'
    )
    fig.text(0.012, 0.01, note, fontsize=8.0, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.84))
    style.save(fig, out / 'dms_vs_retrains')

    # Figure 2: ESM-MSR (sigma = 1) vs zero-shot models
    zs_models = [m for m in EXT_ORDER if m in ('ESM-MSR', 'ESM3-small-open', 'ProteinMPNN', 'ProSST')]
    t_zs, c_zs, df_zs = _build_dms_table(zs_models, P, reps, keys_used)
    atomic_to_csv(df_zs, out / 'dms_sigma_vs_zero_shot.csv', index=False, float_format='%.4f')

    fig2, ax2 = plt.subplots(figsize=(12.5, 5.2))
    style.dot_panel(
        ax2, DMS_COLS, zs_models, t_zs, c_zs,
        highlight=['ESM-MSR'], ylabel='Spearman ρ',
        xlabels=[LABEL.get(x, x) for x in DMS_COLS],
        size=6.4, spread=0.68, ylim=(0.0, 0.90), hollow=hollow,
    )
    _draw_top_legend(fig2, zs_models, bbox=(0.5, 0.942), ncol=4)
    fig2.suptitle(
        f'ESM-MSR (seeds 1-3, σ = 1) vs zero-shot models (σ = 0 and baselines; 95% mutation-level bootstrap, B = {B:,})',
        x=0.012, ha='left', y=0.985, fontsize=10.0, fontweight='bold',
    )
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    style.save(fig2, out / 'dms_sigma_vs_zero_shot')

    # Figure 3: sigma = 0 -> 1 comparison on the 4 datasets (fixed 1x4 grid, no empty axes)
    fig3, axes3 = plt.subplots(1, len(DMS_DS), figsize=(11.2, 3.1), sharey=False)
    for ax_i, col in zip(axes3, DMS_DS):
        v0 = P[(col, 'ESM3-small-open')]
        v1 = P[(col, 'ESM-MSR')]
        xs = [0.0, 1.0]
        ys = [v0[0], v1[0]]
        los = [v0[1], v1[1]]
        his = [v0[2], v1[2]]
        ax_i.plot(xs, ys, '-o', color='k', ms=5.2, lw=1.6)
        ax_i.errorbar(
            xs, ys,
            yerr=[[ys[0] - los[0], ys[1] - los[1]], [his[0] - ys[0], his[1] - ys[1]]],
            fmt='none', ecolor='k', elinewidth=1.15, capsize=2.8,
        )
        ax_i.set_title(LABEL[col], fontsize=9.5)
        ax_i.set_xlabel('LoRA scale σ')
        ax_i.set_xticks([0.0, 1.0])
        ax_i.set_xticklabels(['0', '1'])
        ax_i.set_xlim(-0.25, 1.25)
        ax_i.grid(axis='y', color=style.GRID, lw=0.6)
    axes3[0].set_ylabel('Spearman ρ')
    fig3.suptitle(f'ESM-MSR (seeds 1-3) vs σ = 0 zero-shot control (95% mutation-level bootstrap, B = {B:,})', fontsize=10, y=1.02)
    fig3.tight_layout()
    style.save(fig3, out / 'dms_sigma_sweep')
    return df_all


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    ap.add_argument('--refresh', action='store_true', help='Recompute external frame/bootstrap caches')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    if args.refresh:
        for d in ['s571', 's783', 's2648', 's8754', 'ssym', 'q3421', 's461', 'k2369', 'ptmuld']:
            load_frame(d, refresh=True)

    boot_payload = _get_ext_mutation_boot(B=B, refresh=args.refresh)
    run_core_rho(B=B, ext_boot=boot_payload)
    run_thermostability(boot_payload, B=B, refresh=args.refresh)
    df_dms = run_dms_domainome(boot_payload, B=B)
    active = [m for m in EXT_ORDER if m in set(df_dms['model'])]
    print(df_dms.pivot(index='model', columns='column', values='rho').reindex(active)[DMS_COLS].round(3).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
