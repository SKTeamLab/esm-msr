"""Shared epistasis-hierarchy table builder, domain-cloning 3-seed bootstrapper, and saturation smoother.

Wraps ``esm_msr.epi_metrics`` so ``11d``, ``11n``, ``98``, ``12``, and MGnify epistasis analyses share one
implementation without cross-script ``importlib`` hacks.
"""
from __future__ import annotations

import copy
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd

from esm_msr import epi_metrics as E

from .boot import ci
from .stats import seed_sd
from .structures import parse_mutations

EPI_BASE_METRICS = ('naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank', 'cell_mag')
EPI_EXTENDED_METRICS = (
    'naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank', 'cell_mag', 'cell_mag_pooled', 'cell_mag_gt0',
    'naive_rho_gt1', 'beyond_rho_gt1', 'pair_rho_gt1', 'line_rho_gt1', 'cell_rank_gt1', 'cell_mag_gt1', 'cell_mag_pooled_gt1',
    'naive_rho_sub0', 'beyond_rho_sub0', 'pair_rho_sub0', 'line_rho_sub0', 'cell_rank_sub0', 'cell_mag_sub0', 'cell_mag_pooled_sub0',
)


def parse_mut_key(mut_str: str, code: str) -> tuple[tuple[str, int, str], ...] | None:
    """Parse ``mut_str`` on ``code`` into ``((code, pos, mut_aa), ...)`` as expected by ``esm_msr.epi_metrics.Table``."""
    toks = parse_mutations(mut_str, sep=':')
    if not toks:
        return None
    return tuple((str(code), pos, mut_aa) for _wt, pos, mut_aa in toks)


def smooth_on(x, v, nbins: int = E.SMOOTH_BINS) -> np.ndarray:
    """Piecewise-linear quantile-bin conditional expectation ``E[v | x]`` (delegating to ``esm_msr.epi_metrics.smooth_on``)."""
    return E.smooth_on(np.asarray(x, dtype=float), np.asarray(v, dtype=float), nbins=nbins)


def make_epi_table(
    spine: pd.DataFrame,
    wt,
    mt,
    comb=None,
    heads: tuple[str, ...] = ('epi', 'add'),
    wt_rev=None,
    link=None,
    mask=None,
) -> E.Table:
    """Construct an ``esm_msr.epi_metrics.Table`` from a ``Frame.spine`` and 1D per-row predictions."""
    use = spine['use'].to_numpy(dtype=bool) if 'use' in spine else np.ones(len(spine), dtype=bool)
    if mask is not None:
        use = use & np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(use)
    code_col = 'code' if 'code' in spine else 'code_wt'
    mut_col = 'mut' if 'mut' in spine else 'mut_type'
    ddg_col = 'y' if 'y' in spine else 'ddG_ML'
    code = spine[code_col].astype(str).to_numpy()[idx]
    mut = spine[mut_col].astype(str).to_numpy()[idx]
    keys = [parse_mut_key(m, c) for m, c in zip(mut, code)]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    wt_arr = np.asarray(wt, dtype=float)[idx]
    mt_arr = np.asarray(mt, dtype=float)[idx]
    comb_arr = mt_arr if comb is None else np.asarray(comb, dtype=float)[idx]
    items = dict(
        mut_key=keys,
        subset_type=st,
        ddG=spine[ddg_col].to_numpy(dtype=float)[idx],
        dddG=spine['dddG_ML'].to_numpy(dtype=float)[idx],
        dG_wt=spine['dG_wt'].to_numpy(dtype=float)[idx],
        wt=wt_arr,
        mt=mt_arr,
        comb=comb_arr,
    )
    if wt_rev is not None:
        items['wt_rev'] = np.asarray(wt_rev, dtype=float)[idx]
    return E.Table(items, link, heads=heads)


def subset_epi_table(table: E.Table, mask) -> E.Table:
    """Return a shallow-sliced ``E.Table`` restricted to double rows matching ``mask`` (length ``table.n``)."""
    r = copy.copy(table)
    idx = np.flatnonzero(mask)
    n0 = table.n
    for name, v in vars(table).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    r._mats = None
    return r


def resample_epi_table(table: E.Table, idx: np.ndarray, rep: np.ndarray) -> E.Table:
    """Clone resampled domains cleanly by offsetting ``pair``, ``row``, ``col``, and ``lib`` per domain copy ``rep``."""
    r = copy.copy(table)
    n0 = table.n
    for name, v in vars(table).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    if len(idx):
        r.pair = table.pair[idx] + rep * (int(table.pair.max()) + 1)
        rc_span = int(max(table.row.max(), table.col.max())) + 1
        r.row = table.row[idx] + rep * rc_span
        r.col = table.col[idx] + rep * rc_span
        r.lib = table.lib[idx] + rep * (int(table.lib.max()) + 1)
    r._mats = None
    return r


def _dc_nan(M: np.ndarray, min_line: int = 3, n_iter: int = 8) -> np.ndarray:
    X = M.copy()
    while True:
        fin = np.isfinite(X)
        bad_r = (fin.sum(1) > 0) & (fin.sum(1) < min_line)
        bad_c = (fin.sum(0) > 0) & (fin.sum(0) < min_line)
        if not bad_r.any() and not bad_c.any():
            break
        X[bad_r, :] = np.nan
        X[:, bad_c] = np.nan
    for _ in range(n_iter):
        with np.errstate(invalid='ignore'):
            X = X - np.nanmean(X, axis=1, keepdims=True)
            X = X - np.nanmean(X, axis=0, keepdims=True)
    return X


def _pooled_cell_mag(table: E.Table, head: str) -> float:
    rm, rp = E.beyond_global(table, head)
    blk_m, blk_p = [], []
    for M in table.matrices():
        Rm, Rp = E._lay(M, rm), E._lay(M, rp)
        ok = np.isfinite(Rm) & np.isfinite(Rp)
        if ok.sum() < E.MIN_PAIR_CELLS:
            continue
        Rm_ok, Rp_ok = np.where(ok, Rm, np.nan), np.where(ok, Rp, np.nan)
        blk = E._complete_block(Rm_ok)
        if blk is not None:
            ix = np.ix_(*blk)
            blk_m.extend(E._dc(Rm_ok[ix]).ravel())
            blk_p.extend(E._dc(Rp_ok[ix]).ravel())
    return float(E._rho(blk_p, blk_m)) if len(blk_m) >= 9 else np.nan


def compute_epi_metrics(table: E.Table, head: str = 'epi', extended: bool = True) -> dict[str, float]:
    """Compute canonical epistasis-hierarchy metrics (plus ``x > 0`` / ``x > 1`` / pooled cell-mag when ``extended=True``)."""
    r = E.compute(table, {m: (head,) for m in EPI_BASE_METRICS})
    out = {m: float(r.get(f'epi_{m}_{head}', np.nan)) for m in EPI_BASE_METRICS}
    if not extended:
        return out

    t_gt1 = subset_epi_table(table, table.x > 1.0)
    r_gt1 = E.compute(t_gt1, {m: (head,) for m in EPI_BASE_METRICS})
    t_sub0 = subset_epi_table(table, table.x > 0.0)
    r_sub0 = E.compute(t_sub0, {m: (head,) for m in EPI_BASE_METRICS})
    for m in EPI_BASE_METRICS:
        out[f'{m}_gt1'] = float(r_gt1.get(f'epi_{m}_{head}', np.nan))
        out[f'{m}_sub0'] = float(r_sub0.get(f'epi_{m}_{head}', np.nan))
    out['cell_mag_pooled_gt1'] = _pooled_cell_mag(t_gt1, head)
    out['cell_mag_pooled_sub0'] = _pooled_cell_mag(t_sub0, head)

    rm, rp = E.beyond_global(table, head)
    blk_m, blk_p, gt0_rhos = [], [], []
    for M in table.matrices():
        Rm, Rp, X = E._lay(M, rm), E._lay(M, rp), E._lay(M, table.x)
        ok = np.isfinite(Rm) & np.isfinite(Rp)
        if ok.sum() < E.MIN_PAIR_CELLS:
            continue
        Rm_ok, Rp_ok = np.where(ok, Rm, np.nan), np.where(ok, Rp, np.nan)
        blk = E._complete_block(Rm_ok)
        if blk is not None:
            ix = np.ix_(*blk)
            blk_m.extend(E._dc(Rm_ok[ix]).ravel())
            blk_p.extend(E._dc(Rp_ok[ix]).ravel())
        ok_u = ok & (X > 0.0)
        Rm_u, Rp_u = np.where(ok_u, Rm, np.nan), np.where(ok_u, Rp, np.nan)
        Cm_u, Cp_u = _dc_nan(Rm_u), _dc_nan(Rp_u)
        fin_u = np.isfinite(Cm_u) & np.isfinite(Cp_u)
        if fin_u.sum() >= 9:
            gt0_rhos.append(E._rho(Cp_u[fin_u], Cm_u[fin_u]))
    out['cell_mag_pooled'] = float(E._rho(blk_p, blk_m)) if len(blk_m) >= 9 else np.nan
    out['cell_mag_gt0'] = float(E._mean(gt0_rhos))
    return out


_FORK_STATE: dict = {}


def _worker_one_draw(task):
    counts, seed_weights_by_ns = task
    tables_map = _FORK_STATE['tables']
    models = _FORK_STATE['models']
    metric_names = _FORK_STATE['metric_names']
    extended = _FORK_STATE['extended']
    t0 = _FORK_STATE['ref_table']

    libs = np.unique(t0.lib)
    by = {lib_id: np.flatnonzero(t0.lib == lib_id) for lib_id in libs}
    idx_l, rep_l, k = [], [], 0
    for lib_id, c in zip(libs, counts):
        for _ in range(int(c)):
            idx_l.append(by[lib_id])
            rep_l.append(np.full(len(by[lib_id]), k, dtype=int))
            k += 1
    idx = np.concatenate(idx_l) if idx_l else np.zeros(0, dtype=int)
    rep = np.concatenate(rep_l) if rep_l else np.zeros(0, dtype=int)

    res_cache = {}
    out_row = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    for mi, (_disp, key, head) in enumerate(models):
        sub_tabs = sorted([(s, t) for (k_, s), t in tables_map.items() if k_ == key], key=lambda x: x[0])
        ns = len(sub_tabs)
        sw = seed_weights_by_ns.get(ns, np.ones(ns, dtype=float))
        vals, weights = [], []
        for (s, t), w_s in zip(sub_tabs, sw):
            if w_s <= 0:
                continue
            ck = (key, s)
            if ck not in res_cache:
                res_cache[ck] = resample_epi_table(t, idx, rep)
            md = compute_epi_metrics(res_cache[ck], head=head, extended=extended)
            vals.append([md[m] for m in metric_names])
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


def boot_epistasis_models(
    tables_by_model_seed: dict[tuple[str, int], E.Table],
    models: list[tuple[str, str, str]],
    B: int = 1000,
    seed: int = 0,
    jobs: int = 1,
    extended: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Run joint domain-cloning + actual-``ns`` seed bootstrap across ``models = [(label, key, head), ...]``.

    Returns ``(summary_df, reps_array_B_x_M_x_K, metric_names)``.
    """
    metric_names = list(EPI_EXTENDED_METRICS if extended else EPI_BASE_METRICS)
    first_key = models[0][1]
    ref_table = next(t for (k, _s), t in sorted(tables_by_model_seed.items()) if k == first_key)
    libs = np.unique(ref_table.lib)
    n_libs = len(libs)
    distinct_ns = sorted({
        sum(1 for (k_, _s) in tables_by_model_seed if k_ == key)
        for _disp, key, _head in models
    })

    # Point estimates across seeds
    point_mat = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    sd_mat = np.full((len(models), len(metric_names)), np.nan, dtype=float)
    n_seeds_list = []
    for mi, (_disp, key, head) in enumerate(models):
        sub_tabs = [t for (k_, _s), t in sorted(tables_by_model_seed.items()) if k_ == key]
        n_seeds_list.append(len(sub_tabs))
        seed_vals = np.asarray(
            [[compute_epi_metrics(t, head=head, extended=extended)[m] for m in metric_names] for t in sub_tabs],
            dtype=float,
        )
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

    _FORK_STATE.clear()
    _FORK_STATE.update(tables=tables_by_model_seed, models=models, metric_names=metric_names, extended=extended, ref_table=ref_table)
    try:
        if jobs > 1 and B > 1:
            with get_context('fork').Pool(jobs) as pool:
                draws = pool.map(_worker_one_draw, tasks, chunksize=max(1, B // (jobs * 4)))
        else:
            draws = [_worker_one_draw(t) for t in tasks]
    finally:
        _FORK_STATE.clear()

    reps = np.asarray(draws, dtype=float)  # (B, n_models, n_metrics)
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
