"""Shared statistical tests, multiple-testing corrections, domain×seed bootstraps, and ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .boot import ci


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaN entries preserved in place)."""
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    flat = arr.ravel()
    ok = np.flatnonzero(np.isfinite(flat))
    if len(ok) == 0:
        return out
    p = flat[ok]
    m = len(p)
    order = np.argsort(p, kind='stable')
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    step = np.empty(m, dtype=float)
    step[order] = np.maximum.accumulate(adj)
    out.ravel()[ok] = step
    return out


def boot_pvalue(draws, null: float = 0.0, alternative: str = 'two-sided') -> float:
    """Empirical bootstrap p-value against ``null`` with ``1 / (B + 1)`` floor.

    - ``'two-sided'``: ``max(2 * min(P(d <= null), P(d >= null)), 1 / (B + 1))``
    - ``'greater'`` (H1: theta > null, testing H0: theta <= null): ``max(P(d <= null), 1 / (B + 1))``
    - ``'less'`` (H1: theta < null, testing H0: theta >= null): ``max(P(d >= null), 1 / (B + 1))``
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    floor = 1.0 / (len(d) + 1)
    le = float((d <= null).mean())
    ge = float((d >= null).mean())
    if alternative == 'two-sided':
        return float(max(min(1.0, 2.0 * min(le, ge)), floor))
    if alternative == 'greater':
        return float(max(le, floor))
    if alternative == 'less':
        return float(max(ge, floor))
    raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")


def fmt_p(p: float, B: int | None = None, prefix: str = 'p = ') -> str:
    """Format a p-value consistently across tables and figure annotations."""
    if p is None or not np.isfinite(p):
        return 'n/a'
    if B is not None and p <= 1.0 / (B + 1) + 1e-15:
        bound = 1.0 / (B + 1)
        p_str = f'{bound:.1e}' if bound < 1e-3 else f'{bound:.3g}'
        return f'{prefix.replace("=", "<")}{p_str}' if '=' in prefix else f'< {p_str}'
    if p < 1e-4:
        return f'{prefix}{p:.1e}' if prefix else f'{p:.1e}'
    if p < 1e-3:
        return f'{prefix.replace("=", "<")}0.001' if '=' in prefix else '< 0.001'
    if p < 0.1:
        return f'{prefix}{p:.3f}'
    return f'{prefix}{p:.2f}'


def seed_sd(values) -> float:
    """Sample standard deviation (``ddof=1``) across replicate model seeds; ``NaN`` when fewer than 2 finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    return float(np.std(v, ddof=1))


def paired_boot_test(
    reps_a,
    reps_b,
    point_a: float | None = None,
    point_b: float | None = None,
    level: float = 0.95,
    alternative: str = 'two-sided',
) -> dict[str, float]:
    """Paired bootstrap difference ``a - b`` using point-estimate delta when ``point_a`` and ``point_b`` are supplied."""
    ra = np.asarray(reps_a, dtype=float)
    rb = np.asarray(reps_b, dtype=float)
    d = ra - rb
    fin = d[np.isfinite(d)]
    if len(fin) == 0:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, se=np.nan, p=np.nan, p_two=np.nan, p_one_gt=np.nan, n_boot=0)
    lo, hi = ci(fin, level=level)
    delta = (float(point_a) - float(point_b)) if (point_a is not None and point_b is not None) else float(fin.mean())
    p_two = boot_pvalue(fin, null=0.0, alternative='two-sided')
    p_gt = boot_pvalue(fin, null=0.0, alternative='greater')
    p_main = boot_pvalue(fin, null=0.0, alternative=alternative)
    se = float(np.std(fin, ddof=1)) if len(fin) > 1 else np.nan
    return dict(
        delta=delta,
        lo=float(lo),
        hi=float(hi),
        se=se,
        p=p_main,
        p_two=p_two,
        p_one_gt=p_gt,
        n_boot=int(len(fin)),
    )


def make_domain_seed_draws(
    n_domains: int,
    B: int = 1000,
    seed: int = 0,
    seed_counts: tuple[int, ...] = (2, 3),
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Pre-draw shared ``(B, n_domains)`` domain multiplicities and ``{ns: (B, ns)}`` seed weights for paired comparisons."""
    rng = np.random.default_rng(seed)
    dom_counts = rng.multinomial(n_domains, np.full(n_domains, 1.0 / n_domains), size=B).astype(float) if n_domains > 0 else np.zeros((B, 0))
    seed_w: dict[int, np.ndarray] = {}
    # Draw ns=3 first so standard triplicate RNG streams stay stable across callers
    ordered_ns = [3] + [ns for ns in seed_counts if ns != 3 and ns > 1]
    for ns in ordered_ns:
        seed_w[ns] = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(float) / ns
    return dom_counts, seed_w


def boot_domain_seeds(
    values,
    B: int = 1000,
    seed: int = 0,
    counts: np.ndarray | None = None,
    seed_weights: dict[int, np.ndarray] | np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Joint ``[domain × seed]`` multinomial bootstrap on a ``(n_domains,)`` or ``(n_domains, n_seeds)`` array/DataFrame.

    Returns ``(point_mean, boot_reps)`` where ``point_mean`` is the mean over domains of the per-domain seed mean,
    and ``boot_reps`` has shape ``(B,)`` with both domains and seeds resampled jointly.
    """
    arr = values.to_numpy(dtype=float) if isinstance(values, (pd.DataFrame, pd.Series)) else np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f'Expected 1D or 2D array (n_domains[, n_seeds]), got shape {arr.shape}')
    n_dom, n_seeds = arr.shape
    if n_dom == 0:
        return np.nan, np.full(B, np.nan)

    with np.errstate(invalid='ignore'):
        per_dom = np.nanmean(arr, axis=1)
    point = float(np.nanmean(per_dom)) if np.isfinite(per_dom).any() else np.nan

    if counts is None or (n_seeds > 1 and seed_weights is None):
        c_gen, sw_gen = make_domain_seed_draws(n_dom, B=B, seed=seed, seed_counts=(n_seeds,))
        if counts is None:
            counts = c_gen
        if seed_weights is None:
            seed_weights = sw_gen
    else:
        B = len(counts)

    if n_seeds <= 1:
        sw = np.ones((B, 1), dtype=float)
    elif isinstance(seed_weights, dict):
        if n_seeds not in seed_weights:
            rng_s = np.random.default_rng(seed + 997 + n_seeds)
            sw = rng_s.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=B).astype(float) / n_seeds
        else:
            sw = np.asarray(seed_weights[n_seeds], dtype=float)
    else:
        sw_arr = np.asarray(seed_weights, dtype=float)
        if sw_arr.ndim == 2 and sw_arr.shape == (B, n_seeds):
            sw = sw_arr / np.maximum(sw_arr.sum(axis=1, keepdims=True), 1e-12)
        else:
            rng_s = np.random.default_rng(seed + 997 + n_seeds)
            sw = rng_s.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=B).astype(float) / n_seeds

    fin = np.isfinite(arr)
    arr_zero = np.where(fin, arr, 0.0)
    # Per replicate b and domain d: seed-weighted average over finite seeds of domain d
    num_dom = np.einsum('bs,ds->bd', sw, arr_zero)
    den_dom = np.einsum('bs,ds->bd', sw, fin.astype(float))
    valid_dom = (den_dom > 0) & (counts > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        dom_est = np.where(den_dom > 0, num_dom / den_dom, 0.0)
        w_eff = np.where(valid_dom, counts, 0.0)
        reps = np.sum(w_eff * dom_est, axis=1) / np.sum(w_eff, axis=1)
    reps[np.sum(w_eff, axis=1) == 0] = np.nan
    return point, reps


def ndcg_at_k(pred, y, k: int | None = 96, mode: str = 'rectified') -> float:
    """Compute NDCG@k with ``'rectified'`` (``max(y, 0)``) or ``'shifted'`` (``y - min(y)``) relevance."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(t) < 2:
        return np.nan
    if mode == 'rectified':
        rel = np.where(t > 0.0, t, 0.0)
        if not (rel > 0.0).any():
            return np.nan
    elif mode == 'shifted':
        ymin = float(t.min())
        rel = t - ymin
        if not (rel > 0.0).any():
            return np.nan
    else:
        raise ValueError(f"mode must be 'rectified' or 'shifted', got {mode!r}")
    kk = min(int(k), len(rel)) if k is not None else None
    return float(ndcg_score(rel[None, :], p[None, :], k=kk, ignore_ties=True))
