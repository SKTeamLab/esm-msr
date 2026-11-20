"""Protein-level and pooled variant-screening metrics with exact random/oracle baselines and fixed-cohort K-curves.

Conventions:
- ``y`` is stabilizing-positive ddG (zero = wild type); a *hit* has ``y > tau`` (default ``0.5`` kcal/mol).
- In per-protein screening (``protein_screen``, ``per_protein_table``, ``summarize``, ``boot_summary``),
  ``gain`` is **rectified gain @ K**: ``sum_{i in sel} max(y_i, 0)``.
- In ``pooled_screen``, both ``mean_y`` (unrectified mean ``y`` of the selected variants, also aliased as
  ``gain`` for backwards compatibility with ``08_screening``) and ``rectified_gain`` (rectified sum
  ``sum_{i in sel} max(y_i, 0)``) are returned explicitly, and ties in ``score`` are broken deterministically
  identically to ``protein_screen``.
- For $K$-sweeps (``K = 1 .. K_max``), pass ``keep_groups=eligible_proteins(groups, score, y, K_max)`` or call
  ``per_protein_curve(...)`` so the exact same protein cohort is evaluated at every budget ``K``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln


def _top_idx(score: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(score))
    if k <= 0:
        return np.array([], dtype=int)
    return np.argpartition(-score, k - 1)[:k]


def _jittered_score(score: np.ndarray) -> np.ndarray:
    n = len(score)
    if n == 0:
        return score
    return score + 1e-12 * np.arange(n, dtype=float) / n


def protein_screen(score, y, K: int, tau: float = 0.5, rng=None) -> dict[str, float | int] | None:
    """Compute screening metrics and exact random/oracle baselines on a single protein."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y = score[ok], y[ok]
    n = len(y)
    if n < 2:
        return None
    k = min(int(K), n)
    sel = _top_idx(_jittered_score(score), k)
    ys = y[sel]
    hits = int((y > tau).sum())

    ysort = np.sort(y)
    j = np.arange(1, n + 1)
    logc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    mask = j >= k
    logp = np.full(n, -np.inf)
    logp[mask] = logc(j[mask] - 1, k - 1) - logc(n, k)
    rand_best = float(np.sum(np.exp(logp) * ysort))
    p_rand = 1.0 - (float(np.exp(logc(n - hits, k) - logc(n, k))) if n - hits >= k else 0.0)

    rect = np.maximum(y, 0.0)
    gain = float(rect[sel].sum())
    oracle_gain = float(np.sort(rect)[-k:].sum())
    rand_gain = float(k * rect.mean())

    return dict(
        hit=float((ys > tau).mean()),
        rec=float((ys > tau).any()),
        best=float(ys.max()),
        oracle=float(ysort[-1]),
        gain=gain,
        oracle_gain=oracle_gain,
        rand_gain=rand_gain,
        oracle_hit=float(min(hits, k) / k),
        oracle_rec=float(hits > 0),
        rand_best=rand_best,
        rand_rec=float(p_rand),
        rand_hit=hits / n,
        hits=hits,
        n=n,
    )


def eligible_proteins(groups, score, y, max_K: int, min_variants: int | None = None) -> set:
    """Return the fixed set of proteins having ``>= min_variants`` (default ``max(3, max_K // 2)``) valid variants."""
    req = max(3, int(max_K) // 2) if min_variants is None else int(min_variants)
    sc = np.asarray(score, dtype=float)
    if sc.ndim == 2:
        ok_sc = np.isfinite(sc).all(axis=0)
    else:
        ok_sc = np.isfinite(sc)
    ok = ok_sc & np.isfinite(np.asarray(y, dtype=float))
    g = pd.Series(np.asarray(groups)[ok])
    vc = g.value_counts()
    return set(vc.index[vc >= req])


def per_protein_table(
    groups,
    score,
    y,
    K: int,
    tau: float = 0.5,
    min_variants: int | None = None,
    keep_groups: set | None = None,
) -> pd.DataFrame:
    """Build per-protein screening table at budget ``K`` (pass ``keep_groups`` or ``min_variants`` to lock cohort across ``K``)."""
    req = max(3, int(K) // 2) if (min_variants is None and keep_groups is None) else (int(min_variants) if min_variants is not None else 2)
    rows = []
    order = np.argsort(groups, kind='stable')
    g = np.asarray(groups)[order]
    sc = np.asarray(score, dtype=float)[order]
    yy = np.asarray(y, dtype=float)[order]
    cuts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        grp = g[a]
        if keep_groups is not None and grp not in keep_groups:
            continue
        r = protein_screen(sc[a:b], yy[a:b], K, tau)
        if r is not None and r['n'] >= req:
            r['group'] = grp
            rows.append(r)
    return pd.DataFrame(rows)


def per_protein_curve(groups, score_seeds, y, Ks, tau: float = 0.5, B: int = 1000, seed: int = 0,
                      only_recoverable: bool = True, min_variants: int | None = None) -> pd.DataFrame:
    """Evaluate a screening curve across budgets ``Ks`` on a single fixed protein cohort across all ``K``."""
    Ks = [int(k) for k in Ks]
    arr = np.atleast_2d(np.asarray(score_seeds, dtype=float))
    cohort = eligible_proteins(groups, arr, y, max(Ks), min_variants=min_variants)
    rows = []
    for k in Ks:
        tabs = [per_protein_table(groups, s, y, k, tau=tau, keep_groups=cohort) for s in arr]
        bs = boot_summary(tabs, B=B, seed=seed, only_recoverable=only_recoverable)
        n_pro = len(tabs[0]) if tabs else 0
        n_rec = int((tabs[0]['hits'] > 0).sum()) if n_pro else 0
        for m, (val, lo, hi) in bs.items():
            rows.append(dict(K=k, metric=m, value=val, lo=lo, hi=hi, n_proteins=n_pro, n_recoverable=n_rec))
    return pd.DataFrame(rows)


def summarize(tab: pd.DataFrame, only_recoverable: bool = True) -> dict[str, float | int]:
    """Average per-protein metrics over ``tab`` (``p_recover`` over proteins with ``hits > 0`` when ``only_recoverable``)."""
    if len(tab) == 0:
        return {}
    rec = tab[tab['hits'] > 0] if only_recoverable else tab
    return dict(
        hit_rate=float(tab['hit'].mean()),
        p_recover=float(rec['rec'].mean()) if len(rec) else np.nan,
        best=float(tab['best'].mean()),
        oracle=float(tab['oracle'].mean()),
        oracle_hit=float(tab['oracle_hit'].mean()),
        oracle_rec=float(rec['oracle_rec'].mean()) if len(rec) else np.nan,
        rand_best=float(tab['rand_best'].mean()),
        rand_rec=float(rec['rand_rec'].mean()) if len(rec) else np.nan,
        rand_hit=float(tab['rand_hit'].mean()),
        gain=float(tab['gain'].mean()),
        oracle_gain=float(tab['oracle_gain'].mean()),
        rand_gain=float(tab['rand_gain'].mean()),
        n_proteins=len(tab),
        n_recoverable=len(rec),
    )


def boot_summary(tabs, B: int = 1000, seed: int = 0, only_recoverable: bool = True) -> dict[str, tuple[float, float, float]]:
    """Joint protein-level and seed-level bootstrap of ``summarize(tabs)`` -> ``{metric: (mean, lo, hi)}``."""
    if not isinstance(tabs, (list, tuple)):
        tabs = [tabs]
    rng = np.random.default_rng(seed)
    rng_s = np.random.default_rng(seed + 17)
    n = len(tabs[0])
    n_seeds = len(tabs)
    if n == 0:
        return {}
    cols = ('hit', 'rec', 'best', 'oracle', 'oracle_hit', 'oracle_rec', 'rand_best', 'rand_rec', 'rand_hit', 'gain', 'oracle_gain', 'rand_gain')
    arrs = {m: np.column_stack([t[m].to_numpy() for t in tabs]) for m in cols}
    has = tabs[0]['hits'].to_numpy() > 0
    out = {
        m: [] for m in (
            'hit_rate', 'p_recover', 'best', 'oracle', 'oracle_hit', 'oracle_rec',
            'rand_best', 'rand_rec', 'rand_hit', 'gain', 'oracle_gain', 'rand_gain',
        )
    }
    for _ in range(B):
        c = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        sw = rng_s.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds)).astype(float) / n_seeds if n_seeds > 1 else np.ones(1)
        cw = c * (has if only_recoverable else 1.0)
        out['hit_rate'].append(np.dot(c, arrs['hit'] @ sw) / n)
        out['p_recover'].append(np.dot(cw, arrs['rec'] @ sw) / max(cw.sum(), 1))
        out['oracle_rec'].append(np.dot(cw, arrs['oracle_rec'] @ sw) / max(cw.sum(), 1))
        out['rand_rec'].append(np.dot(cw, arrs['rand_rec'] @ sw) / max(cw.sum(), 1))
        for m in ('best', 'oracle', 'oracle_hit', 'rand_best', 'rand_hit', 'gain', 'oracle_gain', 'rand_gain'):
            out[m].append(np.dot(c, arrs[m] @ sw) / n)
    sums = [summarize(t, only_recoverable) for t in tabs]
    res = {}
    for m, v in out.items():
        lo, hi = np.quantile(v, [0.025, 0.975])
        res[m] = (float(np.mean([s[m] for s in sums])), float(lo), float(hi))
    return res


def pooled_screen(score, y, clusters, frac: float | None = 0.1, tau: float = 0.5,
                  B: int = 1000, seed: int = 0, K: int | None = None) -> dict:
    """Pooled screen over an entire benchmark set with cluster-level protein bootstrap.

    Returns ``{metric: (value, lo, hi)}`` for:
      - ``hit_rate``: fraction of selected variants with ``y > tau``
      - ``best``: maximum measured ``y`` among selected variants
      - ``base_rate``: fraction of all variants in the pool with ``y > tau``
      - ``mean_y`` (also aliased as ``gain``): unrectified mean ``y`` of selected variants
      - ``rectified_gain``: rectified sum ``sum_{i in sel} max(y_i, 0)`` (and ``rectified_gain_mean`` per pick)
    Plus ``reps`` (shape ``(B, 6)``) and ``base`` (tuple of 6 point values).
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y, cl = score[ok], y[ok], np.asarray(clusters)[ok]
    u, ci_ = np.unique(cl.astype(str), return_inverse=True)
    n = len(y)
    order = np.argsort(-_jittered_score(score), kind='stable')
    hit = (y > tau).astype(float)
    rect = np.maximum(y, 0.0)

    def run(w):
        cw = np.cumsum(w[order])
        tot = cw[-1]
        if tot <= 0:
            return (np.nan,) * 6
        kk = float(K) if K is not None else float(frac) * tot
        sel = (cw - w[order]) < kk
        ws = w[order] * sel
        wsum = ws.sum()
        if wsum <= 0:
            return (np.nan,) * 6
        hr = float(np.sum(ws * hit[order]) / wsum)
        yb = y[order][sel & (w[order] > 0)]
        best_y = float(yb.max()) if len(yb) else np.nan
        base_r = float(np.sum(w * hit) / tot)
        mean_y = float(np.sum(ws * y[order]) / wsum)
        rect_sum = float(np.sum(ws * rect[order]))
        rect_mean = float(rect_sum / wsum)
        return hr, best_y, base_r, mean_y, rect_sum, rect_mean

    base = run(np.ones(n, dtype=float))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        c = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u))).astype(float)
        reps.append(run(c[ci_]))
    reps_arr = np.asarray(reps, dtype=float)
    metric_names = ('hit_rate', 'best', 'base_rate', 'mean_y', 'rectified_gain', 'rectified_gain_mean')
    out: dict = {}
    for j, m in enumerate(metric_names):
        lo, hi = np.nanquantile(reps_arr[:, j], [0.025, 0.975])
        out[m] = (float(base[j]), float(lo), float(hi))
    out['gain'] = out['mean_y']  # backwards-compatible alias used by 08_screening ('mean ddG of selection')
    out['reps'] = reps_arr
    out['base'] = base
    return out
