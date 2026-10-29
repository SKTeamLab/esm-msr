"""Screening statistics: what a protein engineer gets from a fixed budget of tested variants.

Convention: ``y`` is ddG with positive = stabilizing; a *hit* is a variant with ``y > tau`` (default 0.5 kcal/mol).

For a pool of variants and a selection of the ``K`` best-scored (or the top fraction):

``hit_rate``        fraction of the selected variants that are hits (precision at K)
``p_recover``       probability that a screen of K variants contains at least one hit.  Per protein, among proteins that have
                    at least one hit at all ("recoverable" proteins), averaged over proteins
``best_ddg``        the best measured ddG among the K selected, averaged over proteins; ``oracle`` is the best any K could do
                    (the top-K of the measured values) and ``random`` the expectation of K random picks, so that
                    ``best_ddg - random`` and ``oracle - best_ddg`` read as gain and regret

Everything is computed per protein (one screen per protein) unless ``pooled`` is asked for (one screen over a whole set, K a
fraction of the set).  Proteins are the resampling unit of every interval.
"""
import numpy as np


def _top_idx(score, k):
    k = min(k, len(score))
    if k <= 0:
        return np.array([], int)
    idx = np.argpartition(-score, k - 1)[:k]
    return idx


def protein_screen(score, y, K, tau=0.5, rng=None):
    """One protein: (hit_rate, recovered(0/1), best_ddg, oracle_best, random_best, n_hits, n)."""
    score = np.asarray(score, float); y = np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y = score[ok], y[ok]
    n = len(y)
    if n < 2:
        return None
    k = min(K, n)
    sel = _top_idx(score + 1e-12 * np.arange(n) / n, k)
    ys = y[sel]
    hits = int((y > tau).sum())
    # expected best of k random picks: exact from the order statistics of y
    ysort = np.sort(y)
    j = np.arange(1, n + 1)
    # P(max of k random picks is the j-th smallest) = C(j-1, k-1) / C(n, k)
    from scipy.special import gammaln
    logc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    mask = j >= k
    logp = np.full(n, -np.inf)
    logp[mask] = logc(j[mask] - 1, k - 1) - logc(n, k)
    rand_best = float(np.sum(np.exp(logp) * ysort))
    # chance of at least one hit among k random picks
    p_rand = 1.0 - (np.exp(logc(n - hits, k) - logc(n, k)) if n - hits >= k else 0.0)
    return dict(hit=float((ys > tau).mean()), rec=float((ys > tau).any()), best=float(ys.max()), oracle=float(ysort[-k:].max()),
                rand_best=rand_best, rand_rec=float(p_rand), rand_hit=hits / n, hits=hits, n=n)


def per_protein_table(groups, score, y, K, tau=0.5):
    """DataFrame-like list of dicts, one per protein with at least K/2 (>= 3) valid variants."""
    import pandas as pd
    rows = []
    order = np.argsort(groups, kind='stable')
    g = np.asarray(groups)[order]
    sc = np.asarray(score)[order]; yy = np.asarray(y)[order]
    cuts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        r = protein_screen(sc[a:b], yy[a:b], K, tau)
        if r is not None and r['n'] >= max(3, K // 2):
            r['group'] = g[a]
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(tab, only_recoverable=True):
    """Averages over proteins of a per_protein_table; ``p_recover`` over proteins with a hit."""
    if len(tab) == 0:
        return {}
    rec = tab[tab['hits'] > 0] if only_recoverable else tab
    return dict(hit_rate=tab['hit'].mean(), p_recover=rec['rec'].mean() if len(rec) else np.nan, best=tab['best'].mean(),
                oracle=tab['oracle'].mean(), rand_best=tab['rand_best'].mean(), rand_rec=rec['rand_rec'].mean() if len(rec) else np.nan,
                rand_hit=tab['rand_hit'].mean(), n_proteins=len(tab), n_recoverable=len(rec))


def boot_summary(tab, B=1000, seed=0, only_recoverable=True):
    """Protein-level bootstrap of ``summarize``: dict metric -> (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(tab)
    if n == 0:
        return {}
    arrs = {m: tab[m].to_numpy() for m in ('hit', 'rec', 'best', 'oracle', 'rand_best', 'rand_rec', 'rand_hit')}
    has = tab['hits'].to_numpy() > 0
    out = {m: [] for m in ('hit_rate', 'p_recover', 'best', 'oracle', 'rand_best', 'rand_rec', 'rand_hit')}
    for _ in range(B):
        c = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        cw = c * (has if only_recoverable else 1.0)
        out['hit_rate'].append(np.dot(c, arrs['hit']) / n)
        out['p_recover'].append(np.dot(cw, arrs['rec']) / max(cw.sum(), 1))
        out['rand_rec'].append(np.dot(cw, arrs['rand_rec']) / max(cw.sum(), 1))
        for m in ('best', 'oracle', 'rand_best', 'rand_hit'):
            out[m].append(np.dot(c, arrs[m]) / n)
    s = summarize(tab, only_recoverable)
    res = {}
    for m, v in out.items():
        lo, hi = np.quantile(v, [0.025, 0.975])
        res[m] = (float(s[m]), float(lo), float(hi))
    return res


def pooled_screen(score, y, clusters, frac, tau=0.5, B=1000, seed=0, K=None):
    """One screen over the whole set: select the top ``frac`` of the set by score (or the top ``K``); the cluster bootstrap
    (whole proteins with replacement, selection recomputed on each replicate) gives intervals.
    Returns {metric: (value, lo, hi)} for hit_rate, best (best measured ddG among the selected) and the random expectation."""
    score = np.asarray(score, float); y = np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y, cl = score[ok], y[ok], np.asarray(clusters)[ok]
    u, ci_ = np.unique(cl.astype(str), return_inverse=True)
    n = len(y)
    order = np.argsort(-score, kind='stable')
    hit = (y > tau).astype(float)

    def run(w):
        cw = np.cumsum(w[order])
        tot = cw[-1]
        kk = K if K is not None else frac * tot
        sel = (cw - w[order]) < kk
        ws = w[order] * sel
        hr = np.sum(ws * hit[order]) / ws.sum()
        yb = y[order][sel & (w[order] > 0)]
        gain = np.sum(ws * y[order]) / ws.sum()
        return hr, (yb.max() if len(yb) else np.nan), np.sum(w * hit) / tot, gain

    base = run(np.ones(n))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        c = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u))).astype(float)
        reps.append(run(c[ci_]))
    reps = np.array(reps)
    out = {}
    for j, m in enumerate(('hit_rate', 'best', 'base_rate', 'gain')):
        lo, hi = np.nanquantile(reps[:, j], [0.025, 0.975])
        out[m] = (float(base[j]), float(lo), float(hi))
    out['reps'] = reps
    out['base'] = base
    return out
