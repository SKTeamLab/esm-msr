"""Count-weighted Spearman's rho bootstrap engine across protein, site, and mutation units.

A bootstrap replicate is a vector of non-negative integer row multiplicities. Mid-ranks with tie averaging
are computed directly from multiplicities in O(n) without re-sorting, matching ``scipy.stats.spearmanr`` on
the expanded sample to machine precision.
"""
from __future__ import annotations

import numpy as np


class _Ranks:
    """Mid-ranks of ``v`` within groups ``g`` under integer row multiplicities."""

    def __init__(self, v, g):
        order = np.lexsort((v, g))
        vs, gs = v[order], g[order]
        n = len(v)
        newtie = np.ones(n, bool)
        newtie[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
        newgrp = np.ones(n, bool)
        newgrp[1:] = gs[1:] != gs[:-1]
        self.order = order
        self.tie_start = np.flatnonzero(newtie)
        self.tie_id = np.cumsum(newtie) - 1
        self.base_idx = np.flatnonzero(newgrp)[np.cumsum(newgrp) - 1]
        inv = np.empty(n, np.int64)
        inv[order] = np.arange(n)
        self.inv = inv

    def ranks(self, w):
        ws = w[self.order]
        before = np.cumsum(ws) - ws
        tie_T = np.add.reduceat(ws, self.tie_start)
        tie_before = before[self.tie_start] - before[self.base_idx[self.tie_start]]
        mid = tie_before + (tie_T + 1.0) / 2.0
        return mid[self.tie_id][self.inv]


class SpearmanEngine:
    """Spearman's rho of ``(x, y)`` per group under row multiplicities."""

    def __init__(self, y, g, n_groups):
        self.g = g
        self.G = n_groups
        self.ry = _Ranks(y, g)

    def _wcorr(self, rx, ry, w):
        g, G = self.g, self.G
        sw = np.bincount(g, w, G)
        with np.errstate(invalid='ignore', divide='ignore'):
            mx = np.bincount(g, w * rx, G) / sw
            my = np.bincount(g, w * ry, G) / sw
            cxy = np.bincount(g, w * rx * ry, G) / sw - mx * my
            vx = np.bincount(g, w * rx * rx, G) / sw - mx * mx
            vy = np.bincount(g, w * ry * ry, G) / sw - my * my
            r = cxy / np.sqrt(vx * vy)
        r[(sw < 3) | ~np.isfinite(r)] = np.nan
        return r

    def prepare(self, x):
        return _Ranks(x, self.g)

    def rho(self, xr, w, ry=None):
        return self._wcorr(xr.ranks(w), self.ry.ranks(w) if ry is None else ry, w)


def _int_codes(values):
    u, inv = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    return inv.astype(np.int64), u


def global_cluster_index(frames_clusters):
    """Map every cluster name across ``frames_clusters`` (``{dataset: array}``) to a shared integer index."""
    allnames = {}
    out = {}
    for ds, names in frames_clusters.items():
        idx = np.empty(len(names), np.int64)
        for i, nm in enumerate(np.asarray(names).astype(str)):
            idx[i] = allnames.setdefault(nm, len(allnames))
        out[ds] = idx
    return out, len(allnames)


def _resolve_seed_weights(seed_w, ns: int) -> np.ndarray | None:
    """Return length-``ns`` non-negative seed weights summing to > 0, or None when unweighted."""
    if seed_w is None or ns <= 1:
        return None
    if callable(seed_w):
        return np.asarray(seed_w(ns), dtype=float)
    if isinstance(seed_w, dict):
        sw = seed_w.get(ns)
        return np.asarray(sw, dtype=float) if sw is not None else None
    arr = np.asarray(seed_w, dtype=float)
    if len(arr) == ns and arr.sum() > 0:
        return arr
    return None


class DatasetBoot:
    """Prepared rankers, valid-row masks, and cluster assignments for bootstrapping one ``Frame``."""

    def __init__(self, frame, keys, per_protein: bool = False, common: bool = False, subset=None):
        sp = frame.spine
        self.frame = frame
        self.per_protein = per_protein
        base = sp['use'].to_numpy() & np.isfinite(sp['y'].to_numpy())
        if subset is not None:
            base &= subset
        self.base = base
        keys = [k for k in keys if k in frame.preds]
        valid = {}
        for k in keys:
            valid[k] = base & np.isfinite(frame.preds[k]).all(axis=0)
        if common and keys:
            m = np.logical_and.reduce([valid[k] for k in keys])
            valid = {k: m for k in keys}
        self.keys = [k for k in keys if valid[k].sum() >= 10]
        self.valid = valid
        self.n_rows = len(sp)
        self.groups_all = _int_codes(sp['group'].to_numpy())[0] if per_protein else np.zeros(self.n_rows, np.int64)
        self._engines = {}
        self._xr = {}
        for k in self.keys:
            m = valid[k]
            mk = m.tobytes()
            if mk not in self._engines:
                idx = np.flatnonzero(m)
                g, _ = _int_codes(self.groups_all[idx]) if per_protein else (np.zeros(len(idx), np.int64), None)
                G = g.max() + 1 if len(g) else 1
                self._engines[mk] = (idx, g, G, SpearmanEngine(sp['y'].to_numpy()[idx], g, G))
            idx, g, G, eng = self._engines[mk]
            self._xr[k] = [eng.prepare(p[idx]) for p in frame.preds[k]]
        self.cluster_names = {
            'protein': sp['cluster'].to_numpy(),
            'site': (sp['cluster'].astype(str) + '#' + sp['site1'].astype(str)).to_numpy(),
        }

    def point(self, k: str):
        """Return ``(seed_mean_rho, per_seed_rhos, n_rows, n_proteins)``."""
        m = self.valid[k]
        idx, g, G, eng = self._engines[m.tobytes()]
        w = np.ones(len(idx))
        per_seed = []
        for xr in self._xr[k]:
            r = eng.rho(xr, w)
            per_seed.append(np.nanmean(r) if self.per_protein else r[0])
        return float(np.mean(per_seed)), np.array(per_seed), int(len(idx)), int(G if self.per_protein else self.n_clusters(k))

    def n_clusters(self, k: str) -> int:
        idx = self._engines[self.valid[k].tobytes()][0]
        return len(np.unique(self.cluster_names['protein'][idx]))

    def replicate(self, w_full, seed_w=None) -> dict[str, float]:
        """Compute Spearman rho of every model under row multiplicities ``w_full`` and seed weights ``seed_w``."""
        out = {}
        ycache = {}
        for k in self.keys:
            mk = self.valid[k].tobytes()
            idx, g, G, eng = self._engines[mk]
            w = w_full[idx]
            if mk not in ycache:
                ycache[mk] = eng.ry.ranks(w)
            rs = np.array([
                np.nanmean(eng.rho(xr, w, ycache[mk])) if self.per_protein else eng.rho(xr, w, ycache[mk])[0]
                for xr in self._xr[k]
            ], dtype=float)
            sw = _resolve_seed_weights(seed_w, len(rs))
            if sw is not None:
                ok = np.isfinite(rs) & (sw > 0)
                out[k] = float(np.dot(sw[ok], rs[ok]) / sw[ok].sum()) if ok.any() else np.nan
            else:
                out[k] = float(np.nanmean(rs))
        return out

    def per_protein_rho(self, k: str):
        """Return ``(protein_names, seed_mean_per_protein_rhos)``."""
        names, r_seeds = self.per_protein_rho_seeds(k)
        return names, np.nanmean(r_seeds, axis=0)

    def per_protein_rho_seeds(self, k: str):
        """Return ``(protein_names, per_seed_per_protein_rhos)`` of shape ``(n_seeds, G)``."""
        idx, g, G, eng = self._engines[self.valid[k].tobytes()]
        w = np.ones(len(idx))
        r_seeds = np.array([eng.rho(xr, w) for xr in self._xr[k]], dtype=float)
        names = np.empty(G, object)
        names[g] = self.frame.spine['group'].to_numpy()[idx]
        return names, r_seeds


def draw_weights(kind: str, boot: DatasetBoot, rng, cluster_counts=None, cluster_idx=None) -> np.ndarray:
    """Draw row multiplicities for one bootstrap replicate ('protein', 'site', or 'mutation')."""
    n = boot.n_rows
    if kind in ('protein', 'site'):
        if cluster_counts is None or cluster_idx is None:
            idx, _ = _int_codes(boot.cluster_names[kind])
            ncl = int(idx.max() + 1) if len(idx) else 0
            counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)) if ncl else np.zeros(0)
            return counts[idx].astype(np.float64) if ncl else np.zeros(n, dtype=np.float64)
        return cluster_counts[cluster_idx].astype(np.float64)
    if kind == 'mutation':
        base = np.flatnonzero(boot.base)
        w = np.zeros(n)
        if boot.per_protein:
            g = boot.groups_all[base]
            order = np.argsort(g, kind='stable')
            gs = g[order]
            starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
            sizes = np.diff(np.r_[starts, len(gs)])
            first = np.repeat(starts, sizes)
            size = np.repeat(sizes, sizes)
            pick = first + np.floor(rng.random(len(gs)) * size).astype(np.int64)
            sel = base[order][pick]
        else:
            sel = base[rng.integers(0, len(base), len(base))]
        np.add.at(w, sel, 1.0)
        return w
    raise ValueError(f"Unknown bootstrap kind {kind!r}; expected 'protein', 'site', or 'mutation'")


def _make_seed_sampler(rng):
    """Draw ``multinomial(ns, [1/ns]*ns)`` keyed by actual ``ns``, pre-seeding ``ns=3`` first for determinism."""
    cache = {3: rng.multinomial(3, np.full(3, 1.0 / 3.0)).astype(float)}

    def get(ns: int) -> np.ndarray:
        if ns <= 1:
            return np.ones(1, dtype=float)
        if ns not in cache:
            cache[ns] = rng.multinomial(ns, np.full(ns, 1.0 / ns)).astype(float)
        return cache[ns]

    return get


def bootstrap_rho(frames, keys, kind: str = 'protein', B: int = 1000, seed: int = 0,
                  per_protein: bool | None = None, common: bool = False, subsets=None):
    """Bootstrap seed-mean Spearman rho for every model on every dataset in ``frames``.

    - ``kind='protein'``: resamples whole protein clusters (using exact per-protein rho aggregation on
      ``per_protein`` datasets, and shared global cluster draws across pooled datasets).
    - ``kind='site'``: resamples ``(protein, site1)`` clusters via ``SpearmanEngine`` on all datasets
      (including ``per_protein=True`` datasets, never falling through to whole-protein resampling).
    - ``kind='mutation'``: resamples individual variants (stratified within each protein when ``per_protein=True``).
    - Multi-seed models resample ``multinomial(ns, 1/ns)`` keyed by each model's actual ``ns`` (never slicing
      ``multinomial(3)`` into an all-zero vector when ``ns=2`` or skipping seed resampling when ``ns>3``).
    """
    if kind not in ('protein', 'site', 'mutation'):
        raise ValueError(f"kind must be 'protein', 'site', or 'mutation', got {kind!r}")
    from .registry import PER_PROTEIN
    rng = np.random.default_rng(seed)
    boots = {}
    for ds, fr in frames.items():
        pp = (ds in PER_PROTEIN) if per_protein is None else per_protein
        sub = None if subsets is None else subsets.get(ds)
        boots[ds] = DatasetBoot(fr, keys, per_protein=pp, common=common, subset=sub)
    point = {ds: {k: b.point(k) for k in b.keys} for ds, b in boots.items()}
    reps = {ds: {k: np.full(B, np.nan) for k in b.keys} for ds, b in boots.items()}

    if kind in ('protein', 'site'):
        # Exact pre-computed per-protein rho shortcut applies ONLY when resampling whole proteins (kind == 'protein')
        use_pp_shortcut = {ds: (kind == 'protein' and b.per_protein) for ds, b in boots.items()}
        names = {ds: b.cluster_names[kind] for ds, b in boots.items() if not use_pp_shortcut[ds]}
        gidx, ncl = global_cluster_index(names)
        pp_tables, pp_pos = {}, {}
        for ds, b in boots.items():
            if use_pp_shortcut[ds]:
                pp_tables[ds] = {k: b.per_protein_rho_seeds(k) for k in b.keys}
                allp, _ = np.unique(b.frame.spine['group'].to_numpy().astype(str), return_inverse=True)
                pp_pos[ds] = {p: i for i, p in enumerate(allp)}
        for bi in range(B):
            counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)) if ncl else np.zeros(0)
            seed_sampler = _make_seed_sampler(rng)
            for ds, b in boots.items():
                if use_pp_shortcut[ds]:
                    npro = len(pp_pos[ds])
                    cnt = rng.multinomial(npro, np.full(npro, 1.0 / npro)).astype(float)
                    for k, (nm, r_seeds) in pp_tables[ds].items():
                        c = cnt[[pp_pos[ds][str(x)] for x in nm]]
                        ns = r_seeds.shape[0]
                        sw = seed_sampler(ns)
                        W = np.outer(sw, c)
                        ok = np.isfinite(r_seeds) & (W > 0)
                        reps[ds][k][bi] = float(np.sum(W[ok] * r_seeds[ok]) / np.sum(W[ok])) if ok.any() else np.nan
                else:
                    w = draw_weights(kind, b, rng, counts, gidx[ds])
                    for k, v in b.replicate(w, seed_w=seed_sampler).items():
                        reps[ds][k][bi] = v
    else:
        for bi in range(B):
            seed_sampler = _make_seed_sampler(rng)
            for ds, b in boots.items():
                w = draw_weights('mutation', b, rng)
                for k, v in b.replicate(w, seed_w=seed_sampler).items():
                    reps[ds][k][bi] = v
    return point, reps


def ci(x, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.quantile(x, [a, 1.0 - a]))


def paired_diff(reps_a, reps_b, point_a: float | None = None, point_b: float | None = None) -> dict[str, float]:
    """Bootstrap distribution summary of ``a - b`` with two-sided p-value (floored at ``1 / (B + 1)``)."""
    d = np.asarray(reps_a, dtype=float) - np.asarray(reps_b, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    delta = (float(point_a) - float(point_b)) if (point_a is not None and point_b is not None) else float(d.mean())
    return dict(mean=delta, lo=float(lo), hi=float(hi), p=float(max(p, 1.0 / (len(d) + 1))))
