"""Bootstrap of Spearman's rho with count weights.

A bootstrap replicate is a vector of non-negative integer *multiplicities*, one per row.  Spearman's rho of the
expanded sample (a row seen k times counts k times; ties, including a row tied with its own copies, share the
average rank) is computed straight from the multiplicities, so a replicate costs O(n) and no re-sorting: the sort
order of every prediction column is fixed once.  The result equals ``scipy.stats.spearmanr`` on the expanded sample
(checked in ``tests/test_boot.py``).

Resampling units (``kind``):

``protein``   whole protein clusters with replacement (the clustered bootstrap).  A replicate draws multiplicities
              for the *union* of clusters over all datasets of a call, so every set and every model sees the same
              proteins: differences between models, and means over datasets, are paired.
``mutation``  individual rows with replacement (treats mutations as independent; for per-protein-mean datasets the
              draw is within each protein).
``site``      clusters = (protein, first mutated position); for single-protein DMS where proteins cannot be resampled.

Per-protein-mean datasets (Tsuboyama, Domainome) report the mean over proteins of a per-protein rho.
"""
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
    """Spearman's rho of (x, y) per group, under row multiplicities.  ``y`` and its ranker can be shared."""

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


# ----------------------------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------------------------
def _int_codes(values):
    u, inv = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    return inv.astype(np.int64), u


def global_cluster_index(frames_clusters):
    """Map every (namespace, cluster name) to one integer shared across datasets.  ``frames_clusters`` is
    {dataset: array of cluster names}.  Datasets resample the same proteins in one replicate."""
    allnames = {}
    out = {}
    for ds, names in frames_clusters.items():
        idx = np.empty(len(names), np.int64)
        for i, nm in enumerate(np.asarray(names).astype(str)):
            idx[i] = allnames.setdefault(nm, len(allnames))
        out[ds] = idx
    return out, len(allnames)


class DatasetBoot:
    """Everything needed to bootstrap one dataset: valid rows per model, rankers, per-protein groups."""

    def __init__(self, frame, keys, per_protein=False, common=False, subset=None):
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
        # one engine per distinct mask (labels are ranked once per mask)
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
        self.cluster_names = {'protein': sp['cluster'].to_numpy(), 'site': (sp['cluster'].astype(str) + '#' + sp['site1'].astype(str)).to_numpy()}

    # -- point estimates ----------------------------------------------------------------------
    def point(self, k):
        """(seed-mean rho, per-seed rhos, n rows, n proteins)."""
        m = self.valid[k]
        idx, g, G, eng = self._engines[m.tobytes()]
        w = np.ones(len(idx))
        per_seed = []
        for xr in self._xr[k]:
            r = eng.rho(xr, w)
            per_seed.append(np.nanmean(r) if self.per_protein else r[0])
        return float(np.mean(per_seed)), np.array(per_seed), int(len(idx)), int(G if self.per_protein else self.n_clusters(k))

    def n_clusters(self, k):
        idx = self._engines[self.valid[k].tobytes()][0]
        return len(np.unique(self.cluster_names['protein'][idx]))

    # -- one replicate for all models --------------------------------------------------------------
    def replicate(self, w_full):
        """rho of every model under full-length multiplicities ``w_full``; returns {key: seed-mean rho}."""
        out = {}
        ycache = {}
        for k in self.keys:
            mk = self.valid[k].tobytes()
            idx, g, G, eng = self._engines[mk]
            w = w_full[idx]
            if mk not in ycache:
                ycache[mk] = eng.ry.ranks(w)
            rs = []
            for xr in self._xr[k]:
                r = eng.rho(xr, w, ycache[mk])
                rs.append(np.nanmean(r) if self.per_protein else r[0])
            out[k] = float(np.mean(rs))
        return out

    def per_protein_rho(self, k):
        """Per-protein rho of every protein (unit weights), seed-averaged; used by the protein-level bootstrap of the
        per-protein-mean datasets, which is exact and needs no re-ranking."""
        idx, g, G, eng = self._engines[self.valid[k].tobytes()]
        w = np.ones(len(idx))
        r = np.nanmean([eng.rho(xr, w) for xr in self._xr[k]], axis=0)
        names = np.empty(G, object)
        names[g] = self.frame.spine['group'].to_numpy()[idx]
        return names, r


def draw_weights(kind, boot, rng, cluster_counts=None, cluster_idx=None):
    """Row multiplicities of one replicate."""
    n = boot.n_rows
    if kind == 'protein':
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
    raise ValueError(kind)


def bootstrap_rho(frames, keys, kind='protein', B=1000, seed=0, per_protein=None, common=False, subsets=None):
    """Bootstrap seed-mean rho for every model on every dataset.

    frames: {dataset: Frame}.  Returns (point, reps) where point[ds][key] = (rho, per-seed rhos, n, n_proteins)
    and reps[ds][key] = array(B,).  ``kind='protein'`` resamples the union of clusters (paired across datasets and
    models); per-protein-mean datasets are resampled at the protein level exactly from the per-protein rhos.
    """
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
        names = {ds: b.cluster_names[kind] for ds, b in boots.items() if not b.per_protein}
        gidx, ncl = global_cluster_index(names)
        # per-protein-mean datasets: exact, from per-protein rhos (clusters = proteins of that dataset)
        pp_tables, pp_pos = {}, {}
        for ds, b in boots.items():
            if b.per_protein:
                pp_tables[ds] = {k: b.per_protein_rho(k) for k in b.keys}
                allp, inv = np.unique(b.frame.spine['group'].to_numpy().astype(str), return_inverse=True)
                pp_pos[ds] = {p: i for i, p in enumerate(allp)}
        for bi in range(B):
            counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)) if ncl else np.zeros(0)
            for ds, b in boots.items():
                if b.per_protein:
                    # each dataset resamples its own proteins; one draw is shared by every model
                    npro = len(pp_pos[ds])
                    cnt = rng.multinomial(npro, np.full(npro, 1.0 / npro))
                    for k, (nm, r) in pp_tables[ds].items():
                        c = cnt[[pp_pos[ds][str(x)] for x in nm]]
                        ok = np.isfinite(r) & (c > 0)
                        reps[ds][k][bi] = float(np.dot(c[ok], r[ok]) / c[ok].sum())
                else:
                    w = draw_weights('protein', b, rng, counts, gidx[ds])
                    for k, v in b.replicate(w).items():
                        reps[ds][k][bi] = v
    else:
        for bi in range(B):
            for ds, b in boots.items():
                w = draw_weights('mutation', b, rng)
                for k, v in b.replicate(w).items():
                    reps[ds][k][bi] = v
    return point, reps


def ci(x, level=0.95):
    a = (1 - level) / 2
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return tuple(np.quantile(x, [a, 1 - a]))


def paired_diff(reps_a, reps_b):
    """Bootstrap distribution of a - b with the p-value of the two-sided test that a - b = 0."""
    d = np.asarray(reps_a) - np.asarray(reps_b)
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=float(max(p, 1.0 / (len(d) + 1))))
