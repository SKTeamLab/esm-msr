"""
Epistasis validation metrics: what a model has learned about how two substitutions combine, level by level, with the assay's saturation and the
other levels kept out of each level's number. Full argument, measurements and reading guide: docs/validation_metrics.md and the report
linked from it.

The data. Each validation double (residue a at position i < j, residue b at j) of a library with wild-type stability c = dG_wt has a measured
ddG_AB, both measured singles ddG_A, ddG_B, and the usual epistasis score dddG = ddG_AB - ddG_A - ddG_B. Its ADDITIVE EXPECTATION is
x = c + ddG_A + ddG_B, the dG the double would have if the two substitutions did not interact. The doubles of one position pair form a matrix
(rows a, columns b), so dddG is organised as

    global  >  position pair  >  line (one fixed residue on one side, the other side varying: a 'pair column')  >  cell

and the levels of the hierarchy are, from the top:

    global   E[dddG | x], a function of the additive expectation alone: the assay's floor and ceiling (it cannot report a dG outside about -1..5,
             and in practice pins unfolded variants near 0..0.5) plus any nonspecific 'diminishing returns'. On the validation doubles this is
             about 56% of the variance of dddG and dominates every statistic computed on raw dddG.
    pair     the mean of a pair matrix beyond the global curve (22%; about 70% of it is between libraries)
    line     a substitution's average coupling with the other position, beyond the pair mean (14%; much of it is the measurement noise of the
             single that every cell of the line shares)
    cell     what depends on the specific combination of the two residues (6%; about the size of the doubles' own measurement noise)

How saturation is kept out. Two devices, each exact for a predictor whose latent is additive (so any credit it gets is not epistasis):

  * BEYOND-GLOBAL RESIDUALS (magnitudes). Measured: r = dG_AB - E[dG_AB | x] (= dddG - E[dddG | x]); a smoother on the measured x. Predicted, for
    head H: its observed-scale double minus its OWN additive prediction pushed through its own link, e = P_AB - obs(L_A + L_B + o) (o the
    head's median second difference, which removes a calibration bias), then minus
    E[e | x_hat] with x_hat = c + L_A + L_B. A head that is additive before its link has e = 0, so it scores exactly 0 at every residual level:
    what the link does to the singles never enters. The pair / line / cell levels are pair means, line effects and double-centred cells of these
    residuals.
  * WITHIN-COLUMN RANKS (orders). A monotone assay cannot change which of two doubles in one column (same library, same partner) is more stable,
    so the order of ddG_AB inside a column is saturation-free. The cell-level rank statistic double-centres column ranks; the confident-flip
    accuracy uses only reversals of order between two columns.

Heads (each head is a rule for the latent ddG of a single or a double; observed = h(c + latent) - c through the run's link, or the latent itself):

    comb  (WT + ~MT)/2: the reported prediction (inference ``combined_pred``)
    mt    ~MT: the MT pass alone (inference ``epistasis_pred``); its non-additive part is the same as comb's, twice as large, without the WT half
    ctx   (WT + ~WT)/2: the WT adapter read on the mutated sequence as well. It never trains on a double, so it is the CONTROL for what the
          backbone plus single-mutant training already know about epistasis; the MT adapter's contribution is comb / mt minus ctx. Needs
          --val_cycle_passes.
    add   WT: the WT adapter on the wild type, additive by construction. Through the link it scores ONLY by saturation: the CONTROL for the naive
          and global metrics (it is exactly 0 / 0.5 on every other one, so it is not logged there).

The metrics (``METRICS`` below; logged as ``val_epi_<name>``):

    naive_rho_{comb,add}      Spearman of predicted against measured dddG over all doubles, observed scale. Holistic and naive: under a link
                              most of it is saturation (add scores about as much as comb). A summary, not a target.
    global_err_{comb,add}     RMSE (kcal/mol) between the predicted and the measured E[dddG | x] (quantile bins of the measured x). How well the
                              model reproduces the global curve; add shows how much of it the link alone carries.
    global_bias_comb          the weighted mean of (predicted - measured) over the same bins: negative = the model under-predicts the
                              nonspecific (saturation) epistasis, e.g. a link floor below the assay's practical floor.
    beyond_rho_{comb,ctx}     Spearman of the beyond-global residuals over all doubles: everything above the global level at once (pair offsets
                              dominate it). The clean holistic number.
    pair_rho_comb             Spearman across position pairs of the pairs' mean residuals. Only about 24 pairs (9 libraries) in validation:
                              report it, do not select on it.
    line_rho_{comb,ctx}       Spearman, pooled over the lines of every matrix (both orientations), of the line means of the residuals centred
                              on their matrix mean.
    cell_mag_comb             each matrix trimmed to a complete block, residuals double-centred, Spearman per block, mean over blocks.
    cell_rank_{comb,mt,ctx}   the same block, but each column of measured ddG_AB and of the head's latent ddG_AB rank-transformed before
                              double-centring (both orientations, mean per block, mean over blocks). Saturation-free; exactly 0 for any
                              predictor whose order within a column does not depend on the partner. (The doubles-based successor of
                              val_rho_flip_pair_mt_avg; r = 0.89 with it across past runs.)
    cell_flipacc_{mt,ctx}     of the 2 x 2 sub-blocks whose measured order REVERSES between two columns by more than ``FLIP_DELTA`` kcal/mol on
                              both sides, the share in which the head's predicted interaction contrast (L_ij - L_i'j - L_ij' + L_i'j', latent;
                              pair, row and column effects cancel exactly) has the reversal's sign; ties count half, so an additive head is
                              exactly 0.5. comb's contrast is half of mt's, so the two are identical.
    cell_sigsd_mt             the standard deviation of the head's double-centred column ranks: how much partner-dependent order it predicts
                              at all (0 = an additive readout; not a skill score).

Counts (logged as ``val_epi_n_<name>``): doubles, pairs, complete blocks, confident flips.
"""
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from esm_msr import routing

NAN = float('nan')
FLAT = 0.02              # a prediction whose spread is below this (kcal/mol, or rank units) is flat: it scores 0 (0.5 for an accuracy). An
                         # additive head's second difference is a constant plus numerical noise (SD up to ~0.013 with bf16 weights); a
                         # real head's spreads by 0.25 or more
FLIP_DELTA = 0.6         # kcal/mol; a measured order reversal counts when both of its differences exceed this (about 2x the noise of a difference)
GLOBAL_BINS = 20         # quantile bins of x for the global curve
SMOOTH_BINS = 25         # quantile bins of the smoother that removes the global curve
MIN_PAIR_CELLS = 20      # a position pair needs this many doubles to form a matrix
MIN_PAIRS = 8            # the pair level needs this many matrices
MIN_LINE = 3             # a line needs this many cells for its mean
MIN_BLOCK = 3            # a complete block needs at least this many rows and columns

HEADS = ('comb', 'mt', 'ctx', 'add')
# name -> heads it is logged for. The order is the reading order of the hierarchy.
METRICS = {
    'naive_rho': ('comb', 'add'),
    'global_err': ('comb', 'add'),
    'global_bias': ('comb',),
    'beyond_rho': ('comb', 'ctx'),
    'pair_rho': ('comb',),
    'line_rho': ('comb', 'ctx'),
    'cell_mag': ('comb',),
    'cell_rank': ('comb', 'mt', 'ctx'),
    'cell_flipacc': ('mt', 'ctx'),
    'cell_sigsd': ('mt',),
}
COUNTS = ('doubles', 'pairs', 'blocks', 'flips')
# Not logged in training (saturation leaks into them; docs/validation_metrics.md): the levels computed on RAW observed-scale dddG, the
# definitions this module replaced. Kept for evaluation reports, beside the residualised levels, to show how much of a number is saturation.
RAW_METRICS = ('pair_raw', 'line_raw', 'cell_raw')


def logged_names():
    """Every ``val_epi_*`` name this module can produce (without the ``val_`` prefix)."""
    return [f'epi_{m}_{h}' for m, hs in METRICS.items() for h in hs] + [f'epi_n_{c}' for c in COUNTS]


# ---------------------------------------------------------------- small statistics
def _rho(a, b, min_n: int = 3) -> float:
    """Spearman; NaN for too few points or a flat label, 0.0 for a flat prediction (standard deviation below FLAT)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return NAN
    a, b = a[ok], b[ok]
    if np.ptp(b) < 1e-12:
        return NAN
    if np.std(a) < FLAT:
        return 0.0
    return float(spearmanr(a, b)[0])


def _mean(xs) -> float:
    xs = [v for v in xs if np.isfinite(v)]
    return float(np.mean(xs)) if xs else NAN


def smooth_on(x, v, nbins: int = SMOOTH_BINS):
    """E[v | x]: piecewise-linear through the means of ``nbins`` quantile bins of x, extended linearly beyond the outer bin centres."""
    x, v = np.asarray(x, float), np.asarray(v, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() < 2 * nbins:
        nbins = max(1, int(ok.sum()) // 2)
    if ok.sum() < 2:
        return out
    qs = np.quantile(x[ok], np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(qs, x[ok], side='right') - 1, 0, nbins - 1)
    cx = np.array([x[ok][idx == k].mean() for k in range(nbins) if (idx == k).any()])
    cv = np.array([v[ok][idx == k].mean() for k in range(nbins) if (idx == k).any()])
    o = np.argsort(cx)
    cx, cv = cx[o], cv[o]
    fin = np.isfinite(x)
    out[fin] = np.interp(x[fin], cx, cv)
    if len(cx) >= 2:
        lo, hi = fin & (x < cx[0]), fin & (x > cx[-1])
        out[lo] = cv[0] + (x[lo] - cx[0]) * (cv[1] - cv[0]) / max(cx[1] - cx[0], 1e-12)
        out[hi] = cv[-1] + (x[hi] - cx[-1]) * (cv[-1] - cv[-2]) / max(cx[-1] - cx[-2], 1e-12)
    return out


def _complete_block(M):
    """Row / column indices of a complete block of M (the most-missing row or column is dropped first), or None below MIN_BLOCK."""
    r, c = np.arange(M.shape[0]), np.arange(M.shape[1])
    while True:
        if len(r) < MIN_BLOCK or len(c) < MIN_BLOCK:
            return None
        bad = ~np.isfinite(M[np.ix_(r, c)])
        if not bad.any():
            return r, c
        rb, cb = bad.sum(1), bad.sum(0)
        if rb.max() / len(c) >= cb.max() / len(r):
            r = np.delete(r, int(np.argmax(rb)))
        else:
            c = np.delete(c, int(np.argmax(cb)))


def _dc(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def _colrank(M):
    return np.column_stack([(rankdata(M[:, j]) - 0.5) / M.shape[0] for j in range(M.shape[1])])


# ---------------------------------------------------------------- the doubles table
class Table:
    """
    The validation doubles that have both singles measured, as aligned arrays, and per head the predicted values the metrics read.

    Measured: ``pair`` (an integer id per position pair), ``row`` / ``col`` (residue indices inside the pair), ``y`` (ddG_AB), ``yA``,
    ``yB``, ``dddG``, ``c`` (dG_wt), ``x`` (= c + yA + yB). Per head ``h`` in ``self.heads``: ``L[h]`` latent ddG_AB, ``LA[h]`` / ``LB[h]`` latent
    singles, ``P[h]`` / ``PA[h]`` / ``PB[h]`` observed-scale ddG_AB and singles, ``PX[h]`` the observed-scale value of the head's additive
    prediction, L_A + L_B plus the head's median second difference (its composition offset, e.g. a calibration bias).
    """

    def __init__(self, items: Dict[str, Sequence], link: Optional[Callable] = None, heads: Sequence[str] = HEADS):
        """
        ``items``: one entry per validation item, pooled over libraries:
          ``mut_key`` (tuple of mutations, each a tuple whose last two elements are position and mutant residue and whose other elements
          identify the library, so keys never pair across libraries), ``subset_type``, ``cens`` (-1/0/+1), ``ddG`` (measured), ``dddG``
          (measured, doubles), ``dG_wt``, and the latent scores ``wt`` (WT pass), ``mt`` (MT pass), ``comb`` and optionally ``wt_rev`` (the WT
          adapter's reverse leg on the mutated sequence, forward sign; --val_cycle_passes).
        ``link``: the run's monotone link h on ABSOLUTE dG (numpy), or None. Observed = h(dG_wt + latent) - dG_wt.
        """
        keys = list(items['mut_key'])
        st = np.asarray([routing.canonical_subset(s) for s in items['subset_type']])
        n = len(keys)
        cens = np.zeros(n, int) if items.get('cens') is None else np.asarray(items['cens'], int)
        gt, dddG = np.asarray(items['ddG'], float), np.asarray(items['dddG'], float)
        c = np.asarray(items['dG_wt'], float) if items.get('dG_wt') is not None else np.full(n, np.nan)
        lat = {'add': np.asarray(items['wt'], float), 'mt': np.asarray(items['mt'], float), 'comb': np.asarray(items['comb'], float)}
        if items.get('wt_rev') is not None and np.isfinite(np.asarray(items['wt_rev'], float)).any():
            lat['ctx'] = 0.5 * (lat['add'] + np.asarray(items['wt_rev'], float))
        self.heads = [h for h in heads if h in lat]

        single = {}
        for i, k in enumerate(keys):
            if st[i] in routing.WT_HEAD_SUBSETS and k is not None and len(k) == 1 and cens[i] == 0 and np.isfinite(gt[i]):
                single[tuple(k[0])] = i
        rec = []
        pair_id, res_id = {}, {}
        for i, k in enumerate(keys):
            if st[i] not in routing.ENSEMBLE_SUBSETS or k is None or len(k) != 2 or cens[i] != 0:
                continue
            if not (np.isfinite(dddG[i]) and np.isfinite(gt[i]) and np.isfinite(c[i])):
                continue
            m1, m2 = tuple(k[0]), tuple(k[1])
            if m1[:-1] > m2[:-1]:
                m1, m2 = m2, m1
            ia, ib = single.get(m1), single.get(m2)
            if ia is None or ib is None:
                continue
            p = pair_id.setdefault((m1[:-1], m2[:-1]), len(pair_id))
            rec.append((i, ia, ib, p, m1[-1], m2[-1]))
        self.n = len(rec)
        I, IA, IB = (np.array([r[j] for r in rec], int) for j in range(3))
        self.pair = np.array([r[3] for r in rec], int)
        rows = [r[4] for r in rec]; cols = [r[5] for r in rec]
        self.row = np.array([res_id.setdefault(('r', p, a), len(res_id)) for p, a in zip(self.pair, rows)], int)
        self.col = np.array([res_id.setdefault(('c', p, b), len(res_id)) for p, b in zip(self.pair, cols)], int)
        self.y, self.yA, self.yB, self.dddG, self.c = (gt[I], gt[IA], gt[IB], dddG[I], c[I]) if self.n else (np.array([]),) * 5
        self.x = self.c + self.yA + self.yB
        obs = (lambda v, cc: link(cc + v) - cc) if link is not None else (lambda v, cc: v)
        self.L, self.LA, self.LB, self.P, self.PA, self.PB, self.PX = ({} for _ in range(7))
        for h in self.heads:
            v = lat[h]
            self.L[h], self.LA[h], self.LB[h] = v[I], v[IA], v[IB]
            self.P[h], self.PA[h], self.PB[h] = obs(v[I], self.c), obs(v[IA], self.c), obs(v[IB], self.c)
            # the head's own composition offset: a calibration bias b makes even an additive head give L_AB - L_A - L_B = -b, which a link
            # would bend into a spurious pattern; the median second difference removes it (for other heads it is a constant of the global level)
            d = v[I] - v[IA] - v[IB]
            off = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
            self.PX[h] = obs(v[IA] + v[IB] + off, self.c)
        self._mats = None

    def head_ok(self, h):
        return h in self.heads and self.n > 0 and np.isfinite(self.L[h]).any()

    def matrices(self):
        """Per position pair with at least MIN_PAIR_CELLS doubles: (index matrix into the table (-1 = no double), rows, cols)."""
        if self._mats is None:
            self._mats = []
            for p in np.unique(self.pair):
                idx = np.where(self.pair == p)[0]
                if len(idx) < MIN_PAIR_CELLS:
                    continue
                ur, ri = np.unique(self.row[idx], return_inverse=True)
                uc, ci = np.unique(self.col[idx], return_inverse=True)
                M = np.full((len(ur), len(uc)), -1, int)
                M[ri, ci] = idx
                self._mats.append(M)
        return self._mats


def _lay(M, v):
    out = np.full(M.shape, np.nan)
    ok = M >= 0
    out[ok] = v[M[ok]]
    return out


# ---------------------------------------------------------------- residuals
def beyond_global(t: Table, h: str):
    """(measured, predicted) beyond-global residuals per double; see the module docstring."""
    rm = (t.c + t.y) - smooth_on(t.x, t.c + t.y)
    e = t.P[h] - t.PX[h]
    if not np.isfinite(e).any() or np.nanstd(e) < FLAT:
        return rm, np.where(np.isfinite(e), 0.0, np.nan)
    return rm, e - smooth_on(t.c + t.LA[h] + t.LB[h], e)


# ---------------------------------------------------------------- levels
def naive_rho(t: Table, h: str) -> float:
    return _rho(t.P[h] - t.PA[h] - t.PB[h], t.dddG)


def global_curve(t: Table, h: str):
    """(RMSE, bias) of the predicted against the measured mean dddG in quantile bins of the measured x."""
    pd_ = t.P[h] - t.PA[h] - t.PB[h]
    ok = np.isfinite(pd_) & np.isfinite(t.x)
    if ok.sum() < 2 * GLOBAL_BINS:
        return NAN, NAN
    qs = np.quantile(t.x[ok], np.linspace(0, 1, GLOBAL_BINS + 1))
    q = np.clip(np.searchsorted(qs, t.x[ok], side='right') - 1, 0, GLOBAL_BINS - 1)
    d, w = [], []
    for k in range(GLOBAL_BINS):
        s = q == k
        if s.any():
            d.append(pd_[ok][s].mean() - t.dddG[ok][s].mean()); w.append(s.sum())
    d, w = np.array(d), np.array(w, float)
    return float(np.sqrt(np.average(d ** 2, weights=w))), float(np.average(d, weights=w))


def residual_levels(t: Table, h: str) -> Dict[str, float]:
    """beyond_rho, pair_rho, line_rho, cell_mag on the beyond-global residuals."""
    rm, rp = beyond_global(t, h)
    out = {'beyond_rho': _rho(rp, rm)}
    mats = t.matrices()
    pm, pp, lm, lp, cells = [], [], [], [], []
    for M in mats:
        Rm, Rp = _lay(M, rm), _lay(M, rp)
        ok = np.isfinite(Rm) & np.isfinite(Rp)
        if ok.sum() < MIN_PAIR_CELLS:
            continue
        Rm, Rp = np.where(ok, Rm, np.nan), np.where(ok, Rp, np.nan)
        mm, mp = np.nanmean(Rm), np.nanmean(Rp)
        pm.append(mm); pp.append(mp)
        for ax in (0, 1):
            cnt = ok.sum(axis=ax)
            with np.errstate(invalid='ignore'):
                lm.extend((np.nanmean(Rm, axis=ax) - mm)[cnt >= MIN_LINE]); lp.extend((np.nanmean(Rp, axis=ax) - mp)[cnt >= MIN_LINE])
        blk = _complete_block(Rm)
        if blk is not None:
            ix = np.ix_(*blk)
            cells.append(_rho(_dc(Rp[ix]).ravel(), _dc(Rm[ix]).ravel()))
    out['pair_rho'] = _rho(pp, pm) if len(pm) >= MIN_PAIRS else NAN
    out['line_rho'] = _rho(lp, lm) if len(lm) >= 2 * MIN_PAIRS else NAN
    out['cell_mag'] = _mean(cells)
    return out


def raw_levels(t: Table, h: str) -> Dict[str, float]:
    """pair_raw, line_raw, cell_raw: the pair / line / double-centred cell levels on RAW dddG (predicted P_AB - P_A - P_B on the observed
    scale against measured dddG). Saturation is non-additive on the observed scale, so it enters all three: a saturation-only head scores
    about 0.71 / 0.48 / 0.10 in simulation. Contrast them with pair_rho / line_rho / cell_mag."""
    pd_ = t.P[h] - t.PA[h] - t.PB[h]
    pm, pp, lm, lp, cells = [], [], [], [], []
    for M in t.matrices():
        Dm, Dp = _lay(M, t.dddG), _lay(M, pd_)
        ok = np.isfinite(Dm) & np.isfinite(Dp)
        if ok.sum() < MIN_PAIR_CELLS:
            continue
        Dm, Dp = np.where(ok, Dm, np.nan), np.where(ok, Dp, np.nan)
        mm, mp = np.nanmean(Dm), np.nanmean(Dp)
        pm.append(mm); pp.append(mp)
        for ax in (0, 1):
            cnt = ok.sum(axis=ax)
            with np.errstate(invalid='ignore'):
                lm.extend((np.nanmean(Dm, axis=ax) - mm)[cnt >= MIN_LINE]); lp.extend((np.nanmean(Dp, axis=ax) - mp)[cnt >= MIN_LINE])
        blk = _complete_block(Dm)
        if blk is not None:
            ix = np.ix_(*blk)
            cells.append(_rho(_dc(Dp[ix]).ravel(), _dc(Dm[ix]).ravel()))
    return {'pair_raw': _rho(pp, pm) if len(pm) >= MIN_PAIRS else NAN, 'line_raw': _rho(lp, lm) if len(lm) >= 2 * MIN_PAIRS else NAN,
            'cell_raw': _mean(cells)}


def rank_levels(t: Table, h: str) -> Dict[str, float]:
    """cell_rank and cell_sigsd on complete blocks of measured / latent ddG_AB; n_blocks."""
    ranks, sig = [], []
    for M in t.matrices():
        Y, L = _lay(M, t.y), _lay(M, t.L[h])
        blk = _complete_block(np.where(np.isfinite(L), Y, np.nan))
        if blk is None:
            continue
        Y, L = Y[np.ix_(*blk)], L[np.ix_(*blk)]
        r, s = [], []
        for A, B in ((Y, L), (Y.T, L.T)):
            sm, sp = _dc(_colrank(A)), _dc(_colrank(B))
            r.append(_rho(sp.ravel(), sm.ravel())); s.append(float(sp.std()))
        ranks.append(_mean(r)); sig.append(float(np.mean(s)))
    return {'cell_rank': _mean(ranks), 'cell_sigsd': _mean(sig), 'n_blocks': len(ranks)}


def flip_accuracy(t: Table, h: str, delta: float = FLIP_DELTA):
    """(accuracy, number of confident reversals); see ``cell_flipacc`` in the module docstring."""
    hits, n = 0.0, 0
    for M in t.matrices():
        Y, L = _lay(M, t.y), _lay(M, t.L[h])
        for A, B in ((Y, L), (Y.T, L.T)):
            R, C = A.shape
            if R < 2 or C < 2:
                continue
            iu = np.triu_indices(R, 1)
            dA, dB = (A[:, None, :] - A[None, :, :])[iu], (B[:, None, :] - B[None, :, :])[iu]    # (row pairs, columns)
            j, k = np.triu_indices(C, 1)
            a, b = dA[:, j], dA[:, k]
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(dB[:, j]) & np.isfinite(dB[:, k])
            flip = ok & (np.sign(a) != np.sign(b)) & (np.abs(a) > delta) & (np.abs(b) > delta)
            if not flip.any():
                continue
            con = (dB[:, j] - dB[:, k])[flip]
            want = np.sign(a[flip])
            hits += float(np.where(np.abs(con) < FLAT, 0.5, (np.sign(con) == want).astype(float)).sum())
            n += int(flip.sum())
    return (hits / n if n else NAN), n


# ---------------------------------------------------------------- everything
def compute(t: Table, metrics: Dict[str, Sequence[str]] = METRICS) -> Dict[str, float]:
    """``{'epi_<metric>_<head>': value, 'epi_n_<count>': value}`` for every requested (metric, head) whose head the table has; never raises
    for missing data (undefined entries are NaN)."""
    out = {'epi_n_doubles': float(t.n), 'epi_n_pairs': float(len(t.matrices()))}
    if t.n == 0:
        return out
    need = {h for hs in metrics.values() for h in hs if t.head_ok(h)}
    for h in sorted(need):
        want = {m for m, hs in metrics.items() if h in hs}
        r = {}
        if 'naive_rho' in want:
            r['naive_rho'] = naive_rho(t, h)
        if want & {'global_err', 'global_bias'}:
            r['global_err'], r['global_bias'] = global_curve(t, h)
        if want & {'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag'}:
            r.update(residual_levels(t, h))
        if want & {'cell_rank', 'cell_sigsd'}:
            rl = rank_levels(t, h)
            out['epi_n_blocks'] = float(rl.pop('n_blocks'))
            r.update(rl)
        if want & set(RAW_METRICS):
            r.update(raw_levels(t, h))
        if 'cell_flipacc' in want:
            r['cell_flipacc'], nf = flip_accuracy(t, h)
            out['epi_n_flips'] = float(nf)
        for m in want:
            out[f'epi_{m}_{h}'] = r.get(m, NAN)
    return out
