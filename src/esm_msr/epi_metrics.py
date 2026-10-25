"""
Epistasis validation metrics: what a model has learned about how two substitutions combine, level by level, with the assay's saturation and the
other levels kept out of each level's number. Full argument, measurements and reading guide: docs/epistasis_metrics_report.html on the devel
branch.

This file is shared VERBATIM by claude/main-baseline and the devel branches - keep it that way, so neither has to resolve it against the other.
It therefore assumes nothing about the link: the caller passes one (or None) and its arity decides what happens. A link taking only an absolute
dG bends at one fixed knee (the baseline's assay clamp); one that also takes a per-item floor bends each double at its own library's floor (the
devel branches' trainable link). Nor does it choose the population: see ``Table`` below.

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

How saturation is kept out. Two devices:

  * BEYOND-GLOBAL RESIDUALS (magnitudes). The same operation on both sides: each side's dddG minus E[dddG | x], its mean as a function of
    that side's own ADDITIVE EXPECTATION x (a 25-quantile-bin smoother, ``smooth_on``). What is removed is the global level - saturation
    at the assay floor plus any nonspecific diminishing returns - which is a function of x alone, so it is not pair/line/cell epistasis.
      measured:   r  = dddG       - E[dddG       | x    ],  x     = c + ddG_A + ddG_B       (c = dG_wt)
      predicted:  r' = dddG'_h    - E[dddG'_h    | x'_h ],  x'_h  = c + the head's own additive prediction (latent)
    The predicted side conditions on the head's OWN x' because a head's saturation-made dddG' is an exact function of x' (it is its link
    bending its own additive prediction), so a head that is additive before its link scores exactly 0 at every residual level. Each head's
    dddG' (``predicted_dddg``):
      comb   P_comb(AB) - obs(comb(A) + comb(B))           x' = c + comb(A) + comb(B)     6 head evaluations
      epi    P_epi(AB)  - P_wt(AB)       ~MT(AB) - WT(AB)  x' = c + WT(AB)                2 evaluations, no singles asked of the MT head
      ctx    P_rev(AB)  - P_wt(AB)       ~WT(AB) - WT(AB)  x' = c + WT(AB)                the same with the WT adapter in the MT seat
    WT(AB) = WT(A) + WT(B) exactly (the WT adapter on the wild type, per-mutation bias), so epi / ctx read the part of the double the WT head's
    singles do not give: the epistasis component the model would actually report, at the cost of the double alone. comb keeps the definition
    used since the suite was introduced, for comparability with earlier runs (only naive_rho_comb is logged). P = observed scale, h(c + latent) - c.
    The pair / line / cell levels are pair means, line effects and double-centred cells of these residuals.
  * WITHIN-COLUMN RANKS (orders). A monotone assay cannot change which of two doubles in one column (same library, same partner) is more stable,
    so the order of ddG_AB inside a column is saturation-free. The cell-level rank statistic double-centres column ranks; the confident-flip
    accuracy uses only reversals of order between two columns. These read each head's latent DOUBLE prediction only (no singles).

Heads (latent ddG rules; observed = h(c + latent) - c through the run's link, or the latent itself):

    comb  (WT + ~MT)/2: the reported prediction (inference ``combined_pred``; logged only for naive_rho)
    epi   ~MT: the epistasis (MT) pass alone, evaluated relative to the WT additive baseline (~MT(AB) - WT(AB))
    ctx   (WT + ~WT)/2 on the rank levels; ~WT alone in its dddG' above. ~WT is the WT adapter read on the mutated sequence. It never trains
          on a double, so it is the CONTROL for what the backbone plus single-mutant training already know about epistasis. Needs
          --val_cycle_passes.
    add   WT: the WT adapter on the wild type, additive by construction. Through the link it scores ONLY by saturation (naive_rho).

The metrics (``METRICS`` below; logged as ``val_epi_<metric>_<head>``).

    naive_rho_{comb,epi,ctx,add}    Spearman of the head's raw observed-scale dddG' against measured dddG over all doubles, nothing removed
                                    (comb / add: P(AB) - P(A) - P(B)). Mostly saturation - add scores about as much as comb. For curiosity.
    global_err_epi                  RMSE (kcal/mol) of the calibrated and linked epistasis double-mutant prediction P_epi against measured ddG_AB.
    global_bias_epi                 Mean of (P_epi - ddG_AB) in kcal/mol on the calibrated and linked epistasis double-mutant predictions.
    beyond_rho_{epi,ctx}            Spearman of r' against r over all doubles: everything above the global level at once (pair and library
                                    offsets dominate it).
    beyond_lib_{epi,ctx}            the same with each library's mean removed from r and r': above the global level and not a library-wide offset.
    beyond_rho_pairavg_{epi,ctx}    mean across position pairs of the Spearman correlation of residuals r' against r within each pair.
    pair_rho_{epi,ctx}              Spearman across position pairs of the pairs' mean residuals. Only about 24 pairs (9 libraries) in
                                    validation: report it, do not select on it.
    line_rho_{epi,ctx}              Spearman, pooled over the lines of every matrix (both orientations), of the line means of the residuals
                                    centred on their matrix mean.
    cell_mag_{epi,ctx}              each matrix trimmed to a complete block, residuals double-centred, Spearman per block, mean over blocks.
    cell_rank_{epi,ctx}             the same block, but each column of measured ddG_AB and of the head's latent ddG_AB rank-transformed before
                                    double-centring (both orientations, mean per block, mean over blocks). Saturation-free; exactly 0 for any
                                    predictor whose order within a column does not depend on the partner.
    cell_flipacc_{epi,ctx}          of the 2 x 2 sub-blocks whose measured order REVERSES between two columns by more than ``FLIP_DELTA``
                                    kcal/mol on both sides, the share in which the head's predicted interaction contrast (L_ij - L_i'j -
                                    L_ij' + L_i'j', latent; pair, row and column effects cancel exactly) has the reversal's sign; ties count
                                    half, so an additive head is exactly 0.5.
    class3_acc_{epi,ctx}            3-class classification accuracy (negative < -0.6, neutral [-0.6, 0.6], positive > 0.6 kcal/mol) on residuals.
    class3_pos_recall_{epi,ctx}     recall on confident positive epistasis (r > 0.6 kcal/mol).
    class3_neg_recall_{epi,ctx}     recall on confident negative epistasis (r < -0.6 kcal/mol).

Changed 2026-10-24:
  - Global calibration/scale metrics anchored to 'epi' (formerly 'mt'): kept only global_err and global_bias, dropping
    global_dddg_bias* (which conflated single-mutant shrinkage with epistasis bias).
  - Deleted all comb epistasis metrics except naive_rho_comb: comb mixes 50% additive WT head, diluting epistatic contrast
    and compressing residuals by 2x.
  - Renamed _mt to _epi across all epistasis metrics to reflect that the head is evaluated as ~MT(AB) - WT(AB) (the epistasis increment).
  - Added beyond_rho_pairavg, class3_acc, class3_pos_recall, and class3_neg_recall.
  - Library-centred beyond_lib.

Counts (logged as ``val_epi_n_<name>``): doubles, pairs, complete blocks, confident flips.
"""
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


NAN = float('nan')
# This branch's subsets (esm_msr.data): singles carry the WT head's measured targets, doubles the ensemble's.
SINGLE_SUBSETS, DOUBLE_SUBSETS = frozenset({'single'}), frozenset({'double'})
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

HEADS = ('comb', 'epi', 'ctx', 'add')
EPI_HEADS = ('epi', 'ctx')
# Heads whose dddG' is their DOUBLE minus the WT-additive double (head -> the latent whose double is read); see predicted_dddg.
WT_REFERENCED = {'epi': 'epi', 'ctx': 'rev', 'mt': 'epi'}
# name -> heads it is logged for. The order is the reading order of the hierarchy.
METRICS = {
    'naive_rho': HEADS,
    'global_err': ('epi',),
    'global_bias': ('epi',),
    'beyond_rho': EPI_HEADS,
    'beyond_lib': EPI_HEADS,
    'beyond_rho_pairavg': EPI_HEADS,
    'pair_rho': EPI_HEADS,
    'line_rho': EPI_HEADS,
    'cell_mag': EPI_HEADS,
    'cell_rank': EPI_HEADS,
    'cell_flipacc': EPI_HEADS,
    'class3_acc': EPI_HEADS,
    'class3_pos_recall': EPI_HEADS,
    'class3_neg_recall': EPI_HEADS,
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


def _takes_offset(link) -> bool:
    """True when ``link`` accepts a per-item floor as its second argument, which the devel branches' link does and a fixed clamp does not."""
    try:
        link(np.zeros(1), np.zeros(1))
        return True
    except TypeError:
        return False


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
    prediction, obs(L_A + L_B).
    """

    def __init__(self, items: Dict[str, Sequence], link: Optional[Callable] = None, heads: Sequence[str] = HEADS):
        """
        ``items``: one entry per validation item, pooled over libraries:
          ``mut_key`` (tuple of mutations, each a tuple whose last two elements are position and mutant residue and whose other elements
          identify the library, so keys never pair across libraries), ``subset_type``, ``ddG`` (measured), ``dddG``
          (measured, doubles), ``dG_wt``, and the latent scores ``wt`` (WT pass), ``mt`` (MT pass), ``comb`` and optionally ``wt_rev`` (the WT
          adapter's reverse leg on the mutated sequence, forward sign; --val_cycle_passes).
        ``link``: the run's monotone link h on ABSOLUTE dG (numpy), or None. Observed = h(dG_wt + latent) - dG_wt.
        """
        keys = list(items['mut_key'])
        st = np.asarray([str(s) for s in items['subset_type']])
        n = len(keys)
        gt, dddG = np.asarray(items['ddG'], float), np.asarray(items['dddG'], float)
        c = np.asarray(items['dG_wt'], float) if items.get('dG_wt') is not None else np.full(n, np.nan)
        mt_arr = items['epi'] if 'epi' in items else items['mt']
        lat = {'add': np.asarray(items['wt'], float), 'epi': np.asarray(mt_arr, float), 'comb': np.asarray(items['comb'], float)}
        if 'mt' in items:
            lat['mt'] = np.asarray(items['mt'], float)
        if items.get('wt_rev') is not None and np.isfinite(np.asarray(items['wt_rev'], float)).any():
            lat['rev'] = np.asarray(items['wt_rev'], float)
            lat['ctx'] = 0.5 * (lat['add'] + lat['rev'])
        self.heads = [h for h in heads if h in lat] + (['rev'] if 'rev' in lat else [])

        single = {}
        for i, k in enumerate(keys):
            if st[i] in SINGLE_SUBSETS and k is not None and len(k) == 1 and np.isfinite(gt[i]):
                single[tuple(k[0])] = i
        rec = []
        pair_id, res_id = {}, {}
        for i, k in enumerate(keys):
            if st[i] not in DOUBLE_SUBSETS or k is None or len(k) != 2:
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
        lib_id = {}
        self.lib = np.array([lib_id.setdefault(tuple(keys[r[0]][0][:-2]), len(lib_id)) for r in rec], int)   # the key minus (pos, aa)
        rows = [r[4] for r in rec]; cols = [r[5] for r in rec]
        self.row = np.array([res_id.setdefault(('r', p, a), len(res_id)) for p, a in zip(self.pair, rows)], int)
        self.col = np.array([res_id.setdefault(('c', p, b), len(res_id)) for p, b in zip(self.pair, cols)], int)
        self.y, self.yA, self.yB, self.dddG, self.c = (gt[I], gt[IA], gt[IB], dddG[I], c[I]) if self.n else (np.array([]),) * 5
        self.x = self.c + self.yA + self.yB
        # the observed scale: h(dG_wt + latent) - dG_wt. Where the link takes a per-item floor, each double bends at its own library's floor
        # (that link clips a NaN floor into its band, so a library without one still gets a sane knee); otherwise it bends at the link's own knee.
        fo = items.get('y_floor')
        fo = np.asarray(fo, float) if fo is not None else None
        if fo is None or len(fo) != n:                     # a field on_validation_epoch_end did not pool
            self.floor_off = np.full(self.n, np.nan)
        else:
            self.floor_off = fo[I]
        if link is None:
            obs = lambda v, cc: v
        elif _takes_offset(link):
            obs = lambda v, cc: link(cc + v, self.floor_off) - cc
        else:
            obs = lambda v, cc: link(cc + v) - cc
        self.L, self.LA, self.LB, self.P, self.PA, self.PB, self.PX = ({} for _ in range(7))
        for h in self.heads:
            v = lat[h]
            self.L[h], self.LA[h], self.LB[h] = v[I], v[IA], v[IB]
            self.P[h], self.PA[h], self.PB[h] = obs(v[I], self.c), obs(v[IA], self.c), obs(v[IB], self.c)
            # No composition offset (until 2026-10-24 the head's median L_AB - L_A - L_B was added here). With --per_mutation_bias an
            # additive head's second difference is exactly 0, and for comb the median is the MT pass's typical diminishing returns, a real
            # global term that the E[dddG' | x'] correction handles anyway.
            self.PX[h] = obs(v[IA] + v[IB], self.c)
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
def predicted_dddg(t: Table, h: str):
    """(dddG'_h on the observed scale, x'_h): the head's predicted epistasis and its own additive expectation; see the module docstring."""
    if h in WT_REFERENCED:
        return t.P[WT_REFERENCED[h]] - t.P['add'], t.c + t.L['add']
    return t.P[h] - t.PX[h], t.c + t.LA[h] + t.LB[h]


def beyond_global(t: Table, h: str):
    """(r, r'): measured and predicted dddG, each minus E[dddG | x] on its own additive expectation x; see the module docstring."""
    r = t.dddG - smooth_on(t.x, t.dddG)
    d, xh = predicted_dddg(t, h)
    if not np.isfinite(d).any() or np.nanstd(d) < FLAT:
        return r, np.where(np.isfinite(d), 0.0, np.nan)
    return r, d - smooth_on(xh, d)


def _lib_centred(v, lib):
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    for g in np.unique(lib[ok]):
        m = ok & (lib == g)
        out[m] = v[m] - v[m].mean()
    return out


# ---------------------------------------------------------------- levels
def naive_rho(t: Table, h: str) -> float:
    d = predicted_dddg(t, h)[0] if h in WT_REFERENCED else t.P[h] - t.PA[h] - t.PB[h]
    return _rho(d, t.dddG)


def global_curve(t: Table, h: str = 'epi'):
    """(RMSE, bias) of the calibrated and linked double-mutant prediction P[h] against measured ddG_AB (y)."""
    d = t.P[h] - t.y
    ok = np.isfinite(d)
    if not ok.any():
        return NAN, NAN
    d = d[ok]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))


def residual_levels(t: Table, h: str) -> Dict[str, float]:
    """beyond_rho, beyond_lib, beyond_rho_pairavg, pair_rho, line_rho, cell_mag, class3_* on the beyond-global residuals."""
    rm, rp = beyond_global(t, h)
    out = {'beyond_rho': _rho(rp, rm), 'beyond_lib': _rho(_lib_centred(rp, t.lib), _lib_centred(rm, t.lib))}
    mats = t.matrices()
    pm, pp, lm, lp, cells, pair_rhos = [], [], [], [], [], []
    for M in mats:
        Rm, Rp = _lay(M, rm), _lay(M, rp)
        ok = np.isfinite(Rm) & np.isfinite(Rp)
        if ok.sum() < MIN_PAIR_CELLS:
            continue
        pair_rhos.append(_rho(Rp[ok], Rm[ok]))
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
    out['beyond_rho_pairavg'] = _mean(pair_rhos)
    out['pair_rho'] = _rho(pp, pm) if len(pm) >= MIN_PAIRS else NAN
    out['line_rho'] = _rho(lp, lm) if len(lm) >= 2 * MIN_PAIRS else NAN
    out['cell_mag'] = _mean(cells)

    ok = np.isfinite(rm) & np.isfinite(rp)
    if ok.any():
        ym, yp = rm[ok], rp[ok]
        cm = np.where(ym < -FLIP_DELTA, -1, np.where(ym > FLIP_DELTA, 1, 0))
        cp = np.where(yp < -FLIP_DELTA, -1, np.where(yp > FLIP_DELTA, 1, 0))
        out['class3_acc'] = float((cm == cp).mean())
        pos_m, neg_m = cm == 1, cm == -1
        out['class3_pos_recall'] = float((cp[pos_m] == 1).mean()) if pos_m.any() else NAN
        out['class3_neg_recall'] = float((cp[neg_m] == -1).mean()) if neg_m.any() else NAN
    else:
        out['class3_acc'] = out['class3_pos_recall'] = out['class3_neg_recall'] = NAN
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
        if want & {'beyond_rho', 'beyond_lib', 'pair_rho', 'line_rho', 'cell_mag'}:
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
