"""
The epistasis hierarchy: how well a predicted dddG reproduces the measured one, level by level.

dddG = ddG_AB - ddG_A - ddG_B for a measured double (positions i < j, residues a at i and b at j). The doubles of one position pair form a
matrix M[a, b] (rows: the residue at the lower position, columns: at the higher one), so the groups are nested:

    all doubles  >  position-pair matrix  >  column (one fixed residue at one position, the other varying)  >  cell

and each nesting level has two complementary statistics. The EFFECT of a level is the variation of its group means (between groups, within the
parent); the RANK of a level is the agreement of the ordering inside each group (within groups). An effect level is computed on group means
MINUS THE PARENT's mean, so it does not repeat what the level above already explains; a rank level is a correlation inside one unit, averaged over
units, which a shift of the parent mean cannot change. Nothing here pools across units except where the level IS the pooling (the global rank, the
effect levels, and the pooled variant of the interaction rank). Saturation (the assay's floor and ceiling, ~60% of dddG variance) enters every
level through the observed-scale dddG; neither device removes the part that bends the surface WITHIN a unit, so a predictor that is additive
before the link is always scored alongside as the control (``wt_add``).

    level                     statistic
    ------------------------  ----------------------------------------------------------------------------------------------------------
    global_rmse               RMSE of predicted against measured dddG, all doubles pooled (the global effect, saturation included)
    global_rho                Spearman of all doubles pooled (the global rank)
    pair_effect               Spearman across position pairs of the mean predicted against the mean measured dddG of the pair's matrix
    matrix_rank               Spearman over the cells of one matrix, averaged over matrices
    partner_effect            Spearman, pooled over every column of every matrix (both orientations), of [column mean - matrix mean]
                              predicted against measured: how much a given partner residue shifts the scored position's mean dddG
    partner_context_rank      Spearman across the cells of one column, averaged over columns: the order of the scored substitutions
                              within a fixed partner
    identity_effect           the residue-pair table: each matrix is double-centred (row, column and grand means removed), the
                              centred cells are averaged per (residue at i, residue at j) over matrices, and the predicted table is
                              correlated with the measured one (Spearman over the combinations)
    interaction_rank_*        double-centre each complete matrix, correlate predicted with measured: ``ranked`` ranks within columns first
                              (no monotone distortion of a column's order, the flip statistic), ``raw`` uses the values; ``per_matrix``
                              averages the per-matrix correlations, ``pooled`` takes one correlation over all matrices

Matrices need ``min_pair_cells`` cells; columns ``min_group`` (effects) or ``min_col`` (ranks); the interaction needs a complete block of at
least ``min_rows`` x ``min_cols`` after the most-missing rows / columns are dropped (centring an incomplete matrix lets missingness itself
correlate with the prediction). A predictor that is flat where it should vary (an additive one, after double-centring; a pair-mean-only one, inside a matrix) scores 0, not NaN; a flat label is NaN.
"""
from typing import Dict, Optional

import numpy as np
from scipy.stats import rankdata, spearmanr

LEVELS = ('global_rmse', 'global_rho', 'pair_effect', 'matrix_rank', 'partner_effect', 'partner_context_rank', 'identity_effect',
          'interaction_rank_ranked_per_matrix', 'interaction_rank_ranked_pooled',
          'interaction_rank_raw_per_matrix', 'interaction_rank_raw_pooled')
NAN = float('nan')


def _rho(a, b, flat_is_zero: bool = True, min_n: int = 3) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return NAN
    a, b = a[ok], b[ok]
    if np.ptp(b) < 1e-12:
        return NAN
    if np.ptp(a) < 1e-12:
        return 0.0 if flat_is_zero else NAN
    return float(spearmanr(a, b)[0])


def _mean(xs) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else NAN


def build_matrices(pred, label, mut_keys, is_double, min_pair_cells: int = 20):
    """
    Position-pair matrices of the doubles that have both a prediction and a measurement.

    ``mut_keys[i]`` is the tuple of item i's two mutations, each ``(..., wt, position, mutant residue)``; everything but the last element
    identifies the position (so pooled keys that start with the library name never mix libraries). Returns a list of
    ``(rows, cols, Mp, Ml)``: sorted residues at the lower and higher position, and the predicted / measured matrices (NaN where missing),
    for the pairs with at least ``min_pair_cells`` cells.
    """
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    pairs: Dict[tuple, list] = {}
    for i, k in enumerate(mut_keys):
        if not bool(is_double[i]) or k is None or len(k) != 2 or not (np.isfinite(pred[i]) and np.isfinite(label[i])):
            continue
        a, b = tuple(k[0]), tuple(k[1])
        lo, hi = (a, b) if a[:-1] <= b[:-1] else (b, a)
        pairs.setdefault((lo[:-1], hi[:-1]), []).append((lo[-1], hi[-1], pred[i], label[i]))
    out = []
    for cells in pairs.values():
        if len(cells) < min_pair_cells:
            continue
        rows, cols = sorted({c[0] for c in cells}), sorted({c[1] for c in cells})
        ri, ci = {r: n for n, r in enumerate(rows)}, {c: n for n, c in enumerate(cols)}
        Mp, Ml = np.full((len(rows), len(cols)), np.nan), np.full((len(rows), len(cols)), np.nan)
        for r, c, p, l in cells:
            Mp[ri[r], ci[c]], Ml[ri[r], ci[c]] = p, l
        out.append((rows, cols, Mp, Ml))
    return out


def _complete_block(M, rows, cols, min_rows: int, min_cols: int):
    """Drops the most-missing row or column until no gap remains; returns (M, rows, cols) or None."""
    M, rows, cols = M.copy(), list(rows), list(cols)
    while True:
        if M.shape[0] < min_rows or M.shape[1] < min_cols:
            return None
        bad = ~np.isfinite(M)
        if not bad.any():
            return M, rows, cols
        rb, cb = bad.sum(1), bad.sum(0)
        if rb.max() / max(1, M.shape[1]) >= cb.max() / max(1, M.shape[0]):
            k = int(np.argmax(rb)); M = np.delete(M, k, axis=0); rows.pop(k)
        else:
            k = int(np.argmax(cb)); M = np.delete(M, k, axis=1); cols.pop(k)


def _double_centre(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def _rank_columns(M):
    R = np.empty(M.shape)
    for j in range(M.shape[1]):
        R[:, j] = (rankdata(M[:, j]) - 0.5) / M.shape[0]
    return R


def _lines(M):
    """Every column and every row of a matrix as index arrays of its observed cells (the two orientations of 'a column')."""
    out = []
    for j in range(M.shape[1]):
        idx = np.where(np.isfinite(M[:, j]))[0]
        out.append((idx, np.full(len(idx), j), 0))
    for i in range(M.shape[0]):
        idx = np.where(np.isfinite(M[i, :]))[0]
        out.append((np.full(len(idx), i), idx, 1))
    return out


def compute(pred, label, mut_keys, is_double, min_pair_cells: int = 20, min_pairs: int = 8, min_group: int = 3, min_col: int = 4,
            min_rows: int = 3, min_cols: int = 3, min_identity_obs: int = 3, min_identities: int = 8) -> Dict[str, float]:
    """All levels for one predictor. ``pred`` is the predicted dddG per item (NaN where undefined), ``label`` the measured one."""
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    is_double = np.asarray(is_double, bool)
    out = {k: NAN for k in LEVELS}
    out.update(n_doubles=0, n_pairs=0, n_columns=0, n_complete_blocks=0, n_identities=0)

    ok = is_double & np.isfinite(pred) & np.isfinite(label)
    out['n_doubles'] = int(ok.sum())
    if ok.sum() >= 3:
        out['global_rmse'] = float(np.sqrt(np.mean((pred[ok] - label[ok]) ** 2)))
        out['global_rho'] = _rho(pred[ok], label[ok])

    mats = build_matrices(pred, label, mut_keys, is_double, min_pair_cells)
    out['n_pairs'] = len(mats)
    if not mats:
        return out

    if len(mats) >= min_pairs:
        out['pair_effect'] = _rho([np.nanmean(Mp) for _, _, Mp, _ in mats], [np.nanmean(Ml) for _, _, _, Ml in mats])

    ranks, ex, ey = [], [], []
    for rows, cols, Mp, Ml in mats:
        okm = np.isfinite(Mp) & np.isfinite(Ml)
        ranks.append(_rho(Mp[okm], Ml[okm]))
        pm, lm = np.nanmean(Mp), np.nanmean(Ml)
        for ri, ci, _ in _lines(Ml):
            if len(ri) >= min_group:
                ex.append(np.mean(Mp[ri, ci]) - pm)
                ey.append(np.mean(Ml[ri, ci]) - lm)
    out['matrix_rank'] = _mean(ranks)
    if len(ex) >= 2 * min_pairs:
        out['partner_effect'] = _rho(ex, ey)

    colr = []
    for rows, cols, Mp, Ml in mats:
        for ri, ci, _ in _lines(Ml):
            if len(ri) >= min_col:
                colr.append(_rho(Mp[ri, ci], Ml[ri, ci]))
    out['n_columns'] = int(np.sum(np.isfinite(colr))) if colr else 0
    out['partner_context_rank'] = _mean(colr)

    # ---- interaction levels, on complete blocks
    ident_p: Dict[tuple, list] = {}
    ident_l: Dict[tuple, list] = {}
    per = {'ranked': [], 'raw': []}
    pool = {'ranked': ([], []), 'raw': ([], [])}
    for rows, cols, Mp, Ml in mats:
        blk = _complete_block(Ml, rows, cols, min_rows, min_cols)
        if blk is None:
            continue
        Lb, rws, cls = blk
        keep_r = [rows.index(r) for r in rws]
        keep_c = [cols.index(c) for c in cls]
        Pb = Mp[np.ix_(keep_r, keep_c)]
        if not np.isfinite(Pb).all():
            continue
        out['n_complete_blocks'] += 1
        dcp, dcl = _double_centre(Pb), _double_centre(Lb)
        for a_i, r in enumerate(rws):
            for b_i, c in enumerate(cls):
                ident_p.setdefault((r, c), []).append(dcp[a_i, b_i])
                ident_l.setdefault((r, c), []).append(dcl[a_i, b_i])
        views = {'raw': (dcp, dcl), 'ranked': (_double_centre(_rank_columns(Pb)), _double_centre(_rank_columns(Lb)))}
        for name, (sp, sl) in views.items():
            per[name].append(_rho(sp.ravel(), sl.ravel()))
            pool[name][0].append(sp.ravel()); pool[name][1].append(sl.ravel())
    for name in ('ranked', 'raw'):
        out[f'interaction_rank_{name}_per_matrix'] = _mean(per[name])
        if pool[name][0]:
            out[f'interaction_rank_{name}_pooled'] = _rho(np.concatenate(pool[name][0]), np.concatenate(pool[name][1]))

    combos = [k for k, v in ident_l.items() if len(v) >= min_identity_obs]
    out['n_identities'] = len(combos)
    if len(combos) >= min_identities:
        out['identity_effect'] = _rho([np.mean(ident_p[k]) for k in combos], [np.mean(ident_l[k]) for k in combos])
    return out
