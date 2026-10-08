"""
The epistasis hierarchy: how well a predicted dddG reproduces the measured one, level by level.

dddG = ddG_AB - ddG_A - ddG_B for a measured double (positions i < j, residues a at i and b at j). The doubles of one position pair form a
matrix M[a, b] (rows: the residue at the lower position, columns: at the higher one), so the groups are nested:

    all doubles  >  position-pair matrix  >  column (one fixed residue at one position, the other varying)  >  cell

and each nesting level has two complementary statistics. The EFFECT of a level is the variation of its group means (between groups, within the
parent); the RANK of a level is the agreement of the ordering inside each group (within groups). An effect level is computed on group means
MINUS THE PARENT's mean, so it does not repeat what the level above already explains; a rank level is a correlation inside one unit, averaged over
units, which a shift of the parent mean cannot change. Nothing here pools across units except where the level IS the pooling (the global rank and the
effect levels). Saturation (the assay's floor and ceiling, ~60% of dddG variance) enters every
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
    interaction_rank_*        double-centre each complete matrix, correlate predicted with measured, one correlation per matrix, averaged:
                              ``ranked`` ranks within columns first (no monotone distortion of a column's order, the flip statistic),
                              ``raw`` uses the values
    *_beyond_add / *_beyond_single   PARTIAL versions of three levels (below), logged when the control is supplied

The partial columns ask whether a level's skill survives once a confounder is held fixed. A partial Spearman correlation is the correlation of
predicted and measured after both have been (rank-)regressed on the control, (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)):

    global_rho_beyond_add     predicted vs measured dddG across all doubles, given the ADDITIVE observed-scale score of the double (the WT head's
                              value for it: sum of its singles passed through the link). Saturation makes dddG a function of how far the additive
                              prediction already is from the floor, so this is the skill that is not just "knows where the assay saturates".
    matrix_rank_beyond_add    the same inside each matrix (the control varies cell to cell), averaged over matrices.
    partner_effect_beyond_single   partner_effect given the MEASURED single ddG of the fixed mutation of the line (centred on the matrix's lines, as
                              the line effects are): a line shift is partly just how destabilising the fixed mutation is (saturation again),
                              partly which residue it pairs with; this keeps the latter.

Matrices need ``min_pair_cells`` cells; columns ``min_group`` (effects) or ``min_col`` (ranks); the interaction needs a complete block of at
least ``min_rows`` x ``min_cols`` after the most-missing rows / columns are dropped (centring an incomplete matrix lets missingness itself
correlate with the prediction). A predictor that is flat where it should vary (an additive one, after double-centring; a pair-mean-only one, inside a matrix) scores 0, not NaN; a flat label is NaN.
"""
from typing import Dict, NamedTuple, Optional

import numpy as np
from scipy.stats import rankdata, spearmanr

LEVELS = ('global_rmse', 'global_rho', 'pair_effect', 'matrix_rank', 'partner_effect', 'partner_context_rank', 'identity_effect',
          'interaction_rank_ranked_per_matrix', 'interaction_rank_raw_per_matrix',
          'global_rho_beyond_add', 'matrix_rank_beyond_add', 'partner_effect_beyond_single')
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


def _partial_rho(x, y, z, min_n: int = 4) -> float:
    """
    Spearman correlation of x and y given z (the correlation of their rank residuals on rank z). A flat label is NaN and a flat predictor 0.0,
    as in ``_rho``; a flat control removes nothing, so the plain correlation is returned; a control that is a monotone function of either
    variable leaves nothing to correlate (NaN).
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < min_n:
        return NAN
    x, y, z = x[ok], y[ok], z[ok]
    if np.ptp(y) < 1e-12:
        return NAN
    if np.ptp(x) < 1e-12:
        return 0.0
    rxy = float(spearmanr(x, y)[0])
    if np.ptp(z) < 1e-12:
        return rxy
    rxz, ryz = float(spearmanr(x, z)[0]), float(spearmanr(y, z)[0])
    den = np.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den) if den > 1e-6 else NAN


def _mean(xs) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else NAN


def predicted_dddG(values, n_mutations, mut_keys):
    """
    Predicted dddG of every double: ``value_AB - value_A - value_B`` of a head's per-item values.

    ``n_mutations[i]`` is item i's number of mutations and ``mut_keys[i]`` the tuple of its mutations (hashable, each ``(..., wt, position, mutant
    residue)``; a pooled key starts with the library name, so a double only finds its own library's singles). NaN for any item that is not a double
    with both singles present, or whose value is not finite.
    """
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    if mut_keys is None or len(mut_keys) != len(values):
        return out
    single = {tuple(k[0]): values[i] for i, k in enumerate(mut_keys) if k is not None and int(n_mutations[i]) == 1 and len(k) == 1}
    for i, k in enumerate(mut_keys):
        if k is not None and int(n_mutations[i]) == 2 and len(k) == 2:
            a, b = single.get(tuple(k[0])), single.get(tuple(k[1]))
            if a is not None and b is not None:
                out[i] = values[i] - a - b
    return out


class Matrix(NamedTuple):
    """One position-pair matrix: sorted residues at the lower / higher position, predicted / measured / control values (NaN where missing),
    and the pair's identity (the two position ids, ``mutation[:-1]``), so a line can be named as a mutation."""
    rows: list
    cols: list
    Mp: np.ndarray
    Ml: np.ndarray
    Mx: Optional[np.ndarray]
    key: tuple


def build_matrices(pred, label, mut_keys, is_double, min_pair_cells: int = 20, extra=None):
    """
    Position-pair matrices of the doubles that have both a prediction and a measurement.

    ``mut_keys[i]`` is the tuple of item i's two mutations, each ``(..., wt, position, mutant residue)``; everything but the last element
    identifies the position (so pooled keys that start with the library name never mix libraries). Returns a list of ``Matrix`` for the pairs
    with at least ``min_pair_cells`` cells. ``extra`` (optional, per item) is laid out in ``Matrix.Mx`` over the same cells; it does not decide
    which cells exist.
    """
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    ext = None if extra is None else np.asarray(extra, float)
    pairs: Dict[tuple, list] = {}
    for i, k in enumerate(mut_keys):
        if not bool(is_double[i]) or k is None or len(k) != 2 or not (np.isfinite(pred[i]) and np.isfinite(label[i])):
            continue
        a, b = tuple(k[0]), tuple(k[1])
        lo, hi = (a, b) if a[:-1] <= b[:-1] else (b, a)
        pairs.setdefault((lo[:-1], hi[:-1]), []).append((lo[-1], hi[-1], pred[i], label[i], np.nan if ext is None else ext[i]))
    out = []
    for key, cells in pairs.items():
        if len(cells) < min_pair_cells:
            continue
        rows, cols = sorted({c[0] for c in cells}), sorted({c[1] for c in cells})
        ri, ci = {r: n for n, r in enumerate(rows)}, {c: n for n, c in enumerate(cols)}
        Mp, Ml, Mx = (np.full((len(rows), len(cols)), np.nan) for _ in range(3))
        for r, c, p, l, x in cells:
            Mp[ri[r], ci[c]], Ml[ri[r], ci[c]], Mx[ri[r], ci[c]] = p, l, x
        out.append(Matrix(rows, cols, Mp, Ml, None if ext is None else Mx, key))
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
            min_rows: int = 3, min_cols: int = 3, min_identity_obs: int = 3, min_identities: int = 8,
            additive=None, single_ddG=None) -> Dict[str, float]:
    """
    All levels for one predictor. ``pred`` is the predicted dddG per item (NaN where undefined), ``label`` the measured one.

    Optional controls for the ``*_beyond_*`` levels (NaN when not supplied): ``additive``, per item, the additive observed-scale score of each
    double (the same control for every head); ``single_ddG``, a dict from a mutation tuple (as it appears in ``mut_keys``) to its measured
    single ddG.
    """
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    is_double = np.asarray(is_double, bool)
    out = {k: NAN for k in LEVELS}
    out.update(n_doubles=0, n_pairs=0, n_columns=0, n_complete_blocks=0, n_identities=0)
    add = None if additive is None else np.asarray(additive, float)

    ok = is_double & np.isfinite(pred) & np.isfinite(label)
    out['n_doubles'] = int(ok.sum())
    if ok.sum() >= 3:
        out['global_rmse'] = float(np.sqrt(np.mean((pred[ok] - label[ok]) ** 2)))
        out['global_rho'] = _rho(pred[ok], label[ok])
        if add is not None:
            out['global_rho_beyond_add'] = _partial_rho(pred[ok], label[ok], add[ok])

    mats = build_matrices(pred, label, mut_keys, is_double, min_pair_cells, extra=add)
    out['n_pairs'] = len(mats)
    if not mats:
        return out

    if len(mats) >= min_pairs:
        out['pair_effect'] = _rho([np.nanmean(m.Mp) for m in mats], [np.nanmean(m.Ml) for m in mats])

    ranks, ranks_add, ex, ey, ez = [], [], [], [], []
    for m in mats:
        okm = np.isfinite(m.Mp) & np.isfinite(m.Ml)
        ranks.append(_rho(m.Mp[okm], m.Ml[okm]))
        if add is not None:
            ranks_add.append(_partial_rho(m.Mp[okm], m.Ml[okm], m.Mx[okm]))
        pm, lm = np.nanmean(m.Mp), np.nanmean(m.Ml)
        zm = []
        for ri, ci, orient in _lines(m.Ml):
            if len(ri) >= min_group:
                ex.append(np.mean(m.Mp[ri, ci]) - pm)
                ey.append(np.mean(m.Ml[ri, ci]) - lm)
                if single_ddG is not None:       # the line's fixed mutation: the column's residue at the higher position, or the row's at the lower
                    fixed = (m.key[1] + (m.cols[ci[0]],)) if orient == 0 else (m.key[0] + (m.rows[ri[0]],))
                    zm.append(single_ddG.get(fixed, NAN))
        if zm:                                   # centred on the matrix's lines, like the line effects themselves
            zm = np.asarray(zm, float)
            ez.extend(zm - np.nanmean(zm) if np.isfinite(zm).any() else zm)
    out['matrix_rank'] = _mean(ranks)
    if add is not None:
        out['matrix_rank_beyond_add'] = _mean(ranks_add)
    if len(ex) >= 2 * min_pairs:
        out['partner_effect'] = _rho(ex, ey)
        if single_ddG is not None:
            out['partner_effect_beyond_single'] = _partial_rho(ex, ey, ez)

    colr = []
    for m in mats:
        for ri, ci, _ in _lines(m.Ml):
            if len(ri) >= min_col:
                colr.append(_rho(m.Mp[ri, ci], m.Ml[ri, ci]))
    out['n_columns'] = int(np.sum(np.isfinite(colr))) if colr else 0
    out['partner_context_rank'] = _mean(colr)

    # ---- interaction levels, on complete blocks
    ident_p: Dict[tuple, list] = {}
    ident_l: Dict[tuple, list] = {}
    per = {'ranked': [], 'raw': []}
    for m in mats:
        blk = _complete_block(m.Ml, m.rows, m.cols, min_rows, min_cols)
        if blk is None:
            continue
        Lb, rws, cls = blk
        keep_r = [m.rows.index(r) for r in rws]
        keep_c = [m.cols.index(c) for c in cls]
        Pb = m.Mp[np.ix_(keep_r, keep_c)]
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
    for name in ('ranked', 'raw'):
        out[f'interaction_rank_{name}_per_matrix'] = _mean(per[name])

    combos = [k for k, v in ident_l.items() if len(v) >= min_identity_obs]
    out['n_identities'] = len(combos)
    if len(combos) >= min_identities:
        out['identity_effect'] = _rho([np.mean(ident_p[k]) for k in combos], [np.mean(ident_l[k]) for k in combos])
    return out
