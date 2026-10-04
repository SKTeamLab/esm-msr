import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score

from esm_msr import routing


def safe_spearman(preds, targets):
    if len(preds) < 2 or len(np.unique(targets)) < 2: return np.nan
    if len(np.unique(preds)) < 2: return np.nan
    rho = spearmanr(preds, targets)[0]
    return np.nan if np.isnan(rho) else float(rho)


def safe_rmse(preds, targets):
    if not len(preds): return np.nan
    return float(np.sqrt(mean_squared_error(targets, preds)))


def compute_ndcg_flexible(pred, true, *,
                          top_n=None, percentile=None, threshold=None,
                          ignore_ties=True, exponential_relevance=False):
    """
    Compute NDCG alongside physical hit-rate metrics for a defined budget (k).

    Returns:
        Tuple: (NDCG_score, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool)
    """
    flags = [top_n is not None, percentile is not None, threshold is not None]
    if sum(flags) != 1:
        raise ValueError("Specify exactly one of top_n, percentile, or threshold.")

    y_score = pred
    y_true = true

    rel_floor = threshold if threshold is not None else 0.0

    # 1. Total Hits in Pool
    total_hits_in_pool = int(np.sum(y_true > rel_floor))

    y_true_processed = np.where(y_true <= rel_floor, 0.0, y_true)

    if total_hits_in_pool == 0:
        return np.nan, 0, 0, 0

    if exponential_relevance:
        y_true_processed = np.exp(y_true_processed) - 1.0

    y_true_processed = y_true_processed.reshape(1, -1)
    n = y_true.size

    if threshold is not None:
        k = None
    elif top_n is not None:
        if top_n <= 0:
            return np.nan, 0, 0, total_hits_in_pool
        k = min(int(top_n), n)
    else:
        k = max(1, int(np.ceil(percentile * n)))
        k = min(k, n)

    # Calculate NDCG
    ndcg_val = ndcg_score(y_true_processed, y_score, k=k, ignore_ties=ignore_ties)

    # 2. Maximum Possible Hits Scored
    ideal_hits_at_k = min(total_hits_in_pool, k) if k is not None else total_hits_in_pool

    # 3. The Model's Actual Hits Scored
    # Sort the true relevances based on the model's predicted ranking
    sorted_indices = np.argsort(-y_score[0])
    if k is not None:
        model_top_k_relevances = y_true_processed[0][sorted_indices][:k]
    else:
        model_top_k_relevances = y_true_processed[0][sorted_indices]

    model_hits_at_k = int(np.sum(model_top_k_relevances > 0))

    return ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool


def safe_ndcg_k96(preds, targets):
    """
    Computes Normalized Discounted Cumulative Gain.
    Filters out negative relevance scores (targets < 0).
    Raises a RuntimeError if it fails rather than silently passing.
    """
    preds = preds.reshape(1, -1)
    targets = targets.reshape(1, -1)
    try:
        ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = compute_ndcg_flexible(preds, targets, top_n=96)
        return ndcg_val
    except Exception as e:
        raise RuntimeError(f"NDCG calculation failed. Underlying error: {str(e)}")


def safe_ndcg_t0(preds, targets):
    """
    Computes Normalized Discounted Cumulative Gain.
    Filters out negative relevance scores (targets < 0).
    Raises a RuntimeError if it fails rather than silently passing.
    """
    preds = preds.reshape(1, -1)
    targets = targets.reshape(1, -1)
    try:
        ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = compute_ndcg_flexible(preds, targets, threshold=0.0)
        return ndcg_val
    except Exception as e:
        raise RuntimeError(f"NDCG calculation failed. Underlying error: {str(e)}")


def epi_full_scores(comb_scores, subset_types, mut_keys):
    """
    ``comb_AB - comb_A - comb_B`` for every double whose two singles are in the same set.

    ``mut_keys[i]`` is the tuple of mutations of item i (each a hashable, e.g. (wt, pos, mt)).
    Singles are items of subset ``single`` with exactly one mutation. Returns an array aligned
    with the inputs: NaN for anything that is not a double with both singles present.
    """
    comb_scores = np.asarray(comb_scores, dtype=np.float64)
    out = np.full(len(comb_scores), np.nan)
    if mut_keys is None or len(mut_keys) != len(comb_scores):
        return out
    st = np.asarray(subset_types)
    single = {}
    for i, k in enumerate(mut_keys):
        if st[i] in routing.WT_HEAD_SUBSETS and len(k) == 1:
            single[tuple(k[0])] = comb_scores[i]
    for i, k in enumerate(mut_keys):
        if st[i] in routing.ENSEMBLE_SUBSETS and len(k) == 2:
            a, b = single.get(tuple(k[0])), single.get(tuple(k[1]))
            if a is not None and b is not None:
                out[i] = comb_scores[i] - a - b
    return out


def _rho_epi_full(comb_scores, subset_types, mut_keys, dddG, is_double):
    e = epi_full_scores(comb_scores, subset_types, mut_keys)
    ok = is_double & np.isfinite(e)
    return safe_spearman(e[ok], dddG[ok])


def delta_single_diagnostics(df, epi_true_col=None):
    """
    How much the MT and WT heads disagree on single mutants, and what that does to the two
    epistasis readouts. Operates on an inference-output DataFrame (``infer_mutants`` columns
    plus ``mut_type``, ``:`` separating the mutations of a multi-mutant).

    With delta_X = mt_X - wt_X on singles, dW = wt_AB - wt_A - wt_B and
    dM = mt_AB - mt_A - mt_B on doubles:

        E_fast = 0.5 * (mt_AB - wt_AB) = 0.5 * (dM - dW) + 0.5 * (delta_A + delta_B)
        E_full = comb_AB - comb_A - comb_B = 0.5 * (dM + dW)

    so E_fast - E_full = 0.5 * (delta_A + delta_B) - dW exactly. ``identity_resid`` checks
    that on the data (should be ~1e-6); ``dW_sd`` checks the WT head's additivity (a constant
    dW has sd ~0). ``delta_term_share`` is sd(0.5 * (delta_A + delta_B)) / sd(0.5 * dM): how
    large the per-substitution contamination of E_fast is relative to the interaction term
    both readouts share. Doubles need both their singles in ``df``.

    Returns a flat dict; undefined entries are NaN. The double-level entries need the
    ``*_dddg_pred`` columns, i.e. an inference run without ``--skip_additive``.
    """
    import pandas as pd
    nan = float('nan')
    keys = ('n_singles', 'delta_mean', 'delta_sd', 'delta_sd_rel_wt', 'slope_mt_on_wt', 'rho_mt_wt_singles',
            'rho_delta_wt_singles', 'n_doubles_paired', 'dW_sd', 'dM_sd', 'delta_term_share',
            'identity_resid', 'rho_fast_vs_full', 'rho_epi_fast', 'rho_epi_full', 'rho_epi_delta_term')
    out = {k: nan for k in keys}
    need = {'mut_type', 'wt_lora_pred', 'mt_lora_pred'}
    if not need.issubset(df.columns):
        return out

    is_dbl = df['mut_type'].astype(str).str.contains(':')
    s = df.loc[~is_dbl].drop_duplicates('mut_type').set_index('mut_type')
    s = s[np.isfinite(s['wt_lora_pred'].astype(float)) & np.isfinite(s['mt_lora_pred'].astype(float))]
    if len(s) < 3:
        return out
    w, m = s['wt_lora_pred'].to_numpy(float), s['mt_lora_pred'].to_numpy(float)
    delta = m - w
    out.update(n_singles=float(len(s)), delta_mean=float(delta.mean()), delta_sd=float(delta.std()),
               delta_sd_rel_wt=float(delta.std() / w.std()) if w.std() > 0 else nan,
               slope_mt_on_wt=float(np.polyfit(w, m, 1)[0]) if w.std() > 0 else nan,
               rho_mt_wt_singles=safe_spearman(m, w), rho_delta_wt_singles=safe_spearman(delta, w))

    cols = {'wt_lora_dddg_pred', 'mt_lora_dddg_pred', 'combined_dddg_pred'}
    if not cols.issubset(df.columns):
        return out
    d = df.loc[is_dbl].copy()
    parts = d['mut_type'].astype(str).str.split(':')
    d = d[parts.str.len() == 2]
    parts = parts[d.index] if d.index.is_unique else parts.loc[d.index]
    delta_s = pd.Series(delta, index=s.index)
    da = parts.str[0].map(delta_s)
    db = parts.str[1].map(delta_s)
    ok = (da.notna() & db.notna()).to_numpy()
    if ok.sum() < 3:
        return out
    d, da, db = d[ok], da[ok].to_numpy(float), db[ok].to_numpy(float)
    dW, dM = d['wt_lora_dddg_pred'].to_numpy(float), d['mt_lora_dddg_pred'].to_numpy(float)
    e_full = d['combined_dddg_pred'].to_numpy(float)
    e_fast = 0.5 * (d['mt_lora_pred'].to_numpy(float) - d['wt_lora_pred'].to_numpy(float))
    dterm = 0.5 * (da + db)
    out.update(n_doubles_paired=float(len(d)), dW_sd=float(np.std(dW)), dM_sd=float(np.std(dM)),
               delta_term_share=float(np.std(dterm) / np.std(0.5 * dM)) if np.std(dM) > 0 else nan,
               identity_resid=float(np.max(np.abs(e_fast - e_full - (dterm - dW)))),
               rho_fast_vs_full=safe_spearman(e_fast, e_full))
    if epi_true_col is not None and epi_true_col in d.columns:
        y = d[epi_true_col].to_numpy(float)
        out.update(rho_epi_fast=safe_spearman(e_fast[np.isfinite(y)], y[np.isfinite(y)]),
                   rho_epi_full=safe_spearman(e_full[np.isfinite(y)], y[np.isfinite(y)]),
                   rho_epi_delta_term=safe_spearman(dterm[np.isfinite(y)], y[np.isfinite(y)]))
    return out


def compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, dddG=None, mut_keys=None):
    """
    The validation metrics for one dataloader (one protein library or benchmark).

    Each head gets two metrics. The ``_valid`` form scores a head only on the items it is
    responsible for in training, which is the number to judge it by; the ``_all`` form scores
    it indiscriminately on every item with a finite target, which is always defined and so
    stays comparable across loaders that lack one subset or another. The gap between them is
    informative in itself: it says how much a head degrades off its own domain.

    * ``rho_wt_valid``   - WT head on plain single mutations in the real wild-type context.
      NaN for a mutant-background library, whose singles are all ``native_cond``.
    * ``rho_wt_all``     - WT head on everything, multi-mutants and conditionals included.
    * ``rho_mt_valid``   - MT head on conditional targets (``cond``, ``native_cond``). NaN for
      a library with no double mutants to derive them from, and for the external benchmarks,
      which are loaded without derived items.
    * ``rho_mt_all``     - MT head on everything.
    * ``rho_combined``   - the reported two-path average, on measured items only.
    * ``rmse_combined``  - calibration of that average in kcal/mol; rank correlation cannot
      see a scale or offset error.
    * ``rho_epi_fast``   - Spearman between predicted epistasis ``comb - wt = 0.5 * (mt - wt)``
      of a double and its measured dddG. Needs only the double itself. Equals the true
      interaction only if the two heads agree on every single mutant; otherwise it carries
      an extra 0.5 * (delta_A + delta_B) per-substitution term, delta_X = mt_X - wt_X.
    * ``rho_epi_full``   - the same correlation for ``comb_AB - comb_A - comb_B``, the second
      difference of the combined prediction, which mirrors how dddG is defined and cancels
      any head-specific single-mutant effect. Needs ``mut_keys`` (one hashable key per item:
      the tuple of its mutations) and both singles of a double in the same loader; doubles
      whose singles are absent are skipped. Because the WT head is additive, this is rank-
      equivalent to the MT-only second difference.

    ``_all`` deliberately mixes quantities: a conditional ddG(X | background) is not a
    wild-type-context ddG, so a correlation pooling them answers "does this head rank
    anything sensibly" rather than "is this head right". Read ``_valid`` first.

    Returns a flat ``{name: value}`` dict; undefined entries are NaN and are not logged.
    """
    gt = np.asarray(ground_truths, dtype=np.float64)
    wt_scores, mt_scores, comb_scores = (np.asarray(x) for x in (wt_scores, mt_scores, comb_scores))
    subset_types = np.asarray([routing.canonical_subset(s) for s in subset_types])
    finite = np.isfinite(gt)

    is_single = np.isin(subset_types, list(routing.WT_HEAD_SUBSETS)) & finite
    is_cond = np.isin(subset_types, list(routing.CONDITIONAL_SUBSETS)) & finite
    is_measured = np.isin(subset_types, list(routing.MEASURED_SUBSETS)) & finite

    metrics = {
        'rho_wt_valid': safe_spearman(wt_scores[is_single], gt[is_single]),
        'rho_wt_all': safe_spearman(wt_scores[finite], gt[finite]),
        'rho_mt_valid': safe_spearman(mt_scores[is_cond], gt[is_cond]),
        'rho_mt_all': safe_spearman(mt_scores[finite], gt[finite]),
        'rho_combined': safe_spearman(comb_scores[is_measured], gt[is_measured]),
        'rmse_combined': safe_rmse(comb_scores[is_measured], gt[is_measured]),
    }

    if dddG is not None:
        dddG = np.asarray(dddG, dtype=np.float64)
        has_dddG = np.isfinite(dddG)
        is_double = np.isin(subset_types, list(routing.ENSEMBLE_SUBSETS)) & has_dddG
        epi_pred = comb_scores - wt_scores
        metrics['rho_epi_fast'] = safe_spearman(epi_pred[is_double], dddG[is_double])
        metrics['rho_epi_full'] = _rho_epi_full(comb_scores, subset_types, mut_keys, dddG, is_double)
    else:
        metrics['rho_epi_fast'] = float('nan')
        metrics['rho_epi_full'] = float('nan')

    return metrics


def flip_signature_rho(pred, target, flip_keys, row_ids, min_len=4, min_rows=3, min_cols=3):
    """Identity-dependent interaction agreement: the validation twin of the report's test.

    A *flip column* is one scored position with one fixed partner identity; ``flip_keys``
    names it and ``row_ids`` is the substitution identity inside it. Columns sharing a
    position pair form a matrix whose rows are substitutions and columns are partners.

    Two steps make this specific to identity-dependent interaction:

    1. Rank within each column. Any monotone function of the underlying stability leaves
       within-column order unchanged, so the assay's saturating response and its
       dynamic-range floor are annihilated rather than corrected for.
    2. Double-centre. Row means carry each substitution's own average effect, column means
       the partners', and the grand mean the pair's overall coupling. Removing all three
       leaves only what depends on the *combination*. Under additivity every column shares
       one ordering, the matrix is constant along rows, and this is exactly zero.

    The block must be COMPLETE before ranking. Cells go missing because the double failed
    to measure, so missingness depends on both substitution effects and is shared between
    the measured and predicted matrix; centring an incomplete matrix admits correlation up
    to +0.95 with no interaction present at all. Rows and columns are therefore trimmed
    (most-missing first) until no gaps remain.

    Returns (rho, n_pairs, n_cells); rho is nan when no usable block exists, and 0.0 when
    the prediction's signature is identically flat - which is what an additive readout gives.
    """
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    row_ids = np.asarray(row_ids)
    pairs = {}
    for i, k in enumerate(flip_keys):
        if not k or not np.isfinite(pred[i]) or not np.isfinite(target[i]):
            continue
        parts = str(k).split('|')
        if len(parts) < 3:
            continue
        pair, col = '|'.join(parts[:2]), parts[2]
        pairs.setdefault(pair, {}).setdefault(col, []).append(i)

    def _sig(M):
        M = np.asarray(M, float).copy()
        while True:
            if M.shape[0] < min_rows or M.shape[1] < min_cols:
                return None
            bad = ~np.isfinite(M)
            if not bad.any():
                break
            rb, cb = bad.sum(1), bad.sum(0)
            if rb.max() / max(1, M.shape[1]) >= cb.max() / max(1, M.shape[0]):
                M = np.delete(M, int(np.argmax(rb)), axis=0)
            else:
                M = np.delete(M, int(np.argmax(cb)), axis=1)
        from scipy.stats import rankdata as _rd
        R = np.empty(M.shape)
        for j in range(M.shape[1]):
            R[:, j] = (_rd(M[:, j]) - 0.5) / M.shape[0]
        return R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()

    A, Bp, n_pairs = [], [], 0
    for cols in pairs.values():
        cols = {c: idx for c, idx in cols.items() if len(idx) >= min_len}
        if len(cols) < min_cols:
            continue
        col_names = sorted(cols)
        rows = sorted({int(row_ids[i]) for idx in cols.values() for i in idx})
        if len(rows) < min_rows:
            continue
        ri = {r: a for a, r in enumerate(rows)}
        Mt = np.full((len(rows), len(col_names)), np.nan)
        Mp = np.full((len(rows), len(col_names)), np.nan)
        for cj, c in enumerate(col_names):
            for i in cols[c]:
                r = ri.get(int(row_ids[i]))
                if r is not None:
                    Mt[r, cj], Mp[r, cj] = target[i], pred[i]
        St = _sig(Mt)
        if St is None:
            continue
        Sp = _sig(np.where(np.isfinite(Mt), Mp, np.nan))
        if Sp is None or Sp.shape != St.shape:
            continue
        A.append(St.ravel())
        Bp.append(Sp.ravel())
        n_pairs += 1
    if not A:
        return float('nan'), 0, 0
    a, b = np.concatenate(A), np.concatenate(Bp)
    if np.ptp(b) < 1e-12 or np.ptp(a) < 1e-12:
        return 0.0, n_pairs, int(len(a))
    return float(spearmanr(a, b)[0]), n_pairs, int(len(a))
