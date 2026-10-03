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


def compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types):
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

    return {
        'rho_wt_valid': safe_spearman(wt_scores[is_single], gt[is_single]),
        'rho_wt_all': safe_spearman(wt_scores[finite], gt[finite]),
        'rho_mt_valid': safe_spearman(mt_scores[is_cond], gt[is_cond]),
        'rho_mt_all': safe_spearman(mt_scores[finite], gt[finite]),
        'rho_combined': safe_spearman(comb_scores[is_measured], gt[is_measured]),
        'rmse_combined': safe_rmse(comb_scores[is_measured], gt[is_measured]),
    }
