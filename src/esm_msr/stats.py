"""Validation metrics for one dataloader (one protein library, or one benchmark)."""
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
    ok = np.isfinite(preds) & np.isfinite(targets)
    if not ok.any(): return np.nan
    return float(np.sqrt(mean_squared_error(targets[ok], preds[ok])))

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

def compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, obs=None):
    """
    The ddG validation metrics for one dataloader.

    * ``rho_singles_wt`` - WT head on the library's single mutations (``rho_wt_valid`` before 2026-10-24).
    * ``rho_singles_combined`` - the reported two-path average on the same singles: what the MT half does to them.
    * ``rho_doubles_wt`` / ``rho_doubles_combined`` - the WT head (additive: the sum of its singles) and the reported
      average on the library's doubles; NaN for a library without doubles. Their difference is what the MT half buys
      on the items it exists for.
    * ``rho_combined``   - the reported two-path average, on measured items only (singles, doubles,
      native-background items).
    * ``rmse_combined``  - calibration of that average in kcal/mol. With ``obs`` (a dict whose ``comb`` holds the
      predictions on the assay's OBSERVED scale, ``h(dG_wt + latent) - dG_wt``, for a model trained with
      ``--link``) it is computed there, so it stays comparable with a model trained without the link; rank metrics
      always use the latent scores, which ``h`` leaves alone because it is monotone.
    * ``ndcg@k=96`` / ``ndcg>0`` - top-of-list retrieval for the WT head and the combined path.

    Retired 2026-10-24: ``rho_wt_all`` / ``rho_mt_all`` (each head on singles and doubles mixed, which answered
    nothing the singles/doubles split does not) and ``rho_mt_valid`` (conditional items; validation relabels
    native_cond as single before this is called, so it was NaN on every library).

    Every item passed in is scored. Censoring is a property of the LOSS, not of the yardstick: scoring only the
    uncensored items makes a run's numbers incomparable to a baseline that censors nothing, and biases them by
    range restriction wherever censoring is common - the mutant-background validation libraries are gap-censored
    at 4.1% against 0.55% elsewhere, which cost them 0.015 rho and read as a model regression that was not there.
    Choosing WHICH items to score is therefore the caller's job (see
    ``ESM3EpistasisLightningModule.on_validation_epoch_end``, which scores the canonical measured population and
    holds back only the rows the assay never fitted).

    Returns a flat ``{name: value}`` dict; undefined entries are NaN.
    """
    gt = np.asarray(ground_truths, dtype=np.float64)
    wt_scores, mt_scores, comb_scores = (np.asarray(x) for x in (wt_scores, mt_scores, comb_scores))
    subset_types = np.asarray(subset_types)
    finite = np.isfinite(gt)

    is_single = np.isin(subset_types, list(routing.WT_HEAD_SUBSETS)) & finite
    is_double = (subset_types == 'double') & finite
    is_measured = np.isin(subset_types, list(routing.MEASURED_SUBSETS)) & finite

    metrics = {
        'rho_singles_wt': safe_spearman(wt_scores[is_single], gt[is_single]),
        'rho_singles_combined': safe_spearman(comb_scores[is_single], gt[is_single]),
        'rho_doubles_wt': safe_spearman(wt_scores[is_double], gt[is_double]),
        'rho_doubles_combined': safe_spearman(comb_scores[is_double], gt[is_double]),
        'rho_combined': safe_spearman(comb_scores[is_measured], gt[is_measured]),
        'rmse_combined': safe_rmse(comb_scores[is_measured], gt[is_measured]),
    }
    if obs is not None:
        oc = np.asarray(obs['comb'], dtype=np.float64)
        ok = is_measured & np.isfinite(oc)
        metrics['rmse_combined'] = safe_rmse(oc[ok], gt[ok]) if ok.any() else np.nan

    if is_measured.any():
        metrics['ndcg@k=96'] = safe_ndcg_k96(comb_scores[is_measured], gt[is_measured])
        metrics['ndcg>0'] = safe_ndcg_t0(comb_scores[is_measured], gt[is_measured])
        metrics['ndcg@k=96_wt'] = safe_ndcg_k96(wt_scores[is_measured], gt[is_measured])
        metrics['ndcg>0_wt'] = safe_ndcg_t0(wt_scores[is_measured], gt[is_measured])

    return metrics
