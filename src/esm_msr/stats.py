import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score

def safe_spearman(preds, targets):
    if len(preds) < 2 or len(np.unique(targets)) < 2: return np.nan 
    if len(np.unique(preds)) < 2: return np.nan 
    rho = spearmanr(preds, targets)[0]
    return np.nan if np.isnan(rho) else float(rho)

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

def compute_metrics(wt_scores, comb_scores, epi_scores, epi_scores_full, ground_truths, dddG_truths, idx_singles, idx_doubles, valid_dddG_mask):
    """
    Consolidated function computing Spearman, NDCG, and RMSE metrics.
    """
    metrics = {
        'rho': {
            'wt': safe_spearman(wt_scores, ground_truths), 
            'combined': safe_spearman(comb_scores, ground_truths)
        },
        'ndcg@k=96': {
            'wt': safe_ndcg_k96(wt_scores, ground_truths),
            'combined': safe_ndcg_k96(comb_scores, ground_truths)
        },
        'ndcg>0': {
            'wt': safe_ndcg_t0(wt_scores, ground_truths),
            'combined': safe_ndcg_t0(comb_scores, ground_truths)
        },
        'rho_singles': {
            'wt': safe_spearman(wt_scores[idx_singles], ground_truths[idx_singles]), 
            'combined': safe_spearman(comb_scores[idx_singles], ground_truths[idx_singles])
        },
        'rho_doubles': {
            'wt': safe_spearman(wt_scores[idx_doubles], ground_truths[idx_doubles]), 
            'combined': safe_spearman(comb_scores[idx_doubles], ground_truths[idx_doubles])
        },
        'rho_dddG_heuristic': {
            'epi': safe_spearman(epi_scores[valid_dddG_mask], dddG_truths[valid_dddG_mask]) if valid_dddG_mask.any() else np.nan
        },
        'rho_dddG': {
            'epi': safe_spearman(epi_scores_full[valid_dddG_mask], dddG_truths[valid_dddG_mask]) if valid_dddG_mask.any() else np.nan
        },
        'rmse': {
            'wt': np.sqrt(mean_squared_error(ground_truths, wt_scores)) if len(wt_scores) else np.nan, 
            'combined': np.sqrt(mean_squared_error(ground_truths, comb_scores)) if len(comb_scores) else np.nan
        }
    }
    return metrics