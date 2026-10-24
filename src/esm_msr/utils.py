import pandas as pd
import numpy as np
import re
import torch

from typing import Dict, Any, Optional

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

### DATA UTILS

def custom_end_gap_alignment(seq1, seq2, allow_gaps=False):
    if allow_gaps:
        print('allowing gaps!')
    # Define the scoring parameters
    match_score = 2  # Reward for matches
    mismatch_penalty = -1  # Penalty for mismatches
    gap_open_penalty = -10 if not allow_gaps else 0 # High penalty to prevent gaps in the middle
    gap_extend_penalty = -10  # High penalty for extending gaps in the middle

    # Align with no penalties for gaps at the ends using pairwise2's global alignment
    alignments = pairwise2.align.globalms(seq1, seq2, 
                                         match_score, 
                                         mismatch_penalty, 
                                         gap_open_penalty, 
                                         gap_extend_penalty,
                                         penalize_extend_when_opening=True, 
                                         penalize_end_gaps=False)

    # Pick the best alignment
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, start, end = best_alignment

    # Determine which sequence is shorter and calculate the offset
    if len(seq1) >= len(seq2):
        shorter_aligned_seq = aligned_seq2
    else:
        shorter_aligned_seq = aligned_seq1
    
    # Count the number of leading gaps in the shorter aligned sequence
    offset = len(shorter_aligned_seq) - len(shorter_aligned_seq.lstrip('-'))
    
    return offset, format_alignment(*best_alignment), score


def determine_diffs(aligned_seq1, aligned_seq2):
    # Get the list of differences (position, wild_type, mutation)
    differences = []
    position = 0  # Position in the aligned sequences

    # Adjust to ensure we're comparing the correct subsequence
    for i, (mut, wt) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if mut == '-' or wt == '-':
            if i < len(aligned_seq1) - 1 and (aligned_seq1[i+1] == '-' or aligned_seq2[i+1] == '-'):
                # Skip over any leading or trailing gaps, keep position unchanged
                continue
        if mut != wt and mut != '-' and wt != '-':
            # Record the position and the difference
            differences.append((wt, position+1, mut))
        if mut != '-':
            # Only increment the position counter when seq1 moves forward
            position += 1

    return differences


def remove_duplicates_with_mean(df, groupby_cols, mean_col, preserve_index=False):
    """
    Remove duplicates by taking the mean of a specific column while keeping other shared values.
    """
    if df.empty:
        return df, df
        
    if preserve_index:
        df_temp = df.copy()
        df_temp['_orig_idx'] = df_temp.index
        
        # Aggregate the mean column, and keep the first original index for the group
        agg_dict = {mean_col: 'mean', '_orig_idx': 'first'}
        result = df_temp.groupby(groupby_cols, dropna=False).agg(agg_dict).reset_index()
        
        # Restore the preserved index
        result.index = result['_orig_idx']
        result.index.name = df.index.name 
        result = result.drop(columns=['_orig_idx'])
    else:
        result = df.groupby(groupby_cols, as_index=False, dropna=False)[mean_col].mean()
        
    duplicates = df[df.duplicated(subset=groupby_cols, keep=False)]
    
    if len(result) == 0 and len(df) > 0:
        raise AssertionError("remove_duplicates_with_mean resulted in an empty dataframe unexpectedly.")
        
    return result, duplicates

### TRAINING UTILS

def _ensure_tensor(x, dtype=None, device=None):
    """Coerce to tensor once; avoid needless copies."""
    if x is None:
        return None
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

def _get_label(batch: Dict[str, Any], key: str, device: str) -> Optional[torch.Tensor]:
    """Accept both new and legacy keys; put onto module device with float dtype when numeric."""
    if key == 'ddG':
        if 'ddG' in batch:
            return _ensure_tensor(batch['ddG'], dtype=torch.float, device=device)
        if 'ground_truth' in batch:  # legacy alias
            return _ensure_tensor(batch['ground_truth'], dtype=torch.float, device=device)
        return None
    elif key == 'dddG':
        if 'dddG' in batch:
            return _ensure_tensor(batch['dddG'], dtype=torch.float, device=device)
        return None
    else:
        return _ensure_tensor(batch.get(key, None), device=device)

def _aa_to_token_id(tokenizer, aa: str) -> int:
    """Convert one-letter amino acid to tokenizer id."""
    try:
        return int(tokenizer.vocab[aa])
    except Exception as e:
        raise KeyError(f"Tokenizer missing amino acid token '{aa}': {e}")
    
def _normalize_batch(batch):
    # unwrap common patterns produced by custom loaders
    if isinstance(batch, list):
        if len(batch) == 1 and isinstance(batch[0], dict):
            return batch[0]
        # You can add other adapter cases here if needed.
        raise TypeError(f"Unexpected batch list structure: {type(batch[0])}")
    if not isinstance(batch, dict):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    return batch
    
# =============================
# Masking: per-modality functions
# =============================
# =============================
# Position mapping & light validation
# =============================
# =============================
# Master masking manager
# =============================
@torch.no_grad()
def l2_weight_norm(params):
    """√(Σ ||p||²). Works whether tensors live on CPU or GPU."""
    total = torch.zeros([], device=params[0][1].device if params else "cpu")
    for _, p in params:
        total += (p.detach() ** 2).sum()
    return total.sqrt().item()

@torch.no_grad()
def l2_grad_norm(params):
    """√(Σ ||∇p||²) over params that have grads."""
    accum = None
    for _, p in params:
        if p.grad is None: 
            continue
        g = p.grad.detach()
        s = (g ** 2).sum()
        if accum is None:
            accum = s
        else:
            accum = accum + s
    if accum is None:
        return 0.0
    return accum.sqrt().item()

@torch.no_grad()
def group_step_norm(params, lr):
    accum = None
    for _, p in params:
        if p.grad is None: continue
        s = (lr * p.grad.detach()).pow(2).sum()
        accum = s if accum is None else (accum + s)
    if accum is None:
        return 0.0
    return accum.sqrt().item()

def slice_batch_by_index(batch: Dict[str, Any], idx: torch.Tensor) -> Dict[str, Any]:
    """
    Return a shallow-sliced view of `batch` selecting rows in `idx` along dim 0.
    Slices tensors with batch-dim == len(batch['mutations']) and lists of that length.
    Leaves other entries untouched.
    """
    B = len(batch['ddG'])
    idx_list = idx.tolist()

    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == B:
            out[k] = v.index_select(0, idx)
        elif isinstance(v, list) and len(v) == B:
            out[k] = [v[i] for i in idx_list]
        else:
            out[k] = v
    return out

def sort_mutations_by_position(df, input_col, output_col='sorted_mutations'):
    """
    Sort mutations from low to high position within each mutation string.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    input_col : str
        Name of column containing mutation strings (e.g., 'D55N:M6N')
    output_col : str
        Name of output column for sorted mutations (default: 'sorted_mutations')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with new column containing sorted mutations
    
    Examples:
    ---------
    >>> df = pd.DataFrame({'mutations': ['D55N:M6N', 'A100G:C20T:B50D']})
    >>> df = sort_mutations_by_position(df, 'mutations')
    >>> df['sorted_mutations'].tolist()
    ['M6N:D55N', 'C20T:B50D:A100G']
    """
    def sort_single_mutation_string(mut_string):
        if pd.isna(mut_string) or mut_string == '':
            return mut_string
        
        # Split by colon to get individual mutations
        mutations = mut_string.split(':')
        
        # Extract position from each mutation using regex
        # Pattern: one or more letters, followed by digits, followed by one or more letters
        def get_position(mutation):
            match = re.search(r'[A-Za-z]+(\d+)[A-Za-z]+', mutation)
            if match:
                return int(match.group(1))
            else:
                # If pattern doesn't match, return a large number to sort it last
                return float('inf')
        
        # Sort mutations by their position
        sorted_mutations = sorted(mutations, key=get_position)
        
        # Join back with colons
        return ':'.join(sorted_mutations)
    
    # Apply the sorting function to the input column
    df = df.copy()
    df[output_col] = df[input_col].apply(sort_single_mutation_string)
    
    return df

def sum_individual_mutation_scores(df, score_column, new_score_column=None):
    """
    Calculates additive scores for higher-order mutations by exploding 
    constituent mutations and merging with single mutation reference data.
    Uses an internal ID system to prevent index corruption.
    """
    if new_score_column is None:
        new_score_column = f"{score_column}_additive"
        
    result_df = df.copy()
    
    # FIX: Generate a strictly unique internal tracking ID to prevent all index broadcasting errors
    result_df['_internal_row_id'] = range(len(result_df))
    result_df[new_score_column] = np.nan
    
    is_combined = result_df['mut_type'].str.contains(':', na=False)
    
    if not is_combined.any():
        return result_df.drop(columns=['_internal_row_id'])
        
    singles_df = result_df[~is_combined]
    
    duplicates_mask = singles_df.duplicated(subset=['mut_type', 'code'], keep=False)
    if duplicates_mask.any():
        # Fixed logic: Count unique pairs of (mut_type, code), not just mut_type
        num_dupes = singles_df[duplicates_mask][['mut_type', 'code']].drop_duplicates().shape[0]
        print(f"Warning: Found {num_dupes} unique single mutation/code pairs with multiple entries. Their scores will be averaged.")
    
    lookup_table = singles_df.groupby(['mut_type', 'code'])[score_column].mean().reset_index()
    lookup_table = lookup_table.rename(columns={'mut_type': 'single_mut_type'})
    
    # Isolate subset and track using our guaranteed unique internal ID
    combined_subset = result_df.loc[is_combined, ['mut_type', 'code', '_internal_row_id']].copy()
    
    # Calculate expected counts and map them to the internal ID
    expected_counts = combined_subset['mut_type'].str.count(':') + 1
    expected_counts.index = combined_subset['_internal_row_id']
    
    combined_subset['constituent'] = combined_subset['mut_type'].str.split(':')
    
    # FIX: Explode natively keeps the _internal_row_id attached to every expanded constituent
    exploded = combined_subset.explode('constituent')
    
    # FIX (Additional): Strip accidental whitespace to prevent silent merge failures
    exploded['constituent'] = exploded['constituent'].str.strip()
    
    merged = pd.merge(
        exploded,
        lookup_table,
        left_on=['constituent', 'code'],
        right_on=['single_mut_type', 'code'],
        how='left'
    )
    
    # FIX: Group by the explicit, unique internal ID
    aggregated = merged.groupby('_internal_row_id').agg(
        total_score=(score_column, 'sum'),
        found_count=(score_column, 'count') 
    )
    
    # FIX: Align expected_counts strictly to the aggregated index
    valid_mask = aggregated['found_count'] == expected_counts.loc[aggregated.index]
    valid_sums = aggregated.loc[valid_mask, 'total_score']
    
    # FIX: Map back to the result dataframe using the internal ID, completely ignoring the original index
    valid_sums_dict = valid_sums.to_dict()
    assignment_mask = result_df['_internal_row_id'].isin(valid_sums_dict.keys())
    result_df.loc[assignment_mask, new_score_column] = result_df.loc[assignment_mask, '_internal_row_id'].map(valid_sums_dict)
    
    missing_count = (~valid_mask).sum()
    if missing_count > 0:
        failed_internal_ids = aggregated[~valid_mask].index
        failed_exploded = merged[merged['_internal_row_id'].isin(failed_internal_ids)]
        
        missing_entirely = failed_exploded['single_mut_type'].isna().sum()
        present_but_nan = (failed_exploded['single_mut_type'].notna() & failed_exploded[score_column].isna()).sum()
        
        #print(f"Warning: {missing_count} combined mutations were skipped.")
        if missing_entirely > 0:
            print(f"  -> {missing_entirely} constituent single mutations are missing from the dataset.")
        if present_but_nan > 0:
            print(f"  -> {present_but_nan} constituent single mutations were found but possess NaN scores.")
            
    # Clean up the internal tracking column before returning
    return result_df.drop(columns=['_internal_row_id'])

def parse_mutation_spec(mut_spec: str) -> dict:
    """
    Parse a single mutation specification like 'A12C' into components.
    
    Args:
        mut_spec: Mutation string in format 'A12C' (from_aa, position, to_aa)
    
    Returns:
        dict with keys: 'wild_type' (from), 'position' (position), 'mutation' (to)
        Returns None values if parsing fails
    """
    # Pattern: single letter, one or more digits, single letter
    match = re.match(r'^([A-Z])(\d+)([A-Z])$', mut_spec.strip())
    
    if match:
        return {
            'wild_type': match.group(1),
            'position': int(match.group(2)),
            'mutation': match.group(3)
        }
    else:
        return {'wild_type': None, 'position': None, 'mutation': None}
    
def parse_multimutant_column(
    df: pd.DataFrame,
    mut_column: str = 'mutation',
    max_mutations: int = 2,
    separator: str = ':',
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Parse colon-separated multi-mutant specifications into separate columns.
    
    Args:
        df: Input DataFrame containing mutation specifications
        mut_column: Name of column containing mutation specs (e.g., 'A12C:R14Q')
        max_mutations: Maximum number of mutations to parse (2 for doubles, 3 for triples, etc.)
        separator: Character separating individual mutations (default ':')
        drop_original: Whether to drop the original mutation column
    
    Returns:
        DataFrame with added columns: wt1, pos1, mut1, wt2, pos2, mut2, ... (up to max_mutations)
    
    Examples:
        >>> df = pd.DataFrame({'mutation': ['A12C:R14Q', 'V5L:G8P:H20Y', 'K3R']})
        >>> parse_multimutant_column(df, max_mutations=3)
           mutation  wt1  pos1 mut1  wt2  pos2  mut2  wt3  pos3  mut3
        0  A12C:R14Q    A    12   C    R  14.0    Q  NaN   NaN  NaN
        1  V5L:G8P:H20Y V     5   L    G   8.0    P    H  20.0    Y
        2        K3R    K     3   R  NaN   NaN  NaN  NaN   NaN  NaN
    """
    result_df = df.copy()
    
    # Initialize columns for each mutation position
    for i in range(1, max_mutations + 1):
        result_df[f'wt{i}'] = None
        result_df[f'pos{i}'] = None
        result_df[f'mut{i}'] = None
    
    # Process each row
    for idx, row in result_df.iterrows():
        mut_spec = str(row[mut_column])
        
        # Split by separator
        individual_muts = mut_spec.split(separator)
        
        # Parse each individual mutation
        for i, mut in enumerate(individual_muts[:max_mutations], start=1):
            parsed = parse_mutation_spec(mut)
            result_df.at[idx, f'wt{i}'] = parsed['wild_type']
            result_df.at[idx, f'pos{i}'] = parsed['position']
            result_df.at[idx, f'mut{i}'] = parsed['mutation']
    
    # Convert position columns to nullable integer type
    for i in range(1, max_mutations + 1):
        result_df[f'pos{i}'] = pd.to_numeric(result_df[f'pos{i}'], errors='coerce')
    
    if drop_original:
        result_df = result_df.drop(columns=[mut_column])
    
    return result_df