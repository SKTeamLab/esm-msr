import pandas as pd
import numpy as np
import re
import os
import atexit
import torch

from esm.utils.constants import esm3 as C

from typing import List, Dict, Any, Optional, Tuple

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.PDB import *
from Bio.PDB.DSSP import DSSP
from Bio.PDB import MMCIFParser, PDBIO, Select

### Constants

RESIDUE_MAP = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
               'MSE': 'M'} # Map Selenomethionine to Methionine

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX' # ProteinMPNN alphabet including 'X' for unknown/mask
AA_ALPHABET = ALPHABET[:-1] # Alphabet excluding 'X'

### PDB utils

# --- Global list to keep track of temporary files ---
_temp_files = []

def _cleanup_temp_files():
    """Remove temporary files created during execution."""
    files_to_remove = list(_temp_files)
    for temp_file in files_to_remove:
        try:
            if os.path.exists(temp_file): 
                os.remove(temp_file)
                _temp_files.remove(temp_file) 
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
        except ValueError:
             pass 

# Register cleanup on import so it runs when the main script exits
atexit.register(_cleanup_temp_files)

def register_temp_file(filepath):
    """Adds a file path to the list of temporary files to be cleaned up."""
    if filepath not in _temp_files:
        _temp_files.append(filepath)

def convert_cif_to_pdb(cif_path, pdb_path):
    """
    Converts a CIF file to a PDB file using Biopython.
    """
    print(f"Converting CIF file: {cif_path} to PDB: {pdb_path}")
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('cif_structure', cif_path)
        print(f"Successfully parsed CIF structure with {len(list(structure.get_models()))} model(s).")

        io = PDBIO()
        io.set_structure(structure)
        
        class StandardResidueSelect(Select):
            def accept_residue(self, residue):
                # Standard residues usually have a blank hetero-field (res.id[0])
                # and are in the RESIDUE_MAP
                return residue.id[0] == ' ' and residue.get_resname() in RESIDUE_MAP

            def accept_atom(self, atom):
                 # Only accept standard ATOM records, not HETATM unless it's MSE
                 parent_resname = atom.get_parent().get_resname()
                 return atom.get_parent().id[0] == ' ' or parent_resname == 'MSE'

        io.save(pdb_path, select=StandardResidueSelect())
        print(f"Successfully wrote PDB file: {pdb_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Input CIF file not found at {cif_path}")
        return False
    except Exception as e:
        print(f"An error occurred during CIF to PDB conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


### DATA UTILS

def get_residue_accessibility(model, filename, target_chain):
    """
    Run DSSP to determine the absolute surface area of each residue
    """
    print(filename)
    dssp_dict = dict(DSSP(model, filename, dssp='mkdssp'))
    # get the target chain only
    df = pd.DataFrame(dssp_dict).T.loc[target_chain, :]
    df.index = pd.Series(df.index).apply(lambda x: x[1])
    df = df.rename({1:'wild_type', 2: 'SS', 3: 'rel_ASA'}, axis=1
        )[['wild_type', 'SS', 'rel_ASA']]
    df = df[df['wild_type'].str.contains('^[ACDEFGHIKLMNPQRSTVWY]+$')]
    df_encoded = pd.get_dummies(df, columns=['SS'])
    #print(df_encoded)
    return df_encoded


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
    
def _double_indices(batch: Dict[str, Any], device: str, finite_dddG: bool = False) -> torch.Tensor:
    """
    Returns long tensor of indices i where:
    - len(mutations[i]) == 2 (double mutant), and
    - dddG[i] is finite (available target) if flagged
    """
    muts = batch['mutations']
    idx_d = torch.tensor([i for i, m in enumerate(muts) if len(m) == 2], device=device, dtype=torch.long)

    if not finite_dddG:
        return idx_d

    else:
        dddG = _get_label(batch, 'dddG', device)
        if dddG is None or idx_d.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=device)

        valid = torch.isfinite(dddG.index_select(0, idx_d))
        return idx_d[valid]
    
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
def _mask_sequence_rows(
    seq: torch.Tensor,                    # [B, L], long
    cols_per_row: List[List[int]],        # zero-based columns per row
    flank: int,
    mask_id: int,
):
    """
    Mask sequence at columns (±flank) for each row, in-place on `seq`.
    """
    B, L = seq.shape
    for i in range(B):
        cols = cols_per_row[i]
        if not cols:
            continue
        # Expand with flanks
        win = set()
        for c in cols:
            a = max(0, c - flank)
            b = min(L - 1, c + flank)
            win.update(range(a, b + 1))
        idx = torch.as_tensor(sorted(win), device=seq.device, dtype=torch.long)
        seq[i, idx] = mask_id
        
# =============================
# Position mapping & light validation
# =============================
def _map_mutations_to_cols_and_validate(
    seq_tensor: torch.Tensor,                      # [B, L]
    mutations_batch: List[List[Tuple[str, int, str]]], 
    tokenizer
) -> List[List[int]]:
    """
    Validates WT consistency and returns 0-based column indices for mutations.
    """
    B, L = seq_tensor.shape
    cols_per_row = []
    
    # We pull this to CPU once to avoid device sync in the loop if tensor is on GPU
    seq_cpu = seq_tensor.detach().cpu()
    
    for i in range(B):
        row_cols = []
        row_muts = mutations_batch[i]
        row_seq = seq_cpu[i]
        
        for (wt, pos, _mt) in row_muts:
            # Contract: positions are 1-based w.r.t sequence_tokens_orig
            # We convert to 0-based index here
            j = pos
            
            if not (0 <= j < L):
                raise AssertionError(f"Row {i}: Mutation position {pos} out of bounds for L={L}.")
            
            # Validation
            expected_id = _aa_to_token_id(tokenizer, wt)
            got_id = int(row_seq[j].item())
            
            if got_id != expected_id:
                raise AssertionError(
                    f"Row {i}: WT mismatch at position {pos}. "
                    f"Expected token id {expected_id} ('{wt}'), got {got_id}. "
                    f"Ensure sequence_tokens_orig has NO BOS/EOS and positions are 0-based/aligned."
                )
            row_cols.append(j)
        cols_per_row.append(row_cols)
        
    return cols_per_row
    
def make_conditional_batch_doubles(
    batch_d: Dict[str, Any],
    which: str,                         # 'A' or 'B' -> the site we are focused on
    tokenizer: Any,
    condition: str = 'wt',              # 'mask' | 'wt' | 'mut'
    rewrite_mutations: bool = True
) -> Dict[str, Any]:
    """
    Prepares a batch for conditional single-mutant inference from double mutants.
    Masks the target site ('which') and sets the context site based on 'condition'.
    """
    assert which in ('A', 'B'), "which must be 'A' or 'B'"
    muts = batch_d['mutations']
    
    mask_id = tokenizer.vocab["<mask>"]

    # Initialize Output Batch, carrying over structural tensors untouched
    out = {k: v for k, v in batch_d.items() if not k.startswith('sequence_tokens')}
    
    seq_new = batch_d['sequence_tokens_orig'].clone()
    Nd, L = seq_new.shape
    
    new_muts = []

    for i in range(Nd):
        (wtA, posA, mtA), (wtB, posB, mtB) = muts[i]
        
        t_wtA, t_mtA = _aa_to_token_id(tokenizer, wtA), _aa_to_token_id(tokenizer, mtA)
        t_wtB, t_mtB = _aa_to_token_id(tokenizer, wtB), _aa_to_token_id(tokenizer, mtB)

        if which == 'A':
            # Target is A: Mask it.
            seq_new[i, posA] = mask_id
            
            # Context is B: Set according to condition.
            if condition == 'mask': 
                seq_new[i, posB] = mask_id
            elif condition == 'wt': 
                seq_new[i, posB] = t_wtB
            else: 
                seq_new[i, posB] = t_mtB
        else:
            # Target is B: Mask it.
            seq_new[i, posB] = mask_id
            
            # Context is A: Set according to condition.
            if condition == 'mask': 
                seq_new[i, posA] = mask_id
            elif condition == 'wt': 
                seq_new[i, posA] = t_wtA
            else: 
                seq_new[i, posA] = t_mtA

        if rewrite_mutations:
            new_muts.append([(wtA, posA, mtA)] if which == 'A' else [(wtB, posB, mtB)])

    # Finalize
    out['sequence_tokens'] = seq_new
    if rewrite_mutations:
        out['mutations'] = new_muts
    
    return out

def _generate_pos_mask(
    batch_size: int,
    length: int,
    cols_per_row: List[List[int]],
    flank: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates a (B, L) boolean tensor masking specifically the mutated positions 
    (and their flanks). Deterministic.
    """
    mask = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
    
    for i, cols in enumerate(cols_per_row):
        if not cols:
            continue
        
        # Optimization: If flank is 0, use direct indexing
        if flank == 0:
            idx = torch.tensor(cols, device=device, dtype=torch.long)
            mask[i, idx] = True
        else:
            # With flanks, iterate to handle range clamping
            for c in cols:
                start = max(0, c - flank)
                end = min(length, c + flank + 1)
                mask[i, start:end] = True
    return mask

def _generate_random_mask(
    batch_size: int,
    max_length: int,
    lengths: torch.Tensor, 
    fraction: float,
    device: torch.device
) -> torch.Tensor:
    """
    Generates a (B, L) boolean tensor masking a random fraction of valid tokens.
    Sampling is done without replacement.
    """
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    
    if fraction <= 0.0:
        return mask
        
    # Iterate rows to sample correct count per sequence length
    for i in range(batch_size):
        # Determine valid length for this sequence
        l_val = int(lengths[i].item()) if lengths is not None else max_length
        if l_val == 0: continue
            
        # If fraction >= 1.0, mask the whole valid sequence
        if fraction >= 1.0:
            mask[i, :l_val] = True
            continue
            
        num_to_mask = int(l_val * fraction + 0.5)
        if num_to_mask == 0: continue
        
        # Random sample without replacement
        perm = torch.randperm(l_val, device=device)[:num_to_mask]
        mask[i, perm] = True
        
    return mask

# =============================
# Master masking manager
# =============================
def apply_masks(
    batch: Dict[str, Any],
    tokenizer: Any,
    *,
    mask_sequence_pos: bool = True,
    mask_structure_pos: bool = False,
    mask_coords_pos: bool = False,
    mask_sequence_fraction: float = 0.0, # Float allows 0.0 (off) to 1.0 (all)
    mask_structure_fraction: float = 0.0,
    mask_coords_fraction: float = 0.0,
    flank_seq: int = 0,
    flank_struct: int = 0,
    flank_coords: int = 0,
    struct_mask_id: int = C.STRUCTURE_MASK_TOKEN,
    from_originals: bool = True,
    skip: bool = False
) -> Dict[str, Any]:
    """
    All-in-one masking entrypoint. 
    
    - _pos args: Deterministically mask mutation sites (+ flanks).
    - _fraction args: Stochastically mask the whole sequence (0.0 to 1.0).
    - If configs match across modalities, the random masks are synchronized.
    """
    if skip:
        try:
            batch['sequence_tokens'] = batch['sequence_tokens_orig']
            batch['structure_tokens'] = batch['structure_tokens_orig']
            batch['coords'] = batch['coords_orig']
            return batch
        except KeyError:
            return batch
    device = batch['sequence_tokens_orig'].device

    # Pull sources
    seq_src = 'sequence_tokens_orig' if from_originals else 'sequence_tokens'
    str_src = 'structure_tokens_orig' if from_originals else 'structure_tokens'
    crd_src = 'coords_orig' if from_originals else 'coords'

    seq_orig = _ensure_tensor(batch.get(seq_src, None), dtype=torch.long, device=device)
    str_orig = _ensure_tensor(batch.get(str_src, None), dtype=torch.long, device=device)
    crd_orig = _ensure_tensor(batch.get(crd_src, None), dtype=torch.float, device=device)

    if seq_orig is None:
        raise ValueError("Batch is missing sequence tokens.")

    B, L_seq = seq_orig.shape
    
    # Attempt to retrieve real lengths for accurate random masking
    lengths = batch.get('lengths')
    if lengths is None:
        # Fallback: assume no padding or infer from mask token if available
        lengths = torch.full((B,), L_seq, device=device, dtype=torch.long)

    # -- 1. Map mutations to columns and validate WT --
    muts: List[List[Tuple[str, int, str]]] = batch['mutations']
    cols_per_row = _map_mutations_to_cols_and_validate(seq_orig, muts, tokenizer)

    # -- 2. Clone working copies --
    out = dict(batch)
    out['sequence_tokens'] = seq_orig.clone()
    if str_orig is not None:
        out['structure_tokens'] = str_orig.clone()
    if crd_orig is not None:
        out['coords'] = crd_orig.clone()

    mask_token_id = tokenizer.vocab["<mask>"]

    # Helpers to store masks for potential reuse (consistency)
    # Keys: 'pos' (from mutation logic) and 'frac' (from random logic)
    seq_masks = {'pos': None, 'frac': None}
    
    # ===========================
    # 3. Sequence Masking
    # ===========================
    final_seq_mask = None
    
    # A. Positional (Mutations)
    if mask_sequence_pos:
        seq_masks['pos'] = _generate_pos_mask(B, L_seq, cols_per_row, flank_seq, device)
        final_seq_mask = seq_masks['pos']
        
    # B. Fractional (Random)
    if mask_sequence_fraction > 0.0:
        seq_masks['frac'] = _generate_random_mask(B, L_seq, lengths, mask_sequence_fraction, device)
        if final_seq_mask is None:
            final_seq_mask = seq_masks['frac']
        else:
            final_seq_mask = final_seq_mask | seq_masks['frac']
            
    if final_seq_mask is not None:
        out['sequence_tokens'].masked_fill_(final_seq_mask, mask_token_id)

    # ===========================
    # 4. Structure Masking
    # ===========================
    if (str_orig is not None) and (mask_structure_pos or mask_structure_fraction > 0.0):
        final_str_mask = None
        str_masks = {'pos': None, 'frac': None}

        # A. Positional (Reuse Seq if config identical, else generate)
        if mask_structure_pos:
            if mask_sequence_pos and flank_struct == flank_seq:
                str_masks['pos'] = seq_masks['pos'] # Reuse
            else:
                str_masks['pos'] = _generate_pos_mask(B, L_seq, cols_per_row, flank_struct, device)
            final_str_mask = str_masks['pos']

        # B. Fractional (Reuse Seq if config identical, else generate)
        if mask_structure_fraction > 0.0:
            if mask_sequence_fraction == mask_structure_fraction:
                str_masks['frac'] = seq_masks['frac'] # Reuse (Consistency!)
            else:
                str_masks['frac'] = _generate_random_mask(B, L_seq, lengths, mask_structure_fraction, device)
            
            if final_str_mask is None:
                final_str_mask = str_masks['frac']
            else:
                final_str_mask = final_str_mask | str_masks['frac']

        # Apply
        if final_str_mask is not None:
            # Broadcast if structure tokens are (B, K, L)
            if out['structure_tokens'].ndim == 3:
                mask_view = final_str_mask.unsqueeze(1)
            else:
                mask_view = final_str_mask
            out['structure_tokens'].masked_fill_(mask_view, struct_mask_id)
            
    # ===========================
    # 5. Coords Masking
    # ===========================
    if (crd_orig is not None) and (mask_coords_pos or mask_coords_fraction > 0.0):
        final_crd_mask = None
        
        # Reuse logic checks Sequence first, then Structure
        
        # A. Positional
        if mask_coords_pos:
            if mask_sequence_pos and flank_coords == flank_seq:
                crd_pos = seq_masks['pos']
            elif mask_structure_pos and flank_coords == flank_struct:
                # Use the locally stored structure mask if generated
                # Note: We rely on the fact that if mask_structure_pos was True, we generated it above
                crd_pos = _generate_pos_mask(B, L_seq, cols_per_row, flank_struct, device) # Re-gen cheaper than tracking 'str_masks' scope complexity
            else:
                crd_pos = _generate_pos_mask(B, L_seq, cols_per_row, flank_coords, device)
            final_crd_mask = crd_pos

        # B. Fractional
        if mask_coords_fraction > 0.0:
            if mask_sequence_fraction == mask_coords_fraction:
                crd_frac = seq_masks['frac']
            elif mask_structure_fraction == mask_coords_fraction and 'str_masks' in locals() and str_masks['frac'] is not None:
                crd_frac = str_masks['frac']
            else:
                crd_frac = _generate_random_mask(B, L_seq, lengths, mask_coords_fraction, device)
            
            if final_crd_mask is None:
                final_crd_mask = crd_frac
            else:
                final_crd_mask = final_crd_mask | crd_frac

        # Apply
        if final_crd_mask is not None:
            if out['coords'].ndim == 4: 
                # (B, L, A, C) -> mask is (B, L) -> unsqueeze last 2
                mask_view = final_crd_mask.unsqueeze(-1).unsqueeze(-1)
            elif out['coords'].ndim == 5:
                # (B, K, L, A, C) -> mask is (B, L) -> unsqueeze 1 (K) and last 2 (A, C)
                mask_view = final_crd_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected coords dimensions: {out['coords'].shape}")
                
            out['coords'].masked_fill_(mask_view, float('nan'))

    return out

def _select_lora_params(named_params):
    """Filter LoRA adapter tensors by name convention (PEFT: 'lora_A', 'lora_B', etc.)."""
    lora = []
    for name, p in named_params:
        if p.requires_grad and ("lora" in name.lower()):  # adjust if your naming differs
            lora.append((name, p))
    return lora

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

def group_step_norm(params, lr):
    s = 0.0
    for _, p in params:
        if p.grad is None: continue
        s += (lr * p.grad.detach()).pow(2).sum().item()
    return s ** 0.5

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

def generate_ids(pdbs, mutations):
    ids = []
    for pdb, mutation in zip(pdbs, mutations):
        #print(pdb, mutation)
        mut = ':'.join([m[0] + str(m[1]) + m[2] for m in mutation])
        id = pdb+'_'+mut
        ids.append(id)
    return ids
    
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