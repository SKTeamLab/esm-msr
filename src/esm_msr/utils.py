import pandas as pd
from sklearn import metrics
import numpy as np

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3
from esm.utils.constants import esm3 as C

from typing import List, Dict, Any, Optional, Tuple

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.PDB import *
from Bio.PDB.DSSP import DSSP

from peft import LoraConfig, get_peft_model
import logging
import re

import os
import atexit
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


### LORA UTILS

def load_ckpt_weights(model, checkpoint_path: str, device: str = 'cuda:0'):
    """
    Loads LoRA weights from a checkpoint into the current model.
    Assumes checkpoint['state_dict'] contains keys with 'lora'.
    Loads with strict=False.
    """
    # Check if the current model instance actually has LoRA parameters
    has_lora_params = any('lora' in name for name, _ in model.named_parameters())
    if not has_lora_params:
        raise KeyError(f"Attempting to load LoRA weights into a model ({type(model).__name__}) that appears to have no LoRA parameters. Skipping load.")

   # try:
    logging.info(f"Attempting to load LoRA weights from checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    # Checkpoint might contain only trainable weights (LoRA/Head) or full state_dict
    state_dict_to_load = checkpoint_data.get('state_dict', checkpoint_data)

    if not state_dict_to_load:
        raise KeyError(f"No state_dict found in checkpoint: {checkpoint_path}")

    # Filter for lora keys specifically from the loaded state_dict
    lora_keys_in_checkpoint = {k: v for k, v in state_dict_to_load.items() if ('lora' in k or 'calibration' in k)}
    if not lora_keys_in_checkpoint:
        raise KeyError(f"No keys containing 'lora' or 'calibration' found in the loaded state_dict from {checkpoint_path}. Cannot load LoRA weights.")

    # Load weights into the current model with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(lora_keys_in_checkpoint, strict=False)

    # Report issues related to LoRA keys specifically
    actual_missing_lora = [k for k in missing_keys if ('lora' in k or 'calibration' in k)]

    if unexpected_keys:
        # This is only a warning if the checkpoint contained more than just LoRA
        logging.warning(f"Unexpected keys found while loading LoRA weights (might be OK if checkpoint had other params): {unexpected_keys}")
    if actual_missing_lora:
            # This is usually an error - model expects LoRA params not in checkpoint
            logging.error(f"LoRA weights MISSING from checkpoint but expected by model: {actual_missing_lora}")

    total_elements = sum(p.numel() for p in lora_keys_in_checkpoint.values())
    logging.info(f"Successfully loaded {len(lora_keys_in_checkpoint)} parameter tensors ({total_elements:,} total elements) from {checkpoint_path} into the current model.")

    #except FileNotFoundError:
    #    logging.error(f"Checkpoint file not found: {checkpoint_path}")
    #except Exception as e:
    #    logging.error(f"Error loading LoRA weights from {checkpoint_path}: {e}", exc_info=True)

    return model

def add_lora_to_esm3(
    model: ESM3, 
    lora_rank: int = 6, 
    lora_alpha: int = 12, 
    lora_dropout: float = 0.15, 
    target_mode: str = "expanded",  # Options: "baseline", "expanded", "all"
    use_dora: bool = False, 
    include_structure_encoder: bool = False, 
    seed: int = None
):
    
    # 1. Force Initialization of Structure Encoder
    # ESM3 loads this lazily. We MUST init it now so we can freeze it.
    # Otherwise, it might load later (e.g. during PEFT setup) with requires_grad=True.
    if hasattr(model, "get_structure_encoder"):
        # This triggers the download/init of the encoder weights
        _ = model.get_structure_encoder()

    # 2. Freeze Global Parameters
    # Now that all submodules (including the encoder) are loaded, freeze everything.
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. Define Target Modules based on Mode
    # "baseline": QKV only (Standard LoRA practice)
    # "expanded": FFN + Output Proj (Your previous default)
    # "all": QKV + FFN + Output Proj (Maximal coverage)
    
    targets = []
    
    if target_mode == "baseline":
        # Target only the fused QKV projection in standard attention
        targets.append(r"layernorm_qkv\.1")

    elif target_mode == "ffn":
        # Targets FFN up/down
        targets = ["1", "3"]
        
    elif target_mode == "expanded":
        # Targets FFN up/down and Attention Output
        targets = ["out_proj", "1", "3"]
        
    elif target_mode == "all":
        # Targets everything linear in the blocks
        targets = ["out_proj", "1", "3", r"layernorm_qkv\.1"]
        
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    target_pattern = "|".join(targets)
    
    # 4. Construct Regex Scope
    # ^transformer matches the main generator.
    # We strictly limit scope to the transformer blocks to avoid hitting heads/embeddings accidentally.
    
    if not include_structure_encoder:
        # Standard: Only main transformer
        # Matches: transformer.blocks.0.attn.layernorm_qkv.1
        target_modules_regex = f"^transformer\\.blocks\\.\\d+\\.(attn|geom_attn|ffn)\\.({target_pattern})$"
    else:
        # Expanded: Main transformer + Structure Encoder (if it exists)
        # Note: Structure Encoder uses 'GeometricReasoning' which lacks 'layernorm_qkv'.
        # So in 'baseline' mode, structure encoder might get 0 adapters if it has no matches.
        target_modules_regex = f".*(?:transformer|structure_encoder)\\.(?:blocks|layers)\\.\\d+\\..*({target_pattern})$"

    print(f"Initializing LoRA with mode '{target_mode}' targeting regex: {target_modules_regex}")

    # 5. Configure LoRA
    lora_config = {
        "target_modules": target_modules_regex,
        "lora_dropout": lora_dropout,
        "lora_alpha": lora_alpha,
        "r": lora_rank,
        "use_rslora": True,
        "bias": "none",
    }
    
    if use_dora:
        lora_config['use_dora'] = True

    config = LoraConfig(**lora_config)

    # 6. Seeding
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 7. Inject Adapters
    peft_model = get_peft_model(model, config)

    # 8. Verification & Stats
    trainable_params = 0
    all_params = 0
    lora_param_count = 0
    
    for name, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if any(x in name for x in ["lora_", "dora_"]):
                lora_param_count += 1
            else:
                print(f"WARNING: Non-LoRA parameter is trainable: {name}")
                param.requires_grad = False
                trainable_params -= param.numel()

    if lora_param_count == 0:
        print("WARNING: No LoRA adapters were added! Check your regex and include_structure_encoder settings.")
        print(f"Attempted regex: {target_modules_regex}")

    print(f"Trainable LoRA parameters: {trainable_params:,} ({trainable_params/all_params:.4%} of total)")
    
    return peft_model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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


def is_fake_mutation(mut_string):
    # Split by colon to handle multiple mutations
    mutations = mut_string.split(':')
    
    for mutation in mutations:
        # Use regex to extract source, position, and target
        match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', mutation)
        if match:
            source = match.group(1)
            target = match.group(3)
            # Check if source equals target (improper mutation)
            if source == target:
                return True
    
    return False


def is_improper_mutation(mutation_string: str) -> bool:
    """
    Checks a string containing protein mutations for conflicts.

    A conflict occurs if two mutations at the same position are inconsistent.
    Specifically, if mutation M1 (X1##Y1) and mutation M2 (X2##Y2) both occur
    at position ##, and M1 appears before M2 in the string, then Y1 (the
    result of M1) must be equal to X2 (the starting amino acid for M2).
    If this condition is violated for any such pair, a conflict exists.

    Mutations are expected in the format X##Y (e.g., L45S), where X and Y are
    uppercase letters and ## is one or more digits.
    The pattern X##Y should not be extracted if X is immediately preceded by a
    digit (e.g., the 'A2H' in '1A2H' will not be considered a mutation).
    Mutations in the string can be separated by underscores ('_'), colons (':'),
    or other characters.

    Args:
        mutation_string: The string to check, e.g., "1ANP_L45S_L73P:G81T"
                         or "A12C_C12G_G12T".

    Returns:
        True if no conflicting mutations are found, False otherwise.
        Returns True for strings with no mutations or only one mutation
        at any given site.
    """
    # Refined Regex: X##Y, not preceded by a digit.
    # (?<!\d): Negative lookbehind asserting not preceded by a digit.
    # ([A-Z]): Group 1 (wild type)
    # (\d+):   Group 2 (position)
    # ([A-Z]): Group 3 (mutant)
    mutation_pattern = re.compile(r"(?<!\d)([A-Z])(\d+)([A-Z])")

    mutations = []
    for match in mutation_pattern.finditer(mutation_string):
        # Groups are still 1, 2, 3 as lookbehind is zero-width
        wild_type = match.group(1)
        position = int(match.group(2))
        mutant = match.group(3)
        mutations.append({
            "wild_type": wild_type,
            "position": position,
            "mutant": mutant,
            "original_string": match.group(0)
        })

    if not mutations:
        return True

    mutations_by_position = {}
    for mut in mutations:
        pos = mut["position"]
        if pos not in mutations_by_position:
            mutations_by_position[pos] = []
        mutations_by_position[pos].append(mut)

    for position, pos_mutations in mutations_by_position.items():
        if len(pos_mutations) > 1:
            for i in range(len(pos_mutations) - 1):
                current_mutation = pos_mutations[i]
                next_mutation = pos_mutations[i+1]
                if current_mutation["mutant"] != next_mutation["wild_type"]:
                    return True
    
    return False

### ANALYSIS UTILS
    
def compute_ndcg_flexible(df, pred_col, true_col, *,
                          top_n=None, percentile=None, threshold=None,
                          ignore_ties=True):
    """
    Compute a variant of NDCG with one of:
      - top_n (int): NDCG@N
      - percentile (float in (0,1]): NDCG@ceil(percentile * n)
      - threshold (float): drop rows with true_col < threshold, then full-list NDCG

    Exactly one of {top_n, percentile, threshold} must be specified.

    Returns:
        float (NDCG) or np.nan on degenerate cases.
    """
    # --- arg validation ---
    flags = [top_n is not None, percentile is not None, threshold is not None]
    if sum(flags) != 1:
        raise ValueError("Specify exactly one of top_n, percentile, or threshold.")

    df = df[[pred_col, true_col]].copy()
    df = df.dropna()
    if df.empty:
        return np.nan

    # --- threshold mode: filter first, then full-list NDCG ---
    if threshold is not None:
        df = df[df[true_col] >= threshold]
        if len(df) < 2:
            return np.nan
        y_score = df[pred_col].to_numpy().reshape(1, -1)
        y_true = df[true_col].to_numpy()

        # ensure nonnegative relevance for ndcg
        min_val = np.min(y_true)
        if min_val < 0:
            y_true = y_true - min_val

        # if all-zero relevance, ndcg undefined
        if np.all(y_true == 0):
            return np.nan

        y_true = y_true.reshape(1, -1)
        return metrics.ndcg_score(y_true, y_score, k=None, ignore_ties=ignore_ties)

    # --- top_n / percentile: choose k on the unfiltered list ---
    y_score = df[pred_col].to_numpy().reshape(1, -1)
    y_true = df[true_col].to_numpy()

    # ensure nonnegative relevance
    min_val = np.min(y_true)
    if min_val < 0:
        y_true = y_true - min_val

    if np.all(y_true == 0) or y_true.size < 2:
        return np.nan

    n = y_true.size
    if top_n is not None:
        if top_n <= 0:
            return np.nan
        k = min(int(top_n), n)
    else:
        # percentile mode
        if not (0 < percentile <= 1):
            raise ValueError("percentile must be in (0, 1].")
        k = max(1, int(np.ceil(percentile * n)))
        k = min(k, n)

    y_true = y_true.reshape(1, -1)
    return metrics.ndcg_score(y_true, y_score, k=k, ignore_ties=ignore_ties)
    

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

def _assert_L_alignment(
    batch: Dict[str, Any],
    *,
    use_orig: bool = True,
    where: str = "pre-mask"
) -> None:
    """
    Asserts that structure_tokens and coords have a residue axis length L matching sequence.

    Args:
    use_orig: check *_orig keys (pre-mask) vs masked keys (post-mask).
    where: label for clearer error messages.

    Raises:
    AssertionError with a precise, actionable message if lengths are incompatible.
    """
    key_seq  = 'sequence_tokens_orig'  if use_orig else 'sequence_tokens'
    key_str  = 'structure_tokens_orig' if use_orig else 'structure_tokens'
    key_crd  = 'coords_orig'           if use_orig else 'coords'

    seq = batch.get(key_seq,  None)
    st  = batch.get(key_str,  None)
    crd = batch.get(key_crd,  None)

    allow_struct_shift = 0
    allow_coords_shift = 0

    if seq is None:
        raise AssertionError(f"[{where}] Missing '{key_seq}' in batch.")

    # Sequence L (we require BOS/EOS-free here; seq is [B, L])
    if not isinstance(seq, torch.Tensor) or seq.ndim != 2:
        raise AssertionError(f"[{where}] '{key_seq}' must be a 2D tensor [B, L], got shape {getattr(seq, 'shape', None)}.")
    L_seq = int(seq.size(-1))

    # Structure L
    if st is not None and not st.sum() < 0:
        if not isinstance(st, torch.Tensor) or st.ndim < 2:
            raise AssertionError(f"[{where}] '{key_str}' must be rank ≥2 tensor, got shape {getattr(st, 'shape', None)}.")
        L_str = int(st.size(-1))
        if (L_str - L_seq) != allow_struct_shift:
            raise AssertionError(
                f"[{where}] structure length mismatch: seq L={L_seq}, struct L={L_str}, "
                f"expected shift={allow_struct_shift}. (shape {tuple(st.shape)})"
            )

    # Coords L
    if crd is not None:
        if not isinstance(crd, torch.Tensor) or crd.ndim < 2:
            raise AssertionError(f"[{where}] '{key_crd}' must be rank ≥2 tensor, got shape {getattr(crd, 'shape', None)}.")
        L_crd = int(crd.size(-3))

        if (L_crd - L_seq) != allow_coords_shift:
            raise AssertionError(
                f"[{where}] coords length mismatch: seq L={L_seq}, coords L={L_crd}, "
                f"expected shift={allow_coords_shift}. (shape {tuple(crd.shape)})"
            )

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

def _mask_structure_rows(
    str_tok: Optional[torch.Tensor],      # [B, K, L] or [B, L] long, or None
    cols_per_row: List[List[int]],
    flank: int,
    struct_mask_id: int,
):
    """
    Mask structure tokens along the L (residue) axis at columns (±flank), in-place.
    """
    if str_tok is None:
        return
    if str_tok.ndim not in (2, 3):
        raise AssertionError("structure tokens should be [B,L] or [B,K,L].")
    B = str_tok.shape[0]
    L = str_tok.shape[-1]
    for i in range(B):
        cols = cols_per_row[i]
        if not cols:
            continue
        win = set()
        for c in cols:
            a = max(0, c - flank)
            b = min(L - 1, c + flank)
            win.update(range(a, b + 1))
        idx = torch.as_tensor(sorted(win), device=str_tok.device, dtype=torch.long)
        if str_tok.ndim == 3:
            str_tok[i, :, idx] = struct_mask_id
        else:
            str_tok[i, idx] = struct_mask_id

def _mask_coords_rows(
    crd: Optional[torch.Tensor],          # [B, L, ...] or [B, K, L, ...] float, or None
    cols_per_row: List[List[int]],
    flank: int,
):
    """
    Mask coordinates (set to NaN) along the L axis at columns (±flank), in-place.
    """
    if crd is None:
        return
    if crd.ndim not in (4, 5):
        raise AssertionError("coords should be [B,L,...] or [B,K,L,...].")
    B = crd.shape[0]
    L = crd.shape[-3] # choose the L axis
    for i in range(B):
        cols = cols_per_row[i]
        if not cols:
            continue
        win = set()
        for c in cols:
            a = max(0, c - flank)
            b = min(L - 1, c + flank)
            win.update(range(a, b + 1))
        idx = torch.as_tensor(sorted(win), device=crd.device, dtype=torch.long)
        if crd.ndim == 4:
            crd[i, idx, ...] = float('nan')
        elif crd.ndim == 5:
            crd[i, :, idx, ...] = float('nan')
        else:
            raise AssertionError
        
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
    Polymorphic conditional batch builder.
    
    Behavior:
    1. Paired Mode: If 'sequence_tokens_mut' exists, it produces 'sequence_tokens_wt' 
       and 'sequence_tokens_mut' for a marginal Delta-LL calculation.
       - condition='wt': WT vs Mutation_at_Which
       - condition='mut': Mutation_at_Other vs Double_Mutation
       
    2. Legacy Mode: If only 'sequence_tokens' exists, it produces a single 
       'sequence_tokens' tensor with the requested masking/identities.
    """
    assert which in ('A', 'B'), "which must be 'A' or 'B'"
    muts = batch_d['mutations']
    
    # Identify Mode
    is_paired = 'sequence_tokens_mut' in batch_d and 'sequence_tokens_wt' in batch_d
    
    # 1. Fetch Base Tensors
    mask_id = tokenizer.vocab["<mask>"]

    # 2. Initialize Output Batch
    out = {k: v for k, v in batch_d.items() if not k.startswith('sequence_tokens') 
           and not k.startswith('coords') and not k.startswith('structure_tokens')}
    
    new_muts = []
    
    # Prepare Output Tensors
    if is_paired:
        seq_wt_new = batch_d['sequence_tokens_wt'].clone()
        seq_mut_new = batch_d['sequence_tokens_mut'].clone()
        Nd, L = seq_wt_new.shape
        
        # Structural tensors for doubles (subs) typically don't change between states,
        # so we clone the original WT ones for both sides of the pair.
        out['coords_wt'] = batch_d['coords_wt'].clone()
        out['coords_mut'] = batch_d['coords_mut'].clone()
        out['structure_tokens_wt'] = batch_d['structure_tokens_wt'].clone()
        out['structure_tokens_mut'] = batch_d['structure_tokens_mut'].clone()
        out['residue_index_wt'] = batch_d['residue_index_wt'].clone()
        out['residue_index_mut'] = batch_d['residue_index_mut'].clone()
    else:
        seq_new = batch_d['sequence_tokens_orig'].clone()
        Nd, L = seq_new.shape
        # Carry over structural tensors as-is
        out['coords'] = batch_d.get('coords', batch_d.get('coords_orig')).clone()
        out['structure_tokens'] = batch_d.get('structure_tokens', batch_d.get('structure_tokens_orig')).clone()
        #out['residue_index'] = batch_d.get('residue_index', batch_d.get('residue_index_orig')).clone()

    for i in range(Nd):
        (wtA, posA, mtA), (wtB, posB, mtB) = muts[i]
        
        # Helper to get tokens
        t_wtA, t_mtA = _aa_to_token_id(tokenizer, wtA), _aa_to_token_id(tokenizer, mtA)
        t_wtB, t_mtB = _aa_to_token_id(tokenizer, wtB), _aa_to_token_id(tokenizer, mtB)

        if is_paired:
            # --- PAIRED LOGIC (For Delta Mean LL) ---
            if which == 'A':
                if condition == 'wt':
                    # Path: WT -> A
                    # seq_wt stays WT
                    seq_mut_new[i, posA] = t_mtA
                elif condition == 'mut':
                    # Path: B -> AB
                    seq_wt_new[i, posB] = t_mtB
                    seq_mut_new[i, posA] = t_mtA
                    seq_mut_new[i, posB] = t_mtB
                elif condition == 'mask':
                    seq_wt_new[i, posA] = mask_id
                    seq_wt_new[i, posB] = mask_id
                    seq_mut_new[i, posA] = t_mtA
                    seq_mut_new[i, posB] = mask_id
            else: # which == 'B'
                if condition == 'wt':
                    # Path: WT -> B
                    seq_mut_new[i, posB] = t_mtB
                elif condition == 'mut':
                    # Path: A -> AB
                    seq_wt_new[i, posA] = t_mtA
                    seq_mut_new[i, posA] = t_mtA
                    seq_mut_new[i, posB] = t_mtB
                elif condition == 'mask':
                    seq_wt_new[i, posA] = mask_id
                    seq_wt_new[i, posB] = mask_id
                    seq_mut_new[i, posB] = t_mtB
                    seq_mut_new[i, posA] = mask_id
        else:
            # --- LEGACY LOGIC (Single Sequence) ---
            if which == 'A':
                seq_new[i, posA] = mask_id
                if condition == 'mask': seq_new[i, posB] = mask_id
                elif condition == 'wt': seq_new[i, posB] = t_wtB
                else: seq_new[i, posB] = t_mtB
            else: # which == 'B'
                seq_new[i, posB] = mask_id
                if condition == 'mask': seq_new[i, posA] = mask_id
                elif condition == 'wt': seq_new[i, posA] = t_wtA
                else: seq_new[i, posA] = t_mtA

        # Update mutation record for the forward pass
        if rewrite_mutations:
            new_muts.append([(wtA, posA, mtA)] if which == 'A' else [(wtB, posB, mtB)])

    # 4. Finalize
    if is_paired:
        out['sequence_tokens_wt'] = seq_wt_new
        out['sequence_tokens_mut'] = seq_mut_new
    else:
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
        
        print(f"Warning: {missing_count} combined mutations were skipped.")
        if missing_entirely > 0:
            print(f"  -> {missing_entirely} constituent single mutations are completely missing from the dataset.")
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