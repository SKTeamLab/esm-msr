"""ProteinMPNN (dauparas/ProteinMPNN at the commit in third_party.lock, unmodified; weights v_48_020): zero-shot
inference on every benchmark, with resource tracking.

The readout is the unmasked WT/MUT log-likelihood score (`score_unmasked_fast`, the one ProteinGym uses): for each
mutation, log p(mut) - log p(wt) with its position decoded last after a random background order, on the wild-type
sequence (mpnn_score_wt_seq) and on the variant's sequence (mpnn_score_mut_seq); mpnn_score is their mean, mpnn_score_additive
the wild-type one, mpnn_score_epistasis their difference. The model has no singles/doubles split: any number of mutations
is scored the same way, so --mutants only chooses which rows are scored.

  everything: python proteinmpnn_inference.py --out DIR      (Tsuboyama val + test, 12 external sets, 7 DMS, Domainome, ProteinGym)
  --mutants {single,multi,both}   which rows to score (default both); the others stay NaN and cost nothing
  --n_samples K --coord_noise A   test-time ensemble over K background orders (and noisy backbones); K=1 is the benchmark default

Rows are the shared benchmark rows (benchmark_rows.py). A mutant-background Tsuboyama library is scored on its own
background sequence (library_sequence), so those libraries are predicted too. Per-dataset and per-protein wall time, CPU,
RAM, VRAM and I/O of the scoring (not of featurizing the structure) are recorded by default
(inference_scripts/resource_tracker.py; --no_resource_tracking to skip).

The earlier row-by-row scorers (calculate_unmasked_scores, calculate_rigorous_scores and `predict`) stay below as the
reference implementation the fast scorer was checked against; they are not run by this script.
"""
# adapted from ProteinMPNN/protein_mpnn_utils.py

import os
import sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                 # paths, benchmark_rows
sys.path.insert(0, str(_HERE.parent))          # resource_tracker, pgym_runner
import paths  # noqa: E402
# ProteinMPNN checkout (dauparas/ProteinMPNN at the commit in third_party.lock: protein_mpnn_utils.py, vanilla_model_weights/)
MPNN_DIR = str(paths.PROTEINMPNN_DIR)
sys.path.append(MPNN_DIR)
from protein_mpnn_utils import tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN
from protein_mpnn_utils import gather_nodes, cat_neighbors_nodes

import os
import warnings
import torch
import copy
import argparse
import time
import itertools
import numpy as np
import random

from pathlib import Path
sys.path.insert(0, str(paths.REPO_ROOT / "src"))
from esm_msr import utils
import resource_tracker  # noqa: E402
import batching  # noqa: E402
import calibration  # noqa: E402
from batching import merge_info  # noqa: E402
from benchmark_rows import EXTERNAL, DMS, external_rows, dms_rows, domainome_rows, library_sequence  # noqa: E402

from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path


REPO_ROOT = paths.REPO_ROOT

def make_tied_positions_for_homomers(pdb_dict_list):
    """Causes identical sequences in a quaternary structure to have likelihoods influenced by each monomer"""
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1,chain_length+1):
            temp_dict = {}
            for _, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict


def calculate_unmasked_scores(model, row, X, S_wt, mask, chain_M, residue_idx, chain_encoding_all, alphabet, N_background_orders=1, batch_size=1):
    """
    Calculates purely contextual unmasked scores by evaluating the mutations 
    once on the full WT sequence and once on the fully Mutated sequence.
    """
    oc = -1
    device = X.device
    start_time = time.time()
    
    try:
        raw_muts = row['mut_type'].split(':')
        parsed_muts = []
        for m in raw_muts:
            parsed_muts.append({'wt': m[0], 'mut': m[-1], 'pos': int(m[1:-1])})
    except Exception as e:
        raise ValueError(f"Failed to parse mutations for row {row.get('mut_type', 'UNKNOWN')}. Exception: {e}")
    
    L = S_wt.shape[1]

    # Create fully mutated background
    S_mut = S_wt[0].clone()
    for pm in parsed_muts:
        p_idx = pm['pos'] + oc
        mut_idx = alphabet.find(pm['mut'])
        S_mut[p_idx] = mut_idx

    base_noise = torch.randn(N_background_orders, L, device=device)
    
    def get_scores_for_sequence(base_S):
        targets = []
        for pm in parsed_muts:
            p_idx = pm['pos'] + oc
            wt_idx = alphabet.find(pm['wt'])
            mut_idx = alphabet.find(pm['mut'])
            for n in range(N_background_orders):
                targets.append((p_idx, mut_idx, wt_idx, n))
                
        total_score = 0.0
        num_items = len(targets)
        all_deltas = []
        
        if num_items > 0:
            for i in range(0, num_items, batch_size):
                chunk_targets = targets[i:i+batch_size]
                B = len(chunk_targets)
                
                S_chunk = base_S.unsqueeze(0).expand(B, -1).contiguous()
                X_chunk = X[0].unsqueeze(0).expand(B, -1, -1, -1).contiguous()
                mask_chunk = mask[0].unsqueeze(0).expand(B, -1).contiguous()
                chain_M_chunk = chain_M[0].unsqueeze(0).expand(B, -1).contiguous()
                residue_idx_chunk = residue_idx[0].unsqueeze(0).expand(B, -1).contiguous()
                chain_encoding_chunk = chain_encoding_all[0].unsqueeze(0).expand(B, -1).contiguous()
                
                decoding_vals = torch.stack([base_noise[t[3]].clone() for t in chunk_targets])
                for b_idx, t in enumerate(chunk_targets):
                    decoding_vals[b_idx, t[0]] = 1000.0 # Target forced to end
                    
                decoding_order_chunk = torch.argsort(decoding_vals)
                
                with torch.no_grad():
                    logits = model.forward(
                        X_chunk, S_chunk, mask_chunk, chain_M_chunk, residue_idx_chunk, chain_encoding_chunk,
                        torch.randn_like(chain_M_chunk), use_input_decoding_order=True, decoding_order=decoding_order_chunk
                    )
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    for b_idx, t in enumerate(chunk_targets):
                        p_idx, mut_idx, wt_idx, _ = t
                        delta = (log_probs[b_idx, p_idx, mut_idx] - log_probs[b_idx, p_idx, wt_idx]).item()
                        all_deltas.append(delta)
                        
        for m_idx in range(len(parsed_muts)):
            start_idx = m_idx * N_background_orders
            end_idx = start_idx + N_background_orders
            scores = all_deltas[start_idx:end_idx]
            if scores:
                total_score += sum(scores) / len(scores)
                
        return total_score

    score_wt = get_scores_for_sequence(S_wt[0])
    score_mut = get_scores_for_sequence(S_mut)
    
    runtime = time.time() - start_time
    return score_wt, score_mut, runtime


def calculate_rigorous_scores(model, row, X, S_wt, mask, chain_M, residue_idx, chain_encoding_all, alphabet, K_paths=4, N_background_orders=1, batch_size=1):
    """
    Calculates Additive and Epistatic scores using a shared set of N background decoding orders.
    This ensures that differences between additive and epistatic scores are due purely to 
    context/coupling, not random autoregressive noise.
    """
    oc = -1
    device = X.device
    start_time = time.time()
    
    # --- 1. Parsing ---
    try:
        raw_muts = row['mut_type'].split(':')
        parsed_muts = []
        for m in raw_muts:
            parsed_muts.append({'wt': m[0], 'mut': m[-1], 'pos': int(m[1:-1])})
    except Exception as e:
        # FLAW FIX: Explicitly raise an error to prevent silent data loss downstream
        raise ValueError(f"Failed to parse mutations for row {row.get('mut_type', 'UNKNOWN')}. Exception: {e}")
    
    mut_indices = [pm['pos'] + oc for pm in parsed_muts]
    L = S_wt.shape[1]

    # --- 2. Generate Shared Background Noise ---
    # We generate N random tensors of shape [L]. 
    # These define the relative order of the background residues.
    # We will reuse these exact tensors for every calculation.
    base_noise = torch.randn(N_background_orders, L, device=device)
    
    # ==========================================
    # PART A: ADDITIVE SCORE (Sum of Singles)
    # ==========================================
    # We score each mutation independently on the WT background.
    
    # Track metadata explicitly to avoid list bloat and brittle pointers
    add_targets = [] # (p_idx, mut_idx, wt_idx, noise_idx)
    
    for pm in parsed_muts:
        p_idx = pm['pos'] + oc
        wt_idx = alphabet.find(pm['wt'])
        mut_idx = alphabet.find(pm['mut'])
        
        for n in range(N_background_orders):
            add_targets.append((p_idx, mut_idx, wt_idx, n))

    additive_score = 0.0
    num_additive_items = len(add_targets)
    all_add_deltas = [] # Store only the scalar deltas to prevent VRAM explosion
    
    if num_additive_items > 0:
        
        # Process forward passes in strict, capped chunks
        for i in range(0, num_additive_items, batch_size):
            chunk_targets = add_targets[i:i+batch_size]
            B = len(chunk_targets)
            
            # Memory-efficient expansion (replaces thousands of physical clones)
            S_chunk = S_wt[0].unsqueeze(0).expand(B, -1).contiguous()
            # FLAW FIX: Expanded to match the 4D shape of X (Batch, Length, Atoms, Coords)
            X_chunk = X[0].unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            mask_chunk = mask[0].unsqueeze(0).expand(B, -1).contiguous()
            chain_M_chunk = chain_M[0].unsqueeze(0).expand(B, -1).contiguous()
            residue_idx_chunk = residue_idx[0].unsqueeze(0).expand(B, -1).contiguous()
            chain_encoding_chunk = chain_encoding_all[0].unsqueeze(0).expand(B, -1).contiguous()
            
            # Construct Decoding Order for the chunk
            decoding_vals = torch.stack([base_noise[t[3]].clone() for t in chunk_targets])
            for b_idx, t in enumerate(chunk_targets):
                decoding_vals[b_idx, t[0]] = 1000.0 # Target forced to end
                
            decoding_order_chunk = torch.argsort(decoding_vals)
            
            with torch.no_grad():
                logits = model.forward(
                    X_chunk, S_chunk, mask_chunk, chain_M_chunk, residue_idx_chunk, chain_encoding_chunk,
                    torch.randn_like(chain_M_chunk), use_input_decoding_order=True, decoding_order=decoding_order_chunk
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Immediately extract scalars to discard massive graph/tensor overhead
                for b_idx, t in enumerate(chunk_targets):
                    p_idx, mut_idx, wt_idx, _ = t
                    delta = (log_probs[b_idx, p_idx, mut_idx] - log_probs[b_idx, p_idx, wt_idx]).item()
                    all_add_deltas.append(delta)
            
        # Accumulate Additive Scores
        for m_idx in range(len(parsed_muts)):
            # Elements appended sequentially: N_background_orders per mutation
            start_idx = m_idx * N_background_orders
            end_idx = start_idx + N_background_orders
            scores = all_add_deltas[start_idx:end_idx]
            additive_score += sum(scores) / len(scores)

    # ==========================================
    # PART B: EPISTATIC SCORE (Chain Rule)
    # ==========================================
    
    paths = []
    if len(mut_indices) == 1:
        paths.append(mut_indices)
    elif len(mut_indices) == 2:
        paths = list(itertools.permutations(mut_indices))
    else:
        for _ in range(K_paths):
            path = list(mut_indices)
            random.shuffle(path)
            paths.append(path)
            
    # Track explicit paths securely
    epi_targets = [] # (path_idx, step_idx, p_idx, mut_idx, wt_idx, noise_idx, current_seq)
    
    for path_idx, path in enumerate(paths):
        current_seq = S_wt[0].clone()
        
        for step_idx, p_idx in enumerate(path):
            pm = next(x for x in parsed_muts if (x['pos'] + oc) == p_idx)
            wt_idx = alphabet.find(pm['wt'])
            mut_idx = alphabet.find(pm['mut'])
            
            for n in range(N_background_orders):
                epi_targets.append((path_idx, step_idx, p_idx, mut_idx, wt_idx, n, current_seq.clone()))
                
            # Commit the mutation to the background context for downstream steps
            current_seq[p_idx] = mut_idx

    epistatic_score = 0.0
    num_epi_items = len(epi_targets)
    all_epi_deltas = []
    
    if num_epi_items > 0:
        # Process forward passes in strict, capped chunks
        for i in range(0, num_epi_items, batch_size):
            chunk_targets = epi_targets[i:i+batch_size]
            B = len(chunk_targets)
            
            # The mutated sequence background is unique per step, so we stack them
            S_chunk = torch.stack([t[6] for t in chunk_targets])
            
            # Expansion for static structure metadata
            # FLAW FIX: Expanded to match the 4D shape of X (Batch, Length, Atoms, Coords)
            X_chunk = X[0].unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            mask_chunk = mask[0].unsqueeze(0).expand(B, -1).contiguous()
            chain_M_chunk = chain_M[0].unsqueeze(0).expand(B, -1).contiguous()
            residue_idx_chunk = residue_idx[0].unsqueeze(0).expand(B, -1).contiguous()
            chain_encoding_chunk = chain_encoding_all[0].unsqueeze(0).expand(B, -1).contiguous()
            
            # Construct Decoding Order
            decoding_vals = torch.stack([base_noise[t[5]].clone() for t in chunk_targets])
            for b_idx, t in enumerate(chunk_targets):
                decoding_vals[b_idx, t[2]] = 1000.0 # Force target
                
            decoding_order_chunk = torch.argsort(decoding_vals)
            
            with torch.no_grad():
                logits = model.forward(
                    X_chunk, S_chunk, mask_chunk, chain_M_chunk, residue_idx_chunk, chain_encoding_chunk,
                    torch.randn_like(chain_M_chunk), use_input_decoding_order=True, decoding_order=decoding_order_chunk
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Immediately extract scalars
                for b_idx, t in enumerate(chunk_targets):
                    p_idx, mut_idx, wt_idx = t[2], t[3], t[4]
                    delta = (log_probs[b_idx, p_idx, mut_idx] - log_probs[b_idx, p_idx, wt_idx]).item()
                    all_epi_deltas.append(delta)
            
        # Accumulate Path Scores
        path_scores = []
        ptr = 0
        
        for path_idx, path in enumerate(paths):
            path_total = 0.0
            for step_idx in range(len(path)):
                step_sum = 0.0
                for _ in range(N_background_orders):
                    # FLAW FIX: Explicit validation of alignment to prevent brittle pointer tracking issues
                    assert epi_targets[ptr][0] == path_idx and epi_targets[ptr][1] == step_idx, \
                        "Fatal: Epistatic aggregation pointer out of sync with target logic."
                    
                    step_sum += all_epi_deltas[ptr]
                    ptr += 1
                path_total += (step_sum / N_background_orders)
            path_scores.append(path_total)
            
        epistatic_score = sum(path_scores) / len(path_scores)

    runtime = time.time() - start_time
    return epistatic_score, additive_score, runtime


def featurize_structure(code, pdb_path, chain, device):
    """ProteinMPNN features for one structure, with `chain` (plus identical-sequence chains, tied) as the designed
    chain(s) and other non-heteroatom chains as fixed context."""
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 
         'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R',
         'TRP': 'W', 'ALA': 'A', 'VAL':'V',  'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}
    pdbparser = PDBParser()
    structure = pdbparser.get_structure(code, pdb_path)

    drop_chains = []

    # get chain sequences and remove chains of only heteroatoms (e.g. DNA)
    for c in structure.get_chains():
        seq = [r.resname for r in c]
        seq = ''.join([d[res] if res in d.keys() else 'X' for res in seq])
        if set(seq) == {'X'}:
            drop_chains.append(c.id)    
    
    homomer=1
    designed_chain_list = []
    fixed_chain_list = []
    #target_chain = pdb_path.split('_')[-1].split('.')[0]
    target_chain = chain

    # identify the target chain and sequence, adding it to the designed chains
    for c in structure.get_chains():
        if c.id == target_chain:
            designed_chain_list.append(target_chain)
            target_seq = [r.resname for r in c]
            target_seq = ''.join([d[res] if res in d.keys() else 'X' for res in target_seq])
            break

    # identify chains with the exact same sequence as the target, adding to designed chains
    for c in structure.get_chains():
        if c.id != target_chain:
            candidate_seq = [r.resname for r in c]
            candidate_seq = ''.join([d[res] if res in d.keys() else 'X' for res in candidate_seq])
            #print(f'target_seq\n{target_seq}')
            #print(f'candid_seq\n{candidate_seq}')
            if candidate_seq == target_seq:
                designed_chain_list.append(c.id)
                homomer += 1
            elif c.id not in drop_chains:
                fixed_chain_list.append(c.id)
    
    chain_list = list(set(designed_chain_list + fixed_chain_list))

    homomer = bool(homomer-1)

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None

    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=100000)

    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None

    protein = dataset_valid[0]
    batch_clones = [copy.deepcopy(protein)]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list,\
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = \
        tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict,
            tied_positions_dict, pssm_dict, bias_by_res_dict)

    return X, S, mask, chain_M, residue_idx, chain_encoding_all


@torch.no_grad()
def mpnn_encode(model, X, mask, residue_idx, chain_encoding_all):
    """The structure half of ProteinMPNN.forward (features + encoder), for one structure (batch 1).

    Nothing here depends on the sequence or the decoding order, so it is computed once and shared by every
    decoder row of that structure. Returns what mpnn_decode_at needs.
    """
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    return {'h_V': h_V, 'h_E': h_E, 'E_idx': E_idx, 'h_EXV_encoder': h_EXV_encoder, 'mask': mask}


@torch.no_grad()
def mpnn_decode_at(model, enc, S, vals, p):
    """Log-probs at position p[b] of ProteinMPNN.forward for row b, given the cached encoder output.

    S [B, L] sequences, vals [B, L] decoding-order keys (a position is decoded before another iff its key is smaller;
    forward's decoding_order is argsort(vals)), p [B] the position read out of each row. Same arithmetic as forward,
    except (a) the order mask is a key comparison at the K neighbours instead of forward's B x L x L one-hot einsum
    (identical values for distinct keys) and (b) the last decoder layer is evaluated only at p (layers are per node,
    so the other nodes' last-layer states are never read). Returns [B, 21] log-probs.
    """
    E_idx, mask = enc['E_idx'], enc['mask']
    B, L = S.shape
    E_idx_b = E_idx.expand(B, -1, -1)
    h_S = model.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, enc['h_E'].expand(B, -1, -1, -1), E_idx_b)
    # order_mask_backward[b, q, E_idx[q, k]] = 1 iff that neighbour is decoded before q
    nb_vals = torch.gather(vals, 1, E_idx_b.reshape(B, -1)).view(E_idx_b.shape)
    mask_attend = (nb_vals < vals.unsqueeze(-1)).float().unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)
    h_EXV_encoder_fw = mask_fw * enc['h_EXV_encoder']
    h_V = enc['h_V'].expand(B, -1, -1)
    for layer in model.decoder_layers[:-1]:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx_b)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)
    rows = torch.arange(B, device=S.device)
    h_ESV_p = torch.cat([h_ES[rows, p], h_V[rows[:, None], E_idx[0, p]]], -1)        # [B, K, 3C]
    h_ESV_p = mask_bw[rows, p] * h_ESV_p + h_EXV_encoder_fw[rows, p]
    h_V_p = model.decoder_layers[-1](h_V[rows, p].unsqueeze(1), h_ESV_p.unsqueeze(1), mask[0, p].unsqueeze(1))
    return torch.log_softmax(model.W_out(h_V_p[:, 0]), dim=-1)


def score_unmasked_fast(model, mut_strings, X, S_wt, mask, residue_idx, chain_encoding_all, alphabet,
                        max_tokens=32768, seed=0, n_samples=1, coord_noise=0.0, batching=None, info_out=None):
    """calculate_unmasked_scores (N_background_orders=1) for every variant of one structure, without the redundant
    work of scoring them one at a time. Returns (score_wt, score_mut) as numpy arrays.

    For mutation (p, wt->mut) of a variant: log p(mut) - log p(wt) at p, p decoded last after a random background
    order, on the WT sequence (score_wt) and on the variant's sequence (score_mut), summed over its mutations.
    Because p is decoded last, nothing reads the residue at p, so the logits at p depend only on p, the background
    order and the residues at the variant's OTHER mutated positions. Two changes from calculate_unmasked_scores:
      - the background order is drawn once per position p (shared by every variant mutating p, in both passes)
        instead of once per variant; same distribution per term, so the same estimator.
      - with that, a decoder row is identified by (p, the other mutations): the WT pass of every variant at p is one
        row, the MUT pass of a single mutant IS that row, and multi-mutants sharing a context share a row.
    Each unique row is one decoder pass on the cached encoder (mpnn_encode / mpnn_decode_at).

    n_samples > 1 averages the scores over that many samples, each with its own background orders and, if
    coord_noise > 0, its own copy of the structure with N(0, coord_noise) A added to every backbone atom (the
    augmentation ProteinMPNN was trained with; v_48_020 used 0.20 A). A sample is shared by every variant, in
    both passes. Also returns the per-sample (score_wt, score_mut), [n_samples, n_variants] each. Sample 0 with
    coord_noise=0 is the single-sample result for the same seed.
    """
    device = X.device
    L = S_wt.shape[1]
    parsed = [[(int(m[1:-1]) - 1, alphabet.find(m[0]), alphabet.find(m[-1])) for m in s.split(':')] for s in mut_strings]
    wt_seq = S_wt[0]
    wt_list = wt_seq.tolist()
    bad = [s for s, muts in zip(mut_strings, parsed) if any(not (0 <= p < L) or wt_list[p] != wi for p, wi, _ in muts)]
    if bad:
        raise ValueError(f"{len(bad)} variants disagree with the structure sequence (first: {bad[0]})")

    key_ids, keys = {}, []
    terms = []    # (variant, pass 0=WT 1=MUT, row, wt_idx, mut_idx)
    def row_of(key):
        k = key_ids.get(key)
        if k is None:
            k = key_ids[key] = len(keys); keys.append(key)
        return k
    for r, muts in enumerate(parsed):
        for p, wi, mi in muts:
            k_wt = row_of((p, ()))
            k_mut = row_of((p, tuple(sorted((q, mq) for q, _, mq in muts if q != p))))
            terms.append((r, 0, k_wt, wi, mi)); terms.append((r, 1, k_mut, wi, mi))

    gen = torch.Generator(device=device).manual_seed(seed)
    t = np.array(terms, dtype=np.int64).reshape(-1, 5)
    B = max(1, max_tokens // L)
    scores = np.zeros((n_samples, len(parsed), 2))
    for s in range(n_samples):
        noise = torch.randn((L, L), generator=gen, device=device)   # row p: background order for target position p
        X_s = X + coord_noise * torch.randn(X.shape, generator=gen, device=device) if coord_noise > 0 else X
        enc = mpnn_encode(model, X_s, mask, residue_idx, chain_encoding_all)
        lp = torch.empty((len(keys), 21), dtype=torch.float32)

        def chunk_fn(i, j, enc=enc, noise=noise, lp=lp):
            chunk = keys[i:j]
            n = j - i
            p_t = torch.tensor([k[0] for k in chunk], device=device)
            vals = noise[p_t].clone()
            vals[torch.arange(n, device=device), p_t] = 1000.0   # target forced to end
            S_chunk = wt_seq.unsqueeze(0).repeat(n, 1)
            sub = [(b, q, mq) for b, k in enumerate(chunk) for q, mq in k[1]]
            if sub:
                b_i, q_i, m_i = (torch.tensor(x, device=device) for x in zip(*sub))
                S_chunk[b_i, q_i] = m_i
            lp[i:j] = mpnn_decode_at(model, enc, S_chunk, vals, p_t).float().cpu()

        if batching is None:                       # the original rule: max_tokens // L decoder rows per call
            for i in range(0, len(keys), B):
                chunk_fn(i, min(i + B, len(keys)))
        else:                                      # baselines/batching.py: dynamic against the GPU, or fixed (default max_tokens // L)
            info = batching.run(len(keys), chunk_fn, label=f'L={L} sample {s}', default_chunk=B)
            if info_out is not None:
                info_out.append(info)
        lp = lp.numpy().astype(np.float64)
        delta = lp[t[:, 2], t[:, 4]] - lp[t[:, 2], t[:, 3]]
        np.add.at(scores[s], (t[:, 0], t[:, 1]), delta)
    print(f"  {len(parsed)} variants -> {len(keys)} decoder rows x {n_samples} sample(s) (L={L}, batch {B})", flush=True)
    score = scores.mean(0)
    if n_samples == 1 and coord_noise == 0:
        return score[:, 0], score[:, 1]
    return score[:, 0], score[:, 1], scores[:, :, 0], scores[:, :, 1]


def make_pgym_score_fn(model, max_tokens, seed=0, n_samples=1, coord_noise=0.0):
    """score_fn for pgym_runner: ProteinMPNN unmasked WT/MUT scores of one ProteinGym DMS (columns as in predict),
    averaged over n_samples (orders, plus backbone noise if coord_noise > 0; see score_unmasked_fast), with each
    sample's mpnn_score as mpnn_score_s<k>."""
    device = next(model.parameters()).device
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    def score_fn(df):
        X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize_structure(
            df['code'].iloc[0], df['pdb_file'].iloc[0], df['chain'].iloc[0], device)
        t0 = time.time()
        tokens = max_tokens
        while True:
            try:
                res = score_unmasked_fast(model, df['mut_type_renumbered'].tolist(), X, S, mask,
                                          residue_idx, chain_encoding_all, alphabet,
                                          max_tokens=tokens, seed=seed, n_samples=n_samples, coord_noise=coord_noise)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tokens <= S.shape[1]:
                    raise
                tokens //= 2
                print(f"OOM; retrying {df['code'].iloc[0]} with max_tokens={tokens}", flush=True)
        score_wt, score_mut = res[:2]
        out = pd.DataFrame({'mpnn_score_wt_seq': score_wt, 'mpnn_score_mut_seq': score_mut})
        out['mpnn_score_additive'] = score_wt
        out['mpnn_score'] = 0.5 * (score_wt + score_mut)
        out['mpnn_score_epistasis'] = out['mpnn_score'] - score_wt
        if len(res) == 4:
            for k, (w, m) in enumerate(zip(res[2], res[3])):
                out[f'mpnn_score_s{k}'] = 0.5 * (w + m)
        out['runtime'] = (time.time() - t0) / len(df)
        torch.cuda.empty_cache()
        return out

    return score_fn


def predict(df, model, K_paths=4, use_masks=False):
    logps = df
    device = torch.device("cuda:0")

    with tqdm(total=len(df)) as pbar:
        for (code, chain), group in df.groupby(['code', 'chain']):
    
            pdb_path = os.path.join(group['pdb_file'].head(1).item())
            X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize_structure(code, pdb_path, chain, device)
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

            # --- Usage Example ---
            with torch.no_grad():
                for uid, row in group.iterrows():
                    
                    if use_masks:
                        epistatic_score, additive_score, runtime = calculate_rigorous_scores(
                            model, row, X, S, mask, chain_M, residue_idx, chain_encoding_all, alphabet, K_paths=K_paths, N_background_orders=1
                        )
                        
                        if additive_score is not None:
                            logps.at[uid, 'mpnn_score_additive'] = additive_score
                            logps.at[uid, 'mpnn_score'] = epistatic_score
                            logps.at[uid, 'mpnn_score_epistasis'] = epistatic_score - additive_score
                            logps.at[uid, 'runtime'] = runtime
                            
                    else:
                        score_wt, score_mut, runtime = calculate_unmasked_scores(
                            model, row, X, S, mask, chain_M, residue_idx, chain_encoding_all, alphabet, N_background_orders=1
                        )
                        
                        if score_wt is not None:
                            logps.at[uid, 'mpnn_score_wt_seq'] = score_wt
                            logps.at[uid, 'mpnn_score_mut_seq'] = score_mut
                            logps.at[uid, 'mpnn_score_additive'] = score_wt
                            logps.at[uid, 'mpnn_score'] = 0.5 * (score_wt + score_mut)
                            logps.at[uid, 'mpnn_score_epistasis'] = logps.at[uid, 'mpnn_score'] - score_wt
                            logps.at[uid, 'runtime'] = runtime
                            
                    if 'pbar' in locals(): pbar.update(1)

    logps.index.name = 'uid'
    return logps


def _limit(df, n):
    """--max_proteins: only the rows of the first n proteins (by `code`) of a frame; everything with n == 0."""
    if not n:
        return df
    return df[df['code'].isin(list(dict.fromkeys(df['code']))[:n])]


def _wanted(n_mut, mutants):
    n = np.asarray(n_mut)
    return {'single': n == 1, 'multi': n >= 2, 'both': np.ones(len(n), dtype=bool)}[mutants]


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
COLS = ('mpnn_score_wt_seq', 'mpnn_score_mut_seq', 'mpnn_score_additive', 'mpnn_score', 'mpnn_score_epistasis')


class Mpnn:
    def __init__(self, args, tracker):
        self.args, self.tracker = args, tracker
        self.batching = args.batching
        self.skipped = []          # structures that could not be featurized: the run exits non-zero if any
        self.device = torch.device('cuda:0')
        with tracker.stage('model_load'):
            model_name = 'v_48_020'                    # 48 edges, 0.20 A training noise
            checkpoint = torch.load(os.path.join(MPNN_DIR, 'vanilla_model_weights', f'{model_name}.pt'), map_location=self.device)
            self.model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                                     num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
                                     k_neighbors=checkpoint['num_edges'])
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

    def featurize(self, code, pdb_path, chain, seq=None):
        """Structure features; `seq` replaces the structure's sequence (a mutant-background Tsuboyama library)."""
        X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize_structure(code, pdb_path, chain, self.device)
        if seq is not None:
            assert S.shape[1] == len(seq), f'{code}: structure has {S.shape[1]} residues, sequence {len(seq)}'
            S = torch.tensor([[ALPHABET.find(c) for c in seq]], device=self.device)
        return X, S, mask, residue_idx, chain_encoding_all

    def score(self, name, feats, mut_strings):
        """The ProteinMPNN readouts of one structure (a dict of arrays aligned with mut_strings). The scoring is one
        `protein` scope; featurizing the structure is not in it."""
        X, S, mask, residue_idx, chain_encoding_all = feats
        a = self.args
        n_mut = np.array([s.count(':') + 1 for s in mut_strings])
        wanted = np.flatnonzero(_wanted(n_mut, a.mutants))
        out = {c: np.full(len(mut_strings), np.nan) for c in COLS}
        if not len(wanted):
            return out
        sel = [mut_strings[i] for i in wanted]
        infos = []
        with self.tracker.scope('protein', name, n_rows=len(wanted), seq_len=int(S.shape[1]), n_samples=a.n_samples,
                                coord_noise=a.coord_noise) as sc:
            with sc.phase('score'):
                res = score_unmasked_fast(self.model, sel, X, S, mask, residue_idx, chain_encoding_all, ALPHABET,
                                          max_tokens=a.max_tokens, seed=a.seed, n_samples=a.n_samples, coord_noise=a.coord_noise,
                                          batching=self.batching, info_out=infos)
            sc.note(**merge_info(infos))
        self.last_infos = infos
        w, m = res[:2]
        out['mpnn_score_wt_seq'][wanted] = w
        out['mpnn_score_mut_seq'][wanted] = m
        out['mpnn_score_additive'][wanted] = w
        out['mpnn_score'][wanted] = 0.5 * (w + m)
        out['mpnn_score_epistasis'][wanted] = 0.5 * (w + m) - w
        if len(res) == 4:
            for k, (wk, mk) in enumerate(zip(res[2], res[3])):
                col = out.setdefault(f'mpnn_score_s{k}', np.full(len(mut_strings), np.nan))
                col[wanted] = 0.5 * (wk + mk)
        torch.cuda.empty_cache()
        return out

    def score_table(self, df, label):
        """Score a rows frame (id, code, chain, pdb_file, mut_type, ddG) structure by structure."""
        cols = {}
        pos = {i: k for k, i in enumerate(df.index)}
        with self.batching.dataset_scope(self.tracker, label, n_rows=len(df)):
            for (pdb_path, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
                try:
                    feats = self.featurize(code, pdb_path, chain)
                except Exception as e:
                    print(f'Skipping {code}{chain}: failed featurizing: {e}')
                    self.skipped.append(f'{label}/{code}{chain}')
                    continue
                r = self.score(f'{code}{chain}', feats, g['mut_type'].tolist())
                idx = [pos[i] for i in g.index]
                for c, v in r.items():
                    cols.setdefault(c, np.full(len(df), np.nan))[idx] = v
        out = df.copy()
        for c, v in cols.items():
            out[c] = v
        return out

    def calibrate(self, args):
        """--calibrate_batch_sizes: the decoder-row batch for the heaviest proteins of every dataset (calibration.py)."""
        from esm_msr import preprocess_megascale
        only = calibration.parse_only(args.calibration_datasets)
        sets = {name: structs for name, (structs, _) in calibration.rows_structures(only).items()}
        want = [s for s in ('val', 'test') if only is None or f'tsuboyama_{s}' in only]
        if want and args.split:
            tsu = paths.TSUBOYAMA_DIR
            ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
                data_file=str(paths.require(tsu / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'the Tsuboyama table')),
                af_model_folder=str(paths.require(tsu / 'AlphaFold_model_PDBs', 'the AlphaFold models')))
            ds.create_training_splits(str(REPO_ROOT / 'data' / f'{args.split}.pkl'), -1)
            for sc in want:
                df = ds.split_dfs[sc]
                df = df[df['mut_structure'].fillna('-') == '-']
                structs = []
                for code in df['code_wt'].unique():
                    seq, pdb = library_sequence(df.loc[df['code_wt'] == code], '-')
                    structs.append({'pdb_file': pdb, 'code': code, 'chain': 'A', 'seq': seq, 'overwrite': seq})
                sets[f'tsuboyama_{sc}'] = structs

        def run_one(struct, muts):
            feats = self.featurize(struct['code'], struct['pdb_file'], struct['chain'], seq=struct.get('overwrite'))
            self.score('calibration', feats, muts)
            return {'batch_max': min(i['batch_max'] for i in self.last_infos), 'batch_ooms': sum(i['batch_ooms'] for i in self.last_infos)}

        sizes, detail = calibration.calibrate(sets, run_one, k=args.calibration_k, cap=args.calibration_cap)
        calibration.save_entries(args.calibrate_batch_sizes, {'proteinmpnn': sizes}, {'proteinmpnn': detail},
                                 calibration.gpu_meta(args.auto_batch_headroom, 'fp32', args.calibration_k, args.calibration_cap))

    def pgym_score_fn(self):
        def score_fn(df):
            feats = self.featurize(df['code'].iloc[0], df['pdb_file'].iloc[0], df['chain'].iloc[0])
            return pd.DataFrame(self.score(str(df['code'].iloc[0]), feats, df['mut_type_renumbered'].tolist()))
        return score_fn

    def tsuboyama(self, split_name, out_dir):
        """The Tsuboyama validation and test splits through the ESM-MSR preprocessor, one library at a time; every row,
        bounded ones included (flagged in `cens`), as esm_msr_testing.py predicts them."""
        from esm_msr import preprocess_megascale
        tsu = paths.TSUBOYAMA_DIR
        ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
            data_file=str(paths.require(tsu / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'the Tsuboyama table')),
            af_model_folder=str(paths.require(tsu / 'AlphaFold_model_PDBs', 'the AlphaFold models')))
        ds.create_training_splits(str(REPO_ROOT / 'data' / f'{split_name}.pkl'), -1)
        for scaffold_ in ('val', 'test'):
            data_scaffold = ds.split_dfs[scaffold_]
            parts = []
            with self.batching.dataset_scope(self.tracker, f'tsuboyama_{scaffold_}', n_rows=len(data_scaffold)):
                for code in data_scaffold['code_wt'].unique()[:self.args.max_proteins or None]:
                    df_true = data_scaffold.loc[data_scaffold['code_wt'] == code].copy()
                    df_true['mut_structure'] = df_true['mut_structure'].fillna('-')
                    df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                    df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']
                    for mut_structure, data in df_true.groupby('mut_structure'):
                        seq, pdb_path = library_sequence(data, mut_structure)
                        try:
                            feats = self.featurize(code, pdb_path, 'A', seq=seq)
                        except Exception as e:
                            print(f'Skipping {code}: failed featurizing: {e}')
                            self.skipped.append(f'tsuboyama_{scaffold_}/{code}/{mut_structure}')
                            continue
                        r = self.score(code if mut_structure == '-' else f'{code}_{mut_structure}', feats, data['mut_type'].tolist())
                        for c, v in r.items():
                            if c not in df_true.columns:
                                df_true[c] = np.nan
                            df_true.loc[data.index, c] = v
                    parts.append(df_true.reset_index().set_index('uid'))
            path = Path(out_dir) / f'tsuboyama_{scaffold_}.csv'
            pd.concat(parts, axis=0).to_csv(path)
            print('wrote', path)


def main(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.calibrate_batch_sizes:
        args.batch_mode, args.no_resource_tracking = 'auto', True
    args.batching = batching.Batching(args, model='proteinmpnn')
    tracker = resource_tracker.Tracker.from_args(
        args, label='proteinmpnn', default_dir=out / 'resources',
        meta={'model': 'ProteinMPNN v_48_020', 'mutants': args.mutants, 'dtype': 'fp32', 'seed': args.seed,
              'n_samples': args.n_samples, 'coord_noise': args.coord_noise, 'max_tokens': args.max_tokens, **args.batching.meta()})
    try:
        mpnn = Mpnn(args, tracker)
        if args.calibrate_batch_sizes:
            return mpnn.calibrate(args)
        if not args.skip_tsuboyama and args.split:
            mpnn.tsuboyama(args.split, out)
        if not args.skip_external:
            for name in EXTERNAL:
                dest = out / f'external_{name}.csv'
                if not (args.skip_existing and dest.exists()):
                    mpnn.score_table(_limit(external_rows(name), args.max_proteins), f'external_{name}').to_csv(dest, index=False)
        if not args.skip_dms:
            for assay in DMS[:args.max_proteins or None]:     # each assay is one protein: the smoke subset is the first N assays
                dest = out / f'dms_{assay}.csv'
                if not (args.skip_existing and dest.exists()):
                    mpnn.score_table(_limit(dms_rows(assay), args.max_proteins), f'dms_{assay}').to_csv(dest, index=False)
        if not args.skip_domainome:
            dest = out / 'domainome.csv'
            if not (args.skip_existing and dest.exists()):
                mpnn.score_table(_limit(domainome_rows(), args.max_proteins), 'domainome').to_csv(dest, index=False)
        if not args.skip_pgym:
            import pgym_runner
            args.batching.dataset = 'proteingym'
            pgym_runner.run_protein_gym(
                args.pgym_dir, out / 'proteingym', mpnn.pgym_score_fn(), pred_col='mpnn_score', dms=args.pgym_dms,
                tag=f'proteinmpnn v_48_020 unmasked seed={args.seed} n_samples={args.n_samples} coord_noise={args.coord_noise} '
                    f'mutants={args.mutants}', extra_cols=('mpnn_score_wt_seq',), tracker=tracker)
    finally:
        tracker.close()
    if mpnn.skipped:
        sys.exit(f'{len(mpnn.skipped)} structure(s) skipped, their predictions are NaN: {mpnn.skipped[:8]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='output directory (CSV per benchmark, proteingym/, resources/)')
    parser.add_argument('--mutants', choices=['single', 'multi', 'both'], default='both',
                        help='which rows to score (the others get NaN predictions and no inference)')
    parser.add_argument('--split', default='hyperopt_splits', help='Tsuboyama split pickle under data/ (without .pkl)')
    for b in ('tsuboyama', 'external', 'dms', 'domainome', 'pgym'):
        parser.add_argument(f'--skip_{b}', action='store_true')
    parser.add_argument('--max_proteins', type=int, default=0, help='smoke test: only the first N proteins of every benchmark (0 = all); ProteinGym: use --pgym_dms')
    parser.add_argument('--skip_existing', action='store_true', help='skip an external/DMS/Domainome output that exists')
    parser.add_argument('--pgym_dir', default=str(paths.PGYM_INPUTS), help='pgym_inputs (preprocessing/pgym_preprocess.py)')
    parser.add_argument('--pgym_dms', default='all', help="comma-separated DMS ids, or 'all'")
    parser.add_argument('--max_tokens', type=int, default=32768, help='fixed mode without --batch_size: decoder rows per call = this // length')
    batching.add_batch_args(parser)
    calibration.add_calibration_args(parser)
    parser.add_argument('--seed', type=int, default=0, help='seed for the background decoding orders')
    parser.add_argument('--n_samples', type=int, default=1, help='average over this many samples, each with its own background orders')
    parser.add_argument('--coord_noise', type=float, default=0.0,
                        help='std (A) of Gaussian noise added to every backbone atom per sample (0.2 = the training noise)')
    resource_tracker.add_tracking_args(parser)
    main(parser.parse_args())
