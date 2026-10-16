# adapted from ProteinMPNN/protein_mpnn_utils.py

import os
import sys
# ProteinMPNN checkout (protein_mpnn_utils.py, vanilla_model_weights/); override with PROTEINMPNN_DIR
MPNN_DIR = os.environ.get('PROTEINMPNN_DIR', '/home/sareeves/software/ProteinMPNN')
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from esm_msr import utils

from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "analysis_notebooks"

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
                        max_tokens=32768, seed=0):
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
    noise = torch.randn((L, L), generator=gen, device=device)   # row p: background order for target position p
    enc = mpnn_encode(model, X, mask, residue_idx, chain_encoding_all)
    B = max(1, max_tokens // L)
    lp = torch.empty((len(keys), 21), dtype=torch.float32)
    for i in range(0, len(keys), B):
        chunk = keys[i:i + B]
        n = len(chunk)
        p_t = torch.tensor([k[0] for k in chunk], device=device)
        vals = noise[p_t].clone()
        vals[torch.arange(n, device=device), p_t] = 1000.0   # target forced to end
        S_chunk = wt_seq.unsqueeze(0).repeat(n, 1)
        sub = [(b, q, mq) for b, k in enumerate(chunk) for q, mq in k[1]]
        if sub:
            b_i, q_i, m_i = (torch.tensor(x, device=device) for x in zip(*sub))
            S_chunk[b_i, q_i] = m_i
        lp[i:i + n] = mpnn_decode_at(model, enc, S_chunk, vals, p_t).float().cpu()

    t = np.array(terms, dtype=np.int64).reshape(-1, 5)
    lp = lp.numpy().astype(np.float64)
    delta = lp[t[:, 2], t[:, 4]] - lp[t[:, 2], t[:, 3]]
    score = np.zeros((len(parsed), 2))
    np.add.at(score, (t[:, 0], t[:, 1]), delta)
    print(f"  {len(parsed)} variants -> {len(keys)} decoder rows (L={L}, batch {B})", flush=True)
    return score[:, 0], score[:, 1]


def make_pgym_score_fn(model, max_tokens, seed=0):
    """score_fn for pgym_runner: ProteinMPNN unmasked WT/MUT scores of one ProteinGym DMS (columns as in predict)."""
    device = next(model.parameters()).device
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    def score_fn(df):
        X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize_structure(
            df['code'].iloc[0], df['pdb_file'].iloc[0], df['chain'].iloc[0], device)
        t0 = time.time()
        tokens = max_tokens
        while True:
            try:
                score_wt, score_mut = score_unmasked_fast(model, df['mut_type_renumbered'].tolist(), X, S, mask,
                                                          residue_idx, chain_encoding_all, alphabet,
                                                          max_tokens=tokens, seed=seed)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tokens <= S.shape[1]:
                    raise
                tokens //= 2
                print(f"OOM; retrying {df['code'].iloc[0]} with max_tokens={tokens}", flush=True)
        out = pd.DataFrame({'mpnn_score_wt_seq': score_wt, 'mpnn_score_mut_seq': score_mut})
        out['mpnn_score_additive'] = score_wt
        out['mpnn_score'] = 0.5 * (score_wt + score_mut)
        out['mpnn_score_epistasis'] = out['mpnn_score'] - score_wt
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


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def main_(args):

    device = torch.device("cuda:0")
    #v_48_010=version with 48 edges 0.10A noise
    model_name = f"v_48_020"
    backbone_noise = 0.00 # Standard deviation of Gaussian noise to add to backbone atoms
    hidden_dim = 128
    num_layers = 3
    model_folder_path = os.path.join(MPNN_DIR, 'vanilla_model_weights')
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, 
        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
        augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mode_suffix = '_masked' if args.use_masks else '_unmasked'
    out_name = f"proteinmpnn_020{mode_suffix}.csv"

    if args.protein_gym:
        if args.use_masks:
            raise AssertionError("--protein_gym scores the unmasked WT/MUT mode only; drop --use_masks")
        if not args.pgym_dir:
            raise AssertionError("--pgym_dir is required with --protein_gym")
        import pgym_runner
        pgym_runner.run_protein_gym(
            args.pgym_dir, args.pgym_out or (Path(args.pgym_dir).resolve().parent / 'pgym_results' / 'proteinmpnn_020_unmasked'),
            make_pgym_score_fn(model, args.pgym_max_tokens, args.pgym_seed), pred_col='mpnn_score',
            dms=args.pgym_dms, tag=f'proteinmpnn {model_name} unmasked seed={args.pgym_seed}', extra_cols=('mpnn_score_wt_seq',))
        return

    if not args.skip_external:
        external_test_dataloaders_names = ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's8754', 's2648', 'ptmul', 'ptmuld']
        #external_test_dataloaders_names = ['ptmul']
        stats_df = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(name)

            df_true = pd.read_csv(DATA_DIR / f'{name}_mapped.csv')
            if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's8754', 's2648']:
                df_true = df_true.reset_index()
                df_true['position_pdb'] = df_true['position']
                df_true['position'] = df_true['seq_pos']
                df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true['mut_type'] = df_true['wild_type'] + df_true['seq_pos'].astype(int).astype(str) + df_true['mutation']
                df_true = df_true.set_index('id')
                if name == 's571':
                    df_true['ddG'] = df_true['dTm']
            else:
                df_true = df_true.reset_index()
                df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            res_df, time_taken = timed_call(predict, df=df_true, model=model, use_masks=args.use_masks)
            print(res_df[['mpnn_score']].head())

            pred_dir = OUT_DIR / f'predictions/{name if name!= "ptmul" else "PTMUL"}/proteinmpnn'
            os.makedirs(pred_dir, exist_ok=True)
            res_df.to_csv(pred_dir / out_name)

            stats_df.at[name, 'spearman'] = res_df[['ddG', 'mpnn_score']].corr('spearman').iloc[0,1]
            #stats_df.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG', top_n=30)
            #stats_df.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG', threshold=0)
            stats_df.at[name, 'time'] = time_taken

            if 'ptmul' not in name:
                assert len(df_true) == len(res_df)
            else:
                pass

            stats_dir = OUT_DIR / 'stats/external/proteinmpnn'
            os.makedirs(stats_dir, exist_ok=True)
            stats_df.to_csv(stats_dir / out_name)

    ############## REPEAT WITH SPECIFIC SPLITS ################

    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        from esm_msr import preprocess_megascale
        ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
            data_file='/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 
            af_model_folder='/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs')

        splits = ds.create_training_splits(str(split_file), -1)

        #if args.remove_spurs_homologs:
        #    ds.remove_homologs_from_scaffold(scaffold='train')
        #    ds.remove_homologs_from_scaffold(scaffold='val')

        for scaffold in ['validation', 'testing']:
            stats_df = pd.DataFrame()
            scaffold_ = scaffold.replace('testing', 'test')
            
            if not os.path.exists(OUT_DIR / f'predictions/{split_name}-{scaffold_}/proteinmpnn/{out_name}'):
                results_list = []

                scaffold_ = {'validation': 'val', 'testing': 'test'}[scaffold]
                data_scaffold = ds.split_dfs[scaffold_]
                
                data_scaffold = utils.parse_multimutant_column(data_scaffold, 'mut_type')
                data_scaffold['id'] = data_scaffold['code'] + '_' + data_scaffold['mut_type']
                data_scaffold = data_scaffold.sort_values('id')

                for code in tqdm(data_scaffold['code_wt'].unique()):

                    df_true = data_scaffold.loc[data_scaffold['code_wt']==code]
                    
                    df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                    df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

                    print(df_true[['dddG_ML']])
                    res_df, time_taken = timed_call(predict, df=df_true, model=model, use_masks=args.use_masks)
                    print(res_df[['mpnn_score', 'mpnn_score_epistasis']].head())

                    stats_df.at[code, 'spearman'] = res_df[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
                    try:
                        stats_df.at[code, 'spearman_epi'] = res_df[['dddG_ML', 'mpnn_score_epistasis']].dropna().corr('spearman').iloc[0,1]
                    except:
                        stats_df.at[code, 'spearman_epi'] = float('nan')
                    #stats_df.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', top_n=30)
                    #stats_df.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', threshold=0)
                    stats_df.at[code, 'time'] = time_taken

                    assert len(df_true) == len(res_df)

                    results_list.append(res_df)

                results_df = pd.concat(results_list, axis=0)

                print(stats_df.mean(axis=0))  

                pred_dir = OUT_DIR / f'predictions/{split_name}-{scaffold_}/proteinmpnn'
                stats_dir = OUT_DIR / f'stats/{split_name}-{scaffold_}/proteinmpnn'
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(stats_dir, exist_ok=True)

                stats_df.mean(axis=0).to_csv(stats_dir / out_name)
                results_df.to_csv(pred_dir / out_name)
                stats_df.to_csv(stats_dir / out_name)

                torch.cuda.empty_cache()

    ######################################

    if not args.skip_dms:

        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain'] #, 'A4_HUMAN_Seuma_2022'] # 'GB1_Wu_2016_binding_domain','A4_HUMAN_Seuma_2022', 
        stats_df = pd.DataFrame()

        results_list = []
    
        for mem_size, prot in zip([4,4,2,2,4,4,2], prots): #4,4,2,2,4,4,2

            df_true = pd.read_csv(DATA_DIR / f'{prot}.csv')
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')
            has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
            if has_doubles:
                df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']
            print(prot, has_doubles)

            prot_name = '_'.join(prot.split('_')[:2])
            if prot_name == 'GB1_Wu':
                prot_name = 'GB1'

            res_df, time_taken = timed_call(predict, df=df_true, model=model, use_masks=args.use_masks)
            print(res_df[['mpnn_score']].head())

            assert len(df_true) == len(res_df)
            
            pred_dir = OUT_DIR / f'predictions/{prot}/proteinmpnn'
            os.makedirs(pred_dir, exist_ok=True)
            res_df.to_csv(pred_dir / out_name)

            results_list.append(res_df)

            stats_df.at[prot, 'spearman'] = res_df[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
            try:
                stats_df.at[prot, 'spearman_epi'] = res_df[['dddG_ML', 'mpnn_score_epistasis']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_df.at[prot, 'spearman_epi'] = float('nan')
            #stats_df.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', top_n=30)
            #stats_df.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', threshold=0)
            stats_df.at[prot, 'time'] = time_taken

        print(stats_df)

        stats_dir = OUT_DIR / 'stats/DMS/proteinmpnn'
        os.makedirs(stats_dir, exist_ok=True)

        results_df = pd.concat(results_list, axis=0)

        stats_df.to_csv(stats_dir / out_name)

        torch.cuda.empty_cache()

        #os.makedirs(os.path.dirname(OUT_DIR / f'stats/DMS/proteinmpnn/{out_name}'), exist_ok=True)
        #prot_stats.to_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/stats/DMS/proteinmpnn/{out_name}{"_alpha"+str(args.lora_alpha)}chain_rule_avg.csv')

    #########################################

    if not args.skip_domainome:

        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/data/domainome1/domainome_mapped_2026.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset='pdb_file')
        df = df[['code', 'mut_type', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        df['chain'] = 'A'
        df = df.dropna(subset=['mut_type'])
        results_list = []
        stats_df = pd.DataFrame()

        for prot in tqdm(df['code'].unique()):

            df_true = df.loc[df['code']==prot]
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true = df_true.set_index('id')

            res_df, time_taken = timed_call(predict, df=df_true, model=model, use_masks=args.use_masks)
            print(res_df[['mpnn_score']].head())

            assert len(df_true) == len(res_df)

            stats_df.at[prot, 'spearman'] = res_df[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
            #stats_df.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', top_n=30)
            #stats_df.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_df, 'mpnn_score', 'ddG_ML', threshold=0)

            stats_df.at[prot, 'time'] = time_taken

            results_list.append(res_df)

        results_df_out = pd.concat(results_list, axis=0)

        print(stats_df.mean(axis=0))

        pred_dir = OUT_DIR / 'predictions/domainome/proteinmpnn'
        stats_dir = OUT_DIR / 'stats/domainome/proteinmpnn'
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        stats_df.mean(axis=0).to_csv(stats_dir / out_name)
        results_df_out.to_csv(pred_dir / out_name)
        stats_df.to_csv(stats_dir / out_name)

        torch.cuda.empty_cache()

    ##########################################

    if not args.skip_functional:

        test_list_DMS = ['D7PM05_CLYGR', 'GFP_AEQVI', 'HIS7_YEAST', 'Q6WV12_9MAXI', 'Q8WTC7_9CNID', 'RASK_HUMAN']

        for mem_size, prot in zip([8,8,8,8,8,8], test_list_DMS):

            strategy = 'masked' if args.use_masks else 'unmasked'

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/lora/DMS/csv_formatted/{prot}.csv')
            df_true['mut_type'] = df_true['MUTS'].apply(lambda x: x.replace(';', ':'))
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true = df_true.set_index('id')
            df_true = utils.parse_multimutant_column(df_true, mut_column='mut_type')
            has_doubles = len(df_true['mut_type'].str.contains(':')) > 0
            print(prot, has_doubles)
            print(df_true)

            pdb = df_true['pdb_file'].head(1).item()

            res_df, time_taken = timed_call(predict, df=df_true, model=model, use_masks=args.use_masks)
        
            #pred_combined = res_df #.set_index('id')

            #res = df_true.join(pred_combined)
            res = res_df
            assert len(df_true) == len(res)
            print(res[['ddG_dir', 'mpnn_score']].corr('spearman').iloc[0,1])
            print(res.head())

            pred_dir = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/predictions/{prot}/proteinmpnn'
            os.makedirs(pred_dir, exist_ok=True)
            res.to_csv(f'{pred_dir}/{out_name}')
            torch.cuda.empty_cache()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--split', type=str, default='hyperopt_splits') #'/home/sareeves/software/esm-msr/data/splits_megascale.csv'
        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--use_masks', action='store_true', default=False, help='Use masked trajectories instead of WT/MUT unmasked inference')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')

        # ProteinGym (all-DMS) benchmark mode
        parser.add_argument('--protein_gym', action='store_true', help='Score all preprocessed ProteinGym DMS benchmarks (unmasked mode) and exit')
        parser.add_argument('--pgym_dir', type=str, default=None, help='Directory with per-DMS input CSVs + manifest.csv (preprocessing/pgym_preprocess.py)')
        parser.add_argument('--pgym_out', type=str, default=None, help='Output directory for ProteinGym results (per-DMS CSVs + summary.csv)')
        parser.add_argument('--pgym_dms', type=str, default='all', help="Comma-separated DMS ids, or 'all'")
        parser.add_argument('--pgym_max_tokens', type=int, default=32768, help='Residues per forward batch (batch size = this // length; halved on OOM)')
        parser.add_argument('--pgym_seed', type=int, default=0, help='Seed for the background decoding orders')

        # Parse known args for main parser
        args, remaining_argv = parser.parse_known_args()

        # Keep track of remaining args after each parse
        current_remaining_argv = list(remaining_argv) # Make a mutable copy

        # Check if any arguments were truly unrecognized by any relevant parser
        if current_remaining_argv:
            parser.error(f"unrecognized arguments: {' '.join(current_remaining_argv)}")

        if args.skip_external:
            print('Skipping benchmark datasets!')
        if args.skip_tsuboyama:
            print('Skipping MegaScale validation and testing datasets!')
        if args.skip_dms:
            print('Skipping DMS assays!')
        if args.skip_functional:
            print('Skipping double mutant functional DMS assays!')
        if args.skip_domainome:
            print('Skipping domainome VAMP assays!')
        if not args.split:
            print('Warning! Not using any specific split file!')

        main_(args)