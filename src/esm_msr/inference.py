import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional
import os
import json
import argparse
import itertools
from tqdm import tqdm

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.constants import esm3 as C

from Bio.PDB import PDBParser


def parse_hparams_to_lora_config(hparams_path: str) -> dict:
    """
    Parses a PyTorch Lightning hparams.yaml file to extract LoRA configuration.
    Raises AssertionErrors for clear, non-silent failure conditions.
    Warns if expected hyperparameters are missing from the configuration.
    """
    if not os.path.isfile(hparams_path):
        raise AssertionError(f"hparams file not found at: {hparams_path}")
        
    try:
        import yaml
    except ImportError:
        raise AssertionError("The 'pyyaml' package is required to parse hparams.yaml. Please install it using 'pip install pyyaml'.")

    try:
        with open(hparams_path, 'r') as f:
            hparams = yaml.safe_load(f)
    except Exception as e:
        raise AssertionError(f"Failed to parse YAML file at {hparams_path}: {e}")
        
    if not isinstance(hparams, dict):
        raise AssertionError(f"Expected hparams.yaml to contain a dictionary, but got {type(hparams)}")

    default_wt = {
        'lora_rank': 2,
        'lora_alpha': 4,
        'lora_dropout': 0.1,
        'target_mode': 'expanded',
        'last_n_layers': 0,
        'use_dora': False,
        'incl_structure_encoder': False,
        'incl_sequence_head': True,
        'unfreeze_layernorms': False,
    }

    default_mt = {
        'lora_rank': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_mode': 'expanded',
        'last_n_layers': 0,
        'use_dora': False,
        'incl_structure_encoder': False,
        'incl_sequence_head': True,
        'unfreeze_layernorms': False,
    }

    # Warn for missing base keys
    for key, default_val in default_wt.items():
        if key + '_wt' not in hparams:
            logging.warning(f"Expected base parameter '{key}' missing from hparams. Defaulting to {default_val}.")

    for key, default_val in default_mt.items():
        if key + '_mt' not in hparams:
            logging.warning(f"Expected base parameter '{key}' missing from hparams. Defaulting to {default_val}.")

    if 'adapter_mode' not in hparams:
        logging.warning("Expected 'adapter_mode' missing from hparams. Defaulting to 'dual'.")
    if 'lora_mode' not in hparams:
        logging.warning("Expected 'lora_mode' missing from hparams. Defaulting to 'ensemble'.")

    wt_config = {
        'lora_rank': hparams.get('lora_rank_wt', default_wt['lora_rank']),
        'lora_alpha': hparams.get('lora_alpha_wt', default_wt['lora_alpha']),
        'lora_dropout': hparams.get('lora_dropout_wt', default_wt['lora_dropout']),
        'target_mode': hparams.get('target_mode_wt', default_wt['target_mode']),
        'last_n_layers': hparams.get('last_n_layers_wt', default_wt['last_n_layers']),
        'use_dora': hparams.get('use_dora_wt', default_wt['use_dora']),
        'incl_structure_encoder': hparams.get('incl_structure_encoder_wt', default_wt['incl_structure_encoder']),
        'incl_sequence_head': hparams.get('incl_sequence_head_wt', default_wt['incl_sequence_head']),
        'unfreeze_layernorms': hparams.get('unfreeze_layernorms_wt', default_wt['unfreeze_layernorms']),
    }

    mt_config = {
        'lora_rank': hparams.get('lora_rank_mt', default_mt['lora_rank']),
        'lora_alpha': hparams.get('lora_alpha_mt', default_mt['lora_alpha']),
        'lora_dropout': hparams.get('lora_dropout_mt', default_mt['lora_dropout']),
        'target_mode': hparams.get('target_mode_mt', default_mt['target_mode']),
        'last_n_layers': hparams.get('last_n_layers_mt', default_mt['last_n_layers']),
        'use_dora': hparams.get('use_dora_mt', default_mt['use_dora']),
        'incl_structure_encoder': hparams.get('incl_structure_encoder_mt', default_mt['incl_structure_encoder']),
        'incl_sequence_head': hparams.get('incl_sequence_head_mt', default_mt['incl_sequence_head']),
        'unfreeze_layernorms': hparams.get('unfreeze_layernorms_mt', default_mt['unfreeze_layernorms']),
    }

    return {
        'wt_config': wt_config,
        'mt_config': mt_config,
        'adapter_mode': hparams.get('adapter_mode', 'dual'),
        'lora_mode': hparams.get('lora_mode', 'ensemble')
    }


def compute_pairwise_heavy_atom_dist_matrix(coords: torch.Tensor, exclude_backbone: bool = True) -> torch.Tensor:
    """
    Computes a fully vectorized pairwise distance matrix between all residues.
    coords: [L, 37, 3] or [1, L, 37, 3] tensor
    Returns: [L, L] tensor of minimum heavy atom distances.
    
    If exclude_backbone is True, computes distance between side-chain heavy atoms 
    (CB and beyond). Intelligently falls back to CA for Glycine or residues with 
    unresolved side-chains to prevent NaN distance failures.
    """
    if coords.dim() == 4:
        if coords.shape[0] == 1:
            coords = coords.squeeze(0)
        else:
            raise AssertionError(f"Expected coords to have a batch size of 1, but got shape {coords.shape}.")
    elif coords.dim() != 3:
        raise AssertionError(f"Expected coords to be 3D [L, 37, 3] or 4D [1, L, 37, 3], but got shape {coords.shape}.")
        
    L = coords.shape[0]
    dist_matrix = torch.full((L, L), float('nan'), device=coords.device)
    
    # Base mask: only consider atoms with finite (non-NaN/Inf) coordinates
    is_finite = torch.isfinite(coords).all(dim=-1)
    
    if exclude_backbone:
        # ESM3/AF2 standard atom37 indices: 0:N, 1:CA, 2:C, 3:O, 4:CB.
        # Side-chain mask: strictly atoms index 4 and above.
        sc_mask = is_finite & (torch.arange(37, device=coords.device) > 3)
        
        # Find which residues actually have valid side-chain atoms
        has_valid_sc = sc_mask.any(dim=-1) # Shape: [L]
        
        # Fallback mask: use CA (index 1) for residues lacking a side-chain (e.g., Glycine)
        ca_mask = is_finite & (torch.arange(37, device=coords.device) == 1)
        
        # Apply side-chain mask normally, but use CA mask where side-chain is missing
        valid_mask = torch.where(has_valid_sc.unsqueeze(-1), sc_mask, ca_mask)
    else:
        # Original behavior: exclude only CA (index 1)
        valid_mask = is_finite & (torch.arange(37, device=coords.device) != 1)

    # Replace invalid atoms with highly distant proxies to avoid torch.cdist NaN explosion
    safe_coords = torch.where(
        valid_mask.unsqueeze(-1), 
        coords, 
        torch.tensor(1e9, dtype=coords.dtype, device=coords.device)
    )
    
    for i in range(L):
        c1 = safe_coords[i][valid_mask[i]] # [N_i, 3]
        if c1.shape[0] == 0:
            continue
            
        c1_batch = c1.unsqueeze(0).expand(L, -1, -1) # [L, N_i, 3]
        dists = torch.cdist(c1_batch.to(torch.float32), safe_coords.to(torch.float32)) # [L, N_i, 37]
        
        # Minimum distance from valid atoms in res i to all valid atoms in res j
        min_dists = dists.amin(dim=(1, 2)) # [L]
        
        # Re-mask distances that involved fallback 1e9 dummy coordinates
        min_dists[min_dists >= 1e8] = float('nan')
        dist_matrix[i] = min_dists
        
    return dist_matrix


def get_pdb_to_seq_mapping(pdb_file, chain_id, original_seq):
    """
    Parses the PDB using BioPython to construct a mapping between 1-based sequence 
    indices and PDB indices.
    Raises an AssertionError if the sequences do not align perfectly.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    chain = structure[0][chain_id]

    pdb_residues = []
    pdb_seq_list = []

    # Standard amino acid mapping
    RESIDUE_MAP = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    for residue in chain.get_residues():
        if residue.get_resname() in RESIDUE_MAP:
            res_id = residue.get_id()
            res_num = res_id[1]
            insertion_code = res_id[2].strip() 
            
            pdb_index = f"{res_num}{insertion_code}"
            pdb_residues.append(pdb_index)
            pdb_seq_list.append(RESIDUE_MAP[residue.get_resname()])

    pdb_sequence_from_parser = "".join(pdb_seq_list)

    if original_seq != pdb_sequence_from_parser:
        raise AssertionError(f"Sequence mismatch! ESM3 Sequence (len {len(original_seq)}): {original_seq}\nBioPython Sequence (len {len(pdb_sequence_from_parser)}): {pdb_sequence_from_parser}\nThis is a critical flaw caused by unaligned parsing rules between ESM3 and BioPython. Ensure the structure has no missing residues or unrecognized records.")

    seq_to_pdb = {i+1: pdb_residues[i] for i in range(len(original_seq))}
    pdb_to_seq = {v: k for k, v in seq_to_pdb.items()}
    
    return seq_to_pdb, pdb_to_seq


def generate_screening_df(args) -> pd.DataFrame:
    """Generates a mutation DataFrame programmatically based on screening parameters."""
    if not os.path.isfile(args.pdb_file):
        raise AssertionError(f"PDB file not found at: {args.pdb_file}")

    logging.info(f"Generating screening DataFrame for {args.pdb_file} (Chain {args.chain})...")
    
    try:
        chain_obj = ProteinChain.from_pdb(args.pdb_file, args.chain)
    except Exception as e:
        raise AssertionError(f"Failed to load chain {args.chain} from {args.pdb_file}: {e}")
        
    original_seq = chain_obj.sequence
    
    # parse the mapping here just to validate PDB indices requested in args
    seq_to_pdb, pdb_to_seq = get_pdb_to_seq_mapping(args.pdb_file, args.chain, original_seq)

    target_positions = []
    
    if args.screen_residues:
        req_res = [r.strip() for r in args.screen_residues.split(',') if r.strip()]
        for r in req_res:
            if r in pdb_to_seq:
                target_positions.append(pdb_to_seq[r])
            else:
                raise AssertionError(f"Residue '{r}' requested in --screen_residues was not found in PDB chain '{args.chain}'.")
        target_positions = list(set(target_positions))
        
    elif args.screen_residues_except:
        exc_res = [r.strip() for r in args.screen_residues_except.split(',') if r.strip()]
        exc_pos = []
        for r in exc_res:
            if r in pdb_to_seq:
                exc_pos.append(pdb_to_seq[r])
            else:
                raise AssertionError(f"Residue '{r}' requested in --screen_residues_except was not found in PDB chain '{args.chain}'.")
        target_positions = [i+1 for i in range(len(original_seq)) if (i+1) not in exc_pos]
        
    else:
        target_positions = [i+1 for i in range(len(original_seq))]

    mut_list = []
    AAs = list('ACDEFGHIKLMNPQRSTVWY')
    modes = ['singles', 'doubles'] if args.mode == 'both' else [args.mode]

    if 'singles' in modes:
        for pos in target_positions:
            wt = original_seq[pos-1]
            for mut in AAs:
                if mut != wt:
                    mut_list.append(f"{wt}{pos}{mut}") # Sequence indices used for inference

    if 'doubles' in modes:
        pairs = list(itertools.combinations(target_positions, 2))
        
        # Vectorized pair filtering
        if args.distance_threshold > 0:
            logging.info(f"Extracting coordinates to filter double mutants within {args.distance_threshold}Å...")
            coords_tensor, _, _ = chain_obj.to_structure_encoder_inputs()
            dist_matrix = compute_pairwise_heavy_atom_dist_matrix(coords_tensor)
            
            valid_pairs = []
            dropped_nan = 0
            for pos1, pos2 in pairs:
                # Sequence 1-based maps directly to matrix 0-based
                dist = dist_matrix[pos1-1, pos2-1].item()
                
                if not np.isnan(dist) and dist <= args.distance_threshold:
                    valid_pairs.append((pos1, pos2))
                elif np.isnan(dist):
                    dropped_nan += 1
                    
            if dropped_nan > 0:
                logging.warning(f"Silently dropped {dropped_nan} mutation combinations due to unresolved/missing coordinates (NaN distances). If you intended to mutate disordered regions, consider disabling distance filtering.")
                
            logging.info(f"Filtered {len(pairs)} theoretical pairs down to {len(valid_pairs)} proximal pairs.")
            pairs = valid_pairs
        
        if len(pairs) > 1000:
            logging.warning(f"This will create {len(pairs)} unique position pairs ({len(pairs) * 19 * 19} double mutations!).")
            
        for pos1, pos2 in pairs:
            wt1 = original_seq[pos1-1]
            wt2 = original_seq[pos2-1]
            for mut1 in AAs:
                if mut1 == wt1: continue
                for mut2 in AAs:
                    if mut2 == wt2: continue
                    mut_list.append(f"{wt1}{pos1}{mut1}:{wt2}{pos2}{mut2}")

    if args.mutations:
        mut_list = [m.strip() for m in args.mutations.split(',')]

    if not mut_list:
        raise AssertionError("The generated mutation list is empty. Ensure you selected valid target residues.")

    df = pd.DataFrame({'mut_type': mut_list})
    df['pdb_file'] = args.pdb_file
    df['code'] = args.code
    df['chain'] = args.chain
    
    logging.info(f"Generated {len(df)} mutation strings.")
    return df


def _handle_mutated_backbone(sequence, coords, structure_tokens, backbone_mutation=None, assert_wt=False, assert_mut=False, mask_ctx=False):
    corrected_seq = sequence
    mutated_backbone_pos: Optional[int] = None
    if backbone_mutation:
        mutated_backbone_pos = int(backbone_mutation[1:-1])
        wt, mut = backbone_mutation[0], backbone_mutation[-1]
        if assert_wt:
            if corrected_seq[mutated_backbone_pos-1] != wt:
                raise AssertionError(f"Expected {wt} at pos {mutated_backbone_pos}, found {corrected_seq[mutated_backbone_pos-1]}. Note: Chain breaks ('|') may shift sequence indices.")
            corrected_seq = list(corrected_seq)
            corrected_seq[mutated_backbone_pos-1] = mut
            corrected_seq = "".join(corrected_seq)
        elif assert_mut:
            if corrected_seq[mutated_backbone_pos-1] != mut:
                raise AssertionError(f"Expected {mut} at pos {mutated_backbone_pos}, found {corrected_seq[mutated_backbone_pos-1]}. Note: Chain breaks ('|') may shift sequence indices.")

    if mask_ctx and mutated_backbone_pos:
        structure_tokens[:, mutated_backbone_pos] = C.STRUCTURE_MASK_TOKEN
        coords[:, mutated_backbone_pos, :, :] = np.nan
        
    return corrected_seq, coords, structure_tokens


def preprocess_structure(model, pdb_path, chain, dev, backbone_mutation, assert_wt=False, assert_mut=False, mask_ctx=False):
    if isinstance(chain, str):
        chains = chain.split(',')
    elif isinstance(chain, (list, tuple)):
        chains = list(chain)
    else:
        chains = [chain]

    if len(chains) == 1:
        protein_chain = ProteinChain.from_pdb(pdb_path, chains[0], is_predicted=True)
        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        seq = protein_chain.sequence
    else:
        protein_chain = ProteinComplex.from_pdb(pdb_path).as_chain(force_conversion=True)
        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        seq = protein_chain.sequence
        
    coords_enc = coords.to(dev, dtype=torch.float32)
    _, structure_tokens = model.structure_encoder.encode(coords_enc, residue_index=residue_index.to(dev))

    coords = F.pad(coords.to(dev, dtype=model.dtype), (0, 0, 0, 0, 1, 1), value=torch.inf)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=C.STRUCTURE_PAD_TOKEN)
    if structure_tokens.shape[1] > 0:
        structure_tokens[:, 0], structure_tokens[:, -1] = C.STRUCTURE_BOS_TOKEN, C.STRUCTURE_EOS_TOKEN

    corrected_seq, masked_coords, masked_struct = _handle_mutated_backbone(
        seq, coords, structure_tokens, backbone_mutation, assert_wt, assert_mut, mask_ctx)

    seq_tokens_list = model.sequence_tokenizer.encode(corrected_seq)
    seq_tokens = torch.tensor(seq_tokens_list, dtype=torch.long).unsqueeze(0).to(dev)
        
    return seq_tokens, masked_coords.to(dev), F.pad(plddt.to(dev, dtype=model.dtype), (1, 1), value=0.0).to(dev), masked_struct.to(dev), corrected_seq


def _prepare_sparse_batch(model, muts_list, dev):
    """
    Constructs a lightweight, memory-efficient sparse batch tensor representation 
    for mutational indexes, mapping AAs directly to model vocab IDs.
    """
    if not muts_list: raise AssertionError("muts_list is empty.")
    B, max_muts = len(muts_list), max(1, max(len(m) for m in muts_list))

    mut_pos_list, wt_id_list, mt_id_list = [], [], []

    for muts in muts_list:
        m_pos, w_id, m_id = [], [], []
        for (w, p, m) in muts:
            m_tid, w_tid = model.vocab.get(m), model.vocab.get(w)
            if m_tid is None or w_tid is None:
                raise AssertionError(f"Failed to map amino acids '{w}' or '{m}' to the tokenizer vocabulary.")
            m_pos.append(p); w_id.append(w_tid); m_id.append(m_tid)
            
        mut_pos_list.append(torch.tensor(m_pos, dtype=torch.long))
        wt_id_list.append(torch.tensor(w_id, dtype=torch.long))
        mt_id_list.append(torch.tensor(m_id, dtype=torch.long))

    mut_pos_stack = torch.nn.utils.rnn.pad_sequence(mut_pos_list, batch_first=True, padding_value=0).to(dev)
    if mut_pos_stack.size(1) < max_muts and mut_pos_stack.dim() > 1:
        pad_size = max_muts - mut_pos_stack.size(1)
        mut_pos_stack = F.pad(mut_pos_stack, (0, pad_size), value=0)
        wt_id_stack = F.pad(torch.nn.utils.rnn.pad_sequence(wt_id_list, batch_first=True, padding_value=C.SEQUENCE_PAD_TOKEN).to(dev), (0, pad_size), value=C.SEQUENCE_PAD_TOKEN)
        mt_id_stack = F.pad(torch.nn.utils.rnn.pad_sequence(mt_id_list, batch_first=True, padding_value=C.SEQUENCE_PAD_TOKEN).to(dev), (0, pad_size), value=C.SEQUENCE_PAD_TOKEN)
    else:
        wt_id_stack = torch.nn.utils.rnn.pad_sequence(wt_id_list, batch_first=True, padding_value=C.SEQUENCE_PAD_TOKEN).to(dev)
        mt_id_stack = torch.nn.utils.rnn.pad_sequence(mt_id_list, batch_first=True, padding_value=C.SEQUENCE_PAD_TOKEN).to(dev)

    mut_mask = torch.zeros(B, max_muts, dtype=torch.bool, device=dev)
    for i, muts in enumerate(muts_list): mut_mask[i, :len(muts)] = True

    return {'mut_pos': mut_pos_stack, 'wt_id': wt_id_stack, 'mt_id': mt_id_stack, 'mut_mask': mut_mask}


@torch.no_grad()
def infer_mutants(model, df: pd.DataFrame, batch_size: int = 16, device=None, backbone_mutation=None, optimize_wt_pass=True, quiet=False, skip_additive=True, skip_reverse=False, mask_strategy=None, calculate_distances=False, ignore_mismatch=True) -> pd.DataFrame:
    """
    A unified, highly interpretable pipeline to evaluate mutational stability.
    Abstracts dense tensor creation, deduplication, and math away from the user.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    all_results = []
    
    for (pdb, code, chain), group_df in df.groupby(['pdb_file', 'code', 'chain'], dropna=False):
        seq_toks, coords, plddt, struct_toks, wt_seq_str = preprocess_structure(model, pdb, chain, dev, backbone_mutation, assert_wt=True)
        
        dist_matrix = None
        if calculate_distances:
            dist_matrix = compute_pairwise_heavy_atom_dist_matrix(coords)
            
        cached_wt_esm3 = None
        if optimize_wt_pass and mask_strategy is None:
            if dev.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=model.dtype):
                    out = model._get_esm3_outputs(seq_toks, coords, struct_toks, plddt)
            else:
                out = model._get_esm3_outputs(seq_toks, coords, struct_toks, plddt)
                
            cached_wt_esm3 = {
                'seq': seq_toks, 
                'logits': model._process_logits(out.sequence_logits.float()), 
                'embeddings': getattr(out, 'embeddings', None)
            }

        valid_rows = []
        for _, row in group_df.iterrows():
            muts, is_valid = [], True
            for m_str in str(row['mut_type']).split(':'):
                if len(m_str) < 3: is_valid = False; break
                wt, mt, pos = m_str[0], m_str[-1], int(m_str[1:-1])
                if pos < 1 or pos > len(wt_seq_str) or mt not in model.vocab:
                    raise IndexError(f"Invalid mutation {m_str} mapping against sequence len {len(wt_seq_str)}")
                elif wt_seq_str[pos-1] != wt:
                    logging.warning(f'Mismatch in inference.infer_mutants: at position {pos}, expected {wt}, got {wt_seq_str[pos-1]}')
                    if not ignore_mismatch:
                        is_valid = False; break
                muts.append((wt, pos, mt))
            if is_valid and muts: valid_rows.append({'mut_type': str(row['mut_type']), 'muts': muts, 'pdb_file': pdb, 'code': code, 'chain': chain})
        
        if not valid_rows: 
            raise AssertionError("No valid rows were produced after checking mutations against the structure.")

        # 1. Compile Unified Target List (Including singles for additive math if requested)
        muts_to_score = set()
        for r in valid_rows:
            muts_tup = tuple(r['muts'])
            muts_to_score.add(muts_tup)
            if not skip_additive and len(muts_tup) > 1:
                for single_m in muts_tup:
                    muts_to_score.add((single_m,))
        
        muts_to_score_list = list(muts_to_score)
        
        # 2. Extract Memory-Efficient Sparse Indices
        sparse_batch = _prepare_sparse_batch(model, muts_to_score_list, dev)
        
        # 3. Dynamic Execution via Unified API (Takes exactly 1 WT sequence reference)
        if not quiet:
            logging.info(f"Scoring {len(muts_to_score_list)} unique mutations via {'DEDUPLICATION' if mask_strategy else 'DENSE'} strategy...")
            
        out = model.score_screening_batch(
            wt_sequence_tokens=seq_toks,
            mut_pos=sparse_batch['mut_pos'],
            wt_id=sparse_batch['wt_id'],
            mt_id=sparse_batch['mt_id'],
            mut_mask=sparse_batch['mut_mask'],
            coords=coords,
            structure_tokens=struct_toks,
            plddt=plddt,
            mask_strategy=mask_strategy,
            batch_size=batch_size,
            skip_reverse=skip_reverse,
            cached_wt_esm3=cached_wt_esm3,
            quiet=quiet
        )
        
        # 4. Store Outputs
        preds = {}
        for i, mut_tup in enumerate(muts_to_score_list):
            preds[mut_tup] = {
                'wt_lora': out['wt_lora_pred'][i].item(),
                'mt_lora': out['mt_lora_pred'][i].item(),
                'combined': out['combined_pred'][i].item()
            }
            
        # 5. Format the Final DataFrame
        for r in tqdm(valid_rows, desc='Constructing output dataframe', disable=quiet):
            muts = r['muts']
            muts_tup = tuple(muts)
            
            res_dict = {
                'pdb_file': r['pdb_file'], 'code': r['code'], 'chain': r['chain'], 'mut_type': r['mut_type']
            }
            
            if skip_additive or len(muts) == 1:
                wt_tot = preds[muts_tup]['wt_lora']
                mt_tot = preds[muts_tup]['mt_lora']
                comb_tot = preds[muts_tup]['combined']
                res_dict.update({
                    'wt_lora_pred': wt_tot, 'mt_lora_pred': mt_tot, 
                    'combined_pred': comb_tot, 'combined_dddg_pred': 0.5 * mt_tot - 0.5 * wt_tot
                })
            else:
                wt_add = sum(preds[(m,)]['wt_lora'] for m in muts)
                mt_add = sum(preds[(m,)]['mt_lora'] for m in muts)
                comb_add = sum(preds[(m,)]['combined'] for m in muts)
                
                wt_tot = preds[muts_tup]['wt_lora']
                mt_tot = preds[muts_tup]['mt_lora']
                comb_tot = preds[muts_tup]['combined']
                
                res_dict.update({
                    'wt_lora_pred_additive': wt_add, 'wt_lora_dddg_pred': wt_tot - wt_add, 'wt_lora_pred': wt_tot, 
                    'mt_lora_pred_additive': mt_add, 'mt_lora_dddg_pred': mt_tot - mt_add, 'mt_lora_pred': mt_tot, 
                    'combined_pred_additive': comb_add, 'combined_dddg_pred': comb_tot - comb_add, 'combined_pred': comb_tot
                })

            if calculate_distances and len(muts) == 2:
                res_dict['dist'] = dist_matrix[muts[0][1], muts[1][1]].item()
                
            all_results.append(res_dict)

    out = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    out = out.dropna(how='all', axis=1)

    return out

if __name__ == "__main__":
    from models import MSRModel
    from huggingface_hub import login, get_token

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run ESM3 mutant stability inference with MSRModel.")

    # Data IO and Execution Args
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV containing: pdb_file, code, chain, mut_type")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV")
    
    # Structure arguments (Used if input_csv is not provided)
    parser.add_argument("--pdb_file", type=str, default=None, help="Path to PDB file (used if --input_csv is not provided)")
    parser.add_argument("--code", type=str, default="protein", help="Protein code")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID")

    # Screening Arguments
    parser.add_argument("--mode", type=str, choices=['singles', 'doubles', 'both'], default='singles', help="Mutation screening mode")
    parser.add_argument("--screen_residues", type=str, default=None, help="Comma-separated list of PDB indices to screen (mutually exclusive with screen_residues_except)")
    parser.add_argument("--screen_residues_except", type=str, default=None, help="Comma-separated list of PDB indices to exclude from screen (mutually exclusive with screen_residues)")
    parser.add_argument("--mutations", type=str, default=None, help="Comma-separated list of sequence-index mutations (e.g., A12C,A12C:D15E) to score directly.")
    parser.add_argument("--distance_threshold", type=float, default=6.0, help="Maximum heavy atom distance (Angstroms) for pairing double mutants. <= 0 means all pairs.")
    parser.add_argument("--calculate_distances", action="store_true", help="Calculate pairwise heavy atom distance for doubles")

    # Runtime configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--backbone_mutation", type=str, default=None, help="Backbone mutation applied across structures")
    parser.add_argument("--no_optimize_wt_pass", action="store_true", help="Disable the caching of the Wild Type pass")
    parser.add_argument('--mask_strategy', type=str, choices=['marginal', 'chain'], default=None)
    parser.add_argument("--skip_additive", action="store_true", help="Skip single mutation sub-calculations for multis")
    parser.add_argument("--skip_reverse", action="store_true", help="Skip the MT pass if predicting purely from WT adapters")

    # Checkpoint configuration
    parser.add_argument("--checkpoint_path", type=str, help="Path to .pt/.ckpt file or a JSON string")
    parser.add_argument("--load_wt_lora_only", action="store_true", help="Discard mutant adapter weights when loading checkpoints")
    parser.add_argument("--lora_config", type=str, default=None, help="Path to JSON file containing LoRA config")
    parser.add_argument("--hparams_path", type=str, default=None, help="Path to lightning hparams.yaml file")
    
    # Model configuration
    parser.add_argument("--log_likelihood", action="store_true", help="Process raw logits into log likelihoods")
    parser.add_argument("--use_plddt", action="store_true", help="Whether to pass pLDDT values to the model")
    parser.add_argument("--quaternary_mode", type=str, default="single_chain", help="How to handle quaternary structure")
    parser.add_argument("--model_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

    # Model source
    parser.add_argument('--base_model_loc', type=str, default=None)
    parser.add_argument('--hf_token', type=str, default=None)

    args = parser.parse_args()

    # 1. See exactly what HuggingFace detects
    print(f"DEBUG: args.hf_token type: {type(args.hf_token)}, value: {args.hf_token}")
    print(f"DEBUG: get_token() cache: {'FOUND' if get_token() else 'NOT FOUND'}")

    token = args.hf_token or get_token()

    if token:
        # 2. Catch the boolean trap before it triggers the prompt
        if not isinstance(token, str):
            raise TypeError(f"Expected token to be a string, but got {type(token)}. Check your argparse definition.")
        
        print("DEBUG: Logging in with detected string token.")
        login(token)
    elif args.base_model_loc:
        os.environ['INFRA_PROVIDER'] = "1"
        os.chdir(args.base_model_loc)
        print(os.getcwd())
        assert os.path.exists(os.path.join(os.getcwd(), 'data/weights/esm3_sm_open_v1.pth'))
    else:
        raise AssertionError('Must provide either a HuggingFace token or have the model installed locally!')

    # Pre-execution validation
    if args.screen_residues and args.screen_residues_except:
        raise AssertionError("Error: --screen_residues and --screen_residues_except cannot be used at the same time.")
        
    has_screen_args = args.screen_residues is not None or args.screen_residues_except is not None

    if args.input_csv and args.mutations:
        raise AssertionError("Error: --input_csv and --mutations cannot be used at the same time.")
        
    if has_screen_args and args.mutations:
        raise AssertionError("Error: --mutations cannot be used with --screen_residues or --screen_residues_except.")
        
    if has_screen_args and args.input_csv:
        raise AssertionError("Error: --input_csv cannot be used with --screen_residues or --screen_residues_except.")

    # Determine Data Source
    if args.input_csv:
        if not os.path.isfile(args.input_csv):
            raise AssertionError(f"Input CSV file does not exist: {args.input_csv}")
        try:
            input_df = pd.read_csv(args.input_csv)
        except Exception as e:
            raise AssertionError(f"Failed to read input CSV: {e}")
            
        required_cols = {'pdb_file', 'code', 'chain', 'mut_type'}
        if not required_cols.issubset(input_df.columns):
            raise AssertionError(f"Input CSV missing required columns. Expected: {required_cols}. Found: {set(input_df.columns)}")
    else:
        if not args.pdb_file:
            raise AssertionError("Either --input_csv or --pdb_file must be provided.")
        input_df = generate_screening_df(args)
    
    # Config loading
    lora_config = None
    adapter_mode = "dual"
    lora_mode = "ensemble"

    if args.lora_config:
        try:
            if os.path.isfile(args.lora_config):
                with open(args.lora_config, 'r') as f:
                    lora_config = json.load(f)
            else:
                lora_config = json.loads(args.lora_config)
                
            adapter_mode = lora_config.get('adapter_mode', 'dual')
            lora_mode = lora_config.get('lora_mode', 'ensemble')
            if 'adapter_mode' not in lora_config:
                logging.warning("JSON config missing 'adapter_mode'. Defaulting to 'dual'.")
            if 'lora_mode' not in lora_config:
                logging.warning("JSON config missing 'lora_mode'. Defaulting to 'ensemble'.")
                
        except Exception as e:
            raise AssertionError(f"Failed to parse explicitly provided --lora_config: {e}")
    else:
        if not args.hparams_path:
             raise AssertionError("Either --lora_config or --hparams_path must be provided.")
        logging.info(f"Extracting LoRA config from hparams file: {args.hparams_path}")
        parsed_config = parse_hparams_to_lora_config(args.hparams_path)
        adapter_mode = parsed_config.get('adapter_mode', 'dual')
        lora_mode = parsed_config.get('lora_mode', 'ensemble')
        lora_config = {
            'wt_config': parsed_config['wt_config'],
            'mt_config': parsed_config['mt_config']
        }

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model_dtype = dtype_map[args.model_dtype]
    device = torch.device(args.device)

    logging.info(f"Initializing MSRModel on {device} (dtype: {model_dtype}, adapter: {adapter_mode}, lora: {lora_mode})...")

    model = MSRModel(
        lora_config=lora_config,
        shared_scale_init=1,
        shared_bias_init=0,
        inference_mode=True,
        log_likelihood=args.log_likelihood,
        use_plddt=args.use_plddt,
        quaternary_mode=args.quaternary_mode,
        model_dtype=model_dtype,
        adapter_mode=adapter_mode,
        lora_mode=lora_mode
    )

    if args.checkpoint_path is not None:
        model.load_lora_weights(checkpoint_path=args.checkpoint_path, load_wt_only=args.load_wt_lora_only)

    model.to(device)
    model.eval()

    logging.info(f"Starting inference for {len(input_df)} mutations...")
    
    results_df = infer_mutants(
        model=model,
        df=input_df,
        batch_size=args.batch_size,
        device=device,
        backbone_mutation=args.backbone_mutation,
        optimize_wt_pass=not args.no_optimize_wt_pass,
        quiet=False,
        skip_additive=args.skip_additive,
        skip_reverse=args.skip_reverse,
        mask_strategy=args.mask_strategy,
        calculate_distances=args.calculate_distances
    )

    if results_df.empty:
        logging.warning("No valid results were produced. Check your inputs or mut_type formatting.")
    else:
        results_df.to_csv(args.output_csv, index=False)
        logging.info(f"Successfully saved {len(results_df)} inference predictions to {args.output_csv}.")