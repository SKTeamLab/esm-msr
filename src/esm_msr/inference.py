import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import argparse
import shutil
from tqdm import tqdm
from pathlib import Path

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.constants import esm3 as C

# Import structural data handling and prep logic from preprocess
from esm_msr.preprocess import (
    download_pdb,
    fix_noncanonical_residues,
    repair_pdb,
    renumber_pdb,
    generate_screening_df,
    standardize_input_df
)


def parse_hparams_to_lora_config(hparams_path: str, sigma: float = 1.0) -> dict:
    """
    Parses a PyTorch Lightning hparams.yaml file to extract LoRA configuration.
    Raises AssertionErrors for clear, non-silent failure conditions.
    Warns if expected hyperparameters are missing from the configuration.
    Multiplies lora_alpha by sigma.
    """
    if sigma <= 0:
        raise AssertionError(f"sigma must be strictly greater than 0. Got: {sigma}")

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
        'lora_alpha': round(float(hparams.get('lora_alpha_wt', default_wt['lora_alpha'])) * sigma, 3),
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
        'lora_alpha': round(float(hparams.get('lora_alpha_mt', default_mt['lora_alpha'])) * sigma, 3),
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


def _handle_mutated_backbone(sequence, coords, structure_tokens, backbone_mutation=None, assert_wt=False, assert_mut=False, mask_ctx=False):
    corrected_seq = sequence
    if backbone_mutation:
        for single_m in backbone_mutation.split(':'):
            mutated_backbone_pos = int(single_m[1:-1])
            wt, mut = single_m[0], single_m[-1]
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
    """
    if df.empty:
        logging.warning("infer_mutants received an empty DataFrame. Returning an empty result.")
        return pd.DataFrame()

    unique_structs = df[['pdb_file', 'code', 'chain']].drop_duplicates()
    if len(unique_structs) > 1:
        raise AssertionError(f"infer_mutants received multiple structures ({len(unique_structs)}). Execution is restricted to a single structure per run.")

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    all_results = []
    
    pdb = unique_structs['pdb_file'].iloc[0]
    code = unique_structs['code'].iloc[0]
    chain = unique_structs['chain'].iloc[0]
    
    seq_toks, coords, plddt, struct_toks, wt_seq_str = preprocess_structure(model, pdb, chain, dev, backbone_mutation, assert_wt=True)
    
    dist_matrix = None
    if calculate_distances:
        from preprocess import compute_pairwise_heavy_atom_dist_matrix
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
    for _, row in df.iterrows():
        muts, is_valid = [], True
        for m_str in str(row['mut_type_renumbered']).split(':'):
            if len(m_str) < 3: 
                is_valid = False
                break
            
            wt, mt = m_str[0], m_str[-1]
            try:
                pos = int(m_str[1:-1])
            except ValueError:
                raise AssertionError(f"Could not parse integer position from mutation string: {m_str}. Are you feeding PDB indices into mut_type_renumbered?")
            
            if pos < 1 or pos > len(wt_seq_str) or mt not in model.vocab:
                raise IndexError(f"Invalid mutation {m_str} mapping against sequence len {len(wt_seq_str)}")
            elif wt_seq_str[pos-1] != wt:
                logging.warning(f'Mismatch in inference.infer_mutants: at position {pos}, expected {wt}, got {wt_seq_str[pos-1]}')
                if not ignore_mismatch:
                    is_valid = False; break
            muts.append((wt, pos, mt))
        
        if is_valid and muts: 
            valid_rows.append({
                'mut_type_renumbered': str(row['mut_type_renumbered']), 
                'mut_type_pdb': str(row['mut_type_pdb']),
                'muts': muts, 
                'pdb_file': pdb, 
                'code': code, 
                'chain': chain
            })
    
    if not valid_rows: 
        raise AssertionError("No valid rows were produced after checking mutations against the structure.")

    muts_to_score = set()
    for r in valid_rows:
        muts_tup = tuple(r['muts'])
        muts_to_score.add(muts_tup)
        if not skip_additive and len(muts_tup) > 1:
            for single_m in muts_tup:
                muts_to_score.add((single_m,))
    
    muts_to_score_list = list(muts_to_score)
    sparse_batch = _prepare_sparse_batch(model, muts_to_score_list, dev)
    
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
    
    preds = {}
    for i, mut_tup in enumerate(muts_to_score_list):
        preds[mut_tup] = {
            'wt_lora': out['wt_lora_pred'][i].item(),
            'mt_lora': out['mt_lora_pred'][i].item(),
            'combined': out['combined_pred'][i].item()
        }
        
    for r in tqdm(valid_rows, desc='Constructing output dataframe', disable=quiet):
        muts = r['muts']
        muts_tup = tuple(muts)
        
        res_dict = {
            'pdb_file': r['pdb_file'], 'code': r['code'], 'chain': r['chain'], 
            'mut_type_renumbered': r['mut_type_renumbered'],
            'mut_type_pdb': r['mut_type_pdb']
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

    parser = argparse.ArgumentParser(description="Run ESM3 mutant stability inference with MSRModel. Enforces single-structure processing.")

    # Data IO and Execution Args
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV. MUST contain only ONE unique pdb_file and chain combination. Must include a mutation column (mut_type, mut_type_renumbered, or mut_type_pdb)")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV")
    
    # Structure arguments (Used if input_csv is not provided)
    parser.add_argument("--pdb_file", type=str, default=None, help="Path to PDB file (used if --input_csv is not provided)")
    parser.add_argument("--code", type=str, default="protein", help="Protein code")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID")

    # Structural Preprocessing Arguments
    parser.add_argument("--model_missing_regions", action="store_true", help="Use Modeller to repair missing loops/atoms in the PDB.")
    parser.add_argument("--skip_fix_noncanonical", action="store_true", help="Skip replacing non-canonical residues with canonical equivalents.")
    parser.add_argument("--renumber_pdb", action="store_true", help="Sequentially renumber the PDB from 1. WARNING: Will break CSV mappings reliant on prior PDB index.")

    # Screening Arguments
    parser.add_argument("--mode", type=str, choices=['singles', 'doubles', 'singles+doubles'], default='singles', help="Mutation screening mode")
    parser.add_argument("--screen_residues", type=str, default=None, help="Comma-separated list of PDB indices to screen (mutually exclusive with screen_residues_except)")
    parser.add_argument("--screen_residues_except", type=str, default=None, help="Comma-separated list of PDB indices to exclude from screen (mutually exclusive with screen_residues)")
    parser.add_argument("--mutations", type=str, default=None, help="Comma-separated list of sequence-index mutations (e.g., A12C,A12C:D15E) to score directly.")
    parser.add_argument("--distance_threshold", type=float, default=-1.0, help="Maximum heavy atom distance (Angstroms) for pairing double mutants. <= 0 means all pairs.")
    parser.add_argument("--calculate_distances", action="store_true", help="Calculate pairwise heavy atom distance for doubles")

    # Runtime configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--backbone_mutation", type=str, default=None, help="Backbone mutation applied across structures")
    parser.add_argument("--no_optimize_wt_pass", action="store_true", help="Disable the caching of the Wild Type pass")
    parser.add_argument('--mask_strategy', type=str, choices=['marginal', 'independent'], default=None)
    parser.add_argument("--skip_additive", action="store_true", help="Skip single mutation sub-calculations for multis")
    parser.add_argument("--skip_reverse", action="store_true", help="Skip the MT pass if predicting purely from WT adapters")

    # Checkpoint configuration
    parser.add_argument("--checkpoint_path", type=str, help="Path to .pt/.ckpt file or a JSON string")
    parser.add_argument("--load_wt_lora_only", action="store_true", help="Discard mutant adapter weights when loading checkpoints")
    parser.add_argument("--lora_config", type=str, default=None, help="Path to JSON file containing LoRA config")
    parser.add_argument("--hparams_path", type=str, default=None, help="Path to lightning hparams.yaml file")
    parser.add_argument("--sigma", type=float, default=1.0, help="Multiplier for the LoRA alpha parameters.")
    
    # Model configuration
    parser.add_argument("--log_likelihood", action="store_true", help="Process raw logits into log likelihoods")
    parser.add_argument("--use_plddt", action="store_true", help="Whether to pass pLDDT values to the model")
    parser.add_argument("--quaternary_mode", type=str, default="single_chain", help="How to handle quaternary structure")
    parser.add_argument("--model_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

    # Model source
    parser.add_argument('--base_model_loc', type=str, default=None)
    #parser.add_argument('--hf_token', type=str, default=None)

    args = parser.parse_args()

    #token = args.hf_token or get_token()

    #if token:
    #    if not isinstance(token, str):
    #        raise TypeError(f"Expected token to be a string, but got {type(token)}. Check your argparse definition.")
    #    login(token)
    if args.base_model_loc:
        os.environ['INFRA_PROVIDER'] = "1"
        base_loc = Path(args.base_model_loc).resolve()
        
        target_rel = Path('data/weights/esm3_sm_open_v1.pth')
        valid_root = None

        if base_loc.is_file() and base_loc.name == 'esm3_sm_open_v1.pth':
            potential_root = base_loc.parents[2]
            if (potential_root / target_rel).is_file():
                valid_root = potential_root
        elif base_loc.is_dir():
            for level in range(3):
                potential_root = base_loc.parents[level-1] if level > 0 else base_loc
                if (potential_root / target_rel).is_file():
                    valid_root = potential_root
                    break

        if not valid_root:
            raise AssertionError(
                f"Could not locate ESM3 weights using base_model_loc: '{args.base_model_loc}'. "
                f"Ensure the path points to the weights file itself, the 'weights' folder, the 'data' folder, or their parent directory."
            )

        saved_path = os.getcwd()
        os.chdir(valid_root)
        assert os.path.exists(os.path.join(os.getcwd(), 'data', 'weights', 'esm3_sm_open_v1.pth')), "Model file verification failed after changing directory."

    if args.screen_residues and args.screen_residues_except:
        raise AssertionError("Error: --screen_residues and --screen_residues_except cannot be used at the same time.")
        
    has_screen_args = args.screen_residues is not None or args.screen_residues_except is not None

    if args.input_csv and args.mutations:
        raise AssertionError("Error: --input_csv and --mutations cannot be used at the same time.")
        
    if has_screen_args and args.mutations:
        raise AssertionError("Error: --mutations cannot be used with --screen_residues or --screen_residues_except.")
        
    if has_screen_args and args.input_csv:
        raise AssertionError("Error: --input_csv cannot be used with --screen_residues or --screen_residues_except.")


    # Extract target structure details
    target_pdb = args.pdb_file
    target_chain = args.chain

    if args.input_csv:
        if not os.path.isfile(args.input_csv):
            raise AssertionError(f"Input CSV file does not exist: {args.input_csv}")
        try:
            input_df = pd.read_csv(args.input_csv)
        except Exception as e:
            raise AssertionError(f"Failed to read input CSV: {e}")
            
        required_cols = {'pdb_file', 'code', 'chain'}
        if not required_cols.issubset(input_df.columns):
            raise AssertionError(f"Input CSV missing required base columns. Expected: {required_cols}. Found: {set(input_df.columns)}")
            
        unique_structs = input_df[['pdb_file', 'chain']].drop_duplicates()
        if len(unique_structs) > 1:
            raise AssertionError("This script processes one structure per execution. Found multiple in CSV.")
            
        target_pdb = unique_structs['pdb_file'].iloc[0]
        target_chain = unique_structs['chain'].iloc[0]

    # Handle Downloading if PDB missing
    if not target_pdb:
        if args.code and args.chain:
            logging.info(f"PDB file not specified. Attempting to download {args.code}...")
            dl_res = download_pdb(args.code, output_dir='./data/structures', file_format='pdb', get_fasta=True)
            if not dl_res.get('pdb'):
                raise AssertionError(f"Failed to download PDB {args.code}. Provide a valid --pdb_file.")
            target_pdb = dl_res['pdb']
        else:
            raise AssertionError("Either --input_csv, or --pdb_file, or both --code and --chain must be provided.")

    # Apply Preprocessing Interventions
    do_prep = (not args.skip_fix_noncanonical) or args.model_missing_regions or args.renumber_pdb
    if do_prep:
        prep_pdb = target_pdb.replace('.pdb', '_inference_prep.pdb')
        shutil.copy(target_pdb, prep_pdb)
        
        if not args.skip_fix_noncanonical:
            logging.info("Replacing non-canonical residues...")
            fix_noncanonical_residues(prep_pdb, prep_pdb, verbose=False)
            
        if args.model_missing_regions:
            logging.info("Repairing missing regions with Modeller...")
            success = repair_pdb(prep_pdb, prep_pdb, chain_id=target_chain)
            if not success:
                raise AssertionError(f"repair_pdb failed for {prep_pdb} chain {target_chain}.")
                
        if args.renumber_pdb:
            logging.info("Renumbering PDB residues sequentially...")
            renumber_pdb(prep_pdb, prep_pdb)
            
        target_pdb = prep_pdb
        args.pdb_file = target_pdb

        if args.input_csv:
            input_df['pdb_file'] = target_pdb

    # Finalize DataFrame
    if args.input_csv:
        input_df = standardize_input_df(input_df, args.backbone_mutation)
    else:
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
            
            # Apply sigma
            if args.sigma <= 0:
                raise AssertionError("sigma must be strictly positive.")
            lora_config['wt_config']['lora_alpha'] = round(float(lora_config['wt_config'].get('lora_alpha', 4)) * args.sigma, 3)
            lora_config['mt_config']['lora_alpha'] = round(float(lora_config['mt_config'].get('lora_alpha', 16)) * args.sigma, 3)

        except Exception as e:
            raise AssertionError(f"Failed to parse explicitly provided --lora_config: {e}")
    else:
        if not args.hparams_path:
            if os.path.exists(os.path.join(os.path.dirname(args.checkpoint_path), 'hparams.yaml')):
                args.hparams_path = os.path.join(os.path.dirname(args.checkpoint_path), 'hparams.yaml')
            else:
                raise AssertionError("Either --lora_config or --hparams_path must be provided.")
        logging.info(f"Extracting LoRA config from hparams file: {args.hparams_path}")
        parsed_config = parse_hparams_to_lora_config(args.hparams_path, sigma=args.sigma)
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

    if args.base_model_loc:
        os.chdir(saved_path)

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