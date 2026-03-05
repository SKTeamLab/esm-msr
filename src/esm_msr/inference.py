import numpy as np
import pandas as pd
import os
import argparse
import tempfile
import sys
import itertools
import atexit
from pathlib import Path
from huggingface_hub import login

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.structure.protein_chain import ProteinChain

from esm_msr import utils, models

from Bio.PDB import PDBParser, PDBList

import warnings
warnings.filterwarnings('ignore')

# --- FIX 5: Guarantee Temp File Cleanup on Exit ---
atexit.register(utils._cleanup_temp_files)

# --- FIX 1: Optimized Heavy Atom Distance Function ---
def get_closest_heavy_atom_distance(pdb_pos1, pdb_pos2, residue_dict):
    res1 = residue_dict.get(str(pdb_pos1))
    res2 = residue_dict.get(str(pdb_pos2))

    if not res1 or not res2:
        return np.nan

    min_dist = float('inf')
    
    for a1 in res1.get_atoms():
        if a1.element == 'H' or a1.name == 'CA': continue
        for a2 in res2.get_atoms():
            if a2.element == 'H' or a2.name == 'CA': continue
            dist = np.linalg.norm(a1.coord - a2.coord)
            if dist < min_dist:
                min_dist = dist

    return min_dist if min_dist != float('inf') else np.nan

def main(args):

    print(args.checkpoint)

    # --- Feature Check: Mutual Exclusivity ---
    if args.screen_residues and args.screen_residues_except:
        print("Error: --screen_residues and --screen_residues_except cannot be used at the same time.")
        sys.exit(1)
        
    has_screen_args = args.screen_residues is not None or args.screen_residues_except is not None

    if args.subset_df and args.mutations:
        print("Error: --subset_df and --mutations cannot be used at the same time.")
        sys.exit(1)
        
    if has_screen_args and args.mutations:
        print("Error: --mutations cannot be used with --screen_residues or --screen_residues_except.")
        sys.exit(1)
        
    if has_screen_args and args.subset_df:
        print("Error: --subset_df cannot be used with --screen_residues or --screen_residues_except.")
        sys.exit(1)

    # --- PDB Download ---
    if len(args.input_structure) == 4 and not os.path.exists(args.input_structure):
        print(f"Detected 4-character code. Attempting to download PDB: {args.input_structure.upper()}")
        pdbl = PDBList(verbose=False)
        downloaded_file = pdbl.retrieve_pdb_file(args.input_structure.upper(), pdir=tempfile.gettempdir(), file_format='mmCif')
        
        if os.path.exists(downloaded_file):
            args.input_structure = downloaded_file
            utils.register_temp_file(downloaded_file)
        else:
            print(f"Error: Could not download PDB structure for code {args.input_structure}.")
            sys.exit(1)

    # --- Determine File Type and Handle CIF Conversion ---
    pdb_path_to_use = args.input_structure

    if not os.path.exists(args.input_structure):
        print(f"Input structure does not exist: {args.input_structure}")
        sys.exit(1)

    if args.input_structure.lower().endswith(('.cif', '.mmcif', '.ent')):
        file_type = 'cif'
    elif args.input_structure.lower().endswith('.pdb'):
        file_type = 'pdb'
    else:
        raise AssertionError('Filetype must be in cif, mmcif, pdb')

    if file_type == 'cif':
        try:
            fd, temp_pdb_path = tempfile.mkstemp(suffix=".pdb", text=True)
            os.close(fd)
            pdb_path_to_use = temp_pdb_path
            utils.register_temp_file(pdb_path_to_use)

            if not utils.convert_cif_to_pdb(args.input_structure, pdb_path_to_use):
                 sys.exit(1) # Cleanup is now handled by atexit
            print(f"Using temporary PDB file: {pdb_path_to_use}")
        except Exception as e:
            print(f"Error creating temporary file for PDB conversion: {e}")
            sys.exit(1)
    else:
        pdb_path_to_use = args.input_structure

    ######################################

    original_seq = ProteinChain.from_pdb(pdb_path_to_use, args.chain_id).sequence

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein_structure", pdb_path_to_use)
    model_ = structure[0]
    chain = model_[args.chain_id]

    pdb_residues = []
    pdb_seq_list = []
    
    # --- FIX 1: Cache BioPython residues for fast distance lookups later ---
    bio_residue_dict = {}

    for residue in chain.get_residues():
        if residue.get_resname() in utils.RESIDUE_MAP:
            res_id = residue.get_id()
            res_num = res_id[1]
            insertion_code = res_id[2].strip() 
            
            pdb_index = f"{res_num}{insertion_code}"
            pdb_residues.append(pdb_index)
            pdb_seq_list.append(utils.RESIDUE_MAP[residue.get_resname()])
            
            bio_residue_dict[pdb_index] = residue

    pdb_sequence_from_parser = "".join(pdb_seq_list)

    assert original_seq == pdb_sequence_from_parser, \
        "Mismatch between original_seq and sequence parsed from PDB. Check chain ID and parsing logic."

    one_to_pdb_index_map = {i+1: pdb_residues[i] for i in range(len(original_seq))}
    pdb_index_to_one_map = {v: k for k, v in one_to_pdb_index_map.items()}
    pos_to_wt = {i+1: original_seq[i] for i in range(len(original_seq))}

    # Model Setup
    base_model = ESM3_sm_open_v0()
    peft_model = utils.add_lora_to_esm3(
        base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_mode='expanded',
        seed=42
    )

    model = models.ESM3LoRAModel(
        peft_model, 
        freeze_lora=True,
        inference_mode=True,
        shared_scale=1,
        shared_bias=0
        ).to(args.device)
    
    if args.checkpoint:
        model = utils.load_ckpt_weights(model, args.checkpoint, device=args.device)
    model.eval()

    utils.print_trainable_parameters(model)

    backbone_mutation = None
    pred_combined = pd.DataFrame()

    # --- FEATURE: Load Data & Parse Mutations ---
    modes_to_run = {args.mode} 
    subset_df_obj = None

    if args.subset_df:
        try:
            subset_df_obj = pd.read_csv(args.subset_df)
            print(f"Loaded subset_df with {len(subset_df_obj)} rows.")
        except Exception as e:
            print(f"Error loading subset_df: {e}")
            sys.exit(1)
            
    elif args.mutations:
        mut_list = [m.strip() for m in args.mutations.split(',')]
        subset_df_obj = pd.DataFrame({'mut_type': mut_list})
        print(f"Created inputs from --mutations flag with {len(subset_df_obj)} rows.")

    elif has_screen_args:
        target_positions = []
        if args.screen_residues:
            req_res = [r.strip() for r in args.screen_residues.split(',') if r.strip()]
            for r in req_res:
                if r in pdb_index_to_one_map:
                    target_positions.append(pdb_index_to_one_map[r])
                else:
                    raise ValueError(f"\n[!] ERROR: Residue '{r}' requested in --screen_residues was not found in PDB chain '{args.chain_id}'. Ensure you are not including chain IDs and that you selected the correct chain.\n")
            target_positions = list(set(target_positions))
            
        elif args.screen_residues_except:
            exc_res = [r.strip() for r in args.screen_residues_except.split(',') if r.strip()]
            exc_pos = []
            for r in exc_res:
                if r in pdb_index_to_one_map:
                    exc_pos.append(pdb_index_to_one_map[r])
                else:
                    raise ValueError(f"\n[!] ERROR: Residue '{r}' requested in --screen_residues_except was not found in PDB chain '{args.chain_id}'.\n")
            target_positions = [i+1 for i in range(len(original_seq)) if (i+1) not in exc_pos]
        
        mut_list = []
        AAs = list('ACDEFGHIKLMNPQRSTVWY')
        
        if args.mode == 'singles':
            for pos in target_positions:
                wt = pos_to_wt[pos]
                pdb_pos = one_to_pdb_index_map[pos] 
                for mut in AAs:
                    if mut != wt:
                        mut_list.append(f"{wt}{pdb_pos}{mut}")
        elif args.mode == 'doubles':
            pairs = list(itertools.combinations(target_positions, 2))
            if len(pairs) > 1000:
                print(f"Warning: This will create {len(pairs)} unique position pairs. This equates to {len(pairs) * 19 * 19} total double mutations!")
            for pos1, pos2 in pairs:
                wt1 = pos_to_wt[pos1]
                pdb_pos1 = one_to_pdb_index_map[pos1]
                wt2 = pos_to_wt[pos2]
                pdb_pos2 = one_to_pdb_index_map[pos2]
                
                for mut1 in AAs:
                    if mut1 == wt1: continue
                    for mut2 in AAs:
                        if mut2 == wt2: continue
                        mut_list.append(f"{wt1}{pdb_pos1}{mut1}:{wt2}{pdb_pos2}{mut2}")

        subset_df_obj = pd.DataFrame({'mut_type': mut_list})
        if subset_df_obj.empty:
            raise ValueError("\n[!] ERROR: The generated mutation list is empty. This often happens if you select '--mode doubles' but provide fewer than 2 valid residues.\n")
            
        print(f"Created inputs from screen residue arguments with {len(subset_df_obj)} rows.")

    if subset_df_obj is not None:
        target_col = 'mut_type'
        
        has_parsed_cols = all(col in subset_df_obj.columns for col in ['wt1', 'pos1', 'mut1'])
        
        if has_parsed_cols and target_col in subset_df_obj.columns:
            print("Inputs contain both raw mutation strings and parsed columns. Trusting existing parsed columns.")
        elif not has_parsed_cols and target_col in subset_df_obj.columns:
            print(f"Parsing multi-mutants from column '{target_col}'...")
            max_muts = subset_df_obj[target_col].astype(str).str.count(':').max() + 1
            subset_df_obj = utils.parse_multimutant_column(subset_df_obj, mut_column=target_col, max_mutations=max_muts)
            
        for i in range(1, 4):  
            pos_col = f'pos{i}'
            wt_col = f'wt{i}'
            
            if pos_col in subset_df_obj.columns:
                # --- NEW CODE: Catch float casting immediately before mapping ---
                def clean_pdb_idx(val):
                    if pd.isna(val): return np.nan
                    if isinstance(val, float) and val.is_integer():
                        return str(int(val)).strip()
                    return str(val).strip()

                subset_df_obj[f'{pos_col}_pdb'] = subset_df_obj[pos_col].apply(clean_pdb_idx)

                def validate_and_convert(row):
                    pdb_idx = row[f'{pos_col}_pdb']
                    if pd.isna(pdb_idx): return np.nan
                    
                    if pdb_idx not in pdb_index_to_one_map:
                        raise ValueError(
                            f"\n[!] ERROR: Position '{pdb_idx}' not found in PDB structure. "
                            f"Are you accidentally using 1-based sequence indices instead of PDB numbering?\n"
                        )
                        
                    one_based_idx = pdb_index_to_one_map[pdb_idx]
                    expected_wt = pos_to_wt[one_based_idx]

                    if str(row[wt_col]) != expected_wt:
                         raise ValueError(
                             f"\n[!] ERROR: Wildtype mismatch at PDB position '{pdb_idx}'. "
                             f"Your input specifies '{row[wt_col]}', but the structure has '{expected_wt}'. "
                             f"Please verify you are using correct PDB numbering.\n"
                         )
                    
                    return one_based_idx

                subset_df_obj[pos_col] = subset_df_obj.apply(validate_and_convert, axis=1)

        # --- NEW CODE: Overwrite/construct PDB-indexed mut_type in input dataframe ---
        def build_pdb_mut_type(row):
            muts = []
            for i in range(1, 4):
                pos_col = f'pos{i}_pdb'
                if pos_col in row and pd.notna(row[pos_col]):
                    muts.append(f"{row[f'wt{i}']}{row[pos_col]}{row[f'mut{i}']}")
            return ":".join(muts) if muts else np.nan

        subset_df_obj['mut_type'] = subset_df_obj.apply(build_pdb_mut_type, axis=1)

        valid_muts_per_row = subset_df_obj[[c for c in subset_df_obj.columns if str(c).startswith('pos') and not str(c).endswith('_pdb')]].notna().sum(axis=1)
        
        if (valid_muts_per_row == 1).any():
            modes_to_run.add('singles')
        if (valid_muts_per_row == 2).any():
            modes_to_run.add('doubles')
        if (valid_muts_per_row > 2).any():
            modes_to_run.add('multi')
            
        print(f"Inferred modes to run: {modes_to_run}")

    # --- Execution Loop ---
    if 'singles' in modes_to_run:
        print("Running singles...")
        pred_singles = model.infer_single_mutants(
            pdb_path_to_use, 
            chain=args.chain_id, 
            strategy=args.strategy, 
            subset_df=subset_df_obj, 
            backbone_mutation=backbone_mutation, 
            quiet=False
        )
        
        pred_singles['pos1_pdb'] = pred_singles['pos1'].map(one_to_pdb_index_map)
        
        # --- NEW CODE: Mut_type renaming and PDB construction ---
        if 'mut_type' in pred_singles.columns:
            pred_singles.rename(columns={'mut_type': 'mut_info_seq_pos'}, inplace=True)
        else:
            # Fallback creation if model didn't output it natively
            pred_singles['mut_info_seq_pos'] = pred_singles['wt1'] + pred_singles['pos1'].astype(str) + pred_singles['mut1']
            
        pred_singles['mut_type'] = pred_singles['wt1'] + pred_singles['pos1_pdb'].astype(str) + pred_singles['mut1']
        
        pred_singles['id'] = (
            args.code + args.chain_id + 
            ('_' + backbone_mutation if backbone_mutation else '') + '_' + 
            pred_singles['mut_type']
        )

        pred_singles = pred_singles.drop('delta_logit1', axis=1)

        pred_combined = pd.concat([pred_combined, pred_singles])

    if 'doubles' in modes_to_run:
        print("Running doubles...")

        if subset_df_obj is not None:
            if 'pos3' not in subset_df_obj.columns:
                subset_df_obj = subset_df_obj.assign(pos3=np.nan)
            
            subset_df_double = subset_df_obj.loc[~(subset_df_obj['pos2'].isna()) & (subset_df_obj['pos3'].isna())].copy()
        else:
            subset_df_double = None

        pred_doubles = model.infer_double_mutants(
            pdb_path_to_use, 
            chain=args.chain_id, 
            strategy=args.strategy, 
            subset_df=subset_df_double, 
            backbone_mutation=backbone_mutation, 
            quiet=False
        )
        
        pred_doubles['pos1_pdb'] = pred_doubles['pos1'].map(one_to_pdb_index_map)
        pred_doubles['pos2_pdb'] = pred_doubles['pos2'].map(one_to_pdb_index_map)

        # --- NEW CODE: Mut_type renaming and PDB construction ---
        if 'mut_type' in pred_doubles.columns:
            pred_doubles.rename(columns={'mut_type': 'mut_info_seq_pos'}, inplace=True)
        else:
            pred_doubles['mut_info_seq_pos'] = (
                pred_doubles['wt1'] + pred_doubles['pos1'].astype(str) + pred_doubles['mut1'] + ':' + 
                pred_doubles['wt2'] + pred_doubles['pos2'].astype(str) + pred_doubles['mut2']
            )

        pred_doubles['mut_type'] = (
            pred_doubles['wt1'] + pred_doubles['pos1_pdb'].astype(str) + pred_doubles['mut1'] + ':' + 
            pred_doubles['wt2'] + pred_doubles['pos2_pdb'].astype(str) + pred_doubles['mut2']
        )

        pred_doubles['id'] = (
            args.code + args.chain_id + 
            ('_' + backbone_mutation if backbone_mutation else '') + '_' + 
            pred_doubles['mut_type']
        )
        
        if args.calculate_distances:
            pred_doubles['dist'] = pred_doubles.apply(
                lambda x: get_closest_heavy_atom_distance(x['pos1_pdb'], x['pos2_pdb'], bio_residue_dict), 
                axis=1
            )

        pred_doubles = pred_doubles.drop(['ctx_geom_1', 'ctx_geom_2'], axis=1) 

        pred_combined = pd.concat([pred_combined, pred_doubles])

    if 'multi' in modes_to_run:
        print("Running multi...")

        if subset_df_obj is not None:
            subset_df_multi = subset_df_obj.loc[~subset_df_obj['pos3'].isna()].copy()
        else:
            subset_df_multi = None

        pred_multi = model.infer_multimutants_sampled(
            pdb_path_to_use, 
            chain=args.chain_id, 
            strategy=args.strategy, 
            subset_df=subset_df_multi, 
            K_paths=args.multi_paths
        )

        def convert_multi_mut_string(mut_str, mapping):
            if pd.isna(mut_str): return mut_str
            converted = []
            for m in mut_str.split(':'):
                wt = m[0]
                mut_aa = m[-1]
                
                try:
                    pos_1based = int(m[1:-1])
                except ValueError:
                    raise ValueError(f"\n[!] ERROR: Could not parse 1-based integer position from mutation string: '{m}'\n")

                if pos_1based not in mapping:
                    raise ValueError(f"\n[!] ERROR: 1-based position '{pos_1based}' generated by the model could not be mapped to a PDB index. Aborting to prevent silent failures.\n")
                    
                pos_pdb = mapping[pos_1based]
                converted.append(f"{wt}{pos_pdb}{mut_aa}")
                
            return ":".join(converted)

        # --- NEW CODE: Rename original and construct PDB mut_type ---
        if 'mut_type' in pred_multi.columns:
            pred_multi.rename(columns={'mut_type': 'mut_info_seq_pos'}, inplace=True)
            
        pred_multi['mut_type'] = pred_multi['mut_info_seq_pos'].apply(
            lambda x: convert_multi_mut_string(x, one_to_pdb_index_map)
        )

        pred_multi['id'] = (
            args.code + args.chain_id + 
            ('_' + backbone_mutation if backbone_mutation else '') + '_' + 
            pred_multi['mut_type']
        )

        pred_combined = pd.concat([pred_combined, pred_multi])

    def has_disallowed_mutation(mut_type, disallowed_str):
        if pd.isna(mut_type): return False
        muts = mut_type.split(':')
        blacklist = disallowed_str.split(',')
        for mut in muts:
            if mut[-1] in blacklist:
                return True
        return False
            
    pred_combined['has_disallowed'] = pred_combined['mut_type'].apply(lambda x: has_disallowed_mutation(x, args.disallow_mutants))
    pred_combined.loc[pred_combined['has_disallowed'], 'ddg_pred'] = np.nan
    
    if not pred_combined.empty:

        if subset_df_obj is not None:
            
            if 'id' not in subset_df_obj.columns and 'mut_type' in subset_df_obj.columns:
                subset_df_obj['id'] = (
                    args.code + args.chain_id + 
                    ('_' + backbone_mutation if backbone_mutation else '') + '_' + 
                    subset_df_obj['mut_type']
                )

            overlap_cols = list(set(subset_df_obj.columns).intersection(set(pred_combined.columns)))
            
            join_target = 'id'

            if join_target in overlap_cols:
                overlap_cols.remove(join_target)
                
            size_before = len(pred_combined)
            
            pred_combined = subset_df_obj.set_index(join_target).join(
                pred_combined.set_index(join_target).drop(overlap_cols, axis=1),
                how='inner'
            ).reset_index()
            
            assert len(pred_combined) == size_before, \
                f"\n[!] ERROR: Assertion failed! Inner join changed dataframe size. Expected {size_before} rows, got {len(pred_combined)} rows. Check for duplicate IDs or dropped rows.\n"

        if args.output_csv:
            pred_combined.to_csv(args.output_csv, index=False)
        else:
            pred_combined.to_csv(f'./{args.code}_{args.chain_id}_{args.mode}_{args.strategy}.csv', index=False)
    else:
        print("No predictions were generated.")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, required=True)
        parser.add_argument('--lora_alpha', type=int, required=True)
        parser.add_argument('--lora_rank', type=int, default=6)
        parser.add_argument('--multi_paths', type=int, default=4)
        parser.add_argument('--input_structure', type=str, required=True)
        parser.add_argument('--code', type=str, default='protein')
        parser.add_argument('--chain_id', type=str, default='A')
        parser.add_argument('--mode', type=str, choices=['singles', 'doubles'], default='singles')
        parser.add_argument('--strategy', type=str, choices=['parallel', 'masked', 'direct'], default='masked')
        parser.add_argument('--calculate_distances', action='store_true')
        parser.add_argument('--output_csv', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda:0')
        parser.add_argument('--subset_df', type=str, default=None, help="Path to CSV file containing subset of mutations")
        parser.add_argument('--mutations', type=str, default=None, help="Comma-separated list of mutations (e.g., A12C,A12C:D15E) to score directly.")
        parser.add_argument('--screen_residues', type=str, default=None, help="Comma-separated list of PDB indices to screen (mutually exclusive with screen_residues_except)")
        parser.add_argument('--screen_residues_except', type=str, default=None, help="Comma-separated list of PDB indices to exclude from screen (mutually exclusive with screen_residues)")
        parser.add_argument('--disallow_mutants', type=str, default='', help="Comma-separated list of mutation identities to replace with N/A (untrusted)")
        parser.add_argument('--hf_token', type=str, default=None)

        args, remaining_argv = parser.parse_known_args()

        if args.hf_token:
            login(args.hf_token)
        else:
            os.environ['INFRA_PROVIDER'] = "1"
            os.chdir(Path(__file__).resolve().parent.parent.parent)

        main(args)