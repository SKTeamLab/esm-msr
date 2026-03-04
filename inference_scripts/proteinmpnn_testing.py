# adapted from ProteinMPNN/protein_mpnn_utils.py

import sys
sys.path.append('/home/sareeves/software/ProteinMPNN/')
from protein_mpnn_utils import tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN

import os
import warnings
import torch
import copy
import argparse
import time
import itertools
import random

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

def calculate_rigorous_scores(model, row, X, S_wt, mask, chain_M, residue_idx, chain_encoding_all, alphabet, K_paths=4, N_background_orders=4):
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
    except:
        return None, None, None
    
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
    
    add_batch_S = []
    add_batch_X = []
    add_batch_mask = []
    add_batch_chain_M = []
    add_batch_residue_idx = []
    add_batch_chain_encoding = []
    
    # We need to track which base_noise index to use for each batch item
    add_noise_indices = [] 
    add_targets = [] # (batch_idx, pos_idx, mut_idx, wt_idx)
    
    batch_counter = 0
    
    for pm in parsed_muts:
        p_idx = pm['pos'] + oc
        wt_idx = alphabet.find(pm['wt'])
        mut_idx = alphabet.find(pm['mut'])
        
        for n in range(N_background_orders):
            add_batch_S.append(S_wt.clone()[0])
            add_batch_X.append(X[0])
            add_batch_mask.append(mask[0])
            add_batch_chain_M.append(chain_M[0])
            add_batch_residue_idx.append(residue_idx[0])
            add_batch_chain_encoding.append(chain_encoding_all[0])
            
            add_noise_indices.append(n) # Remember which noise to use
            add_targets.append((batch_counter, p_idx, mut_idx, wt_idx))
            batch_counter += 1

    additive_score = 0.0
    
    if len(add_batch_S) > 0:
        S_batch = torch.stack(add_batch_S)
        X_batch = torch.stack(add_batch_X)
        mask_batch = torch.stack(add_batch_mask)
        chain_M_batch = torch.stack(add_batch_chain_M)
        residue_idx_batch = torch.stack(add_batch_residue_idx)
        chain_encoding_batch = torch.stack(add_batch_chain_encoding)
        
        # Construct Decoding Order
        # 1. Start with the specific base_noise for this sample
        # 2. Force the target position to be last (1000.0)
        decoding_vals = torch.stack([base_noise[n].clone() for n in add_noise_indices])
        
        for b_idx, p_idx, _, _ in add_targets:
            decoding_vals[b_idx, p_idx] = 1000.0
            
        decoding_order = torch.argsort(decoding_vals)
        
        with torch.no_grad():
            logits = model.forward(
                X_batch, S_batch, mask_batch, chain_M_batch, residue_idx_batch, chain_encoding_batch,
                torch.randn_like(chain_M_batch), use_input_decoding_order=True, decoding_order=decoding_order
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            
        # Accumulate Additive Scores
        # Strategy: Sum over mutations, Average over N orders
        # To do this correctly: Group by mutation, average, then sum totals.
        mutation_scores_map = {pm['pos']+oc: [] for pm in parsed_muts}
        
        for b_idx, p_idx, mut_idx, wt_idx in add_targets:
            score = (log_probs[b_idx, p_idx, mut_idx] - log_probs[b_idx, p_idx, wt_idx]).item()
            mutation_scores_map[p_idx].append(score)
            
        for p_idx in mutation_scores_map:
            scores = mutation_scores_map[p_idx]
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
            
    epi_batch_S = []
    epi_batch_X = []
    epi_batch_mask = []
    epi_batch_chain_M = []
    epi_batch_residue_idx = []
    epi_batch_chain_encoding = []
    
    epi_noise_indices = []
    epi_targets = []
    
    batch_counter = 0
    
    for path in paths:
        current_seq = S_wt.clone()[0]
        
        for p_idx in path:
            pm = next(x for x in parsed_muts if (x['pos'] + oc) == p_idx)
            wt_idx = alphabet.find(pm['wt'])
            mut_idx = alphabet.find(pm['mut'])
            
            # Expand by SAME N orders
            for n in range(N_background_orders):
                epi_batch_S.append(current_seq.clone())
                epi_batch_X.append(X[0])
                epi_batch_mask.append(mask[0])
                epi_batch_chain_M.append(chain_M[0])
                epi_batch_residue_idx.append(residue_idx[0])
                epi_batch_chain_encoding.append(chain_encoding_all[0])
                
                epi_noise_indices.append(n) # Reusing the same base noise
                epi_targets.append((batch_counter, p_idx, mut_idx, wt_idx))
                batch_counter += 1
            
            # Update context
            current_seq[p_idx] = mut_idx

    epistatic_score = 0.0
    
    if len(epi_batch_S) > 0:
        S_batch = torch.stack(epi_batch_S)
        X_batch = torch.stack(epi_batch_X)
        mask_batch = torch.stack(epi_batch_mask)
        chain_M_batch = torch.stack(epi_batch_chain_M)
        residue_idx_batch = torch.stack(epi_batch_residue_idx)
        chain_encoding_batch = torch.stack(epi_batch_chain_encoding)
        
        # Reuse the SAME base noise
        decoding_vals = torch.stack([base_noise[n].clone() for n in epi_noise_indices])
        
        # Force targets to end
        for b_idx, p_idx, _, _ in epi_targets:
            decoding_vals[b_idx, p_idx] = 1000.0
            
        decoding_order = torch.argsort(decoding_vals)
        
        with torch.no_grad():
            logits = model.forward(
                X_batch, S_batch, mask_batch, chain_M_batch, residue_idx_batch, chain_encoding_batch,
                torch.randn_like(chain_M_batch), use_input_decoding_order=True, decoding_order=decoding_order
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            
        # Accumulate Path Scores
        path_scores = []
        ptr = 0
        
        for path in paths:
            path_total = 0.0
            for _ in path:
                # Average over N orders for this step
                step_sum = 0.0
                for _ in range(N_background_orders):
                    b_idx, p_idx, mut_idx, wt_idx = epi_targets[ptr]
                    delta = log_probs[b_idx, p_idx, mut_idx] - log_probs[b_idx, p_idx, wt_idx]
                    step_sum += delta.item()
                    ptr += 1
                path_total += (step_sum / N_background_orders)
            path_scores.append(path_total)
            
        epistatic_score = sum(path_scores) / len(path_scores)

    runtime = time.time() - start_time
    return epistatic_score, additive_score, runtime


def predict(df, model, K_paths=4):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 
         'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R',
         'TRP': 'W', 'ALA': 'A', 'VAL':'V',  'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}
    
    pdbparser = PDBParser()
    
    logps = df
    device = torch.device("cuda:0")

    with tqdm(total=len(df)) as pbar:
        for (code, chain), group in df.groupby(['code', 'chain']):
    
            drop_chains = []

            # get chain sequences and remove chains of only heteroatoms (e.g. DNA)
            pdb_path = os.path.join(group['pdb_file'].head(1).item())
            structure = pdbparser.get_structure(code, pdb_path)
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

            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

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

            # --- Usage Example ---
            with torch.no_grad():
                for uid, row in group.iterrows():
                    epistatic_score, additive_score, runtime = calculate_rigorous_scores(
                        model, row, X, S, mask, chain_M, residue_idx, chain_encoding_all, alphabet, K_paths=K_paths, N_background_orders=3
                    )
                    
                    if additive_score is not None:
                        logps.at[uid, 'mpnn_score_additive'] = additive_score
                        logps.at[uid, 'mpnn_score'] = epistatic_score
                        logps.at[uid, 'mpnn_score_epistasis'] = epistatic_score - additive_score
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
    model_folder_path = os.path.join('/home/sareeves/software/ProteinMPNN', 'vanilla_model_weights')
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

    if not args.skip_external:
        external_test_dataloaders_names = ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
        #external_test_dataloaders_names = ['ptmul']
        stats_masked = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(name)

            df_true = pd.read_csv(DATA_DIR / f'{name}_mapped_new.csv')
            if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369']:
                df_true = df_true.reset_index()
                df_true['position_pdb'] = df_true['position']
                df_true['position'] = df_true['seq_pos']
                df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true['mut_type'] = df_true['wild_type'] + df_true['seq_pos'].astype(int).astype(str) + df_true['mutation']
                df_true = df_true.set_index('id')

            else:
                df_true = df_true.reset_index()
                df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            res_masked, time_masked = timed_call(predict, df=df_true, model=model)
            print(res_masked[['mpnn_score']].head())

            os.makedirs(os.path.dirname(DATA_DIR / f'predictions/{name if name!= "ptmul" else "PTMUL"}/proteinmpnn/_'), exist_ok=True)
            res_masked.to_csv(DATA_DIR / f'predictions/{name if name!= "ptmul" else "PTMUL"}/proteinmpnn/proteinmpnn_020.csv')

            stats_masked.at[name, 'spearman'] = res_masked[['ddG', 'mpnn_score']].corr('spearman').iloc[0,1]
            stats_masked.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG', top_n=30)
            stats_masked.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG', threshold=0)
            stats_masked.at[name, 'time'] = time_masked

            if 'ptmul' not in name:
                assert len(df_true) == len(res_masked)
            else:
                pass

            os.makedirs(os.path.dirname(DATA_DIR / f'stats/external/proteinmpnn/_'), exist_ok=True)
            stats_masked.to_csv(DATA_DIR / f'stats/external/proteinmpnn/proteinmpnn_020.csv')

    ############## REPEAT WITH SPECIFIC SPLITS ################

    if args.split is not None and not args.skip_tsuboyama:
        splits = pd.read_csv(args.split, index_col=0)
        split_name = args.split.split('/')[-1].split('splits_')[1].split('.csv')[0]

        for scaffold in ['validation', 'testing']:

            results_masked = []

            stats_masked = pd.DataFrame()

            scaffold_ = scaffold.replace('testing', 'test')
            test_list = eval(splits.loc['stability', scaffold])
            #test_list = test_list[:3]
            #test_list = [str(t).replace('|', '+') for t in test_list]

            tsu = pd.read_csv('~/PSLMs/data/preprocessed/tsuboyama_all_subs_corrected.csv')
            tsu = tsu.loc[~tsu['mut_type'].apply(utils.is_fake_mutation)]
            tsu = tsu.loc[~tsu['uid'].apply(utils.is_improper_mutation)]
            
            tsu = utils.parse_multimutant_column(tsu, 'mut_type')
            tsu['id'] = tsu['code'] + '_' + tsu['mut_type']
            tsu = tsu.sort_values('id')
            tsu['mut_seq'] = tsu['aa_seq']
            tsu = tsu.set_index('uid')

            for code in tqdm(test_list):

                df_true = tsu.loc[tsu['code'].str.contains(code, regex=False)]
                if not code.startswith('v2_'):
                    df_true = df_true.loc[~df_true['code'].str.startswith('v2_')]
                df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

                print(df_true[['dddG_ML']])
                res_masked, time_masked = timed_call(predict, df=df_true, model=model)
                print(res_masked[['mpnn_score', 'mpnn_score_epistasis']].head())

                stats_masked.at[code, 'spearman'] = res_masked[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
                try:
                    stats_masked.at[code, 'spearman_epi'] = res_masked[['dddG_ML', 'mpnn_score_epistasis']].dropna().corr('spearman').iloc[0,1]
                except:
                    stats_masked.at[code, 'spearman_epi'] = float('nan')
                stats_masked.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', top_n=30)
                stats_masked.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', threshold=0)
                stats_masked.at[code, 'time'] = time_masked

                assert len(df_true) == len(res_masked)

                results_masked.append(res_masked)

            results_masked = pd.concat(results_masked, axis=0)

            print(stats_masked.mean(axis=0))  

            os.makedirs(os.path.dirname((DATA_DIR / f'predictions/{split_name}-{scaffold_}/proteinmpnn/proteinmpnn_020.csv')), exist_ok=True)
            os.makedirs(os.path.dirname((DATA_DIR / f'stats/{split_name}-{scaffold_}/proteinmpnn/proteinmpnn_020.csv')), exist_ok=True)

            stats_masked.mean(axis=0).to_csv(DATA_DIR / f'stats/{split_name}-{scaffold_}/proteinmpnn/proteinmpnn_020.csv')

            results_masked.to_csv(DATA_DIR / f'predictions/{split_name}-{scaffold_}/proteinmpnn/proteinmpnn_020.csv')

            stats_masked.to_csv(DATA_DIR / f'stats/{split_name}-{scaffold_}/proteinmpnn/proteinmpnn_020.csv')

            torch.cuda.empty_cache()

    ######################################

    if not args.skip_dms:

        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain'] #, 'A4_HUMAN_Seuma_2022'] # 'GB1_Wu_2016_binding_domain','A4_HUMAN_Seuma_2022', 
        stats_masked = pd.DataFrame()

        results_masked = []
    
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

            res_masked, time_masked = timed_call(predict, df=df_true, model=model)
            print(res_masked[['mpnn_score']].head())

            assert len(df_true) == len(res_masked)

            os.makedirs(os.path.dirname((DATA_DIR / f'predictions/{prot}/proteinmpnn/proteinmpnn_020.csv')), exist_ok=True)
            res_masked.to_csv(DATA_DIR / f'predictions/{prot}/proteinmpnn/proteinmpnn_020.csv')

            results_masked.append(res_masked)

            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
            try:
                stats_masked.at[prot, 'spearman_epi'] = res_masked[['dddG_ML', 'mpnn_score_epistasis']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_masked.at[prot, 'spearman_epi'] = float('nan')
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', threshold=0)
            stats_masked.at[prot, 'time'] = time_masked

        print(stats_masked)

        os.makedirs(os.path.dirname((DATA_DIR / f'stats/DMS/proteinmpnn/proteinmpnn_020.csv')), exist_ok=True)

        results_masked = pd.concat(results_masked, axis=0)

        stats_masked.to_csv(DATA_DIR / f'stats/DMS/proteinmpnn/proteinmpnn_020.csv')

        torch.cuda.empty_cache()

        #os.makedirs(os.path.dirname(DATA_DIR / f'stats/DMS/proteinmpnn/proteinmpnn_020.csv'), exist_ok=True)
        #prot_stats.to_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/stats/DMS/proteinmpnn/proteinmpnn_020.csv{"_alpha"+str(args.lora_alpha)}chain_rule_avg.csv')

    #########################################

    if not args.skip_domainome:

        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/domainome1/domainome_mapped_new.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset='pdb_file')
        df = df[['code', 'mut_seq', 'mut_info', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        df['mut_type'] = df['mut_info']
        df['chain'] = 'A'
        results_masked = []
        stats_masked = pd.DataFrame()

        for prot in tqdm(df['code'].unique()):

            strategy = 'masked'

            df_true = df.loc[df['code']==prot]
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')

            pdb = df_true['pdb_file'].head(1).item()

            res_masked, time_masked = timed_call(predict, df=df_true, model=model)
            print(res_masked[['mpnn_score']].head())

            assert len(df_true) == len(res_masked)

            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'mpnn_score']].corr('spearman').iloc[0,1]
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'mpnn_score', 'ddG_ML', threshold=0)

            stats_masked.at[prot, 'time'] = time_masked

            results_masked.append(res_masked)

        results_masked_out = pd.concat(results_masked, axis=0)

        print(stats_masked.mean(axis=0))

        os.makedirs(os.path.dirname(DATA_DIR / f'predictions/domainome/proteinmpnn/proteinmpnn_020.csv'), exist_ok=True)
        os.makedirs(os.path.dirname(DATA_DIR / f'stats/domainome/proteinmpnn/proteinmpnn_020.csv'), exist_ok=True)

        stats_masked.mean(axis=0).to_csv(DATA_DIR / f'stats/domainome/proteinmpnn/proteinmpnn_020.csv')

        results_masked_out.to_csv(DATA_DIR / f'predictions/domainome/proteinmpnn/proteinmpnn_020.csv')

        stats_masked.to_csv(DATA_DIR / f'stats/domainome/proteinmpnn/proteinmpnn_020.csv')

        torch.cuda.empty_cache()

    ##########################################

    if not args.skip_functional:

        test_list_DMS = ['D7PM05_CLYGR', 'GFP_AEQVI', 'HIS7_YEAST', 'Q6WV12_9MAXI', 'Q8WTC7_9CNID', 'RASK_HUMAN']

        for mem_size, prot in zip([8,8,8,8,8,8], test_list_DMS):

            strategy = 'masked'

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/lora/DMS/csv_formatted/{prot}.csv')
            df_true['mut_type'] = df_true['MUTS'].apply(lambda x: x.replace(';', ':'))
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true = df_true.set_index('id')
            df_true = utils.parse_multimutant_column(df_true, mut_column='mut_type')
            has_doubles = len(df_true['mut_type'].str.contains(':')) > 0
            print(prot, has_doubles)
            print(df_true)

            pdb = df_true['pdb_file'].head(1).item()

            res_masked, time_masked = timed_call(predict, df=df_true, model=model)
        
            #pred_combined = res_masked #.set_index('id')

            #res = df_true.join(pred_combined)
            res = res_masked
            assert len(df_true) == len(res)
            print(res[['ddG_dir', 'mpnn_score']].corr('spearman').iloc[0,1])
            print(res.head())

            os.makedirs(os.path.dirname(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/predictions/{prot}/proteinmpnn/proteinmpnn_020.csv'))
            res.to_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/predictions/{prot}/proteinmpnn/proteinmpnn_020.csv')
            torch.cuda.empty_cache()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--split', type=str) #'/home/sareeves/software/esm-msr/data/splits_megascale.csv'
        parser.add_argument('--loc', type=str, default='inference_scripts')
        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')

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
        if args.skip_functional:
            print('Skipping double mutant DMS assays!')
        if args.skip_domainome:
            print('Skipping domainome VAMP assays!')
        if not args.split:
            print('Warning! Not using any specific split file!')

        main_(args)