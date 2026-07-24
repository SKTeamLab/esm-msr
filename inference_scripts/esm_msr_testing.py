import pandas as pd
import os
import torch
from tqdm import tqdm
import argparse
import time
import logging

from huggingface_hub import login, get_token

from esm_msr import stats, utils, models, inference, preprocess_megascale
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "LoRA_models"

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def safe_spearman(df, col1, col2):
    """Safely computes spearman correlation, returning NaN if insufficient valid data."""
    valid_df = df[[col1, col2]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    return valid_df.corr('spearman').iloc[0, 1]

def safe_ndcg(df, col1, col2, top_n=None, threshold=None):
    """Safely computes NDCG, returning NaN if insufficient valid data."""
    valid_df = df[[col1, col2]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    preds = valid_df[col1].to_numpy().reshape(1, -1)
    truths = valid_df[col2].to_numpy().reshape(1, -1)
    ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = stats.compute_ndcg_flexible(preds, truths, top_n=top_n, threshold=threshold)
    return ndcg_val

def update_stats(stats_df, row_name, res_df, true_col, pred_col, epi_true_col='dddG_ML', epi_pred_col=None, time_val=None):
    """Helper to cleanly extract subset metrics for a specific predictive branch."""
    if pred_col not in res_df.columns:
        # Added explicit print to prevent silent failures
        print(f"Warning: Skipping stats update for '{row_name}'; column '{pred_col}' not found in predictions.")
        return stats_df
        
    stats_df.at[row_name, 'spearman_all'] = safe_spearman(res_df, true_col, pred_col)
    
    if 'mut_type' in res_df.columns:
        is_single = ~res_df['mut_type'].str.contains(':')
        is_double = res_df['mut_type'].str.contains(':')
        
        stats_df.at[row_name, 'spearman_singles'] = safe_spearman(res_df[is_single], true_col, pred_col)
        stats_df.at[row_name, 'n_singles'] = is_single.sum()
        
        stats_df.at[row_name, 'spearman_doubles'] = safe_spearman(res_df[is_double], true_col, pred_col)
        stats_df.at[row_name, 'n_doubles'] = is_double.sum()
    else:
        stats_df.at[row_name, 'spearman_singles'] = float('nan')
        stats_df.at[row_name, 'n_singles'] = 0
        stats_df.at[row_name, 'spearman_doubles'] = float('nan')
        stats_df.at[row_name, 'n_doubles'] = 0

    if epi_pred_col and epi_true_col in res_df.columns and epi_pred_col in res_df.columns:
        stats_df.at[row_name, 'spearman_doubles_epi'] = safe_spearman(res_df, epi_true_col, epi_pred_col)
    else:
        stats_df.at[row_name, 'spearman_doubles_epi'] = float('nan')

    stats_df.at[row_name, 'ndcg@96'] = safe_ndcg(res_df, pred_col, true_col, top_n=96)
    stats_df.at[row_name, 'ndcg>0'] = safe_ndcg(res_df, pred_col, true_col, threshold=0)
    
    if time_val is not None:
        stats_df.at[row_name, 'time'] = time_val
        
    return stats_df


def main_(args):

    CHECKPOINT_STR = str(args.checkpoint) if args.checkpoint else "zeroshot"

    print('\n\n\n\n\n')
    print(f"Running Inference for Checkpoint: {CHECKPOINT_STR}")
    print('\n\n\n\n\n')

    os.makedirs('tmp', exist_ok=True)

    if CHECKPOINT_STR != 'zeroshot':
        hparams_path = os.path.join(MODEL_DIR, os.path.dirname(args.checkpoint), 'hparams.yaml')
        parsed_config = inference.parse_hparams_to_lora_config(hparams_path)
        adapter_mode = parsed_config.get('adapter_mode', 'dual')
        lora_mode = parsed_config.get('lora_mode', 'ensemble')
        if args.lora_epsilon != 1:
            parsed_config['wt_config']['lora_alpha'] *= args.lora_epsilon
            parsed_config['mt_config']['lora_alpha'] *= args.lora_epsilon

        lora_config = {
            'wt_config': parsed_config['wt_config'],
            'mt_config': parsed_config['mt_config']
        }
    
    else:
        mt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "incl_structure_encoder": False, "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        wt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "incl_structure_encoder": False, "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        adapter_mode = 'dual'
        lora_mode = 'ensemble'
        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': args.seed}        

    model = models.MSRModel(
        lora_config=lora_config, shared_scale_init=1, shared_bias_init=0, adapter_mode=adapter_mode,
        lora_mode=lora_mode, model_dtype=torch.float32, inference_mode=True
    ).to('cuda:0')

    # ---------------------------------------------------------
    # Robust Checkpoint Loading
    # ---------------------------------------------------------
    if args.checkpoint:
        ckpt_path = str(MODEL_DIR / args.checkpoint)
        model.load_lora_weights(ckpt_path)
    else:
        print('Zero shot mode!')
    
    model.eval()

    # =========================================================================
    # EXTERNAL BENCHMARKS
    # =========================================================================
    if not args.skip_external:
        external_test_dataloaders_names = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
        
        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(f"Processing External Dataset: {name}")

            res_combined = []
            total_time = 0

            df_true = pd.read_csv(DATA_DIR / f"{name}_mapped.csv")
            df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
            if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's2648', 's8754']:
                df_true = df_true.reset_index()
                df_true['position_pdb'] = df_true['position']
                df_true['position'] = df_true['seq_pos']
                df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                if 'dTm' in df_true.columns:
                    df_true['ddG'] = df_true['dTm']
            else:
                df_true = df_true.reset_index()
                df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            for (pdb, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                
                unique_data = data[~data.index.duplicated(keep='first')]
                
                input_data = inference.standardize_input_df(unique_data, quiet=True)
                pred_df, t_inf = timed_call(
                    inference.infer_mutants, 
                    model=model, df=input_data, batch_size=1, quiet=True, mask_strategy=args.mask_strategy, 
                    optimize_wt_pass=(args.mask_strategy is None), skip_reverse=args.skip_reverse
                )
                pred_df['id'] = code + chain + '_' + pred_df['mut_type_renumbered']
                pred_df = pred_df.set_index('id')

                overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                res_partial = data.join(pred_df.drop(overlap_cols, axis=1))

                res_combined.append(res_partial)
                total_time += t_inf

            res_df = pd.concat(res_combined)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Extract Metrics based on New Output Schema
            stats_wt = update_stats(stats_wt, name, res_df, 'ddG', 'wt_lora_pred', 'dddG', 'wt_lora_dddg_pred', total_time)
            
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, name, res_df, 'ddG', 'mt_lora_pred', 'dddG', 'mt_lora_dddg_pred', total_time)
                stats_cmb = update_stats(stats_cmb, name, res_df, 'ddG', 'combined_pred', 'dddG', 'combined_dddg_pred', total_time)

            if 'ptmul' not in name:
                assert len(df_true) == len(res_df), f"Lost samples during join for {name}!"

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')

    # =========================================================================
    # TSUBOYAMA SPLITS
    # =========================================================================
    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
            data_file='/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 
            af_model_folder='/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs')
        splits = ds.create_training_splits(str(split_file), -1)

        if args.remove_spurs_homologs:
            ds.remove_homologs_from_scaffold(scaffold='train')
            ds.remove_homologs_from_scaffold(scaffold='val')

        for scaffold in ['validation', 'testing']:
            res_combined = []

            stats_wt = pd.DataFrame()
            stats_mt = pd.DataFrame()
            stats_cmb = pd.DataFrame()

            scaffold_ = {'validation': 'val', 'testing': 'test'}[scaffold]
            data_scaffold = ds.split_dfs[scaffold_]
            
            data_scaffold = utils.parse_multimutant_column(data_scaffold, 'mut_type')
            data_scaffold['id'] = data_scaffold['code'] + '_' + data_scaffold['mut_type']
            data_scaffold = data_scaffold.sort_values('id')

            time_per_code = {} # Track time per protein properly

            for code in tqdm(data_scaffold['code_wt'].unique()):
                df_true = data_scaffold.loc[data_scaffold['code_wt']==code].copy()
                df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
                assert len(df_true) > 0
                df_true['mut_structure'] = df_true['mut_structure'].fillna('-')

                t_total = 0

                for mut_structure, data in df_true.groupby('mut_structure'):
                    backbone_mutation = mut_structure if mut_structure != '-' else None

                    data = data.set_index('id')
                    data = utils.sum_individual_mutation_scores(data, 'ddG_ML', new_score_column='ddG_additive_ML')
                    data['dddG_ML'] = data['ddG_ML'] - data['ddG_additive_ML']

                    unique_data = data[~data.index.duplicated(keep='first')]
                    input_data = inference.standardize_input_df(unique_data, backbone_mutation=backbone_mutation, quiet=True)

                    # Unified Inference
                    pred_df, t = timed_call(
                        inference.infer_mutants, 
                        model=model, df=input_data, batch_size=16, backbone_mutation=backbone_mutation, quiet=True, 
                        skip_additive=False, mask_strategy=args.mask_strategy, optimize_wt_pass=(args.mask_strategy is None),
                        skip_reverse=args.skip_reverse
                    )
                    pred_df['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_df['mut_type_renumbered']
                    pred_df = pred_df.set_index('id')

                    overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                    res_partial = data.join(pred_df.drop(overlap_cols, axis=1))
                    res_combined.append(res_partial)
                    t_total += t
                
                time_per_code[code] = t_total # Store the accumulated time for this specific code

            # Aggregate DataFrames
            res_df = pd.concat(res_combined)

            # File Operations
            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Metrics
            for code, group in res_df.groupby('code_wt'):
                current_time = time_per_code.get(code, float('nan'))
                stats_wt = update_stats(stats_wt, code, group, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', current_time)
                if not args.skip_reverse:
                    stats_mt = update_stats(stats_mt, code, group, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', current_time)
                    stats_cmb = update_stats(stats_cmb, code, group, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', current_time)

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
                stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
                stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

            torch.cuda.empty_cache()

    # =========================================================================
    # DMS DATASETS
    # =========================================================================
    if not args.skip_dms:
        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']
        mem_sizes = [4, 4, 8, 8, 1, 1, 8]
        
        assert len(prots) == len(mem_sizes), f"Length mismatch: {len(prots)} proteins vs {len(mem_sizes)} memory sizes."

        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()
        
        res_combined = []

        for mem_size, prot in zip(mem_sizes, prots): 
            batch_sz = mem_size * 32

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/data/preprocessed/{prot}.csv')
            df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')
            
            has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
            if has_doubles:
                df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

            prot_name = '_'.join(prot.split('_')[:2])
            if prot_name == 'GB1_Wu':
                prot_name = 'GB1'

            unique_data = df_true[~df_true.index.duplicated(keep='first')]
            input_data = inference.standardize_input_df(unique_data, quiet=True)

            pred_df, t_inf = timed_call(
                inference.infer_mutants, 
                model=model, df=input_data, batch_size=batch_sz, quiet=False, skip_additive=False, 
                mask_strategy=args.mask_strategy, optimize_wt_pass=(args.mask_strategy is None),
                skip_reverse=args.skip_reverse
            )
            pred_df['id'] = prot_name + '_' + pred_df['mut_type_renumbered']
            pred_df = pred_df.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_df.columns)))
            res = df_true.join(pred_df.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res), f"Merge error on DMS {prot}"
            res_combined.append(res)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res.to_csv(out_path)

            stats_wt = update_stats(stats_wt, prot, res, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', t_inf)
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, prot, res, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', t_inf)
                stats_cmb = update_stats(stats_cmb, prot, res, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', t_inf)

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)

            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')

            torch.cuda.empty_cache()

    # =========================================================================
    # DOMAINOME DATASET
    # =========================================================================
    
    if not args.skip_domainome:
        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/data/domainome1/domainome_mapped_2026.csv'
        df = pd.read_csv(path)
        df['pdb_file'] = df['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset=['pdb_file', 'position'])
        df = df[['code', 'mut_type', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        
        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()

        res_combined = []
        
        should_skip_reverse_dom = args.skip_reverse or args.skip_reverse_domainome

        for prot in tqdm(df['code'].unique()):
            df_true = df.loc[df['code']==prot].copy()
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true['chain'] = 'A'
            df_true = df_true.set_index('id')

            unique_data = df_true[~df_true.index.duplicated(keep='first')]
            input_data = inference.standardize_input_df(unique_data, quiet=True)

            pred_df, t_inf = timed_call(
                inference.infer_mutants, 
                model=model, df=input_data, batch_size=32, quiet=True, 
                skip_reverse=should_skip_reverse_dom, mask_strategy=args.mask_strategy, 
                optimize_wt_pass=(args.mask_strategy is None)
            )
            pred_df['id'] = prot + '_' + pred_df['mut_type_renumbered']
            pred_df = pred_df.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_df.columns)))
            res = df_true.join(pred_df.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res)
            res_combined.append(res)

            stats_wt = update_stats(stats_wt, prot, res, 'ddG_ML', 'wt_lora_pred', time_val=t_inf)
            if not should_skip_reverse_dom:
                stats_mt = update_stats(stats_mt, prot, res, 'ddG_ML', 'mt_lora_pred', time_val=t_inf)
                stats_cmb = update_stats(stats_cmb, prot, res, 'ddG_ML', 'combined_pred', time_val=t_inf)

        res_df = pd.concat(res_combined, axis=0)

        out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res_df.to_csv(out_path)

        stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
        os.makedirs(os.path.dirname(stats_base), exist_ok=True)

        stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
        stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')

        if not should_skip_reverse_dom:
            stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
            
            stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

        torch.cuda.empty_cache()


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--split', type=str)
        parser.add_argument('--seed', type=int, required=False, default=42)
        parser.add_argument('--lora_epsilon', type=float, required=False, default=1)
        
        # Precision Argument
        parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['16', '16-mixed', '32', 'bf16-mixed'])

        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--mask_strategy', type=str, choices=['marginal', 'chain'], default=None)
        parser.add_argument('--mask_structure_pos', action='store_true')
        parser.add_argument('--mask_coords_pos', action='store_true')
        parser.add_argument('--mask_coords', action='store_true')
        parser.add_argument('--regenerate_results', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_ctx', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')
        parser.add_argument('--skip_additive', action='store_true')
        parser.add_argument('--skip_reverse', action='store_true')
        parser.add_argument('--skip_reverse_domainome', action='store_true')
        parser.add_argument('--use_dora', action='store_true')
        
        parser.add_argument('--local_path_to_structures', type=str, default='/home/sareeves/software/esm-msr/data/structures')
        parser.add_argument('--hf_token', type=str, default=None)
        parser.add_argument('--remove_spurs_homologs', action='store_true', help='Remove homologous sequences to SPURS training data from the Tsuboyama splits to test generalization to non-homologous sequences')

        args, remaining_argv = parser.parse_known_args()
        current_remaining_argv = list(remaining_argv) 

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
        if args.skip_reverse:
            print('Skipping all reverse mutational passes!')
        if args.mask_structure_pos or args.mask_coords_pos:
            print('Masking one or more inputs!')
        if not args.split:
            print('Warning! Not using any specific split file!')
        if args.split and 'mega' in args.split and not args.remove_spurs_homologs:
            print('Warning: not removing SPURS homologs')

        token = args.hf_token or get_token()

        if token:
            login(token)
            print('Using token')
        else:
            os.environ['INFRA_PROVIDER'] = "1"
            os.chdir(Path(__file__).resolve().parent.parent)
            print(f'Using local model which should be located at {os.path.join(os.getcwd(), "data/weights/esm3_sm_open_v1.pth")}')

        main_(args)