import pandas as pd
import os
import torch
from tqdm import tqdm
import argparse
import torch
import time

from huggingface_hub import login

from esm.pretrained import ESM3_sm_open_v0

from esm_msr import utils, models

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "logs"

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def main_(args):

    CHECKPOINT = args.checkpoint

    print('\n\n\n\n\n')
    print(CHECKPOINT)
    print('\n\n\n\n\n')

    os.makedirs('tmp', exist_ok=True)

    # Model
    base_model = ESM3_sm_open_v0()
    peft_model = utils.add_lora_to_esm3(
        base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_mode='all' if 'qkv' in CHECKPOINT else 'expanded',
        seed=42
    )

    if 'll' in CHECKPOINT:
        print('Using log_likelihoods in inference')
    else:
        print('Not using log_likelihoods in inference')

    model = models.ESM3LoRAModel(
        peft_model,
        freeze_lora=True,
        inference_mode=True,
        log_likelihood='ll' in CHECKPOINT,
        shared_scale=0,
        shared_bias=0,       
        ).to('cuda:0')
    
    if args.checkpoint:
        model = utils.load_ckpt_weights(model, MODEL_DIR / args.checkpoint, device='cuda:0')
    else:
        # zero shot
        model.calibration_head_shared = models.CalibrationHead(init_scale=None, init_bias=0, requires_grad=False)
    model.eval()

    utils.print_trainable_parameters(model)   

    if not args.skip_external:
        external_test_dataloaders_names = ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
        stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()
        stats_direct = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(name)

            res_parallel = []
            res_masked = []
            res_direct = []

            time_parallel = 0
            time_masked = 0    
            time_direct = 0

            df_true = pd.read_csv(DATA_DIR / f"{name}_mapped_new.csv")
            if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369']:
                df_true = df_true.reset_index()
                df_true['position_pdb'] = df_true['position']
                df_true['position'] = df_true['seq_pos']
                df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')

            else:
                df_true = df_true.reset_index()
                df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            for (pdb, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):

                pred_combined_parallel, t_parallel = timed_call(model.infer_mutants, df=data, strategy='parallel', quiet=True)
                pred_combined_parallel['id'] = code + chain + '_' + pred_combined_parallel['mut_type']
                pred_combined_masked, t_masked = timed_call(model.infer_mutants, df=data, strategy='masked', quiet=True)
                pred_combined_masked['id'] = code + chain + '_' + pred_combined_masked['mut_type']
                pred_combined_direct, t_direct = timed_call(model.infer_mutants, df=data, strategy='direct', quiet=True)
                pred_combined_direct['id'] = code + chain + '_' + pred_combined_masked['mut_type']
                
                pred_combined_parallel = pred_combined_parallel.set_index('id')
                pred_combined_masked = pred_combined_masked.set_index('id')
                pred_combined_direct = pred_combined_direct.set_index('id')

                overlap_cols = list(set(data.columns).intersection(set(pred_combined_parallel.columns)))
                res_partial_parallel = data.join(pred_combined_parallel.drop(overlap_cols, axis=1))
                res_partial_masked = data.join(pred_combined_masked.drop(overlap_cols, axis=1))
                res_partial_direct = data.join(pred_combined_direct.drop(overlap_cols, axis=1))

                res_parallel.append(res_partial_parallel)
                res_masked.append(res_partial_masked)
                res_direct.append(res_partial_direct)

                time_parallel += t_parallel
                time_masked += t_masked
                time_direct += t_direct

            res_parallel = pd.concat(res_parallel)
            res_masked = pd.concat(res_masked)
            res_direct = pd.concat(res_direct)

            os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}'), exist_ok=True)
            res_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
            res_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
            res_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

            stats_parallel.at[name, 'spearman'] = res_parallel[['ddG', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_parallel.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG', top_n=30)
            stats_parallel.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG', threshold=0)

            stats_masked.at[name, 'spearman'] = res_masked[['ddG', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_masked.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG', top_n=30)
            stats_masked.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG', threshold=0)

            stats_direct.at[name, 'spearman'] = res_direct[['ddG', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_direct.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG', top_n=30)
            stats_direct.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG', threshold=0)

            stats_parallel.at[name, 'time'] = time_parallel
            stats_masked.at[name, 'time'] = time_masked
            stats_direct.at[name, 'time'] = time_direct

            if 'ptmul' not in name:
                assert len(df_true) == len(res_parallel)
                assert len(df_true) == len(res_masked)
                assert len(df_true) == len(res_direct)
            else:
                print(len(df_true), len(res_parallel))

            os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT}'), exist_ok=True)
            stats_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
            stats_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
            stats_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

    ############## REPEAT WITH SPECIFIC SPLITS ################

    if args.split is not None and not args.skip_tsuboyama:
        splits = pd.read_csv(args.split, index_col=0)
        split_name = args.split.split('/')[-1].split('splits_')[1].split('.csv')[0]

        for scaffold in ['validation', 'testing']:
            results_parallel = []
            results_parallel_ctx = []
            results_masked = []
            results_masked_ctx = []
            results_direct = []

            stats_parallel = pd.DataFrame()
            stats_parallel_ctx = pd.DataFrame()
            stats_masked = pd.DataFrame()
            stats_masked_ctx = pd.DataFrame()     
            stats_direct = pd.DataFrame()

            scaffold_ = scaffold.replace('testing', 'test')
            test_list = eval(splits.loc['stability', scaffold])

            tsu = pd.read_csv(DATA_DIR / f"tsuboyama_all_subs_final.csv")
            tsu = tsu.loc[~tsu['mut_type'].apply(utils.is_fake_mutation)]
            tsu = tsu.loc[~tsu['uid'].apply(utils.is_improper_mutation)]
            
            tsu = utils.parse_multimutant_column(tsu, 'mut_type')
            tsu['id'] = tsu['code'] + '_' + tsu['mut_type']
            tsu = tsu.sort_values('id')

            for code in tqdm(test_list):

                df_true = tsu.loc[tsu['code'].str.contains(code, regex=False)]
                if not code.startswith('v2_'):
                    df_true = df_true.loc[~df_true['code'].str.startswith('v2_')]
                pdb = df_true['pdb_file'].head(1).item()
                df_true['mut_structure'] = df_true['mut_structure'].fillna('-')

                res_parallel = []
                res_parallel_ctx = []
                res_masked = []
                res_masked_ctx = []
                res_direct = []

                time_parallel = 0
                time_parallel_ctx = 0
                time_masked = 0
                time_masked_ctx = 0    
                time_direct = 0   

                for mut_structure, data in df_true.groupby('mut_structure'):
                    
                    if mut_structure != '-':
                        backbone_mutation = mut_structure
                    else:
                        backbone_mutation = None

                    data = data.set_index('id')
                    data = utils.sum_individual_mutation_scores(data, 'ddG_ML', new_score_column='ddG_additive_ML')
                    data['dddG_ML'] = data['ddG_ML'] - data['ddG_additive_ML']

                    pred_combined_parallel, t_parallel = timed_call(model.infer_mutants, df=data, strategy='parallel', backbone_mutation=backbone_mutation, quiet=True)
                    pred_combined_parallel['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_combined_parallel['mut_type']
                    pred_combined_parallel = pred_combined_parallel.set_index('id')

                    if not args.skip_ctx:
                        pred_combined_parallel_ctx, t_parallel_ctx = timed_call(model.infer_mutants, df=data, strategy='parallel', backbone_mutation=backbone_mutation, quiet=True, use_modeled_context_structs=True, mut_structs_root='/home/sareeves/PSLMs/data/lora/FINAL_results')
                        pred_combined_parallel_ctx['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_combined_parallel_ctx['mut_type']
                        pred_combined_parallel_ctx = pred_combined_parallel_ctx.set_index('id')

                    pred_combined_masked, t_masked = timed_call(model.infer_mutants, df=data, strategy='masked', backbone_mutation=backbone_mutation, quiet=True)
                    pred_combined_masked['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_combined_masked['mut_type']
                    pred_combined_masked = pred_combined_masked.set_index('id')

                    if not args.skip_ctx:
                        pred_combined_masked_ctx, t_masked_ctx = timed_call(model.infer_mutants, df=data, strategy='masked', backbone_mutation=backbone_mutation, quiet=True, use_modeled_context_structs=True, mut_structs_root='/home/sareeves/PSLMs/data/lora/FINAL_results')
                        pred_combined_masked_ctx['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_combined_masked_ctx['mut_type']
                        pred_combined_masked_ctx = pred_combined_masked_ctx.set_index('id')

                    pred_combined_direct, t_direct = timed_call(model.infer_mutants, df=data, strategy='direct', backbone_mutation=backbone_mutation, quiet=True)
                    pred_combined_direct['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_combined_masked['mut_type']
                    pred_combined_direct = pred_combined_direct.set_index('id')

                    overlap_cols = list(set(data.columns).intersection(set(pred_combined_parallel.columns)))
                    res_partial_parallel = data.join(pred_combined_parallel.drop(overlap_cols, axis=1))
                    res_partial_masked = data.join(pred_combined_masked.drop(overlap_cols, axis=1))
                    res_partial_direct = data.join(pred_combined_direct.drop(overlap_cols, axis=1))

                    res_parallel.append(res_partial_parallel)
                    res_masked.append(res_partial_masked)
                    res_direct.append(res_partial_direct)

                    time_parallel += t_parallel
                    time_masked += t_masked
                    time_direct += t_direct

                    if not args.skip_ctx:
                        res_partial_parallel_ctx = data.join(pred_combined_parallel_ctx.drop(overlap_cols, axis=1))
                        res_partial_masked_ctx = data.join(pred_combined_masked_ctx.drop(overlap_cols, axis=1))
                        res_parallel_ctx.append(res_partial_parallel_ctx)
                        res_masked_ctx.append(res_partial_masked_ctx)
                        time_parallel_ctx += t_parallel_ctx
                        time_masked_ctx += t_masked_ctx

                res_parallel = pd.concat(res_parallel)
                res_direct = pd.concat(res_direct)
                res_masked = pd.concat(res_masked)

                if not args.skip_ctx:
                    res_parallel_ctx = pd.concat(res_parallel_ctx)
                    res_masked_ctx = pd.concat(res_masked_ctx)
                    
                    stats_parallel_ctx.at[code, 'spearman'] = res_parallel_ctx[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
                    try:
                        stats_parallel_ctx.at[code, 'spearman_epi'] = res_parallel_ctx[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
                    except:
                        stats_parallel_ctx.at[code, 'spearman_epi'] = float('nan')
                    stats_parallel_ctx.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel_ctx, 'ddg_pred', 'ddG_ML', top_n=30)
                    stats_parallel_ctx.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel_ctx, 'ddg_pred', 'ddG_ML', threshold=0)

                    stats_masked_ctx.at[code, 'spearman'] = res_masked_ctx[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
                    try:
                        stats_masked_ctx.at[code, 'spearman_epi'] = res_masked_ctx[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
                    except:
                        stats_masked_ctx.at[code, 'spearman_epi'] = float('nan')
                    stats_masked_ctx.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked_ctx, 'ddg_pred', 'ddG_ML', top_n=30)
                    stats_masked_ctx.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked_ctx, 'ddg_pred', 'ddG_ML', threshold=0)

                    stats_parallel_ctx.at[code, 'time'] = time_parallel_ctx
                    stats_masked_ctx.at[code, 'time'] = time_masked_ctx

                    results_parallel_ctx.append(res_parallel_ctx.reset_index(drop=True).set_index('uid'))
                    results_masked_ctx.append(res_masked_ctx.reset_index(drop=True).set_index('uid'))

                    assert len(df_true) == len(res_parallel_ctx)
                    assert len(df_true) == len(res_masked_ctx)

                stats_parallel.at[code, 'spearman'] = res_parallel[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
                try:
                    stats_parallel.at[code, 'spearman_epi'] = res_parallel[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
                except Exception as e:
                    #print(e)
                    stats_parallel.at[code, 'spearman_epi'] = float('nan')
                stats_parallel.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', top_n=30)
                stats_parallel.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', threshold=0)

                stats_masked.at[code, 'spearman'] = res_masked[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
                try:
                    stats_masked.at[code, 'spearman_epi'] = res_masked[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
                except:
                    stats_masked.at[code, 'spearman_epi'] = float('nan')
                stats_masked.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', top_n=30)
                stats_masked.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', threshold=0)

                stats_direct.at[code, 'spearman'] = res_direct[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
                try:
                    stats_direct.at[code, 'spearman_epi'] = res_direct[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
                except:
                    stats_direct.at[code, 'spearman_epi'] = float('nan')
                stats_direct.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', top_n=30)
                stats_direct.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', threshold=0)

                stats_parallel.at[code, 'time'] = time_parallel
                stats_masked.at[code, 'time'] = time_masked
                stats_direct.at[code, 'time'] = time_direct

                assert len(df_true) == len(res_parallel)
                assert len(df_true) == len(res_masked)
                assert len(df_true) == len(res_direct)

                results_parallel.append(res_parallel.reset_index(drop=True).set_index('uid'))
                results_masked.append(res_masked.reset_index(drop=True).set_index('uid'))
                results_direct.append(res_direct.reset_index(drop=True).set_index('uid'))

            results_parallel = pd.concat(results_parallel, axis=0)
            results_masked = pd.concat(results_masked, axis=0)
            results_direct = pd.concat(results_direct, axis=0)

            print(stats_parallel.mean(axis=0))
            print(stats_masked.mean(axis=0))

            os.makedirs(os.path.dirname((REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}')), exist_ok=True)
            os.makedirs(os.path.dirname((REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}')), exist_ok=True)

            stats_parallel.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel_avg.csv')
            stats_masked.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked_avg.csv')
            stats_direct.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct_avg.csv')

            results_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
            results_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
            results_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

            stats_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
            stats_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
            stats_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

            if not args.skip_ctx:
                results_parallel_ctx = pd.concat(results_parallel_ctx, axis=0)
                results_masked_ctx = pd.concat(results_masked_ctx, axis=0)

                stats_parallel_ctx.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_parallel_avg.csv')
                stats_masked_ctx.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_masked_avg.csv')
                results_parallel_ctx.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_parallel.csv')
                results_masked_ctx.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_masked.csv')
                stats_parallel_ctx.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_parallel.csv')
                stats_masked_ctx.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_mut_ctx_masked.csv')

            torch.cuda.empty_cache()

    ######################################

    if not args.skip_dms:

        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain'] #, 'A4_HUMAN_Seuma_2022'] # 'GB1_Wu_2016_binding_domain','A4_HUMAN_Seuma_2022', 
        #prots = ['GB1_Wu_2016_binding_domain']
        stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()
        stats_direct = pd.DataFrame()

        results_parallel = []
        results_masked = []
        results_direct = []
    
        for mem_size, prot in zip([2,2,2,2,2,2,1], prots): #4,4,2,2,4,4,2

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/preprocessed/{prot}.csv')
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')
            has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
            if has_doubles:
                df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

            prot_name = '_'.join(prot.split('_')[:2])
            if prot_name == 'GB1_Wu':
                prot_name = 'GB1'

            pred_combined_parallel, t_parallel = timed_call(model.infer_mutants, df=df_true, strategy='parallel', mem_scale=mem_size, quiet=True)
            pred_combined_parallel['id'] = prot_name + '_' + pred_combined_parallel['mut_type']
            pred_combined_masked, t_masked = timed_call(model.infer_mutants, df=df_true, strategy='masked', mem_scale=mem_size, quiet=True)
            pred_combined_masked['id'] = prot_name + '_' + pred_combined_masked['mut_type']
            pred_combined_direct, t_direct = timed_call(model.infer_mutants, df=df_true, strategy='direct', mem_scale=mem_size, quiet=True)
            pred_combined_direct['id'] = prot_name + '_' + pred_combined_masked['mut_type']
        
            pred_combined_parallel = pred_combined_parallel.set_index('id')
            pred_combined_masked = pred_combined_masked.set_index('id')
            pred_combined_direct = pred_combined_direct.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_combined_parallel.columns)))

            res_parallel = df_true.join(pred_combined_parallel.drop(overlap_cols, axis=1))
            res_masked = df_true.join(pred_combined_masked.drop(overlap_cols, axis=1))
            res_direct = df_true.join(pred_combined_direct.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res_parallel)
            assert len(df_true) == len(res_masked)
            assert len(df_true) == len(res_direct)

            os.makedirs(os.path.dirname((REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}')), exist_ok=True)
            res_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
            res_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
            res_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

            results_parallel.append(res_parallel)
            results_masked.append(res_masked)
            results_direct.append(res_direct)

            stats_parallel.at[prot, 'spearman'] = res_parallel[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            try:
                stats_parallel.at[prot, 'spearman_epi'] = res_parallel[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_parallel.at[prot, 'spearman_epi'] = float('nan')
            stats_parallel.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_parallel.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            try:
                stats_masked.at[prot, 'spearman_epi'] = res_masked[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_masked.at[prot, 'spearman_epi'] = float('nan')
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_direct.at[prot, 'spearman'] = res_direct[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            try:
                stats_direct.at[prot, 'spearman_epi'] = res_direct[['dddG_ML', 'dddg_pred']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_direct.at[prot, 'spearman_epi'] = float('nan')
            stats_direct.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_direct.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_parallel.at[prot, 'time'] = t_parallel
            stats_masked.at[prot, 'time'] = t_masked
            stats_direct.at[prot, 'time'] = t_direct

        print(stats_parallel)
        print(stats_masked)
        print(stats_direct)

        os.makedirs(os.path.dirname((REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT}')), exist_ok=True)

        results_parallel = pd.concat(results_parallel, axis=0)
        results_masked = pd.concat(results_masked, axis=0)
        results_direct = pd.concat(results_direct, axis=0)

        stats_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
        stats_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
        stats_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

        torch.cuda.empty_cache()

    #########################################

    if not args.skip_domainome:

        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/domainome1/domainome_mapped_new.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset='pdb_file')
        df['mut_type'] = df['mut_info']
        df = df[['code', 'mut_seq', 'mut_type', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        
        results_parallel = []
        results_masked = []
        results_direct = []
        
        stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()
        stats_direct = pd.DataFrame()

        for prot in tqdm(df['code'].unique()):

            df_true = df.loc[df['code']==prot]
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true['chain'] = 'A'
            df_true = df_true.set_index('id')

            pred_singles_parallel, t_parallel = timed_call(model.infer_mutants, df=df_true, strategy='parallel') #, quiet=True)
            pred_singles_parallel['id'] = prot + '_' + pred_singles_parallel['mut_type']
            pred_singles_masked, t_masked = timed_call(model.infer_mutants, df=df_true, strategy='masked') #, quiet=True)
            pred_singles_masked['id'] = prot + '_' + pred_singles_masked['mut_type']

            pred_singles_direct = pred_singles_masked
            t_direct = t_masked

            overlap_cols = list(set(df_true.columns).intersection(set(pred_singles_parallel.columns)))

            res_parallel = df_true.join(pred_singles_parallel.set_index('id').drop(overlap_cols, axis=1))
            res_masked = df_true.join(pred_singles_masked.set_index('id').drop(overlap_cols, axis=1))
            res_direct = df_true.join(pred_singles_direct.set_index('id').drop(overlap_cols, axis=1))

            assert len(df_true) == len(res_parallel)
            assert len(df_true) == len(res_masked)
            assert len(df_true) == len(res_direct)

            stats_parallel.at[prot, 'spearman'] = res_parallel[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_parallel.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_parallel.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_direct.at[prot, 'spearman'] = res_direct[['ddG_ML', 'ddg_pred']].corr('spearman').iloc[0,1]
            stats_direct.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', top_n=30)
            stats_direct.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_direct, 'ddg_pred', 'ddG_ML', threshold=0)

            stats_parallel.at[prot, 'time'] = t_parallel
            stats_masked.at[prot, 'time'] = t_masked
            stats_direct.at[prot, 'time'] = t_direct

            results_parallel.append(res_parallel)
            results_masked.append(res_masked)
            results_direct.append(res_direct)

        results_parallel_out = pd.concat(results_parallel, axis=0)
        results_masked_out = pd.concat(results_masked, axis=0)
        results_direct_out = pd.concat(results_direct, axis=0)

        print(stats_parallel.mean(axis=0))
        print(stats_masked.mean(axis=0))
        print(stats_direct.mean(axis=0))

        os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT}'), exist_ok=True)
        os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}'), exist_ok=True)

        stats_parallel.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel_avg.csv')
        stats_masked.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked_avg.csv')
        stats_direct.mean(axis=0).to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct_avg.csv')

        results_parallel_out.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
        results_masked_out.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
        results_direct_out.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

        stats_parallel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_parallel.csv')
        stats_masked.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_masked.csv')
        stats_direct.to_csv(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg_direct.csv')

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

            pred_combined, t = timed_call(model.infer_mutants, df=df_true, strategy=strategy, mem_scale=mem_size, quiet=True)
            pred_combined['id'] = prot + '_' + pred_combined['mut_type']
        
            pred_combined = pred_combined.set_index('id')
            overlap_cols = list(set(df_true.columns).intersection(set(pred_combined.columns)))

            res = df_true.join(pred_combined.drop(overlap_cols, axis=1))
            assert len(df_true) == len(res)
            print(res[['ddG_dir', 'ddg_pred']].corr('spearman').iloc[0,1])

            os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}'), exist_ok=True)
            res.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg.csv')
            torch.cuda.empty_cache()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--additive_condition', type=str, required=True, choices=['mask', 'wt', 'mut'])
        parser.add_argument('--split', type=str)
        parser.add_argument('--lora_alpha', type=int, required=True)
        parser.add_argument('--lora_rank', type=int, required=True)
        parser.add_argument('--loc', type=str, default='inference_scripts')
        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--mask_sequence_pos', type=bool, default=True)
        parser.add_argument('--mask_structure_pos', action='store_true')
        parser.add_argument('--mask_coords_pos', action='store_true')
        parser.add_argument('--mask_coords', action='store_true')
        parser.add_argument('--regenerate_cache', action='store_true')
        parser.add_argument('--regenerate_results', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_ctx', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')
        parser.add_argument('--hf_token', type=str, default=None)

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
        if not args.regenerate_cache:
            print('WARNING: Using an existing cache!')
        if args.mask_structure_pos or args.mask_coords_pos:
            print('Masking one or more inputs!')
        if not args.split:
            print('Warning! Not using any specific split file!')

        if args.hf_token:
            login(args.hf_token)
        else:
            os.environ['INFRA_PROVIDER'] = "1"
            os.chdir(Path(__file__).resolve().parent.parent)

        main_(args)