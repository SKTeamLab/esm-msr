import sys
import pkg_resources
from unittest.mock import MagicMock
import importlib.machinery

sys.path.append('/home/sareeves/software/esm-msr/src')

# 1. Patch pkg_resources for the version check
mock_dist = MagicMock()
mock_dist.version = '0.11.0'
mock_dist.project_name = 'torchtext'
pkg_resources.get_distribution = MagicMock(return_value=mock_dist)

# 2. Create a recursive mock for torchtext and all sub-modules
def recursive_mock(*args, **kwargs):
    return MagicMock()

mock_torchtext = MagicMock()
mock_torchtext.data = MagicMock()
mock_torchtext.utils = MagicMock()
mock_torchtext.functional = MagicMock()
mock_torchtext.vocab = MagicMock()

dummy_spec = importlib.machinery.ModuleSpec(name="torchtext", loader=None, origin="mocked")
for m in [mock_torchtext, mock_torchtext.data, mock_torchtext.utils, mock_torchtext.vocab]:
    m.__spec__ = dummy_spec
    m.__path__ = []

# 3. Inject the main modules and the specific failing sub-module into sys.modules
sys.modules["torchtext"] = mock_torchtext
sys.modules["torchtext.data"] = mock_torchtext.data
sys.modules["torchtext.data.functional"] = mock_torchtext.data.functional
sys.modules["torchtext.utils"] = mock_torchtext.utils
sys.modules["torchtext.vocab"] = mock_torchtext.vocab

import importlib.machinery
from contextlib import contextmanager

@contextmanager
def mock_esm_context():
    """
    Temporarily hijacks sys.modules to provide mock objects for 'esm'
    and its submodules. Restores original state upon exit.
    """
    mocked_keys = ["esm", "esm.pretrained", "esm.models.esm3", "esm.utils.constants", "esm.utils.structure.protein_chain"]
    original_modules = {}
    
    for key in mocked_keys:
        if key in sys.modules:
            original_modules[key] = sys.modules[key]

    mock_esm = MagicMock()
    mock_esm.pretrained = MagicMock()
    mock_esm.models.esm3 = MagicMock()
    mock_esm.utils.constants = MagicMock()
    mock_esm.utils.structure.protein_chain = MagicMock()

    dummy_spec = importlib.machinery.ModuleSpec(name="esm", loader=None, origin="mocked")
    for m in [mock_esm, mock_esm.pretrained, mock_esm.models.esm3, mock_esm.utils.constants, mock_esm.utils.structure.protein_chain]:
        m.__spec__ = dummy_spec
        m.__path__ = []

    sys.modules["esm"] = mock_esm
    sys.modules["esm.pretrained"] = mock_esm.pretrained
    sys.modules["esm.models.esm3"] = mock_esm.models.esm3
    sys.modules["esm.utils.constants"] = mock_esm.utils.constants
    sys.modules["esm.utils.structure.protein_chain"] = mock_esm.utils.structure.protein_chain

    try:
        yield
    finally:
        for key in mocked_keys:
            if key in sys.modules:
                del sys.modules[key]
            if key in original_modules:
                sys.modules[key] = original_modules[key]

with mock_esm_context():
    from esm_msr.utils import parse_multimutant_column, sort_mutations_by_position, sum_individual_mutation_scores
    from esm_msr.preprocessing import MegaScaleDatasetPreprocessor

import pandas as pd
import os
import torch
from tqdm import tqdm
import argparse
import time
from pathlib import Path
import warnings
import numpy as np

from omegaconf import OmegaConf

# SPURS Imports
from spurs.utils import seed_everything
from spurs.models.stability.spurs import SPURS
from spurs.models.stability.spurs_multi import SPURSMulti
from spurs.datamodules.datasets.data_utils import Alphabet
from spurs.inference import parse_pdb, get_SPURS_from_hub, get_SPURS_multi_from_hub, parse_pdb_for_mutation

warnings.filterwarnings('ignore') # Consider removing this to catch legitimate deprecations

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = "/home/sareeves/software/SPURS/checkpoints"

def extract_ddg_prediction(mutation: str, ddg_matrix: torch.Tensor, assert_wt_zero: bool = True) -> float:
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    pos = int(mutation[1:-1]) - 1  

    mut_idx = ALPHABET.index(mut_aa)
    
    if assert_wt_zero:
        wt_idx = ALPHABET.index(wt_aa)
        wt_score = ddg_matrix[pos, wt_idx].item()
        # Upgraded to strictly raise instead of silently printing and passing
        assert abs(wt_score) <= 1e-6, f"Expected WT score at index {wt_idx} to be ~0.0, but got {wt_score}"

    return ddg_matrix[pos, mut_idx].item()

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def load_spurs_models(checkpoint_single, checkpoint_multi):
    print("Loading SPURS models...")
    
    if checkpoint_single is not None and (checkpoint_single != 'hf'):
        cfg_single = OmegaConf.load('/home/sareeves/software/SPURS/data/checkpoints/spurs/.hydra/config.yaml')
        del cfg_single['model']['_target_']
        seed_everything(cfg_single['train']['seed'])
        
        model_single = SPURS(cfg_single['model']).to('cuda')
        ckpt_single = torch.load(os.path.join(MODEL_DIR,checkpoint_single), map_location=torch.device('cpu'), weights_only=False)['state_dict']
        ckpt_remove_model_single = {k[6:]:v for k, v in ckpt_single.items() if 'model.' in k}
        model_single.load_state_dict(ckpt_remove_model_single, strict=True)
        model_single.eval()
    else:
        model_single, cfg_single = get_SPURS_from_hub()
        model_single.eval()

    if (checkpoint_multi is not None) and (checkpoint_multi != 'hf'):
        cfg_multi = OmegaConf.load('/home/sareeves/software/SPURS/data/checkpoints/spurs/.hydra/config_multi.yaml')
        del cfg_multi['model']['_target_']
        seed_everything(cfg_multi['train']['seed'])
        
        model_multi = SPURSMulti(cfg_multi['model']).to('cuda')
        ckpt_multi = torch.load(os.path.join(MODEL_DIR,checkpoint_multi), map_location=torch.device('cpu'), weights_only=False)['state_dict']
        ckpt_remove_model_multi = {k[6:]:v for k, v in ckpt_multi.items() if 'model.' in k}
        model_multi.load_state_dict(ckpt_remove_model_multi, strict=True)
        model_multi.eval()
    else:
        model_multi, cfg_multi = get_SPURS_multi_from_hub()
        model_multi.eval()

    return model_single, model_multi, cfg_single, cfg_multi

def main_(args):
    os.makedirs('tmp', exist_ok=True)
    
    model_single, model_multi, cfg_single, cfg_multi = load_spurs_models(args.checkpoint_single, args.checkpoint_multi)

    # ================= External Processing =================
    if not args.skip_external:
        external_test_dataloaders_names = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
        stats_single = pd.DataFrame()
        stats_multi = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(f"Processing External Dataset: {name}")
            
            df_true = pd.read_csv(DATA_DIR / f"{name}_mapped.csv")
            
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
                df_true = sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            df_true['SPURS_single'] = float('nan')
            df_true['SPURS_single_additive'] = float('nan')
            df_true['SPURS_multi'] = float('nan')
            df_true['SPURS_multi_additive'] = float('nan')
            mut_col = 'mut_type'

            time_single_total = 0
            time_multi_total = 0

            for (pdb_path, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                pdb_data = parse_pdb(pdb_path, code, chain, cfg_single) 
                
                # 1. Single Model Inference
                ddg_local, t_single = timed_call(model_single, pdb_data, return_logist=True)
                time_single_total += t_single
                
                # 2. Extract unique single mutations for SPURS_multi_additive
                all_singles = set()
                for _, row in data.iterrows():
                    mut_string = row['wild_type'] + str(int(row['position'])) + row['mutation'] if 'wild_type' in row else row[mut_col]
                    for m in mut_string.split(':'):
                        all_singles.add(m)
                all_singles = list(all_singles)
                
                multi_single_scores = {}
                if len(all_singles) > 0:
                    single_mut_lists = [[m] for m in all_singles]
                    mut_ids_singles, append_tensors_singles = parse_pdb_for_mutation(single_mut_lists)
                    pdb_data_singles = pdb_data.copy()
                    pdb_data_singles['mut_ids'] = mut_ids_singles
                    pdb_data_singles['append_tensors'] = append_tensors_singles.to('cuda')
                    
                    with torch.no_grad():
                        ddg_multi_singles, t_multi_singles = timed_call(model_multi, pdb_data_singles)
                    time_multi_total += t_multi_singles
                    
                    ddg_multi_singles_vals = ddg_multi_singles.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi_singles) else ddg_multi_singles
                    multi_single_scores = dict(zip(all_singles, ddg_multi_singles_vals))

                # 3. Populate Additive Scores
                for i, row in data.iterrows():
                    mut_string = row['wild_type'] + str(int(row['position'])) + row['mutation'] if 'wild_type' in row else row[mut_col]
                    muts = mut_string.split(':')
                    
                    single_add = sum(extract_ddg_prediction(m, ddg_local) for m in muts)
                    multi_add = sum(multi_single_scores[m] for m in muts)
                    
                    df_true.at[i, 'SPURS_single_additive'] = single_add
                    df_true.at[i, 'SPURS_multi_additive'] = multi_add
                    if len(muts) == 1:
                        df_true.at[i, 'SPURS_single'] = single_add

                # 4. Multi Model Inference for target mutations
                data_multi_lengths = data[mut_col].apply(lambda x: len(x.split(':')))
                for num_muts, group in data.groupby(data_multi_lengths):
                    mut_strings = [row['wild_type'] + str(int(row['position'])) + row['mutation'] if 'wild_type' in row else row[mut_col] for _, row in group.iterrows()]
                    mutation_lists = [m.split(':') for m in mut_strings]
                    mut_ids, append_tensors = parse_pdb_for_mutation(mutation_lists)
                    
                    pdb_data_group = pdb_data.copy()
                    pdb_data_group['mut_ids'] = mut_ids
                    pdb_data_group['append_tensors'] = append_tensors.to('cuda') 
                    
                    with torch.no_grad():
                        ddg_multi, t_multi = timed_call(model_multi, pdb_data_group)
                    
                    time_multi_total += t_multi
                    ddg_multi_vals = ddg_multi.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi) else ddg_multi
                    
                    for i, pred in zip(group.index, ddg_multi_vals):
                        df_true.at[i, 'SPURS_multi'] = pred

            df_true['SPURS_single'] *= -1
            df_true['SPURS_single_additive'] *= -1
            df_true['SPURS_multi'] *= -1
            df_true['SPURS_multi_additive'] *= -1

            # Store metrics for Single Model
            stats_single.at[name, 'spearman'] = df_true[['ddG', 'SPURS_single_additive']].corr('spearman').iloc[0,1]
            #stats_single.at[name, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG', top_n=30)
            #stats_single.at[name, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG', threshold=0)
            stats_single.at[name, 'time'] = time_single_total

            # Store metrics for Multi Model
            stats_multi.at[name, 'spearman'] = df_true[['ddG', 'SPURS_multi']].corr('spearman').iloc[0,1]
            #stats_multi.at[name, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG', top_n=30)
            #stats_multi.at[name, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG', threshold=0)
            stats_multi.at[name, 'time'] = time_multi_total

            stats_dir = REPO_ROOT / 'analysis_notebooks' / 'stats/external/SPURS'
            os.makedirs(stats_dir, exist_ok=True)

            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/SPURS'
            os.makedirs(out_dir, exist_ok=True)

            stats_single.to_csv(stats_dir / f'{args.checkpoint_single}_single_stats.csv', na_rep='', float_format='%.6f')
            stats_multi.to_csv(stats_dir / f'{args.checkpoint_multi}_multi_stats.csv', na_rep='', float_format='%.6f')
            
            # Export Consolidated Predictions 
            df_true.to_csv(out_dir / f'{args.checkpoint_single}+{args.checkpoint_multi}.csv')

    # ================= REPEAT WITH SPECIFIC SPLITS =================
    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        ds = MegaScaleDatasetPreprocessor(
            data_file='/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 
            af_model_folder='/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs',
            spurs_override=args.remove_spurs_homologs)

        splits = ds.create_training_splits(str(split_file), -1)

        for scaffold in ['validation', 'testing']:
            results_combined_list = []
            stats_single = pd.DataFrame()
            stats_multi = pd.DataFrame()
            
            scaffold_ = {'validation': 'val', 'testing': 'test'}[scaffold]
            data_scaffold = ds.split_dfs[scaffold_]
            
            for code in tqdm(data_scaffold['code_wt'].unique()):
                df_true = data_scaffold.loc[data_scaffold['code_wt']==code]
                assert len(df_true) > 0
                
                df_true['mut_structure'] = df_true['mut_structure'].fillna('-')
                
                df_true['SPURS_single'] = float('nan')
                df_true['SPURS_single_additive'] = float('nan')
                df_true['SPURS_multi'] = float('nan')
                df_true['SPURS_multi_additive'] = float('nan')

                df_true = sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']
                mut_col = 'mut_type'

                time_single_total = 0
                time_multi_total = 0

                for mut_structure, data in df_true.groupby('mut_structure'):

                    data_ = data.loc[~data['mut_type'].str.contains(':')]
                    pdb_path = data_['pdb_file'].head(1).item() 
                    mutbb_seq = data_['aa_seq'].head(1).item()
                    mutation0 = data_[mut_col].head(1).item()
                    wt, pos, mut = mutation0[0], mutation0[1:-1], mutation0[-1]
                    mutbb_seq = list(mutbb_seq)
                    assert mutbb_seq[int(pos)-1] == mut, f"Expected position {pos} to be {mut}, but got {mutbb_seq[int(pos)-1]}"
                    mutbb_seq[int(pos)-1] = wt
                    mutbb_seq = ''.join(mutbb_seq)
                    
                    if mut_structure != '-':
                        wt, pos, mut = mut_structure[0], mut_structure[1:-1], mut_structure[-1]
                        mutbb_seq = list(mutbb_seq)
                        assert mutbb_seq[int(pos)-1] == mut, f"Expected position {pos} to be {mut}, but got {mutbb_seq[int(pos)-1]}"
                        mutbb_seq[int(pos)-1] = mut
                        mutbb_seq = ''.join(mutbb_seq)

                    chain = 'A' 
                    
                    try:
                        pdb_data = parse_pdb(pdb_path, code, chain, cfg_single, overwrite_seq=mutbb_seq)
                    except Exception as e:
                        print(f"Skipping sequence {code} - Failed parsing PDB: {e}")
                        continue
                    
                    # 1. Single Model 
                    ddg_local, t_single = timed_call(model_single, pdb_data, return_logist=True)
                    time_single_total += t_single
                    
                    # 2. Extract unique single mutations for SPURS_multi_additive
                    all_singles = set()
                    for _, row in data.iterrows():
                        for m in row[mut_col].split(':'):
                            all_singles.add(m)
                    all_singles = list(all_singles)
                    
                    multi_single_scores = {}
                    if len(all_singles) > 0:
                        single_mut_lists = [[m] for m in all_singles]
                        mut_ids_singles, append_tensors_singles = parse_pdb_for_mutation(single_mut_lists)
                        pdb_data_singles = pdb_data.copy()
                        pdb_data_singles['mut_ids'] = mut_ids_singles
                        pdb_data_singles['append_tensors'] = append_tensors_singles.to('cuda')
                        
                        with torch.no_grad():
                            ddg_multi_singles, t_multi_singles = timed_call(model_multi, pdb_data_singles)
                        time_multi_total += t_multi_singles
                        
                        ddg_multi_singles_vals = ddg_multi_singles.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi_singles) else ddg_multi_singles
                        multi_single_scores = dict(zip(all_singles, ddg_multi_singles_vals))

                    # 3. Populate Additive Scores
                    for i, row in data.iterrows():
                        muts = row[mut_col].split(':')
                        single_add = sum(extract_ddg_prediction(m, ddg_local) for m in muts)
                        multi_add = sum(multi_single_scores[m] for m in muts)
                        
                        df_true.loc[i, 'SPURS_single_additive'] = single_add
                        df_true.loc[i, 'SPURS_multi_additive'] = multi_add
                        if len(muts) == 1:
                            df_true.loc[i, 'SPURS_single'] = single_add

                    # 4. Multi Model
                    data_multi_lengths = data[mut_col].apply(lambda x: len(x.split(':')))
                    for num_muts, group in data.groupby(data_multi_lengths):
                        mutation_lists = group[mut_col].apply(lambda x: x.split(':')).tolist()
                        mut_ids, append_tensors = parse_pdb_for_mutation(mutation_lists)
                        
                        pdb_data_group = pdb_data.copy()
                        pdb_data_group['mut_ids'] = mut_ids
                        pdb_data_group['append_tensors'] = append_tensors.to('cuda')

                        with torch.no_grad():
                            ddg_multi, t_multi = timed_call(model_multi, pdb_data_group)
                        
                        time_multi_total += t_multi
                        ddg_multi_vals = ddg_multi.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi) else ddg_multi
                        
                        for i, pred in zip(group.index, ddg_multi_vals):
                            df_true.loc[i, 'SPURS_multi'] = pred

                df_true['SPURS_single'] *= -1
                df_true['SPURS_single_additive'] *= -1
                df_true['SPURS_multi'] *= -1
                df_true['SPURS_multi_additive'] *= -1
                df_true['SPURS_multi_epi'] = df_true['SPURS_multi'] - df_true['SPURS_multi_additive']

                # Single Model Metrics
                stats_single.at[code, 'spearman_all'] = df_true[['ddG_ML', 'SPURS_single_additive']].corr('spearman').iloc[0,1]
                stats_single.at[code, 'spearman_singles'] = df_true.loc[~df_true['mut_type'].str.contains(':'), ['ddG_ML', 'SPURS_single']].corr('spearman').iloc[0,1]
                stats_single.at[code, 'n_singles'] = len(df_true.loc[(~df_true['mut_type'].str.contains(':')) & (~df_true['SPURS_single'].isna())])
                stats_single.at[code, 'spearman_doubles'] = df_true.loc[df_true['mut_type'].str.contains(':'), ['ddG_ML', 'SPURS_single_additive']].corr('spearman').iloc[0,1]
                stats_single.at[code, 'spearman_doubles_epi'] = float('nan') # Additive models inherently have 0 epistasis variance
                #stats_single.at[code, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG_ML', top_n=30)
                #stats_single.at[code, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG_ML', threshold=0)
                stats_single.at[code, 'time'] = time_single_total

                # Multi Model Metrics
                stats_multi.at[code, 'spearman_all'] = df_true[['ddG_ML', 'SPURS_multi']].corr('spearman').iloc[0,1]
                stats_multi.at[code, 'spearman_singles'] = df_true.loc[~df_true['mut_type'].str.contains(':'), ['ddG_ML', 'SPURS_multi']].corr('spearman').iloc[0,1]
                stats_multi.at[code, 'n_singles'] = stats_single.at[code, 'n_singles']
                stats_multi.at[code, 'spearman_doubles'] = df_true.loc[df_true['mut_type'].str.contains(':'), ['ddG_ML', 'SPURS_multi']].corr('spearman').iloc[0,1]
                stats_multi.at[code, 'spearman_doubles_epi'] = df_true[['dddG_ML', 'SPURS_multi_epi']].dropna().corr('spearman').iloc[0,1]
                #stats_multi.at[code, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG_ML', top_n=30)
                #stats_multi.at[code, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG_ML', threshold=0)
                stats_multi.at[code, 'time'] = time_multi_total

                nd = len(df_true.loc[(df_true['mut_type'].str.contains(':')) & (~df_true['SPURS_single_additive'].isna())])
                if nd > 0:
                    stats_single.at[code, 'n_doubles'] = nd
                    stats_multi.at[code, 'n_doubles'] = nd
                
                results_combined_list.append(df_true.reset_index().set_index('uid'))

            results_combined_df = pd.concat(results_combined_list, axis=0)

            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/SPURS'
            stats_dir = REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/SPURS'
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            stats_single.mean(axis=0).to_csv(stats_dir / f'{args.checkpoint_single}_single_avg.csv')
            stats_single.to_csv(stats_dir / f'{args.checkpoint_single}_single_stats.csv', na_rep='', float_format='%.6f')

            stats_multi.mean(axis=0).to_csv(stats_dir / f'{args.checkpoint_multi}_multi_avg.csv')
            stats_multi.to_csv(stats_dir / f'{args.checkpoint_multi}_multi_stats.csv', na_rep='', float_format='%.6f')
            
            results_combined_df.to_csv(out_dir / f'{args.checkpoint_single}+{args.checkpoint_multi}.csv')
                
            torch.cuda.empty_cache()

    # ================= DMS Processing =================
    if not args.skip_dms:
        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']
        stats_single = pd.DataFrame()
        stats_multi = pd.DataFrame()
    
        for prot in prots:
            df_true = pd.read_csv(f'/home/sareeves/software/esm-msr/data/preprocessed/{prot}.csv')
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            
            df_true = df_true.set_index('id')
            mut_col = 'mut_info'
            
            if len(df_true.loc[df_true[mut_col].str.contains(':')]) > 0:
                df_true = sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

            df_true['SPURS_single'] = float('nan')
            df_true['SPURS_single_additive'] = float('nan')
            df_true['SPURS_multi'] = float('nan')
            df_true['SPURS_multi_additive'] = float('nan')
                
            time_single_total = 0
            time_multi_total = 0

            for (pdb_path, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                pdb_data = parse_pdb(pdb_path, code, chain, cfg_single)
                
                # 1. Single Model
                ddg_local, t_single = timed_call(model_single, pdb_data, return_logist=True)
                time_single_total += t_single

                # 2. Extract unique single mutations for SPURS_multi_additive
                all_singles = set()
                for _, row in data.iterrows():
                    for m in row[mut_col].split(':'):
                        all_singles.add(m)
                all_singles = list(all_singles)
                
                multi_single_scores = {}
                if len(all_singles) > 0:
                    single_mut_lists = [[m] for m in all_singles]
                    mut_ids_singles, append_tensors_singles = parse_pdb_for_mutation(single_mut_lists)
                    pdb_data_singles = pdb_data.copy()
                    pdb_data_singles['mut_ids'] = mut_ids_singles
                    pdb_data_singles['append_tensors'] = append_tensors_singles.to('cuda')
                    
                    with torch.no_grad():
                        ddg_multi_singles, t_multi_singles = timed_call(model_multi, pdb_data_singles)
                    time_multi_total += t_multi_singles
                    
                    ddg_multi_singles_vals = ddg_multi_singles.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi_singles) else ddg_multi_singles
                    multi_single_scores = dict(zip(all_singles, ddg_multi_singles_vals))

                # 3. Populate Additive Scores
                for i, row in data.iterrows():
                    muts = row[mut_col].split(':')
                    single_add = sum(extract_ddg_prediction(m, ddg_local) for m in muts)
                    multi_add = sum(multi_single_scores[m] for m in muts)
                    
                    df_true.loc[i, 'SPURS_single_additive'] = single_add
                    df_true.loc[i, 'SPURS_multi_additive'] = multi_add
                    if len(muts) == 1:
                        df_true.loc[i, 'SPURS_single'] = single_add

                # 4. Multi Model
                data_multi_lengths = data[mut_col].apply(lambda x: len(x.split(':')))
                for num_muts, group in data.groupby(data_multi_lengths):
                    mutation_lists = group[mut_col].apply(lambda x: x.split(':')).tolist()
                    mut_ids, append_tensors = parse_pdb_for_mutation(mutation_lists)
                    
                    pdb_data_group = pdb_data.copy()
                    pdb_data_group['mut_ids'] = mut_ids
                    pdb_data_group['append_tensors'] = append_tensors.to('cuda')

                    with torch.no_grad():
                        ddg_multi, t_multi = timed_call(model_multi, pdb_data_group)
                    
                    time_multi_total += t_multi
                    ddg_multi_vals = ddg_multi.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi) else ddg_multi
                    
                    for i, pred in zip(group.index, ddg_multi_vals):
                        df_true.loc[i, 'SPURS_multi'] = pred

            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/SPURS'
            os.makedirs(out_dir, exist_ok=True)
            
            df_true['SPURS_single'] *= -1
            df_true['SPURS_single_additive'] *= -1
            df_true['SPURS_multi'] *= -1
            df_true['SPURS_multi_additive'] *= -1
            df_true['SPURS_multi_epi'] = df_true['SPURS_multi'] - df_true['SPURS_multi_additive']

            df_true.to_csv(out_dir / f'{args.checkpoint_single}+{args.checkpoint_multi}.csv')

            # Metric Calculation
            stat_df = stats_single
            pred_col = 'SPURS_single_additive'
            t_tot = time_single_total
            stat_df.at[prot, 'spearman_all'] = df_true[['ddG_ML', pred_col]].corr('spearman').iloc[0,1]
            stat_df.at[prot, 'spearman_singles'] = df_true.loc[~df_true['mut_info'].str.contains(':'), ['ddG_ML', 'SPURS_single']].corr('spearman').iloc[0,1]
            stat_df.at[prot, 'n_singles'] = len(df_true.loc[(~df_true['mut_info'].str.contains(':')) & (~df_true['SPURS_single'].isna())])
            
            nd = len(df_true.loc[(df_true['mut_info'].str.contains(':')) & (~df_true[pred_col].isna())])
            if nd > 0:
                stat_df.at[prot, 'spearman_doubles'] = df_true.loc[df_true['mut_info'].str.contains(':'), ['ddG_ML', pred_col]].corr('spearman').iloc[0,1]
                stat_df.at[prot, 'n_doubles'] = nd
                stat_df.at[prot, 'spearman_doubles_epi'] = float('nan')
            
            #stat_df.at[prot, 'ndcg@30'] = compute_ndcg_flexible(df_true, pred_col, 'ddG_ML', top_n=30)
            #stat_df.at[prot, 'ndcg>0'] = compute_ndcg_flexible(df_true, pred_col, 'ddG_ML', threshold=0)
            stat_df.at[prot, 'time'] = t_tot

            stat_df = stats_multi
            pred_col = 'SPURS_multi'
            t_tot = time_multi_total
            stat_df.at[prot, 'spearman_all'] = df_true[['ddG_ML', pred_col]].corr('spearman').iloc[0,1]
            stat_df.at[prot, 'spearman_singles'] = df_true.loc[~df_true['mut_info'].str.contains(':'), ['ddG_ML', pred_col]].corr('spearman').iloc[0,1]
            stat_df.at[prot, 'n_singles'] = stats_single.at[prot, 'n_singles']
            
            if nd > 0:
                stat_df.at[prot, 'spearman_doubles'] = df_true.loc[df_true['mut_info'].str.contains(':'), ['ddG_ML', pred_col]].corr('spearman').iloc[0,1]
                stat_df.at[prot, 'n_doubles'] = nd
                if 'dddG_ML' in df_true.columns:
                    stat_df.at[prot, 'spearman_doubles_epi'] = df_true[['dddG_ML', 'SPURS_multi_epi']].dropna().corr('spearman').iloc[0,1]
            
            #stat_df.at[prot, 'ndcg@30'] = compute_ndcg_flexible(df_true, pred_col, 'ddG_ML', top_n=30)
            #stat_df.at[prot, 'ndcg>0'] = compute_ndcg_flexible(df_true, pred_col, 'ddG_ML', threshold=0)
            stat_df.at[prot, 'time'] = t_tot

        stats_dir = REPO_ROOT / 'analysis_notebooks' / 'stats/DMS/SPURS'
        os.makedirs(stats_dir, exist_ok=True)
        stats_single.to_csv(stats_dir / f'{args.checkpoint_single}_single_stats.csv', na_rep='', float_format='%.6f')
        stats_multi.to_csv(stats_dir / f'{args.checkpoint_multi}_multi_stats.csv', na_rep='', float_format='%.6f')
        
        torch.cuda.empty_cache()

    # ================= Domainome Processing =================
    if not args.skip_domainome:
        path = '/home/sareeves/software/esm-msr/data/domainome1/domainome_mapped_2026.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = -df['scaled_fitness']
        df = df.dropna(subset=['pdb_file', 'mut_type'])
        df = df[['code', 'mut_type', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        
        stats_single = pd.DataFrame()
        stats_multi = pd.DataFrame()
        mut_col = 'mut_type'

        all_dfs = []

        for prot in tqdm(df['code'].unique()):
            df_true = df.loc[df['code']==prot].copy()
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true['chain'] = 'A'
            df_true = df_true.set_index('id')
            
            df_true['SPURS_single'] = float('nan')
            df_true['SPURS_single_additive'] = float('nan')
            df_true['SPURS_multi'] = float('nan')
            df_true['SPURS_multi_additive'] = float('nan')
            
            time_single_total = 0
            time_multi_total = 0

            for (pdb_path, code, chain), data in df_true.groupby(['pdb_file', 'code', 'chain']):
                pdb_data = parse_pdb(pdb_path, code, chain, cfg_single)
                
                # 1. Single Model 
                ddg_local, t_single = timed_call(model_single, pdb_data, return_logist=True)
                time_single_total += t_single

                # 2. Extract unique single mutations for SPURS_multi_additive
                all_singles = set()
                for _, row in data.iterrows():
                    for m in row[mut_col].split(':'):
                        all_singles.add(m)
                all_singles = list(all_singles)
                
                multi_single_scores = {}
                if len(all_singles) > 0:
                    single_mut_lists = [[m] for m in all_singles]
                    mut_ids_singles, append_tensors_singles = parse_pdb_for_mutation(single_mut_lists)
                    pdb_data_singles = pdb_data.copy()
                    pdb_data_singles['mut_ids'] = mut_ids_singles
                    pdb_data_singles['append_tensors'] = append_tensors_singles.to('cuda')
                    
                    with torch.no_grad():
                        ddg_multi_singles, t_multi_singles = timed_call(model_multi, pdb_data_singles)
                    time_multi_total += t_multi_singles
                    
                    ddg_multi_singles_vals = ddg_multi_singles.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi_singles) else ddg_multi_singles
                    multi_single_scores = dict(zip(all_singles, ddg_multi_singles_vals))

                # 3. Populate Additive Scores
                for i, row in data.iterrows():
                    muts = row[mut_col].split(':')
                    single_add = sum(extract_ddg_prediction(m, ddg_local) for m in muts)
                    multi_add = sum(multi_single_scores[m] for m in muts)
                    
                    df_true.loc[i, 'SPURS_single_additive'] = single_add
                    df_true.loc[i, 'SPURS_multi_additive'] = multi_add
                    if len(muts) == 1:
                        df_true.loc[i, 'SPURS_single'] = single_add
                
                # 4. Multi Model
                data_multi_lengths = data[mut_col].apply(lambda x: len(x.split(':')))
                for num_muts, group in data.groupby(data_multi_lengths):
                    mutation_lists = group[mut_col].apply(lambda x: x.split(':')).tolist()
                    mut_ids, append_tensors = parse_pdb_for_mutation(mutation_lists)
                    
                    pdb_data_group = pdb_data.copy()
                    pdb_data_group['mut_ids'] = mut_ids
                    pdb_data_group['append_tensors'] = append_tensors.to('cuda')
                    
                    with torch.no_grad():
                        ddg_multi, t_multi = timed_call(model_multi, pdb_data_group)
                    
                    time_multi_total += t_multi
                    ddg_multi_vals = ddg_multi.view(-1).cpu().tolist() if torch.is_tensor(ddg_multi) else ddg_multi
                    
                    for i, pred in zip(group.index, ddg_multi_vals):
                        df_true.loc[i, 'SPURS_multi'] = pred

            df_true['SPURS_single'] *= -1
            df_true['SPURS_single_additive'] *= -1
            df_true['SPURS_multi'] *= -1
            df_true['SPURS_multi_additive'] *= -1

            # Record Single Metrics
            stats_single.at[prot, 'spearman'] = df_true[['ddG_ML', 'SPURS_single_additive']].corr('spearman').iloc[0,1]
            #stats_single.at[prot, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG_ML', top_n=30)
            #stats_single.at[prot, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_single_additive', 'ddG_ML', threshold=0)
            stats_single.at[prot, 'time'] = time_single_total

            # Record Multi Metrics
            stats_multi.at[prot, 'spearman'] = df_true[['ddG_ML', 'SPURS_multi']].corr('spearman').iloc[0,1]
            #stats_multi.at[prot, 'ndcg@30'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG_ML', top_n=30)
            #stats_multi.at[prot, 'ndcg>0'] = compute_ndcg_flexible(df_true, 'SPURS_multi', 'ddG_ML', threshold=0)
            stats_multi.at[prot, 'time'] = time_multi_total

            all_dfs.append(df_true)

        out_dir = REPO_ROOT / 'analysis_notebooks' / 'predictions/domainome/SPURS'
        os.makedirs(out_dir, exist_ok=True)
        
        df_out = pd.concat(all_dfs, axis=0)
        df_out.to_csv(out_dir / f'{args.checkpoint_single}+{args.checkpoint_multi}.csv')

        stats_dir = REPO_ROOT / 'analysis_notebooks' / 'stats/domainome/SPURS'
        os.makedirs(stats_dir, exist_ok=True)

        stats_single.mean(axis=0).to_csv(stats_dir / f'{args.checkpoint_single}_single_avg.csv')
        stats_single.to_csv(stats_dir / f'{args.checkpoint_single}_single_stats.csv', na_rep='', float_format='%.6f')
        
        stats_multi.mean(axis=0).to_csv(stats_dir / f'{args.checkpoint_multi}_multi_avg.csv')
        stats_multi.to_csv(stats_dir / f'{args.checkpoint_multi}_multi_stats.csv', na_rep='', float_format='%.6f')

        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_single', type=str, default='hf')
    parser.add_argument('--checkpoint_multi', type=str, default='hf')
    parser.add_argument('--split', type=str)
    parser.add_argument('--loc', type=str, default='inference_scripts')
    parser.add_argument('--skip_external', action='store_true')
    parser.add_argument('--skip_tsuboyama', action='store_true')
    parser.add_argument('--skip_ctx', action='store_true')
    parser.add_argument('--skip_dms', action='store_true')
    parser.add_argument('--skip_functional', action='store_true')
    parser.add_argument('--skip_domainome', action='store_true')
    parser.add_argument('--remove_spurs_homologs', action='store_true', help='Remove homologous sequences to SPURS training data from the Tsuboyama splits to test generalization to non-homologous sequences')

    args, remaining_argv = parser.parse_known_args()

    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    if args.split and 'mega' in args.split and not args.remove_spurs_homologs:
        print('Warning: not removing SPURS homologs')

    main_(args)