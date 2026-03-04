import sys
sys.path.append('/home/sareeves/software/ThermoMPNN-D/')

import os
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from copy import deepcopy
import pickle
import argparse

import torch
from torch.utils.data import DataLoader

from esm_msr import utils

from omegaconf import OmegaConf
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.train_thermompnn import parse_cfg
from thermompnn.protein_mpnn_utils import alt_parse_PDB, parse_PDB

from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, ddgBenchDatasetv2, tied_featurize_mut
from thermompnn.inference.inference_utils import get_metrics_full
from thermompnn.inference.v2_inference import zero_shot_convert


from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_multi_mutation_additive_prediction(model, model_path, cfg, prefix, max_mutations=10):
    """
    Run additive predictions for multiple mutations (up to max_mutations).
    
    Args:
        model: The prediction model
        model_path: Path to the model
        cfg: Configuration object
        prefix: Directory prefix
        max_mutations: Maximum number of mutations to handle (default: 10)
    """
    # Create output directory
    os.makedirs(REPO_ROOT / 'analysis_notebooks' / f'predictions/PTMUL/thermompnnd', exist_ok=True)
    
    # Set base configuration
    cfg.training.batch_size = 1
    cfg.data.dataset = 'ptmul'
    cfg.data.splits = ['alt']
    cfg.data.mut_types = ['double', 'higher']
    keep = True
    
    # First, load the full dataset to get mutation counts
    full_dataset_df = pd.read_csv(os.path.join(prefix, 'preprocessed/ptmul_mapped_new.csv'))
    
    # Calculate the number of mutations for each row
    full_dataset_df['mutation_count'] = full_dataset_df['mut_info'].apply(lambda x: len(x.split(':')))
    full_dataset_df = full_dataset_df.loc[full_dataset_df['mutation_count']<=max_mutations]

    full_dataset_df['WT_name'] = full_dataset_df['code'] + full_dataset_df['chain'] 
    
    # Create a list to store all individual results
    all_individual_results = {}  # Use a dictionary to map mutation counts to results
    
    # Process each mutation position
    for position in range(1, max_mutations + 1):
        print(f"Processing mutation position {position}/{max_mutations}")
        
        # Filter the dataset for entries with at least 'position' mutations
        position_df = full_dataset_df[full_dataset_df['mutation_count'] >= position]
        
        # Skip if no mutations at this position
        if len(position_df) == 0:
            print(f"No proteins with {position} or more mutations found, skipping position {position}")
            continue
        
        # Save the filtered dataset to a temporary file
        temp_csv_path = os.path.join(prefix, f'preprocessed/ptmuld_mapped_pos{position}_temp.csv')
        position_df.to_csv(temp_csv_path, index=False)
        
        # Create config for this position
        cfg_i = deepcopy(cfg)
        cfg_i.data.pick = position - 1  # Adjust for 0-indexing
        
        try:
            # Create dataset for this position using the filtered CSV
            ptmul_dataset = ddgBenchDatasetv3(
                cfg_i, 
                pdb_dir='PTMUL', 
                csv_fname=temp_csv_path, 
                multi=True, 
                invert=True
            )
            
            # Run prediction for this position
            results_i = run_prediction_batched(
                name=f'ThermoMPNN_additive_pos{position}',
                model=model,
                dataset_name=f'ptmul-{position}',
                results=[],
                dataset=ptmul_dataset,
                keep=keep,
                zero_shot=False,
                cfg=cfg_i
            )
            
            # Store the results for this position
            all_individual_results[position] = results_i
            
            # Cleanup
            del ptmul_dataset
        finally:
            # Remove the temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
    
    # Now combine results to get additive predictions
    if not all_individual_results:
        print("No mutation data found!")
        return
    
    # Get all unique mutation counts from the results
    mutation_counts = set(full_dataset_df['mutation_count'])
    
    # Process each mutation count separately
    final_results = []
    
    for mut_count in sorted(mutation_counts):
        if mut_count < 2:
            continue  # Skip single mutations
            
        print(f"Processing additive predictions for proteins with {mut_count} mutations")
        
        # Get proteins with this mutation count
        count_df = full_dataset_df[full_dataset_df['mutation_count'] == mut_count]
        
        # Extract unique identifiers to filter results
        identifiers = set(count_df['WT_name'] + '_' + count_df['mut_info'])
        print(sorted(list(identifiers)))
        
        # Start with the first position's results for these proteins
        if 1 not in all_individual_results:
            print(f"Missing results for position 1, skipping count {mut_count}")
            continue
            
        base_results = all_individual_results[1].copy()
        
        # Filter to only proteins with this mutation count
        base_results['temp_uid'] = base_results['WT_name'] + '_' + base_results['mut_type'].str.replace(';', ':')
        count_results = base_results[base_results['uid'].isin(identifiers)].copy()
        
        if len(count_results) == 0:
            print(f"No matching proteins found for count {mut_count}, skipping")
            continue
            
        # Initialize the additive prediction with the first position
        count_results['ddG_pred_additive'] = 0
        count_results = count_results.drop('ddG_pred', axis=1)
        
        # Add predictions from other positions up to the mutation count
        for pos in range(1, mut_count + 1):
            if pos not in all_individual_results:
                print(f"Missing results for position {pos}, skipping")
                continue
                
            # Get results for this position
            pos_results = all_individual_results[pos].copy()
            #pos_results['temp_uid'] = pos_results['WT_name'] + '_' + pos_results['mut_type'].str.replace(';', ':')
            
            # Filter to matching proteins
            pos_results = pos_results[pos_results['uid'].isin(identifiers)]
            
            # Create a mapping of identifiers to predictions
            pred_map = dict(zip(pos_results['uid'], pos_results['ddG_pred']))
            
            # Add this position's predictions to the additive total
            count_results['ddG_pred_additive'] += count_results['uid'].map(pred_map).fillna(0)
            
            # For debugging: store individual position predictions
            count_results[f'ddG_pred_pos{pos}'] = count_results['uid'].map(pred_map).fillna(0)
        
        # Remove the temporary UID column
        count_results = count_results.drop('uid', axis=1).rename({'temp_uid': 'uid'}, axis=1)
        
        # Append to the final results
        final_results.append(count_results)
    
    # Combine all results
    if not final_results:
        print("No results to combine!")
        return
        
    combined_results = pd.concat(final_results, ignore_index=True)
    
    # Post-process results
    combined_results['code'] = combined_results['WT_name'].str[:-1]
    combined_results['chain'] = combined_results['WT_name'].str[-1]
    combined_results = combined_results.set_index('uid').sort_index()
    
    return combined_results
    

def run_prediction_batched(name, model, dataset_name, dataset, results, keep=True, zero_shot=False, cfg=None):
    """Standard inference for CSV/PDB based dataset in batched models"""

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    max_batches = None
    metrics = {
        "ddG": get_metrics_full(),
    }
    for m in metrics['ddG'].values():
        m = m.to(device)
    
    model = model.eval()
    model = model.cuda()
    
    print('\nTesting Model %s on dataset %s\n' % (name, dataset_name))
    preds, ddgs = [], []

    loader = DataLoader(dataset, collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.get('side_chains', False)), 
                        shuffle=False, num_workers=cfg.training.get('num_workers', 8), batch_size=cfg.training.get('batch_size', 256))

    batches = []
    for i, batch in enumerate(tqdm(loader)):

        if batch is None:
            continue
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
        X = X.to(device)
        S = S.to(device)
        mask = mask.to(device)
        lengths = torch.Tensor(lengths).to(device)
        chain_M = chain_M.to(device)
        chain_encoding_all = chain_encoding_all.to(device)
        residue_idx = residue_idx.to(device)
        mut_positions = mut_positions.to(device)
        mut_wildtype_AAs = mut_wildtype_AAs.to(device)
        mut_mutant_AAs = mut_mutant_AAs.to(device)
        mut_ddGs = mut_ddGs.to(device)
        atom_mask = torch.Tensor(atom_mask).to(device)

        if cfg.model.get('aggregation', '') == 'siamese':
            # average both siamese network passes
            predA, predB = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            pred = torch.mean(torch.cat([predA, predB], dim=-1), dim=-1)
        elif not zero_shot:
            pred, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
        else:
            # non-epistatic (single mut) zero-shot
            pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)[-2]
            pred = zero_shot_convert(pred, mut_positions, mut_mutant_AAs, mut_wildtype_AAs)

            # epistatic zero-shot predictions
            # pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, None, mut_positions)[-2]

            # pred1 = zero_shot_convert(pred, mut_positions[:, 0].unsqueeze(-1), mut_mutant_AAs[:, 0].unsqueeze(-1), mut_wildtype_AAs[:, 0].unsqueeze(-1))
            # pred2 = zero_shot_convert(pred, mut_positions[:, 1].unsqueeze(-1), mut_mutant_AAs[:, 1].unsqueeze(-1), mut_wildtype_AAs[:, 1].unsqueeze(-1))
            # pred = pred1 + pred2

        if len(pred.shape) == 1:
            pred = pred.unsqueeze(-1)

        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(pred, dim=-1), torch.squeeze(mut_ddGs, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(pred, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())
        batches += [i for p in range(len(pred))]   
    
    print('%s mutations evaluated' % (str(len(ddgs))))
    
    if keep:
        preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)

        print(dataset.df.head())

        if 'megascale' in dataset_name:
            tmp = pd.DataFrame({'uid': dataset.df.uid,
            'ddG_pred': preds, 
            'ddG_true': ddgs, 
            'batch': batches, 
            'mut_type': dataset.df.mut_type, 
            'WT_name': dataset.df.WT_name})

        else:
            if 'ptmul' in dataset_name: # manually correct for subset inference df size mismatch
                #dataset.df = dataset.df.loc[dataset.df.NMUT < 3].reset_index(drop=True)
                tmp = pd.DataFrame({'uid': dataset.df.uid,
                'ddG_pred': preds, 
                'ddG_true': ddgs, 
                'batch': batches, 
                'mut_type': dataset.df.MUTS, 
                'WT_name': dataset.df.PDB})

            else:
                tmp = pd.DataFrame({'uid': dataset.df.uid,
                    'ddG_pred': preds, 
                    'ddG_true': ddgs, 
                    'batch': batches, 
                    'mut_type': dataset.df.MUT, 
                    'WT_name': dataset.df.PDB})

    else:
        tmp = pd.DataFrame()
        # tmp.to_csv(f'ThermoMPNN_{os.path.basename(name).removesuffix(".ckpt")}_{dataset_name}_preds.csv')

    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(met_name, column[f"{dtype} {met_name}"])
            except ValueError:
                pass

    return tmp

class ddgBenchDatasetv3(ddgBenchDatasetv2):
    def __init__(self, cfg, pdb_dir, csv_fname, flip=False, multi=False, invert=False, gen_mut=True):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.rev = flip  # "reverse" mutation testing
        print('Reverse mutations: %s' % str(self.rev))
        df = pd.read_csv(csv_fname)
        if not 'PDB' in df.columns:
            df['PDB'] = df['code'] + df['chain']
        if not 'MUT' in df.columns and not 'MUTS' in df.columns or gen_mut:
            if not multi:
                df['MUT'] = df['wild_type'] + df['seq_pos'].astype(int).astype(str) + df['mutation']
            else:
                df['MUTS'] = df['mut_info_seq_pos'].str.replace(':', ';') #.apply(lambda x: len(x.split(':')))
                df['NMUT'] = df['MUTS'].apply(lambda x: len(x.split(';')))
        elif 'MUT' in df.columns:
            if not 'uid' in df.columns:
                df['uid'] = df['code'] +'_' + df['MUT']
        elif 'MUTS' in df.columns:
            if not 'uid' in df.columns:
                df['uid'] = df['code'] +'_' + df['MUTS'].str.replace(';', ':')            
        try:
            df['DDG'] = (-1 if invert else 1 ) * df['ddG']
        except KeyError:
            df['DDG'] = (-1 if invert else 1 ) * df['ddG_dir']
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()
                 
        self.pdb_data = {}
        self.side_chains = self.cfg.data.get('side_chains', False)
        # parse all PDBs first - treat each row as its own PDB
        for i, row in self.df.iterrows():
            #print(row)
            fname = row.PDB[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            pdb_file = row['pdb_file']
            chain = [row.PDB[-1]]
            pdb = alt_parse_PDB(pdb_file, input_chain_list=chain, side_chains=self.side_chains)
            self.pdb_data[i] = pdb[0]

class DomainomeDataset(ddgBenchDatasetv2):
    def __init__(self, cfg, pdb_dir, csv_fname, flip=False,  invert=False):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.rev = flip  # "reverse" mutation testing
        print('Reverse mutations: %s' % str(self.rev))
        df = pd.read_csv(csv_fname)
        df = df.dropna(subset='pdb_file')
        #df = df.loc[df['domain_ID']!='Q13472_PF06839_896']
        df['PDB'] = df['domain_ID']

        df['MUT'] = df['uniprot_ID_mutation'].apply(lambda x: x.split('_')[-1])
        #df['MUT'] = df['mut_info']
        df['uid'] = df['PDB'] + '_' + df['mut_info']
        df = df.loc[~df['MUT'].str.endswith('*')]
        df = df.loc[~df['MUT'].str.contains('NANA')]
        #df['MUT'] = df['MUT'].apply(lambda x: x[0] + str(int(x[1:-1])+1) + x[-1])

        df['DDG'] = (-1 if invert else 1 ) * df['scaled_fitness']

        self.df = df
        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()
                 
        self.pdb_data = {}
        self.side_chains = self.cfg.data.get('side_chains', False)
        # parse all PDBs first - treat each row as its own PDB
        for i, row in tqdm(self.df.reset_index().iterrows(), total=len(self.df)):
            pdb_file = row['pdb_file']
            chain = 'A'
            pdb = alt_parse_PDB(pdb_file, input_chain_list=chain, side_chains=self.side_chains)
            self.pdb_data[i] = pdb[0]
            #if i == 1330:
            #    print(pdb[0])

def extract_uid(row):
    uid = row['WT_name'].split(".pdb")[0]#.replace("|","+")
    if row['WT_name'].split(".pdb")[1]:
        uid += row['WT_name'].split(".pdb")[1]
    uid += '_' + row['mut_type']
    return uid

def split_and_process_domainome(model, model_path, cfg, prefix, n_splits=5):
    """
    Split the domainome_mapped.csv into n parts and process each part separately.
    
    Args:
        model: The model to use for predictions
        model_path: Path to the model
        cfg: Configuration object
        prefix: Directory prefix
        n_splits: Number of splits to create
    """
    # Create output directory
    output_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Read the original CSV file
    domainome_csv_path = os.path.join(prefix, 'preprocessed/domainome_mapped_new.csv')
    domainome_df = pd.read_csv(domainome_csv_path)
    
    # Calculate the number of rows per split
    total_rows = len(domainome_df)
    rows_per_split = int(np.ceil(total_rows / n_splits))
    
    # Create temporary split CSV files
    split_paths = []
    for i in range(n_splits):
        start_idx = i * rows_per_split
        end_idx = min((i + 1) * rows_per_split, total_rows)
        
        split_df = domainome_df.iloc[start_idx:end_idx]
        split_path = os.path.join(prefix, f'preprocessed/domainome_mapped_split_{i}.csv')
        split_df.to_csv(split_path, index=False)
        split_paths.append(split_path)
    
    # Process each split
    all_results = []
    cfg.training.batch_size = 64
    keep = True
    
    for i, split_path in enumerate(tqdm(split_paths, desc="Processing domainome splits")):
        print(f"Processing split {i+1}/{n_splits}")
        
        # Load the split dataset
        domainome_split = DomainomeDataset(cfg, csv_fname=split_path, pdb_dir='domainome')
        
        # Run predictions on this split
        df_preds_split = run_prediction_batched(
            name='ThermoMPNN_single', 
            model=model, 
            dataset_name=f'domainome_split_{i}', 
            results=None, 
            dataset=domainome_split, 
            keep=keep, 
            zero_shot=False, 
            cfg=cfg
        )
        
        # Clean up
        del domainome_split
        
        # Process results
        print(df_preds_split.head())
        #df_preds_split['WT_name'] = df_preds_split['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
        #df_preds_split['uid'] = df_preds_split['WT_name'] + '_' + df_preds_split['mut_type']
        df_preds_split['code'] = df_preds_split['WT_name']
        
        # Append to the combined results
        all_results.append(df_preds_split)
        
        # Optional: Save intermediate results
        df_preds_split.set_index('uid').sort_index().to_csv(f'{output_dir}_split_{i}.csv')
    
    # Combine all results
    df_preds_domainome = pd.concat(all_results)
    df_preds_domainome = df_preds_domainome.set_index('uid').sort_index()
    df_preds_domainome.to_csv(f'{output_dir}.csv')
    
    # Clean up temporary split files
    for split_path in split_paths:
        if os.path.exists(split_path):
            os.remove(split_path)

    return df_preds_domainome

class MegaScaleDatasetv3(MegaScaleDatasetv2):
    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve

        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"])
        df = df.loc[~df['WT_name'].str.contains('pross')]
        df['uid'] = df.apply(extract_uid, axis=1).str.replace('|', '_')
        df['code'] = df['uid'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        df = df.loc[~df['mut_type'].apply(utils.is_fake_mutation)]
        df = df.loc[~df['uid'].apply(utils.is_improper_mutation)]
        # remove unreliable data and insertion/deletion mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)

        mut_list = []
        if 'single' in self.cfg.data.mut_types:
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))
        
        if 'double' in self.cfg.data.mut_types:
            tmp = df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True)

            tmp['dupe'] = tmp['code'] + '_' + tmp['mut_type']
            tmp = tmp.drop_duplicates(subset=['dupe']).reset_index(drop=True)
            mut_list.append(tmp)
            
        self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data
        
        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)

        self.wt_names = [s.replace('+', '_') for s in splits[self.split]]

        ### CHANGED to keep destabilized backbones with .pdb_XnY 
        expanded_names = []
        for ename in list(self.df.WT_name.unique()):
            for wname in list(self.wt_names):
                if wname in ename.replace('|', '_'):
                    expanded_names.append(ename)
        self.wt_names = set(expanded_names)
        ### END ADDED

        # pre-loading wildtype structures - can avoid later file I/O for 50% of data points
        self.side_chains = self.cfg.data.get('side_chains', False)
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            #wt_name = wt_name.replace("|",":") #.split(".pdb")[0].replace("|",":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, wt_name.split('.pdb')[0].replace('|', '_') +'.pdb')
            pdb = parse_PDB(pdb_file, side_chains=self.side_chains)
            self.pdb_data[wt_name] = pdb[0]

        # filter df for only data with structural data
        self.df = self.df.loc[self.df.WT_name.isin(self.wt_names)].reset_index(drop=True)
        
        df_list = []
        # pick which mutations to use (data augmentation)
        if ('single' in self.cfg.data.mut_types) or ('double' in self.cfg.data.mut_types):
            print('Including %s direct single/double mutations' % str(self.df.shape[0]))
            self.df['DIRECT'] = True
            self.df['wt_orig'] = self.df['mut_type'].str[0]  # mark original WT for file loading use
            df_list.append(self.df)
            
        if 'double-aug' in cfg.data.mut_types:
            # grab single mutants even if not included in mutation type list
            tmp = df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True)
            tmp = tmp.loc[tmp.WT_name.isin(self.wt_names)].reset_index(drop=True) # filter by split
            
            double_aug = self._augment_double_mutants(tmp, c=1)
            print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
            double_aug['DIRECT'] = False
            self.tmp = tmp
            df_list.append(double_aug)
        
        if self.split == 'test':
            self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        else:
            self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
        
        epi = cfg.data.epi if 'epi' in cfg.data else False
        if epi:
            self._generate_epi_dataset()

        self._sort_dataset()
        #print(sorted(list(self.pdb_data.keys())))
        #print(sorted(self.df.WT_name.unique()))
        print(self.df.columns)


def main(args):
# default single mutant config

    data_prefix = REPO_ROOT / 'data'
    
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(args.thermompnn_root, 'examples/configs/local.yaml')), OmegaConf.load(os.path.join(args.thermompnn_root, 'examples/configs/single.yaml')))
    cfg = parse_cfg(cfg) 
    cfg.data_loc.megascale_splits = '/home/sareeves/software/esm-msr/data/lora/Megascale/splits.pkl'

    # load single mutant model
    model_path = os.path.join(args.thermompnn_root, 'thermompnn/checkpoints/', args.single_checkpoint)
    model = TransferModelPLv2.load_from_checkpoint(checkpoint_path=model_path, cfg=cfg, device='gpu')

    cfg.training.batch_size = 1
    cfg.data.mut_types = ['single']
    keep = True
    outs = {}
    for name in ['s669', 'ssym', 'q3421', 'k3822']:
        if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
            os.makedirs(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/', exist_ok=True)
            dataset = ddgBenchDatasetv3(cfg, csv_fname=os.path.join(data_prefix, f'preprocessed/{name}_mapped_new.csv'), pdb_dir=name)
            results = run_prediction_batched(name='ThermoMPNN_single', model=model, dataset_name=name, results=None, dataset=dataset, keep=keep, zero_shot=False, cfg=cfg)
            #results['WT_name'] = results['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
            results['uid'] = results['WT_name'] + '_' + results['mut_type']
            results['code'] = results['WT_name']
            results = results.set_index('uid').sort_index()
            print(results.head())
            results['ddG_true'] *= -1
            results['ddG_pred'] *= -1
            results.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')
            outs[name] = results
            print(results.head())
            del dataset
        else:
            outs[name] = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0)

    for name, df in outs.items():
        
        name = name.upper()
        print(name)
        print(df[['ddG_true', 'ddG_pred']].corr('spearman').iloc[0, 1])
        if name == 'S669':
            s461 = pd.read_csv(os.path.join(data_prefix, 'preprocessed/s461_mapped_new.csv'), index_col=0)
            s461['uid'] = s461['code'] + s461['chain'] + '_' + s461['wild_type'] + s461['seq_pos'].astype(int).astype(str) + s461['mutation']
            s461 = s461.set_index('uid')
            new_df = s461.join(df[['ddG_pred']], how='inner')
            assert(len(new_df)==461)
            if new_df['ddG'].mean() > 0:
                new_df['ddG'] *= -1
            print('S461')
            print(new_df[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1])
        if name == 'K3822':
            k2369 = pd.read_csv(os.path.join(data_prefix, 'preprocessed/k2369_mapped_new.csv'), index_col=0)
            #k2369.loc[k2369['code']=='3PGK', 'seq_pos'] -= 1
            k2369['uid'] = k2369['code'] + k2369['chain'] + '_' + k2369['wild_type'] + k2369['seq_pos'].astype(int).astype(str) + k2369['mutation']
            new_df = df.loc[k2369['uid']]
            assert len(new_df) == 2369
            if new_df['ddG_true'].mean() > 0:
                new_df['ddG_true'] *= -1
            print('K2369')
            print(new_df[['ddG_true', 'ddG_pred']].corr('spearman').iloc[0, 1])

    ########## DOMAINOME ################
                
    if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
        df_preds_domainome = split_and_process_domainome(model, model_path, cfg, data_prefix)
    else:
        df_preds_domainome = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 
        if not 'code' in df_preds_domainome.columns:
            df_preds_domainome['code'] = df_preds_domainome['WT_name']
        df_preds_domainome['ddG_pred'] *= -1
        df_preds_domainome['ddG_true'] *= -1
        df_preds_domainome.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')


    ############## REPEAT WITH SPECIFIC SPLIT ################

    if args.split and not args.split == '/home/sareeves/software/esm-msr/data/lora/Megascale/splits.pkl':

        cfg_new = deepcopy(cfg)
        cfg_new.data_loc.megascale_splits = args.split
        split_name = args.split.split('/')[-1].split('.pkl')[0].split('splits_')[1]
        for scaffold in ['validation', 'test']:
            if scaffold == 'validation':
                scaffold_ = 'val'
            else:
                scaffold_ = 'test'
            if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_singles.csv') or args.regenerate_results:
                os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
                # load single mutant dataset of choice - Megascale-S test set
                cfg_new.training.batch_size = 128
                cfg_new.data.dataset = 'megascale'
                cfg_new.data.splits = [scaffold_]
                cfg_new.data.mut_types = ['single']
                keep = True # this will return the raw predictions if True
                dataset = MegaScaleDatasetv3(cfg_new, split=scaffold_)
                df_preds_megascale_singles = run_prediction_batched(name='ThermoMPNN_single', model=model, dataset_name=f'megascale-S-{split_name}-{scaffold}', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg_new)
                #df_preds_megascale_singles['WT_name'] = df_preds_megascale_singles['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
                df_preds_megascale_singles['WT_name'] = df_preds_megascale_singles['WT_name'].apply(lambda x: x.split('.')[0])
                #df_preds_megascale_singles['uid'] = df_preds_megascale_singles['WT_name'] + '_' + df_preds_megascale_singles['mut_type']
                df_preds_megascale_singles['code'] = df_preds_megascale_singles['WT_name']
                df_preds_megascale_singles = df_preds_megascale_singles.set_index('uid').sort_index()
                df_preds_megascale_singles['ddG_pred'] *= -1
                df_preds_megascale_singles['ddG_true'] *= -1  
                df_preds_megascale_singles.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_singles.csv')
                del dataset
            else:
                df_preds_megascale_singles = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_singles.csv', index_col=0)   


            if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_doubles.csv') or args.regenerate_results:
                os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
                # Megascale-D test set
                cfg_new.training.batch_size = 128
                cfg_new.data.dataset = 'megascale'
                cfg_new.data.splits = [scaffold_]
                cfg_new.data.mut_types = ['double']
                cfg_new.data.pick = 0
                # load double mutant dataset twice, once for each mutation
                cfg_new2 = deepcopy(cfg_new)
                cfg_new2.data.pick = 1
                keep = True
                dataset_1 = MegaScaleDatasetv3(cfg_new, split=scaffold_) # first mutation
                dataset_2 = MegaScaleDatasetv3(cfg_new2, split=scaffold_) # second mutation
                results_1 = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name=f'megascale-D-{split_name}-{scaffold}-1', results=[], dataset=dataset_1, keep=keep, zero_shot=False, cfg=cfg_new)
                results_2 = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name=f'megascale-D-{split_name}-{scaffold}-2', results=[], dataset=dataset_2, keep=keep, zero_shot=False, cfg=cfg_new)
                # add single mutant ddGs to get additive prediction
                df_preds_megascale_doubles_additive = results_1.copy(deep=True)
                df_preds_megascale_doubles_additive['ddG_pred_additive'] =  results_1.ddG_pred + results_2.ddG_pred
                #print(df_preds_megascale_doubles_additive.head())
                #df_preds_megascale_doubles_additive['WT_name'] = df_preds_megascale_doubles_additive['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
                df_preds_megascale_doubles_additive['WT_name'] = df_preds_megascale_doubles_additive['WT_name'].apply(lambda x: x.split('.')[0])
                #df_preds_megascale_doubles_additive['uid'] = df_preds_megascale_doubles_additive['WT_name'] + '_' + df_preds_megascale_doubles_additive['mut_type']
                df_preds_megascale_doubles_additive['code'] = df_preds_megascale_doubles_additive['WT_name']
                df_preds_megascale_doubles_additive = df_preds_megascale_doubles_additive.set_index('uid').sort_index()
                df_preds_megascale_doubles_additive['ddG_pred_additive'] *= -1
                df_preds_megascale_doubles_additive['ddG_true'] *= -1
                df_preds_megascale_doubles_additive.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_doubles.csv') 
                del dataset_1
                del dataset_2
            else:
                df_preds_megascale_doubles_additive = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}_doubles.csv', index_col=0) 

            df_preds_megascale_doubles_additive['ddG_pred'] = df_preds_megascale_doubles_additive['ddG_pred_additive']
            results_megascale_combined = pd.concat([df_preds_megascale_singles, df_preds_megascale_doubles_additive], axis=0)
            print(f'Length of combined megascale: {len(results_megascale_combined)}')
            results_megascale_combined.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 

            

    ############## REPEAT WITH Fitness ################

    for batch_size, prot in zip([16,16,16,16,4,512, 4], ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'GB1_Wu_2016_binding_domain']): #]):#, 'GB1_Wu_2016_binding_domain']):
    #for batch_size, prot in zip([4,32,8], ['DLG4_HUMAN_Faure_2021', 'GRB2_HUMAN_Faure_2021', 'MYO_HUMAN_Kung_2025']):
        short_name = prot.split('_')[0] + '_' + prot.split('_')[2]
        short_name = short_name.lower()
        if short_name == 'gb1_2016':
            short_name = 'gb1_wu'

        cfg_cur = deepcopy(cfg)
        cfg_cur.data_loc.megascale_csv = f'/home/sareeves/PSLMs/data/preprocessed/{prot}.csv'
        cfg_cur.data_loc.megascale_splits = f'/home/sareeves/PSLMs/data/preprocessed/{short_name}_splits.pkl'
        cfg_cur.data_loc.megascale_pdbs = '/home/sareeves/PSLMs/structures/'  

        if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
            os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
            # load single mutant dataset of choice - Megascale-S test set
            cfg_cur.training.batch_size = 24
            cfg_cur.data.dataset = 'megascale'
            cfg_cur.data.splits = ['test']
            cfg_cur.data.mut_types = ['single']
            keep = True # this will return the raw predictions if True
            dataset = MegaScaleDatasetv3(cfg_cur, split='test')
            df_preds_cur = run_prediction_batched(name='ThermoMPNN_single', model=model, dataset_name=f'megascale-S-{prot}', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg_cur)
            #df_preds_cur['WT_name'] = df_preds_cur['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
            df_preds_cur['WT_name'] = df_preds_cur['WT_name'].apply(lambda x: x.split('.')[0])
            #df_preds_cur['uid'] = df_preds_cur['WT_name'] + '_' + df_preds_cur['mut_type']
            df_preds_cur['code'] = df_preds_cur['WT_name']
            df_preds_cur = df_preds_cur.set_index('uid').sort_index()
            df_preds_cur['ddG_pred'] *= -1
            df_preds_cur['ddG_true'] *= -1   
            df_preds_cur.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')
            del dataset
        else:
            df_preds_cur = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 
 

        if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
            os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
            # Megascale-D test set
            cfg_cur.training.batch_size = 24
            cfg_cur.data.dataset = 'seuma'
            cfg_cur.data.splits = ['test']
            cfg_cur.data.mut_types = ['double']
            cfg_cur.data.pick = 0
            # load double mutant dataset twice, once for each mutation
            cfg_cur2 = deepcopy(cfg_cur)
            cfg_cur2.data.pick = 1
            keep = True
            dataset_1 = MegaScaleDatasetv3(cfg_cur, split='test') # first mutation
            dataset_2 = MegaScaleDatasetv3(cfg_cur2, split='test') # second mutation
            results_1 = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name=f'megascale-D-{prot}-1', results=[], dataset=dataset_1, keep=keep, zero_shot=False, cfg=cfg_cur)
            results_2 = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name=f'megascale-D-{prot}-2', results=[], dataset=dataset_2, keep=keep, zero_shot=False, cfg=cfg_cur)
            # add single mutant ddGs to get additive prediction
            results_1['ddG_pred_additive'] =  results_1.ddG_pred + results_2.ddG_pred
            #print(results_1.head())
            #results_1['WT_name'] = results_1['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
            results_1['WT_name'] = results_1['WT_name'].apply(lambda x: x.split('.')[0])
            #results_1['uid'] = results_1['WT_name'] + '_' + results_1['mut_type']
            results_1['code'] = results_1['WT_name']
            results_1 = results_1.set_index('uid').sort_index()
            results_1['ddG_pred_additive'] *= -1
            results_1['ddG_true'] *= -1  
            results_1.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 
            del dataset_1
            del dataset_2
        else:
            results_1 = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 

        results_1['ddG_pred'] = results_1['ddG_pred_additive']
        results_cur_combined = pd.concat([df_preds_cur, results_1], axis=0)
        print(f'Length of combined seuma: {len(results_cur_combined)}')


    ############## REPEAT WITH ESTA_BACSU ################

    cfg_nutschel = deepcopy(cfg)
    cfg_nutschel.data_loc.megascale_csv = '/home/sareeves/PSLMs/data/preprocessed/ESTA_BACSU_Nutschel_2020_dTm.csv'
    cfg_nutschel.data_loc.megascale_splits = '/home/sareeves/PSLMs/data/preprocessed/esta_nutschel_splits.pkl'
    cfg_nutschel.data_loc.megascale_pdbs = '/home/sareeves/PSLMs/structures/'  

    if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/ESTA_BACSU_Nutschel_2020_dTm/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
        os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/ESTA_BACSU_Nutschel_2020_dTm/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
        # load single mutant dataset of choice - Megascale-S test set
        cfg_nutschel.training.batch_size = 24
        cfg_nutschel.data.dataset = 'megascale'
        cfg_nutschel.data.splits = ['test']
        cfg_nutschel.data.mut_types = ['single']
        keep = True # this will return the raw predictions if True
        dataset = MegaScaleDatasetv3(cfg_nutschel, split='test')
        df_preds_nutschel = run_prediction_batched(name='ThermoMPNN_single', model=model, dataset_name=f'megascale-S-ESTA_BACSU_Nutschel_2020_dTm', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg_nutschel)
        #df_preds_nutschel['WT_name'] = df_preds_nutschel['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
        df_preds_nutschel['WT_name'] = df_preds_nutschel['WT_name'].apply(lambda x: x.split('.')[0])
        #df_preds_nutschel['uid'] = df_preds_nutschel['WT_name'] + '_' + df_preds_nutschel['mut_type']
        df_preds_nutschel['code'] = df_preds_nutschel['WT_name']
        df_preds_nutschel = df_preds_nutschel.set_index('uid').sort_index()
        df_preds_nutschel['ddG_pred'] *= -1
        df_preds_nutschel['ddG_true'] *= -1  
        df_preds_nutschel.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ESTA_BACSU_Nutschel_2020_dTm/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')
        del dataset
    else:
        df_preds_nutschel = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ESTA_BACSU_Nutschel_2020_dTm/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 


    if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
        os.makedirs(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd', exist_ok=True)
        # Alternate benchmark: ptmuld dataset
        cfg.training.batch_size = 1
        cfg.data.dataset = 'ptmul'
        cfg.data.splits = ['alt']
        cfg.data.mut_types = ['double']
        cfg.data.pick = 0
        cfg2 = deepcopy(cfg)
        cfg2.data.pick = 1
        keep = True

        ptmuld1 = ddgBenchDatasetv3(cfg, pdb_dir='PTMUL', csv_fname=os.path.join(data_prefix, 'preprocessed/ptmuld_mapped_new.csv'), multi=True, invert=True)
        ptmuld2 = ddgBenchDatasetv3(cfg2, pdb_dir='PTMUL', csv_fname=os.path.join(data_prefix, 'preprocessed/ptmuld_mapped_new.csv'), multi=True, invert=True)
        results_1_ptmuld = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name='ptmul-D-1', results=[], dataset=ptmuld1, keep=keep, zero_shot=False, cfg=cfg)
        results_2 = run_prediction_batched(name='ThermoMPNN_additive', model=model, dataset_name='ptmul-D-2', results=[], dataset=ptmuld2, keep=keep, zero_shot=False, cfg=cfg)
        # add single mutant ddGs to get additive prediction
        results_1_ptmuld['ddG_pred_additive'] =  results_1_ptmuld.ddG_pred + results_2.ddG_pred
        results_1_ptmuld['WT_name'] = results_1_ptmuld['WT_name'].apply(lambda x: x.split('_')[0]) # + '_' + x.split('_')[-1])
        results_1_ptmuld['uid'] = results_1_ptmuld['WT_name'] + '_' + results_1_ptmuld['mut_type']
        results_1_ptmuld['code'] = results_1_ptmuld['WT_name']
        results_1_ptmuld = results_1_ptmuld.set_index('uid').sort_index()
        results_1_ptmuld['ddG_pred_additive'] *= -1
        results_1_ptmuld['ddG_true'] *= -1  
        results_1_ptmuld.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 
    else:
        results_1_ptmuld = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 

    print('ptmuld additive')
    print(results_1_ptmuld[['ddG_true', 'ddG_pred_additive']].corr('spearman').iloc[0, 1])
    print(len(results_1_ptmuld[['ddG_true', 'ddG_pred_additive']].dropna()))

    if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/PTMUL/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
        results_1_ptmul = run_multi_mutation_additive_prediction(model, model_path, cfg, data_prefix, max_mutations=10)
        results_1_ptmul['ddG_pred_additive'] *= -1
        results_1_ptmul.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/PTMUL/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')
    else:
        results_1_ptmul = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/PTMUL/thermompnnd/additive_{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 

    print('PTMUL additive')
    print(results_1_ptmul[['ddG_true', 'ddG_pred_additive']].corr('spearman').iloc[0, 1])
    print(len(results_1_ptmul[['ddG_true', 'ddG_pred_additive']].dropna()))

    thermompnn_root = '/home/sareeves/software/ThermoMPNN-D'
    # default epistatic double mutant config
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(thermompnn_root,'examples/configs/local.yaml')), OmegaConf.load(os.path.join(thermompnn_root,'examples/configs/epistatic.yaml')))
    cfg = parse_cfg(cfg)

    if args.epistatic_checkpoint is not None:
        # load epistatic double mutant model
        model_path = os.path.join(args.thermompnn_root, 'thermompnn/checkpoints/', args.epistatic_checkpoint)
        model = TransferModelPLv2Siamese.load_from_checkpoint(checkpoint_path=model_path, cfg=cfg, device='gpu')

        ############## REPEAT WITH SPECIFIC SPLIT ################

        if args.split and not args.split == '/home/sareeves/software/esm-msr/data/lora/Megascale/splits.pkl':

            cfg_new = deepcopy(cfg)
            cfg_new.data_loc.megascale_splits = args.split
            split_name = args.split.split('/')[-1].split('.pkl')[0].split('splits_')[1]

            for scaffold in ['validation', 'test']:
                if scaffold == 'validation':
                    scaffold_ = 'val'
                else:
                    scaffold_ = 'test'
                if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
                    os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
                    # Megascale-D test set
                    cfg_new.training.batch_size = 128
                    cfg_new.data.dataset = 'megascale'
                    cfg_new.data.splits = [scaffold_]
                    cfg_new.data.mut_types = ['double']
                    keep = True
                    dataset = MegaScaleDatasetv3(cfg_new, split=scaffold_) # double mutation
                    results_megascale_epistatic = run_prediction_batched(name='ThermoMPNN_epistatic', model=model, dataset_name=f'megascale-D-{split_name}-{scaffold}', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg_new)
                    #results_megascale_epistatic['WT_name'] = results_megascale_epistatic['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
                    results_megascale_epistatic['WT_name'] = results_megascale_epistatic['WT_name'].apply(lambda x: x.split('.')[0])
                    #results_megascale_epistatic['uid'] = results_megascale_epistatic['WT_name'] + '_' + results_megascale_epistatic['mut_type']
                    results_megascale_epistatic['code'] = results_megascale_epistatic['WT_name']
                    results_megascale_epistatic = results_megascale_epistatic.set_index('uid').sort_index()
                    results_megascale_epistatic['ddG_pred'] *= -1
                    results_megascale_epistatic['ddG_true'] *= -1
                    results_megascale_epistatic.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 
                else:
                    results_megascale_epistatic = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 
                


        ############## REPEAT WITH EXTERNAL ################

        for batch_size, prot in zip([16,16,16,16,4,512], ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'GB1_Wu_2016_binding_domain']):#, 'GB1_Wu_2016_binding_domain']):
            short_name = prot.split('_')[0] + '_' + prot.split('_')[2]
            short_name = short_name.lower()

            if short_name == 'gb1_2016':
                short_name = 'gb1_wu'

            cfg_cur = deepcopy(cfg)
            cfg_cur.data_loc.megascale_csv = f'/home/sareeves/PSLMs/data/preprocessed/{prot}.csv'
            cfg_cur.data_loc.megascale_splits = f'/home/sareeves/PSLMs/data/preprocessed/{short_name}_splits.pkl'
            cfg_cur.data_loc.megascale_pdbs = '/home/sareeves/PSLMs/structures/'  

            if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
                os.makedirs(os.path.dirname(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}'), exist_ok=True)
                # Megascale-D test set
                cfg_cur.training.batch_size = 24
                cfg_cur.data.dataset = 'megascale'
                cfg_cur.data.splits = ['test']
                cfg_cur.data.mut_types = ['double']
                keep = True
                dataset = MegaScaleDatasetv3(cfg_cur, split='test') # double mutation
                results_cur_epistatic = run_prediction_batched(name='ThermoMPNN_epistatic', model=model, dataset_name=f'megascale-D-{prot}', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg_cur)
                #results_cur_epistatic['WT_name'] = results_cur_epistatic['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
                results_cur_epistatic['WT_name'] = results_cur_epistatic['WT_name'].apply(lambda x: x.split('.')[0])
                #results_cur_epistatic['uid'] = results_cur_epistatic['WT_name'] + '_' + results_cur_epistatic['mut_type']
                results_cur_epistatic['code'] = results_cur_epistatic['WT_name']
                results_cur_epistatic = results_cur_epistatic.set_index('uid').sort_index()
                results_cur_epistatic['ddG_pred'] *= -1
                results_cur_epistatic['ddG_true'] *= -1  
                results_cur_epistatic.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 
            else:
                results_cur_epistatic = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 

            

        #########################################################

        if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
            # ptmuld dataset
            cfg.training.batch_size = 1
            cfg.data.dataset = 'ptmul'
            cfg.data.splits = ['alt']
            cfg.data.mut_types = ['double']
            keep = True
            dataset = ddgBenchDatasetv3(cfg, pdb_dir='PTMUL', csv_fname=os.path.join(data_prefix, 'preprocessed/ptmuld_mapped_new.csv'), multi=True, invert=True)
            results_ptmuld_epistatic = run_prediction_batched(name='ThermoMPNN_epistatic', model=model, dataset_name='ptmul-D-test-epi', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg)
            results_ptmuld_epistatic['WT_name'] = results_ptmuld_epistatic['WT_name'].apply(lambda x: x.split('_')[0]) # + '_' + x.split('_')[-1])
            results_ptmuld_epistatic['uid'] = results_ptmuld_epistatic['WT_name'] + '_' + results_ptmuld_epistatic['mut_type'].str.replace(';', ':')
            results_ptmuld_epistatic['code'] = results_ptmuld_epistatic['WT_name']
            results_ptmuld_epistatic = results_ptmuld_epistatic.set_index('uid').sort_index()
            results_ptmuld_epistatic['ddG_pred'] *= -1
            results_ptmuld_epistatic['ddG_true'] *= -1  
            results_ptmuld_epistatic.to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') 
        else:
            results_ptmuld_epistatic = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/ptmuld/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0) 

        print('ptmuld epistatic')
        print(results_ptmuld_epistatic[['ddG_true', 'ddG_pred']].corr('spearman').iloc[0, 1])
    

        outs = {}
        for name in ['D7PM05_CLYGR', 'GFP_AEQVI', 'HIS7_YEAST', 'Q6WV12_9MAXI', 'Q8WTC7_9CNID', 'RASK_HUMAN']:
            if not os.path.exists(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv') or args.regenerate_results:
                os.makedirs(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/', exist_ok=True)
                cfg.training.batch_size = 32
                cfg.data.dataset = 'ptmul'
                cfg.data.splits = ['alt']
                cfg.data.mut_types = ['double']
                dataset = ddgBenchDatasetv3(cfg, csv_fname=os.path.join(data_prefix, f'lora/DMS/csv_formatted/{name}.csv'), pdb_dir='PTMUL', multi=True, gen_mut=False)
                results_dms_epistatic = run_prediction_batched(name='ThermoMPNN_epistatic', model=model, dataset_name='ptmul_fake', results=[], dataset=dataset, keep=keep, zero_shot=False, cfg=cfg)
                results_dms_epistatic['WT_name'] = results_dms_epistatic['WT_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[-1])
                results_dms_epistatic['uid'] = results_dms_epistatic['WT_name'].str[:-1] + '_' + results_dms_epistatic['mut_type'].str.replace(':', ';')
                results_dms_epistatic['code'] = results_dms_epistatic['WT_name'].str[:-1]
                #results_dms_epistatic['ddG_pred'] *= -1
                #results_dms_epistatic['ddG_true'] *= -1 
                results_dms_epistatic = results_dms_epistatic.set_index('uid').sort_index()
                outs[name] = results_dms_epistatic
                outs[name].to_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv')
            else:
                outs[name] = pd.read_csv(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name}/thermompnnd/{model_path.split("/")[-1]+("_excl_destab" if args.excl_destab else "")}.csv', index_col=0)

        for name, df in outs.items():  
            name = name.upper()
            print(name)
            print(df[['ddG_true', 'ddG_pred']].corr('spearman').iloc[0, 1])

    

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--thermompnn_root', type=str, default='/home/sareeves/software/ThermoMPNN-D/')
        #parser.add_argument('--output', type=str, default='../data/lora/test_scores_final.csv')
        parser.add_argument('--split', type=str) #'/home/sareeves/PSLMs/data/lora/Megascale/splits.pkl'
        parser.add_argument('--single_checkpoint', type=str) #single_epoch=99_val_ddG_spearman=0.73.ckpt'
        parser.add_argument('--epistatic_checkpoint', type=str) # esm_epistatic_epoch=50_val_ddG_spearman=0.78.ckpt'
        parser.add_argument('--regenerate_results', action='store_true') #'/home/sareeves/PSLMs/data/lora/Megascale/splits.pkl'
        parser.add_argument('--excl_destab', action='store_true')
        args = parser.parse_args()
        
        main(args)