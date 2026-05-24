import os
import re
import argparse
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from esm_msr import utils
from esm_msr.data import ( 
    ProteinStructureMutationEpistasisDataset,
    collate_fn_twopass,
    ProteinCyclingDataLoader,
    SubsetRestrictedProteinCyclingDataLoader,
    PooledDataLoader
)


def is_fake_mutation(mut_string):
    # Split by colon to handle multiple mutations
    mutations = mut_string.split(':')
    
    for mutation in mutations:
        # Use regex to extract source, position, and target
        match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', mutation)
        if match:
            source = match.group(1)
            target = match.group(3)
            # Check if source equals target (improper mutation)
            if source == target:
                return True
    
    return False


def is_improper_mutation(mutation_string: str) -> bool:
    """
    Checks a string containing protein mutations for conflicts.

    A conflict occurs if two mutations at the same position are inconsistent.
    Specifically, if mutation M1 (X1##Y1) and mutation M2 (X2##Y2) both occur
    at position ##, and M1 appears before M2 in the string, then Y1 (the
    result of M1) must be equal to X2 (the starting amino acid for M2).
    If this condition is violated for any such pair, a conflict exists.

    Mutations are expected in the format X##Y (e.g., L45S), where X and Y are
    uppercase letters and ## is one or more digits.
    The pattern X##Y should not be extracted if X is immediately preceded by a
    digit (e.g., the 'A2H' in '1A2H' will not be considered a mutation).
    Mutations in the string can be separated by underscores ('_'), colons (':'),
    or other characters.

    Args:
        mutation_string: The string to check, e.g., "1ANP_L45S_L73P:G81T"
                         or "A12C_C12G_G12T".

    Returns:
        True if no conflicting mutations are found, False otherwise.
        Returns True for strings with no mutations or only one mutation
        at any given site.
    """

    mutation_pattern = re.compile(r"(?<!\d)([A-Z])(\d+)([A-Z])")

    mutations = []
    for match in mutation_pattern.finditer(mutation_string):
        wild_type = match.group(1)
        position = int(match.group(2))
        mutant = match.group(3)
        mutations.append({
            "wild_type": wild_type,
            "position": position,
            "mutant": mutant,
            "original_string": match.group(0)
        })

    if not mutations:
        return True

    mutations_by_position = {}
    for mut in mutations:
        pos = mut["position"]
        if pos not in mutations_by_position:
            mutations_by_position[pos] = []
        mutations_by_position[pos].append(mut)

    for position, pos_mutations in mutations_by_position.items():
        if len(pos_mutations) > 1:
            for i in range(len(pos_mutations) - 1):
                current_mutation = pos_mutations[i]
                next_mutation = pos_mutations[i+1]
                if current_mutation["mutant"] != next_mutation["wild_type"]:
                    return True
    
    return False

class MegaScaleDatasetPreprocessor:
    """Handles raw data loading, preprocessing, and dataset creation."""
    def __init__(self, data_file: str, af_model_folder: str = '/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs/', spurs_override: bool = False):
        self.data_file = data_file
        self.af_model_folder = af_model_folder
        self.df = pd.DataFrame()
        self.split_dfs = {}
        self.spurs_override = spurs_override

        self.preprocess()

    def preprocess(self) -> None:
        """Adds UID and removes fake/improper mutations."""
        if self.spurs_override:
            logging.info('Preprocessing according to SPURS methodology.')
            self._preprocess_spurs()
            return
        
        try:
            self.df = pd.read_csv(self.data_file)
            logging.info(f"Loaded raw data: {len(self.df)} entries from {self.data_file}")
        except FileNotFoundError:
            logging.error(f"Data file not found: {self.data_file}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

        self.df = self.df[['aa_seq', 'mut_type', 'WT_name', 'ddG_ML']]
        self.df['ddG_ML'] = pd.to_numeric(self.df['ddG_ML'], errors='coerce')
        self.df = self.df.loc[self.df['ddG_ML'].notna()]
        self.df = self.df.loc[~self.df['mut_type'].str.contains('wt')]
        self.df = self.df.loc[~self.df['mut_type'].str.contains('ins')]
        self.df = self.df.loc[~self.df['mut_type'].str.contains('del')]

        # removed because it has mislabelled rows, seq doesn't match bb mutation, unclear which is correct
        self.df = self.df.loc[self.df['WT_name'] != '1UBQ.pdb_L43A']

        self.df['mut_structure'] = self.df['WT_name'].apply(lambda x: x.split('.pdb_')[1] if '.pdb_' in x else None)
        self.df['code_wt'] = self.df['WT_name'].apply(lambda x: x.split('.pdb')[0])
        self.df['code'] = self.df.apply(
            lambda row: f"{row['code_wt']}_{row['mut_structure']}" if pd.notnull(row['mut_structure']) else row['code_wt'], 
            axis=1
        )
        self.df['chain'] = 'A'
        self.df['uid'] = self.df['code'].astype(str) + '_' + self.df['mut_type'].astype(str)

        groupby_cols = list(self.df.columns.drop(['ddG_ML']))
        self.df, dup = utils.remove_duplicates_with_mean(self.df, groupby_cols, 'ddG_ML', preserve_index=True)

        self.df['uid'] = self.df['uid'].str.replace('|', '_', regex=False)
        self.df['code'] = self.df['code'].str.replace('|', '_', regex=False)
        self.df['code_wt'] = self.df['code_wt'].str.replace('|', '_', regex=False)

        base_dir = os.path.join(self.af_model_folder, '')
        self.df['pdb_file'] = base_dir + self.df['code_wt'] + '.pdb'

        logging.info(f"Reduced to: {len(self.df)} entries after basic filtering")

    def _preprocess_spurs(self) -> None:
        # reproduce preprocessing steps from SPURS with compatibility

        df = pd.read_csv(self.data_file, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)

        df1 = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)
        df2 = df.loc[(~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & df.mut_type.str.contains(":")) | (df.mut_type == 'wt'), :].reset_index(drop=True)
        
        # Concat with ignore_index=False to preserve duplicate indices from df1 and df2
        df = pd.concat([df1,df2], axis=0, ignore_index=False)

        orig_index = df.index

        self.df = df

        self.df['ddG_ML'] = pd.to_numeric(self.df['ddG_ML'], errors='coerce')
        self.df = self.df.loc[self.df['ddG_ML'].notna()]

        self.df['mut_structure'] = self.df['WT_name'].apply(lambda x: x.split('.pdb_')[1] if '.pdb_' in x else None)
        self.df['code_wt'] = self.df['WT_name'].apply(lambda x: x.split('.pdb')[0])
        self.df['code'] = self.df.apply(
            lambda row: f"{row['code_wt']}_{row['mut_structure']}" if pd.notnull(row['mut_structure']) else row['code_wt'], 
            axis=1
        )
        self.df['chain'] = 'A'
        self.df['uid'] = self.df['code'].astype(str) + '_' + self.df['mut_type'].astype(str)

        groupby_cols = list(self.df.columns.drop(['ddG_ML']))
        self.df, dup = utils.remove_duplicates_with_mean(self.df, groupby_cols, 'ddG_ML', preserve_index=True)

        self.df['uid'] = self.df['uid'].str.replace('|', '_', regex=False)
        self.df['code'] = self.df['code'].str.replace('|', '_', regex=False)
        self.df['code_wt'] = self.df['code_wt'].str.replace('|', '_', regex=False)

        base_dir = os.path.join(self.af_model_folder, '')
        self.df['pdb_file'] = base_dir + self.df['code_wt'] + '.pdb'

        logging.info(f"Total size: {len(self.df)} according to SPURS preprocessing")

        # assert that the indices are retained from the pre-concat state
        # orig_index.equals() will fail if duplicates were actually removed
        assert self.df.index.isin(orig_index).all(), "Index mismatch detected: Indices were flattened or lost."

    def _filter_data(self, df, scaffold) -> pd.DataFrame:

        if self.spurs_override and scaffold != 'test':
            # second half of SPURS preprocessing
            orig_wt_names = df['WT_name'].unique()
            
            mmseq_wt_search = os.path.join(self.af_model_folder, '../../mmseq_mut_search_0.25.m8')
            ret = []
            with open(mmseq_wt_search, 'r') as f:
                for line in f.readlines():
                    second_column_value = int(line.split("\t")[1])
                    ret.append(second_column_value)
            # we dont want the rows in the ret
            previous_len = len(df)
            df = df.loc[~df.index.isin(ret), :]#.reset_index(drop=True)
            cur_len = len(df)
            if (previous_len - cur_len) > 0:
                logging.info(f"removed {previous_len - cur_len} rows from the dataset due to SPURS filtering")

            removed_wt_names = []
            for wt_name in orig_wt_names:
                wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
                mut_rows = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
                
                # Verify both WT and mutation rows exist
                if len(wt_rows) == 0 or len(mut_rows) == 0:
                    removed_wt_names.append(wt_name)
                    df = df.loc[df['WT_name']!=wt_name]

            if len(removed_wt_names) > 0:
                logging.warning(f"Removed {removed_wt_names} from the dataset")

        orig_len = len(df)
        
        df = df.loc[df['mut_type'] != 'wt']
        df = df.loc[~df['mut_structure'].fillna('').str.startswith('pross')]
        df = df.loc[~df['mut_type'].apply(is_fake_mutation)]
        df = df.loc[~df['uid'].apply(is_improper_mutation)]

        #if orig_len - len(df) > 0:
        #    logging.info(f"Removed {orig_len - len(df)} fake or improper mutations (wt == mut)")

        if len(df) == 0:
            logging.warning(f"Filtration removed all valid rows for scaffold '{scaffold}'")

        return df

    def create_training_splits(self, split_file: Optional[str], max_train_proteins: int = -1) -> Dict[str, List[str]]:
        """Derives train/val/test splits from a pickled dictionary or unique codes."""
        splits_dict = {'train': [], 'val': [], 'test': []}
        
        if split_file:
            try:
                with open(split_file, 'rb') as f:
                    splits = pickle.load(f)
                
                splits_dict['train'] = [c.split('.pdb')[0] for c in splits['train']]
                splits_dict['val'] = [c.split('.pdb')[0] for c in splits['val']]
                splits_dict['test'] = [c.split('.pdb')[0] for c in splits['test']]
                logging.info(f"Using split file {split_file}.")
                
            except FileNotFoundError: 
                raise FileNotFoundError(f"Split file not found: {split_file}")
            except KeyError as e: 
                raise KeyError(f"Missing expected key in pickle dictionary: {e}")
            except pickle.UnpicklingError as e:
                raise RuntimeError(f"Failed to unpickle {split_file}. Is it a valid .pkl file? Error: {e}")
            except Exception as e: 
                raise RuntimeError(f"Error reading split file {split_file}: {e}")
        else:
            protein_list = self.df['code_wt'].unique().tolist()
            
            # Default behavior if no split file: dump all into train
            splits_dict['train'] = protein_list
            logging.info("No split file provided. All unique codes assigned to train.")

        if max_train_proteins > 0:
            splits_dict['train'] = splits_dict['train'][:max_train_proteins]
            logging.info(f"Limiting training to {len(splits_dict['train'])} proteins.")

        for scaffold in ['train', 'val', 'test']:
            
            items = []

            for prot_code in tqdm(splits_dict[scaffold], desc=f"Creating {scaffold} datasets"):
                df_prot = self.df[self.df['code_wt']==prot_code]

                df_prot = self._filter_data(df_prot, scaffold=scaffold)
                    
                if df_prot.empty: 
                    logging.warning(f"Skipping {scaffold} protein '{prot_code}': Doesn't appear to be in the dataframe")
                    continue

                items.append(df_prot)

            if not items:
                raise AssertionError(f"Scaffold '{scaffold}' contains no valid data items to concatenate.")

            combined_df = pd.concat(items)
            self.split_dfs[scaffold] = combined_df 

        self.get_splits()
        return splits_dict

    def get_splits(self) -> Dict:
        splits = {}
        for scaffold in ['train', 'val', 'test']:
            df = self.split_dfs[scaffold]
            splits[scaffold] = list(df['code'].unique())
            print('Scaffold', scaffold, ':', len(splits[scaffold]), 'unique structures:')
            print(splits[scaffold])
        return splits

    def create_protein_dataloaders(
        self, 
        tokenizer: Any, 
        structure_encoder: Any, 
        scaffold: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool = False,
        cache_path: str = '',
        score_name: str = '',
        generate_cache: bool = False,
        mut_structures_root: str = '',
        incl_destab_bb: bool = True,
        incl_singles: bool = True,
        incl_doubles: bool = False,
        incl_reversions: bool = False,
        incl_mut_ctx: bool = False,
        combine_validation: bool = False,
    ) -> Tuple[List[DataLoader], List[str]]:
        """Generates a list of dataloaders for a specific list of protein codes."""
        loaders = []
        loader_names = []

        df = self.split_dfs[scaffold]
        if combine_validation:
            df = pd.concat([df, self.split_dfs['val']])
            
        protein_list = list(df['code_wt'].unique())
        print(scaffold)
        print(df['code'].unique())

        logging.info(f'Creating {scaffold} datasets for {len(protein_list)} proteins...')
        for wt_code in tqdm(protein_list, desc=f"Creating {scaffold} datasets"):
        
            df_prot = df.loc[df['code_wt']==wt_code]
            for code in df_prot['code'].unique():
                df_prot_ = df_prot.loc[df_prot['code']==code]
       
                dataset = ProteinStructureMutationEpistasisDataset(
                    dms_df=df_prot_, tokenizer=tokenizer, structure_encoder=structure_encoder,
                    dms_name=code, path=cache_path, score_name=score_name,
                    generate=generate_cache, mut_structs_root=mut_structures_root,
                    incl_destab_bb=incl_destab_bb,
                    incl_singles=incl_singles,
                    incl_doubles=incl_doubles,
                    incl_reversions=incl_reversions, 
                    incl_mut_ctx=incl_mut_ctx,
                )
                if len(dataset) == 0:
                        logging.warning(f"{scaffold.capitalize()} dataset for '{code}' is empty. Skipping.")
                        continue
                        
                loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_twopass,
                                    num_workers=num_workers, shuffle=shuffle, pin_memory=True)
                loaders.append(loader)
                loader_names.append(code)

        return loaders, loader_names


def load_benchmark_datasets(data_path_base: str, tokenizer: Any, structure_encoder: Any, cache_path: str, generate_cache: bool) -> Tuple[List[DataLoader], List[str]]:
    """Loads static benchmark datasets."""
    val_dataloaders = []
    val_loader_names = []
    benchmark_datasets = {'ptmuld': 'ptmuld_mapped.csv', 's461': 's461_mapped.csv', 'ssym': 'ssym_mapped.csv'}
    
    for name, filename in benchmark_datasets.items():
        filepath = os.path.join(data_path_base, filename)
        if os.path.exists(filepath):
            try:
                df_bench = pd.read_csv(filepath)
                score_name_bench = 'ddG'
                if score_name_bench not in df_bench.columns:
                     logging.warning(f"Benchmark '{name}' missing '{score_name_bench}' column. Skipping.")
                     continue

                dataset = ProteinStructureMutationEpistasisDataset(
                    dms_df=df_bench, tokenizer=tokenizer, structure_encoder=structure_encoder,
                    dms_name=name, path=cache_path, score_name=score_name_bench,
                    generate=generate_cache, mut_structs_root='',
                    incl_destab_bb=False, 
                    incl_singles=True,
                    incl_doubles=True,
                    incl_reversions=False,
                    incl_mut_ctx=False
                )
                
                if len(dataset) == 0: continue
                
                loader = PooledDataLoader([DataLoader(dataset, batch_size=1, collate_fn=collate_fn_twopass, num_workers=0, shuffle=False, pin_memory=True)], batch_size=3)
                val_dataloaders.append(loader)
                val_loader_names.append(name)
                logging.info(f"Added benchmark validation dataset: {name}")
            except Exception as e: logging.warning(f"Could not load benchmark {name}: {e}", exc_info=True)
        else:
            logging.warning(f'{filepath} could not be found! Skipping this loader.')
            
    return val_dataloaders, val_loader_names


def setup_dataloaders(args: argparse.Namespace, tokenizer: Any, structure_encoder: Any, add_benchmarks_to_val = False) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader], List[str], List[str], List[str]]:
    """Main orchestrator utilizing MegaScaleDatasetPreprocessor."""
    
    preprocessor = MegaScaleDatasetPreprocessor(args.raw_data_file, args.af_model_folder, spurs_override=args.remove_spurs_homologs)
    splits = preprocessor.create_training_splits(args.split_file, args.max_train_proteins)
    
    overlap = set(splits.get('train', [])) & set(splits.get('val', []))
    if overlap: 
        logging.warning(f"Overlap detected between train/val protein codes: {len(overlap)} proteins.")

    struct_enc_arg = None if args.incl_structure_encoder_mt else structure_encoder

    # Build Train Loaders
    train_dataloaders, train_loader_names = preprocessor.create_protein_dataloaders(
        tokenizer=tokenizer, structure_encoder=struct_enc_arg,
        scaffold='train', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        cache_path=args.cache_path, score_name=args.score_column,
        generate_cache=args.regenerate_cache,
        mut_structures_root=args.mut_structures_root, combine_validation=args.combine_validation,
        incl_singles=args.incl_singles,
        incl_doubles=args.incl_doubles,
        incl_mut_ctx=args.incl_mut_ctx, 
        incl_reversions=args.incl_reversions,
    )

    if not train_dataloaders: 
        raise RuntimeError("No valid training dataloaders created.")

    # Build Validation Loaders
    val_dataloaders, val_loader_names = preprocessor.create_protein_dataloaders(
        tokenizer=tokenizer, structure_encoder=struct_enc_arg,
        scaffold='val', batch_size=64, num_workers=args.num_workers, shuffle=False,
        cache_path=args.cache_path, score_name=args.score_column,
        generate_cache=args.regenerate_cache,
        incl_singles=True,
        incl_doubles=True,
        incl_reversions=False,
        incl_mut_ctx=False,
    )

    # Add Benchmarks
    bench_loaders, bench_names = load_benchmark_datasets(
        args.benchmark_data_path, tokenizer, struct_enc_arg, args.cache_path, args.regenerate_cache
    )
    
    if add_benchmarks_to_val:
        val_dataloaders.extend(bench_loaders)
        val_loader_names.extend(bench_names)

    if not val_dataloaders: 
        logging.warning("No validation dataloaders created.")

    # Final Assembly (Train only)
    if args.dataloading == 'cycle':
        if not args.use_subset_restrict:
            logging.info("Using ProteinCyclingDataLoader")
            train_combined_loader = ProteinCyclingDataLoader(
                train_dataloaders, args.batch_size, train_loader_names, collate_fn_twopass,
                strategy=args.loader_strategy,
            )
        else:
            subset_configs = getattr(args, 'subset_balance_configs', None)
            subset_caps = getattr(args, 'subset_caps', None)
            logging.info("Using SubsetRestrictedProteinCyclingDataLoader")
            logging.info(f"Subset balance configs: {subset_configs}, Subset caps: {subset_caps}")
            train_combined_loader = SubsetRestrictedProteinCyclingDataLoader(
                train_dataloaders, args.batch_size, train_loader_names, collate_fn_twopass,
                strategy=args.loader_strategy, 
                subset_balance_configs=subset_configs,        
                subset_caps=subset_caps
            )
        train_dataloaders_final = [train_combined_loader]
        train_loader_names_final = ['cycled_train']
        
    elif args.dataloading == 'pool':
        logging.info("Using PooledDataLoader.")
        train_combined_loader = PooledDataLoader(train_dataloaders, args.batch_size, train_loader_names, strategy=args.loader_strategy)
        train_dataloaders_final = [train_combined_loader]
        train_loader_names_final = ['pooled_train']
        
    else:
        logging.warning(f"Unsupported loader_strategy '{args.loader_strategy}'. Using individual loaders.")
        train_dataloaders_final = train_dataloaders
        train_loader_names_final = train_loader_names

    logging.info(f"Setup complete. Final Training loaders: {len(train_dataloaders_final)} (Names: {train_loader_names_final}), Final Validation loaders: {len(val_dataloaders)} (Names: {val_loader_names})")
    
    return train_dataloaders_final, val_dataloaders, bench_loaders, train_loader_names_final, val_loader_names, bench_names