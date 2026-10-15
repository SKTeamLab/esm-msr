import os
import re
import argparse
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from esm_msr.censoring import SRC_NONE, SRC_RANGE, range_tag
from esm_msr.data import (
    MEAS_COLUMNS,
    MutationStabilityDataset,
    library_floor,
    shrink_floors,
    collate_fn_twopass,
    create_consolidated_dataloader,
    PooledDataLoader
)

# The columns that identify one measured variant. Replicate rows share all of them and are collapsed by taking the
# mean of every measurement column (`_collapse_replicates`); nothing else may enter the key, or replicates whose
# confidence intervals differ slightly would stop collapsing.
IDENTITY_COLS = ['aa_seq', 'mut_type', 'WT_name', 'mut_structure', 'code_wt', 'code', 'chain', 'uid']


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

def _tag_range_rows(df: pd.DataFrame, admit: bool) -> pd.DataFrame:
    """
    Mark the rows the assay refused to fit but still bounded, and drop the ones it bounded not at all.

    Must run BEFORE the numeric coercion, which is what destroys the ``<-1`` / ``>5`` tags. Keeping these rows is
    what stops the double set being selected on its own outcome; see ``esm_msr.censoring`` for why that matters and
    ``--censor_range`` for the switch. With ``admit=False`` the old behaviour is restored exactly: every row the
    assay declined to fit is dropped.
    """
    tags = [range_tag(dg, ddg) for dg, ddg in zip(df.get('dG_ML', pd.Series(index=df.index, dtype=object)),
                                                  df.get('ddG_ML', pd.Series(index=df.index, dtype=object)))]
    out = df.copy()
    out['cens'] = [t[0] for t in tags]
    out['dG_bound'] = [t[1] for t in tags]
    out['cens_src'] = np.where(out['cens'] != 0, SRC_RANGE, SRC_NONE)
    unfitted = out['ddG_ML'].astype(str).str.strip() == '-'
    if not admit:
        return out.loc[~unfitted]
    n_tag, n_nil = int((out['cens'] != 0).sum()), int((unfitted & (out['cens'] == 0)).sum())
    logging.info(f"range censoring: admitting {n_tag} bounded rows the assay declined to fit "
                 f"({int((out['cens'] < 0).sum())} '<-1', {int((out['cens'] > 0).sum())} '>5'); "
                 f"dropping {n_nil} that carry no bound either")
    return out.loc[~unfitted | (out['cens'] != 0)]


def _fill_range_values(df: pd.DataFrame, dG_wt: Dict[str, float]) -> pd.DataFrame:
    """
    Give every range-tagged row the numbers the rest of the pipeline needs, derived from its bound.

    ``dG_ML`` becomes the tag's threshold and ``ddG_ML`` that threshold minus the library's wild-type dG, so the
    row's label IS its bound on its own ddG scale. ``apply_item_censoring`` then derives ``cens_bound`` through the
    usual ``dG_bound + (ddG - dG_meas)`` offset and gets the label back unchanged, exactly as for ``SRC_GAP``.

    The wild-type dG comes from the library's ``wt`` rows. A library whose wild type is itself off-scale has none,
    and on this dataset those libraries turn out to have no measured rows at all - every row is a tag - so there is
    no reference to place them against and nothing to recover: 40k of the 96k tagged rows are dropped here, all of
    them from such libraries. A tagged row with no wild-type dG keeps NaN and the caller's notna filter removes it.
    """
    if 'cens' not in df.columns or not (df['cens'] != 0).any():
        return df
    key = (df['WT_name'].str.replace('.pdb_', '_', regex=False)
           .str.replace('.pdb', '', regex=False).str.replace('|', '_', regex=False))
    wt = key.map(dG_wt).astype(float)
    tagged = df['cens'] != 0
    out = df.copy()
    out.loc[tagged, 'dG_ML'] = out.loc[tagged, 'dG_bound']
    out.loc[tagged, 'ddG_ML'] = out.loc[tagged, 'dG_bound'] - wt[tagged]
    lost = int((tagged & wt.isna()).sum())
    if lost:
        logging.info(f"range censoring: dropping {lost} tagged rows whose library has no wild-type dG to place "
                     f"them against")
    return out


def _drop_all_bounded_libraries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any library left with no measured rows at all.

    Admitting the bounded rows resurrects a handful of libraries whose every variant the assay declined to fit -
    on this dataset 22 of them, three of which sit in a split. Every range-censored item of a library carries the
    same label (its bound is ``DG_RANGE_LOW - dG_wt``, a library constant), so such a library has no label variance:
    it can teach the hinge nothing it does not already know from the other libraries, and as a validation library
    it would be scored entirely against a constant. The point of admitting these rows is to restore the missing
    destabilised tail INSIDE libraries that have measurements, not to invent libraries made only of bounds.
    """
    if 'cens' not in df.columns or 'code' not in df.columns:
        return df
    measured = df.loc[df['cens'] == 0, 'code'].unique()
    keep = df['code'].isin(measured)
    if not keep.all():
        gone = sorted(set(df.loc[~keep, 'code'].unique()))
        logging.info(f"range censoring: dropping {len(gone)} libraries with no measured rows "
                     f"({int((~keep).sum())} items): {gone[:8]}{' ...' if len(gone) > 8 else ''}")
    return df.loc[keep]


def _collapse_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse replicate rows of the same variant, averaging every measurement column.

    Replicates are rows agreeing on all of ``IDENTITY_COLS``. The former code grouped by "every column except
    ddG_ML", which was the same thing while only four columns were kept, but silently stops collapsing once the
    per-protease values and their confidence intervals are carried along, since replicates differ in those. The
    original index of the first row of each group is preserved, because the SPURS path filters on raw row indices.
    """
    meas_cols = [c for c in MEAS_COLUMNS if c in df.columns]
    agg = {c: 'mean' for c in meas_cols}
    # The censoring a row declares is a property of the variant, not a measurement to average over.
    agg.update({c: 'first' for c in ('cens', 'cens_src', 'dG_bound') if c in df.columns})
    agg['_orig_idx'] = 'first'

    def collapse(frame):
        tmp = frame.copy()
        tmp['_orig_idx'] = tmp.index
        out = tmp.groupby(IDENTITY_COLS, dropna=False).agg(agg).reset_index()
        out.index = out['_orig_idx']
        out.index.name = df.index.name
        return out.drop(columns=['_orig_idx'])

    if 'cens' not in df.columns:
        return collapse(df)

    # A variant whose replicates disagree about whether it could be fitted at all must not have the two averaged:
    # the bounded replicate's dG_ML has been overwritten with the bound itself, so averaging it into a real
    # measurement silently corrupts that measurement (and the mean of -1.0 and a real value is neither). The fitted
    # replicates win outright, and a variant counts as bounded only when every replicate of it was.
    measured, bounded = df.loc[df['cens'] == 0], df.loc[df['cens'] != 0]
    if not len(bounded):
        return collapse(measured)
    m_out = collapse(measured) if len(measured) else measured
    b_out = collapse(bounded)
    if len(m_out):
        seen = set(map(tuple, m_out[IDENTITY_COLS].to_numpy()))
        b_out = b_out.loc[[t not in seen for t in map(tuple, b_out[IDENTITY_COLS].to_numpy())]]
    return pd.concat([m_out, b_out]) if len(m_out) else b_out


class MegaScaleDatasetPreprocessor:
    """Handles raw data loading, preprocessing, and dataset creation."""
    def __init__(self, data_file: str, af_model_folder: str = '/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs/', spurs_override: bool = False,
                 censor_range: bool = True):
        self.data_file = data_file
        self.af_model_folder = af_model_folder
        self.df = pd.DataFrame()
        self.split_dfs = {}
        self.spurs_override = spurs_override
        self.censor_range = censor_range
        # Measured dG of each library's starting sequence (the WT, or the background for mutation-suffixed codes),
        # keyed by `code`. The 'wt' rows are filtered out of self.df, so this is the only place they survive. Each
        # dataset needs it to put an item on the assay's observed scale (dG_obs = h(dG_wt + latent)) and to compute
        # the additive expectation its doubles-conditioned floor estimate uses.
        self.dG_wt: Dict[str, float] = {}
        # Per-library floor statistics, keyed by `code`: where the assay's floor is for that library and whether the
        # library shows any evidence of having one. Computed from the raw table every run (not cached), so the
        # estimator is a free knob. See data.library_floor / data.floor_evidence.
        self.floors: Dict[str, Dict[str, float]] = {}
        self.dataset_floor: float = float('nan')

        self.preprocess()

    @staticmethod
    def _wt_dG_by_code(df: pd.DataFrame) -> Dict[str, float]:
        """
        Each library's wild-type dG, keyed like the `code` column, recovered from the labels themselves.

        ``ddG_ML`` is defined against ONE designated wild-type replicate, not against the mean of them: in 265 of
        the 412 libraries there is a ``wt`` row whose own ``ddG_ML`` is exactly 0, and in all 265 its ``dG_ML`` is
        the constant every other row was referenced to. So ``dG_ML - ddG_ML`` returns that constant from any
        measured row, with a within-library spread of exactly zero, and this is an identity rather than an
        estimate - the median below merely picks one of many identical values.

        Averaging the ``wt`` rows instead (what this used to do) matched it in only 7 of 412 libraries, with a
        median error of 0.015 and a maximum of 0.197 kcal/mol; their median matches in 250 but still misses by up
        to 0.146. Either way ``dG_wt + ddG`` would not equal the item's own ``dG_meas``, which is the axis the
        link places its knee on - and the error is a per-library constant, so it biases a library's whole knee.
        """
        if 'dG_ML' not in df.columns or 'ddG_ML' not in df.columns:
            return {}
        sub = df[['WT_name', 'dG_ML', 'ddG_ML']].copy()
        if 'mut_type' in sub.columns or 'mut_type' in df.columns:
            sub = sub.loc[df['mut_type'] != 'wt'] if 'mut_type' in df.columns else sub
        dg = pd.to_numeric(sub['dG_ML'], errors='coerce')
        ddg = pd.to_numeric(sub['ddG_ML'], errors='coerce')
        ref = (dg - ddg).rename('ref')
        code = (sub['WT_name'].str.replace('.pdb_', '_', regex=False)
                .str.replace('.pdb', '', regex=False).str.replace('|', '_', regex=False))
        ok = ref.notna()
        grouped = ref[ok].groupby(code[ok])
        spread = grouped.std().fillna(0.0)
        loud = spread[spread > 1e-3]
        if len(loud):
            logging.warning(f"dG_wt: {len(loud)} libraries disagree about their own reference "
                            f"(max sd {loud.max():.4f}); the median is used. Worst: {list(loud.nlargest(3).index)}")
        return grouped.median().to_dict()

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

        self.dG_wt = self._wt_dG_by_code(self.df)
        keep = ['aa_seq', 'mut_type', 'WT_name'] + [c for c in MEAS_COLUMNS if c in self.df.columns]
        missing = [c for c in MEAS_COLUMNS if c not in self.df.columns]
        if missing:
            logging.warning(f"Raw table is missing measurement columns {missing}; they will be NaN on every item.")
        self.df = self.df[keep]
        self.df = _tag_range_rows(self.df, self.censor_range)
        for c in MEAS_COLUMNS:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        self.df = _fill_range_values(self.df, self.dG_wt)
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

        self.df = _drop_all_bounded_libraries(self.df)
        self.df = _collapse_replicates(self.df)

        self.df['uid'] = self.df['uid'].str.replace('|', '_', regex=False)
        self.df['code'] = self.df['code'].str.replace('|', '_', regex=False)
        self.df['code_wt'] = self.df['code_wt'].str.replace('|', '_', regex=False)

        base_dir = os.path.join(self.af_model_folder, '')
        self.df['pdb_file'] = base_dir + self.df['code_wt'] + '.pdb'

        logging.info(f"Reduced to: {len(self.df)} entries after basic filtering")

    def _preprocess_spurs(self) -> None:
        # reproduce preprocessing steps from SPURS with compatibility

        cols = ["mut_type", "WT_name", "aa_seq"] + list(MEAS_COLUMNS)
        df = pd.read_csv(self.data_file, usecols=lambda c: c in set(cols), low_memory=False)
        df = _tag_range_rows(df, self.censor_range).reset_index(drop=True)

        df1 = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)
        df2 = df.loc[(~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & df.mut_type.str.contains(":")) | (df.mut_type == 'wt'), :].reset_index(drop=True)
        
        # Concat with ignore_index=False to preserve duplicate indices from df1 and df2
        df = pd.concat([df1,df2], axis=0, ignore_index=False)

        orig_index = df.index

        self.df = df

        self.dG_wt = self._wt_dG_by_code(self.df)
        for c in MEAS_COLUMNS:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        self.df = _fill_range_values(self.df, self.dG_wt)
        self.df = self.df.loc[self.df['ddG_ML'].notna()]

        self.df['mut_structure'] = self.df['WT_name'].apply(lambda x: x.split('.pdb_')[1] if '.pdb_' in x else None)
        self.df['code_wt'] = self.df['WT_name'].apply(lambda x: x.split('.pdb')[0])
        self.df['code'] = self.df.apply(
            lambda row: f"{row['code_wt']}_{row['mut_structure']}" if pd.notnull(row['mut_structure']) else row['code_wt'], 
            axis=1
        )
        self.df['chain'] = 'A'
        self.df['uid'] = self.df['code'].astype(str) + '_' + self.df['mut_type'].astype(str)

        self.df = _drop_all_bounded_libraries(self.df)
        self.df = _collapse_replicates(self.df)

        self.df['uid'] = self.df['uid'].str.replace('|', '_', regex=False)
        self.df['code'] = self.df['code'].str.replace('|', '_', regex=False)
        self.df['code_wt'] = self.df['code_wt'].str.replace('|', '_', regex=False)

        base_dir = os.path.join(self.af_model_folder, '')
        self.df['pdb_file'] = base_dir + self.df['code_wt'] + '.pdb'

        logging.info(f"Total size: {len(self.df)} according to SPURS preprocessing")

        # assert that the indices are retained from the pre-concat state
        # orig_index.equals() will fail if duplicates were actually removed
        assert self.df.index.isin(orig_index).all(), "Index mismatch detected: Indices were flattened or lost."

    def compute_floors(self, percentile: float = 5.0, **evidence_kw) -> None:
        """
        Estimate every library's assay floor in one pass, then shrink the accepted ones toward their median.

        Done here rather than per-dataset because shrinkage and the fallback knee need all the libraries at once,
        and because the 'wt' rows this needs (through ``dG_wt``) are filtered out of the per-library frames.
        """
        if 'code' not in self.df.columns:
            return
        for code, df_c in self.df.groupby('code'):
            self.floors[code] = library_floor(df_c, percentile, self.dG_wt.get(code, float('nan')), str(code),
                                              **evidence_kw)
        self.dataset_floor = shrink_floors(self.floors)

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
        incl_doubles: bool = True,
        incl_cond: bool = True,
        incl_native_cond: bool = True,
        censor_floor: bool = True,
        gap_max=2.0,
        native_cond_as_single: bool = False,
        censor_range: bool = True,
        cache_version: str = None,
        combine_validation: bool = False,
    ) -> Tuple[List[DataLoader], List[str]]:
        """Generates a list of dataloaders for a specific list of protein codes."""
        loaders = []
        loader_names = []
        floors = []

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
       
                dataset = MutationStabilityDataset(
                    dms_df=df_prot_, tokenizer=tokenizer, structure_encoder=structure_encoder,
                    dms_name=code, path=cache_path, score_name=score_name,
                    generate=generate_cache, mut_structs_root=mut_structures_root,
                    incl_destab_bb=incl_destab_bb,
                    incl_singles=incl_singles,
                    incl_doubles=incl_doubles,
                    incl_cond=incl_cond,
                    incl_native_cond=incl_native_cond,
                    dG_wt=self.dG_wt.get(code),
                    floor_stats=self.floors.get(code),
                    censor_floor=censor_floor,
                    gap_max=gap_max,
                    native_cond_as_single=native_cond_as_single,
                    censor_range=censor_range,
                    cache_version=cache_version,
                )
                if len(dataset) == 0:
                        logging.warning(f"{scaffold.capitalize()} dataset for '{code}' is empty. Skipping.")
                        continue
                floors.append((code, dataset.floor_stats.get('y_floor', float('nan')),
                               dataset.floor_stats.get('floor_pctl', float('nan')),
                               dataset.floor_stats.get('evidence', 0.0)))
                        
                loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_twopass,
                                    num_workers=num_workers, shuffle=shuffle, pin_memory=True)
                loaders.append(loader)
                loader_names.append(code)

        _log_floor_summary(scaffold, floors)
        return loaders, loader_names


def _log_floor_summary(scaffold: str, floors: List[Tuple[str, float, float, float]]) -> None:
    """How many libraries of one split show floor evidence, where their floors sit, and how far the percentile
    estimate would have put them - the comparison that motivated the evidence gate."""
    if not floors:
        return
    y = np.array([f[1] for f in floors], dtype=float)
    pc = np.array([f[2] for f in floors], dtype=float)
    ev = np.array([f[3] for f in floors], dtype=float) > 0
    msg = f"floors ({scaffold}): {int(ev.sum())}/{len(floors)} libraries have floor evidence (the rest are not censored)"
    if ev.any():
        msg += (f"; accepted floors median {np.nanmedian(y[ev]):.3f}, "
                f"range [{np.nanmin(y[ev]):.3f}, {np.nanmax(y[ev]):.3f}]")
    both = ev & np.isfinite(pc)
    if both.sum() >= 3:
        d = pc[both] - y[both]
        msg += (f"; the {len(d)} accepted libraries' percentile estimate differs by a median {np.median(d):+.3f} "
                f"(SD {np.std(d):.3f}, correlation {np.corrcoef(pc[both], y[both])[0, 1]:.2f})")
    nopc = (~ev) & np.isfinite(pc)
    if nopc.any():
        msg += (f"; the {int(nopc.sum())} libraries WITHOUT evidence would have been given a percentile floor of "
                f"median {np.nanmedian(pc[nopc]):.3f} (max {np.nanmax(pc[nopc]):.3f}) and censored against it")
    logging.info(msg)


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

                dataset = MutationStabilityDataset(
                    dms_df=df_bench, tokenizer=tokenizer, structure_encoder=structure_encoder,
                    dms_name=name, path=cache_path, score_name=score_name_bench,
                    generate=generate_cache, mut_structs_root='',
                    incl_destab_bb=False, 
                    incl_singles=True,
                    incl_doubles=True,
                    incl_cond=False,
                    incl_native_cond=False,
                    censor_floor=False,
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
    
    preprocessor = MegaScaleDatasetPreprocessor(args.raw_data_file, args.af_model_folder, spurs_override=args.remove_spurs_homologs,
                                                censor_range=args.censor_range)
    preprocessor.compute_floors(percentile=args.floor_percentile, x_max=args.floor_x_max, min_n=args.floor_min_n,
                                max_sd=args.floor_max_sd, band=(args.link_floor_min, args.link_floor_max))
    # Initialises the link's learned global floor at the median of the libraries that do have one (a measured
    # quantity); reaches the module through hparams, since main() builds it from vars(args) after this returns.
    args.dataset_floor = preprocessor.dataset_floor
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
        incl_cond=args.incl_cond,
        incl_native_cond=args.incl_native_cond,
        censor_floor=args.censor_floor,
        gap_max=args.censor_gap_max if args.censor_gap else None,
        native_cond_as_single=args.native_cond_as_single,
        censor_range=args.censor_range,
        cache_version=args.cache_version,
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
        incl_cond=False,
        incl_native_cond=True,
        censor_floor=args.censor_floor,
        gap_max=args.censor_gap_max if args.censor_gap else None,
        native_cond_as_single=args.native_cond_as_single,
        censor_range=args.censor_range,
        cache_version=args.cache_version,
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
        
        consolidated_loader = create_consolidated_dataloader(
            dataloaders=train_dataloaders,
            train_list=train_loader_names,
            batch_size=args.batch_size,
            collate_fn=collate_fn_twopass,
            strategy=args.loader_strategy,
            subset_caps=args.subset_caps,
            num_workers=args.num_workers
        )

        train_dataloaders_final = [consolidated_loader]
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