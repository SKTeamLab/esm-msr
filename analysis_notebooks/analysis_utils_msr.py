import pandas as pd
import numpy as np
import colorsys
import re
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics

from tqdm.notebook import tqdm
from scipy.stats import spearmanr, pearsonr
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import FixedLocator
import matplotlib.ticker as ticker
import matplotlib.collections as mcoll

from scipy.stats import mode as scipy_mode
from scipy.stats import gaussian_kde, pearsonr

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score


# convert names used in inference outputs to those used in figures
remap_names = {
    'esmif_monomer': 'ESM-IF(M)', 
    'esmif_multimer': 'ESM-IF', 
    'mpnn_mean': 'ProteinMPNN mean', 
    'esm2_mean': 'ESM-2 mean',
    'esmif_mean': 'ESM-IF mean',
    'mif_mean': 'MIF mean',
    'msa_transformer_mean': 'MSA-T mean',
    'msa_transformer_median': 'MSA-T median',
    'esm1v_mean': 'ESM-1V mean',
    'esm1v_median': 'ESM-1V median',
    'esm2_150M': 'ESM-2 150M',
    'esm2_650M': 'ESM-2 650M',
    'esm2_3B': 'ESM-2 3B',
    'esm2_15B_half': 'ESM-2 15B',
    'esm3-small-open': 'ESM-3 1.4B',
    'esm3-medium': 'ESM-3 7B',
    'esm3-large': 'ESM-3 98B',
    'esm3_sm': 'ESM-3 1.4B',
    'esm3_med': 'ESM-3 7B',
    'esm3_lg': 'ESM-3 98B',
    'esm-msr': 'ESM-MSR',
    'esm_msr': 'ESM-MSR',
    'esm3_sm_plddt': 'ESM-3 1.4B pLDDT',
    'mif': 'MIF', 
    'mifst': 'MIF-ST', 
    'monomer_ddg': 'Ros_ddG_monomer', 
    'cartesian_ddg': 'Rosetta CartDDG', 
    'mpnn_10_00': 'ProteinMPNN 0.1', 
    'mpnn_20_00': 'ProteinMPNN 0.2', 
    'mpnn_20_00_wt': 'ProteinMPNN 0.2 wt',
    'mpnn_20_00_mut': 'ProteinMPNN 0.2 mut', 
    'mpnn_30_00': 'ProteinMPNN 0.3', 
    'mpnn_sol_10_00': 'ProteinMPNN_sol 0.1', 
    'mpnn_sol_20_00': 'ProteinMPNN_sol 0.2', 
    'mpnn_sol_30_00': 'ProteinMPNN_sol 0.3', 
    'tranception': 'Tranception (reduced)', 
    'tranception_weights': 'Tranception',
    'tranception_original': 'Tranception_original',
    'tranception_reproduced': 'Tranception_reproduced',
    'tranception_target': 'Tranception_target',
    'esm1v_2': 'ESM-1V 2', 
    'msa_1': 'MSA-T 1', 
    'korpm': 'KORPM',
    'Evo': 'EvoEF',
    'msa_transformer_median': 'MSA-T median',
    'ankh': 'Ankh',
    'saprot_pdb': 'SaProt PDB',
    'saprot_af2': 'SaProt AF2',
    'structural': 'Structural',
    'evolutionary': 'Evolutionary',
    'supervised': 'Supervised',
    'clustered_ensemble': 'Clustered Ensemble',
    'mpnn_rosetta': 'Rosetta/ProtMPNN',
    'mutcomputex': 'MutComputeX',
    'stability-oracle': 'Stability Oracle',
    'delta_kdh': 'Δ hydrophobicity', 
    'delta_vol': 'Δ volume', 
    'delta_chg': 'Δ charge',
    'rel_ASA': 'relative ASA',
    'q3421_pslm_rfa_2': 'Ensemble* 2 Feats',
    'q3421_pslm_rfa_3': 'Ensemble* 3 Feats',
    'q3421_pslm_rfa_4': 'Ensemble* 4 Feats',
    'q3421_pslm_rfa_5': 'Ensemble* 5 Feats',
    'q3421_pslm_rfa_6': 'Ensemble* 6 Feats',
    'q3421_pslm_rfa_7': 'Ensemble* 7 Feats',
    'q3421_pslm_rfa_8': 'Ensemble* 8 Feats',
    'K1566_pslm_rfa_2': 'Ensemble 2 Feats',
    'K1566_pslm_rfa_3': 'Ensemble 3 Feats',
    'K1566_pslm_rfa_4': 'Ensemble 4 Feats',
    'K1566_pslm_rfa_5': 'Ensemble 5 Feats',
    'K1566_pslm_rfa_6': 'Ensemble 6 Feats',
    'K1566_pslm_rfa_7': 'Ensemble 7 Feats',
    'K1566_pslm_rfa_8': 'Ensemble 8 Feats',
    'random': 'Random Scores',
    'ddG': 'ΔΔG label', 
    'dTm': 'ΔTm label',
    'upper_bound': 'Theoretical Max',
    'esm3_lora': 'ESM-MSR',
    'thermompnn': 'ThermoMPNN',
    'thermompnn_combined': 'ThermoMPNN',
    'mutate_everything': 'MutateEverything'
    }
#    'random_1': 'Gaussian Noise',

# predictions will have dir in their name to specify direct mutation
remap_names_2 = {f"{key}_dir": value for key, value in remap_names.items()}

remap_cols = {  'auprc': 'AUPRC', 
                'spearman': 'Spearman\'s ρ', 
                'auppc': 'mean PPC', 
                'aumsc': 'mean MSC', 
                'weighted_ndcg': 'wNDCG', 
                'ndcg': 'NDCG',
                'weighted_spearman': 'wρ', 
                'weighted_auprc': 'wAUPRC', 
                'tp': 'True Positives', 
                'sensitivity': 'Sensitivity', 
                'mean_stabilization': 'Mean Stabilization',
                'net_stabilization': 'Net Stabilization',
                'mean_squared_error': 'MSE',
                'accuracy': 'Accuracy', 
                'mean_reciprocal_rank': 'MRR', 
                'n': 'n', 
                'MCC': 'MCC', 
                'recall@k1.0': 'Recall @ k1',
                'recall@k0.0': 'Recall @ k0'
                }

# check if these substrings are in the name of a model in order to assign colors
evolutionary = ['tranception', 'msa_transformer', 'esm1v', 'msa', 'esm2', 'ankh', 'esm3', 'ESM3']
structural = ['mpnn', 'mif', 'mifst', 'esmif', 'mutcomputex', 'sapro', 'ProteinMPNN']
supervised = ['MAESTRO', 'ThermoNet', 'INPS', 'PremPS', 'mCSM', 'DUET', 'I-Mutant3.0', 'SAAFEC', 'MUpro', 'MuPro', 'esm_msr', 'ESM-MSR']
untrained = ['DDGun']
transfer = ['stability-oracle', 'ACDC', 'mutate_everything', 'MutateEverything']
potential = ['KORPM', 'PopMusic', 'SDM', 'korpm', 'PoPMuSiC', 'thermompnnd', 'ThermoMPNN']
biophysical = ['cartesian_ddg', 'FoldX', 'Evo', 'CartDDG', 'Cartesian DDG']
ensemble = ['ens', 'mpnn_rosetta', 'rfa', ' + ']
unknown = ['ddG', 'dTm', 'random', 'delta', 'ASA', 'Dynamut', 'upper_bound']

categories = tuple(['struc. PSLM', 'seq. PSLM', 'transfer', 'biophysical', 'potential', 'untrained', 'supervised', 'unknown', 'unused', 'ensemble'])
colors = tuple(list(sns.color_palette('tab10'))[:len(categories)])
custom_colors = dict(zip(categories, colors)) 

mapping_categories = {  'ensemble': ensemble,
                        'unknown': unknown,
                        'struc. PSLM': structural,
                        'seq. PSLM': evolutionary,
                        'supervised': supervised,
                        'untrained': untrained,
                        'transfer': transfer,
                        'potential': potential,
                        'biophysical': biophysical,
                      }

def determine_category(model):
    for k,v in mapping_categories.items():
        if any(v_ in str(model) for v_ in v):
            category = k
            return category

def determine_base_color(model):
    return custom_colors[determine_category(model)]

def generate_palette(base_color):
    # Generate the base palette
    palette = [sns.light_palette(base_color, n_colors=4, reverse=True)[0]]
    for p in range(1, 4):
        palette.append(sns.light_palette(base_color, n_colors=7, reverse=True)[::2][p])
        palette.append(sns.dark_palette(base_color, n_colors=7, reverse=True)[::2][p])

    # Predefined offsets to create variation
    # Ensure these offsets keep the colors within the [0, 1] range after application
    offsets = [
        (0, 0, 0),  # First color unchanged
        (0.08, 0.08, 0),  # Slightly increase contrast for the second color
        (-0.08, 0.08, -0.),  # Increase contrast for the third color
        (0.07, -0.07, 0.07),  # Significantly alter the fourth color for more distinction
        (-0.06, 0.6, -0.06),  # Minor adjustments for the fifth to balance the palette
        (0.04, -0.04, 0.04),  # Continue with subtle changes
        (-0.04, 0.04, -0.04)  # And further subtle changes
    ]

    # Apply deterministic offsets to each color in the palette
    deterministic_palette_hex = []
    for color, offset in zip(palette, offsets):
        # Adjust each color component within the clipping bounds
        adjusted_color = np.clip(np.array([c + o for c, o in zip(color, offset)]), 0, 1)
        deterministic_palette_hex.append(tuple(adjusted_color))

    return deterministic_palette_hex

# Function to stochastically select a color
def select_color_from_palette(palette, used_colors):
    i = 0
    color = palette[i]
    while color in used_colors:
        i += 1
        color = palette[i]
        #print(i)
    return color

# Function to assign color
def assign_color(model, used_colors, palette):
    selected_color = select_color_from_palette(palette, used_colors)
    return sns.color_palette([selected_color])[0]  # Convert to RGB

def get_color_mapping(data, column='variable'):
    used_colors = set()
    palettes = {}
    color_mapping = {}
    for var in data[column].unique():
        #print(var)
        base_color = determine_base_color(var)
        if base_color in palettes.keys():
            palette = palettes[base_color]
        else:
            palette = generate_palette(base_color)
            palettes[base_color] = palette

        color_mapping[var] = assign_color(var, used_colors, palette)
        used_colors.add(color_mapping[var])
    return color_mapping

def deduplicate_and_check(df, dataset_name, df_name="DataFrame", pred_tolerance=0.1, meas_tolerance=1000):
    """
    Checks dataset size against its name suffix, and deduplicates 
    by taking the mean of numerical columns for duplicate indices,
    applying separate tolerance thresholds for predictions and measurements.
    """
    global diffs
    if df.empty:
        return df
    
    try:
        df = df.set_index('uid')
    except:
        pass
        
    # --- COLUMN FILTERING ---
    metadata_cols = {'code', 'mut_type', 'chain', 'wild_type', 'seq_pos', 'mutation', 'wt_code', 'mut_info', 'prefix'}
    
    cols_to_keep = []
    for col in df.columns:
        if col in metadata_cols:
            cols_to_keep.append(col)
        elif 'ddg' in col.lower() or 'ground_truth' in col:
            cols_to_keep.append(col)
        elif any(model in col for model in ['ThermoMPNN', 'MutateEverything', 'ESM', 'SPURS', 'Rosetta', 'ProteinMPNN']):
            cols_to_keep.append(col)
            
    dropped = set(df.columns) - set(cols_to_keep)
    if dropped:
        print(f"Info: {df_name} dropped irrelevant columns before aggregation: {list(dropped)}")
        df = df[cols_to_keep].copy()
    # ------------------------

    expected_size = None
    match = re.search(r'(\d+)$', dataset_name)
    if match:
        expected_size = int(match.group(1))

    initial_len = len(df)

    if expected_size is not None and initial_len != expected_size:
        print(f"Warning: {df_name} initial length is {initial_len}, expected {expected_size} based on dataset '{dataset_name}'.")

    if df.index.has_duplicates:
        print(f"Warning: {df_name} has duplicate indices. Validating numeric tolerances before deduplicating.")
        
        num_df = df.select_dtypes(include='number')
        
        if not num_df.empty:
            grouped = num_df.groupby(level=0)
            diffs = grouped.max() - grouped.min()
            
            # Isolate ground truth columns from prediction columns to apply separate thresholds
            meas_cols = [c for c in num_df.columns if 'ddg' in c.lower() or 'ground_truth' in c.lower()]
            pred_cols = [c for c in num_df.columns if c not in meas_cols]
            
            violating_meas = (diffs[meas_cols] > meas_tolerance).any(axis=1) if meas_cols else pd.Series(False, index=diffs.index)
            violating_pred = (diffs[pred_cols] > pred_tolerance).any(axis=1) if pred_cols else pd.Series(False, index=diffs.index)
            
            violating_mask = violating_meas | violating_pred
            
            if violating_mask.any():
                violating_indices = diffs[violating_mask].index.tolist()
                print(diffs[violating_mask])
                raise AssertionError(
                    f"Fatal Error: {df_name} contains duplicate indices with differing numerical values "
                    f"exceeding tolerances (Measurement: {meas_tolerance}, Prediction: {pred_tolerance}). "
                    f"Cannot safely aggregate. Violating indices: {violating_indices[:10]}"
                    f"{'...' if len(violating_indices) > 10 else ''}"
                )
            
            num_df_processed = grouped.mean()
        else:
            # Handle edge case where DataFrame only has string columns
            num_df_processed = pd.DataFrame(index=df.index.unique())
            
        str_df = df.select_dtypes(exclude='number').groupby(level=0).first()
        df = num_df_processed.join(str_df)
    
    final_len = len(df)
    
    if expected_size is not None:
        if final_len > expected_size:
            raise AssertionError(f"Fatal Error: After deduplication, {df_name} length is {final_len}, which is strictly greater than expected {expected_size}.")
        elif final_len < expected_size:
            print(f"Warning: After deduplication, {df_name} length is {final_len}, which is less than expected {expected_size}.")
    else:
        print(f"Info: {df_name} ('{dataset_name}') length went from {initial_len} to {final_len} after deduplication.")
        
    return df


def assess_all_models(df_preds, df_scores, func, 
                      splits=('hyperopt_splits-S-test', 'hyperopt_splits-D-test', 'hyperopt_splits-test'),
                      models=('ESM-MSR_1', 'ESM-MSR_2', 'ESM-MSR_3', 
                              'ESM-MSR_alpha_med_1', 'ESM-MSR_alpha_med_2', 'ESM-MSR_alpha_med_3', 
                              'ESM-MSR_alpha_low_1', 'ESM-MSR_alpha_low_2', 'ESM-MSR_alpha_low_3',
                              'ESM-MSR_single_1', 'ESM-MSR_single_2', 'ESM-MSR_single_3', 
                              'ESM-MSR_reg_1', 'ESM-MSR_reg_2', 'ESM-MSR_reg_3',
                              'ESM-MSR_masked_1', 'ESM-MSR_masked_2', 'ESM-MSR_masked_3',
                              'ESM-MSR_chain_1', 'ESM-MSR_chain_2', 'ESM-MSR_chain_3', 
                              'SPURS_1', 'SPURS_2', 'SPURS_3', 
                              'ThermoMPNN(-D)_1', 'ThermoMPNN(-D)_2', 'ThermoMPNN(-D)_3', 
                              'MutateEverything_1', 'MutateEverything_2', 'MutateEverything_3', 
                              'ESM3-small-open', 'ESM3-small-open_chain',
                              'ESM3-small', 'ESM3-medium', 'ESM3-large', 
                              'Rosetta Cartesian DDG_1', 'Rosetta Cartesian DDG_2', 'Rosetta Cartesian DDG_3', 
                              'ProteinMPNN'),
                      add_suffix=''
                     ):
    
    for model in models:
        print(model)
        # We start with the original name
        target_model = model
        
        if add_suffix != '':
            # Identify if it's a replicate model (ends in _1, _2, _3)
            # We check the penultimate character for '_'
            if len(model) > 2 and model[-2] == '_':
                orig_suffix = model[-2:]  # e.g., "_1"
                model_base = model[:-2]   # e.g., "ESM-MSR" or "ThermoMPNN(-D)"
                
                # Check if the suffix is already there to avoid "_additive_additive"
                if not model_base.endswith(add_suffix):
                    target_model = f"{model_base}{add_suffix}{orig_suffix}"
            else:
                # Standard model without _N suffix
                if not model.endswith(add_suffix):
                    target_model = model + add_suffix

        # Filtering logic per your requirements
        if '_additive_additive' in target_model:
            continue

        # Validation: Ensure the column actually exists in your data
        if target_model not in df_preds.columns:
            print(f"Error: Calculated model name '{target_model}' not found in df_preds.")
            # Raising error as requested instead of silent failure
            raise AssertionError(f"Column {target_model} missing from input DataFrame.")

        for split in splits:
            if "-S-" in split:
                df_subset = df_preds.loc[~df_preds['mut_type'].str.contains(':')]
            elif "-D-" in split:
                df_subset = df_preds.loc[df_preds['mut_type'].str.contains(':')]
            else:             
                df_subset = df_preds
            
            # Using target_model for both the model identifier and the column name
            df_scores = func(df_subset, df_scores, target_model, dataset=split, pred_col=target_model, label='ddG')

    return df_scores


def assess_classification_metric(
    df_preds: pd.DataFrame, 
    df_scores: pd.DataFrame, 
    name: str, 
    dataset: str, 
    metric_name: str,
    pred_col: str = 'esm_msr', 
    label: str = 'ddG_ML', 
    quiet: bool = True,
    label_threshold: float = 0.0,
    pred_threshold: float = 0.0,
    flip_labels: bool = False
) -> pd.DataFrame:
    """
    Evaluates a specific classification metric and updates the scores dataframe.
    """
    assert pred_col in df_preds.columns and label in df_preds.columns, f"Missing prediction ({pred_col}) or label ({label}) columns in df_preds"
    
    metric_name = metric_name.lower()
    
    # Map string names to their respective sklearn metric functions
    metrics_map = {
        'accuracy': accuracy_score,
        'auroc': roc_auc_score,
        'auprc': average_precision_score,
        'mcc': matthews_corrcoef,
        'f1': f1_score
    }
    
    if metric_name not in metrics_map:
        raise NotImplementedError(f"Metric '{metric_name}' is not implemented. Please choose from {list(metrics_map.keys())}.")
        
    # Automatically binarize the ground truth labels
    y_true = (df_preds[label] >= label_threshold).astype(int)
    y_pred_scores = df_preds[pred_col]
    
    # Metrics like accuracy, MCC, and F1 require hard class labels (0 or 1)
    if metric_name in ['accuracy', 'mcc', 'f1']:
        y_pred = (y_pred_scores >= pred_threshold).astype(int)
    else:
        y_pred = y_pred_scores

    if flip_labels:
        # Invert the ground truth (1 becomes 0, 0 becomes 1)
        y_true = 1 - y_true
        
        # Invert the predictions to match the new positive class
        if metric_name in ['accuracy', 'mcc', 'f1']:
            y_pred = 1 - y_pred
        else:
            # For threshold-agnostic metrics (AUROC/AUPRC), invert the ranking direction
            y_pred = -y_pred
            
    try:
        score = metrics_map[metric_name](y_true, y_pred)
    except ValueError as e:
        print(f"Failed to calculate {metric_name.upper()}. Ensure your labels are valid.")
        raise e

    if not quiet:
        print(f'{metric_name.upper()} for {dataset}: {score:.4f}')
        
    df_scores.loc[(f'{dataset}', 'ungrouped'), name] = score
    df_scores.loc[(f'{dataset}', 'ungrouped'), f'{name}_n'] = len(y_true)

    return df_scores

def assess_grouped_spearman(df_preds, df_scores, name, dataset, pred_col='esm_msr', label='ddG_ML', quiet=True):
    # 1. Structural Robustness: Ensure all required columns exist
    required_cols = [pred_col, label, 'code']
    for col in required_cols:
        if col not in df_preds.columns:
            # Raising error as requested in saved preferences
            raise AssertionError(f"Column '{col}' missing from df_preds")

    # 2. Alignment Robustness: Keep 'code' attached to the data during filtering
    # This prevents the "duplicate labels" reindexing error
    valid_data = df_preds[required_cols].dropna(subset=[label, pred_col])

    if valid_data.empty:
        if not quiet:
            print(f"Warning: No valid data for {dataset} - {name}")
        return df_scores

    # --- Ungrouped Calculation ---
    ungrouped_rho = valid_data[[label, pred_col]].corr('spearman').iloc[0, 1]
    df_scores.loc[(dataset, 'ungrouped'), name] = ungrouped_rho
    df_scores.loc[(dataset, 'ungrouped'), name + '_n'] = len(valid_data)

    # --- Grouped Calculation ---
    # We group by the column name directly, so index alignment is irrelevant
    grouped = valid_data.groupby('code')
    
    # Calculate spearman for each group more cleanly
    # Note: .corr() returns a Series with a MultiIndex; we select the cross-correlation
    def get_spearman(group):
        if len(group) < 2:
            return np.nan
        return group[label].corr(group[pred_col], method='spearman')

    corrs = grouped.apply(get_spearman, include_groups=False)
    
    # Remove groups that couldn't produce a correlation (e.g., n < 2 or zero variance)
    valid_corrs = corrs.dropna()

    if not quiet:
        print(f'Ungrouped rho for {dataset}: {ungrouped_rho:.4f}')
        print(f'Grouped rho for {dataset}: {valid_corrs.mean():.4f}')

    df_scores.loc[(dataset, 'grouped'), name] = valid_corrs.mean()
    
    # Average n observations across groups and total number of groups
    group_counts = grouped.size()
    df_scores.loc[(dataset, 'grouped'), name + '_n'] = group_counts.mean()
    df_scores.loc[(dataset, 'grouped'), name + '_groups'] = len(group_counts)

    return df_scores

def assess_grouped_gain(df_preds, df_scores, name, dataset, pred_col='esm_msr', label='ddG_ML', quiet=True, n=96, rectified=True):
    assert pred_col in df_preds.columns and label in df_preds.columns, "Missing prediction or label columns in df_preds"

    tmp = df_preds.copy(deep=True)
    tmp = tmp.sort_values(pred_col, ascending=False)
    tmp = tmp.head(n)
    if rectified:
        tmp.loc[tmp[label]<0, label] = 0
    if not quiet:
        print(f'Ungrouped rho for {dataset} splits test data')
        print(tmp[label].sum())
        
    df_scores.loc[(f'{dataset}', 'ungrouped'), name] = tmp[label].sum()
    # Using len(tmp) instead of parameter 'n' in case the actual dataframe has fewer rows than 'n'
    df_scores.loc[(f'{dataset}', 'ungrouped'), name+'_n'] = len(tmp)

    groups = 0
    gain = 0
    group_ns = []

    for code, group in df_preds.groupby('code'):
        tmp = group.copy(deep=True)
        tmp = tmp.sort_values(pred_col, ascending=False)
        tmp = tmp.head(n)
        if rectified:
            tmp.loc[tmp[label]<0, label] = 0
        if not quiet:
            print(f'Ungrouped rho for {dataset} splits test data')
            print(tmp[label].sum())
            
        groups += 1
        gain += tmp[label].sum()
        group_ns.append(len(tmp))

    assert groups > 0, f"No valid groups found for dataset {dataset}"

    df_scores.loc[(f'{dataset}', 'grouped'), name] = gain / groups
    df_scores.loc[(f'{dataset}', 'grouped'), name+'_n'] = sum(group_ns) / groups
    df_scores.loc[(f'{dataset}', 'grouped'), name+'_groups'] = groups

    return df_scores

def assess_grouped_ndcg(df_preds, df_scores, name, dataset, pred_col='esm_msr', label='ddG_ML', top_n=None, threshold=None):
    """
    Unified wrapper to evaluate NDCG and update a scoring dataframe.
    Handles both grouped (per-protein) and ungrouped (global) evaluations.
    Requires exactly one of `top_n` or `threshold` to be specified.
    """
    if sum([top_n is not None, threshold is not None]) != 1:
        raise ValueError("Specify exactly one of top_n or threshold.")

    # Drop rows without ground truth labels to prevent length mismatches
    df_eval = df_preds.dropna(subset=[label, pred_col]).copy()
    assert len(df_eval) > 0, f"Dataframe became empty after dropping NaNs for dataset {dataset}"

    # ==========================================
    # 1. UNGROUPED EVALUATION (Global Dataset)
    # ==========================================
    # Pass the ENTIRE dataframe. Do not pre-truncate.
    u_ndcg, u_model_hits, u_ideal_hits, u_total_pool = compute_ndcg_flexible(
        df_eval, pred_col, label, top_n=top_n, threshold=threshold
    )
    
    df_scores.loc[(f'{dataset}', 'ungrouped'), name] = u_ndcg
    # Store the relevant available hit count
    df_scores.loc[(f'{dataset}', 'ungrouped'), f'{name}_n'] = u_ideal_hits if top_n is not None else u_total_pool

    # ==========================================
    # 2. GROUPED EVALUATION (Per-Protein)
    # ==========================================
    g_ndcgs = []
    g_ns = []
    
    for code, group in df_eval.groupby('code'):
        g_ndcg, g_model_hits, g_ideal_hits, g_total_pool = compute_ndcg_flexible(
            group, pred_col, label, top_n=top_n, threshold=threshold
        )
        
        g_ndcgs.append(g_ndcg)
        g_ns.append(g_ideal_hits if top_n is not None else g_total_pool)

    # Convert to Series to safely calculate mean (ignores np.nan from 0-hit proteins)
    s_ndcgs = pd.Series(g_ndcgs)
    s_ns = pd.Series(g_ns)
    
    df_scores.loc[(f'{dataset}', 'grouped'), name] = s_ndcgs.mean()
    df_scores.loc[(f'{dataset}', 'grouped'), f'{name}_n'] = s_ns.mean()
    df_scores.loc[(f'{dataset}', 'grouped'), f'{name}_groups'] = len(s_ndcgs)

    return df_scores


def sum_individual_mutation_scores_with_missing(df, score_column, new_score_column=None):
    """
    Vectorized version for summing individual mutation scores for combined mutations.
    
    For mutations with a colon in the mut_type column (indicating combined mutations),
    find the rows where both mut_type AND code match, and sum their scores.
    
    This implementation is fully vectorized and avoids all row-by-row iterations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing mutation data
    score_column : str
        Column name containing the scores to sum
    new_score_column : str, optional
        Column name for the summed scores. If None, defaults to f"{score_column}_additive"
        
    Returns:
    --------
    pandas.DataFrame
        Copy of input DataFrame with the new score column added
    """
    
    # Create a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Set default name for the new score column if not provided
    if new_score_column is None:
        new_score_column = f"{score_column}_additive"
    
    # Initialize the new column with NaN values
    result_df[new_score_column] = np.nan
    
    # Find rows with a colon in mut_type (combined mutations)
    combined_mutation_mask = result_df['mut_type'].str.contains(':', na=False)
    
    # Only process rows with combined mutations
    if combined_mutation_mask.sum() == 0:
        return result_df  # No combined mutations found
    
    # Extract all combined mutations
    combined_df = result_df[combined_mutation_mask].copy()
    
    # Split each combined mutation into exactly 2 parts
    split_mutations = combined_df['mut_type'].str.split(':', expand=True)
    
    # Filter to only process rows with exactly 2 mutations
    if split_mutations.shape[1] < 2:
        print("Warning: No valid combined mutations with exactly 2 parts found")
        return result_df
    
    # Keep only rows with exactly 2 mutations (non-null in both columns)
    valid_mask = split_mutations[0].notna() & split_mutations[1].notna()
    if split_mutations.shape[1] > 2:
        # Check that there are no additional mutations beyond the first 2
        for col in range(2, split_mutations.shape[1]):
            valid_mask &= split_mutations[col].isna()
    
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"Warning: {n_invalid} rows don't have exactly 2 mutations and will be skipped")
        combined_df = combined_df[valid_mask]
        split_mutations = split_mutations[valid_mask]
    
    if len(combined_df) == 0:
        return result_df
    
    # Create lookup keys for both individual mutations
    combined_df = combined_df.copy()  # Avoid SettingWithCopyWarning
    combined_df['mutation1'] = split_mutations[0].values
    combined_df['mutation2'] = split_mutations[1].values
    
    # Create key columns for vectorized lookup
    combined_df['key1'] = combined_df['mutation1'] + '|' + combined_df['code']
    combined_df['key2'] = combined_df['mutation2'] + '|' + combined_df['code']
    
    # Create lookup dictionary once
    df_lookup = df.copy()
    df_lookup['lookup_key'] = df_lookup['mut_type'] + '|' + df_lookup['code']
    lookup_dict = df_lookup.set_index('lookup_key')[score_column].to_dict()
    
    # Vectorized lookup for both mutations
    score1_series = combined_df['key1'].map(lookup_dict)
    score2_series = combined_df['key2'].map(lookup_dict)
    
    # Calculate sum only where both scores exist
    both_exist_mask = score1_series.notna() & score2_series.notna()
    summed_scores = score1_series + score2_series
    
    # Update the result dataframe with vectorized assignment
    valid_indices = combined_df.index[both_exist_mask]
    result_df.loc[valid_indices, new_score_column] = summed_scores[both_exist_mask]
    
    # Optional: Print statistics about missing mutations
    missing_mask = ~both_exist_mask

    missing_df = pd.DataFrame()

    if missing_mask.any():
        n_missing = missing_mask.sum()
        print(f"Warning: {n_missing} combined mutations couldn't be processed due to missing individual mutations")
        
        # If you want detailed info about what's missing (comment out for speed):
        missing_df = combined_df[missing_mask]
        missing1 = score1_series[missing_mask].isna()
        missing2 = score2_series[missing_mask].isna()
        # if missing1.any():
        #     print(f"  - {missing1.sum()} missing first mutations")
        # if missing2.any():
        #     print(f"  - {missing2.sum()} missing second mutations")
    
    return result_df, missing_df, missing1, missing2


def sum_individual_mutation_scores(df, score_column, new_score_column=None):
    """
    Vectorized version for summing individual mutation scores for combined mutations.
    
    For mutations with a colon in the mut_type column (indicating combined mutations),
    find the rows where both mut_type AND code match, and sum their scores.
    
    This implementation is fully vectorized and avoids all row-by-row iterations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing mutation data
    score_column : str
        Column name containing the scores to sum
    new_score_column : str, optional
        Column name for the summed scores. If None, defaults to f"{score_column}_additive"
        
    Returns:
    --------
    pandas.DataFrame
        Copy of input DataFrame with the new score column added
    """
    
    # Create a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Set default name for the new score column if not provided
    if new_score_column is None:
        new_score_column = f"{score_column}_additive"
    
    # Initialize the new column with NaN values
    result_df[new_score_column] = np.nan
    
    # Find rows with a colon in mut_type (combined mutations)
    combined_mutation_mask = result_df['mut_type'].str.contains(':', na=False)
    
    # Only process rows with combined mutations
    if combined_mutation_mask.sum() == 0:
        return result_df  # No combined mutations found
    
    # Extract all combined mutations
    combined_df = result_df[combined_mutation_mask].copy()
    
    # Split each combined mutation into exactly 2 parts
    split_mutations = combined_df['mut_type'].str.split(':', expand=True)
    
    # Filter to only process rows with exactly 2 mutations
    if split_mutations.shape[1] < 2:
        print("Warning: No valid combined mutations with exactly 2 parts found")
        return result_df
    
    # Keep only rows with exactly 2 mutations (non-null in both columns)
    valid_mask = split_mutations[0].notna() & split_mutations[1].notna()
    if split_mutations.shape[1] > 2:
        # Check that there are no additional mutations beyond the first 2
        for col in range(2, split_mutations.shape[1]):
            valid_mask &= split_mutations[col].isna()
    
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"Warning: {n_invalid} rows don't have exactly 2 mutations and will be skipped")
        combined_df = combined_df[valid_mask]
        split_mutations = split_mutations[valid_mask]
    
    if len(combined_df) == 0:
        return result_df
    
    # Create lookup keys for both individual mutations
    combined_df = combined_df.copy()  # Avoid SettingWithCopyWarning
    combined_df['mutation1'] = split_mutations[0].values
    combined_df['mutation2'] = split_mutations[1].values
    
    # Create key columns for vectorized lookup
    combined_df['key1'] = combined_df['mutation1'] + '|' + combined_df['code']
    combined_df['key2'] = combined_df['mutation2'] + '|' + combined_df['code']
    
    # Create lookup dictionary once
    df_lookup = df.copy()
    df_lookup['lookup_key'] = df_lookup['mut_type'] + '|' + df_lookup['code']
    lookup_dict = df_lookup.set_index('lookup_key')[score_column].to_dict()
    
    # Vectorized lookup for both mutations
    score1_series = combined_df['key1'].map(lookup_dict)
    score2_series = combined_df['key2'].map(lookup_dict)
    
    # Calculate sum only where both scores exist
    both_exist_mask = score1_series.notna() & score2_series.notna()
    summed_scores = score1_series + score2_series
    
    # Update the result dataframe with vectorized assignment
    valid_indices = combined_df.index[both_exist_mask]
    result_df.loc[valid_indices, new_score_column] = summed_scores[both_exist_mask]
    
    # Optional: Print statistics about missing mutations
    missing_mask = ~both_exist_mask
    if missing_mask.any():
        n_missing = missing_mask.sum()
        print(f"Warning: {n_missing} combined mutations couldn't be processed due to missing individual mutations")
        
    return result_df


def unify_similar_columns(df, delimiter='', quiet=True):
    """
    Identify columns that differ only by a numeric suffix and replace them with 
    unified columns containing the mean and standard deviation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    delimiter : str, default=''
        The delimiter between the column base name and the numeric suffix.
        For example, for columns like 'value_1', 'value_2', use delimiter='_'
        For columns like 'value1', 'value2', use delimiter=''
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with similar columns replaced by mean and std columns
    """
    import pandas as pd
    import re
    import numpy as np
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Find column groups that differ only by numeric suffixes
    column_groups = {}
    pattern = re.compile(r'(.+' + re.escape(delimiter) + r')(\d+)$')

    for col in df.columns:
        if not quiet:
            print(col)
        match = pattern.match(col)
        if match:
            base_name = match.group(1)
            if base_name not in column_groups:
                column_groups[base_name] = []
            column_groups[base_name].append(col)
    
    # Process each group of similar columns
    for base_name, columns in column_groups.items():
        # Skip if only one column with this base name
        if len(columns) <= 1:
            continue
        
        # Calculate mean and standard deviation
        mean_values = result_df[columns].mean(axis=1)
        std_values = result_df[columns].std(axis=1)
        
        # Add new columns
        result_df[f'{base_name}mean'] = mean_values
        result_df[f'{base_name}std'] = std_values
        
        # Remove the original columns
        result_df = result_df.drop(columns=columns)
    
    return result_df


def create_triple_metric_chart_with_pvals(
    series_bar1,
    series_bar2,
    series_overall,
    model_groups,
    *,
    highlighted_model=None,
    title=None,
    y_label=None,
    figsize=(14, 6),
    ylim=(0, 1),
    bar_width=0.8,
    shadow_alpha=0.15,
    hatch_bar1="///",
    hatch_bar2=None,
    edgecolor="black",
    fontsize_multiplier=1.3,
    capsize=4,
    base_colors=None,
    highlight_gray='#e0e0e0',
    error_bar_color='red',
    enforce_global_n=True,
    is_grouped=True,
    label_min_sep=0.05,
    p_value_y_coord=None,
    legend_loc="upper right",
    label_bar1="Singles",
    label_bar2="Doubles",
    label_overall="Overall"
):
    """
    Plots two adjacent bars with an overarching shadow background bar.
    Utilizes Standard Error (SEM) for error bars and annotations.
    Preserves p-value calculations against a highlighted model for the overall series.
    """
    
    only_overall = (series_bar1 is None and series_bar2 is None)
    
    def _collect(s, base):
        if s is None: return None
        reps = [k for k in s.index if re.fullmatch(rf"{re.escape(base)}_?\d+", k)]
        values = None
        
        if reps:
            values = s[reps].astype(float).values
        elif base in s.index:
            values = np.array([float(s[base])])

        if values is None or len(values) == 0 or not np.all(np.isfinite(values)):
            return None

        n = int(len(values))
        mean = float(np.mean(values))
        # Standard Error of the Mean (SEM) = std / sqrt(n)
        if n > 1:
            sem = float(np.std(values, ddof=1) / np.sqrt(n))
        else:
            sem = 0.0

        return {
            "mean": mean,
            "err": sem,
            "n": n,
            "has_replicates": n > 1,
            "values": values,
        }

    def _get_stats_from_series(s, base):
        if s is None: return None
        
        reps = [k for k in s.index if re.fullmatch(rf"{re.escape(base)}_?\d+", k)]
        keys_used = reps if reps else ([base] if base in s.index else [])
        
        if not keys_used:
            return None # Model not present in this series
            
        stats_dict = {}
        vals_n = []
        vals_g = []
        
        for k in keys_used:
            for stat_key in [f"{k}_n", f"{base}_n"]:
                if stat_key in s.index:
                    try: 
                        vals_n.append(float(s[stat_key]))
                        break
                    except ValueError as e: 
                        raise AssertionError(f"Expected numeric value for {stat_key}") from e
                        
            if is_grouped:
                for stat_key in [f"{k}_groups", f"{base}_groups"]:
                    if stat_key in s.index:
                        try: 
                            vals_g.append(int(s[stat_key]))
                            break
                        except ValueError as e: 
                            raise AssertionError(f"Expected numeric value for {stat_key}") from e
                            
        if vals_n:
            stats_dict['n'] = max(set(vals_n), key=vals_n.count)
            
        if is_grouped:
            if vals_g:
                stats_dict['groups'] = max(set(vals_g), key=vals_g.count)
            else:
                raise AssertionError(f"is_grouped=True but missing groups key for '{base}' in series.")
                
        return stats_dict if stats_dict else None

    def _determine_consensus_stats(s1, s2, sov, model_groups, enforce=True):
        def _consensus(s, tag):
            if s is None: return None
            vals_n = []
            vals_g = []
            for m in model_groups:
                st = _get_stats_from_series(s, m)
                if st:
                    if 'n' in st: vals_n.append(st['n'])
                    if 'groups' in st: vals_g.append(st['groups'])
                
            res = {}
            if vals_n:
                uniq_n = sorted(set(vals_n))
                if enforce and len(uniq_n) > 1:
                    raise AssertionError(f"Inconsistent {tag} n: {uniq_n}")
                res['n'] = uniq_n[0]
                
            if vals_g:
                uniq_g = sorted(set(vals_g))
                if enforce and len(uniq_g) > 1:
                    raise AssertionError(f"Inconsistent {tag} groups: {uniq_g}")
                res['groups'] = uniq_g[0]
                
            return res if res else None
            
        return _consensus(s1, 'bar1'), _consensus(s2, 'bar2'), _consensus(sov, 'overall')

    stats_bar1, stats_bar2, stats_overall = _determine_consensus_stats(
        series_bar1, series_bar2, series_overall, model_groups, enforce=enforce_global_n
    )
    
    base_color_list = base_colors or ['#d35400', '#34495e', '#3498db', '#2ecc71', '#bbbbbb', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    model_color_map = {}
    color_index = 0
    for m in model_groups:
        if highlighted_model is not None and m == highlighted_model:
            model_color_map[m] = None
        else:
            model_color_map[m] = base_color_list[color_index % len(base_color_list)]
            color_index += 1

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(model_groups))
    half = bar_width / 2.0
    
    overall_data = {}
    per_model_label_entries = {m: [] for m in model_groups}
    valid_data_found = False

    # Step 1: Data Collection & Bar Drawing
    for i, model in enumerate(model_groups):
        base_col = model_color_map.get(model)
        is_highlight = (highlighted_model is not None and model == highlighted_model)
        lw = 2.5 if is_highlight else 1.2

        overall = _collect(series_overall, model)
        b1 = _collect(series_bar1, model)
        b2 = _collect(series_bar2, model)

        # Plot Overall (Wide Shadow Bar / Or Solid Single Bar)
        if overall is not None:
            valid_data_found = True
            overall_data[model] = {**overall, 'xpos': x[i]}
            
            if only_overall:
                # Single bar mode: pastel color, highlight with yellow border
                if is_highlight:
                    overall_color = highlight_gray
                    overall_edge = 'yellow'
                else:
                    overall_color = tuple(0.5 + 0.5 * c for c in mcolors.to_rgb(base_col))
                    overall_edge = edgecolor
                overall_alpha = 1.0
            else:
                overall_color = highlight_gray if is_highlight else (base_col or highlight_gray)
                overall_edge = edgecolor
                overall_alpha = shadow_alpha
            
            # Draw standard shadow bar behind the inner bars
            ax.bar(x[i], overall["mean"], width=bar_width, color=overall_color, alpha=overall_alpha, 
                   edgecolor=overall_edge, linewidth=lw, zorder=1)
            
            if only_overall and overall["has_replicates"]:
                ax.errorbar(x[i], overall["mean"], yerr=overall["err"], fmt="none", 
                            ecolor=error_bar_color, elinewidth=lw, capsize=capsize, zorder=3)
            
            # Detect obstruction (>95% magnitude on all existing foreground bars)
            obstructed = False
            bars_present = []
            if b1 is not None: bars_present.append(b1["mean"])
            if b2 is not None: bars_present.append(b2["mean"])
            
            if bars_present:
                if overall["mean"] >= 0:
                    obstructed = all(val > overall["mean"] * 0.95 for val in bars_present)
                else:
                    obstructed = all(val < overall["mean"] * 0.95 for val in bars_present)
            
            if obstructed:
                # White box in front to visually lighten the foreground bars
                # WARNING: zorder=4 will draw over error bars (zorder=3)
                ax.bar(x[i], overall["mean"], width=bar_width, color='white', alpha=0.4, 
                       edgecolor='white', linewidth=lw, zorder=4)

            per_model_label_entries[model].append((x[i], overall["mean"], overall["err"], "overall", model))

        # Plot Bar 1 (Left narrow bar)
        if b1 is not None:
            valid_data_found = True
            c_b1 = 'black' if is_highlight else base_col
            edge_b1 = '#bbbbbb' if is_highlight else edgecolor
            hatch = '///' if is_highlight else hatch_bar1
            
            ax.bar(x[i] - half/2.0, b1["mean"], width=half, color=c_b1, hatch=hatch, 
                   edgecolor=edge_b1, linewidth=lw, zorder=2)
            
            if b1["has_replicates"]:
                ax.errorbar(x[i] - half/2.0, b1["mean"], yerr=b1["err"], fmt="none", 
                            ecolor=error_bar_color, elinewidth=lw, capsize=capsize, zorder=3)
                            
            if is_highlight:
                ax.bar(x[i] - half/2.0, b1["mean"], width=half, color='none', edgecolor='yellow', linewidth=lw, zorder=2.5)
                
            per_model_label_entries[model].append((x[i] - half/2.0, b1["mean"], b1["err"], "bar1", model))

        # Plot Bar 2 (Right narrow bar)
        if b2 is not None:
            valid_data_found = True
            if is_highlight:
                c_b2 = '#444444'
                edge_b2 = 'yellow'
            else:
                rgb = mcolors.to_rgb(base_col)
                c_b2 = tuple(0.5 + 0.5 * c for c in rgb)
                edge_b2 = edgecolor
                
            ax.bar(x[i] + half/2.0, b2["mean"], width=half, color=c_b2, hatch=hatch_bar2, 
                   edgecolor=edge_b2, linewidth=lw, zorder=3)
                   
            if b2["has_replicates"]:
                ax.errorbar(x[i] + half/2.0, b2["mean"], yerr=b2["err"], fmt="none", 
                            ecolor=error_bar_color, elinewidth=lw, capsize=capsize, zorder=3)
                            
            per_model_label_entries[model].append((x[i] + half/2.0, b2["mean"], b2["err"], "bar2", model))

    if not valid_data_found:
        raise AssertionError("No valid data found matching the provided model_groups in the series.")

    ax.set_xticks(x)
    ax.set_xticklabels(model_groups, rotation=45, ha="right", fontsize=10 * fontsize_multiplier)
    if y_label: ax.set_ylabel(y_label, fontsize=16 * fontsize_multiplier, fontweight="bold")
    if title: ax.set_title(title, fontsize=16 * fontsize_multiplier, fontweight="bold")

    # Step 2: P-Value Calculation (Only on Overall data)
    pvals_overall = {m: np.nan for m in model_groups}
    if highlighted_model is not None and highlighted_model in overall_data:
        hl = overall_data[highlighted_model]
        pvals_overall[highlighted_model] = 1.0
        for m in model_groups:
            if m not in overall_data or m == highlighted_model: continue
            cur = overall_data[m]
            if hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_ind(cur['values'], hl['values'], equal_var=False)
            elif hl['has_replicates'] and not cur['has_replicates']:
                _, p = stats.ttest_1samp(hl['values'], cur['mean'])
            elif not hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_1samp(cur['values'], hl['mean'])
            else: 
                print(f"Warning: Cannot calculate overall p-value for '{m}' vs '{highlighted_model}' (both lack replicates).")
                p = np.nan
            pvals_overall[m] = p

    # Step 3: Text Annotation & Collision Handling
    span_now = ylim[1] - ylim[0] if ylim else (ax.get_ylim()[1] - ax.get_ylim()[0])
    per_model_ytexts = {m: [] for m in model_groups}

    for m in model_groups:
        for (xpos, mean, err, kind, model_name) in per_model_label_entries[m]:
            if kind == 'overall' and p_value_y_coord is not None:
                ytext, va = p_value_y_coord, 'center'
            else:
                offset = 0.02 * span_now
                if mean >= 0: ytext, va = mean + err + offset, 'bottom'
                else: ytext, va = mean - err - offset, 'top'
            
            if kind == 'overall':
                p = pvals_overall.get(model_name, np.nan)
                p_text = "NA" if np.isnan(p) else f"{p:.2e}"
                
                if err > 1e-9: # Safe comparison for zero err
                    text = f"{mean:.3f}±{err:.3f}\np={p_text}"
                else:
                    text = f"{mean:.3f}\np={p_text}"
            else:
                text = f"{mean:.3f}"
                
            per_model_ytexts[m].append([xpos, ytext, text, va])

    # Note: This is an O(N^2) greedy collision resolution. 
    extra_offset = 0.025 * span_now
    for m, entries in per_model_ytexts.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if abs(entries[i][1] - entries[j][1]) < label_min_sep:
                    if entries[i][1] >= entries[j][1]: entries[i][1] += extra_offset
                    else: entries[j][1] += extra_offset

    for m, entries in per_model_ytexts.items():
        for xpos, ytext, text, va in entries:
            ax.text(xpos, ytext, text, ha='center', va=va, fontsize=10 * fontsize_multiplier, fontweight='bold')

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if ylim: ax.set_ylim(ylim)
    
    # Construct Legend dynamically based on extracted Ns and Groups
    def _format_legend(base_label, stats_dict):
        if not stats_dict: return base_label
        parts = []
        if is_grouped:
            if 'n' in stats_dict and stats_dict['n'] is not None:
                val = stats_dict['n']
                parts.append(f"avg_muts={int(val) if float(val).is_integer() else f'{float(val):.1f}'}")
            if 'groups' in stats_dict and stats_dict['groups'] is not None:
                parts.append(f"n_domains={stats_dict['groups']}")
        else:
            if 'n' in stats_dict and stats_dict['n'] is not None:
                val = stats_dict['n']
                parts.append(f"n={int(val) if float(val).is_integer() else f'{float(val):.1f}'}")
                
        return f"{base_label} ({', '.join(parts)})" if parts else base_label

    lbl_ov = _format_legend(label_overall, stats_overall)
    
    if only_overall:
        legend_elements = [
            Patch(facecolor=highlight_gray, edgecolor=edgecolor, alpha=1.0, label=lbl_ov)
        ]
    else:
        lbl_1 = _format_legend(label_bar1, stats_bar1)
        lbl_2 = _format_legend(label_bar2, stats_bar2)
        legend_elements = [
            Patch(facecolor="grey", edgecolor=edgecolor, hatch=hatch_bar1, label=lbl_1),
            Patch(facecolor="#ffffff", edgecolor=edgecolor, hatch=hatch_bar2, label=lbl_2),
            Patch(facecolor="#ffffff", edgecolor=edgecolor, alpha=shadow_alpha, label=lbl_ov),
        ]
        
    ax.legend(handles=legend_elements, loc=legend_loc, framealpha=0.9, fontsize=10 * fontsize_multiplier)
    ax.tick_params(axis='both', which='major', labelsize=10 * fontsize_multiplier)
    plt.tight_layout()
    return fig, ax


def create_metric_comparison_chart_epistatic(
    series, 
    model_groups, 
    *,
    highlighted_model=None, 
    output_file=None, 
    title=None, 
    y_label=None, 
    fontsize_multiplier=1.3, 
    bracket_height_factor=0.05,
    bar_width=0.8,
    figsize=(12,7), 
    legend_loc="upper right", 
    show_significance=True,
    is_grouped=True,
    series2=None,
    series_names=('Series 1', 'Series 2'),
    colors=('#d35400', '#34495e', '#3498db', '#2ecc71', "#bbbbbb", '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', ),
    ylim=None
):
    """
    Create a bar chart comparing different model groups.
    
    Corrections applied:
    - Hatches are ALWAYS on the Left bar.
    - Left bar is ALWAYS the DARKER color (Saturated or Black).
    - Right bar is ALWAYS the LIGHTER color (Pastel or Light Grey).
    - Highlighted model: Left is Black w/ Light Grey hatches, Right is Light Grey.
    """
    
    # Set plot style
    plt.style.use('default')
    
    # --- Helper: Robust Data Collection ---
    def _collect(src_series, base_name):
        """Collects mean, std, and values for stats."""
        if src_series is None: return None
        
        # 1. Identify Replicates
        pattern = re.compile(rf"^{re.escape(base_name)}[_]?(\d+)$")
        reps = [k for k in src_series.index if pattern.match(k) and not k.endswith(('_n', '_groups'))]
        
        keys_used = []
        values = None
        
        # 2. Extract Values
        if reps:
            reps = sorted(reps)
            keys_used = reps
            values = src_series[reps].astype(float).values
        elif base_name in src_series.index:
            keys_used = [base_name]
            values = np.array([float(src_series[base_name])])
            
        if values is None or len(values) == 0:
            return None

        # 3. Extract Underlying N and Groups
        n_vals_found = []
        g_vals_found = []
        
        has_global_n = False
        if f"{base_name}_n" in src_series.index:
            try: 
                n_vals_found.append(float(src_series[f"{base_name}_n"]))
                has_global_n = True
            except ValueError as e: 
                raise AssertionError(f"Expected numeric value for {base_name}_n") from e
            
        has_global_g = False
        if is_grouped:
            if f"{base_name}_groups" in src_series.index:
                try: 
                    g_vals_found.append(int(src_series[f"{base_name}_groups"]))
                    has_global_g = True
                except ValueError as e: 
                    raise AssertionError(f"Expected numeric value for {base_name}_groups") from e
            
        for k in keys_used:
            if f"{k}_n" in src_series.index:
                try: n_vals_found.append(float(src_series[f"{k}_n"]))
                except ValueError as e: raise AssertionError(f"Expected numeric value for {k}_n") from e
            elif not has_global_n and k != base_name:
                pass
                
            if is_grouped:
                if f"{k}_groups" in src_series.index:
                    try: g_vals_found.append(int(src_series[f"{k}_groups"]))
                    except ValueError as e: raise AssertionError(f"Expected numeric value for {k}_groups") from e
                elif not has_global_g and k != base_name:
                    raise AssertionError(f"is_grouped=True but missing {k}_groups and global {base_name}_groups")

        n_underlying = max(set(n_vals_found), key=n_vals_found.count) if n_vals_found else None
        g_underlying = max(set(g_vals_found), key=g_vals_found.count) if g_vals_found else None
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0.0,
            'values': values,
            'has_replicates': len(values) > 1,
            'n_underlying': n_underlying,
            'groups_underlying': g_underlying
        }

    # --- Step 1: Organize Data ---
    pairs_data = []
    mode = "series" if series2 is not None else "epistatic"
    
    for model in model_groups:
        if mode == "series":
            data_left = _collect(series, model)
            data_right = _collect(series2, model)
            if data_left:
                pairs_data.append({
                    'name': model,
                    'left': data_left,
                    'right': data_right,
                    'is_highlight': (model == highlighted_model)
                })
        else:
            # Epistatic Mode
            data_left = _collect(series, model)
            data_right = _collect(series, f"{model}_additive")
            
            # Fallback for prefix pattern
            if not data_right:
                pat = re.compile(rf"^{re.escape(model)}[_]?(\d+)_additive$")
                add_reps = [k for k in series.index if pat.match(k)]
                if add_reps:
                    vals = series[add_reps].astype(float).values
                    n_vals_found = []
                    g_vals_found = []
                    
                    has_global_n = False
                    if f"{model}_additive_n" in series.index:
                        try:
                            n_vals_found.append(float(series[f"{model}_additive_n"]))
                            has_global_n = True
                        except ValueError as e: raise AssertionError(f"Expected numeric value for {model}_additive_n") from e
                        
                    has_global_g = False
                    if is_grouped and f"{model}_additive_groups" in series.index:
                        try:
                            g_vals_found.append(int(series[f"{model}_additive_groups"]))
                            has_global_g = True
                        except ValueError as e: raise AssertionError(f"Expected numeric value for {model}_additive_groups") from e
                    
                    for r in add_reps:
                        prefix = r[:-len("_additive")] if r.endswith("_additive") else r
                        
                        n_found = False
                        for c_n in [f"{r}_n", f"{prefix}_n_additive"]:
                            if c_n in series.index:
                                 try: 
                                     n_vals_found.append(float(series[c_n]))
                                     n_found = True
                                     break
                                 except ValueError as e: raise AssertionError(f"Expected numeric value for {c_n}") from e
                        
                        if is_grouped:
                            g_found = False
                            for c_g in [f"{r}_groups", f"{prefix}_groups_additive"]:
                                if c_g in series.index:
                                     try: 
                                         g_vals_found.append(int(series[c_g]))
                                         g_found = True
                                         break
                                     except ValueError as e: raise AssertionError(f"Expected numeric value for {c_g}") from e
                                     
                            if not g_found and not has_global_g:
                                 raise AssertionError(f"is_grouped=True but missing {r}_groups and global {model}_additive_groups")
                             
                    n_und = max(set(n_vals_found), key=n_vals_found.count) if n_vals_found else None
                    g_und = max(set(g_vals_found), key=g_vals_found.count) if g_vals_found else None
                    
                    data_right = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)) if len(vals)>1 else 0.0,
                        'values': vals, 
                        'has_replicates': len(vals)>1, 
                        'n_underlying': n_und,
                        'groups_underlying': g_und
                    }

            if data_left:
                pairs_data.append({
                    'name': model,
                    'left': data_left, 
                    'right': data_right,
                    'is_highlight': (model == highlighted_model)
                })

    if not pairs_data:
        print("No valid data found matching model_groups.")
        return plt.subplots()

    # --- Step 2: Consensus Stats Calculation ---
    def get_consensus(data_list, key, stat_key):
        ns = [p[key][stat_key] for p in data_list if p[key] and p[key][stat_key] is not None]
        if not ns: return None
        return max(set(ns), key=ns.count)

    consensus_left_n = get_consensus(pairs_data, 'left', 'n_underlying')
    consensus_left_g = get_consensus(pairs_data, 'left', 'groups_underlying')
    consensus_right_n = get_consensus(pairs_data, 'right', 'n_underlying')
    consensus_right_g = get_consensus(pairs_data, 'right', 'groups_underlying')

    xtick_labels = []
    for p in pairs_data:
        label = p['name']
        flag = False
        if p['left']:
            if p['left']['n_underlying'] not in (None, consensus_left_n) or \
               p['left']['groups_underlying'] not in (None, consensus_left_g):
                flag = True
        if p['right']:
            if p['right']['n_underlying'] not in (None, consensus_right_n) or \
               p['right']['groups_underlying'] not in (None, consensus_right_g):
                flag = True
        if flag: label += "*"
        xtick_labels.append(label)

    # --- Step 3: Setup Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    base_colors = colors
    color_map = {}
    c_idx = 0
    for p in pairs_data:
        if p['is_highlight']:
            color_map[p['name']] = None 
        else:
            color_map[p['name']] = base_colors[c_idx % len(base_colors)]
            c_idx += 1

    group_indices = np.arange(len(pairs_data))
    half = bar_width / 2.0
    
    y_max_data = -np.inf
    y_min_data = np.inf
    group_visual_max_y = {} 
    
    # Data Span pass
    for p in pairs_data:
        l, r = p['left'], p['right']
        l_top = l['mean'] + (l['std'] if l['has_replicates'] else 0)
        l_bot = l['mean'] - (l['std'] if l['has_replicates'] else 0)
        y_max_data = max(y_max_data, l_top)
        y_min_data = min(y_min_data, l_bot)
        
        if r:
            r_top = r['mean'] + (r['std'] if r['has_replicates'] else 0)
            r_bot = r['mean'] - (r['std'] if r['has_replicates'] else 0)
            y_max_data = max(y_max_data, r_top)
            y_min_data = min(y_min_data, r_bot)

    span = (y_max_data - y_min_data) if (y_max_data != y_min_data) else 1.0
    text_buffer = span * 0.08 
    
    # --- Step 4: Draw Bars ---
    for i, p in enumerate(pairs_data):
        model = p['name']
        left = p['left']
        right = p['right']
        
        # Determine Colors based on Highlight or Normal
        if p['is_highlight']:
            c_left = 'black'
            edge_c_left = '#bbbbbb' 
            c_right = '#444444'
            edge_c_right = 'black'
        else:
            base_c = color_map[model]
            rgb = mcolors.to_rgb(base_c)
            c_pastel = tuple(0.5 + 0.5 * c for c in rgb)
            c_left = base_c
            edge_c_left = 'black'
            c_right = c_pastel
            edge_c_right = 'black'

        # LEFT BAR LOGIC
        pos_left = i - half/2.0
        hatch_left = '///'

        ax.bar(pos_left, left['mean'], width=half, color=c_left, hatch=hatch_left,
               edgecolor=edge_c_left, 
               linewidth=2 if p['is_highlight'] else 1)
        
        if p['is_highlight']:
            ax.bar(pos_left, left['mean'], width=half, color='none', edgecolor='yellow', linewidth=2, zorder=2)
        
        err_left = left['std'] if left['has_replicates'] else 0
        if err_left > 0:
            ax.errorbar(pos_left, left['mean'], yerr=err_left, fmt='none', ecolor='red', capsize=4)
            
        text_y_left = left['mean'] + err_left + (span * 0.01)
        va_left = 'bottom'
        if left['mean'] < 0:
            text_y_left = left['mean'] - err_left - (span * 0.01)
            va_left = 'top'
        ax.text(pos_left, text_y_left, f"{left['mean']:.3f}", 
                ha='center', va=va_left, fontsize=9*fontsize_multiplier, fontweight='bold')
        
        if left['mean'] >= 0: visual_top_left = text_y_left + text_buffer
        else: visual_top_left = max(0, left['mean'] + err_left)
        
        # RIGHT BAR LOGIC
        visual_top_right = -np.inf
        if right:
            pos_right = i + half/2.0
            hatch_right = None
            
            ax.bar(pos_right, right['mean'], width=half, color=c_right, hatch=hatch_right,
                   edgecolor=edge_c_right, linewidth=2 if p['is_highlight'] else 1)
            
            err_right = right['std'] if right['has_replicates'] else 0
            if err_right > 0:
                ax.errorbar(pos_right, right['mean'], yerr=err_right, fmt='none', ecolor='red', capsize=4)
            
            text_y_right = right['mean'] + err_right + (span * 0.01)
            va_right = 'bottom'
            if right['mean'] < 0:
                text_y_right = right['mean'] - err_right - (span * 0.01)
                va_right = 'top'
            ax.text(pos_right, text_y_right, f"{right['mean']:.3f}", 
                    ha='center', va=va_right, fontsize=9*fontsize_multiplier, fontweight='bold')
            
            if right['mean'] >= 0: visual_top_right = text_y_right + text_buffer
            else: visual_top_right = max(0, right['mean'] + err_right)

        group_visual_max_y[i] = max(visual_top_left, visual_top_right)

    # Initialize final_y_max
    final_y_max = max(group_visual_max_y.values()) if group_visual_max_y else y_max_data

    # --- Step 5: Statistics (Optional) ---
    if show_significance:
        comparisons = []
        for i, p in enumerate(pairs_data):
            if not p['right']: continue
            l, r = p['left'], p['right']
            
            if not l['has_replicates'] and not r['has_replicates']:
                continue
            
            p_val = np.nan
            with np.errstate(all='ignore'):
                if l['has_replicates'] and r['has_replicates']:
                    _, p_val = stats.ttest_ind(l['values'], r['values'], equal_var=False)
                elif l['has_replicates'] and not r['has_replicates']:
                    _, p_val = stats.ttest_1samp(l['values'], r['mean'])
                elif not l['has_replicates'] and r['has_replicates']:
                    _, p_val = stats.ttest_1samp(r['values'], l['mean'])

            comparisons.append({
                'idx': i,
                'p': p_val,
                'base_y': group_visual_max_y[i]
            })

        for comp in comparisons:
            i = comp['idx']
            p_val = comp['p']
            base_y = comp['base_y']
            
            bottom = base_y + (span * bracket_height_factor) 
            arm_h = span * 0.02 
            top = bottom + arm_h
            
            x1 = i - half/2.0
            x2 = i + half/2.0
            
            if np.isnan(p_val): label = "NA"
            elif p_val < 0.001: label = "***"
            elif p_val < 0.01: label = "**"
            elif p_val < 0.05: label = "*"
            else: label = "ns"
            
            if not np.isnan(p_val) and label != "ns": label += f"\np={p_val:.1e}"
            elif not np.isnan(p_val) and label == "ns": label += f"\n({p_val:.2f})"

            ax.plot([x1, x1, x2, x2], [bottom, top, top, bottom], c='black', lw=1.5)
            ax.text((x1+x2)/2, top + (span*0.01), label, 
                    ha='center', va='bottom', fontsize=8*fontsize_multiplier)
            
            final_y_max = max(final_y_max, top + text_buffer)

    # --- Step 6: Formatting ---
    ax.set_xticks(group_indices)
    ax.set_xticklabels(xtick_labels, fontsize=10*fontsize_multiplier, rotation=45, ha='right')
    
    if y_label: ax.set_ylabel(y_label, fontsize=16*fontsize_multiplier, fontweight='bold')
    if title: ax.set_title(title, fontsize=16*fontsize_multiplier, fontweight='bold')
    
    if not ylim:
        pad_top = 0.05 * span
        pad_bot = 0.05 * span
        if y_min_data >= 0:
            ax.set_ylim(0, final_y_max + pad_top)
        else:
            ax.set_ylim(y_min_data - pad_bot, final_y_max + pad_top)
    else:
        ax.set_ylim(ylim)
    
    def _format_legend(base_label, cons_n, cons_g):
        parts = []
        if is_grouped:
            if cons_n is not None:
                parts.append(f"avg_muts={int(cons_n) if float(cons_n).is_integer() else f'{float(cons_n):.1f}'}")
            if cons_g is not None:
                parts.append(f"n_domains={cons_g}")
        else:
            if cons_n is not None:
                parts.append(f"n={int(cons_n) if float(cons_n).is_integer() else f'{float(cons_n):.1f}'}")
                
        return f"{base_label} ({', '.join(parts)})" if parts else base_label

    l1_text = _format_legend("Epistatic" if mode == 'epistatic' else series_names[0], consensus_left_n, consensus_left_g)
    l2_text = _format_legend("Additive" if mode == 'epistatic' else series_names[1], consensus_right_n, consensus_right_g)
    
    # Legend: Left (Dark/Hatched) vs Right (Light/Solid)
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', hatch='///', label=l1_text),
        Patch(facecolor='white', edgecolor='black', label=l2_text)
    ]
        
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=10*fontsize_multiplier, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', linestyle='', alpha=0)
    plt.subplots_adjust(bottom=0.25)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
    return fig, ax

def create_pos_neg_overall_chart(
    series,
    model_groups,
    *,
    highlighted_model=None,
    title=None,
    y_label=None,
    figsize=(14, 6),
    ylim=(0, 1),
    bar_width=0.8,
    shadow_alpha=0.15,
    hatch_pos="///",
    hatch_neg=None,
    edgecolor="black",
    fontsize_multiplier=1.3,
    capsize=4,
    base_colors=None,
    highlight_gray='#e0e0e0',
    error_bar_color='red',
    enforce_global_n=True,
    is_grouped=True,
    epistasis=True,
    label_min_sep=0.05,
    p_value_y_coord=None,
    legend_loc="upper right"
):
    """
    Plots overall, positive, and negative performance for model groups.
    Updated to use Standard Error (SEM) for error bars and annotations.
    """
    def _collect(series, base, suffix=None):
        if suffix is None:
            direct = base
            reps = [k for k in series.index if re.fullmatch(rf"{re.escape(base)}_?\d+", k)]
        else:
            direct = f"{base}_{suffix}"
            pat1 = rf"{re.escape(base)}_{re.escape(suffix)}_?\d+"
            pat2 = rf"{re.escape(base)}_?\d+_{re.escape(suffix)}"
            reps = [k for k in series.index if re.fullmatch(pat1, k) or re.fullmatch(pat2, k)]

        values = None
        if reps:
            values = series[reps].astype(float).values
        elif direct in series.index:
            values = np.array([float(series[direct])])

        if values is None or len(values) == 0 or not np.all(np.isfinite(values)):
            return None

        n = int(len(values))
        mean = float(np.mean(values))
        # Standard Error of the Mean (SEM) = std / sqrt(n)
        if n > 1:
            sem = float(np.std(values, ddof=1) / np.sqrt(n))
        else:
            sem = 0.0

        return {
            "mean": mean,
            "err": sem,  # Switched from std to sem
            "n": n,
            "has_replicates": n > 1,
            "values": values,
        }

    def _get_stats_from_series(series, base, subset=None):
        if subset is None:
            direct = base
            reps = [k for k in series.index if re.fullmatch(rf"{re.escape(base)}_?\d+", k)]
        else:
            direct = f"{base}_{subset}"
            pat1 = rf"{re.escape(base)}_{re.escape(subset)}_?\d+"
            pat2 = rf"{re.escape(base)}_?\d+_{re.escape(subset)}"
            reps = [k for k in series.index if re.fullmatch(pat1, k) or re.fullmatch(pat2, k)]

        keys_used = reps if reps else ([direct] if direct in series.index else [])
        if not keys_used:
            return None # Model not present

        stats = {}
        vals_n = []
        vals_g = []

        def _find_stat(base_key, stat_type):
            candidates = [f"{base_key}_{stat_type}"]
            if subset:
                if base_key.endswith(f"_{subset}"):
                    prefix = base_key[:-(len(subset)+1)]
                    candidates.append(f"{prefix}_{stat_type}_{subset}")
                    candidates.append(f"{prefix}_{subset}_{stat_type}")
                elif base_key.startswith(f"{base}_{subset}"):
                    suffix = base_key[len(f"{base}_{subset}"):]
                    if suffix.startswith("_"): suffix = suffix[1:]
                    if suffix:
                        candidates.append(f"{base}_{stat_type}_{subset}_{suffix}")
                        candidates.append(f"{base}_{subset}_{stat_type}_{suffix}")
                        
                candidates.append(f"{base}_{stat_type}_{subset}")
                candidates.append(f"{base}_{subset}_{stat_type}")
            else:
                candidates.append(f"{base}_{stat_type}")
                
            for c in candidates:
                if c in series.index:
                    return series[c]
            return None

        for k in keys_used:
            n_val = _find_stat(k, 'n')
            if n_val is not None:
                try: 
                    vals_n.append(float(n_val))
                except ValueError as e: 
                    raise AssertionError(f"Expected numeric for '{k}_n'") from e
                
            if is_grouped:
                g_val = _find_stat(k, 'groups')
                if g_val is not None:
                    try: 
                        vals_g.append(int(g_val))
                    except ValueError as e: 
                        raise AssertionError(f"Expected numeric for '{k}_groups'") from e

        if vals_n: 
            stats['n'] = max(set(vals_n), key=vals_n.count)
        
        if is_grouped:
            if vals_g:
                stats['groups'] = max(set(vals_g), key=vals_g.count)
            else:
                subset_tag = f" ({subset})" if subset else ""
                raise AssertionError(f"is_grouped=True but missing expected groups key for '{base}'{subset_tag}")
                
        return stats if stats else None

    def _determine_consensus_stats(series, model_groups, enforce=True):
        def _consensus(vals, tag):
            if not vals: return None
            uniq = sorted(set(vals))
            if enforce and len(uniq) > 1:
                raise AssertionError(f"Inconsistent {tag}: {uniq}")
            return uniq[0]

        pos_vals_n, neg_vals_n, all_vals_n = [], [], []
        pos_vals_g, neg_vals_g, all_vals_g = [], [], []

        for m in model_groups:
            p = _get_stats_from_series(series, m, 'pos')
            if p:
                if 'n' in p: pos_vals_n.append(p['n'])
                if 'groups' in p: pos_vals_g.append(p['groups'])
                
            n = _get_stats_from_series(series, m, 'neg')
            if n:
                if 'n' in n: neg_vals_n.append(n['n'])
                if 'groups' in n: neg_vals_g.append(n['groups'])
                
            a = _get_stats_from_series(series, m, None)
            if a:
                if 'n' in a: all_vals_n.append(a['n'])
                if 'groups' in a: all_vals_g.append(a['groups'])

        return {
            'pos': {'n': _consensus(pos_vals_n, 'pos n'), 'groups': _consensus(pos_vals_g, 'pos groups')},
            'neg': {'n': _consensus(neg_vals_n, 'neg n'), 'groups': _consensus(neg_vals_g, 'neg groups')},
            'overall': {'n': _consensus(all_vals_n, 'overall n'), 'groups': _consensus(all_vals_g, 'overall groups')}
        }

    stats_cons = _determine_consensus_stats(series, model_groups, enforce=enforce_global_n)
    
    base_color_list = base_colors or ['#d35400', '#34495e', '#3498db', '#2ecc71', '#bbbbbb', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', "#ffffff"]
    
    model_color_map = {}
    color_index = 0
    for m in model_groups:
        if highlighted_model is not None and m == highlighted_model:
            model_color_map[m] = None
        else:
            model_color_map[m] = base_color_list[color_index % len(base_color_list)]
            color_index += 1

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(model_groups))
    half = bar_width / 2.0
    per_model_label_entries = {m: [] for m in model_groups}
    overall_data = {}

    for i, model in enumerate(model_groups):
        base_col = model_color_map.get(model)
        is_highlight = (highlighted_model is not None and model == highlighted_model)
        lw = 2.5 if is_highlight else 1.2

        overall = _collect(series, model, suffix=None)
        pos = _collect(series, model, suffix="pos")
        neg = _collect(series, model, suffix="neg")

        if overall is not None:
            overall_data[model] = {**overall, 'xpos': x[i]}
            overall_color = highlight_gray if is_highlight else (base_col or highlight_gray)
            ax.bar(x[i], overall["mean"], width=bar_width, color=overall_color, alpha=shadow_alpha, 
                   edgecolor=edgecolor, linewidth=lw, zorder=1)
                   
            # Detect obstruction (>95% magnitude on all existing foreground bars)
            obstructed = False
            bars_present = []
            if pos is not None: bars_present.append(pos["mean"])
            if neg is not None: bars_present.append(neg["mean"])
            
            if bars_present:
                if overall["mean"] >= 0:
                    obstructed = all(val > overall["mean"] * 0.95 for val in bars_present)
                else:
                    # Inverts check logic to ensure negative magnitude is correctly evaluated
                    obstructed = all(val < overall["mean"] * 0.95 for val in bars_present)
            
            if obstructed:
                # White box in front to visually lighten the foreground bars
                ax.bar(x[i], overall["mean"], width=bar_width, color='white', alpha=0.4, 
                       edgecolor='white', linewidth=lw, zorder=4)
                       
            per_model_label_entries[model].append((x[i], overall["mean"], overall["err"], "overall", model))

        if pos is not None:
            current_pos_color = 'black' if is_highlight else base_col
            current_edgecolor = '#bbbbbb' if is_highlight else edgecolor
            ax.bar(x[i] - half/2.0, pos["mean"], width=half, color=current_pos_color, hatch=hatch_pos if not is_highlight else '///', 
                   edgecolor=current_edgecolor, linewidth=lw, zorder=2)
            if pos["has_replicates"]:
                ax.errorbar(x[i] - half/2.0, pos["mean"], yerr=pos["err"], fmt="none", ecolor=error_bar_color, elinewidth=lw, capsize=capsize, zorder=3)
            if is_highlight:
                ax.bar(x[i] - half/2.0, pos["mean"], width=half, color='none', edgecolor='yellow', linewidth=lw, zorder=2.5)
            per_model_label_entries[model].append((x[i] - half/2.0, pos["mean"], pos["err"], "pos", model))

        if neg is not None:
            if is_highlight:
                neg_color = '#444444'
            else:
                rgb = mcolors.to_rgb(base_col)
                neg_color = tuple(0.5 + 0.5 * c for c in rgb)
            ax.bar(x[i] + half/2.0, neg["mean"], width=half, color=neg_color, edgecolor=edgecolor if not is_highlight else 'yellow', linewidth=lw, zorder=3)
            if neg["has_replicates"]:
                ax.errorbar(x[i] + half/2.0, neg["mean"], yerr=neg["err"], fmt="none", ecolor=error_bar_color, elinewidth=lw, capsize=capsize, zorder=3)
            per_model_label_entries[model].append((x[i] + half/2.0, neg["mean"], neg["err"], "neg", model))

    ax.set_xticks(x)
    ax.set_xticklabels(model_groups, rotation=45, ha="right", fontsize=10 * fontsize_multiplier)
    if y_label: ax.set_ylabel(y_label, fontsize=16 * fontsize_multiplier, fontweight="bold")
    if title: ax.set_title(title, fontsize=16 * fontsize_multiplier, fontweight="bold")

    pvals_overall = {m: np.nan for m in model_groups}
    if highlighted_model is not None and highlighted_model in overall_data:
        hl = overall_data[highlighted_model]
        pvals_overall[highlighted_model] = 1.0
        for m in model_groups:
            if m not in overall_data or m == highlighted_model: continue
            cur = overall_data[m]
            if hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_ind(cur['values'], hl['values'], equal_var=False)
            elif hl['has_replicates'] and not cur['has_replicates']:
                _, p = stats.ttest_1samp(hl['values'], cur['mean'])
            elif not hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_1samp(cur['values'], hl['mean'])
            else: p = np.nan
            pvals_overall[m] = p

    span_now = ax.get_ylim()[1] - ax.get_ylim()[0]
    per_model_ytexts = {m: [] for m in model_groups}

    for m in model_groups:
        for (xpos, mean, err, kind, model_name) in per_model_label_entries[m]:
            if kind == 'overall' and p_value_y_coord is not None:
                ytext, va = p_value_y_coord, 'center'
            else:
                offset = 0.02 * span_now
                if mean >= 0: ytext, va = mean + err + offset, 'bottom'
                else: ytext, va = mean - err - offset, 'top'
            
            if kind == 'overall':
                p = pvals_overall.get(model_name, np.nan)
                p_text = "NA" if np.isnan(p) else f"{p:.2e}"
                if err > 1e-9: # Safe comparison for zero err
                    text = f"{mean:.3f}±{err:.3f}\np={p_text}"
                else:
                    text = f"{mean:.3f}\np={p_text}"
            else: text = f"{mean:.3f}"
            per_model_ytexts[m].append([xpos, ytext, text, va])

    extra_offset = 0.025 * span_now
    for m, entries in per_model_ytexts.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if abs(entries[i][1] - entries[j][1]) < label_min_sep:
                    if entries[i][1] >= entries[j][1]: entries[i][1] += extra_offset
                    else: entries[j][1] += extra_offset

    for m, entries in per_model_ytexts.items():
        for xpos, ytext, text, va in entries:
            ax.text(xpos, ytext, text, ha='center', va=va, fontsize=10 * fontsize_multiplier, fontweight='bold')

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(ylim)
    
    def _format_legend(base_label, stats_dict):
        if not stats_dict: return base_label
        parts = []
        
        if is_grouped:
            if 'n' in stats_dict and stats_dict['n'] is not None:
                val = stats_dict['n']
                parts.append(f"avg_muts={int(val) if float(val).is_integer() else f'{float(val):.1f}'}")
            if 'groups' in stats_dict and stats_dict['groups'] is not None:
                parts.append(f"n_domains={stats_dict['groups']}")
        else:
            if 'n' in stats_dict and stats_dict['n'] is not None:
                val = stats_dict['n']
                parts.append(f"n={int(val) if float(val).is_integer() else f'{float(val):.1f}'}")
                
        return f"{base_label} ({', '.join(parts)})" if parts else base_label

    pos_base = f"{'Positive epistasis' if epistasis else 'Stabilizing'}; ΔΔG > 0"
    neg_base = f"{'Negative epistasis' if epistasis else 'Destabilizing'}; ΔΔG <= 0"
    
    pos_lbl = _format_legend(pos_base, stats_cons['pos'])
    neg_lbl = _format_legend(neg_base, stats_cons['neg'])
    all_lbl = _format_legend("Overall (shadow)", stats_cons['overall'])

    legend_elements = [
        Patch(facecolor="grey", edgecolor=edgecolor, hatch=hatch_pos, label=pos_lbl),
        Patch(facecolor="#ffffff", edgecolor=edgecolor, label=neg_lbl),
        Patch(facecolor="#ffffff", edgecolor=edgecolor, alpha=shadow_alpha, label=all_lbl),
    ]
    ax.legend(handles=legend_elements, loc=legend_loc, framealpha=0.9, fontsize=10 * fontsize_multiplier)
    ax.tick_params(axis='both', which='major', labelsize=10 * fontsize_multiplier)
    plt.tight_layout()
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde, pearsonr
from scipy.optimize import curve_fit

def density_scatter(
    x, y, ax=None, sort=True, s=5, point_alpha=0.5,
    cmap="viridis", log=False, vmin=None, vmax=None, bw_method="scott", 
    hide_marginal_legend=False, stats=False, stats_color='red', include_p_value=False,
    nonlinear_fit=False, nonlinear_color='orange'
):
    """
    Scatter colored by KDE density, with semi-transparent points,
    marginal distributions, and ground truth overlay on marginals.
    Optionally adds line of best fit and Pearson correlation.

    Parameters
    ----------
    stats : bool, default=False
        If True, adds a linear line of best fit and a text box with Pearson r and p-value.
    stats_color : str, default='red'
        Color for the linear best fit line.
    nonlinear_fit : bool, default=False
        If True, attempts to fit a generalized logistic (sigmoid) curve to model global epistasis.
    nonlinear_color : str, default='orange'
        Color for the non-linear fit line.
    """
    # Capture pandas series names before numpy cast
    x_name = x.name if hasattr(x, 'name') else 'Predictions'
    y_name = y.name if hasattr(y, 'name') else 'Ground Truth'

    # --- Global Masking Must Happen Before KDE ---
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]

    if len(x_clean) == 0:
        raise AssertionError("No finite data points available to plot.")

    # Create figure with marginal axes - Fixed 2x3 GridSpec to remove empty bottom row
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=300)
        gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.25,
                              width_ratios=[0.2, 4, 1], height_ratios=[1, 4])
        ax_main = fig.add_subplot(gs[1, 1])
        ax_top = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[1, 2])  # right marginal
        ax_cbar = fig.add_subplot(gs[1, 0])   # colorbar on left
    else:
        # If ax provided, use it directly (no marginals in this case)
        ax_main = ax
        ax_top = None
        ax_right = None
        ax_cbar = None

    # KDE density estimate for scatter using cleaned data
    xy = np.vstack([x_clean, y_clean])
    z = gaussian_kde(xy, bw_method=bw_method)(xy)

    # Determine color scale
    if vmin is None:
        vmin = float(np.min(z))
    if vmax is None:
        vmax = float(np.max(z))
    if log:
        eps = np.finfo(float).eps
        norm = LogNorm(max(eps, vmin), vmax)
    else:
        norm = Normalize(vmin, vmax)

    # Map density to RGBA, then inject alpha per point
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(z))
    colors[:, 3] = np.clip(point_alpha, 0.0, 1.0)

    # Optionally sort so dense points draw last (on top)
    if sort:
        idx = z.argsort()
        x_, y_, colors = x_clean[idx], y_clean[idx], colors[idx]
    else:
        x_, y_ = x_clean, y_clean

    # Draw main scatter
    ax_main.scatter(x_, y_, c=colors, s=s, edgecolors="none", rasterized=False)

    # Add Line of Best Fit and Stats
    if stats:
        if len(x_clean) > 1:
            # Linear Regression (Polyfit degree 1)
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            
            # Plot line across the current data range
            line_x = np.array([np.min(x_clean), np.max(x_clean)])
            line_y = slope * line_x + intercept
            ax_main.plot(line_x, line_y, color=stats_color, linestyle='--', linewidth=2, label='Linear Fit')

            # Pearson Correlation
            r, p = pearsonr(x_clean, y_clean)

            text_str = f"$r = {r:.2f}$"
            
            if include_p_value:
                # Format p-value
                if p < 0.001:
                    p_str = "< 0.001"
                else:
                    p_str = f"{p:.3f}"
                text_str += f"\n$p = {p_str}$"
            
            # Place text box in top-left
            ax_main.text(0.05, 0.95, text_str, transform=ax_main.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
        else:
            raise AssertionError("Cannot compute linear stats: fewer than 2 valid (finite) data points available.")
    
    # Add Non-Linear Fit (Global Epistasis Sigmoid)
    if nonlinear_fit:
        if len(x_clean) > 4:
            def sigmoid(x_val, L, U, k, x0):
                # Clip the exponent to avoid overflow errors
                exp_term = np.clip(-k * (x_val - x0), -100, 100)
                return L + (U - L) / (1 + np.exp(exp_term))

            # Establish bounded initial guesses to aid convergence
            L_guess = np.min(y_clean)
            U_guess = np.max(y_clean)
            x0_guess = np.median(x_clean)
            k_guess = 1.0 

            try:
                popt, pcov = curve_fit(
                    sigmoid, x_clean, y_clean, 
                    p0=[L_guess, U_guess, k_guess, x0_guess],
                    bounds=([L_guess - 2, U_guess - 2, 0.01, np.min(x_clean)], 
                            [L_guess + 2, U_guess + 2, 10, np.max(x_clean)]),
                    maxfev=10000
                )
                
                x_curve = np.linspace(np.min(x_clean), np.max(x_clean), 200)
                y_curve = sigmoid(x_curve, *popt)
                ax_main.plot(x_curve, y_curve, color=nonlinear_color, linestyle='-', linewidth=2, label='Non-linear Fit')
                
            except RuntimeError as e:
                print(f"Global epistasis optimization failed to converge. Inspect data boundaries. Internal error: {e}")
            except ValueError as e:
                raise AssertionError(f"Curve fitting failed due to invalid data constraints or bounds. Internal error: {e}")
        else:
            raise AssertionError("Cannot fit non-linear curve: insufficient valid data points (requires > 4).")

    # Move y-axis label to right side
    ax_main.yaxis.set_label_position('right')
    ax_main.set_ylabel(y_name, rotation=270, labelpad=10)
    
    # Move x-axis label to top (between marginal and main plot)
    ax_main.set_xlabel('')  # Remove default x-label
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel(x_name)

    # --- ADDED: Force integer tick locations and 0 decimal formatting on main axes ---
    ax_main.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_main.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax_main.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_main.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Add marginal distributions if axes were created
    if ax_top is not None and ax_right is not None:
        # Set the marginal axis limits to match the main plot
        ax_top.set_xlim(ax_main.get_xlim())
        ax_right.set_ylim(ax_main.get_ylim())
        
        # Top marginal: predictions (x-axis) - Use cleaned data
        ax_top.hist(x_clean, bins=50, alpha=0.7, color='steelblue', density=True, label=x.name)
        ax_top.hist(y_clean, bins=50, alpha=0.3, color='coral', density=True, label=y.name)
        ax_top.set_ylabel('Density')
        ax_top.tick_params(labelbottom=False)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

        # Right marginal: ground truth (y-axis) - Use cleaned data
        ax_right.hist(y_clean, bins=50, alpha=0.7, color='coral', density=True, 
                      orientation='horizontal', label=y.name)
        ax_right.hist(x_clean, bins=50, alpha=0.3, color='steelblue', density=True,
                      orientation='horizontal', label=x.name)
        ax_right.set_xlabel('Density')
        ax_right.tick_params(labelleft=False)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)

        # Unified Full-Alpha Legend aligned with the right edge of ax_right and top edge of ax_top
        if not hide_marginal_legend:
            import matplotlib.patches as mpatches
            from matplotlib.transforms import blended_transform_factory
            
            pred_patch = mpatches.Patch(color='steelblue', alpha=1.0, label=x.name)
            gt_patch = mpatches.Patch(color='coral', alpha=1.0, label=y.name)
            
            # Blend the transforms so x=1.0 is the right edge of ax_right, and y=1.0 is the top edge of ax_top
            transform = blended_transform_factory(ax_right.transAxes, ax_top.transAxes)
            ax_top.legend(
                handles=[pred_patch, gt_patch],
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=transform,
                fontsize=8
            )

    # Create colorbar on the left side
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    
    if ax_cbar is not None:
        cbar = plt.colorbar(sm, cax=ax_cbar)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    else:
        cbar = plt.colorbar(sm, ax=ax_main)
        
    cbar.set_label("Density")

    # Lock format to prevent bounding box shifts downstream
    formatter = FormatStrFormatter('%.1e') if log else FormatStrFormatter('%.2f')
    cbar.ax.yaxis.set_major_formatter(formatter)

    return ax_main


def calculate_ppc(group, pred_col, percentile_values, meas='ddG', threshold=1):
    result = {}
    ground_truth = set(group.loc[group[meas] > threshold].index)
    sorted_predictions = group.sort_values(pred_col, ascending=False)
    
    for p in percentile_values:
        #k = (p - 100) / 100
        k = p / 100
        l = max(int(len(group) * k), 1)
        #print(pred_col, l)
        kth_prediction = set(sorted_predictions.head(l).index)
        result[f"{p}%"] = len(ground_truth.intersection(kth_prediction))
        result[f"pos_{p}%"] = len(kth_prediction)
        result[f"frac_{p}%"] = len(ground_truth.intersection(kth_prediction)) / len(kth_prediction)
    
    return pd.Series(result)

def calculate_msc(group, pred_col, percentile_values, meas='ddG'):
    result = {}
    sorted_predictions = group.sort_values(pred_col, ascending=False)
    
    for p in percentile_values:
        #k = (p - 100) / 100
        k = p / 100
        l = max(int(len(group) * k), 1)
        #print(pred_col, l)
        kth_prediction = list(set(sorted_predictions.head(l).index))
        result[f"{p}$"] = group.loc[kth_prediction, meas].sum()
        result[f"pos_{p}$"] = len(kth_prediction)
        result[f"frac_{p}$"] = group.loc[kth_prediction, meas].sum() / len(kth_prediction)
    
    return pd.Series(result)


def compute_stats_multi(
    df, 
    split_col=None, split_val=None, split_col_2=None, split_val_2=None, 
    measurements=('ddG', 'dTm'), stats=(), n_classes=2, quiet=False, 
    grouper=('code'), n_bootstraps=-1, split_first=True, split_last=True,
    threshold=0, duplicates=True,
    ):
    """
    Computes all per-protein and per-dataset stats, including when splitting
    into more than one feature-based scaffold. Splitting is done by specifying
    split_cols (the feature names) and split_vals (the threshold for splitting
    on the respective features). Specifying only split_col and split_val will
    create two scaffolds. Specifying only split_col with split_val > 
    split_val_2 will create 3 scaffolds, with high, intermediate and low values.
    Specifying different split_col and split_col_2 will create 4 scaffolds
    based on high and low values of 2 features. You can pass in a tuple of stats
    to only calculate a subset of the possible stats. You can use n_classes=3
    to eliminate the near-neutral mutations.
    """
    assert (split_first or split_last)
    if n_bootstraps > 0:
        dbs_bs = bootstrap_by_grouper(df, n_bootstraps, grouper=grouper, drop=False, duplicates=duplicates)
    else:
        dbs_bs = [df]
    dfs_out = []

    for db_gt_preds in tqdm(dbs_bs) if not quiet else dbs_bs:
        #db_gt_preds.to_csv('test.csv')
        split_col_ = split_col
        split_col_2_ = split_col_2

        # make sure to not accidentally modify the input
        db_internal = db_gt_preds.copy(deep=True)
        if grouper is not None:
            index_names = db_internal.index.names
            if index_names == [None]:
                db_internal.index.name = 'uid_sym'
                index_names = ['uid_sym']
            db_grouper = db_internal[grouper].reset_index().drop_duplicates()
            db_grouper = db_grouper.set_index(index_names)
            db_internal = db_internal.drop(grouper, axis=1)
        # currently, grouper cant be None!
        else:
            db_grouper = db_internal[[]]

        # eliminate the neutral mutations
        if n_classes == 3:
            db_internal = db_internal.loc[
                ~((db_internal['ddG'] > -1) & (db_internal['ddG'] < 1))
                ]
            if 'dTm' in db_internal.columns:
                db_internal = db_internal.loc[
                    ~((db_internal['dTm'] > -2) & (db_internal['dTm'] < 2))
                    ]

        # case where there are two split_vals on the same column
        if split_col_2_ is None and split_val_2 is not None:
            split_col_2_ = split_col_
        # case where there is no split (default)
        if (split_col_ is None) or (split_val is None):
            split_col_ = 'tmp'
            split_val = 0
            db_internal['tmp'] = -1
        # case where there is only one split (2 scaffolds)
        if (split_col_2_ is None) or (split_val_2 is None):
            split_col_2_ = 'tmp2'
            split_val_2 = 0
            db_internal['tmp2'] = -1

        #print(db_internal)
        # there may be missing features for some entries
        db_internal = db_internal.dropna(subset=[split_col_, split_col_2_])

        # db_discrete will change the continuous measurements into binary labels
        db_discrete = db_internal.copy(deep=True)
        
        # default case
        # stability threshold is defined exactly at 0 kcal/mol or deg. K
        if n_classes == 2:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 0, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < 0, 'ddG'] = 0
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] > 0, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] < 0, 'dTm'] = 0

        # stabilizing mutations now need to be >= 1 kcal/mol or deg. K
        elif n_classes == 3:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 1, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < -1, 'ddG'] = -1
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] >= 2, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] <= -2, 'dTm'] = -1

        # for creating a multi-index later
        cols = db_discrete.columns.drop(measurements + [split_col_, split_col_2_])
        
        # db_discrete_bin has discrete labels and binarized (discrete) predictions
        # drop the split_col_s so they do not get binarized
        db_discrete_bin = db_discrete.copy(deep=True).drop(
            [split_col_, split_col_2_], axis=1).astype(float)

        # binarize predictions (>0 stabilizing, assigned positive prediction)
        db_discrete_bin[db_discrete_bin > 0] = 1
        db_discrete_bin[db_discrete_bin < 0] = 0

        # retrieve the original split_col_(s)
        db_discrete_new = db_discrete[
            [split_col_] + ([split_col_2_] if split_col_2_ != split_col_ else [])]
        # make sure the indices align
        assert all(db_discrete_new.index == db_discrete_bin.index)
        # reunite with split_col_s
        db_discrete_bin = pd.concat([db_discrete_bin, db_discrete_new], axis=1)

        # create labels to assign to different scaffolds
        # case no split
        if split_col_ == 'tmp' and split_col_2_ == 'tmp2':
            split = ['']
        # case only one split col
        elif split_col_2_ == 'tmp2':
            split = [f'{split_col_} <= {split_val}', f'{split_col_} > {split_val}',]
        # case 2 splits on same col
        elif split_col_ == split_col_2_:
            split = [f'{split_col_} <= {split_val_2}',
                     f'{split_val} >= {split_col_} > {split_val_2}', 
                     f'{split_col_} > {split_val}']
        # case 3 splits total
        elif split_last == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_} > {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_2_} <= {split_val_2}']
        # case 3 splits total
        elif split_first == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_} > {split_val}']
        # case 2 splits on 2 cols
        else:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_} > {split_val} & {split_col_2_} <= {split_val_2}', 
                     f'{split_col_} > {split_val} & {split_col_2_} > {split_val_2}']
            #s2 = []
            #for keep, scaffold in zip(keep_scaffolds, split):
            #    if keep:
            #        s2.append(scaffold)
            #split = s2
                
        # separate statistics by measurement, feature scaffold, prediction
        idx = pd.MultiIndex.from_product([['dTm', 'ddG'], split, cols])
        df_out = pd.DataFrame(index=idx)

        # iterate through measurements and splits
        for meas in measurements:
            for sp in split:

                # get new copies that get reduced per scaffold / measurement
                cur_df_bin = db_discrete_bin.copy(deep=True)
                cur_df_discrete = db_discrete.copy(deep=True)
                cur_df_cont = db_internal.copy(deep=True)

                # the following section contains the logic for splitting based on
                # which scaffold is being considered and is self-explanatory
                # there is no logic needed if there is no split requested

                if split_col_ != 'tmp' and split_col_2_ != 'tmp2' and split_col_ != split_col_2_:
                    # case where there are 4 scaffolds
                    if len(sp.split('&')) > 1:
                        if '>' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                        elif '<=' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]

                        if '>' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2]
                        elif '<=' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2]

                    # case where there are 3 scaffolds
                    elif len(sp.split('&')) == 1:
                        if not split_first:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]

                        elif not split_last:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2]   

                # case where there are 3 scaffolds (on the same feature)
                elif split_col_ == split_col_2_:

                    if ('>' in sp and not '>=' in sp):
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    elif '<=' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val_2]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val_2]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val_2]
                    else:
                        cur_df_bin = cur_df_bin.loc[(cur_df_bin[split_col_] > split_val_2) & (cur_df_bin[split_col_] <= split_val)]
                        cur_df_discrete = cur_df_discrete.loc[(cur_df_discrete[split_col_] > split_val_2) & (cur_df_discrete[split_col_] <= split_val)]
                        cur_df_cont = cur_df_cont.loc[(cur_df_cont[split_col_] > split_val_2) & (cur_df_cont[split_col_] <= split_val)]
                        
                # case where there are two scaffolds on one feature
                elif split_col_2_ == 'tmp2' and split_col_ != 'tmp':

                    if '>' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    else:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]                  
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val] 
                
                # in this next section we compute the statistics one model at a time
                # all predictions should have the suffix _dir to designate direction mutations
                #for col in (tqdm([col for col in cols if ('_dir' in col and not 'runtime' in col)]) \
                #    if not quiet else [col for col in cols if ('_dir' in col and not 'runtime' in col)]):
                for col in cols:
                    
                    # get a reduced version of cur_df_cont for the relevant model
                    try:
                        pred_df_cont = cur_df_cont[[col,meas,f'runtime_{col}']].dropna()
                        # we only care about the total runtime for this function
                        df_out.loc[(meas,sp,col), 'runtime'] = pred_df_cont[f'runtime_{col}'].sum()
                        pred_df_cont = pred_df_cont.drop(f'runtime_{col}', axis=1)
                    except KeyError:
                        #if not quiet:
                        #    print('e', col)
                        pred_df_cont = cur_df_cont[[col,meas]].dropna()
                        df_out.loc[(meas,sp,col), 'runtime'] = np.nan    

                    # get a reduced version of the classification-task predictions and labels
                    pred_df_bin = cur_df_bin[[col,meas]].dropna()
                    #print(pred_df_bin)

                    if 'n' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'n'] = len(pred_df_bin)
                        saved_n = len(pred_df_bin)
                    if len(pred_df_bin) == 0:
                        raise AssertionError(f'There are no {col} predictions in this scaffold ({sp})!')
                    
                    # compute the 'easy' whole-dataset statistics
                    try:
                        tn, fp, fn, tp = metrics.confusion_matrix(pred_df_bin[meas], pred_df_bin[col]).ravel()
                    except:
                        tn, fp, fn, tp = 1,1,1,1
                    # compute each statistic by default (when stats==())
                    if 'tp' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'tp'] = tp
                    if 'fp' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'fp'] = fp
                    if 'tn' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'tn'] = tn 
                    if 'fn' in stats or stats == ():  
                        df_out.loc[(meas,sp,col), 'fn'] = fn   
                    if 'sensitivity' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'sensitivity'] = tp/(tp+fn)
                    if 'specificity' in stats or stats == ():         
                        df_out.loc[(meas,sp,col), 'specificity'] = tn/(tn+fp)
                    if 'PPV' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'PPV'] = tp/(tp+fp)
                    if 'pred_positives' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'pred_positives'] = tp+fp
                    if 'accuracy' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'accuracy'] = metrics.accuracy_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'f1_score' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'f1_score'] = metrics.f1_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'MCC' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'MCC'] = metrics.matthews_corrcoef(pred_df_bin[meas], pred_df_bin[col])

                    # get a reduced version of the model's predictions with discrete ground truth labels
                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna()
                    # discrete labels allow testing different thresholds of continuous predictions
                    # e.g. for area-under-curve methods
                    try:
                        pred_df_discrete[meas] = pred_df_discrete[meas].astype(int)
                        auroc = metrics.roc_auc_score(pred_df_discrete[meas], pred_df_discrete[col])
                        auprc = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                        if 'auroc' in stats or stats == (): 
                            df_out.loc[(meas,sp,col), 'auroc'] = auroc
                        if 'auprc' in stats or stats == (): 
                            df_out.loc[(meas,sp,col), 'auprc'] = auprc
                    # might fail for small scaffolds
                    except Exception as e:
                        if not quiet:
                            print('Couldn\'t compute AUC:', e)

                    # using the full (continous) predictions and labels now
                    pred_df_cont = cur_df_cont[[col,meas]].dropna().join(db_grouper)

                    # recall of the top-k predicted-most-stable proteins across the whole slice of data
                    for stat in [s for s in stats if 'recall@' in s] if stats != () else ['recall@k0.0', 'recall@k1.0']:
                        k = stat.split('@')[-1].strip('k')
                        if k == '':
                            k = 0.
                        else:
                            k = float(k)
                        
                        pred_df_discrete_k = pred_df_cont.copy(deep=True).drop_duplicates()
                        pred_df_discrete_k[meas] = pred_df_discrete_k[meas].apply(lambda x: 1 if x > k else 0)
                        stable_ct = pred_df_discrete_k[meas].sum()

                        gain = pred_df_cont.loc[pred_df_cont[meas] > k, meas].sum()
                        #print(stable_ct)
                        #print(stable_ct)
                        df_out.loc[(meas,sp,col), f'{k}_n_stable'] = stable_ct
                    
                        sorted_preds = pred_df_discrete_k.sort_values(col, ascending=False).index
                        df_out.loc[(meas,sp,col), f'recall@k{k}'] = pred_df_discrete_k.loc[sorted_preds[:stable_ct], meas].sum() / stable_ct
                        df_out.loc[(meas,sp,col), f'gain@k{k}'] = pred_df_cont.drop_duplicates().loc[(sorted_preds[:stable_ct]), meas].sum() / gain

                    # average experimental stabilization of predicted positives
                    if 'mean_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].mean()
                    # average experimental stabilization of predicted positives
                    if 'net_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'net_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].sum()
                    # average predicted score for experimentally stabilizing mutants
                    if 'mean_stable_pred' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stable_pred'] = pred_df_cont.loc[pred_df_cont[meas]>0, col].mean()
                    # mean squared error
                    if 'mse' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_squared_error'] = metrics.mean_squared_error(pred_df_cont[meas], pred_df_cont[col])

                    # top-1 score, e.g. the experimental stabilization achieved on 
                    # average for the top-scoring mutant of each protein
                    if ('mean_t1s' in stats) or (stats == ()): 
                        top_1_stab = 0
                        for code, group in pred_df_cont.groupby(grouper):
                            top_1_stab += group.sort_values(col, ascending=False)[meas].head(1).item()
                        df_out.loc[(meas,sp,col), 'mean_t1s'] = top_1_stab / len(pred_df_cont[grouper].unique())

                    # inverse of the assigned rank of the number one most stable protein per group
                    if ('mean_reciprocal_rank' in stats) or (stats == ()): 
                        reciprocal_rank_sum = 0
                        unique_groups = pred_df_cont[grouper].unique()
                        for code, group in pred_df_cont.groupby(grouper):
                            group = group.drop_duplicates()
                            sorted_group = group.sort_values(col, ascending=False)
                            highest_meas_rank = sorted_group[meas].idxmax()

                            rank_of_highest_meas = sorted_group.index.get_loc(highest_meas_rank)
                            if type(rank_of_highest_meas) in [slice, list, bool]:
                                print('Something went wrong with MRR for', col, code)
                                continue
                            try:
                                rank_of_highest_meas += 1
                            except:
                                print('Something went wrong with MRR for', col, code)
                                continue

                            reciprocal_rank_sum += 1 / rank_of_highest_meas

                        mean_reciprocal_rank = reciprocal_rank_sum / len(unique_groups)
                        df_out.loc[(meas, sp, col), 'mean_reciprocal_rank'] = mean_reciprocal_rank
                    
                    # normalized discounted cumulative gain, a measure of information retrieval ability
                    if ('ndcg' in stats) or (stats == ()):
                        # whole-dataset version (not presented in study)
                        df_out.loc[(meas,sp,col), 'ndcg'] = compute_ndcg_flexible(pred_df_cont, col, meas, threshold=threshold)
                        cum_ndcg = 0
                        w_cum_ndcg = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        # iterate over unique proteins (wild-type structures)
                        for code, group in pred_df_cont.groupby(grouper): 
                            # must be more than one to retrieve, and their stabilities should be different
                            if len(group.loc[group[meas]>threshold]) > 1 and not all(group[meas]==group[meas][0]):
                                cur_ndcg = compute_ndcg_flexible(group, col, meas, threshold=threshold)
                                # can happen if there are no stable mutants
                                if np.isnan(cur_ndcg):
                                    continue
                                # running-total (cumulative)
                                cum_ndcg += cur_ndcg
                                cum_d += 1
                                # weighted running-total (by log(num mutants))
                                w_cum_ndcg += cur_ndcg * np.log(len(group.loc[group[meas]>0]))
                                w_cum_d += np.log(len(group.loc[group[meas]>0]))
                                cum_muts += len(group.loc[group[meas]>0])
                        df_out.loc[(meas,sp,col), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)
                        # may be less than the number of proteins in the dataset based on the if statement               
                        df_out.loc[(meas,sp,col), 'n_proteins_ndcg'] = cum_d
                        # may be less than the number of mutants based on the if statement
                        df_out.loc[(meas,sp,col), 'n_muts_ndcg'] = cum_muts
                    
                    if ('pearson' in stats) or (stats == ()):
                        whole_r, _ = pearsonr(pred_df_cont[col], pred_df_cont[meas])
                        df_out.loc[(meas,sp,col), 'pearson'] = whole_r

                    # Spearman's rho, rank-order version of Pearson's r
                    # follows same logic as above
                    if ('spearman' in stats) or (stats == ()):
                        whole_p, _ = spearmanr(pred_df_cont[col], pred_df_cont[meas])
                        df_out.loc[(meas,sp,col), 'spearman'] = whole_p
                        cum_p = 0
                        w_cum_p = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for code, group in pred_df_cont.groupby(grouper):
                            if len(group) > 1 and not all(group[meas]==group[meas][0]):
                                spearman, _ = spearmanr(group[col], group[meas])
                                # can happen if all predictions are the same
                                # in which case ranking ability is poor since we 
                                # already checked that the measurements are different
                                if np.isnan(spearman):
                                    spearman=0
                                cum_p += spearman
                                cum_d += 1
                                w_cum_p += spearman * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_spearman'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_spearman'] = cum_muts
                        if cum_muts > saved_n:
                            print(cum_muts, saved_n, sp, col)

                    # Spearman's rho, rank-order version of Pearson's r
                    # follows same logic as above
                    if ('gain' in stats) or (stats == ()):
                        cum_recalled = 0
                        cum_gain = 0
                        cum_stable_ct = 0
                        cum_possible_gain = 0
                        
                        k_ = 0.5
                        for i, (code, group) in enumerate(pred_df_cont.groupby(grouper)):
                            #group = group.drop_duplicates()
                            group_discrete_k_ = group.copy(deep=True)
                            group_discrete_k_[meas] = group_discrete_k_[meas].apply(lambda x: 1 if x > k_ else 0)
                            stable_ct = group_discrete_k_[meas].sum()
                            cum_stable_ct += stable_ct

                            possible_gain = group.loc[group[meas] > k_, meas].sum()
                            cum_possible_gain += possible_gain
                            df_out.loc[(meas,sp,col), f'{k_}_n_stable'] = stable_ct
                        
                            sorted_preds = group_discrete_k_.sort_values(col, ascending=False).index
                            cum_recalled += group_discrete_k_.loc[sorted_preds[:stable_ct], meas].sum()
                            cum_gain += group.loc[sorted_preds[:stable_ct], meas].sum()
                            #if i == 0:
                            #    print(code, meas, sp, col)
                            #    print(stable_ct)
                            #    print(group.loc[sorted_preds[:stable_ct], [col, meas]])

                        df_out.loc[(meas,sp,col), f'frac_cum_recall@k_{k_}'] = cum_recalled / cum_stable_ct
                        df_out.loc[(meas,sp,col), f'frac_cum_gain@k_{k_}'] = cum_gain / cum_possible_gain
                        df_out.loc[(meas,sp,col), f'mean_gain@k_{k_}'] = cum_gain / cum_stable_ct
                        df_out.loc[(meas,sp,col), f'n_muts_mean_gain@k_{k_}'] = cum_stable_ct
                        df_out.loc[(meas,sp,col), f'n_muts_gain'] = cum_stable_ct

                    # refresh the discrete dataframe
                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna().join(db_grouper)
                    #pred_df_discrete['code'] = pred_df_discrete.index.str[:4] 
                    
                    # calculate area under the precision recall curve per protein as with the above stats
                    if ('auprc' in stats) or (stats == ()):
                        #df_out.loc[(meas,sp,col), 'auprc'] = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                        cum_ps = 0
                        w_cum_ps = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for _, group in pred_df_discrete.groupby(grouper): 
                            if len(group) > 1:
                                #group[meas] = group[meas].astype(int)
                                cur_ps = metrics.average_precision_score(group[meas], group[col])
                                # NaN if there is only one class in this scaffold for this protein
                                if np.isnan(cur_ps):
                                    continue
                                cum_ps += cur_ps
                                cum_d += 1
                                w_cum_ps += cur_ps * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_auprc'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_auprc'] = cum_muts

                    # these are the expensive statistics (calculated at 100 thresholds)
                    # it would take too long to compute them per-scaffold
                    if split_col_ == 'tmp':
                        if ('auppc' in stats) or (stats == ()):
                            percentiles = [str(int(s)/10)+'%' for s in range(1, 1000)]
                            percentile_values = [float(s.split('%')[0]) for s in percentiles]
                        else:
                            percentiles = [s for s in stats if '%' in s]
                            percentile_values = [float(s.split('%')[0]) for s in percentiles]

                        if grouper is not None:
                            # Apply the function to each group and reset the index
                            results_df = pred_df_cont.groupby(grouper).apply(
                                calculate_ppc, pred_col=col, meas=meas, percentile_values=percentile_values, threshold=threshold
                                ).reset_index()
                        else:
                            results_df = calculate_ppc(pred_df_cont, pred_col=col, meas=meas, percentile_values=percentile_values, threshold=threshold)
                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                #stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                                stat_dict[stat] = results_df[f"frac_{stat}"].mean()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[(meas, sp, col), percentiles] = list(stat_dict.values())
                        df_out.loc[(meas, sp, col), 'auppc'] = df_out.loc[(meas, sp, col), percentiles].mean()

                        # mean stability vs prediction percentile curve
                        if ('aumsc' in stats) or (stats == ()):
                            percentiles = [str(int(s)/10)+'$' for s in range(1, 1000)]
                            percentile_values = [float(s.split('$')[0]) for s in percentiles]
                        else:
                            percentiles = [s for s in stats if '$' in s]
                            percentile_values = [float(s.split('$')[0]) for s in percentiles]

                        if grouper is not None:
                            # Apply the function to each group and reset the index
                            results_df = pred_df_cont.groupby(grouper).apply(
                                calculate_msc, pred_col=col, meas=meas, percentile_values=percentile_values
                                ).reset_index()
                        else:
                            results_df = calculate_msc(pred_df_cont,
                                pred_col=col, meas=meas, percentile_values=percentile_values
                                )             

                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                #stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                                stat_dict[stat] = results_df[f"frac_{stat}"].mean()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[(meas, sp, col), percentiles] = list(stat_dict.values())
                        df_out.loc[(meas, sp, col), 'aumsc'] = df_out.loc[(meas, sp, col), percentiles].mean()
        dfs_out.append(df_out)

    if n_bootstraps > 0:

        concatenated_df = pd.concat(dfs_out, axis=0)

        # Reset the index to a simple range index, then set it back to a multi-index
        concatenated_df.reset_index(inplace=True)
        concatenated_df.set_index(['level_0', 'level_1', 'level_2'], inplace=True)

        # Now perform the groupby operation and compute mean and std
        mean_df = concatenated_df.groupby(level=['level_0', 'level_1', 'level_2']).mean()
        std_df = concatenated_df.groupby(level=['level_0', 'level_1', 'level_2']).std()

        # Create new DataFrame with _mean and _std columns
        result_df = pd.DataFrame(index=mean_df.index)

        for col in mean_df.columns:
            result_df[f"{col}_mean"] = mean_df[col]
            result_df[f"{col}_std"] = std_df[col]

        df_out = result_df

    else:
        df_out = dfs_out[0]

    df_out = df_out.reset_index()
    
    # add labels for the input information used by the model
    #df_out['model_type'] = 'structural'
    for k,v in mapping_categories.items():
        for m in v:
            # there are many variants of the models so just check if their base name matches
            df_out.loc[df_out['level_2'].str.contains(m), 'model_type'] = k
    df_out = df_out.rename({'level_0': 'measurement', 'level_1': 'class', 'level_2': 'model'}, axis=1)

    df_out = df_out.set_index(['measurement', 'model_type', 'model', 'class'])
    # sort by measurement type, and then model type within each measurement type
    # class is the scaffold
    df_out = df_out.sort_index(level=1).sort_index(level=0)

    return df_out.dropna(how='all')

def compute_stats_multi_per_group(
    df, 
    split_col=None, split_val=None, split_col_2=None, split_val_2=None, 
    measurements=('ddG', 'dTm'), stats=(), n_classes=2, quiet=False, 
    grouper=('code'), split_first=True, split_last=True,
    threshold=0, duplicates=True,
    ):
    """
    Computes all per-protein and per-dataset stats, iterating per group specified
    by 'grouper' instead of using bootstrapping.
    
    Returns a DataFrame containing statistics for each group individually.
    """
    assert (split_first or split_last)
    
    # Group the dataframe by the grouper instead of bootstrapping
    try:
        grouped = df.groupby(grouper)
    except KeyError as e:
        print(f"Error grouping by {grouper}: {e}")
        return None

    # Use the grouped object as the iterator
    iterator = tqdm(grouped) if not quiet else grouped
    dfs_out = []

    for group_name, db_gt_preds in iterator:
        split_col_ = split_col
        split_col_2_ = split_col_2
        split_val_2_ = split_val_2

        # make sure to not accidentally modify the input
        db_internal = db_gt_preds.copy(deep=True)
        
        # Handle the grouper column extraction to mimic original logic
        # (Original code dropped grouper from db_internal but kept it in db_grouper for later joins)
        if grouper is not None:
            index_names = db_internal.index.names
            if index_names == [None]:
                db_internal.index.name = 'uid_sym'
                index_names = ['uid_sym']
            
            # Handle both string and list grouper inputs
            if isinstance(grouper, str) or (isinstance(grouper, tuple) and len(grouper) == 1):
                db_grouper = db_internal[grouper].reset_index().drop_duplicates()
            else:
                db_grouper = db_internal[grouper].reset_index().drop_duplicates()
            
            db_grouper = db_grouper.set_index(index_names)
            db_internal = db_internal.drop(grouper, axis=1)
        else:
            db_grouper = db_internal[[]]

        # eliminate the neutral mutations
        if n_classes == 3:
            db_internal = db_internal.loc[
                ~((db_internal['ddG'] > -1) & (db_internal['ddG'] < 1))
                ]
            if 'dTm' in db_internal.columns:
                db_internal = db_internal.loc[
                    ~((db_internal['dTm'] > -2) & (db_internal['dTm'] < 2))
                    ]

        # case where there are two split_vals on the same column
        if split_col_2_ is None and split_val_2 is not None:
            split_col_2_ = split_col_
        # case where there is no split (default)
        if (split_col_ is None) or (split_val is None):
            split_col_ = 'tmp'
            split_val = 0
            db_internal['tmp'] = -1
        # case where there is only one split (2 scaffolds)
        if (split_col_2_ is None) or (split_val_2 is None):
            split_col_2_ = 'tmp2'
            split_val_2_ = 0
            db_internal['tmp2'] = -1

        # there may be missing features for some entries
        db_internal = db_internal.dropna(subset=[split_col_, split_col_2_])

        # db_discrete will change the continuous measurements into binary labels
        db_discrete = db_internal.copy(deep=True)
        
        # default case
        if n_classes == 2:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 0, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < 0, 'ddG'] = 0
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] > 0, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] < 0, 'dTm'] = 0

        # stabilizing mutations now need to be >= 1 kcal/mol or deg. K
        elif n_classes == 3:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 1, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < -1, 'ddG'] = -1
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] >= 2, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] <= -2, 'dTm'] = -1

        cols = db_discrete.columns.drop(measurements + [split_col_, split_col_2_])
        
        # db_discrete_bin has discrete labels and binarized (discrete) predictions
        db_discrete_bin = db_discrete.copy(deep=True).drop(
            [split_col_, split_col_2_], axis=1).astype(float)

        db_discrete_bin[db_discrete_bin > 0] = 1
        db_discrete_bin[db_discrete_bin < 0] = 0

        db_discrete_new = db_discrete[
            [split_col_] + ([split_col_2_] if split_col_2_ != split_col_ else [])]
        assert all(db_discrete_new.index == db_discrete_bin.index)
        db_discrete_bin = pd.concat([db_discrete_bin, db_discrete_new], axis=1)

        # create labels to assign to different scaffolds
        if split_col_ == 'tmp' and split_col_2_ == 'tmp2':
            split = ['']
        elif split_col_2_ == 'tmp2':
            split = [f'{split_col_} <= {split_val}', f'{split_col_} > {split_val}',]
        elif split_col_ == split_col_2_:
            split = [f'{split_col_} <= {split_val_2_}',
                     f'{split_val} >= {split_col_} > {split_val_2_}', 
                     f'{split_col_} > {split_val}']
        elif split_last == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2_}',
                     f'{split_col_} > {split_val} & {split_col_2_} > {split_val_2_}',
                     f'{split_col_2_} <= {split_val_2_}']
        elif split_first == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2_}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2_}',
                     f'{split_col_} > {split_val}']
        else:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2_}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2_}',
                     f'{split_col_} > {split_val} & {split_col_2_} <= {split_val_2_}', 
                     f'{split_col_} > {split_val} & {split_col_2_} > {split_val_2_}']
                
        idx = pd.MultiIndex.from_product([['dTm', 'ddG'], split, cols])
        df_out = pd.DataFrame(index=idx)

        for meas in measurements:
            for sp in split:
                cur_df_bin = db_discrete_bin.copy(deep=True)
                cur_df_discrete = db_discrete.copy(deep=True)
                cur_df_cont = db_internal.copy(deep=True)

                if split_col_ != 'tmp' and split_col_2_ != 'tmp2' and split_col_ != split_col_2_:
                    if len(sp.split('&')) > 1:
                        if '>' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                        elif '<=' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]

                        if '>' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2_]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2_]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2_]
                        elif '<=' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2_]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2_]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2_]

                    elif len(sp.split('&')) == 1:
                        if not split_first:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]
                        elif not split_last:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2_]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2_]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2_]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2_]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2_]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2_]

                elif split_col_ == split_col_2_:
                    if ('>' in sp and not '>=' in sp):
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    elif '<=' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val_2_]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val_2_]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val_2_]
                    else:
                        cur_df_bin = cur_df_bin.loc[(cur_df_bin[split_col_] > split_val_2_) & (cur_df_bin[split_col_] <= split_val)]
                        cur_df_discrete = cur_df_discrete.loc[(cur_df_discrete[split_col_] > split_val_2_) & (cur_df_discrete[split_col_] <= split_val)]
                        cur_df_cont = cur_df_cont.loc[(cur_df_cont[split_col_] > split_val_2_) & (cur_df_cont[split_col_] <= split_val)]
                        
                elif split_col_2_ == 'tmp2' and split_col_ != 'tmp':
                    if '>' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    else:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]                  
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val] 
                
                for col in cols:
                    try:
                        pred_df_cont = cur_df_cont[[col,meas,f'runtime_{col}']].dropna()
                        df_out.loc[(meas,sp,col), 'runtime'] = pred_df_cont[f'runtime_{col}'].sum()
                        pred_df_cont = pred_df_cont.drop(f'runtime_{col}', axis=1)
                    except KeyError:
                        pred_df_cont = cur_df_cont[[col,meas]].dropna()
                        df_out.loc[(meas,sp,col), 'runtime'] = np.nan    

                    pred_df_bin = cur_df_bin[[col,meas]].dropna()

                    if 'n' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'n'] = len(pred_df_bin)
                        saved_n = len(pred_df_bin)
                    if len(pred_df_bin) == 0:
                        continue
                    #    raise AssertionError(f'There are no {col} predictions in this scaffold ({sp}) for group {group_name}!')
                    
                    try:
                        tn, fp, fn, tp = metrics.confusion_matrix(pred_df_bin[meas], pred_df_bin[col]).ravel()
                    except:
                        tn, fp, fn, tp = 1,1,1,1
                    
                    if 'tp' in stats or stats == (): df_out.loc[(meas,sp,col), 'tp'] = tp
                    if 'fp' in stats or stats == (): df_out.loc[(meas,sp,col), 'fp'] = fp
                    if 'tn' in stats or stats == (): df_out.loc[(meas,sp,col), 'tn'] = tn 
                    if 'fn' in stats or stats == (): df_out.loc[(meas,sp,col), 'fn'] = fn   
                    if 'sensitivity' in stats or stats == (): df_out.loc[(meas,sp,col), 'sensitivity'] = tp/(tp+fn)
                    if 'specificity' in stats or stats == (): df_out.loc[(meas,sp,col), 'specificity'] = tn/(tn+fp)
                    if 'PPV' in stats or stats == (): df_out.loc[(meas,sp,col), 'PPV'] = tp/(tp+fp)
                    if 'pred_positives' in stats or stats == (): df_out.loc[(meas,sp,col), 'pred_positives'] = tp+fp
                    if 'accuracy' in stats or stats == (): df_out.loc[(meas,sp,col), 'accuracy'] = metrics.accuracy_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'f1_score' in stats or stats == (): df_out.loc[(meas,sp,col), 'f1_score'] = metrics.f1_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'MCC' in stats or stats == (): df_out.loc[(meas,sp,col), 'MCC'] = metrics.matthews_corrcoef(pred_df_bin[meas], pred_df_bin[col])

                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna()
                    try:
                        pred_df_discrete[meas] = pred_df_discrete[meas].astype(int)
                        auroc = metrics.roc_auc_score(pred_df_discrete[meas], pred_df_discrete[col])
                        auprc = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                        if 'auroc' in stats or stats == (): df_out.loc[(meas,sp,col), 'auroc'] = auroc
                        if 'auprc' in stats or stats == (): df_out.loc[(meas,sp,col), 'auprc'] = auprc
                    except Exception as e:
                        if not quiet: print('Couldn\'t compute AUC:', e)

                    pred_df_cont = cur_df_cont[[col,meas]].dropna().join(db_grouper)

                    for stat in [s for s in stats if 'recall@' in s] if stats != () else ['recall@k0.0', 'recall@k1.0']:
                        k = stat.split('@')[-1].strip('k')
                        k = 0. if k == '' else float(k)
                        
                        pred_df_discrete_k = pred_df_cont.copy(deep=True).drop_duplicates()
                        pred_df_discrete_k[meas] = pred_df_discrete_k[meas].apply(lambda x: 1 if x > k else 0)
                        stable_ct = pred_df_discrete_k[meas].sum()
                        gain = pred_df_cont.loc[pred_df_cont[meas] > k, meas].sum()

                        df_out.loc[(meas,sp,col), f'{k}_n_stable'] = stable_ct
                        sorted_preds = pred_df_discrete_k.sort_values(col, ascending=False).index
                        
                        if stable_ct > 0:
                            df_out.loc[(meas,sp,col), f'recall@k{k}'] = pred_df_discrete_k.loc[sorted_preds[:stable_ct], meas].sum() / stable_ct
                        else:
                            df_out.loc[(meas,sp,col), f'recall@k{k}'] = 0
                            
                        if gain > 0:
                            df_out.loc[(meas,sp,col), f'gain@k{k}'] = pred_df_cont.drop_duplicates().loc[(sorted_preds[:stable_ct]), meas].sum() / gain
                        else:
                            df_out.loc[(meas,sp,col), f'gain@k{k}'] = 0

                    if 'mean_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].mean()
                    if 'net_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'net_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].sum()
                    if 'mean_stable_pred' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stable_pred'] = pred_df_cont.loc[pred_df_cont[meas]>0, col].mean()
                    if 'mse' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_squared_error'] = metrics.mean_squared_error(pred_df_cont[meas], pred_df_cont[col])

                    if ('mean_t1s' in stats) or (stats == ()): 
                        top_1_stab = 0
                        unique_groups_len = len(pred_df_cont[grouper].unique())
                        if unique_groups_len > 0:
                            for code, group in pred_df_cont.groupby(grouper):
                                top_1_stab += group.sort_values(col, ascending=False)[meas].head(1).item()
                            df_out.loc[(meas,sp,col), 'mean_t1s'] = top_1_stab / unique_groups_len

                    if ('mean_reciprocal_rank' in stats) or (stats == ()): 
                        reciprocal_rank_sum = 0
                        unique_groups = pred_df_cont[grouper].unique()
                        if len(unique_groups) > 0:
                            for code, group in pred_df_cont.groupby(grouper):
                                group = group.drop_duplicates()
                                sorted_group = group.sort_values(col, ascending=False)
                                highest_meas_rank = sorted_group[meas].idxmax()
                                rank_of_highest_meas = sorted_group.index.get_loc(highest_meas_rank)
                                
                                if type(rank_of_highest_meas) in [slice, list, bool]:
                                    print('Something went wrong with MRR for', col, code)
                                    continue
                                try:
                                    rank_of_highest_meas += 1
                                    reciprocal_rank_sum += 1 / rank_of_highest_meas
                                except:
                                    print('Something went wrong with MRR for', col, code)
                                    continue
                            df_out.loc[(meas, sp, col), 'mean_reciprocal_rank'] = reciprocal_rank_sum / len(unique_groups)
                    
                    if ('ndcg' in stats) or (stats == ()):
                        df_out.loc[(meas,sp,col), 'ndcg'] = compute_ndcg_flexible(pred_df_cont, col, meas, threshold=threshold)
                        cum_ndcg = 0
                        w_cum_ndcg = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for code, group in pred_df_cont.groupby(grouper): 
                            if len(group.loc[group[meas]>threshold]) > 1 and not all(group[meas]==group[meas].iloc[0]):
                                cur_ndcg = compute_ndcg_flexible(group, col, meas, threshold=threshold)
                                if np.isnan(cur_ndcg): continue
                                cum_ndcg += cur_ndcg
                                cum_d += 1
                                w_cum_ndcg += cur_ndcg * np.log(len(group.loc[group[meas]>0]))
                                w_cum_d += np.log(len(group.loc[group[meas]>0]))
                                cum_muts += len(group.loc[group[meas]>0])
                        df_out.loc[(meas,sp,col), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_ndcg'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_ndcg'] = cum_muts
                    
                    if ('pearson' in stats) or (stats == ()):
                        if len(pred_df_cont) > 1:
                            whole_r, _ = pearsonr(pred_df_cont[col], pred_df_cont[meas])
                            df_out.loc[(meas,sp,col), 'pearson'] = whole_r
                        else:
                             df_out.loc[(meas,sp,col), 'pearson'] = np.nan

                    if ('spearman' in stats) or (stats == ()):
                        if len(pred_df_cont) > 1:
                            whole_p, _ = spearmanr(pred_df_cont[col], pred_df_cont[meas])
                            df_out.loc[(meas,sp,col), 'spearman'] = whole_p
                        else:
                            df_out.loc[(meas,sp,col), 'spearman'] = np.nan
                        
                        cum_p = 0
                        w_cum_p = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for code, group in pred_df_cont.groupby(grouper):
                            if len(group) > 1 and not all(group[meas]==group[meas].iloc[0]):
                                spearman, _ = spearmanr(group[col], group[meas])
                                if np.isnan(spearman): spearman=0
                                cum_p += spearman
                                cum_d += 1
                                w_cum_p += spearman * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_spearman'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_spearman'] = cum_muts

                    if ('gain' in stats) or (stats == ()):
                        cum_recalled = 0
                        cum_gain = 0
                        cum_stable_ct = 0
                        cum_possible_gain = 0
                        k_ = 0.5
                        for i, (code, group) in enumerate(pred_df_cont.groupby(grouper)):
                            group_discrete_k_ = group.copy(deep=True)
                            group_discrete_k_[meas] = group_discrete_k_[meas].apply(lambda x: 1 if x > k_ else 0)
                            stable_ct = group_discrete_k_[meas].sum()
                            cum_stable_ct += stable_ct
                            possible_gain = group.loc[group[meas] > k_, meas].sum()
                            cum_possible_gain += possible_gain
                            
                            sorted_preds = group_discrete_k_.sort_values(col, ascending=False).index
                            cum_recalled += group_discrete_k_.loc[sorted_preds[:stable_ct], meas].sum()
                            cum_gain += group.loc[sorted_preds[:stable_ct], meas].sum()

                        df_out.loc[(meas,sp,col), f'{k_}_n_stable'] = cum_stable_ct
                        if cum_stable_ct > 0:
                            df_out.loc[(meas,sp,col), f'frac_cum_recall@k_{k_}'] = cum_recalled / cum_stable_ct
                            df_out.loc[(meas,sp,col), f'mean_gain@k_{k_}'] = cum_gain / cum_stable_ct
                        else:
                            df_out.loc[(meas,sp,col), f'frac_cum_recall@k_{k_}'] = 0
                            df_out.loc[(meas,sp,col), f'mean_gain@k_{k_}'] = 0
                            
                        if cum_possible_gain > 0:
                            df_out.loc[(meas,sp,col), f'frac_cum_gain@k_{k_}'] = cum_gain / cum_possible_gain
                        else:
                            df_out.loc[(meas,sp,col), f'frac_cum_gain@k_{k_}'] = 0
                            
                        df_out.loc[(meas,sp,col), f'n_muts_mean_gain@k_{k_}'] = cum_stable_ct
                        df_out.loc[(meas,sp,col), f'n_muts_gain'] = cum_stable_ct

                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna().join(db_grouper)
                    
                    if ('auprc' in stats) or (stats == ()):
                        cum_ps = 0
                        w_cum_ps = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for _, group in pred_df_discrete.groupby(grouper): 
                            if len(group) > 1:
                                cur_ps = metrics.average_precision_score(group[meas], group[col])
                                if np.isnan(cur_ps): continue
                                cum_ps += cur_ps
                                cum_d += 1
                                w_cum_ps += cur_ps * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_auprc'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_auprc'] = cum_muts

                    if split_col_ == 'tmp':
                        if ('auppc' in stats) or (stats == ()):
                            percentiles = [str(int(s)/10)+'%' for s in range(1, 1000)]
                            percentile_values = [float(s.split('%')[0]) for s in percentiles]
                        else:
                            percentiles = [s for s in stats if '%' in s]
                            percentile_values = [float(s.split('%')[0]) for s in percentiles]

                        if grouper is not None:
                            results_df = pred_df_cont.groupby(grouper).apply(
                                calculate_ppc, pred_col=col, meas=meas, percentile_values=percentile_values, threshold=threshold
                                ).reset_index()
                        else:
                            results_df = calculate_ppc(pred_df_cont, pred_col=col, meas=meas, percentile_values=percentile_values, threshold=threshold)
                        
                        stat_dict = {}
                        for stat in percentiles:
                            try:
                                stat_dict[stat] = results_df[f"frac_{stat}"].mean()
                            except (ZeroDivisionError, KeyError):
                                stat_dict[stat] = 0

                        df_out.loc[(meas, sp, col), percentiles] = list(stat_dict.values())
                        df_out.loc[(meas, sp, col), 'auppc'] = df_out.loc[(meas, sp, col), percentiles].mean()

                        if ('aumsc' in stats) or (stats == ()):
                            percentiles_msc = [str(int(s)/10)+'$' for s in range(1, 1000)]
                            percentile_values_msc = [float(s.split('$')[0]) for s in percentiles_msc]
                        else:
                            percentiles_msc = [s for s in stats if '$' in s]
                            percentile_values_msc = [float(s.split('$')[0]) for s in percentiles_msc]

                        if grouper is not None:
                            results_df_msc = pred_df_cont.groupby(grouper).apply(
                                calculate_msc, pred_col=col, meas=meas, percentile_values=percentile_values_msc
                                ).reset_index()
                        else:
                            results_df_msc = calculate_msc(pred_df_cont, pred_col=col, meas=meas, percentile_values=percentile_values_msc)            

                        stat_dict_msc = {}
                        for stat in percentiles_msc:
                            try:
                                stat_dict_msc[stat] = results_df_msc[f"frac_{stat}"].mean()
                            except (ZeroDivisionError, KeyError):
                                stat_dict_msc[stat] = 0

                        df_out.loc[(meas, sp, col), percentiles_msc] = list(stat_dict_msc.values())
                        df_out.loc[(meas, sp, col), 'aumsc'] = df_out.loc[(meas, sp, col), percentiles_msc].mean()

        # Add group name to the dataframe for this group
        df_out['group'] = str(group_name) if not isinstance(group_name, (str, int, float)) else group_name
        dfs_out.append(df_out)

    if len(dfs_out) > 0:
        df_out = pd.concat(dfs_out)
    else:
        return pd.DataFrame()

    df_out = df_out.reset_index()
    
    # add labels for the input information used by the model
    # relying on global mapping_categories as in original
    try:
        for k,v in mapping_categories.items():
            for m in v:
                df_out.loc[df_out['level_2'].str.contains(m, na=False), 'model_type'] = k
    except NameError:
        pass

    df_out = df_out.rename({'level_0': 'measurement', 'level_1': 'class', 'level_2': 'model'}, axis=1)

    # Include 'group' in the index to differentiate rows
    df_out = df_out.set_index(['measurement', 'model_type', 'model', 'class', 'group'])
    # sort by measurement type, and then model type within each measurement type
    df_out = df_out.sort_index(level=1).sort_index(level=0)

    return df_out.dropna(how='all')

def annotate_points(ax, data, x_col, y_col, hue_col, x_values, text_offset=(0, 0), spacing=0.02):
    line_colors = {}
    for line in ax.lines:
        label = line.get_label()
        color = line.get_color()
        line_colors[label] = color

    for x_val in x_values:
        models_and_points = []
        for model, model_data in data.groupby(hue_col):
            value_row = model_data.loc[model_data[x_col] == x_val]
            if not value_row.empty:
                if len(value_row) > 1:
                    x, y = value_row[x_col].values[0], value_row[y_col].values.mean()
                else:
                    x, y = value_row[x_col].values[0], value_row[y_col].values[0]
                models_and_points.append((model, x, y))

        # Sort models_and_points by y values to space them evenly
        models_and_points = sorted(models_and_points, key=lambda x: x[2], reverse=True)
        print(models_and_points)

        # Calculate annotation positions and add annotations
        y_annot = max(y for _, _, y in models_and_points) + text_offset[1]
        for model, x, y in models_and_points:
            ax.annotate(f"{y:.2f}", (x, y),
                        xytext=(x + text_offset[0], y_annot),
                        arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                        fontsize=12, color=line_colors[model])
            y_annot -= spacing
            ax.axvline(x=x, color='r', linestyle='dashed')
            

def compute_dddg(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every column X that has a matching column X_additive,
    compute df[X] - df[X_additive] and store the result in a new column
    where '_additive' is replaced with 'dddG'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe with new 'dddG' columns added.
    """
    for col in df.columns:
        if col.endswith("_additive"):
            base_col = col[:-9]  # strip "_additive"
            if base_col in df.columns:
                new_col = base_col + "_dddG"
                df[new_col] = df[base_col] - df[col]
        if col[:-2].endswith("_additive"):
            base_col = col[:-11] + col[-2:]  # strip "_additive_x"
            if base_col in df.columns:
                new_col = base_col + "_dddG"
                df[new_col] = df[base_col] - df[col]
    return df

def annotate_points(ax, data, x_col, y_col, hue_col, x_values, text_offset=(0, 0), spacing=0.02):
    line_colors = {}
    for line in ax.lines:
        label = line.get_label()
        color = line.get_color()
        line_colors[label] = color

    for x_val in x_values:
        models_and_points = []
        for model, model_data in data.groupby(hue_col):
            value_row = model_data.loc[model_data[x_col] == x_val]
            if not value_row.empty:
                if len(value_row) > 1:
                    x, y = value_row[x_col].values[0], value_row[y_col].values.mean()
                else:
                    x, y = value_row[x_col].values[0], value_row[y_col].values[0]
                models_and_points.append((model, x, y))

        # Sort models_and_points by y values to space them evenly
        models_and_points = sorted(models_and_points, key=lambda x: x[2], reverse=True)
        print(models_and_points)

        # Calculate annotation positions and add annotations
        y_annot = max(y for _, _, y in models_and_points) + text_offset[1]
        for model, x, y in models_and_points:
            ax.annotate(f"{y:.2f}", (x, y),
                        xytext=(x + text_offset[0], y_annot),
                        arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                        fontsize=12, color=line_colors[model])
            y_annot -= spacing
            ax.axvline(x=x, color='r', linestyle='dashed')
            

def recovery_curves(rcv, models=['cartesian_ddg', 'ddG', 'dTm', 'random'], measurements=('ddG'), plots=('auppc', 'aumsc'), title='Dataset'):

    font = {'size': 12}
    matplotlib.rc('font', **font)

    if len(plots) == 1:
        if len(measurements) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300)
            ax_list = [axes]  
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
            ax_list = [axes[0], axes[1]]        
    elif len(plots) == 2:
        if len(measurements) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
            ax_list = [axes[0], axes[1]]
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), dpi=300)
            ax_list = [axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]]

    d5 = rcv.reset_index()
    d5 = d5.loc[d5['model'].isin(models)].set_index(['measurement', 'model_type', 'model', 'class'])
    d5 = d5.drop([c for c in d5.columns if 'stab_' in c], axis=1)

    # Function to extract base model name (removing numeric suffixes)
    def get_base_model_name(model_name):
        # Match patterns like "model_name_1", "model_name_2.0", etc.
        match = re.match(r'^(.*?)_\d+(.*)?$', model_name)
        if match:
            full_match = match.group(1)
            if match.group(2):
                full_match += match.group(2)
            return full_match
        return model_name

    i = 0

    if 'ddG' in measurements:
        # for plotting recovery over thresholds
        if 'auppc' in plots:
            recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
            recov = recov.loc[recov['model'] != 'dTm']
            recov = recov.loc[recov['measurement'] == 'ddG']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            
            # Add base model column
            recov['base_model'] = recov['model'].apply(get_base_model_name)
            
            # Process data
            melted_1 = recov.melt(id_vars=['model', 'base_model'], value_vars=[c for c in recov.columns if c not in ['model', 'base_model']], 
                                  var_name="variable", value_name="value")
            recov = melted_1
            recov['variable'] = recov['variable'].str.strip('%').astype(float)
            
            # Sort random to the end
            recov['sort_helper'] = recov['model'] == 'random'
            recov = recov.sort_values(by='sort_helper').drop('sort_helper', axis=1)
            
            # Get unique base models to plot
            unique_base_models = sorted(recov['base_model'].unique().tolist())
            if 'random' in unique_base_models:
                unique_base_models.remove('random')
                unique_base_models.append('random')  # Move to end
            
            cmap = get_color_mapping(pd.DataFrame({'model': unique_base_models}), 'model')
            print(cmap)
            
            # For each base model, calculate mean and std
            for base_model in unique_base_models:
                subset = recov[recov['base_model'] == base_model]
                
                # Group by variable and calculate mean and std
                grouped = subset.groupby('variable').agg({'value': ['mean', 'std']})
                grouped.columns = ['mean', 'std']
                grouped = grouped.reset_index()
                
                color = cmap[base_model]
                
                # Plot mean line
                ax_ = sns.lineplot(data=grouped, x='variable', y='mean', ax=ax_list[i], 
                                  label=base_model, color=color)
                
                # Add shaded area for standard deviation
                if len(subset['model'].unique()) > 1:  # Only add std if we have more than one model in the group
                    ax_list[i].fill_between(grouped['variable'], 
                                          grouped['mean'] - grouped['std'],
                                          grouped['mean'] + grouped['std'], 
                                          alpha=0.2, color=color)
                
                # Special handling for zero shot
                if 'esm3' in base_model:
                    ax_.lines[-1].set_linestyle('--')

                # Special handling for MSR
                if 'esm_msr' in base_model:
                    ax_.lines[-1].set_linestyle('-')

                # Special handling for random
                if 'random' in base_model:
                    ax_.lines[-1].set_linestyle('--')

            if len(measurements) > 1:
                ax_list[i].set_xlabel('')
            else:
                ax_list[i].set_xlabel('top x% of ranked predictions')
            ax_list[i].set_ylabel('fraction stabilizing (ΔΔG > 0.5 kcal/mol)')
            #annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=left_text_offset, spacing=left_spacing/2)
            i += 1

        if 'aumsc' in plots:
            recov = d5[[c for c in d5.columns if '$' in c]].reset_index()
            recov = recov.loc[recov['model'] != 'dTm']
            recov = recov.loc[recov['measurement'] == 'ddG']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            
            # Add base model column
            recov['base_model'] = recov['model'].apply(get_base_model_name)
            
            # Process data
            recov = recov.melt(id_vars=['model', 'base_model'], var_name="variable", value_name="value")
            recov['variable'] = recov['variable'].str.strip('$').astype(float)
            
            # Sort random to the end
            recov['sort_helper'] = recov['model'] == 'random'
            recov = recov.sort_values(by='sort_helper').drop('sort_helper', axis=1)
            
            # Get unique base models to plot
            unique_base_models = sorted(recov['base_model'].unique().tolist())
            if 'random' in unique_base_models:
                unique_base_models.remove('random')
                unique_base_models.append('random')  # Move to end
            
            cmap = get_color_mapping(pd.DataFrame({'model': unique_base_models}), 'model')
            if not cmap:
                cmap = get_color_mapping(pd.DataFrame({'model': unique_base_models}), 'model')
            
            # For each base model, calculate mean and std
            for base_model in unique_base_models:
                subset = recov[recov['base_model'] == base_model]
                
                # Group by variable and calculate mean and std
                grouped = subset.groupby('variable').agg({'value': ['mean', 'std']})
                grouped.columns = ['mean', 'std']
                grouped = grouped.reset_index()
                
                color = cmap[base_model]
                
                # Plot mean line
                ax_ = sns.lineplot(data=grouped, x='variable', y='mean', ax=ax_list[i], 
                                  label=base_model, color=color)
                
                # Add shaded area for standard deviation
                if len(subset['model'].unique()) > 1:  # Only add std if we have more than one model in the group
                    ax_list[i].fill_between(grouped['variable'], 
                                          grouped['mean'] - grouped['std'],
                                          grouped['mean'] + grouped['std'], 
                                          alpha=0.2, color=color)

                # Special handling for zero shot
                if 'esm3' in base_model:
                    ax_.lines[-1].set_linestyle('--')

                # Special handling for MSR
                if 'esm_msr' in base_model:
                    ax_.lines[-1].set_linestyle('-')

                # Special handling for random
                if 'random' in base_model:
                    ax_.lines[-1].set_linestyle('--')
                    
            ax_list[i].set_xlabel('top x% of ranked predictions')
            ax_list[i].set_ylabel('mean stabilization (kcal/mol)')
            #annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=right_text_offset, spacing=right_spacing*12)
            i += 1

    # Handle legends
    handles, labels = ax_list[0].get_legend_handles_labels()
    
    if len(ax_list) > 1:
        for ax in ax_list:
            try:
                ax.get_legend().remove()
                ax.set_title(title)
            except:
                pass
    else:
        try:
            ax_list[0].get_legend().remove()
            ax_list[0].set_title(title)
        except:
            pass

    try:
        labels = [remap_names[name] if name in remap_names.keys() else name for name in labels]
    except NameError:
        pass  # remap_names_2 not defined

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.), ncol=2)
    plt.tight_layout()

    plt.show()
    return fig


def compute_ndcg_flexible(df, pred_col, true_col, *,
                          top_n=None, percentile=None, threshold=None,
                          ignore_ties=True, exponential_relevance=False):
    """
    Compute NDCG alongside physical hit-rate metrics for a defined budget (k).
    
    Returns:
        Tuple: (NDCG_score, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool)
    """
    flags = [top_n is not None, percentile is not None, threshold is not None]
    if sum(flags) != 1:
        raise ValueError("Specify exactly one of top_n, percentile, or threshold.")

    df = df[[pred_col, true_col]].dropna()
    if len(df) < 2:
        return np.nan, 0, 0, 0

    y_score = df[pred_col].to_numpy().reshape(1, -1)
    y_true = df[true_col].to_numpy()

    rel_floor = threshold if threshold is not None else 0.0
    
    # 1. Total Hits in Pool
    total_hits_in_pool = int(np.sum(y_true > rel_floor))
    
    y_true_processed = np.where(y_true <= rel_floor, 0.0, y_true)

    if total_hits_in_pool == 0:
        return np.nan, 0, 0, 0

    if exponential_relevance:
        y_true_processed = np.exp(y_true_processed) - 1.0

    y_true_processed = y_true_processed.reshape(1, -1)
    n = y_true.size

    if threshold is not None:
        k = None
    elif top_n is not None:
        if top_n <= 0:
            return np.nan, 0, 0, total_hits_in_pool
        k = min(int(top_n), n)
    else:
        k = max(1, int(np.ceil(percentile * n)))
        k = min(k, n)

    # Calculate NDCG
    ndcg_val = metrics.ndcg_score(y_true_processed, y_score, k=k, ignore_ties=ignore_ties)
    
    # 2. Maximum Possible Hits Scored
    ideal_hits_at_k = min(total_hits_in_pool, k) if k is not None else total_hits_in_pool
    
    # 3. The Model's Actual Hits Scored
    # Sort the true relevances based on the model's predicted ranking
    sorted_indices = np.argsort(-y_score[0])
    if k is not None:
        model_top_k_relevances = y_true_processed[0][sorted_indices][:k]
    else:
        model_top_k_relevances = y_true_processed[0][sorted_indices]
        
    model_hits_at_k = int(np.sum(model_top_k_relevances > 0))

    return ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool


def custom_rho(df_preds, pred_col='esm_msr', label='ddG_ML', grouped=False):

    df_preds = df_preds.dropna(subset=label)

    if grouped:
        corrs = df_preds.groupby('code')[[label, pred_col]].corr('spearman').reset_index()
        #print(corrs)
        corrs = corrs.loc[corrs['level_1']==label].set_index('code').drop(['level_1'], axis=1)
        #print(corrs.mean()[pred_col])
        corr = corrs.mean()[pred_col]
        n = len(df_preds[[label, pred_col]].dropna())

    else:
        #print(df_preds[[label, pred_col]].corr('spearman').iloc[0, 1])
        corr = df_preds[[label, pred_col]].corr('spearman').iloc[0, 1]
        n = len(df_preds[[label, pred_col]].dropna())

    return corr, n


def custom_ndcg(df_preds, pred_col='esm_msr', label='ddG_ML', grouped=False, percentile=None, top_n=None, threshold=None):
    
    # Drop rows without ground truth labels
    df_preds = df_preds.dropna(subset=[label])

    if grouped:
        ndcgs = []
        ns = []
        
        for code, group in df_preds.groupby('code'):
            ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = compute_ndcg_flexible(
                group, pred_col, label, percentile=percentile, top_n=top_n, threshold=threshold
            )
            
            ndcgs.append(ndcg_val)
            
            # Catch percentile as well as top_n
            if top_n is not None or percentile is not None:
                ns.append(ideal_hits_at_k)
            elif threshold is not None:
                ns.append(total_hits_in_pool)
                
        ndcgs = pd.Series(ndcgs)
        ns = pd.Series(ns)
        
        # Optional: Warn if groups were dropped due to no hits
        valid_groups = ndcgs.notna().sum()
        if valid_groups < len(ndcgs):
             print(f"Warning: {len(ndcgs) - valid_groups} groups were excluded from NDCG mean due to 0 hits in the plate/pool.")

        ndcg = ndcgs.mean()
        n = ns.sum()
        
        return ndcg, n

    else:
        # We only need the prediction and label columns for the un-grouped evaluation
        df_eval = df_preds[[label, pred_col]]
        
        ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = compute_ndcg_flexible(
            df_eval, pred_col, label, percentile=percentile, top_n=top_n, threshold=threshold
        )
        
        if top_n is not None or percentile is not None:
            n = ideal_hits_at_k
        elif threshold is not None:
            n = total_hits_in_pool
            
        return ndcg_val, n


def custom_rmse(df_preds, pred_col='esm_msr', label='ddG_ML', grouped=False):
    """
    Computes Root Mean Squared Error (RMSE) between prediction and label columns.
    
    If grouped=True, computes RMSE for each group defined by 'code' column 
    and returns the mean of those RMSEs.
    """
    # FLAW FIXED: Original code only dropped subset=label. 
    # RMSE requires valid values in both prediction and label columns.
    df_preds = df_preds.dropna(subset=[label, pred_col]).copy()

    if grouped:
        # Calculate RMSE for each group individually
        # Formula: sqrt(mean((pred - label)^2))
        group_rmses = df_preds.groupby('code').apply(
            lambda x: np.sqrt(((x[pred_col] - x[label]) ** 2).mean())
        )
        
        # Return the mean of the RMSEs (Macro-average across groups)
        metric = group_rmses.mean()
        n = len(df_preds)

    else:
        # Global RMSE calculation
        diff = df_preds[pred_col] - df_preds[label]
        metric = np.sqrt((diff ** 2).mean())
        n = len(df_preds)

    return metric, n


def custom_gain(df_preds, pred_col='esm_msr', label='ddG_ML', threshold=0, exclude_negative=True, normalize=True):

    df_preds = df_preds.dropna(subset=label)

    df_preds = df_preds[['code', label, pred_col]].dropna()
    df_preds = df_preds.loc[df_preds[pred_col] > threshold]
    out = df_preds[label].sum()

    if exclude_negative:
        #assert normalize, "Normalization is required when excluding negative predictions to maintain interpretability."
        df_preds.loc[df_preds[pred_col]<threshold, label] = 0

    if normalize:    
        out /= len(df_preds)

    return out


def bootstrap_by_grouper(dbf, n_bootstraps, grouper='code', drop=True, noise=0, target='ddG', duplicates=True):
    if grouper == 'code' and not 'code' in dbf.columns:
        dbf['code'] = dbf.index.str[:4]
    if grouper is not None:
        groups = list(dbf[grouper].unique())
    else:
        groups = list(set(dbf.index))
    out = []
    for i in range(n_bootstraps):
        redraw = []
        if grouper is not None:
            while len(redraw) < len(groups):
                group = random.choice(groups)
                new_db = dbf.loc[dbf[grouper]==group]
                if drop:
                    new_db = new_db.drop(grouper, axis=1)
                redraw.append(new_db)
            df_bs = pd.concat(redraw, axis=0)
        else:
            df_bs = dbf.sample(frac=1, replace=True)
        if noise > 0:
            df_bs[target] += np.random.normal(scale=noise, size=len(df_bs))
        if not duplicates:
            df_bs = df_bs.drop_duplicates()
        out.append(df_bs)
    return out

# --- Helper Functions for Log Transformation ---

def signed_log_transform(x, linear_threshold=1.0):
    """
    Applies a signed log transformation: sign(x) * log10(|x| + 1).
    Shifted by 1 to handle 0 gracefully and maintain continuity.
    """
    return np.sign(x) * np.log10(np.abs(x) + 1)

def inverse_signed_log_transform(y, linear_threshold=1.0):
    """
    Inverse of the signed log transformation.
    """
    return np.sign(y) * (10**np.abs(y) - 1)

def set_log_ticks(ax, min_val, max_val, axis='y'):
    """
    Manually sets ticks at powers of 10 (and negatives).
    """
    # Generate powers of 10 based on range
    max_log = int(np.ceil(np.log10(max(abs(min_val), abs(max_val)) + 1)))
    
    # Create ticks: 0, +/-1, +/-10, +/-100...
    ticks_raw = [0]
    for i in range(0, max_log + 1):
        val = 10**i
        if val <= abs(max_val) + 1: # Add buffer check
            ticks_raw.append(val)
        if -val >= min_val - 1:
            ticks_raw.append(-val)
    
    ticks_raw = sorted(list(set(ticks_raw)))
    
    # Transform positions
    ticks_pos = signed_log_transform(np.array(ticks_raw))
    
    if axis == 'y':
        ax.yaxis.set_major_locator(FixedLocator(ticks_pos))
        ax.set_yticklabels([str(t) for t in ticks_raw])
    else:
        ax.xaxis.set_major_locator(FixedLocator(ticks_pos))
        ax.set_xticklabels([str(t) for t in ticks_raw])

# --- Plotting Functions ---

def custom_plot(data, x, y, hue, order, ax, 
                   use_color=None, legend_labels=None, legend_colors=None,
                   group_plot_width=0.8, hue_spacing=0.05, 
                   orientation='vertical', alpha=0.8,
                   # Retaining unused args to maintain compatibility
                   std=True, capsize=5, bar_group_spacing_factor=0.95, 
                   count_label_fontsize=14, highlight_outline_width=1.5, 
                   variable_width=True, min_width_ratio=0.1, 
                   annotation_position=None, count_columns=None):
    """
    Generates a 'Raincloud' style plot (Half-Violin + Boxplot + Scatter) for the given data.
    Returns (legend_elements, medians_dict).
    medians_dict is structure {x_val: {hue_val: median_value}}
    """
    
    # Identify bootstrap columns for y using regex to match {y}_{digits}
    pattern = re.compile(f"^{re.escape(y)}_\\d+$")
    y_cols = [c for c in data.columns if pattern.match(c)]
    
    # Fallback: if no bootstrap columns found, check if y itself is a column
    if not y_cols:
        if y in data.columns:
            y_cols = [y]
        else:
            return [], {}

    if not order:
        unique_x_values = sorted(list(data[x].unique()))
    else:
        unique_x_values = [c for c in order if c in list(data[x].unique())]
        
    if not unique_x_values:
        return [], {}

    # Prepare Colors
    lut = {}
    unique_hue_values = []
    if legend_labels is not None:
        unique_hue_values = legend_labels
        if legend_colors is not None:
             lut = dict(zip(legend_labels, legend_colors))
    else:
        unique_hue_values = sorted(list(data[hue].unique()))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        palette_to_use = [default_colors[i % len(default_colors)] for i in range(len(unique_hue_values))]
        lut = dict(zip(unique_hue_values, palette_to_use))

    num_hues = len(unique_hue_values)
    
    # Calculate widths
    if num_hues > 1:
        width_per_violin = (group_plot_width * 0.9) / num_hues 
    else:
        width_per_violin = group_plot_width * 0.8

    legend_elements = []
    created_legend_labels = set()
    medians_dict = {}

    for x_idx, x_val in enumerate(unique_x_values):
        medians_dict[x_val] = {}
        
        # Calculate start position for this group
        # This math centers the group of violins on the integer x_idx
        total_group_width = num_hues * width_per_violin
        start_offset = -total_group_width / 2 + width_per_violin / 2
        
        for h_idx, hue_val in enumerate(unique_hue_values):
            
            # Center position for this specific violin
            center_pos = x_idx + start_offset + (h_idx * width_per_violin)
            
            # Extract data
            subset = data[(data[x] == x_val) & (data[hue] == hue_val)]
            if subset.empty:
                continue
            
            # Flatten the bootstrap columns
            values = subset[y_cols].values.flatten()
            values = values[~np.isnan(values)]
            
            if len(values) == 0:
                continue

            medians_dict[x_val][hue_val] = np.median(values)
                
            # Plot setup
            color = use_color if use_color else lut.get(hue_val, 'grey')
            vert = True if orientation == 'vertical' else False # Enforce vertical for metric on Y
            
            # 1. Wide Background Boxplot
            # Plotted behind (zorder=1). 'showfliers=False' because we plot points later.
            ax.boxplot([values], positions=[center_pos], widths=width_per_violin * 0.8,
                       patch_artist=True, showfliers=False, vert=vert,
                       boxprops={'facecolor': color, 'alpha': 0.15, 'edgecolor': color, 'linewidth': 0.5},
                       whiskerprops={'color': color, 'alpha': 0.3},
                       capprops={'color': color, 'alpha': 0.3},
                       medianprops={'color': 'black', 'alpha': 0.7, 'linewidth': 2},
                       zorder=1)
            
            # 2. Half Violin (Left Side)
            parts = ax.violinplot([values], positions=[center_pos], vert=vert, widths=width_per_violin,
                                  showmeans=False, showmedians=False, showextrema=False)
            
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(None)
                pc.set_alpha(alpha)
                pc.set_zorder(2)
                
                # Clip to left half
                # Modify vertices: set any x > center_pos to center_pos
                path = pc.get_paths()[0]
                verts = path.vertices
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, center_pos)
            
            # 3. Scatter Points (Right Side)
            # Add jitter to right side
            # Jitter width is half the violin width roughly
            jitter_width = width_per_violin * 0.3
            x_jitter = np.abs(np.random.normal(0, jitter_width/2, size=len(values)))
            x_jitter = np.clip(x_jitter, 0, width_per_violin/2) # Bound jitter
            
            # Downsample for visualization if too many points (optional, keeping all for now)
            plot_vals = values
            plot_jitter = x_jitter
            
            ax.scatter(center_pos + plot_jitter, plot_vals, s=4, color=color, alpha=0.8, 
                       edgecolors='none', zorder=3)

            # Legend
            if not use_color and hue_val not in created_legend_labels:
                legend_elements.append(Patch(facecolor=color, label=str(hue_val), alpha=alpha))
                created_legend_labels.add(hue_val)

    # Basic Tick Setup (Overridden by compare_performance usually)
    if orientation == 'vertical':
        ax.set_xticks(range(len(unique_x_values)))
        ax.set_xticklabels(unique_x_values)
        ax.set_xlim(-0.6, len(unique_x_values) - 0.4)
        ax.set_xlabel(x)
    else:
        ax.set_yticks(range(len(unique_x_values)))
        ax.set_yticklabels(unique_x_values)
        ax.set_ylim(-0.6, len(unique_x_values) - 0.4)
        ax.set_ylabel(x)
        
    return legend_elements, medians_dict

def compare_performance_per_group(dbc,
                        threshold_1 = 1.5, 
                        threshold_2 = None, 
                        split_col = 'hbonds', 
                        split_col_2 = None, 
                        measurement = 'ddG',
                        statistic = 'MCC',
                        count_proteins = False,
                        count_muts = False,
                        subset = None,
                        grouper = 'cluster',
                        duplicates = False,
                        order = None,
                        legend_order = None,
                        drop_label = False,
                        asterisk = (),       
                        double_asterisk = (), 
                        split_first = True,
                        split_last = True,
                        legend_loc = 'below',
                        figsize = (12, 12), 
                        orientation = 'vertical',
                        alt_stat_name = None,
                        use_signed_log_transform = True,
                        n_classes = 2,
                        split_left_model = False,
                        left_group_label = None,
                        right_group_label = None,
                        fillna_value = 0,
                        annotation_stat = 'mean' # Options: 'mean', 'median'
                        ):
    """
    Compares performance across different models and data splits using group-wise statistics.
    Updated to include toggle for mean/median text annotations.
    """
    
    my_palette = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']
    rename_dict = {'delta_kdh': 'Δ hydrophobicity', 'delta_vol': 'Δ volume', 'rel_ASA': 'relative ASA', 'neff': 'N eff. seqs'}

    # Data Filtering
    db_complete = dbc.copy(deep=True)
    if subset is not None:
        measurement_cols_to_keep = [m_col for m_col in db_complete.columns if measurement in m_col]
        cols_to_keep = subset + [c for c in dbc.columns if '_dir' not in c or c in measurement_cols_to_keep]
        db_complete = db_complete[list(set(cols_to_keep))]

    db_complete = db_complete.dropna(subset=[measurement]) 
    if drop_label and 'ddG_dir' in db_complete.columns: 
        db_complete = db_complete.drop('ddG_dir', axis=1)

    # --- Compute Split Statistics Per Group ---
    if count_proteins:
        base_n_col = f'n_proteins_{statistic}'
    elif count_muts:
        base_n_col = f'n_muts_{statistic}'
    else:
        base_n_col = 'n'

    stats_to_compute_split = [statistic] + ['n']

    df_stats_split = compute_stats_multi_per_group(db_complete, quiet=True,
                                        split_col=split_col, split_col_2=split_col_2, 
                                        split_val=threshold_1, split_val_2=threshold_2, 
                                        measurements=[measurement],
                                        stats=list(set(stats_to_compute_split)), 
                                        grouper=grouper, split_first=split_first, split_last=split_last,
                                        n_classes=n_classes, duplicates=duplicates)

    df_stats_split[statistic] = df_stats_split[statistic].fillna(fillna_value)
    
    # Reset index to have 'model', 'class', 'group' as columns
    df_stats_split = df_stats_split.reset_index()
    splits_perf = df_stats_split.rename(columns={base_n_col: 'count'})
    
    if 'count' not in splits_perf.columns and base_n_col != 'n':
         if 'n' in splits_perf.columns:
             splits_perf = splits_perf.rename(columns={'n': 'count'})

    stat_col = statistic
    count_col = 'count'
    
    # Calculate mean for sorting
    splits_perf[f'{statistic}_mean'] = splits_perf.groupby('model')[stat_col].transform('mean')
    
    # Determine sorting order
    if order is None:
        model_means = splits_perf.groupby('model')[f'{statistic}_mean'].max().sort_values(ascending=False)
        ordered_models = model_means.index.tolist()
    else:
        ordered_models = [m for m in order if m in splits_perf['model'].unique()]
        if not drop_label and 'ddG_dir' in splits_perf['model'].unique() and 'ddG_dir' not in ordered_models:
            ordered_models = ['ddG_dir'] + ordered_models

    # --- Pre-calculate Means (before transform) if requested ---
    # We calculate the arithmetic mean of the raw data. 
    # If using log transform, we will project this mean onto the log scale for Y-positioning.
    perf_means_dict = {}
    count_means_dict = {}
    
    if annotation_stat == 'mean':
        def extract_means(df, col):
            # Returns {model: {class: mean_value}}
            return df.groupby(['model', 'class'])[col].mean().unstack().to_dict(orient='index')
        
        perf_means_dict = extract_means(splits_perf, stat_col)
        count_means_dict = extract_means(splits_perf, count_col)

    # --- Apply Data Transformations ---
    if use_signed_log_transform:
            splits_perf[stat_col] = signed_log_transform(splits_perf[stat_col].values)
            splits_perf[count_col] = signed_log_transform(splits_perf[count_col].values)
            
            # If using means, we must also transform the mean values to match the plot's Y-axis coordinate system
            if annotation_stat == 'mean':
                for m in perf_means_dict:
                    for c in perf_means_dict[m]:
                        perf_means_dict[m][c] = signed_log_transform(perf_means_dict[m][c])
                for m in count_means_dict:
                    for c in count_means_dict[m]:
                        count_means_dict[m][c] = signed_log_transform(count_means_dict[m][c])

    if legend_order is None:
        unique_classes = sorted(splits_perf['class'].unique())
    else:
        unique_classes = legend_order

    palette_to_use = [my_palette[i % len(my_palette)] for i in range(len(unique_classes))]
    lut = dict(zip(unique_classes, palette_to_use))
    
    # --- Figure Creation and Layout ---
    perf_axes = []
    count_axes = []
    model_groups = []
    
    if split_left_model and len(ordered_models) > 1:
        width_ratios = [1, len(ordered_models) - 1]
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300,
                                    gridspec_kw={'height_ratios': [1, 1], 'width_ratios': width_ratios, 
                                                'hspace': 0.05, 'wspace': 0.1},
                                    sharey='row', sharex='col')
        fig.patch.set_facecolor('white')
        (ax1_l, ax1_r), (ax2_l, ax2_r) = axes
        perf_axes = [ax1_l, ax1_r]
        count_axes = [ax2_l, ax2_r]
        model_groups = [[ordered_models[0]], ordered_models[1:]]
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, dpi=300, 
                                        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
        fig.patch.set_facecolor('white')
        perf_axes = [ax1]
        count_axes = [ax2]
        model_groups = [ordered_models]

    # --- Plotting Loop ---
    legend_handles = []
    perf_medians_all = {}
    count_medians_all = {}
    
    for i, models_subset in enumerate(model_groups):
        ax_p = perf_axes[i]
        ax_c = count_axes[i]
        
        # Plot Performance
        l_h, medians = custom_plot(
            data=splits_perf, x='model', y=stat_col, hue='class',
            order=models_subset, ax=ax_p, legend_labels=unique_classes,
            legend_colors=palette_to_use, orientation=orientation,
            group_plot_width=0.8
        )
        if l_h: legend_handles = l_h
        perf_medians_all.update(medians)
        
        # Plot Counts
        _, c_medians = custom_plot(
            data=splits_perf, x='model', y=count_col, hue='class',
            order=models_subset, ax=ax_c, legend_labels=unique_classes,
            legend_colors=palette_to_use, orientation=orientation,
            group_plot_width=0.8
        )
        count_medians_all.update(c_medians)
        
        # --- Axis Formatting ---
        stat_display = alt_stat_name if alt_stat_name else statistic
        default_p_label = f"{stat_display}" if use_signed_log_transform else stat_display
        default_c_label = "per domain" if use_signed_log_transform else "stabilizing per domain (N)"

        p_label, c_label = "", ""
        if split_left_model and len(model_groups) > 1:
            if i == 0:
                p_label = left_group_label if left_group_label else default_p_label
                c_label = 'Total # substitutions ' + default_c_label
            elif i == 1:
                p_label = right_group_label if right_group_label else ""
                c_label = 'Predicted most stabilizing ' + default_c_label
                ax_p.set_ylabel(p_label, fontsize=18, labelpad=10)
                ax_c.set_ylabel(c_label, fontsize=18, labelpad=10)
        else:
            if i == 0:
                p_label, c_label = default_p_label, default_c_label

        ax_p.set_ylabel(p_label, fontsize=14)
        ax_c.set_ylabel(c_label, fontsize=14)
            
        final_tick_labels = []
        for m in models_subset:
            label = remap_names_2.get(m, m) if 'remap_names_2' in globals() else m
            if m in asterisk: label += '*'
            if m in double_asterisk: label += '**'
            final_tick_labels.append(label)
        
        if orientation == 'vertical':
            plt.setp(ax_p.get_xticklabels(), visible=False)
            ax_p.set_xlabel('')
            ax_c.set_xticks(range(len(models_subset)))
            ax_c.set_xticklabels(final_tick_labels, rotation=45, ha='right', fontsize=16)
            ax_c.set_xlabel('', fontsize=14)

        ax_p.grid(axis='y', linestyle='--', alpha=0.3)
        ax_c.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Select Stats for Annotation (Mean or Median) ---
    if annotation_stat == 'mean':
        final_perf_stats = perf_means_dict
        final_count_stats = count_means_dict
    else:
        # Default to median (values returned by custom_plot)
        final_perf_stats = perf_medians_all
        final_count_stats = count_medians_all

    # --- Text Annotations ---
    def add_text_annotations(axes_list, model_subsets, stats_dict, is_count=False):
        # Parameters for fallback manual adjustment
        fallback_width_scale = 1.0 
        fallback_x_offset = -0.12   
        
        for ax_idx, models_subset in enumerate(model_subsets):
            ax = axes_list[ax_idx]
            
            # Strategy: Extract exact violin positions from the plot objects
            violins = [c for c in ax.collections if isinstance(c, mcoll.PolyCollection)]
            
            valid_violins = []
            for v in violins:
                paths = v.get_paths()
                if paths and len(paths) > 0:
                    ext = paths[0].get_extents()
                    if ext.width > 0 and ext.height > 0:
                        valid_violins.append(v)
            
            expected_count = len(models_subset) * len(unique_classes)
            
            violin_x_centers = []
            if len(valid_violins) == expected_count:
                for v in valid_violins:
                    ext = v.get_paths()[0].get_extents()
                    violin_x_centers.append((ext.x0 + ext.x1) / 2)
                violin_x_centers.sort()
            
            use_extracted = (len(violin_x_centers) == expected_count)
            
            group_width = 0.8
            flat_idx = 0 
            
            for m_idx, model_name in enumerate(models_subset):
                offsets = np.linspace(-group_width/2, group_width/2, len(unique_classes) + 1)
                centers = (offsets[:-1] + offsets[1:]) / 2
                
                for h_idx, h_val in enumerate(unique_classes):
                    
                    if use_extracted:
                        x_pos = violin_x_centers[flat_idx]
                    else:
                        x_pos = m_idx + (centers[h_idx] * fallback_width_scale) + fallback_x_offset

                    flat_idx += 1 
                    
                    # Retrieve Stat Value (could be Mean or Median)
                    if model_name in stats_dict and h_val in stats_dict[model_name]:
                        val = stats_dict[model_name][h_val]
                        
                        # Use inverse transform for label text to show real unit
                        orig_val = inverse_signed_log_transform(val) if use_signed_log_transform else val
                        
                        if is_count:
                            label_text = f"{orig_val:.2f}" #f"{int(round(orig_val))}"
                        else:
                            label_text = f"{int(orig_val)}" if abs(orig_val) >= 100 else f"{orig_val:.2f}"
                        
                        # Place Text
                        ax.text(x_pos, val, label_text, 
                                rotation=90, va='center', ha='center',
                                color='black', fontsize=12, #color=lut.get(h_val, 'black'),
                                zorder=20, #fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))
                        
                        # 2. Add the dash-like rectangle
                        # Adjust 'offset' to control the gap between the text and the dash
                        offset = -0.1 
                        rect_width = 0.05
                        rect_height = 0.01

                        ax.add_patch(Rectangle(
                            (x_pos + offset, val - (rect_height / 2)), # (x, y) bottom-left corner
                            rect_width,                                # width (thinness)
                            rect_height,                               # height (length of dash)
                            color=lut.get(h_val),
                            zorder=20
                        ))

    add_text_annotations(perf_axes, model_groups, final_perf_stats, is_count=False)
    add_text_annotations(count_axes, model_groups, final_count_stats, is_count=True)

    # --- Dashed Lines (Comparison Stats) ---
    # Using the same stat selected for annotation (Mean or Median)
    target_axes_indices = range(len(perf_axes))
    if split_left_model and len(perf_axes) > 1:
        target_axes_indices = range(1, len(perf_axes))

    if len(ordered_models) >= 2:
        second_model = ordered_models[1]
        for stats_dict, axes in [(final_perf_stats, perf_axes), (final_count_stats, count_axes)]:
            if second_model in stats_dict:
                hue_stats = stats_dict[second_model]
                for h_val, val in hue_stats.items():
                    color = lut.get(h_val, 'black')
                    for idx in target_axes_indices:
                        axes[idx].axhline(y=val, color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=5)

    # --- Final Ticks Adjustment ---
    if use_signed_log_transform:
        all_stat_vals = splits_perf[stat_col].dropna()
        orig_stat_vals = inverse_signed_log_transform(all_stat_vals)
        all_count_vals = splits_perf[count_col].dropna()
        orig_count_vals = inverse_signed_log_transform(all_count_vals)
        
        set_log_ticks(perf_axes[0], np.min(orig_stat_vals), np.max(orig_stat_vals), axis='y')
        set_log_ticks(count_axes[0], np.min(orig_count_vals), np.max(orig_count_vals), axis='y')

    # --- Legend ---
    if legend_loc == 'below':
            fig.legend(handles=legend_handles, loc='lower center', 
                    bbox_to_anchor=(0.5, -0.15), 
                    ncol=2 if len(unique_classes) == 4 else len(unique_classes), 
                    fontsize=14, frameon=False)
    else:
            perf_axes[0].legend(handles=legend_handles, loc=legend_loc)

    return splits_perf, None, fig


def sort_mutations_by_position(df, input_col, output_col='sorted_mutations'):
    """
    Sort mutations from low to high position within each mutation string.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    input_col : str
        Name of column containing mutation strings (e.g., 'D55N:M6N')
    output_col : str
        Name of output column for sorted mutations (default: 'sorted_mutations')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with new column containing sorted mutations
    
    Examples:
    ---------
    >>> df = pd.DataFrame({'mutations': ['D55N:M6N', 'A100G:C20T:B50D']})
    >>> df = sort_mutations_by_position(df, 'mutations')
    >>> df['sorted_mutations'].tolist()
    ['M6N:D55N', 'C20T:B50D:A100G']
    """
    def sort_single_mutation_string(mut_string):
        if pd.isna(mut_string) or mut_string == '':
            return mut_string
        
        # Split by colon to get individual mutations
        mutations = mut_string.split(':')
        
        # Extract position from each mutation using regex
        # Pattern: one or more letters, followed by digits, followed by one or more letters
        def get_position(mutation):
            match = re.search(r'[A-Za-z]+(\d+)[A-Za-z]+', mutation)
            if match:
                return int(match.group(1))
            else:
                # If pattern doesn't match, return a large number to sort it last
                return float('inf')
        
        # Sort mutations by their position
        sorted_mutations = sorted(mutations, key=get_position)
        
        # Join back with colons
        return ':'.join(sorted_mutations)
    
    # Apply the sorting function to the input column
    df = df.copy()
    df[output_col] = df[input_col].apply(sort_single_mutation_string)
    
    return df

def add_additive_predictions(df_preds, df_ref):
    """
    Calculates additive predictions for multimutants in df_ref by summing 
    single-mutant predictions from df_preds.

    Args:
        df_preds (pd.DataFrame): Dataframe containing single mutant predictions.
                                 Must have columns: ['code', 'mut_type', 'ddG_pred']
        df_ref (pd.DataFrame):   Reference dataframe containing multimutants 
                                 (colon-separated).
                                 Must have columns: ['code', 'mut_type']

    Returns:
        pd.DataFrame: A copy of df_ref with a new 'ddG_pred_additive' column.
    """
    # 1. Prepare the reference dataframe
    # We create a temporary ID to ensure we can group back to the exact original rows later
    df_out = df_ref.copy()
    df_out['_temp_id'] = df_out.index

    # 2. Explode the multimutants in the reference df
    # Select only necessary columns to keep the operation lightweight
    exploded = df_out[['_temp_id', 'code', 'mut_type']].copy()
    
    # Split "A1G:T2C" -> ["A1G", "T2C"]
    exploded['single_mut'] = exploded['mut_type'].str.split(':')
    
    # Explode into separate rows: one row per single mutation
    exploded = exploded.explode('single_mut')

    # 3. Merge with the predictions
    # We perform a left join to attach the 'ddG_pred' from df_preds to our exploded list
    # matching on 'code' (pdb_id) and the specific mutation
    merged = exploded.merge(
        df_preds[['code', 'mut_type', 'ddG_pred']],
        left_on=['code', 'single_mut'],
        right_on=['code', 'mut_type'],
        how='left'
    )

    # 4. Sum the predictions
    # We group by the temporary ID (original row) and sum the ddG values.
    # min_count=1 ensures that if a mutation is missing (NaN), the result is NaN 
    # (or partial sum) rather than blindly returning 0.
    additive_sums = merged.groupby('_temp_id')['ddG_pred'].sum(min_count=1)

    # 5. Assign result back to the reference dataframe
    df_out['ddG_pred_additive'] = additive_sums
    
    # Cleanup
    del df_out['_temp_id']
    
    return df_out


def calculate_epistasis(df, val_col=None, mut_col='mut_type'):
    """
    Calculates additive predictions and epistasis (_dddG) for multi-mutants.
    Can process a single column, a list of columns, or automatically detect 
    all valid numeric columns if val_col is None.
    
    Args:
        df (pd.DataFrame): Dataframe containing single and multi mutant predictions.
        val_col (str, list, or None): The column(s) to process. If None, targets all numeric cols.
        mut_col (str): The name of the column containing the mutation strings.
        
    Returns:
        pd.DataFrame: A copy of the original dataframe with new _additive and _dddG columns.
    """
    assert mut_col in df.columns, f"AssertionError: Mutation column '{mut_col}' not found in dataframe."
    assert 'code' in df.columns, "AssertionError: Column 'code' not found in dataframe."

    # 1. Determine which columns to process
    if val_col is None:
        val_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns 
            if col != 'code' and not col.endswith('_additive') and not col.endswith('_dddG')
        ]
        if not val_cols:
            print("Warning: val_col was None, but no valid numeric columns were found to process.")
            return df.copy()
    elif isinstance(val_col, str):
        val_cols = [val_col]
    elif isinstance(val_col, list):
        val_cols = val_col
    else:
        raise TypeError("AssertionError: val_col must be a string, a list of strings, or None.")

    for col in val_cols:
        assert col in df.columns, f"AssertionError: Value column '{col}' not found in dataframe."

    df_out = df.copy()

    # Initialize new columns with NaNs
    for col in val_cols:
        df_out[f"{col}_additive"] = np.nan
        df_out[f"{col}_dddG"] = np.nan

    # 2. Identify singles vs. multis based on the colon delimiter
    is_multi = df_out[mut_col].astype(str).str.contains(':', na=False)
    
    if is_multi.sum() == 0:
        print(f"Warning: No multi-mutants found (no ':' detected in '{mut_col}'). Returning dataframe with NaNs in epistasis columns.")
        return df_out

    # 3. Isolate single mutants to act as the reference lookup table
    cols_to_keep = ['code', mut_col] + val_cols
    df_singles = df_out[~is_multi][cols_to_keep].copy()

    # 4. Isolate and explode the multi-mutants
    df_multis = df_out[is_multi].copy()
    exploded = df_multis[['code', mut_col]].copy()
    exploded['_temp_id'] = exploded.index  # Track original row index
    
    exploded['single_mut'] = exploded[mut_col].str.split(':')
    exploded = exploded.explode('single_mut')

    # 5. Merge exploded multis with their constituent single mutant values
    merged = exploded.merge(
        df_singles,
        left_on=['code', 'single_mut'],
        right_on=['code', mut_col],
        how='left'
    )

    # 6. Vectorized calculation across all target columns simultaneously
    additive_sums = merged.groupby('_temp_id')[val_cols].sum(min_count=1)
    
    # Identify any multi-mutant groups that are missing at least one single mutant 
    # for each specific value column.
    groups_with_nans = merged[val_cols].isna().groupby(merged['_temp_id']).any()
    
    # Overwrite the invalid partial sums with NaN
    additive_sums = additive_sums.mask(groups_with_nans)

    # 7. Map sums back to the original rows and calculate epistasis
    for col in val_cols:
        additive_col = f"{col}_additive"
        dddG_col = f"{col}_dddG"
        
        df_out.loc[additive_sums.index, additive_col] = additive_sums[col]
        df_out[dddG_col] = df_out[col] - df_out[additive_col]

    return df_out

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*(int(round(c*255)) for c in rgb))

def _adjust_lightness(hex_color, delta=0.0):
    r,g,b = _hex_to_rgb(hex_color)
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    l = min(max(l + delta, 0.0), 1.0)
    r2,g2,b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r2,g2,b2))

def _base_name(col):
    """Remove repeat suffix like _1, _2, etc. (only at the end)."""
    return col.rsplit('_', 1)[0] if re.search(r'_(\d+)$', col) else col

def _strip_variant_suffix(name):
    """
    Handle -D, _additive suffix, and SPURS logic for determining filled vs empty markers.
    """
    if 'SPURS-multi' in name:
        stripped = name.replace('SPURS-multi', 'SPURS')
        return stripped, 'full'
    elif 'SPURS' in name:
        return name, 'additive'

    # Check for -D suffix. The regex (?:\s*\(|$) ensures we only strip -D if it's 
    # at the absolute end of the string OR immediately followed by a parameter in parentheses.
    # E.g., 'ESM-MSR-D (sigma=1)' -> 'ESM-MSR (sigma=1)', 'full'
    if re.search(r'-D(?:\s*\(|$)', name):
        stripped = re.sub(r'-D(?=\s*\(|$)', '', name)
        return stripped, 'full'
    
    # Check for _additive anywhere in name
    if '_additive' in name:
        stripped = name.replace('_additive', '')
        return stripped, 'additive'
    
    return name, 'unknown'

def _root_name(name):
    """
    Return the name without splitting off parentheses. 
    By keeping '(sigma=1.25)' attached, the pipeline inherently treats it as a 
    completely different model with its own canonical grouping and colors.
    """
    return name.strip()

def _canonical_name(base):
    """
    Get the canonical name for grouping: strip -D suffix, then get root.
    This determines which items share the same x-offset and marker family.
    """
    stripped, _ = _strip_variant_suffix(base)
    return _root_name(stripped)

def visualize_model_performance(
    data, figsize=(12, 6), title="Model Performance Across Datasets", 
    ylabel="Performance Score", 
    colors=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#d35400', '#34495e'],
    highlighted_model=None, ylim=None, debug=False, 
    additive_offset=0.05, group_spread=0.1, legend_loc='lower left',
    yscale='linear', symlog_thresh=1e-3
):
    """
    Visualize model performance across datasets with automatic grouping.
    Replaces vertical lines with alternating grey background panels.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Map columns -> base (repeat-aggregated) names
    col_to_base = {col: _base_name(col) for col in data.columns}
    bases_in_order = []
    seen_bases = set()
    for col in data.columns:
        b = col_to_base[col]
        if b not in seen_bases:
            bases_in_order.append(b)
            seen_bases.add(b)

    # Base info gathering
    base_info = {}
    for base in bases_in_order:
        stripped, variant_type = _strip_variant_suffix(base)
        canonical = _canonical_name(base)
        base_info[base] = {
            'stripped': stripped,
            'variant_type': variant_type,
            'canonical': canonical
        }
    
    # Determine 'is_additive' logic
    stripped_with_D = {base_info[b]['stripped'] for b in bases_in_order if base_info[b]['variant_type'] == 'full'}
    stripped_with_additive = {base_info[b]['stripped'] for b in bases_in_order if base_info[b]['variant_type'] == 'additive'}
    
    for base in bases_in_order:
        info = base_info[base]
        if info['variant_type'] == 'full':
            info['is_additive'] = False
        elif info['variant_type'] == 'additive':
            info['is_additive'] = True
        else:
            if info['stripped'] in stripped_with_D:
                info['is_additive'] = True
            elif info['stripped'] in stripped_with_additive:
                info['is_additive'] = False
            else:
                info['is_additive'] = False

    # Aggregation
    aggregated_data = {}
    for base in bases_in_order:
        model_cols = [c for c in data.columns if col_to_base[c] == base]
        model_df = data[model_cols]
        aggregated_data[base] = {
            'mean': model_df.mean(axis=1),
            'std': model_df.std(axis=1) if len(model_cols) > 1 else np.zeros(len(model_df)),
            'has_repeats': len(model_cols) > 1
        }
    
    if debug:
        print(f"Bases in order: {bases_in_order}")

    # Layout logic: Unique canonicals determine broad groupings
    canonicals_in_order = []
    seen_canonicals = set()
    for base in bases_in_order:
        c = base_info[base]['canonical']
        if c not in seen_canonicals:
            canonicals_in_order.append(c)
            seen_canonicals.add(c)

    # Color & Marker Assignments
    canonical_base_color = {c: colors[i % len(colors)] for i, c in enumerate(canonicals_in_order)}
    
    canonical_to_bases = {c: {'normal': [], 'additive': []} for c in canonicals_in_order}
    for base in bases_in_order:
        info = base_info[base]
        category = 'additive' if info['is_additive'] else 'normal'
        canonical_to_bases[info['canonical']][category].append(base)

    base_colors = {}
    for canonical in canonicals_in_order:
        base_hex = canonical_base_color[canonical]
        normal_family = canonical_to_bases[canonical]['normal']
        additive_family = canonical_to_bases[canonical]['additive']
        
        n = len(normal_family)
        normal_deltas = list(np.linspace(-0.12, 0.12, n)) if n > 1 else [0.0] if n == 1 else []
        
        for b, d in zip(normal_family, normal_deltas):
            base_colors[b] = _adjust_lightness(base_hex, d)
        
        for additive_base in additive_family:
            stripped = base_info[additive_base]['stripped']
            matching_normal = next((nb for nb in normal_family if base_info[nb]['stripped'] == stripped), None)
            
            if matching_normal:
                base_colors[additive_base] = _adjust_lightness(base_colors[matching_normal], 0.25)
            else:
                base_colors[additive_base] = _adjust_lightness(base_hex, 0.3)

    markers = ['s', 'D', 'p', 'h', '^', 'v', '<', '>', 'H', 'o', '*', '+', 'x']
    stripped_in_order = []
    seen_stripped = set()
    for base in bases_in_order:
        s = base_info[base]['stripped']
        if s not in seen_stripped:
            stripped_in_order.append(s)
            seen_stripped.add(s)
    
    stripped_markers = {s: markers[i % len(markers)] for i, s in enumerate(stripped_in_order)}

    # Plot Setup
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, len(data.index) - 0.5)
    
    # Set Y-axis scale
    if yscale == 'symlog':
        ax.set_yscale('symlog', linthresh=symlog_thresh)
    else:
        ax.set_yscale(yscale)

    def y_formatter(x, pos):
        if abs(x) < 10:
            return f"{x:g}"
        return ticker.LogFormatterSciNotation()(x, pos)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

    # Background Panels (Grey panels with white spaces)
    # We color every second dataset region with a light grey span
    for i in range(len(data.index)):
        if i % 2 == 1: # Color odd indices to create alternating effect
            ax.axvspan(i - 0.5, i + 0.5, color='#f0f0f0', alpha=1.0, zorder=0, linewidth=0)
    
    # Ensure remaining grid lines (horizontal) remain grey
    ax.grid(axis='y', color='#e0e0e0', linestyle='--', alpha=0.7, zorder=1)

    # Offsets calculation
    num_canonicals = len(canonicals_in_order)
    if num_canonicals == 1:
        offsets = {canonicals_in_order[0]: 0.0}
    else:
        center = (num_canonicals - 1) * group_spread / 2.0
        offsets = {canonical: (i * group_spread) - center for i, canonical in enumerate(canonicals_in_order)}

    stripped_has_both = set()
    if additive_offset != 0:
        s_add = {base_info[b]['stripped'] for b in bases_in_order if base_info[b]['is_additive']}
        s_full = {base_info[b]['stripped'] for b in bases_in_order if not base_info[b]['is_additive']}
        stripped_has_both = s_add & s_full

    # Plotting loop
    plot_handles = {}
    for base in bases_in_order:
        info = base_info[base]
        x_pos = np.arange(len(data.index)) + offsets[info['canonical']]
        
        if additive_offset != 0 and info['stripped'] in stripped_has_both:
            shift = additive_offset / 2.0
            x_pos = x_pos + shift if info['is_additive'] else x_pos - shift

        means, stds = aggregated_data[base]['mean'], aggregated_data[base]['std']
        color = base_colors[base]
        marker = stripped_markers[info['stripped']]
        is_highlighted = (highlighted_model == base)
        
        plot_color = 'black' if is_highlighted else color
        edge_color = 'yellow' if is_highlighted else color

        if aggregated_data[base]['has_repeats']:
            handle = ax.errorbar(
                x_pos, means, yerr=stds, fmt=marker, markersize=10,
                markerfacecolor='white' if info['is_additive'] else plot_color,
                markeredgecolor=color if info['is_additive'] else edge_color,
                markeredgewidth=1, ecolor=plot_color, capsize=8, 
                capthick=2 if not info['is_additive'] else 1,
                elinewidth=2 if not info['is_additive'] else 1,
                label=base, alpha=1, zorder=4 if is_highlighted else 3
            )
        else:
            handle, = ax.plot(
                x_pos, means, linestyle='', marker=marker, markersize=10,
                markerfacecolor='white' if info['is_additive'] else plot_color,
                markeredgecolor=color if info['is_additive'] else edge_color,
                markeredgewidth=1, label=base, alpha=1, zorder=4 if is_highlighted else 3
            )
        plot_handles[base] = {'handle': handle, 'label': base, 
                              'stripped': info['stripped'], 'is_additive': info['is_additive']}

    # Legend Construction
    ordered_handles, ordered_labels = [], []
    for stripped in stripped_in_order:
        for b in bases_in_order:
            if plot_handles[b]['stripped'] == stripped and not plot_handles[b]['is_additive']:
                ordered_handles.append(plot_handles[b]['handle'])
                ordered_labels.append(plot_handles[b]['label'])
        for b in bases_in_order:
            if plot_handles[b]['stripped'] == stripped and plot_handles[b]['is_additive']:
                ordered_handles.append(plot_handles[b]['handle'])
                ordered_labels.append(plot_handles[b]['label'])

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, rotation=30, ha='right', fontsize=12)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(ordered_handles, ordered_labels, loc=legend_loc, ncol=2, 
              columnspacing=1.0, handletextpad=0.5, frameon=True)
    
    plt.tight_layout()
    return fig, ax

