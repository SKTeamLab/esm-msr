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

from scipy.stats import mode as scipy_mode
from scipy.stats import gaussian_kde, pearsonr


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

def assess_grouped_spearman(df_preds, df_scores, name, dataset, pred_col='esm_msr', label='ddG_ML', quiet=True):

    if not quiet:
        print(f'Ungrouped rho for {dataset} splits test data')
        print(df_preds[[label, pred_col]].corr('spearman').iloc[0, 1])
    df_scores.loc[(f'{dataset}', 'ungrouped'), name] = df_preds[[label, pred_col]].corr('spearman').iloc[0, 1]
    df_scores.loc[(f'{dataset}', 'ungrouped'), name+'_n'] = len(df_preds[[label, pred_col]].dropna())

    corrs = df_preds.groupby('code')[[label, pred_col]].corr('spearman').reset_index()
    corrs = corrs.loc[corrs['level_1']==label].set_index('code').drop(['level_1'], axis=1)
    if not quiet:
        print(f'Grouped rho for {dataset} splits test data')
        print(corrs.mean()[pred_col])
    df_scores.loc[(f'{dataset}', 'grouped'), name] = corrs.mean()[pred_col]
    df_scores.loc[(f'{dataset}', 'grouped'), name+'_n'] = len(df_preds[[label, pred_col]].dropna())
    if not quiet:
        print(name)
    return df_scores

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
    import pandas as pd
    import numpy as np
    
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
    series2=None,
    series_names=('Series 1', 'Series 2')
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
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Helper: Robust Data Collection ---
    def _collect(src_series, base_name):
        """Collects mean, std, and values for stats."""
        if src_series is None: return None
        
        # 1. Identify Replicates
        pattern = re.compile(rf"^{re.escape(base_name)}[_]?(\d+)$")
        reps = [k for k in src_series.index if pattern.match(k) and not k.endswith('_n')]
        
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

        # 3. Extract Underlying N (from _n columns)
        n_vals_found = []
        if f"{base_name}_n" in src_series.index:
            try: n_vals_found.append(int(src_series[f"{base_name}_n"]))
            except: pass
            
        for k in keys_used:
            if f"{k}_n" in src_series.index:
                try: n_vals_found.append(int(src_series[f"{k}_n"]))
                except: pass

        n_underlying = None
        if n_vals_found:
            n_underlying = max(set(n_vals_found), key=n_vals_found.count)
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0.0,
            'values': values,
            'has_replicates': len(values) > 1,
            'n_underlying': n_underlying 
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
                    for r in add_reps:
                        if f"{r}_n" in series.index:
                             try: n_vals_found.append(int(series[f"{r}_n"]))
                             except: pass
                    n_und = max(set(n_vals_found), key=n_vals_found.count) if n_vals_found else None
                    data_right = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)) if len(vals)>1 else 0.0,
                        'values': vals, 'has_replicates': len(vals)>1, 'n_underlying': n_und
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

    # --- Step 2: Consensus N Calculation ---
    def get_consensus(data_list, key):
        ns = [p[key]['n_underlying'] for p in data_list if p[key] and p[key]['n_underlying'] is not None]
        if not ns: return None
        return max(set(ns), key=ns.count)

    consensus_left = get_consensus(pairs_data, 'left')
    consensus_right = get_consensus(pairs_data, 'right')

    xtick_labels = []
    for p in pairs_data:
        label = p['name']
        flag = False
        if p['left']:
            n_l = p['left']['n_underlying']
            if n_l is not None and consensus_left is not None and n_l != consensus_left: 
                print('left', n_l)
                flag = True
        if p['right']:
            n_r = p['right']['n_underlying']
            if n_r is not None and consensus_right is not None and n_r != consensus_right: 
                print('right', n_r)
                flag = True
        if flag: label += "*"
        xtick_labels.append(label)

    # --- Step 3: Setup Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    base_colors = ['#34495e', '#3498db', '#2ecc71', "#bbbbbb", '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#d35400']
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
            # Highlight: Left is Black (Darkest) with Light Grey Hatches
            c_left = 'black'
            # Setting edgecolor changes hatch color (and border)
            edge_c_left = '#bbbbbb' 
            
            # Highlight: Right is Light Grey (Lighter)
            c_right = '#444444'
            edge_c_right = 'black'
        else:
            # Normal: Calculate Pastel vs Saturated
            base_c = color_map[model]
            rgb = mcolors.to_rgb(base_c)
            c_pastel = tuple(0.5 + 0.5 * c for c in rgb)
            c_saturated = base_c
            
            # Left gets the Darker (Saturated)
            c_left = c_saturated
            edge_c_left = 'black'
            
            # Right gets the Lighter (Pastel)
            c_right = c_pastel
            edge_c_right = 'black'

        # LEFT BAR LOGIC
        # Always Hatched, Always Darker
        pos_left = i - half/2.0
        hatch_left = '///'

        ax.bar(pos_left, left['mean'], width=half, color=c_left, hatch=hatch_left,
               edgecolor=edge_c_left, 
               linewidth=2 if p['is_highlight'] else 1)
        
        # If highlighted, add the yellow border loop
        # This overlays a yellow border, which is helpful because the base bar 
        # might have a light grey border due to the hatch color setting.
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
        # Always Solid, Always Lighter
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
    
    pad_top = 0.05 * span
    pad_bot = 0.05 * span
    if y_min_data >= 0:
        ax.set_ylim(0, final_y_max + pad_top)
    else:
        ax.set_ylim(y_min_data - pad_bot, final_y_max + pad_top)
    
    l1_text = "Epistatic" if mode == 'epistatic' else series_names[0]
    l2_text = "Additive" if mode == 'epistatic' else series_names[1]
    
    if consensus_left is not None: l1_text += f" (n={consensus_left})"
    if consensus_right is not None: l2_text += f" (n={consensus_right})"
    
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
    bar_width=0.8,                # overall width; pos/neg each take half
    shadow_alpha=0.15,            # transparency for overall "shadow"
    hatch_pos="///",
    hatch_neg=None,
    edgecolor="black",
    fontsize_multiplier=1.3,
    capsize=4,
    output_file=None,
    base_colors=None,             # optional list of colors; defaults to Matplotlib cycle
    highlight_gray='#e0e0e0',     # face color for highlighted model
    enforce_global_n=True,
    epistasis=True,
    # --- New: label overlap control ---
    label_min_sep=0.05,            # if labels for a single model are within this y-diff, push the higher one up
    legend_loc="upper right"
):
    """
    For each model in `model_groups`, draw three aligned bars at one x-position:
      - Overall bar (key: '{model}') as a light 'shadow' (full width, alpha)
      - Pos bar (key: '{model}_pos') as left half-width, hatched
      - Neg bar (key: '{model}_neg') as right half-width, solid

    Adds significance brackets and p-values comparing the OVERALL (shadow) values of
    each model *against the highlighted model*, using:
      - two-sample Welch t-test when both have replicates
      - one-sample t-test when only one side has replicates
      - NA when neither side has replicates

    Also nudges overlapping value labels within a model if any two are within `label_min_sep` on the y-axis.
    """

    # --- helpers ---
    def _collect(series, base, suffix=None):
        """Return dict: mean, std, n, has_replicates, keys_used. Prefers replicates."""
        if suffix is None:
            direct = base
            reps = [k for k in series.index if re.fullmatch(rf"{re.escape(base)}_?\d+", k)]
        else:
            direct = f"{base}_{suffix}"
            pat1 = rf"{re.escape(base)}_{re.escape(suffix)}_?\d+"   # base_suffix_N
            pat2 = rf"{re.escape(base)}_?\d+_{re.escape(suffix)}"   # base_N_suffix
            reps = [k for k in series.index if re.fullmatch(pat1, k) or re.fullmatch(pat2, k)]

        keys_used = []
        values = None
        if reps:
            keys_used = reps
            values = series[reps].astype(float).values
        elif direct in series.index:
            keys_used = [direct]
            values = np.array([float(series[direct])])

        if values is None or len(values) == 0 or not np.all(np.isfinite(values)):
            return None  # missing

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
            "has_replicates": len(values) > 1,
            "keys": keys_used,
            "values": values,
        }

    def _get_n_from_series(series, base, subset=None, fallback=None):
        """Priority for n columns:
           pos: base_n_pos > base_pos_n
           neg: base_n_neg > base_neg_n
           all: base_n
           If none found, uses `fallback` if provided, else returns None.
        """
        if subset == 'pos':
            keys = [f"{base}_n_pos", f"{base}_pos_n"]
        elif subset == 'neg':
            keys = [f"{base}_n_neg", f"{base}_neg_n"]
        else:
            keys = [f"{base}_n"]
        for k in keys:
            if k in series.index:
                try:
                    return int(series[k])
                except Exception:
                    pass
        return None if fallback is None else int(fallback)

    def _determine_consensus_n(series, model_groups, enforce=True):
        """Determine consensus n with the following rules:
        - Prefer POS counts across models; if none, use NEG; if none, use OVERALL.
        - If POS, NEG, and OVERALL are all present, assert POS + NEG == OVERALL.
        Returns (legend_n, pos_consensus, neg_consensus, overall_consensus).
        """
        def _consensus(vals, tag):
            if not vals:
                return None
            uniq = sorted(set(int(v) for v in vals))
            if enforce and len(uniq) > 1:
                raise ValueError(f"Expected a single consensus {tag} n across models, found: {uniq}")
            return uniq[0]

        pos_vals, neg_vals, all_vals = [], [], []
        for m in model_groups:
            p = _get_n_from_series(series, m, 'pos')
            if p is not None:
                pos_vals.append(int(p))
            n = _get_n_from_series(series, m, 'neg')
            if n is not None:
                neg_vals.append(int(n))
            a = _get_n_from_series(series, m, None)
            if a is not None:
                all_vals.append(int(a))

        pos_c = _consensus(pos_vals, 'pos')
        neg_c = _consensus(neg_vals, 'neg')
        all_c = _consensus(all_vals, 'overall')
        print(pos_c, neg_c, all_c)

        legend_n = pos_c if pos_c is not None else (neg_c if neg_c is not None else all_c)
        if legend_n is None:
            raise ValueError("No n could be determined from *_n columns; please supply n in legend explicitly or add *_n columns.")

        if enforce and pos_c is not None and neg_c is not None and all_c is not None:
            if pos_c + neg_c != all_c:
                raise ValueError(f"Expected pos_n + neg_n == overall_n, got {pos_c} + {neg_c} != {all_c}")

        return legend_n, pos_c, neg_c, all_c

    # compute consensus n across all models/subsets
    consensus_n, _pos_consensus, _neg_consensus, _overall_consensus = _determine_consensus_n(series, model_groups, enforce=enforce_global_n)

    # --- Exact coloring system from epistatic chart ---
    base_color_list = base_colors or ['#34495e', '#3498db', '#2ecc71', '#bbbbbb', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#d35400']
    # Map each model to a base color (skip the highlighted: it gets grey/white treatment later)
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

    # Track extremes for y-limits (include error bars when present)
    label_entries = []  # (xpos, mean, err)
    per_model_label_entries = {m: [] for m in model_groups}  # track labels per model for overlap fixing

    y_min, y_max = 0.0, 0.0
    drawn_any = False

    # We'll store OVERALL info for significance testing
    overall_data = {}  # model -> dict from _collect (with values)

    for i, model in enumerate(model_groups):
        base_col = model_color_map.get(model)  # may be None for highlighted

        is_highlight = (highlighted_model is not None and model == highlighted_model)
        lw = 2.5 if is_highlight else 1.2

        # Gather data
        overall = _collect(series, model, suffix=None)
        pos     = _collect(series, model, suffix="pos")
        neg     = _collect(series, model, suffix="neg")

        # Save overall for testing later
        if overall is not None:
            overall_data[model] = {**overall, 'xpos': x[i]}

        # Overall (shadow)
        if overall is not None:
            if is_highlight:
                overall_color = highlight_gray
            else:
                overall_color = base_col if base_col is not None else highlight_gray
            ax.bar(
                x[i], overall["mean"], width=bar_width,
                color=overall_color, alpha=shadow_alpha,
                edgecolor=edgecolor, linewidth=lw, zorder=1,
            )
            hi = overall["mean"] + (overall["std"] if overall["has_replicates"] else 0)
            lo = overall["mean"] - (overall["std"] if overall["has_replicates"] else 0)
            y_max = max(y_max, hi, 0)
            y_min = min(y_min, lo, 0)
            label_entries.append((x[i], overall["mean"], (overall["std"] if overall["has_replicates"] else 0.0)))
            per_model_label_entries[model].append((x[i], overall["mean"], (overall["std"] if overall["has_replicates"] else 0.0), "overall", model))

        # POS (left)
        if pos is not None:
            if is_highlight:
                # Use white edge to make hatches white against black face
                current_pos_color = 'black'
                current_edgecolor = '#bbbbbb'
                current_hatch = '///' 
            else:
                current_pos_color = highlight_gray if is_highlight else base_col
                current_edgecolor = edgecolor
                current_hatch = hatch_pos

            ax.bar(
                x[i] - half/2.0, pos["mean"], width=half,
                color=current_pos_color, hatch=current_hatch,
                edgecolor=current_edgecolor, linewidth=lw, zorder=2,
            )

            if pos["has_replicates"] and pos["std"] > 0:
                ax.errorbar(
                    x[i] - half/2.0, pos["mean"], yerr=pos["std"],
                    # Use 'black' for error bars specifically if the bar is black
                    fmt="none", ecolor='red', 
                    elinewidth=lw, capsize=capsize, zorder=3,
                )

            # If highlighted, overlay a hollow bar to restore the black border
            if is_highlight:
                ax.bar(
                    x[i] - half/2.0, pos["mean"], width=half,
                    color='none', edgecolor='yellow', linewidth=lw, zorder=2.5
                )
            
            hi = pos["mean"] + (pos["std"] if pos["has_replicates"] else 0)
            lo = pos["mean"] - (pos["std"] if pos["has_replicates"] else 0)
            y_max = max(y_max, hi, 0)
            y_min = min(y_min, lo, 0)
            label_entries.append((x[i] - half/2.0, pos["mean"], (pos["std"] if pos["has_replicates"] else 0.0)))
            per_model_label_entries[model].append((x[i] - half/2.0, pos["mean"], (pos["std"] if pos["has_replicates"] else 0.0), "pos", model))

        # NEG (right)
        if neg is not None:
            if is_highlight:
                neg_color = '#444444'  # Solid dark grey
            else:
                # lighter shade of the base color (blend with white)
                rgb = mcolors.to_rgb(base_col)
                lighter_rgb = tuple(0.5 + 0.5 * c for c in rgb)
                neg_color = lighter_rgb

            ax.bar(
                x[i] + half/2.0, neg["mean"], width=half,
                color=neg_color, hatch=None,
                edgecolor=edgecolor if not is_highlight else 'yellow', linewidth=lw, zorder=3,
            )
            if neg["has_replicates"] and neg["std"] > 0:
                ax.errorbar(
                    x[i] + half/2.0, neg["mean"], yerr=neg["std"],
                    fmt="none", ecolor='red', elinewidth=lw, capsize=capsize, zorder=3,
                )

            hi = neg["mean"] + (neg["std"] if neg["has_replicates"] else 0)
            lo = neg["mean"] - (neg["std"] if neg["has_replicates"] else 0)
            y_max = max(y_max, hi, 0)
            y_min = min(y_min, lo, 0)
            label_entries.append((x[i] + half/2.0, neg["mean"], (neg["std"] if neg["has_replicates"] else 0.0)))
            per_model_label_entries[model].append((x[i] + half/2.0, neg["mean"], (neg["std"] if neg["has_replicates"] else 0.0), "neg", model))

        drawn_any = drawn_any or (overall is not None or pos is not None or neg is not None)

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_groups, rotation=45, ha="right", fontsize=10 * fontsize_multiplier)

    # Labels/title
    if y_label:
        ax.set_ylabel(y_label, fontsize=16 * fontsize_multiplier, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16 * fontsize_multiplier, fontweight="bold")

    # Y limits with symmetric-ish padding around extremes (handles negatives)
    if drawn_any:
        span = y_max - y_min
        pad = 0.08 * span if span > 0 else 1.0
        ymin_init, ymax_init = (y_min - pad, y_max + pad)
        ax.set_ylim(ymin_init, ymax_init)
    else:
        ymin_init, ymax_init = (0.0, 1.0)

    # --- Compute p-values for OVERALL vs highlighted; p=1 for reference ---
    pvals_overall = {m: np.nan for m in model_groups}
    if highlighted_model is not None and highlighted_model in overall_data:
        hl = overall_data[highlighted_model]
        pvals_overall[highlighted_model] = 1.0
        for m in model_groups:
            if m not in overall_data or m == highlighted_model:
                continue
            cur = overall_data[m]
            if hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_ind(cur['values'], hl['values'], equal_var=False)
            elif hl['has_replicates'] and not cur['has_replicates']:
                _, p = stats.ttest_1samp(hl['values'], cur['mean'])
            elif not hl['has_replicates'] and cur['has_replicates']:
                _, p = stats.ttest_1samp(cur['values'], hl['mean'])
            else:
                p = np.nan
            pvals_overall[m] = p

    # --- Draw numeric value labels (now with p-values for OVERALL) ---
    def _compute_label_y(mean, err, span):
        offset = 0.02 * span
        if mean >= 0:
            return mean + err + offset, 'bottom'
        else:
            return mean - err - offset, 'top'

    span_now = ax.get_ylim()[1] - ax.get_ylim()[0]
    per_model_ytexts = {m: [] for m in model_groups}  # store (x,y,text,va)

    for m in model_groups:
        for (xpos, mean, err, kind, model_name) in per_model_label_entries[m]:
            ytext, va = _compute_label_y(mean, err, span_now)
            if kind == 'overall':
                p = pvals_overall.get(model_name, np.nan)
                p_text = "NA" if np.isnan(p) else f"{p:.2e}"
                text = f"{mean:.3f}\np={p_text}"
            else:
                text = f"{mean:.3f}"
            per_model_ytexts[m].append([xpos, ytext, text, va])

    # Overlap fix: within each model, if any two labels are within `label_min_sep` vertically, bump the higher one
    extra_offset = 0.025 * span_now
    for m, entries in per_model_ytexts.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                yi = entries[i][1]
                yj = entries[j][1]
                if abs(yi - yj) < label_min_sep:
                    if yi >= yj:
                        entries[i][1] = yi + extra_offset
                    else:
                        entries[j][1] = yj + extra_offset

    for m, entries in per_model_ytexts.items():
        for xpos, ytext, text, va in entries:
            ax.text(
                xpos, ytext, text,
                ha='center', va=va,
                fontsize=10 * fontsize_multiplier,
                fontweight='bold'
            )

    # Grid

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle="", alpha=0.0)

    # Legend
    if epistasis:
        pos_label = f"Positive epistasis; δΔΔG > 0 (n={_pos_consensus})" if consensus_n is not None else "Positive; δΔΔG > 0 epistasis"
        neg_label = f"Negative epistasis; δΔΔG <= 0 (n={_neg_consensus})" if consensus_n is not None else "Negative; δΔΔG <= 0 epistasis"
    else:
        pos_label = f"Stabilizing mutations; ΔΔG > 0 (n={_pos_consensus})" if consensus_n is not None else "Stabilizing mutations; ΔΔG > 0"
        neg_label = f"Destabilizing mutations; ΔΔG <= 0 (n={_neg_consensus})" if consensus_n is not None else "Destabilizing mutations; ΔΔG <= 0"

    legend_elements = [
        Patch(facecolor="grey", edgecolor=edgecolor, hatch=hatch_pos, label=pos_label),
        Patch(facecolor="#ffffff", edgecolor=edgecolor, hatch=hatch_neg, label=neg_label),
        Patch(facecolor="#ffffff", edgecolor=edgecolor, alpha=shadow_alpha, label=f"Overall (shadow) (n={_overall_consensus})"),
    ]

    ax.set_ylim(ylim)
    ax.legend(handles=legend_elements, loc=legend_loc, framealpha=0.9, fontsize=10 * fontsize_multiplier)

    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=10 * fontsize_multiplier)

    plt.tight_layout()
    #if output_file:
    #    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig, ax


def density_scatter(
    x, y, ax=None, sort=True, s=5, point_alpha=0.5,
    cmap="viridis", log=False, vmin=None, vmax=None, bw_method="scott", 
    hide_marginal_legend=False, stats=False, stats_color='red', include_p_value=False
):
    """
    Scatter colored by KDE density, with semi-transparent points,
    marginal distributions, and ground truth overlay on marginals.
    Optionally adds line of best fit and Pearson correlation.

    Parameters
    ----------
    stats : bool, default=False
        If True, adds a line of best fit and a text box with Pearson r and p-value.
    stats_color : str, default='red'
        Color for the best fit line.
    """
    # Create figure with marginal axes - swapped left marginal and colorbar positions
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(3, 3, hspace=0.15, wspace=0.25,
                             width_ratios=[0.2, 4, 1], height_ratios=[1, 4, 0.2])
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

    # KDE density estimate for scatter
    # Note: This will fail if x or y contain NaNs
    xy = np.vstack([np.asarray(x), np.asarray(y)])
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
        x_, y_, colors = np.asarray(x)[idx], np.asarray(y)[idx], colors[idx]
    else:
        x_, y_ = np.asarray(x), np.asarray(y)

    # Draw main scatter
    ax_main.scatter(x_, y_, c=colors, s=s, edgecolors="none", rasterized=False)
    
    # --- NEW: Add Line of Best Fit and Stats ---
    if stats:
        # Ensure we compute stats on finite data only
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean, y_clean = x_arr[mask], y_arr[mask]

        if len(x_clean) > 1:
            # Linear Regression (Polyfit degree 1)
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            
            # Plot line across the current data range
            line_x = np.array([np.min(x_clean), np.max(x_clean)])
            line_y = slope * line_x + intercept
            ax_main.plot(line_x, line_y, color=stats_color, linestyle='--', linewidth=2, label='Best Fit')

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
    # -------------------------------------------

    # Move y-axis label to right side
    ax_main.yaxis.set_label_position('right')
    ax_main.set_ylabel(y.name if hasattr(y, 'name') else 'Ground Truth', rotation=270, labelpad=10)
    
    # Move x-axis label to top (between marginal and main plot)
    ax_main.set_xlabel('')  # Remove default x-label
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel(x.name if hasattr(x, 'name') else 'Predictions')

    # Add marginal distributions if axes were created
    if ax_top is not None and ax_right is not None:
        # Set the marginal axis limits to match the main plot
        ax_top.set_xlim(ax_main.get_xlim())
        ax_right.set_ylim(ax_main.get_ylim())
        
        # Top marginal: predictions (x-axis)
        ax_top.hist(x, bins=50, alpha=0.7, color='steelblue', density=True, label='Predictions')
        ax_top.hist(y, bins=50, alpha=0.3, color='coral', density=True, label='Ground Truth')
        ax_top.set_ylabel('Density')
        ax_top.tick_params(labelbottom=False)
        if not hide_marginal_legend:
            ax_top.legend(loc='upper right', fontsize=9)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

        # Right marginal: ground truth (y-axis)
        ax_right.hist(y, bins=50, alpha=0.7, color='coral', density=True, 
                      orientation='horizontal', label='Ground Truth')
        ax_right.hist(x, bins=50, alpha=0.3, color='steelblue', density=True,
                      orientation='horizontal', label='Predictions')
        ax_right.set_xlabel('Density')
        ax_right.tick_params(labelleft=False)
        if not hide_marginal_legend:
            ax_right.legend(loc='upper right', fontsize=9)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)

    # Create colorbar on the left side
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    if ax_cbar is not None:
        cbar = plt.colorbar(sm, cax=ax_cbar)
        # Move colorbar ticks and label to the left side
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    else:
        cbar = plt.colorbar(sm, ax=ax_main)
    cbar.set_label("Density")

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
                          ignore_ties=True):
    """
    Compute a variant of NDCG with one of:
      - top_n (int): NDCG@N
      - percentile (float in (0,1]): NDCG@ceil(percentile * n)
      - threshold (float): drop rows with true_col < threshold, then full-list NDCG

    Exactly one of {top_n, percentile, threshold} must be specified.

    Returns:
        float (NDCG) or np.nan on degenerate cases.
    """
    # --- arg validation ---
    flags = [top_n is not None, percentile is not None, threshold is not None]
    if sum(flags) != 1:
        raise ValueError("Specify exactly one of top_n, percentile, or threshold.")

    df = df[[pred_col, true_col]].copy()
    df = df.dropna()
    if df.empty:
        return np.nan

    # --- threshold mode: filter first, then full-list NDCG ---
    if threshold is not None:
        df = df[df[true_col] >= threshold]
        if len(df) < 2:
            return np.nan
        y_score = df[pred_col].to_numpy().reshape(1, -1)
        y_true = df[true_col].to_numpy()

        y_true_processed = np.where(y_true < threshold, 0, y_true)

        # if all-zero relevance, ndcg undefined
        if np.all(y_true_processed == 0):
            return np.nan

        y_true_processed = y_true_processed.reshape(1, -1)
        return metrics.ndcg_score(y_true_processed, y_score, k=None, ignore_ties=ignore_ties)

    # --- top_n / percentile: choose k on the unfiltered list ---
    y_score = df[pred_col].to_numpy().reshape(1, -1)
    y_true = df[true_col].to_numpy()

    y_true_processed = np.where(y_true < 0, 0, y_true)

    if np.all(y_true_processed == 0) or y_true_processed.size < 2:
        return np.nan

    n = y_true_processed.size
    if top_n is not None:
        if top_n <= 0:
            return np.nan
        k = min(int(top_n), n)
    else:
        # percentile mode
        if not (0 < percentile <= 1):
            raise ValueError("percentile must be in (0, 1].")
        k = max(1, int(np.ceil(percentile * n)))
        k = min(k, n)

    y_true_processed = y_true_processed.reshape(1, -1)
    return metrics.ndcg_score(y_true_processed, y_score, k=k, ignore_ties=ignore_ties)


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

    df_preds = df_preds.dropna(subset=label)

    if grouped:
        ndcgs = []
        ns = []
        for code, group in df_preds.groupby('code'):
            ndcg = compute_ndcg_flexible(group, pred_col, label, percentile=percentile, top_n=top_n, threshold=threshold)
            ndcgs.append(ndcg)
            ns.append(len(group))
            
        ndcgs = pd.Series(ndcgs)
        ns = pd.Series(ns)

        ndcg = ndcgs.mean()
        n = ns.sum()
        return ndcg, n

    else:
        df_preds = df_preds[['code', label, pred_col]].dropna()
        ndcg = compute_ndcg_flexible(df_preds, pred_col, label, percentile=percentile, top_n=top_n, threshold=threshold)
        ndf = df_preds[[label, pred_col]].dropna()
        n = len(ndf)
        return ndcg, n


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

def custom_barplot(data, x, y, hue, order, ax, 
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
                       medianprops={'color': color, 'alpha': 1, 'linewidth': 2},
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
            
            ax.scatter(center_pos + plot_jitter, plot_vals, s=2, color=color, alpha=0.4, 
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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

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
        l_h, medians = custom_barplot(
            data=splits_perf, x='model', y=stat_col, hue='class',
            order=models_subset, ax=ax_p, legend_labels=unique_classes,
            legend_colors=palette_to_use, orientation=orientation,
            group_plot_width=0.8
        )
        if l_h: legend_handles = l_h
        perf_medians_all.update(medians)
        
        # Plot Counts
        _, c_medians = custom_barplot(
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
        # Default to median (values returned by custom_barplot)
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
                            color='black',
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
    Handle -D and _additive suffix logic for determining filled vs empty markers.
    
    Logic:
    - If name has -D suffix (before parenthetical or at end): filled marker, strip -D for grouping
    - If name has _additive anywhere: empty marker (additive), strip _additive for grouping
    - Otherwise: needs to check if a corresponding -D version exists to determine if additive
    
    Returns (stripped_name, variant_type) where variant_type is:
    - 'full' if has -D suffix (filled marker)
    - 'additive' if has _additive (empty marker)
    - 'unknown' if neither (will be determined later based on pairing)
    """
    # Check for -D suffix (before parenthetical or at end)
    if re.search(r'-D(?:\s*\(|$)', name):
        stripped = re.sub(r'-D(?=\s*\(|$)', '', name)
        return stripped, 'full'
    
    # Check for _additive anywhere in name
    if '_additive' in name:
        stripped = name.replace('_additive', '')
        return stripped, 'additive'
    
    return name, 'unknown'

def _root_name(name):
    """Everything before the first '(' (trimmed)."""
    return name.split('(')[0].strip()

def _canonical_name(base):
    """
    Get the canonical name for grouping: strip -D suffix, then get root.
    This determines which items share the same x-offset and marker.
    """
    stripped, _ = _strip_variant_suffix(base)
    return _root_name(stripped)

def visualize_model_performance(
    data, figsize=(12, 6), title="Model Performance Across Datasets", 
    ylabel="Performance Score", 
    colors=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#d35400', '#34495e'],
    highlighted_model=None, ylim=(0, 1), debug=False, additive_offset=0.05, legend_loc='lower left'
):
    """
    Visualize model performance across datasets with automatic grouping.
    
    Grouping logic:
    1. Columns ending with _N (where N is a digit) are treated as repeats and aggregated.
    2. Models with -D suffix are the "full" models (filled markers). Models without -D
       but with a corresponding -D version are "additive" (empty markers, lighter color).
    3. Models with _additive are additive versions. Models without _additive but with
       a corresponding _additive version are the "full" models.
    4. Models sharing the same prefix before '(' are grouped at the same x-offset.
    
    Parameters:
    -----------
    data : DataFrame or ndarray
        Performance data with models as columns and datasets as rows.
    debug : bool
        If True, print diagnostic information about how columns are being processed.
    additive_offset : float
        Horizontal offset between non-additive and additive markers when both are present.
        Set to 0 to disable offsetting. Default is 0.05.
    """
    # Convert to DataFrame if numpy array
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

    # For each base, determine: stripped name, variant_type, and canonical name
    base_info = {}
    for base in bases_in_order:
        stripped, variant_type = _strip_variant_suffix(base)
        canonical = _canonical_name(base)
        base_info[base] = {
            'stripped': stripped,
            'variant_type': variant_type,  # 'full', 'additive', or 'unknown'
            'canonical': canonical
        }
    
    # For 'unknown' variants, determine if they're additive by checking for corresponding -D version
    # OR if they're the "full" version because a corresponding _additive version exists
    
    # Collect all stripped names that have a -D (full) version
    stripped_with_D = set()
    for base in bases_in_order:
        if base_info[base]['variant_type'] == 'full':
            stripped_with_D.add(base_info[base]['stripped'])
    
    # Collect all stripped names that have an _additive version
    stripped_with_additive = set()
    for base in bases_in_order:
        if base_info[base]['variant_type'] == 'additive':
            stripped_with_additive.add(base_info[base]['stripped'])
    
    # Now determine is_additive for each base
    for base in bases_in_order:
        info = base_info[base]
        if info['variant_type'] == 'full':
            info['is_additive'] = False  # Has -D = filled marker
        elif info['variant_type'] == 'additive':
            info['is_additive'] = True   # Has _additive = empty marker
        else:
            # 'unknown' - check if there's a corresponding -D version (then this is additive)
            # OR if there's a corresponding _additive version (then this is the full/non-additive one)
            if info['stripped'] in stripped_with_D:
                info['is_additive'] = True  # Has a -D counterpart, so this is additive
            elif info['stripped'] in stripped_with_additive:
                info['is_additive'] = False  # Has an _additive counterpart, so this is the full version
            else:
                info['is_additive'] = False  # Standalone model, treat as full

    # Aggregate repeats for each base (mean/std across repeats per dataset)
    aggregated_data = {}
    for base in bases_in_order:
        model_cols = [c for c in data.columns if col_to_base[c] == base]
        model_df = data[model_cols]
        aggregated_data[base] = {
            'mean': model_df.mean(axis=1),
            'std': model_df.std(axis=1) if len(model_cols) > 1 else np.zeros(len(model_df)),
            'has_repeats': len(model_cols) > 1
        }
    
    # Debug output
    if debug:
        print("\n=== DEBUG: Column Processing ===")
        print(f"Original columns: {list(data.columns)}")
        print(f"\nColumn -> Base mapping:")
        for col, base in col_to_base.items():
            print(f"  '{col}' -> '{base}'")
        print(f"\nBase info:")
        for base in bases_in_order:
            info = base_info[base]
            agg = aggregated_data[base]
            cols = [c for c in data.columns if col_to_base[c] == base]
            print(f"  '{base}':")
            print(f"    stripped='{info['stripped']}', variant_type='{info['variant_type']}', is_additive={info['is_additive']}")
            print(f"    columns={cols}, has_repeats={agg['has_repeats']}")
        print("=== END DEBUG ===\n")

    # Get unique canonical names in order of first appearance (for x-offset assignment)
    canonicals_in_order = []
    seen_canonicals = set()
    for base in bases_in_order:
        c = base_info[base]['canonical']
        if c not in seen_canonicals:
            canonicals_in_order.append(c)
            seen_canonicals.add(c)

    # Assign a base color per canonical from provided palette
    canonical_base_color = {}
    for i, canonical in enumerate(canonicals_in_order):
        canonical_base_color[canonical] = colors[i % len(colors)]

    # Within each canonical, collect bases (items) in appearance order
    # Separate into non-additive and additive for color assignment
    canonical_to_bases = {c: {'normal': [], 'additive': []} for c in canonicals_in_order}
    for base in bases_in_order:
        info = base_info[base]
        if info['is_additive']:
            canonical_to_bases[info['canonical']]['additive'].append(base)
        else:
            canonical_to_bases[info['canonical']]['normal'].append(base)

    # Build per-item colors
    # Normal items get slight lightness variation among themselves
    # Additive items get a much lighter version of their corresponding normal item's color
    base_colors = {}
    for canonical in canonicals_in_order:
        base_hex = canonical_base_color[canonical]
        normal_family = canonical_to_bases[canonical]['normal']
        additive_family = canonical_to_bases[canonical]['additive']
        
        # Assign colors to normal items with slight variation
        n = len(normal_family)
        if n == 0:
            normal_deltas = []
        elif n == 1:
            normal_deltas = [0.0]
        else:
            max_delta = 0.12
            if n == 2:
                normal_deltas = [-max_delta/2, max_delta/2]
            else:
                normal_deltas = list(np.linspace(-max_delta, max_delta, n))
        
        for b, d in zip(normal_family, normal_deltas):
            base_colors[b] = _adjust_lightness(base_hex, d)
        
        # Assign colors to additive items - much lighter versions
        # Try to match with corresponding normal item if exists
        for additive_base in additive_family:
            stripped = base_info[additive_base]['stripped']
            # Find matching normal item
            matching_normal = None
            for normal_base in normal_family:
                normal_stripped = base_info[normal_base]['stripped']
                if normal_stripped == stripped:
                    matching_normal = normal_base
                    break
            
            if matching_normal is not None:
                # Use same base but much lighter
                base_colors[additive_base] = _adjust_lightness(base_colors[matching_normal], 0.25)
            else:
                # No matching normal, use canonical base color but lighter
                base_colors[additive_base] = _adjust_lightness(base_hex, 0.3)

    # Assign markers per canonical group
    # Items in the same canonical group (including additive variants) share the same marker
    markers = ['s', 'D', 'p', '^', 'v', '<', '>', 'h', 'H', 'o', '*', '+', 'x',]
    canonical_markers = {}
    for i, canonical in enumerate(canonicals_in_order):
        canonical_markers[canonical] = markers[i % len(markers)]

    # Within a canonical, we may have multiple "variants" (different parenthetical suffixes)
    # Each variant (normal + its additive) should share the same marker
    # But different variants within the same canonical should have different markers
    
    # Actually, let's refine: group by stripped name (after removing _additive but before removing parenthetical)
    # This way, "Model (A)" and "Model (A)_additive" share marker, 
    # "Model (B)" and "Model (B)_additive" share a different marker
    
    stripped_to_bases = {}
    for base in bases_in_order:
        stripped = base_info[base]['stripped']
        if stripped not in stripped_to_bases:
            stripped_to_bases[stripped] = []
        stripped_to_bases[stripped].append(base)
    
    # Get unique stripped names in order
    stripped_in_order = []
    seen_stripped = set()
    for base in bases_in_order:
        s = base_info[base]['stripped']
        if s not in seen_stripped:
            stripped_in_order.append(s)
            seen_stripped.add(s)
    
    # Assign markers per stripped name
    stripped_markers = {}
    for i, stripped in enumerate(stripped_in_order):
        stripped_markers[stripped] = markers[i % len(markers)]

    # Set up plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Tight x-limits around datasets
    ax.set_xlim(-0.5, len(data.index) - 0.5)

    # Background shading and dividers
    ax.axvspan(-0.5, len(data.index) - 0.5, alpha=0.1, color='gray', zorder=0)
    for i in range(1, len(data.index)):
        ax.axvline(x=i - 0.5, color='white', linewidth=2, zorder=1)

    # Horizontal offsets based on number of unique CANONICALS
    num_canonicals = len(canonicals_in_order)
    if num_canonicals == 1:
        offsets = {canonicals_in_order[0]: 0.0}
    else:
        spread = 0.09
        center = (num_canonicals - 1) * spread / 2.0
        offsets = {canonical: (i * spread) - center for i, canonical in enumerate(canonicals_in_order)}

    # Determine which stripped names have both additive and non-additive versions
    stripped_has_both = set()
    if additive_offset != 0:
        stripped_has_additive = set()
        stripped_has_full = set()
        for base in bases_in_order:
            info = base_info[base]
            if info['is_additive']:
                stripped_has_additive.add(info['stripped'])
            else:
                stripped_has_full.add(info['stripped'])
        stripped_has_both = stripped_has_additive & stripped_has_full

    # Plot per base (item), using canonical offset and stripped marker
    plot_handles = {}  # Store handles for legend reordering
    for base in bases_in_order:
        info = base_info[base]
        canonical = info['canonical']
        stripped = info['stripped']
        is_additive = info['is_additive']
        
        # Base x position from canonical offset
        x_pos = np.arange(len(data.index)) + offsets[canonical]
        
        # Apply additive offset if this stripped name has both versions
        if additive_offset != 0 and stripped in stripped_has_both:
            if is_additive:
                x_pos = x_pos + additive_offset / 2
            else:
                x_pos = x_pos - additive_offset / 2

        means = aggregated_data[base]['mean']
        stds = aggregated_data[base]['std']
        has_repeats = aggregated_data[base]['has_repeats']

        # color & marker
        color = base_colors[base]
        marker = stripped_markers[stripped]

        # Handle highlight (by base name)
        is_highlighted = (highlighted_model is not None and base == highlighted_model)
        plot_color = '#000000' if is_highlighted else color
        edge_color = 'black' if is_highlighted else color

        # Additive models: use unfilled markers (just edge)
        if is_additive:
            facecolor = 'white'
            edgewidth = 2
        else:
            facecolor = plot_color
            edgewidth = 1

        # Plot - only show error bars if there are actual repeats
        if has_repeats:
            # For additive models with error bars, we need to control whisker/cap colors separately
            if is_additive:
                # Additive: white fill, thin colored edge, thin colored whiskers
                handle = ax.errorbar(
                    x_pos, means, yerr=stds,
                    fmt=marker, markersize=10,
                    markerfacecolor='white',
                    markeredgecolor=color,  # Use original color, not plot_color
                    markeredgewidth=1,  # Thin edge to match non-additive
                    ecolor=color,  # Whisker color matches the model color
                    capsize=10, capthick=1.5, elinewidth=1.5,  # Thinner whiskers to match
                    label=f'{base} (±σ)', alpha=1
                )
            else:
                # Non-additive: filled with color
                handle = ax.errorbar(
                    x_pos, means, yerr=stds,
                    fmt=marker, markersize=10,
                    markerfacecolor=plot_color,
                    markeredgecolor=edge_color if not is_highlighted else 'yellow',
                    markeredgewidth=1,  # Changed from edgewidth variable
                    ecolor=plot_color,  # Whisker color matches the marker
                    capsize=10, capthick=2.25, elinewidth=2.25,
                    label=f'{base} (±σ)', alpha=1
                )
        else:
            # No repeats - just markers
            if is_additive:
                facecolor = 'white'
                edgewidth = 1  # Thin edge
            else:
                facecolor = plot_color
                edgewidth = 1  # Consistent thin edge

            handle, = ax.plot(
                x_pos, means, linestyle='', marker=marker,
                color=plot_color, markersize=10,
                markerfacecolor=facecolor,
                markeredgecolor=edge_color if not is_highlighted else 'yellow',
                markeredgewidth=edgewidth,
                label=base, alpha=1
            )

        # Store handle info for legend reordering
        plot_handles[base] = {
            'handle': handle,
            'label': base, #f'{base} (±σ)' if has_repeats else base,
            'stripped': stripped,
            'is_additive': is_additive
        }

    # Build paired legend: group by stripped name, non-additive first then additive
    # This creates a 2-column legend where pairs are on the same row
    ordered_handles = []
    ordered_labels = []
    
    # Get unique stripped names in order
    seen_stripped = []
    for base in bases_in_order:
        s = base_info[base]['stripped']
        if s not in seen_stripped:
            seen_stripped.append(s)
    
    # For each stripped name, add non-additive first, then additive
    for stripped in seen_stripped:
        # Find non-additive version(s)
        for base in bases_in_order:
            if plot_handles[base]['stripped'] == stripped and not plot_handles[base]['is_additive']:
                ordered_handles.append(plot_handles[base]['handle'])
                ordered_labels.append(plot_handles[base]['label'])
        # Find additive version(s)
        for base in bases_in_order:
            if plot_handles[base]['stripped'] == stripped and plot_handles[base]['is_additive']:
                ordered_handles.append(plot_handles[base]['handle'])
                ordered_labels.append(plot_handles[base]['label'])
    
    # Add any remaining (unpaired) models
    for base in bases_in_order:
        if plot_handles[base]['handle'] not in ordered_handles:
            ordered_handles.append(plot_handles[base]['handle'])
            ordered_labels.append(plot_handles[base]['label'])

    # Labels & title
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)

    # X ticks are datasets
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=16)

    # Y range
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    # Two-column legend with pairs aligned
    ax.legend(ordered_handles, ordered_labels, loc=legend_loc, ncol=2,
              frameon=False, fancybox=True, shadow=True,
              facecolor='white', edgecolor='gray', fontsize=11,
              columnspacing=1.0, handletextpad=0.5)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, ax