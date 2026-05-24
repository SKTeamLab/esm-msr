import os
import time
import argparse
import pickle
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.manifold import MDS

from tqdm import tqdm

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align

from esm.utils.structure.protein_chain import ProteinChain
from esm_msr.preprocessing import MegaScaleDatasetPreprocessor
import subprocess


def dataframe_to_fasta(df, name_col, seq_col, output_file):
    if name_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"Columns '{name_col}' or '{seq_col}' not found in the DataFrame")
    
    records = []
    for index, row in df.iterrows():
        record = SeqRecord(
            Seq(row[seq_col]),
            id=row[name_col],
            name=row[name_col],
            description=""
        )
        records.append(record)
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    print(f"FASTA file '{output_file}' has been created successfully.")

def generate_splits_from_clusters(candidate_datasets: pd.DataFrame, 
                                  clusters: Dict[str, List[str]],
                                  test_ids: Set[str],
                                  functional_ids: Set[str],
                                  allow_redundancy: bool = True,
                                  n_validation: int = 15,
                                  n_test_tsuboyama: int = 15,
                                  seed: int = 42) -> Dict:
    
    print(f'Target internal test proteins: {n_test_tsuboyama}')
    print(f'Target validation proteins: {n_validation}')

    options = candidate_datasets.copy()
    candidate_ids = set()
    
    for _, row in options.iterrows():
        candidate_ids.add(row['name'])

    quarantine_ids = test_ids.union(functional_ids)
    quarantined_clusters = set()
    for rep, members in clusters.items():
        if any(m in quarantine_ids for m in members):
            quarantined_clusters.add(rep)

    safe_clusters = [rep for rep in clusters if rep not in quarantined_clusters]
    
    print(f"Total Clusters: {len(clusters)} | Quarantined: {len(quarantined_clusters)} | Safe: {len(safe_clusters)}")

    safe_clusters.sort()
    
    random.seed(seed)
    np.random.seed(seed)
    np.random.shuffle(safe_clusters)

    test_set = set()
    training_set = set()
    validation_set = set()

    for rep in clusters:
        clusters[rep].sort()

    for rep in safe_clusters:
        members = [m for m in clusters[rep] if m in candidate_ids]
        if not members:
            continue

        if allow_redundancy:
            selected = members
        else:
            selected = [rep] if rep in members else [members[0]]

        if len(test_set) < n_test_tsuboyama:
            target_set = test_set
        elif len(validation_set) < n_validation:    
            target_set = validation_set
        elif len(training_set) < 1000:
            target_set = training_set
        else:
            continue

        for s in selected:
            dms_id = options.loc[options['name'] == s, 'DMS_id'].iloc[0]
            target_set.add(dms_id)

    if len(training_set) < 1 or len(validation_set) < 1 or (len(test_set) < 1 and n_test_tsuboyama > 0):
        print(f"Warning: Low sequence counts after clustering. Training: {len(training_set)}, Validation: {len(validation_set)}, Test: {len(test_set)}")
    
    splits = {
        'train': sorted([c.replace('|','_')+'.pdb' for c in training_set]),
        'val': sorted([c.replace('|','_')+'.pdb' for c in validation_set]),
        'test': sorted([c.replace('|','_')+'.pdb' for c in test_set]),
        'thermostability': sorted([c.replace('|','_')+'.pdb' for c in test_ids])
    }
    
    print(f"Split sizes - Training: {len(training_set)}, Validation: {len(validation_set)}, Testing: {len(splits['test'])} candidates, Thermostability: {len(splits['thermostability'])}")

    return splits


# ==========================================
# Rigorous Validations & Visualizations
# ==========================================

def calculate_rigorous_identity(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_open_penalty=-10, gap_extend_penalty=-1):
    len1, len2 = len(seq1), len(seq2)

    if len1 == 0 and len2 == 0: return 1.0, ""
    if len1 == 0 or len2 == 0: return 0.0, ""

    try:
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = gap_open_penalty
        aligner.extend_gap_score = gap_extend_penalty
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

        alignments = aligner.align(seq1, seq2)
    except Exception as e:
        raise AssertionError(f"Alignment fundamentally failed. Investigate the sequences. Error: {e}\nSeq1: {seq1[:30]}\nSeq2: {seq2[:30]}")

    try:
        alignment = next(alignments)
    except StopIteration:
        return 0.0, ""

    aligned_seq1 = alignment[0]
    aligned_seq2 = alignment[1]
    aln_len = len(aligned_seq1)

    if aln_len == 0: return 0.0, ""

    matches = sum(1 for i in range(aln_len) if aligned_seq1[i] == aligned_seq2[i] and aligned_seq1[i] != '-')
    return matches / min(len1, len2), str(alignment)


def visualize_protein_similarity_mds_improved(
        protein_sets_dict: Dict[str, Dict[str, str]],
        identity_threshold: float,
        figsize: Tuple[int, int] = (10, 10),
        node_size: int = 300,
        font_size: int = 8,
        node_alpha: float = 0.6, 
        min_edge_width: float = 0.5, 
        max_edge_width: float = 4.0,  
        close_proximity_threshold_mds_units: float = 0.02,
        mds_random_state: int = 42,
        output_filename: str = "protein_similarity_mds_improved.png",
        precomputed_distance_matrix: np.ndarray = None,
        protein_ids_order: List[str] = None,
        alignment_params: Dict[str, Any] = None,
        calculate_identity_func = None,
        sizes: Dict[str, float] = None):
    
    print(f"Starting MDS visualization for: {output_filename}")

    if alignment_params is None: alignment_params = {'gap_open_penalty': -10, 'gap_extend_penalty': -1} 

    all_proteins_dict: Dict[str, str] = {}
    protein_set_membership: Dict[str, List[str]] = defaultdict(list)

    # Note: domainome is explicitly ignored upstream, but sanity check to prevent pollution
    if 'domainome' in protein_sets_dict:
        raise AssertionError("Domainome explicitly disallowed in MDS sets per user request. Fix dict creation.")

    for set_name, proteins_in_set in protein_sets_dict.items():
        if not isinstance(proteins_in_set, dict): continue
        for protein_id, sequence in proteins_in_set.items():
            if protein_id not in all_proteins_dict:
                all_proteins_dict[protein_id] = sequence
            elif all_proteins_dict[protein_id] != sequence:
                 pass
            protein_set_membership[protein_id].append(set_name)
    
    protein_to_primary_set_map: Dict[str, str] = {pid: sets[0] for pid, sets in protein_set_membership.items() if sets}

    protein_ids = protein_ids_order if protein_ids_order else list(all_proteins_dict.keys()) 
    protein_ids = [pid for pid in protein_ids if pid in all_proteins_dict]

    protein_sequences = [all_proteins_dict[pid] for pid in protein_ids]
    n_proteins = len(protein_ids)

    if n_proteins < 2:
        print("Too few proteins (<2) for MDS. Skipping.")
        return

    distance_matrix = np.zeros((n_proteins, n_proteins))
    for i in range(n_proteins):
        for j in range(i + 1, n_proteins):
            seq1, seq2 = protein_sequences[i], protein_sequences[j]
            if calculate_identity_func is not None:
                identity, _ = calculate_identity_func(seq1, seq2, **alignment_params)
            else:
                identity = sum(1 for a, b in zip(seq1, seq2) if a == b) / max(len(seq1), len(seq2))
            
            distance = np.sqrt(max(0.0, 1.0 - identity))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance 

    if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0) 

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=mds_random_state, n_init=4, max_iter=300, normalized_stress=False) 

    if n_proteins <= mds.n_components: coords_2d = np.random.rand(n_proteins, 2) 
    else: coords_2d = mds.fit_transform(distance_matrix)
    
    pos = {protein_ids[i]: coords_2d[i] for i in range(n_proteins)}

    inter_set_identity_edges = []
    for i in range(n_proteins):
        for j in range(i + 1, n_proteins):
            pid1, pid2 = protein_ids[i], protein_ids[j]
            identity = np.round(1.0 - (distance_matrix[i, j] ** 2), 6)
            if identity > identity_threshold:
                if protein_to_primary_set_map.get(pid1) != protein_to_primary_set_map.get(pid2):
                    inter_set_identity_edges.append((pid1, pid2, identity))

    fig, ax = plt.subplots(figsize=figsize) 
    G = nx.Graph() 
    G.add_nodes_from(protein_ids) 

    critical_colors = {
        'train': '#000000',  # Solid Black (highest visual weight)
        'val': '#E31A1C',    # Strong Red 
        'test': '#2CA02C'    # Strong Green
    }

    # Categorical palette with high variance, explicitly stripped of Black/Red/Green 
    # to ensure secondary datasets never masquerade as training/validation data.
    available_colors = [
        '#377eb8', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#f781bf', 
        '#999999', '#66c2a5', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#8dd3c7', '#ffffb3', 
        '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69'
    ]

    set_names_ordered = list(protein_sets_dict.keys())
    color_map = {}
    color_idx = 0

    for s_name in set_names_ordered:
        if s_name in critical_colors:
            color_map[s_name] = critical_colors[s_name]
        else:
            if color_idx >= len(available_colors):
                raise AssertionError(
                    f"Visualization failed: Ran out of unique colors for dataset '{s_name}'. "
                    f"You have {len(set_names_ordered)} sets but only {len(available_colors) + len(critical_colors)} colors available. "
                    f"Expand the 'available_colors' list to ensure strict visual distinctness."
                )
            color_map[s_name] = available_colors[color_idx]
            color_idx += 1
    # ==========================================

    multi_set_proteins = {pid for pid, sets in protein_set_membership.items() if len(sets) > 1}
    
    node_colors_list, node_edge_colors, node_edge_widths, node_sizes_list = [], [], [], []
    if sizes is None: sizes = {}
    
    for pid_node in protein_ids:
        primary_set_name = protein_to_primary_set_map.get(pid_node)
        node_colors_list.append(color_map.get(primary_set_name, 'lightgrey') if primary_set_name else 'lightgrey')
        node_sizes_list.append(node_size * sizes.get(primary_set_name, 1.0) if primary_set_name else node_size)
        node_edge_colors.append('black' if pid_node in multi_set_proteins else 'darkgray')
        node_edge_widths.append(3.0 if pid_node in multi_set_proteins else 0.5)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes_list, node_color=node_colors_list, 
                           alpha=node_alpha, edgecolors=node_edge_colors, linewidths=node_edge_widths)

    if inter_set_identity_edges:
        for pid1, pid2, identity_val in inter_set_identity_edges:
            if pid1 not in pos or pid2 not in pos: continue
            pos1, pos2 = pos[pid1], pos[pid2]
            mid_point = (pos1 + pos2) / 2.0
            color1 = color_map.get(protein_to_primary_set_map.get(pid1), 'grey')
            color2 = color_map.get(protein_to_primary_set_map.get(pid2), 'grey')
            line_width = min_edge_width + (max_edge_width - min_edge_width) * max(0.0, min(1.0, identity_val))
            ax.plot([pos1[0], mid_point[0]], [pos1[1], mid_point[1]], color=color1, linewidth=line_width, alpha=0.75, solid_capstyle='round', zorder=1)
            ax.plot([mid_point[0], pos2[0]], [mid_point[1], pos2[1]], color=color2, linewidth=line_width, alpha=0.75, solid_capstyle='round', zorder=1)

    legend_handles = []
    set_protein_counts = defaultdict(int)
    for pid in protein_ids: set_protein_counts[protein_to_primary_set_map.get(pid, 'Unknown')] += 1

    for s_name in set_names_ordered:
        if s_name in color_map and set_protein_counts[s_name] > 0:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{s_name} ({set_protein_counts[s_name]})', markersize=10, markerfacecolor=color_map[s_name], markeredgecolor='darkgray', markeredgewidth=0.5, alpha=node_alpha))

    if legend_handles:
        ax.legend(handles=legend_handles, title="Protein Sets", loc='best', fontsize=max(8, font_size), frameon=True)

    ax.set_title(f"Protein Similarity Network (n={len(protein_ids)} unique)\nEdges > {identity_threshold * 100:.0f}% identity", fontsize=font_size + 2, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_intraset_matrices(results: Dict, figsize: Tuple[int, int], output_prefix: str):
    for set_name, set_data in results.items():
        if set_name not in ['train', 'val', 'test']: continue
        if set_data['n_proteins'] < 2: continue
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(set_data['identity_matrix'], dtype=bool), k=1)
        show_labels = set_data['n_proteins'] <= 50
        sns.heatmap(set_data['identity_matrix'], mask=mask, cmap='RdYlBu_r', vmin=0, vmax=1, center=0.5, square=True, linewidths=0.5 if show_labels else 0, cbar_kws={"shrink": 0.8, "label": "Sequence Identity"}, annot=set_data['n_proteins'] <= 20, fmt='.2f', annot_kws={'size': 8}, xticklabels=show_labels, yticklabels=show_labels)
        
        if show_labels:
            ax.set_xticks(np.arange(len(set_data['protein_ids'])) + 0.5); ax.set_yticks(np.arange(len(set_data['protein_ids'])) + 0.5)
            ax.set_xticklabels(set_data['protein_ids'], rotation=90, ha='right'); ax.set_yticklabels(set_data['protein_ids'], rotation=0)
        else:
            ax.set_xlabel(f'Protein Index'); ax.set_ylabel(f'Protein Index')
        ax.set_title(f"Intra-set Similarity Matrix: {set_name} ({set_data['n_proteins']} proteins)", fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.savefig(f"{output_prefix}_matrix_{set_name}.png", dpi=300, bbox_inches='tight'); plt.close()
    
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(8, len(results)*0.4 + 2))
        set_names = list(results.keys())
        stats_matrix = np.array([[results[s]['min_identity'], results[s]['median_identity'], results[s]['mean_identity'], results[s]['max_identity']] for s in set_names])
        sns.heatmap(stats_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0, vmax=1, yticklabels=set_names, xticklabels=['Min', 'Median', 'Mean', 'Max'], ax=ax)
        ax.set_title('Identity Distribution Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.savefig(f"{output_prefix}_summary_matrices.png", dpi=300, bbox_inches='tight'); plt.close()

def create_summary_statistics_table(results: Dict, mut_counts: Dict[str, int], output_prefix: str):
    table_data = []
    # Iterate over mut_counts to capture everything, including granular functional sets excluded from vis
    for set_name, n_mut in mut_counts.items():
        if set_name in results:
            n_proteins = results[set_name]['n_proteins']
        elif any(f in set_name for f in ['DLG4_', 'GRB2_', 'MYO_', 'ESTA_', 'GB1_']):
            n_proteins = 1 # Functional sub-datasets are inherently single-protein
        else:
            n_proteins = 0
            
        table_data.append({
            'Set': set_name, 
            'N Proteins': n_proteins, 
            'N Mutations': n_mut, 
            'N Mutations per protein (avg)': f"{n_mut / n_proteins if n_proteins > 0 else 0:.2f}"
        })
        
    df = pd.DataFrame(table_data)
    df.to_csv(f"{output_prefix}_set_sizes.csv", index=False)
    print(df.to_string(index=False))

def analyze_intraset_similarity(protein_sets_dict: Dict[str, Dict[str, str]], mut_counts: Dict[str, int], alignment_params: Dict[str, Any] = None, figsize_matrix: Tuple[int, int] = (10, 8), output_prefix: str = "intraset_similarity", calculate_identity_func = None):
    results = {}
    for set_name, proteins in protein_sets_dict.items():
        if not isinstance(proteins, dict): continue
        protein_ids = list(proteins.keys())
        n_proteins = len(protein_ids)
        if n_proteins < 2: continue
        
        identity_matrix = np.ones((n_proteins, n_proteins))
        identities_list = []
        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                identity, _ = calculate_identity_func(proteins[protein_ids[i]], proteins[protein_ids[j]], **(alignment_params or {}))
                identity_matrix[i, j] = identity_matrix[j, i] = identity
                identities_list.append(identity)
        
        results[set_name] = {'protein_ids': protein_ids, 'identity_matrix': identity_matrix, 'n_proteins': n_proteins, 'mean_identity': np.mean(identities_list) if identities_list else 0, 'median_identity': np.median(identities_list) if identities_list else 0, 'min_identity': np.min(identities_list) if identities_list else 1, 'max_identity': np.max(identities_list) if identities_list else 1}
    
    create_intraset_matrices(results, figsize_matrix, output_prefix)
    create_summary_statistics_table(results, mut_counts, output_prefix)
    return results

def run_intraset_analysis(protein_sets_dict, mut_counts, calculate_rigorous_identity, output_prefix, alignment_params=None):
    return analyze_intraset_similarity(protein_sets_dict=protein_sets_dict, mut_counts=mut_counts, alignment_params=alignment_params, figsize_matrix=(10, 8), output_prefix=output_prefix, calculate_identity_func=calculate_rigorous_identity)

def analyze_interset_max_overlaps(protein_sets_dict: Dict[str, Dict[str, str]], calculate_identity_func, output_prefix: str):
    sets = list(protein_sets_dict.keys())
    max_overlaps = pd.DataFrame(index=sets, columns=sets, dtype=float)
    
    for i, set_a in enumerate(sets):
        for j, set_b in enumerate(sets):
            if i > j:
                max_overlaps.loc[set_a, set_b] = max_overlaps.loc[set_b, set_a]
                continue
            
            max_id = 0.0
            if i == j:
                seqs_a = list(protein_sets_dict[set_a].values())
                for idx1 in range(len(seqs_a)):
                    for idx2 in range(idx1 + 1, len(seqs_a)):
                        ident, _ = calculate_identity_func(seqs_a[idx1], seqs_a[idx2])
                        if ident > max_id: max_id = ident
                if len(seqs_a) <= 1: max_id = 1.0
            else:
                for seq_a in protein_sets_dict[set_a].values():
                    for seq_b in protein_sets_dict[set_b].values():
                        ident, _ = calculate_identity_func(seq_a, seq_b)
                        if ident > max_id: max_id = ident
                        
            max_overlaps.loc[set_a, set_b] = max_id

    max_overlaps.to_csv(f"{output_prefix}_max_overlaps.csv")
    fig, ax = plt.subplots(figsize=(max(8, len(sets)*0.6), max(6, len(sets)*0.6)))
    sns.heatmap(max_overlaps, annot=len(sets)<=25, fmt='.2f', cmap='Reds', vmin=0, vmax=1, cbar_kws={"label": "Max Sequence Identity"}, ax=ax)
    ax.set_title('Maximum Overlaps Between Protein Sets', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(); plt.savefig(f"{output_prefix}_max_overlaps_matrix.png", dpi=300, bbox_inches='tight'); plt.close()
    return max_overlaps

# ==========================================
# Main Execution Logic
# ==========================================

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    REPO_ROOT = Path(__file__).resolve().parent.parent
    homology_dir = os.path.join(REPO_ROOT, "data/tsuboyama/homology")
    
    if os.path.exists(homology_dir):
        import shutil
        shutil.rmtree(homology_dir)
    os.makedirs(homology_dir)
    
    functional_proteins = {
        'GRB2': 'TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN', 
        'DLG4': 'PRRIVIHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKP',
        'EstA': 'AEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN',
        'Myo': 'MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG',
        'GB1': 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    }

    ds = MegaScaleDatasetPreprocessor(
        data_file=os.path.join(REPO_ROOT, 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'), 
        af_model_folder=os.path.join(REPO_ROOT, 'data/tsuboyama/AlphaFold_model_PDBs')
    )
    df = ds.df
    ref = df.groupby('code_wt').first().reset_index()
    ref['name'] = ref['code_wt']
        
    dataframe_to_fasta(ref.reset_index(), 'name', 'aa_seq', os.path.join(homology_dir, 'tsuboyama_seqs.fasta'))

    thermo_datasets = ['ssym', 'ptmul', 's461', 'k2369', 'q3421', 's571', 's783', 's2648', 's8754']
    thermo_dict = {dset: {} for dset in thermo_datasets}
    thermo_mut_counts = {}

    if args.train_doubles:
        test1 = df.loc[~df['mut_type'].str.contains(':'), ['code', 'chain', 'pdb_file']]
        test_df = pd.concat([test1]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()    
    elif args.test_doubles:
        test1 = df.loc[df['mut_type'].str.contains(':'), ['code', 'chain', 'pdb_file']]
        test2 = pd.read_csv(os.path.join(REPO_ROOT, f'data/preprocessed/ptmul_mapped.csv'), index_col=0)[['code', 'chain', 'pdb_file']]
        test_df = pd.concat([test1, test2]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    elif args.train_synthetic:
        test1 = df.loc[df['code'].str.len()==4, ['code', 'chain', 'pdb_file']]
        test2 = df.loc[df['code'].str.startswith('v2_'), ['code', 'chain', 'pdb_file']]
        test_df = pd.concat([test1, test2]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    elif args.test_synthetic:
        test1 = df.loc[~(df['code'].str.len()==4) & ~(df['code'].str.startswith('v2_')), ['code', 'chain', 'pdb_file']]
        test_df = pd.concat([test1]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    else:
        test_dfs = []
        for dset in thermo_datasets:
            print(dset)
            # Load the full dataframe first to access mutation columns
            raw_dset_df = pd.read_csv(os.path.join(REPO_ROOT, f'data/preprocessed/{dset}_mapped.csv'), index_col=0)
            
            if 'mut_type' not in raw_dset_df.columns:
                try:
                    raw_dset_df['mut_type'] = raw_dset_df['mut_info']
                except:
                    raw_dset_df['mut_type'] = raw_dset_df['wild_type'] + raw_dset_df['position'].astype(str) + raw_dset_df['mutation']
            
            # Deduplicate by protein code and mutation identifier
            dedup_dset_df = raw_dset_df.drop_duplicates(subset=['code', 'mut_type']) 
            
            thermo_mut_counts[dset] = dedup_dset_df.shape[0]
            
            # Subset down to what the sequence extraction needs
            dset_df = dedup_dset_df[['code', 'chain', 'pdb_file', 'mut_type']].copy()
            dset_df['source_dset'] = dset
            test_dfs.append(dset_df)
        
        thermo_raw_df = pd.concat(test_dfs)
        test_df = thermo_raw_df.drop_duplicates(subset=['code', 'chain']).copy().reset_index(drop=True)

    test_df['pdb_seq'] = test_df.apply(lambda x: ProteinChain.from_pdb(os.path.join(REPO_ROOT, 'data/structures', x['pdb_file']), x['chain']).sequence, axis=1)
    test_df['name'] = 'test_' + test_df['code'] + '_' + test_df['chain']
    dataframe_to_fasta(test_df, 'name', 'pdb_seq', os.path.join(homology_dir, 'test_seqs.fasta'))

    if not (args.train_doubles or args.test_doubles or args.train_synthetic or args.test_synthetic):
        # Repopulate original dataset splits with their extracted sequences to allow separated metrics
        for dset in thermo_datasets:
            sub_df = thermo_raw_df[thermo_raw_df['source_dset'] == dset].drop_duplicates(subset=['code', 'chain']).copy()
            sub_df['name'] = 'test_' + sub_df['code'] + '_' + sub_df['chain']
            merged = sub_df.merge(test_df[['name', 'pdb_seq']], on='name', how='left')
            thermo_dict[dset] = {row['name']: row['pdb_seq'] for _, row in merged.iterrows() if pd.notna(row['pdb_seq'])}

    # Load Domainome
    domainome_path = os.path.join(REPO_ROOT, 'data/domainome1/domainome_mapped_2026.csv')
    if not os.path.exists(domainome_path): raise FileNotFoundError(f"Missing essential scaffold dataset at {domainome_path}.")
        
    domainome_raw_df = pd.read_csv(domainome_path).dropna(subset=['position', 'pdb_file', 'scaled_fitness'])
    domainome_raw_df['code'] = domainome_raw_df['domain_ID']
    domainome_raw_df['chain'] = 'A'
    domainome_ref = domainome_raw_df.drop_duplicates(subset=['code', 'chain']).copy()
    domainome_ref['pdb_seq'] = domainome_ref.apply(lambda x: ProteinChain.from_pdb(os.path.join(REPO_ROOT, 'data/structures', x['pdb_file']), x['chain']).sequence, axis=1)
    domainome_ref['name'] = 'domainome_' + domainome_ref['code'] + '_' + domainome_ref['chain']
    domainome_proteins = {row['name']: row['pdb_seq'] for _, row in domainome_ref.iterrows()}

    sequences = list(SeqIO.parse(os.path.join(homology_dir, 'tsuboyama_seqs.fasta'), "fasta")) + list(SeqIO.parse(os.path.join(homology_dir, 'test_seqs.fasta'), "fasta"))
    sequences.extend([SeqRecord(Seq(seq), id=pid, name=pid, description="") for pid, seq in functional_proteins.items()])
    sequences.extend([SeqRecord(Seq(seq), id=pid, name=pid, description="") for pid, seq in domainome_proteins.items()])
    SeqIO.write(sequences, os.path.join(homology_dir, 'combined.fasta'), "fasta")

    splits = None
    if args.input_split and os.path.exists(args.input_split):
        with open(args.input_split, 'rb') as f: raw_splits = pickle.load(f)
        splits = {'train': raw_splits.get('train', []), 'val': raw_splits.get('val', [])}
        splits['test'] = raw_splits.get('test_internal', raw_splits.get('test', []))
        splits['thermostability'] = raw_splits.get('test_external', raw_splits.get('thermostability', []))
    else:
        commands = [
            f"cd {homology_dir} && mmseqs createdb combined.fasta DB && " +
            f"mmseqs cluster DB clu tmp --min-seq-id {args.prescreen_identity} -c 0.0 --alignment-mode 2 --threads 1 && " +
            "mmseqs createtsv DB DB clu cluster.tsv"
        ]
        for cmd in commands: subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)

        clusters = defaultdict(list)
        with open(os.path.join(homology_dir, 'cluster.tsv'), 'r') as f:
            for line in f: rep, mem = line.strip().split('\t'); clusters[rep].append(mem)

        candidate_datasets = ref.copy()
        candidate_datasets['DMS_id'] = candidate_datasets['name']
        splits = generate_splits_from_clusters(candidate_datasets, clusters, set(test_df['name'].tolist()), set(functional_proteins.keys()), args.allow_redundancy, args.n_validation, args.n_test_tsuboyama, args.seed)
        pd.Series({k:str(v) for k,v in splits.items()}).to_csv(os.path.join(REPO_ROOT, 'data', f'{args.output}.csv'))
        with open(os.path.join(REPO_ROOT, 'data', f'{args.output}.pkl'), 'wb') as f: pickle.dump(splits, f)

    split_file_path = args.input_split if args.input_split else os.path.join(REPO_ROOT, 'data', f'{args.output}.pkl')
    ds_post = MegaScaleDatasetPreprocessor(data_file=os.path.join(REPO_ROOT, 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'), af_model_folder=os.path.join(REPO_ROOT, 'data/tsuboyama/AlphaFold_model_PDBs'))
    post_processed_splits = ds_post.create_training_splits(str(split_file_path), -1)
    splits.update({k: post_processed_splits[k] for k in ['train', 'val', 'test']})

    functional_prots_map = {
        'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4', 
        'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4', 
        'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2', 
        'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2', 
        'MYO_HUMAN_Kung_2025_display': 'Myo', 
        'ESTA_BACSU_Nutschel_2020_dTm': 'EstA', 
        'GB1_Wu_2016_binding_domain': 'GB1'
    }
    
    functional_mut_pools = defaultdict(set)
    func_split_mut_counts = {}
    func_dict = {}

    for prot_file, prot_key in functional_prots_map.items():
        df_func_path = os.path.join(REPO_ROOT, f'data/preprocessed/{prot_file}.csv')
        if os.path.exists(df_func_path):
            df_func = pd.read_csv(df_func_path)
            df_func['id'] = df_func['code'] + '_' + df_func['mut_info']
            functional_mut_pools[prot_key].update(df_func['id'].tolist())
            
            # Keep the full suffix for the mutation count table
            func_split_mut_counts[prot_file] = len(df_func['id'].unique())
            
            # Strip the suffix for the overlap/MDS visualizers (grouping by prot_key)
            func_dict[prot_key] = {prot_key: functional_proteins[prot_key]}
        else: 
            raise AssertionError(f"Functional dataset not found: {df_func_path}")

    seq_map = {rec.id.replace('|', '_'): str(rec.seq) for rec in sequences}
    protein_sets_dict_pooled = {
        'train': {str(pid).replace('.pdb', ''): seq_map[str(pid).replace('.pdb', '')] for pid in splits['train'] if str(pid).replace('.pdb', '') in seq_map},
        'val': {str(pid).replace('.pdb', ''): seq_map[str(pid).replace('.pdb', '')] for pid in splits['val'] if str(pid).replace('.pdb', '') in seq_map},
        'test': {str(pid).replace('.pdb', ''): seq_map[str(pid).replace('.pdb', '')] for pid in splits['test'] if str(pid).replace('.pdb', '') in seq_map},
        'thermostability': {str(pid).replace('.pdb', ''): seq_map[str(pid).replace('.pdb', '')] for pid in splits['thermostability'] if str(pid).replace('.pdb', '') in seq_map},
        'functional': functional_proteins,
        'domainome': domainome_proteins
    }

    vis_dir = os.path.join(REPO_ROOT, 'data/visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_output_prefix = os.path.join(vis_dir, args.output)

    alignment_cache = {}
    def memoized_calculate_identity(seq1, seq2, **kwargs):
        pair_key = frozenset([seq1, seq2])
        if pair_key in alignment_cache: return alignment_cache[pair_key]
        res = calculate_rigorous_identity(seq1, seq2, **kwargs)
        alignment_cache[pair_key] = res
        return res

    items_to_drop = {'train': set(), 'val': set()}
    for tgt_set in ['train', 'val']:
        for tgt_id, tgt_seq in protein_sets_dict_pooled[tgt_set].items():
            for ref_set in ['test', 'thermostability', 'functional']:
                for ref_id, ref_seq in protein_sets_dict_pooled[ref_set].items():
                    if tgt_id in items_to_drop[tgt_set]: continue
                    identity, _ = memoized_calculate_identity(tgt_seq, ref_seq)
                    if identity >= args.rigorous_identity:
                        items_to_drop[tgt_set].add(tgt_id)

    for tgt_id, tgt_seq in protein_sets_dict_pooled['val'].items():
        for ref_id, ref_seq in protein_sets_dict_pooled['train'].items():
            if tgt_id in items_to_drop['val']: continue
            identity, _ = memoized_calculate_identity(tgt_seq, ref_seq)
            if identity >= args.rigorous_identity:
                items_to_drop['val'].add(tgt_id)

    splits_modified = False
    for tgt_set in ['train', 'val']:
        for pid in items_to_drop[tgt_set]:
            del protein_sets_dict_pooled[tgt_set][pid]
            splits_modified = True

    if splits_modified:
        splits['train'] = sorted([pid + '.pdb' for pid in protein_sets_dict_pooled['train']])
        splits['val'] = sorted([pid + '.pdb' for pid in protein_sets_dict_pooled['val']])
        pd.Series({k:str(v) for k,v in splits.items()}).to_csv(os.path.join(REPO_ROOT, 'data', f'{args.output}.csv'))
        with open(os.path.join(REPO_ROOT, 'data', f'{args.output}.pkl'), 'wb') as f: pickle.dump(splits, f)

    if not protein_sets_dict_pooled['test'] and not protein_sets_dict_pooled['thermostability']: raise AssertionError("Both test sets are completely empty.")

    # Mut counts and split dictionaries initialization
    mut_counts_pooled = {
        'train': ds_post.split_dfs['train'].shape[0], 
        'val': ds_post.split_dfs['val'].shape[0],
        'test': ds_post.split_dfs['test'].shape[0], 
        
        # Deduplicate the pooled thermostability frame
        'thermostability': thermo_raw_df.drop_duplicates(subset=['code', 'chain', 'mut_type']).shape[0] if 'thermo_raw_df' in locals() else test_df.shape[0],
        
        # Deduplicate domainome (assuming 'position' distinguishes mutations)
        'domainome': domainome_raw_df.drop_duplicates(subset=['code', 'position']).shape[0], 
        
        'functional': sum(len(pool) for pool in functional_mut_pools.values())
    }

    protein_sets_dict_split = {
        'train': protein_sets_dict_pooled['train'], 'val': protein_sets_dict_pooled['val'],
        'test': protein_sets_dict_pooled['test'], 'domainome': protein_sets_dict_pooled['domainome']
    }
    mut_counts_split = {k: mut_counts_pooled[k] for k in ['train', 'val', 'test', 'domainome']}

    if not (args.train_doubles or args.test_doubles or args.train_synthetic or args.test_synthetic):
        surviving_thermo = set(protein_sets_dict_pooled['thermostability'].keys())
        for dset in thermo_datasets:
            protein_sets_dict_split[dset] = {pid: seq for pid, seq in thermo_dict[dset].items() if pid in surviving_thermo}
        mut_counts_split.update(thermo_mut_counts)
    else:
        raise AssertionError("Cannot separate thermostability datasets into constituents when alternative dataset flags (e.g., --train_doubles) are used, as the 10 base datasets are not loaded into memory.")

    for dset, seq_dict in func_dict.items(): protein_sets_dict_split[dset] = seq_dict
    mut_counts_split.update(func_split_mut_counts)

    # --- POOLED EXECUTIONS ---
    analyze_interset_max_overlaps(protein_sets_dict_pooled, memoized_calculate_identity, f"{vis_output_prefix}_pooled")
    mds_pooled_external = {k: v for k, v in protein_sets_dict_pooled.items() if k in ['train', 'val', 'thermostability', 'functional'] and k != 'domainome'}
    visualize_protein_similarity_mds_improved(mds_pooled_external, args.rigorous_identity, output_filename=f"{vis_output_prefix}_pooled_mds_external_other.png", calculate_identity_func=memoized_calculate_identity)
    mds_pooled_internal = {k: protein_sets_dict_pooled[k] for k in ['train', 'val', 'test'] if k in protein_sets_dict_pooled}
    visualize_protein_similarity_mds_improved(mds_pooled_internal, args.rigorous_identity, output_filename=f"{vis_output_prefix}_pooled_mds_internal.png", calculate_identity_func=memoized_calculate_identity)
    mds_pooled_domainome = {k: protein_sets_dict_pooled[k] for k in ['train', 'domainome'] if k in protein_sets_dict_pooled}
    visualize_protein_similarity_mds_improved(mds_pooled_internal, args.rigorous_identity, output_filename=f"{vis_output_prefix}_pooled_mds_domainome.png", calculate_identity_func=memoized_calculate_identity)
    run_intraset_analysis(protein_sets_dict_pooled, mut_counts_pooled, memoized_calculate_identity, f"{vis_output_prefix}_pooled_intraset")

    # --- SPLIT EXECUTIONS ---
    analyze_interset_max_overlaps(protein_sets_dict_split, memoized_calculate_identity, f"{vis_output_prefix}_split")
    mds_split_external = {k: v for k, v in protein_sets_dict_split.items() if k != 'domainome' and k != 'test'}
    visualize_protein_similarity_mds_improved(mds_split_external, args.rigorous_identity, output_filename=f"{vis_output_prefix}_split_mds_external_other.png", calculate_identity_func=memoized_calculate_identity)
    run_intraset_analysis(protein_sets_dict_split, mut_counts_split, memoized_calculate_identity, f"{vis_output_prefix}_split_intraset")

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--input_split', type=str, default=None)
        parser.add_argument('--allow_redundancy', action='store_true')
        parser.add_argument('--train_doubles', action='store_true')
        parser.add_argument('--test_doubles', action='store_true')
        parser.add_argument('--train_synthetic', action='store_true')
        parser.add_argument('--test_synthetic', action='store_true')
        parser.add_argument('--n_validation', type=int, default=20)
        parser.add_argument('--n_test_tsuboyama', type=int, default=30)
        parser.add_argument('--prescreen_identity', type=float, default=0.25)
        parser.add_argument('--rigorous_identity', type=float, default=0.30)
        parser.add_argument('--seed', type=int, default=42)

        args = parser.parse_args()
        main(args)