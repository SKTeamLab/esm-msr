"""Measure sequence and structure homology overlap between the sets of one or more split files.

Reads the pair table written by split_tsuboyama.py (`data/<output>_homology_pairs.csv`:
MMseqs2, Foldseek and pairwise identity for every pair with evidence) and, for each split
pickle, reports how many libraries in each set have a homolog in each other set, the
strongest identity / TM-score / E-values between sets, and the composition of each set
(natural vs designed libraries, single and double mutation counts). A pair counts as
homologous under the same rule split_tsuboyama.py splits by (homology.homology_edge_mask).

Example:
    python split_overlap_report.py \
        --pairs ../data/splits_oct06_capped_homology_pairs.csv \
        --library_table ../data/splits_oct06_capped_library_assignment.csv \
        --splits current=/path/hyperopt_splits.pkl new=../data/splits_oct06_capped.pkl \
        --out_prefix ../data/visualizations/split_overlap
"""
import argparse
import json
import pickle
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from homology import homology_edge_mask

INTERNAL = ['train', 'val', 'test']
REFERENCE_SETS = ['train', 'val', 'test', 'external', 'functional']
FUNCTIONAL = {'GRB2', 'DLG4', 'EstA', 'Myo', 'GB1'}


def membership_from_split(split_file: str, pair_ids) -> Dict[str, str]:
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)
    mem = {}
    for s in INTERNAL:
        for c in splits.get(s, []):
            mem[str(c).replace('.pdb', '')] = s
    for pid in pair_ids:
        if pid.startswith('test_'):
            mem[pid] = 'external'
        elif pid in FUNCTIONAL:
            mem[pid] = 'functional'
    return mem


def best_partners(pairs: pd.DataFrame, mem: Dict[str, str]) -> pd.DataFrame:
    """Per internal library and reference set: strongest structural and sequence evidence."""
    p = pairs.copy()
    p['set_a'], p['set_b'] = p['a'].map(mem), p['b'].map(mem)
    p = p.dropna(subset=['set_a', 'set_b'])
    both = pd.concat([
        p.rename(columns={'a': 'lib', 'set_a': 'lib_set', 'b': 'partner', 'set_b': 'ref_set'}),
        p.rename(columns={'b': 'lib', 'set_b': 'lib_set', 'a': 'partner', 'set_a': 'ref_set'}),
    ])
    both = both.loc[both['lib_set'].isin(INTERNAL)]
    both = both.sort_values('struct_evalue', na_position='last')
    agg = both.groupby(['lib', 'lib_set', 'ref_set']).agg(
        struct_evalue=('struct_evalue', 'min'), struct_tm_max=('struct_tm_max', 'max'),
        seq_evalue=('seq_evalue', 'min'), seq_fident=('seq_fident', 'max'), identity=('identity', 'max'),
        is_homolog=('is_homolog', 'any'), best_struct_partner=('partner', 'first')).reset_index()
    return agg


def overlap_matrix(best: pd.DataFrame, mem: Dict[str, str]) -> pd.DataFrame:
    """Fraction of libraries in each internal set with >=1 homolog in each reference set."""
    n = pd.Series(mem).value_counts()
    hom = best.loc[best['is_homolog'] & (best['lib_set'] != best['ref_set'])]
    counts = hom.groupby(['lib_set', 'ref_set'])['lib'].nunique().unstack(fill_value=0)
    counts = counts.reindex(index=INTERNAL, columns=REFERENCE_SETS, fill_value=0)
    frac = counts.div(n.reindex(INTERNAL), axis=0)
    for s in INTERNAL:
        frac.loc[s, s] = np.nan
    return frac


def cross_set_maxima(pairs: pd.DataFrame, mem: Dict[str, str]) -> pd.DataFrame:
    """Per set pair, the extreme of each similarity measure taken independently.

    Each column is maximised (E-values minimised) over all pairs joining the two sets, so one
    row generally describes several different pairs. These are reporting bounds; what may
    straddle a boundary is decided by `homology_edge_mask`, and the `is_homolog` columns
    above report actual violations.
    """
    sets = INTERNAL + ['external', 'functional']
    p = pairs.assign(sa=pairs['a'].map(mem), sb=pairs['b'].map(mem)).dropna(subset=['sa', 'sb'])
    p = p.loc[(p['sa'] != p['sb']) & (p['sa'].isin(INTERNAL) | p['sb'].isin(INTERNAL))]
    key = p[['sa', 'sb']].apply(lambda r: ' vs '.join(sorted(r, key=sets.index)), axis=1)
    aggs = {c: f for c, f in [('identity', 'max'), ('struct_tm_max', 'max'), ('struct_tm_min', 'max'),
                              ('seq_evalue', 'min'), ('struct_evalue', 'min')] if c in p}
    return p.groupby(key).agg(aggs)


def composition(lib: pd.DataFrame, mem: Dict[str, str]) -> pd.DataFrame:
    l = lib.copy()
    l['set'] = l.index.map(mem)
    l = l.loc[l['set'].isin(INTERNAL)]
    out = l.groupby('set').agg(
        libraries=('natural', 'size'), natural_libs=('natural', 'sum'),
        libs_with_doubles=('n_double', lambda x: int((x > 0).sum())),
        n_single=('n_single', 'sum'), n_double=('n_double', 'sum')).reindex(INTERNAL)
    out['designed_libs'] = out['libraries'] - out['natural_libs']
    out['frac_designed'] = out['designed_libs'] / out['libraries']
    out['doubles_per_single'] = out['n_double'] / out['n_single']
    out['share_of_all_doubles'] = out['n_double'] / out['n_double'].sum()
    return out


def plot_report(results: Dict[str, dict], struct_evalue: float, out_prefix: str) -> None:
    names = list(results)
    colors = {'val→train': '#2a78d6', 'test→train': '#eb6834', 'train/val→external': '#1baf7a'}

    # 1. best cross-set structural partner per library
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        b = results[name]['best']
        groups = {
            'val→train': b[(b.lib_set == 'val') & (b.ref_set == 'train')],
            'test→train': b[(b.lib_set == 'test') & (b.ref_set == 'train')],
            'train/val→external': b[b.lib_set.isin(['train', 'val']) & b.ref_set.isin(['external', 'functional'])]
                .sort_values('struct_evalue').drop_duplicates('lib'),
        }
        for label, g in groups.items():
            y = -np.log10(g['struct_evalue'].clip(lower=1e-12))
            ax.scatter(g['struct_tm_max'], y, s=22, c=colors[label], alpha=0.75, edgecolors='white', linewidths=0.6, label=label)
        ax.axhline(-np.log10(struct_evalue), color='#52514e', lw=1, ls='--')
        ax.text(0.02, -np.log10(struct_evalue) + 0.15, f'homology cut (E ≤ {struct_evalue:g})', fontsize=8, color='#52514e', transform=ax.get_yaxis_transform())
        ax.set_title(name)
        ax.set_xlabel('TM-score of best partner (shorter-chain normalised)')
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel('−log10 Foldseek E-value of best partner')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_best_partner.png', dpi=200)
    plt.close(fig)

    # 2. overlap matrices
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        m = results[name]['overlap']
        ax.imshow(m.to_numpy(dtype=float), cmap='Blues', vmin=0, vmax=0.3)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iloc[i, j]
                ax.text(j, i, '—' if pd.isna(v) else f'{v:.0%}', ha='center', va='center', fontsize=9,
                        color='white' if (not pd.isna(v) and v > 0.18) else '#0b0b0b')
        ax.set_xticks(range(m.shape[1]), m.columns, rotation=30)
        ax.set_yticks(range(m.shape[0]), m.index)
        ax.set_title(f'{name}: libraries with a homolog in…', fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_overlap_matrix.png', dpi=200)
    plt.close(fig)

    # 3. composition
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        c = results[name]['composition']
        x = np.arange(len(INTERNAL)) + k * width
        axes[0].bar(x, c['frac_designed'], width * 0.92, label=name)
        axes[1].bar(x, c['doubles_per_single'], width * 0.92, label=name)
    for ax, title in zip(axes, ['Fraction of libraries that are designed', 'Double / single mutation ratio']):
        ax.set_xticks(np.arange(len(INTERNAL)) + width * (len(names) - 1) / 2, INTERNAL)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_composition.png', dpi=200)
    plt.close(fig)


def main(args):
    pairs = pd.read_csv(args.pairs)
    pairs['is_homolog'] = homology_edge_mask(pairs, args.seq_evalue, args.struct_evalue,
                                             args.max_identity, args.max_tm, args.tm_norm)
    for col in ['struct_evalue', 'struct_tm_max', 'seq_evalue', 'seq_fident']:
        if col not in pairs:
            pairs[col] = np.nan
    lib = pd.read_csv(args.library_table, index_col=0)
    pair_ids = set(pairs['a']) | set(pairs['b'])

    results, export = {}, {}
    for spec in args.splits:
        name, path = spec.split('=', 1)
        mem = membership_from_split(path, pair_ids)
        best = best_partners(pairs, mem)
        res = {'best': best, 'overlap': overlap_matrix(best, mem), 'composition': composition(lib, mem),
               'maxima': cross_set_maxima(pairs, mem)}
        results[name] = res
        leaks = best.loc[best['is_homolog'] & (best['lib_set'] != best['ref_set'])].sort_values('struct_evalue')
        leaks.to_csv(f'{args.out_prefix}_{name}_homolog_pairs.csv', index=False)
        print(f'\n=== {name} ({path}) ===')
        print(res['composition'].to_string())
        print(f'\nFraction of libraries with a homolog (seq E<={args.seq_evalue:g}, struct E<={args.struct_evalue:g}, '
              f'identity>{args.max_identity}, TM_{args.tm_norm}>{args.max_tm}) in:')
        print(res['overlap'].round(3).to_string())
        print('\nPer-measure extremes between sets (each column is a different pair):')
        print(res['maxima'].round(4).to_string())
        res['maxima'].to_csv(f'{args.out_prefix}_{name}_cross_set_maxima.csv')
        export[name] = {
            'composition': res['composition'].reset_index().to_dict(orient='records'),
            'overlap': res['overlap'].reset_index().rename(columns={'index': 'set'}).to_dict(orient='records'),
            'best': best.to_dict(orient='records'),
            'maxima': res['maxima'].reset_index().rename(columns={'index': 'sets'}).to_dict(orient='records'),
        }
    with open(f'{args.out_prefix}_data.json', 'w') as f:
        json.dump(export, f, default=lambda x: None if pd.isna(x) else x)
    plot_report(results, args.struct_evalue, args.out_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, help='homology_pairs.csv from split_tsuboyama.py')
    parser.add_argument('--library_table', required=True, help='<output>_library_assignment.csv from split_tsuboyama.py')
    parser.add_argument('--splits', nargs='+', required=True, help='name=path.pkl entries')
    parser.add_argument('--seq_evalue', type=float, default=1e-3)
    parser.add_argument('--struct_evalue', type=float, default=1e-3)
    parser.add_argument('--max_identity', type=float, default=0.40)
    parser.add_argument('--max_tm', type=float, default=0.70)
    parser.add_argument('--tm_norm', choices=['max', 'min'], default='max')
    parser.add_argument('--out_prefix', required=True)
    main(parser.parse_args())
