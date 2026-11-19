"""Heat maps of the maximum sequence identity and the maximum TM-score between the protein sets of the split (``leak_audit/22_set_overlap_matrix.py``).

Sets: Homology-Restricted MegaScale train / val / test, the nine thermostability sets, and the held-out versions of ProteinGym, Domainome, MGnify and the Rocklin designs (the ones in the summary figure).
Cell = the largest value over all pairs of proteins of the two sets (identity: MMseqs2 identity over the shorter sequence, i.e. the whole-protein normalisation of the split's own identity; TM-score: Foldseek, shorter-chain normalised).  Only the lower triangle is drawn (the matrices are
symmetric) and the diagonal is blank; a grey cell has no MMseqs2 hit (E <= 10) between its two sets.

Reads ``reference/split_set_overlap_{identity,tm}.csv``; writes ``results/00_supplementary_figs/split_construction/split_max_{identity,tm}_heatmap.{png,pdf}``.

    python analysis/20b_split_overlap_heatmaps.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.common import out_dir

HERE = Path(__file__).resolve().parent
OUT = out_dir('00_supplementary_figs') / 'split_construction'
LABEL = {'train': 'Train', 'val': 'Validation', 'test': 'Test', 'ssym': 'Ssym', 'ptmul': 'PTMUL', 's461': 'S461', 'k2369': 'K2369', 'q3421': 'Q3421', 's571': 'S571', 's783': 'S783', 's2648': 'S2648',
         's8754': 'S8754', 'ProteinGym': 'ProteinGym', 'Domainome': 'Domainome', 'MGnify': 'MGnify', 'Rocklin designs': 'Rocklin designs'}
GROUPS = [3, 9, 4]                                   # internal sets | thermostability sets | held-out large benchmarks
CUT = {'identity': 0.40, 'identity_sig': 0.40, 'tm': 0.70}
STEM = {'identity': 'split_max_identity_heatmap', 'identity_sig': 'split_max_identity_significant_heatmap', 'tm': 'split_max_tm_heatmap'}
TITLE = {'identity': 'Max sequence identity (MMseqs2, normalised by the shorter sequence)', 'identity_sig': 'Max sequence identity (MMseqs2, hits with E ≤ 1e-3)', 'tm': 'Max TM-score (Foldseek, shorter-chain normalised)'}


def draw(m, kind, stem):
    sets = list(m.index)
    n = len(sets)
    fig, ax = plt.subplots(figsize=(10.4, 9.0))
    v = m.to_numpy(float).copy()
    v[np.triu_indices(n)] = np.nan                                   # lower triangle only; the diagonal is blank
    v = np.ma.masked_invalid(v)
    cmap = plt.get_cmap('Reds').copy()
    im = ax.imshow(v, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    for i in range(n):
        for j in range(i):
            x = m.iloc[i, j]
            if np.isfinite(x):
                ax.text(j, i, f'{x:.2f}', ha='center', va='center', fontsize=7.6, color='white' if x > 0.55 else '#222222')
            else:                                                  # identity: no significant MMseqs2 hit (E <= 1e-3) between the two sets; TM-score: not computed
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#e4e4e4', ec='none'))
                ax.text(j, i, 'n.s.' if kind == 'identity_sig' else '–', ha='center', va='center', fontsize=7.4, color='#777777')
    edges = np.cumsum(GROUPS)[:-1] - 0.5
    for e in edges:                                                # group borders, drawn inside the triangle only
        ax.plot([-0.5, e], [e, e], color='black', lw=1.1)
        ax.plot([e, e], [e, n - 0.5], color='black', lw=1.1)
    ax.set_xlim(-0.5, n - 1.5)                                       # a lower triangle has no cells in the first row and the last column
    ax.set_ylim(n - 0.5, 0.5)
    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(1, n))
    ax.set_xticklabels([LABEL[s] for s in sets[:-1]], rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
    ax.set_yticklabels([LABEL[s] for s in sets[1:]], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(TITLE[kind], fontsize=9.5)
    cb.ax.axhline(CUT[kind], color='black', lw=1.2)                 # the rule of the split: an edge above it (identity > 0.40, TM-score > 0.70)
    ticks = sorted(set(np.round(np.arange(0, 1.01, 0.2), 2)) | {CUT[kind]})
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:.2f}'.rstrip('0').rstrip('.') if t != CUT[kind] else f'{t:.2f} (rule)' for t in ticks])
    fig.tight_layout()
    fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / f'{stem}.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in ('identity', 'tm'):
        m = pd.read_csv(HERE / 'reference' / f'split_set_overlap_{kind}.csv', index_col=0)
        draw(m, kind, STEM[kind])


if __name__ == '__main__':
    main()
