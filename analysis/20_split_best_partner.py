"""Best cross-set partner of every library, by sequence (left) and by structure (right): the evidence behind the homology rule of the Homology-Restricted MegaScale split.

For each validation / test library against the training libraries, and each training / validation library against the external benchmarks (the thermostability sets and the five functional DMS targets), the
strongest evidence over all partners (each measure maximised / minimised over the partners on its own, as in ``preprocessing/split_overlap_report.py``):

  left   x = BLOSUM62 local sequence identity (shorter-chain normalised, Smith-Waterman), y = -log10 of the smallest MMseqs2 E-value; a library with no MMseqs2 hit (E > 10) sits on the bottom line
  right  x = Foldseek TM-score (shorter-chain normalised), y = -log10 of the smallest Foldseek E-value; libraries with no hit are not drawn

Dashed lines are the four parts of the split's rule, two per search (MMseqs2: E <= 1e-3 and identity > 0.40; Foldseek: E <= 1e-3 and TM-score > 0.70): a pair above any of them could not straddle a split.  ``--split capped`` (the default) is the Homology-Restricted MegaScale split
(``data/splits_oct06_capped.pkl``) -> ``results/00_supplementary_figs/split_construction/split_best_partner_capped``; ``--split hyperopt`` the earlier hyperparameter-search split
(``data/hyperopt_splits.pkl``) -> ``results/for_reviewers/split_best_partner_hyperopt``.

Needs the pair table of ``preprocessing/split_tsuboyama.py`` (``homology_pairs.csv``; ``$HOMOLOGY_PAIRS_CSV``, default the salvaged Oct-06 run under ``~/playground/esm-msr-devel/analysis/salvaged_2026-10-16``).

    python analysis/20_split_best_partner.py [--split capped|hyperopt]
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'preprocessing'))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.registry import RESULTS  # noqa: E402

spec = importlib.util.spec_from_file_location('split_overlap_report', HERE.parent / 'preprocessing' / 'split_overlap_report.py')
sor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sor)

PAIRS = Path(os.environ.get('HOMOLOGY_PAIRS_CSV', Path.home() / 'playground' / 'esm-msr-devel' / 'analysis' / 'salvaged_2026-10-16' / 'structure-homology-splits-e5ca04' / 'data' / 'tsuboyama' / 'homology' / 'homology_pairs.csv'))
SPLITS = {'capped': ('splits_oct06_capped.pkl', out_dir('00_supplementary_figs') / 'split_construction', 'split_best_partner_capped'),
          'hyperopt': ('hyperopt_splits.pkl', RESULTS / 'for_reviewers', 'split_best_partner_hyperopt')}
COLORS = {'val → train': '#2a78d6', 'test → train': '#eb6834', 'train / val → external': '#1baf7a'}
E_CUT, ID_CUT, TM_CUT = 1e-3, 0.40, 0.70
NO_HIT = -1.0                                                  # -log10 10: the search reports hits down to E = 10


def groups_of(pairs, mem):
    p = pairs.copy()
    p['set_a'], p['set_b'] = p['a'].map(mem), p['b'].map(mem)
    p = p.dropna(subset=['set_a', 'set_b'])
    both = pd.concat([p.rename(columns={'a': 'lib', 'set_a': 'lib_set', 'b': 'partner', 'set_b': 'ref_set'}), p.rename(columns={'b': 'lib', 'set_b': 'lib_set', 'a': 'partner', 'set_a': 'ref_set'})])
    both = both[both['lib_set'].isin(sor.INTERNAL) & (both['lib_set'] != both['ref_set'])]
    out = {}
    for label, ls, rs in (('val → train', ['val'], ['train']), ('test → train', ['test'], ['train']), ('train / val → external', ['train', 'val'], ['external', 'functional'])):
        g = both[both['lib_set'].isin(ls) & both['ref_set'].isin(rs)]
        a = g.groupby('lib').agg(seq_evalue=('seq_evalue', 'min'), identity=('identity', 'max'), struct_evalue=('struct_evalue', 'min'), tm=('struct_tm_max', 'max'))
        libs = sorted(mem_libs(mem, ls))
        out[label] = a.reindex(libs)                          # a library with no pair at all has no evidence either way
    return out


def mem_libs(mem, sets):
    return [k for k, v in mem.items() if v in sets and not k.startswith('test_')]


def draw(groups, split):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for label, g in groups.items():
        c = COLORS[label]
        ys = (-np.log10(g['seq_evalue'].clip(lower=1e-12))).fillna(NO_HIT)
        a.scatter(g['identity'], ys, s=22, c=c, alpha=0.75, edgecolors='white', linewidths=0.6, label=f'{label} (n = {len(g)})')
        h = g.dropna(subset=['struct_evalue'])
        b.scatter(h['tm'], -np.log10(h['struct_evalue'].clip(lower=1e-12)), s=22, c=c, alpha=0.75, edgecolors='white', linewidths=0.6)
    for ax in (a, b):
        ax.axhline(-np.log10(E_CUT), color='#52514e', lw=1, ls='--')
        ax.grid(alpha=0.2)
        ax.axhline(NO_HIT, color='#bbbbbb', lw=0.6, zorder=0)
    a.axvline(ID_CUT, color='#52514e', lw=1, ls='--')
    b.axvline(TM_CUT, color='#52514e', lw=1, ls='--')
    a.text(0.01, -np.log10(E_CUT) + 0.12, 'E-value ≤ 10⁻³', fontsize=8, color='#52514e', ha='left', transform=a.get_yaxis_transform())
    b.text(0.01, -np.log10(E_CUT) + 0.12, 'E-value ≤ 10⁻³', fontsize=8, color='#52514e', ha='left', transform=b.get_yaxis_transform())
    a.text(ID_CUT + 0.003, 0.99, 'identity > 0.40', fontsize=8, color='#52514e', rotation=90, va='top', transform=a.get_xaxis_transform())
    b.text(TM_CUT + 0.003, 0.99, 'TM-score > 0.70', fontsize=8, color='#52514e', rotation=90, va='top', transform=b.get_xaxis_transform())
    a.set_xlabel('Sequence identity of best partner (BLOSUM62 local, shorter chain)')
    b.set_xlabel('TM-score of best partner (shorter-chain normalised)')
    a.set_ylabel('−log10 E-value of best partner')
    a.text(0.0, 1.03, 'MMseqs2 E-value and identity (criteria 1 and 2)', transform=a.transAxes, fontsize=9.5, fontweight='bold', color='#444444')
    b.text(0.0, 1.03, 'Foldseek E-value and TM-score (criteria 3 and 4)', transform=b.transAxes, fontsize=9.5, fontweight='bold', color='#444444')
    a.text(-0.12, 1.07, 'a', transform=a.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')
    b.text(-0.04, 1.07, 'b', transform=b.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')
    top = int(np.ceil(max(a.get_ylim()[1], b.get_ylim()[1], 4.4) - 0.3))
    a.set_ylim(NO_HIT - 0.3, top + 0.3)
    ticks = list(range(0, top + 1))
    a.set_yticks([-1] + ticks)
    a.set_yticklabels(['no hit'] + [str(t) for t in ticks])
    a.set_xlim(a.get_xlim()[0], max(a.get_xlim()[1], ID_CUT + 0.02))
    b.set_xlim(b.get_xlim()[0], max(b.get_xlim()[1], TM_CUT + 0.02))
    a.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=list(SPLITS), default='capped')
    a = ap.parse_args()
    pkl, out, stem = SPLITS[a.split]
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS)
    mem = sor.membership_from_split(str(HERE.parent / 'data' / pkl), set(pairs['a']) | set(pairs['b']))
    groups = groups_of(pairs, mem)
    fig = draw(groups, a.split)
    fig.savefig(out / f'{stem}.png', dpi=300)
    fig.savefig(out / f'{stem}.pdf')
    rows = pd.concat({k: v.reset_index() for k, v in groups.items()}, names=['comparison']).reset_index(level=0)
    rows.to_csv(out / f'{stem}.csv', index=False, float_format='%.5g')
    for k, g in groups.items():
        print(f'{a.split} {k}: {len(g)} libraries, {int(g.seq_evalue.notna().sum())} with a MMseqs2 hit, {int(g.struct_evalue.notna().sum())} with a Foldseek hit; '
              f'above the cuts: E {int((g.seq_evalue <= E_CUT).sum())} / identity {int((g.identity > ID_CUT).sum())} (seq), E {int((g.struct_evalue <= E_CUT).sum())} / TM {int((g.tm > TM_CUT).sum())} (struct)')
    print('wrote', out / f'{stem}.png')


if __name__ == '__main__':
    main()
