"""Supplementary figure: the four-part homology filter on the held-out benchmarks.

For ProteinGym assays, Domainome domains, MGnify scaffolds and Rocklin designs, how many proteins are homologous to the training libraries under the rule that built the Homology-Restricted MegaScale split
(any one of: 1 MMseqs2 E-value <= 1e-3, 2 identity > 0.40, 3 Foldseek E-value <= 1e-3, 4 Foldseek TM-score > 0.70; ``leak_audit/21_tsuboyama_rule.py``, ``reference/tsuboyama_rule_flags.csv``), and which of
the four criteria catch them (a protein can meet several).  Left: share of each benchmark removed, in total and by criterion; right: how many proteins only one criterion catches, which shows what each part adds.

Output ``results/00_supplementary_figs/holdout_filter.{png,pdf}`` and ``holdout_filter_counts.csv``.

    python analysis/21b_holdout_filter_figure.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.common import out_dir
from msr_analysis.heldout import flags

OUT = out_dir('00_supplementary_figs')
SETS = [('proteingym', 'ProteinGym\nassays'), ('domainome', 'Domainome\ndomains'), ('mgnify', 'MGnify\nscaffolds'), ('rocklin_denovo', 'Rocklin\ndesigns')]
CRIT = [('f_seq_e', 'MMseqs2 E-value ≤ 1e-3', '#2a78d6'), ('f_identity', 'Identity > 0.40', '#7fb3e8'), ('f_struct_e', 'Foldseek E-value ≤ 1e-3', '#d35400'), ('f_tm', 'Foldseek TM-score > 0.70', '#f0a36b')]


def main():
    f = flags()
    rows = []
    for ds, _ in SETS:
        g = f[f.dataset == ds]
        n = len(g)
        r = dict(dataset=ds, n=n, flagged=int(g.homologous.sum()), kept=n - int(g.homologous.sum()))
        for c, _, _ in CRIT:
            r[c] = int(g[c].sum())
            r[c + '_only'] = int((g[c] & (g[[k for k, _, _ in CRIT]].sum(axis=1) == 1)).sum())
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'holdout_filter_counts.csv', index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.arange(len(SETS))
    w = 0.15
    cols = [('homologous', 'Removed (any criterion)', '#333333')] + [(c, lab, col) for c, lab, col in CRIT]
    for j, (c, lab, col) in enumerate(cols):
        v = [100 * (T.loc[i, 'flagged'] if c == 'homologous' else T.loc[i, c]) / T.loc[i, 'n'] for i in range(len(SETS))]
        a.bar(x + (j - 2) * w, v, w, color=col, edgecolor='black', lw=0.6, label=lab, zorder=3)
        for xi, vi in zip(x + (j - 2) * w, v):
            a.text(xi, vi + 0.4, f'{vi:.0f}' if vi >= 1 else f'{vi:.1f}', ha='center', va='bottom', fontsize=7.6, zorder=4)
    a.set_xticks(x)
    a.set_xticklabels([f'{lab}\n(n = {int(T.loc[i, "n"]):,})' for i, (_, lab) in enumerate(SETS)], fontsize=9.5)
    a.set_ylabel('% of the benchmark homologous to the training set')
    a.grid(axis='y', color='#dddddd', lw=0.6, zorder=0)
    a.set_axisbelow(True)
    a.set_ylim(0, 1.3 * a.get_ylim()[1])
    a.legend(fontsize=8.2, frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left')
    for s_ in ('top', 'right'):
        a.spines[s_].set_visible(False)
    bottom = np.zeros(len(SETS))
    for c, lab, col in CRIT:
        v = np.array([T.loc[i, c + '_only'] for i in range(len(SETS))], float)
        share = v / T['flagged'].to_numpy() * 100
        b.bar(x, share, 0.6, bottom=bottom, color=col, edgecolor='black', lw=0.6, label=lab, zorder=3)
        for xi, bi, si in zip(x, bottom, share):
            if si >= 6:
                b.text(xi, bi + si / 2, f'{si:.0f}%', ha='center', va='center', fontsize=8, zorder=4)
        bottom += share
    b.bar(x, 100 - bottom, 0.6, bottom=bottom, color='#bbbbbb', edgecolor='black', lw=0.6, label='two or more criteria', zorder=3)
    for xi, bi, si in zip(x, bottom, 100 - bottom):
        if si >= 6:
            b.text(xi, bi + si / 2, f'{si:.0f}%', ha='center', va='center', fontsize=8, zorder=4)
    b.set_xticks(x)
    b.set_xticklabels([lab for _, lab in SETS], fontsize=9.5)
    b.set_ylabel('% of the removed proteins')
    b.set_ylim(0, 100)
    b.legend(fontsize=8.2, frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3)
    for s_ in ('top', 'right'):
        b.spines[s_].set_visible(False)
    a.text(-0.1, 1.04, 'a', transform=a.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='right')
    b.text(-0.12, 1.04, 'b', transform=b.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='right')
    a.set_title('Caught by each criterion', fontsize=10.5, loc='left')
    b.set_title('Caught by one criterion only', fontsize=10.5, loc='left')
    fig.tight_layout()
    fig.savefig(OUT / 'holdout_filter.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'holdout_filter.pdf', bbox_inches='tight')
    print(T.to_string(index=False))


if __name__ == '__main__':
    main()
