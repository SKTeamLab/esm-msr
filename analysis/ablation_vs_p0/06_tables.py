#!/usr/bin/env python
"""Tables of the ablation-versus-p0 comparison.

  results/table_composites.md   delta rho [95% interval] of every ablation on the headline cells (the thermostability composites, Homology-Restricted MegaScale test, Domainome, the DMS composite,
                                ProteinGym average), WT head and combined prediction
  results/table_counts.md       per ablation and head: how many set cells have an interval above 0 / below 0, and the thermostability cells alone
  results/table_significant.md  every cell whose interval excludes 0

    python analysis/ablation_vs_p0/06_tables.py
"""
import sys

import numpy as np
import pandas as pd

import abl as A

HEADLINE = [('ext_singles', 'ext_singles_composite', 'all', 'Thermostability singles (10 sets, composite)'), ('ext_multi', 'ext_multi_composite', 'all', 'PTMUL / PTMUL-D (composite)'),
            ('tsuboyama', A.C.LABEL['tsuboyama_capped_test'], 'all', 'MegaScale test, all'), ('tsuboyama', A.C.LABEL['tsuboyama_capped_test'], 'doubles', 'MegaScale test, doubles'),
            ('tsuboyama', A.C.LABEL['tsuboyama_capped_test'], 'epistasis (dddG)', 'MegaScale test, epistasis'), ('domainome', 'Domainome', 'all', 'Domainome'),
            ('dms', 'dms_composite', 'all', 'DMS benchmark (3, composite)'), ('proteingym', 'PG avg', 'all', 'ProteinGym average (held-out assays)')]


def f(r):
    return '-' if r is None else f'{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]' + ('*' if (r.lo > 0 or r.hi < 0) else '')


def main():
    D = A.load_all()
    L = []
    for head in A.HEADS:
        L += [f'**{"WT head" if head == "wt" else "Combined prediction"}** (delta rho = family - p0; * = interval excludes 0)', '', '| set | ' + ' | '.join(A.SHORT[fam] for fam in A.ABL) + ' |', '|---|' + '---|' * len(A.ABL)]
        for g, s, sub, lab in HEADLINE:
            cells = []
            for fam in A.ABL:
                hh = 'epi' if sub.startswith('epistasis') else head
                r = D[(D.group == g) & (D.set == s) & (D.subset == sub) & (D.family == fam) & (D['head'] == hh)]
                if sub.startswith('epistasis') and head == 'wt':
                    r = r.iloc[0:0]
                cells.append(f(r.iloc[0]) if len(r) else '-')
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
        L.append('')
    (A.OUT / 'table_composites.md').write_text('\n'.join(L))
    cells = D[~D.set.str.contains('composite|PG avg')]
    C_ = ['| ablation | head | cells | interval above 0 | interval below 0 | thermostability cells | above 0 | below 0 |', '|---|---|---|---|---|---|---|---|']
    for fam in A.ABL:
        for head in A.HEADS + ['epi']:
            c = cells[(cells.family == fam) & (cells['head'] == head)]
            if not len(c):
                continue
            t = c[c.group.isin(['ext_singles', 'ext_multi'])]
            C_.append(f'| {A.SHORT[fam]} | {head} | {len(c)} | {int((c.lo > 0).sum())} | {int((c.hi < 0).sum())} | {len(t) or "-"} | {int((t.lo > 0).sum()) if len(t) else "-"} | {int((t.hi < 0).sum()) if len(t) else "-"} |')
    (A.OUT / 'table_counts.md').write_text('\n'.join(C_) + '\n')
    S = ['| group | set | subset | ablation | head | delta rho | 95% interval | p |', '|---|---|---|---|---|---|---|---|']
    sig = D[((D.lo > 0) | (D.hi < 0)) & ~D.set.str.contains('composite')].sort_values(['group', 'set', 'family', 'head'])
    for r in sig.itertuples():
        S.append(f'| {r.group} | {r.set} | {r.subset} | {A.SHORT[r.family]} | {r.head} | {r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p:.2g} |')
    (A.OUT / 'table_significant.md').write_text('\n'.join(S) + '\n')
    print('\n'.join(L))
    print('\n'.join(C_))


if __name__ == '__main__':
    sys.exit(main())
