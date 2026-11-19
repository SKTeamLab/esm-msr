#!/usr/bin/env python
"""Tables of the combined-versus-WT comparison.

  results/table_sets.md     one row per set and variant class: rho of the WT head and of the combined prediction (p0) and delta rho = rho(combined) - rho(WT head) with its 95% interval for p0, p1, pw0
  results/table_groups.md   delta rho averaged over the sets of a group (equal weight per set), and the ProteinGym categories (mean over assays, bootstrap p of ``04_fig_proteingym.py``)
  results/table_nmut.md     delta rho by number of mutations

    python analysis/combined_vs_wt/07_tables.py
"""
import sys

import numpy as np
import pandas as pd

import cvw as C

GROUPS = [('Stability / abundance, single mutants', [('tsuboyama_capped_val', 'singles'), ('tsuboyama_capped_test', 'singles'), ('domainome', 'all')] + [(d, 'all') for d in C.EXT_SINGLE]),
          ('Homology-Restricted MegaScale, single mutants', [('tsuboyama_capped_val', 'singles'), ('tsuboyama_capped_test', 'singles')]),
          ('External single-mutant sets (10)', [(d, 'all') for d in C.EXT_SINGLE]),
          ('Stability, multi-mutants (MegaScale doubles, PTMUL, PTMUL-D)', [('tsuboyama_capped_val', 'multi'), ('tsuboyama_capped_test', 'multi'), ('ptmul', 'all'), ('ptmuld', 'all')]),
          ('Homology-Restricted MegaScale, doubles', [('tsuboyama_capped_val', 'multi'), ('tsuboyama_capped_test', 'multi')]),
          ('DMS benchmark sets (3, all variants)', [(d, 'all') for d in C.DMS]),
          ('DMS sets, single mutants only', [(d, 'singles') for d in C.DMS]),
          ('DMS sets, multi-mutants only', [(d, 'multi') for d in C.DMS])]


def f(d, lo, hi):
    return f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def main():
    D = pd.read_csv(C.OUT / 'benchmark_delta.csv')
    PG = pd.read_csv(C.OUT / 'pg_delta_pvalues.csv')
    # per set
    L = ['| set | variants | n | rho WT-view | rho Dual-view | delta ESM-MSR | delta p1 | delta pw0 |', '|---|---|---|---|---|---|---|---|']
    sets = C.EXT_SINGLE + C.EXT_MULTI + ['domainome'] + C.TSU + C.DMS
    for ds in sets:
        for cid in [c for c in ('all', 'singles', 'multi') if ((D.dataset == ds) & (D.cell == c)).any()]:
            r = {fam: D[(D.dataset == ds) & (D.cell == cid) & (D.family == fam)] for fam in C.FAMS}
            p0 = r['FINAL_p0'].iloc[0]
            cells = [f(*r[fam].iloc[0][['d_comb', 'd_comb_lo', 'd_comb_hi']]) if len(r[fam]) else '-' for fam in C.FAMS]
            L.append(f'| {C.LABEL[ds]} | {p0.subset} | {int(p0.n):,} | {p0.rho_wt:.3f} | {p0.rho_comb:.3f} | ' + ' | '.join(cells) + ' |')
    (C.OUT / 'table_sets.md').write_text('\n'.join(L) + '\n')
    # groups
    G = ['| group | sets | delta p0 | delta p1 | delta pw0 |', '|---|---|---|---|---|']
    for name, items in GROUPS:
        vals = {fam: [D[(D.dataset == ds) & (D.cell == c) & (D.family == fam)].d_comb.iloc[0] for ds, c in items if ((D.dataset == ds) & (D.cell == c) & (D.family == fam)).any()] for fam in C.FAMS}
        n = len(vals['FINAL_p0'])
        cells = [f'{np.mean(vals[fam]):+.3f}' if len(vals[fam]) == n else '-' for fam in C.FAMS]
        G.append(f'| {name} | {n} | ' + ' | '.join(cells) + ' |')
    G += ['', '| ProteinGym group | assays | delta p0 | p (Holm) | delta p1 | p (Holm) |', '|---|---|---|---|---|---|']
    for row, gid in [(r.row, r.group) for r in PG[PG.family == 'FINAL_p0'].itertuples()]:
        a = PG[(PG.row == row) & (PG.group == gid)].set_index('family')
        lab = C.CAT_LABEL.get(gid, gid.replace('_', ', ').replace('Other', 'Other categories'))
        pf = lambda p: C.fmt_p(p) if np.isfinite(p) else '-'
        G.append(f'| {lab}{" (all variants)" if row == "top" else ""} | {int(a.loc["FINAL_p0", "n_assays"])} | {a.loc["FINAL_p0", "mean_delta"]:+.3f} | {pf(a.loc["FINAL_p0", "p_holm"])} | '
                 f'{a.loc["FINAL_p1", "mean_delta"]:+.3f} | {pf(a.loc["FINAL_p1", "p_holm"])} |')
    (C.OUT / 'table_groups.md').write_text('\n'.join(G) + '\n')
    # by number of mutations
    N = ['| set | mutations | n | delta p0 | delta p1 | delta pw0 |', '|---|---|---|---|---|---|']
    for ds in ['tsuboyama_capped_val', 'tsuboyama_capped_test', 'ptmul'] + C.DMS:
        for cid in sorted(D[(D.dataset == ds) & D.cell.str.startswith('n')].cell.unique(), key=lambda c: int(c[1])):
            r = {fam: D[(D.dataset == ds) & (D.cell == cid) & (D.family == fam)] for fam in C.FAMS}
            p0 = r['FINAL_p0'].iloc[0]
            N.append(f'| {C.LABEL[ds]} | {cid[1:]} | {int(p0.n):,} | ' + ' | '.join(f(*r[fam].iloc[0][['d_comb', 'd_comb_lo', 'd_comb_hi']]) if len(r[fam]) else '-' for fam in C.FAMS) + ' |')
    (C.OUT / 'table_nmut.md').write_text('\n'.join(N) + '\n')
    print('\n'.join(G))


if __name__ == '__main__':
    sys.exit(main())
