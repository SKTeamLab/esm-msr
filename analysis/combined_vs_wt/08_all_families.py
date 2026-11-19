#!/usr/bin/env python
"""Combined versus WT head inside every FINAL_* family (not only p0, p1, pw0): delta rho = rho(combined) - rho(WT head) of the same checkpoint, on the same variants.

Sets: the DMS benchmark (DLG4 binding, GRB2 binding, MYO; composite = mean of the three rho), the 10 external single-mutant sets (composite), PTMUL + PTMUL-D (composite), Homology-Restricted MegaScale test
(single mutants, doubles).  Same bootstrap as ``01_benchmark_delta.py`` (units: sites for the DMS sets, proteins for the others; one draw shared by the heads and families of a cell; B = 100).
Writes ``results/combined_minus_wt_all_families.csv`` and ``results/table_all_families.md``.

    python analysis/combined_vs_wt/08_all_families.py [--B 100]
"""
import argparse
import sys

import numpy as np
import pandas as pd

import cvw as C
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.final_runs import FAMILIES

ALL = [f for f in FAMILIES if '@' not in f]
KEYS = [f'{f}:{h}' for f in ALL for h in ('wt', 'comb')]


def rows_for(name, point, reps, B, rows):
    for fam in ALL:
        k1, k0 = f'{fam}:comb', f'{fam}:wt'
        if k1 not in point or k0 not in point:
            continue
        d = reps[k1] - reps[k0]
        lo, hi = ci(d)
        rows.append(dict(set=name, family=fam, rho_wt=point[k0], rho_comb=point[k1], delta=point[k1] - point[k0], lo=lo, hi=hi, p=max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / (B + 1))))


def composite(point, reps, names):
    keys = set.intersection(*[set(point[c]) for c in names])
    return {k: float(np.nanmean([point[c][k] for c in names])) for k in keys}, {k: np.nanmean([reps[c][k] for c in names], axis=0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=C.B_DEFAULT)
    B = ap.parse_args().B
    rows = []
    for group, names, kind, label in (('dms', C.DMS, 'site', 'DMS benchmark (3, composite)'), ('ext_singles', C.EXT_SINGLE, 'protein', 'External single-mutant sets (10, composite)'),
                                      ('ext_multi', C.EXT_MULTI, 'protein', 'PTMUL / PTMUL-D (composite)')):
        fr = {d: C.load_final_frame(d, families=ALL) for d in names}
        point, reps = bootstrap_rho(fr, KEYS, kind=kind, B=B, seed=16, per_protein=False)
        pt = {d: {k: v[0] for k, v in point[d].items()} for d in names}
        if group == 'dms':
            for d in names:
                rows_for(C.LABEL[d], pt[d], reps[d], B, rows)
        cp, cr = composite(pt, reps, names)
        rows_for(label, cp, cr, B, rows)
        print(f'  {group}', flush=True)
    for cid, mask_fn, label in (('single', lambda nm, use: use & (nm == 1), 'MegaScale test, single mutants'), ('multi', lambda nm, use: use & (nm > 1), 'MegaScale test, doubles')):
        fr = C.load_final_frame('tsuboyama_capped_test', families=ALL)
        nm, use = fr.spine['n_mut'].to_numpy(), fr.spine['use'].to_numpy()
        point, reps = bootstrap_rho({cid: fr}, KEYS, kind='protein', B=B, seed=16, per_protein=True, subsets={cid: mask_fn(nm, use)})
        rows_for(label, {k: v[0] for k, v in point[cid].items()}, reps[cid], B, rows)
    D = pd.DataFrame(rows)
    D.to_csv(C.OUT / 'combined_minus_wt_all_families.csv', index=False, float_format='%.5g')
    order = ['DMS benchmark (3, composite)', C.LABEL[C.DMS[0]], C.LABEL[C.DMS[1]], C.LABEL[C.DMS[2]], 'External single-mutant sets (10, composite)', 'PTMUL / PTMUL-D (composite)', 'MegaScale test, single mutants',
             'MegaScale test, doubles']
    short = {f: FAMILIES[f]['short'] for f in ALL}
    L = ['| set | ' + ' | '.join(short[f] for f in ALL) + ' |', '|---|' + '---|' * len(ALL)]
    for s in order:
        cells = []
        for f in ALL:
            r = D[(D.set == s) & (D.family == f)]
            cells.append('-' if not len(r) else f'{r.delta.iloc[0]:+.3f} [{r.lo.iloc[0]:+.3f}, {r.hi.iloc[0]:+.3f}]' + ('*' if (r.lo.iloc[0] > 0 or r.hi.iloc[0] < 0) else ''))
        L.append(f'| {s} | ' + ' | '.join(cells) + ' |')
    (C.OUT / 'table_all_families.md').write_text('combined - WT head within each family (* = interval excludes 0)\n\n' + '\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
