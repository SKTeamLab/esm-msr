#!/usr/bin/env python
"""Combined versus WT head on the benchmark sets: paired rho(combined) - rho(WT) per set and variant class, for p0, p1 and pw0.

For every set and every subset of its measured variants (all; single mutants; multi-mutants; by number of mutations) and every family, the Spearman rho with the measurement of the WT head, the MT head
and the combined prediction (1/2 (MT + WT)), all read from the same prediction file (sigma = 1, the family's native masking, seeds 1-3), and the paired differences with a bootstrap interval.

  external single-mutant sets (10)           pooled rho; the bootstrap resamples proteins
  PTMUL, PTMUL-D                            pooled rho over the multi-mutants; proteins
  DLG4 binding, GRB2 binding, MYO (DMS)      pooled rho; the bootstrap resamples sites (first mutated position)
  Domainome, Homology-Restricted MegaScale   mean over proteins of the per-protein rho; the bootstrap resamples proteins (exact, from the per-protein rho)
  epistasis                                  rho of the predicted epistasis comb(AB) - WT(A) - WT(B) with the measured dddG, Tsuboyama doubles with both singles measured (pooled; proteins)

The bootstrap draws are shared by the heads and families of a cell (one draw per replicate, ``msr_analysis.boot.bootstrap_rho``), so the differences are paired.  B = 100 (draft).
Writes ``results/benchmark_delta.csv`` (one row per cell x family), ``results/epistasis_rho.csv``.

    python analysis/combined_vs_wt/01_benchmark_delta.py [--B 100]
"""
import argparse
import sys

import numpy as np
import pandas as pd

import cvw as C
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import epi_rows, subframe
from msr_analysis.registry import PER_PROTEIN

KEYS = [f'{f}:{h}' for f in C.FAMS for h in C.HEADS]


def cell_list(ds, sp):
    """[(cell id, label of the subset, boolean mask over the spine rows)] of one dataset."""
    nm = sp['n_mut'].to_numpy()
    use = sp['use'].to_numpy()
    top = int(nm[use].max())
    out = [('all', 'all', use)]
    if (nm[use] == 1).any() and top > 1:
        out += [('singles', 'single mutants', use & (nm == 1)), ('multi', 'multi-mutants', use & (nm > 1))]
    if top > 1:
        edges = [(1, 1), (2, 2), (3, 3), (4, 4)] if top <= 4 else [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
        if ds in ('ptmul',):
            edges = [(2, 2), (3, 3), (4, 4), (5, 99)]
        for lo, hi in edges:
            if lo > top:
                continue
            m = use & (nm >= lo) & (nm <= hi)
            if m.sum() >= 50:
                out.append((f'n{lo}' if lo == hi else f'n{lo}+', f'{lo} mutation{"s" if lo > 1 else ""}' if lo == hi else f'{lo}+ mutations', m))
    return out


def run(B):
    rows, epi_rows_out = [], []
    sets = C.EXT_SINGLE + C.EXT_MULTI + C.DMS + ['domainome'] + C.TSU
    for ds in sets:
        fr = C.load_final_frame(ds, families=C.FAMS)
        kind = 'site' if ds in C.DMS else 'protein'
        pp = ds in PER_PROTEIN
        for cid, clabel, mask in cell_list(ds, fr.spine):
            keys = [k for k in KEYS if k in fr.preds]
            point, reps = bootstrap_rho({cid: fr}, keys, kind=kind, B=B, seed=16, per_protein=pp, subsets={cid: mask})
            pt, rp = point[cid], reps[cid]
            for fam in C.FAMS:
                if not all(f'{fam}:{h}' in pt for h in C.HEADS):
                    continue
                r = {h: pt[f'{fam}:{h}'][0] for h in C.HEADS}
                row = dict(dataset=ds, cell=cid, subset=clabel, family=fam, n=pt[f'{fam}:wt'][2], n_proteins=pt[f'{fam}:wt'][3], rho_wt=r['wt'], rho_mt=r['mt'], rho_comb=r['comb'])
                for a, b, nm_ in (('comb', 'wt', 'd_comb'), ('mt', 'wt', 'd_mt')):
                    d = rp[f'{fam}:{a}'] - rp[f'{fam}:{b}']
                    lo, hi = ci(d)
                    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
                    row.update({nm_: r[a] - r[b], f'{nm_}_lo': lo, f'{nm_}_hi': hi, f'{nm_}_p': max(p, 1 / (B + 1))})
                rows.append(row)
            print(f'  {ds} / {cid}: n = {int(mask.sum())}', flush=True)
        if ds in C.TSU:
            sp = fr.spine
            m = epi_rows(fr) & sp['wt_backbone'].to_numpy()
            ekeys = [f'{f}:epi' for f in C.FAMS if f'{f}:epi' in fr.preds]
            efr = subframe(fr, m, y=sp['dddG_ML'].to_numpy(dtype=float), keys=ekeys)
            efr.notes = fr.notes
            point, reps = bootstrap_rho({'epi': efr}, ekeys, kind='protein', B=B, seed=16, per_protein=False)
            for k in ekeys:
                lo, hi = ci(reps['epi'][k])
                epi_rows_out.append(dict(dataset=ds, family=k.split(':')[0], n=point['epi'][k][2], rho_epi=point['epi'][k][0], lo=lo, hi=hi))
    D = pd.DataFrame(rows)
    D.to_csv(C.OUT / 'benchmark_delta.csv', index=False, float_format='%.4f')
    E = pd.DataFrame(epi_rows_out)
    E.to_csv(C.OUT / 'epistasis_rho.csv', index=False, float_format='%.4f')
    return D, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=C.B_DRAFT)
    a = ap.parse_args()
    D, E = run(a.B)
    pd.set_option('display.width', 250, 'display.max_rows', 400)
    print(D.query("cell in ('all','singles','multi')").pivot_table(index=['dataset', 'cell'], columns='family', values='d_comb').round(3).to_string())
    print(E.round(3).to_string())


if __name__ == '__main__':
    sys.exit(main())
