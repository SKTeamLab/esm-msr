#!/usr/bin/env python
"""Every FINAL_* ablation against FINAL_p0, paired, on every benchmark set: delta rho = rho(family) - rho(p0) under the same readout (WT head; combined 1/2 (MT + WT)).

Per set and variant class, and family x head: the seed-mean Spearman rho of the family and of p0, the difference and its 95% bootstrap interval and two-sided p.  The bootstrap resamples the same units for
p0 and for every family of a cell (one draw per replicate, ``msr_analysis.boot.bootstrap_rho``; the triplicate seeds are resampled too), so the interval is that of the paired difference.  FINAL_p0_mega
has seeds 1-2 on the benchmark sets (p0 has 1-3), so its pairing with p0 is approximate; it is not scored on the Tsuboyama sets (it trains on a different split).  Units: proteins (the external sets,
PTMUL), sites (DLG4 binding, GRB2 binding, MYO: first mutated position), the per-protein mean over proteins (Homology-Restricted MegaScale, Domainome).  B = 100 (draft).

  ext_singles    the 10 external single-mutant stability / thermostability sets and their composite (mean of the 10 rho; one protein draw shared by the 10 sets)
  ext_multi      PTMUL and PTMUL-D and their composite
  tsuboyama      Homology-Restricted MegaScale val and test: all variants, single mutants, doubles; the predicted epistasis comb(AB) - WT(A) - WT(B) against dddG (``epi`` head, combined only)
  domainome      the 522 Domainome domains
  dms            DLG4 binding, GRB2 binding, MYO

Writes ``results/paired_vs_p0.csv``; ProteinGym is ``02_pg_vs_p0.py``.

    python analysis/ablation_vs_p0/01_paired_vs_p0.py [--B 100]
"""
import argparse
import sys

import numpy as np
import pandas as pd

import abl as A
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import epi_rows, subframe
from msr_analysis.registry import PER_PROTEIN

ALL = [A.REF] + A.ABL
KEYS = [f'{f}:{h}' for f in ALL for h in A.HEADS]


def emit(rows, group, name, subset, point, reps, B, heads=A.HEADS, n=None):
    """Rows of one cell: ``point[key]`` the seed-mean rho, ``reps[key]`` its B bootstrap replicates."""
    for fam in A.ABL:
        for h in heads:
            k, k0 = f'{fam}:{h}', f'{A.REF}:{h}'
            if k not in point or k0 not in point:
                continue
            d = reps[k] - reps[k0]
            lo, hi = ci(d)
            p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
            rows.append(dict(group=group, set=name, subset=subset, family=fam, head=h, rho_family=point[k], rho_p0=point[k0], delta=point[k] - point[k0], lo=lo, hi=hi, p=max(p, 1 / (B + 1)), n=n))


def run_group(frames, keys, kind, pp, B, subsets=None):
    point, reps = bootstrap_rho(frames, keys, kind=kind, B=B, seed=16, per_protein=pp, subsets=subsets)
    return {c: {k: v[0] for k, v in point[c].items()} for c in point}, reps


def composite(point, reps, names):
    keys = set.intersection(*[set(point[c]) for c in names])
    return {k: float(np.nanmean([point[c][k] for c in names])) for k in keys}, {k: np.nanmean([reps[c][k] for c in names], axis=0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=A.C.B_DRAFT)
    B = ap.parse_args().B
    rows = []
    # external sets: composites share one protein draw
    for group, names in (('ext_singles', A.C.EXT_SINGLE), ('ext_multi', A.C.EXT_MULTI)):
        fr = {d: A.C.load_final_frame(d, families=ALL) for d in names}
        point, reps = run_group(fr, KEYS, 'protein', False, B)
        for d in names:
            emit(rows, group, A.C.LABEL[d], 'all', point[d], reps[d], B, n=int(fr[d].spine['use'].sum()))
        cp, cr = composite(point, reps, names)
        emit(rows, group, f'{group}_composite', 'all', cp, cr, B)
        print(f'  {group}: {len(names)} sets', flush=True)
    # DMS benchmark: sites
    fr = {d: A.C.load_final_frame(d, families=ALL) for d in A.C.DMS}
    point, reps = run_group(fr, KEYS, 'site', False, B)
    for d in A.C.DMS:
        emit(rows, 'dms', A.C.LABEL[d], 'all', point[d], reps[d], B, n=int(fr[d].spine['use'].sum()))
    cp, cr = composite(point, reps, A.C.DMS)
    emit(rows, 'dms', 'dms_composite', 'all', cp, cr, B)
    print('  dms', flush=True)
    # Homology-Restricted MegaScale: per-protein mean
    for ds in A.C.TSU:
        fr = A.C.load_final_frame(ds, families=ALL)
        sp = fr.spine
        nm, use = sp['n_mut'].to_numpy(), sp['use'].to_numpy()
        for cid, mask in (('all', use), ('single mutants', use & (nm == 1)), ('doubles', use & (nm > 1))):
            point, reps = run_group({cid: fr}, KEYS, 'protein', True, B, subsets={cid: mask})
            emit(rows, 'tsuboyama', A.C.LABEL[ds], cid, point[cid], reps[cid], B, n=int(mask.sum()))
        m = epi_rows(fr) & sp['wt_backbone'].to_numpy()
        ekeys = [f'{f}:epi' for f in ALL if f'{f}:epi' in fr.preds]
        efr = subframe(fr, m, y=sp['dddG_ML'].to_numpy(dtype=float), keys=ekeys)
        efr.notes = fr.notes
        point, reps = run_group({'epi': efr}, ekeys, 'protein', False, B)
        emit(rows, 'tsuboyama', A.C.LABEL[ds], 'epistasis (dddG)', point['epi'], reps['epi'], B, heads=['epi'], n=int(m.sum()))
        print(f'  {ds}', flush=True)
    # Domainome
    fr = A.C.load_final_frame('domainome', families=ALL)
    point, reps = run_group({'domainome': fr}, [k for k in KEYS if k in fr.preds], 'protein', True, B)
    emit(rows, 'domainome', 'Domainome', 'all', point['domainome'], reps['domainome'], B, n=int(fr.spine['use'].sum()))
    D = pd.DataFrame(rows)
    D.to_csv(A.OUT / 'paired_vs_p0.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    t = D[D.set.str.contains('composite') | (D.group == 'tsuboyama')]
    print(t.pivot_table(index=['set', 'subset', 'head'], columns='family', values='delta').round(3).to_string())


if __name__ == '__main__':
    sys.exit(main())
