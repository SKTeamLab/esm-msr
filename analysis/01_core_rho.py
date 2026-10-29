#!/usr/bin/env python
"""Spearman rho of every model on every dataset with bootstrap intervals (protein-, site- and mutation-level).

Writes ``results/00_core/rho_bootstrap.csv`` (one row per model x dataset x resampling unit) and the replicate
matrices ``cache/boot_<kind>.pkl`` that the figure scripts use for *paired* differences between models.

rho conventions are those of BENCHMARKS: seed means for multi-seed models; pooled rho per set for the external sets and
DMS; mean of per-protein rho for Tsuboyama (per library) and Domainome (per domain); measured rows only.
Resampling units: ``protein`` (clusters of whole proteins, drawn jointly over the 12 external sets so that models and sets
stay paired), ``site`` (first mutated position, for the single-protein DMS), ``mutation`` (single rows; within a protein for
the per-protein means).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import (ALL_DATASETS, CACHE, DMS7, EXT12, MODELS, RESULTS, TSU)  # noqa: E402

OUT = RESULTS / '00_core'
OUT.mkdir(parents=True, exist_ok=True)


def keys_for(fr):
    return [k for k in MODELS if k in fr.preds]


def n_boot(n):
    return 1000 if n <= 12000 else (500 if n <= 70000 else 200)


def run(kind, groups, label):
    rows, allreps = [], {}
    for name, dss in groups.items():
        t0 = time.time()
        frames = {d: load_frame(d) for d in dss}
        keys = sorted({k for f in frames.values() for k in keys_for(f)})
        B = min(n_boot(len(f)) for f in frames.values())
        if kind == 'mutation':
            B = max(100, B // 2)
        point, reps = bootstrap_rho(frames, keys, kind=kind, B=B, seed=11)
        for ds in dss:
            for k, (rho, per_seed, n, ng) in point[ds].items():
                lo, hi = ci(reps[ds][k])
                rows.append(dict(dataset=ds, model=k, kind=kind, rho=rho, lo=lo, hi=hi, se=float(np.nanstd(reps[ds][k], ddof=1)),
                                 n=n, n_proteins=ng, seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan, B=B))
        allreps.update({ds: reps[ds] for ds in dss})
        print(f'{kind:8s} {name:12s} B={B:4d} {len(keys):2d} models  {time.time() - t0:6.1f}s', flush=True)
    with open(CACHE / f'boot_{kind}.pkl', 'wb') as fh:
        pickle.dump(allreps, fh, protocol=4)
    return rows


def main():
    rows = []
    # protein level: the external sets resample the union of their proteins; each other set on its own
    rows += run('protein', {'external': EXT12, 'tsu_val': [TSU[0]], 'tsu_test': [TSU[1]], 'domainome': ['domainome']}, 'protein')
    # site level for the single-protein DMS (proteins cannot be resampled there)
    rows += run('site', {d: [d] for d in DMS7}, 'site')
    # mutation level everywhere (Domainome / GB1 with few replicates)
    rows += run('mutation', {**{'external': EXT12}, **{d: [d] for d in DMS7 + TSU + ['domainome']}}, 'mutation')
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'rho_bootstrap.csv', index=False, float_format='%.5f')
    print(d.groupby('kind').size())


if __name__ == '__main__':
    main()
