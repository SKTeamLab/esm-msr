#!/usr/bin/env python
"""Recompute every cell of BENCHMARKS sections 1.1-1.3 from the per-variant predictions and compare it with the
number printed in the document (``reference/all_by_all_long.csv``, ``reference/spurs_retrain_oct06.csv``).

This is the gate for everything else in ``analysis/``: the figures reuse the same loaders and the same Spearman engine,
so a cell that agrees here is a cell the figures show correctly.  Output: ``results/00_core/validation_vs_benchmarks.csv``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis.boot import DatasetBoot  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import ALL_DATASETS, PER_PROTEIN, REFERENCE, RESULTS  # noqa: E402

OUT = RESULTS / '00_core'
OUT.mkdir(parents=True, exist_ok=True)

# model key -> (reference label, reference n_seeds)
REF = {'msr:comb': ('ESM-MSR (released)', 3), 'msr:wt': ('ESM-MSR (released) WT', 3),
       'msrdev:comb': ('ESM-MSR (devel)', 3), 'msrdev:wt': ('ESM-MSR (devel) WT', 1),
       'msrrel:comb': ('ESM-MSR (released)', 1), 'msrrel:wt': ('ESM-MSR (released) WT', 1),
       'msrrel@0:comb': ('ESM3-small-open (σ=0, MSR readout)', 1),
       'me1': ('ME stage1', 3), 'me2': ('ME stage2', 3), 'me1+': ('ME stage1+', 3), 'me2+': ('ME stage2+', 3),
       'tmpnn': ('TMPNN', 3), 'tmpnnd': ('TMPNN+D', 3), 'spurs': ('SPURS (retrained)', 3), 'spursm': ('SPURS multi (retrained)', 3),
       'esmtherm': ('ESMtherm', 1), 'ddgemb': ('DDGemb', 1), 'ifum_wt': ('IFUM (WT backbone)', 1),
       'ifum_esmfold': ('IFUM (native, ESMFold)', 1), 'geoddg': ('GeoDDG-3D', 1), 'esm3dg': ('ESM3dG', 1),
       'prosst': ('ProSST', 1), 'pmpnn': ('ProteinMPNN', 1), 'esm3s': ('ESM3-small zero-shot', 1),
       'esm3m': ('ESM3-medium zero-shot', 1), 'esm3l': ('ESM3-large zero-shot', 1)}
for s in ('0.25', '0.5', '0.75'):
    REF[f'msrdev@{s}:comb'] = (f'ESM-MSR (devel) σ={s}', 3)
    REF[f'msrrel@{s}:comb'] = (f'ESM-MSR (released) σ={s}', 1)
DS_REF = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}


def main():
    ref = pd.read_csv(REFERENCE / 'all_by_all_long.csv')
    rows = []
    for ds in ALL_DATASETS:
        fr = load_frame(ds)
        b = DatasetBoot(fr, list(REF), per_protein=ds in PER_PROTEIN)
        for k in b.keys:
            rho, per_seed, n, ng = b.point(k)
            lab, ns = REF[k]
            r = ref[(ref.model == lab) & (ref.dataset == DS_REF.get(ds, ds)) & (ref.n_seeds == ns)]
            doc = float(r['rho'].iloc[0]) if len(r) else np.nan
            rows.append(dict(dataset=ds, model=k, rho=rho, doc=doc, diff=rho - doc, n=n, seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'validation_vs_benchmarks.csv', index=False, float_format='%.5f')
    ok = d.dropna(subset=['doc'])
    print(f"{len(ok)} cells with a documented value; |diff| median {ok['diff'].abs().median():.5f}, "
          f"max {ok['diff'].abs().max():.4f}; > 0.005 in {(ok['diff'].abs() > 0.005).sum()} cells")
    bad = ok[ok['diff'].abs() > 0.005].sort_values('diff', key=np.abs, ascending=False)
    print(bad.head(40).to_string(index=False))
    print('no documented value:', len(d) - len(ok))


if __name__ == '__main__':
    main()
