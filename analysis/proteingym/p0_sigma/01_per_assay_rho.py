#!/usr/bin/env python
"""Per-assay Spearman rho of the WT adapter at sigma = 0, 0.5 and 1, for all ProteinGym assays and the three DMS-benchmark sets that are not in ProteinGym.

  sigma = 1, 0.5   FINAL_p0 seed 1 (``wt_lora_pred``)
  sigma = 0        ESM3-small-open, unmasked (the released model with the adapter scaled to 0)

rho is over all variants of the assay present in all three prediction files (singles and multi-mutants; the additive WT-head prediction for the multi-mutants), ProteinGym's convention.
Writes ``results/per_assay_rho.csv`` (assay, source, uniprot, category, n, train_overlap, fallback, rho_0.0, rho_0.5, rho_1.0: the WT head at every sigma; then has_multi and, for the deployed readout of the ProteinGym
assays, rho_dep_0.0, rho_dep_0.5, rho_dep_1.0 and dep_head_<sigma>: the combined prediction on the assays with multi-mutants, the WT head on the others (``pgsigma.load_assay_dep``; sigma 0.5 reads the WT head
on a multi-mutant assay until its combined run exists, sigma 0 is the combined read of the ESM-MSR-small checkpoint's sigma-0 run)); ``train_overlap`` flags the 23 assays that overlap the ESM-MSR training data
(``pgsigma.train_overlap``), which the figures leave out (``pgsigma.per_assay``).

    python analysis/proteingym/p0_sigma/01_per_assay_rho.py
"""
import pandas as pd

import pgsigma as P


def main():
    ref = pd.read_csv(P.ROOT / 'ProteinGym' / 'DMS_substitutions.csv').set_index('DMS_id')
    rows, missing, leaked = [], [], P.train_overlap()
    for dms, r in ref.iterrows():
        d = P.load_assay(dms)
        if d is None:
            missing.append(dms)
            continue
        dep, has_multi, heads = P.load_assay_dep(dms)
        rows.append(dict(assay=dms, source='ProteinGym', uniprot=r.UniProt_ID, category=r.coarse_selection_type, n=len(d), train_overlap=dms in leaked, fallback=' '.join(str(x) for x in P.fallback_sigmas(dms)),
                         **{f'rho_{s}': P.rho(d[f's{s}'].to_numpy(float), d['y'].to_numpy(float))[0] for s in P.SIGMAS}, has_multi=has_multi,
                         **{f'rho_dep_{s}': P.rho(dep[f'd{s}'].to_numpy(float), dep['y'].to_numpy(float))[0] for s in P.SIGMAS}, **{f'dep_head_{s}': heads[s] for s in P.SIGMAS}))
    for name in P.DMS_SETS:
        d = P.load_assay(name)
        rows.append(dict(assay=name, source='DMS benchmark', uniprot=name.split('_Faure')[0].split('_Kung')[0], category='DMS benchmark', n=len(d), train_overlap=False, fallback='',
                         **{f'rho_{s}': P.rho(d[f's{s}'].to_numpy(float), d['y'].to_numpy(float))[0] for s in P.SIGMAS}, has_multi=False,
                         **{f'rho_dep_{s}': P.rho(d[f's{s}'].to_numpy(float), d['y'].to_numpy(float))[0] for s in P.SIGMAS}, **{f'dep_head_{s}': 'wt' for s in P.SIGMAS}))
    t = pd.DataFrame(rows)
    t.to_csv(P.OUT / 'per_assay_rho.csv', index=False, float_format='%.4f')
    print(f'{len(t)} assays ({int((t.source == "ProteinGym").sum())} ProteinGym); missing in at least one sigma: {missing}')
    print('assays with a sigma read from a fallback file:', t.loc[t.fallback.fillna('') != '', ['assay', 'fallback']].to_dict('records'))
    pg = t[(t.source == 'ProteinGym') & ~t.train_overlap]
    mm = t[(t.source == 'ProteinGym') & t.has_multi]
    print(f'{len(mm)} ProteinGym assays with multi-mutants; combined prediction found at sigma 0 / 0.5 / 1 for', ' / '.join(str(int((mm[f"dep_head_{s}"] == "combined").sum())) for s in P.SIGMAS))
    print(f'{int(t.train_overlap.sum())} assays overlap the training data and are left out of the figures; {len(pg)} ProteinGym assays remain')
    print(pg.groupby('category')[['rho_0.0', 'rho_0.5', 'rho_1.0']].agg(['mean']).round(3).assign(n=pg.groupby('category').size()).to_string())


if __name__ == '__main__':
    main()
