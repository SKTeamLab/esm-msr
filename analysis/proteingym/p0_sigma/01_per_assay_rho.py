#!/usr/bin/env python
"""Per-assay Spearman rho of the WT adapter at sigma = 0, 0.5 and 1, for all ProteinGym assays and the three DMS-benchmark sets that are not in ProteinGym.

  sigma = 1, 0.5   FINAL_p0 seeds 1-3 (``wt_lora_pred``): the per-assay rho of every seed, then the mean over the seeds (the combined read at sigma 0.5 exists for the 69 multi-mutant assays at seeds 1-3 since Nov 19: the mean over the seeds that have it)
  sigma = 0        ESM3-small-open, unmasked (the released model with the adapter scaled to 0)

rho is over all variants of the assay present in all three prediction files (singles and multi-mutants; the additive WT-head prediction for the multi-mutants), ProteinGym's convention.
Writes ``results/per_assay_rho.csv`` (assay, source, uniprot, category, n, train_overlap, fallback, rho_0.0, rho_0.5, rho_1.0: the WT head at every sigma; then has_multi and, for the deployed readout of the ProteinGym
assays, rho_dep_0.0, rho_dep_0.5, rho_dep_1.0 and dep_head_<sigma>: the combined prediction on the assays with multi-mutants, the WT head on the others (``pgsigma.load_assay_dep``; sigma 0.5 reads the WT head
on a multi-mutant assay until its combined run exists, sigma 0 is the combined read of the ESM-MSR-small checkpoint's sigma-0 run)); ``train_overlap`` flags the 23 assays that overlap the ESM-MSR training data
(``pgsigma.train_overlap``), which the figures leave out (``pgsigma.per_assay``).

    python analysis/proteingym/p0_sigma/01_per_assay_rho.py
"""
import numpy as np
import pandas as pd

import pgsigma as P


def seed_mean(rows):
    """Mean over the seeds of a list of dicts of per-seed values (the seeds that have the files); NaN where no seed has one."""
    return {k: (float(np.nanmean([r[k] for r in rows])) if rows else np.nan) for k in (rows[0] if rows else {})}


def main():
    ref = pd.read_csv(P.ROOT / 'ProteinGym' / 'DMS_substitutions.csv').set_index('DMS_id')
    rows, missing, leaked = [], [], P.train_overlap()
    for dms, r in ref.iterrows():
        per, dep_per, heads_per, n, has_multi = [], [], [], None, False
        for k in P.SEEDS:                                              # FINAL_p0 seeds 1-3: the per-assay rho of every seed, then the mean
            d = P.load_assay(dms, k)
            if d is None:
                continue
            dep, has_multi, heads = P.load_assay_dep(dms, k)
            n = len(d) if n is None else n
            per.append({f'rho_{s}': P.rho(d[f's{s}'].to_numpy(float), d['y'].to_numpy(float))[0] for s in P.SIGMAS})
            dep_per.append({f'rho_dep_{s}': P.rho(dep[f'd{s}'].to_numpy(float), dep['y'].to_numpy(float))[0] for s in P.SIGMAS})
            heads_per.append(heads)
        if not per:
            missing.append(dms)
            continue
        # the deployed read at a sigma: the mean over the seeds that have the combined prediction (sigma 0.5: seeds 1-3 on the 69 multi-mutant assays); an assay without one at any seed reads the WT head of every seed
        dep, heads = {}, {}
        for s in P.SIGMAS:
            ks = [i for i, h in enumerate(heads_per) if h[s] == 'combined'] or list(range(len(heads_per)))
            dep[f'rho_dep_{s}'] = float(np.nanmean([dep_per[i][f'rho_dep_{s}'] for i in ks]))
            heads[s] = heads_per[ks[0]][s]
        rows.append(dict(assay=dms, source='ProteinGym', uniprot=r.UniProt_ID, category=r.coarse_selection_type, n=n, train_overlap=dms in leaked, fallback=' '.join(str(x) for x in P.fallback_sigmas(dms)),
                         **seed_mean(per), has_multi=has_multi, **dep, **{f'dep_head_{s}': heads[s] for s in P.SIGMAS}, n_seeds=len(per)))
    for name in P.DMS_SETS:
        per, n_set = [], None
        for k in P.SEEDS:
            d = P.load_assay(name, k)
            if d is not None:
                n_set = len(d) if n_set is None else n_set
                per.append({f'rho_{s}': P.rho(d[f's{s}'].to_numpy(float), d['y'].to_numpy(float))[0] for s in P.SIGMAS})
        m = seed_mean(per)
        rows.append(dict(assay=name, source='DMS benchmark', uniprot=name.split('_Faure')[0].split('_Kung')[0], category='DMS benchmark', n=n_set, train_overlap=False, fallback='', **m, has_multi=False,
                         **{k.replace('rho_', 'rho_dep_'): v for k, v in m.items()}, **{f'dep_head_{s}': 'wt' for s in P.SIGMAS}, n_seeds=len(per)))
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
