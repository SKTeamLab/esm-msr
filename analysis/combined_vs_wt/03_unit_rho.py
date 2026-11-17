#!/usr/bin/env python
"""Per-protein rho of the WT head, the MT head and the combined prediction, for every protein of the per-protein-mean sets (Homology-Restricted MegaScale val and test, Domainome).

One row per set x subset (all / single mutants / multi-mutants) x protein x family: the seed-mean Spearman rho of each head with the measurement over the protein's variants of that subset, the number of
variants, and the Spearman correlation between the WT and the MT head (seed 1).  ProteinGym's per-assay table is ``02_pg_per_assay.py``.  Writes ``results/unit_rho.csv``.

    python analysis/combined_vs_wt/03_unit_rho.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cvw as C
from msr_analysis.boot import DatasetBoot

SETS = C.TSU + ['domainome']


def main():
    rows = []
    for ds in SETS:
        fr = C.load_final_frame(ds, families=C.FAMS)
        sp = fr.spine
        nm = sp['n_mut'].to_numpy()
        use = sp['use'].to_numpy()
        subs = {'all': use}
        if (nm[use] > 1).any():
            subs.update(single=use & (nm == 1), multi=use & (nm > 1))
        for sub, mask in subs.items():
            for fam in C.FAMS:
                keys = [f'{fam}:{h}' for h in C.HEADS]
                if not all(k in fr.preds for k in keys):
                    continue
                b = DatasetBoot(fr, keys, per_protein=True, subset=mask)
                per = {}
                for k in keys:
                    names, r = b.per_protein_rho_seeds(k)
                    per[k.split(':')[1]] = pd.Series(np.nanmean(r, axis=0), index=names)
                idx = np.flatnonzero(b.valid[keys[0]])
                g = sp['group'].to_numpy()[idx]
                wt1, mt1 = fr.preds[f'{fam}:wt'][0][idx], fr.preds[f'{fam}:mt'][0][idx]
                n = pd.Series(1, index=g).groupby(level=0).sum()
                rwm = {}
                for name in per['wt'].index:
                    m = g == name
                    rwm[name] = float(spearmanr(wt1[m], mt1[m])[0]) if m.sum() >= 10 else np.nan
                t = pd.DataFrame({f'rho_{h}': per[h] for h in C.HEADS})
                t['n'] = n
                t['r_wt_mt'] = pd.Series(rwm)
                t = t.rename_axis('protein').reset_index()
                t.insert(0, 'subset', sub)
                t.insert(0, 'family', fam)
                t.insert(0, 'dataset', ds)
                rows.append(t)
                print(f'  {ds} {sub} {fam}: {len(t)} proteins', flush=True)
    T = pd.concat(rows, ignore_index=True)
    T['d_comb'] = T.rho_comb - T.rho_wt
    T['d_mt'] = T.rho_mt - T.rho_wt
    T.to_csv(C.OUT / 'unit_rho.csv', index=False, float_format='%.4f')
    print(T.groupby(['dataset', 'subset', 'family'])[['d_comb', 'd_mt', 'r_wt_mt']].mean().round(3).to_string())


if __name__ == '__main__':
    sys.exit(main())
