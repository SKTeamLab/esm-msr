"""Score MutateEverything's ProteinGym predictions the way the BENCHMARKS ProteinGym rows are built.

Reads the per-assay files collected under <root>/baselines/retrained/mutate_everything/predictions/<stage>/seed<k>/proteingym/ and the labels in
<root>/pgym_inputs/<DMS>.csv, and prints per-seed and seed-mean PG avg / Stability / DMS mean and the five category means.

  python analysis_scripts/score_me_proteingym.py --stage stage2 --out /tmp/me_st2_pg_per_dms.csv     # corrected stage 2 (raw upstream result files)
  python analysis_scripts/score_me_proteingym.py --stage stage1                                      # the 2026-10-16 stage-1 files (labelled), + a check against the document

Two file formats are read. The stage-2 files (from the fixed-slot reruns, `outputs/tracked/me_st2_s*`) are ME's own result files: pdb_id, mut_info, scores,
ddg_pred, no labels; ddg_pred is destabilizing-positive, so the prediction is -ddg_pred, and the label is joined on mut_info == the label table's
mut_type_renumbered (one row per mutant). The stage-1 files have the labels and ME's readouts attached (pred_combined, DMS_score).
Per-DMS rho is Spearman over the rows that have a prediction; PG avg is the mean of the five category means, DMS mean the plain mean over the 213 assays
every model scored (the four in PG_DROP are left out), exactly as analysis_scripts/benchmark_matrix.py summarises the other models.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PG_DROP = {'BRCA1_HUMAN_Findlay_2018', 'SCN5A_HUMAN_Glazer_2019', 'POLG_CXB3N_Mattenberger_2021', 'BRCA2_HUMAN_Erwood_2022_HEK293T'}
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']


def rho(pred, y):
    """Spearman over the rows that have a prediction (benchmark_matrix.py's / pgym_runner.safe_spearman's rule)."""
    pred, y = np.asarray(pred, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(pred)
    pred, y = pred[ok], y[ok]
    if len(pred) < 2 or len(np.unique(y)) < 2 or len(np.unique(pred)) < 2:
        return np.nan
    r = spearmanr(pred, y)[0]
    return np.nan if np.isnan(r) else float(r)


def score_assay(res, labels):
    """rho of one assay. ``res`` is either an upstream result file (mut_info, ddg_pred) or a labelled one (pred_combined, DMS_score)."""
    if 'pred_combined' in res:
        return rho(res['pred_combined'], res['DMS_score'])
    j = labels[['mut_type_renumbered', 'DMS_score']].merge(res[['mut_info', 'ddg_pred']], left_on='mut_type_renumbered', right_on='mut_info', how='left')
    return rho(-j['ddg_pred'].to_numpy(), j['DMS_score'].to_numpy())


def summarise(per_dms, cat):
    """PG avg, Stability, DMS mean and the category means over the 213 common assays; ``per_dms`` is indexed by DMS id, ``cat`` maps DMS id -> category."""
    keep = [d for d in cat.index if d not in PG_DROP]
    s = per_dms.reindex(keep)
    g = s.groupby(cat.reindex(keep)).mean()
    out = {c: float(g[c]) for c in CATS}
    out.update({'PG avg': float(g.mean()), 'DMS mean': float(s.mean()), 'n': int(s.notna().sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--root', default=os.path.expanduser('~/playground/esm-msr-devel'))
    ap.add_argument('--out', help='write the per-DMS rho of every seed here')
    a = ap.parse_args()
    cat = pd.read_csv(f'{a.root}/baselines/pgym_per_dms/pgym_all_models_per_dms.csv').set_index('DMS_id')
    per = {}
    for k in a.seeds:
        r = {}
        for f in sorted(glob.glob(f'{a.root}/baselines/retrained/mutate_everything/predictions/{a.stage}/seed{k}/proteingym/*.csv')):
            name = os.path.basename(f)[:-4]
            if name == 'summary':
                continue
            r[name] = score_assay(pd.read_csv(f, low_memory=False), pd.read_csv(f'{a.root}/pgym_inputs/{name}.csv', usecols=['mut_type_renumbered', 'DMS_score']))
        per[k] = pd.Series(r)
    per = pd.DataFrame(per)
    rows = {f'seed{k}': summarise(per[k], cat['category']) for k in a.seeds}
    rows['mean'] = summarise(per.mean(axis=1), cat['category'])
    tab = pd.DataFrame(rows).T
    print(tab.round(4).to_string())
    sd = tab.loc[[f'seed{k}' for k in a.seeds], ['PG avg', 'Stability', 'DMS mean']].std(ddof=1)
    print('seed sd (ddof 1): ' + '  '.join(f'{c} {v:.4f}' for c, v in sd.items()))
    if a.out:
        per.to_csv(a.out)
    if a.stage == 'stage1':       # the same rule has to give the document's column
        keep = [d for d in cat.index if d not in PG_DROP]
        print('max |recomputed - pgym_all_models_per_dms "MutateEverything single"| over the 213 DMS: %.1e'
              % (per.mean(axis=1).reindex(keep) - cat.loc[keep, 'MutateEverything single']).abs().max())


if __name__ == '__main__':
    main()
