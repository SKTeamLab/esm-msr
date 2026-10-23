"""Per-DMS Spearman of the ProteinGym authors' own published zero-shot scores.

ProteinGym ships every benchmarked model's predictions for every DMS (zero_shot_substitutions_scores.zip, one CSV per
DMS with one column per model, ~99 columns). For a model the authors have already run there is nothing to compute:
no GPU, no weights, no structures. This reads two columns (ProSST-2048, which is the checkpoint we would have run, and
ProteinMPNN, which is an independent check on our own ProteinMPNN implementation) and scores them two ways:

  rho_mapped  on exactly the variants our other models were scored on (pgym_inputs/<DMS>.csv, joined on `mutant`):
              some DMS are only partly covered by the AF2 structure (map_mode af2_subset), and every other model
              in the comparison sees only the covered variants, so this is the like-for-like number.
  rho_full    on every variant in the public file, which is the official ProteinGym protocol.

  curl -sL -o zero_shot_substitutions_scores.zip \\
      https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip
  python analysis_scripts/pgym_public_scores.py --out pgym_public_per_dms.csv
"""
import argparse
import io
import zipfile

import numpy as np
import pandas as pd

ZIP = '/home/sreeves/stability-baselines/pgym_public/zero_shot_substitutions_scores.zip'
INP = '/home/sreeves/esm-msr-devel/pgym_inputs'
CATEGORY = '/home/sreeves/esm-msr-devel/outputs/pgym/pgym_all_models_per_dms.csv'


def rho(a, b):
    v = pd.DataFrame({'a': a, 'b': b}).dropna()
    return float(v.corr('spearman').iloc[0, 1]) if len(v) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pgym_public_per_dms.csv')
    ap.add_argument('--models', default='ProSST-2048,ProteinMPNN')
    a = ap.parse_args()
    models = a.models.split(',')
    z = zipfile.ZipFile(ZIP)
    cat = pd.read_csv(CATEGORY).set_index('DMS_id').category
    rows = []
    for dms in cat.index:
        name = f'{dms}.csv'
        if name not in z.namelist():
            print('not in archive:', dms)
            continue
        with z.open(name) as f:
            pub = pd.read_csv(io.TextIOWrapper(f), usecols=['mutant', 'DMS_score'] + models)
        mine = pd.read_csv(f'{INP}/{dms}.csv', usecols=['mutant'])
        sub = pub[pub.mutant.isin(set(mine.mutant))]
        r = dict(DMS_id=dms, category=cat[dms], n_public=len(pub), n_mapped=len(sub))
        for m in models:
            r[f'{m} rho_mapped'] = rho(sub[m], sub.DMS_score)
            r[f'{m} rho_full'] = rho(pub[m], pub.DMS_score)
            r[f'{m} n_scored'] = int(pub[m].notna().sum())
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    print(f'{len(D)} DMS -> {a.out}')
    for m in models:
        print(f'{m}: mean rho_mapped {D[f"{m} rho_mapped"].mean():.4f}  rho_full {D[f"{m} rho_full"].mean():.4f}  '
              f'(DMS with any score: {int(D[f"{m} n_scored"].gt(0).sum())})')


if __name__ == '__main__':
    main()
