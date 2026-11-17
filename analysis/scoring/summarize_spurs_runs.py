"""Three-seed summary CSV of the retrained SPURS / SPURS multi (docs/benchmarks/spurs_retrain_oct06.csv) from the per-seed scorer outputs.

Input: <scored>/{SPURS,SPURSm}_<tag>_s{1,2,3}_summary.csv written by score_baseline_runs.py, and the predictions' domainome.csv (the low-homology
Domainome column is the mean per-domain rho over the 458 domains outside the 64 `P`/`H` domains of reference/leak_audit_structure_homology.csv).
Rows: mean over seeds and the seed SD (ddof 1) of `spearman_all`, in the row order of the document's tables.

  python analysis/scoring/summarize_spurs_runs.py --scored baselines/retrained/spurs_unfrozen/scored --tag unf \\
      --predictions baselines/retrained/spurs/predictions --source "<text>" --out docs/benchmarks/spurs_retrain_oct06.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
DMS = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']
KEYS = [('tsu_capped_val', 'tsuboyama_val_all'), ('tsu_capped_test', 'tsuboyama_test_all')] + [(s, f'external_{s}') for s in SETS] \
    + [(d, f'dms_{d}') for d in DMS] + [('domainome', 'domainome_mean_over_domains'), ('PG avg', 'proteingym_PG avg'),
                                       ('PG Stability', 'proteingym_PG Stability'), ('PG DMS mean', 'proteingym_PG DMS mean')]


def low_id(pred_dir, col, excluded):
    out = []
    for k in (1, 2, 3):
        d = pd.read_csv(os.path.join(pred_dir, f'seed{k}', 'domainome.csv'), usecols=['code', 'ddG', col], low_memory=False).dropna()
        out.append(np.mean([spearmanr(g[col], g.ddG)[0] for c, g in d.groupby('code') if str(c) not in excluded and len(g) > 2]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = pd.read_csv(os.path.join(os.path.dirname(HERE), 'reference', 'leak_audit_structure_homology.csv'))
    excluded = set(ref[(ref.dataset == 'domainome') & (ref.class_with_struct.isin(['P', 'H']))].item_id.astype(str))
    rows = []
    for model, prefix, col in (('SPURS (retrained)', 'SPURS', 'SPURS_single_additive'), ('SPURS multi (retrained)', 'SPURSm', 'SPURS_multi')):
        S = [pd.read_csv(os.path.join(a.scored, f'{prefix}_{a.tag}_s{k}_summary.csv'), index_col=0) for k in (1, 2, 3)]
        for name, key in KEYS:
            v = np.array([s.loc[key, 'spearman_all'] for s in S])
            rows.append((model, name, v.mean(), v.std(ddof=1), 3, a.source))
        v = low_id(a.predictions, col, excluded)
        rows.append((model, 'domainome_low_id', v.mean(), v.std(ddof=1), 3, a.source))
    pd.DataFrame(rows, columns=['model', 'dataset', 'rho', 'sd', 'n_seeds', 'source']).to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(rows)} rows')


if __name__ == '__main__':
    main()
