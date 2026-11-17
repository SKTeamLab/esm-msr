"""Apply the Tobit link post hoc to a finished MGnify run (esm_msr_testing.py --mgnify) and score it.

The link is not trained: ``observed ddG = h(dG_wt + latent) - dG_wt`` with ``h`` the closed-form expected reading of the noisy-floor Tobit model
(``esm_msr.val_stats.tobit_link``) and the three constants the checkpoint was trained with (hparams.yaml ``tobit_sigma_y`` / ``tobit_mu_f`` / ``tobit_sigma_f``;
FINAL_p0: 0.28 / -0.14 / 0.5). ``predictions.csv.gz`` already holds the latent predictions and each scaffold's measured ``dG_wt``, so no GPU run is needed.

  python analysis/scoring/mgnify_link.py final_mgnify/FINAL_p0/s1/heldout [--hparams final_checkpoints/FINAL_p0/s1/hparams.yaml]

Labels come from the v2 table (analysis/scoring/mgnify_labels.py) unless --v1_labels.
Writes ``stats_linked.csv`` (regime x subset x head, latent and linked side by side; same regimes as the runner's stats.csv) next to the predictions.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgnify_labels import apply_v2_labels  # noqa: E402

HEADS = ['wt_lora_pred', 'mt_lora_pred', 'combined_pred']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--hparams', default=None, help='the checkpoint\'s hparams.yaml (default: the Tobit constants of FINAL_p0)')
    ap.add_argument('--v1_labels', action='store_true', help='keep the labels stored in the run (v1 table) instead of the v2 table')
    a = ap.parse_args()
    c = dict(tobit_sigma_y=0.28, tobit_mu_f=-0.14, tobit_sigma_f=0.5)
    if a.hparams:
        hp = yaml.safe_load(open(a.hparams))
        c = {k: float(hp[k]) for k in c}
    h = tobit_link(c['tobit_sigma_y'], c['tobit_mu_f'], c['tobit_sigma_f'])
    d = pd.read_csv(os.path.join(a.run_dir, 'predictions.csv.gz'), low_memory=False)
    if not a.v1_labels:
        d = apply_v2_labels(d)
    dg = d['dG_wt'].astype(float)
    for k in HEADS:
        d[k + '_linked'] = h(dg + d[k]) - dg
    in_range = d['in_range_1_5'].astype(bool)
    regimes = {'in_range_1_5': in_range, 'all_rows': pd.Series(True, index=d.index)}
    subsets = {'all': pd.Series(True, index=d.index), 'singles': d['n_mut'] == 1, 'doubles': d['n_mut'] == 2}
    rows = []
    for rn, rm in regimes.items():
        for sn, sm in subsets.items():
            for k in HEADS:
                s = d[rm & sm]
                s = s[s['ddG_ML'].notna() & s[k].notna()]
                if len(s) < 5:
                    continue
                row = dict(regime=rn, subset=sn, head=k, n=len(s))
                for tag, col in (('latent', k), ('linked', k + '_linked')):
                    row[f'spearman_{tag}'] = spearmanr(s[col], s['ddG_ML'])[0]
                    row[f'pearson_{tag}'] = pearsonr(s[col], s['ddG_ML'])[0]
                    row[f'rmse_{tag}'] = float(np.sqrt(np.mean((s[col] - s['ddG_ML']) ** 2)))
                rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(a.run_dir, 'stats_linked.csv')
    out.to_csv(path, index=False, float_format='%.4f')
    print(f'wrote {path} (Tobit constants {c})')
    print(out[out['subset'].isin(['all'])].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
