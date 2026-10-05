"""Which parts of the measured epistasis does a model's prediction capture?

Takes the model's predicted dddG for doubles and correlates it with each component of the measured dddG (the same hierarchy as
anova_variance_partition.py): the global saturation curve, the position-pair offset, the per-substitution row and column effects, and the
interaction left over. A model that learns the identity-independent biology scores high on the first three; one that has learned the
interaction scores on the last. Each component is a deterministic function of the MEASURED data of the evaluated cells.

Usage:
  PYTHONPATH=src python analysis_notebooks/anova/decompose_predictions.py PREDICTIONS.csv [--pred_col combined_dddg_pred] [--out out.json]
PREDICTIONS.csv is an esm_msr_testing.py output: it needs `code` (or code_wt), `mut_type` ('A12G:K30R' rows are the doubles) and the prediction column.
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, 'analysis_notebooks/anova')
import anova_variance_partition as A  # noqa: E402

PAT = re.compile(r'^([A-Z])(\d+)([A-Z]):([A-Z])(\d+)([A-Z])$')


def measured_components(T: pd.DataFrame, iso: IsotonicRegression) -> pd.DataFrame:
    """Adds the measured components g (global), m (pair offset), rc (row + column) and eps (what is left) to a doubles table."""
    T = T.copy()
    T['g'] = iso.predict(T.x) - T.x
    T['r1'] = T.dddG - T.g
    T['m'] = T.groupby('pair').r1.transform('mean')
    T['r2'] = T.r1 - T.m
    R, C = A.fit_rowcol(T.assign(r2=T.r2), 'r2')
    T['rc'] = [R[p][a] + C[p][b] for p, a, b in zip(T.pair, T.a, T.b)]
    T['eps'] = T.r2 - T.rc
    return T


def decompose(T: pd.DataFrame, pred: np.ndarray, iso: IsotonicRegression) -> dict:
    """Spearman of ``pred`` with dddG and with each measured component, over the cells of ``T`` (aligned with ``pred``)."""
    C = measured_components(T, iso)
    ok = np.isfinite(pred)
    out = {'n_cells': int(ok.sum()), 'n_pairs': int(C.pair[ok].nunique())}
    for name, col in (('measured dddG', 'dddG'), ('global saturation', 'g'), ('position-pair offset', 'm'),
                      ('substitution row + column effects', 'rc'), ('interaction (leftover)', 'eps')):
        v = C[col].to_numpy()
        out[name] = float(spearmanr(pred[ok], v[ok])[0]) if np.ptp(v[ok]) > 0 else float('nan')
    out['share_of_measured_variance'] = {col: float(np.var(C[col][ok]) / np.var(C['dddG'][ok]))
                                         for col in ('g', 'm', 'rc', 'eps')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred_csv')
    ap.add_argument('--pred_col', default='combined_dddg_pred')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    T, _ = A.build_table()                                  # all measured doubles, with dG_wt, pair keys, dddG
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(T.x, T.dG_AB)
    P = pd.read_csv(args.pred_csv, low_memory=False)
    code_col = 'code' if 'code' in P.columns else 'code_wt'
    rows = []
    for code, mt, pr in zip(P[code_col].astype(str), P['mut_type'].astype(str), P[args.pred_col]):
        m = PAT.match(mt)
        if m:
            _, p1, a1, _, p2, a2 = m.groups()
            p1, p2 = int(p1), int(p2)
            if p1 > p2:
                p1, p2, a1, a2 = p2, p1, a2, a1
            rows.append((code.replace('|', '_'), p1, p2, a1, a2, pr))
    D = pd.DataFrame(rows, columns=['code', 'p', 'q', 'a', 'b', 'pred'])
    M = T.merge(D, on=['code', 'p', 'q', 'a', 'b'], how='inner').reset_index(drop=True)
    print(f"{len(M)} doubles matched between the measured table and {args.pred_csv}")
    res = decompose(M, M['pred'].to_numpy(float), iso)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
