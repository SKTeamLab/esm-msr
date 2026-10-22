"""Fit and check the label-uncertainty model behind ``--ci_weight`` (esm_msr.ci_weights; docs/ci_weighting.md section 2). CPU only, ~1 minute.

    PYTHONPATH=src python analysis_scripts/ci_weighting_fit.py --raw_data_file ... --af_model_folder ... --split_file ... \
        [--val_dump <log_dir>/0/val_dump_e3.npz]

1. rebuilds the processed train/val tables with the repo's own preprocessor (so the keys and censoring tags are the training run's);
2. refits the depth prior m(x) = E[min(g, 3)^2 | x] on the TRAIN doubles and prints it next to the constants shipped in ci_weights.py;
3. with ``--val_dump`` (any run's ``--val_dump`` npz), calibrates the model's held-out squared error against the label-variance ingredients on
   the validation items: the disagreement coefficient (theory 1/4), the interval's, and what the depth prior adds for doubles;
4. reports, for the weights ``ci_weights`` would produce on the TRAIN items, their spread, effective sample size and weighting efficiency.
"""
import argparse
import json
import logging
import re

import numpy as np
import pandas as pd

from esm_msr import ci_weights
from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor

NUM = ['dG_ML', 'ddG_ML', 'deltaG_t', 'deltaG_c', 'deltaG_95CI']


def prep(df, dG_wt):
    df = df.copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['sgap'] = ci_weights.signed_gap(df.deltaG_t, df.deltaG_c)
    off = {code: ci_weights.library_offset(g.sgap.values, g.dG_ML.values) for code, g in df.groupby('code')}
    df['offset'] = df.code.map(off)
    df['g'] = (df.sgap - df.offset).abs().clip(upper=ci_weights.GAP_CAP)
    df['p'] = df.mut_type.map(lambda mt: [(int(m.group(2)), m.group(3)) for m in
                                         (re.match(r'([A-Za-z])(\d+)([A-Za-z])', t) for t in str(mt).split(':')) if m])
    df['n'] = df.p.map(len)
    s = df[(df.n == 1) & df.ddG_ML.notna()]
    single = {(c, p[0]): v for c, p, v in zip(s.code, s.p, s.ddG_ML)}
    df['x'] = [(single.get((c, p[0]), np.nan) + single.get((c, p[1]), np.nan) + dG_wt.get(c, np.nan)) if len(p) == 2 else np.nan
               for c, p in zip(df.code, df.p)]
    df['L'] = ci_weights.label_variance(df.deltaG_95CI, df.deltaG_t, df.deltaG_c, df.offset, df.x.where(df.n == 2))
    return df


def ols(X, cols, y='r2'):
    A = np.c_[np.ones(len(X)), *[X[c].values for c in cols]]
    coef, *_ = np.linalg.lstsq(A, X[y].values, rcond=None)
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * ((X[y].values - A @ coef) ** 2).mean())
    return '  '.join('%s=%.3f(%.3f)' % (n, c, s) for n, c, s in zip(['a'] + cols, coef, se))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_data_file', required=True)
    ap.add_argument('--af_model_folder', required=True)
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--val_dump')
    ap.add_argument('--tau', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75])
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)

    p = MegaScaleDatasetPreprocessor(a.raw_data_file, a.af_model_folder, spurs_override=False, censor_range=True)
    p.compute_floors(percentile=5.0, x_max=-1.5, min_n=20, max_sd=0.5, band=(0.0, 0.001))
    p.create_training_splits(a.split_file, 0)
    tr, val = prep(p.split_dfs['train'], p.dG_wt), prep(p.split_dfs['val'], p.dG_wt)

    # ---- 2. the depth prior
    dd = tr[(tr.n == 2) & tr.x.notna() & tr.g.notna()]
    edges = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4.0])
    fit = dd.groupby(pd.cut(dd.x.clip(-5, 4), edges), observed=True).g.apply(lambda v: (v ** 2).mean()).values
    print('depth prior m(x), x at the centres')
    print('  fitted : ', np.round(fit, 3).tolist())
    print('  shipped: ', list(ci_weights.DEPTH_M), '  (max |diff| %.3f)' % np.abs(fit - np.array(ci_weights.DEPTH_M)).max())

    # ---- 3. calibration against a trained model's held-out squared error
    if a.val_dump:
        d = np.load(a.val_dump, allow_pickle=True)
        val['kk'] = list(zip(val.code, val.p.map(frozenset)))
        dk = [(l, frozenset((int(q[1]), q[2]) for q in json.loads(m))) for l, m in zip(d['loader'], d['mut_key'])]
        D = pd.DataFrame({'kk': dk, 'y': d['ground_truth'], 'comb': d['comb'], 'src': d['cens_src'], 'subset': d['subset_type']})
        M = D.merge(val[['kk', 'dG_ML', 'g', 'deltaG_95CI', 'x', 'L']], on='kk', how='left')
        M = M[M.src != 1].copy()                                    # a range-bounded row's label is a bound, not a value
        M['r2'] = (M.y - M.comb) ** 2
        M['g2'] = M.g ** 2
        M['ci2'] = (M.deltaG_95CI / ci_weights.CI_Z95) ** 2
        M['mx'] = ci_weights.depth_prior(M.x)
        inr = M[(M.dG_ML > 1) & (M.dG_ML < 4.5)]
        print('\nheld-out squared error of the model (%d items, rmse %.3f); IN RANGE (1-4.5), theory: coefficient on g^2 is 1/4' % (
            len(M), np.sqrt(M.r2.mean())))
        print('  all items  : mse =', ols(inr, ['g2', 'ci2']))
        print('  singles    : mse =', ols(inr[inr.subset == 'single'], ['g2']))
        print('  doubles    : mse =', ols(inr[inr.subset == 'double'], ['g2']))
        dbl = M[(M.subset == 'double') & M.mx.notna()]
        print('doubles, all readings (%d): mse =' % len(dbl), ols(dbl, ['g2']), '\n                          mse =', ols(dbl, ['g2', 'mx']),
              '\n                          mse =', ols(dbl, ['mx']))

    # ---- 4. the weights on the train items
    # two populations: every item with a variance (range-bounded rows excluded: they are bounds), and the items the queued recipe KEEPS as values:
    # singles read above 0.5 and doubles read above 0 (the floor bounds of mt_frozenwt_dthr0_*), with the protease-gap rule off.
    t = tr[tr.L.notna() & (tr.cens == 0)]
    kept = t[((t.n == 1) & (t.dG_ML > 0.5)) | ((t.n == 2) & (t.dG_ML > 0.0))]
    for name, pop in (('every uncensored-by-range item', t), ('items kept as values by the dthr0 recipe', kept)):
        print('\nweights on %d %s, by tau   [efficiency = E[tau^2+L] E[1/(tau^2+L)], the most variance optimal weights can remove]' % (len(pop), name))
        print('  tau    w05   w50   w95   max   ESS%%  efficiency   weight share of doubles with x<0 (count share %.1f%%)   of x<-2 (count %.1f%%)' % (
            100 * ((pop.n == 2) & (pop.x < 0)).mean(), 100 * ((pop.n == 2) & (pop.x < -2)).mean()))
        for tau in a.tau:
            ref = ci_weights.reference_weight(pop.L.values, tau)
            sm = ci_weights.summarise(pop.L.values, tau, ref)
            w = 1.0 / (tau ** 2 + pop.L.values) / ref
            d0, d2 = ((pop.n == 2) & (pop.x < 0)).values, ((pop.n == 2) & (pop.x < -2)).values
            print('  %.2f  %5.2f %5.2f %5.2f %5.1f  %4.0f  %.3f       %5.1f%%                                              %5.1f%%' % (
                tau, sm['p05'], sm['p50'], sm['p95'], sm['max'], 100 * sm['ess_frac'], sm['efficiency'],
                100 * w[d0].sum() / w.sum(), 100 * w[d2].sum() / w.sum()))
    protease_gap = tr[(tr.cens == 0) & (tr.dG_ML > 0)].assign(gap_abs=lambda d: d.sgap.abs())
    print('\nitems the hard gap rule (|dG_t - dG_c| > 2) would censor among those kept as values: singles %.2f%%, doubles %.2f%%' % (
        100 * (protease_gap[protease_gap.n == 1].gap_abs > 2).mean(), 100 * (protease_gap[protease_gap.n == 2].gap_abs > 2).mean()))
    singles = tr[(tr.n == 1) & tr.L.notna()]
    print('label variance L: singles median %.3f p95 %.3f ; doubles median %.3f p95 %.3f' % (
        singles.L.median(), singles.L.quantile(.95), t[t.n == 2].L.median(), t[t.n == 2].L.quantile(.95)))


if __name__ == '__main__':
    main()
