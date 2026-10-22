"""What the assay's own confidence interval says about label noise, and how hard ``--ci_weight`` would bite on it.

Reads the raw Tsuboyama table (the file ``--raw_data_file`` points at) and reports, for the numeric double mutants:
the CI-implied label SD against the protease disagreement, the CI by dG band, and, for a range of ``--ci_weight_tau``, the
spread of the weights, their effective sample size and the share of total weight that the doubles read at or below the censoring
threshold keep. These are the numbers in docs/ci_weighting.md section 2. CPU only.

    python analysis_scripts/ci_weighting_stats.py /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv [--threshold 0.5]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

Z95 = 3.92


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw_csv')
    ap.add_argument('--threshold', type=float, default=0.5, help='the doubles censoring threshold the weights are compared with')
    a = ap.parse_args()

    df = pd.read_csv(a.raw_csv, low_memory=False)
    for c in ('dG_ML', 'ddG_ML', 'deltaG', 'deltaG_95CI', 'deltaG_t', 'deltaG_c'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    mt = df['mut_type'].astype(str)
    clean = ~mt.str.contains('ins|del|wt', case=False)
    dbl = df[mt.str.contains(':') & clean & df.ddG_ML.notna()].copy()
    sgl = df[~mt.str.contains(':') & clean & df.ddG_ML.notna()].copy()
    dbl['gap'] = (dbl.deltaG_t - dbl.deltaG_c).abs()
    sig = dbl.deltaG_95CI / Z95
    print('numeric doubles %d, singles %d; dG_ML == deltaG for every double: %s' % (
        len(dbl), len(sgl), bool(np.allclose(dbl.dG_ML, dbl.deltaG))))
    inr = dbl[(dbl.dG_ML > 0.5) & (dbl.dG_ML < 4.5)]
    print('CI-implied label SD, in range (dG 0.5-4.5): median %.3f   vs  protease disagreement implies a per-protease SD of %.3f' % (
        (inr.deltaG_95CI / Z95).median(), inr.gap.median() / 0.954))
    print('Spearman(CI width, protease gap): in range %.3f, all doubles %.3f' % (
        spearmanr(inr.deltaG_95CI, inr.gap, nan_policy='omit')[0], spearmanr(dbl.deltaG_95CI, dbl.gap, nan_policy='omit')[0]))

    bins = [-1.01, -0.5, 0, 0.5, 1, 2, 3, 4, 4.5, 5.01]
    for name, x in (('singles', sgl), ('doubles', dbl)):
        x = x.assign(band=pd.cut(x.dG_ML, bins), gap=(x.deltaG_t - x.deltaG_c).abs())
        g = x.groupby('band', observed=True).agg(n=('dG_ML', 'size'), CI_median=('deltaG_95CI', 'median'),
                                                  gap_median=('gap', 'median'))
        print('\n%s by dG_ML band' % name)
        print(g.round(3).to_string())

    low = (dbl.dG_ML <= a.threshold).values
    s2 = sig.values ** 2
    print('\ndoubles at/below dG %.2f: %.1f%% of all (a uniform weighting would give them that share of the loss)' % (
        a.threshold, 100 * low.mean()))
    print('CI width quantiles (10/50/90%%): at/below %s   above %s' % (
        dbl.deltaG_95CI[low].quantile([.1, .5, .9]).round(3).tolist(), dbl.deltaG_95CI[~low].quantile([.1, .5, .9]).round(3).tolist()))
    print('\n tau   w95/w05   ESS%   share kept by the censored-population   max/mean')
    for tau in (0, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4):
        w = 1 / (s2 + tau ** 2 + 1e-12)
        w = w / w.mean()
        print(' %.2f  %7.1f  %5.1f    %5.1f%%                                %5.1f' % (
            tau, np.quantile(w, .95) / np.quantile(w, .05), 100 * w.sum() ** 2 / (w ** 2).sum() / len(w),
            100 * w[low].sum() / w.sum(), w.max()))


if __name__ == '__main__':
    main()
