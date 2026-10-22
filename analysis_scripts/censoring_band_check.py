"""Does the dead-versus-alive decision at the censoring boundary carry anything a model could gain from? (docs/ci_weighting.md section 6.) CPU only.

    PYTHONPATH=src python analysis_scripts/censoring_band_check.py --raw_data_file ... --af_model_folder ... --split_file ... \
        --log_dir dc_runs/logs --ref_run mt_frozenwt_r16a16_s1 --new_run mt_frozenwt_dthr0_r16a16_s1 --epoch 3

Compares two paired runs (same seed, same frozen WT; they differ only in the doubles' censoring threshold) at one matched epoch:
1. the validation metrics, epoch by epoch;
2. the validation doubles split by reading and by the additive expectation x (an exogenous depth): where the two models' squared errors differ, and
   whether their within-library rank correlation with the label differs there;
3. on the training doubles, the within-library rank agreement of the TWO PROTEASES' readings grouped by x (not by the reading, which would restrict
   its range and bias the correlation toward 0): the rank signal that exists to be learned.
Also prints the WT-only threshold sweep if its runs are present (singles thresholds 0.3 / 0.5 / 0.6).
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ci_weighting_fit import prep
from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor

COLS = ['val_rho_combined_avg', 'val_rmse_combined_avg', 'val_epi_cell_rank_mt', 'val_epi_cell_mag_comb', 'val_epi_cell_flipacc_mt', 'val_epi_beyond_rho_mt']


def epochs(log_dir, run):
    m = pd.read_csv(f'{log_dir}/{run}/0/metrics.csv')
    return m.dropna(subset=['val_rho_combined_avg']).groupby('epoch')[[c for c in COLS if c in m.columns]].last()


def with_dump(log_dir, run, epoch, val):
    d = np.load(f'{log_dir}/{run}/0/val_dump_e{epoch}.npz', allow_pickle=True)
    keys = [(l, frozenset((int(q[1]), q[2]) for q in json.loads(m))) for l, m in zip(d['loader'], d['mut_key'])]
    D = pd.DataFrame({'kk': keys, 'y': d['ground_truth'], 'pred': d['comb'], 'src': d['cens_src'], 'subset': d['subset_type'], 'loader': d['loader']})
    return D.merge(val[['kk', 'dG_ML', 'x']], on='kk', how='left')


def lib_rho(g, a, b, minn=12):
    r = [spearmanr(h[a], h[b])[0] for _, h in g.groupby('loader') if len(h) >= minn and h[a].nunique() > 3 and h[b].nunique() > 3]
    return (float(np.nanmean(r)) if r else float('nan')), len(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ('raw_data_file', 'af_model_folder', 'split_file', 'log_dir', 'ref_run', 'new_run'):
        ap.add_argument('--' + a, required=True)
    ap.add_argument('--epoch', type=int, default=3)
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)

    A, B = epochs(a.log_dir, a.ref_run), epochs(a.log_dir, a.new_run)
    print('1. PAIRED validation metrics, %s (ref) | %s (new) | diff' % (a.ref_run, a.new_run))
    for e in sorted(set(A.index) & set(B.index)):
        print('  epoch %d  ' % e + '  '.join('%s %.4f|%.4f(%+.4f)' % (c[4:22], A.loc[e, c], B.loc[e, c], B.loc[e, c] - A.loc[e, c]) for c in A.columns if c in B))

    p = MegaScaleDatasetPreprocessor(a.raw_data_file, a.af_model_folder, spurs_override=False, censor_range=True)
    p.compute_floors(percentile=5.0, x_max=-1.5, min_n=20, max_sd=0.5, band=(0.0, 0.001))
    p.create_training_splits(a.split_file, 0)
    tr, val = prep(p.split_dfs['train'], p.dG_wt), prep(p.split_dfs['val'], p.dG_wt)
    val['kk'] = list(zip(val.code, val.p.map(frozenset)))

    ref, new = with_dump(a.log_dir, a.ref_run, a.epoch, val), with_dump(a.log_dir, a.new_run, a.epoch, val)
    assert (ref.kk == new.kk).all()
    M = ref[['kk', 'y', 'subset', 'loader', 'dG_ML', 'x', 'src']].assign(pa=ref.pred.values, pb=new.pred.values)
    M = M[(M.src != 1) & (M.subset == 'double') & M.x.notna()].copy()
    M['ea'], M['eb'] = (M.y - M.pa) ** 2, (M.y - M.pb) ** 2
    r, x = M.dG_ML, M.x
    groups = {'A read (0,0.5], expected dead (x<-1)': (r > 0) & (r <= 0.5) & (x < -1), 'B read (0,0.5], x >= -1': (r > 0) & (r <= 0.5) & (x >= -1),
              'C read <= 0': r <= 0, 'D read (0.5,1.5]': (r > 0.5) & (r <= 1.5), 'E read > 1.5': r > 1.5}
    tot = (M.eb - M.ea).sum()
    print('\n2. VALIDATION doubles at epoch %d (%d with an expectation): squared error ref -> new, share of the total change, within-library rank corr with label' % (a.epoch, len(M)))
    for k, m in groups.items():
        g = M[m]
        print('  %-40s n=%5d  mse %.3f -> %.3f  (%+5.1f%% of the change)   rho(pred, label) %6.3f -> %6.3f' % (
            k, len(g), g.ea.mean(), g.eb.mean(), 100 * (g.eb - g.ea).sum() / tot, lib_rho(g, 'pa', 'y')[0], lib_rho(g, 'pb', 'y')[0]))

    d = tr[(tr.n == 2) & tr.x.notna() & tr.dG_ML.notna() & (tr.cens == 0)]
    print('\n3. TRAIN doubles by the exogenous expectation x: within-library rank agreement of the trypsin and chymotrypsin readings')
    for lo, hi in [(-20, -3), (-3, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 9)]:
        g = d[(d.x > lo) & (d.x <= hi)].rename(columns={'code': 'loader'})
        rho = [spearmanr(h.deltaG_t, h.deltaG_c)[0] for _, h in g.groupby('loader') if len(h) >= 15 and h.deltaG_t.nunique() > 3 and h.deltaG_c.nunique() > 3]
        print('  x in (%5.1f,%5.1f]  n=%6d  rho_t,c %5.2f   reading <= 0 / (0,0.5] / > 0.5: %3.0f%% / %3.0f%% / %3.0f%%' % (
            lo, hi, len(g), float(np.nanmean(rho)) if rho else float('nan'), 100 * (g.dG_ML <= 0).mean(), 100 * ((g.dG_ML > 0) & (g.dG_ML <= 0.5)).mean(), 100 * (g.dG_ML > 0.5).mean()))

    sweep = [f'wt_r2a4_f0_cthr0{t}_s2' for t in (3, 5, 6)]
    if all(os.path.exists(f'{a.log_dir}/{s}/0/metrics.csv') for s in sweep):
        print('\n4. WT-only threshold sweep on singles (seed 2), best val_rho_wt_valid_avg:')
        for s in sweep:
            m = pd.read_csv(f'{a.log_dir}/{s}/0/metrics.csv')
            print('  %-26s %.4f' % (s, m['val_rho_wt_valid_avg'].max()))


if __name__ == '__main__':
    main()
