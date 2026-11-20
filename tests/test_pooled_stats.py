"""Pooled validation correlations: one Spearman over all libraries at once, on the per-library population."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esm_msr import censoring, pooled_stats, stats  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    lib = np.repeat(np.arange(3), 100)
    gt = rng.normal(size=n) + 2.0 * lib                                    # a library offset the per-library means cannot see
    comb = gt + rng.normal(scale=0.5, size=n)
    wt = gt + rng.normal(scale=0.7, size=n)
    st = np.where(rng.random(n) < 0.6, 'single', 'double')
    return wt, comb, gt, st, lib


def test_pooled_equals_the_spearman_of_the_pooled_items():
    wt, comb, gt, st, _ = _data()
    out = pooled_stats.pooled_rho(wt, comb, gt, st)
    assert abs(out['rho_combined_pooled'] - spearmanr(comb, gt)[0]) < 1e-12
    s = st == 'single'
    assert abs(out['rho_singles_wt_pooled'] - spearmanr(wt[s], gt[s])[0]) < 1e-12


def test_pooled_sees_a_library_offset_that_the_per_library_mean_does_not():
    wt, comb, gt, st, lib = _data()
    per = np.mean([stats.compute_metrics(wt[lib == i], wt[lib == i], comb[lib == i], gt[lib == i], st[lib == i])['rho_combined'] for i in range(3)])
    pooled = pooled_stats.pooled_rho(wt, comb, gt, st)['rho_combined_pooled']
    shifted = comb - 2.0 * lib                                              # remove the offset from the predictions: per-library rho unchanged, pooled drops
    per2 = np.mean([stats.compute_metrics(wt[lib == i], wt[lib == i], shifted[lib == i], gt[lib == i], st[lib == i])['rho_combined'] for i in range(3)])
    pooled2 = pooled_stats.pooled_rho(wt, shifted, gt, st)['rho_combined_pooled']
    assert abs(per - per2) < 1e-9 and pooled2 < pooled - 0.2


def test_population_is_the_canonical_measured_one():
    wt, comb, gt, st, _ = _data()
    src = np.zeros(len(gt), dtype=int)
    src[:40] = censoring.SRC_RANGE                                         # rows the assay only bounded: held back
    gt2 = gt.copy(); gt2[40:60] = np.nan                                   # no label: never scored
    out = pooled_stats.pooled_rho(wt, comb, gt2, st, src)
    keep = np.arange(len(gt)) >= 60
    assert abs(out['rho_combined_pooled'] - spearmanr(comb[keep], gt[keep])[0]) < 1e-12
    st2 = np.where(st == 'single', 'native_cond', st)                       # relabelled as singles, as validation does
    assert abs(pooled_stats.pooled_rho(wt, comb, gt, st2)['rho_singles_wt_pooled'] - pooled_stats.pooled_rho(wt, comb, gt, st)['rho_singles_wt_pooled']) < 1e-12


def test_undefined_is_nan():
    out = pooled_stats.pooled_rho([1.0], [1.0], [1.0], ['single'])
    assert np.isnan(out['rho_combined_pooled']) and np.isnan(out['rho_singles_wt_pooled'])
