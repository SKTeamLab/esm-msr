"""Tests for msr_analysis.stats, msr_analysis.calib, and msr_analysis.tobit."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msr_analysis import calib, stats, tobit  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402


def test_holm_step_down_and_nan_preservation():
    raw = np.array([0.04, np.nan, 0.01, 0.20, 0.005])
    adj = stats.holm(raw)
    assert np.isnan(adj[1])
    # Finite p-values sorted: [0.005 (x4=0.02), 0.01 (x3=0.03), 0.04 (x2=0.08), 0.20 (x1=0.20)]
    assert adj[4] == pytest.approx(0.02)
    assert adj[2] == pytest.approx(0.03)
    assert adj[0] == pytest.approx(0.08)
    assert adj[3] == pytest.approx(0.20)
    # Enforce step-down monotonicity when a later multiplied p is smaller than an earlier one
    mono = stats.holm([0.01, 0.012, 0.5])
    assert mono[0] <= mono[1] <= mono[2] <= 1.0


def test_boot_pvalue_floor_and_sidedness():
    draws = np.ones(199) * 0.5
    assert stats.boot_pvalue(draws, null=0.0, alternative='two-sided') == pytest.approx(1.0 / 200.0)
    assert stats.boot_pvalue(draws, null=0.0, alternative='greater') == pytest.approx(1.0 / 200.0)
    assert stats.boot_pvalue(draws, null=0.0, alternative='less') == pytest.approx(1.0)

    mixed = np.array([-1.0] * 10 + [1.0] * 90)
    assert stats.boot_pvalue(mixed, null=0.0, alternative='greater') == pytest.approx(0.10)
    assert stats.boot_pvalue(mixed, null=0.0, alternative='two-sided') == pytest.approx(0.20)


def test_fmt_p_and_seed_sd_ddof1():
    assert stats.fmt_p(np.nan) == 'n/a'
    assert stats.fmt_p(1.0 / 1001.0, B=1000) == 'p < 1.0e-03'
    assert stats.fmt_p(1.0 / 20001.0, B=20000) == 'p < 5.0e-05'
    assert stats.fmt_p(0.0234) == 'p = 0.023'
    assert stats.fmt_p(0.346) == 'p = 0.35'

    vals = [0.80, 0.82, 0.84]
    assert stats.seed_sd(vals) == pytest.approx(np.std(vals, ddof=1))
    assert np.isnan(stats.seed_sd([0.80]))


def test_paired_boot_test_and_boot_domain_seeds():
    ra = np.array([0.80, 0.83, 0.79, 0.86, 0.82])
    rb = np.array([0.70, 0.71, 0.69, 0.72, 0.70])
    # ra - rb = [0.10, 0.12, 0.10, 0.14, 0.12] (boot mean = 0.116), while point_a - point_b = 0.125
    res = stats.paired_boot_test(ra, rb, point_a=0.830, point_b=0.705)
    assert res['delta'] == pytest.approx(0.125)
    assert res['delta'] != pytest.approx(float((ra - rb).mean()))
    assert res['lo'] <= res['delta'] <= res['hi']
    assert res['n_boot'] == 5

    rng = np.random.default_rng(3)
    dom_mat = rng.normal(loc=0.65, scale=0.08, size=(25, 3))
    # Inject seed-to-seed offset so seed resampling measurably contributes variance
    dom_mat[:, 0] -= 0.05
    dom_mat[:, 2] += 0.05
    pt, reps = stats.boot_domain_seeds(dom_mat, B=300, seed=5)
    assert pt == pytest.approx(dom_mat.mean())
    assert reps.shape == (300,) and np.isfinite(reps).all()
    # Compare against domain-only bootstrap on seed mean: joint [domain x seed] has strictly larger SD
    _, reps_dom_only = stats.boot_domain_seeds(dom_mat.mean(axis=1), B=300, seed=5)
    assert np.std(reps) > np.std(reps_dom_only)


def test_ndcg_at_k_rectified_and_shifted():
    y_neg = np.array([-3.0, -2.0, -1.0, -0.5])
    pred = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isnan(stats.ndcg_at_k(pred, y_neg, k=2, mode='rectified'))
    assert stats.ndcg_at_k(pred, y_neg, k=2, mode='shifted') == pytest.approx(1.0)

    y_pos = np.array([-1.0, 0.2, 0.8, 1.5])
    assert stats.ndcg_at_k(y_pos, y_pos, k=2, mode='rectified') == pytest.approx(1.0)


def test_calib_selects_deployed_head_per_dataset():
    sp_single = pd.DataFrame({'n_mut': [1, 1, 1, 1], 'use': [True] * 4, 'y': [-1.0, 0.0, 1.0, 2.0]})
    fr_single = Frame('singles_ds', sp_single, {
        'msrp0:wt': np.array([[-0.5, 0.0, 0.5, 1.0]]),
        'msrp0:comb': np.array([[99.0, 99.0, 99.0, 99.0]]),  # wrong head if picked on singles
    })
    assert calib.calibration_key('msrp0', fr_single) == 'msrp0:wt'
    ab_s = calib.fit_frame_scale_bias(fr_single, 'msrp0')
    assert ab_s is not None and ab_s[0][0] == pytest.approx(2.0) and ab_s[0][1] == pytest.approx(0.0)

    sp_multi = pd.DataFrame({'n_mut': [1, 1, 2, 2], 'use': [True] * 4, 'y': [-2.0, 0.0, 2.0, 4.0]})
    fr_multi = Frame('multi_ds', sp_multi, {
        'msrp0:wt': np.array([[50.0, -50.0, 12.0, -12.0]]),  # wrong head if picked on multi-mutants
        'msrp0:comb': np.array([[-1.0, 0.0, 1.0, 2.0]]),
    })
    assert calib.calibration_key('msrp0', fr_multi) == 'msrp0:comb'
    cal_multi = calib.calibrated_preds(fr_multi, 'msrp0', ab_s)
    assert cal_multi is not None
    assert calib.rmse(cal_multi[0], sp_multi['y'].to_numpy()) == pytest.approx(0.0, abs=1e-12)


def test_tobit_linked_and_analytical_floor_inverses():
    dg_wt = np.array([1.5, 2.0, 3.0, 0.8])
    lat_1d = np.array([-2.0, -0.5, 0.0, 1.2])
    lat_2d = np.vstack([lat_1d, lat_1d * 0.5])
    out_1d = tobit.linked(lat_1d, dg_wt)
    out_2d = tobit.linked(lat_2d, dg_wt)
    assert out_1d.shape == (4,) and out_2d.shape == (2, 4)
    assert np.allclose(out_2d[0], out_1d)

    z = np.linspace(0.5, 5.0, 30)
    y_sp = tobit.softplus_link(z, floor=0.0, tau=0.59)
    assert np.allclose(tobit.inv_softplus_link(y_sp, floor=0.0, tau=0.59), z, atol=1e-6)

    y_lk = tobit.leaky_link(z, knee=0.0, tau=0.59, alpha=0.10, clamp=-1.0, width=0.25)
    assert np.allclose(tobit.inv_leaky_link(y_lk, knee=0.0, tau=0.59, alpha=0.10, clamp=-1.0, width=0.25), z, atol=1e-6)
