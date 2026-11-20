"""scripts/revision_ablations_rescore.py and the pure parts of scripts/revision_ablations_comet.py, on a synthetic validation dump."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import revision_ablations_comet as C  # noqa: E402
import revision_ablations_rescore as R  # noqa: E402


def _dump(path, kind='tobit', seed=0):
    rng = np.random.default_rng(seed)
    n, libs = 120, ['L1', 'L2']
    gt = rng.normal(size=n)
    wt = gt + rng.normal(scale=0.4, size=n) - 1.0                      # latent below the observed floor: the link matters
    mt = gt + rng.normal(scale=0.4, size=n) - 1.0
    st = np.where(np.arange(n) % 3 == 0, 'double', 'single')
    keys = [json.dumps([[libs[i % 2], i, 'A']] if s == 'single' else [[libs[i % 2], i, 'A'], [libs[i % 2], i + 500, 'C']]) for i, s in enumerate(st)]
    ls = {'kind': 'tobit', 'sigma_y': 0.28, 'mu_f': -0.14, 'sigma_f': 0.5} if kind == 'tobit' else {'kind': 'none'}
    np.savez(path, wt=wt, mt=mt, comb=0.5 * (wt + mt), dddG=np.full(n, np.nan), ground_truth=gt, dG_wt=np.full(n, 1.5), cens=np.zeros(n, int), cens_src=np.zeros(n, int),
             n_mut=np.where(st == 'single', 1, 2), subset_type=st, loader=np.array([libs[i % 2] for i in range(n)]), mut_key=np.array(keys), link_summary=np.array(json.dumps(ls)))


def test_the_link_changes_only_the_observed_scale_statistics_and_none_is_the_identity(tmp_path):
    p = tmp_path / 'val_dump_e0.npz'
    _dump(p)
    tob = R.rescore_dump(p, 'summary')
    none = R.rescore_dump(p, 'none')
    assert set(tob) == set(none)
    for k in tob:
        if R.affected(k):
            assert abs(tob[k] - none[k]) > 1e-3, k                         # RMSE and naive_rho_comb move with the link
        else:
            assert tob[k] == pytest.approx(none[k], abs=1e-12), k          # ranks, ndcg and the pooled correlations do not
    assert 'val_rho_combined_pooled' in none and 'val_rho_singles_wt_pooled' in none
    assert none['val_rmse_combined_avg'] > tob['val_rmse_combined_avg']      # latent predictions below the floor are not forgiven without the link


def test_auto_follows_the_regression_loss(tmp_path):
    p = tmp_path / 'val_dump_e0.npz'
    _dump(p)
    assert R.rescore_dump(p, 'auto', 'tobit') == R.rescore_dump(p, 'summary')
    assert R.rescore_dump(p, 'auto', 'none') == R.rescore_dump(p, 'none')


def test_an_older_dump_without_subset_type_or_cens_src_still_scores(tmp_path):
    p = tmp_path / 'val_dump_e0.npz'
    _dump(p, kind='none')
    z = dict(np.load(p, allow_pickle=True))
    for k in ('subset_type', 'cens_src', 'dG_wt'):
        z.pop(k)
    np.savez(p, **z)
    out = R.rescore_dump(p, 'auto', 'tobit')
    assert np.isfinite(out['val_rho_combined_pooled']) and np.isfinite(out['val_rho_combined_avg'])


def test_corrected_points_replace_only_what_changed_and_pooled_points_are_added():
    old = [{'metricName': 'val_rmse_combined_avg', 'step': 7, 'epoch': 0, 'metricValue': 1.0}, {'metricName': 'val_rho_combined_avg', 'step': 7, 'epoch': 0, 'metricValue': 0.8},
           {'metricName': 'val_rmse_combined_avg', 'step': 7, 'epoch': 0, 'metricValue': 1.0}]                  # a duplicated point of the original
    r = {'corrected': [{'metric': 'val_rmse_combined_avg', 'step': 7, 'epoch': 0, 'old': 1.0, 'new': 1.7}], 'new': [{'metric': 'val_rho_combined_pooled', 'step': 7, 'epoch': 0, 'value': 0.5}]}
    got = {(p['metricName'], p['step']): p['metricValue'] for p in C.desired_points(old, r)}
    assert got == {('val_rmse_combined_avg', 7): 1.7, ('val_rho_combined_avg', 7): 0.8, ('val_rho_combined_pooled', 7): 0.5}
    assert len(C.desired_points(old, r)) == 3
    assert C.aligned(r, old) == (True, [])
    r['new'][0]['step'] = 8
    assert C.aligned(r, old) == (False, [8])
