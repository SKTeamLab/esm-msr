"""Validation hands the epistasis table the run's observed-scale link: the Tobit expected reading for a Tobit run, nothing for any other.

Only the observed-scale metrics (naive_rho) read it; the residual, pair, line, cell and xgt1 metrics read the latent scores, so a monotone link leaves them
alone (tests/test_epi_metrics.py: test_the_rank_metrics_ignore_a_monotone_measurement).
"""
import types

import numpy as np

from esm_msr import epi_metrics, training, val_stats
from esm_msr.training import ESM3EpistasisLightningModule as M


def _pooled():
    n = 6
    keys = [(('L', 1, 'A'),), (('L', 2, 'A'),), (('L', 1, 'A'), ('L', 2, 'A')), (('L', 1, 'G'),), (('L', 2, 'G'),), (('L', 1, 'G'), ('L', 2, 'G'))]
    z = np.zeros(n)
    return {'mut_key': keys, 'subset_type': ['single', 'single', 'double'] * 2, 'cens': z.astype(int), 'cens_src': z.astype(int),
            'ground_truths': np.array([0.5, -0.2, 0.1, 0.3, 0.2, 0.9]), 'dddG': np.array([np.nan, np.nan, -0.2, np.nan, np.nan, 0.4]),
            'dG_wt': np.full(n, 2.0), 'y_floor': np.zeros(n), 'wt_scores': np.arange(n) * 0.1, 'mt_scores': np.arange(n) * 0.2, 'comb_scores': np.arange(n) * 0.15,
            'wt_rev_scores': z, 'n_mut': np.array([1, 1, 2] * 2), 'loader': ['L'] * n}


def _tables_seen(monkeypatch, obs_link):
    seen = []
    real = epi_metrics.Table
    monkeypatch.setattr(epi_metrics, 'Table', lambda items, link=None, **kw: (seen.append(link), real(items, link, **kw))[1])
    stub = types.SimpleNamespace(hparams=types.SimpleNamespace(val_dump=False), logger=None, obs_link=obs_link, log=lambda *a, **k: None)
    M._log_epistasis(stub, _pooled(), dump=False)
    return seen


def test_a_run_that_did_not_train_with_tobit_scores_epistasis_as_it_is(monkeypatch):
    assert _tables_seen(monkeypatch, None) == [None]


def test_a_tobit_run_hands_the_table_its_link(monkeypatch):
    link = val_stats.tobit_link(0.28, 0.05, 0.38)
    assert _tables_seen(monkeypatch, link) == [link]


def test_the_link_is_the_runs_only_when_it_trained_with_tobit_and_did_not_opt_out():
    hp = lambda **k: types.SimpleNamespace(get=lambda key, d=None: k.get(key, d))
    assert training.observed_scale_link(hp()) is None
    assert training.observed_scale_link(hp(reg_nll='none')) is None
    assert training.observed_scale_link(hp(reg_nll='tobit', val_tobit_link=False)) is None
    h = training.observed_scale_link(hp(reg_nll='tobit'))
    ref = val_stats.tobit_link(0.28, 0.05, 0.38)
    z = np.linspace(-2.0, 6.0, 17)
    np.testing.assert_allclose(h(z), ref(z), rtol=0, atol=0)
    wide = training.observed_scale_link(hp(reg_nll='tobit', tobit_sigma_f=0.8))               # the constants are the run's own flags
    assert not np.allclose(wide(z), ref(z))
