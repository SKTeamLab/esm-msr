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
            'dG_wt': np.full(n, 2.0), 'wt_scores': np.arange(n) * 0.1, 'mt_scores': np.arange(n) * 0.2, 'comb_scores': np.arange(n) * 0.15,
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


def test_the_observed_scale_link_is_the_tobit_expected_reading_with_the_runs_constants():
    hp = lambda **k: types.SimpleNamespace(get=lambda key, d=None: k.get(key, d))
    h = training.observed_scale_link(hp())
    ref = val_stats.tobit_link(0.28, 0.05, 0.38)
    z = np.linspace(-2.0, 6.0, 17)
    np.testing.assert_allclose(h(z), ref(z), rtol=0, atol=0)
    wide = training.observed_scale_link(hp(tobit_sigma_f=0.8))               # the constants are the run's own flags
    assert not np.allclose(wide(z), ref(z))
