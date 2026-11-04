"""Validation scores epistasis on the latent scale: the run's link is not passed to the epistasis table."""
import types

import numpy as np

from esm_msr import epi_metrics
from esm_msr.training import ESM3EpistasisLightningModule as M


def _pooled():
    n = 6
    keys = [(('L', 1, 'A'),), (('L', 2, 'A'),), (('L', 1, 'A'), ('L', 2, 'A')), (('L', 1, 'G'),), (('L', 2, 'G'),), (('L', 1, 'G'), ('L', 2, 'G'))]
    z = np.zeros(n)
    return {'mut_key': keys, 'subset_type': ['single', 'single', 'double'] * 2, 'cens': z.astype(int), 'cens_src': z.astype(int),
            'ground_truths': np.array([0.5, -0.2, 0.1, 0.3, 0.2, 0.9]), 'dddG': np.array([np.nan, np.nan, -0.2, np.nan, np.nan, 0.4]),
            'dG_wt': np.full(n, 2.0), 'y_floor': np.zeros(n), 'wt_scores': np.arange(n) * 0.1, 'mt_scores': np.arange(n) * 0.2, 'comb_scores': np.arange(n) * 0.15,
            'wt_rev_scores': z, 'n_mut': np.array([1, 1, 2] * 2), 'loader': ['L'] * n}


def test_the_epistasis_table_gets_no_link(monkeypatch):
    seen = []
    real = epi_metrics.Table
    monkeypatch.setattr(epi_metrics, 'Table', lambda items, link=None, **kw: (seen.append(link), real(items, link, **kw))[1])

    def forbidden():
        raise AssertionError('the epistasis scoring read the run link')
    stub = types.SimpleNamespace(hparams=types.SimpleNamespace(val_dump=False), logger=None, link_head=types.SimpleNamespace(numpy=forbidden, summary=forbidden),
                                 log=lambda *a, **k: None)
    M._log_epistasis(stub, _pooled(), dump=False)
    assert seen == [None]
