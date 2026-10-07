"""on_validation_epoch_end with the link on and a loader whose batches carry no dG_wt (no observed-scale outputs).

Regression test for a crash: the pooled observed-scale arrays were built only from loaders that had them, so their length
differed from the pooled latent arrays (ValueError at the end of the first validation).
"""
import types
import unittest
from collections import defaultdict

import numpy as np

from esm_msr import training


def outputs(n, with_obs, rng):
    o = {'wt_scores': rng.normal(size=n), 'mt_scores': rng.normal(size=n), 'comb_scores': rng.normal(size=n),
         'ground_truths': rng.normal(size=n), 'dddG': np.full(n, np.nan), 'cens': np.zeros(n, dtype=int),
         'subset_type': ['single'] * n, 'flip_key': [''] * n, 'row_id': np.full(n, -1), 'mut_key': [((('A', 1, 'C'),)[0],)] * n}
    if with_obs:
        for k in ('wt', 'mt', 'comb'):
            o[f'{k}_obs'] = rng.normal(size=n)
    return o


class Stub:
    def __init__(self, link):
        self.link_head = link
        self.hparams = types.SimpleNamespace(flip_list_min=4)
        self.val_dataloader_names = ['with_obs', 'without_obs']
        self.logged = {}
        for k in ('_VAL_PER_PROTEIN', '_VAL_AVG', '_VAL_POOLED', '_VAL_PROGRESS_BAR'):
            setattr(self, k, getattr(training.ESM3EpistasisLightningModule, k))
        self.trainer = types.SimpleNamespace(sanity_checking=True, current_epoch=0)

    def log(self, name, value, **kw):
        self.logged[name] = value


class TestPooling(unittest.TestCase):
    def test_mixed_loaders_do_not_crash(self):
        rng = np.random.default_rng(0)
        s = Stub(link=object())
        s.validation_step_outputs = defaultdict(list, {0: [outputs(30, True, rng)], 1: [outputs(25, False, rng)]})
        training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
        self.assertIn('val_rmse_combined_pooled', s.logged)

    def test_logged_names_are_the_trimmed_set(self):
        rng = np.random.default_rng(1)
        s = Stub(link=None)
        s.validation_step_outputs = defaultdict(list, {0: [outputs(40, False, rng)], 1: [outputs(30, False, rng)]})
        training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
        names = set(s.logged)
        per_protein = {n for n in names if '/' in n}
        self.assertTrue(per_protein <= {f'val_{m}/{lib}' for m in ('rho_combined', 'rmse_combined', 'rho_wt_valid', 'rho_flip_pair_mt')
                                        for lib in ('with_obs', 'without_obs')}, per_protein)
        for banned in ('auc_hyper', 'rho_epi_fast', 'rho_wt_all', 'rho_mt_all', 'val_flip_pairs', 'val_flip_pair_matrices/'):
            self.assertFalse(any(banned in n for n in names), banned)
        for needed in ('val_rho_combined_avg', 'val_rho_wt_valid_avg', 'val_rmse_combined_pooled', 'val_n_flip_pair_matrices'):
            self.assertIn(needed, names)


if __name__ == '__main__':
    unittest.main()
