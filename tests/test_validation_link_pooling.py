"""on_validation_epoch_end: what is logged, with and without the link, the cycle passes and the observed-scale outputs.

Regression test for a crash: the pooled observed-scale arrays were built only from loaders that had them, so their length differed from the
pooled latent arrays (ValueError at the end of the first validation).
"""
import types
import unittest
from collections import defaultdict

import numpy as np

from esm_msr import epi_metrics, training
from esm_msr.link import MonotoneLink


def outputs(n, with_obs, rng):
    o = {'wt_scores': rng.normal(size=n), 'mt_scores': rng.normal(size=n), 'comb_scores': rng.normal(size=n),
         'ground_truths': rng.normal(size=n), 'dddG': np.full(n, np.nan), 'dG_wt': np.full(n, 2.0), 'cens': np.zeros(n, dtype=int),
         'subset_type': ['single'] * n, 'flip_key': [''] * n, 'mut_key': [(('A', i, 'C'),) for i in range(n)]}
    if with_obs:
        o['comb_obs'] = rng.normal(size=n)
    return o


class Stub:
    def __init__(self, link):
        self.link_head = link
        self.hparams = types.SimpleNamespace(flip_list_min=4)
        self.val_dataloader_names = ['with_obs', 'without_obs']
        self.logged = {}
        for k in ('_VAL_PER_PROTEIN', '_VAL_AVG', '_VAL_PROGRESS_BAR'):
            setattr(self, k, getattr(training.ESM3EpistasisLightningModule, k))
        self.trainer = types.SimpleNamespace(sanity_checking=True, current_epoch=0)

    def log(self, name, value, **kw):
        self.logged[name] = value


class TestPooling(unittest.TestCase):
    def test_mixed_loaders_do_not_crash(self):
        rng = np.random.default_rng(0)
        s = Stub(link=MonotoneLink())
        s.validation_step_outputs = defaultdict(list, {0: [outputs(30, True, rng)], 1: [outputs(25, False, rng)]})
        training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
        self.assertIn('val_rmse_combined_avg', s.logged)

    def test_logged_names_are_the_documented_set(self):
        rng = np.random.default_rng(1)
        s = Stub(link=None)
        s.validation_step_outputs = defaultdict(list, {0: [outputs(40, False, rng)], 1: [outputs(30, False, rng)]})
        training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
        names = set(s.logged)
        per_protein = {n for n in names if '/' in n}
        self.assertEqual(per_protein, {f'val_{m}/{lib}' for m in ('rho_combined', 'rmse_combined') for lib in ('with_obs', 'without_obs')})
        self.assertEqual(names - per_protein, {'val_rho_combined_avg', 'val_rmse_combined_avg', 'val_rho_wt_valid_avg',
                                               'val_epi_n_doubles', 'val_epi_n_pairs'})       # no doubles here: counts only


def library_with_doubles(rng, with_cycle, n_pairs=9):
    """n_pairs position pairs, each a 5 x 5 matrix of doubles with its 10 singles, in the shape validation_step produces."""
    A, B = list('ACDEF'), list('GHIKL')
    muts, sub = [], []
    for p in range(n_pairs):
        i, j = 10 * p + 1, 10 * p + 5
        muts += [(('X', i, a),) for a in A] + [(('Y', j, b),) for b in B]
        sub += ['single'] * 10
        for a in A:
            for b in B:
                muts.append((('X', i, a), ('Y', j, b))); sub.append('double')
    n = len(muts)
    gt = rng.normal(size=n)
    o = {'wt_scores': rng.normal(size=n), 'mt_scores': rng.normal(size=n), 'comb_scores': rng.normal(size=n),
         'ground_truths': gt, 'dddG': np.where(np.array(sub) == 'double', rng.normal(size=n), np.nan), 'dG_wt': np.full(n, 2.5),
         'cens': np.zeros(n, dtype=int), 'subset_type': sub, 'flip_key': [''] * n, 'mut_key': muts, 'comb_obs': rng.normal(size=n)}
    if with_cycle:
        o['wt_rev_scores'] = rng.normal(size=n)
    return o


class TestEpistasisLogging(unittest.TestCase):
    def run_epoch(self, with_cycle, link=True):
        rng = np.random.default_rng(1)
        s = Stub(link=MonotoneLink() if link else None)
        s.val_dataloader_names = ['lib']
        s.hparams = types.SimpleNamespace(flip_list_min=4, val_cycle_passes=with_cycle)
        s.validation_step_outputs = defaultdict(list, {0: [library_with_doubles(rng, with_cycle)]})
        training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
        return s.logged

    def test_every_documented_metric_is_logged_with_the_cycle_passes(self):
        logged = self.run_epoch(True)
        for name in epi_metrics.logged_names():
            if name not in ('epi_pair_rho_comb',):              # needs MIN_PAIRS matrices of MIN_PAIR_CELLS doubles
                self.assertIn(f'val_{name}', logged, name)
        self.assertEqual(logged['val_epi_n_doubles'], 9 * 25.0)

    def test_without_the_cycle_passes_only_the_ctx_metrics_are_missing(self):
        logged = self.run_epoch(False)
        self.assertFalse([k for k in logged if k.endswith('_ctx')])
        for name in ('val_epi_cell_rank_mt', 'val_epi_naive_rho_add', 'val_epi_global_err_comb'):
            self.assertIn(name, logged)

    def test_without_a_link_the_latent_is_the_observed_scale(self):
        logged = self.run_epoch(True, link=False)
        self.assertIn('val_epi_cell_rank_comb', logged)

    def test_the_retired_names_are_gone(self):
        logged = self.run_epoch(True)
        for gone in ('val_rho_flip_pair_mt_avg', 'val_rho_colrank_mt_avg', 'val_epi_global_rho_comb', 'val_epi_matrix_rank_comb',
                     'val_rho_mt_valid_avg', 'val_auc_dead_wt_avg', 'val_n_flip_pair_matrices', 'val_rho_combined_pooled'):
            self.assertNotIn(gone, logged)


if __name__ == '__main__':
    unittest.main()
