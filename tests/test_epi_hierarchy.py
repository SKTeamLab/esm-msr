"""The epistasis hierarchy on synthetic position-pair matrices with known structure."""
import unittest

import numpy as np

from esm_msr import epi_hierarchy as H

AA = list('ACDEFGHIKLMNPQRSTVWY')


def build(n_pairs=12, seed=0, n_rows=8, n_cols=8, pair_sd=0.0, row_sd=0.0, col_sd=0.0, table_sd=0.0, noise_sd=0.0):
    """label[a, b] = pair offset + row effect + column effect + shared residue-pair table + noise, for n_pairs position pairs."""
    rng = np.random.default_rng(seed)
    table = rng.normal(0, table_sd, (20, 20))
    keys, label, parts = [], [], []
    for p in range(n_pairs):
        rows = rng.choice(20, n_rows, replace=False); cols = rng.choice(20, n_cols, replace=False)
        g = rng.normal(0, pair_sd); r = rng.normal(0, row_sd, 20); c = rng.normal(0, col_sd, 20)
        for a in rows:
            for b in cols:
                keys.append((('lib', 'X', 10 + p, AA[a]), ('lib', 'Y', 50 + p, AA[b])))
                parts.append((g, r[a], c[b], table[a, b]))
                label.append(g + r[a] + c[b] + table[a, b] + rng.normal(0, noise_sd))
    return keys, np.array(label), np.array(parts)


class TestHierarchy(unittest.TestCase):
    def run_levels(self, pred, label, keys):
        return H.compute(pred, label, keys, np.ones(len(label), bool))

    def test_a_perfect_prediction_scores_one_at_every_level_and_zero_error(self):
        keys, y, _ = build(pair_sd=0.5, row_sd=0.4, col_sd=0.4, table_sd=0.3)
        o = self.run_levels(y.copy(), y, keys)
        self.assertAlmostEqual(o['global_rmse'], 0.0, places=9)
        for lvl in H.LEVELS[1:]:
            self.assertAlmostEqual(o[lvl], 1.0, places=6, msg=lvl)

    def test_a_pair_mean_only_prediction_scores_on_the_pair_effect_and_nothing_below(self):
        keys, y, parts = build(pair_sd=0.8, row_sd=0.4, col_sd=0.4, table_sd=0.3)
        pred = parts[:, 0]
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['pair_effect'], 0.95)
        for lvl in ('matrix_rank', 'partner_effect', 'partner_context_rank', 'identity_effect',
                    'interaction_rank_ranked_per_matrix', 'interaction_rank_raw_pooled'):
            self.assertAlmostEqual(o[lvl], 0.0, places=9, msg=lvl)             # flat inside every matrix: no information, not NaN

    def test_an_additive_prediction_has_no_interaction_but_ranks_the_columns_and_matrix(self):
        keys, y, parts = build(pair_sd=0.5, row_sd=0.6, col_sd=0.6, table_sd=0.15)
        pred = parts[:, 0] + parts[:, 1] + parts[:, 2]                       # the whole additive part, none of the interaction
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['matrix_rank'], 0.9)
        self.assertGreater(o['partner_context_rank'], 0.7)
        for lvl in ('interaction_rank_ranked_pooled', 'interaction_rank_raw_pooled', 'interaction_rank_raw_per_matrix', 'identity_effect'):
            self.assertAlmostEqual(o[lvl], 0.0, places=9, msg=lvl)

    def test_the_shared_residue_pair_table_is_found_by_the_identity_effect_and_the_raw_interaction_rank(self):
        keys, y, parts = build(n_pairs=30, n_rows=12, n_cols=12, pair_sd=0.3, row_sd=0.5, col_sd=0.5, table_sd=0.6)
        pred = parts[:, 3]                                                   # only the residue-pair table
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['identity_effect'], 0.8)
        self.assertGreater(o['interaction_rank_raw_pooled'], 0.6)
        self.assertAlmostEqual(o['pair_effect'], o['pair_effect'])           # defined (no NaN level that should exist)
        self.assertFalse(np.isnan(o['interaction_rank_ranked_per_matrix']))

    def test_partner_effect_is_computed_on_means_minus_the_matrix_mean(self):
        keys, y, parts = build(pair_sd=2.0, row_sd=0.0, col_sd=0.7, table_sd=0.0)
        o = self.run_levels(parts[:, 2] + parts[:, 0], y, keys)              # column effect (+ the large pair offset)
        self.assertGreater(o['partner_effect'], 0.9)
        o2 = self.run_levels(parts[:, 0], y, keys)                           # the pair offset alone must not leak into the partner effect
        self.assertAlmostEqual(o2['partner_effect'], 0.0, places=9)

    def test_noise_lowers_the_cell_levels_more_than_the_pooled_ones(self):
        keys, y, parts = build(pair_sd=0.6, row_sd=0.3, col_sd=0.3, table_sd=0.1, noise_sd=0.5)
        o = self.run_levels(parts.sum(1), y, keys)                           # exact structure, noisy label
        self.assertGreater(o['pair_effect'], 0.8)
        self.assertLess(o['partner_context_rank'], o['pair_effect'])
        self.assertGreater(o['global_rmse'], 0.4)

    def test_undersized_inputs_give_nan_not_errors(self):
        keys, y, _ = build(n_pairs=2, n_rows=3, n_cols=3, row_sd=0.3)         # 9 cells per pair: below the matrix minimum
        o = self.run_levels(y.copy(), y, keys)
        self.assertEqual(o['n_pairs'], 0)
        self.assertTrue(np.isnan(o['pair_effect'])); self.assertAlmostEqual(o['global_rho'], 1.0)

    def test_censored_or_missing_items_are_left_out(self):
        keys, y, _ = build(pair_sd=0.5, row_sd=0.4)
        pred = y.copy(); pred[::7] = np.nan
        ok = np.ones(len(y), bool); ok[3::11] = False
        o = H.compute(pred, y, keys, ok)
        self.assertLess(o['n_doubles'], len(y)); self.assertAlmostEqual(o['global_rho'], 1.0)


if __name__ == '__main__':
    unittest.main()


class TestPredictedDddG(unittest.TestCase):
    def test_second_difference_of_a_head_within_a_library(self):
        m = lambda lib, p, r: (lib, 'X', p, r)
        keys = [(m('L', 1, 'A'),), (m('L', 2, 'C'),), (m('L', 1, 'A'), m('L', 2, 'C')), (m('M', 1, 'A'), m('M', 2, 'C')), (m('L', 1, 'A'), m('L', 9, 'G'))]
        n = [1, 1, 2, 2, 2]
        out = H.predicted_dddG([1.0, 2.0, 5.0, 9.0, 7.0], n, keys)
        self.assertEqual(out[2], 5.0 - 1.0 - 2.0)
        self.assertTrue(np.isnan(out[:2]).all())           # singles have none
        self.assertTrue(np.isnan(out[3]))                  # the library M has no singles
        self.assertTrue(np.isnan(out[4]))                  # one of its singles is absent

    def test_without_keys_everything_is_nan(self):
        self.assertTrue(np.isnan(H.predicted_dddG([1.0, 2.0], [1, 2], None)).all())
