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
    def run_levels(self, pred, label, keys, **controls):
        return H.compute(pred, label, keys, np.ones(len(label), bool), **controls)

    def test_a_perfect_prediction_scores_one_at_every_level_and_zero_error(self):
        keys, y, _ = build(pair_sd=0.5, row_sd=0.4, col_sd=0.4, table_sd=0.3)
        o = self.run_levels(y.copy(), y, keys)
        self.assertAlmostEqual(o['global_rmse'], 0.0, places=9)
        for lvl in H.LEVELS[1:]:
            if not lvl.endswith(('_beyond_add', '_beyond_single')):           # the partial levels need their control
                self.assertAlmostEqual(o[lvl], 1.0, places=6, msg=lvl)
            else:
                self.assertTrue(np.isnan(o[lvl]), lvl)

    def test_a_perfect_prediction_stays_perfect_given_any_control(self):
        keys, y, parts = build(pair_sd=0.5, row_sd=0.4, col_sd=0.4, table_sd=0.3, noise_sd=0.2)
        rng = np.random.default_rng(5)
        single = {k[1]: v for k, v in zip(keys, rng.normal(size=len(keys)))}
        o = self.run_levels(y.copy(), y, keys, additive=rng.normal(size=len(y)) + 0.3 * y, single_ddG=single)
        for lvl in ('global_rho_beyond_add', 'matrix_rank_beyond_add', 'partner_effect_beyond_single'):
            self.assertAlmostEqual(o[lvl], 1.0, places=6, msg=lvl)

    def test_a_pair_mean_only_prediction_scores_on_the_pair_effect_and_nothing_below(self):
        keys, y, parts = build(pair_sd=0.8, row_sd=0.4, col_sd=0.4, table_sd=0.3)
        pred = parts[:, 0]
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['pair_effect'], 0.95)
        for lvl in ('matrix_rank', 'partner_effect', 'partner_context_rank', 'identity_effect',
                    'interaction_rank_ranked_per_matrix', 'interaction_rank_raw_per_matrix'):
            self.assertAlmostEqual(o[lvl], 0.0, places=9, msg=lvl)             # flat inside every matrix: no information, not NaN

    def test_an_additive_prediction_has_no_interaction_but_ranks_the_columns_and_matrix(self):
        keys, y, parts = build(pair_sd=0.5, row_sd=0.6, col_sd=0.6, table_sd=0.15)
        pred = parts[:, 0] + parts[:, 1] + parts[:, 2]                       # the whole additive part, none of the interaction
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['matrix_rank'], 0.9)
        self.assertGreater(o['partner_context_rank'], 0.7)
        for lvl in ('interaction_rank_ranked_per_matrix', 'interaction_rank_raw_per_matrix', 'identity_effect'):
            self.assertAlmostEqual(o[lvl], 0.0, places=9, msg=lvl)

    def test_the_shared_residue_pair_table_is_found_by_the_identity_effect_and_the_raw_interaction_rank(self):
        keys, y, parts = build(n_pairs=30, n_rows=12, n_cols=12, pair_sd=0.3, row_sd=0.5, col_sd=0.5, table_sd=0.6)
        pred = parts[:, 3]                                                   # only the residue-pair table
        o = self.run_levels(pred, y, keys)
        self.assertGreater(o['identity_effect'], 0.8)
        self.assertGreater(o['interaction_rank_raw_per_matrix'], 0.6)
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


class TestPartialLevels(unittest.TestCase):
    """*_beyond_add and *_beyond_single: the skill that is left once a confounder is held fixed."""

    def test_partial_rho_is_the_correlation_of_rank_residuals_and_handles_degenerate_inputs(self):
        from scipy.stats import pearsonr, rankdata
        rng = np.random.default_rng(3)
        z = rng.normal(size=400); x = z + rng.normal(size=400); y = -z + 0.5 * x + rng.normal(size=400)
        A = np.vstack([rankdata(z), np.ones(400)]).T
        res = lambda v: rankdata(v) - A @ np.linalg.lstsq(A, rankdata(v), rcond=None)[0]
        self.assertAlmostEqual(H._partial_rho(x, y, z), pearsonr(res(x), res(y))[0], places=9)
        self.assertAlmostEqual(H._partial_rho(np.zeros(400), y, z), 0.0)               # flat predictor: no information
        self.assertTrue(np.isnan(H._partial_rho(x, np.zeros(400), z)))                # flat label
        self.assertAlmostEqual(H._partial_rho(x, y, np.zeros(400)), H._rho(x, y))     # flat control removes nothing
        self.assertTrue(np.isnan(H._partial_rho(z, y, 2 * z + 1)))                    # predictor IS the control: nothing left

    def test_a_predictor_that_only_tracks_the_control_loses_its_skill(self):
        keys, y, parts = build(n_pairs=14, n_rows=9, n_cols=9, pair_sd=0.2, row_sd=0.5, col_sd=0.5)
        rng = np.random.default_rng(1)
        sat = rng.normal(size=len(y))                                   # stand-in for the additive score
        label = y + 1.5 * sat                                           # the label is mostly the control, plus structure
        only_control = sat + rng.normal(0, 0.05, len(y))
        o = H.compute(only_control, label, keys, np.ones(len(y), bool), additive=sat)
        self.assertGreater(o['global_rho'], 0.7); self.assertLess(abs(o['global_rho_beyond_add']), 0.3)
        self.assertGreater(o['matrix_rank'], 0.7); self.assertLess(abs(o['matrix_rank_beyond_add']), 0.3)

    def test_a_predictor_with_independent_skill_keeps_it(self):
        keys, y, parts = build(n_pairs=14, n_rows=9, n_cols=9, pair_sd=0.2, row_sd=0.5, col_sd=0.5)
        rng = np.random.default_rng(2)
        sat = rng.normal(size=len(y))
        label = y + 1.5 * sat
        pred = y + rng.normal(0, 0.1, len(y))                           # knows the structure, ignores the control
        o = H.compute(pred, label, keys, np.ones(len(y), bool), additive=sat)
        self.assertGreater(o['global_rho_beyond_add'], o['global_rho'] - 0.05)
        self.assertGreater(o['matrix_rank_beyond_add'], 0.5)

    def test_the_partner_effect_is_partialled_on_the_single_of_the_fixed_mutation_in_both_orientations(self):
        for col_not_row in (True, False):
            keys, _, _ = build(n_pairs=14, n_rows=9, n_cols=9)
            f = 1 if col_not_row else 0                                  # the fixed mutation of a column is at the HIGHER position
            rng = np.random.default_rng(4)
            e1 = {k[f]: rng.normal() for k in keys}                      # the part of a mutation's effect its single ddG shows
            e2 = {k[f]: rng.normal() for k in keys}                      # the part it does not
            label = np.array([e1[k[f]] + e2[k[f]] for k in keys])
            ok = np.ones(len(keys), bool)
            only_single = np.array([e1[k[f]] for k in keys]) + rng.normal(0, 0.02, len(keys))
            only_rest = np.array([e2[k[f]] for k in keys]) + rng.normal(0, 0.02, len(keys))
            a = H.compute(only_single, label, keys, ok, single_ddG=e1)
            b = H.compute(only_rest, label, keys, ok, single_ddG=e1)
            self.assertGreater(a['partner_effect'], 0.4, col_not_row)
            self.assertTrue(np.isfinite(a['partner_effect_beyond_single']), col_not_row)       # the fixed mutation was found
            self.assertLess(abs(a['partner_effect_beyond_single']), 0.25, col_not_row)         # all it knew was the single
            self.assertGreater(b['partner_effect_beyond_single'], b['partner_effect'], col_not_row)   # what the single does not explain is kept

    def test_levels_without_their_control_are_nan(self):
        keys, y, _ = build(row_sd=0.4)
        o = H.compute(y.copy(), y, keys, np.ones(len(y), bool))
        for lvl in ('global_rho_beyond_add', 'matrix_rank_beyond_add', 'partner_effect_beyond_single'):
            self.assertTrue(np.isnan(o[lvl]), lvl)


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
