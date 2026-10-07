"""Pair offset, row / column effects and within-column rank agreement, on synthetic doubles with known structure."""
import unittest

import numpy as np

from esm_msr import stats


def make(n_pairs=12, rows=8, cols=8, seed=0):
    """Doubles over several position pairs; truth = pair offset + row effect + column effect + interaction + noise."""
    rng = np.random.default_rng(seed)
    keys, truth, parts = [], [], []
    for p in range(n_pairs):
        off = rng.normal(0, 1.0)
        r_eff = rng.normal(0, 0.7, rows)
        c_eff = rng.normal(0, 0.7, cols)
        inter = rng.normal(0, 0.3, (rows, cols))
        for i in range(rows):
            for j in range(cols):
                keys.append((('lib', 'A', 10 + p, 'X%d' % i), ('lib', 'B', 50 + p, 'Y%d' % j)))
                keys[-1] = (('lib', 'A', 10 + p, i), ('lib', 'B', 50 + p, j))
                truth.append(off + r_eff[i] + c_eff[j] + inter[i, j])
                parts.append((p, i, j, off, r_eff[i], c_eff[j], inter[i, j]))
    return keys, np.array(truth), parts


class TestEpiComponents(unittest.TestCase):
    def test_perfect_prediction_scores_one_on_every_component(self):
        keys, truth, _ = make()
        out = stats.epi_component_rhos(truth, truth, keys, np.ones(len(keys), bool))
        for k in ('rho_pair_offset', 'rho_subst_effect'):
            self.assertAlmostEqual(out[k], 1.0, places=6, msg=k)

    def test_pair_offset_prediction_scores_only_the_offset(self):
        keys, truth, parts = make()
        off = np.array([p[3] for p in parts])
        out = stats.epi_component_rhos(off, truth, keys, np.ones(len(keys), bool))
        self.assertGreater(out['rho_pair_offset'], 0.9)
        # a prediction that is constant within a pair carries no row or column information (it is centred away)
        self.assertTrue(np.isnan(out['rho_subst_effect']) or abs(out['rho_subst_effect']) < 0.25)

    def test_row_effect_prediction_scores_the_row_effect_not_the_offset(self):
        keys, truth, parts = make(seed=1)
        row = np.array([p[4] for p in parts])
        out = stats.epi_component_rhos(row, truth, keys, np.ones(len(keys), bool))
        # one side's substitution effects are predicted, the other side's are not: the pooled score sits well above zero but below one
        self.assertGreater(out['rho_subst_effect'], 0.4)
        self.assertLess(out['rho_subst_effect'], 0.95)

    def test_too_few_pairs_gives_nan(self):
        keys, truth, _ = make(n_pairs=3)
        out = stats.epi_component_rhos(truth, truth, keys, np.ones(len(keys), bool))
        self.assertTrue(np.isnan(out['rho_pair_offset']))

    def test_colrank_is_one_for_a_perfect_prediction_and_ignores_scale(self):
        rng = np.random.default_rng(2)
        keys, target = [], []
        for c in range(6):
            for r in range(10):
                keys.append(f'col{c}')
                target.append(rng.normal())
        target = np.array(target)
        rho, n = stats.colrank_rho(np.exp(target), target, keys)       # any monotone transform of the truth
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertEqual(n, 6)

    def test_colrank_keeps_a_row_effect_that_the_flip_metric_removes(self):
        # prediction = row effect only: the same ordering in every column. Within-column ranking agrees with the data; double-centred it does not
        rng = np.random.default_rng(3)
        rows, cols = 10, 10
        r = rng.normal(0, 1, rows)
        truth = r[:, None] + rng.normal(0, 0.3, (rows, cols))
        pred = np.repeat(r[:, None], cols, axis=1)
        keys = [f'code|1|2_{c}' for c in range(cols) for _ in range(rows)]
        rid = np.array([i for _ in range(cols) for i in range(rows)])
        p = np.array([pred[i, c] for c in range(cols) for i in range(rows)])
        t = np.array([truth[i, c] for c in range(cols) for i in range(rows)])
        rho_col, _ = stats.colrank_rho(p, t, keys)
        self.assertGreater(rho_col, 0.8)
        rho_flip, _, _ = stats.flip_signature_rho(p, t, keys, rid, min_len=4, by_partner_position=True)
        self.assertAlmostEqual(rho_flip, 0.0, places=6)


if __name__ == '__main__':
    unittest.main()


class TestPartnerBlindScores(unittest.TestCase):
    def test_conditional_items_take_the_score_of_the_plain_single_with_their_mutation(self):
        m = lambda p, a: ('L', p, a)
        keys = [(m(3, 'A'),), (m(5, 'G'),), (m(3, 'A'),), (m(5, 'G'),), (m(7, 'K'),)]
        sub = ['single', 'single', 'cond', 'cond', 'cond']
        wt = [0.5, -1.0, 9.0, 9.0, 9.0]                       # the in-context WT scores of the cond items (9.0) must not be used
        out = stats.partner_blind_scores(wt, sub, keys)
        self.assertEqual(list(out[:2]), [0.5, -1.0])           # singles keep their own score
        self.assertEqual(list(out[2:4]), [0.5, -1.0])
        self.assertTrue(np.isnan(out[4]))                      # no single for L7K in this loader

    def test_without_mutation_keys_everything_is_nan(self):
        self.assertTrue(np.isnan(stats.partner_blind_scores([1.0, 2.0], ['single', 'cond'], None)).all())
