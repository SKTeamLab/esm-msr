import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parents[1] / 'analysis_notebooks' / 'anova'))
import decompose_predictions as D  # noqa: E402


def synthetic(seed=0):
    """Doubles on 3 position pairs x 8 x 8 residues with a known structure: row, column, pair offset, interaction, saturation."""
    rng = np.random.default_rng(seed)
    rows = []
    for pi, (p, q) in enumerate(((3, 9), (3, 14), (11, 20))):
        offs = rng.normal(0, 0.6)
        ra, cb = rng.normal(0, 0.8, 8), rng.normal(0, 0.8, 8)
        eps = rng.normal(0, 0.4, (8, 8))
        for i in range(8):
            for j in range(8):
                x = rng.uniform(-1, 3)                       # additive-predicted dG, mid-range (no saturation here)
                dddG = offs + ra[i] + cb[j] + eps[i, j]
                rows.append(dict(code='LIB', p=p, q=q, a='ACDEFGHI'[i], b='KLMNPQRS'[j], x=x, dddG=dddG,
                                 dG_AB=x + dddG, pair=f'LIB|{p}|{q}', offs=offs, rc=ra[i] + cb[j], eps=eps[i, j]))
    return pd.DataFrame(rows)


class TestDecompose(unittest.TestCase):
    def setUp(self):
        self.T = synthetic()
        self.iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(self.T.x, self.T.dG_AB)

    def test_components_add_back_to_the_measured_value(self):
        C = D.measured_components(self.T, self.iso)
        np.testing.assert_allclose(C.g + C.m + C.rc + C.eps, C.dddG, atol=1e-9)

    def test_a_prediction_of_only_the_interaction_scores_on_the_interaction(self):
        C = D.measured_components(self.T, self.iso)
        res = D.decompose(self.T, C['eps'].to_numpy(), self.iso)
        self.assertGreater(res['interaction (leftover)'], 0.95)
        self.assertLess(abs(res['position-pair offset']), 0.2)

    def test_a_prediction_of_only_row_and_column_effects_scores_on_them(self):
        C = D.measured_components(self.T, self.iso)
        res = D.decompose(self.T, C['rc'].to_numpy(), self.iso)
        self.assertGreater(res['substitution row + column effects'], 0.95)
        self.assertLess(abs(res['interaction (leftover)']), 0.3)

    def test_variance_shares_are_reported(self):
        res = D.decompose(self.T, self.T['dddG'].to_numpy(), self.iso)
        self.assertAlmostEqual(res['measured dddG'], 1.0, places=6)
        self.assertEqual(set(res['share_of_measured_variance']), {'g', 'm', 'rc', 'eps'})


if __name__ == '__main__':
    unittest.main()
