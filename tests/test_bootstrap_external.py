import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_notebooks'))
import bootstrap_external as be  # noqa: E402


def make_df(rng, n=300, n_codes=12):
    code = rng.choice([f'p{i}' for i in range(n_codes)], n)
    y = np.round(rng.normal(size=n), 1)                       # ties in the ground truth
    a = np.round(y + rng.normal(scale=0.8, size=n), 1)        # ties in predictions
    b = rng.normal(size=n)
    b[rng.random(n) < 0.15] = np.nan                          # model with missing rows
    return pd.DataFrame({'code': code, 'ground_truth': y, 'a': a, 'b': b})


def reference(df, w, grouped, mult=None):
    """Spearman on the expanded sample, via scipy; grouped = (multiplicity-weighted) mean over codes."""
    out = {}
    for m in ('a', 'b'):
        if not grouped:
            keep = df[m].notna().values
            idx = np.repeat(np.flatnonzero(keep), w[keep])
            out[m] = spearmanr(df['ground_truth'].values[idx], df[m].values[idx])[0]
        else:
            rs, ms = [], []
            for c, sub in df.groupby('code', sort=True):
                ww = w[sub.index.values]
                keep = sub[m].notna().values & (ww > 0)
                idx = np.repeat(np.flatnonzero(keep), ww[keep])
                if len(idx) < 2:
                    continue
                r = spearmanr(sub['ground_truth'].values[idx], sub[m].values[idx])[0]
                if np.isfinite(r):
                    rs.append(r)
                    ms.append(1.0 if mult is None else mult[list(sorted(df['code'].unique())).index(c)])
            out[m] = float(np.average(rs, weights=ms)) if rs else np.nan
    return out


class TestWeightedSpearman(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.df = make_df(self.rng).sort_values('code', kind='stable').reset_index(drop=True)

    def test_observed_matches_pandas_and_scipy(self):
        eng = be.SpearmanBootstrap(self.df, ['a', 'b'], grouped=False)
        obs = eng.observed()
        for m in ('a', 'b'):
            self.assertAlmostEqual(obs[m], self.df[['ground_truth', m]].corr('spearman').iloc[0, 1], places=10)
        eng_g = be.SpearmanBootstrap(self.df, ['a', 'b'], grouped=True)
        pandas_grouped = {m: self.df.groupby('code')[['ground_truth', m]].corr('spearman').xs('ground_truth', level=1)[m].mean()
                          for m in ('a', 'b')}
        for m, v in eng_g.observed().items():
            self.assertAlmostEqual(v, pandas_grouped[m], places=10)

    def test_random_integer_weights_match_expanded_sample(self):
        for grouped in (False, True):
            eng = be.SpearmanBootstrap(self.df, ['a', 'b'], grouped=grouped)
            for _ in range(5):
                w = self.rng.poisson(1.0, len(self.df))
                got = eng.rho(w)
                ref = reference(self.df, w, grouped)
                for m in ('a', 'b'):
                    self.assertAlmostEqual(got[m], ref[m], places=9, msg=f'{m} grouped={grouped}')

    def test_protein_draw_gives_whole_proteins_and_matches_reference(self):
        eng = be.SpearmanBootstrap(self.df, ['a', 'b'], grouped=True)
        w, mult = eng.draw(np.random.default_rng(1), 'protein')
        self.assertEqual(mult.sum(), eng.n_proteins)
        # every row of a protein carries the protein's multiplicity
        self.assertTrue((w == mult[eng.codes]).all())
        got, ref = eng.rho(w, mult=mult), reference(self.df, w, True, mult)
        for m in ('a', 'b'):
            self.assertAlmostEqual(got[m], ref[m], places=9)

    def test_mutation_draw_stays_within_group(self):
        eng = be.SpearmanBootstrap(self.df, ['a'], grouped=True)
        w, _ = eng.draw(np.random.default_rng(2), 'mutation')
        per_group = np.bincount(eng.group, weights=w, minlength=eng.n_groups)
        self.assertTrue((per_group == eng.gsize).all())

    def test_holm_matches_hand_computation(self):
        np.testing.assert_allclose(be.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(be.holm(np.array([0.5, 0.001])), [0.5, 0.002])

    def test_joint_protein_draw_shares_proteins_across_datasets(self):
        rng = np.random.default_rng(3)
        d1, d2 = make_df(rng, 200, 10), make_df(rng, 150, 10)   # same protein ids p0..p9
        e1 = be.SpearmanBootstrap(d1.sort_values('code'), ['a'], False)
        e2 = be.SpearmanBootstrap(d2.sort_values('code'), ['a'], False)
        glob = np.unique(np.concatenate([e1.code_labels, e2.code_labels]))
        kg = rng.multinomial(len(glob), np.full(len(glob), 1 / len(glob)))
        w1 = kg[np.searchsorted(glob, e1.code_labels)[e1.codes]]
        w2 = kg[np.searchsorted(glob, e2.code_labels)[e2.codes]]
        for c in glob:   # a protein carries the same multiplicity in both datasets
            m1, m2 = (e1.code_labels == c), (e2.code_labels == c)
            self.assertEqual(w1[e1.codes == np.flatnonzero(m1)[0]][0], w2[e2.codes == np.flatnonzero(m2)[0]][0])

    def test_p_value_helpers(self):
        self.assertAlmostEqual(be.one_sided_p(np.array([1.0] * 99)), 1 / 100)
        self.assertAlmostEqual(be.one_sided_p(np.array([-1.0] * 99)), 1.0)
        self.assertAlmostEqual(be.one_sided_p(np.array([1.0, np.nan, 1.0])), 1 / 3)


if __name__ == '__main__':
    unittest.main()
