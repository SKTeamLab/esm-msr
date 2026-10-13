"""The WT-in-context control (ctx) in inference and esm_msr_testing.py: score_screening_batch runs the reverse WT leg when asked (also on
the WT-only cached path), the testing metrics report every epistasis metric for comb, mt, ctx and add (recalibrated too), and rmse_obs_*
puts latent predictions on the observed scale through the link."""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

from esm_msr import epi_metrics
from esm_msr.link import numpy_link
from esm_msr.models import MSRModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference_scripts'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


class Stub:
    """forward_batch / forward_cycle that record the mutant sequences they are given; predictions encode them."""
    training = False

    def __init__(self):
        self.seen = []

    def forward_batch(self, b, cached_wt_esm3=None, skip_reverse=False, mask_strategy=None):
        n = b['mt_sequence_tokens'].shape[0]
        return {'wt_lora_pred': torch.ones(n), 'mt_lora_pred': torch.full((n,), 2.0), 'combined_pred': torch.full((n,), 1.5)}

    def forward_cycle(self, b, mask_strategy=None, legs=('wt_rev', 'mt_fwd')):
        self.seen.append(b['mt_sequence_tokens'].clone())
        return {'wt_rev': b['mt_sequence_tokens'].sum(dim=1).float()}


def screen(stub, wt_reverse, skip_reverse=False, cached=False):
    wt = torch.tensor([[1, 2, 3, 4]])
    kw = dict(mut_pos=torch.tensor([[1], [3]]), wt_id=torch.tensor([[2], [4]]), mt_id=torch.tensor([[9], [7]]),
              mut_mask=torch.tensor([[True], [True]]), batch_size=8, skip_reverse=skip_reverse, quiet=True, wt_reverse=wt_reverse,
              cached_wt_esm3={'logits': torch.zeros(1, 4, 5), 'embeddings': None} if cached else None)
    return MSRModel.score_screening_batch(stub, wt, **kw)


class TestScreeningReverseLeg(unittest.TestCase):
    def test_off_by_default_and_nan(self):
        s = Stub(); out = screen(s, wt_reverse=False)
        self.assertTrue(torch.isnan(out['wt_rev_pred']).all()); self.assertEqual(s.seen, [])

    def test_the_leg_reads_each_mutated_sequence(self):
        s = Stub(); out = screen(s, wt_reverse=True)
        torch.testing.assert_close(out['wt_rev_pred'], torch.tensor([1 + 9 + 3 + 4, 1 + 2 + 3 + 7], dtype=torch.float32))   # pos 1 -> 9, pos 3 -> 7

    def test_the_wt_only_cached_path_still_builds_the_mutants(self):
        s = Stub(); out = screen(s, wt_reverse=True, skip_reverse=True, cached=True)
        torch.testing.assert_close(out['wt_rev_pred'], torch.tensor([17.0, 13.0]))


ROWS, COLS = 'ACDEFG', 'HIKLMN'


def predictions(seed=0, n_pairs=10):
    """A predictions table like inference writes for MegaScale libraries: singles and doubles of n_pairs 6 x 6 position-pair matrices."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pairs):
        lib = f'L{p % 2}'
        i0, j0 = 10 * p + 1, 10 * p + 5
        a, b = rng.normal(-1, 1, 6), rng.normal(-1, 1, 6)
        eps = rng.normal(0, 0.5, (6, 6))
        for i, x in enumerate(ROWS):
            rows.append(dict(code=lib, mut_type=f'W{i0}{x}', ddG_ML=a[i], wt=a[i], ctxe=0.0, mte=0.0))
        for j, y in enumerate(COLS):
            rows.append(dict(code=lib, mut_type=f'W{j0}{y}', ddG_ML=b[j], wt=b[j], ctxe=0.0, mte=0.0))
        for i, x in enumerate(ROWS):
            for j, y in enumerate(COLS):
                rows.append(dict(code=lib, mut_type=f'W{i0}{x}:W{j0}{y}', ddG_ML=a[i] + b[j] + eps[i, j], wt=a[i] + b[j],
                                 ctxe=0.5 * eps[i, j] + rng.normal(0, 0.5), mte=eps[i, j]))
    df = pd.DataFrame(rows)
    df['wt_lora_pred'] = df.wt
    df['mt_lora_pred'] = df.wt + df.mte
    df['ctx_pred'] = df.wt + df.ctxe
    df['combined_pred'] = 0.5 * (df.wt_lora_pred + df.mt_lora_pred)
    dbl = df.mut_type.str.contains(':')
    single = dict(zip(df.loc[~dbl, 'code'] + '|' + df.loc[~dbl, 'mut_type'], df.loc[~dbl, 'ddG_ML']))
    df['dddG_ML'] = np.nan
    for k in df.index[dbl]:
        m1, m2 = df.at[k, 'mut_type'].split(':')
        df.at[k, 'dddG_ML'] = df.at[k, 'ddG_ML'] - single[df.at[k, 'code'] + '|' + m1] - single[df.at[k, 'code'] + '|' + m2]
    return df


class TestTestingMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr_testing as T
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f'esm_msr_testing not importable: {e}')
        cls.T = T

    def test_every_metric_for_every_head_including_ctx(self):
        df = predictions()
        sc = self.T.epi_scores(self.T.epi_items(df, 'ddG_ML', 'dddG_ML', dG_wt={'L0': 2.0, 'L1': 2.5}), None)
        self.assertEqual(set(sc), {'comb', 'mt', 'ctx', 'add'})
        for head in ('comb', 'mt', 'ctx'):
            for m in epi_metrics.METRICS:
                self.assertIn(f'epi_{m}', sc[head], (head, m))
        self.assertGreater(sc['mt']['epi_cell_rank'], sc['ctx']['epi_cell_rank'])          # ctx knows half the cell, with noise
        self.assertGreater(sc['ctx']['epi_cell_rank'], 0.1)
        self.assertEqual(sc['add']['epi_cell_rank'], 0.0)

    def test_without_the_reverse_leg_there_is_no_ctx_head(self):
        df = predictions().drop(columns='ctx_pred')
        self.assertNotIn('ctx', self.T.epi_scores(self.T.epi_items(df, 'ddG_ML', 'dddG_ML'), None))

    def test_recalibration_fits_and_scores_ctx_too(self):
        df = predictions()
        out, fits = self.T.recalibrate(df, 'linear', 'ddG_ML')
        self.assertIn('ctx_pred', fits)
        sc = self.T.epi_scores(self.T.epi_items(out, 'ddG_ML', 'dddG_ML'), self.T.recal_links(fits))
        self.assertIn('ctx', sc)

    def test_observed_scale_rmse(self):
        df = predictions()
        h = numpy_link(-1.0, 5.0, 0.3, 0.3)
        dgw = {'L0': 0.5, 'L1': 0.8}
        st = self.T.update_obs_rmse(pd.DataFrame(), 'x', df, 'ddG_ML', 'combined_pred', h, dgw)
        c = df.code.map(dgw).to_numpy()
        want = np.sqrt(np.mean((h(c + df.combined_pred.to_numpy()) - c - df.ddG_ML.to_numpy()) ** 2))
        self.assertAlmostEqual(st.at['x', 'rmse_obs_all'], want, places=9)
        self.assertIn('rmse_obs_doubles', st.columns)
        self.assertEqual(len(self.T.update_obs_rmse(pd.DataFrame(), 'x', df, 'ddG_ML', 'combined_pred', None, dgw).columns), 0)


if __name__ == '__main__':
    unittest.main()
