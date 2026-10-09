"""--lambda_mt_flip: the confident-reversal loss. Finds exactly the measured order reversals beyond the margin, is log 2 per reversal for an
additive prediction, rewards a prediction with the reversal's sign, and is off (nothing computed or logged) by default."""
import math
import types
import unittest

import numpy as np
import torch

try:
    from tests.test_compose_censoring import make_hp, StubModel
except ImportError:  # run from inside tests/
    from test_compose_censoring import make_hp, StubModel

from esm_msr.losses import ListMLELoss


def _cls():
    from esm_msr import training
    return next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_reversal_tetrads'))


def matrix(eps, a=(0.0, -1.0, -2.0, -3.0, -4.0), b=(0.0, -0.5), pair='LIB|5|7', partners='AC'):
    """Conditional targets of one position pair: target(i|j) = a_i + b_j + eps[i, j]. Returns flip keys, row ids, targets in row-major
    order (cell (i, j) is item 2 i + j)."""
    keys, rows, t = [], [], []
    for i, ai in enumerate(a):
        for j, c in enumerate(partners):
            keys.append(f'{pair}{c}'); rows.append(i); t.append(ai + b[j] + eps[i][j])
    return keys, np.array(rows), np.array(t)


class TestTetrads(unittest.TestCase):
    def setUp(self):
        self.fn = _cls()._reversal_tetrads

    def test_additive_targets_have_no_reversal(self):
        keys, rows, t = matrix(np.zeros((5, 2)))
        self.assertEqual(len(self.fn(t, np.ones(len(t), bool), keys, rows, 0.6)[0]), 0)

    def test_a_reversal_beyond_the_margin_is_found_with_its_sign(self):
        eps = np.zeros((5, 2)); eps[1, 1] = 2.0                 # in column C row 1 rises above row 0 (-1 + 2 = 1 vs 0)
        keys, rows, t = matrix(eps)
        a, b, c, d, s = self.fn(t, np.ones(len(t), bool), keys, rows, 0.6)
        self.assertEqual(len(a), 1)
        self.assertEqual((int(a[0]), int(b[0]), int(c[0]), int(d[0]), float(s[0])), (0, 2, 1, 3, 1.0))   # rows 0/1, columns A/C
        self.assertGreater(s[0] * (t[a] - t[b] - t[c] + t[d])[0], 0)                                      # the truth's contrast agrees

    def test_the_margin_applies_to_both_columns(self):
        eps = np.zeros((5, 2)); eps[1, 1] = 1.5                 # column C: row 1 above row 0 by only 0.5
        keys, rows, t = matrix(eps)
        self.assertEqual(len(self.fn(t, np.ones(len(t), bool), keys, rows, 0.6)[0]), 0)
        self.assertEqual(len(self.fn(t, np.ones(len(t), bool), keys, rows, 0.4)[0]), 1)

    def test_invalid_rows_and_keys_without_a_partner_position_are_ignored(self):
        eps = np.zeros((5, 2)); eps[1, 1] = 2.0
        keys, rows, t = matrix(eps)
        valid = np.ones(len(t), bool); valid[3] = False          # cell (1|C)
        self.assertEqual(len(self.fn(t, valid, keys, rows, 0.6)[0]), 0)
        self.assertEqual(len(self.fn(t, np.ones(len(t), bool), ['LIB|5|native'] * len(t), rows, 0.6)[0]), 0)


def flip_batch(eps, feat_eps):
    """One pair (5 rows x 2 partners) of cond items: targets with interaction ``eps``; the stub predicts feature = a_i + b_j + feat_eps."""
    keys, rows, t = matrix(eps)
    _, _, feat = matrix(feat_eps)
    B = len(t)
    return {
        'ddG': torch.tensor(t, dtype=torch.float32), 'feat': torch.tensor(feat, dtype=torch.float32), 'mut_mask': torch.ones(B, 1, dtype=torch.bool),
        'subset_type': ['cond'] * B, 'flip_key': keys, 'mt_id': torch.tensor(rows).view(-1, 1),
        'reg_ok': torch.ones(B, dtype=torch.bool), 'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': torch.full((B,), float('nan')),
        'cens_src': torch.zeros(B, dtype=torch.long), 'dG_wt': torch.full((B,), 2.0), 'bg_offset': torch.zeros(B),
    }


def run(batch, **hp_kw):
    cls = _cls()
    stub = types.SimpleNamespace()
    stub.hparams = make_hp(**{'flip_delta': 0.6, 'flip_scale': 0.25, **hp_kw})
    stub.model = StubModel()
    stub.link_head = None
    stub.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    stub.crit_reg = torch.nn.MSELoss(reduction='none')
    stub.crit_rank_wt, stub.crit_rank_mt = ListMLELoss(), ListMLELoss()
    stub._warned_unrouted = True
    stub.global_step = 0
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_compute_flip_loss', '_compute_block_components', '_aligned_chunks', '_subset_weights', '_plan_units'):
        setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    stub._reversal_tetrads = cls._reversal_tetrads
    out = cls._compose_losses_streaming_and_backward(stub, batch)
    return out, stub


ONLY_FLIP = dict(lambda_reg_mt_master=0.0, lambda_mt_colrank=0.0, lambda_reg_wt=0.0, lambda_rank_wt=0.0)


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.eps = np.zeros((5, 2)); self.eps[1, 1] = 2.0

    def test_off_by_default(self):
        out, stub = run(flip_batch(self.eps, np.zeros((5, 2))))
        self.assertNotIn('L_flip_mt', out)
        self.assertEqual(stub._rev_diag[0], 0)

    def test_an_additive_prediction_costs_log2_per_reversal(self):
        out, stub = run(flip_batch(self.eps, np.zeros((5, 2))), lambda_mt_flip=1.0, **ONLY_FLIP)
        self.assertAlmostEqual(out['L_flip_mt'], math.log(2.0), places=5)
        self.assertEqual(stub._rev_diag[0], 1)

    def test_the_gradient_reaches_the_mt_pass_only(self):
        small = np.zeros((5, 2)); small[1, 1] = 0.2               # the stub's only MT parameter scales the prediction: it needs a contrast
        _, stub = run(flip_batch(self.eps, small), lambda_mt_flip=1.0, **ONLY_FLIP)
        g = stub.model.w.grad
        self.assertEqual(float(g[0]), 0.0)
        self.assertLess(float(g[1]), 0.0)                        # growing the right-signed contrast lowers the loss

    def test_predicting_the_reversal_lowers_the_loss(self):
        right = np.zeros((5, 2)); right[1, 1] = 1.5
        wrong = np.zeros((5, 2)); wrong[1, 1] = -1.5
        l_right = run(flip_batch(self.eps, right), lambda_mt_flip=1.0, **ONLY_FLIP)[0]['L_flip_mt']
        l_wrong = run(flip_batch(self.eps, wrong), lambda_mt_flip=1.0, **ONLY_FLIP)[0]['L_flip_mt']
        self.assertLess(l_right, math.log(2.0))
        self.assertGreater(l_wrong, math.log(2.0))

    def test_it_alone_creates_the_mt_unit(self):
        out, _ = run(flip_batch(self.eps, np.zeros((5, 2))), lambda_mt_flip=0.5, **ONLY_FLIP)
        self.assertEqual({k for k in out if k.startswith('L_')}, {'L_flip_mt'})


if __name__ == '__main__':
    unittest.main()
