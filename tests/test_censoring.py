import math
import unittest

import numpy as np
import torch

from esm_msr import censoring, stats
from esm_msr.losses import ListMLELoss


def lse(*xs):
    return math.log(sum(math.exp(x) for x in xs))


class TestTwoSidedListMLE(unittest.TestCase):
    def setUp(self):
        self.c = ListMLELoss()

    def test_no_censored_member_equals_forward(self):
        g = torch.Generator().manual_seed(0)
        p, t = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
        m = torch.rand(5, 8, generator=g) > 0.2
        a = self.c.forward_censored(p, t, m, torch.zeros(5, 8, dtype=torch.long))
        self.assertTrue(torch.allclose(a, self.c(p, t, mask=m), atol=1e-6))

    def test_lower_only_equals_score_mask_path(self):
        g = torch.Generator().manual_seed(1)
        p, t = torch.randn(6, 9, generator=g), torch.randn(6, 9, generator=g)
        m = torch.rand(6, 9, generator=g) > 0.2
        cens = torch.zeros(6, 9, dtype=torch.long)
        cens[torch.rand(6, 9, generator=g) > 0.7] = -1
        a = self.c.forward_censored(p, t, m, cens)
        self.assertTrue(torch.allclose(a, self.c(p, t, mask=m, score_mask=(cens == 0)), atol=1e-6))

    def test_upper_only_closed_form(self):
        # members a (better), b (worse) ordinary, h upper-censored. Reverse PL, worst first:
        # -[(-s_b) - lse(-s_b,-s_a,-s_h)] - [(-s_a) - lse(-s_a,-s_h)]
        s_a, s_b, s_h = 0.4, -0.3, 1.1
        pred = torch.tensor([[s_a, s_b, s_h]])
        gt = torch.tensor([[2.0, 1.0, 99.0]])
        cens = torch.tensor([[0, 0, 1]])
        want = (s_b + lse(-s_b, -s_a, -s_h)) + (s_a + lse(-s_a, -s_h))
        got = float(self.c.forward_censored(pred, gt, torch.ones(1, 3, dtype=torch.bool), cens))
        self.assertAlmostEqual(got, want, places=5)

    def test_both_sides_average_the_two_passes(self):
        # a, b ordinary; d dead (lower-censored); h hyper (upper-censored)
        s = dict(a=0.4, b=-0.3, d=-1.2, h=1.1)
        pred = torch.tensor([[s['a'], s['b'], s['d'], s['h']]])
        gt = torch.tensor([[2.0, 1.0, -99.0, 99.0]])
        cens = torch.tensor([[0, 0, -1, 1]])
        # forward pass leaves h out and ties d at the bottom; reverse pass leaves d out and ties h at the end
        fwd = (lse(s['a'], s['b'], s['d']) - s['a']) + (lse(s['b'], s['d']) - s['b'])
        rev = (s['b'] + lse(-s['b'], -s['a'], -s['h'])) + (s['a'] + lse(-s['a'], -s['h']))
        got = float(self.c.forward_censored(pred, gt, torch.ones(1, 4, dtype=torch.bool), cens))
        self.assertAlmostEqual(got, 0.5 * (fwd + rev), places=5)

    def test_order_within_each_censored_block_is_free(self):
        pred = torch.tensor([[0.7, 0.1, -0.4, -0.9, 1.3, 0.2]])
        gt = torch.tensor([[3.0, 2.0, 0.0, 0.0, 9.0, 9.0]])
        cens = torch.tensor([[0, 0, -1, -1, 1, 1]])
        m = torch.ones(1, 6, dtype=torch.bool)
        base = float(self.c.forward_censored(pred, gt, m, cens))
        swapped = pred.clone()
        swapped[0, 2], swapped[0, 3] = pred[0, 3], pred[0, 2]       # permute the dead pair
        swapped[0, 4], swapped[0, 5] = pred[0, 5], pred[0, 4]       # permute the hyper pair
        self.assertAlmostEqual(float(self.c.forward_censored(swapped, gt, m, cens)), base, places=6)

    def test_pushes_hyper_up_and_dead_down(self):
        gt = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
        cens = torch.tensor([[0, 0, -1, 1]])
        m = torch.ones(1, 4, dtype=torch.bool)
        pred = torch.tensor([[0.5, -0.5, 0.1, -0.1]], requires_grad=True)   # dead above ordinary, hyper below
        self.c.forward_censored(pred, gt, m, cens).backward()
        # descending the loss must lower the dead member and raise the hyper member
        self.assertGreater(float(pred.grad[0, 2]), 0)
        self.assertLess(float(pred.grad[0, 3]), 0)

    def test_list_with_only_censored_members_is_skipped(self):
        pred = torch.randn(1, 4, requires_grad=True)
        loss = self.c.forward_censored(pred, torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
                                       torch.tensor([[-1, -1, 1, 1]]))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.all(pred.grad == 0))


class TestRangeParsing(unittest.TestCase):
    def test_strings_become_bounds_on_the_ddG_scale(self):
        raw = np.array(['1.5', '<-1', '>5', '-', 0.2], dtype=object)
        wt = np.array([3.0, 3.0, 3.0, 3.0, np.nan])
        cens, ddG = censoring.range_censoring(raw, wt)
        self.assertEqual(list(cens), [0, -1, 1, 0, 0])
        self.assertAlmostEqual(ddG[0], 1.5 - 3.0)
        self.assertAlmostEqual(ddG[1], -1.0 - 3.0)      # dead: ddG at most -4
        self.assertAlmostEqual(ddG[2], 5.0 - 3.0)       # hyper: ddG at least +2
        self.assertTrue(np.isnan(ddG[3]))               # '-' = no estimate, not dead
        self.assertTrue(np.isnan(ddG[4]))               # unknown dG_wt: no bound can be formed


class TestApplyItemCensoring(unittest.TestCase):
    def _items(self):
        return [
            {'cens': -1, 'cens_src': censoring.SRC_RANGE, 'dG_meas': -1.0, 'dG_bound': -1.0, 'ddG': -4.0, 'flip_key': ''},
            {'cens': 1, 'cens_src': censoring.SRC_RANGE, 'dG_meas': 5.0, 'dG_bound': 5.0, 'ddG': 2.0, 'flip_key': ''},
            {'cens': 0, 'dG_meas': 1.0, 'ddG': -2.0, 'flip_key': ''},
        ]

    def test_dropped_unless_requested(self):
        kept, c = censoring.apply_item_censoring(self._items(), include_out_of_range=False)
        self.assertEqual(len(kept), 1)
        self.assertEqual(c['dropped_range'], 2)

    def test_kept_with_bounds_on_item_scale(self):
        kept, c = censoring.apply_item_censoring(self._items(), include_out_of_range=True)
        self.assertEqual([i['cens'] for i in kept], [-1, 1, 0])
        self.assertAlmostEqual(kept[0]['cens_bound'], -4.0)
        self.assertAlmostEqual(kept[1]['cens_bound'], 2.0)
        self.assertTrue(np.isnan(kept[2]['cens_bound']))

    def test_conditional_item_bound_carries_the_partner_single(self):
        # a dead double (dG bound -1, ddG_AB = -4) with partner single ddG_B = -1: ddG(A|B) = ddG_AB - ddG_B = -3 is at most -3
        item = {'cens': -1, 'cens_src': censoring.SRC_RANGE, 'dG_meas': -1.0, 'dG_bound': -1.0, 'ddG': -3.0, 'flip_key': 'k'}
        kept, _ = censoring.apply_item_censoring([item], include_out_of_range=True)
        self.assertAlmostEqual(kept[0]['cens_bound'], -3.0)


class TestCensoredRegression(unittest.TestCase):
    def test_only_the_wrong_side_of_the_bound_is_penalised(self):
        crit = torch.nn.MSELoss(reduction='none')
        pred = torch.tensor([-5.0, -3.0, 4.0, 1.0])
        bound = torch.tensor([-4.0, -4.0, 2.0, 2.0])
        cens = torch.tensor([-1, -1, 1, 1])
        out = censoring.censored_regression_loss(crit, pred, bound, cens)
        # dead, predicted below the bound: free. dead predicted above it: penalised. hyper above: free. hyper below: penalised.
        self.assertEqual(out.tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_gradient_pushes_toward_the_bound_only_when_violated(self):
        crit = torch.nn.MSELoss(reduction='none')
        pred = torch.tensor([-3.0, -6.0], requires_grad=True)
        censoring.censored_regression_loss(crit, pred, torch.tensor([-4.0, -4.0]), torch.tensor([-1, -1])).sum().backward()
        self.assertGreater(float(pred.grad[0]), 0)       # too high: pushed down
        self.assertEqual(float(pred.grad[1]), 0)         # already below: untouched


class TestAuroc(unittest.TestCase):
    def test_values(self):
        self.assertEqual(censoring.auroc(np.array([3., 4.]), np.array([1., 2.])), 1.0)
        self.assertEqual(censoring.auroc(np.array([1., 2.]), np.array([3., 4.])), 0.0)
        self.assertEqual(censoring.auroc(np.array([1., 1.]), np.array([1., 1.])), 0.5)
        self.assertTrue(np.isnan(censoring.auroc(np.array([]), np.array([1.]))))


class TestValidationMetricsWithCensoring(unittest.TestCase):
    def test_existing_metrics_ignore_censored_items_and_new_aucs_appear(self):
        wt = np.array([1.0, 2.0, 3.0, -5.0, 9.0])
        gt = np.array([1.1, 1.9, 3.2, -4.0, 2.0])
        sub = ['single'] * 5
        base = stats.compute_metrics(wt[:3], wt[:3], wt[:3], gt[:3], sub[:3])
        cens = np.array([0, 0, 0, -1, 1])
        full = stats.compute_metrics(wt, wt, wt, gt, sub, cens=cens)
        self.assertAlmostEqual(base['rho_wt_valid'], full['rho_wt_valid'], places=9)
        self.assertEqual(full['auc_dead_wt'], 1.0)       # dead (-5) scored below all ordinary singles
        self.assertEqual(full['auc_hyper_wt'], 1.0)      # hyper (9) scored above them
        self.assertNotIn('auc_dead_wt', base)

    def test_no_cens_argument_gives_the_old_metrics(self):
        wt = np.array([1.0, 2.0, 3.0])
        m = stats.compute_metrics(wt, wt, wt, wt + 0.1, ['single'] * 3)
        self.assertNotIn('auc_dead_wt', m)


if __name__ == '__main__':
    unittest.main()


class TestWtRankHelperUnchanged(unittest.TestCase):
    """The WT head's list loss must be exactly what it was unless a row of the lists is censored."""

    @classmethod
    def setUpClass(cls):
        try:
            from esm_msr import training
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")
        cls.fn = staticmethod(next(v for v in vars(training).values()
                                   if isinstance(v, type) and hasattr(v, '_compute_rank_loss'))._compute_rank_loss)

    def _data(self):
        g = torch.Generator().manual_seed(3)
        return torch.randn(32, generator=g), torch.randn(32, generator=g), torch.rand(32, generator=g) > 0.1

    def test_none_and_all_zero_cens_are_bit_identical_to_the_plain_call(self):
        pred, targ, mask = self._data()
        crit = ListMLELoss()
        plain = self.fn(None, pred, targ, mask, 16, crit)
        for cens in (None, torch.zeros(32, dtype=torch.long)):
            got = self.fn(None, pred, targ, mask, 16, crit, cens=cens)
            self.assertTrue(torch.equal(plain[0], got[0]))
            self.assertTrue(torch.equal(plain[1], got[1]))
            self.assertEqual(plain[2], got[2])
        direct = crit(pred.view(-1, 16), targ.view(-1, 16), mask=mask.view(-1, 16))
        self.assertTrue(torch.equal(plain[1], direct.detach()))

    def test_a_censored_row_changes_the_loss_through_the_censored_path(self):
        pred, targ, mask = self._data()
        crit = ListMLELoss()
        cens = torch.zeros(32, dtype=torch.long)
        cens[3] = -1
        plain = self.fn(None, pred, targ, mask, 16, crit)
        got = self.fn(None, pred, targ, mask, 16, crit, cens=cens)
        self.assertFalse(torch.equal(plain[1], got[1]))
        want = crit.forward_censored(pred.view(-1, 16), targ.view(-1, 16), mask.view(-1, 16), cens.view(-1, 16))
        self.assertTrue(torch.allclose(got[1], want.detach()))
