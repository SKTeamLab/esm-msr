import math
import re
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from esm_msr.losses import ListMLELoss


def reference_listmle(pred, gt, mask=None):
    """Naive per-list Plackett-Luce NLL, deliberately independent of the vectorised code."""
    total, n = 0.0, 0
    for b in range(pred.shape[0]):
        keep = [j for j in range(pred.shape[1]) if mask is None or bool(mask[b, j])]
        if len(keep) < 2:
            continue
        order = sorted(keep, key=lambda j: -float(gt[b, j]))
        s = [float(pred[b, j]) for j in order]
        nll = 0.0
        for k in range(len(s)):
            nll -= s[k] - math.log(sum(math.exp(x) for x in s[k:]))
        total += nll
        n += 1
    return total / n


class TestListMLEUnchanged(unittest.TestCase):
    """The WT head's rank loss must be bit-for-bit what it was: score_mask is opt-in."""

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.pred = torch.randn(5, 8, generator=g)
        self.gt = torch.randn(5, 8, generator=g)
        self.mask = torch.rand(5, 8, generator=g) > 0.2
        self.crit = ListMLELoss()

    def test_matches_naive_reference(self):
        for mask in (None, self.mask):
            got = float(self.crit(self.pred, self.gt, mask=mask))
            self.assertAlmostEqual(got, reference_listmle(self.pred, self.gt, mask), places=4)

    def test_all_true_score_mask_is_identical(self):
        a = self.crit(self.pred, self.gt, mask=self.mask)
        b = self.crit(self.pred, self.gt, mask=self.mask, score_mask=torch.ones_like(self.mask))
        self.assertTrue(torch.equal(a, b))

    def test_censored_loss_is_reached_only_through_the_two_rank_helpers(self):
        # forward_censored may be called from _compute_rank_loss (WT / combined lists, only when a row is censored)
        # and _compute_flip_loss, and nowhere else in the training module.
        src = (Path(__file__).parents[1] / 'src/esm_msr/training.py').read_text()
        self.assertEqual(len(re.findall(r'forward_censored\(', src)), 2)
        rank = src.split('def _compute_rank_loss')[1].split('def _compute_flip_loss')[0]
        flip = src.split('def _compute_flip_loss')[1].split('def _subset_weights')[0]
        self.assertEqual(rank.count('forward_censored('), 1)
        self.assertEqual(flip.count('forward_censored('), 1)


class TestCensoredListMLE(unittest.TestCase):
    def setUp(self):
        self.crit = ListMLELoss()

    def test_closed_form_one_scored_item(self):
        pred = torch.tensor([[0.3, -0.2, 1.1, 0.4]])
        gt = torch.tensor([[2.0, -1.0, -1.0, -1.0]])
        sm = torch.tensor([[True, False, False, False]])
        want = float(torch.logsumexp(pred[0], 0) - pred[0, 0])
        self.assertAlmostEqual(float(self.crit(pred, gt, score_mask=sm)), want, places=5)

    def test_all_censored_list_is_zero_with_grad(self):
        pred = torch.randn(1, 5, requires_grad=True)
        loss = self.crit(pred, torch.randn(1, 5), score_mask=torch.zeros(1, 5, dtype=torch.bool))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.all(pred.grad == 0))

    def test_order_among_censored_is_free(self):
        pred = torch.tensor([[1.0, 0.5, -0.5, 0.2, -1.0]])
        gt = torch.tensor([[3.0, 2.0, -1.0, -1.0, -1.0]])
        sm = torch.tensor([[True, True, False, False, False]])
        base = float(self.crit(pred, gt, score_mask=sm))
        # permute the predictions among the censored members: no change at all
        p2 = pred.clone()
        p2[0, 2:] = pred[0, [4, 2, 3]]
        self.assertAlmostEqual(float(self.crit(p2, gt, score_mask=sm)), base, places=6)
        # permute the *targets* among the censored members: no change either
        g2 = gt.clone()
        g2[0, 2:] = torch.tensor([-5.0, -1.0, -3.0])
        self.assertAlmostEqual(float(self.crit(pred, g2, score_mask=sm)), base, places=6)

    def test_still_pushes_scored_above_censored(self):
        gt = torch.tensor([[2.0, 1.0, -1.0, -1.0]])
        sm = torch.tensor([[True, True, False, False]])
        right = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
        wrong = torch.tensor([[-1.0, -2.0, 2.0, 1.0]])   # censored items scored above real ones
        self.assertLess(float(self.crit(right, gt, score_mask=sm)),
                        float(self.crit(wrong, gt, score_mask=sm)))
        pred = wrong.clone().requires_grad_(True)
        self.crit(pred, gt, score_mask=sm).backward()
        self.assertTrue(torch.all(pred.grad[0, :2] < 0))   # descending the loss raises scored items

    def test_scored_order_still_matters(self):
        gt = torch.tensor([[2.0, 1.0, -1.0, -1.0]])
        sm = torch.tensor([[True, True, False, False]])
        good = torch.tensor([[2.0, 1.0, -3.0, -3.0]])
        swapped = torch.tensor([[1.0, 2.0, -3.0, -3.0]])
        self.assertLess(float(self.crit(good, gt, score_mask=sm)), float(self.crit(swapped, gt, score_mask=sm)))

    def test_padding_mask_and_censoring_compose(self):
        pred = torch.tensor([[0.5, 0.1, -0.3, 9.0]])
        gt = torch.tensor([[2.0, 1.0, -1.0, 7.0]])
        mask = torch.tensor([[True, True, True, False]])        # last slot is padding
        sm = torch.tensor([[True, True, False, True]])
        a = float(self.crit(pred, gt, mask=mask, score_mask=sm))
        b = float(self.crit(pred[:, :3], gt[:, :3], score_mask=sm[:, :3]))
        self.assertAlmostEqual(a, b, places=6)

    def test_rejects_invert(self):
        with self.assertRaises(AssertionError):
            ListMLELoss(invert=True)(torch.zeros(1, 3), torch.zeros(1, 3), score_mask=torch.ones(1, 3, dtype=torch.bool))


class TestFlipLossCensoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from esm_msr.training import ESM3EpistasisLightningModule  # noqa: F401
        except Exception as e:  # pragma: no cover - heavy deps absent
            raise unittest.SkipTest(f"training module not importable: {e}")

    def _call(self, censored, keys, pred, targets, min_len=3):
        censored = None if censored is None else censored.long()
        from esm_msr import training
        cls = next(v for v in vars(training).values()
                   if isinstance(v, type) and hasattr(v, '_compute_flip_loss'))
        stub = types.SimpleNamespace()
        valid = torch.ones(len(keys), dtype=torch.bool)
        out = cls._compute_flip_loss(stub, pred, targets, valid, keys, ListMLELoss(), min_len,
                                     cens=censored)
        return out, stub

    def test_all_censored_column_is_dropped_and_diag_counts_uncensored(self):
        keys = ['a'] * 4 + ['b'] * 4
        targets = torch.tensor([3., 2., -1., -1., -1., -1., -1., -1.])
        pred = torch.randn(8)
        cens = torch.tensor([False, False, True, True, True, True, True, True])
        (loss, _, G), stub = self._call(cens, keys, pred, targets)
        self.assertEqual(G, 1)                       # column 'b' carries no information
        self.assertEqual(stub._flip_diag[3], 2)      # uncensored members that entered the loss

    def test_no_censoring_matches_uncensored_call(self):
        keys = ['a'] * 5
        targets = torch.tensor([3., 2., 1., 0., -1.])
        pred = torch.randn(5)
        (l1, *_), _ = self._call(None, keys, pred, targets)
        (l2, *_), _ = self._call(torch.zeros(5, dtype=torch.bool), keys, pred, targets)
        self.assertAlmostEqual(float(l1), float(l2), places=6)


class TestFloorCensoring(unittest.TestCase):
    def test_marks_on_measured_dG_only_in_columns(self):
        from esm_msr import censoring
        items = [
            {'flip_key': 'c', 'dG_meas': -1.5, 'ddG': -3.0},      # at/below floor
            {'flip_key': 'c', 'dG_meas': 0.0, 'ddG': -1.5},       # exactly at floor -> censored
            {'flip_key': 'c', 'dG_meas': 0.4, 'ddG': -1.1},       # above floor: genuine compensator keeps ordering
            {'flip_key': 'c', 'dG_meas': np.nan, 'ddG': 0.0},     # unknown -> never censored
            {'flip_key': '', 'dG_meas': -3.0, 'ddG': -4.0},       # not a flip item
        ]
        kept, counts = censoring.apply_item_censoring(items, include_out_of_range=False, censor_floor=0.0)
        self.assertEqual([i['cens'] for i in kept], [-1, -1, 0, 0, 0])
        self.assertEqual(counts['floor'], 2)
        # the bound sits on the item's own ddG scale: dG_bound + (ddG - dG_meas) = 0.0 + (-1.5 - 0.0)
        self.assertAlmostEqual(kept[1]['cens_bound'], -1.5)
        self.assertAlmostEqual(kept[0]['cens_bound'], 0.0 + (-3.0 - -1.5))

    def test_disabled_marks_nothing(self):
        from esm_msr import censoring
        items = [{'flip_key': 'c', 'dG_meas': -9.0, 'ddG': -9.0}]
        kept, _ = censoring.apply_item_censoring(items, include_out_of_range=False, censor_floor=None)
        self.assertEqual(kept[0]['cens'], 0)


if __name__ == '__main__':
    unittest.main()
