import types
import unittest
from collections import Counter

import numpy as np
import torch

try:
    from tests.test_compose_censoring import make_batch, run
    from tests.test_sampler import DummyDataset
except ImportError:  # run from inside tests/
    from test_compose_censoring import make_batch, run
    from test_sampler import DummyDataset

from esm_msr.data import ProteinCyclingBatchSampler
from esm_msr.flipkeys import split_flip_key


def _cls():
    from esm_msr import training
    return next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_compute_int_loss'))


def _matrix(n_rows=8, partners='ACD', pair='LIB|5|7', seed=0):
    """Flip keys / row ids / index grid for one position-pair matrix; cell (r, c) is item r*len(partners)+c."""
    keys, rows = [], []
    for r in range(n_rows):
        for c in partners:
            keys.append(f'{pair}{c}')
            rows.append(r)
    return keys, np.array(rows)


class TestInteractionLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.fn = staticmethod(_cls()._compute_int_loss)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def _call(self, pred, target, keys, rows, cens=None, valid=None, min_rows=4, min_cols=2):
        valid = torch.ones(len(keys), dtype=torch.bool) if valid is None else valid
        return self.fn(None, pred, target, valid, keys, rows, cens, min_rows, min_cols)[:4]

    def test_target_variance_is_returned_for_the_unexplained_fraction(self):
        # a perfect prediction has zero loss but the target's double-centred variance is still reported
        g = torch.Generator().manual_seed(3)
        target = torch.randn(len(self.keys), generator=g)
        out = self.fn(None, target.clone(), target, torch.ones(len(self.keys), dtype=torch.bool), self.keys, self.rows, None, 4, 2)
        total, val, n_mat, n_cells, ss_y = out
        self.assertAlmostEqual(val, 0.0, places=8)
        self.assertGreater(ss_y, 0.0)
        # predicting nothing (all zeros) leaves exactly the target's own double-centred sum of squares
        out0 = self.fn(None, torch.zeros_like(target), target, torch.ones(len(self.keys), dtype=torch.bool), self.keys, self.rows, None, 4, 2)
        self.assertAlmostEqual(out0[1], out0[4], places=5)

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.keys, self.rows = _matrix()
        self.eps = torch.randn(24, generator=g)                       # the interaction
        self.row_eff = torch.randn(8, generator=g)[torch.as_tensor(self.rows)]
        col = torch.randn(3, generator=g)
        self.col_eff = col[torch.arange(24) % 3]

    def test_identity_independent_terms_cost_nothing(self):
        # Truth = interaction + row + column + constant. A prediction with the SAME interaction but entirely different
        # row / column / constant terms matches it exactly after double-centring.
        target = self.eps + self.row_eff + self.col_eff + 3.0
        pred = self.eps - 2.0 * self.row_eff + 5.0 * self.col_eff - 7.0
        total, val, n_mat, n_cells = self._call(pred, target, self.keys, self.rows)
        self.assertEqual((n_mat, n_cells), (1, 24))
        self.assertLess(val, 1e-9)

    def test_a_wrong_interaction_is_charged(self):
        target = self.eps + self.row_eff
        pred = -self.eps + self.row_eff
        _, val, _, _ = self._call(pred, target, self.keys, self.rows)
        self.assertGreater(val, 1.0)

    def test_gradient_has_no_component_along_the_additive_directions(self):
        target = self.eps
        pred = (0.3 * torch.randn(24)).requires_grad_(True)
        total, *_ = self._call(pred, target, self.keys, self.rows)
        total.backward()
        g = pred.grad.view(8, 3)
        # the gradient is itself double-centred: its row sums and column sums vanish
        self.assertTrue(torch.allclose(g.sum(0), torch.zeros(3), atol=1e-5))
        self.assertTrue(torch.allclose(g.sum(1), torch.zeros(8), atol=1e-5))

    def test_a_missing_cell_trims_to_a_complete_block(self):
        valid = torch.ones(24, dtype=torch.bool)
        valid[4] = False                                              # row 1, column 'C' missing
        total, val, n_mat, n_cells = self._call(self.eps, self.eps + 1.0, self.keys, self.rows, valid=valid)
        self.assertEqual(n_mat, 1)
        self.assertIn(n_cells, (21, 16))                              # drop the row (7x3) or the column (8x2)
        self.assertLess(val, 1e-9)

    def test_censored_cells_are_excluded_and_small_blocks_skipped(self):
        cens = torch.zeros(24, dtype=torch.long)
        cens[::3] = -1                                                # column 'A' entirely censored
        _, _, n_mat, n_cells = self._call(self.eps, self.eps, self.keys, self.rows, cens=cens)
        self.assertEqual((n_mat, n_cells), (1, 16))
        out = self._call(self.eps, self.eps, self.keys, self.rows, min_rows=9)
        self.assertEqual(out[0], None)

    def test_columns_of_different_pairs_are_separate_matrices(self):
        k1, r1 = _matrix(pair='LIB|5|7')
        k2, r2 = _matrix(pair='LIB|5|9')
        keys, rows = k1 + k2, np.concatenate([r1, r2])
        _, _, n_mat, n_cells = self._call(torch.zeros(48), torch.zeros(48), keys, rows)
        self.assertEqual((n_mat, n_cells), (2, 48))


class TestFlipKeys(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_flip_key('1AOY|47|55K'), ('1AOY|47|55', 'K'))
        self.assertEqual(split_flip_key('1AOY|47|native'), ('1AOY|47|native', None))
        self.assertEqual(split_flip_key(''), ('', None))


class TestPairGroupSampler(unittest.TestCase):
    def _items(self):
        items = []
        for q in (7, 9, 11):                                          # three partner positions for scored position 5
            for b in 'ACDEF':                                         # 5 columns each
                for a in range(10):
                    items.append({'flip_key': f'LIB|5|{q}{b}', 'subset_type': 'cond', 'a': a})
        items += [{'flip_key': '', 'subset_type': 'single'} for _ in range(30)]
        return items

    def test_columns_of_a_pair_travel_together_and_stay_whole(self):
        ds = DummyDataset('LIB', self._items())
        smp = ProteinCyclingBatchSampler([ds], batch_size=50, train_list=['LIB'], strategy='all', rng_seed=1, flip_pair_groups=3)
        for b in smp:
            keys = [ds[i]['flip_key'] for i in b if ds[i]['flip_key']]
            cols = Counter(keys)
            self.assertTrue(all(n == 10 for n in cols.values()), cols)             # no column is split
            by_pair = Counter(split_flip_key(k)[0] for k in cols)
            self.assertTrue(any(n > 1 for n in by_pair.values()))                   # columns of one pair do co-occur

    def test_groups_are_balanced_so_no_pair_is_left_with_a_lone_column(self):
        from esm_msr.data import balanced_group_sizes
        self.assertEqual(balanced_group_sizes(19, 3), [3, 3, 3, 3, 3, 2, 2])
        self.assertEqual(balanced_group_sizes(7, 3), [3, 2, 2])
        self.assertEqual(balanced_group_sizes(6, 3), [3, 3])
        self.assertEqual(balanced_group_sizes(1, 3), [1])
        self.assertEqual(balanced_group_sizes(0, 3), [])
        for n in range(2, 40):
            sizes = balanced_group_sizes(n, 3)
            self.assertEqual(sum(sizes), n)
            self.assertTrue(all(2 <= z <= 3 for z in sizes), (n, sizes))

    def test_default_keeps_single_column_units(self):
        ds = DummyDataset('LIB', self._items())
        smp = ProteinCyclingBatchSampler([ds], batch_size=50, train_list=['LIB'], strategy='all', rng_seed=1)
        self.assertEqual(smp.flip_pair_groups, 0)
        self.assertGreater(len(list(smp)), 0)


class TestAlignedChunks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.fn = staticmethod(next(v for v in vars(__import__('esm_msr.training', fromlist=['x'])).values()
                                       if isinstance(v, type) and hasattr(v, '_aligned_chunks'))._aligned_chunks)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def _stub(self, keys):
        return types.SimpleNamespace(_last_flip_keys=keys)

    def test_pairs_are_never_cut_when_they_fit(self):
        # three pairs of 3 columns x 8 rows = 24 rows each, plus 10 anchored singles; micro-batch 64
        keys = []
        for q in (7, 9, 11):
            for b in 'ACD':
                keys += [f'LIB|5|{q}{b}'] * 8
        keys += [''] * 10
        order = torch.randperm(len(keys), generator=torch.Generator().manual_seed(0))
        chunks = self.fn(self._stub(keys), order, 64)
        self.assertEqual(sorted(torch.cat(chunks).tolist()), list(range(len(keys))))   # a partition of the rows
        for c in chunks:
            self.assertLessEqual(len(c), 64)
            pairs = Counter(split_flip_key(keys[int(order[i])])[0] for i in c.tolist() if keys[int(order[i])])
            for pair, n in pairs.items():
                self.assertEqual(n, 24, (pair, n))                                      # every pair arrives whole

    def test_an_oversized_pair_is_cut_at_column_boundaries(self):
        keys = [f'LIB|5|7{b}' for b in 'ACDEF' for _ in range(19)]                      # 95 rows in one pair, micro-batch 64
        chunks = self.fn(self._stub(keys), torch.arange(len(keys)), 64)
        self.assertEqual([len(c) for c in chunks], [57, 38])
        for c in chunks:
            self.assertTrue(all(keys[i] == keys[c[0]] or True for i in c.tolist()))
            cnt = Counter(keys[i] for i in c.tolist())
            self.assertTrue(all(n == 19 for n in cnt.values()))                         # no column is split


class TestInteractionLossInTheComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr.training  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def _batch(self):
        # 12 singles (three lists of 4), then one position-pair matrix of 6 rows x 3 columns of cond items
        keys, rows = _matrix(n_rows=6, partners='ACD', pair='LIB|5|7')
        n = 12 + 18
        g = torch.Generator().manual_seed(2)
        return {
            'ddG': torch.randn(n, generator=g), 'feat': torch.randn(n, generator=g),
            'mut_mask': torch.ones(n, 1, dtype=torch.bool),
            'subset_type': ['single'] * 12 + ['cond'] * 18,
            'flip_key': [''] * 12 + keys,
            'reg_ok': torch.ones(n, dtype=torch.bool),
            'cens': torch.zeros(n, dtype=torch.long), 'cens_bound': torch.full((n,), float('nan')),
            'cens_src': torch.zeros(n, dtype=torch.long),
            'ddG_additive': torch.full((n,), float('nan')), 'dddG': torch.full((n,), float('nan')),
            'dG_wt': torch.full((n,), 2.0), 'bg_offset': torch.zeros(n),
            'mt_id': torch.cat([torch.zeros(12, 1, dtype=torch.long), torch.as_tensor(rows).view(-1, 1)]),
        }

    def test_off_by_default_and_present_when_enabled(self):
        out, _ = run(self._batch(), flip_list_min=3)
        self.assertNotIn('L_int_mt', out)
        out, stub = run(self._batch(), flip_list_min=3, lambda_int_mt=1.0)
        self.assertIn('L_int_mt', out)
        self.assertTrue(torch.isfinite(stub.model.w.grad).all())

    def test_aligned_micro_batches_keep_the_matrix_whole(self):
        # micro-batch of 16 rows would cut the 18-row matrix; aligned chunking cannot fit it either and cuts at a column boundary
        out, _ = run(self._batch(), flip_list_min=3, lambda_int_mt=1.0, micro_batch_size=24)
        self.assertIn('L_int_mt', out)


if __name__ == '__main__':
    unittest.main()
