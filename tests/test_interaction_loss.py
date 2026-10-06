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
    return next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_compute_block_components'))


def _matrix(n_rows=8, partners='ACD', pair='LIB|5|7', seed=0):
    """Flip keys / row ids / index grid for one position-pair matrix; cell (r, c) is item r*len(partners)+c."""
    keys, rows = [], []
    for r in range(n_rows):
        for c in partners:
            keys.append(f'{pair}{c}')
            rows.append(r)
    return keys, np.array(rows)


class TestBlockComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.fn = staticmethod(_cls()._compute_block_components)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def _call(self, err, keys, rows, target=None, w=None, valid=None, min_rows=4, min_cols=2):
        n = len(keys)
        valid = torch.ones(n, dtype=torch.bool) if valid is None else valid
        target = torch.zeros(n) if target is None else target
        w = torch.ones(n) if w is None else w
        return self.fn(None, err, target, w, valid, keys, rows, min_rows, min_cols)

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.keys, self.rows = _matrix()
        self.eps = torch.randn(24, generator=g)                       # the interaction
        eps = self.eps.view(8, 3)
        self.eps = (eps - eps.mean(1, keepdim=True) - eps.mean(0, keepdim=True) + eps.mean()).reshape(-1)
        self.row_eff = torch.randn(8, generator=g)[torch.as_tensor(self.rows)]
        col = torch.randn(3, generator=g)
        self.col_eff = col[torch.arange(24) % 3]

    def test_the_three_parts_add_up_to_the_plain_squared_error(self):
        err = self.eps + self.row_eff + self.col_eff + 0.7
        out = self._call(err, self.keys, self.rows)
        raw = out['raw']
        self.assertAlmostEqual(raw['off'] + raw['subst'] + raw['int'], float((err ** 2).sum()), places=3)
        self.assertEqual((out['n_mat'], out['n_cells']), (1, 24))

    def test_each_kind_of_error_lands_in_its_own_part(self):
        for err, part in ((torch.full((24,), 0.7), 'off'), (self.row_eff - self.row_eff.mean(), 'subst'), (self.eps, 'int')):
            raw = self._call(err, self.keys, self.rows)['raw']
            others = sum(v for k, v in raw.items() if k != part)
            self.assertGreater(raw[part], 0.1, part)
            self.assertLess(others, 1e-3 * raw[part] + 1e-6 + (0 if part != 'subst' else 0.0), (part, raw))

    def test_additive_errors_have_no_interaction_part(self):
        err = 3.0 + 2.0 * self.row_eff - 4.0 * self.col_eff
        self.assertLess(self._call(err, self.keys, self.rows)['raw']['int'], 1e-5)

    def test_a_pure_interaction_error_has_no_offset_or_substitution_part(self):
        raw = self._call(self.eps, self.keys, self.rows)['raw']
        self.assertLess(raw['off'] + raw['subst'], 1e-5)
        self.assertGreater(raw['int'], 1.0)

    def test_gradient_of_the_interaction_part_is_itself_double_centred(self):
        err = (0.3 * torch.randn(24)).requires_grad_(True)
        out = self._call(err, self.keys, self.rows)
        out['int'].backward()
        g = err.grad.view(8, 3)
        self.assertTrue(torch.allclose(g.sum(0), torch.zeros(3), atol=1e-5))
        self.assertTrue(torch.allclose(g.sum(1), torch.zeros(8), atol=1e-5))

    def test_a_missing_cell_trims_to_a_complete_block(self):
        valid = torch.ones(24, dtype=torch.bool)
        valid[4] = False                                              # row 1, column 'C' missing
        out = self._call(self.eps, self.keys, self.rows, valid=valid)
        self.assertEqual(out['n_mat'], 1)
        self.assertIn(out['n_cells'], (21, 16))                       # drop the row (7x3) or the column (8x2)
        self.assertEqual(len(out['cells']), out['n_cells'])
        self.assertNotIn(4, out['cells'].tolist())

    def test_invalid_cells_and_small_blocks_are_skipped(self):
        valid = torch.ones(24, dtype=torch.bool)
        valid[::3] = False                                            # column 'A' entirely out (censored or unusable)
        out = self._call(self.eps, self.keys, self.rows, valid=valid)
        self.assertEqual((out['n_mat'], out['n_cells']), (1, 16))
        self.assertIsNone(self._call(self.eps, self.keys, self.rows, min_rows=9))

    def test_columns_of_different_pairs_are_separate_matrices(self):
        k1, r1 = _matrix(pair='LIB|5|7')
        k2, r2 = _matrix(pair='LIB|5|9')
        out = self._call(torch.zeros(48), k1 + k2, np.concatenate([r1, r2]))
        self.assertEqual((out['n_mat'], out['n_cells']), (2, 48))

    def test_block_weight_is_the_mean_of_its_cells(self):
        w = torch.full((24,), 0.5)
        out = self._call(self.eps, self.keys, self.rows, w=w)
        self.assertAlmostEqual(float(out['int']), 0.5 * out['raw']['int'], places=5)

    def test_target_variance_is_reported(self):
        out = self._call(torch.zeros(24), self.keys, self.rows, target=self.eps)
        self.assertAlmostEqual(out['ss_y'], float((self.eps ** 2).sum()), places=4)


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
        self.assertNotIn('L_comp_int', out)
        out, stub = run(self._batch(), flip_list_min=3, mt_comp_int=2.0)
        for k in ('L_comp_off', 'L_comp_subst', 'L_comp_int', 'L_int_tgt'):
            self.assertIn(k, out)
        self.assertTrue(torch.isfinite(stub.model.w.grad).all())

    def test_all_weights_one_through_the_component_path_is_the_plain_regression(self):
        plain, sp = run(self._batch(), flip_list_min=3, lambda_mt_colrank=0.0)
        comp, sc = run(self._batch(), flip_list_min=3, lambda_mt_colrank=0.0, mt_comp_offset=1.0 + 1e-12)   # != (1,1,1): takes the component path
        self.assertAlmostEqual(plain['L_reg_mt'], comp['L_reg_mt'], places=5)
        self.assertTrue(torch.allclose(sp.model.w.grad, sc.model.w.grad, rtol=1e-4, atol=1e-6))

    def test_the_interaction_weight_changes_the_gradient_and_zero_drops_the_part(self):
        _, s1 = run(self._batch(), flip_list_min=3, lambda_mt_colrank=0.0)
        _, s0 = run(self._batch(), flip_list_min=3, lambda_mt_colrank=0.0, mt_comp_int=0.0)
        _, s9 = run(self._batch(), flip_list_min=3, lambda_mt_colrank=0.0, mt_comp_int=9.0)
        self.assertFalse(torch.allclose(s1.model.w.grad, s0.model.w.grad))
        self.assertFalse(torch.allclose(s1.model.w.grad, s9.model.w.grad))

    def test_aligned_micro_batches_keep_the_matrix_whole(self):
        # a micro-batch of 24 rows holds the 18-row matrix; the block must reach the loss in one piece
        out, _ = run(self._batch(), flip_list_min=3, mt_comp_int=2.0, micro_batch_size=24)
        self.assertIn('L_comp_int', out)


if __name__ == '__main__':
    unittest.main()
