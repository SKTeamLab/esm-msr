"""--reg_balance: regression weights in units of the rank loss's gradient (constants measured by scripts/grad_share_probe.py)."""
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from esm_msr import training
from esm_msr.config import MAX_COLUMN_LEN, parse_arguments

BASE = ['x', '--experiment_name', 't', '--raw_data_file', 'a', '--af_model_folder', 'b']


def parse(*extra):
    with mock.patch.object(sys, 'argv', BASE + list(extra)):
        return parse_arguments()


class TestBalance(unittest.TestCase):
    def test_constants_cover_every_regression_term_and_are_positive(self):
        self.assertEqual(set(training.REG_BALANCE), {'reg_wt', 'reg_mt', 'comp_off', 'comp_subst', 'comp_int'})
        self.assertTrue(all(v > 0 for v in training.REG_BALANCE.values()))

    def test_off_is_exactly_one_and_on_is_the_constant(self):
        for name, k in training.REG_BALANCE.items():
            self.assertEqual(training.balance({'reg_balance': False}, name), 1.0)
            self.assertEqual(training.balance({}, name), 1.0)
            self.assertEqual(training.balance({'reg_balance': True}, name), k)

    def test_balance_forces_the_component_path_even_at_unit_weights(self):
        self.assertFalse(training.comp_on({'mt_comp_offset': 1.0, 'mt_comp_subst': 1.0, 'mt_comp_int': 1.0}))
        self.assertTrue(training.comp_on({'mt_comp_offset': 1.0, 'mt_comp_subst': 1.0, 'mt_comp_int': 1.0, 'reg_balance': True}))

    def test_config_derives_pair_groups_and_needs_two_columns_per_micro_batch(self):
        a = parse('--reg_balance', '--micro_batch_size', '64')
        self.assertTrue(a.reg_balance)
        self.assertEqual(a.flip_pair_groups, 64 // MAX_COLUMN_LEN)
        self.assertEqual(parse('--micro_batch_size', '64').flip_pair_groups, 0)
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(sys, 'argv', BASE + ['--reg_balance', '--micro_batch_size', '16']):
            with self.assertRaises(SystemExit):
                parse_arguments()


class TestTap(unittest.TestCase):
    def test_no_probe_returns_the_term_untouched_even_for_a_stand_in_object(self):
        t = torch.tensor(1.5)
        self.assertIs(training._tap(SimpleNamespace(), 'x', t), t)
        self.assertIs(training._tap(SimpleNamespace(_probe=None), 'x', t), t)

    def test_probe_records_names_in_order(self):
        obj = SimpleNamespace(_probe=[])
        a, b = torch.tensor(1.0), torch.tensor(2.0)
        self.assertIs(training._tap(obj, 'rank_wt', a), a)
        training._tap(obj, 'reg_wt', b)
        self.assertEqual([n for n, _ in obj._probe], ['rank_wt', 'reg_wt'])


if __name__ == '__main__':
    unittest.main()
