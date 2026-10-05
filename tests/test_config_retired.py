"""Retired training flags: old commands keep working when the flag is harmless, and fail loudly when ignoring it would change the model."""
import contextlib
import io
import logging
import sys
import unittest
from unittest import mock

from esm_msr.config import MAX_COLUMN_LEN, RETIRED_FLAGS, parse_arguments

BASE = ['x', '--experiment_name', 't', '--raw_data_file', 'a', '--af_model_folder', 'b']


def parse(*extra):
    with mock.patch.object(sys, 'argv', BASE + list(extra)):
        return parse_arguments()


def parse_fails(*extra):
    with contextlib.redirect_stderr(io.StringIO()) as err, mock.patch.object(sys, 'argv', BASE + list(extra)):
        try:
            parse_arguments()
        except SystemExit:
            return err.getvalue()
    return None


class TestRetiredFlags(unittest.TestCase):
    def test_harmless_values_are_accepted_and_never_reach_hparams(self):
        with self.assertLogs(level=logging.WARNING) as logs:
            a = parse('--lora_mode', 'ensemble', '--detach_ensemble_input', '--reg_loss', 'mse', '--lambda_reg_combined', '0',
                      '--flip_align_units', '--flip_pair_groups', '3', '--dedup_backbone')
        derived = {'flip_pair_groups', 'incl_singles', 'incl_doubles', 'incl_cond', 'incl_reversions', 'incl_native_cond'}   # set from other flags
        for name in set(RETIRED_FLAGS) - derived:
            self.assertFalse(hasattr(a, name), name)
        self.assertTrue(any('--lora_mode is retired' in m for m in logs.output))

    def test_a_value_that_used_to_change_behaviour_is_an_error(self):
        for flags in (['--lora_mode', 'corrector'], ['--lambda_rank_combined', '1.0'], ['--reg_loss', 'huber'], ['--no-dedup_backbone'],
                      ['--detach_regression'], ['--freeze_wt_on_convergence'], ['--use_plddt'], ['--int_min_rows', '2']):
            self.assertIsNotNone(parse_fails(*flags), flags)

    def test_subset_size_is_the_wt_list_size(self):
        self.assertEqual(parse().wt_list_size, 16)
        self.assertEqual(parse('--subset_size', '8').wt_list_size, 8)
        self.assertEqual(parse('--wt_list_size', '32').wt_list_size, 32)

    def test_doubles_and_reversions_are_not_training_subsets(self):
        self.assertIsNotNone(parse_fails('--subset_caps', 'single=None', 'double=0.5'))
        self.assertIsNotNone(parse_fails('--subset_caps', 'single=None', 'reversion=None'))
        a = parse('--subset_caps', 'single=None', 'cond=None', 'native_cond=None')
        self.assertEqual((a.incl_singles, a.incl_cond, a.incl_native_cond, a.incl_doubles, a.incl_reversions), (True, True, True, False, False))

    def test_pair_groups_follow_the_micro_batch_when_the_interaction_loss_is_on(self):
        self.assertEqual(MAX_COLUMN_LEN, 19)
        self.assertEqual(parse('--lambda_int_mt', '30', '--micro_batch_size', '64').flip_pair_groups, 3)     # what the I-series used
        self.assertEqual(parse('--lambda_int_mt', '30', '--micro_batch_size', '128').flip_pair_groups, 6)
        self.assertEqual(parse('--micro_batch_size', '64').flip_pair_groups, 0)
        self.assertIsNotNone(parse_fails('--lambda_int_mt', '30', '--micro_batch_size', '32'))                # cannot hold two columns


if __name__ == '__main__':
    unittest.main()
