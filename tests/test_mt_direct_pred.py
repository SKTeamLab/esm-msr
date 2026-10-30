"""`--mt_direct_pred`: the MT adapter predicts directly, the zero-shot WT adapter is frozen, no teacher forcing.

The flag is a named combination of primitive settings derived in config.parse_arguments; off (the default) it must change nothing.
"""
import sys

import pytest

from esm_msr import config

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_off_by_default_and_the_default_parse_is_unchanged():
    a, b = parse(), parse('--no-mt_direct_pred')
    assert a.mt_direct_pred is False and a.freeze_wt_adapter is False
    assert vars(a) == vars(b)


def test_the_flag_freezes_the_wt_adapter_and_zeroes_its_losses():
    base, a = parse(), parse('--mt_direct_pred')
    assert a.mt_direct_pred is True and a.freeze_wt_adapter is True
    assert a.lambda_reg_wt == 0.0 and a.lambda_rank_wt == 0.0
    assert a.lambda_reg_mt == base.lambda_reg_mt and a.lambda_rank_mt == base.lambda_rank_mt      # the MT objective is untouched
    assert a.tf_mix == base.tf_mix


def test_the_epistasis_metric_that_needs_a_wt_head_is_rerouted():
    a = parse('--mt_direct_pred', '--mt_convergence_metric', 'epi_beyond_rho_epi', '--monitor_metric', 'val_epi_beyond_rho_epi')
    assert a.mt_convergence_metric == 'rho_combined' and a.monitor_metric == 'val_rho_combined_avg'


def test_fused_adapters_are_refused():
    with pytest.raises(SystemExit):
        parse('--mt_direct_pred', '--adapter_mode', 'fused')
