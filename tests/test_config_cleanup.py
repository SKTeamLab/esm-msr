"""Retired training flags: still accepted (launch scripts pass them), never reaching hparams; removed ones are refused."""
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


def test_the_launch_scripts_retired_flags_are_still_accepted_and_dropped():
    a = parse('--rank_loss', 'listmle', '--detach_ensemble_input')
    assert not hasattr(a, 'rank_loss') and not hasattr(a, 'detach_ensemble_input')


@pytest.mark.parametrize('extra', [('--no-detach_ensemble_input',), ('--rank_loss', 'mse')])
def test_a_retired_flag_asking_for_the_old_behaviour_is_an_error(extra):
    with pytest.raises(SystemExit):
        parse(*extra)


@pytest.mark.parametrize('flag', ['--residual_wd', '--residual_lr_mult', '--lr_total_steps', '--wt_convergence_patience'])
def test_removed_flags_are_unrecognised(flag):
    with pytest.raises(SystemExit):
        parse(flag, '1')


@pytest.mark.parametrize('flag', ['--detach_regression', '--freeze_wt_on_convergence'])
def test_removed_switches_are_unrecognised(flag):
    with pytest.raises(SystemExit):
        parse(flag)


def test_none_of_them_reaches_hparams():
    keys = set(vars(parse()))
    assert not keys & {'rank_loss', 'detach_ensemble_input', 'residual_wd', 'residual_lr_mult', 'lr_total_steps',
                       'detach_regression', 'freeze_wt_on_convergence', 'wt_convergence_patience'}
    assert 'wt_convergence_metric' in keys          # still read by the WT plateau
