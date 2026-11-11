"""A mutant-background library ('2M7O_I21A': one introduced mutation, then a scan of single mutants) is a set of ordinary singles to both heads. The only switch is
``--exclude_mutbg``, which leaves such libraries out of training (validation still scores them)."""
import sys

import pytest

from esm_msr import config


def parse(*argv):
    old = sys.argv
    try:
        sys.argv = ['training.py', '--experiment_name', 'x'] + list(argv)
        return config.parse_arguments()
    finally:
        sys.argv = old


def test_the_default_trains_on_them_and_the_flag_leaves_them_out():
    assert parse().exclude_mutbg is False
    assert parse('--exclude_mutbg').exclude_mutbg is True
    assert parse('--exclude_mutbg', '--no-exclude_mutbg').exclude_mutbg is False


def test_the_retirement_message_names_the_replacement(capsys):
    with pytest.raises(SystemExit):
        parse('--no-native_cond_for_wt')
    assert '--exclude_mutbg' in capsys.readouterr().err
