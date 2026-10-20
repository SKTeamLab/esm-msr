"""`--incl_*` and `--subset_caps` must agree — both directions.

Both spellings exist: `--incl_doubles True/False` and `--subset_caps double=<cap>`, where a cap of 0 means "never
load it" and `None` means "unrestricted". They disagreed: `parse_arguments` derived the flags from the caps only,
so `--no-incl_doubles` was parsed and then silently discarded (the cap, None by default, won). A run that asked to
drop doubles kept them, and nothing said so.

Now an explicitly passed `--no-incl_*` forces that subset's cap to 0.0, so the dataset, the batch sampler and the
recorded hparams all agree. The reverse direction still works: a cap of 0.0 turns the flag off, and a positive cap
turns it on even for the subsets whose flag defaults to False (cond, native_cond) — which is why the fix keys on the
flag being passed EXPLICITLY rather than on its value.
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


def test_defaults_load_singles_and_doubles():
    a = parse()
    assert a.incl_singles is True and a.incl_doubles is True
    assert a.subset_caps['single'] is None and a.subset_caps['double'] is None


def test_no_incl_doubles_is_honoured_and_sets_the_cap():
    """The regression: this used to leave incl_doubles True."""
    a = parse('--no-incl_doubles')
    assert a.incl_doubles is False
    assert a.subset_caps['double'] == 0.0           # the cap now says the same thing as the flag


def test_no_incl_singles_is_honoured_and_sets_the_cap():
    a = parse('--no-incl_singles')
    assert a.incl_singles is False
    assert a.subset_caps['single'] == 0.0


def test_a_zero_cap_still_turns_the_flag_off():
    """The other direction, unchanged: the cap is the sampler's own semantic."""
    a = parse('--subset_caps', 'double=0.0')
    assert a.incl_doubles is False
    assert a.subset_caps['double'] == 0.0


def test_a_positive_cap_keeps_the_subset_and_its_value():
    a = parse('--subset_caps', 'double=0.5')
    assert a.incl_doubles is True
    assert a.subset_caps['double'] == 0.5


def test_a_requested_cap_survives_a_flag_that_defaults_to_false():
    """cond/native_cond default to False; asking for a cap must still turn them on."""
    a = parse('--subset_caps', 'cond=0.2')
    assert a.incl_cond is True and a.subset_caps['cond'] == 0.2
    a = parse('--subset_caps', 'native_cond=0.3')
    assert a.incl_native_cond is True and a.subset_caps['native_cond'] == 0.3


def test_the_frozen_wt_paradigm_keeps_singles_by_default(tmp_path):
    """--frozen_wt_mt_only no longer forces a doubles-only stream (2026-10-19): singles stay in unless asked out."""
    ckpt = tmp_path / 'wt.ckpt'
    import torch
    torch.save({'state_dict': {}}, ckpt)
    a = parse('--frozen_wt_mt_only', '--init_wt_from_checkpoint', str(ckpt))
    assert a.incl_singles is True and a.subset_caps['single'] is None
    assert a.incl_doubles is True and a.subset_caps['double'] is None      # doubles unrestricted, not capped
    # and the old doubles-only stream is still reachable, coherently: the flag and the cap agree
    b = parse('--frozen_wt_mt_only', '--init_wt_from_checkpoint', str(ckpt), '--no-incl_singles')
    assert b.incl_singles is False and b.subset_caps['single'] == 0.0


def test_the_paradigm_refuses_a_run_with_nothing_to_train_on(tmp_path):
    ckpt = tmp_path / 'wt.ckpt'
    import torch
    torch.save({'state_dict': {}}, ckpt)
    with pytest.raises(SystemExit):
        parse('--frozen_wt_mt_only', '--init_wt_from_checkpoint', str(ckpt), '--subset_caps', 'double=0.0')
