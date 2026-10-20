"""
--alternate_heads: one head per batch, the WT head on singles and the MT head on multi-mutants.

Every training batch is ONE library and the median training library holds no multi-mutants at all, so a strict
alternation would waste every other step on most libraries. The decision therefore falls back to the other head
when a batch cannot serve its turn.
"""
import pytest

from esm_msr.training import alternating_phase as phase


def test_it_alternates_when_a_batch_can_serve_either():
    assert [phase(i, True, True) for i in range(6)] == ['wt', 'mt', 'wt', 'mt', 'wt', 'mt']


def test_a_batch_with_no_multi_mutants_always_trains_wt():
    """The common case: the median training library has no doubles, so an 'mt' turn would do nothing."""
    assert [phase(i, True, False) for i in range(4)] == ['wt'] * 4


def test_a_batch_with_no_singles_always_trains_mt():
    assert [phase(i, False, True) for i in range(4)] == ['mt'] * 4


def test_an_empty_batch_is_skipped():
    assert phase(0, False, False) == 'none'
    assert phase(1, False, False) == 'none'


def test_the_turns_are_balanced_over_many_mixed_batches():
    turns = [phase(i, True, True) for i in range(1000)]
    assert turns.count('wt') == turns.count('mt') == 500


def test_the_flags_default_off():
    from esm_msr import config
    import sys
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        a = config.parse_arguments()
    finally:
        sys.argv = argv
    assert a.alternate_heads is False
    # tf_delink flipped to ON by default on 2026-10-19: an observed-scale additive label must not be injected into a
    # latent half (see config's help). --no-tf_delink restores the old behaviour.
    assert a.tf_delink is True
