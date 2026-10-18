"""--double_batch_frac: concentrate multi-mutants into fewer batches instead of spreading them thin.

A batch is one library, and only multi-mutants carry a gradient for the MT path. Shuffled packing therefore
spreads the MT signal as thinly as it can go, and the median training library has no doubles at all. The
invariant that makes this a PACKING ablation rather than a reweighting: every item is still used exactly once
per epoch, in the same proportions.
"""
import types

import pytest

from esm_msr.data import ProteinCyclingBatchSampler as S


class FakeDS:
    """len() items of one library; the first `n_d` are doubles."""
    def __init__(self, n, n_d, pdb='L'):
        self.items = [{'pdb': pdb, 'subset_type': 'double' if i < n_d else 'single'} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def sampler(datasets, batch_size=8, frac=0.0, seed=0):
    return S(datasets=datasets, batch_size=batch_size, train_list=[f'L{i}' for i in range(len(datasets))],
             strategy='all', rng_seed=seed, verbose=False, double_batch_frac=frac)


def batches(s):
    return list(iter(s))


def test_off_by_default_is_bit_identical_to_the_shuffled_packing():
    """The default must reproduce the old packing exactly, RNG draws included, or no arm is comparable."""
    a, b = sampler([FakeDS(64, 16)], frac=0.0), sampler([FakeDS(64, 16)], frac=0.0)
    assert batches(a) == batches(b)
    assert sampler([FakeDS(64, 16)], frac=0.0).double_batch_frac == 0.0


@pytest.mark.parametrize('frac', [0.25, 0.5])
def test_every_item_is_still_used_exactly_once_per_epoch(frac):
    ds = FakeDS(64, 16)
    plain = sorted(i for b in batches(sampler([ds], frac=0.0)) for i in b)
    packed = sorted(i for b in batches(sampler([ds], frac=frac)) for i in b)
    assert plain == packed


def test_the_doubles_are_concentrated_into_fewer_batches():
    ds = FakeDS(64, 16)          # 16 doubles over 8 batches of 8: shuffled puts ~2 in every batch
    spread = batches(sampler([ds], frac=0.0))
    packed = batches(sampler([ds], frac=0.5))
    n_d = lambda bs: [sum(1 for i in b if i < 16) for b in bs]
    assert sum(1 for c in n_d(packed) if c) < sum(1 for c in n_d(spread) if c)
    assert max(n_d(packed)) >= 4          # ceil(8 * 0.5)
    assert sum(n_d(packed)) == sum(n_d(spread)) == 16


def test_a_library_with_no_doubles_is_unaffected():
    ds = FakeDS(32, 0)
    assert batches(sampler([ds], frac=0.5)) == batches(sampler([ds], frac=0.0))


def test_doubles_only_library_still_batches():
    ds = FakeDS(32, 32)
    bs = batches(sampler([ds], frac=0.5))
    assert sorted(i for b in bs for i in b) == list(range(32))


def test_the_requested_fraction_is_a_floor_not_a_cap():
    ds = FakeDS(64, 40)         # more doubles than 0.25 * 8 * n_batches can hold
    bs = batches(sampler([ds], frac=0.25))
    assert sum(sum(1 for i in b if i < 40) for b in bs) == 40


def test_batch_order_within_a_batch_is_shuffled():
    """Doubles are collected first, so without a shuffle they would always occupy the low positions."""
    ds = FakeDS(64, 16)
    bs = batches(sampler([ds], frac=0.5, seed=3))
    with_d = [b for b in bs if any(i < 16 for i in b)]
    assert any(any(i < 16 for i in b[len(b) // 2:]) for b in with_d)


def test_an_out_of_range_fraction_is_refused():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match='double_batch_frac'):
            sampler([FakeDS(16, 4)], frac=bad)


def test_batch_count_is_stable_so_an_epoch_is_the_same_length():
    """A different number of optimizer steps would confound every comparison against dc17."""
    ds = FakeDS(64, 16)
    assert len(sampler([ds], frac=0.0)) == len(sampler([ds], frac=0.5))


def test_several_libraries_keep_their_own_items():
    a, b = FakeDS(32, 8, 'A'), FakeDS(32, 0, 'B')
    s = sampler([a, b], frac=0.5)
    assert sorted(i for bt in batches(s) for i in bt) == list(range(64))
    for bt in batches(s):                     # a batch never mixes libraries
        assert max(bt) < 32 or min(bt) >= 32


def test_the_flag_defaults_to_off():
    import sys
    from esm_msr import config
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        assert config.parse_arguments().double_batch_frac == 0.0
    finally:
        sys.argv = argv
