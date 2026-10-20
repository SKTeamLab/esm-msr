"""The MT pass on native_cond must see the WT pass's input (the background / pseudo-WT), not the fully mutated one.

A mutant-background library's mt_sequence_tokens carries the background mutation AND the scored mutation. The MT
head's task there is the conditional effect from the library's starting sequence, so it gets the same tokens the WT
pass sees; the remaining sequence/structure mismatch is the MT head's domain.
"""
import torch

from esm_msr.models import native_cond_mt_batch


def _batch(subset_types):
    B = len(subset_types)
    wt = torch.tensor([[1, 2, 3, 4]] * B)
    mt = torch.tensor([[1, 9, 3, 4]] * B)          # the mutant carries the scored mutation at position 1
    return {'subset_type': list(subset_types), 'wt_sequence_tokens': wt, 'mt_sequence_tokens': mt}


def test_native_cond_rows_get_the_wt_sequence():
    b = _batch(['single', 'native_cond', 'double'])
    out = native_cond_mt_batch(b)
    assert torch.equal(out['mt_sequence_tokens'][0], b['mt_sequence_tokens'][0])   # single: untouched
    assert torch.equal(out['mt_sequence_tokens'][1], b['wt_sequence_tokens'][1])   # the native_cond row
    assert torch.equal(out['mt_sequence_tokens'][2], b['mt_sequence_tokens'][2])   # double: untouched


def test_no_native_cond_returns_the_same_batch():
    b = _batch(['single', 'double'])
    assert native_cond_mt_batch(b) is b


def test_the_input_batch_is_not_mutated():
    b = _batch(['native_cond'])
    before = b['mt_sequence_tokens'].clone()
    native_cond_mt_batch(b)
    assert torch.equal(b['mt_sequence_tokens'], before)
