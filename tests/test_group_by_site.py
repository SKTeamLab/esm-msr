"""``--group_by_site`` and ``--mask_dist_wt/_mt``: the throughput paradigm.

Batches carry whole sites (the singles at one position, the doubles of one position pair); a pass whose input is the same sequence for
every item of a group is read once; ListMLE lists are drawn at random across the micro-batch; and a pass is masked with a pattern
drawn from (no site, exactly one site, all sites) weights. The backbone is the stand-in of test_pass_masking.py.
"""
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import config
from esm_msr.data import ProteinCyclingBatchSampler
from esm_msr.training import ESM3EpistasisLightningModule as M

from test_pass_masking import MASK, _double, _model
from test_four_pass import _cat, _single

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def _train(m):
    """Put the stand-in in training mode the way the real one is: its PEFT adapters report .training, which switches off the eval-only shortcuts."""
    m.train()
    m.peft_wt, m.peft_mt = types.SimpleNamespace(training=True), types.SimpleNamespace(training=True)
    return m


def _rows(m):
    seen, inner = [], m._get_esm3_outputs

    def counting(seq, *a, **k):
        seen.append(int(seq.shape[0]))
        return inner(seq, *a, **k)
    m._get_esm3_outputs = counting
    return seen


# ---------------------------------------------------------------------------------------------------------------- the flags

def test_the_flags_and_their_validation():
    a = parse('--group_by_site', '--mask_dist_wt', '1', '2', '1', '--mask_dist_mt', '0', '1', '0')
    assert a.group_by_site is True and a.mask_dist_wt == [1.0, 2.0, 1.0] and a.mask_dist_mt == [0.0, 1.0, 0.0]
    assert parse().group_by_site is False and parse().mask_dist_wt is None and parse().mask_dist_mt is None
    for bad in (('-1', '1', '1'), ('0', '0', '0')):
        with pytest.raises(SystemExit):
            parse('--mask_dist_mt', *bad)


# ---------------------------------------------------------------------------------------------------------------- one forward per distinct sequence

def _batch(n=6):
    b = _cat([_single(site=1, mut=9 + k) for k in range(n)])
    b['pdb'] = ['LIB'] * n
    return b


@pytest.mark.parametrize('pass_type', ['wt', 'mt'])
def test_masked_singles_at_one_site_share_one_forward_and_the_numbers_do_not_change(pass_type):
    b = _batch()
    ref, new = _model(), _model()
    new.dedup_rows = True
    for m in (ref, new):
        _train(m)
        m.random_mask_dist_wt = m.random_mask_dist_mt = (0.0, 0.0, 1.0)                      # every mutated site masked: the six mutants are one sequence
    r_ref, r_new = _rows(ref), _rows(new)
    a = ref.forward_partitioned(b, pass_type=pass_type, mask_strategy='random')
    c = new.forward_partitioned(b, pass_type=pass_type, mask_strategy='random')
    assert r_ref == [6] and r_new == [1], (r_ref, r_new)
    assert torch.allclose(a['pred_calibrated'], c['pred_calibrated']) and torch.allclose(a['unsummed_llr'], c['unsummed_llr'])


def test_an_unmasked_wt_pass_is_one_forward_for_the_whole_group_and_an_unmasked_mt_pass_is_not():
    b = _batch()
    m = _train(_model())
    m.dedup_rows = True
    rows = _rows(m)
    m.forward_partitioned(b, pass_type='wt', mask_strategy=None)        # every row carries the wild-type sequence
    m.forward_partitioned(b, pass_type='mt', mask_strategy=None)        # six different mutant sequences
    assert rows == [1, 6], rows


def test_the_graph_survives_the_dedup():
    b = _batch()
    m = _train(_model())
    m.dedup_rows = True
    m.base = torch.nn.Parameter(m.base.clone())                         # the stand-in's only learnable thing
    out = m.forward_partitioned(b, pass_type='wt', mask_strategy=None)['pred_calibrated']
    out.sum().backward()
    assert m.base.grad is not None and float(m.base.grad.abs().sum()) > 0


def test_dedup_is_off_unless_asked_so_existing_runs_keep_their_per_row_forwards():
    m = _train(_model())
    rows = _rows(m)
    m.forward_partitioned(_batch(), pass_type='wt', mask_strategy=None)
    assert rows == [6]


# ---------------------------------------------------------------------------------------------------------------- the mask pattern

def _pattern_counts(dist, n=600, double=True):
    torch.manual_seed(0)
    m = _model()
    m.train()
    m.random_mask_dist_mt = dist
    seen = _rows(m)
    inner = m._get_esm3_outputs
    masks = []

    def rec(seq, *a, **k):
        masks.append(int((seq == MASK).sum()))
        return inner(seq, *a, **k)
    m._get_esm3_outputs = rec
    b = _double() if double else _single()
    for _ in range(n):
        m.forward_partitioned(b, pass_type='mt', mask_strategy='random')
    return np.bincount(masks, minlength=3) / n


def test_a_pattern_draw_masks_none_exactly_one_or_all_sites_of_a_double():
    assert _pattern_counts((1, 0, 0)).tolist() == [1.0, 0.0, 0.0]
    assert _pattern_counts((0, 1, 0)).tolist() == [0.0, 1.0, 0.0]            # exactly one, never two, never none
    assert _pattern_counts((0, 0, 1)).tolist() == [0.0, 0.0, 1.0]
    got = _pattern_counts((1, 2, 1))                                          # the weights of an independent p = 0.5 draw
    assert got == pytest.approx([0.25, 0.5, 0.25], abs=0.07)


def test_exactly_one_picks_either_site_of_a_double():
    torch.manual_seed(3)
    m = _model()
    m.train()
    m.random_mask_dist_mt = (0, 1, 0)
    b = _double()
    which = set()
    inner = m._get_esm3_outputs

    def rec(seq, *a, **k):
        which.add(tuple((seq[0, b['mut_pos'][0]] == MASK).tolist()))
        return inner(seq, *a, **k)
    m._get_esm3_outputs = rec
    for _ in range(60):
        m.forward_partitioned(b, pass_type='mt', mask_strategy='random')
    assert which == {(True, False), (False, True)}


def test_for_a_single_one_and_all_are_the_same_event():
    got = _pattern_counts((0, 1, 0), double=False)
    assert got[1] == 1.0 and _pattern_counts((0, 0, 1), double=False)[1] == 1.0


# ---------------------------------------------------------------------------------------------------------------- the sampler

class _Ds:
    """Items of one library: singles at several positions and the doubles of one pair, as the cache lays them out."""

    def __init__(self, n_pos=6, per_pos=5, n_pair=12):
        self.items = []
        for p in range(n_pos):
            for k in range(per_pos):
                self.items.append({'pdb': 'LIB', 'subset_type': 'single', 'mut_pos': np.array([p])})
        for k in range(n_pair):
            self.items.append({'pdb': 'LIB', 'subset_type': 'double', 'mut_pos': np.array([2, 4])})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _sites(ds, batch):
    return [tuple(sorted(ds[i]['mut_pos'].tolist())) for i in batch]


def test_grouped_batches_keep_each_site_together():
    ds = _Ds()
    s = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=0, group_by_site=True)
    batches = list(s)
    assert len(batches) == len(ds) // 8
    flat = [i for b in batches for i in b]
    keys = _sites(ds, flat)
    runs = 1 + sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    assert runs <= len(set(keys)) + len(batches)                  # a site is one run, apart from a cut at a batch boundary
    plain = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=0)
    pk = _sites(ds, [i for b in plain for i in b])
    assert 1 + sum(1 for a, b in zip(pk, pk[1:]) if a != b) > runs      # the ungrouped order is far more fragmented


def test_grouped_sampling_is_reproducible_and_resumable():
    ds = _Ds()
    a = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=5, group_by_site=True)
    b = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=5, group_by_site=True)
    assert list(a) == list(b)
    e1 = list(a)
    c = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=5, group_by_site=True)
    c.set_epoch(1)
    assert list(c) == e1


def test_the_default_sampler_is_unchanged():
    ds = _Ds()
    a = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=2)
    b = ProteinCyclingBatchSampler([ds], 8, ['LIB'], rng_seed=2, group_by_site=False)
    assert list(a) == list(b)


# ---------------------------------------------------------------------------------------------------------------- rank lists

class _Rec:
    """A ListMLE stand-in that records the targets of every list it is handed."""

    def __init__(self):
        self.lists = []

    def forward_censored(self, p, t, m, c):
        self.lists.append(t[0].tolist())
        return p.sum() * 0.0


def _rank(shuffle):
    torch.manual_seed(0)
    stub = types.SimpleNamespace()
    n = 32
    pred = torch.randn(n, requires_grad=True)
    tgt = torch.arange(n, dtype=torch.float32)                           # rows 0-15 = one site, 16-31 = another
    crit = _Rec()
    M._compute_rank_loss(stub, pred, tgt, torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long), 16, crit, shuffle=shuffle)
    return crit.lists


def test_lists_are_consecutive_rows_unless_shuffled_and_then_they_cross_sites():
    plain = _rank(False)
    assert plain[0] == list(range(16)) and plain[1] == list(range(16, 32))
    mixed = _rank(True)
    assert sorted(sum(mixed, [])) == list(range(32))                     # nothing lost or repeated
    assert any(x < 16 for x in mixed[0]) and any(x >= 16 for x in mixed[0])
