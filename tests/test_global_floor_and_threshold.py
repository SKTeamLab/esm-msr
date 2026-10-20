"""A learned global floor with a prior, and a censoring threshold decoupled from the link (2026-10-20).

* ``--assay_floor X --learn_global_floor`` (floor_source 'global'): every item carries y_floor NaN, so the knee is the link's
  global_floor, initialised at X and held near it by ``--link_floor_penalty`` / ``--link_floor_prior_sd``.
* ``--censor_threshold T`` (singles) / ``--censor_threshold_double T2`` (multi-mutants, default T): an item read at or below its
  threshold is lower-censored in every library, bounded by it; a single censored that way supplies no teacher-forcing label.
Unset, every recipe behaves as before.
"""
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import censoring, config
from esm_msr.link import SaturatingLink
from esm_msr.training import teacher_forcing_unusable

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', '--link']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ config
def test_floor_source_is_derived_and_recorded():
    assert parse().floor_source == 'library'
    assert parse('--assay_floor', '0.0', '--link_floor_min', '0.0', '--link_floor_max', '0.001').floor_source == 'assay'
    a = parse('--assay_floor', '0.0', '--learn_global_floor', '--link_floor_min', '-1.0', '--link_floor_max', '1.0')
    assert a.floor_source == 'global'


def test_a_learned_floor_keeps_censoring_at_the_assay_floor():
    a = parse('--assay_floor', '0.0', '--learn_global_floor', '--link_floor_min', '-1.0', '--link_floor_max', '1.0')
    assert a.censor_threshold == 0.0 and a.censor_threshold_double == 0.0
    b = parse('--assay_floor', '0.0', '--learn_global_floor', '--link_floor_min', '-1.0', '--link_floor_max', '1.0',
              '--censor_threshold', '0.3')
    assert b.censor_threshold == 0.3


def test_a_learned_floor_needs_room_to_move():
    with pytest.raises(SystemExit):
        parse('--assay_floor', '0.0', '--learn_global_floor', '--link_floor_min', '0.0', '--link_floor_max', '0.001')


def test_double_threshold_defaults_to_the_single_one_and_needs_censor_floor():
    a = parse('--censor_threshold', '0.3')
    assert a.censor_threshold == 0.3 and a.censor_threshold_double == 0.3
    assert parse('--censor_threshold', '0.3', '--censor_threshold_double', '0.5').censor_threshold_double == 0.5
    with pytest.raises(SystemExit):
        parse('--censor_threshold', '0.3', '--no-censor_floor')


def test_defaults_leave_the_old_recipe_alone():
    a = parse('--assay_floor', '0.0', '--link_floor_min', '0.0', '--link_floor_max', '0.001')
    assert a.censor_threshold is None and a.censor_threshold_double is None and a.link_floor_penalty == 0.0


# ------------------------------------------------------------------ link prior
def test_floor_penalty_is_zero_at_the_anchor_quadratic_away_and_off_when_frozen():
    link = SaturatingLink(floor_min=-1.0, floor_max=1.0, global_floor_init=0.0, learn_global_floor=True)
    assert float(link.floor_penalty(1.0, 0.0, 0.25)) == pytest.approx(0.0)
    with torch.no_grad():
        link.global_floor.fill_(0.25)
    assert float(link.floor_penalty(1.0, 0.0, 0.25)) == pytest.approx(1.0)
    assert float(link.floor_penalty(2.0, 0.0, 0.5)) == pytest.approx(0.5)
    frozen = SaturatingLink(floor_min=-1.0, floor_max=1.0, global_floor_init=0.3, learn_global_floor=False)
    assert not frozen.floor_penalty(1.0, 0.0, 0.25).requires_grad and float(frozen.floor_penalty(1.0, 0.0, 0.25)) == 0.0


def test_floor_penalty_pulls_the_floor_back():
    link = SaturatingLink(floor_min=-1.0, floor_max=1.0, global_floor_init=0.4, learn_global_floor=True)
    link.floor_penalty(1.0, 0.0, 0.25).backward()
    assert float(link.global_floor.grad) > 0          # gradient descent lowers it toward the anchor


# ------------------------------------------------------------------ dataset
def _ds(items, floor_stats, assay_floor, floor_source=None, t1=None, t2=None):
    from esm_msr.data import MutationStabilityDataset as D
    d = types.SimpleNamespace(data=items, floor_stats=floor_stats, assay_floor=assay_floor, floor_source=floor_source,
                              censor_threshold=t1, censor_threshold_double=t2, dms_name='LIB')
    d._stamp_floor = types.MethodType(D._stamp_floor, d)
    d._apply_censoring = types.MethodType(D._apply_censoring, d)
    return d


def _item(dG_meas, subset='single', dG_wt=2.0):
    return {'dG_meas': dG_meas, 'ddG': dG_meas - dG_wt, 'meas': None, 'subset_type': subset}


def test_global_floor_source_stamps_nan_everywhere():
    d = _ds([_item(0.5), _item(-0.2)], {'evidence': 1.0, 'y_floor': -0.3}, 0.0, floor_source='global')
    d._stamp_floor()
    assert all(np.isnan(i['y_floor']) for i in d.data) and not d.has_floor_evidence


def test_threshold_censors_every_library_with_its_own_value_per_subset():
    items = [_item(0.2), _item(0.4), _item(0.4, 'double'), _item(0.6, 'double')]
    d = _ds(items, {'evidence': 0.0}, None, t1=0.3, t2=0.5)     # a library with NO floor evidence
    d._stamp_floor(); d._apply_censoring(True, None)
    assert [i['cens'] for i in d.data] == [-1, 0, -1, 0]
    assert d.data[0]['cens_src'] == censoring.SRC_FLOOR
    assert d.data[0]['cens_bound'] == pytest.approx(0.3 - 2.0)   # the threshold, on the item's own ddG scale
    assert d.data[2]['cens_bound'] == pytest.approx(0.5 - 2.0)


def test_without_a_threshold_censoring_is_unchanged():
    d = _ds([_item(0.2), _item(-0.1)], {'evidence': 0.0}, 0.0)
    d._stamp_floor(); d._apply_censoring(True, None)
    assert [i['cens'] for i in d.data] == [0, -1]                 # censored at the asserted floor 0, as before


# ------------------------------------------------------------------ teacher forcing
def test_a_double_with_a_single_at_or_below_the_cut_gets_no_label():
    dG_wt = torch.tensor([2.0, 2.0, 2.0])
    ddG_A = torch.tensor([-1.0, -1.8, -1.0])          # readings 1.0, 0.2, 1.0
    ddG_B = torch.tensor([-0.5, -0.5, -1.7])          # readings 1.5, 1.5, 0.3
    out = teacher_forcing_unusable(dG_wt, ddG_A, ddG_B, torch.full((3,), 0.3))
    assert out.tolist() == [False, True, True]


# ------------------------------------------------------------------ teacher forcing, through the real loss path
def test_threshold_drops_a_doubles_label_when_one_single_reads_below_it():
    from test_tf_mix import _stub, _batch, _combined_loss
    b = _batch()
    b['ddG_A'] = torch.full((6,), -1.8)                 # reads 0.2 (dG_wt 2)
    b['ddG_B'] = torch.full((6,), -0.5)                 # reads 1.5
    assert _combined_loss(_stub(), b) == pytest.approx((0.5 * 1.0 + 1.0) ** 2, abs=1e-6)   # above y_floor -0.2: label used
    s = _stub(); s.hparams['censor_threshold'] = 0.3
    assert _combined_loss(s, b) == pytest.approx((0.5 * 2.0 + 1.0) ** 2, abs=1e-6)          # WT pass instead


def test_a_censored_single_supplies_no_label_for_itself_under_a_threshold():
    from test_tf_mix import _stub, _batch, _combined_loss
    b = _batch(doubles=False)
    b['cens'] = torch.full((6,), -1, dtype=torch.long)
    b['cens_bound'] = torch.full((6,), 0.3 - 2.0)
    # without a threshold the forced half is the single's own (bounded) reading: comb -0.5, hinge vs -1.7
    assert _combined_loss(_stub(), b) == pytest.approx((-0.5 + 1.7) ** 2, abs=1e-6)
    s = _stub(); s.hparams['censor_threshold'] = 0.3
    # with it the WT pass (2.0) forms the half: comb 1.0, hinge vs -1.7
    assert _combined_loss(s, b) == pytest.approx((1.0 + 1.7) ** 2, abs=1e-6)
