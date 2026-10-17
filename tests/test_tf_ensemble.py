"""
The reimplemented teacher-forced ensemble, and specifically the ratio that decides where the WT half comes from.

The legacy objective always used the measured additive label when it had one. That is the regime the MT head will
NOT be in at inference: there is no label then, only the WT head's own prediction, which carries its own error
the MT head never saw. --tf_label_prob mixes the two regimes, resampled every step.
"""
import pytest
import torch

from esm_msr.training import tf_wt_half

NAN = float('nan')
LBL = torch.tensor([-1.0, -2.0, NAN, 0.5])        # the third item has no measured additive label
PRED = torch.tensor([-1.2, -1.7, -0.3, 0.9])


def test_ratio_one_is_the_legacy_behaviour():
    """Always the label where there is one; the unlabelled item still falls back to the prediction."""
    half, took = tf_wt_half(LBL, PRED, 1.0)
    torch.testing.assert_close(half, torch.tensor([-1.0, -2.0, -0.3, 0.5]))
    assert took.tolist() == [True, True, False, True]


def test_ratio_zero_is_fully_free_running():
    half, took = tf_wt_half(LBL, PRED, 0.0)
    torch.testing.assert_close(half, PRED)
    assert took.tolist() == [False] * 4


def test_an_item_without_a_label_never_takes_one():
    for p in (0.0, 0.3, 0.75, 1.0):
        _, took = tf_wt_half(LBL, PRED, p)
        assert took[2].item() is False, p


def test_the_realised_ratio_matches_the_requested_one():
    torch.manual_seed(0)
    lbl = torch.zeros(20000)                       # all labelled, so every item is eligible
    pred = torch.ones(20000)
    for p in (0.25, 0.5, 0.9):
        _, took = tf_wt_half(lbl, pred, p)
        assert abs(took.float().mean().item() - p) < 0.02, p


def test_the_draw_is_resampled_so_an_item_sees_both_regimes():
    torch.manual_seed(1)
    lbl, pred = torch.zeros(2000), torch.ones(2000)
    a = tf_wt_half(lbl, pred, 0.5)[1]
    b = tf_wt_half(lbl, pred, 0.5)[1]
    assert not torch.equal(a, b), "the same mask twice: the ratio would be a fixed per-item split, not a regime mix"


def test_the_half_is_exactly_one_of_the_two_sources_per_item():
    """No blending: an item takes the label or the prediction, never an average of them."""
    torch.manual_seed(2)
    half, took = tf_wt_half(LBL, PRED, 0.5)
    for i in range(4):
        src = LBL[i] if took[i] else PRED[i]
        assert half[i] == src, (i, half[i], src)


def test_it_is_off_by_default():
    from esm_msr import config
    import sys
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        a = config.parse_arguments()
    finally:
        sys.argv = argv
    assert a.tf_ensemble is False
    assert a.tf_label_prob == 1.0          # so turning it on alone reproduces the legacy regime
