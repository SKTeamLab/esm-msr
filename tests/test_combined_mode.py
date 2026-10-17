"""
The double-mutant ablation turns on how the WT and MT heads are combined, so the three modes have to agree
between training, validation and screening - all of which now go through ``combine_predictions`` - and the
default has to be bit-identical to the released 0.5/0.5 behaviour, because a queued run reads the live worktree.

What the modes mean against a measured ddG target:
  mean  combined = 0.5 wt + 0.5 mt, so MT learns 2*ddG - wt and carries the additive signal too.
  sum   combined = wt + mt, so MT learns ddG - wt: the EPISTASIS term, driven to 0 on singles rather than told.
  route combined = wt for singles, mt otherwise: MT as a standalone double predictor that never sees a single.
"""
import pytest
import torch

from esm_msr.models import COMBINE_MODES, combine_predictions

WT = torch.tensor([-1.0, -2.0, 0.5, 3.0])
MT = torch.tensor([-1.5, 0.5, -0.25, 1.0])
SINGLE = torch.tensor([True, False, True, False])


def test_mean_is_the_released_formula_bit_for_bit():
    torch.testing.assert_close(combine_predictions(WT, MT), 0.5 * WT + 0.5 * MT, atol=0, rtol=0)
    torch.testing.assert_close(combine_predictions(WT, MT, 'mean', SINGLE), 0.5 * WT + 0.5 * MT, atol=0, rtol=0)


def test_sum_makes_mt_the_epistasis_term():
    torch.testing.assert_close(combine_predictions(WT, MT, 'sum'), WT + MT, atol=0, rtol=0)
    # the property that matters: fitting combined to a target leaves MT holding target - wt
    target = torch.tensor([-2.0, -1.0, 0.25, 4.0])
    implied_mt = target - WT
    torch.testing.assert_close(combine_predictions(WT, implied_mt, 'sum'), target, atol=1e-6, rtol=0)


def test_route_reports_wt_for_singles_and_mt_for_the_rest():
    got = combine_predictions(WT, MT, 'route', SINGLE)
    torch.testing.assert_close(got, torch.where(SINGLE, WT, MT), atol=0, rtol=0)
    assert got[0] == WT[0] and got[1] == MT[1]


def test_route_without_is_single_is_an_error_not_a_silent_mean():
    with pytest.raises(AssertionError, match='is_single'):
        combine_predictions(WT, MT, 'route')


def test_an_unknown_mode_is_rejected():
    with pytest.raises(AssertionError, match='combined_mode'):
        combine_predictions(WT, MT, 'average')


def test_the_modes_are_the_documented_three():
    assert COMBINE_MODES == ('mean', 'sum', 'route')


def test_gradients_flow_to_both_heads_in_every_mode():
    for mode in COMBINE_MODES:
        wt = WT.clone().requires_grad_(True)
        mt = MT.clone().requires_grad_(True)
        combine_predictions(wt, mt, mode, SINGLE).sum().backward()
        assert wt.grad is not None and mt.grad is not None, mode
        if mode == 'route':      # each head only gets gradient where it is reported
            assert bool((wt.grad[~SINGLE] == 0).all()) and bool((mt.grad[SINGLE] == 0).all())
        else:
            assert bool((wt.grad != 0).all()) and bool((mt.grad != 0).all()), mode


def test_the_defaults_leave_training_untouched():
    """A run that sets neither flag must behave exactly as before."""
    from esm_msr import config
    import sys
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        a = config.parse_arguments()
    finally:
        sys.argv = argv
    assert a.combined_mode == 'mean'
    assert a.wt_trains_doubles is True


def test_an_unset_hparam_arriving_as_none_means_the_default():
    """getattr on a stubbed hparams namespace yields None rather than raising, and None must mean 'mean'."""
    torch.testing.assert_close(combine_predictions(WT, MT, None), 0.5 * WT + 0.5 * MT, atol=0, rtol=0)
