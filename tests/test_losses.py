import pytest
import torch
from torch import nn

from esm_msr.losses import ListMLELoss, hinge_regression_loss


def _list(preds, gts, cens=None):
    p = torch.tensor([preds], dtype=torch.float32)
    g = torch.tensor([gts], dtype=torch.float32)
    m = torch.ones_like(p, dtype=torch.bool)
    c = torch.zeros_like(p, dtype=torch.long) if cens is None else torch.tensor([cens])
    return p, g, m, c


def test_censored_equals_plain_without_censoring():
    crit = ListMLELoss()
    p, g, m, c = _list([2.0, 1.0, 0.0, -1.0], [3.0, 2.0, 1.0, 0.0])
    assert float(crit.forward_censored(p, g, m, c)) == pytest.approx(float(crit(p, g, m)), abs=1e-6)


def test_order_within_a_lower_censored_block_is_not_scored():
    """Items 3 and 4 are both at the floor: swapping the model's opinion of which is worse must not change the loss."""
    crit = ListMLELoss()
    g, cens = [3.0, 2.0, -1.0, -1.0], [0, 0, -1, -1]
    a = crit.forward_censored(*_list([2.0, 1.0, 0.5, -0.5], g, cens))
    b = crit.forward_censored(*_list([2.0, 1.0, -0.5, 0.5], g, cens))
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_order_within_an_upper_censored_block_is_not_scored():
    crit = ListMLELoss()
    g, cens = [9.0, 9.0, 2.0, 1.0], [1, 1, 0, 0]
    a = crit.forward_censored(*_list([3.0, 2.5, 1.0, 0.0], g, cens))
    b = crit.forward_censored(*_list([2.5, 3.0, 1.0, 0.0], g, cens))
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_censored_items_are_still_pushed_below_the_measurable_ones():
    """The block is unscored internally but still has to sit at the bottom."""
    crit = ListMLELoss()
    g, cens = [3.0, 2.0, -1.0, -1.0], [0, 0, -1, -1]
    right = crit.forward_censored(*_list([2.0, 1.0, -3.0, -3.0], g, cens))
    wrong = crit.forward_censored(*_list([-3.0, -3.0, 2.0, 2.0], g, cens))
    assert float(wrong) > float(right)


def test_mixed_censoring_averages_both_directions():
    crit = ListMLELoss()
    g, cens = [9.0, 3.0, 2.0, -1.0], [1, 0, 0, -1]
    loss = crit.forward_censored(*_list([3.0, 1.0, 0.5, -2.0], g, cens))
    assert torch.isfinite(loss) and float(loss) > 0


def test_all_censored_list_is_dropped_but_keeps_gradients():
    crit = ListMLELoss()
    p, g, m, c = _list([1.0, 0.0], [-1.0, -1.0], [-1, -1])
    p.requires_grad_(True)
    loss = crit.forward_censored(p, g, m, c)
    assert float(loss) == 0.0
    loss.backward()
    assert p.grad is not None


def test_censored_rejects_invert():
    with pytest.raises(AssertionError):
        crit = ListMLELoss(invert=True)
        crit.forward_censored(*_list([1.0, 0.0], [1.0, 0.0]))


def test_masked_entries_ignored():
    crit = ListMLELoss()
    p = torch.tensor([[2.0, 1.0, 99.0]])
    g = torch.tensor([[3.0, 2.0, 99.0]])
    m = torch.tensor([[True, True, False]])
    c = torch.zeros_like(p, dtype=torch.long)
    small = crit.forward_censored(p[:, :2], g[:, :2], m[:, :2], c[:, :2])
    assert float(crit.forward_censored(p, g, m, c)) == pytest.approx(float(small), abs=1e-6)


# --------------------------------------------------------------------------- hinge regression

def test_hinge_is_free_below_the_bound_and_costs_above_it():
    crit = nn.MSELoss(reduction='none')
    cens = torch.tensor([-1, -1])
    bound = torch.tensor([-1.0, -1.0])
    target = torch.full((2,), float('nan'))          # a censored item has no measured value
    L = hinge_regression_loss(crit, torch.tensor([-4.0, 1.0]), target, bound, cens)
    assert float(L[0]) == 0.0                         # predicted at/below the floor: nothing to correct
    assert float(L[1]) == pytest.approx(4.0)          # predicted 2.0 above the bound


def test_hinge_upper_censored_is_mirrored():
    crit = nn.MSELoss(reduction='none')
    cens = torch.tensor([1, 1])
    bound = torch.tensor([5.0, 5.0])
    target = torch.full((2,), float('nan'))
    L = hinge_regression_loss(crit, torch.tensor([7.0, 3.0]), target, bound, cens)
    assert float(L[0]) == 0.0
    assert float(L[1]) == pytest.approx(4.0)


def test_hinge_leaves_uncensored_items_alone():
    crit = nn.MSELoss(reduction='none')
    pred, target = torch.tensor([1.0, -2.0]), torch.tensor([0.0, 0.0])
    L = hinge_regression_loss(crit, pred, target, torch.full((2,), float('nan')), torch.zeros(2, dtype=torch.long))
    assert torch.allclose(L, crit(pred, target))


def test_hinge_drops_censored_items_without_a_finite_bound():
    """A censored item whose library has no dG_wt has no bound on the ddG scale: it must contribute 0, not NaN."""
    crit = nn.MSELoss(reduction='none')
    L = hinge_regression_loss(crit, torch.tensor([2.0]), torch.tensor([float('nan')]),
                              torch.tensor([float('nan')]), torch.tensor([-1]))
    assert float(L[0]) == 0.0
    assert torch.isfinite(L).all()


def test_hinge_gradient_only_flows_from_violations():
    crit = nn.MSELoss(reduction='none')
    pred = torch.tensor([-4.0, 1.0], requires_grad=True)
    L = hinge_regression_loss(crit, pred, torch.full((2,), float('nan')),
                              torch.tensor([-1.0, -1.0]), torch.tensor([-1, -1]))
    L.sum().backward()
    assert float(pred.grad[0]) == 0.0
    assert float(pred.grad[1]) > 0
