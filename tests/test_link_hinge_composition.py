"""How the floor link and the hinge compose, which is not obvious and is easy to get backwards.

With the link ON, a prediction on the observed scale is h(dG_wt + latent) - dG_wt, and h is bounded below by the
library's floor. A censored item's bound is that same floor expressed on the item's own ddG scale. So the prediction
can never fall strictly below the bound, the hinge's "wrong side" test is always true, and the hinge mask does
nothing - but the squared error against the bound decays to zero as the latent drops, because h flattens onto the
floor. The link therefore implements item 5's requirement smoothly on its own, and the explicit hinge is what
delivers it when the link is OFF.
"""
import numpy as np
import pytest
import torch
from torch import nn

from esm_msr.link import SaturatingLink
from esm_msr.losses import hinge_regression_loss

CRIT = nn.MSELoss(reduction='none')


def _censored_loss(latent, dG_wt=3.0, y_floor=-0.3, link=True):
    lat = torch.tensor([float(latent)])
    wt = torch.tensor([float(dG_wt)])
    fl = torch.tensor([float(y_floor)])
    bound = fl - wt                                   # the floor on this item's ddG scale
    pred = SaturatingLink()(wt + lat, fl) - wt if link else lat
    return float(hinge_regression_loss(CRIT, pred, torch.tensor([float('nan')]), bound,
                                       torch.tensor([-1]))[0])


def test_with_the_link_a_deeper_prediction_always_costs_less():
    """Monotone decreasing in how far below the floor the model puts the variant, and vanishing in the limit."""
    losses = [_censored_loss(z) for z in (-1.0, -3.0, -5.0, -8.0, -12.0)]
    assert all(a > b for a, b in zip(losses, losses[1:])), losses
    assert losses[-1] == pytest.approx(0.0, abs=1e-6)


def test_with_the_link_predicting_above_the_floor_is_what_costs():
    """A variant measured at the floor, predicted as comfortably stable, pays the full squared error."""
    assert _censored_loss(+2.0) > 1.0
    assert _censored_loss(+2.0) > _censored_loss(-2.0)


def test_without_the_link_the_hinge_is_what_zeroes_it():
    """On the latent scale the prediction CAN sit below the bound, and then the hinge must make it free."""
    bound = -0.3 - 3.0                                  # y_floor - dG_wt = -3.3
    assert _censored_loss(-5.0, link=False) == 0.0      # latent -5.0 is below the bound: free
    assert _censored_loss(-2.0, link=False) > 0.0       # latent -2.0 is above it: penalised
    assert bound < -2.0


def test_an_uncensored_item_is_untouched_by_either_mechanism():
    lat, wt, fl = torch.tensor([-1.0]), torch.tensor([3.0]), torch.tensor([-0.3])
    target = torch.tensor([-1.0])
    pred = SaturatingLink()(wt + lat, fl) - wt
    got = hinge_regression_loss(CRIT, pred, target, fl - wt, torch.tensor([0]))
    assert torch.allclose(got, CRIT(pred, target))
    # Well inside the dynamic range the link is near-identity, so this is ordinary regression on ddG.
    assert float(pred) == pytest.approx(-1.0, abs=0.02)


def test_gradient_pushes_a_censored_item_down_not_up():
    lat = torch.tensor([-1.0], requires_grad=True)
    wt, fl = torch.tensor([3.0]), torch.tensor([-0.3])
    pred = SaturatingLink()(wt + lat, fl) - wt
    hinge_regression_loss(CRIT, pred, torch.tensor([float('nan')]), fl - wt, torch.tensor([-1])).sum().backward()
    assert float(lat.grad) > 0        # descending -grad lowers the latent, i.e. pushes it toward the floor


def test_doubles_latent_target_is_not_treated_as_censored():
    """A double's WT-path target is the sum of its singles, a latent quantity with no bound; training forces
    cens=0 there, and this is what that guarantees numerically."""
    pred, target = torch.tensor([-6.0]), torch.tensor([-6.2])
    got = hinge_regression_loss(CRIT, pred, target, torch.tensor([-3.3]), torch.tensor([0]))
    assert float(got) == pytest.approx(0.04, abs=1e-5)
