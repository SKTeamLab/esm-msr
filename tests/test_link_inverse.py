"""
``SaturatingLink.inverse`` / ``latent_ddG``: putting a MEASURED label back on the latent scale.

Needed because teacher forcing injects ddG_A + ddG_B - a sum of two SATURATED readings - into a latent sum that
then goes through the link. 

The point that is easy to get wrong: the inverse is nonlinear, so de-linking the SUM is not de-linking the
singles. The label has to be inverted one single at a time and then added.
"""
import numpy as np
import pytest
import torch

from esm_msr.link import SaturatingLink

FLOOR = torch.tensor([-0.3, -0.3, -0.3, -0.3])
NAN = float('nan')


def link():
    return SaturatingLink(tau=0.59, floor=-0.3)


def test_inverse_round_trips_the_forward():
    L = link()
    z = torch.tensor([0.5, 1.0, 2.0, 5.0])
    y = L(z, FLOOR)
    torch.testing.assert_close(L.inverse(y, FLOOR), z, atol=1e-4, rtol=1e-4)


def test_obs_ddG_and_latent_ddG_are_exact_inverses():
    L = link()
    dG_wt = torch.tensor([2.0, 2.0, 3.5, 3.5])
    latent = torch.tensor([-0.5, -1.5, -0.2, -2.0])
    obs = L.obs_ddG(latent, dG_wt, FLOOR)
    back = L.latent_ddG(obs, dG_wt, FLOOR)
    torch.testing.assert_close(back, latent, atol=1e-4, rtol=1e-4)


def test_a_reading_at_or_below_the_floor_is_not_invertible():
    """The floor is an asymptote: such a reading is consistent with any low latent value, so NaN, not a clamp."""
    L = link()
    f = L.effective_floor(FLOOR)
    y = torch.stack([f[0] - 1.0, f[0], f[0] + 1e-6, f[0] + 1.0])
    out = L.inverse(y, FLOOR)
    assert torch.isnan(out[0]) and torch.isnan(out[1]), out
    assert torch.isfinite(out[3]), out


def test_far_above_the_floor_the_link_is_the_identity():
    L = link()
    z = torch.tensor([20.0, 40.0, 60.0, 80.0])
    torch.testing.assert_close(L.inverse(z, FLOOR), z, atol=1e-3, rtol=1e-5)


def test_no_overflow_or_nan_in_the_large_v_branch():
    L = link()
    y = torch.tensor([50.0, 100.0, 200.0, 400.0])
    out = L.inverse(y, FLOOR)
    assert torch.isfinite(out).all(), out


def test_de_linking_the_sum_is_NOT_de_linking_the_singles():
    """The reason the implementation inverts ddG_A and ddG_B separately."""
    L = link()
    dG_wt = torch.tensor([2.0, 2.0, 2.0, 2.0])
    ddG_A = torch.tensor([-0.8, -1.2, -0.4, -1.6])
    ddG_B = torch.tensor([-0.7, -0.9, -0.5, -0.3])
    per_single = L.latent_ddG(ddG_A, dG_wt, FLOOR) + L.latent_ddG(ddG_B, dG_wt, FLOOR)
    as_a_sum = L.latent_ddG(ddG_A + ddG_B, dG_wt, FLOOR)
    gap = (per_single - as_a_sum).abs()
    assert (gap > 1e-3).any(), "the two agree, so this test is not exercising the nonlinearity"
    assert gap.max() > 0.05, f"expected a materially different answer, got {gap.max():.4f}"


def test_the_gradient_flows_through_the_inverse():
    L = link()
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    L.inverse(y, FLOOR).sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
