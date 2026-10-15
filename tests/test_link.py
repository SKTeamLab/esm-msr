import math

import numpy as np
import pytest
import torch

from esm_msr.link import FLOOR_MAX_DEFAULT, FLOOR_MIN_DEFAULT, TAU_DEFAULT, SaturatingLink, numpy_link, numpy_link_from_state_dict


def test_knee_value_and_asymptotes():
    """h(y_floor) = y_floor + tau*ln2; h -> y_floor below, h -> z above, and never below the floor."""
    link = SaturatingLink()
    f = torch.tensor(-0.3)
    assert float(link(f, f)) == pytest.approx(float(f) + TAU_DEFAULT * math.log(2), abs=1e-5)
    assert float(link(torch.tensor(-50.0), f)) == pytest.approx(float(f), abs=1e-6)
    assert float(link(torch.tensor(20.0), f)) == pytest.approx(20.0, abs=1e-6)
    # Nothing is ever reported below the floor. (Far below it the float32 result saturates to exactly the floor;
    # mathematically h > floor strictly, which holds wherever the difference is representable.)
    z = torch.linspace(-30, 30, 2001)
    assert (link(z, f.expand_as(z)) >= f).all()
    near = torch.linspace(-5.0, 30.0, 2001)
    assert (link(near, f.expand_as(near)) > f).all()


def test_monotone():
    """Non-decreasing everywhere (it goes exactly flat far below the floor, where float32 saturates), and strictly
    increasing from the knee upwards, which is the range ranking and regression actually live in."""
    link = SaturatingLink()
    z = torch.linspace(-20, 20, 4001)
    out = link(z, torch.full_like(z, -0.3))
    assert (out.diff() >= 0).all()
    near = torch.linspace(-2.0, 20.0, 4001)
    assert (link(near, torch.full_like(near, -0.3)).diff() > 0).all()


def test_floor_is_clipped_to_band():
    """A wild cached estimate is pulled into [floor_min, floor_max]; the floor of the output follows the clipped value."""
    link = SaturatingLink()
    far_below = link(torch.tensor(-60.0), torch.tensor(-7.0))
    far_above = link(torch.tensor(-60.0), torch.tensor(+9.0))
    assert float(far_below) == pytest.approx(FLOOR_MIN_DEFAULT, abs=1e-5)
    assert float(far_above) == pytest.approx(FLOOR_MAX_DEFAULT, abs=1e-5)


def test_nan_floor_falls_back_to_the_learned_global_floor():
    """A library with no floor of its own bends at the shared, learned knee."""
    link = SaturatingLink(global_floor_init=-0.4)
    out = link(torch.tensor(-60.0), torch.tensor(float('nan')))
    assert float(out.detach()) == pytest.approx(-0.4, abs=1e-5)
    # ... and it is a real parameter that receives gradient from exactly those items.
    out.backward()
    assert link.global_floor.grad is not None and abs(float(link.global_floor.grad)) > 0


def test_a_library_with_its_own_floor_does_not_train_the_global_one():
    link = SaturatingLink(global_floor_init=-0.4)
    link(torch.tensor(-60.0), torch.tensor(-0.2)).backward()
    assert link.global_floor.grad is None or float(link.global_floor.grad) == 0.0


def test_global_floor_is_clipped_to_the_band_too():
    link = SaturatingLink(global_floor_init=-9.0)
    assert float(link(torch.tensor(-60.0), torch.tensor(float('nan'))).detach()) == pytest.approx(FLOOR_MIN_DEFAULT, abs=1e-5)


def test_obs_ddG_matches_definition():
    link = SaturatingLink()
    latent, dG_wt, y = torch.tensor([-6.0, 0.5]), torch.tensor([3.0, 3.0]), torch.tensor([-0.2, -0.2])
    got = link.obs_ddG(latent, dG_wt, y)
    want = link(dG_wt + latent, y) - dG_wt
    assert torch.allclose(got, want)
    # A deeply destabilised variant is reported near the floor, not at its latent value.
    assert float(got[0] + dG_wt[0]) == pytest.approx(-0.2, abs=0.02)


def test_obs_ddG_bg_offset_saturates_as_the_parent_double():
    """A conditional item's background offset moves it down the same curve."""
    link = SaturatingLink()
    latent, dG_wt, y = torch.tensor([-1.0]), torch.tensor([2.0]), torch.tensor([-0.1])
    plain = link.obs_ddG(latent, dG_wt, y)
    with_bg = link.obs_ddG(latent, dG_wt, y, bg_offset=torch.tensor([-3.0]))
    assert float(with_bg) < float(plain)


def test_nan_dG_wt_propagates():
    link = SaturatingLink()
    out = link.obs_ddG(torch.tensor([0.5]), torch.tensor([float('nan')]), torch.tensor([-0.2]))
    assert not torch.isfinite(out).any()


def test_numpy_link_matches_torch():
    link = SaturatingLink()
    h = link.numpy()
    z = np.linspace(-10, 10, 101)
    y = np.full_like(z, -0.25)
    got = h(z, y)
    want = link(torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)).detach().numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_numpy_link_from_state_dict_roundtrip():
    link = SaturatingLink(tau=0.4, floor_min=-0.8, floor_max=0.3)
    h = numpy_link_from_state_dict({f'link_head.{k}': v for k, v in link.state_dict().items()})
    assert h is not None
    assert h(-50.0, -7.0) == pytest.approx(-0.8, abs=1e-5)
    assert h(0.0, 0.0) == pytest.approx(float(link(torch.tensor(0.0), torch.tensor(0.0))), abs=1e-5)
    assert numpy_link_from_state_dict({'other.weight': torch.zeros(1)}) is None


def test_tau_is_fixed_by_default_and_learnable_on_request():
    assert SaturatingLink().raw_tau.requires_grad is False
    assert SaturatingLink(learn_tau=True).raw_tau.requires_grad is True
    assert float(SaturatingLink().tau()) == pytest.approx(TAU_DEFAULT, abs=1e-4)


def test_floor_bounds_receive_gradient_when_learnable():
    link = SaturatingLink(learn_floor_bounds=True)
    # A cached floor below the band is clipped at floor_min, so the gradient reaches floor_min.
    link(torch.tensor(-60.0), torch.tensor(-5.0)).backward()
    assert link.floor_min.grad is not None and abs(float(link.floor_min.grad)) > 0
    assert SaturatingLink().floor_min.requires_grad is False


def test_band_cannot_invert():
    link = SaturatingLink(floor_min=-1.0, floor_max=0.2, learn_floor_bounds=True)
    with torch.no_grad():
        link.floor_max.fill_(-3.0)
    lo, hi = link.bounds()
    assert float(hi.detach()) > float(lo.detach())


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        SaturatingLink(floor_min=0.5, floor_max=-0.5)
    with pytest.raises(ValueError):
        SaturatingLink(tau=0.0)
