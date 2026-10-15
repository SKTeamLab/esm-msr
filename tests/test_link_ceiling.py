"""The second soft clamp: a learned ceiling, its own temperature, and the anchor that keeps both near 0.59."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from esm_msr.link import (CEILING_DEFAULT, FLOOR_MAX_DEFAULT, TAU_DEFAULT, SaturatingLink, numpy_link,
                          numpy_link_from_state_dict)

NAN = torch.tensor(float('nan'))


def _two_sided(**kw):
    return SaturatingLink(ceiling=CEILING_DEFAULT, global_floor_init=-0.4, **kw)


def test_saturates_at_both_ends_and_stays_monotone():
    link = _two_sided()
    z = torch.linspace(-30.0, 30.0, 400)
    yf = torch.full_like(z, float('nan'))
    h = link(z, yf)
    assert bool((h.diff() >= 0).all()), "the link must be non-decreasing"
    assert float(h[0]) == pytest.approx(-0.4, abs=1e-3)              # the global floor
    assert float(h[-1]) == pytest.approx(CEILING_DEFAULT, abs=1e-3)  # the ceiling
    mid = (z > 1.0) & (z < 3.0)                                      # well inside the range: close to identity
    assert torch.allclose(h[mid], z[mid], atol=0.05)


def test_ceiling_cannot_cross_the_floor_band_however_it_is_pushed():
    """Parameterised as floor_max + softplus(raw), so even a hugely negative raw keeps it above the band."""
    link = _two_sided()
    with torch.no_grad():
        link.raw_ceiling_span.fill_(-50.0)
    assert float(link.ceiling()) > float(link.bounds()[1])
    z = torch.linspace(-10.0, 10.0, 200)
    assert bool((link(z, torch.full_like(z, float('nan'))).diff() >= 0).all())


def test_floor_only_is_unchanged_by_the_new_code():
    """ceiling=None must reproduce the floor-only link exactly, so dc3-dc5 semantics are untouched."""
    a = SaturatingLink(global_floor_init=-0.4)
    z = torch.linspace(-20.0, 20.0, 300)
    yf = torch.full_like(z, float('nan'))
    f = a.effective_floor(yf)
    expected = f + a.tau() * F.softplus((z - f) / a.tau())
    assert torch.allclose(a(z, yf), expected, atol=1e-6)
    assert 'ceiling' not in a.summary()


def test_tau_hi_defaults_to_tau_lo_and_is_separately_learnable():
    link = _two_sided(learn_tau=True)
    assert float(link.tau_hi()) == pytest.approx(TAU_DEFAULT, abs=1e-5)
    with torch.no_grad():
        link.raw_tau_hi.fill_(float(np.log(np.expm1(1.5))))
    assert float(link.tau_hi()) == pytest.approx(1.5, abs=1e-4)
    assert float(link.tau()) == pytest.approx(TAU_DEFAULT, abs=1e-5)   # unchanged


def test_tau_penalty_is_zero_at_the_anchor_and_grows_quadratically():
    link = _two_sided(learn_tau=True)
    assert float(link.tau_penalty(1.0)) == pytest.approx(0.0, abs=1e-8)
    with torch.no_grad():
        link.raw_tau.fill_(float(np.log(np.expm1(TAU_DEFAULT + 0.3))))
    assert float(link.tau_penalty(1.0)) == pytest.approx(0.09, abs=1e-4)
    assert float(link.tau_penalty(10.0)) == pytest.approx(0.9, abs=1e-3)


def test_tau_penalty_ignores_frozen_temperatures():
    """With --no-learn_link_tau there is nothing to anchor, so the term must not add a constant to the loss."""
    assert float(_two_sided(learn_tau=False).tau_penalty(1.0)) == 0.0


def test_tau_penalty_carries_gradient_to_both_temperatures():
    link = _two_sided(learn_tau=True)
    with torch.no_grad():
        link.raw_tau.fill_(1.0)
        link.raw_tau_hi.fill_(-1.0)
    link.tau_penalty(1.0).backward()
    assert link.raw_tau.grad is not None and abs(float(link.raw_tau.grad)) > 0
    assert link.raw_tau_hi.grad is not None and abs(float(link.raw_tau_hi.grad)) > 0


def test_numpy_link_matches_torch_with_a_ceiling():
    link = _two_sided(learn_tau=True)
    with torch.no_grad():
        link.raw_tau_hi.fill_(float(np.log(np.expm1(0.8))))
    h = link.numpy()
    z = np.linspace(-12.0, 12.0, 160)
    yf = np.full_like(z, np.nan)
    got = h(z, yf)
    want = link(torch.tensor(z, dtype=torch.float32), torch.tensor(yf, dtype=torch.float32)).detach().numpy()
    assert np.allclose(got, want, atol=2e-4)


def test_state_dict_roundtrip_recovers_the_ceiling():
    link = _two_sided(learn_tau=True)
    with torch.no_grad():
        link.raw_tau_hi.fill_(float(np.log(np.expm1(0.8))))
    h = numpy_link_from_state_dict(link.state_dict(), prefix='')
    assert h is not None
    z = np.linspace(-12.0, 12.0, 120)
    yf = np.full_like(z, np.nan)
    assert np.allclose(h(z, yf), link.numpy()(z, yf), atol=1e-9)
    assert h(np.array([1e6]), np.array([np.nan]))[0] == pytest.approx(float(link.ceiling()), abs=1e-3)


def test_a_library_floor_still_overrides_the_global_one_under_a_ceiling():
    link = _two_sided()
    z = torch.tensor([-20.0, -20.0])
    yf = torch.tensor([-0.9, float('nan')])
    h = link(z, yf)
    assert float(h[0]) == pytest.approx(-0.9, abs=1e-3)
    assert float(h[1]) == pytest.approx(-0.4, abs=1e-3)


def test_obs_ddG_saturates_upward_for_a_hyperstable_prediction():
    """A '>5' item's bound is reachable: the observed ddG cannot exceed ceiling - dG_wt however high the latent."""
    link = _two_sided()
    dG_wt = torch.tensor([2.5, 2.5])
    out = link.obs_ddG(torch.tensor([50.0, 500.0]), dG_wt, torch.tensor([float('nan')] * 2))
    assert torch.allclose(out, torch.full_like(out, CEILING_DEFAULT - 2.5), atol=1e-3)


def test_an_upper_bound_above_the_link_is_clamped_to_what_it_can_emit():
    """Mirror of the floor clamp. A '>5' row is bounded at 5 - dG_wt, but the link tops out at ceiling - dG_wt;
    once the LEARNED ceiling drops below 5 that bound is unreachable and the hinge would push the latent up
    without limit."""
    link = _two_sided()
    with torch.no_grad():                      # ceiling learns down to ~3.0
        link.raw_ceiling_span.fill_(float(np.log(np.expm1(3.0 - FLOOR_MAX_DEFAULT - 1e-3))))
    dG_wt = torch.tensor([2.5])
    offset = -dG_wt                            # a plain single: ddG = dG - dG_wt
    raw_bound = torch.tensor([5.0]) + offset   # 2.5, what the '>5' tag states
    ceil_ddG = link.ceiling().detach() + offset
    clamped = torch.minimum(raw_bound, ceil_ddG)
    assert float(clamped) == pytest.approx(float(link.ceiling()) - 2.5, abs=1e-4)
    assert float(clamped) < float(raw_bound)
    # and the clamped bound is now what the link actually approaches from below
    reached = link.obs_ddG(torch.tensor([500.0]), dG_wt, torch.tensor([float('nan')]))
    assert float(reached) == pytest.approx(float(clamped), abs=1e-3)


def test_the_clamp_leaves_a_reachable_bound_alone():
    link = _two_sided()
    dG_wt = torch.tensor([2.5])
    offset = -dG_wt
    raw_bound = torch.tensor([5.0]) + offset
    ceil_ddG = link.ceiling().detach() + offset
    assert float(torch.minimum(raw_bound, ceil_ddG)) == pytest.approx(float(raw_bound), abs=1e-3)
