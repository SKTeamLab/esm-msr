import math

import numpy as np
import pytest
import torch

from esm_msr.link import FLOOR_BAND, TAU_DEFAULT, SaturatingLink, numpy_link, numpy_link_from_state_dict


def test_knee_value_and_asymptotes():
    """h(floor) = floor + tau*ln2; h -> floor below, h -> z above, and never below the floor."""
    link = SaturatingLink(floor=-0.3)
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
    link = SaturatingLink(floor=-0.3)
    z = torch.linspace(-20, 20, 4001)
    out = link(z, torch.full_like(z, -0.3))
    assert (out.diff() >= 0).all()
    near = torch.linspace(-2.0, 20.0, 4001)
    assert (link(near, torch.full_like(near, -0.3)).diff() > 0).all()


def test_the_knee_is_the_one_floor_whatever_an_item_carries():
    """An item's own y_floor is clipped into [floor, floor + FLOOR_BAND]: a run has one knee. (A dump from the era of per-library
    floors is rescored with numpy_link instead, which still takes them.)"""
    link = SaturatingLink(floor=0.0)
    for carried in (-7.0, -0.3, 0.0, 0.5, 9.0):
        out = float(link(torch.tensor(-60.0), torch.tensor(carried)))
        assert 0.0 - 1e-6 <= out <= FLOOR_BAND + 1e-6, (carried, out)


def test_nan_floor_falls_back_to_the_one_floor():
    link = SaturatingLink(floor=-0.4)
    out = link(torch.tensor(-60.0), torch.tensor(float('nan')))
    assert float(out) == pytest.approx(-0.4, abs=1e-5)


def test_nothing_in_the_link_is_trainable():
    link = SaturatingLink()
    assert list(link.parameters()) == []
    # ... but its numbers are in the state dict under the names old checkpoints used, so link_head.* keys load and rescore
    assert set(link.state_dict()) == {'raw_tau', 'floor_min', 'floor_max', 'global_floor'}


def test_obs_ddG_matches_definition():
    link = SaturatingLink(floor=-0.2)
    latent, dG_wt, y = torch.tensor([-6.0, 0.5]), torch.tensor([3.0, 3.0]), torch.tensor([-0.2, -0.2])
    got = link.obs_ddG(latent, dG_wt, y)
    want = link(dG_wt + latent, y) - dG_wt
    assert torch.allclose(got, want)
    # A deeply destabilised variant is reported near the floor, not at its latent value.
    assert float(got[0] + dG_wt[0]) == pytest.approx(-0.2, abs=0.02)


def test_nan_dG_wt_propagates():
    link = SaturatingLink(floor=-0.2)
    out = link.obs_ddG(torch.tensor([0.5]), torch.tensor([float('nan')]), torch.tensor([-0.2]))
    assert not torch.isfinite(out).any()


def test_numpy_link_matches_torch():
    link = SaturatingLink(floor=-0.25)
    h = link.numpy()
    z = np.linspace(-10, 10, 101)
    y = np.full_like(z, -0.25)
    got = h(z, y)
    want = link(torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)).detach().numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_numpy_link_from_state_dict_roundtrip():
    link = SaturatingLink(tau=0.4, floor=-0.8)
    h = numpy_link_from_state_dict({f'link_head.{k}': v for k, v in link.state_dict().items()})
    assert h is not None
    assert h(-50.0, -0.8) == pytest.approx(-0.8, abs=1e-5)
    assert h(0.0, -0.8) == pytest.approx(float(link(torch.tensor(0.0), torch.tensor(-0.8))), abs=1e-5)
    assert numpy_link_from_state_dict({'other.weight': torch.zeros(1)}) is None


def test_numpy_link_still_reads_an_old_checkpoint_with_a_ceiling_and_a_band():
    """Rescoring a run from the era of per-library floors and a learned ceiling: the keys that existed then are read as before."""
    sd = {'link_head.raw_tau': torch.tensor(0.0), 'link_head.floor_min': torch.tensor(-1.0), 'link_head.floor_max': torch.tensor(0.2),
          'link_head.global_floor': torch.tensor(-0.1), 'link_head.raw_ceiling_span': torch.tensor(1.0)}
    h = numpy_link_from_state_dict(sd)
    assert h is not None
    assert h(-60.0, -7.0) == pytest.approx(-1.0, abs=0.05)          # a per-library floor clipped into the old band (the ceiling bends it a little)
    assert h(60.0, 0.0) < 60.0                                      # and the upper clamp is there


def test_tau_is_what_was_asked():
    assert float(SaturatingLink().tau()) == pytest.approx(TAU_DEFAULT, abs=1e-4)
    assert float(SaturatingLink(tau=0.3).tau()) == pytest.approx(0.3, abs=1e-5)


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        SaturatingLink(tau=0.0)
