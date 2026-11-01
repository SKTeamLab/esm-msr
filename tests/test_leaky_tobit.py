"""``LeakyFloorLink`` (knee and asymptote are different numbers), ``tobit_regression_loss`` (noisy-floor mixture likelihood), and the flags that
select them (``--link_kind leaky``, ``--reg_nll tobit``, ``--tf_cut``).

The point of each: the audit of 2026-11-01 found that (a) one number ``f`` cannot be both the knee and the lowest reportable dG, and (b) a
censor threshold at or below the floor feeds readings the inverse link cannot invert. These pin the properties the replacements were chosen for.
"""
import math
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import config
from esm_msr.link import LeakyFloorLink, SaturatingLink, numpy_link_from_state_dict
from esm_msr.losses import tobit_regression_loss as tobit
from esm_msr.training import ESM3EpistasisLightningModule as M, build_link

NAN = float('nan')
BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ------------------------------------------------------------------------------------------------ the link
def test_leaky_inverse_round_trips_everywhere_above_the_clamp():
    L = LeakyFloorLink()
    z = torch.tensor([-12.0, -6.0, -3.0, -1.0, 0.0, 0.3, 1.0, 4.0, 30.0], dtype=torch.float64)
    torch.testing.assert_close(L.inverse(L(z)), z, atol=1e-6, rtol=1e-6)


def test_leaky_has_no_nan_for_any_numeric_reading_but_does_at_the_clamp():
    """Every numeric MegaScale reading is > -1: the point of putting the asymptote there while the knee stays at 0."""
    L = LeakyFloorLink()
    y = torch.tensor([-0.999, -0.5, -0.1, 0.0, 0.05, 0.3, 2.0])
    assert torch.isfinite(L.inverse(y)).all()
    out = L.inverse(torch.tensor([-1.0, -1.5]))
    assert torch.isnan(out).all()


def test_leaky_knee_stays_at_the_knee_not_the_clamp():
    """A dead double (z = -2.5) is reported near the knee, not near the asymptote; SP(floor=-1) is what got this wrong."""
    leaky, sp_m1 = LeakyFloorLink(tau=0.65, alpha=0.10), SaturatingLink(tau=1.0, floor=-1.0)
    z = torch.tensor([-2.5])
    assert float(leaky(z)) > -0.45
    assert float(sp_m1(z, torch.tensor([-1.0]))) < float(leaky(z)) - 0.3 or float(sp_m1(z, torch.tensor([-1.0]))) < -0.7


def test_leaky_inverse_slope_is_bounded_where_the_softplus_inverse_diverges():
    """The inverse's slope is what amplifies a reading's noise into the latent. Measured at alpha = 0.10 (values rounded up)."""
    L = LeakyFloorLink(alpha=0.10)
    for lo, bound in ((0.0, 4.5), (-0.5, 12.0), (-0.9, 31.0)):
        y = torch.linspace(lo, 0.5, 1401, dtype=torch.float64)
        assert float((torch.diff(L.inverse(y)) / torch.diff(y)).max()) <= bound, lo
    sp = SaturatingLink(tau=0.59, floor=0.0)
    y2 = torch.linspace(0.005, 0.5, 1000)
    assert float((torch.diff(sp.inverse(y2, torch.zeros(1000))) / torch.diff(y2)).max()) > 100.0


def test_leaky_is_identity_far_above_and_monotone():
    L = LeakyFloorLink()
    z = torch.linspace(-8, 40, 4001, dtype=torch.float64)
    h = L(z)
    assert bool((torch.diff(h) > 0).all())
    torch.testing.assert_close(L(torch.tensor([30.0, 60.0])), torch.tensor([30.0, 60.0]), atol=1e-6, rtol=0)


def test_leaky_obs_and_latent_ddG_are_exact_inverses():
    L = LeakyFloorLink()
    dG_wt = torch.tensor([2.0, 2.0, 3.5, 3.5], dtype=torch.float64)
    lat = torch.tensor([-0.5, -2.5, -0.2, -6.0], dtype=torch.float64)
    torch.testing.assert_close(L.latent_ddG(L.obs_ddG(lat, dG_wt), dG_wt), lat, atol=1e-6, rtol=1e-6)


def test_leaky_numpy_state_dict_and_summary_agree_with_the_module():
    L = LeakyFloorLink(tau=0.7, knee=0.1, alpha=0.2, clamp=-0.9, clamp_width=0.3)
    z = np.linspace(-6, 6, 25)
    expect = L(torch.tensor(z)).numpy()
    np.testing.assert_allclose(L.numpy()(z), expect, atol=1e-6)
    h = numpy_link_from_state_dict({'link_head.' + k: v for k, v in L.state_dict().items()})
    np.testing.assert_allclose(h(z), expect, atol=1e-6)
    s = L.summary()
    assert s['global_floor'] == pytest.approx(0.1) and s['leaky_alpha'] == pytest.approx(0.2) and s['clamp'] == pytest.approx(-0.9)


def test_softplus_state_dict_still_loads_as_softplus():
    sp = SaturatingLink(tau=0.59, floor=0.0)
    h = numpy_link_from_state_dict({'link_head.' + k: v for k, v in sp.state_dict().items()})
    z = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(h(z), sp(torch.tensor(z), torch.zeros(13).double()).numpy(), atol=1e-6)


def test_build_link_selects_the_kind():
    hp = types.SimpleNamespace(link=True, link_tau=0.65, link_floor=0.0, link_kind='leaky', link_alpha=0.1, link_clamp=-1.0)
    hp.get = lambda k, d=None: getattr(hp, k, d)
    assert isinstance(build_link(hp), LeakyFloorLink)
    hp.link_kind = 'softplus'
    assert isinstance(build_link(hp), SaturatingLink)


# ------------------------------------------------------------------------------------------------ the likelihood
def _t(z, y, b=NAN, c=0):
    return float(tobit(torch.tensor([float(z)]), torch.tensor([float(y)]), torch.tensor([float(b)]), torch.tensor([c])))


def test_tobit_is_mse_far_above_the_floor():
    for z, y in ((2.0, 3.0), (4.0, 3.1), (3.0, 3.0)):
        assert _t(z, y) == pytest.approx((y - z) ** 2, abs=1e-4)


def test_tobit_a_dead_double_read_at_the_floor_costs_almost_nothing_and_a_false_stable_prediction_a_lot():
    dead = _t(-3.0, 0.05)
    assert dead < 0.1
    assert _t(1.5, 0.05) > 10 * dead + 1.0


def test_tobit_single_item_minimiser_is_the_noise_aware_latent_not_the_inverse_link():
    """For a reading of +0.2 the minimiser is ~+0.08: the deterministic inverse of SP(0, 0.59) would claim about -0.5."""
    zs = torch.linspace(-4, 3, 7001)
    for y, z_star in ((0.2, 0.08), (0.05, -0.12), (1.0, 1.0), (1.5, 1.5)):
        L = tobit(zs, torch.full_like(zs, y), torch.full_like(zs, NAN), torch.zeros_like(zs, dtype=torch.long))
        assert float(zs[int(L.argmin())]) == pytest.approx(z_star, abs=0.07), y


def test_tobit_range_tag_is_a_one_sided_survival_term():
    z = torch.tensor([-4.0, -1.0, 0.0, 1.0, 3.0])
    lo = tobit(z, torch.full_like(z, NAN), torch.full_like(z, -1.0), -torch.ones(5, dtype=torch.long))
    assert float(lo[0]) < 1e-6 and float(lo[1]) == pytest.approx(2 * 0.28 ** 2 * math.log(2), abs=1e-5)
    assert bool((torch.diff(lo) > 0).all())                     # claiming it is MORE stable than the tag allows costs more and more
    hi = tobit(z, torch.full_like(z, NAN), torch.full_like(z, 5.0), torch.ones(5, dtype=torch.long))
    assert bool((torch.diff(hi) < 0).all())


def test_tobit_censored_item_with_no_bound_contributes_nothing_and_grads_are_finite():
    z = torch.tensor([0.5, -0.5, 2.0, -3.0], requires_grad=True)
    y = torch.tensor([NAN, NAN, 1.0, 0.1])
    b = torch.tensor([NAN, NAN, NAN, NAN])
    c = torch.tensor([-1, 1, 0, 0])
    L = tobit(z, y, b, c)
    assert float(L[0]) == 0.0 and float(L[1]) == 0.0
    L.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_tobit_downward_gradient_vanishes_for_a_confidently_dead_prediction():
    """KNOWN LIMIT, pinned: a variant read at +1.0 whose latent is predicted far below the floor gets (almost) no restoring gradient, as with the
    softplus link. It is why the first runs watch the singles' correlation rather than only the loss."""
    z = torch.tensor([-3.0, 0.0], requires_grad=True)
    tobit(z, torch.tensor([1.0, 1.0]), torch.full((2,), NAN), torch.zeros(2, dtype=torch.long)).sum().backward()
    assert abs(float(z.grad[0])) < 0.05 * abs(float(z.grad[1]))


# ------------------------------------------------------------------------------------------------ the flags
def test_defaults_leave_the_released_behaviour_untouched():
    a = parse()
    assert a.link_kind == 'softplus' and a.reg_nll == 'none' and a.tf_cut is None


def test_flags_parse_and_validate():
    a = parse('--link_kind', 'leaky', '--link_tau', '0.65', '--reg_nll', 'tobit', '--tf_cut', '0.5')
    assert a.link_kind == 'leaky' and a.reg_nll == 'tobit' and a.tf_cut == 0.5 and a.link_alpha == 0.10 and a.link_clamp == -1.0
    assert parse('--tf_cut', 'none').tf_cut is None
    for bad in (('--link_kind', 'leaky', '--link_clamp', '0.5'), ('--link_kind', 'leaky', '--link_alpha', '1.0'),
                ('--reg_nll', 'tobit', '--no-link'),):
        with pytest.raises(SystemExit):
            parse(*bad)


# ------------------------------------------------------------------------------------------------ the composition (stub self, CPU)
from test_tf_mix import _stub, _batch, _combined_loss        # noqa: E402  (the same hand-checkable stub the --tf_mix tests use)
from esm_msr.losses import TOBIT_SIGMA_Y, TOBIT_MU_F, TOBIT_SIGMA_F        # noqa: E402


def _tobit_stub():
    s = _stub()
    s.tobit = True
    s.tobit_params = dict(sigma_y=TOBIT_SIGMA_Y, mu_f=TOBIT_MU_F, sigma_f=TOBIT_SIGMA_F)
    return s


def test_tobit_composition_scores_the_combined_latent_on_absolute_dG():
    """Stub: dG_wt 2.0, ddG label -1.0 (y = 1.0), teacher-forced WT half +1.0, MT 0.0 -> combined latent +0.5 -> z = 2.5."""
    got = _combined_loss(_tobit_stub())
    want = float(tobit(torch.tensor([2.5]), torch.tensor([1.0]), torch.tensor([NAN]), torch.tensor([0]))[0])
    assert got == pytest.approx(want, abs=1e-5)


def test_tobit_ignores_the_floor_censor_flag_but_not_the_assay_tag():
    """A floor-censored item (cens_src = SRC_FLOOR) has a number the mixture uses; only SRC_RANGE items are bounds."""
    from esm_msr import censoring
    b = _batch()
    base = _combined_loss(_tobit_stub(), b)
    b2 = _batch()
    b2['cens'] = -torch.ones(6, dtype=torch.long)
    b2['cens_src'] = torch.full((6,), censoring.SRC_FLOOR, dtype=torch.long)
    b2['cens_bound'] = torch.full((6,), 0.4)
    assert _combined_loss(_tobit_stub(), b2) == pytest.approx(base, abs=1e-6)
    b3 = _batch()
    b3['cens'] = -torch.ones(6, dtype=torch.long)
    b3['cens_src'] = torch.full((6,), censoring.SRC_RANGE, dtype=torch.long)
    b3['cens_bound'] = torch.full((6,), -2.0)          # on the item's ddG scale: absolute bound = dG_wt + (-2.0) = 0.0
    want = float(tobit(torch.tensor([2.5]), torch.tensor([NAN]), torch.tensor([0.0]), torch.tensor([-1]))[0])
    assert _combined_loss(_tobit_stub(), b3) == pytest.approx(want, abs=1e-5)


def test_tf_cut_chooses_the_teacher_forcing_cut_on_its_own():
    """Singles read at dG_wt + 0.5 = 2.5: above a cut of 1.0 the label (+1.0 on the ddG scale) is used; at or below 2.6 it is not and the WT head's +2.0 is."""
    def with_cut(cut):
        s = _stub()
        s.hparams['tf_cut'] = cut
        return _combined_loss(s)
    assert with_cut(1.0) == pytest.approx((0.5 * 1.0 + 1.0) ** 2, abs=1e-6)
    assert with_cut(2.6) == pytest.approx((0.5 * 2.0 + 1.0) ** 2, abs=1e-6)
    assert with_cut(None) == pytest.approx(_combined_loss(_stub()), abs=1e-7)       # unset: the old coupling, unchanged
