"""``--tf_label`` / ``--tf_no_label``: what the WT half of a block's combined prediction is built from.

The posterior is checked against brute-force integration of prior x likelihood, so the closed form cannot drift from the Tobit loss it claims to invert.
"""
import sys

import pytest
import torch

from esm_msr import censoring, config
from esm_msr.link import LeakyFloorLink, SaturatingLink
from esm_msr.losses import TF_PRIOR_SD, TOBIT_SIGMA_Y, tobit_posterior_mean, tobit_regression_loss
from esm_msr.training import ESM3EpistasisLightningModule as M, TF_CLAMP_Y0, tf_single_label
from test_blocks import _blk, _module


def _brute(y, m, cens=0, s=TF_PRIOR_SD):
    """The posterior mean by quadrature: the prior times exp(-loss / (2 sigma_y^2)), which is the likelihood up to a constant."""
    z = torch.linspace(m - 12 * s, m + 12 * s, 48001)
    ones = torch.ones_like(z)
    loss = tobit_regression_loss(z, ones * y, ones * y, torch.full_like(z, cens, dtype=torch.long)).double()
    post = torch.exp(-0.5 * ((z.double() - m) / s) ** 2 - loss / (2 * TOBIT_SIGMA_Y ** 2))
    return float((z.double() * post).sum() / post.sum())


@pytest.mark.parametrize('y,m', [(3.0, 2.5), (2.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.5, 0.0), (0.25, 1.0), (0.05, 3.0), (0.0, -3.0), (-0.5, 1.0), (-0.9, -2.0), (4.0, 0.0)])
def test_the_closed_form_is_prior_times_the_tobit_likelihood(y, m):
    got = tobit_posterior_mean(torch.tensor([y]), torch.tensor([m]), torch.tensor([0]))
    assert float(got) == pytest.approx(_brute(y, m), abs=3e-3)


@pytest.mark.parametrize('b,m,cens', [(-1.0, 1.0, -1), (-1.0, -3.0, -1), (-1.0, 0.0, -1), (5.0, 4.0, 1), (5.0, 6.0, 1)])
def test_a_range_tag_is_the_truncated_mean_of_the_survival_term(b, m, cens):
    got = tobit_posterior_mean(torch.tensor([b]), torch.tensor([m]), torch.tensor([cens]))
    assert float(got) == pytest.approx(_brute(b, m, cens=cens), abs=3e-3)


def test_it_is_continuous_in_the_reading_with_no_cut():
    y = torch.linspace(-1.0, 5.0, 6001)
    for m in (-3.0, -1.0, 0.0, 1.0, 3.0):
        z = tobit_posterior_mean(y, torch.full_like(y, m), torch.zeros_like(y, dtype=torch.long))
        assert torch.isfinite(z).all()
        assert float(z.diff().abs().max()) < 0.02, m                                # no jump anywhere: the inverse link has one at its asymptote


def test_the_limits_above_the_floor_at_the_floor_and_with_no_reading():
    t = lambda v: torch.tensor([float(v)])
    none = torch.tensor([0])
    w = TF_PRIOR_SD ** 2 / (TF_PRIOR_SD ** 2 + TOBIT_SIGMA_Y ** 2)
    assert float(tobit_posterior_mean(t(3.0), t(2.0), none)) == pytest.approx(w * 3.0 + (1 - w) * 2.0, abs=1e-3)      # well above the floor: the precision-weighted mean
    assert float(tobit_posterior_mean(t(0.0), t(-3.0), none)) == pytest.approx(-3.0, abs=0.02)                         # a deep prior stays deep at the floor
    assert float(tobit_posterior_mean(t(0.0), t(3.0), none)) < 0.5                                                     # an optimistic prior is cut down to the floor
    assert float(tobit_posterior_mean(t(float('nan')), t(1.2), none)) == pytest.approx(1.2)                            # no reading: the prior


def test_it_stays_finite_at_the_extremes_in_single_precision():
    y = torch.tensor([-1.0, 5.0, -1.0, 5.0, 40.0, -40.0])
    m = torch.tensor([60.0, -60.0, -60.0, 60.0, -60.0, 60.0])
    for c in (0, -1, 1):
        assert torch.isfinite(tobit_posterior_mean(y, m, torch.full_like(y, c, dtype=torch.long))).all()


def test_the_single_label_only_treats_the_assays_own_tags_as_bounds():
    dG, wt, fl = torch.tensor(3.0), torch.tensor(-1.0), torch.tensor(0.0)           # the WT head reads this single at absolute dG 2
    y = torch.tensor(-3.5)                                                          # read at absolute -0.5
    link = SaturatingLink()
    tag = tf_single_label('posterior', link, y, torch.tensor(-1), torch.tensor(censoring.SRC_RANGE), wt, dG, fl)
    num = tf_single_label('posterior', link, y, torch.tensor(-1), torch.tensor(censoring.SRC_FLOOR), wt, dG, fl)
    assert float(num) == pytest.approx(float(tobit_posterior_mean(dG + y, dG + wt, torch.tensor(0))) - 3.0)
    assert float(tag) != pytest.approx(float(num), abs=1e-3)


@pytest.mark.parametrize('link', [SaturatingLink(tau=0.59), LeakyFloorLink(tau=0.59)])
def test_clamped_inverts_one_fixed_reading_below_y0_and_is_nan_with_no_reading(link):
    dG, fl = torch.tensor(3.0), torch.tensor(0.0)
    zero = torch.tensor(0)

    def at(v):
        return tf_single_label('clamped', link, torch.tensor(float(v)) - 3.0, zero, zero, torch.tensor(0.0), dG, fl)
    want = float(link.inverse(torch.tensor(TF_CLAMP_Y0), fl)) - 3.0
    assert float(at(-0.9)) == pytest.approx(want, abs=1e-5) and float(at(0.0)) == pytest.approx(want, abs=1e-5)
    assert float(at(2.0)) == pytest.approx(float(link.inverse(torch.tensor(2.0), fl)) - 3.0, abs=1e-5)
    assert torch.isnan(at(float('nan')))


# ---------------------------------------------------------------------------------------------------------------- through the block step

def _spy(s):
    """Record what the first call hands the WT half."""
    seen = {}
    forced = s._block_forced

    def wrap(c, wt_cal, wt_raw, n, lat, has, live):
        seen.setdefault('lat', lat.detach().clone())
        seen.setdefault('has', has.detach().clone())
        seen.setdefault('n', n.clone())
        return forced(c, wt_cal, wt_raw, n, lat, has, live)
    s._block_forced = wrap
    return seen


def test_the_default_label_is_the_inverse_whether_or_not_the_flags_are_set():
    full = lambda v: torch.full((20, 20), v)
    a, c = _module(lambda_rank_mt=0.0), _module(lambda_rank_mt=0.0, tf_label='inverse', tf_no_label='wt')
    sa, sc = _spy(a), _spy(c)
    M._block_step(a, _blk('pair', yA=full(0.3), yB=full(0.2)))
    M._block_step(c, _blk('pair', yA=full(0.3), yB=full(0.2)))
    assert sa['has'].any() and torch.equal(sa['has'], sc['has']) and torch.equal(sa['lat'].nan_to_num(), sc['lat'].nan_to_num())
    assert float(a.model.w_mt.grad) == pytest.approx(float(c.model.w_mt.grad), rel=1e-7)


def test_the_posterior_labels_every_double_even_with_a_tagged_single():
    s = _module(lambda_rank_mt=0.0, tf_label='posterior')
    seen = _spy(s)
    b = _blk('pair')
    b['si_y'] = torch.full_like(b['si_y'], -3.0)                                    # every single at i tagged '<-1', so the cached yA / yB are NaN
    b['si_cens'] = torch.full_like(b['si_cens'], -1)
    b['si_cens_src'] = torch.full_like(b['si_cens_src'], censoring.SRC_RANGE)
    b['yA'], b['yB'] = torch.full((20, 20), float('nan')), torch.full((20, 20), float('nan'))
    logs = M._block_step(s, b)
    assert seen['has'].all() and torch.isfinite(seen['lat']).all()
    assert logs['L_reg_combined'] == logs['L_reg_combined']
    assert s.model.w_mt.grad is not None and torch.isfinite(s.model.w_mt.grad)


def test_the_posterior_is_one_label_per_single_and_the_wild_type_has_no_effect():
    s = _module(lambda_rank_mt=0.0, tf_label='posterior')
    seen = _spy(s)
    M._block_step(s, _blk('pair'))
    lat, n = seen['lat'], seen['n']
    assert float(lat[n == 0].abs().max()) == 0.0
    idx = torch.nonzero(n == 2)
    r, k = idx[0].tolist()
    r2, k2 = idx[-1].tolist()
    assert float(lat[r, k] + lat[r2, k2]) == pytest.approx(float(lat[r, k2] + lat[r2, k]), abs=1e-4)     # every cell is its row's single plus its column's


def test_tf_no_label_mt_changes_only_the_doubles_without_a_label():
    nolab = lambda: _blk('pair', yA=torch.full((20, 20), float('nan')), yB=torch.full((20, 20), float('nan')))
    wt, mt = _module(lambda_rank_mt=0.0), _module(lambda_rank_mt=0.0, tf_no_label='mt')
    M._block_step(wt, nolab())
    M._block_step(mt, nolab())
    assert float(wt.model.w_mt.grad) != pytest.approx(float(mt.model.w_mt.grad), rel=1e-3)
    lab = lambda: _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2))
    a, c = _module(lambda_rank_mt=0.0), _module(lambda_rank_mt=0.0, tf_no_label='mt')
    M._block_step(a, lab())
    M._block_step(c, lab())
    assert float(a.model.w_mt.grad) == pytest.approx(float(c.model.w_mt.grad), rel=1e-6)


def test_the_singles_block_takes_the_same_labels():
    for kind in ('posterior', 'clamped'):
        s = _module(lambda_rank_mt=0.0, lambda_rank_wt=0.0, tf_label=kind)
        logs = M._block_step(s, _blk('single'))
        assert logs['L_reg_combined'] == logs['L_reg_combined'], kind
        assert s.model.w_mt.grad is not None and torch.isfinite(s.model.w_mt.grad), kind


# ---------------------------------------------------------------------------------------------------------------- flags

def _parse(monkeypatch, *a):
    monkeypatch.setattr(sys, 'argv', ['x', '--experiment_name', 't', *a])
    return config.parse_arguments()


def test_the_flags_default_to_todays_behaviour_and_refuse_what_they_cannot_mean(monkeypatch):
    a = _parse(monkeypatch)
    assert a.tf_label == 'inverse' and a.tf_no_label == 'wt'
    assert _parse(monkeypatch, '--block_training', '--tf_label', 'posterior').tf_label == 'posterior'
    for bad in (['--tf_label', 'posterior'], ['--tf_no_label', 'mt'], ['--block_training', '--tf_label', 'posterior', '--tf_cut', '0.5'],
                ['--block_training', '--tf_label', 'clamped', '--tf_no_label', 'mt']):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *bad)
