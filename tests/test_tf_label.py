"""``--tf_label`` / ``--tf_no_label``: what the WT half of a block's combined prediction is built from.

The posterior is checked against brute-force integration of prior x likelihood, so the closed form cannot drift from the Tobit loss it claims to invert.
"""
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import blocks, censoring, config
from esm_msr.data import stamp_tf_singles
from esm_msr.losses import TF_PRIOR_SD, TOBIT_SIGMA_Y, tobit_posterior_mean, tobit_regression_loss
from esm_msr.training import ESM3EpistasisLightningModule as M, tf_single_label
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
        assert float(z.diff().abs().max()) < 0.02, m                                # no jump anywhere


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
    dG, wt = torch.tensor(3.0), torch.tensor(-1.0)                                  # the WT head reads this single at absolute dG 2
    y = torch.tensor(-3.5)                                                          # read at absolute -0.5
    tag = tf_single_label(y, torch.tensor(-1), torch.tensor(censoring.SRC_RANGE), wt, dG)
    num = tf_single_label(y, torch.tensor(-1), torch.tensor(censoring.SRC_FLOOR), wt, dG)
    assert float(num) == pytest.approx(float(tobit_posterior_mean(dG + y, dG + wt, torch.tensor(0))) - 3.0)
    assert float(tag) != pytest.approx(float(num), abs=1e-3)


def test_the_posterior_label_with_no_reading_is_the_prior():
    dG, wt = torch.tensor(3.0), torch.tensor(-1.0)
    zero = torch.tensor(0)
    assert float(tf_single_label(torch.tensor(float('nan')), zero, zero, wt, dG)) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------------------------------------------- through the block step

def _spy(s):
    """Record what the first call hands the WT half."""
    seen = {}
    forced = s._block_forced

    def wrap(wt_cal, lat, has):
        seen.setdefault('lat', lat.detach().clone())
        seen.setdefault('has', has.detach().clone())
        return forced(wt_cal, lat, has)
    s._block_forced = wrap
    return seen


def _reads(s, b):
    """The WT head's calibrated read of each single at the block's two positions, as the block computes it; alphabet-indexed."""
    i, j = int(b['positions'][0]), int(b['positions'][1])
    wt_i, wt_j = int(b['wt_ids'][0]), int(b['wt_ids'][1])
    wt_rows, *_ = blocks.pair_sequences(b['seq'], i, j, wt_i, wt_j)
    llr = s.model.forward_site_llr(wt_rows, torch.as_tensor([i, j]), torch.as_tensor([wt_i, wt_j]), 'wt').detach()
    cal = s._block_cal('wt')
    w = [blocks.TOKEN_TO_AA[wt_i], blocks.TOKEN_TO_AA[wt_j]]
    return [cal(llr[k], (torch.arange(20) != w[k]).long()).detach() for k in (0, 1)], w


def _single_tensors(y_i, y_j, **kw):
    t = lambda v: torch.full((20,), float(v))
    return dict(si_y=t(y_i), sj_y=t(y_j), **kw)


def test_a_single_below_the_cut_costs_only_itself_and_its_partner_keeps_its_measurement():
    s = _module(lambda_rank_mt=0.0, tf_cut=0.5)
    seen = _spy(s)
    b = _blk('pair', **_single_tensors(-1.9, 0.5))                                 # at the stub's dG_wt of 2: i reads 0.1 (below the cut), j reads 2.5 (above it)
    (ri, _), (wi, wj) = _reads(s, b)
    M._block_step(s, b)
    lat, has = seen['lat'], seen['has']
    x, y = [a for a in range(20) if a != wi][0], [a for a in range(20) if a != wj][0]
    assert bool(has[x, y])
    assert float(lat[x, y]) == pytest.approx(float(ri[x]) + 0.5, abs=1e-5)         # the WT head's read of i, the measurement of j


def test_two_dead_singles_are_two_reads_and_two_live_ones_two_measurements():
    s = _module(lambda_rank_mt=0.0, tf_cut=0.5)
    seen = _spy(s)
    b = _blk('pair', **_single_tensors(-1.9, -1.8))
    reads, (wi, wj) = _reads(s, b)
    M._block_step(s, b)
    x, y = [a for a in range(20) if a != wi][0], [a for a in range(20) if a != wj][0]
    assert float(seen['lat'][x, y]) == pytest.approx(float(reads[0][x] + reads[1][y]), abs=1e-5) and not bool(seen['has'][x, y])
    s = _module(lambda_rank_mt=0.0, tf_cut=0.5)
    seen = _spy(s)
    M._block_step(s, _blk('pair', **_single_tensors(0.3, 0.2)))
    assert float(seen['lat'][x, y]) == pytest.approx(0.5, abs=1e-5) and bool(seen['has'][x, y])


def test_a_tagged_or_missing_single_is_read_from_the_wt_head_and_the_wild_type_is_zero():
    s = _module(lambda_rank_mt=0.0)
    seen = _spy(s)
    b = _blk('pair', **_single_tensors(-3.0, 0.5))
    b['si_cens'], b['si_cens_src'] = torch.full_like(b['si_cens'], -1), torch.full_like(b['si_cens_src'], censoring.SRC_RANGE)
    b['sj_y'] = b['sj_y'].clone()
    reads, (wi, wj) = _reads(s, b)
    b['sj_y'][(wj + 1) % 20] = float('nan')                                        # one single at j has no reading at all
    M._block_step(s, b)
    x, y = [a for a in range(20) if a != wi][0], [a for a in range(20) if a not in (wj, (wj + 1) % 20)][0]
    assert float(seen['lat'][x, y]) == pytest.approx(float(reads[0][x]) + 0.5, abs=1e-5) and bool(seen['has'][x, y])
    assert float(seen['lat'][x, (wj + 1) % 20]) == pytest.approx(float(reads[0][x] + reads[1][(wj + 1) % 20]), abs=1e-5)
    assert float(seen['lat'][wi, wj]) == 0.0 and bool(seen['has'][wi, wj])


def test_the_posterior_labels_every_double_even_with_a_tagged_single():
    s = _module(lambda_rank_mt=0.0, tf_label='posterior')
    seen = _spy(s)
    b = _blk('pair')
    b['si_y'] = torch.full_like(b['si_y'], -3.0)                                    # every single at i tagged '<-1', so the item's own ddG_A / ddG_B would be NaN
    b['si_cens'] = torch.full_like(b['si_cens'], -1)
    b['si_cens_src'] = torch.full_like(b['si_cens_src'], censoring.SRC_RANGE)
    logs = M._block_step(s, b)
    assert seen['has'].all() and torch.isfinite(seen['lat']).all()
    assert logs['L_reg_combined'] == logs['L_reg_combined']
    assert s.model.w_mt.grad is not None and torch.isfinite(s.model.w_mt.grad)


def test_the_posterior_is_one_label_per_single_and_the_wild_type_has_no_effect():
    s = _module(lambda_rank_mt=0.0, tf_label='posterior')
    seen = _spy(s)
    b = _blk('pair')
    wi, wj = blocks.TOKEN_TO_AA[int(b['wt_ids'][0])], blocks.TOKEN_TO_AA[int(b['wt_ids'][1])]
    M._block_step(s, b)
    lat = seen['lat']
    a = torch.arange(20)
    n = (a != wi).long().view(-1, 1) + (a != wj).long().view(1, -1)                 # mutations per cell
    assert float(lat[n == 0].abs().max()) == 0.0
    idx = torch.nonzero(n == 2)
    r, k = idx[0].tolist()
    r2, k2 = idx[-1].tolist()
    assert float(lat[r, k] + lat[r2, k2]) == pytest.approx(float(lat[r, k2] + lat[r2, k]), abs=1e-4)     # every cell is its row's single plus its column's


def test_the_singles_block_takes_the_same_labels():
    for kind in ('posterior', 'measured'):
        s = _module(lambda_rank_mt=0.0, lambda_rank_wt=0.0, tf_label=kind)
        logs = M._block_step(s, _blk('single'))
        assert logs['L_reg_combined'] == logs['L_reg_combined'], kind
        assert s.model.w_mt.grad is not None and torch.isfinite(s.model.w_mt.grad), kind


# ---------------------------------------------------------------------------------------------------------------- the per-item path

def test_stamp_tf_singles_keeps_each_measured_single_where_the_cache_blanks_both():
    def item(kind, pos, mts, y=0.0, src=0, **kw):
        return dict(subset_type=kind, mut_pos=np.array(pos), mt_id=np.array(mts), ddG=y, cens_src=src, **kw)
    nan = float('nan')
    items = [item('single', [3], [10], y=-1.2), item('single', [7], [20], y=-3.0, src=censoring.SRC_RANGE), item('single', [7], [21], y=0.4),
             item('double', [3, 7], [10, 20], ddG_A=nan, ddG_B=nan),    # the single at 7 is only a tag
             item('double', [3, 7], [10, 22], ddG_A=nan, ddG_B=nan),    # the single at 7 was never measured
             item('double', [7, 3], [21, 10], ddG_A=0.4, ddG_B=-1.2)]   # slot 0 is the item's first mutation
    stamp_tf_singles(items)
    assert items[3]['tf_A'] == pytest.approx(-1.2) and np.isnan(items[3]['tf_B'])
    assert items[4]['tf_A'] == pytest.approx(-1.2) and np.isnan(items[4]['tf_B'])
    assert items[5]['tf_A'] == pytest.approx(0.4) and items[5]['tf_B'] == pytest.approx(-1.2)
    assert 'tf_A' not in items[0]






