"""``--ci_weight`` and ``--no-censor_doubles``: confidence-interval weights in place of censoring on doubles.

* ``esm_msr.ci_weights``: w = [1/(sigma^2 + tau^2)] / population mean, sigma = deltaG_95CI / 3.92; 1.0 for anything censored or
  without an interval.
* ``ListMLELoss``: a weight scales the Plackett-Luce term in which its item is CHOSEN, never its place in a denominator.
* the loss composer: the regression numerator takes the weights, its denominator stays the sum of the SUBSET weights, and
  ``--ci_weight_losses`` picks which losses see them.
* ``censoring.apply_item_censoring(censor_multi=False)``: no floor/threshold/gap rule touches a measured double; singles and
  assay-bounded rows are exactly as before.
"""
import math
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import censoring, ci_weights, config
from esm_msr.data import MEAS_COLUMNS
from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M, ci_item_weights

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
CI_COL = MEAS_COLUMNS.index('deltaG_95CI')


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ the weights themselves

def test_weight_is_normalised_inverse_variance():
    ci = torch.tensor([0.392, 0.784])                       # sigma 0.1 and 0.2
    w = ci_weights.item_weights(ci, torch.ones(2, dtype=torch.bool), tau=0.0, ref=100.0)
    assert w.tolist() == pytest.approx([1.0, 0.25])         # (1/0.01)/100 and (1/0.04)/100


def test_tau_adds_to_every_items_variance_and_flattens_the_weights():
    ci = torch.tensor([0.392, 0.784])
    spread = lambda tau: float(ci_weights.item_weights(ci, torch.ones(2, dtype=torch.bool), tau, 1.0).max()
                               / ci_weights.item_weights(ci, torch.ones(2, dtype=torch.bool), tau, 1.0).min())
    assert spread(0.0) == pytest.approx(4.0)
    assert spread(0.1) == pytest.approx((0.04 + 0.01) / (0.01 + 0.01))
    assert spread(5.0) == pytest.approx((0.04 + 25.0) / (0.01 + 25.0))     # ~1.001      # tau at the model's scale: no discrimination left


def test_the_reference_makes_the_population_mean_one():
    rng = np.random.default_rng(0)
    widths = rng.gamma(4.0, 0.04, size=5000)
    ref = ci_weights.reference_weight(widths, tau=0.02)
    w = ci_weights.item_weights(torch.tensor(widths), torch.ones(5000, dtype=torch.bool), 0.02, ref)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-4)


def test_offsets_average_zero_over_the_weighted_population():
    """So a list that mixes weighted doubles with unweighted singles is not shifted against them."""
    rng = np.random.default_rng(1)
    widths = torch.tensor(rng.gamma(4.0, 0.04, size=4000))
    ref = ci_weights.reference_weight(widths.numpy(), 0.02)
    logmean = ci_weights.reference_log_weight(widths.numpy(), 0.02, ref)
    assert logmean < 0                                       # the weights average 1, so their logs average below 0
    applies = torch.ones(4000, dtype=torch.bool)
    w = ci_weights.item_weights(widths, applies, 0.02, ref)
    off = ci_weights.item_offsets(w, ci_weights.usable_mask(widths, applies), logmean)
    assert float(off.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-4)


def test_unusable_or_unapplied_items_stay_at_one():
    ci = torch.tensor([float('nan'), -0.1, 0.4, 0.4])
    applies = torch.tensor([True, True, False, True])
    w = ci_weights.item_weights(ci, applies, 0.0, 50.0)
    assert w[:3].tolist() == [1.0, 1.0, 1.0] and float(w[3]) != 1.0


def test_reference_ignores_missing_intervals_and_refuses_an_empty_population():
    assert ci_weights.reference_weight([0.392, float('nan'), -1.0], 0.0) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        ci_weights.reference_weight([float('nan')], 0.0)


def test_zero_tau_on_a_zero_width_interval_is_finite():
    w = ci_weights.item_weights(torch.tensor([0.0]), torch.ones(1, dtype=torch.bool), 0.0, 1.0)
    assert torch.isfinite(w).all()


def test_batch_helper_weights_only_uncensored_items_of_the_chosen_subsets():
    B = 5
    meas = torch.full((B, len(MEAS_COLUMNS)), float('nan'))
    meas[:, CI_COL] = 0.392
    batch = {'meas': meas}
    subsets = ['double', 'double', 'single', 'double', 'double']
    cens = torch.tensor([0, -1, 0, 0, 1])                    # item 1 and 4 are bounds: no value, no interval to weight by
    meas[3, CI_COL] = float('nan')                           # and item 3 has no interval at all
    w, off = ci_item_weights(batch, subsets, cens, {'double'}, 0.0, 50.0, logmean=0.25)
    assert float(w[0]) == pytest.approx(2.0)                 # (1/0.01)/50
    assert w[1:].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert float(off[0]) == pytest.approx(math.log(2.0) - 0.25)      # centred log w for the weighted item
    assert off[1:].tolist() == [0.0, 0.0, 0.0, 0.0]                  # and nothing for the rest


# ------------------------------------------------------------------ weighted Plackett-Luce

def _pl_by_hand(scores, order, weights):
    """-sum_k w_k * ((s_k + log w_k) - log sum_{j>=k} exp(s_j + log w_j)): the term scaled by the chosen item's weight, and
    every score (numerator and denominators) offset by its item's log weight."""
    out = 0.0
    for k, i in enumerate(order):
        rest = [scores[j] + math.log(weights[j]) for j in order[k:]]
        out += weights[i] * (scores[i] + math.log(weights[i]) - math.log(sum(math.exp(r) for r in rest)))
    return -out


def test_weighted_term_matches_the_hand_computed_likelihood():
    scores = [0.3, 1.1, -0.4, 0.8]
    truth = [2.0, 3.0, -1.0, 2.5]                            # descending order: 1, 3, 0, 2
    w = [0.5, 2.0, 0.1, 1.0]
    p, g = torch.tensor([scores]), torch.tensor([truth])
    m = torch.ones_like(p, dtype=torch.bool)
    got = ListMLELoss().forward(p, g, m, torch.tensor([w]))
    assert float(got) == pytest.approx(_pl_by_hand(scores, [1, 3, 0, 2], w), abs=1e-5)


def test_unit_weights_reproduce_the_unweighted_loss_exactly():
    crit = ListMLELoss()
    p = torch.tensor([[2.0, 1.0, 0.5, -1.0], [0.1, 0.9, -0.3, 0.0]])
    g = torch.tensor([[3.0, 2.0, 1.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
    m = torch.ones_like(p, dtype=torch.bool)
    c = torch.tensor([[0, 0, 0, -1], [0, 0, 0, 0]])
    ones = torch.ones_like(p)
    assert float(crit(p, g, m, ones)) == float(crit(p, g, m))
    assert float(crit.forward_censored(p, g, m, c, ones)) == float(crit.forward_censored(p, g, m, c))


def test_a_uniform_weight_just_rescales_the_loss():
    """Weights that do not discriminate must not change what is learned: the offset cancels in every softmax and the terms scale."""
    crit = ListMLELoss()
    p = torch.tensor([[2.0, 1.0, 0.5, -1.0]], requires_grad=True)
    g = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    m = torch.ones_like(p, dtype=torch.bool)
    base = crit(p, g, m)
    (gb,) = torch.autograd.grad(base, p)
    scaled = crit(p, g, m, torch.full_like(p, 2.5))
    (gs,) = torch.autograd.grad(scaled, p)
    assert float(scaled) == pytest.approx(2.5 * float(base), rel=1e-5)
    assert torch.allclose(gs, 2.5 * gb, atol=1e-5)


def test_uninformative_labels_exert_no_net_force_on_any_item():
    """The reason the weight is BOTH an offset and a term scale. With every score equal and the label order averaged over all
    24 orders of four items (labels that say nothing), the gradient on each score is exactly zero: a poorly measured item is not
    pushed down, a precisely measured one is not pushed up. Scaling only the chosen term fails this - the lowest-weight item is
    pushed down by +0.70 here and the highest up by -1.09 - which is a bias the features cannot explain and fits as noise
    (the simulation in docs/ci_weighting.md: -0.07 Spearman at high noise)."""
    import itertools
    w = torch.tensor([[0.2, 1.0, 1.0, 3.0]])
    n = 4

    def mean_grad(loss_of):
        total = torch.zeros(n)
        perms = list(itertools.permutations(range(n)))
        for perm in perms:
            s = torch.zeros(1, n, requires_grad=True)
            y = torch.zeros(1, n)
            for rank, item in enumerate(perm):
                y[0, item] = float(n - rank)
            loss_of(s, y).backward()
            total += s.grad[0]
        return total / len(perms)

    ones = torch.ones(1, n, dtype=torch.bool)
    ours = mean_grad(lambda s, y: ListMLELoss().forward(s, y, None, w))
    chosen_only = mean_grad(lambda s, y: ListMLELoss()._per_list(s, y, ones, ones, w)[0].sum())
    assert float(ours.abs().max()) < 1e-6
    assert float(chosen_only[0]) > 0.5 and float(chosen_only[3]) < -0.5      # the rejected design, for contrast


def test_a_mixed_list_is_balanced_only_when_the_term_weight_is_the_exponential_of_the_offset():
    """Two singles (weight 1, offset 0) and two weighted doubles in one list, uninformative labels, equal scores. The composer's
    pairing - term weight exp(offset), offsets centred on the weighted population - exerts no net force on any item. Keeping the
    arithmetic-mean-1 weights next to the centred offsets does (a few percent of a term's force, with a direction), which is
    what an epistasis study cannot afford between doubles and singles."""
    import itertools
    wd = torch.tensor([0.5, 1.8])
    logmean = -0.256                                          # the default tau's value on the doubles
    o = torch.cat([torch.zeros(2), torch.log(wd) - logmean])

    def force(weights):
        total, perms = torch.zeros(4), list(itertools.permutations(range(4)))
        for perm in perms:
            s = torch.zeros(1, 4, requires_grad=True)
            y = torch.zeros(1, 4)
            for rank, item in enumerate(perm):
                y[0, item] = float(4 - rank)
            ListMLELoss().forward(s, y, None, weights[None], o[None]).backward()
            total += s.grad[0]
        return total / len(perms)

    assert float(force(torch.exp(o)).abs().max()) < 1e-6
    assert float(force(torch.cat([torch.ones(2), wd])).abs().max()) > 0.03


def test_censored_members_carry_the_neutral_weight_and_ordinary_ones_move_the_loss():
    crit = ListMLELoss()
    p = torch.tensor([[3.0, 1.0, 0.5, -2.0]])
    g = torch.tensor([[9.0, 3.0, 2.0, -1.0]])
    m = torch.ones_like(p, dtype=torch.bool)
    c = torch.tensor([[1, 0, 0, -1]])                       # an upper bound, two ordinary members, a lower bound
    base = float(crit.forward_censored(p, g, m, c))
    assert float(crit.forward_censored(p, g, m, c, torch.ones_like(p))) == pytest.approx(base, abs=1e-6)
    assert float(crit.forward_censored(p, g, m, c, torch.tensor([[1.0, 2.0, 1.0, 1.0]]))) != pytest.approx(base, abs=1e-4)


def test_weights_keep_the_gradient_alive_on_every_member():
    p = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    g = torch.tensor([[3.0, 2.0, 1.0]])
    m = torch.ones_like(p, dtype=torch.bool)
    ListMLELoss().forward_censored(p, g, m, torch.zeros_like(g, dtype=torch.long), torch.tensor([[1.0, 0.0, 3.0]])).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0


# ------------------------------------------------------------------ the composer

class _CalHead(torch.nn.Module):
    use_scale = False
    bias = None

    def bias_for(self, n_mut):
        return 0.0


class _EnsembleModel(torch.nn.Module):
    lora_mode = 'ensemble'

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self.calibration_head_wt = _CalHead()

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        out = torch.full((int(micro['ddG'].shape[0]),), 0.0 if pass_type == 'mt' else 2.0) * self.w[0]
        return {'pred_calibrated': out, 'pred_raw': out}


class _HP(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def get(self, k, default=None):
        return dict.get(self, k, default)


def _stub(**extra):
    base = dict(subset_size=3, micro_batch_size=3, micro_batch_tokens=0, double_weight=1.0, cond_weight=1.0,
                native_cond_weight=1.0, wt_double_target='latent', mt_reg_mask='all', mask_strategy=None,
                lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0,
                combined_mode='mean', detach_regression=False)
    base.update(extra)
    s = types.SimpleNamespace()
    s.hparams = _HP(base)
    s.model = _EnsembleModel()
    s.ns_head = None
    s.link_head = None
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.backward_losses = []
    s.manual_backward = lambda loss, retain_graph=False: s.backward_losses.append(float(loss.detach()))
    for name in ('_compute_rank_loss', '_reg_loss'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    s._micro_batch = M._micro_batch
    return s


def _batch(ci_widths, targets, cens=None):
    """Doubles whose teacher-forced label is +1.0, so the combined prediction is 0.5 for every item and each item's squared
    error is (0.5 - target)^2, readable by hand."""
    B = len(ci_widths)
    f = lambda v: torch.full((B,), float(v))
    meas = torch.full((B, len(MEAS_COLUMNS)), float('nan'))
    meas[:, CI_COL] = torch.tensor(ci_widths)
    return {
        'ddG': torch.tensor(targets, dtype=torch.float32), 'ddG_additive': f(1.0), 'meas': meas,
        'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long), 'mut_mask': torch.ones(B, 2, dtype=torch.bool),
        'subset_type': ['double'] * B, 'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': f(1.0),
        'cens': torch.zeros(B, dtype=torch.long) if cens is None else torch.tensor(cens), 'cens_bound': f(float('nan')),
        'ddG_A': f(float('nan')), 'ddG_B': f(float('nan')),
    }


# six doubles, three precise (sigma 0.1) with error^2 2.25, three loose (sigma 0.2) with error^2 0.25
CI = [0.392] * 3 + [0.784] * 3
TARGET = [-1.0] * 3 + [1.0] * 3


def test_regression_numerator_is_weighted_and_the_denominator_stays_the_subset_sum():
    stub = _stub(ci_weight=True, ci_weight_tau=0.0, ci_weight_ref=100.0, ci_weight_subsets=['double'], ci_weight_losses='both')
    logs = M._compose_losses_streaming_and_backward(stub, _batch(CI, TARGET))
    # weights 1.0 and 0.25; the summed backward losses are sum(w * err^2) / sum(subset weights) = 6.9375 / 6
    assert sum(stub.backward_losses) == pytest.approx((3 * 1.0 * 2.25 + 3 * 0.25 * 0.25) / 6.0, abs=1e-5)
    # the logged value is the weighted MEAN (sum w err^2 / sum w), a readable training loss
    assert float(logs['L_reg_combined']) == pytest.approx((3 * 2.25 + 3 * 0.25 * 0.25) / 3.75, abs=1e-5)


def test_ci_weight_off_is_the_plain_mean():
    stub = _stub()
    M._compose_losses_streaming_and_backward(stub, _batch(CI, TARGET))
    assert sum(stub.backward_losses) == pytest.approx((3 * 2.25 + 3 * 0.25) / 6.0, abs=1e-5)


def test_rank_only_leaves_the_regression_unweighted():
    stub = _stub(ci_weight=True, ci_weight_tau=0.0, ci_weight_ref=100.0, ci_weight_subsets=['double'], ci_weight_losses='rank')
    M._compose_losses_streaming_and_backward(stub, _batch(CI, TARGET))
    assert sum(stub.backward_losses) == pytest.approx((3 * 2.25 + 3 * 0.25) / 6.0, abs=1e-5)


def test_censored_doubles_keep_weight_one_in_the_regression():
    stub = _stub(ci_weight=True, ci_weight_tau=0.0, ci_weight_ref=100.0, ci_weight_subsets=['double'], ci_weight_losses='reg')
    # item 0 is a lower-censored bound at -2.0 (the prediction 0.5 sits above it: a hinge error of 6.25); its weight must be 1,
    # not the 1.0 its interval would have given, so move its interval to something that WOULD change it if it were read
    batch = _batch([0.1] + CI[1:], TARGET, cens=[-1, 0, 0, 0, 0, 0])
    batch['cens_bound'][0] = -2.0
    M._compose_losses_streaming_and_backward(stub, batch)
    expected = (1.0 * 6.25 + 2 * 1.0 * 2.25 + 3 * 0.25 * 0.25) / 6.0
    assert sum(stub.backward_losses) == pytest.approx(expected, abs=1e-5)


def test_rank_weights_reach_the_list_loss():
    pred = torch.tensor([0.9, 0.1, -0.5])
    targ = torch.tensor([3.0, 2.0, 1.0])
    mask = torch.ones(3, dtype=torch.bool)
    cens = torch.zeros(3, dtype=torch.long)
    stub = _stub()
    plain, _, _ = M._compute_rank_loss(stub, pred, targ, mask, cens, 3, ListMLELoss())
    unit, _, _ = M._compute_rank_loss(stub, pred, targ, mask, cens, 3, ListMLELoss(), torch.ones(3))
    w = torch.tensor([2.0, 1.0, 0.5])
    weighted, _, _ = M._compute_rank_loss(stub, pred, targ, mask, cens, 3, ListMLELoss(), w)
    assert float(unit) == float(plain)
    direct = ListMLELoss().forward_censored(pred[None], targ[None], mask[None], cens[None], w[None])    # one list of the full size
    assert float(weighted) == pytest.approx(float(direct), abs=1e-6)
    assert float(weighted) != pytest.approx(float(plain), abs=1e-3)
    # the composer hands the CENTRED offsets down, which differ from log w by a constant for a list of weighted items only
    off = torch.log(w) + 0.3
    centred, _, _ = M._compute_rank_loss(stub, pred, targ, mask, cens, 3, ListMLELoss(), w, off)
    assert float(centred) == pytest.approx(float(weighted), abs=1e-6)          # a constant shift of a whole list changes nothing


# ------------------------------------------------------------------ censoring switch

def _item(subset, dG, gap=0.0, **kw):
    d = dict(subset_type=subset, dG_meas=dG, ddG=dG - 3.0, protease_gap=gap, y_floor=0.0, cens=0, cens_src=0)
    d.update(kw)
    return d


def test_no_censor_doubles_spares_every_value_rule_on_a_double_only():
    items = [_item('single', -0.5), _item('double', -0.5), _item('double', 2.0, gap=3.0), _item('cond', -0.5),
             _item('single', 2.0, gap=3.0)]
    on, _ = censoring.apply_item_censoring([dict(i) for i in items], True, 2.0, threshold=0.5, threshold_multi=0.5)
    off, c = censoring.apply_item_censoring([dict(i) for i in items], True, 2.0, threshold=0.5, threshold_multi=0.5,
                                            censor_multi=False)
    assert [i['cens'] for i in on] == [-1, -1, -1, -1, -1]            # the status quo: all five are censored
    assert [i['cens'] for i in off] == [-1, 0, 0, 0, -1]               # only the singles are, by threshold and by gap
    assert all(np.isnan(i['cens_bound']) for i in off if i['cens'] == 0)
    assert c['floor_by_subset'].get('double', 0) == 0 and c['gap_by_subset'].get('double', 0) == 0


def test_no_censor_doubles_keeps_an_assay_bounded_double_as_a_bound():
    bounded = _item('double', -1.0, cens=-1, cens_src=censoring.SRC_RANGE, dG_bound=-1.0)
    out, c = censoring.apply_item_censoring([bounded], True, 2.0, threshold=0.5, threshold_multi=0.5, censor_multi=False)
    assert out[0]['cens'] == -1 and out[0]['cens_src'] == censoring.SRC_RANGE and c['range'] == 1


def test_default_is_unchanged():
    items = [_item('double', -0.5), _item('single', -0.5)]
    a, _ = censoring.apply_item_censoring([dict(i) for i in items], True, 2.0, threshold=0.5, threshold_multi=0.5)
    b, _ = censoring.apply_item_censoring([dict(i) for i in items], True, 2.0, threshold=0.5, threshold_multi=0.5,
                                          censor_multi=True)
    assert a == b


# ------------------------------------------------------------------ flags

def test_defaults_leave_both_features_off():
    a = parse()
    assert a.ci_weight is False and a.censor_doubles is True
    assert a.ci_weight_subsets == ['double'] and a.ci_weight_losses == 'both' and a.ci_weight_ref is None


def test_flags_parse_and_validate():
    a = parse('--ci_weight', '--no-censor_doubles', '--ci_weight_tau', '0.05', '--ci_weight_losses', 'reg',
              '--ci_weight_subsets', 'double', 'single', 'double')
    assert a.ci_weight and not a.censor_doubles and a.ci_weight_tau == 0.05 and a.ci_weight_losses == 'reg'
    assert a.ci_weight_subsets == ['double', 'single']                # de-duplicated, sorted
    for bad in (['--ci_weight', '--ci_weight_tau', '-0.1'], ['--ci_weight', '--ci_weight_ref', '0'],
                ['--ci_weight_losses', 'nope'], ['--ci_weight_subsets', 'triple']):
        with pytest.raises(SystemExit):
            parse(*bad)
