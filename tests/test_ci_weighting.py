"""``--ci_weight`` and ``--no-censor_doubles``: label-uncertainty weights in place of censoring on doubles.

* ``esm_msr.ci_weights``: a label's variance is its fit interval's plus a quarter of its LIBRARY-CENTRED squared protease disagreement (averaged
  with a depth prior for a double), and the weight is 1 / (tau^2 + variance) normalised to mean 1; 1.0 for anything censored or without readings.
* ``ListMLELoss``: a weight scales the Plackett-Luce term in which its item is chosen AND offsets its score in every denominator.
* the loss composer: the regression numerator takes the weights, its denominator stays the sum of the SUBSET weights, every uncensored item of
  every subset is weighted alike, and ``--ci_weight_losses`` picks which losses see them.
* ``censoring.apply_item_censoring(censor_multi=False)``: no floor/threshold/gap rule touches a measured double; singles and assay-bounded rows
  are exactly as before.
"""
import itertools
import math
import sys
import types

import numpy as np
import pytest
import torch

from esm_msr import censoring, ci_weights, config
from esm_msr.data import MEAS_COLUMNS, _MEAS_INDEX
from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M, ci_item_weights

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ the label variance

def test_label_variance_is_the_interval_plus_a_quarter_of_the_squared_deviation():
    ci = 0.392                                                # s = 0.1
    L = ci_weights.label_variance(ci, 2.4, 1.6, offset=0.0)   # gap 0.8
    assert float(L) == pytest.approx(0.1 ** 2 + 0.25 * 0.8 ** 2)


def test_a_library_wide_protease_offset_cancels():
    """The offset is shared by the whole library and cancels in ddG = dG - dG_wt, so a variant that sits on it has no deviation."""
    on_offset = ci_weights.label_variance(0.392, 2.0 + 0.6, 2.0, offset=0.6)
    deviating = ci_weights.label_variance(0.392, 2.0 + 0.6 + 0.8, 2.0, offset=0.6)
    assert float(on_offset) == pytest.approx(0.1 ** 2)
    assert float(deviating) == pytest.approx(0.1 ** 2 + 0.25 * 0.8 ** 2)


def test_the_deviation_is_capped():
    """dG_ML is clamped to [-1, 5]; a 15 kcal/mol disagreement (an unbounded per-protease dG of -15) is not a 15 kcal/mol label error."""
    assert float(ci_weights.label_variance(0.0, -15.0, 2.0, 0.0)) == pytest.approx(0.25 * ci_weights.GAP_CAP ** 2)


def test_missing_readings_give_no_variance_and_hence_weight_one():
    assert np.isnan(ci_weights.label_variance(0.3, float('nan'), 2.0, 0.0))
    w = ci_weights.item_weights(torch.tensor([float('nan'), 0.04]), torch.ones(2, dtype=torch.bool), tau=0.5, ref=2.0)
    assert float(w[0]) == 1.0 and float(w[1]) != 1.0


def test_depth_prior_falls_with_the_additive_expectation_and_is_flat_beyond_its_ends():
    x = np.array([-9.0, -4.5, -3.0, -1.5, -0.25, 0.75, 2.5])
    m = ci_weights.depth_prior(x)
    assert (np.diff(m) <= 1e-9).all()                        # the deeper the expectation, the larger the expected disagreement
    assert m[0] == m[1] == ci_weights.DEPTH_M[0]              # flat below the first centre
    assert float(ci_weights.depth_prior(100.0)) == ci_weights.DEPTH_M[-1]
    assert np.isnan(ci_weights.depth_prior(float('nan')))


def test_a_double_with_a_deep_expectation_is_less_certain_than_a_shallow_one_with_the_same_deviation():
    kw = dict(ci_width=0.1, dG_t=2.2, dG_c=2.0, offset=0.0)
    shallow = float(ci_weights.label_variance(x=2.0, **kw))
    deep = float(ci_weights.label_variance(x=-3.0, **kw))
    single = float(ci_weights.label_variance(**kw))          # no expectation: its own deviation only
    assert deep > 10 * shallow and single < deep
    own = (0.1 / 3.92) ** 2 + 0.25 * 0.2 ** 2
    assert single == pytest.approx(own)
    assert deep == pytest.approx((0.1 / 3.92) ** 2 + 0.25 * 0.5 * (0.2 ** 2 + float(ci_weights.depth_prior(-3.0))))


def test_library_offset_is_the_median_over_in_range_variants_only():
    gaps = np.array([0.5] * 12 + [3.0] * 20)                 # 12 in range sit at 0.5; 20 floor variants disagree wildly
    reads = np.array([2.0] * 12 + [-0.5] * 20)
    assert ci_weights.library_offset(gaps, reads) == pytest.approx(0.5)
    assert ci_weights.library_offset(gaps[:5], reads[:5]) == 0.0         # too few in-range variants to estimate it


def _meas(**kw):
    a = np.full(len(MEAS_COLUMNS), np.nan, dtype=np.float32)
    for k, v in kw.items():
        a[_MEAS_INDEX[k]] = v
    return a


def test_stamping_a_library_uses_its_own_offset_and_the_depth_prior_for_doubles_only():
    items = [{'subset_type': 'single', 'meas': _meas(dG_ML=2.0, deltaG_t=2.6, deltaG_c=2.0, deltaG_95CI=0.392)} for _ in range(20)]
    items.append({'subset_type': 'single', 'meas': _meas(dG_ML=2.0, deltaG_t=3.4, deltaG_c=2.0, deltaG_95CI=0.392)})      # 0.8 off the offset
    items.append({'subset_type': 'double', 'dG_wt': 2.0, 'ddG_additive': -5.0,
                  'meas': _meas(dG_ML=0.0, deltaG_t=0.6, deltaG_c=0.0, deltaG_95CI=0.392)})                               # on the offset, but deep
    items.append({'subset_type': 'single', 'meas': None})
    out = ci_weights.stamp_label_variance(items, _MEAS_INDEX)
    assert out['offset'] == pytest.approx(0.6, abs=1e-6)
    assert items[0]['label_var'] == pytest.approx(0.1 ** 2, abs=1e-6)                                 # on the library's offset
    assert items[20]['label_var'] == pytest.approx(0.1 ** 2 + 0.25 * 0.8 ** 2, abs=1e-5)
    assert items[21]['label_var'] == pytest.approx(0.1 ** 2 + 0.25 * 0.5 * float(ci_weights.depth_prior(-3.0)), abs=1e-5)
    assert math.isnan(items[22]['label_var'])


# ------------------------------------------------------------------ the weights

def test_weight_is_normalised_inverse_total_variance():
    L = torch.tensor([0.01, 0.04])
    w = ci_weights.item_weights(L, torch.ones(2, dtype=torch.bool), tau=0.0, ref=100.0)
    assert w.tolist() == pytest.approx([1.0, 0.25])


def test_tau_flattens_the_weights():
    L = torch.tensor([0.01, 0.5])
    spread = lambda tau: float(ci_weights.item_weights(L, torch.ones(2, dtype=torch.bool), tau, 1.0).max()
                               / ci_weights.item_weights(L, torch.ones(2, dtype=torch.bool), tau, 1.0).min())
    assert spread(0.0) == pytest.approx(50.0)
    assert spread(0.5) == pytest.approx((0.25 + 0.5) / (0.25 + 0.01))
    assert spread(5.0) == pytest.approx((25.0 + 0.5) / (25.0 + 0.01))


def test_the_reference_makes_the_population_mean_one_and_the_offsets_average_zero():
    rng = np.random.default_rng(0)
    L = rng.gamma(0.7, 0.3, size=6000)
    ref = ci_weights.reference_weight(L, 0.5)
    logmean = ci_weights.reference_log_weight(L, 0.5, ref)
    applies = torch.ones(6000, dtype=torch.bool)
    w = ci_weights.item_weights(torch.tensor(L), applies, 0.5, ref)
    off = ci_weights.item_offsets(w, ci_weights.usable_mask(torch.tensor(L), applies), logmean)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-4)
    assert logmean < 0                                        # the weights average 1, so their logs average below 0
    assert float(off.mean()) == pytest.approx(0.0, abs=1e-4)


def test_summary_reports_the_most_that_weighting_can_gain():
    flat = ci_weights.summarise(np.full(100, 0.1), 0.5, ci_weights.reference_weight(np.full(100, 0.1), 0.5))
    assert flat['efficiency'] == pytest.approx(1.0) and flat['ess_frac'] == pytest.approx(1.0)
    L = np.r_[np.full(50, 0.01), np.full(50, 2.0)]
    s = ci_weights.summarise(L, 0.5, ci_weights.reference_weight(L, 0.5))
    assert s['efficiency'] > 1.2 and s['ess_frac'] < 1.0


def test_reference_refuses_an_empty_population():
    with pytest.raises(ValueError):
        ci_weights.reference_weight([float('nan')], 0.5)


def test_batch_helper_weights_every_subset_alike_and_leaves_bounds_alone():
    batch = {'label_var': torch.tensor([0.01, 0.01, 0.01, float('nan'), 0.01])}
    cens = torch.tensor([0, 0, -1, 0, 1])                    # items 2 and 4 are bounds; item 3 has no readings
    w, off = ci_item_weights(batch, cens, tau=0.0, ref=50.0, logmean=0.25)
    assert w[:2].tolist() == pytest.approx([2.0, 2.0])        # a single and a double alike: (1/0.01)/50
    assert w[2:].tolist() == [1.0, 1.0, 1.0]
    assert off[:2].tolist() == pytest.approx([math.log(2.0) - 0.25] * 2)
    assert off[2:].tolist() == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ weighted Plackett-Luce

def _pl_by_hand(scores, order, weights, offsets):
    """-sum_k w_k * ((s_k + o_k) - log sum_{j>=k} exp(s_j + o_j)): each term scaled by its chosen item's weight, every score offset."""
    out = 0.0
    for k, i in enumerate(order):
        rest = [scores[j] + offsets[j] for j in order[k:]]
        out += weights[i] * (scores[i] + offsets[i] - math.log(sum(math.exp(r) for r in rest)))
    return -out


def test_weighted_term_matches_the_hand_computed_likelihood():
    scores = [0.3, 1.1, -0.4, 0.8]
    truth = [2.0, 3.0, -1.0, 2.5]                            # descending order: 1, 3, 0, 2
    w = [0.5, 2.0, 0.1, 1.0]
    p, g = torch.tensor([scores]), torch.tensor([truth])
    m = torch.ones_like(p, dtype=torch.bool)
    got = ListMLELoss().forward(p, g, m, torch.tensor([w]))
    assert float(got) == pytest.approx(_pl_by_hand(scores, [1, 3, 0, 2], w, [math.log(x) for x in w]), abs=1e-5)
    off = [0.2, -0.1, 0.4, 0.0]                              # and with explicit (centred) offsets in place of log w
    got = ListMLELoss().forward(p, g, m, torch.tensor([w]), torch.tensor([off]))
    assert float(got) == pytest.approx(_pl_by_hand(scores, [1, 3, 0, 2], w, off), abs=1e-5)


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


def _mean_force(loss_of, n=4):
    """The gradient on each score at equal scores, averaged over every label order of n items: labels that say nothing."""
    total, perms = torch.zeros(n), list(itertools.permutations(range(n)))
    for perm in perms:
        s = torch.zeros(1, n, requires_grad=True)
        y = torch.zeros(1, n)
        for rank, item in enumerate(perm):
            y[0, item] = float(n - rank)
        loss_of(s, y).backward()
        total += s.grad[0]
    return total / len(perms)


def test_uninformative_labels_exert_no_net_force_when_the_weighting_is_uniform():
    """The reason the weight is BOTH an offset and a term scale. With every score equal and the label order averaged over all 24 orders of
    four items, the gradient on each score is exactly zero for the shipped pairing (term scale w, centred offset log w - mean log w): a poorly
    measured item is not pushed down, a precisely measured one is not pushed up. Scaling only the chosen term fails this - the lowest-weight
    item is pushed down with +0.70 here and the highest up with -1.09 - a bias the features cannot explain, which fits as noise (the
    simulation in docs/ci_weighting.md: -0.07 Spearman at high noise)."""
    w = torch.tensor([[0.2, 1.0, 1.0, 3.0]])
    off = torch.log(w) + 0.3                                  # centring is a constant shift of the offsets
    ones = torch.ones(1, 4, dtype=torch.bool)
    ours = _mean_force(lambda s, y: ListMLELoss().forward(s, y, None, w, off))
    chosen_only = _mean_force(lambda s, y: ListMLELoss()._per_list(s, y, ones, ones, w)[0].sum())
    assert float(ours.abs().max()) < 1e-6
    assert float(chosen_only[0]) > 0.5 and float(chosen_only[3]) < -0.5      # the rejected design, for contrast


def test_leaving_one_termed_item_unweighted_breaks_the_balance():
    """Why the weighting must be uniform over the items that carry terms: give item 0 weight 1 and offset 0 beside weighted others and the
    same average now pushes on it. (Censored items are fine - they have no term - which is why only uncensored items are weighted.)"""
    w = torch.tensor([[1.0, 0.5, 1.8, 0.7]])
    off = torch.cat([torch.zeros(1, 1), torch.log(w[:, 1:]) + 0.3], dim=1)
    force = _mean_force(lambda s, y: ListMLELoss().forward(s, y, None, w, off))
    assert float(force.abs().max()) > 0.03


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


def _batch(label_vars, targets, cens=None, subsets=None):
    """Doubles whose teacher-forced label is +1.0, so the combined prediction is 0.5 for every item and each item's squared
    error is (0.5 - target)^2, readable by hand."""
    B = len(label_vars)
    f = lambda v: torch.full((B,), float(v))
    return {
        'ddG': torch.tensor(targets, dtype=torch.float32), 'ddG_additive': f(1.0), 'label_var': torch.tensor(label_vars, dtype=torch.float32),
        'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long), 'mut_mask': torch.ones(B, 2, dtype=torch.bool),
        'subset_type': subsets or ['double'] * B, 'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': f(1.0),
        'cens': torch.zeros(B, dtype=torch.long) if cens is None else torch.tensor(cens), 'cens_bound': f(float('nan')),
        'ddG_A': f(float('nan')), 'ddG_B': f(float('nan')),
    }


# six items: three precise (variance 0.01) with error^2 2.25, three loose (variance 0.04) with error^2 0.25
LV = [0.01] * 3 + [0.04] * 3
TARGET = [-1.0] * 3 + [1.0] * 3
ON = dict(ci_weight=True, ci_weight_tau=0.0, ci_weight_ref=100.0, ci_weight_losses='both')


def test_regression_numerator_is_weighted_and_the_denominator_stays_the_subset_sum():
    stub = _stub(**ON)
    logs = M._compose_losses_streaming_and_backward(stub, _batch(LV, TARGET))
    # weights 1.0 and 0.25; the summed backward losses are sum(w * err^2) / sum(subset weights) = 6.9375 / 6
    assert sum(stub.backward_losses) == pytest.approx((3 * 1.0 * 2.25 + 3 * 0.25 * 0.25) / 6.0, abs=1e-5)
    # the logged value is the weighted MEAN (sum w err^2 / sum w), a readable training loss
    assert float(logs['L_reg_combined']) == pytest.approx((3 * 2.25 + 3 * 0.25 * 0.25) / 3.75, abs=1e-5)


def test_every_subset_is_weighted_alike():
    """Uniform on purpose: nothing in the loss depends on whether an item is a single or a double."""
    stub = _stub(**ON)
    M._compose_losses_streaming_and_backward(stub, _batch(LV, TARGET, subsets=['single', 'double'] * 3))
    assert sum(stub.backward_losses) == pytest.approx((3 * 1.0 * 2.25 + 3 * 0.25 * 0.25) / 6.0, abs=1e-5)


def test_ci_weight_off_is_the_plain_mean():
    stub = _stub()
    M._compose_losses_streaming_and_backward(stub, _batch(LV, TARGET))
    assert sum(stub.backward_losses) == pytest.approx((3 * 2.25 + 3 * 0.25) / 6.0, abs=1e-5)


def test_rank_only_leaves_the_regression_unweighted():
    stub = _stub(**{**ON, 'ci_weight_losses': 'rank'})
    M._compose_losses_streaming_and_backward(stub, _batch(LV, TARGET))
    assert sum(stub.backward_losses) == pytest.approx((3 * 2.25 + 3 * 0.25) / 6.0, abs=1e-5)


def test_a_censored_item_keeps_weight_one_in_the_regression():
    stub = _stub(**{**ON, 'ci_weight_losses': 'reg'})
    # item 0 is a lower-censored bound at -2.0 (the prediction 0.5 sits above it: a hinge error of 6.25); its variance would give it weight
    # 4.0 if it were read, but a bound is not a value with a variance, so it must count once
    batch = _batch([0.0025] + LV[1:], TARGET, cens=[-1, 0, 0, 0, 0, 0])
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
    off = torch.log(w) + 0.3                                  # the composer hands CENTRED offsets: a constant shift of a whole list
    centred, _, _ = M._compute_rank_loss(stub, pred, targ, mask, cens, 3, ListMLELoss(), w, off)
    assert float(centred) == pytest.approx(float(weighted), abs=1e-6)


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
    assert a.ci_weight_tau == 0.5 and a.ci_weight_losses == 'both' and a.ci_weight_ref is None
    assert not hasattr(a, 'ci_weight_subsets')                 # the weighting is uniform: there is no subset to choose


def test_flags_parse_and_validate():
    a = parse('--ci_weight', '--no-censor_doubles', '--ci_weight_tau', '0.1', '--ci_weight_losses', 'reg')
    assert a.ci_weight and not a.censor_doubles and a.ci_weight_tau == 0.1 and a.ci_weight_losses == 'reg'
    for bad in (['--ci_weight', '--ci_weight_tau', '-0.1'], ['--ci_weight', '--ci_weight_ref', '0'],
                ['--ci_weight_losses', 'nope'], ['--ci_weight_subsets', 'double']):
        with pytest.raises(SystemExit):
            parse(*bad)
