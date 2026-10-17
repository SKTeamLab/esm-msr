"""
The reimplemented teacher-forced ensemble: where the WT half comes from, what its label is, and the ratio between them.

The legacy objective always used the label when it had one. That is the regime the MT head will NOT be in at
inference: there is no label then, only the WT head's own prediction, which carries its own error the MT head never
saw. --tf_label_prob mixes the two regimes, resampled every step, and --tf_label_prob_end anneals it.

The label is the item's OWN target (latent under the link), not ddG_additive: the WT half of a cond item is the WT
pass on that item, which scores the target mutation on the background, and ddG_A + ddG_B is the label of a double's
WT pass. The integration tests below pin that by giving every item a ddG_additive 1 kcal/mol off its target.
"""
import os
import sys

import pytest
import torch

from esm_msr.link import MonotoneLink
from esm_msr.training import tf_label, tf_label_prob_at, tf_wt_half

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # the stub harness in test_compose_censoring
from test_compose_censoring import StubModel, make_batch, run  # noqa: E402

NAN = float('nan')
LBL = torch.tensor([-1.0, -2.0, NAN, 0.5])        # the third item has no usable label
PRED = torch.tensor([-1.2, -1.7, -0.3, 0.9])


# ---------------- the ratio ----------------

def test_ratio_one_always_takes_the_label_where_there_is_one():
    """The unlabelled item still falls back to the prediction."""
    half, took = tf_wt_half(LBL, PRED, 1.0)
    torch.testing.assert_close(half, torch.tensor([-1.0, -2.0, -0.3, 0.5]))
    assert took.tolist() == [True, True, False, True]


def test_ratio_zero_is_fully_free_running():
    half, took = tf_wt_half(LBL, PRED, 0.0)
    torch.testing.assert_close(half, PRED)
    assert took.tolist() == [False] * 4


def test_an_item_without_a_label_never_takes_one():
    for p in (0.0, 0.3, 0.75, 1.0):
        _, took = tf_wt_half(LBL, PRED, p)
        assert took[2].item() is False, p


def test_the_realised_ratio_matches_the_requested_one():
    torch.manual_seed(0)
    lbl = torch.zeros(20000)                       # all labelled, so every item is eligible
    pred = torch.ones(20000)
    for p in (0.25, 0.5, 0.9):
        _, took = tf_wt_half(lbl, pred, p)
        assert abs(took.float().mean().item() - p) < 0.02, p


def test_the_draw_is_resampled_so_an_item_sees_both_regimes():
    torch.manual_seed(1)
    lbl, pred = torch.zeros(2000), torch.ones(2000)
    a = tf_wt_half(lbl, pred, 0.5)[1]
    b = tf_wt_half(lbl, pred, 0.5)[1]
    assert not torch.equal(a, b), "the same mask twice: the ratio would be a fixed per-item split, not a regime mix"


def test_the_half_is_exactly_one_of_the_two_sources_per_item():
    """No blending: an item takes the label or the prediction, never an average of them."""
    torch.manual_seed(2)
    half, took = tf_wt_half(LBL, PRED, 0.5)
    for i in range(4):
        src = LBL[i] if took[i] else PRED[i]
        assert half[i] == src, (i, half[i], src)


# ---------------- the label ----------------

def test_without_a_link_the_label_is_the_target_and_a_bound_has_none():
    ddG = torch.tensor([-1.0, 0.3, -2.5])
    lbl = tf_label(ddG, cens=torch.tensor([0, 0, -1]))
    torch.testing.assert_close(lbl[:2], ddG[:2])
    assert torch.isnan(lbl[2])


def test_with_a_link_the_label_is_the_latent_that_reproduces_the_observation():
    """h(dG_wt + bg + label) - dG_wt is the observed target ddG + bg: the MT regression's own convention."""
    link = MonotoneLink()
    dGwt, bg = torch.full((3,), 2.0), torch.tensor([0.0, -0.8, 0.4])
    ddG = torch.tensor([-1.2, 0.5, 1.0])               # absolute observed dG 0.8, 1.7, 3.4: inside the knees
    z = tf_label(ddG, link, dGwt, bg)
    torch.testing.assert_close(link.obs_ddG(z, dGwt, bg).detach(), ddG + bg, atol=1e-4, rtol=0)
    assert not torch.allclose(z, ddG, atol=1e-2), "the link is nowhere near the identity at the knees' margins"


def test_with_a_link_an_observation_beyond_the_knees_has_no_label():
    """At the floor or the ceiling the latent is unidentified, so the item takes the WT head's prediction instead."""
    link = MonotoneLink()                              # knees at -1 and 5
    lo_, hi_, _, _ = (float(v) for v in link.params())
    dGwt, bg = torch.full((3,), 2.0), torch.zeros(3)
    obs_abs = torch.tensor([float(link(torch.tensor(lo_ - 0.5))), 2.0, float(link(torch.tensor(hi_ + 0.5)))])
    z = tf_label(obs_abs - dGwt, link, dGwt, bg)
    assert torch.isnan(z[0]) and torch.isfinite(z[1]) and torch.isnan(z[2]), z


# ---------------- the schedule ----------------

class HP(dict):
    __getattr__ = dict.__getitem__


def test_without_an_end_the_ratio_is_fixed():
    hp = HP(tf_label_prob=0.7, tf_label_prob_end=None, tf_anneal_steps=0)
    assert [tf_label_prob_at(hp, s) for s in (0, 10, 10 ** 6)] == [0.7] * 3


def test_the_anneal_is_linear_from_where_the_mt_learning_rate_starts_and_then_held():
    hp = HP(tf_label_prob=1.0, tf_label_prob_end=0.0, tf_anneal_steps=1000, mt_lora_delay_steps=200)
    assert tf_label_prob_at(hp, 0) == 1.0
    assert tf_label_prob_at(hp, 200) == 1.0
    assert tf_label_prob_at(hp, 700) == pytest.approx(0.5)
    assert tf_label_prob_at(hp, 1200) == 0.0
    assert tf_label_prob_at(hp, 5000) == 0.0


# ---------------- through the real training step (stub model) ----------------

def _cond_batch(link=None):
    """test_compose_censoring's batch, no censoring, every item with an absurd ddG_additive, and the cond items' feature
    (= the stub's MT prediction at w = 1) set exactly to their label, so the MT regression is satisfied exactly."""
    b = make_batch(with_cens=False)
    n = len(b['ddG'])
    cond = torch.tensor([s == 'cond' for s in b['subset_type']])
    b['ddG'] = torch.where(cond, 0.6 * b['ddG'].clamp(-2, 2), b['ddG'])      # observed dG 2 + ddG stays inside the knees
    # wrong by a destabilising background's worth, like ddG_additive on a real cond item, and small enough that a
    # combined prediction built on it stays inside the link's range, where it would still have a gradient
    b['ddG_additive'] = b['ddG'] - 1.0
    b['bg_offset'] = torch.where(cond, torch.linspace(-0.5, 0.5, n), torch.zeros(n))
    lbl = tf_label(b['ddG'], link, b['dG_wt'], b['bg_offset'])
    b['feat'] = torch.where(cond, lbl, b['feat'])
    return b


def _mt_grad(batch, link=None, **kw):
    _, stub = run(batch, link_head=link, lambda_mt_colrank=0.0, subfloor_rank_only=False, **kw)
    return stub.model.w.grad[1].clone()


@pytest.mark.parametrize('use_link', [False, True])
def test_a_prediction_that_satisfies_the_mt_regression_also_satisfies_the_teacher_forced_term(use_link):
    """The bug this pins: with ddG_additive as the label, the teacher-forced term pulled the MT head away from its own target."""
    link = MonotoneLink() if use_link else None
    b = _cond_batch(link)
    g = _mt_grad(b, link, tf_ensemble=True, tf_label_prob=1.0, lambda_reg_mt_master=1e-6, lambda_tf_reg=1.0)
    assert abs(float(g)) < 1e-4, g


def test_without_a_link_the_labelled_term_is_a_quarter_weight_copy_of_the_mt_regression():
    b = _cond_batch()
    b['feat'] = b['feat'] + 0.3                               # off target, so there is a gradient to compare
    g_reg = _mt_grad(b, tf_ensemble=False)
    g_both = _mt_grad(b, tf_ensemble=True, tf_label_prob=1.0, lambda_tf_reg=1.0)
    torch.testing.assert_close(g_both - g_reg, 0.25 * g_reg)


def test_a_fully_labelled_unit_skips_the_wt_forward():
    calls = []
    orig = StubModel.forward_partitioned

    def counting(self, micro, pass_type, **kw):
        calls.append((pass_type, len(micro['feat'])))
        return orig(self, micro, pass_type, **kw)
    StubModel.forward_partitioned = counting
    try:
        _mt_grad(_cond_batch(), tf_ensemble=True, tf_label_prob=1.0)
        n_wt_at_one = sum(1 for k, _ in calls if k == 'wt')
        calls.clear()
        _mt_grad(_cond_batch(), tf_ensemble=True, tf_label_prob=0.0)
        n_wt_at_zero = sum(1 for k, _ in calls if k == 'wt')
    finally:
        StubModel.forward_partitioned = orig
    assert n_wt_at_zero == n_wt_at_one + 1, (n_wt_at_one, n_wt_at_zero)


# ---------------- the flags ----------------

def _parse(*extra):
    from esm_msr import config
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', *extra]
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_it_is_off_by_default():
    a = _parse()
    assert a.tf_ensemble is False
    assert a.tf_label_prob == 1.0 and a.tf_label_prob_end is None and a.tf_anneal_steps == 0


@pytest.mark.parametrize('extra', [
    ('--tf_label_prob_end', '0.0'),                                 # nothing to anneal over
    ('--tf_anneal_steps', '100'),                                   # nothing to anneal to
    ('--tf_label_prob', '1.5'),
    ('--tf_label_prob_end', '-0.1', '--tf_anneal_steps', '100'),
])
def test_an_incoherent_schedule_is_refused(extra):
    with pytest.raises(SystemExit):
        _parse(*extra)
