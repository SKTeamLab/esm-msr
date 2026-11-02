"""``--joint_doubles``: the WT and MT adapters train TOGETHER on the real doubles.

``off`` (every run before this flag): a double's WT half is the measured additive label or the WT head's detached prediction, so the
WT head learns from singles alone. ``both``: the WT half of a double is the WT head's LIVE prediction, ``0.5 * (WT + MT)`` is regressed on
the measured double, and the gradient reaches both adapters. ``alternate``: the same loss, but a double keeps only ONE adapter's graph -
the other half is read under no_grad - and the owner alternates; the owner's loss is doubled so the expected update is ``both``'s.

The stub model is linear in one parameter per adapter (``WT = wp * x_wt``, ``MT = mp * x_mt``), so every gradient below is checked
against a hand-derived one, and it records which activations are alive when each forward starts, which is the memory claim.
"""
import gc
import sys
import types
import weakref

import pytest
import torch

from esm_msr import config
from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M

from _stubs import compose_hparams, with_identity_link

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------------------------------------------- the flag

def test_off_by_default_and_the_choices_are_the_three_modes():
    assert parse().joint_doubles == 'off'
    for m in ('off', 'both', 'alternate'):
        assert parse('--joint_doubles', m).joint_doubles == m
    with pytest.raises(SystemExit):
        parse('--joint_doubles', 'sometimes')


@pytest.mark.parametrize('extra', [('--mt_direct_pred',), ('--adapter_mode', 'fused'), ('--lambda_reg_mt', '0'), ('--subsets', 'single'),
                                   ('--freeze_mt_adapter',)])
def test_a_combination_with_nothing_to_joint_is_refused(extra):
    with pytest.raises(SystemExit):
        parse('--joint_doubles', 'both', *extra)


def test_a_frozen_wt_head_is_refused(tmp_path):
    ck = tmp_path / 'w.ckpt'
    ck.write_bytes(b'x')
    with pytest.raises(SystemExit):
        parse('--joint_doubles', 'alternate', '--init_wt_from_checkpoint', str(ck), '--frozen_wt_mt_only')


def test_it_composes_with_the_per_pass_masks():
    """WT sees WHERE the mutations are (marginal masking), MT sees WHAT they are (no masking): flags that already existed."""
    a = parse('--joint_doubles', 'alternate', '--mask_strategy_wt', 'marginal')
    assert (a.joint_doubles, a.mask_strategy_wt, a.mask_strategy_mt) == ('alternate', 'marginal', None)


# ---------------------------------------------------------------------------------------------------------------- the stub

class _CalHead(torch.nn.Module):
    use_scale = False
    bias = None

    def bias_for(self, n_mut):
        return 0.0


class _Model(torch.nn.Module):
    """WT = wp * x_wt, MT = mp * x_mt (linear in the one parameter of each adapter), and a bookkeeping of what each forward saw."""
    lora_mode = 'ensemble'

    def __init__(self):
        super().__init__()
        self.wp = torch.nn.Parameter(torch.tensor(1.0))
        self.mp = torch.nn.Parameter(torch.tensor(1.0))
        self.calibration_head_wt = _CalHead()
        self.calls, self.refs = [], []

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        gc.collect()
        self.calls.append(dict(pass_type=pass_type, grad=torch.is_grad_enabled(), mask=mask_strategy, n=int(micro['ddG'].shape[0]),
                               alive=[(p, g, r() is not None) for p, g, r in self.refs]))
        w, x = (self.wp, micro['x_wt']) if pass_type == 'wt' else (self.mp, micro['x_mt'])
        h = x.unsqueeze(1).expand(-1, 64) * w              # the stand-in for the saved activations of a real graph
        self.refs.append((pass_type, torch.is_grad_enabled(), weakref.ref(h)))
        out = h.mean(dim=1) + 0.0 * (h * h).sum(dim=1)     # == w * x; the second term only makes the mul save h for backward
        return {'pred_calibrated': out, 'pred_raw': out}


class _HP(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def get(self, k, default=None):
        return dict.get(self, k, default)


def _batch(n_single=0, n_double=4, y=None, labels=True):
    B = n_single + n_double
    f = lambda v: torch.full((B,), float(v))
    mask = torch.zeros(B, 2, dtype=torch.bool)
    mask[:, 0] = True
    mask[n_single:, 1] = True
    g = torch.Generator().manual_seed(0)
    return {
        'ddG': torch.tensor(y if y is not None else [-1.0, 0.5, 2.0, -0.3, 1.2, 0.1, -0.7, 0.9][:B]),
        'x_wt': torch.randn(B, generator=g), 'x_mt': torch.randn(B, generator=g),
        'ddG_additive': f(1.0), 'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long), 'mut_mask': mask,
        'subset_type': ['single'] * n_single + ['double'] * n_double,
        'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': f(1.0),
        'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': f(float('nan')),
        'ddG_A': f(0.5) if labels else f(float('nan')), 'ddG_B': f(0.5) if labels else f(float('nan')),
    }


def _stub(mode='off', mask_wt=None, mask_mt=None, **hp):
    base = compose_hparams(micro_batch_size=4, tf_mix=1.0, joint_doubles=mode, **hp)
    s = types.SimpleNamespace()
    s.hparams = _HP(base)
    s.model = _Model()
    with_identity_link(s)
    s.mask_wt, s.mask_mt = mask_wt, mask_mt
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_reg_loss', '_forward_rows', '_joint_reg'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    s._joint_owners = M._joint_owners
    s._micro_batch = M._micro_batch
    return s


def _grads(stub, batch):
    logs = M._compose_losses_streaming_and_backward(stub, batch)
    g = lambda p: None if p.grad is None else p.grad.clone()
    return logs, g(stub.model.wp), g(stub.model.mp)


def _analytic(batch):
    """Gradients of  sum_i (0.5 (wp x_wt + mp x_mt) - y)^2 / B  at wp = mp = 1, over the doubles."""
    d = batch['mut_mask'].sum(1) == 2
    xw, xm, y = batch['x_wt'][d], batch['x_mt'][d], batch['ddG'][d]
    r = 0.5 * (xw + xm) - y
    B = batch['ddG'].shape[0]
    return (r * xw).sum() / B, (r * xm).sum() / B, (r ** 2).sum() / B


# ---------------------------------------------------------------------------------------------------------------- the loss

def test_both_regresses_the_live_wt_half_and_ignores_the_additive_label():
    batch = _batch(labels=True)                                    # tf_mix 1.0 and a usable label: off would use the label
    logs, gw, gm = _grads(_stub('both', lambda_reg_wt=0.0), batch)
    g_wt, g_mt, loss = _analytic(batch)
    assert gw.item() == pytest.approx(g_wt.item(), abs=1e-6) and gm.item() == pytest.approx(g_mt.item(), abs=1e-6)
    assert logs['L_reg_joint'] == pytest.approx(loss.item() * 4 / 4, abs=1e-6)
    assert logs['L_reg_combined'] == pytest.approx(logs['L_reg_joint'])


def test_off_never_sends_a_double_s_gradient_to_the_wt_head():
    _, gw, gm = _grads(_stub('off', lambda_reg_wt=0.0), _batch(labels=True))
    assert gw is None                                              # the WT graph is never in a double's loss
    assert abs(gm.item()) > 0


def test_alternate_averaged_over_the_two_owners_is_both():
    """Each double is owned by WT on one turn and by MT on the other; the owner's loss is doubled. So the mean of the two turns' gradients
    is the gradient of 'both' - the claim that makes alternating an unbiased replacement."""
    batch = _batch()
    _, bw, bm = _grads(_stub('both', lambda_reg_wt=0.0), batch)
    turns = []
    for t in (0, 1):
        s = _stub('alternate', lambda_reg_wt=0.0)
        s._joint_turn = t
        turns.append(_grads(s, batch))
    assert torch.allclose((turns[0][1] + turns[1][1]) / 2, bw, atol=1e-6)
    assert torch.allclose((turns[0][2] + turns[1][2]) / 2, bm, atol=1e-6)
    assert not torch.allclose(turns[0][1], turns[1][1])            # the two turns really are different updates


def test_alternate_gives_each_adapter_a_share_of_the_doubles_in_one_micro_batch():
    s = _stub('alternate', lambda_reg_wt=0.0)
    logs, gw, gm = _grads(s, _batch())
    assert abs(gw.item()) > 0 and abs(gm.item()) > 0               # both adapters trained from one micro-batch of four doubles
    own_w, own_m = M._joint_owners(torch.tensor([True, True, True, True]), 'alternate', 0)
    assert own_w.tolist() == [True, False, True, False] and own_m.tolist() == [False, True, False, True]
    own_w, _ = M._joint_owners(torch.tensor([False, True, True, True]), 'alternate', 1)
    assert own_w.tolist() == [False, False, True, False]            # counted among the joint rows only, and the start flips with the turn


def test_singles_are_untouched_and_a_singles_only_batch_is_the_off_batch():
    batch = _batch(n_single=4, n_double=0)
    _, w_off, m_off = _grads(_stub('off', lambda_reg_wt=0.0), batch)
    for mode in ('both', 'alternate'):
        _, w, m = _grads(_stub(mode, lambda_reg_wt=0.0), batch)
        assert (w is None and w_off is None) or torch.equal(w, w_off)
        assert torch.equal(m, m_off), mode


def test_a_double_with_no_dg_wt_is_not_a_joint_double():
    """Without a measured dG_wt the double cannot be put on the observed scale: it leaves the linked regression, joint or not."""
    batch = _batch()
    batch['dG_wt'][2] = float('nan')
    logs, _, _ = _grads(_stub('both', lambda_reg_wt=0.0), batch)
    keep = [0, 1, 3]
    want = ((0.5 * (batch['x_wt'][keep] + batch['x_mt'][keep]) - batch['ddG'][keep]) ** 2).sum().item() / len(keep)
    assert logs['L_reg_joint'] == pytest.approx(want, abs=1e-6)


def test_a_frozen_path_falls_back_to_the_detached_wt_half():
    s = _stub('both', lambda_reg_wt=0.0)
    s.peft_manager.wt_path_is_frozen = True
    logs, _, gm = _grads(s, _batch(labels=False))
    assert 'L_reg_joint' not in logs and s.model.wp.grad is None and abs(gm.item()) > 0


# ---------------------------------------------------------------------------------------------------------------- memory and masks

def _alive_at_each_forward(mode):
    s = _stub(mode, lambda_reg_wt=0.0, mask_wt='marginal', mask_mt=None)
    s._joint_turn = 0
    M._compose_losses_streaming_and_backward(s, _batch(n_single=2, n_double=2))
    return s.model.calls


def test_alternate_never_holds_the_wt_and_mt_graphs_together_but_both_does():
    def wt_graph_alive_at_mt_grad_forward(calls):
        return [any(p == 'wt' and g and a for p, g, a in c['alive']) for c in calls if c['pass_type'] == 'mt' and c['grad']]
    assert not any(wt_graph_alive_at_mt_grad_forward(_alive_at_each_forward('alternate')))
    assert all(wt_graph_alive_at_mt_grad_forward(_alive_at_each_forward('both')))


def test_alternate_reads_the_half_it_does_not_own_under_no_grad():
    calls = _alive_at_each_forward('alternate')
    nog = [c for c in calls if not c['grad']]
    assert {c['pass_type'] for c in nog} == {'mt', 'wt'} and all(c['n'] < 4 for c in nog)
    assert all(c['n'] <= 3 for c in calls if c['grad'])           # no pass builds a graph over the whole micro-batch


def test_each_pass_keeps_its_own_masking_in_every_forward():
    for mode in ('both', 'alternate'):
        for c in _alive_at_each_forward(mode):
            assert c['mask'] == ('marginal' if c['pass_type'] == 'wt' else None), (mode, c)
