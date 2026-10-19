"""``--nonspecific_term fx``: nonspecific epistasis f(x) as its own learned term (port of the archived
main-baseline commit 70be32e onto this branch).

f is a function of a multi-mutant's ADDITIVE expectation (dG_wt + the additive latent, detached), added to its
combined latent before the link and never to a single, so the link keeps modelling the readout alone. Three
properties matter and are tested here: the interpolation itself, that the shift lands on doubles only (and on
the MT readout twice, so comb = (WT + MT) / 2 holds), and that f neither trains the WT half nor is trained by
it - only the combined losses give it gradient.
"""
import inspect
import sys
import types

import pytest
import torch

from esm_msr import config
from esm_msr.link import NonspecificTerm
from esm_msr.training import ESM3EpistasisLightningModule as M


# ----------------------------------------------------------------------------------------------------------------
# the interpolation
# ----------------------------------------------------------------------------------------------------------------

def test_piecewise_linear_and_constant_beyond_the_knots():
    f = NonspecificTerm(lo=-1.0, hi=1.0, step=1.0)
    with torch.no_grad():
        f.values.copy_(torch.tensor([0.4, 0.2, 0.0]))
    x = torch.tensor([-5.0, -1.0, -0.5, 0.5, 1.0, 7.0])
    torch.testing.assert_close(f(x), torch.tensor([0.4, 0.4, 0.3, 0.1, 0.0, 0.0]))


def test_default_table_is_zero_and_serialisable():
    """Initialised at 0 (so the term starts as a no-op) and the table is plain JSON - it is written into the val dumps."""
    import json
    f = NonspecificTerm()
    assert len(f.values) == int(round((5.0 - (-3.0)) / 0.5)) + 1 == 17
    assert torch.all(f.values == 0)
    t = f.table()
    assert t['knots'][0] == -3.0 and t['knots'][-1] == 5.0 and len(t['knots']) == len(t['values'])
    assert json.loads(json.dumps(t)) == t
    on = {float(k): v for k, v in zip(t['knots'], t['values'])}
    assert on[-1.0] == 0.0


def test_a_trainable_parameter_with_gradient():
    f = NonspecificTerm()
    f(torch.tensor([0.0, 1.0])).sum().backward()
    assert f.values.grad is not None and float(f.values.grad.abs().sum()) > 0.0


# ----------------------------------------------------------------------------------------------------------------
# the validation shift
# ----------------------------------------------------------------------------------------------------------------

class _Head(torch.nn.Module):
    def __init__(self, bias=None, scale=None):
        super().__init__()
        self.use_scale = scale is not None
        self.bias = None if bias is None else torch.nn.Parameter(torch.tensor(float(bias)))
        if scale is not None:
            self.scale = torch.nn.Parameter(torch.tensor(float(scale)))


class _Model(torch.nn.Module):
    lora_mode = 'corrector'

    def __init__(self, head=None):
        super().__init__()
        if head is not None:
            self.calibration_head_wt = head


class _HP(dict):
    """hparams is read-only on LightningModule; this stub mirrors AttributeDict, which RAISES on a missing key."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


    def get(self, k, default=None):
        return dict.get(self, k, default)


def _ns_stub(f, per_mutation_bias=False, head=None):
    s = types.SimpleNamespace()
    s.ns_head = f
    s.model = _Model(head)
    s.hparams = _HP(per_mutation_bias=per_mutation_bias)
    return s


def _ids():
    # item 0 is a double (two mutated positions), item 1 a single
    return torch.tensor([[True, True], [True, False]])


def _out_dict():
    return {'wt_lora_pred': torch.tensor([1.0, 1.0]), 'mt_lora_pred': torch.tensor([0.5, 0.5]),
            'combined_pred': torch.tensor([0.75, 0.75]), 'epi_pred': torch.tensor([0.25, 0.25])}


def _apply(stub, out=None, dG_wt=(2.0, 2.0)):
    out = _out_dict() if out is None else out
    batch = {'mut_mask': _ids()}
    res = M._apply_nonspecific(stub, out, batch, torch.tensor(dG_wt))
    return out, res


def test_comb_gains_f_and_the_mt_readout_twice_f():
    f = NonspecificTerm()
    with torch.no_grad():
        f.values.fill_(0.3)
    before, after = _apply(_ns_stub(f))
    # double: comb += f, mt += 2f, epi += f
    assert float(after['combined_pred'][0]) == pytest.approx(0.75 + 0.3, abs=1e-6)
    assert float(after['mt_lora_pred'][0]) == pytest.approx(0.5 + 0.6, abs=1e-6)
    assert float(after['epi_pred'][0]) == pytest.approx(0.25 + 0.3, abs=1e-6)
    # single: untouched
    for k in ('combined_pred', 'mt_lora_pred', 'epi_pred'):
        assert float(after[k][1]) == pytest.approx(float(before[k][1]), abs=1e-6)
    # comb = (WT + MT) / 2 still holds on the adjusted values
    assert float(after['combined_pred'][0]) == pytest.approx(0.5 * (float(after['wt_lora_pred'][0]) + float(after['mt_lora_pred'][0])), abs=1e-6)
    # the input dict is not modified in place
    assert float(before['combined_pred'][0]) == pytest.approx(0.75, abs=1e-6)


def test_item_without_dG_wt_is_left_alone():
    f = NonspecificTerm()
    with torch.no_grad():
        f.values.fill_(0.3)
    before, after = _apply(_ns_stub(f), dG_wt=(float('nan'), 2.0))
    assert float(after['combined_pred'][0]) == pytest.approx(float(before['combined_pred'][0]), abs=1e-6)
    assert float(after['combined_pred'][1]) == pytest.approx(float(before['combined_pred'][1]), abs=1e-6)


def test_x_is_the_additive_expectation_and_the_bias_counts_once_per_extra_mutation():
    """With f(x) = x the shift IS x, so the test reads x directly: dG_wt (2.0) + the WT pass (1.0) + one bias per
    EXTRA mutation (a double has two, the WT pass carries the bias once) - and nothing extra under --per_mutation_bias,
    which already put bias * n_mut into the WT pass."""
    f = NonspecificTerm(lo=-10.0, hi=10.0, step=1.0)
    with torch.no_grad():
        f.values.copy_(f.knots)
    _, no_bias = _apply(_ns_stub(f))
    assert float(no_bias['combined_pred'][0]) == pytest.approx(0.75 + 3.0, abs=1e-5)

    head = _Head(bias=0.5)
    _, with_bias = _apply(_ns_stub(f, per_mutation_bias=False, head=head))
    assert float(with_bias['combined_pred'][0]) == pytest.approx(0.75 + 3.5, abs=1e-5)

    _, already = _apply(_ns_stub(f, per_mutation_bias=True, head=head))
    assert float(already['combined_pred'][0]) == pytest.approx(0.75 + 3.0, abs=1e-5)


def test_no_term_means_no_change():
    out = _out_dict()
    stub = _ns_stub(None)
    assert M._apply_nonspecific(stub, out, {'mut_mask': _ids()}, torch.tensor([2.0, 2.0])) is out


# ----------------------------------------------------------------------------------------------------------------
# the training step: f learns from doubles, never trains the WT half
# ----------------------------------------------------------------------------------------------------------------

class _StepModel(torch.nn.Module):
    """Minimal stand-in for MSRModel: the additive expectation of a double is its ddG_additive label."""
    lora_mode = 'corrector'

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        i = 0 if pass_type == 'wt' else 1
        add = micro.get('ddG_additive')
        base = torch.where(torch.isfinite(add), add, micro['ddG'].float()) if add is not None else micro['ddG'].float()
        pred = base * self.w[i]
        return {'pred_calibrated': pred, 'pred_raw': pred}


def _batch(doubles):
    from esm_msr.losses import ListMLELoss          # noqa: F401  (kept local: only the stub below needs it)
    B = 6
    f = lambda v: torch.full((B,), float(v))
    if doubles:
        mask = torch.ones(B, 2, dtype=torch.bool)
        add, subset = torch.linspace(-1.0, 1.0, B), ['double'] * B
    else:
        mask = torch.ones(B, 1, dtype=torch.bool)
        add, subset = f(float('nan')), ['single'] * B
    return {
        'ddG': torch.linspace(-3.0, 1.0, B), 'ddG_additive': add,
        'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long), 'mut_mask': mask, 'subset_type': subset,
        'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': torch.linspace(-1.0, 3.0, B),
        'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': f(float('nan')),
        'ddG_A': f(float('nan')), 'ddG_B': f(float('nan')),
    }


def _step_stub(ns, **hp):
    from esm_msr.losses import ListMLELoss
    base = dict(subset_size=4, micro_batch_size=3, micro_batch_tokens=0, double_weight=1.0, cond_weight=1.0,
                native_cond_weight=1.0, wt_double_target='latent', mt_reg_mask='all', mask_strategy=None,
                lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0)
    base.update(hp)
    s = types.SimpleNamespace()
    s.hparams = _HP(base)
    s.model = _StepModel()
    s.ns_head = ns
    s.link_head = None
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_reg_loss'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    s._micro_batch = M._micro_batch
    return s


def test_f_learns_from_doubles():
    f = NonspecificTerm()
    stub = _step_stub(f)
    M._compose_losses_streaming_and_backward(stub, _batch(doubles=True))
    assert f.values.grad is not None and float(f.values.grad.abs().sum()) > 0.0


def test_a_singles_only_batch_gives_f_nothing():
    f = NonspecificTerm()
    stub = _step_stub(f)
    M._compose_losses_streaming_and_backward(stub, _batch(doubles=False))
    assert f.values.grad is None or float(f.values.grad.abs().sum()) == 0.0


def test_the_term_does_not_train_the_wt_half():
    """The WT half is detached (and x is detached), so the WT adapter's gradient must be bit-identical whether the
    term is present or not."""
    grads = []
    for ns in (None, NonspecificTerm()):
        stub = _step_stub(ns, lambda_reg_wt=1.0, lambda_rank_wt=1.0)
        M._compose_losses_streaming_and_backward(stub, _batch(doubles=True))
        grads.append(stub.model.w.grad.detach().clone())
    torch.testing.assert_close(grads[0], grads[1])


# ----------------------------------------------------------------------------------------------------------------
# wiring: flag, checkpoint, optimizer
# ----------------------------------------------------------------------------------------------------------------

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def _parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_flag_defaults_off_and_is_accepted():
    args = _parse()
    assert args.nonspecific_term == 'none'
    args = _parse('--nonspecific_term', 'fx')
    assert args.nonspecific_term == 'fx'
    assert args.ns_knot_lo == -3.0 and args.ns_knot_hi == 5.0 and args.ns_knot_step == 0.5 and args.ns_lr == 5e-3


def test_fx_without_the_link_is_refused():
    with pytest.raises(SystemExit):
        _parse('--no-link', '--nonspecific_term', 'fx')


def test_the_checkpoint_keeps_the_term():
    """Its f table defines the predictions the run was scored on, so it has to survive the checkpoint filter."""
    src = inspect.getsource(M.on_save_checkpoint)
    assert 'ns_head' in src


def test_the_optimizer_gives_the_term_its_own_group():
    src = inspect.getsource(M.configure_optimizers)
    assert '"name": "nonspecific"' in src and '"nonspecific"' in src
