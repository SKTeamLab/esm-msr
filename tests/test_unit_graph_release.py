"""A micro-batch with no loss term runs no backward, so its activation graph is not freed by manual_backward and
stays alive through the loop's prediction variables - while the NEXT micro-batch's forward allocates a second copy.
Peak memory doubles. Fixed on devel as acf6026 ("a lossless unit pinned ~20 GB"); this is the same guard for the
loop in this branch.

The stub keeps a weak reference to each forward's intermediate activation and asserts the previous one is gone by
the time the next forward starts.
"""
import gc
import types
import weakref

import pytest
import torch

from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M


class _HP(dict):
    """hparams is a read-only property on LightningModule, so the stub carries a plain attribute-dict."""
    __getattr__ = dict.get

    def get(self, k, default=None):
        return dict.get(self, k, default)


class TrackingStub(torch.nn.Module):
    lora_mode = 'corrector'          # avoids the teacher-forcing path, which needs a calibration head

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.refs, self.alive_at_forward = [], []

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        gc.collect()
        self.alive_at_forward.append((pass_type, [r() is not None for r in self.refs]))
        i = 0 if pass_type == 'wt' else 1
        # a wide intermediate standing in for the activations a real graph saves for backward
        h = micro['ddG'].float().unsqueeze(1).expand(-1, 64) * self.w[i]
        self.refs.append(weakref.ref(h))
        raw = (h * h).sum(dim=1) / 64.0            # mul saves h for backward, so h lives exactly as long as the graph
        return {'pred_calibrated': raw, 'pred_raw': raw}


def _batch(B=8):
    f = lambda v: torch.full((B,), float(v))
    return {
        'ddG': torch.linspace(-3.0, 1.0, B),
        'ddG_additive': f(float('nan')),
        'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long),
        'mut_mask': torch.ones(B, 1, dtype=torch.bool),       # all singles
        'subset_type': ['single'] * B,
        'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': torch.linspace(-1.0, 3.0, B),
        'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': f(float('nan')),
        'ddG_A': f(float('nan')), 'ddG_B': f(float('nan')),
    }


def _stub(**hp):
    base = dict(subset_size=4, micro_batch_size=4, micro_batch_tokens=0,
                double_weight=1.0, cond_weight=1.0, native_cond_weight=1.0,
                wt_double_target='latent', mt_reg_mask='all', mask_strategy=None,
                lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0)
    base.update(hp)
    s = types.SimpleNamespace()
    s.hparams = _HP(base)
    s.model = TrackingStub()
    s.link_head = None
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_reg_loss'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    s._micro_batch = M._micro_batch
    return s


def _run(stub):
    M._compose_losses_streaming_and_backward(stub, _batch())
    calls = stub.model.alive_at_forward
    assert len(calls) >= 4, f"expected at least two micro-batches of two passes, got {len(calls)}"
    return calls


def _assert_released(calls):
    """Checked at the start of each new micro-batch, i.e. at every WT forward after the first.

    Within one micro-batch the WT activation is alive during the MT forward BY DESIGN - the ensemble consumes
    wt_pred_cal - so only the boundary between micro-batches is a leak.
    """
    wt_forwards = [k for k, (kind, _) in enumerate(calls) if kind == 'wt']
    assert len(wt_forwards) >= 2, f"need at least two micro-batches, got {len(wt_forwards)}"
    for k in wt_forwards[1:]:
        kind, alive = calls[k]
        assert not any(alive), (f"micro-batch starting at forward #{k} began with {sum(alive)} activation(s) from "
                                f"an earlier micro-batch still alive")


def test_a_lossless_wt_pass_does_not_pin_its_graph_through_later_forwards():
    """With both WT lambdas at 0 the WT pass produces no loss and runs no backward, which is the exact case where
    nothing else frees its graph."""
    _assert_released(_run(_stub()))


def test_still_released_when_every_pass_does_backward():
    """The ordinary case, which manual_backward already handles - checked so the guard cannot regress it."""
    _assert_released(_run(_stub(lambda_reg_wt=1.0, lambda_rank_wt=1.0)))


def test_released_when_the_wt_path_is_frozen():
    """Frozen WT skips its backward for the whole run, so without the guard every unit leaks one graph."""
    stub = _stub(lambda_reg_wt=1.0, lambda_rank_wt=1.0)
    stub.peft_manager.wt_path_is_frozen = True
    _assert_released(_run(stub))
