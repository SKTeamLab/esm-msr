"""A work unit with no loss term runs no backward, so its activation graph used to stay alive (via the loop's prediction variables)
while the NEXT unit's forward allocated a second copy: peak memory doubled, giving OOM / driver errors with pair-group batches.

The stub keeps a weak reference to each forward's intermediate activation and asserts that the previous one is gone when the
next forward starts.
"""
import gc
import types
import unittest
import weakref

import torch

from tests.test_compose_censoring import make_hp, make_batch
from esm_msr.losses import ListMLELoss


class TrackingStub(torch.nn.Module):
    dedup_backbone = False

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.refs, self.alive_at_forward = [], []

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        gc.collect()
        self.alive_at_forward.append((pass_type, [r() is not None for r in self.refs]))
        i = 0 if pass_type == 'wt' else 1
        h = micro['feat'].unsqueeze(1).expand(-1, 64) * self.w[i]       # the "activation" the graph would keep for backward
        self.refs.append(weakref.ref(h))
        raw = (h * h).sum(dim=1) / 64.0                       # mul saves h for backward, so h lives exactly as long as the graph
        return {'pred_calibrated': raw, 'pred_raw': raw}


class TestUnitGraphRelease(unittest.TestCase):
    def test_lossless_unit_does_not_pin_its_graph_through_the_next_forward(self):
        from esm_msr import training
        cls = next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_compose_losses_streaming_and_backward'))
        n_single = 0
        batch = make_batch(n_single=n_single, with_cens=False)
        B = 6
        # first four rows (the first MT unit at micro_batch_size 4): dead cond items with no usable bound and no flip column -> no loss
        batch['cens'][:4] = -1
        batch['cens_bound'][:4] = float('nan')
        batch['cens_src'][:4] = 1                      # assay-range censoring: no ordinary regression, and no hinge without a finite bound
        batch['flip_key'] = [''] * B
        stub = types.SimpleNamespace()
        stub.hparams = make_hp(micro_batch_size=4, include_out_of_range=True, lambda_mt_colrank=0.0, lambda_rank_wt=0.0)
        stub.model = TrackingStub()
        stub.link_head = None
        stub.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
        stub.crit_reg = torch.nn.MSELoss(reduction='none')
        stub.crit_rank_wt, stub.crit_rank_mt = ListMLELoss(), ListMLELoss()
        stub._warned_unrouted = True
        stub.global_step = 0
        stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
        for name in ('_compute_rank_loss', '_compute_flip_loss', '_compute_block_components', '_aligned_chunks', '_subset_weights', '_plan_units'):
            setattr(stub, name, types.MethodType(getattr(cls, name), stub))
        cls._compose_losses_streaming_and_backward(stub, batch)
        calls = stub.model.alive_at_forward
        self.assertGreaterEqual(len(calls), 2, 'the scenario must produce at least two MT units')
        for k, (kind, alive) in enumerate(calls[1:], start=1):
            self.assertFalse(any(alive), f"forward #{k} ({kind}) started while an earlier unit's activation was still alive")


if __name__ == '__main__':
    unittest.main()
