"""A pass with no loss term runs no backward, so manual_backward never frees its activation graph, which then stays alive
through the loop's prediction variables while the NEXT micro-batch's forward allocates a second copy: peak memory doubles.
Ported from devel-revision (5c5e9e9; devel acf6026, "a lossless unit pinned ~20 GB").

The stub keeps a weak reference to each forward's intermediate activation and asserts that none from an earlier micro-batch
is alive when the next micro-batch's WT forward starts. Within one micro-batch the WT activation is alive during the MT forward
by design (the ensemble consumes wt_pred_cal), so only the boundary is checked.
"""
import gc
import types
import unittest
import weakref

import torch

from esm_msr.losses import ListMLELoss


class CalHead(torch.nn.Module):
    use_scale, bias = False, None


class TrackingModel(torch.nn.Module):
    lora_mode = 'ensemble'

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.calibration_head_wt = CalHead()
        self.refs, self.alive_at_forward = [], []

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, detach_calibration=False):
        gc.collect()
        self.alive_at_forward.append((pass_type, [r() is not None for r in self.refs]))
        h = micro['ddG'].float().unsqueeze(1).expand(-1, 64) * self.w[0 if pass_type == 'wt' else 1]
        self.refs.append(weakref.ref(h))
        raw = (h * h).sum(dim=1) / 64.0        # mul saves h for backward, so h lives exactly as long as the graph
        return {'pred_calibrated': raw, 'pred_raw': raw}


def make_batch(n=32):
    return {'ddG': torch.linspace(-3.0, 1.0, n), 'mut_mask': torch.ones(n, 1, dtype=torch.bool), 'row': torch.arange(n),
            'valid_dddG_mask': torch.zeros(n, dtype=torch.bool), 'ddG_additive': torch.full((n,), float('nan')),
            'dddG': torch.full((n,), float('nan')), 'subset_type': ['single'] * n, 'dG_wt': torch.full((n,), 2.0),
            'mutations': [[('A', i, 'G')] for i in range(n)]}


def run(wt_frozen=False, **lambdas):
    from esm_msr import training
    cls = training.ESM3EpistasisLightningModule
    hp = dict(subset_size=16, micro_batch_size=16, mt_reg_mask='all', zero_epistasis_for_singles=True, detach_ensemble_input=True,
              mask_strategy=None, detach_regression=False, lambda_reg_wt=1.0, lambda_rank_wt=1.0, lambda_reg_combined=1.0,
              lambda_rank_combined=1.0, lambda_epi_combined=0.0, double_weight=1.0, mut_ctx_weight=0.5, reversion_weight=0.5)
    hp.update(lambdas)
    stub = types.SimpleNamespace(hparams=types.SimpleNamespace(**hp), model=TrackingModel(),
                                 peft_manager=types.SimpleNamespace(wt_path_is_frozen=wt_frozen, mt_path_is_frozen=False),
                                 crit_reg=torch.nn.MSELoss(reduction='none'), crit_rank_wt=ListMLELoss(), crit_rank_combined=ListMLELoss())
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    stub._compute_rank_loss = types.MethodType(cls._compute_rank_loss, stub)
    cls._compose_losses_streaming_and_backward(stub, make_batch())
    return stub.model.alive_at_forward


class TestGraphRelease(unittest.TestCase):
    def assert_released(self, calls):
        wt_forwards = [k for k, (kind, _) in enumerate(calls) if kind == 'wt']
        self.assertGreaterEqual(len(wt_forwards), 2, "need at least two micro-batches")
        for k in wt_forwards[1:]:
            self.assertFalse(any(calls[k][1]), f"the micro-batch starting at forward #{k} began with an earlier graph still alive")

    def test_a_lossless_wt_pass_does_not_pin_its_graph(self):
        self.assert_released(run(lambda_reg_wt=0.0, lambda_rank_wt=0.0))

    def test_a_frozen_wt_path_does_not_pin_its_graph(self):
        self.assert_released(run(wt_frozen=True))

    def test_still_released_when_every_pass_does_backward(self):
        self.assert_released(run())


if __name__ == '__main__':
    unittest.main()
