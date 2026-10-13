"""--micro_batch_size changes speed and memory only: the losses and every gradient are the same for micro-batches of 16 (one ranking
list, the released setting) and 64, including WT lists with masked items (doubles without both singles measured) and lists with fewer
than two valid items. Holds with and without --link."""
import types
import unittest

import numpy as np
import torch

from esm_msr.link import MonotoneLink
from esm_msr.losses import ListMLELoss


class CalHead(torch.nn.Module):
    use_scale, bias = False, None


class PerItemModel(torch.nn.Module):
    """pred = feature * w[pass] + theta[pass, item]: one free parameter per item and pass, so any reweighting of an item shows."""
    lora_mode = 'ensemble'

    def __init__(self, n):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.theta = torch.nn.Parameter(torch.zeros(2, n))
        self.calibration_head_wt = CalHead()

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, detach_calibration=False):
        i = 0 if pass_type == 'wt' else 1
        v = micro['feat'] * self.w[i] + self.theta[i, micro['row']]
        return {'pred_calibrated': v, 'pred_raw': v}


def make_batch(n=64, seed=0):
    rng = np.random.default_rng(seed)
    n_mut = rng.choice([1, 2], n, p=[0.6, 0.4])
    ddG = rng.normal(-1, 1.2, n)
    add = np.where(n_mut == 2, ddG + rng.normal(0, 0.5, n), np.nan)
    add[(n_mut == 2) & (rng.random(n) < 0.4)] = np.nan              # doubles without both singles: masked in the WT lists
    add[48:63] = np.nan; n_mut[48:63] = 2                             # a list with a single valid WT item
    mm = torch.zeros(n, 2, dtype=torch.bool)
    for i, k in enumerate(n_mut):
        mm[i, :k] = True
    return {'ddG': torch.tensor(ddG, dtype=torch.float32), 'feat': torch.tensor(ddG + rng.normal(0, 0.8, n), dtype=torch.float32),
            'row': torch.arange(n), 'mut_mask': mm, 'valid_dddG_mask': torch.tensor(n_mut == 2),
            'ddG_additive': torch.tensor(add, dtype=torch.float32), 'dddG': torch.full((n,), float('nan')),
            'subset_type': ['single' if k == 1 else 'double' for k in n_mut], 'dG_wt': torch.full((n,), 2.0),
            'mutations': [[('A', i, 'G')] * k for i, k in enumerate(n_mut)]}


def run(batch, micro, link=None):
    from esm_msr import training
    cls = training.ESM3EpistasisLightningModule
    hp = types.SimpleNamespace(subset_size=16, micro_batch_size=micro, mt_reg_mask='all', zero_epistasis_for_singles=True,
                               detach_ensemble_input=True, mask_strategy=None, detach_regression=False, lambda_reg_wt=1.0, lambda_rank_wt=1.0,
                               lambda_reg_combined=1.0, lambda_rank_combined=1.0, lambda_epi_combined=0.0, double_weight=1.0,
                               mut_ctx_weight=0.5, reversion_weight=0.5)
    stub = types.SimpleNamespace(hparams=hp, model=PerItemModel(len(batch['ddG'])), link_head=link,
                                 peft_manager=types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False),
                                 crit_reg=torch.nn.MSELoss(reduction='none'), crit_rank_wt=ListMLELoss(), crit_rank_combined=ListMLELoss())
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    stub._compute_rank_loss = types.MethodType(cls._compute_rank_loss, stub)
    cls._compose_losses_streaming_and_backward(stub, batch)
    return stub


class TestMicroBatchInvariance(unittest.TestCase):
    def _check(self, link_fn):
        batch = make_batch()
        a, b = run(batch, 16, link_fn()), run(batch, 64, link_fn())
        torch.testing.assert_close(a.model.theta.grad, b.model.theta.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a.model.w.grad, b.model.w.grad, atol=1e-6, rtol=1e-5)
        self.assertGreater(float(a.model.theta.grad[0].abs().sum()), 0.0)

    def test_16_and_64_give_the_same_gradients(self):
        self._check(lambda: None)

    def test_16_and_64_give_the_same_gradients_through_the_link(self):
        self._check(lambda: MonotoneLink())


if __name__ == '__main__':
    unittest.main()
