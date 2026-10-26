"""Unit tests for the Part IV modeling & inference strategies:
1. Two-knob assay adaptation (`epistasis_weight` in `combine_predictions` and `MSRModel`)
2. Antisymmetric cycle readout for single-adapter (`fused`) mode (`fused_antisymmetric`)
3. Local VQ-VAE structure token masking on `native_cond` scaffolds (`mask_native_cond_structure_batch`)
4. Within-partner / within-pair contrastive ListMLE (`_compute_partner_rank_loss`)
"""
from types import SimpleNamespace
import pytest
import torch
from esm.utils.constants import esm3 as C

from esm_msr.losses import ListMLELoss
from esm_msr.models import CalibrationHead, MSRModel, combine_predictions, mask_native_cond_structure_batch
from esm_msr.training import ESM3EpistasisLightningModule


def test_combine_predictions_epistasis_weight():
    wt = torch.tensor([1.0, -2.0])
    mt = torch.tensor([3.0, 2.0])
    assert torch.allclose(combine_predictions(wt, mt), torch.tensor([2.0, 0.0]))
    assert torch.allclose(combine_predictions(wt, mt, epistasis_weight=0.0), wt)
    assert torch.allclose(combine_predictions(wt, mt, epistasis_weight=1.0), mt)
    assert torch.allclose(combine_predictions(wt, mt, epistasis_weight=0.25), 0.75 * wt + 0.25 * mt)


def test_mask_native_cond_structure_batch():
    batch = {
        'pdb': ['2KT8_G54S', '1A32', '2KYB_G2S'],
        'subset_type': ['native_cond', 'single', 'native_cond'],
        'structure_tokens': torch.full((3, 60), 100, dtype=torch.long),
    }
    orig = batch['structure_tokens'].clone()
    out = mask_native_cond_structure_batch(batch)
    # Input tensor is not mutated in-place
    assert torch.equal(batch['structure_tokens'], orig)
    # Row 0 (2KT8_G54S) has pos 54 masked to C.STRUCTURE_MASK_TOKEN (4096)
    assert int(out['structure_tokens'][0, 54].item()) == int(C.STRUCTURE_MASK_TOKEN)
    assert int(out['structure_tokens'][0, 53].item()) == 100
    # Row 1 (ordinary single) is untouched
    assert torch.equal(out['structure_tokens'][1], orig[1])
    # Row 2 (2KYB_G2S) has pos 2 masked
    assert int(out['structure_tokens'][2, 2].item()) == int(C.STRUCTURE_MASK_TOKEN)


def test_fused_antisymmetric_cancels_bias_and_flips_reverse():
    m = MSRModel.__new__(MSRModel)
    torch.nn.Module.__init__(m)
    m.adapter_mode = 'fused'
    m.fused_antisymmetric = True
    m.epistasis_weight = 0.5
    m.native_cond_mt_input = False
    m.mask_native_cond_struct = False
    m.log_likelihood = False
    m.calibration_head_fused = CalibrationHead(init_scale=0.5, init_bias=0.2, per_mutation=True)
    m.peft_fused = SimpleNamespace(training=False)

    # Synthetic logits [2, 5, 32]
    logits = torch.zeros((2, 5, 32), dtype=torch.float32)
    logits[:, 1, 5] = 2.0   # mt token 5 at pos 1 has logit 2.0, wt token 4 has logit 0.0 -> LLR(w->m) = +2.0
    m._get_esm3_outputs = lambda seq, c, s, p, active_model=None: SimpleNamespace(sequence_logits=logits[:seq.shape[0]])

    batch = {
        'wt_sequence_tokens': torch.zeros((2, 5), dtype=torch.long),
        'mt_sequence_tokens': torch.ones((2, 5), dtype=torch.long),
        'mut_pos': torch.tensor([[1, 0], [1, 2]], dtype=torch.long),
        'wt_id': torch.tensor([[4, 0], [4, 4]], dtype=torch.long),
        'mt_id': torch.tensor([[5, 0], [5, 5]], dtype=torch.long),
        'mut_mask': torch.tensor([[True, False], [True, False]], dtype=torch.bool),
        'coords': None, 'structure_tokens': None, 'plddt': None,
    }
    m.eval()
    wt_out = m.forward_partitioned(batch, pass_type='wt')
    mt_out = m.forward_partitioned(batch, pass_type='mt')

    s = float(m.calibration_head_fused.scale.item())
    b = float(m.calibration_head_fused.bias.item())
    # WT pass: s * 2.0 + b * 1; MT pass (antisymmetric): - (s * (-2.0) + b * 1) = s * 2.0 - b * 1
    assert torch.allclose(wt_out['pred_calibrated'], torch.full((2,), s * 2.0 + b))
    assert torch.allclose(mt_out['pred_calibrated'], torch.full((2,), s * 2.0 - b))
    # Combined mean cancels +b and -b identically
    fb = m.forward_batch(batch)
    assert torch.allclose(fb['combined_pred'], torch.full((2,), s * 2.0))


def test_compute_partner_rank_loss_groups_doubles():
    mod = ESM3EpistasisLightningModule.__new__(ESM3EpistasisLightningModule)
    crit = ListMLELoss()
    B = 8
    pred = torch.linspace(-1.0, 1.0, B, requires_grad=True)
    targets = torch.linspace(-0.5, 1.5, B)
    mask = torch.ones(B, dtype=torch.bool)
    cens = torch.zeros(B, dtype=torch.long)
    micro = {
        'mut_pos': torch.tensor([[2, 5], [2, 6], [2, 5], [3, 7], [2, 6], [3, 7], [4, 0], [5, 0]], dtype=torch.long),
        'mt_id': torch.tensor([[10, 11], [10, 12], [10, 13], [14, 15], [10, 16], [14, 17], [8, 0], [9, 0]], dtype=torch.long),
        'mut_mask': torch.tensor([
            [True, True], [True, True], [True, True], [True, True],
            [True, True], [True, True], [True, False], [True, False],
        ], dtype=torch.bool),
    }
    epi_pred = pred * 0.5
    epi_targ = targets * 0.2
    epi_mask = torch.tensor([True, True, True, True, True, True, False, False])

    loss, val, n_list = mod._compute_partner_rank_loss(
        pred, targets, mask, cens, list_size=4, crit_fn=crit,
        micro=micro, epi_pred=epi_pred, epi_targ=epi_targ, epi_mask=epi_mask
    )
    assert loss is not None and loss.requires_grad
    assert n_list == 2
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
