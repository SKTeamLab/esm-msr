"""Unit tests verifying ESM3PredictorBase._get_esm3_outputs per_res_plddt conditioning.

Matches official ESM3.logits() default conditioning (per_res_plddt = 1.0 wherever
structure coordinates are finite, 0.0 for BOS/EOS/missing residues) instead of
ESM3.forward()'s zero-filled default.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from esm_msr.models import ESM3PredictorBase


def _make_stub(use_plddt: bool = False):
    captured: dict = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(sequence_logits=torch.zeros(1, 4, 33))

    stub_esm = SimpleNamespace(model=fake_forward)
    pred = ESM3PredictorBase.__new__(ESM3PredictorBase)
    torch.nn.Module.__init__(pred)
    pred.model = stub_esm
    pred.use_plddt = use_plddt
    return pred, captured


def test_default_per_res_plddt_ones_where_coords_finite():
    pred, cap = _make_stub(use_plddt=False)
    seq = torch.tensor([[0, 5, 6, 2]])
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    # BOS (0) and EOS (3) have NaN coords; residue 1 has partial finite coords; residue 2 has all NaN
    coords = torch.full((1, 4, 3, 3), float('nan'))
    coords[0, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
    # Even if raw B-factors are passed while use_plddt=False, default finite-mask 1.0s are used
    raw_bfac = torch.tensor([[0.0, 92.5, 81.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=raw_bfac)

    plddt = cap['per_res_plddt']
    assert plddt is not None
    assert plddt.shape == (1, 4)
    assert plddt.dtype == torch.float32
    assert torch.equal(plddt, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_default_per_res_plddt_handles_unbatched_3d_coords():
    pred, cap = _make_stub(use_plddt=False)
    seq = torch.tensor([0, 5, 2])
    coords = torch.full((3, 3, 3), float('nan'))
    coords[1] = 1.0

    pred._get_esm3_outputs(seq, coords, None)

    assert cap['structure_coords'].shape == (1, 3, 3, 3)
    assert torch.equal(cap['per_res_plddt'], torch.tensor([[0.0, 1.0, 0.0]]))


def test_sequence_only_passes_none_plddt():
    pred, cap = _make_stub(use_plddt=False)
    seq = torch.tensor([[0, 5, 6, 2]])

    pred._get_esm3_outputs(seq, structure_coords=None, structure_tokens=None)

    assert cap['structure_coords'] is None
    assert cap['per_res_plddt'] is None


def test_use_plddt_true_forwards_caller_plddt():
    pred, cap = _make_stub(use_plddt=True)
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.ones((1, 4, 3, 3))
    custom = torch.tensor([[0.0, 0.95, 0.80, 0.0]])

    pred._get_esm3_outputs(seq, coords, None, per_res_plddt=custom)

    assert torch.allclose(cap['per_res_plddt'], custom)
