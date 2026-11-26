"""Unit tests verifying opt-in per_res_plddt=1.0 conditioning in ESM3PredictorBase._get_esm3_outputs.

Default behavior (`default_plddt_one=False`, `use_plddt=False`) passes `per_res_plddt=None`
to preserve exact manuscript `FINAL_p0` conditioning (ESM3.forward zero-filled default).
Opt-in `default_plddt_one=True` passes `1.0` wherever structure coordinates are finite
(matching official `ESM3.logits()` conditioning, `0.0` for BOS/EOS/missing residues).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import yaml

from esm_msr.config import parse_arguments
from esm_msr.inference import parse_hparams_to_lora_config
from esm_msr.models import ESM3PredictorBase


def _make_stub(use_plddt: bool = False, default_plddt_one: bool = False):
    captured: dict = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(sequence_logits=torch.zeros(1, 4, 33))

    stub_esm = SimpleNamespace(model=fake_forward)
    pred = ESM3PredictorBase.__new__(ESM3PredictorBase)
    torch.nn.Module.__init__(pred)
    pred.model = stub_esm
    pred.use_plddt = use_plddt
    pred.default_plddt_one = default_plddt_one
    return pred, captured


def test_default_passes_none_plddt_for_final_p0_parity():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=False)
    seq = torch.tensor([[0, 5, 6, 2]])
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    coords = torch.ones((1, 4, 3, 3))
    raw_bfac = torch.tensor([[0.0, 92.5, 81.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=raw_bfac)

    assert cap['per_res_plddt'] is None


def test_opt_in_default_plddt_one_sets_ones_where_coords_finite():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=True)
    seq = torch.tensor([[0, 5, 6, 2]])
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    # BOS (0) and EOS (3) have NaN coords; residue 1 has partial finite coords; residue 2 has all NaN
    coords = torch.full((1, 4, 3, 3), float('nan'))
    coords[0, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
    raw_bfac = torch.tensor([[0.0, 92.5, 81.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=raw_bfac)

    plddt = cap['per_res_plddt']
    assert plddt is not None
    assert plddt.shape == (1, 4)
    assert plddt.dtype == torch.float32
    assert torch.equal(plddt, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_opt_in_default_plddt_one_handles_unbatched_3d_coords():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=True)
    seq = torch.tensor([0, 5, 2])
    coords = torch.full((3, 3, 3), float('nan'))
    coords[1] = 1.0

    pred._get_esm3_outputs(seq, coords, None)

    assert cap['structure_coords'].shape == (1, 3, 3, 3)
    assert torch.equal(cap['per_res_plddt'], torch.tensor([[0.0, 1.0, 0.0]]))


def test_sequence_only_passes_none_plddt_even_when_opted_in():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=True)
    seq = torch.tensor([[0, 5, 6, 2]])

    pred._get_esm3_outputs(seq, structure_coords=None, structure_tokens=None)

    assert cap['structure_coords'] is None
    assert cap['per_res_plddt'] is None


def test_use_plddt_true_forwards_caller_plddt():
    pred, cap = _make_stub(use_plddt=True, default_plddt_one=True)
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.ones((1, 4, 3, 3))
    custom = torch.tensor([[0.0, 0.95, 0.80, 0.0]])

    pred._get_esm3_outputs(seq, coords, None, per_res_plddt=custom)

    assert torch.allclose(cap['per_res_plddt'], custom)


def test_training_config_and_hparams_default_plddt_one_is_opt_in(tmp_path):
    args_def = parse_arguments(['--experiment_name', 'unit_test'])
    assert args_def.default_plddt_one is False
    args_on = parse_arguments(['--experiment_name', 'unit_test', '--default_plddt_one'])
    assert args_on.default_plddt_one is True

    hp_old = tmp_path / 'hparams_old.yaml'
    hp_old.write_text(yaml.safe_dump({'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
                                      'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1}))
    assert parse_hparams_to_lora_config(str(hp_old))['default_plddt_one'] is False

    hp_new = tmp_path / 'hparams_new.yaml'
    hp_new.write_text(yaml.safe_dump({'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
                                      'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1,
                                      'default_plddt_one': True}))
    assert parse_hparams_to_lora_config(str(hp_new))['default_plddt_one'] is True
