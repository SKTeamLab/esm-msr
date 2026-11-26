"""Unit tests verifying opt-in --sequence_only ablation mode in ESM3PredictorBase._get_esm3_outputs."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import yaml

from esm_msr.config import parse_arguments
from esm_msr.inference import parse_hparams_to_lora_config
from esm_msr.models import ESM3PredictorBase


def _make_stub(sequence_only: bool = False, use_plddt: bool = False, default_plddt_one: bool = False):
    captured: dict = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(sequence_logits=torch.zeros(1, 4, 33))

    stub_esm = SimpleNamespace(model=fake_forward)
    pred = ESM3PredictorBase.__new__(ESM3PredictorBase)
    torch.nn.Module.__init__(pred)
    pred.model = stub_esm
    pred.sequence_only = sequence_only
    pred.use_plddt = use_plddt
    pred.default_plddt_one = default_plddt_one
    return pred, captured


def test_default_passes_structure_tensors():
    pred, cap = _make_stub(sequence_only=False)
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.ones((1, 4, 3, 3))
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])

    pred._get_esm3_outputs(seq, coords, struct_tok)

    assert cap['sequence_tokens'].shape == (1, 4)
    assert cap['structure_coords'] is not None
    assert cap['structure_tokens'] is not None
    assert cap['per_res_plddt'] is None


def test_sequence_only_omits_coords_struct_tokens_and_plddt():
    pred, cap = _make_stub(sequence_only=True, use_plddt=True, default_plddt_one=True)
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.ones((1, 4, 3, 3))
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    plddt = torch.tensor([[0.0, 0.9, 0.9, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=plddt)

    assert torch.equal(cap['sequence_tokens'], seq)
    assert cap['structure_coords'] is None
    assert cap['structure_tokens'] is None
    assert cap['per_res_plddt'] is None


def test_sequence_only_handles_unbatched_sequence():
    pred, cap = _make_stub(sequence_only=True)
    seq = torch.tensor([0, 5, 6, 2])
    coords = torch.ones((4, 3, 3))
    struct_tok = torch.tensor([4098, 10, 20, 4097])

    pred._get_esm3_outputs(seq, coords, struct_tok)

    assert cap['sequence_tokens'].shape == (1, 4)
    assert cap['structure_coords'] is None
    assert cap['structure_tokens'] is None
    assert cap['per_res_plddt'] is None


def test_sequence_only_config_and_hparams_opt_in(tmp_path):
    assert parse_arguments(['--experiment_name', 't']).sequence_only is False
    assert parse_arguments(['--experiment_name', 't', '--sequence_only']).sequence_only is True

    hp_old = tmp_path / 'old.yaml'
    hp_old.write_text(yaml.safe_dump({'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
                                      'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1}))
    assert parse_hparams_to_lora_config(str(hp_old))['sequence_only'] is False

    hp_seq = tmp_path / 'seq.yaml'
    hp_seq.write_text(yaml.safe_dump({'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
                                      'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1,
                                      'sequence_only': True}))
    assert parse_hparams_to_lora_config(str(hp_seq))['sequence_only'] is True
