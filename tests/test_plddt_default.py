"""Unit tests verifying --plddt_mode {zero, one, infer} conditioning across ESM-MSR.

Default behavior (`plddt_mode='zero'`, when no flag is passed) passes `per_res_plddt=None`
to preserve exact manuscript `FINAL_p0` conditioning (ESM3.forward zero-filled default).
`plddt_mode='one'` passes `1.0` wherever structure coordinates are finite (matching official
`ESM3.logits()` conditioning, `0.0` for BOS/EOS/missing residues).
`plddt_mode='infer'` infers real per-residue and average pLDDTs normalized to `[0, 1]`
for predicted structures (both `[0, 100]` AF2/ColabFold and `[0, 1]` ESMFold scales),
while safely falling back to `1.0` on crystal B-factors (>100 or experimental PDB headers)
and all-zero placeholders (e.g. Rosetta de novo designs).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from esm_msr.blocks import BlockDataset, build_library
from esm_msr.config import parse_arguments
from esm_msr.inference import parse_hparams_to_lora_config
from esm_msr.models import ESM3PredictorBase, MSRModel
from esm_msr.plddt import is_experimental_pdb, plddt_mode_from_hparams, resolve_plddt_mode


def _make_stub(use_plddt: bool = False, default_plddt_one: bool = False, plddt_mode: str | None = None):
    captured: dict = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        B, L = kwargs['sequence_tokens'].shape
        return SimpleNamespace(sequence_logits=torch.zeros(B, L, 33))

    stub_esm = SimpleNamespace(model=fake_forward)
    pred = ESM3PredictorBase.__new__(ESM3PredictorBase)
    torch.nn.Module.__init__(pred)
    pred.model = stub_esm
    pred.use_plddt = use_plddt
    pred.default_plddt_one = default_plddt_one
    pred.plddt_mode = plddt_mode if plddt_mode is not None else 'zero'
    pred.sequence_only = False
    return pred, captured


def test_default_passes_none_plddt_for_final_p0_parity():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=False)
    seq = torch.tensor([[0, 5, 6, 2]])
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    coords = torch.ones((1, 4, 3, 3))
    raw_bfac = torch.tensor([[0.0, 92.5, 81.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=raw_bfac)

    assert cap['per_res_plddt'] is None
    assert 'average_plddt' not in cap


def test_plddt_mode_one_sets_ones_where_coords_finite():
    pred, cap = _make_stub(plddt_mode='one')
    seq = torch.tensor([[0, 5, 6, 2]])
    struct_tok = torch.tensor([[4098, 10, 20, 4097]])
    coords = torch.full((1, 4, 3, 3), float('nan'))
    coords[0, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
    raw_bfac = torch.tensor([[0.0, 92.5, 81.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, struct_tok, per_res_plddt=raw_bfac)

    plddt = cap['per_res_plddt']
    assert plddt is not None
    assert plddt.shape == (1, 4)
    assert plddt.dtype == torch.float32
    assert torch.equal(plddt, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert 'average_plddt' not in cap


def test_legacy_default_plddt_one_attribute_still_maps_to_one():
    pred, cap = _make_stub(use_plddt=False, default_plddt_one=True)
    seq = torch.tensor([0, 5, 2])
    coords = torch.full((3, 3, 3), float('nan'))
    coords[1] = 1.0

    pred._get_esm3_outputs(seq, coords, None)

    assert cap['structure_coords'].shape == (1, 3, 3, 3)
    assert torch.equal(cap['per_res_plddt'], torch.tensor([[0.0, 1.0, 0.0]]))


def test_sequence_only_passes_none_plddt_even_in_infer_mode():
    pred, cap = _make_stub(plddt_mode='infer')
    seq = torch.tensor([[0, 5, 6, 2]])

    pred._get_esm3_outputs(seq, structure_coords=None, structure_tokens=None)

    assert cap['structure_coords'] is None
    assert cap['per_res_plddt'] is None


def test_infer_normalizes_af2_0_100_plddt_and_sets_average_plddt():
    pred, cap = _make_stub(plddt_mode='infer')
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.full((1, 4, 3, 3), float('nan'))
    coords[0, 1] = 1.0
    coords[0, 2] = 2.0
    raw_af2 = torch.tensor([[0.0, 95.0, 80.0, 0.0]])

    pred._get_esm3_outputs(seq, coords, None, per_res_plddt=raw_af2)

    assert torch.allclose(cap['per_res_plddt'], torch.tensor([[0.0, 0.95, 0.80, 0.0]]))
    assert torch.allclose(cap['average_plddt'], torch.full((1, 4), 0.875))


def test_infer_preserves_0_1_scale_plddt():
    pred, cap = _make_stub(plddt_mode='infer')
    seq = torch.tensor([[0, 5, 6, 2]])
    coords = torch.full((1, 4, 3, 3), float('nan'))
    coords[0, 1] = 1.0
    coords[0, 2] = 2.0
    raw_01 = torch.tensor([[0.0, 0.92, 0.74, 0.0]])

    pred._get_esm3_outputs(seq, coords, None, per_res_plddt=raw_01)

    assert torch.allclose(cap['per_res_plddt'], torch.tensor([[0.0, 0.92, 0.74, 0.0]]))
    assert torch.allclose(cap['average_plddt'], torch.full((1, 4), 0.83))


def test_infer_falls_back_to_ones_for_all_zero_rosetta_and_gt100_crystal_bfactors():
    pred, cap = _make_stub(plddt_mode='infer')
    seq = torch.tensor([[0, 5, 6, 2], [0, 5, 6, 2]])
    coords = torch.full((2, 4, 3, 3), float('nan'))
    coords[:, 1:3] = 1.0
    # Row 0: Rosetta all-zero B-factors; Row 1: Crystal B-factors > 100 A^2
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 25.0, 145.5, 0.0],
    ])

    pred._get_esm3_outputs(seq, coords, None, per_res_plddt=raw)

    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    assert torch.equal(cap['per_res_plddt'], expected)
    assert torch.allclose(cap['average_plddt'], torch.ones((2, 4)))


def test_is_experimental_pdb_distinguishes_crystal_from_af2_dummy_cryst1(tmp_path):
    af2_pdb = tmp_path / 'af2.pdb'
    af2_pdb.write_text(
        "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 94.50           C\n"
    )
    assert is_experimental_pdb(af2_pdb) is False

    cryst_expdta = tmp_path / 'xray.pdb'
    cryst_expdta.write_text(
        "EXPDTA    X-RAY DIFFRACTION\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 18.20           C\n"
    )
    assert is_experimental_pdb(cryst_expdta) is True

    cryst_cell = tmp_path / 'lattice.pdb'
    cryst_cell.write_text(
        "CRYST1   52.100   61.300   74.800  90.00  90.00  90.00 P 21 21 21    4\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 22.10           C\n"
    )
    assert is_experimental_pdb(cryst_cell) is True


def test_forward_site_llr_and_blocks_plumb_plddt_to_esm3():
    # Verify build_library + BlockDataset carry plddt and forward_site_llr forwards it
    item = {
        'subset_type': 'single',
        'mut_pos': np.array([1]),
        'wt_id': np.array([5]),   # A
        'mt_id': np.array([23]),  # C
        'ddG': 0.5,
        'cens': 0,
        'cens_bound': np.nan,
        'dG_meas': 2.0,
        'dG_wt': 1.5,
        'cens_src': 0,
        'wt_sequence_tokens': np.array([0, 5, 6, 2], dtype=np.int64),
        'coords_orig': np.ones((1, 4, 37, 3), dtype=np.float32),
        'structure_tokens_orig': np.array([4098, 10, 20, 4097], dtype=np.int64),
        'plddt': np.array([[0.0, 90.0, 80.0, 0.0]], dtype=np.float32),
    }
    item['coords_orig'][0, 0] = np.nan
    item['coords_orig'][0, 3] = np.nan
    lib = build_library('1TEST', [item])
    assert lib is not None and lib.plddt is not None
    assert lib.plddt.shape == (1, 4)

    ds = BlockDataset([lib])
    batch = ds[('s', 0, [0])]
    assert 'plddt' in batch
    assert torch.equal(batch['plddt'], torch.tensor([[0.0, 90.0, 80.0, 0.0]]))

    # Verify forward_site_llr routes plddt through _get_esm3_outputs
    captured: dict = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        B, L = kwargs['sequence_tokens'].shape
        return SimpleNamespace(sequence_logits=torch.zeros(B, L, 33))

    stub = MSRModel.__new__(MSRModel)
    torch.nn.Module.__init__(stub)
    stub.model = SimpleNamespace(model=fake_forward)
    stub.peft_wt = stub.model
    stub.peft_mt = stub.model
    stub.adapter_mode = 'dual'
    stub.log_likelihood = False
    stub.plddt_mode = 'infer'
    stub.sequence_only = False
    stub.register_buffer('canonical_idx_tensor', torch.arange(4, 24, dtype=torch.long))

    seqs = torch.tensor([[0, 32, 6, 2], [0, 5, 32, 2]], dtype=torch.long)
    read_pos = torch.tensor([1, 2], dtype=torch.long)
    wt_at = torch.tensor([5, 6], dtype=torch.long)
    out = stub.forward_site_llr(seqs, read_pos, wt_at, 'wt', batch['coords'], batch['structure_tokens'], batch['plddt'])
    assert out.shape == (2, 20)
    assert torch.allclose(captured['per_res_plddt'], torch.tensor([[0.0, 0.90, 0.80, 0.0], [0.0, 0.90, 0.80, 0.0]]))
    assert torch.allclose(captured['average_plddt'], torch.full((2, 4), 0.85))


def test_training_config_and_hparams_plddt_mode_and_legacy_aliases(tmp_path):
    args_def = parse_arguments(['--experiment_name', 'unit_test'])
    assert args_def.plddt_mode == 'zero'
    assert args_def.default_plddt_one is False

    args_one = parse_arguments(['--experiment_name', 'unit_test', '--plddt_mode', 'one'])
    assert args_one.plddt_mode == 'one'
    assert args_one.default_plddt_one is True

    args_infer = parse_arguments(['--experiment_name', 'unit_test', '--plddt_mode', 'infer'])
    assert args_infer.plddt_mode == 'infer'
    assert args_infer.default_plddt_one is False

    # Backward-compatible legacy alias
    args_legacy = parse_arguments(['--experiment_name', 'unit_test', '--default_plddt_one'])
    assert args_legacy.plddt_mode == 'one'
    assert args_legacy.default_plddt_one is True

    # Conflicting flags fail cleanly
    with pytest.raises(SystemExit):
        parse_arguments(['--experiment_name', 'unit_test', '--plddt_mode', 'infer', '--default_plddt_one'])

    # hparams.yaml round-trip (legacy missing, legacy default_plddt_one, and new plddt_mode)
    base_hp = {'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
               'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1}

    hp_old = tmp_path / 'hparams_old.yaml'
    hp_old.write_text(yaml.safe_dump(base_hp))
    c_old = parse_hparams_to_lora_config(str(hp_old))
    assert c_old['plddt_mode'] == 'zero'
    assert c_old['default_plddt_one'] is False

    hp_one = tmp_path / 'hparams_one.yaml'
    hp_one.write_text(yaml.safe_dump({**base_hp, 'default_plddt_one': True}))
    c_one = parse_hparams_to_lora_config(str(hp_one))
    assert c_one['plddt_mode'] == 'one'
    assert c_one['default_plddt_one'] is True

    hp_infer = tmp_path / 'hparams_infer.yaml'
    hp_infer.write_text(yaml.safe_dump({**base_hp, 'plddt_mode': 'infer'}))
    c_infer = parse_hparams_to_lora_config(str(hp_infer))
    assert c_infer['plddt_mode'] == 'infer'
    assert c_infer['default_plddt_one'] is False
    assert plddt_mode_from_hparams({'plddt_mode': 'infer'}) == 'infer'
    assert resolve_plddt_mode(use_plddt=True) == 'infer'
