"""Inference rebuilds the calibration heads a checkpoint recorded, and applies the project defaults where it recorded none.

The defaults are the same everywhere (training config, the model constructor, this fallback): tie_mt_scale OFF, an
independent MT scale, and per_mutation_bias ON, the bias once per mutated position. A checkpoint whose hparams lack the
keys (the released models, every run before the flags) was trained with an independent scale, so tying it would give
the released small model's MT half the WT scale (0.340 instead of its own 0.217); it was also trained with one bias per
item, so the per-mutation default shifts its multi-mutants by (k - 1) * bias and leaves singles alone. That second
change is a decision, and results computed per item before it have to be rerun.
"""
import pytest

from esm_msr.inference import calibration_from_hparams, resolve_calibration


def test_missing_keys_mean_the_project_defaults():
    assert calibration_from_hparams({}) == {'tie_mt_scale': False, 'per_mutation_bias': True}


def test_recorded_settings_are_used():
    assert calibration_from_hparams({'tie_mt_scale': True, 'per_mutation_bias': True}) == \
        {'tie_mt_scale': True, 'per_mutation_bias': True}


@pytest.mark.parametrize('ckpt', [None, {'tie_mt_scale': True, 'per_mutation_bias': True}])
def test_explicit_flags_win(ckpt):
    assert resolve_calibration(ckpt, tie_mt_scale=False, per_mutation_bias=False) == \
        {'tie_mt_scale': False, 'per_mutation_bias': False}
    assert resolve_calibration(ckpt) == (ckpt or {'tie_mt_scale': False, 'per_mutation_bias': True})


def test_model_defaults_match_the_project_defaults():
    import inspect
    from esm_msr.models import MSRModel
    p = inspect.signature(MSRModel.__init__).parameters
    assert p['tie_mt_scale'].default is False
    assert p['per_mutation_bias'].default is True


def test_parse_hparams_extracts_asymmetric_mask_strategy(tmp_path):
    import yaml
    from esm_msr.inference import parse_hparams_to_lora_config
    hp_file = tmp_path / "hparams.yaml"
    hp_file.write_text(yaml.safe_dump({
        'adapter_mode': 'dual',
        'lora_rank_wt': 2, 'lora_alpha_wt': 4, 'lora_dropout_wt': 0.1,
        'lora_rank_mt': 16, 'lora_alpha_mt': 16, 'lora_dropout_mt': 0.1,
        'mask_strategy_wt': None,
        'mask_strategy_mt': 'independent',
        'native_cond_for_mt': False,
        'mask_native_cond_struct': True,
        'fused_antisymmetric': False,
    }))
    cfg = parse_hparams_to_lora_config(str(hp_file))
    assert cfg['mask_strategy'] == {'wt': None, 'mt': 'independent'}
    assert cfg['mask_native_cond_struct'] is True
    assert cfg['fused_antisymmetric'] is False

