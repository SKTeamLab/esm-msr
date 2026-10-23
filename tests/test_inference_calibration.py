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
