"""Inference must rebuild the calibration heads a checkpoint was trained with, not the current training defaults.

tie_mt_scale and per_mutation_bias default ON for new training runs. A checkpoint whose hparams lack the keys (the
released models, every run before the flags) was trained untied with one bias per item; scoring it tied would give the
released small model's MT half the WT scale (0.340 instead of its own 0.217).
"""
import pytest

from esm_msr.inference import calibration_from_hparams, resolve_calibration


def test_missing_keys_mean_the_released_behaviour():
    assert calibration_from_hparams({}) == {'tie_mt_scale': False, 'per_mutation_bias': False}


def test_recorded_settings_are_used():
    assert calibration_from_hparams({'tie_mt_scale': True, 'per_mutation_bias': True}) == \
        {'tie_mt_scale': True, 'per_mutation_bias': True}


@pytest.mark.parametrize('ckpt', [None, {'tie_mt_scale': True, 'per_mutation_bias': True}])
def test_explicit_flags_win(ckpt):
    assert resolve_calibration(ckpt, tie_mt_scale=False, per_mutation_bias=False) == \
        {'tie_mt_scale': False, 'per_mutation_bias': False}
    assert resolve_calibration(ckpt) == (ckpt or {'tie_mt_scale': False, 'per_mutation_bias': False})


def test_model_default_is_untied():
    import inspect
    from esm_msr.models import MSRModel
    assert inspect.signature(MSRModel.__init__).parameters['tie_mt_scale'].default is False
