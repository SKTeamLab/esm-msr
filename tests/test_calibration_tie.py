"""The MT calibration head can share the WT head's scale (--tie_mt_scale, default OFF: independent scales).

The calibrated outputs are the common latent scale; with an independent MT scale the raw-scale rank loss weights the
two halves differently from the calibrated combined prediction. The bias stays per head.
"""
import sys

import pytest
import torch

from esm_msr.models import CalibrationHead, _ScaleSource


def test_mt_scale_follows_the_wt_scale():
    wt = CalibrationHead(init_scale=0.37, init_bias=0.0, per_mutation=True)
    mt = CalibrationHead(init_scale=0.9, init_bias=0.0, per_mutation=True)
    mt.scale_ref = _ScaleSource(wt)
    assert float(mt.scale) == pytest.approx(float(wt.scale), abs=1e-7)
    with torch.no_grad():
        wt.raw_scale.add_(0.5)
    assert float(mt.scale) == pytest.approx(float(wt.scale), abs=1e-7), "the tie must read the WT scale live"


def test_an_untied_mt_scale_stays_its_own():
    wt = CalibrationHead(init_scale=0.37, init_bias=0.0)
    mt = CalibrationHead(init_scale=0.9, init_bias=0.0)
    assert float(mt.scale) != pytest.approx(float(wt.scale), abs=1e-3)


def test_per_mutation_bias_is_on_by_default_and_cannot_be_disabled_with_a_link():
    from esm_msr import config
    base = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
    argv = sys.argv
    try:
        sys.argv = base
        assert config.parse_arguments().per_mutation_bias is True     # mandatory default
        sys.argv = base + ['--no-per_mutation_bias']                  # ... and it cannot be turned off
        with pytest.raises(SystemExit):
            config.parse_arguments()
    finally:
        sys.argv = argv


def test_tie_mt_scale_is_off_by_default_and_can_be_turned_on():
    """Independent MT scale unless asked to tie (the released models, every MT run so far, and the trimmed recipes). The setting is a real flag
    and is recorded in hparams.yaml, because inference reads it back."""
    from esm_msr import config
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--experiment_name', 'x']
        assert config.parse_arguments().tie_mt_scale is False
        sys.argv = ['training.py', '--experiment_name', 'x', '--tie_mt_scale']
        assert config.parse_arguments().tie_mt_scale is True
        sys.argv = ['training.py', '--experiment_name', 'x', '--no-tie_mt_scale']      # the old launchers' spelling of the default
        assert config.parse_arguments().tie_mt_scale is False
    finally:
        sys.argv = argv
