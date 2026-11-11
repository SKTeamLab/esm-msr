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


