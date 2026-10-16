"""The WT path's learning rate must follow its OWN metric.

The shared ReduceLROnPlateau watches val_rho_combined_avg, which the MT path keeps improving long after
rho_wt_valid has stopped (it peaks around epoch 2). Without a separate rule the WT adapter keeps taking
full-rate steps on a metric it is no longer improving, drifting, and moving the teacher-forced target the MT
path is fitting against.
"""
import types

import numpy as np
import pytest

from esm_msr.training import ESM3EpistasisLightningModule as M


class _HP(dict):
    __getattr__ = dict.get

    def get(self, k, default=None):
        return dict.get(self, k, default)


def _stub(patience=1, factor=0.1, freeze_after=0, min_lr=1e-7):
    s = types.SimpleNamespace()
    s.hparams = _HP(wt_convergence_metric='rho_wt_valid', wt_plateau_patience=patience,
                    wt_plateau_factor=factor, wt_plateau_min_lr=min_lr, wt_freeze_after_decays=freeze_after)
    s.groups = [{'name': 'lora_wt', 'lr': 2e-4}, {'name': 'calib_wt', 'lr': 4e-3},
                {'name': 'lora_mt', 'lr': 2e-4}, {'name': 'calib_mt', 'lr': 4e-3}, {'name': 'link', 'lr': 5e-3}]
    s.optimizers = lambda: types.SimpleNamespace(param_groups=s.groups)
    s.peft_manager = types.SimpleNamespace(has_transitioned=False)
    s.froze = []
    s._transition_to_mt = lambda: (s.froze.append(True), setattr(s.peft_manager, 'has_transitioned', True))
    s._wt_plateau_step = types.MethodType(M._wt_plateau_step, s)
    return s


def _lr(s, name):
    return next(g['lr'] for g in s.groups if g['name'] == name)


def feed(s, values):
    for v in values:
        s._wt_plateau_step({'rho_combined': [0.8], 'rho_wt_valid': [v]})


def test_improving_metric_never_decays():
    s = _stub()
    feed(s, [0.80, 0.81, 0.82, 0.83])
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)
    assert _lr(s, 'calib_wt') == pytest.approx(4e-3)


def test_plateau_decays_only_the_wt_groups():
    s = _stub(patience=1)
    feed(s, [0.80, 0.80])                      # one epoch of no improvement at patience 1
    assert _lr(s, 'lora_wt') == pytest.approx(2e-5)
    assert _lr(s, 'calib_wt') == pytest.approx(4e-4)
    assert _lr(s, 'lora_mt') == pytest.approx(2e-4), "the MT path must be untouched"
    assert _lr(s, 'calib_mt') == pytest.approx(4e-3), "the MT path must be untouched"
    assert _lr(s, 'link') == pytest.approx(5e-3), "the link must be untouched"


def test_patience_is_respected():
    s = _stub(patience=3)
    feed(s, [0.80, 0.80, 0.80])                # best + 2 bad epochs: not yet
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)
    feed(s, [0.80])                            # third bad epoch
    assert _lr(s, 'lora_wt') == pytest.approx(2e-5)


def test_a_new_best_resets_the_counter():
    s = _stub(patience=2)
    feed(s, [0.80, 0.80, 0.85, 0.85])          # bad, then a new best, then one bad
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)


def test_decay_repeats_and_respects_min_lr():
    s = _stub(patience=1, min_lr=1e-5)
    feed(s, [0.80] + [0.80] * 6)
    assert _lr(s, 'lora_wt') == pytest.approx(1e-5), "must not fall below --wt_plateau_min_lr"


def test_freezes_after_the_configured_number_of_decays():
    s = _stub(patience=1, freeze_after=2)
    feed(s, [0.80, 0.80])
    assert s.froze == [], "one decay is not enough"
    feed(s, [0.80])
    assert s.froze == [True], "the second decay must freeze the WT path"


def test_disabled_by_default_patience_zero():
    s = _stub(patience=0)
    feed(s, [0.80] * 5)
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)


def test_does_nothing_once_already_transitioned():
    s = _stub(patience=1)
    s.peft_manager.has_transitioned = True
    feed(s, [0.80, 0.80])
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)


def test_a_missing_or_nan_metric_is_ignored():
    s = _stub(patience=1)
    s._wt_plateau_step({'rho_combined': [0.8]})                       # key absent
    s._wt_plateau_step({'rho_wt_valid': []})                          # empty
    s._wt_plateau_step({'rho_wt_valid': [float('nan')]})              # NaN
    assert _lr(s, 'lora_wt') == pytest.approx(2e-4)
