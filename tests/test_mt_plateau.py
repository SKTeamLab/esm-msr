"""
The MT path's own plateau, keyed on epi_beyond_rho_mt.

Measured across dc6/dc7/dc8_s1 that metric peaks early and then falls back without regaining the peak - dc6 at
epoch 4 then -0.0190, dc8 at epoch 3 then -0.0079 - which is the overshoot this scheduler exists to stop. It is
also NOISY, swinging +-0.03 with no trend, and dc7_s1 peaked at its LAST epoch after two dips. So these tests pin
both halves: that a sustained decline decays the MT groups, and that only the MT groups are ever touched.
"""
import types

import numpy as np
import pytest

from esm_msr import training

CLS = training.ESM3EpistasisLightningModule
GROUPS = ('lora_wt', 'calib_wt', 'lora_mt', 'calib_mt', 'link')


def stub(patience=1, factor=0.3, metric='epi_beyond_rho_mt', freeze_after=0, mt_frozen=False):
    s = types.SimpleNamespace()
    s.hparams = types.SimpleNamespace(plateau_patience=patience, plateau_factor=factor,
                                      mt_convergence_metric=metric, freeze_after_decays=freeze_after)
    s.peft_manager = types.SimpleNamespace(mt_path_is_frozen=mt_frozen, froze=[])
    def _freeze():                      # the real PEFTStateManager sets this, and _mt_plateau_step returns
        s.peft_manager.froze.append('mt')     # early on it, so the freeze cannot repeat
        s.peft_manager.mt_path_is_frozen = True
    s.peft_manager.freeze_mt_components = _freeze
    s.peft_manager.enforce_freezing = lambda *a, **k: None
    s._groups = [{'name': n, 'lr': 2e-4} for n in GROUPS]
    s.optimizers = lambda: types.SimpleNamespace(param_groups=s._groups)
    return s


def feed(s, values):
    for v in values:
        CLS._mt_plateau_step(s, {'epi_beyond_rho_mt': [v]})
    return {g['name']: g['lr'] for g in s._groups}


def test_dc6s1_trajectory_decays_after_the_peak():
    """dc6_s1's actual epi_beyond_rho_mt: peak at e4, then a sustained decline."""
    s = stub(patience=1)
    lrs = feed(s, [0.4887, 0.5012, 0.5218, 0.5182, 0.5395, 0.5374, 0.5231, 0.5111, 0.5225, 0.5205])
    assert s._mt_decays >= 1
    assert lrs['lora_mt'] < 2e-4 and lrs['calib_mt'] < 2e-4


def test_only_the_mt_groups_are_touched():
    s = stub(patience=1)
    lrs = feed(s, [0.50, 0.49, 0.48])
    assert lrs['lora_wt'] == 2e-4 and lrs['calib_wt'] == 2e-4 and lrs['link'] == 2e-4
    assert lrs['lora_mt'] < 2e-4 and lrs['calib_mt'] < 2e-4


def test_a_rising_metric_never_decays():
    s = stub(patience=1)
    lrs = feed(s, [0.41, 0.46, 0.51, 0.53, 0.54])
    assert getattr(s, '_mt_decays', 0) == 0
    assert all(v == 2e-4 for v in lrs.values())


def test_dc7s1_shows_the_cost_of_patience_1():
    """dc7_s1 peaked at its LAST epoch. Patience 1 decays it at e4, long before that - recorded, not asserted away."""
    dc7 = [0.4145, 0.4617, 0.5169, 0.5308, 0.5095, 0.5059, 0.5321, 0.5276, 0.5237, 0.5367]
    s1 = stub(patience=1); feed(s1, dc7)
    s2 = stub(patience=2); feed(s2, dc7)
    assert s1._mt_decays > s2._mt_decays, "patience 2 must be no more trigger-happy than patience 1"
    assert s1._mt_decays >= 2, "patience 1 fires repeatedly on this run's noise"


def test_patience_zero_is_off():
    s = stub(patience=0)
    lrs = feed(s, [0.54, 0.40, 0.30])
    assert getattr(s, '_mt_decays', 0) == 0 and all(v == 2e-4 for v in lrs.values())


def test_a_frozen_mt_path_is_left_alone():
    s = stub(patience=1, mt_frozen=True)
    lrs = feed(s, [0.54, 0.40, 0.30])
    assert getattr(s, '_mt_decays', 0) == 0 and all(v == 2e-4 for v in lrs.values())


def test_decays_compound_per_head():
    s = stub(patience=1, factor=0.1)
    out = feed(s, [0.5] + [0.4] * 3)
    assert out['lora_mt'] == pytest.approx(2e-4 * 0.1 ** 3) and out['lora_wt'] == pytest.approx(2e-4)


def test_freezing_after_n_decays():
    s = stub(patience=1, freeze_after=2)
    feed(s, [0.54, 0.50, 0.49, 0.48, 0.47])
    assert s.peft_manager.froze == ['mt'], f"froze more than once: {s.peft_manager.froze}"
    assert s._mt_decays == 2, "kept decaying after the freeze"



def test_a_missing_or_nan_metric_is_ignored():
    s = stub(patience=1)
    CLS._mt_plateau_step(s, {})
    CLS._mt_plateau_step(s, {'epi_beyond_rho_mt': [float('nan')]})
    assert getattr(s, '_mt_decays', 0) == 0


def test_the_metric_name_is_configurable():
    s = stub(patience=1, metric='epi_cell_rank_mt')
    for v in (0.20, 0.19, 0.18):
        CLS._mt_plateau_step(s, {'epi_cell_rank_mt': [v]})
    assert s._mt_decays >= 1


def test_defaults_turn_it_on_with_patience_one():
    from esm_msr import config
    import sys
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        a = config.parse_arguments()
    finally:
        sys.argv = argv
    assert a.plateau_patience == 1
    assert a.mt_convergence_metric == 'epi_beyond_lib_comb'
    assert a.plateau_factor == 0.3           # gentle, because the metric is noisy: a patience-1 decay can fire on a wobble


def test_a_missing_metric_warns_instead_of_sitting_silent(caplog):
    """dc12 burned 8 epochs with an inert plateau because the metric never reached avg_metrics and nothing said so."""
    import logging
    s = stub(patience=1)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            CLS._mt_plateau_step(s, {'rho_combined': [0.84]})      # the key it wants is absent
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 1, f"expected exactly one warning, got {len(msgs)}"
    assert 'will never fire' in msgs[0] and 'epi_beyond_rho_mt' in msgs[0]
    assert getattr(s, '_mt_decays', 0) == 0


def test_log_epistasis_returns_its_results():
    """The bug itself: it logged res and fell off the end, so the caller merged None into avg_metrics."""
    import inspect
    src = inspect.getsource(CLS._log_epistasis)
    assert 'return res' in src, "_log_epistasis must hand its scalars back for the plateaus to see them"
