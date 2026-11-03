"""The redundant ``_epi``-head epistasis metrics are no longer logged, and a flag that still names one is rewritten to its kept twin."""
import sys

import numpy as np
import pytest

from esm_msr import config, epi_metrics
from esm_msr.training import RETIRED_EPI_LOG, epi_log_names

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def synthetic_res():
    names = epi_metrics.logged_names()
    return {n: 0.1 + 0.01 * i for i, n in enumerate(names)}, {
        'epi_clean_err_comb': 0.5, 'epi_clean_err_epi': 0.5, 'epi_clean_bias_comb': 0.1, 'epi_clean_bias_epi': 0.1,
        'epi_cell_rank_mt_legacy': 0.2, 'wt_doubles_clean_rmse': 1.0, 'wt_doubles_clean_slope': 0.9}


def test_no_epi_head_scalar_reaches_the_logger():
    res, diag = synthetic_res()
    assert any(RETIRED_EPI_LOG.match(k) for k in res)                     # the module still produces them
    out = epi_log_names(res, diag)
    assert not [k for k in out if k.endswith('_epi')]
    assert 'val_epi_beyond_lib_comb' in out and 'val_epi_cell_rank_comb' in out and 'val_epi_global_err_comb' in out
    assert 'val_epi_naive_rho_add' in out                                 # the saturation-only control is a different head, not a duplicate
    assert 'val_diag_epi_cell_rank_mt_legacy' in out and 'val_diag_epi_clean_err_comb' in out
    assert 'val_diag_epi_clean_err_epi' not in out and 'val_diag_epi_clean_bias_epi' not in out


def test_the_unmasked_regime_logs_the_mt_cell_rank_and_no_other_diagnostic():
    res, diag = synthetic_res()
    out = epi_log_names(res, diag, 'unmasked_')
    assert 'val_unmasked_diag_epi_cell_rank_mt_legacy' in out
    assert not [k for k in out if k.startswith('val_unmasked_diag_') and not k.endswith('mt_legacy')]
    assert all(k.startswith('val_unmasked_') for k in out)
    assert not [k for k in out if k.endswith('_epi')]


def test_non_finite_values_are_dropped():
    out = epi_log_names({'epi_cell_rank_comb': float('nan'), 'epi_pair_rho_comb': 0.3}, {})
    assert out == {'val_epi_pair_rho_comb': 0.3}


def _noisy_table(seed=0):
    """Two libraries of 6 x 6 doubles, a noisy WT head and a noisy MT head that learned some of the epistasis."""
    rng = np.random.default_rng(seed)
    keys, st, ddg, dddg, dg_wt, wt, mt, lib = [], [], [], [], [], [], [], []
    for L, c in (('L1', 2.0), ('L2', 3.0)):
        a, b = rng.normal(0, 1, 6), rng.normal(0, 1, 6)
        eps = rng.normal(0, 0.7, (6, 6))
        wa, wb = a + rng.normal(0, 0.3, 6), b + rng.normal(0, 0.3, 6)
        for i in range(6):
            keys.append(((L, 10, chr(65 + i)),)); st.append('single'); ddg.append(a[i]); dddg.append(np.nan); dg_wt.append(c)
            wt.append(wa[i]); mt.append(wa[i]); lib.append(L)
        for j in range(6):
            keys.append(((L, 20, chr(70 + j)),)); st.append('single'); ddg.append(b[j]); dddg.append(np.nan); dg_wt.append(c)
            wt.append(wb[j]); mt.append(wb[j]); lib.append(L)
        for i in range(6):
            for j in range(6):
                keys.append(((L, 10, chr(65 + i)), (L, 20, chr(70 + j)))); st.append('double'); ddg.append(a[i] + b[j] + eps[i, j])
                dddg.append(eps[i, j]); dg_wt.append(c); wt.append(wa[i] + wb[j]); lib.append(L)
                mt.append(wa[i] + wb[j] + 2.0 * (0.6 * eps[i, j] + rng.normal(0, 0.4)))
    wt, mt = np.array(wt), np.array(mt)
    return epi_metrics.Table({'mut_key': keys, 'subset_type': st, 'cens': np.zeros(len(keys), int), 'ddG': ddg, 'dddG': dddg, 'dG_wt': dg_wt,
                              'wt': wt, 'mt': mt, 'comb': 0.5 * (wt + mt)}, link=None)


def test_the_scale_free_levels_equal_their_comb_twin_and_cell_rank_epi_is_the_mt_legacy_rank():
    """The identities the removal rests on: comb is half the epi residual, so every level that ignores scale is exact, and cell_rank_epi
    is the MT head's own cell rank (val_diag_epi_cell_rank_mt_legacy), which is kept."""
    t = _noisy_table()
    res, diag = epi_metrics.compute(t), epi_metrics.compute_clean_diagnostics(t)
    for m in ('beyond_rho', 'beyond_lib', 'beyond_rho_pairavg', 'pair_rho', 'line_rho'):
        assert res[f'epi_{m}_epi'] == pytest.approx(res[f'epi_{m}_comb'], abs=1e-12, nan_ok=True), m
    assert res['epi_cell_rank_epi'] == pytest.approx(diag['epi_cell_rank_mt_legacy'], abs=1e-12)
    assert diag['epi_clean_err_epi'] == diag['epi_clean_err_comb'] and diag['epi_clean_bias_epi'] == diag['epi_clean_bias_comb']
    # what is NOT redundant: the scale-dependent metrics measure 2x epistasis for the epi head and 1x for the reported prediction
    assert abs(res['epi_global_err_epi'] - res['epi_global_err_comb']) > 1e-3
    assert abs(res['epi_cell_rank_comb'] - res['epi_cell_rank_epi']) > 1e-3
    # and the filter keeps every one of the non-redundant names
    kept = epi_log_names(res, diag)
    assert 'val_epi_global_err_comb' in kept and 'val_epi_cell_rank_comb' in kept and 'val_diag_epi_cell_rank_mt_legacy' in kept


def test_a_retired_monitor_is_rewritten():
    a = parse('--monitor_metric', 'val_epi_beyond_lib_epi', '--mt_convergence_metric', 'epi_beyond_rho_epi')
    assert a.monitor_metric == 'val_epi_beyond_lib_comb' and a.mt_convergence_metric == 'epi_beyond_rho_comb'
    a = parse('--monitor_metric', 'val_epi_cell_rank_epi')
    assert a.monitor_metric == 'val_diag_epi_cell_rank_mt_legacy'
    a = parse('--monitor_metric', 'val_unmasked_epi_cell_rank_epi')
    assert a.monitor_metric == 'val_unmasked_diag_epi_cell_rank_mt_legacy'
    a = parse('--monitor_metric', 'val_unmasked_epi_pair_rho_epi')
    assert a.monitor_metric == 'val_unmasked_epi_pair_rho_comb'


def test_a_monitor_with_no_exact_twin_is_refused_but_a_plateau_metric_is_left_alone():
    with pytest.raises(SystemExit):
        parse('--monitor_metric', 'val_epi_global_err_epi')
    assert parse('--mt_convergence_metric', 'epi_global_err_epi').mt_convergence_metric == 'epi_global_err_epi'   # read from the internal results


def test_the_defaults_name_logged_metrics():
    a = parse()
    assert a.monitor_metric == 'val_epi_beyond_rho_comb' and a.mt_convergence_metric == 'epi_beyond_rho_comb'
    assert parse('--monitor_metric', 'val_epi_composite').monitor_metric == 'val_epi_composite'
