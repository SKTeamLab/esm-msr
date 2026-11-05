"""A flag that still names a retired ``_epi``-head epistasis metric is rewritten to its kept twin, and a monitor must be a statistic validation logs."""
import sys

import pytest

from esm_msr import config

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_a_retired_monitor_is_rewritten():
    a = parse('--monitor_metric', 'val_epi_beyond_lib_epi', '--mt_convergence_metric', 'epi_beyond_rho_epi')
    assert a.monitor_metric == 'val_epi_beyond_lib_comb' and a.mt_convergence_metric == 'epi_beyond_rho_comb'
    a = parse('--monitor_metric', 'val_unmasked_epi_pair_rho_epi')
    assert a.monitor_metric == 'val_unmasked_epi_pair_rho_comb'


def test_a_monitor_with_no_exact_twin_is_refused_but_a_plateau_metric_is_left_alone():
    for retired in ('val_epi_global_err_epi', 'val_epi_cell_rank_epi', 'val_unmasked_epi_cell_rank_epi'):
        with pytest.raises(SystemExit):
            parse('--monitor_metric', retired)
    assert parse('--mt_convergence_metric', 'epi_global_err_epi').mt_convergence_metric == 'epi_global_err_epi'   # read from the internal results


def test_a_monitor_that_validation_does_not_log_is_refused_at_parse_time():
    for gone in ('val_cycle_gap_avg', 'val_diag_epi_cell_rank_mt_legacy', 'val_epi_global_err_comb', 'val_rho_singles_combined_avg', 'val_typo_avg'):
        with pytest.raises(SystemExit):
            parse('--monitor_metric', gone)
    for ok in ('val_rho_combined_avg', 'val_rho_singles_wt_avg', 'val_unmasked_rho_combined_avg', 'val_masked_epi_composite', 'val_epi_bias_xgt1_comb'):
        assert parse('--monitor_metric', ok).monitor_metric == ok


def test_the_defaults_and_the_recipes_name_logged_metrics():
    a = parse()
    assert a.monitor_metric == 'val_epi_beyond_rho_comb' and a.mt_convergence_metric == 'epi_beyond_rho_comb'
    assert parse('--monitor_metric', 'val_epi_composite').monitor_metric == 'val_epi_composite'
    import os, yaml
    from esm_msr import val_stats
    for recipe in ('joint', 'joint_clean', 'mt_frozen', 'wt_phase'):
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', f'{recipe}.yaml')))
        assert cfg['monitor_metric'] in val_stats.logged_summary_names(), recipe
