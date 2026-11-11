"""The statistic that picks checkpoints is not a flag: ``config.monitor_metric`` reads the validation regime the MT head trained on, and it is always one validation logs."""
import os

import pytest
import yaml

from esm_msr import config, val_stats


def test_the_monitored_regime_is_the_one_the_mt_head_trained_on():
    assert config.monitor_metric({'block_mask_mt': 1.0}) == 'val_rho_combined_avg'
    assert config.monitor_metric({'block_mask_mt': 0.0}) == 'val_unmasked_rho_combined_avg'
    assert config.monitor_metric({'block_mask_mt': 0.5}) == 'val_unmasked_rho_combined_avg'      # visible at least half the time
    assert config.monitor_metric({'block_mask_mt': 0.75}) == 'val_rho_combined_avg'
    assert config.monitor_metric({}) == 'val_rho_combined_avg'                                  # the default is the masked read


@pytest.mark.parametrize('p', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_inferred_monitor_is_a_statistic_validation_logs(p):
    assert config.monitor_metric({'block_mask_mt': p}) in val_stats.logged_summary_names()


def test_the_recipe_parses_and_monitors_the_unmasked_regime():
    a = config.parse_arguments(['--config', os.path.join(os.path.dirname(__file__), '..', 'configs', 'recipe.yaml'), '--experiment_name', 'x'])
    assert config.monitor_metric(a) == 'val_unmasked_rho_combined_avg'
    assert yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'recipe.yaml')))['block_mask_mt'] == 0.0
