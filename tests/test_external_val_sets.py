"""OOD validation benchmarks are tracked separately and kept out of every aggregate.

The three external sets (ptmuld, s461, ssym) are out of the training distribution, so including them in
``val_*_avg`` / the pooled global ranking / the epistasis levels made those numbers incomparable across runs and
inconsistent with what the model trained on. They are still scored per loader for diagnostics, and summarised as
``val_rho_combined_external`` / ``val_rmse_combined_external``.
"""
import sys

import numpy as np
import pytest

from esm_msr import config
from esm_msr.training import summarise_external_benchmarks

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_default_set_is_the_three_benchmarks():
    assert parse().external_val_sets == 'ptmuld,s461,ssym'


def test_it_can_be_overridden_or_emptied():
    assert parse('--external_val_sets', 's461').external_val_sets == 's461'
    assert parse('--external_val_sets', '').external_val_sets == ''


def test_summary_uses_only_the_external_loaders():
    per_loader = {
        'ptmuld': {'rho_combined': 0.6, 'rmse_combined': 1.0},
        's461': {'rho_combined': 0.8, 'rmse_combined': 1.4},
        '1A0N': {'rho_combined': 0.1, 'rmse_combined': 9.0},     # in-distribution: must not enter
        '2K9D': {'rho_combined': 0.2, 'rmse_combined': 9.0},
    }
    out = summarise_external_benchmarks(per_loader, {'ptmuld', 's461', 'ssym'})
    assert out['val_rho_combined_external'] == pytest.approx(0.7)
    assert out['val_rmse_combined_external'] == pytest.approx(1.2)


def test_absent_or_nan_external_loaders_are_skipped():
    per_loader = {
        'ptmuld': {'rho_combined': float('nan'), 'rmse_combined': float('nan')},
        's461': {'rho_combined': 0.8, 'rmse_combined': 1.4},
        '1A0N': {'rho_combined': 0.1, 'rmse_combined': 9.0},
    }
    out = summarise_external_benchmarks(per_loader, {'ptmuld', 's461', 'ssym'})
    assert out['val_rho_combined_external'] == pytest.approx(0.8)
    assert out['val_rmse_combined_external'] == pytest.approx(1.4)


def test_no_external_loader_means_no_external_metric():
    assert summarise_external_benchmarks({'1A0N': {'rho_combined': 0.5}}, {'ptmuld'}) == {}


def test_the_external_summary_differs_from_an_all_loader_mean():
    """The point of the change: an aggregate over every loader is NOT the reported average any more."""
    per_loader = {'ptmuld': {'rho_combined': 0.0}, '1A0N': {'rho_combined': 0.9}}
    ext = summarise_external_benchmarks(per_loader, {'ptmuld'})['val_rho_combined_external']
    all_mean = np.mean([v['rho_combined'] for v in per_loader.values()])
    assert ext == pytest.approx(0.0) and all_mean == pytest.approx(0.45)
