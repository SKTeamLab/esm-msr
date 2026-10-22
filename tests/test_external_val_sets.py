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
from esm_msr.link import SaturatingLink
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


# ---------------------------------------------------------------------------------------------------------------
# The end-to-end guard. The pure helper above was tested, but the method that CALLS it was not - which is how a
# missing `ext_per_loader = {}` reached a 7-minute run and killed it at the end of its first validation epoch with
# `NameError: name 'ext_per_loader' is not defined` (2026-10-19 12:22, wt_ref_floor0_r16a16_d02). This drives
# on_validation_epoch_end itself, with one in-distribution and one OOD loader, and asserts the contract both ways.
def _bare_module():
    """A LightningModule shell with only the attributes on_validation_epoch_end reads (no ESM3, no GPU)."""
    import torch
    from types import SimpleNamespace
    from esm_msr.training import ESM3EpistasisLightningModule

    obj = object.__new__(ESM3EpistasisLightningModule)
    torch.nn.Module.__init__(obj)
    obj.logged = {}
    obj.log = lambda k, v, **kw: obj.logged.setdefault(k, v)
    obj._VAL_PROGRESS_BAR = set()
    obj.external_val_sets = {'ptmuld'}
    obj.val_dataloader_names = ['LIB1', 'ptmuld']
    obj._hparams = SimpleNamespace(freeze_wt_after_epoch=99, lr_warmup_steps=0, val_dump=False)
    obj._trainer = SimpleNamespace(sanity_checking=True, current_epoch=0, global_step=0)
    obj._fabric = None
    obj.link_head = SaturatingLink()
    obj._mut_keys = lambda pdb, muts: [('lib', str(i)) for i in range(len(muts))]
    return obj


def _loader_output(pdb, n=24, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.normal(size=n)
    wt = gt * 0.8 + shift + rng.normal(scale=0.2, size=n)
    mt = gt * 0.5 + shift + rng.normal(scale=0.2, size=n)
    return dict(wt_scores=wt, mt_scores=mt, comb_scores=0.5 * (wt + mt), ground_truths=gt,
                cens=np.zeros(n), dddG_truths=gt, cens_src=np.zeros(n),
                subset_types=['single'] * n, pdb=pdb, mutations=['A12G'] * n,
                dG_wt=np.zeros(n), y_floor=np.zeros(n), wt_rev_scores=wt, mt_fwd_scores=mt)


def test_validation_epoch_end_runs_with_an_ood_loader_and_keeps_it_out_of_the_average():
    from esm_msr.training import ESM3EpistasisLightningModule

    obj = _bare_module()
    obj.validation_step_outputs = {0: [_loader_output('LIB1', seed=0)],
                                   1: [_loader_output('ptmuld', shift=0.5, seed=1)]}
    ESM3EpistasisLightningModule.on_validation_epoch_end(obj)     # raised NameError before the fix

    got = obj.logged
    assert 'val_rho_combined_external' in got and 'val_rho_combined/LIB1' in got
    assert got['val_rho_combined_avg'] == pytest.approx(got['val_rho_combined/LIB1'])
    assert got['val_rho_combined_external'] == pytest.approx(got['val_rho_combined/ptmuld'])
    # ...and the two are genuinely different here, so the assertion above is not vacuous.
    assert abs(got['val_rho_combined/LIB1'] - got['val_rho_combined/ptmuld']) > 1e-6
