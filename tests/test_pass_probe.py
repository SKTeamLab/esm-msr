"""
The validation pass probe (esm_msr.pass_probe): every way of reading a mutation, scored on what it gets right and on WHERE it puts its
non-additivity.

The synthetic library has a known answer: doubles are additive + a pair offset + a residue-residue interaction. The views are built to
carry exactly what their input regime allows - the wild-type view nothing, the pair-masked view a pair offset and a line effect but no
interaction, the conditional view the interaction - and the probe has to say so.
"""
import types
from collections import defaultdict

import numpy as np
import pytest

from esm_msr import pass_probe, training

N_PAIRS, ROWS, COLS = 9, list('ACDEF'), list('GHIKL')


def library(seed=0):
    rng = np.random.default_rng(seed)
    keys, st, gt, views_s, views_d = [], [], [], [], []
    truth_single = {}
    off = rng.normal(size=N_PAIRS)
    for p in range(N_PAIRS):
        i, j = 10 * p + 1, 10 * p + 5
        line = {a: rng.normal() * 0.5 for a in ROWS}                 # the pair-masked view's row effect: knows the pair, not the partner
        inter = rng.normal(size=(len(ROWS), len(COLS)))
        inter = inter - inter.mean(1, keepdims=True) - inter.mean(0, keepdims=True) + inter.mean()
        for a in ROWS:
            truth_single[('L', i, a)] = rng.normal()
        for b in COLS:
            truth_single[('L', j, b)] = rng.normal()
        for a in ROWS:
            keys.append((('L', i, a),)); st.append('single'); gt.append(truth_single[('L', i, a)])
        for b in COLS:
            keys.append((('L', j, b),)); st.append('single'); gt.append(truth_single[('L', j, b)])
        for ia, a in enumerate(ROWS):
            for ib, b in enumerate(COLS):
                add = truth_single[('L', i, a)] + truth_single[('L', j, b)]
                keys.append((('L', i, a), ('L', j, b))); st.append('double')
                gt.append(add + off[p] + inter[ia, ib])
                views_d.append((len(keys) - 1, add, off[p] + line[a], off[p] + inter[ia, ib]))
    n = len(keys)
    gt = np.array(gt)
    single_score = np.where(np.array(st) == 'single', gt, np.nan)
    u_wt, pair, end = single_score.copy(), single_score.copy(), single_score.copy()
    for k, add, pair_part, epi in views_d:
        u_wt[k] = add                       # additive by construction
        pair[k] = add + pair_part           # pair offset + line effect, no interaction
        end[k] = add + 2 * epi              # an end-of-path view: the sum of the two conditional effects
    dddG = np.full(n, np.nan)
    for k, add, _, epi in views_d:
        dddG[k] = epi
    items = {'mut_key': keys, 'subset_type': st, 'cens': np.zeros(n, int), 'ddG': gt, 'dddG': dddG, 'dG_wt': np.full(n, 2.0)}
    views = {'wt.u_wt': u_wt, 'wt.pair': pair, 'wt.u_mt': end, 'wt.i_mt': end,
             'mt.u_wt': u_wt, 'mt.pair': pair, 'mt.u_mt': end, 'mt.i_mt': end}
    return items, views


@pytest.fixture(scope='module')
def scored():
    items, views = library()
    return pass_probe.score(items, views, ['L'] * len(items['ddG']))


def test_every_view_and_trapezoid_is_scored(scored):
    assert set(scored) == set(pass_probe.VIEWS) | set(pass_probe.TRAPEZOIDS)
    assert len(scored) == 12
    for row in scored.values():
        assert set(row) == set(pass_probe.METRICS)


def test_the_wild_type_view_predicts_no_non_additivity(scored):
    r = scored['wt.u_wt']
    assert r['nonadd_sd'] < 1e-9
    assert np.isnan(r['nonadd_cell'])                   # no variance, so no shares
    assert r['cell_rank'] == 0.0


def test_the_pair_masked_view_carries_pair_and_line_but_no_interaction(scored):
    r = scored['wt.pair']
    assert r['nonadd_sd'] > 0.3
    assert r['nonadd_cell'] < 1e-9
    assert r['nonadd_pair'] + r['nonadd_line'] == pytest.approx(1.0)
    assert r['nonadd_line'] > 0.05
    assert r['cell_rank'] == 0.0                         # its order within a column does not depend on the partner


def test_the_conditional_view_carries_the_interaction(scored):
    r = scored['mt.i_mt']
    assert r['nonadd_cell'] > 0.5
    # not 1: an end view carries TWICE the interaction on top of the same additive part, so its within-column ranks differ from the truth's
    assert r['cell_rank'] > 0.7 and r['cell_rank'] > scored['wt.pair']['cell_rank'] + 0.5


def test_the_trapezoid_of_an_additive_start_and_a_conditional_end_is_the_double(scored):
    """(add + (add + 2 dddG)) / 2 = ddG_AB exactly."""
    assert scored['tz.u_wt+i_mt']['rho_double'] == pytest.approx(1.0)
    assert scored['tz.u_wt+i_mt']['rho_double'] > scored['wt.u_wt']['rho_double']


def test_singles_are_scored_per_library(scored):
    assert scored['wt.u_wt']['rho_single'] == pytest.approx(1.0)


def test_log_names_are_flat():
    assert pass_probe.log_name('tz.pair+i_mt', 'cell_rank') == 'probe_tz_pair_i_mt_cell_rank'
    assert pass_probe.log_name('wt.u_wt', 'rho_single') == 'probe_wt_u_wt_rho_single'


# ---------------- run_views ----------------

def test_run_views_reads_every_view_once_and_reuses_the_runs_own_passes():
    import torch
    calls = []

    class M:
        def forward_partitioned(self, batch, pass_type, mask_strategy=None, adapter=None):
            calls.append((adapter, pass_type, mask_strategy))
            return {'pred_calibrated': torch.full((len(batch['ddG']),), float(len(calls)))}
    batch = {'ddG': torch.zeros(5), 'x': torch.arange(5)}
    known = {('wt', 'wt', None): np.arange(5.0), ('mt', 'mt', None): -np.arange(5.0)}
    out = pass_probe.run_views(M(), batch, torch.tensor([0, 2, 3]), known)
    assert set(out) == set(pass_probe.VIEWS)
    assert len(calls) == 6 and set(calls) == {v for v in pass_probe.VIEWS.values()} - set(known)
    np.testing.assert_array_equal(out['wt.u_wt'], [0, np.nan, 2, 3, np.nan])
    np.testing.assert_array_equal(out['mt.u_mt'], [-0.0, np.nan, -2, -3, np.nan])
    assert np.isnan(out['wt.pair'][[1, 4]]).all() and np.isfinite(out['wt.pair'][[0, 2, 3]]).all()


# ---------------- through validation ----------------

class Stub:
    def __init__(self, every=2):
        self.hparams = {'val_pass_probe': every}
        self.trainer = types.SimpleNamespace(sanity_checking=False)


def test_the_probe_runs_on_the_zero_shot_validation_and_then_every_nth():
    s = Stub(every=2)
    seen = []
    for sanity in (True, False, False, False, False, False):
        s.trainer.sanity_checking = sanity
        training.ESM3EpistasisLightningModule.on_validation_epoch_start(s)
        seen.append(s._probe_now)
    assert seen == [False, True, False, True, False, True]          # sanity check, zs, e0, e1, e2, e3


def test_off_by_default():
    s = Stub(every=0)
    training.ESM3EpistasisLightningModule.on_validation_epoch_start(s)
    assert s._probe_now is False


def test_a_probe_epoch_logs_the_probe_and_dumps_the_views(tmp_path):
    items, views = library()
    n = len(items['ddG'])
    out = {'wt_scores': views['wt.u_wt'], 'mt_scores': views['mt.u_mt'], 'comb_scores': 0.5 * (views['wt.u_wt'] + views['mt.u_mt']),
           'ground_truths': items['ddG'], 'dddG': items['dddG'], 'dG_wt': items['dG_wt'], 'cens': items['cens'],
           'subset_type': items['subset_type'], 'flip_key': [''] * n, 'mut_key': items['mut_key'],
           **{f'probe_{v}': a for v, a in views.items()}}
    s = types.SimpleNamespace(link_head=None, hparams=types.SimpleNamespace(flip_list_min=4, val_cycle_passes=False),
                              val_dataloader_names=['lib'], logged={}, _probe_now=True, global_step=5, current_epoch=1,
                              # sanity_checking keeps the stand-in out of the early-stop and plateau code; the probe ignores it here
                              trainer=types.SimpleNamespace(sanity_checking=True, current_epoch=1),
                              logger=types.SimpleNamespace(log_dir=str(tmp_path)))
    for k in ('_VAL_PER_PROTEIN', '_VAL_AVG', '_VAL_PROGRESS_BAR'):
        setattr(s, k, getattr(training.ESM3EpistasisLightningModule, k))
    s.log = lambda name, value, **kw: s.logged.__setitem__(name, value)
    s.validation_step_outputs = defaultdict(list, {0: [out]})
    training.ESM3EpistasisLightningModule.on_validation_epoch_end(s)
    assert s.logged['val_probe_mt_i_mt_cell_rank'] > 0.7
    assert s.logged['val_probe_wt_pair_nonadd_cell'] < 1e-9
    d = np.load(tmp_path / 'val_dump_zs.npz')
    assert {f'lib|probe_{v}' for v in pass_probe.VIEWS} <= set(d.files)


def test_the_offline_script_scores_a_probe_dump_with_every_trapezoid(tmp_path):
    import os
    import subprocess
    import sys
    test_a_probe_epoch_logs_the_probe_and_dumps_the_views(tmp_path)          # writes val_dump_zs.npz through the real epoch end
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = tmp_path / 'probe.csv'
    r = subprocess.run([sys.executable, os.path.join(root, 'scripts', 'pass_probe_from_dump.py'), str(tmp_path / 'val_dump_zs.npz'),
                        '--csv', str(csv)], env={**os.environ, 'PYTHONPATH': os.path.join(root, 'src')}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    df = pd.read_csv(csv, index_col=0)
    assert len(df) == 8 + 16
    assert df.loc['tz.wt.u_wt+mt.i_mt', 'rho_double'] == pytest.approx(1.0)
    assert df.loc['wt.pair', 'nonadd_cell'] < 1e-9
