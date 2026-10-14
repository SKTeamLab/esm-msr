"""The per-library floor: when we believe a library has one, where it is, and the censoring it drives."""
import numpy as np
import pandas as pd
import pytest

from esm_msr import censoring
from esm_msr.data import floor_evidence, library_floor, shrink_floors


def _df(rows):
    """rows: (mut_type, dG_ML)"""
    return pd.DataFrame(rows, columns=['mut_type', 'dG_ML'])


def _pinned_library(floor=-0.3, n_deep=30, spread=0.05, dG_wt=3.0, n_healthy=60, seed=0):
    """A library whose deeply destabilised doubles all pile up at one value: a real floor."""
    rng = np.random.default_rng(seed)
    rows = [('A1G', dG_wt - 3.0), ('A2G', dG_wt - 3.0)]            # two singles, each ddG = -3 -> x = dG_wt - 6
    rows += [(f'H{i}G', v) for i, v in enumerate(np.linspace(dG_wt - 1.0, dG_wt, n_healthy))]
    rows += [(f'A1G:A2G_{i}'.replace(f'_{i}', ''), v) for i, v in
             zip(range(n_deep), rng.normal(floor, spread, n_deep))]
    return rows


# --------------------------------------------------------------------------- the evidence test

def test_accepts_a_tight_cluster_of_deeply_destabilised_variants():
    dG = np.array([-0.30, -0.28, -0.33, -0.31] * 6)
    x = np.full(len(dG), -3.0)
    ev = floor_evidence(dG, x, dG_wt=3.0)
    assert ev['evidence'] == 1.0
    assert ev['floor'] == pytest.approx(np.median(dG), abs=1e-6)
    assert ev['n'] == len(dG)


def test_rejects_too_few_expected_pinned_variants():
    dG = np.array([-0.3] * 5)
    assert floor_evidence(dG, np.full(5, -3.0), dG_wt=3.0)['evidence'] == 0.0


def test_rejects_a_spread_out_cluster():
    """Variants predicted far below the floor that come back spread over 2 kcal/mol are being measured, not pinned."""
    rng = np.random.default_rng(1)
    dG = rng.normal(-0.3, 1.0, 40)
    assert floor_evidence(dG, np.full(40, -3.0), dG_wt=3.0)['evidence'] == 0.0


def test_rejects_a_cluster_outside_the_assay_band():
    """A tight cluster of deeply destabilised variants reading +1.5 is compensation, not a floor."""
    dG = np.full(40, 1.5) + np.random.default_rng(2).normal(0, 0.05, 40)
    ev = floor_evidence(dG, np.full(40, -3.0), dG_wt=3.0)
    assert ev['evidence'] == 0.0
    assert np.isfinite(ev['median'])        # still reported, just not believed


def test_rejects_without_dG_wt():
    """No dG_wt means no additive expectation, so there is no way to know which variants should be pinned."""
    dG = np.full(40, -0.3)
    assert floor_evidence(dG, np.full(40, -3.0), dG_wt=float('nan'))['evidence'] == 0.0


def test_only_variants_expected_below_x_max_count():
    dG = np.full(40, -0.3)
    assert floor_evidence(dG, np.full(40, +2.0), dG_wt=3.0)['evidence'] == 0.0      # none are expected dead


# --------------------------------------------------------------------------- library_floor

def test_pinned_library_gets_a_floor():
    out = library_floor(_df(_pinned_library(floor=-0.3)), dG_wt=3.0, name='pinned')
    assert out['evidence'] == 1.0
    assert out['y_floor'] == pytest.approx(-0.3, abs=0.05)
    assert out['n_deep'] >= 20 and out['sd_deep'] < 0.5


def test_mild_library_gets_no_floor_and_would_have_been_given_a_bogus_percentile():
    """The case the evidence gate exists for: a library of mild variants has no floor population, but its 5th
    percentile is a perfectly healthy dG that censoring would have treated as the floor."""
    mild = _df([(f'A{i}G', v) for i, v in enumerate(np.linspace(1.0, 4.0, 100))])
    out = library_floor(mild, percentile=5.0, dG_wt=4.0, name='mild')
    assert out['evidence'] == 0.0
    assert not np.isfinite(out['y_floor'])
    assert out['floor_pctl'] > 1.0                      # what the percentile estimator would have used


def test_percentile_is_still_reported_for_comparison():
    out = library_floor(_df(_pinned_library()), percentile=5.0, dG_wt=3.0)
    assert np.isfinite(out['floor_pctl'])
    assert np.isfinite(out['n_dG']) and out['n_dG'] > 0


def test_missing_dG_column_yields_nothing():
    out = library_floor(pd.DataFrame({'mut_type': ['A1G'], 'ddG': [0.5]}))
    assert not np.isfinite(out['y_floor']) and out['evidence'] == 0.0


# --------------------------------------------------------------------------- shrinkage and the fallback

def test_shrinkage_pulls_thin_estimates_toward_the_dataset_floor():
    stats = {
        'big':  {'y_floor': 0.00, 'n_deep': 500.0, 'evidence': 1.0},
        'thin': {'y_floor': -0.90, 'n_deep': 20.0, 'evidence': 1.0},
        'mid':  {'y_floor': -0.10, 'n_deep': 100.0, 'evidence': 1.0},
    }
    glob = shrink_floors(stats, shrink=5.0)
    assert glob == pytest.approx(-0.10)
    assert stats['big']['y_floor'] == pytest.approx(0.0, abs=0.002)        # barely moved
    assert -0.90 < stats['thin']['y_floor'] < -0.10                        # pulled most of the way in
    assert abs(stats['thin']['y_floor'] - -0.90) > 0.1


def test_libraries_without_evidence_take_the_dataset_floor_as_a_knee_only():
    stats = {
        'a': {'y_floor': -0.2, 'n_deep': 100.0, 'evidence': 1.0},
        'b': {'y_floor': float('nan'), 'n_deep': 0.0, 'evidence': 0.0},
    }
    glob = shrink_floors(stats)
    assert stats['b']['y_floor'] == pytest.approx(glob)      # a sane knee for the link
    assert stats['b']['evidence'] == 0.0                     # but nothing may be censored against it


def test_shrinkage_with_no_accepted_floors_leaves_nan():
    stats = {'a': {'y_floor': float('nan'), 'n_deep': 0.0, 'evidence': 0.0}}
    assert not np.isfinite(shrink_floors(stats))
    assert not np.isfinite(stats['a']['y_floor'])


# --------------------------------------------------------------------------- censoring driven by the floor

def _item(ddG, dG_meas, y_floor, subset='single'):
    return {'ddG': ddG, 'dG_meas': dG_meas, 'y_floor': y_floor, 'subset_type': subset}


def test_item_at_or_below_the_floor_is_lower_censored():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6, -0.5), _item(-1.0, 1.5, -0.5)])
    assert items[0]['cens'] == censoring.CENS_LO
    assert items[0]['cens_src'] == censoring.SRC_FLOOR
    assert items[1]['cens'] == 0
    assert counts['floor'] == 1


def test_bound_is_on_the_items_own_ddG_scale():
    """dG_meas is the measured dG of the state the label describes, so the bound shifts by the same offset the
    label does: a conditional item derived from a floored double is bounded its partner's single higher."""
    dbl = _item(-4.0, -0.6, -0.5, 'double')
    cond = _item(-1.5, -0.6, -0.5, 'cond')
    items, _ = censoring.apply_item_censoring([dbl, cond])
    assert items[0]['cens_bound'] == pytest.approx(-0.5 + (-4.0 - -0.6))
    assert items[1]['cens_bound'] == pytest.approx(-0.5 + (-1.5 - -0.6))
    assert items[1]['cens_bound'] > items[0]['cens_bound']


def test_censoring_off_censors_nothing():
    """This is the path a library without floor evidence takes."""
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6, -0.5)], censor_floor=False)
    assert items[0]['cens'] == 0 and counts['floor'] == 0
    assert not np.isfinite(items[0]['cens_bound'])


def test_library_without_a_floor_value_censors_nothing():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6, float('nan'))])
    assert items[0]['cens'] == 0 and counts['floor'] == 0


def test_item_without_a_measured_dG_censors_nothing():
    items, counts = censoring.apply_item_censoring([_item(-3.5, float('nan'), -0.5)])
    assert items[0]['cens'] == 0 and counts['floor'] == 0


def test_auroc_basics():
    assert censoring.auroc([3, 4], [1, 2]) == pytest.approx(1.0)
    assert censoring.auroc([1, 2], [3, 4]) == pytest.approx(0.0)
    assert censoring.auroc([1, 2], [1, 2]) == pytest.approx(0.5)
    assert not np.isfinite(censoring.auroc([], [1]))


# --------------------------------------------------------------------------- the two-protease source

def _gitem(ddG, dG_meas, gap, y_floor=float('nan'), subset='single'):
    return {'ddG': ddG, 'dG_meas': dG_meas, 'y_floor': y_floor, 'protease_gap': gap, 'subset_type': subset}


def test_protease_disagreement_censors_against_the_items_own_value():
    """No library floor is needed: a variant whose two fits disagree is bounded by what it itself reported."""
    items, counts = censoring.apply_item_censoring([_gitem(-2.5, -0.4, 6.0)], censor_floor=False, gap_max=2.0)
    assert items[0]['cens'] == censoring.CENS_LO
    assert items[0]['cens_src'] == censoring.SRC_GAP
    assert items[0]['cens_bound'] == pytest.approx(-2.5)      # its own ddG
    assert counts['gap'] == 1


def test_agreeing_proteases_are_left_alone():
    items, counts = censoring.apply_item_censoring([_gitem(-2.5, -0.4, 0.3)], censor_floor=False, gap_max=2.0)
    assert items[0]['cens'] == 0 and counts['gap'] == 0


def test_gap_source_can_be_disabled():
    items, counts = censoring.apply_item_censoring([_gitem(-2.5, -0.4, 6.0)], censor_floor=False, gap_max=None)
    assert items[0]['cens'] == 0 and counts['gap'] == 0


def test_the_two_sources_are_a_union():
    """One item reachable only by the floor, one only by the gap."""
    floor_only = _gitem(-3.0, -0.6, 0.2, y_floor=-0.5)
    gap_only = _gitem(-2.0, 1.5, 7.0, y_floor=-0.5)
    items, counts = censoring.apply_item_censoring([floor_only, gap_only], censor_floor=True, gap_max=2.0)
    assert items[0]['cens_src'] == censoring.SRC_FLOOR
    assert items[1]['cens_src'] == censoring.SRC_GAP
    assert counts['floor'] == 1 and counts['gap'] == 1


def test_floor_wins_when_both_apply():
    """The library's central pinning value is the more robust bound, so it is checked first."""
    both = _gitem(-3.0, -0.6, 7.0, y_floor=-0.5)
    items, _ = censoring.apply_item_censoring([both], censor_floor=True, gap_max=2.0)
    assert items[0]['cens_src'] == censoring.SRC_FLOOR
    assert items[0]['cens_bound'] == pytest.approx(-0.5 + (-3.0 - -0.6))


def test_gap_needs_a_measured_dG():
    items, counts = censoring.apply_item_censoring([_gitem(-2.5, float('nan'), 6.0)], censor_floor=False, gap_max=2.0)
    assert items[0]['cens'] == 0 and counts['gap'] == 0
