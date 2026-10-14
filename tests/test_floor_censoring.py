"""The per-library floor estimate and the censoring it drives."""
import numpy as np
import pandas as pd
import pytest

from esm_msr import censoring
from esm_msr.data import library_floor


def _df(rows):
    """rows: (mut_type, dG_ML)"""
    return pd.DataFrame(rows, columns=['mut_type', 'dG_ML'])


# --------------------------------------------------------------------------- the percentile estimate

def test_percentile_is_the_nth_percentile_of_numeric_dG():
    dG = list(np.linspace(-1.0, 4.0, 101))
    out = library_floor(_df([(f'A{i}G', v) for i, v in enumerate(dG)]), percentile=5.0)
    assert out['y_floor'] == pytest.approx(np.percentile(dG, 5.0))
    assert out['n_dG'] == 101
    assert out['frac_cens_lo'] == 0.0


def test_percentile_follows_the_requested_N():
    dG = list(np.linspace(-1.0, 4.0, 101))
    df = _df([(f'A{i}G', v) for i, v in enumerate(dG)])
    assert library_floor(df, percentile=1.0)['y_floor'] < library_floor(df, percentile=25.0)['y_floor']


def test_censoring_tags_are_excluded_from_the_percentile_but_counted():
    """'<-1' rows have no number; they must not be read as values, and their share is reported."""
    rows = [('A1G', -0.5), ('A2G', 0.0), ('A3G', 1.0), ('A4G', '<-1'), ('A5G', '<-1'), ('A6G', '-')]
    out = library_floor(_df(rows), percentile=5.0)
    assert out['n_dG'] == 3
    assert out['frac_cens_lo'] == pytest.approx(2 / 6)
    assert out['y_floor'] >= -0.5          # driven by the numeric rows only


def test_estimate_tracks_library_composition_not_only_the_assay():
    """The known weakness, pinned so it cannot change silently: the same floor, two sensitivities, two estimates.

    Both libraries are measured by an assay whose floor is -1. The mild one never reaches it, so its 5th
    percentile sits far above the floor; the saturated one piles up on it. A good floor estimator would return
    about -1 for both.
    """
    mild = _df([(f'A{i}G', v) for i, v in enumerate(np.linspace(1.0, 4.0, 100))])
    saturated = _df([(f'A{i}G', v) for i, v in enumerate([-1.0] * 50 + list(np.linspace(0.0, 4.0, 50)))])
    assert library_floor(mild, percentile=5.0)['y_floor'] > 1.0
    assert library_floor(saturated, percentile=5.0)['y_floor'] == pytest.approx(-1.0, abs=0.05)


def test_doubles_conditioned_estimate_recovers_the_floor_where_the_percentile_does_not():
    """The alternative recorded alongside: conditioning on an additive expectation far below the floor finds the
    floor itself even in a library whose measured tail is mostly healthy."""
    dG_wt = 3.0
    rows = [('A1G', 1.0), ('A2G', 1.0)]                      # two singles, each ddG = -2
    rows += [(f'B{i}G', v) for i, v in enumerate(np.linspace(1.5, 4.0, 60))]
    # The double of the two: additive expectation 3 + (-2) + (-2) = -1 < -1.5 is false, so make them harsher.
    rows = [('A1G', 0.0), ('A2G', 0.0)] + rows[2:]           # ddG = -3 each -> x = -3
    rows += [('A1G:A2G', -0.95)]                             # reported at the floor
    out = library_floor(_df(rows), percentile=5.0, dG_wt=dG_wt)
    assert out['n_floor_cond'] == 1
    assert out['floor_cond'] == pytest.approx(-0.95)
    # The percentile, by contrast, is dragged up by the healthy majority.
    assert out['y_floor'] > out['floor_cond']


def test_doubles_conditioned_estimate_is_nan_without_dG_wt():
    rows = [('A1G', 0.0), ('A2G', 0.0), ('A1G:A2G', -0.95)]
    assert not np.isfinite(library_floor(_df(rows), dG_wt=float('nan'))['floor_cond'])


def test_missing_dG_column_yields_no_estimate():
    out = library_floor(pd.DataFrame({'mut_type': ['A1G'], 'ddG': [0.5]}))
    assert not np.isfinite(out['y_floor'])


# --------------------------------------------------------------------------- censoring driven by the floor

def _item(ddG, dG_meas, y_floor, subset='single'):
    return {'ddG': ddG, 'dG_meas': dG_meas, 'y_floor': y_floor, 'subset_type': subset}


def test_item_at_or_below_the_floor_is_lower_censored():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6, -0.5), _item(-1.0, 1.5, -0.5)])
    assert items[0]['cens'] == censoring.CENS_LO
    assert items[0]['cens_src'] == censoring.SRC_FLOOR
    assert items[1]['cens'] == 0
    assert counts['floor'] == 1
    assert dict(counts['floor_by_subset']) == {'single': 1}


def test_bound_is_on_the_items_own_ddG_scale():
    """dG_meas is the measured dG of the state the label describes, so the bound shifts by the same offset the
    label does: a conditional item derived from a floored double is bounded its partner's single higher."""
    dbl = _item(-4.0, -0.6, -0.5, 'double')                   # ddG_AB = -4.0, dG_AB = -0.6
    cond = _item(-1.5, -0.6, -0.5, 'cond')                    # ddG(A|B) = ddG_AB - ddG_B = -1.5
    items, _ = censoring.apply_item_censoring([dbl, cond])
    assert items[0]['cens_bound'] == pytest.approx(-0.5 + (-4.0 - -0.6))
    assert items[1]['cens_bound'] == pytest.approx(-0.5 + (-1.5 - -0.6))
    assert items[1]['cens_bound'] > items[0]['cens_bound']


def test_no_censoring_when_disabled():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6, -0.5)], censor_floor=False)
    assert items[0]['cens'] == 0
    assert counts['floor'] == 0
    assert not np.isfinite(items[0]['cens_bound'])


def test_library_without_a_floor_estimate_censors_nothing():
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
