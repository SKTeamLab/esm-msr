"""``--censor_threshold``: one number that makes an item a bound. The estimator of per-library floors, the protease-gap rule and the
separate threshold for doubles were removed (docs/trim_audit.md §2.2); their tests went with them."""
import numpy as np
import pytest

from esm_msr import censoring


def _item(ddG, dG_meas, subset='single'):
    return {'ddG': ddG, 'dG_meas': dG_meas, 'subset_type': subset}


def test_item_at_or_below_the_threshold_is_lower_censored():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6), _item(-1.0, 1.5), _item(-2.0, 0.5)], threshold=0.5)
    assert items[0]['cens'] == censoring.CENS_LO and items[0]['cens_src'] == censoring.SRC_FLOOR
    assert items[1]['cens'] == 0
    assert items[2]['cens'] == censoring.CENS_LO                     # AT the threshold counts
    assert counts['floor'] == 2


def test_one_threshold_serves_singles_and_doubles():
    items, _ = censoring.apply_item_censoring([_item(-3.0, 0.3, 'single'), _item(-3.0, 0.3, 'double'), _item(-3.0, 0.3, 'native_cond')],
                                              threshold=0.5)
    assert [i['cens'] for i in items] == [censoring.CENS_LO] * 3


def test_bound_is_on_the_items_own_ddG_scale():
    """dG_meas is the measured dG of the state the label describes, so the bound shifts by the same offset the label does."""
    items, _ = censoring.apply_item_censoring([_item(-4.0, -0.6, 'double'), _item(-1.5, -0.6, 'cond')], threshold=0.5)
    assert items[0]['cens_bound'] == pytest.approx(0.5 + (-4.0 - -0.6))
    assert items[1]['cens_bound'] == pytest.approx(0.5 + (-1.5 - -0.6))
    assert items[1]['cens_bound'] > items[0]['cens_bound']


def test_no_threshold_censors_nothing():
    items, counts = censoring.apply_item_censoring([_item(-3.5, -0.6)], threshold=None)
    assert items[0]['cens'] == 0 and counts['floor'] == 0
    assert not np.isfinite(items[0]['cens_bound'])


def test_item_without_a_measured_dG_censors_nothing():
    items, counts = censoring.apply_item_censoring([_item(-3.5, float('nan'))], threshold=0.5)
    assert items[0]['cens'] == 0 and counts['floor'] == 0


def test_the_protease_gap_no_longer_censors():
    it = {'ddG': -1.0, 'dG_meas': 3.0, 'subset_type': 'single', 'protease_gap': 9.0}
    items, counts = censoring.apply_item_censoring([it], threshold=0.5)
    assert items[0]['cens'] == 0 and 'gap' not in counts


def test_auroc_basics():
    assert censoring.auroc([3, 4], [1, 2]) == pytest.approx(1.0)
    assert censoring.auroc([1, 2], [3, 4]) == pytest.approx(0.0)
    assert censoring.auroc([1, 2], [1, 2]) == pytest.approx(0.5)
    assert not np.isfinite(censoring.auroc([], [1]))
