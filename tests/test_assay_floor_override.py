"""--assay_floor replaces every library's measured knee with one asserted value, for the link AND for censoring.

Both read item['y_floor'], so overriding it in _stamp_floor moves them together; forcing has_floor_evidence is
what extends censoring to the 364 of 411 libraries that have no floor of their own. Without that flag the knee
would move while the censored population stayed put - the inconsistent half-state."""
import types
import unittest

import numpy as np


def ds(items, floor_stats, assay_floor):
    from esm_msr.data import MutationStabilityDataset as D
    d = types.SimpleNamespace(data=items, floor_stats=floor_stats, assay_floor=assay_floor)
    d._stamp_floor = types.MethodType(D._stamp_floor, d)
    return d


def item(dG_meas, ddG=-1.0, gap=None):
    # 'meas' is a flat array indexed by data._MEAS_INDEX, not a mapping; None means "no raw columns" -> NaN.
    meas = None
    if gap is not None:
        from esm_msr.data import MEAS_COLUMNS, _MEAS_INDEX
        meas = [float('nan')] * len(MEAS_COLUMNS)
        meas[_MEAS_INDEX['deltaG_t']] = 2.0
        meas[_MEAS_INDEX['deltaG_c']] = 2.0 - gap
    return {'dG_meas': dG_meas, 'ddG': ddG, 'meas': meas}


class TestStampFloor(unittest.TestCase):
    def test_without_the_override_a_library_with_no_evidence_carries_nan(self):
        d = ds([item(0.5)], {'evidence': 0.0, 'y_floor': -0.3}, None)
        d._stamp_floor()
        self.assertTrue(np.isnan(d.data[0]['y_floor']))
        self.assertFalse(d.has_floor_evidence)

    def test_without_the_override_an_evidenced_library_carries_its_own_floor(self):
        d = ds([item(0.5)], {'evidence': 1.0, 'y_floor': -0.3}, None)
        d._stamp_floor()
        self.assertAlmostEqual(d.data[0]['y_floor'], -0.3, places=6)
        self.assertTrue(d.has_floor_evidence)

    def test_the_override_reaches_a_library_that_has_no_evidence(self):
        d = ds([item(0.5), item(-0.2)], {'evidence': 0.0, 'y_floor': float('nan')}, 0.0)
        d._stamp_floor()
        self.assertTrue(d.has_floor_evidence, "censoring would otherwise skip this library entirely")
        self.assertEqual([i['y_floor'] for i in d.data], [0.0, 0.0])
        self.assertEqual(d.y_floor_used, 0.0)

    def test_the_override_replaces_a_measured_floor_too(self):
        d = ds([item(0.5)], {'evidence': 1.0, 'y_floor': -0.641}, 0.0)
        d._stamp_floor()
        self.assertEqual(d.data[0]['y_floor'], 0.0)

    def test_it_still_stamps_the_protease_gap(self):
        it = item(0.5, gap=0.6)
        d = ds([it], {'evidence': 0.0}, 0.0)
        d._stamp_floor()
        self.assertAlmostEqual(it['protease_gap'], 0.6, places=6)


class TestCensoringFollowsTheOverride(unittest.TestCase):
    def test_a_zero_floor_censors_everything_at_or_below_zero(self):
        from esm_msr import censoring
        items = [{'dG_meas': -0.2, 'ddG': -2.0, 'y_floor': 0.0, 'protease_gap': 0.0},
                 {'dG_meas': 0.3, 'ddG': -1.0, 'y_floor': 0.0, 'protease_gap': 0.0}]
        out, counts = censoring.apply_item_censoring(items, True, 2.0)
        self.assertEqual(out[0]['cens'], censoring.CENS_LO)
        self.assertEqual(out[1]['cens'], 0)
        self.assertEqual(counts['floor'], 1)

    def test_with_the_library_floor_the_same_item_is_a_measurement(self):
        from esm_msr import censoring
        items = [{'dG_meas': -0.2, 'ddG': -2.0, 'y_floor': -0.641, 'protease_gap': 0.0}]
        out, counts = censoring.apply_item_censoring(items, True, 2.0)
        self.assertEqual(out[0]['cens'], 0, "-0.2 sits above a -0.641 floor, so it is a real reading")
        self.assertEqual(counts['floor'], 0)


if __name__ == '__main__':
    unittest.main()
