"""Validation drops libraries in which no item has a numeric ddG (every item only a bound)."""
import unittest

from esm_msr.preprocess_megascale import has_numeric_values

NAN = float('nan')


class TestHasNumericValues(unittest.TestCase):
    def test_all_bounds_is_false(self):
        self.assertFalse(has_numeric_values([{'ddG': 3.0, 'cens': 1}, {'ddG': -4.0, 'cens': -1}]))

    def test_one_measurement_is_enough(self):
        self.assertTrue(has_numeric_values([{'ddG': 3.0, 'cens': 1}, {'ddG': -0.5, 'cens': 0}]))

    def test_a_measurement_without_a_value_does_not_count(self):
        self.assertFalse(has_numeric_values([{'ddG': NAN, 'cens': 0}, {'ddG': 2.0, 'cens': 1}]))

    def test_items_without_a_censoring_field_are_measurements(self):
        self.assertTrue(has_numeric_values([{'ddG': 0.3}]))

    def test_empty_is_false(self):
        self.assertFalse(has_numeric_values([]))


if __name__ == '__main__':
    unittest.main()
