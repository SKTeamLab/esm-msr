import unittest
import numpy as np
from esm_msr import stats


class TestStatsEpistasis(unittest.TestCase):
    def test_compute_metrics_with_and_without_dddG(self):
        # 2 singles, 2 doubles
        wt_scores = np.array([1.0, 2.0, 1.5, 3.5])
        mt_scores = np.array([1.0, 2.0, 2.5, 5.5])
        comb_scores = 0.5 * wt_scores + 0.5 * mt_scores  # [1.0, 2.0, 2.0, 4.5]
        ground_truths = np.array([1.1, 1.9, 2.1, 4.4])
        subset_types = ['single', 'single', 'double', 'double']

        # Without dddG
        m_no_dddG = stats.compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types)
        self.assertTrue(np.isnan(m_no_dddG['rho_epi']))
        self.assertFalse(np.isnan(m_no_dddG['rho_combined']))

        # With dddG:
        # doubles: epi_pred = comb - wt = [0.5, 1.0]
        # ground truth dddG = [0.2, 0.4] -> perfect rank correlation = 1.0
        dddG = np.array([np.nan, np.nan, 0.2, 0.4])
        m_with_dddG = stats.compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, dddG=dddG)
        self.assertAlmostEqual(m_with_dddG['rho_epi'], 1.0, places=5)

    def test_compute_metrics_no_doubles_returns_nan_rho_epi(self):
        # Only singles
        wt_scores = np.array([1.0, 2.0])
        mt_scores = np.array([1.0, 2.0])
        comb_scores = np.array([1.0, 2.0])
        ground_truths = np.array([1.1, 1.9])
        subset_types = ['single', 'single']
        dddG = np.array([np.nan, np.nan])

        m = stats.compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, dddG=dddG)
        self.assertTrue(np.isnan(m['rho_epi']))


if __name__ == '__main__':
    unittest.main()
