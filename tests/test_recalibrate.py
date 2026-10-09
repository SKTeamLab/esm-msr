"""esm_msr_testing.recalibrate: a linear fit recovers a known scale and offset, keeps the latent columns, transforms dddG exactly; the
nonlinear fit beats the linear one on saturated data and is monotone within a library."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference_scripts'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def frame(seed=0, floor=None):
    rng = np.random.default_rng(seed)
    a = rng.normal(-1.0, 1.2, 40)
    rows = [dict(code='L', mut_type=f'A{i + 1}G', lat=a[i]) for i in range(40)]
    for i in range(0, 40, 2):
        rows.append(dict(code='L', mut_type=f'A{i + 1}G:A{i + 2}G', lat=a[i] + a[i + 1]))
    df = pd.DataFrame(rows)
    true = 2.0 * df.lat + 0.5
    if floor is not None:
        true = np.maximum(true, floor + 0.05 * (true - floor))                 # a saturating assay
    df['ddG_ML'] = true + rng.normal(0, 0.05, len(df))
    df['combined_pred'] = df.lat
    df['combined_dddg_pred'] = np.where(df.mut_type.str.contains(':'), 0.0, np.nan)
    return df


class TestRecalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr_testing as T
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f'esm_msr_testing not importable: {e}')
        cls.T = T

    def test_linear_recovers_scale_and_offset(self):
        out, fits = self.T.recalibrate(frame(), 'linear', 'ddG_ML')
        f = fits['combined_pred']
        self.assertAlmostEqual(f['s'], 2.0, delta=0.05)
        self.assertAlmostEqual(f['b'], 0.5, delta=0.05)
        self.assertLess(f['rmse_after'], f['rmse_before'])
        self.assertIn('combined_pred_uncal', out)
        dbl = out.mut_type.str.contains(':')
        np.testing.assert_allclose(out.combined_dddg_pred[dbl], -f['b'])        # s * 0 + (1 - 2) * b

    def test_nonlinear_beats_linear_on_saturated_data_and_keeps_order(self):
        df = frame(1, floor=-1.0)
        _, lin = self.T.recalibrate(df, 'linear', 'ddG_ML')
        out, non = self.T.recalibrate(df, 'nonlinear', 'ddG_ML')
        self.assertLess(non['combined_pred']['rmse_after'], lin['combined_pred']['rmse_after'])
        o = np.argsort(out.combined_pred_uncal.to_numpy())
        self.assertTrue(np.all(np.diff(out.combined_pred.to_numpy()[o]) >= -1e-9))


if __name__ == '__main__':
    unittest.main()
