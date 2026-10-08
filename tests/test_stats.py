import unittest
import numpy as np
from esm_msr import stats


class TestComputeMetrics(unittest.TestCase):
    def test_the_ddG_metrics_and_nothing_epistatic(self):
        wt = np.array([1.0, 2.0, 1.5, 3.5])
        mt = np.array([1.0, 2.0, 2.5, 5.5])
        m = stats.compute_metrics(wt, mt, 0.5 * wt + 0.5 * mt, np.array([1.1, 1.9, 2.1, 4.4]), ['single', 'single', 'double', 'double'])
        self.assertEqual(set(m), {'rho_wt_valid', 'rho_wt_all', 'rho_mt_valid', 'rho_mt_all', 'rho_combined', 'rmse_combined'})
        self.assertGreater(m['rho_combined'], 0.9)
        self.assertTrue(np.isnan(m['rho_mt_valid']))           # no conditional items


class TestDeltaSingleDiagnostics(unittest.TestCase):
    def _df(self, seed=0, nonadd_wt=0.0):
        import pandas as pd
        rng = np.random.default_rng(seed)
        sing = {f'{c}{i}X': (rng.normal(), rng.normal()) for c in 'AB' for i in range(1, 7)}  # (wt, mt)
        rows = [dict(mut_type=k, wt_lora_pred=w, mt_lora_pred=m) for k, (w, m) in sing.items()]
        names = list(sing)
        for a in names[:6]:
            for b in names[6:]:
                (wa, ma), (wb, mb) = sing[a], sing[b]
                wt_ab = wa + wb - 0.2 + nonadd_wt * rng.normal()        # WT head: additive up to a constant
                mt_ab = ma + mb + rng.normal() * 0.5
                comb_ab = 0.5 * (wt_ab + mt_ab)
                comb_add = 0.5 * (wa + wb + ma + mb)
                rows.append(dict(mut_type=f'{a}:{b}', wt_lora_pred=wt_ab, mt_lora_pred=mt_ab,
                                 wt_lora_dddg_pred=wt_ab - wa - wb, mt_lora_dddg_pred=mt_ab - ma - mb,
                                 combined_dddg_pred=comb_ab - comb_add, dddG=rng.normal()))
        return pd.DataFrame(rows)

    def test_algebraic_identity_and_wt_additivity(self):
        out = stats.delta_single_diagnostics(self._df(), 'dddG')
        self.assertEqual(out['n_singles'], 12)
        self.assertEqual(out['n_doubles_paired'], 36)
        self.assertLess(out['identity_resid'], 1e-9)      # E_fast - E_full = 0.5*(dA+dB) - dW
        self.assertLess(out['dW_sd'], 1e-9)               # WT head additive
        self.assertGreater(out['delta_term_share'], 0)

    def test_identical_heads_make_the_readouts_agree(self):
        df = self._df()
        df['mt_lora_pred'] = np.where(df['mut_type'].str.contains(':'), df['mt_lora_pred'], df['wt_lora_pred'])
        # re-derive the double-level dddg columns for the modified singles
        sing = df[~df['mut_type'].str.contains(':')].set_index('mut_type')
        for i, r in df[df['mut_type'].str.contains(':')].iterrows():
            a, b = r['mut_type'].split(':')
            df.at[i, 'mt_lora_dddg_pred'] = r['mt_lora_pred'] - sing.at[a, 'mt_lora_pred'] - sing.at[b, 'mt_lora_pred']
            df.at[i, 'combined_dddg_pred'] = 0.5 * (r['wt_lora_dddg_pred'] + df.at[i, 'mt_lora_dddg_pred'])
        out = stats.delta_single_diagnostics(df, 'dddG')
        self.assertAlmostEqual(out['delta_sd'], 0.0, places=9)
        self.assertLess(out['identity_resid'], 1e-9)
        self.assertAlmostEqual(out['delta_term_share'], 0.0, places=9)

    def test_missing_columns_give_nan(self):
        out = stats.delta_single_diagnostics(self._df().drop(columns=['combined_dddg_pred']), 'dddG')
        self.assertFalse(np.isnan(out['delta_sd']))
        self.assertTrue(np.isnan(out['identity_resid']))


class TestPairLevelFlip(unittest.TestCase):
    """The per-pair flip metric credits only what depends on the partner RESIDUE."""

    def _data(self, seed=0):
        rng = np.random.default_rng(seed)
        rows = list(range(8))
        partners = {10: 'ACDEFGHI', 20: 'KLMNPQRS'}          # two partner positions, 8 residues each
        r = {q: rng.normal(0, 2.0, 8) for q in partners}       # substitution effect that depends on the partner POSITION only
        keys, rid, truth, only_pos, eps_only = [], [], [], [], []
        for q, aas in partners.items():
            for b in aas:
                for a in rows:
                    e = rng.normal(0, 0.3)
                    keys.append(f'LIB|5|{q}{b}')
                    rid.append(a)
                    truth.append(r[q][a] + e)
                    only_pos.append(r[q][a])                    # knows a and q, not b
                    eps_only.append(e)                          # knows only the interaction
        return keys, np.array(rid), np.array(truth), np.array(only_pos), np.array(eps_only)

    def test_partner_residue_ignorant_predictor_scores_only_on_the_pooled_metric(self):
        keys, rid, truth, only_pos, _ = self._data()
        pooled, _, _ = stats.flip_signature_rho(only_pos, truth, keys, rid, min_len=4)
        pair, n_pair, _ = stats.flip_signature_rho(only_pos, truth, keys, rid, min_len=4, by_partner_position=True)
        self.assertGreater(pooled, 0.15)             # earns credit for a x partner-position structure
        self.assertEqual(pair, 0.0)                  # one matrix per pair: constant along every row, signature flat
        self.assertEqual(n_pair, 2)

    def test_true_interaction_is_credited_by_both(self):
        keys, rid, truth, only_pos, eps = self._data()
        for by_pos in (False, True):
            rho, _, _ = stats.flip_signature_rho(truth, truth, keys, rid, min_len=4, by_partner_position=by_pos)
            self.assertAlmostEqual(rho, 1.0, places=6)
        # the interaction alone, on top of the position-specific part, still correlates strongly with the truth in a matrix
        rho, _, _ = stats.flip_signature_rho(only_pos + eps, truth, keys, rid, min_len=4, by_partner_position=True)
        self.assertAlmostEqual(rho, 1.0, places=6)

    def test_keys_without_a_partner_position_keep_the_default_grouping(self):
        keys = [f'LIB|5|native' for _ in range(20)]
        rid = np.tile(np.arange(5), 4)
        # all 'native' keys collapse into a single column per scored position: too few columns either way
        rho_a, *_ = stats.flip_signature_rho(np.arange(20.), np.arange(20.), keys, rid, min_len=4)
        rho_b, *_ = stats.flip_signature_rho(np.arange(20.), np.arange(20.), keys, rid, min_len=4, by_partner_position=True)
        self.assertEqual(str(rho_a), str(rho_b))


class TestObservedScaleMetrics(unittest.TestCase):
    def test_rmse_uses_the_observed_scale_and_ranks_the_latent_scale(self):
        # two singles and a double; the double is far below the floor, so its LATENT is much lower than what is measured
        sub = ['single', 'single', 'double']
        gt = np.array([-2.0, -2.5, -3.0])             # measured ddG (saturated at dG -1 with dG_wt 2.0 -> -3.0 floor)
        latent = np.array([-2.0, -2.5, -9.0])         # latent: the double is predicted 9 kcal/mol less stable
        obs = np.array([-2.0, -2.5, -3.0])            # what the assay would report for those latents
        m = stats.compute_metrics(latent, latent, latent, gt, sub, obs={'comb': obs})
        self.assertAlmostEqual(m['rmse_combined'], 0.0, places=9)           # the saturated measurement is matched
        plain = stats.compute_metrics(latent, latent, latent, gt, sub)
        self.assertGreater(plain['rmse_combined'], 3.0)                      # the latent scale is penalised for it
        self.assertAlmostEqual(m['rho_combined'], plain['rho_combined'])     # rank metrics are untouched


if __name__ == '__main__':
    unittest.main()
