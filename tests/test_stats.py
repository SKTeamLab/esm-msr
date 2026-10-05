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
        self.assertTrue(np.isnan(m_no_dddG['rho_epi_fast']))
        self.assertFalse(np.isnan(m_no_dddG['rho_combined']))

        # With dddG:
        # doubles: epi_pred = comb - wt = [0.5, 1.0]
        # ground truth dddG = [0.2, 0.4] -> perfect rank correlation = 1.0
        dddG = np.array([np.nan, np.nan, 0.2, 0.4])
        m_with_dddG = stats.compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, dddG=dddG)
        self.assertAlmostEqual(m_with_dddG['rho_epi_fast'], 1.0, places=5)

    def test_compute_metrics_no_doubles_returns_nan_rho_epi(self):
        # Only singles
        wt_scores = np.array([1.0, 2.0])
        mt_scores = np.array([1.0, 2.0])
        comb_scores = np.array([1.0, 2.0])
        ground_truths = np.array([1.1, 1.9])
        subset_types = ['single', 'single']
        dddG = np.array([np.nan, np.nan])

        m = stats.compute_metrics(wt_scores, mt_scores, comb_scores, ground_truths, subset_types, dddG=dddG)
        self.assertTrue(np.isnan(m['rho_epi_fast']))


class TestEpiFull(unittest.TestCase):
    def _lib(self, delta=0.0):
        """Singles A1,B1,A2,B2 plus doubles. WT head additive; MT head = WT + delta per single
        + interaction eps on doubles. Returns arrays in compute_metrics order."""
        a = {'A1': 1.0, 'A2': 2.5, 'B1': -0.5, 'B2': 0.7}
        eps = {('A1', 'B1'): 0.3, ('A1', 'B2'): -0.4, ('A2', 'B1'): 0.9, ('A2', 'B2'): 0.1}
        keys, st, wt, mt, ddd = [], [], [], [], []
        for n, v in a.items():
            keys.append((('X', 1, n),)); st.append('single')
            wt.append(v); mt.append(v + delta); ddd.append(np.nan)
        for (x, y), e in eps.items():
            keys.append((('X', 1, x), ('X', 1, y))); st.append('double')
            wt.append(a[x] + a[y]); mt.append(a[x] + a[y] + 2 * e + delta * 2); ddd.append(e)
        wt, mt = np.array(wt), np.array(mt)
        return wt, mt, 0.5 * wt + 0.5 * mt, np.array(ddd), st, keys

    def test_full_recovers_epsilon_exactly(self):
        wt, mt, comb, ddd, st, keys = self._lib()
        e = stats.epi_full_scores(comb, st, keys)
        self.assertTrue(np.all(np.isnan(e[:4])))
        np.testing.assert_allclose(e[4:], ddd[4:], atol=1e-9)

    def test_full_cancels_single_disagreement_but_fast_does_not(self):
        # The two heads disagree on singles by a per-substitution delta.
        wt, mt, comb, ddd, st, keys = self._lib(delta=0.0)
        mt = mt.copy()
        bias = {'A1': 3.0, 'A2': -2.0, 'B1': 1.5, 'B2': -1.0}
        names = ['A1', 'A2', 'B1', 'B2']
        for i, n in enumerate(names):
            mt[i] += bias[n]
        for j, (x, y) in enumerate([('A1', 'B1'), ('A1', 'B2'), ('A2', 'B1'), ('A2', 'B2')]):
            mt[4 + j] += bias[x] + bias[y]      # MT head carries its own single effects into the double
        comb = 0.5 * wt + 0.5 * mt
        e = stats.epi_full_scores(comb, st, keys)
        np.testing.assert_allclose(e[4:], ddd[4:], atol=1e-9)           # second difference: exact
        fast = (comb - wt)[4:]
        self.assertFalse(np.allclose(fast, ddd[4:], atol=1e-3))          # between-head difference: polluted
        m = stats.compute_metrics(wt, mt, comb, ddd, st, dddG=ddd, mut_keys=keys)
        self.assertAlmostEqual(m['rho_epi_full'], 1.0, places=6)
        self.assertLess(m['rho_epi_fast'], 1.0)

    def test_missing_singles_or_keys_are_nan(self):
        wt, mt, comb, ddd, st, keys = self._lib()
        m = stats.compute_metrics(wt, mt, comb, ddd, st, dddG=ddd)       # no mut_keys
        self.assertTrue(np.isnan(m['rho_epi_full']))
        drop = [i for i, k in enumerate(keys) if st[i] == 'double' or k != (('X', 1, 'A1'),)]   # remove single A1
        e = stats.epi_full_scores(comb[drop], [st[i] for i in drop], [keys[i] for i in drop])
        involves_a1 = [any(m[2] == 'A1' for m in keys[i]) for i in drop if st[i] == 'double']
        self.assertEqual(list(np.isnan(e[-4:])), involves_a1)            # only doubles missing a single are NaN


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


if __name__ == '__main__':
    unittest.main()
