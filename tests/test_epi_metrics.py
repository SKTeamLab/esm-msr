"""esm_msr.epi_metrics: each level credits only its own component, and saturation earns nothing outside the naive / global metrics."""
import unittest

import numpy as np

from esm_msr import epi_metrics as E

ROWS, COLS = 'ACDEFG', 'HIKLMN'


def assay_clamp(lo=-1.0, hi=5.0, tau=0.3):
    """The assay as two soft clamps in series, a floor at ``lo`` then a ceiling at ``hi``.

    Written out here rather than imported from a branch's link module: this file tests epi_metrics, which takes
    whatever link the caller hands it, so the test should not care which link module the branch happens to ship.
    That also keeps this file byte-identical on claude/main-baseline and the devel branches.
    """
    def h(z):
        z = np.asarray(z, dtype=np.float64)
        u = lo + tau * np.logaddexp(0.0, (z - lo) / tau)
        return hi - tau * np.logaddexp(0.0, (hi - u) / tau)
    return h


FLOOR = assay_clamp()


def synthetic(seed=0, n_pairs=10, measure=FLOOR):
    """True latent singles and epistasis components on n_pairs position-pair matrices (6 x 6) in two libraries, measured through ``measure``."""
    rng = np.random.default_rng(seed)
    pairs = []
    for p in range(n_pairs):
        lib, c = ('L1', 1.5) if p % 2 else ('L2', 2.5)
        a, b = rng.normal(-1.2, 1.0, len(ROWS)), rng.normal(-1.2, 1.0, len(COLS))
        m = rng.normal(0, 0.6)
        r, s = rng.normal(0, 0.4, len(ROWS)), rng.normal(0, 0.4, len(COLS))
        eps = rng.normal(0, 0.4, (len(ROWS), len(COLS)))
        eps = eps - eps.mean(1, keepdims=True) - eps.mean(0, keepdims=True) + eps.mean()
        pairs.append(dict(lib=lib, c=c, pos=(10 * p + 1, 10 * p + 5), a=a, b=b, m=m, r=r, s=s, eps=eps))
    return pairs, measure


def items(pairs, measure, predict, link_pred=None):
    """Validation items for epi_metrics.Table. ``predict(P, i, j)`` gives a head's latent epistasis for a cell (the head's latent singles are
    the true ones). All four latent heads use the same prediction, wt is additive."""
    obs = (lambda z: z) if measure is None else measure
    out = {k: [] for k in ('mut_key', 'subset_type', 'cens', 'ddG', 'dddG', 'dG_wt', 'wt', 'mt', 'comb', 'wt_rev')}

    def add(key, st, ddG, dddG, c, wt, pred):
        out['mut_key'].append(key); out['subset_type'].append(st); out['cens'].append(0); out['ddG'].append(ddG); out['dddG'].append(dddG)
        out['dG_wt'].append(c); out['wt'].append(wt); out['mt'].append(pred); out['comb'].append(0.5 * (wt + pred)); out['wt_rev'].append(pred)

    for P in pairs:
        c, (p, q) = P['c'], P['pos']
        ka = [(P['lib'], 'W', p, x) for x in ROWS]
        kb = [(P['lib'], 'W', q, x) for x in COLS]
        yA = obs(c + P['a']) - obs(c)
        yB = obs(c + P['b']) - obs(c)
        for i in range(len(ROWS)):
            add((ka[i],), 'single', yA[i], np.nan, c, P['a'][i], P['a'][i])
        for j in range(len(COLS)):
            add((kb[j],), 'single', yB[j], np.nan, c, P['b'][j], P['b'][j])
        for i in range(len(ROWS)):
            for j in range(len(COLS)):
                e = P['m'] + P['r'][i] + P['s'][j] + P['eps'][i, j]
                y = obs(c + P['a'][i] + P['b'][j] + e) - obs(c)
                add((ka[i], kb[j]), 'double', y, y - yA[i] - yB[j], c, P['a'][i] + P['b'][j],
                    P['a'][i] + P['b'][j] + predict(P, i, j))
    return {k: (v if k in ('mut_key', 'subset_type') else np.asarray(v, float)) for k, v in out.items()}


KNOW = {
    'nothing': lambda P, i, j: 0.0,
    'pair': lambda P, i, j: P['m'],
    'line': lambda P, i, j: P['r'][i] + P['s'][j],
    'cell': lambda P, i, j: P['eps'][i, j],
    'all': lambda P, i, j: P['m'] + P['r'][i] + P['s'][j] + P['eps'][i, j],
}
ALL_HEADS = {m: ('comb', 'epi', 'ctx', 'add') for m in tuple(E.METRICS) + ('cell_sigsd',)}


def scores(know, measure=FLOOR, link=FLOOR, seed=0):
    pairs, meas = synthetic(seed, measure=measure)
    return E.compute(E.Table(items(pairs, meas, KNOW[know]), link), ALL_HEADS)


class TestLevelIsolation(unittest.TestCase):
    def test_the_rank_and_flip_metrics_credit_only_the_cell(self):
        for know in ('nothing', 'pair', 'line'):
            s = scores(know)
            self.assertEqual(s['epi_cell_rank_epi'], 0.0, know)
            self.assertEqual(s['epi_cell_flipacc_epi'], 0.5, know)
            self.assertEqual(s['epi_cell_sigsd_epi'], 0.0, know)
        s = scores('cell')
        # rank space is not additive: without the line effects the double-centred ranks of the truth are not reproduced exactly
        self.assertGreater(s['epi_cell_rank_epi'], 0.5)
        self.assertGreater(s['epi_cell_flipacc_epi'], 0.9)

    def test_each_residual_level_is_carried_by_its_own_component(self):
        pair, line, cell = (scores(k, measure=None, link=None) for k in ('pair', 'line', 'cell'))
        self.assertGreater(pair['epi_pair_rho_epi'], 0.7)
        self.assertGreater(line['epi_line_rho_epi'], 0.7)
        self.assertGreater(cell['epi_cell_mag_epi'], 0.7)
        # a higher level does not score at the cell level (up to the smoother's curvature). Through a saturating link it can: line effects
        # pushed through the floor are non-additive on the observed scale, for the measurement and a model that knows them alike
        self.assertLess(abs(pair['epi_cell_mag_epi']), 0.25)
        self.assertLess(abs(line['epi_cell_mag_epi']), 0.25)

    def test_a_perfect_predictor_scores_at_the_top_of_every_level(self):
        s = scores('all', measure=None, link=None)
        for m in ('beyond_lib', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag'):
            self.assertGreater(s[f'epi_{m}_epi'], 0.85, m)
        self.assertAlmostEqual(s['epi_cell_rank_epi'], 1.0, places=9)
        self.assertEqual(s['epi_cell_flipacc_epi'], 1.0)


class TestSaturation(unittest.TestCase):
    def test_an_additive_head_through_the_link_scores_only_on_the_naive_and_global_metrics(self):
        s = scores('nothing')
        self.assertGreater(s['epi_naive_rho_add'], 0.5)              # saturation alone: the confound the naive metric cannot see past
        for m in ('beyond_lib', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag', 'cell_rank'):
            self.assertEqual(s[f'epi_{m}_add'], 0.0, m)
        self.assertEqual(s['epi_cell_flipacc_add'], 0.5)
        s_all = scores('all')
        self.assertLess(s_all['epi_global_err_epi'], s['epi_global_err_epi'])

    def test_the_rank_metrics_ignore_a_monotone_measurement(self):
        self.assertAlmostEqual(scores('all', measure=None, link=None)['epi_cell_rank_epi'], 1.0, places=9)
        self.assertGreater(scores('all', measure=FLOOR, link=None)['epi_cell_rank_epi'], 0.6)     # floor ties blur, never invent
        self.assertEqual(scores('nothing', measure=FLOOR, link=None)['epi_cell_rank_epi'], 0.0)
        self.assertEqual(scores('all', measure=FLOOR, link=None)['epi_cell_flipacc_epi'], 1.0)

    def test_without_a_link_the_additive_head_is_flat(self):
        s = scores('nothing', measure=None, link=None)
        self.assertEqual(s['epi_naive_rho_add'], 0.0)
        self.assertEqual(s['epi_beyond_rho_add'], 0.0)


class TestTable(unittest.TestCase):
    def test_doubles_pair_only_with_their_own_librarys_singles(self):
        pairs, meas = synthetic(1)
        it = items(pairs, meas, KNOW['all'])
        t = E.Table(it, FLOOR)
        self.assertEqual(t.n, 10 * len(ROWS) * len(COLS))
        # relabel one library's singles: its doubles lose their singles and drop out
        it2 = dict(it)
        it2['mut_key'] = [tuple(('X',) + m[1:] for m in k) if (st == 'single' and k[0][0] == 'L1') else k
                          for k, st in zip(it['mut_key'], it['subset_type'])]
        self.assertEqual(E.Table(it2, FLOOR).n, 5 * len(ROWS) * len(COLS))

    def test_an_unmeasured_single_takes_its_doubles_with_it(self):
        pairs, meas = synthetic(2)
        it = items(pairs, meas, KNOW['all'])
        gt = it['ddG'].copy()
        first_single = it['subset_type'].index('single')
        gt[first_single] = np.nan                                  # a dead single: its 6 doubles have no additive expectation
        it['ddG'] = gt
        self.assertEqual(E.Table(it, FLOOR).n, 10 * len(ROWS) * len(COLS) - len(COLS))

    def test_the_caller_chooses_the_population_and_cens_is_not_read(self):
        """Table used to drop cens != 0 items itself. It no longer does: scoring only the uncensored items is a
        range restriction on the very population a censoring run exists to extend, which biased such runs' metrics
        down and made them incomparable to a run without censoring. The caller picks the population instead - see
        training.on_validation_epoch_end, which restricts to the canonical set before pooling."""
        pairs, meas = synthetic(2)
        it = items(pairs, meas, KNOW['all'])
        n_all = E.Table(it, FLOOR).n
        cens = np.zeros(len(it['ddG']), int)
        cens[it['subset_type'].index('single')] = -1
        cens[it['subset_type'].index('double')] = +1
        it['cens'] = cens
        self.assertEqual(E.Table(it, FLOOR).n, n_all)

    def test_ctx_needs_the_reverse_leg(self):
        pairs, meas = synthetic(3)
        it = items(pairs, meas, KNOW['cell'])
        self.assertIn('epi_cell_rank_ctx', E.compute(E.Table(it, FLOOR)))
        it['wt_rev'] = np.full(len(it['wt']), np.nan)
        res = E.compute(E.Table(it, FLOOR))
        self.assertFalse([k for k in res if k.endswith('_ctx')])
        self.assertIn('epi_cell_rank_epi', res)

    def test_no_doubles_gives_counts_only(self):
        it = {'mut_key': [(('L', 'W', 1, 'A'),)], 'subset_type': ['single'], 'cens': np.zeros(1), 'ddG': np.ones(1), 'dddG': np.full(1, np.nan),
              'dG_wt': np.ones(1), 'wt': np.ones(1), 'mt': np.ones(1), 'comb': np.ones(1), 'wt_rev': None}
        self.assertEqual(E.compute(E.Table(it, None)), {'epi_n_doubles': 0.0, 'epi_n_pairs': 0.0})

    def test_logged_names_cover_the_documented_set(self):
        names = E.logged_names()
        self.assertEqual(len([n for n in names if not n.startswith('epi_n_')]), 4 + 1 + 1 + 11 * 2)
        s = E.compute(E.Table(items(*synthetic(0), KNOW['all']), FLOOR))
        for n in names:
            self.assertIn(n, s)

    def test_every_metric_is_logged_for_its_configured_heads(self):
        for m, hs in E.METRICS.items():
            if m.startswith('global_'):
                self.assertEqual(tuple(hs), ('epi',), m)
            elif m == 'naive_rho':
                self.assertEqual(set(hs), {'comb', 'epi', 'ctx', 'add'}, m)
            else:
                self.assertEqual(set(hs), {'epi', 'ctx'}, m)

    def test_comb_and_epi_agree_on_the_reversals(self):
        t = E.Table(items(*synthetic(5), KNOW['cell']), FLOOR)
        self.assertEqual(E.flip_accuracy(t, 'comb')[0], E.flip_accuracy(t, 'epi')[0])


class TestWTReference(unittest.TestCase):
    def test_epi_ignores_its_own_singles(self):
        """epi's dddG' is ~MT(AB) - WT(AB): what the MT/epi head says about singles never enters."""
        pairs, meas = synthetic(6, measure=None)
        it = items(pairs, meas, KNOW['all'])
        base = E.compute(E.Table(it, None))
        st = np.asarray(it['subset_type'])
        it2 = dict(it)
        it2['mt'] = np.where(st == 'single', it['mt'] + np.random.default_rng(0).normal(0, 2.0, len(st)), it['mt'])
        it2['comb'] = 0.5 * (it2['wt'] + it2['mt'])
        moved = E.compute(E.Table(it2, None))
        for m in ('naive_rho_epi', 'beyond_rho_epi', 'beyond_lib_epi', 'pair_rho_epi', 'line_rho_epi', 'cell_mag_epi'):
            self.assertAlmostEqual(base[f'epi_{m}'], moved[f'epi_{m}'], places=9, msg=m)

    def test_library_centring_removes_a_library_wide_offset(self):
        """A per-library constant on residuals is a library offset, not epistasis: beyond_lib does not move, beyond_rho does."""
        pairs, meas = synthetic(7, measure=None)
        it = items(pairs, meas, KNOW['cell'])
        t = E.Table(it, None)
        rm, rp = E.beyond_global(t, 'epi')
        rp_shifted = rp + np.where(t.lib == 0, 3.0, 0.0)
        base = E._rho(E._lib_centred(rp, t.lib), E._lib_centred(rm, t.lib))
        moved = E._rho(E._lib_centred(rp_shifted, t.lib), E._lib_centred(rm, t.lib))
        uncentered_base = E._rho(rp, rm)
        uncentered_moved = E._rho(rp_shifted, rm)
        self.assertAlmostEqual(base, moved, places=9)
        self.assertNotAlmostEqual(uncentered_base, uncentered_moved, places=3)

    def test_the_measured_residual_is_dddG_minus_its_global_curve(self):
        t = E.Table(items(*synthetic(8), KNOW['all']), FLOOR)
        r, _ = E.beyond_global(t, 'epi')
        np.testing.assert_allclose(r, (t.c + t.y) - E.smooth_on(t.x, t.c + t.y), atol=1e-9)    # the form used before 2026-10-24


class TestGlobalErrAndBias(unittest.TestCase):
    def test_zero_for_a_perfect_head_and_signed(self):
        pairs, meas = synthetic(9, measure=None)
        it = items(pairs, meas, KNOW['all'])
        r = E.compute(E.Table(it, None))
        self.assertAlmostEqual(r['epi_global_err_epi'], 0.0, places=9)
        self.assertAlmostEqual(r['epi_global_bias_epi'], 0.0, places=9)
        # shift the double predictions by -0.5: bias drops by -0.5, err becomes 0.5
        st = np.asarray(it['subset_type'])
        it['mt'] = it['mt'] - np.where(st == 'double', 0.5, 0.0)
        r2 = E.compute(E.Table(it, None))
        self.assertAlmostEqual(r2['epi_global_bias_epi'], -0.5, places=9)
        self.assertAlmostEqual(r2['epi_global_err_epi'], 0.5, places=9)


class TestSmoother(unittest.TestCase):
    def test_linear_data_is_reproduced_including_beyond_the_outer_bins(self):
        x = np.linspace(-3, 3, 400)
        np.testing.assert_allclose(E.smooth_on(x, 2 * x + 1), 2 * x + 1, atol=1e-9)


if __name__ == '__main__':
    unittest.main()


class TestRawLevels(unittest.TestCase):
    def test_saturation_alone_scores_on_the_raw_levels_and_not_on_the_residual_ones(self):
        pairs, meas = synthetic(4)
        res = E.compute(E.Table(items(pairs, meas, KNOW['nothing']), FLOOR), {m: ('add',) for m in E.RAW_METRICS + ('pair_rho', 'line_rho', 'cell_mag')})
        self.assertGreater(res['epi_pair_raw_add'], 0.3)
        self.assertGreater(res['epi_line_raw_add'], 0.3)
        for m in ('pair_rho', 'line_rho', 'cell_mag'):
            self.assertEqual(res[f'epi_{m}_add'], 0.0)
