"""esm_msr.epi_metrics: each level credits only its own component, and saturation earns nothing outside the naive / global metrics."""
import unittest

import numpy as np

from esm_msr import epi_metrics as E
from esm_msr.link import numpy_link

ROWS, COLS = 'ACDEFG', 'HIKLMN'
FLOOR = numpy_link(-1.0, 5.0, 0.3, 0.3)          # the assay: a soft floor at -1 and ceiling at 5


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
ALL_HEADS = {m: ('comb', 'mt', 'ctx', 'add') for m in E.METRICS}


def scores(know, measure=FLOOR, link=FLOOR, seed=0):
    pairs, meas = synthetic(seed, measure=measure)
    return E.compute(E.Table(items(pairs, meas, KNOW[know]), link), ALL_HEADS)


class TestLevelIsolation(unittest.TestCase):
    def test_the_rank_and_flip_metrics_credit_only_the_cell(self):
        for know in ('nothing', 'pair', 'line'):
            s = scores(know)
            self.assertEqual(s['epi_cell_rank_mt'], 0.0, know)
            self.assertEqual(s['epi_cell_flipacc_mt'], 0.5, know)
            self.assertEqual(s['epi_cell_sigsd_mt'], 0.0, know)
        s = scores('cell')
        # rank space is not additive: without the line effects the double-centred ranks of the truth are not reproduced exactly
        self.assertGreater(s['epi_cell_rank_mt'], 0.5)
        self.assertGreater(s['epi_cell_flipacc_mt'], 0.9)

    def test_each_residual_level_is_carried_by_its_own_component(self):
        pair, line, cell = (scores(k, measure=None, link=None) for k in ('pair', 'line', 'cell'))
        self.assertGreater(pair['epi_pair_rho_comb'], 0.7)
        self.assertGreater(line['epi_line_rho_comb'], 0.7)
        self.assertGreater(cell['epi_cell_mag_comb'], 0.7)
        # a higher level does not score at the cell level (up to the smoother's curvature). Through a saturating link it can: line effects
        # pushed through the floor are non-additive on the observed scale, for the measurement and a model that knows them alike
        self.assertLess(abs(pair['epi_cell_mag_comb']), 0.25)
        self.assertLess(abs(line['epi_cell_mag_comb']), 0.25)

    def test_a_perfect_predictor_scores_at_the_top_of_every_level(self):
        s = scores('all', measure=None, link=None)
        for m in ('beyond_rho', 'pair_rho', 'line_rho', 'cell_mag'):
            self.assertGreater(s[f'epi_{m}_comb'], 0.85, m)
        self.assertAlmostEqual(s['epi_cell_rank_mt'], 1.0, places=9)
        self.assertEqual(s['epi_cell_flipacc_mt'], 1.0)
        # comb carries half the epistasis on an additive base, which reorders columns (FINDINGS section 4): lower in rank space only
        self.assertLess(s['epi_cell_rank_comb'], s['epi_cell_rank_mt'])


class TestSaturation(unittest.TestCase):
    def test_an_additive_head_through_the_link_scores_only_on_the_naive_and_global_metrics(self):
        s = scores('nothing')
        self.assertGreater(s['epi_naive_rho_add'], 0.5)              # saturation alone: the confound the naive metric cannot see past
        for m in ('beyond_rho', 'pair_rho', 'line_rho', 'cell_mag', 'cell_rank'):
            self.assertEqual(s[f'epi_{m}_add'], 0.0, m)
        self.assertEqual(s['epi_cell_flipacc_add'], 0.5)
        self.assertLess(s['epi_global_err_add'], scores('nothing', link=None)['epi_global_err_add'])

    def test_the_rank_metrics_ignore_a_monotone_measurement(self):
        self.assertAlmostEqual(scores('all', measure=None, link=None)['epi_cell_rank_mt'], 1.0, places=9)
        self.assertGreater(scores('all', measure=FLOOR, link=None)['epi_cell_rank_mt'], 0.6)     # floor ties blur, never invent
        self.assertEqual(scores('nothing', measure=FLOOR, link=None)['epi_cell_rank_mt'], 0.0)
        self.assertEqual(scores('all', measure=FLOOR, link=None)['epi_cell_flipacc_mt'], 1.0)

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

    def test_censored_or_unmeasured_doubles_and_singles_are_left_out(self):
        pairs, meas = synthetic(2)
        it = items(pairs, meas, KNOW['all'])
        cens = it['cens'].copy()
        first_single = it['subset_type'].index('single')
        cens[first_single] = -1                                    # a dead single: its 6 doubles have no additive expectation
        it['cens'] = cens
        self.assertEqual(E.Table(it, FLOOR).n, 10 * len(ROWS) * len(COLS) - len(COLS))

    def test_ctx_needs_the_reverse_leg(self):
        pairs, meas = synthetic(3)
        it = items(pairs, meas, KNOW['cell'])
        self.assertIn('epi_cell_rank_ctx', E.compute(E.Table(it, FLOOR)))
        it['wt_rev'] = np.full(len(it['wt']), np.nan)
        res = E.compute(E.Table(it, FLOOR))
        self.assertFalse([k for k in res if k.endswith('_ctx')])
        self.assertIn('epi_cell_rank_mt', res)

    def test_no_doubles_gives_counts_only(self):
        it = {'mut_key': [(('L', 'W', 1, 'A'),)], 'subset_type': ['single'], 'cens': np.zeros(1), 'ddG': np.ones(1), 'dddG': np.full(1, np.nan),
              'dG_wt': np.ones(1), 'wt': np.ones(1), 'mt': np.ones(1), 'comb': np.ones(1), 'wt_rev': None}
        self.assertEqual(E.compute(E.Table(it, None)), {'epi_n_doubles': 0.0, 'epi_n_pairs': 0.0})

    def test_logged_names_cover_the_documented_set(self):
        names = E.logged_names()
        self.assertEqual(len([n for n in names if not n.startswith('epi_n_')]), 17)
        s = scores('all')
        for n in names:
            self.assertIn(n, s)


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
