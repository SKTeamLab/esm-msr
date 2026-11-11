"""The validation statistics suite (esm_msr.val_stats), shared verbatim with main-baseline: which names are logged, how libraries are grouped, what the external summary holds.
Nothing here is branch-specific, so this file is identical on both."""
import numpy as np
import pytest

from esm_msr import epi_metrics, stats, val_stats

EXPECTED_EPI = ({f'epi_{m}_{h}' for m in ('beyond_lib', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag', 'cell_rank', 'cell_flipacc') for h in ('comb', 'ctx')}
                | {'epi_naive_rho_comb', 'epi_err_xgt1_comb', 'epi_bias_xgt1_comb'})


def _table(seed=0, ctx=True):
    """Two libraries of 6 x 6 doubles (additive expectation above 1 kcal/mol for most), a noisy WT head and a noisy MT head that learned some of the epistasis."""
    rng = np.random.default_rng(seed)
    keys, st, ddg, dddg, dg_wt, wt, mt, rev = [], [], [], [], [], [], [], []
    for L, c in (('L1', 3.0), ('L2', 3.5), ('L3', 3.2)):
        for p0 in (10, 30, 50):
            a, b = rng.normal(0, 0.5, 6), rng.normal(0, 0.5, 6)
            eps = rng.normal(0, 0.7, (6, 6))
            wa, wb = a + rng.normal(0, 0.3, 6), b + rng.normal(0, 0.3, 6)
            for i in range(6):
                keys.append(((L, p0, chr(65 + i)),)); st.append('single'); ddg.append(a[i]); dddg.append(np.nan); dg_wt.append(c)
                wt.append(wa[i]); mt.append(wa[i]); rev.append(wa[i])
            for j in range(6):
                keys.append(((L, p0 + 10, chr(70 + j)),)); st.append('single'); ddg.append(b[j]); dddg.append(np.nan); dg_wt.append(c)
                wt.append(wb[j]); mt.append(wb[j]); rev.append(wb[j])
            for i in range(6):
                for j in range(6):
                    keys.append(((L, p0, chr(65 + i)), (L, p0 + 10, chr(70 + j)))); st.append('double'); ddg.append(a[i] + b[j] + eps[i, j])
                    dddg.append(eps[i, j]); dg_wt.append(c); wt.append(wa[i] + wb[j]); rev.append(wa[i] + wb[j] + rng.normal(0, 0.2))
                    mt.append(wa[i] + wb[j] + 2.0 * (0.6 * eps[i, j] + rng.normal(0, 0.4)))
    wt, mt = np.array(wt), np.array(mt)
    items = {'mut_key': keys, 'subset_type': st, 'cens': np.zeros(len(keys), int), 'ddG': ddg, 'dddG': dddg, 'dG_wt': dg_wt, 'wt': wt, 'mt': mt, 'comb': 0.5 * (wt + mt)}
    if ctx:
        items['wt_rev'] = np.array(rev)
    return epi_metrics.Table(items, link=None)


def test_exactly_the_kept_epistasis_names_are_logged_and_the_module_produces_every_one_of_them():
    t = _table()
    res, diag = epi_metrics.compute(t), epi_metrics.compute_clean_diagnostics(t)
    out = val_stats.epi_log_names(res, diag)
    assert set(out) == {f'val_{n}' for n in EXPECTED_EPI}, sorted(set(out) ^ {f'val_{n}' for n in EXPECTED_EPI})
    assert all(np.isfinite(v) for v in out.values())
    assert val_stats.EPI_LOGGED | set(val_stats.EPI_DIAG_RENAMED.values()) == EXPECTED_EPI


def test_what_is_culled_stays_out():
    t = _table()
    out = val_stats.epi_log_names(epi_metrics.compute(t), epi_metrics.compute_clean_diagnostics(t))
    gone = ('global_err', 'global_bias', 'beyond_rho_pairavg', 'class3', 'epi_n_', 'naive_rho_add', 'naive_rho_ctx', 'mt_legacy', 'wt_doubles_clean', 'clean')
    assert not [k for k in out if any(g in k for g in gone) or k.endswith('_epi')], sorted(out)


def test_without_the_cycle_passes_there_is_no_ctx_but_every_comb_name_survives():
    t = _table(ctx=False)
    out = val_stats.epi_log_names(epi_metrics.compute(t), epi_metrics.compute_clean_diagnostics(t))
    assert out and not [k for k in out if k.endswith('_ctx')]
    assert 'val_epi_cell_rank_comb' in out and 'val_epi_bias_xgt1_comb' in out and 'val_epi_naive_rho_comb' in out


def test_the_clean_diagnostics_are_the_comb_error_and_bias_where_the_additive_expectation_is_above_one():
    t = _table()
    diag = epi_metrics.compute_clean_diagnostics(t)
    out = val_stats.epi_log_names({}, diag, 'unmasked_')
    assert out == {'val_unmasked_epi_err_xgt1_comb': pytest.approx(diag['epi_clean_err_comb']), 'val_unmasked_epi_bias_xgt1_comb': pytest.approx(diag['epi_clean_bias_comb'])}


def test_non_finite_values_are_dropped_and_the_prefix_is_applied():
    out = val_stats.epi_log_names({'epi_cell_rank_comb': float('nan'), 'epi_pair_rho_comb': 0.3, 'epi_pair_rho_ctx': 0.1}, {}, 'masked_')
    assert out == {'val_masked_epi_pair_rho_comb': 0.3, 'val_masked_epi_pair_rho_ctx': 0.1}


def test_library_groups():
    for name, g in (('1A32', 'nat'), ('2KT8_G54S', 'mutbg'), ('2M7O_I21A', 'mutbg'), ('v2K43S_2KVV', 'mutbg'), ('HEEH_KT_rd6_0280', 'des'), ('r6_560_TrROS_Hall', 'des'),
                    ('EHEE_rd1_0284', 'des'), ('HHH_rd4_0816', 'des')):
        assert val_stats.library_group(name) == g, name
    assert val_stats.is_mutbg_library('5GU9_V60D') and not val_stats.is_mutbg_library('5GU9')


def test_the_external_summary_holds_four_means_and_nothing_inside_an_average():
    per = {'ptmuld': {'rho_combined': 0.5, 'rho_wt': 0.4, 'rmse_combined': 1.0},        # multi-mutants only: no singles, so no rmse_singles_wt
           's461': {'rho_combined': 0.7, 'rho_wt': 0.6, 'rmse_combined': 0.8, 'rmse_singles_wt': 0.9},
           'ssym': {'rho_combined': 0.3, 'rho_wt': 0.2, 'rmse_combined': 1.2, 'rmse_singles_wt': 1.1},
           '1A0N': {'rho_combined': 0.9, 'rho_wt': 0.9, 'rmse_combined': 0.1, 'rmse_singles_wt': 0.1}}
    out = val_stats.summarise_external(per, {'ptmuld', 's461', 'ssym'})
    assert out == {'val_rho_combined_external': pytest.approx(0.5), 'val_rho_wt_external': pytest.approx(0.4), 'val_rmse_combined_external': pytest.approx(1.0),
                   'val_rmse_singles_wt_external': pytest.approx(1.0)}
    assert val_stats.summarise_external(per, {'ptmuld'}, 'unmasked_')['val_unmasked_rho_wt_external'] == pytest.approx(0.4)
    assert val_stats.summarise_external({'1A0N': {'rho_combined': 0.5}}, {'ptmuld'}) == {}


def test_the_ddg_metrics_a_library_reports():
    rng = np.random.default_rng(1)
    n = 60
    st = np.array(['single'] * 40 + ['double'] * 20)
    gt = rng.normal(0, 1, n)
    wt, mt = gt + rng.normal(0, 0.5, n), gt + rng.normal(0, 0.5, n)
    comb = 0.5 * (wt + mt)
    m = stats.compute_metrics(wt, mt, comb, gt, st)
    for k in val_stats.PER_LIBRARY_LOGGED + ('rho_wt',):
        assert np.isfinite(m[k]), k
    assert m['rmse_singles_wt'] == pytest.approx(float(np.sqrt(np.mean((wt[:40] - gt[:40]) ** 2))))
    assert m['rho_wt'] == pytest.approx(stats.safe_spearman(wt, gt))
    for k in val_stats.AVG_LOGGED:
        assert k in m, k
    obs = {'comb': comb + 1.0, 'wt': wt + 2.0}              # optional observed-scale dict (e.g. from tobit_link)
    m2 = stats.compute_metrics(wt, mt, comb, gt, st, obs=obs)
    assert m2['rmse_combined'] > m['rmse_combined'] and m2['rmse_singles_wt'] > m['rmse_singles_wt'] and m2['rho_combined'] == m['rho_combined']


SY, MU, SF = 0.28, 0.05, 0.38


def test_the_tobit_link_is_the_mean_of_the_larger_of_two_normals():
    """E[max(X, F)], X ~ N(z, sy^2), F ~ N(mu, sf^2): the closed form against a Monte Carlo of the observation model itself."""
    h = val_stats.tobit_link(SY, MU, SF)
    rng = np.random.default_rng(0)
    n = 2_000_000
    f = rng.normal(MU, SF, n)
    for z in (-2.0, -0.5, 0.0, 0.3, 1.0, 3.0):
        mc = np.maximum(rng.normal(z, SY, n), f).mean()
        assert float(h(z)) == pytest.approx(mc, abs=2e-3), z


def test_the_tobit_link_is_a_smooth_monotone_map_between_the_floor_and_the_identity():
    h = val_stats.tobit_link(SY, MU, SF)
    z = np.linspace(-8.0, 8.0, 4001)
    out = h(z)
    assert np.all(np.diff(out) >= -1e-12)                                      # monotone, so no rank is ever changed (flat to rounding far below the floor)
    mid = (z > -2.0) & (z < 6.0)
    assert np.all(np.diff(out[mid]) > 0)
    slope = np.diff(out) / np.diff(z)
    assert slope.min() >= -1e-9 and slope.max() <= 1.0 + 1e-9                  # it only compresses (the slope reaches 1 only far above the floor)
    assert slope[z[1:] < 1.0].max() < 0.99                                    # the slope is Phi(a): still bending at one kcal/mol above the floor
    assert float(h(-30.0)) == pytest.approx(MU, abs=1e-9)                      # far below the floor: the floor's mean
    assert float(h(30.0)) == pytest.approx(30.0, abs=1e-9)                     # far above it: the identity
    assert np.all(out >= np.maximum(z, MU) - 1e-12)                            # a mean of a maximum is at least the larger mean
    assert h(np.array([[0.0, 1.0], [2.0, 3.0]])).shape == (2, 2) and np.isfinite(h(np.array([np.nan, 1.0]))[1])


def test_the_link_summary_holds_what_a_dump_needs_to_rescore_itself():
    assert val_stats.tobit_link_summary(SY, MU, SF) == {'kind': 'tobit', 'sigma_y': SY, 'mu_f': MU, 'sigma_f': SF}


def test_the_observed_scale_rmse_of_a_tobit_run_reads_the_prediction_through_the_link():
    """A latent prediction below the floor is penalised against a floored reading unless it is read through the link."""
    h = val_stats.tobit_link(SY, MU, SF)
    dg = np.full(6, 2.0)
    gt = np.array([-2.0, -2.0, -2.0, 0.5, 0.2, 0.0])                 # three variants read at the floor (absolute dG 0), the rest above it
    latent = np.array([-4.0, -3.0, -2.5, 0.5, 0.2, 0.0])             # the model says they are far below it
    st = np.array(['single'] * 6)
    plain = stats.compute_metrics(latent, latent, latent, gt, st)
    obs = {'comb': h(dg + latent) - dg, 'wt': h(dg + latent) - dg}
    linked = stats.compute_metrics(latent, latent, latent, gt, st, obs=obs)
    assert linked['rmse_combined'] < plain['rmse_combined']
    assert linked['rho_combined'] == plain['rho_combined']           # the link is monotone


def test_a_library_without_singles_has_no_wt_rmse_and_the_run_does_not_fail():
    m = stats.compute_metrics(np.arange(5.), np.arange(5.), np.arange(5.), np.arange(5.) + 0.1, np.array(['double'] * 5))
    assert np.isnan(m['rmse_singles_wt']) and np.isfinite(m['rho_wt'])


def test_the_monitor_names_validation_can_actually_log():
    names = val_stats.logged_summary_names()
    for ok in ('val_rho_combined_avg', 'val_rho_singles_wt_avg', 'val_rho_doubles_combined_avg', 'val_rmse_combined_avg', 'val_ndcg>0_wt_avg', 'val_ndcg@k=96_avg',
               'val_nat_rho_combined_avg', 'val_mutbg_rho_combined_avg', 'val_des_rho_combined_avg', 'val_rho_combined_external', 'val_rho_wt_external',
               'val_rmse_combined_external', 'val_rmse_singles_wt_external', 'val_epi_cell_rank_comb', 'val_epi_beyond_lib_ctx', 'val_epi_composite',
               'val_epi_bias_xgt1_comb', 'val_unmasked_rho_combined_avg', 'val_masked_epi_pair_rho_comb'):
        assert ok in names, ok
    for gone in ('val_cycle_gap_avg', 'val_rho_ungrouped_combined', 'val_diag_norm_lora_mt', 'val_rho_singles_combined_avg', 'val_epi_global_err_comb', 'val_diag_epi_clean_err_comb'):
        assert gone not in names, gone
