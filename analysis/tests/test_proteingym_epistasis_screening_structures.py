"""Tests for msr_analysis.{proteingym,epistasis,screening,structures,tables,style}."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msr_analysis import epistasis, proteingym, screening, structures, style, tables  # noqa: E402


def test_pgym_reference_canonical_173_and_176_counts():
    ref_173 = proteingym.pgym_reference()
    assert len(ref_173) == 173
    assert proteingym.heldout_category_counts() == {
        'Activity': 38,
        'Binding': 13,
        'Expression': 18,
        'OrganismalFitness': 67,
        'Stability': 37,
    }
    ref_176 = proteingym.pgym_reference(include_fallback_small=True)
    assert len(ref_176) == 176
    ref_213 = proteingym.pgym_reference(heldout=False, drop_large=True)
    assert len(ref_213) == 213


def test_boot_pgym_categories_and_sigma_panel():
    rng = np.random.default_rng(11)
    cats_list = ['Activity'] * 6 + ['Binding'] * 5 + ['Expression'] * 5 + ['OrganismalFitness'] * 6 + ['Stability'] * 6
    dms_ids = [f'DMS_{i}' for i in range(len(cats_list))]
    cats = pd.Series(cats_list, index=dms_ids)

    base_means = rng.uniform(0.35, 0.75, size=len(dms_ids))
    no_seed_var = pd.DataFrame({'s1': base_means, 's2': base_means, 's3': base_means}, index=dms_ids)
    with_seed_var = pd.DataFrame({
        's1': base_means - 0.06,
        's2': base_means,
        's3': base_means + 0.06,
    }, index=dms_ids)

    summary, reps = proteingym.boot_pgym_categories(
        {'ModelFixed': no_seed_var, 'ModelSeeded': with_seed_var},
        cats,
        B=400,
        seed=9,
    )
    pg_fixed = summary[(summary['model'] == 'ModelFixed') & (summary['column'] == 'PG avg')].iloc[0]
    pg_seeded = summary[(summary['model'] == 'ModelSeeded') & (summary['column'] == 'PG avg')].iloc[0]
    assert pg_fixed['value'] == pytest.approx(pg_seeded['value'])
    # Joint [assay x seed] bootstrap captures triplicate seed variance, widening the CI
    assert np.std(reps['ModelSeeded']['PG avg']) > np.std(reps['ModelFixed']['PG avg'])
    assert pg_seeded['seed_sd_pg_avg'] > 0.05

    # Verify draw_property_sigma_panel renders cleanly and leaves rcParams untouched
    rc_before = dict(plt.rcParams)
    sig_df = pd.DataFrame({
        'assay': dms_ids,
        'category': cats_list,
        'rho_dep_0.0': base_means - 0.05,
        'rho_dep_0.5': base_means + 0.02,
        'rho_dep_1.0': base_means + 0.05,
        'd_0.5': np.full(len(dms_ids), 0.07),
        'd_1.0': np.full(len(dms_ids), 0.10),
        'leaked': [False] * len(dms_ids),
    })
    pvals = proteingym.sigma_delta_pvalues(sig_df, B=200, seed=3)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    proteingym.draw_property_sigma_panel(ax, sig_df, pvals, compact=True)
    plt.close(fig)
    assert plt.rcParams['savefig.dpi'] == rc_before['savefig.dpi']


def _toy_epistasis_spine() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    aas = ('C', 'D', 'F', 'H', 'I')
    rows = []
    wt_preds, comb_preds = [], []
    for d_idx in range(3):
        code = f'DOM{d_idx}'
        dg_wt = 2.2 + 0.3 * d_idx
        singles_y = {}
        # Singles at positions 1 and 2 (5 substitutions each -> 25 complete double cells per domain)
        for pos in (1, 2):
            for aa_i, aa in enumerate(aas):
                y_s = -1.2 + 0.45 * aa_i + rng.normal(scale=0.05)
                singles_y[(pos, aa)] = y_s
                rows.append({
                    'code': code, 'mut': f'A{pos}{aa}', 'y': y_s,
                    'dddG_ML': np.nan, 'dG_wt': dg_wt, 'use': True,
                })
                wt_preds.append(y_s + rng.normal(scale=0.04))
                comb_preds.append(y_s + rng.normal(scale=0.04))
        for aa1 in aas:
            for aa2 in aas:
                add_y = singles_y[(1, aa1)] + singles_y[(2, aa2)]
                epi_true = 0.35 * singles_y[(1, aa1)] * singles_y[(2, aa2)] + rng.normal(scale=0.03)
                y_d = add_y + epi_true
                rows.append({
                    'code': code, 'mut': f'A1{aa1}:A2{aa2}', 'y': y_d,
                    'dddG_ML': epi_true, 'dG_wt': dg_wt, 'use': True,
                })
                wt_p = add_y + rng.normal(scale=0.05)
                cb_p = add_y + 0.85 * epi_true + rng.normal(scale=0.05)
                wt_preds.append(wt_p)
                comb_preds.append(cb_p)
    return pd.DataFrame(rows), np.asarray(wt_preds), np.asarray(comb_preds)


def test_epistasis_make_resample_and_boot_models():
    sp, wt_p, cb_p = _toy_epistasis_spine()
    tab = epistasis.make_epi_table(sp, wt_p, cb_p, comb=cb_p, heads=('epi', 'add'))
    assert tab.n == 75
    m_base = epistasis.compute_epi_metrics(tab, head='epi', extended=False)
    assert m_base['naive_rho'] > 0.6

    # Cloning domain 0 twice must produce 2 distinct pair matrices rather than colliding in one matrix
    dom0 = np.flatnonzero(tab.lib == 0)
    idx = np.concatenate([dom0, dom0])
    rep = np.concatenate([np.zeros(len(dom0), dtype=int), np.ones(len(dom0), dtype=int)])
    cloned = epistasis.resample_epi_table(tab, idx, rep)
    assert len(np.unique(cloned.pair)) == 2 * len(np.unique(tab.pair[dom0]))
    assert len(cloned.matrices()) == 2

    tables_map = {
        ('msrp0', 1): tab,
        ('msrp0', 2): epistasis.make_epi_table(sp, wt_p, cb_p + 0.01, comb=cb_p + 0.01, heads=('epi', 'add')),
        ('msrp0', 3): epistasis.make_epi_table(sp, wt_p, cb_p - 0.01, comb=cb_p - 0.01, heads=('epi', 'add')),
        ('single_seed', 1): tab,
    }
    models_spec = [('ESM-MSR', 'msrp0', 'epi'), ('SingleSeed', 'single_seed', 'epi')]
    df_out, reps, mnames = epistasis.boot_epistasis_models(tables_map, models_spec, B=25, seed=4, jobs=1, extended=False)
    assert reps.shape == (25, 2, len(mnames))
    assert np.isfinite(reps[:, :, mnames.index('naive_rho')]).all()
    assert len(df_out) == 2 * len(mnames)


def test_screening_pooled_gain_disambiguation_tie_jitter_and_fixed_cohort():
    # Pool where top-scoring items include destabilizing (negative y) variants
    scores = np.array([10.0, 9.0, 1.0, 0.0, -1.0, -2.0])
    y = np.array([1.0, -0.4, -1.0, -1.5, -2.0, -2.5])
    clusters = np.array(['P1', 'P1', 'P2', 'P2', 'P3', 'P3'])
    pooled = screening.pooled_screen(scores, y, clusters, K=2, tau=0.5, B=50, seed=0)
    assert pooled['mean_y'][0] == pytest.approx((1.0 - 0.4) / 2.0)
    assert pooled['gain'][0] == pytest.approx(pooled['mean_y'][0])
    assert pooled['rectified_gain'][0] == pytest.approx(1.0)
    assert pooled['rectified_gain_mean'][0] == pytest.approx(0.5)

    # Identical tied scores pick deterministic prefix in both protein_screen and pooled_screen
    tied = np.zeros(6)
    ps = screening.protein_screen(tied, y, K=2, tau=0.5)
    ps_pool = screening.pooled_screen(tied, y, clusters, K=2, tau=0.5, B=20, seed=0)
    assert ps is not None
    assert ps['gain'] == pytest.approx(ps_pool['rectified_gain'][0])

    # Fixed cohort across K-sweep: protein P_small (6 variants) is excluded at K=1 when max(Ks)=20 requires >= 10 variants
    groups = np.array(['P_small'] * 6 + ['P_big1'] * 15 + ['P_big2'] * 15)
    sc_curve = np.linspace(0.0, 1.0, len(groups))
    y_curve = sc_curve - 0.3
    curve = screening.per_protein_curve(groups, sc_curve, y_curve, Ks=[1, 5, 10, 20], B=40, seed=0)
    assert set(curve['n_proteins'].unique()) == {2}


def test_structures_pdb_virtual_cb_and_mutation_parsers(tmp_path):
    assert structures.parse_mutations('WT') == []
    assert structures.parse_mutations('A12C:M-3V,G45*') == [('A', 12, 'C'), ('M', -3, 'V'), ('G', 45, '*')]
    assert structures.parse_mutations('bad_token') is None

    pdb_text = '\n'.join([
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
        'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
        'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
        'ATOM      4  CB  ALA A   1       1.988  -0.773  -1.206  1.00 20.00           C  ',
        'ATOM      5  CB BALA A   1       9.999   9.999   9.999  0.50 20.00           C  ',  # altloc B ignored
        'ATOM      6  N   GLY A   2       3.300   1.600   0.000  1.00 20.00           N  ',
        'ATOM      7  CA  GLY A   2       3.950   2.900   0.000  1.00 20.00           C  ',
        'ATOM      8  C   GLY A   2       5.460   2.800   0.000  1.00 20.00           C  ',
    ]) + '\n'
    pdb_file = tmp_path / 'toy.pdb'
    pdb_file.write_text(pdb_text)

    res = structures.load_pdb_residues(pdb_file)
    seq, nums = structures.pdb_sequence(res)
    assert seq == 'AG' and nums == [1, 2]
    # Altloc B must be ignored on Ala 1
    assert np.allclose(res[1]['cb'], [1.988, -0.773, -1.206])
    # Glycine 2 has no CB atom -> reconstructed tetrahedral virtual CB (~1.52 A from CA)
    gly_ca_cb = np.linalg.norm(res[2]['cb'] - res[2]['ca'])
    assert 1.45 < gly_ca_cb < 1.60

    dmat = structures.cb_distance_matrix(res)
    assert dmat.shape == (3, 3) and np.isnan(dmat[0, 1])
    assert dmat[1, 2] == pytest.approx(structures.pair_cb_distance(res, 1, 2))


def test_tables_write_stat_table_and_raincloud_ci_whiskers(tmp_path):
    df = pd.DataFrame({
        'model': ['ESM-MSR', 'ThermoMPNN_1'],
        'rho': [0.71234, 0.64567],
        'delta': [0.0, -0.06667],
    })
    bold = pd.DataFrame({'model': [True, False], 'rho': [True, False], 'delta': [False, False]})
    cols = {'model': 'Model', 'rho': 'Spearman ρ', 'delta': 'Δρ'}
    written = tables.write_stat_table(df, tmp_path / 'bench_summary', columns=cols, bold_mask=bold, tex=True, caption='Summary_1')
    assert set(written.keys()) == {'csv', 'md', 'tex'}
    md_txt = written['md'].read_text()
    tex_txt = written['tex'].read_text()
    assert '**0.7123**' in md_txt
    assert r'\textbf{0.7123}' in tex_txt and r'ThermoMPNN\_1' in tex_txt and r'$\rho$' in tex_txt

    # Verify raincloud renders [lo, hi] 95% CI whisker container (fixing dropped-CI bug in 14_property_category_gain)
    fig, ax = plt.subplots(figsize=(3, 3))
    vals = np.linspace(0.2, 0.8, 20)
    style.raincloud(ax, 0.0, vals, color='#2a78d6', mean=0.50, lo=0.41, hi=0.59)
    err_containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(err_containers) == 1
    plt.close(fig)
