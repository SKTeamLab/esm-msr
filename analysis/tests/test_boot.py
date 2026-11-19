"""Tests for count-weighted SpearmanEngine, arbitrary-seed-count bootstrap, and site vs protein resampling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msr_analysis.boot import SpearmanEngine, bootstrap_rho, paired_diff  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402


def _expand(x, y, w):
    idx = np.repeat(np.arange(len(x)), w.astype(int))
    return x[idx], y[idx]


def test_pooled_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    n = 300
    x = np.round(rng.normal(size=n), 1)
    y = np.round(x + rng.normal(size=n), 1)
    w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
    eng = SpearmanEngine(y, np.zeros(n, np.int64), 1)
    got = eng.rho(eng.prepare(x), w)[0]
    xe, ye = _expand(x, y, w)
    assert abs(got - spearmanr(xe, ye)[0]) < 1e-10


def test_grouped_matches_scipy():
    rng = np.random.default_rng(1)
    n, G = 600, 6
    g = rng.integers(0, G, n)
    x = rng.normal(size=n)
    y = x * rng.uniform(0.2, 1.0, n) + rng.normal(size=n)
    w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
    eng = SpearmanEngine(y, g, G)
    got = eng.rho(eng.prepare(x), w)
    for k in range(G):
        m = g == k
        xe, ye = _expand(x[m], y[m], w[m])
        assert abs(got[k] - spearmanr(xe, ye)[0]) < 1e-10


def _synthetic_frame(seed: int = 42) -> Frame:
    rng = np.random.default_rng(seed)
    rows = []
    for p_idx in range(6):
        prot = f'P{p_idx}'
        slope = 0.4 + 0.12 * p_idx
        for site in range(1, 9):
            site_bias = 0.35 * ((-1) ** site) * site
            for aa_i, aa in enumerate(('A', 'V', 'L')):
                y_val = site_bias + slope * aa_i + rng.normal(scale=0.15)
                rows.append({
                    'id': f'{prot}_G{site}{aa}',
                    'row_id': len(rows),
                    'code': prot,
                    'group': prot,
                    'cluster': prot,
                    'mut': f'G{site}{aa}',
                    'n_mut': 1,
                    'site1': site,
                    'y': y_val,
                    'use': True,
                })
    sp = pd.DataFrame(rows)
    y = sp['y'].to_numpy(dtype=float)
    preds = {}
    for ns in (1, 2, 3, 5):
        preds[f'm_s{ns}'] = np.vstack([y + rng.normal(scale=0.4 + 0.05 * s, size=len(y)) for s in range(ns)])
    return Frame('synthetic', sp, preds)


@pytest.mark.parametrize('per_protein', [False, True])
@pytest.mark.parametrize('kind', ['protein', 'site', 'mutation'])
def test_bootstrap_rho_arbitrary_seed_counts_never_produce_nan(per_protein, kind):
    fr = _synthetic_frame()
    keys = ['m_s1', 'm_s2', 'm_s3', 'm_s5']
    point, reps = bootstrap_rho({'synthetic': fr}, keys, kind=kind, B=160, seed=13, per_protein=per_protein)
    for k in keys:
        arr = reps['synthetic'][k]
        assert arr.shape == (160,)
        assert np.isfinite(arr).all(), f'NaN replicate in {k} (kind={kind}, per_protein={per_protein})'
        pt_rho, per_seed, n_rows, _ = point['synthetic'][k]
        assert np.isfinite(pt_rho) and len(per_seed) == int(k.split('_s')[1]) and n_rows == len(fr)


def test_site_bootstrap_on_per_protein_dataset_resamples_sites_not_whole_proteins():
    fr = _synthetic_frame(seed=99)
    _, reps_prot = bootstrap_rho({'synthetic': fr}, ['m_s3'], kind='protein', B=150, seed=7, per_protein=True)
    _, reps_site = bootstrap_rho({'synthetic': fr}, ['m_s3'], kind='site', B=150, seed=7, per_protein=True)
    # Previously kind='site' with per_protein=True fell through to the whole-protein shortcut; verify they differ
    assert not np.allclose(reps_prot['synthetic']['m_s3'], reps_site['synthetic']['m_s3'])
    assert np.std(reps_site['synthetic']['m_s3']) > 0


def test_paired_diff_uses_point_estimates_and_floors_pvalue():
    reps_a = np.linspace(0.70, 0.80, 100)
    reps_b = np.linspace(0.50, 0.60, 100)
    res = paired_diff(reps_a, reps_b, point_a=0.76, point_b=0.54)
    assert res['mean'] == pytest.approx(0.22)
    assert res['p'] == pytest.approx(1.0 / 101.0)
