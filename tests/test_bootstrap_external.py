"""analysis/msr_analysis/boot.py: count-weighted Spearman bootstrap engine and dataset drivers against scipy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
from msr_analysis import boot  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402


def _make_frame(rng, name='ext', n=300, n_codes=12):
    code = rng.choice([f'p{i}' for i in range(n_codes)], n)
    site1 = rng.integers(1, 40, n)
    y = np.round(rng.normal(size=n), 1)                                # ties in ground truth
    a = np.round(y + rng.normal(scale=0.8, size=n), 1)[None, :]        # single-seed model with ties
    b = rng.normal(size=(1, n))
    b[0, rng.random(n) < 0.15] = np.nan                                # model with missing rows
    c = np.vstack([
        y + rng.normal(scale=0.7, size=n),
        y + rng.normal(scale=0.9, size=n),
        y + rng.normal(scale=0.8, size=n),
    ])                                                                 # 3-seed model
    spine = pd.DataFrame({
        'use': np.ones(n, dtype=bool),
        'y': y,
        'group': code,
        'cluster': code,
        'site1': site1,
    })
    return Frame(name=name, spine=spine, preds={'a': a, 'b': b, 'c': c})


def _reference_rho(frame, w, per_protein):
    sp = frame.spine
    out = {}
    for k in ('a', 'b'):
        p = frame.preds[k][0]
        valid = sp['use'].to_numpy() & np.isfinite(sp['y'].to_numpy()) & np.isfinite(p)
        if not per_protein:
            keep = valid & (w > 0)
            idx = np.repeat(np.flatnonzero(keep), w[keep].astype(int))
            out[k] = float(spearmanr(sp['y'].to_numpy()[idx], p[idx])[0])
        else:
            rs = []
            for _, sub in sp[valid].groupby('group', sort=True):
                rows = sub.index.to_numpy()
                ww = w[rows].astype(int)
                if ww.sum() < 3:
                    continue
                idx = np.repeat(rows, ww)
                r = spearmanr(sp['y'].to_numpy()[idx], p[idx])[0]
                if np.isfinite(r):
                    rs.append(r)
            out[k] = float(np.mean(rs)) if rs else np.nan
    return out


def test_point_estimates_match_pandas_and_scipy():
    rng = np.random.default_rng(0)
    fr = _make_frame(rng)
    b_pool = boot.DatasetBoot(fr, ['a', 'b', 'c'], per_protein=False)
    for k in ('a', 'b'):
        df = pd.DataFrame({'y': fr.spine['y'], 'p': fr.preds[k][0]})
        want = df.corr('spearman').iloc[0, 1]
        assert b_pool.point(k)[0] == pytest.approx(want, abs=1e-10)
    # 3-seed point estimate is the mean of per-seed Spearman rhos
    seed_rhos = [spearmanr(fr.spine['y'], s)[0] for s in fr.preds['c']]
    mean_c, per_seed_c, n_rows, n_prot = b_pool.point('c')
    assert mean_c == pytest.approx(np.mean(seed_rhos), abs=1e-10)
    np.testing.assert_allclose(per_seed_c, seed_rhos, atol=1e-10)
    assert n_rows == len(fr) and n_prot == fr.spine['cluster'].nunique()

    b_grp = boot.DatasetBoot(fr, ['a', 'b'], per_protein=True)
    for k in ('a', 'b'):
        df = pd.DataFrame({'group': fr.spine['group'], 'y': fr.spine['y'], 'p': fr.preds[k][0]})
        want = df.groupby('group')[['y', 'p']].corr('spearman').xs('y', level=1)['p'].mean()
        assert b_grp.point(k)[0] == pytest.approx(want, abs=1e-10)


@pytest.mark.parametrize('per_protein', [False, True])
def test_random_integer_weights_match_expanded_scipy_sample(per_protein):
    rng = np.random.default_rng(1)
    fr = _make_frame(rng, n=320, n_codes=10)
    db = boot.DatasetBoot(fr, ['a', 'b'], per_protein=per_protein)
    for _ in range(5):
        w = rng.poisson(1.2, len(fr)).astype(float)
        got = db.replicate(w)
        ref = _reference_rho(fr, w, per_protein=per_protein)
        for k in ('a', 'b'):
            assert got[k] == pytest.approx(ref[k], abs=1e-9)


def test_joint_protein_draw_shares_proteins_across_datasets():
    rng = np.random.default_rng(2)
    f1 = _make_frame(rng, 'd1', n=200, n_codes=10)
    f2 = _make_frame(rng, 'd2', n=160, n_codes=10)
    b1 = boot.DatasetBoot(f1, ['a', 'b'], per_protein=False)
    b2 = boot.DatasetBoot(f2, ['a', 'b'], per_protein=False)
    gidx, ncl = boot.global_cluster_index({'d1': b1.cluster_names['protein'], 'd2': b2.cluster_names['protein']})
    counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl))
    w1 = boot.draw_weights('protein', b1, rng, counts, gidx['d1'])
    w2 = boot.draw_weights('protein', b2, rng, counts, gidx['d2'])
    shared = set(f1.spine['cluster']) & set(f2.spine['cluster'])
    assert shared
    for p in shared:
        m1 = w1[f1.spine['cluster'].to_numpy() == p]
        m2 = w2[f2.spine['cluster'].to_numpy() == p]
        assert np.all(m1 == m1[0]) and np.all(m2 == m2[0]) and m1[0] == m2[0]
    for db, fr, w in ((b1, f1, w1), (b2, f2, w2)):
        got = db.replicate(w)
        ref = _reference_rho(fr, w, per_protein=False)
        for k in ('a', 'b'):
            assert got[k] == pytest.approx(ref[k], abs=1e-9)


def test_mutation_draw_preserves_within_protein_and_total_sizes():
    rng = np.random.default_rng(3)
    fr = _make_frame(rng, n=240, n_codes=8)
    b_grp = boot.DatasetBoot(fr, ['a'], per_protein=True)
    w_grp = boot.draw_weights('mutation', b_grp, rng)
    for _, sub in fr.spine.groupby('group'):
        assert w_grp[sub.index.to_numpy()].sum() == len(sub)

    b_pool = boot.DatasetBoot(fr, ['a'], per_protein=False)
    w_pool = boot.draw_weights('mutation', b_pool, rng)
    assert w_pool.sum() == len(fr)


@pytest.mark.parametrize('kind', ['protein', 'site', 'mutation'])
def test_bootstrap_rho_and_interval_helpers(kind):
    rng = np.random.default_rng(4)
    frames = {'d1': _make_frame(rng, 'd1', 180, 10), 'd2': _make_frame(rng, 'd2', 180, 10)}
    point, reps = boot.bootstrap_rho(frames, ['a', 'b', 'c'], kind=kind, B=30, seed=7, per_protein=False)
    for ds in ('d1', 'd2'):
        assert set(point[ds]) == {'a', 'b', 'c'}
        assert reps[ds]['a'].shape == (30,) and np.isfinite(reps[ds]['a']).all()
        lo, hi = boot.ci(reps[ds]['a'])
        assert lo <= point[ds]['a'][0] <= hi or lo <= hi

    assert boot.ci([np.nan, np.nan]) == (np.nan, np.nan)
    diff_pos = boot.paired_diff(np.ones(99), np.zeros(99))
    assert diff_pos['mean'] == pytest.approx(1.0) and diff_pos['p'] == pytest.approx(1.0 / 100)
    diff_sym = boot.paired_diff(np.array([-1.0, 1.0, -1.0, 1.0]), np.zeros(4))
    assert diff_sym['mean'] == pytest.approx(0.0) and diff_sym['p'] == pytest.approx(1.0)
