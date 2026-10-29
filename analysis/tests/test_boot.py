"""The count-weighted Spearman engine equals scipy's spearmanr on the expanded sample."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msr_analysis.boot import SpearmanEngine  # noqa: E402


def _expand(x, y, w):
    idx = np.repeat(np.arange(len(x)), w.astype(int))
    return x[idx], y[idx]


def test_pooled_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    n = 300
    x = np.round(rng.normal(size=n), 1)            # many ties
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
