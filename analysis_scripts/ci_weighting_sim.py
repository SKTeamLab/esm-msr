"""Does weighting by label precision help a regression, and which form of it helps a Plackett-Luce (ListMLE) ranking loss?

The evidence behind docs/ci_weighting.md section 3 and the design of ``ListMLELoss(weights=...)``. CPU only, a minute or two.

A linear scorer s = X b is fitted to labels y = mu + sigma_i * eps, mu = X b* + u, where ``u`` is model misfit (a part of mu the
scorer cannot express) and sigma_i is heteroscedastic (lognormal, scaled by ``noise``). The score is Spearman(X b, X b*) on clean
held-out items, paired over seeds (mean difference to the unweighted fit +- its standard error). The weights are the --ci_weight
shape, 1 / (sigma_i^2 + 0.02^2): the label SD is the only noise they can see, whatever the misfit is.

Ranking variants, all fitted by L-BFGS on the same lists:
  chosen       each Plackett-Luce term scaled by its chosen item's weight (the first design; REJECTED)
  offset(+)    every score offset by +log w, terms unscaled
  offset(-)    the wrong sign, as a control for the mechanism
  chosen+off   offset by +log w AND each term scaled by w   (what ListMLELoss does)
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

N, D, LIST, NTEST, TAU = 3200, 20, 16, 3000, 0.02
VARIANTS = {'none': (False, 0.0), 'chosen': (True, 0.0), 'offset(+)': (False, 1.0), 'offset(-)': (False, -1.0),
            'chosen+off': (True, 1.0)}


def make(seed, noise, misfit):
    g = np.random.default_rng(seed)
    b = g.normal(size=D) / np.sqrt(D)
    X = g.normal(size=(N + NTEST, D)).astype(np.float32)
    lin = X @ b
    lin = (lin - lin.mean()) / lin.std()
    u = g.normal(size=N + NTEST) * misfit
    sigma = noise * 0.5 * np.exp(0.8 * g.normal(size=N + NTEST))          # median sigma 0.5 * noise, heavy right tail
    y = lin + u + sigma * g.normal(size=N + NTEST)
    return y[:N].astype(np.float32), sigma[:N].astype(np.float32), X[:N], X[N:], lin[N:]


def fit_reg(X, y, w):
    Xb = np.c_[X, np.ones(len(X))]
    beta = np.linalg.solve(Xb.T @ (Xb * w[:, None]) + 1e-6 * np.eye(D + 1), Xb.T @ (y * w))
    return beta[:-1]


def fit_rank(X, y, w, variant, seed, steps=250):
    scale_term, c = VARIANTS[variant]
    g = torch.Generator().manual_seed(seed)
    Xt, yt, wt = torch.tensor(X), torch.tensor(y), torch.tensor(w / w.mean())
    beta = torch.zeros(D, requires_grad=True)
    opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=steps, line_search_fn='strong_wolfe')
    lists = torch.cat([torch.randperm(N, generator=g)[: (N // LIST) * LIST].view(-1, LIST) for _ in range(6)])
    lists = torch.gather(lists, 1, torch.argsort(-yt[lists], dim=1))      # each list in its label order
    w_o = wt[lists]

    def closure():
        opt.zero_grad()
        s = Xt[lists] @ beta + c * torch.log(w_o)
        lse = torch.flip(torch.logcumsumexp(torch.flip(s, [1]), 1), [1])
        loss = -((w_o if scale_term else torch.ones_like(w_o)) * (s - lse)).sum(1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return beta.detach().numpy()


def run(noise, misfit, seeds):
    rows = []
    for sd in range(seeds):
        y, sigma, Xtr, Xte, truth = make(sd, noise, misfit)
        w = 1.0 / (sigma ** 2 + TAU ** 2)
        r = {'reg': spearmanr(Xte @ fit_reg(Xtr, y, np.ones(N)), truth)[0],
             'reg weighted': spearmanr(Xte @ fit_reg(Xtr, y, w), truth)[0]}
        for v in VARIANTS:
            r['rank ' + v] = spearmanr(Xte @ fit_rank(Xtr, y, w, v, sd), truth)[0]
        rows.append(r)
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    d = lambda a, b: '%+.4f+-%.4f' % ((A[a] - A[b]).mean(), (A[a] - A[b]).std(ddof=1) / np.sqrt(seeds))
    print('noise x%-4.2f misfit %.1f | rho (unweighted) reg %.4f rank %.4f | reg weighted %s | rank: %s' % (
        noise, misfit, A['reg'].mean(), A['rank none'].mean(), d('reg weighted', 'reg'),
        '  '.join('%s %s' % (v, d('rank ' + v, 'rank none')) for v in VARIANTS if v != 'none')), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--noise', type=float, nargs='+', default=[0.25, 1.0, 3.0])
    ap.add_argument('--misfit', type=float, nargs='+', default=[0.0, 0.5])
    a = ap.parse_args()
    torch.set_num_threads(2)
    print('change in Spearman vs the unweighted fit, paired over %d seeds (mean +- s.e.)' % a.seeds)
    for m in a.misfit:
        for n in a.noise:
            run(n, m, a.seeds)
