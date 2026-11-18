#!/usr/bin/env python
"""How well can ``cell_mag`` be predicted on the Faure datasets?  Blocks, noise ceilings and ESM-MSR.

``cell_mag`` (``esm_msr.epi_metrics``): the doubles of one position pair form a matrix (rows: residue at position i, columns: residue at j); each matrix is trimmed to a complete block (>= 3 x 3, pairs with
at least 20 doubles), the beyond-global residuals r = eps - E[eps | x] are double-centred (row, column and block means removed), and the Spearman rho between predicted and measured double-centred residuals
is averaged over blocks.  Double-centring removes everything that is additive in the two residues, so the noise of the singles drops out: only the double's own error sigma_AB is left.

  1. which sets have blocks (PDZ3: none, each pair has one fixed residue on one side; SH3: hundreds);
  2. the noise ceiling: the expected ``cell_mag`` of a perfect predictor of the true cell effects, under three noise models of the double's own measurement
       nominal        the reported error of each double (DiMSum)
       R = 0.87, 0.92 constant noise, calibrated so that two replicates correlate at Pearson R (Faure et al. report 0.87-0.92) and the reported value is the mean of three replicates
     The true double-centred cell effects are taken as the observed ones shrunk to the signal variance (signal variance = observed variance of the double-centred residual - noise variance, the
     shape of the distribution kept), measured = true + noise, and the mean block Spearman is averaged over simulations.  Given for all blocks and for the blocks above the floor (mean additive
     expectation x > -0.5).
  3. ESM-MSR (FINAL_p0, seeds 1-3; predicted epistasis comb(AB) - WT(A) - WT(B)) on the same blocks, and its pooled beyond-global rho.

Writes ``results/cell_mag_ceiling.csv`` and ``results/cell_mag_blocks.csv``; figure ``cell_mag_ceiling``.

    python analysis/faure_epistasis/03_cell_ceiling.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import faure as F
from esm_msr import epi_metrics as E
from msr_analysis.final_runs import load_final_frame

FLOOR_X = -0.5
NSIM = 30
R_REP = (0.87, 0.92)
SEED = 7


def dc(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def dc_var(S2):
    """Variance of each cell of dc(e) for independent errors e with variances S2 (R x C)."""
    R, C = S2.shape
    Pr, Pc = np.eye(R) - 1.0 / R, np.eye(C) - 1.0 / C
    return (Pr ** 2) @ S2 @ (Pc ** 2).T


def blocks_of(d):
    """Complete blocks (>= 3 x 3) of the pairs with at least ``MIN_PAIR_CELLS`` doubles: list of (row index matrix into d, mean x)."""
    out = []
    for (p1, p2), g in d.groupby(['pos1', 'pos2']):
        if len(g) < E.MIN_PAIR_CELLS:
            continue
        ra, ca = sorted(g.aa1.unique()), sorted(g.aa2.unique())
        ri, ci = {a: i for i, a in enumerate(ra)}, {b: j for j, b in enumerate(ca)}
        M = np.full((len(ra), len(ca)), np.nan)
        I = np.full((len(ra), len(ca)), -1, int)
        for ix, a, b, r in zip(g.index, g.aa1, g.aa2, g.r):
            M[ri[a], ci[b]] = r
            I[ri[a], ci[b]] = ix
        blk = E._complete_block(M)
        if blk is None:
            continue
        sub = I[np.ix_(*blk)]
        out.append((sub, float(d.x.to_numpy()[sub.ravel()].mean()), p1, p2))
    return out


def noise_matrices(d, sub, model, var_f):
    if model == 'nominal':
        return (d.s.to_numpy() ** 2)[sub]
    R = float(model.split('=')[1])
    return np.full(sub.shape, var_f * (1 - R) / (3 * R))            # merged triplicate: one replicate has three times the error variance


def ceiling(d, blocks, model, var_f, rng):
    """(mean block Spearman of a perfect predictor, Pearson ceiling, signal SD, noise SD) under one noise model."""
    C = [dc(d.r.to_numpy()[b[0]]) for b in blocks]
    S2 = [noise_matrices(d, b[0], model, var_f) for b in blocks]
    V = [dc_var(s) for s in S2]
    c2 = float(np.mean(np.concatenate([c.ravel() ** 2 for c in C])))
    nv = float(np.mean(np.concatenate([v.ravel() for v in V])))
    sig = max(c2 - nv, 0.0)
    kappa = np.sqrt(sig / c2)
    sims = []
    for _ in range(NSIM):
        rh = []
        for c, s2 in zip(C, S2):
            T = kappa * c
            meas = T + dc(rng.normal(0, np.sqrt(s2)))
            rh.append(spearmanr(T.ravel(), meas.ravel())[0] if sig > 0 else 0.0)
        sims.append(np.nanmean(rh))
    return float(np.mean(sims)), float(np.sqrt(sig / (sig + nv))) if sig > 0 else 0.0, float(np.sqrt(sig)), float(np.sqrt(nv))


def model_preds(ds, d):
    """ESM-MSR (FINAL_p0) beyond-global predicted residuals on the doubles, one row per seed (NaN where the model has no prediction), and the predicted epistasis."""
    fr = load_final_frame(ds, families=['FINAL_p0'])
    pos = {k: i for i, k in enumerate(fr.spine['id'].astype(str))}
    ix = np.array([pos.get(i, -1) for i in d.id])
    ok = ix >= 0
    res, epis = [], []
    for s in range(fr.preds['FINAL_p0:epi'].shape[0]):
        epi = fr.preds['FINAL_p0:epi'][s][np.where(ok, ix, 0)]
        add = fr.preds['FINAL_p0:wt+'][s][np.where(ok, ix, 0)]
        full = np.full(len(d), np.nan)
        good = ok & np.isfinite(epi) & np.isfinite(add)
        full[good] = epi[good] - E.smooth_on(add[good], epi[good])
        res.append(full)
        epis.append(np.where(good, epi, np.nan))
    return np.array(res), np.array(epis)


def model_cell_mag(P, d, blocks):
    """Mean over blocks (and seeds) of the Spearman rho between the double-centred predicted and measured residuals."""
    vals = []
    for full in P:
        rb = []
        for b in blocks:
            sub = b[0]
            if np.isfinite(full[sub]).all():
                rb.append(spearmanr(dc(full[sub]).ravel(), dc(d.r.to_numpy()[sub]).ravel())[0])
        vals.append(np.nanmean(rb) if rb else np.nan)
    return float(np.mean(vals))


def pooled_rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0])


def main():
    rng = np.random.default_rng(SEED)
    rows, brows = [], []
    for ds in F.SETS:
        single, d, raw = F.load(ds)
        d['r'] = d.eps - E.smooth_on(d.x.to_numpy(), d.eps.to_numpy())
        allv = raw[raw.NMUTS <= 2]
        var_f = float(allv.truth.var() - (allv.sigma ** 2).mean())           # variance of the true fitness
        bl = blocks_of(d)
        for b in bl:
            brows.append(dict(set=ds, pos1=b[2], pos2=b[3], rows=b[0].shape[0], cols=b[0].shape[1], mean_x=b[1]))
        P, epis = model_preds(ds, d)
        base = dict(set=ds, doubles=len(d), pairs_ge20=int((d.groupby(['pos1', 'pos2']).size() >= E.MIN_PAIR_CELLS).sum()),
                    model_beyond_rho=float(np.mean([pooled_rho(p, d.r.to_numpy()) for p in P])), model_naive_rho=float(np.mean([pooled_rho(e, d.eps.to_numpy()) for e in epis])))
        if not bl:
            rows.append(dict(base, stratum='no complete block', blocks=0, cells=0))
            print(ds, 'no blocks; model beyond_rho %.3f naive %.3f' % (base['model_beyond_rho'], base['model_naive_rho']), flush=True)
            continue
        for stratum, sel in (('all blocks', bl), ('blocks above floor', [b for b in bl if b[1] > FLOOR_X])):
            if len(sel) < 5:
                continue
            mc = model_cell_mag(P, d, sel)
            for model in ('nominal', f'R={R_REP[0]}', f'R={R_REP[1]}'):
                cs, pc, ssd, nsd = ceiling(d, sel, model, var_f, rng)
                rows.append(dict(base, stratum=stratum, noise_model=model, blocks=len(sel), cells=int(sum(b[0].size for b in sel)), ceiling_spearman=cs, ceiling_pearson=pc, signal_sd=ssd, noise_sd=nsd,
                                 model_cell_mag=mc))
        print(ds, 'blocks', len(bl), flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(F.OUT / 'cell_mag_ceiling.csv', index=False, float_format='%.4g')
    pd.DataFrame(brows).to_csv(F.OUT / 'cell_mag_blocks.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(C.drop(columns=['doubles']).round(3).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
