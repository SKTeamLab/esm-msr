#!/usr/bin/env python
"""Is the ``cell_mag`` of ESM-MSR (FINAL_p0) on the GRB2 datasets different from 0?  Bootstrap and permutation tests (``03_cell_ceiling.py`` reports the values).

Per complete block (``blocks_of``) the Spearman rho between the double-centred predicted and measured beyond-global residuals, averaged over the three seeds; ``cell_mag`` is the mean over the blocks.
Three tests, for all blocks and for the blocks above the floor (mean additive expectation > -0.5):

  block bootstrap      the blocks are resampled (B = 1000; two-sided p = 2 min(P(mean <= 0), P(mean >= 0)), at least 1 / (B + 1))
  position bootstrap   the 55-56 positions are resampled with replacement and a block (i, j) gets the weight count(i) x count(j): blocks that share a position are not independent
  permutation          the predicted residuals are permuted among the cells of each block (before double-centring, the same permutation for the three seeds), 200 times: the null of a predictor with no
                       information about the cell, including the small bias double-centring gives a random predictor

Writes ``results/cell_mag_significance.csv``.

    python analysis/faure_epistasis/05_cell_mag_significance.py
"""
import importlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import faure as F
from esm_msr import epi_metrics as E

C3 = importlib.import_module('03_cell_ceiling')
B = 1000
NPERM = 200
FLOOR_X = C3.FLOOR_X


def rho(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def two_sided(x):
    return max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1.0 / (len(x) + 1))


def main():
    rng = np.random.default_rng(11)
    rows = []
    for ds in F.SETS[2:]:
        single, d, raw = F.load(ds)
        d['r'] = d.eps - E.smooth_on(d.x.to_numpy(), d.eps.to_numpy())
        bl = C3.blocks_of(d)
        P, _ = C3.model_preds(ds, d)
        R = d.r.to_numpy()
        for stratum, sel in (('all blocks', bl), ('blocks above floor', [b for b in bl if b[1] > FLOOR_X])):
            keep = [b for b in sel if all(np.isfinite(p[b[0]]).all() for p in P)]
            idx = [b[0] for b in keep]
            meas = [C3.dc(R[i]).ravel() for i in idx]
            per_seed = np.array([[rho(C3.dc(p[i]).ravel(), m) for i, m in zip(idx, meas)] for p in P])      # seeds x blocks
            v = per_seed.mean(axis=0)
            point = float(v.mean())
            bb = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
            pos = sorted({int(x) for b in keep for x in (b[2], b[3])})
            pidx = {p: k for k, p in enumerate(pos)}
            b1, b2 = np.array([pidx[b[2]] for b in keep]), np.array([pidx[b[3]] for b in keep])
            pb = []
            for _ in range(B):
                cnt = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
                w = cnt[b1] * cnt[b2]
                pb.append(float((w * v).sum() / w.sum()) if w.sum() > 0 else np.nan)
            pb = np.array(pb)
            null = []
            for _ in range(NPERM):
                vals = []
                for i, m in zip(idx, meas):
                    perm = rng.permutation(i.size).reshape(i.shape)
                    vals.append(np.mean([rho(C3.dc(p[i].ravel()[perm.ravel()].reshape(i.shape)).ravel(), m) for p in P]))
                null.append(np.mean(vals))
            null = np.array(null)
            rows.append(dict(set=ds, stratum=stratum, blocks=len(keep), cells=int(sum(i.size for i in idx)), cell_mag=point, seed_values='/'.join(f'{x:.3f}' for x in per_seed.mean(axis=1)),
                             block_boot_lo=np.quantile(bb, .025), block_boot_hi=np.quantile(bb, .975), block_boot_p=two_sided(bb), pos_boot_lo=np.nanquantile(pb, .025), pos_boot_hi=np.nanquantile(pb, .975),
                             pos_boot_p=two_sided(pb[np.isfinite(pb)]), null_mean=float(null.mean()), null_sd=float(null.std()), perm_p=float(max((np.sum(null >= point) + 1) / (NPERM + 1), 1 / (NPERM + 1))),
                             share_blocks_positive=float((v > 0).mean())))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(F.OUT / 'cell_mag_significance.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(D.round(4).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
