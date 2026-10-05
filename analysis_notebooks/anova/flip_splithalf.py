"""Unbiased test of how much of the validation-style flip metric a predictor can earn WITHOUT knowing the partner residue.
The predictor for a cell is the mean of ddG_AB over a DISJOINT half of the partner residues (so it never sees the cell itself
or its column): it knows the scored substitution and the partner POSITION, nothing about the partner residue."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis_notebooks/anova')
from esm_msr import stats
import anova_variance_partition as A
OUT = sys.argv[1]
T = pd.read_pickle(f'{OUT}/T.pkl'); rng = np.random.default_rng(2)
aa = {c: i for i, c in enumerate(A.AA)}
res = {}
for rep in range(3):
    # random halves of partner residues, separately for each pair and each orientation
    hb = {(p, b): rng.random() < 0.5 for p in T.pair.unique() for b in A.AA}
    ha = {(p, a): rng.random() < 0.5 for p in T.pair.unique() for a in A.AA}
    T['hb'] = [hb[(p, b)] for p, b in zip(T.pair, T.b)]
    T['ha'] = [ha[(p, a)] for p, a in zip(T.pair, T.a)]
    # orientation 1: scored a, partner residue b. predictor = mean over cells with hb True of row a, applied to cells with hb False
    m1 = T[T.hb].groupby(['pair', 'a']).ddG_AB.mean().rename('pred1')
    P1 = T.join(m1, on=['pair', 'a']); P1.loc[P1.hb, 'pred1'] = np.nan
    # orientation 2: scored b, partner residue a. predictor = mean over cells with ha True of column b, applied to cells with ha False
    m2 = T[T.ha].groupby(['pair', 'b']).ddG_AB.mean().rename('pred2')
    P2 = T.join(m2, on=['pair', 'b']); P2.loc[P2.ha, 'pred2'] = np.nan
    k1 = (T.code + '|' + T.p.astype(str) + '|' + T.q.astype(str) + T.b).tolist()
    k2 = (T.code + '|' + T.q.astype(str) + '|' + T.p.astype(str) + T.a).tolist()
    r1, r2 = T.a.map(aa).to_numpy(), T.b.map(aa).to_numpy()
    tgt = np.concatenate([T.ddG_AB, T.ddG_AB])
    pred = np.concatenate([P1.pred1, P2.pred2])
    rho_v = stats.flip_signature_rho(pred, tgt, k1 + k2, np.concatenate([r1, r2]), min_len=4)
    kp = (T.code + '|' + T.p.astype(str) + '_' + T.q.astype(str) + '|' + T.b).tolist()
    rho_p = stats.flip_signature_rho(P1.pred1.to_numpy(), T.ddG_AB.to_numpy(), kp, r1, min_len=4)
    res[f'rep{rep}'] = {'validation-style': rho_v, 'pair-level': rho_p}
    print(rep, res[f'rep{rep}'], flush=True)
json.dump(res, open(f'{OUT}/splithalf.json', 'w'), indent=1)
