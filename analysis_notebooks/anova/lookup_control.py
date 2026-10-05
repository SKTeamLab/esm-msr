"""Control for the per-pair flip metric: how much can a 20x20 lookup of (scored residue, partner residue) earn, with no
knowledge of position, structure or context?  The table is the mean measured ddG_AB over TRAINING libraries for each ordered
(scored substitution X, partner residue Y); it is applied to VALIDATION libraries (held out), per-pair flip metric.
Variants: 'raw' table (includes the per-X main effect), 'interaction' table (X x Y part left after removing row and column
means, so the within-column ordering is the SAME as an X-only predictor plus a partner-dependent correction), and the X-only
control (table averaged over Y: must score exactly 0)."""
import pickle, sys, json
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], '/dev/null']
sys.path.insert(0, 'analysis_notebooks/anova')
from esm_msr import stats
import anova_variance_partition as A

T, _ = A.build_table()
sp = pickle.load(open(A.SPLITS, 'rb'))
stem = lambda l: [x.replace('.pdb', '') for x in l]
tr, va = set(stem(sp['train'])), set(stem(sp['val']))
T['split'] = np.where(T.code_wt.isin(tr), 'train', np.where(T.code_wt.isin(va), 'val', 'other'))
print(T.split.value_counts().to_dict(), flush=True)
Tt, Tv = T[T.split == 'train'], T[T.split == 'val'].reset_index(drop=True)
# ordered cells (scored X, partner Y, value) from both orientations
def cells(df):
    return pd.concat([pd.DataFrame({'X': df.a, 'Y': df.b, 'v': df.ddG_AB}), pd.DataFrame({'X': df.b, 'Y': df.a, 'v': df.ddG_AB})])
C = cells(Tt)
M = C.pivot_table(index='X', columns='Y', values='v', aggfunc='mean').reindex(index=list(A.AA), columns=list(A.AA))
N = C.pivot_table(index='X', columns='Y', values='v', aggfunc='count').reindex(index=list(A.AA), columns=list(A.AA))
print('table cells with data: %d/400, median n per cell %d' % (M.notna().sum().sum(), np.nanmedian(N.values)), flush=True)
M = M.fillna(M.stack().mean())
rowm, colm, g = M.mean(1), M.mean(0), M.values.mean()
I = M - rowm.values[:, None] - colm.values[None, :] + g          # interaction part
tables = {'raw (X,Y) table': M, 'interaction part + X main effect': I.add(rowm, axis=0), 'X-only (control, expect 0)': M.mean(1).to_frame().reindex(columns=list(A.AA)).ffill(axis=1)}
tables['X-only (control, expect 0)'] = pd.DataFrame(np.repeat(M.mean(1).values[:, None], 20, 1), index=M.index, columns=M.columns)
aa = {c: i for i, c in enumerate(A.AA)}
out = {}
for name, tab in tables.items():
    f = lambda X, Y: tab.values[aa[X], aa[Y]]
    p1 = np.array([f(a, b) for a, b in zip(Tv.a, Tv.b)]); p2 = np.array([f(b, a) for a, b in zip(Tv.a, Tv.b)])
    k1 = (Tv.code + '|' + Tv.p.astype(str) + '_' + Tv.q.astype(str) + '|' + Tv.b).tolist()
    k2 = (Tv.code + '|' + Tv.q.astype(str) + '_' + Tv.p.astype(str) + '|' + Tv.a).tolist()
    r1, r2 = Tv.a.map(aa).to_numpy(), Tv.b.map(aa).to_numpy()
    tgt = Tv.ddG_AB.to_numpy()
    one = stats.flip_signature_rho(p1, tgt, k1, r1, min_len=4)
    both = stats.flip_signature_rho(np.concatenate([p1, p2]), np.concatenate([tgt, tgt]), k1 + k2, np.concatenate([r1, r2]), min_len=4)
    pooled = stats.flip_signature_rho(np.concatenate([p1, p2]), np.concatenate([tgt, tgt]),
        (Tv.code + '|' + Tv.p.astype(str) + '|' + Tv.q.astype(str) + Tv.b).tolist() + (Tv.code + '|' + Tv.q.astype(str) + '|' + Tv.p.astype(str) + Tv.a).tolist(),
        np.concatenate([r1, r2]), min_len=4)
    out[name] = {'per-pair, one orientation': one, 'per-pair, both orientations': both, 'pooled (validation-style)': pooled}
    print(name, {k: tuple(round(x, 4) if isinstance(x, float) else x for x in v) for k, v in out[name].items()}, flush=True)
json.dump(out, open('docs/anova/lookup_control.json', 'w'), indent=1)
