"""Variance partition of measured double-mutant epistasis (Tsuboyama MegaScale).

Splits  dddG = ddG_AB - ddG_A - ddG_B  (measured, kcal/mol) into
  noise | global (assay saturation) | position-pair mean | per-substitution row+column effects | interaction
and reports what each identity-independent term would explain out-of-sample, the measurement-noise
ceiling, and what the validation flip metric does and does not reward.

Usage: PYTHONPATH=src python analysis_notebooks/anova/anova_variance_partition.py [outdir]
Writes results.json and T.pkl (the doubles table) to outdir. See docs/anova_epistasis_report.md.
"""
import json, os, pickle, re, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

RAW = '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
SPLITS = '/home/sareeves/software/esm-msr/data/hyperopt_splits.pkl'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'analysis_notebooks/anova/out'
CLIP_MISSING = '--clip-missing' in sys.argv      # include '<-1' / '>5' variants at -1 / 5 instead of dropping them
SKIP_FLIP = '--skip-flip' in sys.argv
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
PAT = re.compile(r'^([A-Z])(\d+)([A-Z])$')
AA = 'ACDEFGHIKLMNPQRSTVWY'


def build_table():
    d = pd.read_csv(RAW, low_memory=False)
    raw_dG = d['dG_ML'].astype(str)
    for c in ['dG_ML', 'ddG_ML', 'deltaG_t', 'deltaG_c']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['dG_clipped'] = d['dG_ML'].copy()
    d.loc[raw_dG == '<-1', 'dG_clipped'] = -1.0          # confidently below the assay range
    d.loc[raw_dG == '>5', 'dG_clipped'] = 5.0            # confidently above it
    d['code'] = (d['WT_name'].str.replace('.pdb_', '_', regex=False).str.replace('.pdb', '', regex=False)
                 .str.replace('|', '_', regex=False))
    d['code_wt'] = d['WT_name'].str.split('.pdb').str[0].str.replace('|', '_', regex=False)
    wt = d[d.mut_type == 'wt'].groupby('code').dG_ML.mean()
    if CLIP_MISSING:
        d['ddG_use'] = d['dG_clipped'] - d['code'].map(wt)
        d['dG_use'] = d['dG_clipped']
    else:
        d['ddG_use'], d['dG_use'] = d['ddG_ML'], d['dG_ML']
    d = d[d.ddG_use.notna() & d.dG_use.notna() & ~d.mut_type.str.contains('wt|ins|del', na=False) & (d.WT_name != '1UBQ.pdb_L43A')]
    # noise: protease-to-protease disagreement in the well-measured range, centred
    s = d[['deltaG_t', 'deltaG_c', 'dG_ML']].dropna()
    s = s[s.dG_ML.between(0, 4) & s.deltaG_t.between(-1, 5) & s.deltaG_c.between(-1, 5)]
    diff = s.deltaG_t - s.deltaG_c
    sigma_single_protease = float(diff.std() / np.sqrt(2))
    g = d.groupby(['code', 'mut_type'], as_index=False).agg(dG=('dG_use', 'mean'), ddG=('ddG_use', 'mean'),
                                                          code_wt=('code_wt', 'first'))
    g['nm'] = g.mut_type.str.count(':') + 1
    S = g[g.nm == 1].set_index(['code', 'mut_type'])
    rows = []
    for r in g[g.nm == 2].itertuples():
        ms = [PAT.match(t) for t in r.mut_type.split(':')]
        if any(m is None for m in ms):
            continue
        (a1, p1, b1), (a2, p2, b2) = [(m.group(1), int(m.group(2)), m.group(3)) for m in ms]
        if p1 == p2:
            continue
        if p1 > p2:
            (a1, p1, b1), (a2, p2, b2) = (a2, p2, b2), (a1, p1, b1)
        k1, k2 = (r.code, f'{a1}{p1}{b1}'), (r.code, f'{a2}{p2}{b2}')
        if k1 not in S.index or k2 not in S.index:
            continue
        rows.append((r.code, r.code_wt, p1, p2, b1, b2, r.dG, r.ddG, S.at[k1, 'ddG'], S.at[k2, 'ddG']))
    T = pd.DataFrame(rows, columns=['code', 'code_wt', 'p', 'q', 'a', 'b', 'dG_AB', 'ddG_AB', 'ddG_A', 'ddG_B'])
    T['dG_wt'] = T.code.map(wt)
    T = T.dropna(subset=['dG_wt']).reset_index(drop=True)
    T['S'] = T.ddG_A + T.ddG_B
    T['x'] = T.dG_wt + T.S                      # additive-predicted dG of the double
    T['dddG'] = T.ddG_AB - T.S
    T['pair'] = T.code + '|' + T.p.astype(str) + '|' + T.q.astype(str)
    return T, sigma_single_protease


def fit_rowcol(df, col, n_iter=15):
    """Backfit additive row (a) + column (b) effects per pair on residual `col`. Returns dicts keyed by pair."""
    R, C = {}, {}
    for pair, g in df.groupby('pair'):
        a, b, y = g.a.to_numpy(), g.b.to_numpy(), g[col].to_numpy()
        ra, cb = {k: 0.0 for k in set(a)}, {k: 0.0 for k in set(b)}
        for _ in range(n_iter):
            for k in ra:
                m = a == k
                ra[k] = float(np.mean(y[m] - np.array([cb[v] for v in b[m]])))
            for k in cb:
                m = b == k
                cb[k] = float(np.mean(y[m] - np.array([ra[v] for v in a[m]])))
        R[pair], C[pair] = ra, cb
    return R, C


def predict_levels(train, test):
    """Cumulative predictions of dddG for `test` cells using only `train` cells."""
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(train.x, train.dG_AB)
    g_tr, g_te = iso.predict(train.x) - train.x, iso.predict(test.x) - test.x
    tr = train.assign(r1=train.dddG - g_tr)
    m = tr.groupby('pair').r1.mean()
    m_te = test.pair.map(m).fillna(0.0).to_numpy()
    tr['r2'] = tr.r1 - tr.pair.map(m)
    R, C = fit_rowcol(tr, 'r2')
    rc = np.array([R.get(p, {}).get(a, 0.0) + C.get(p, {}).get(b, 0.0)
                   for p, a, b in zip(test.pair, test.a, test.b)])
    base = float(train.dddG.mean())
    return {'M0 constant': np.full(len(test), base), 'M1 +global': np.asarray(g_te, float) + 0 * base,
            'M2 +pair mean': np.asarray(g_te, float) + m_te,
            'M3 +row/col': np.asarray(g_te, float) + m_te + rc}


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def flip_rho(items, pred, target):
    from esm_msr import stats
    keys, rows = items
    return stats.flip_signature_rho(pred, target, keys, rows, min_len=4)


def main():
    T, sig1 = build_table()
    T.to_pickle(f'{OUT}/T.pkl')
    res = {'n_doubles': int(len(T)), 'n_libs': int(T.code.nunique()), 'n_pairs': int(T.pair.nunique()),
           'cells_per_pair_median': float(T.groupby('pair').size().median()),
           'var_dddG': float(T.dddG.var()), 'mean_dddG': float(T.dddG.mean()),
           'sigma_single_protease': sig1, 'sigma_ml_optimistic': sig1 / np.sqrt(2)}
    V = res['var_dddG']
    for tag, s in (('opt', sig1 / np.sqrt(2)), ('pess', sig1)):
        res[f'noise_total_share_{tag}'] = 3 * s * s / V          # sigma_Y^2 + sigma_A^2 + sigma_B^2
        res[f'noise_in_residual_share_{tag}'] = s * s / V        # only the double's own noise lands in the residual

    # ---- cell-level 5-fold CV of the hierarchy ----
    K = 5
    fold = rng.integers(0, K, len(T))
    preds = {k: np.zeros(len(T)) for k in ['M0 constant', 'M1 +global', 'M2 +pair mean', 'M3 +row/col']}
    for f in range(K):
        tr, te = T[fold != f], T[fold == f]
        for k, v in predict_levels(tr, te).items():
            preds[k][np.where(fold == f)[0]] = v
    y = T.dddG.to_numpy()
    res['cv'] = {k: {'r2': float(r2(y, p)), 'rho': float(spearmanr(y, p)[0])} for k, p in preds.items()}
    res['cv']['M0 constant']['rho'] = float('nan')
    # incremental shares of total variance
    ks = list(preds)
    res['cv_increment'] = {ks[i]: res['cv'][ks[i]]['r2'] - (res['cv'][ks[i - 1]]['r2'] if i else 0.0) for i in range(len(ks))}
    res['cv_resid_share'] = 1 - res['cv']['M3 +row/col']['r2']

    # ---- in-sample (for contrast) ----
    p_in = predict_levels(T, T)
    res['insample'] = {k: {'r2': float(r2(y, p))} for k, p in p_in.items()}

    # ---- per-x-bin: how much is global, how much is residual ----
    bins = [-10, -2, -1, 0, 1, 2, 3, 10]
    T['xb'] = pd.cut(T.x, bins)
    res['by_x'] = {str(b): {'n': int(len(g)), 'mean_dddG': float(g.dddG.mean()),
                            'var_dddG': float(g.dddG.var()),
                            'resid_var_M3': float(np.var(g.dddG - preds['M3 +row/col'][g.index]))}
                   for b, g in T.groupby('xb', observed=True)}

    # ---- out-of-protein baseline vs the model's validation libraries ----
    sp = pickle.load(open(SPLITS, 'rb'))
    cw = {k: {c.replace('.pdb', '').replace('|', '_') for c in sp[k]} for k in sp}
    tr_m, va_m = T.code_wt.isin(cw['train']), T.code_wt.isin(cw['val'])
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(T[tr_m].x, T[tr_m].dG_AB)
    V_ = T[va_m].copy()
    V_['g'] = iso.predict(V_.x) - V_.x
    per_lib = [spearmanr(g.g, g.dddG)[0] for _, g in V_.groupby('code') if len(g) >= 20]
    res['val_split'] = {'n_doubles': int(va_m.sum()), 'n_libs': int(V_.code.nunique()),
                        'rho_global_only_pooled': float(spearmanr(V_.g, V_.dddG)[0]),
                        'rho_global_only_per_lib_mean': float(np.nanmean(per_lib)),
                        'r2_global_only': float(r2(V_.dddG.to_numpy(), V_.g.to_numpy()))}

    if SKIP_FLIP:
        json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)
        print(json.dumps(res, indent=1))
        return
    # ---- flip metric: what it rewards ----
    aa = {c: i for i, c in enumerate(AA)}
    def expand(df, pair_level):
        k1 = (df.code + '|' + df.p.astype(str) + ('_' + df.q.astype(str) if pair_level else '') + '|' + df.q.astype(str) + df.b).tolist() \
            if not pair_level else (df.code + '|' + df.p.astype(str) + '_' + df.q.astype(str) + '|' + df.b).tolist()
        r1 = df.a.map(aa).to_numpy()
        if pair_level:
            return (k1, r1), None
        k2 = (df.code + '|' + df.q.astype(str) + '|' + df.p.astype(str) + df.a).tolist()
        r2_ = df.b.map(aa).to_numpy()
        return (k1 + k2, np.concatenate([r1, r2_])), None
    F = T.copy()
    # per-(pair, row) and per-(pair, col) means of the measured double = substitution effects that ignore the partner identity
    F['row_mean'] = F.groupby(['pair', 'a']).ddG_AB.transform('mean')
    F['col_mean'] = F.groupby(['pair', 'b']).ddG_AB.transform('mean')
    F['h'] = iso.predict(F.x) - F.dG_wt             # global-only prediction of ddG_AB
    out = {}
    for level, pair_level in (('validation-style (columns pooled over partner positions)', False),
                              ('pair-level (one matrix per position pair)', True)):
        items, _ = expand(F, pair_level)
        tgt = np.concatenate([F.ddG_AB, F.ddG_AB]) if not pair_level else F.ddG_AB.to_numpy()
        pr = {}
        pr['measured (perfect)'] = tgt
        pr['global only: h(additive)'] = np.concatenate([F.h, F.h]) if not pair_level else F.h.to_numpy()
        rm = np.concatenate([F.row_mean, F.col_mean]) if not pair_level else F.row_mean.to_numpy()
        # NB: this in-sample row/column mean includes the scored cell itself and is optimistic; the unbiased version is
        # flip_splithalf.py. A 'measured + independent noise' ceiling was removed: it re-uses the measured noise on both sides.
        pr['substitution mean, partner residue ignored (in-sample, biased up)'] = rm
        pr['random'] = rng.normal(size=len(tgt))
        out[level] = {}
        for k, p in pr.items():
            rho, npairs, ncells = flip_rho(items, p, tgt)
            out[level][k] = {'rho': rho, 'n_pairs': npairs, 'n_cells': ncells}
    res['flip'] = out
    json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
