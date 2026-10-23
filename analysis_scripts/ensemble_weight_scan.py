"""How much of the MT head's predicted epistasis should a multi-mutant prediction keep?

Rescores saved ESM-MSR predictions under a one-parameter family of WT/MT ensembles, with no GPU: every
per-variant prediction file already carries wt_lora_pred, mt_lora_pred and the mutation count, so any weighting
can be evaluated offline. Results and conclusions: docs/BENCHMARKS_OCT2026.md section 2.6.

Writing A = wt_lora_pred (additive by construction) and M = mt_lora_pred (the direct multi-mutant head),
    (1-L)*A + L*M  ==  A + L*(M - A)
so the ensemble weight L IS the shrinkage applied to the MT head's implied epistasis e = M - A.
The shipped recipe combined_pred = 0.5*(A+M) is already L = 0.5, i.e. half the predicted interaction.
L = 1/N is the (N-1)/N * WT + 1/N * MT proposal. This scans L overall and per mutation count.
Run under msr_venv on the cluster; CPU, reads the saved per-variant predictions.

  python analysis_scripts/ensemble_weight_scan.py
"""
import glob, os, sys
import numpy as np, pandas as pd

D = '/home/sreeves/esm-msr-devel/analysis_notebooks/predictions'
CKPT = {'capped': 'esm-msr-baseline-w2a4-m16a16-capped',
        'dc8_s1': 'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s1'}
SETS = ['PTMUL', 'ptmuld', 'GB1_Wu_2016_binding_domain', 'splits_oct06_capped-test',
        'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain']
GRID = [0.0, 0.1, 0.2, 0.25, 1/3, 0.4, 0.5, 0.6, 0.75, 1.0]


def rho(a, b):
    v = pd.DataFrame({'a': a, 'b': b}).dropna()
    return np.nan if len(v) < 3 else v.corr('spearman').iloc[0, 1]


def load(ds, ck):
    f = sorted(glob.glob(f'{D}/{ds}/{CKPT[ck]}/*_epsilon1_unmasked_predictions.csv'))
    if not f:
        return None
    d = pd.read_csv(f[0], low_memory=False)
    truth = next((c for c in ['ddG_ML', 'ddG', 'DDG'] if c in d.columns and d[c].notna().any()), None)
    if truth is None or 'wt_lora_pred' not in d.columns:
        return None
    if 'NMUT' in d.columns and d['NMUT'].notna().any():
        n = d['NMUT'].astype(float)
    else:
        mt = next(c for c in ['mut_type', 'mut_info'] if c in d.columns)
        n = d[mt].astype(str).str.count(':') + 1
    out = pd.DataFrame({'A': d.wt_lora_pred, 'M': d.mt_lora_pred, 'y': d[truth], 'N': n})
    if 'code_wt' in d.columns:
        out['grp'] = d.code_wt
    elif 'code' in d.columns:
        out['grp'] = d.code
    else:
        out['grp'] = 'all'
    chk = (0.5 * (d.wt_lora_pred + d.mt_lora_pred) - d.combined_pred).abs().max() if 'combined_pred' in d else np.nan
    return out.dropna(subset=['A', 'M', 'y']), chk


def scored(df, L, per_group):
    """rho of (1-L)A + LM against truth; L may be a scalar or a per-row array."""
    p = (1 - L) * df.A + L * df.M
    if not per_group:
        return rho(p, df.y)
    v = [rho(g.p, g.y) for _, g in pd.DataFrame({'p': p, 'y': df.y, 'grp': df.grp}).groupby('grp') if len(g) > 2]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


print('A = WT head (additive), M = MT head.  (1-L)A + LM  ==  A + L*(M-A), so L shrinks the predicted epistasis.')
for ck in CKPT:
    print(f'\n################ checkpoint: {ck}')
    for ds in SETS:
        r = load(ds, ck)
        if r is None:
            print(f'  {ds}: no predictions'); continue
        df, chk = r
        per_group = ds.startswith('splits_oct06')          # Tsuboyama is reported per protein
        multi = df[df.N > 1]
        if not len(multi):
            continue
        row = {f'L={L:.2f}': round(scored(multi, L, per_group), 4) for L in GRID}
        row['L=1/N'] = round(scored(multi, 1.0 / multi.N, per_group), 4)
        best = max(GRID, key=lambda L: (scored(multi, L, per_group) if not np.isnan(scored(multi, L, per_group)) else -9))
        print(f'\n  == {ds}  (multi-mutants only: n={len(multi)}, N up to {int(multi.N.max())}, '
              f'combined_pred check {chk:.1e})')
        print('    ' + '  '.join(f'{k}:{v}' for k, v in row.items()))
        print(f'    best constant L on the grid = {best:.2f}   (shipped recipe is L=0.50)')
        by = []
        for n, g in multi.groupby(multi.N.astype(int)):
            if len(g) < 30:
                continue
            vals = {L: scored(g, L, per_group) for L in GRID}
            bn = max(vals, key=lambda L: (vals[L] if not np.isnan(vals[L]) else -9))
            by.append(dict(N=n, n_rows=len(g), rho_L050=round(vals[0.5], 4), rho_best=round(vals[bn], 4),
                           best_L=round(bn, 2), rho_L_1overN=round(scored(g, 1.0 / n, per_group), 4)))
        if by:
            print(pd.DataFrame(by).to_string(index=False).replace('\n', '\n    ').rjust(4))
