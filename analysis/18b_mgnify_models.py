"""ESM-MSR p0 and the baselines on the MGnify scaffolds and the Rocklin designs: the per-scaffold mean rho and the pooled rho (the two inferred-ddG columns of ``01_summary``'s pg_categories figures).

Homologs are excluded with the rule that built the Homology-Restricted MegaScale split (MMseqs2 E <= 1e-3, local identity > 0.40; no structure search was run on these sets; flags in
``reference/tsuboyama_rule_flags.csv`` from ``leak_audit/21_tsuboyama_rule.py``).  MGnify is every scaffold left after that (not the audit's low-identity stratum); the Rocklin designs likewise.

``per scaffold``  the mean over the scaffolds (designs) with at least 10 measured variants of the scaffold's own Spearman rho between prediction and ddG_ML, taken per seed and then averaged over the
                  seeds.  A within-scaffold rank cannot change with the Tobit link (dG_wt is constant in a scaffold), so every model is read as its raw ddG score.
``pooled``        one Spearman rho over all measured variants of the set (per seed, then averaged).  Here the scale matters: the baselines learn the assay's saturation implicitly, so ESM-MSR is read on the
                  observed scale, ``h(dG_wt + latent) - dG_wt`` (``esm_msr.val_stats.tobit_link``, the checkpoints' constants), and the baselines raw.
95% intervals resample scaffolds (``--B`` draws, the same draws for every model of a set).

Readouts follow the deployed rule: both sets hold multi-mutants, so the multi-mutant version of each model is read (ESM-MSR combined head, SPURS-multi, ThermoMPNN-D; ProteinMPNN and ProSST score every
variant directly).  ESM3-small-open (sigma 0) is the exception: its run holds the WT head only (the combined columns are placeholders).  A model enters
a set on the scaffolds it has scored completely (the baselines write whole chunks); ``coverage`` is that share of the set's scaffolds, and a model with partial coverage also gets an ESM-MSR row on the same
scaffolds / rows (``ESM-MSR, same scaffolds as ...``).

Needs ``final_mgnify/`` (FINAL_p0 and ``baselines/``).  Output: ``results/18_mgnify/mgnify_models_per_scaffold.csv`` and ``mgnify_models_pooled.csv``.
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from msr_analysis.common import out_dir
from msr_analysis.mgnify import RUNS, load_mgnify
from msr_analysis.registry import REFERENCE

OUT = out_dir('18_mgnify')
BASE = RUNS / 'baselines'
MIN_N = 10
SETS = {'MGnify': dict(run='all', src='mgnify', flag='mgnify'), 'Rocklin designs': dict(run='rdenovo', src='rocklin_denovo', flag='rocklin_denovo')}
# model -> (chunk dir template, column, seeds, readout)
BASELINES = {
    'SPURS': ('spurs_unfrozen_s{}', 'SPURS_multi', (1, 2, 3), 'SPURS-multi'),
    'ThermoMPNN': ('thermompnn_s{}', 'pred_combined', (1, 2, 3), 'ThermoMPNN-D'),
    'ProteinMPNN': ('proteinmpnn', 'mpnn_score', (None,), 'direct'),
    'ProSST': ('prosst', 'prosst', (None,), 'direct'),
}


def chunks(tpl, seed, src, col):
    fs = glob.glob(str(BASE / (tpl.format(seed) if seed else tpl) / 'mgnify' / src / 'chunk_*.csv.gz'))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, usecols=['id', col], low_memory=False) for f in fs]).drop_duplicates('id')
    return d.set_index('id')[col]


def esm3_wt(ids):
    """ESM3-small-open (sigma 0) WT head on the rows ``ids``: the finished run's ``predictions.csv.gz`` (MGnify scaffolds and Rocklin designs together), else its shards so far."""
    f = BASE / 'esm3_zeroshot' / 'mgnify' / 'predictions.csv.gz'
    if f.exists():
        d = pd.read_csv(f, usecols=['uid', 'wt_lora_pred'], low_memory=False)
    else:
        fs = glob.glob(str(BASE / 'esm3_zeroshot' / 'mgnify' / 'shards' / 'shard_*.csv'))
        if not fs:
            return None
        d = pd.concat([pd.read_csv(g, usecols=['uid', 'wt_lora_pred'], low_memory=False) for g in fs])
    return d.drop_duplicates('uid').set_index('uid')['wt_lora_pred'].reindex(ids).to_numpy(float)


def scaffold_rho(code, pred, y):
    """Spearman rho within each scaffold (``code``: integer scaffold index per row); NaN for a scaffold with any missing prediction."""
    df = pd.DataFrame({'g': code, 'p': pred, 'y': y})
    bad = df['p'].isna().groupby(df['g']).any()
    for c in ('p', 'y'):
        r = df.groupby('g')[c].rank(method='average')
        df['r' + c] = r - r.groupby(df['g']).transform('mean')
    num = (df['rp'] * df['ry']).groupby(df['g']).sum()
    den = np.sqrt((df['rp'] ** 2).groupby(df['g']).sum() * (df['ry'] ** 2).groupby(df['g']).sum())
    r = (num / den).replace([np.inf, -np.inf], np.nan)
    r[bad.reindex(r.index).to_numpy()] = np.nan
    return r


def pooled_rho(p, y):
    ok = np.isfinite(p)
    return float(np.corrcoef(rankdata(p[ok]), rankdata(y[ok]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000, help='scaffold bootstraps of the per-scaffold means')
    ap.add_argument('--B_pooled', type=int, default=200, help='scaffold bootstraps of the pooled rho (each redraw re-ranks every model)')
    a = ap.parse_args()
    flags = pd.read_csv(REFERENCE / 'tsuboyama_rule_flags.csv')
    rng = np.random.default_rng(3)
    rows_s, rows_p = [], []
    for name, cfg in SETS.items():
        fr = load_mgnify(run=cfg['run'])
        sp = fr.spine
        bad = set(flags.loc[(flags['dataset'] == cfg['flag']) & flags['homologous'], 'item_id'].astype(str))
        m = sp['use'].to_numpy() & ~sp['code_wt'].astype(str).isin(bad).to_numpy()
        d = sp[m].reset_index(drop=True)
        n_all = sp.loc[sp['use'], 'code_wt'].nunique()
        print(f'{name}: {d.code_wt.nunique()} of {n_all} scaffolds with a measured variant kept ({len(bad)} homologous by the rule), {len(d)} variants')
        code, uniq = pd.factorize(d['code_wt'])
        y = d['y'].to_numpy(float)
        cnt = np.bincount(code)
        big = cnt >= MIN_N
        preds = {'ESM-MSR': ('combined (latent)', [fr.preds['comb'][i, m] for i in range(fr.preds['comb'].shape[0])],
                             [fr.preds['comb~'][i, m] for i in range(fr.preds['comb~'].shape[0])])}
        for mn, (tpl, col, seeds, readout) in BASELINES.items():
            P = []
            for s in seeds:
                c = chunks(tpl, s, cfg['src'], col)
                P.append(None if c is None else c.reindex(d['id'].values).to_numpy(float))
            if not any(p is None for p in P):
                preds[mn] = (readout, P, P)
        if True:
            e3 = esm3_wt(d['id'].values)
            if e3 is not None and np.isfinite(e3).any():
                preds['ESM3-small-open'] = ('WT head (sigma 0)', [e3], [e3])
        # per-scaffold mean rho over the scaffolds with >= MIN_N variants; scaffolds drawn once for the set
        n_big = int(big.sum())
        draws = rng.integers(0, n_big, (a.B, n_big))
        per = {mn: pd.concat([scaffold_rho(code, p, y) for p in P], axis=1).mean(axis=1, skipna=False).to_numpy(float)[big] for mn, (_, P, _) in preds.items()}

        def boot_mean(v, ok):
            out = [np.mean(v[dr[ok[dr]]]) if ok[dr].any() else np.nan for dr in draws]
            return np.nanpercentile(out, [2.5, 97.5])
        for mn, (readout, _, _) in preds.items():
            v = per[mn]
            ok = np.isfinite(v)
            if not ok.any():
                continue
            lo, hi = boot_mean(v, ok)
            rows_s.append(dict(dataset=name, model=mn, readout=readout, rho=float(v[ok].mean()), lo=lo, hi=hi, n_scaffolds=int(ok.sum()), n_set=n_big, coverage=float(ok.mean())))
            if ok.mean() < 0.999 and mn != 'ESM-MSR':
                b = per['ESM-MSR'].copy(); b[~ok] = np.nan
                lo2, hi2 = boot_mean(b, np.isfinite(b))
                rows_s.append(dict(dataset=name, model=f'ESM-MSR, same scaffolds as {mn}', readout='combined (latent)', rho=float(np.nanmean(b)), lo=lo2, hi=hi2, n_scaffolds=int(ok.sum()), n_set=n_big, coverage=float(ok.mean())))
        # pooled rho over every measured variant (scaffold bootstrap; the draws re-rank every model)
        order = np.argsort(code, kind='stable')
        bounds = np.searchsorted(code[order], np.arange(len(uniq) + 1))
        pdraws = [np.concatenate([order[bounds[i]:bounds[i + 1]] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(a.B_pooled)]

        def pooled(P, mask=None):
            def f(idx):
                return float(np.mean([pooled_rho(p[idx], y[idx]) for p in P]))
            pt = f(np.flatnonzero(mask) if mask is not None else np.arange(len(y)))
            bs = [f(dr[mask[dr]] if mask is not None else dr) for dr in pdraws]
            return pt, *np.nanpercentile(bs, [2.5, 97.5])
        for mn, (readout, _, P) in preds.items():
            ok = np.all([np.isfinite(p) for p in P], axis=0)
            if ok.mean() < 0.02:
                continue
            msk = None if ok.all() else ok
            pt, lo, hi = pooled(P, msk)
            rows_p.append(dict(dataset=name, model=mn, readout=readout + (', Tobit-linked' if mn == 'ESM-MSR' else ''), rho=pt, lo=lo, hi=hi, n_variants=int(ok.sum()), n_set=len(y), coverage=float(ok.mean())))
            if mn != 'ESM-MSR' and msk is not None:
                pt2, lo2, hi2 = pooled(preds['ESM-MSR'][2], ok)
                rows_p.append(dict(dataset=name, model=f'ESM-MSR, same rows as {mn}', readout='combined (Tobit-linked)', rho=pt2, lo=lo2, hi=hi2, n_variants=int(ok.sum()), n_set=len(y), coverage=float(ok.mean())))
    for rows, f in ((rows_s, 'mgnify_models_per_scaffold.csv'), (rows_p, 'mgnify_models_pooled.csv')):
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f, index=False, float_format='%.4f')
        with pd.option_context('display.width', 220, 'display.max_columns', 20):
            print(f'-- {f}\n' + res.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
