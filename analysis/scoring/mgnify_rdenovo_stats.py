"""Triplicate stats of ESM-MSR (FINAL_p0, seeds 1-3) on the 1,000 Rocklin de novo designs of data/mgnify_v2 (source == rocklin_denovo).

The designs' variants are sequential edits of the design scaffold (n_mut 1 / 2 / 3), so the mutation-count subsets are 10,873 / 45,889 / 153,056 rows.
Measured epistasis (a double vs its two assayed singles) exists for only a few hundred rows: singles are sparse (~11 per scaffold).  Stats:
  pooled      Spearman of ddG_ML vs each head, by subset (all / n=1 / n=2 / n>=3 / multi) and regime (in-range 1-5 / all rows); latent and Tobit-linked
              (post hoc, as mgnify_link.py), plus the additive composition of the WT singles (wt_lora_pred_additive) for the multi subsets
  within      mean over scaffolds (>= 10 rows in the subset) of the within-scaffold Spearman, same cells
  epistasis   measured: doubles with both singles assayed and uncensored, dddG_ML vs epi_msr = comb(AB) - WT_A - WT_B (the tables' readout) and the
                        reference readouts combined_dddg_pred / mt_lora_dddg_pred: Spearman, Pearson, sign agreement, OLS slope; (n is small)
              model-referenced: every multi-mutant in range, ddG_ML - wt_lora_pred_additive vs epi_msr (the model's predicted non-additivity against
                        the measured deviation from the model's own additive composition; includes the single-mutation prediction errors, not
                        a measured epistasis); pooled and within-scaffold
              within-scaffold ranking of multi-mutants: combined - additive composition
Every interval resamples whole scaffolds; the same draw is applied to the three seeds and the interval is for the seed mean.  Point values: seed mean,
with the across-seed SD alongside.

  python analysis/scoring/mgnify_rdenovo_stats.py final_mgnify/FINAL_p0 [--boot 100] [--out DIR]
"""
import argparse
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402
sys.path.insert(0, HERE)
from mgnify_labels import apply_v2_labels  # noqa: E402

SEEDS = (1, 2, 3)
LINK = tobit_link(0.28, -0.14, 0.5)


def sp(x, y):
    if len(x) < 5:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pe(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 5 else np.nan


def within(codes, x, y, min_n=10):
    """Mean over scaffolds with >= min_n rows of the within-scaffold Spearman."""
    order = np.argsort(codes, kind='stable')
    c = codes[order]
    cuts = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True])
    vals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= min_n:
            i = order[a:b]
            vals.append(sp(x[i], y[i]))
    vals = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def load(root):
    ds = []
    for s in SEEDS:
        d = pd.read_csv(os.path.join(root, f's{s}', 'rdenovo', 'predictions.csv.gz'), low_memory=False)
        d = apply_v2_labels(d)
        ds.append(d.sort_values('uid', ignore_index=True))
    assert all((ds[0]['uid'].to_numpy() == x['uid'].to_numpy()).all() for x in ds)
    return ds


def prep(ds):
    """Per-seed derived columns and the shared flags on the seed-1 frame."""
    for d in ds:
        dg = d['dG_wt'].astype(float)
        for k in ('wt_lora_pred', 'mt_lora_pred', 'combined_pred'):
            d[k + '_linked'] = LINK(dg + d[k]) - dg
        d['epi_msr'] = d['combined_pred'] - d['wt_lora_pred_additive']
        d['resid_add'] = d['ddG_ML'] - d['wt_lora_pred_additive']
    b = ds[0]
    sing = b[b['n_mut'] == 1].set_index(['code_wt', 'mut_type'])['cens']
    dbl = b[b['n_mut'] == 2]
    ok = np.array([all(sing.get((r.code_wt, x), np.nan) == 0 for x in r.mut_type.split(':')) for r in dbl.itertuples()])
    b['singles_uncens'] = False
    b.loc[dbl.index[ok], 'singles_uncens'] = True
    for d in ds[1:]:
        d['singles_uncens'] = b['singles_uncens']
    for d in ds:
        d['in_range'] = d['in_range_1_5'].astype(bool)
    return ds


def draws(codes, B, rng):
    u, inv = np.unique(codes, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        yield np.concatenate([order[bounds[i]:bounds[i + 1]] for i in pick])


def cell(ds, mask, fn, B, rng_seed):
    """fn(frame) -> float (or (float, n)); returns seed mean, seed SD, lo, hi (bootstrap of the seed mean over scaffolds), n."""
    idx = np.flatnonzero(mask)
    sub = [d.iloc[idx].reset_index(drop=True) for d in ds]
    if len(idx) < 10:
        return None
    def one(frames):
        r = [fn(f) for f in frames]
        return np.array([x[0] if isinstance(x, tuple) else x for x in r], float), (r[0][1] if isinstance(r[0], tuple) else len(frames[0]))
    pt, n = one(sub)
    bs = []
    rng = np.random.default_rng(rng_seed)
    for j in draws(sub[0]['code_wt'].to_numpy(), B, rng):
        bs.append(np.nanmean(one([f.iloc[j] for f in sub])[0]))
    return dict(value=float(np.nanmean(pt)), sd_seeds=float(np.nanstd(pt)), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                n=int(n), n_scaffolds=int(sub[0]['code_wt'].nunique()))


DS = None
TASKS = []


def _run(i):
    meta, mask, fn, seed = TASKS[i]
    return cell(DS, mask, fn, ARGS.boot, seed)


def run_all(meta_mask_fn_seed, jobs):
    """Evaluate the queued cells in forked workers (DS / TASKS are inherited; nothing is pickled but the index)."""
    global TASKS
    TASKS = meta_mask_fn_seed
    with get_context('fork').Pool(jobs) as p:
        return p.map(_run, range(len(TASKS)), chunksize=1)


def main():
    global DS, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='final_mgnify/FINAL_p0 (holds s1/rdenovo ...)')
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--out', default=None)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    ARGS = a
    out = a.out or os.path.join(a.root, 'rdenovo_stats')
    os.makedirs(out, exist_ok=True)
    ds = prep(load(a.root))
    DS = ds
    b = ds[0]
    ok = b['ddG_ML'].notna().to_numpy()
    nm = b['n_mut'].to_numpy()
    subsets = {'all': nm >= 1, 'n=1': nm == 1, 'n=2': nm == 2, 'n>=3': nm >= 3, 'multi(n>=2)': nm >= 2}
    regimes = {'in_range': b['in_range'].to_numpy(), 'all_rows': np.ones(len(b), bool)}
    heads = {'wt head': 'wt_lora_pred', 'mt head': 'mt_lora_pred', 'combined': 'combined_pred'}
    rows, q = [], []
    for rn, rm in regimes.items():
        for sn, sm in subsets.items():
            m = ok & rm & sm
            cols = [(h, 'latent', c) for h, c in heads.items()] + [(h, 'linked', c + '_linked') for h, c in heads.items()]
            if sn in ('n=2', 'n>=3', 'multi(n>=2)'):
                cols.append(('additive composition (WT singles)', 'latent', 'wt_lora_pred_additive'))
            for h, form, col in cols:
                for kind, fn in (('pooled', lambda f, col=col: sp(f[col].to_numpy(), f['ddG_ML'].to_numpy())),
                                 ('within-scaffold', lambda f, col=col: within(f['code_wt'].to_numpy(), f[col].to_numpy(), f['ddG_ML'].to_numpy()))):
                    q.append((dict(kind=kind, regime=rn, subset=sn, head=h, form=form), m, fn, 0))
    for (meta, *_), r in zip(q, run_all(q, a.jobs)):
        if r:
            rows.append(dict(meta, **r))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'stats.csv'), index=False, float_format='%.4f')

    er, eq = [], []
    # measured epistasis: doubles with both singles assayed and uncensored
    m = (nm == 2) & b['has_all_singles'].fillna(False).astype(bool).to_numpy() & b['dddG_ML'].notna().to_numpy() & b['singles_uncens'].to_numpy()
    for head in ('epi_msr', 'combined_dddg_pred', 'mt_lora_dddg_pred'):
        for stat, fn in (('spearman', lambda f, h=head: sp(f[h].to_numpy(), f['dddG_ML'].to_numpy())),
                         ('pearson', lambda f, h=head: pe(f[h].to_numpy(), f['dddG_ML'].to_numpy())),
                         ('sign agreement', lambda f, h=head: float((np.sign(f[h]) == np.sign(f['dddG_ML'])).mean())),
                         ('OLS slope (measured on predicted)', lambda f, h=head: float(np.polyfit(f[h].to_numpy(), f['dddG_ML'].to_numpy(), 1)[0]))):
            eq.append((dict(test='measured dddG (doubles, singles assayed)', readout=head, stat=stat), m & ds[0][head].notna().to_numpy(), fn, 1))
    # model-referenced non-additivity on multi-mutants
    for sn, sm in (('multi(n>=2)', nm >= 2), ('n=2', nm == 2), ('n>=3', nm >= 3)):
        mm = sm & b['in_range'].to_numpy() & ok
        for kind, fn in (('pooled', lambda f: sp(f['epi_msr'].to_numpy(), f['resid_add'].to_numpy())),
                         ('within-scaffold', lambda f: within(f['code_wt'].to_numpy(), f['epi_msr'].to_numpy(), f['resid_add'].to_numpy()))):
            eq.append((dict(test=f'model-referenced non-additivity, {sn}, in range', readout='epi_msr vs ddG - additive', stat=f'{kind} spearman'), mm, fn, 2))
    for (meta, *_), r in zip(eq, run_all(eq, a.jobs)):
        if r:
            er.append(dict(meta, **r))
    pd.DataFrame(er).to_csv(os.path.join(out, 'epistasis.csv'), index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(pd.DataFrame(er).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
