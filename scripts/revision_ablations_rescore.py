#!/usr/bin/env python
"""Rescore a run's validation statistics offline from its validation dumps: the link correction and the pooled correlations, per epoch and per masking regime.

Two jobs, one pass over the dumps (training_logs/<run>/0/val_dump_{zs,e0,e1,...}.npz next to metrics.csv; never needs hparams.yaml, which holds the Comet key):

1. LINK CORRECTION. Validation reads the predictions through the Tobit expected-reading link for the observed-scale statistics (every RMSE and the observed-scale epistasis
   naive_rho_comb). That is right for a run trained with the Tobit likelihood and wrong for one trained with plain MSE (--reg_nll none), whose predictions already are on the
   observed scale; the external benchmark sets are never linked. Runs before 2026-11-20 applied it to the MSE runs. This recomputes the affected statistics with the link a
   run SHOULD have had (--link auto: none for --reg_nll none, the dump's own link otherwise; --link none / summary force it) and reports every statistic whose logged value
   differs (a Tobit run: none).
2. POOLED CORRELATIONS. val_[<regime>_]rho_combined_pooled and val_[<regime>_]rho_singles_wt_pooled (esm_msr.pooled_stats: one Spearman over every in-distribution item).

--check recomputes with the dump's OWN link (what the run logged) and compares every statistic it can with metrics.csv: the proof that this rescoring is the training code's
scoring (per-library RMSE and the _avg, the rho family, naive_rho_comb) before any corrected value is trusted.

Output: <out>/<experiment name>.json = {corrected: [{metric, step, epoch, old, new}], new: [{metric, step, epoch, value}], check: {...}} for scripts/revision_ablations_comet.py.

    PYTHONPATH=src python scripts/revision_ablations_rescore.py RUN_DIR [RUN_DIR ...] --out DIR [--link auto|none|summary] [--reg-nll none|tobit] [--check]

RUN_DIR holds metrics.csv and the val_dump_*.npz; its name is the Comet experiment name unless --name is given (one run).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from esm_msr import censoring, epi_metrics as E, pooled_stats, stats, val_stats

REGIMES = (('', ('wt', 'mt', 'comb')), ('unmasked_', ('wt_unmasked', 'mt_unmasked', 'comb_unmasked')))
TOL = 1e-6                       # a logged value this close to the desired one is not a correction


def dump_order(run_dir):
    """The dumps of a run in validation order: zs (the sanity pass, step 0), then e0, e1, ..."""
    ds = {re.fullmatch(r'val_dump_(zs|e\d+)\.npz', p.name).group(1): p for p in Path(run_dir).glob('val_dump_*.npz') if re.fullmatch(r'val_dump_(zs|e\d+)\.npz', p.name)}
    keys = sorted(ds, key=lambda k: -1 if k == 'zs' else int(k[1:]))
    return [(k, ds[k]) for k in keys]


def validation_rows(run_dir):
    """metrics.csv rows that hold a validation, in step order: (step, epoch, {metric: value})."""
    m = pd.read_csv(Path(run_dir) / 'metrics.csv', low_memory=False)
    v = m.dropna(subset=['val_rho_combined_avg']).sort_values('step').drop_duplicates('step', keep='last')
    return [(int(r['step']), int(r['epoch']), {k: float(r[k]) for k in v.columns if k.startswith('val_') and pd.notna(r[k])}) for _, r in v.iterrows()]


def dump_link(z):
    ls = z['link_summary'] if 'link_summary' in z.files else None
    s = json.loads(str(ls.item() if ls.ndim == 0 else ls)) if ls is not None else {'kind': 'none'}
    return val_stats.tobit_link(s['sigma_y'], s['mu_f'], s['sigma_f']) if s.get('kind') == 'tobit' else None


def library_metrics(z, fields, link):
    """{logged name without the regime prefix: value}: the per-library suite stats.compute_metrics gives, as validation logs it, with the observed scale read through `link`."""
    wt, mt, comb = (np.asarray(z[f], dtype=np.float64) for f in fields)
    gt, dgw = np.asarray(z['ground_truth'], dtype=np.float64), np.asarray(z['dG_wt'], dtype=np.float64)
    st = np.where(np.asarray(z['subset_type']) == 'native_cond', 'single', np.asarray(z['subset_type']))
    canon = np.asarray(z['cens_src']) != censoring.SRC_RANGE
    ok = np.isfinite(dgw)
    zz = np.where(ok, dgw, 0.0)
    obs = None if link is None else {'wt': np.where(ok, link(zz + wt) - zz, wt), 'comb': np.where(ok, link(zz + comb) - zz, comb)}
    out, per = {}, {m: [] for m in val_stats.AVG_LOGGED}
    loaders = np.asarray(z['loader']).astype(str)
    for lib in sorted(set(loaders)):
        c = (loaders == lib) & canon
        met = stats.compute_metrics(wt[c], mt[c], comb[c], gt[c], st[c], obs=None if obs is None else {k: v[c] for k, v in obs.items()})
        for m in val_stats.PER_LIBRARY_LOGGED:
            if np.isfinite(met.get(m, np.nan)):
                out[f'{m}/{lib}'] = met[m]
        for m in val_stats.AVG_LOGGED:
            if np.isfinite(met.get(m, np.nan)):
                per[m].append(met[m])
    for m, vals in per.items():
        if vals:
            out[f'{m}_avg'] = float(np.nanmean(vals))
    return out


def epi_naive(z, fields, link):
    wt, mt, comb = (np.asarray(z[f], dtype=float) for f in fields)
    keys = [tuple(tuple(m) for m in json.loads(s)) for s in z['mut_key']]
    items = {'mut_key': keys, 'subset_type': z['subset_type'], 'cens': np.asarray(z['cens']).astype(int), 'ddG': z['ground_truth'], 'dddG': z['dddG'], 'dG_wt': z['dG_wt'],
             'wt': wt, 'mt': mt, 'comb': comb, 'wt_rev': None, 'cens_src': z['cens_src']}
    canon = np.asarray(items['cens_src']) != censoring.SRC_RANGE                  # the canonical population, as validation scores it (the dump holds every mutant)
    scored = {k: (None if v is None else ([x for x, c in zip(v, canon) if c] if isinstance(v, list) else np.asarray(v)[canon])) for k, v in items.items()}
    return E.compute(E.Table(scored, link)).get('epi_naive_rho_comb', np.nan)


class _Dump(dict):
    """A dump as a dict with the fields a rescoring needs, filled in for the older flat format (the released-code baseline): subset_type from n_mut, no range-tagged rows, no dG_wt."""
    @property
    def files(self):
        return list(self)


def open_dump(path):
    z = np.load(path, allow_pickle=True)
    d = _Dump({k: z[k] for k in z.files})
    n = len(d['wt'])
    if 'ground_truth' not in d and 'ground_truths' in d:
        d['ground_truth'] = d['ground_truths']
    if 'subset_type' not in d:
        nm = np.asarray(d['n_mut'])
        d['subset_type'] = np.where(nm == 1, 'single', np.where(nm == 2, 'double', 'other'))
    d.setdefault('cens_src', np.zeros(n, dtype=int))
    d.setdefault('cens', np.zeros(n, dtype=int))
    d.setdefault('dG_wt', np.full(n, np.nan))
    return d


def rescore_dump(path, link_mode, reg_nll=None):
    """{logged metric name: value} of one dump, plus the link the values were computed with ('dump' = what the run logged)."""
    z = open_dump(path)
    own = dump_link(z)
    if link_mode == 'summary' or (link_mode == 'auto' and reg_nll != 'none'):
        link = own
    else:
        link = None
    out = {}
    for prefix, fields in REGIMES:
        if fields[0] not in z.files:
            continue
        for k, v in library_metrics(z, fields, link).items():
            out[f'val_{prefix}{k}'] = v
        nv = epi_naive(z, fields, link)
        if np.isfinite(nv):
            out[f'val_{prefix}epi_naive_rho_comb'] = float(nv)
        pooled = pooled_stats.pooled_rho(z[fields[0]], z[fields[2]], z['ground_truth'], z['subset_type'], z['cens_src'])
        for k, v in pooled.items():
            if np.isfinite(v):
                out[f'val_{prefix}{k}'] = float(v)
    return out


def affected(name):
    """The statistics the link touches: every observed-scale RMSE and the observed-scale epistasis naive_rho_comb (never an external set's: those are not in a dump)."""
    return 'rmse' in name or name.endswith('epi_naive_rho_comb')


def process(run_dir, name, link_mode, reg_nll, check):
    run_dir = Path(run_dir)
    dumps, rows = dump_order(run_dir), validation_rows(run_dir)
    if len(dumps) != len(rows):
        sys.exit(f'{name}: {len(dumps)} dumps but {len(rows)} validations in metrics.csv: cannot pair them')
    corrected, new, worst, checked = [], [], {}, 0
    for (tag, path), (step, epoch, logged) in zip(dumps, rows):
        want = rescore_dump(path, link_mode, reg_nll)
        own = rescore_dump(path, 'summary', reg_nll) if (check or link_mode != 'summary') else want
        # alignment + reproduction: what the run's own link gives must be what it logged
        for k, v in own.items():
            if k in logged and not any(x in k for x in pooled_stats.NAMES):
                d = abs(v - logged[k])
                fam = re.sub(r'/.*', '', k)
                worst[fam] = max(worst.get(fam, 0.0), d)
                checked += 1
        for k, v in want.items():
            if any(x in k for x in pooled_stats.NAMES):
                new.append({'metric': k, 'step': step, 'epoch': epoch, 'value': float(v), 'dump': tag})
            elif k in logged and affected(k) and abs(v - logged[k]) > TOL:
                corrected.append({'metric': k, 'step': step, 'epoch': epoch, 'old': logged[k], 'new': float(v), 'dump': tag})
    ok = max(worst.values() or [0.0])
    res = {'name': name, 'run_dir': str(run_dir), 'link_mode': link_mode, 'reg_nll': reg_nll, 'n_validations': len(rows), 'checked_values': checked,
           'check_max_abs_diff_by_family': {k: float(f'{v:.3g}') for k, v in sorted(worst.items())}, 'check_ok': bool(ok < 2e-4), 'corrected': corrected, 'new': new}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--link', choices=['auto', 'none', 'summary'], default='auto')
    ap.add_argument('--reg-nll', choices=['none', 'tobit'], default=None, help="what the run trained with (needed by --link auto: none = plain MSE, no link); hparams.yaml is never read")
    ap.add_argument('--name', default=None, help='Comet experiment name (a single run); default: the run directory name')
    ap.add_argument('--check', action='store_true', help='only report how well the dump-own link reproduces metrics.csv')
    a = ap.parse_args()
    if a.name and len(a.runs) > 1:
        sys.exit('--name needs a single run')
    if a.link == 'auto' and a.reg_nll is None:
        sys.exit('--link auto needs --reg-nll (none|tobit) for each batch of runs of one kind')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for r in a.runs:
        name = a.name or Path(r).name
        res = process(r, name, a.link, a.reg_nll, a.check)
        (Path(a.out) / f'{name}.json').write_text(json.dumps(res, indent=1))
        c = {}
        for x in res['corrected']:
            c.setdefault(re.sub(r'/.*', '', x['metric']), []).append(abs(x['new'] - x['old']))
        print(f"{name}: {res['n_validations']} validations, {res['checked_values']} logged values re-derived with the run's own link, max |diff| by family {res['check_max_abs_diff_by_family']} "
              f"-> reproduction {'OK' if res['check_ok'] else 'FAILED'}; {len(res['corrected'])} corrected points in {len(c)} families "
              f"{ {k: f'max {max(v):.3f}' for k, v in sorted(c.items())} }; {len(res['new'])} new pooled points", flush=True)


if __name__ == '__main__':
    main()
