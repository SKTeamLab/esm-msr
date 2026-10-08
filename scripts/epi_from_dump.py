"""
The epistasis hierarchy recomputed offline from one or more val_dump_<tag>.npz files (training_logs/<run>/0/), with the current definitions in
esm_msr.epi_hierarchy. Reads both dump formats: the baseline's flat arrays (wt, mt, comb, wt_rev, dddG, n_mut, ground_truth, loader, mut_key) and the
devel run's per-library '<library>|<field>' arrays. Heads as in training: wt_add, wt_ctx (needs the wt_rev leg), comb. A devel dump holds
observed-scale values only for wt / mt / comb (and wt_ctx when it was logged with the link); otherwise the latent is taken as the observed scale.

Usage: PYTHONPATH=src python scripts/epi_from_dump.py DUMP.npz [DUMP.npz ...] [--levels a,b,c]
"""
import argparse, json, sys
import numpy as np

from esm_msr import epi_hierarchy as H, routing, stats

HEADS = ('wt_add', 'wt_ctx', 'comb')


def _load_baseline(z):
    keys = [tuple(tuple(m) for m in json.loads(s)) for s in z['mut_key']]
    n = np.asarray(z['n_mut'])
    st = np.where(n == 1, 'single', np.where(n == 2, 'double', 'other'))
    cens = np.zeros(len(n), dtype=int)
    wr = z['wt_rev'] if 'wt_rev' in z.files else None
    vals = {'wt_add': np.asarray(z['wt'], float), 'comb': np.asarray(z['comb'], float),
            'wt_ctx': None if wr is None else 0.5 * (np.asarray(z['wt'], float) + np.asarray(wr, float))}
    return keys, st, cens, np.asarray(z['dddG'], float), np.asarray(z['ground_truth'], float), vals


def _load_devel(z):
    libs = sorted({k.split('|')[0] for k in z.files if '|' in k})
    keys, st, cens, dd, gt = [], [], [], [], []
    vals = {h: [] for h in HEADS}
    for lib in libs:
        g = lambda f: z[f'{lib}|{f}'] if f'{lib}|{f}' in z.files else None
        mk = [tuple(tuple(m) for m in json.loads(s)) for s in g('mut_key')]
        keys += [tuple((lib,) + m for m in k) for k in mk]          # numbered per library: tag them as in training
        st += list(g('subset_type')); cens += list(g('cens')); dd += list(g('dddG')); gt += list(g('ground_truths'))
        obs = g('comb_obs') is not None
        wt, comb, wr = (g('wt_obs'), g('comb_obs'), g('wt_ctx_obs')) if obs else (g('wt_scores'), g('comb_scores'), None)
        if not obs and g('wt_rev_scores') is not None:
            wr = 0.5 * (g('wt_scores') + g('wt_rev_scores'))
        nan = np.full(len(mk), np.nan)
        vals['wt_add'].append(wt); vals['comb'].append(comb); vals['wt_ctx'].append(nan if wr is None else wr)
    vals = {h: np.concatenate([np.asarray(v, float) for v in vs]) for h, vs in vals.items()}
    return keys, np.array([routing.canonical_subset(s) for s in st]), np.array(cens), np.asarray(dd, float), np.asarray(gt, float), vals


def levels_for(path):
    z = np.load(path, allow_pickle=True)
    keys, st, cens, dddG, gt, vals = _load_baseline(z) if 'n_mut' in z.files else _load_devel(z)
    is_double = (st == 'double') & np.isfinite(dddG) & (cens == 0)
    additive = vals['wt_add']
    single = {tuple(k[0]): float(gt[i]) for i, k in enumerate(keys)
              if st[i] in routing.WT_HEAD_SUBSETS and len(k) == 1 and cens[i] == 0 and np.isfinite(gt[i])}
    out = {}
    for head in HEADS:
        v = vals[head]
        if v is None or not np.isfinite(v[is_double]).any():
            continue
        out[head] = H.compute(stats.epi_full_scores(v, st, keys), dddG, keys, is_double, additive=additive, single_ddG=single)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--levels', default=','.join(H.LEVELS))
    a = ap.parse_args()
    lv = a.levels.split(',')
    for path in a.dumps:
        res = levels_for(path)
        print(f'\n{path}   (n_doubles {next(iter(res.values()))["n_doubles"]}, pairs {next(iter(res.values()))["n_pairs"]})')
        print(f'  {"level":38s}' + ''.join(f'{h:>9s}' for h in res))
        for l in lv:
            print(f'  {l:38s}' + ''.join(f'{res[h][l]:9.3f}' for h in res))
