"""
The epistasis metrics (esm_msr.epi_metrics) recomputed offline from validation dumps (training_logs/<run>/0/val_dump_<tag>.npz), so any past run
can be scored with the current definitions.

Reads both dump formats: devel's per-library '<library>|<field>' arrays and the released-code baseline's flat arrays (wt, mt, comb, wt_rev, dddG,
n_mut, ground_truth, loader, mut_key). The link comes from the dump's 'link_summary' (none: the latent is the observed scale). dG_wt comes from the
dump ('<library>|dG_wt', written since this script's version) or, for older dumps, from the library caches (--cache: the singles' dG_meas - ddG).

  PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e5.npz [more dumps] [--cache DIR] [--boot 200] [--csv out.csv]

--boot N adds a percentile 90% interval from N resamples of the position pairs (each resampled copy of a pair keeps its own singles), the
sampling uncertainty of the validation set itself for a fixed model.
"""
import argparse, json, os, pickle, sys
import numpy as np

from esm_msr import epi_metrics as E
from esm_msr.link import numpy_link

_DGWT = {}


def cache_dg_wt(cache, lib):
    """A library's dG_wt from its cache (median of dG_meas - ddG over its uncensored singles); NaN when unknown."""
    if (cache, lib) not in _DGWT:
        val = np.nan
        for name in sorted(os.listdir(cache)) if cache and os.path.isdir(cache) else []:
            if name.startswith(f'{lib}_') and name.endswith('.pkl') and 'premask' not in name:
                items = pickle.load(open(os.path.join(cache, name), 'rb'))
                v = [it['dG_meas'] - it['ddG'] for it in items if it.get('subset_type') in ('single', 'native_cond') and it.get('cens', 0) == 0
                     and np.isfinite(it.get('dG_meas', np.nan)) and np.isfinite(it.get('ddG', np.nan))]
                val = float(np.median(v)) if v else np.nan
                break
        _DGWT[(cache, lib)] = val
    return _DGWT[(cache, lib)]


def load_items(path, cache=None):
    """(items for epi_metrics.Table, link or None) from one dump. Mutation keys are tagged with their library, as in training."""
    z = np.load(path, allow_pickle=True)
    if 'n_mut' in z.files:                                       # the baseline's flat format
        libs = z['loader'].astype(str)
        keys = [tuple(tuple(m) for m in json.loads(s)) for s in z['mut_key']]       # already (library, wt, pos, mt)
        n = np.asarray(z['n_mut'])
        items = {'mut_key': keys, 'subset_type': np.where(n == 1, 'single', np.where(n == 2, 'double', 'other')),
                 'cens': np.zeros(len(n), int), 'ddG': z['ground_truth'], 'dddG': z['dddG'],
                 'dG_wt': np.array([cache_dg_wt(cache, l) for l in libs]),
                 'wt': z['wt'], 'mt': z['mt'], 'comb': z['comb'], 'wt_rev': z['wt_rev'] if 'wt_rev' in z.files else None}
        return items, None
    libs = sorted({k.split('|')[0] for k in z.files if '|' in k})
    acc = {k: [] for k in ('mut_key', 'subset_type', 'cens', 'ddG', 'dddG', 'dG_wt', 'wt', 'mt', 'comb', 'wt_rev')}
    for lib in libs:
        g = lambda f: z[f'{lib}|{f}'] if f'{lib}|{f}' in z.files else None
        mk = [tuple(tuple(m) for m in json.loads(s)) for s in g('mut_key')]
        n = len(mk)
        acc['mut_key'] += [tuple((lib,) + m for m in k) for k in mk]
        acc['subset_type'] += list(g('subset_type'))
        acc['cens'].append(g('cens')); acc['ddG'].append(g('ground_truths')); acc['dddG'].append(g('dddG'))
        acc['dG_wt'].append(g('dG_wt') if g('dG_wt') is not None else np.full(n, cache_dg_wt(cache, lib)))
        acc['wt'].append(g('wt_scores')); acc['mt'].append(g('mt_scores')); acc['comb'].append(g('comb_scores'))
        acc['wt_rev'].append(g('wt_rev_scores') if g('wt_rev_scores') is not None else np.full(n, np.nan))
    items = {k: (v if k in ('mut_key', 'subset_type') else np.concatenate([np.asarray(a, float) for a in v])) for k, v in acc.items()}
    items['cens'] = items['cens'].astype(int)
    link = None
    if 'link_summary' in z.files:
        s = json.loads(str(z['link_summary']))
        link = numpy_link(s['lo'], s['hi'], s['tau_lo'], s['tau_hi'])
    return items, link


def _pairs_of(items):
    """For every double, its position-pair key; for every single, the set of pair keys it belongs to."""
    pk = {}
    for i, (k, s) in enumerate(zip(items['mut_key'], items['subset_type'])):
        if s == 'double' and k is not None and len(k) == 2:
            a, b = sorted([tuple(k[0])[:-1], tuple(k[1])[:-1]])
            pk[i] = (a, b)
    return pk


def bootstrap(items, link, n_boot, seed=0):
    """Percentile intervals from resampling the position pairs that form matrices (at least MIN_PAIR_CELLS doubles); each copy of a pair is
    relabelled so it forms its own matrix with its own singles. Doubles outside those matrices stay in every draw once."""
    rng = np.random.default_rng(seed)
    pk = _pairs_of(items)
    by_pair = {}
    for i, q in pk.items():
        by_pair.setdefault(q, []).append(i)
    pairs = sorted(p for p, v in by_pair.items() if len(v) >= E.MIN_PAIR_CELLS)
    in_mat = {i for p in pairs for i in by_pair[p]}
    single_idx = {}
    for i, (k, s) in enumerate(zip(items['mut_key'], items['subset_type'])):
        if s == 'single' and k is not None and len(k) == 1:
            single_idx.setdefault(tuple(k[0])[:-1], []).append(i)
    fixed = [i for i in range(len(items['mut_key'])) if i not in in_mat]        # every single once, and the doubles outside matrices
    arrays = [k for k in items if k not in ('mut_key', 'subset_type') and items[k] is not None]
    draws = []
    for _ in range(n_boot):
        idx = list(fixed)
        keys = [items['mut_key'][i] for i in fixed]
        st = [items['subset_type'][i] for i in fixed]
        for copy, p in enumerate(rng.choice(len(pairs), len(pairs), replace=True)):
            P, tag = pairs[p], f'#{copy}'
            for i in by_pair[P] + single_idx.get(P[0], []) + single_idx.get(P[1], []):
                idx.append(i)
                keys.append(tuple((m[0] + tag,) + tuple(m[1:]) for m in items['mut_key'][i]))
                st.append(items['subset_type'][i])
        idx = np.array(idx)
        sub = {k: np.asarray(items[k])[idx] for k in arrays}
        sub.update(mut_key=keys, subset_type=st)
        draws.append(E.compute(E.Table(sub, link)))
    names = sorted({k for d in draws for k in d})
    return {k: np.nanpercentile([d.get(k, np.nan) for d in draws], [5, 95]) for k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--cache', default='/home/sareeves/playground/esm-msr-devel/cache_v7')
    ap.add_argument('--boot', type=int, default=0)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    rows = []
    for path in a.dumps:
        items, link = load_items(path, a.cache)
        res = E.compute(E.Table(items, link))
        ci = bootstrap(items, link, a.boot) if a.boot else {}
        print(f'\n{path}  (link: {"yes" if link else "no"}; doubles {int(res["epi_n_doubles"])}, pairs {int(res["epi_n_pairs"])})')
        for name in E.logged_names():
            if name in res:
                extra = f'   [{ci[name][0]:.3f}, {ci[name][1]:.3f}]' if name in ci else ''
                print(f'  val_{name:28s} {res[name]:8.3f}{extra}')
        rows.append({'dump': path, **res, **{f'{k}_lo': v[0] for k, v in ci.items()}, **{f'{k}_hi': v[1] for k, v in ci.items()}})
    if a.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.csv, index=False)


if __name__ == '__main__':
    sys.exit(main())
