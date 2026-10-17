"""
Score every ensemble of the four forward passes, offline, from a run's val_dump.

The passes are a 2 x 2 of {which adapter} x {which sequence it was given}:

                     given the WILD-TYPE sequence      given the MUTATED sequence
    WT adapter       wt        (the additive head)     wt_rev    (reverse mutation, negated)
    MT adapter       mt_fwd    (a regime it never      mt        (the reported MT head)
                               trains on: a probe)

``forward_cycle`` already computes all four under --val_cycle_passes and the dump carries them, so this needs no
GPU and no rerun. Only ``wt`` + ``mt`` (as ``comb``) and ``wt_rev`` (as ``ctx``) are scored during training;
``mt_fwd`` is dumped and otherwise unused, and no ensemble beyond those two is ever looked at.

Every metric here is RANK based - per-library Spearman, and the epistasis cell rank - so a monotone link cannot
change any of them and none of this depends on reconstructing the run's link. That is also why RMSE is absent:
it would need the link, and on the latent scale it is meaningless.

  PYTHONPATH=src python analysis_scripts/ensemble_rescore.py DUMP [DUMP ...] [--csv out.csv] [--no-epi]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from esm_msr import epi_metrics                                       # noqa: E402
from esm_msr.stats import safe_spearman                               # noqa: E402

LEGS = ('wt', 'wt_rev', 'mt_fwd', 'mt')


def ensembles(d):
    """Name -> latent score, for each leg and each averaging of them worth looking at."""
    wt, wt_rev, mt_fwd, mt = (d[k].astype(float) for k in LEGS)
    mean = lambda *a: np.mean(np.stack(a), axis=0)
    return {
        # the four legs on their own
        'wt (WT adapter, wt seq)': wt,
        'wt_rev (WT adapter, mut seq)': wt_rev,
        'mt_fwd (MT adapter, wt seq)': mt_fwd,
        'mt (MT adapter, mut seq)': mt,
        # the two the run already scores
        'comb = (wt + mt)/2  [reported]': mean(wt, mt),
        'ctx = (wt + wt_rev)/2  [WT sym]': mean(wt, wt_rev),
        # the mirror of ctx, which nothing currently computes
        'msym = (mt + mt_fwd)/2  [MT sym]': mean(mt, mt_fwd),
        # by which sequence the adapter was shown
        'fwd = (wt + mt_fwd)/2  [wt seq]': mean(wt, mt_fwd),
        'rev = (wt_rev + mt)/2  [mut seq]': mean(wt_rev, mt),
        # the off-diagonal: the two passes neither head is trained on
        'cross = (wt_rev + mt_fwd)/2': mean(wt_rev, mt_fwd),
        'quad = mean of all four': mean(wt, wt_rev, mt_fwd, mt),
    }


def per_library_rho(score, gt, loader, keep):
    """Mean over libraries of each library's Spearman, the same shape as the run's val_rho_*_avg metrics."""
    out = []
    for lib in dict.fromkeys(loader):
        m = keep & (loader == lib)
        if m.sum() < 3:
            continue
        r = safe_spearman(score[m], gt[m])
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else np.nan, len(out)


def score_dump(path, want_epi=True):
    d = np.load(path, allow_pickle=False)
    missing = [k for k in LEGS if k not in d.files]
    if missing:
        sys.exit(f"{path}: missing {missing} - was the run launched without --val_cycle_passes?")
    gt = d['ground_truth'].astype(float)
    loader = d['loader'].astype(str)
    st = d['subset_type'].astype(str)
    cens_src = d['cens_src'].astype(int) if 'cens_src' in d.files else np.zeros(len(gt), int)
    # The canonical population, as the run's own validation uses: everything the assay FITTED. SRC_RANGE rows are
    # held back because their label is their bound, a library constant, so any ranking against it is noise.
    canon = np.isfinite(gt) & (cens_src != 3)
    rows = {}
    for name, sc in ensembles(d).items():
        ok = canon & np.isfinite(sc)
        r_all, n_lib = per_library_rho(sc, gt, loader, ok)
        r_sgl, _ = per_library_rho(sc, gt, loader, ok & (st == 'single'))
        r_dbl, _ = per_library_rho(sc, gt, loader, ok & (st == 'double'))
        row = {'rho_avg': r_all, 'rho_single': r_sgl, 'rho_double': r_dbl,
               'rho_pooled': safe_spearman(sc[ok], gt[ok]), 'n_lib': n_lib}
        if want_epi:
            # Reuse the tested module unmodified by feeding each ensemble through its 'comb' slot.
            keys = [tuple(tuple(m) for m in json.loads(k)) for k in d['mut_key']]
            items = {'mut_key': keys, 'subset_type': list(st), 'ddG': gt, 'dddG': d['dddG'].astype(float),
                     'dG_wt': d['dG_wt'].astype(float), 'wt': d['wt'].astype(float), 'mt': sc, 'comb': sc,
                     'wt_rev': None, 'y_floor': d['y_floor'].astype(float) if 'y_floor' in d.files else None}
            res = epi_metrics.compute(epi_metrics.Table(items), {'cell_rank': ('comb',), 'cell_flipacc': ('comb',)})
            row['epi_cell_rank'] = res.get('epi_cell_rank_comb', np.nan)
            row['epi_cell_flipacc'] = res.get('epi_cell_flipacc_comb', np.nan)
        rows[name] = row
    return pd.DataFrame(rows).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dumps', nargs='+')
    p.add_argument('--csv')
    p.add_argument('--no-epi', action='store_true')
    a = p.parse_args()
    frames = []
    for path in a.dumps:
        df = score_dump(path, want_epi=not a.no_epi)
        df.insert(0, 'dump', os.path.basename(os.path.dirname(os.path.dirname(path))) + '/' + os.path.basename(path))
        frames.append(df)
        print(f"\n=== {path} ===")
        with pd.option_context('display.width', 200, 'display.max_columns', 30):
            print(df.drop(columns='dump').astype(float).round(4).to_string())
    if a.csv:
        pd.concat(frames).to_csv(a.csv)
        print(f"\nwrote {a.csv}")


if __name__ == '__main__':
    main()
