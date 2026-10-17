"""
The validation pass probe (esm_msr.pass_probe) scored offline from a dump written on a --val_pass_probe epoch, with every trapezoid -
each start view (wt.u_wt, wt.pair, mt.u_wt, mt.pair) paired with each end view (wt.u_mt, wt.i_mt, mt.u_mt, mt.i_mt), 16 in all - instead of
the four logged during training. No GPU: the views are in the dump as '<library>|probe_<view>'.

  PYTHONPATH=src python scripts/pass_probe_from_dump.py training_logs/<run>/0/val_dump_zs.npz [more dumps] [--csv out.csv] [--logged-only]

One table per dump, a row per head, the columns of pass_probe.METRICS. Read rho_* for prediction, cell_rank / cell_flipacc for the
interaction, and nonadd_* for where each view puts its own non-additivity.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epi_from_dump import load_items                      # noqa: E402
from esm_msr import pass_probe                            # noqa: E402


def load_views(path):
    """{view: scores} in load_items' item order (libraries sorted), and each item's library; None when the dump has no probe."""
    z = np.load(path, allow_pickle=True)
    libs = sorted({k.split('|')[0] for k in z.files if '|' in k})
    if not all(f'{lib}|probe_{v}' in z.files for lib in libs for v in pass_probe.VIEWS):
        return None, None
    views = {v: np.concatenate([np.asarray(z[f'{lib}|probe_{v}'], float) for lib in libs]) for v in pass_probe.VIEWS}
    loader = np.concatenate([[lib] * len(z[f'{lib}|subset_type']) for lib in libs])
    return views, loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dumps', nargs='+')
    p.add_argument('--csv')
    p.add_argument('--logged-only', action='store_true', help='only the four trapezoids training logs')
    a = p.parse_args()
    frames = []
    for path in a.dumps:
        views, loader = load_views(path)
        if views is None:
            print(f"{path}: no probe views (not a --val_pass_probe epoch)", file=sys.stderr)
            continue
        items, link = load_items(path)
        items = {k: items[k] for k in ('mut_key', 'subset_type', 'cens', 'ddG', 'dddG', 'dG_wt')}
        tz = pass_probe.TRAPEZOIDS if a.logged_only else pass_probe.ALL_TRAPEZOIDS
        df = pd.DataFrame(pass_probe.score(items, views, loader, link, trapezoids=tz)).T[list(pass_probe.METRICS)]
        print(f"\n=== {path} ===")
        with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 100):
            print(df.astype(float).round(3).to_string())
        df.insert(0, 'dump', path)
        frames.append(df)
    if a.csv and frames:
        pd.concat(frames).to_csv(a.csv)
        print(f"\nwrote {a.csv}")


if __name__ == '__main__':
    main()
