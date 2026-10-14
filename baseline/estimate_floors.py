"""
Per-library floor offsets for --link_floor_file.

The assay infers dG by comparing a variant's measured K50 against a K50 predicted for its UNFOLDED sequence from composition. A variant that
is unfolded anyway has K50 = its true unfolded K50, so its inferred dG is the error of that prediction for that sequence: a constant of the
library (one wild type plus point mutants is one sequence, for composition purposes) and of the protease. That constant is the library's
PRACTICAL FLOOR, and it is what the link's floor knee should be.

Estimator: the median measured dG of the library's doubles whose additive expectation x = dG_wt + ddG_A + ddG_B is below ``--x_max`` (default
-1.5), i.e. the variants expected far below any floor, which therefore pin at it. Requires ``--min_n`` such doubles (default 20); libraries
with fewer are shrunk toward the global median with weight ``--shrink`` (default 5 pseudo-counts), which leaves a well-measured library
essentially untouched (standard error about 0.05 against a spread of 0.31) and keeps thin ones sane. dG_wt is not a usable fallback: it
correlates 0.16 with the floor.

The file stores the OFFSET from the global median, so the link's own learned ``lo`` keeps its meaning (the floor of a typical library) and the
offsets are centred at 0.

Doubles reported '<-1' have no numeric dG and are excluded, so this is the floor of the population the regression loss actually sees; the
censored ones are handled by the hinge. The two are consistent, not alternatives: a library's floor and its rate of '<-1' reports are the same
quantity seen twice (correlation -0.61; libraries with floor +0.29 censor 0.5% of their deep doubles, those with floor -0.30 censor 69%).

Usage: python baseline/estimate_floors.py OUT.json [--x_max -1.5] [--min_n 20] [--shrink 5] [--split_file PATH --splits train]
  --split_file/--splits restricts ESTIMATION to libraries of those splits (others get offset 0): the stricter control, for showing that
  conditioning on the floor during training helps even when it is withheld at validation.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

RAW = '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'


def code_of(wt_name: pd.Series) -> pd.Series:
    """The library key, exactly as preprocess_megascale._wt_dG_by_code builds it."""
    return (wt_name.str.replace('.pdb_', '_', regex=False).str.replace('.pdb', '', regex=False).str.replace('|', '_', regex=False))


def doubles_table(raw: str) -> pd.DataFrame:
    """Every double with both singles measured, with the library's dG_wt, the double's dG and its additive expectation."""
    df = pd.read_csv(raw, usecols=['WT_name', 'mut_type', 'dG_ML'], low_memory=False)
    df = df[~df.mut_type.str.contains('ins|del', na=False)].copy()
    df['code'] = code_of(df['WT_name'])
    df['dG'] = pd.to_numeric(df['dG_ML'], errors='coerce')
    g = df.groupby(['code', 'mut_type'], as_index=False).agg(dG=('dG', 'mean'))
    wt = g[g.mut_type == 'wt'].set_index('code')['dG']
    rows = []
    for code, G in g[g.mut_type != 'wt'].groupby('code'):
        c = wt.get(code, np.nan)
        if not np.isfinite(c):
            continue
        sing = {m: d for m, d in zip(G.mut_type, G.dG) if ':' not in m and np.isfinite(d)}
        for m, d in zip(G.mut_type, G.dG):
            if m.count(':') != 1:
                continue
            a, b = m.split(':')
            if a not in sing or b not in sing or not all(re.match(r'^[A-Z]\d+[A-Z]$', s) for s in (a, b)):
                continue
            rows.append((code, c, d, sing[a] + sing[b] - c))          # x = c + (dA - c) + (dB - c)
    return pd.DataFrame(rows, columns=['code', 'dG_wt', 'dG', 'x'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--raw', default=RAW)
    p.add_argument('--x_max', type=float, default=-1.5, help='a double counts as pinned when its additive expectation is below this')
    p.add_argument('--min_n', type=int, default=20, help='pinned doubles needed before a library gets its own estimate')
    p.add_argument('--shrink', type=float, default=5.0, help='pseudo-counts of the global median mixed into every estimate')
    p.add_argument('--split_file', default=None)
    p.add_argument('--splits', nargs='*', default=['train'])
    a = p.parse_args()

    D = doubles_table(a.raw)
    deep = D[(D.x < a.x_max) & np.isfinite(D.dG)]
    est = deep.groupby('code').dG.agg(['median', 'size']).rename(columns={'median': 'floor', 'size': 'n'})
    keep = est[est.n >= a.min_n]
    globl = float(keep.floor.median())

    allowed = None
    if a.split_file:
        import pickle
        sp = pickle.load(open(a.split_file, 'rb'))
        allowed = {k.replace('.pdb', '').replace('|', '_') for s in a.splits for k in sp[s]}

    out, used = {}, 0
    for code, r in est.iterrows():
        if allowed is not None and code.split('_')[0] not in {c.split('_')[0] for c in allowed} and code not in allowed:
            continue
        n = float(r['n'])
        floor = (n * float(r['floor']) + a.shrink * globl) / (n + a.shrink)     # shrunk toward the global median
        out[code] = round(floor - globl, 4)
        used += r['n'] >= a.min_n
    meta = {'_global_floor': round(globl, 4), '_x_max': a.x_max, '_min_n': a.min_n, '_shrink': a.shrink,
            '_n_libraries': len(out), '_n_own_estimate': int(used), '_splits': (a.splits if allowed is not None else 'all'),
            '_doc': 'offset from _global_floor, added to the link floor knee; see baseline/estimate_floors.py'}
    json.dump({**meta, **out}, open(a.out, 'w'), indent=0, sort_keys=True)
    o = np.array(list(out.values()))
    print(f'{len(D)} doubles with both singles; {len(deep)} pinned (x < {a.x_max}) in {est.shape[0]} libraries')
    print(f'global floor {globl:+.3f} kcal/mol from {len(keep)} libraries with >= {a.min_n} pinned doubles')
    print(f'wrote {len(out)} offsets to {a.out}: {used} from their own estimate, {len(out) - used} shrunk')
    print(f'offsets: SD {o.std():.3f}, 10th-90th pct {np.quantile(o, 0.1):+.3f} to {np.quantile(o, 0.9):+.3f}, '
          f'min {o.min():+.3f}, max {o.max():+.3f}')


if __name__ == '__main__':
    main()
