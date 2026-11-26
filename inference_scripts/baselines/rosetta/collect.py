"""Step 4: parse the cartesian_ddg outputs into the harness prediction file (login node, light):
  python collect.py --out rosetta_predictions_rocklin --dest results/rosetta_rocklin.csv
ddg_rosetta = mean over rounds of MUT total - mean over rounds of WT total (positive = destabilising); pred = -ddg_rosetta (higher = more
stable, like ddG_ML); pred_additive = -(sum of the component singles' ddg_rosetta) where every component single of the same scaffold was
computed (NaN otherwise).  Lists unfinished chunks in tasks/missing_chunks.txt (array indices to resubmit).
--singles DIR (repeatable) adds the singles of prepare_single_tasks.py's runs to the lookup behind pred_additive; their rows are not written to --dest."""
import argparse
import os
import re

import numpy as np
import pandas as pd

LINE = re.compile(r'^COMPLEX:\s+Round(\d+):\s+(\S+):\s+(-?[\d.]+)')


def parse_ddg(path):
    scores = {}
    for line in open(path):
        m = LINE.match(line)
        if m:
            scores.setdefault(m.group(2), []).append(float(m.group(3)))
    wt = scores.pop('WT', None)
    return {lab: np.mean(v) - np.mean(wt) for lab, v in scores.items()} if wt else {}


def load(out):
    """(tasks/mutations.csv of a run with its ddg_rosetta column, NaN where not computed); writes tasks/missing_chunks.txt."""
    muts = pd.read_csv(f'{out}/tasks/mutations.csv', keep_default_na=False)
    n_chunks = len(pd.read_csv(f'{out}/tasks/chunks.csv'))
    ddg, missing = {}, []
    for k in range(n_chunks):
        f = f'{out}/ddg/chunk_{k:05d}/chunk_{k:05d}.ddg'
        res = parse_ddg(f) if os.path.exists(f) and os.path.exists(os.path.dirname(f) + '/DONE') else {}
        if not res:
            missing.append(k)
        ddg[k] = res
    muts['ddg_rosetta'] = [ddg[c].get(lab, np.nan) for c, lab in zip(muts.chunk, muts.label)]
    partial = sorted(set(muts.loc[muts.ddg_rosetta.isna(), 'chunk']) - set(missing))
    print(f'{out}: {muts.ddg_rosetta.notna().sum()}/{len(muts)} mutations computed; {len(missing)} chunks missing, {len(partial)} done but incomplete')
    if missing or partial:
        open(f'{out}/tasks/missing_chunks.txt', 'w').write(','.join(map(str, sorted(set(missing) | set(partial)))) + '\n')
    return muts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='rosetta_predictions_rocklin')
    p.add_argument('--dest', default='rosetta_rocklin_predictions.csv')
    p.add_argument('--singles', action='append', default=[], help='extra run directory holding constituent singles (for pred_additive)')
    a = p.parse_args()
    muts = load(a.out)
    runs = [muts] + [load(d) for d in a.singles]
    single = pd.concat([r[~r.mut_type.str.contains(':')] for r in runs]).set_index(['code', 'mut_type'])['ddg_rosetta']
    assert single.index.is_unique, 'a single was computed in two run directories'
    single = single.to_dict()
    add = [v if ':' not in mt else sum(single.get((c, m), np.nan) for m in mt.split(':')) for c, mt, v in zip(muts.code, muts.mut_type, muts.ddg_rosetta)]
    muts['pred'] = -muts.ddg_rosetta
    muts['pred_additive'] = -np.array(add, dtype=float)
    muts[['uid', 'code', 'mut_type', 'ddg_rosetta', 'pred', 'pred_additive']].to_csv(a.dest, index=False)
    print(f'wrote {a.dest} ({muts.pred_additive.notna().sum()}/{len(muts)} rows with pred_additive)')


if __name__ == '__main__':
    main()
