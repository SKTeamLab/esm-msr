#!/usr/bin/env python
"""ProteinGym: every FINAL_* ablation against FINAL_p0, paired per assay, by category.

Per assay: rho of the WT head (``final_pgym_wt/<family>/s<seed>``, the mean over seeds 1-3) of the family minus that of p0 (the same seeds), and for p1 also the combined prediction
(``final_pgym_comb``: the mean over the complete seeds, 1 and 3 for p1; the only family besides p0 with a combined ProteinGym run).  The ProteinGym assays homologous to the training set under the split's four-part rule are left out (``msr_analysis.heldout``).  Per category and over the
five categories (the mean of the five category means, ``PG avg`` of the ablation table): mean delta rho over the assays, its 95% interval and the two-sided p of a bootstrap over assays (20000
resamples, one draw of assays per category shared by every family; the draws of ``msr_analysis.boot.paired_diff``'s convention), Holm-adjusted over the families x categories of a head.
A family that lacks more than 3 of the held-out assays is skipped.  Writes ``results/paired_vs_p0_pg.csv`` in the columns of ``paired_vs_p0.csv``.

    python analysis/ablation_vs_p0/02_pg_vs_p0.py
"""
import sys

import numpy as np
import pandas as pd

import abl as A
import pgsigma as P
from msr_analysis.final_runs import pgym_per_dms_seeds

B = 20000
CATS = A.C.CATS


def main():
    leak = pd.read_csv(P.OUT / 'leak_per_assay.csv').set_index('assay')
    keep = [a for a in leak.index if a not in P.train_overlap()]
    cat = leak.loc[keep, 'category']
    per = {}
    for fam in [A.REF] + A.ABL:
        r, ns = pgym_per_dms_seeds(fam, 'wt', dms_ids=keep)                    # the mean over the seeds with a complete run
        if r is None or r.notna().sum() < len(keep) - 3:
            print(f'  skipped {fam}: {None if r is None else int(r.notna().sum())} assays')
            continue
        per[(fam, 'wt')] = r
    t = pd.read_csv(A.C.OUT / 'pg_per_assay.csv', index_col='assay').loc[keep]
    per[(A.REF, 'comb')] = t['p0_all_comb']
    per[('FINAL_p1', 'comb')] = t['p1_all_comb']
    rng = np.random.default_rng(16)
    draws = {c: rng.choice(np.flatnonzero((cat == c).to_numpy()), (B, int((cat == c).sum()))) for c in CATS}
    rows = []
    for h in A.HEADS:
        ref = per.get((A.REF, h))
        for fam in A.ABL:
            r = per.get((fam, h))
            if ref is None or r is None:
                continue
            d = (r - ref).to_numpy(float)
            cm = {c: np.nanmean(d[(cat == c).to_numpy()]) for c in CATS}
            reps = {c: np.nanmean(d[draws[c]], axis=1) for c in CATS}
            reps['PG avg'] = np.mean([reps[c] for c in CATS], axis=0)
            cm['PG avg'] = float(np.mean([cm[c] for c in CATS]))
            for c in CATS + ['PG avg']:
                x = reps[c]
                p = max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1 / (B + 1))
                rows.append(dict(group='proteingym', set=c, subset='all', family=fam, head=h, rho_family=float(np.nanmean(r[(cat == c).to_numpy()])) if c in CATS else np.nan,
                                 rho_p0=float(np.nanmean(ref[(cat == c).to_numpy()])) if c in CATS else np.nan, delta=cm[c], lo=np.quantile(x, .025), hi=np.quantile(x, .975), p=p,
                                 n=int((cat == c).sum()) if c in CATS else len(d)))
    D = pd.DataFrame(rows)
    D['p_holm'] = np.nan
    for h in A.HEADS:
        i = D['head'] == h
        D.loc[i, 'p_holm'] = A.C.holm(D.loc[i, 'p'])
    D.to_csv(A.OUT / 'paired_vs_p0_pg.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 250, 'display.max_rows', 200)
    print(D.pivot_table(index=['head', 'set'], columns='family', values='delta').round(3).to_string())
    print(D[D.p_holm < 0.05][['set', 'family', 'head', 'delta', 'p_holm']].round(4).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
