"""Swap the labels of a finished MGnify run for those of the v2 table (``data/mgnify/mgnify_mapped.csv``).

The v2 preprocessing keeps every v1 row (same uid, sequences, structures, split) but re-derives the WT dG of 1,772 scaffolds from additional WT
measurements, which changes dG_wt / ddG_ML on 6,035 rows (dddG_ML on 13).  Predictions do not depend on those labels, so a v1 run is re-scored by joining
the v2 label columns on ``uid``.  ``cens`` becomes a string in v2; it is mapped back to the legacy integers the scorers use.
"""
import os

import pandas as pd

LABEL_COLS = ['dG_wt', 'dG_ML', 'ddG', 'ddG_ML', 'dddG_ML', 'ddG_additive_ML', 'wt_CI', 'cens', 'in_range_1_5', 'qc_pass_both', 'has_all_singles']
CENS_LEGACY = {'default': 0, 'mut_below_neg1': -1, 'wt_out_of_range': -2, 'mut_above_pos5': 1}


def default_table():
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    while d != os.path.dirname(d):
        for sub in ('data/mgnify', 'repo/data/mgnify'):
            f = os.path.join(d, sub, 'mgnify_mapped.csv')
            if os.path.exists(f):
                return f
        d = os.path.dirname(d)
    return os.path.join('data', 'mgnify', 'mgnify_mapped.csv')


def legacy_cens(s):
    return s.map(lambda v: CENS_LEGACY.get(v, v)).astype(int) if s.dtype == object else s


def apply_v2_labels(d, table=None):
    """Replace the label columns of prediction frame ``d`` by the v2 values (matched on uid); returns the frame (labels not found stay)."""
    table = table or default_table()
    cols = [c for c in LABEL_COLS if c in pd.read_csv(table, nrows=0).columns]
    v2 = pd.read_csv(table, usecols=['uid'] + cols, low_memory=False).set_index('uid')
    v2['cens'] = legacy_cens(v2['cens'])
    keep = d['uid'].isin(v2.index)
    out = d.copy()
    ch = {}
    for c in cols:
        new = out['uid'].map(v2[c])
        if c in out.columns and c in ('dG_wt', 'ddG_ML', 'dddG_ML'):
            ch[c] = int(((out[c] - new).abs() > 1e-6).sum())
        out[c] = new.where(keep, out[c]) if c in out.columns else new
    print(f'v2 labels applied to {int(keep.sum()):,}/{len(d):,} rows; rows changed: {ch}')
    return out
