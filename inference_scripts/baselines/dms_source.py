"""
The one place the DMS assay tables are read from, with a completeness guard.

WHY THIS EXISTS. There were two copies of the preprocessed DMS tables:

    <repo>/data/preprocessed/                  <- complete (the repo's own, use this)
    <esm-msr checkout>/data/preprocessed/      <- GB1 is missing V39A, V39C, V39D (an old copy, renamed *.BROKEN_...)

GB1_Wu_2016_binding_domain is a 4-site saturation library: 19 substitutions at each of positions 39, 40, 41, 54,
so 76 singles. The old esm-msr copy has 73, and every pipeline read it (thermompnn_inference.py,
build_dms_me_data.py, and esm_msr_testing.py's DMS block all hardcoded that path). The 149,357 rows the two copies
share are bit-identical, so the only damage is the three absent singles -- but a double is usable by the epistasis
metrics only when BOTH its singles are measured, so those three cost 152 of GB1's 2,091 two-mutation variants
(1,939 were scored). The broken copy has been renamed to *.BROKEN_MISSING_SINGLES_DO_NOT_USE.

``load(assay)`` reads the good copy and runs ``validate``, which fails loudly on exactly this class of defect:
every component single of every two-mutation variant must be present as its own row, and a saturation library must
have 19 substitutions at each position it mutates. Call it from anything that builds or scores DMS data.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

PRE = str(paths.PREPROCESSED)
# Assays whose single-mutant table must be saturated (19 substitutions at every mutated position).
SATURATED = {'GB1_Wu_2016_binding_domain'}
# Pinned shapes of the complete tables, so a silent swap back to a truncated copy is caught.
EXPECT = {'GB1_Wu_2016_binding_domain': dict(rows=149360, singles=76)}


def validate(df, assay, strict=True):
    """Raise (or warn) if ``df`` cannot support the epistasis metrics. Returns a dict of what it found."""
    mt = df['mut_type'].astype(str)
    n = mt.str.count(':') + 1
    singles = set(mt[n == 1])
    pos = {}
    for m in singles:
        pos.setdefault(int(m[1:-1]), set()).add(m[-1])

    orphan = [m for m in mt[n == 2] if not set(m.split(':')) <= singles]
    thin = {p: len(v) for p, v in pos.items() if len(v) != 19}
    found = dict(rows=len(df), singles=len(singles), doubles=int((n == 2).sum()),
                 orphan_doubles=len(orphan), positions=len(pos))

    # An orphan double is a DEFECT only where the library is meant to be saturated (GB1). The domain-wide assays
    # simply did not measure every single, so there it is a fact about the assay: warn, never fail.
    problems, warnings = [], []
    if mt.duplicated().any():
        problems.append(f'{int(mt.duplicated().sum())} duplicated mut_type rows')
    if orphan:
        where = problems if assay in SATURATED else warnings
        where.append(f'{len(orphan)} two-mutation variants whose component single is absent, e.g. {orphan[:3]}')
    if assay in SATURATED and thin:
        problems.append(f'not saturated: positions with != 19 substitutions {thin}')
    if warnings:
        print(f'[note] {assay}: ' + '; '.join(warnings) + ' (inherent to this assay, not a defect)')
    exp = EXPECT.get(assay)
    if exp and any(found[k] != v for k, v in exp.items()):
        problems.append(f'shape {{rows: {found["rows"]}, singles: {found["singles"]}}} != expected {exp}')

    if problems:
        msg = f'{assay}: ' + '; '.join(problems)
        if strict:
            raise AssertionError(msg)
        print(f'[WARN] {msg}')
    return found


def load(assay, strict=True):
    """The complete preprocessed table for one DMS assay, validated, with pdb_file pointed at this machine's structures."""
    df = pd.read_csv(os.path.join(PRE, f'{assay}.csv'), low_memory=False)
    if 'pdb_file' in df.columns:
        df['pdb_file'] = paths.fix_pdb_column(df['pdb_file'])
    validate(df, assay, strict=strict)
    return df


if __name__ == '__main__':
    import sys
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(PRE)
                                   if f.endswith('.csv') and not f.endswith('_mapped.csv'))
    for a in names:
        try:
            df = pd.read_csv(os.path.join(PRE, f'{a}.csv'), low_memory=False)
            print(f'{a:45s} {validate(df, a, strict=False)}')
        except Exception as e:                                        # noqa: BLE001
            print(f'{a:45s} FAILED: {e}')
