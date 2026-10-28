"""Quarantine the superseded per-item-bias prediction/stats files.

`X_sigma1_Y` and `X_sigma1.0_Y` are two *generations* of the same artefact, not two spellings of a duplicate:
the `sigma1` file is the 2026-10-08 inference under the old per-item calibration bias
(mean(wt_lora_dddg_pred) = -0.13), the `sigma1.0` file is the 2026-10-23 regeneration under per-mutation bias
(mean = 0.000), which is the convention the project standardised on.

Newer runs (trim, release-baseline) also write `sigma1`, so the token alone cannot identify the stale generation --
only the presence of a `sigma1.0` sibling can. Those files are moved into a `_pre_pmb/` subdirectory rather than
renamed with a prefix, because any prefix would still match a `*_sigma1_...` glob.

A second pass (--stale) moves `_sigma1_` files that have NO `_sigma1.0_` sibling but were written before the
2026-10-23 per-mutation-bias switch, i.e. the old-generation files nothing replaced (the released checkpoint's
10-08 statistics). It deliberately leaves alone (a) every `_sigma1_` file written on or after 10-23: the baseline
triplicate, the trim and frozen-WT seeds and the masked devel runs all write the spelling `_sigma1_` and are the
NEWEST scoring, not the oldest; (b) Domainome files, which are singles-only (per-mutation bias does not move a
single) and have no regenerated copy; (c) backup suffixes (`.incl_bounded`, `.pre_gb1_fix`).

Dry-run unless --apply.
"""
import glob
import os
import sys

ROOT = '/home/sreeves/esm-msr-devel/analysis_notebooks'
APPLY = '--apply' in sys.argv
QUAR = '_pre_pmb'

moves = []
for top in ('predictions', 'stats'):
    for f in glob.glob(os.path.join(ROOT, top, '**', '*_sigma1_*'), recursive=True):
        if f'{os.sep}{QUAR}{os.sep}' in f:
            continue
        sib = f.replace('_sigma1_', '_sigma1.0_')
        if os.path.exists(sib):
            moves.append((f, os.path.join(os.path.dirname(f), QUAR, os.path.basename(f))))

if '--stale' in sys.argv:
    import time
    CUT = time.mktime((2026, 10, 23, 0, 0, 0, 0, 0, -1))
    BACKUP = ('.incl_bounded', '.pre_gb1_fix')
    for top in ('predictions', 'stats'):
        for f in glob.glob(os.path.join(ROOT, top, '**', '*_sigma1_*'), recursive=True):
            rel = os.path.relpath(f, os.path.join(ROOT, top))
            if (f'{os.sep}{QUAR}{os.sep}' in f or rel.split(os.sep)[0] == 'domainome' or f.endswith(BACKUP)
                    or os.path.exists(f.replace('_sigma1_', '_sigma1.0_')) or os.path.getmtime(f) >= CUT):
                continue
            moves.append((f, os.path.join(os.path.dirname(f), QUAR, os.path.basename(f))))

if APPLY:
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
print(f'{"quarantined" if APPLY else "planned"}={len(moves)}')
for s, d in moves[:3]:
    print('  ', os.path.relpath(s, ROOT), '->', QUAR + '/' + os.path.basename(d))
