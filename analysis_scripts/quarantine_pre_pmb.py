"""Quarantine the superseded per-item-bias prediction/stats files.

`X_sigma1_Y` and `X_sigma1.0_Y` are two *generations* of the same artefact, not two spellings of a duplicate:
the `sigma1` file is the 2026-10-08 inference under the old per-item calibration bias
(mean(wt_lora_dddg_pred) = -0.13), the `sigma1.0` file is the 2026-10-23 regeneration under per-mutation bias
(mean = 0.000), which is the convention the project standardised on.

Newer runs (trim, release-baseline) also write `sigma1`, so the token alone cannot identify the stale generation --
only the presence of a `sigma1.0` sibling can. Those files are moved into a `_pre_pmb/` subdirectory rather than
renamed with a prefix, because any prefix would still match a `*_sigma1_...` glob.

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

if APPLY:
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
print(f'{"quarantined" if APPLY else "planned"}={len(moves)}')
for s, d in moves[:3]:
    print('  ', os.path.relpath(s, ROOT), '->', QUAR + '/' + os.path.basename(d))
