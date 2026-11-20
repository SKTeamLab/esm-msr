#!/usr/bin/env python
"""Run the Oct-2026 literature leakage audit steps (`01`..`20`) sequentially and sync exports via ``export_reference``.

Usage: ``python -m audits.homology_and_splits.oct2026_leak_audit.run_all [start_step_num]``
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[2]
STEPS = [
    '01_esmtherm_inventory.py',
    '02_ifum_inventory.py',
    '03_megascale.py',
    '04_testsets.py',
    '05_inventory_proteins.py',
    '06_sequence_homology.py',
    '07_structure_homology.py',
    '08_families.py',
    '09_split_audit.py',
    '10_overlap_engine.py',
    '11_homology_classes.py',
    '12_literature_exact.py',
    '13_mgnify_search.py',
    '14_esm3dg_homology.py',
    '15_esm3dg_exact.py',
    '16_rosetta.py',
    '17_proteingym.py',
    '18_matrix.py',
    '20_mgnify_vs_capped_train.py',
]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    start = int(args[0]) if args else 1
    env = dict(**__import__('os').environ)
    py_path = f"{ANALYSIS_DIR.parent / 'src'}:{ANALYSIS_DIR}"
    env['PYTHONPATH'] = f"{py_path}:{env['PYTHONPATH']}" if env.get('PYTHONPATH') else py_path
    for s in STEPS:
        n = int(s.split('_', 1)[0])
        if n < start:
            continue
        t0 = time.time()
        print(f'=== [{n:02d}] {s} ===', flush=True)
        rc = subprocess.call([sys.executable, str(HERE / s)], env=env)
        print(f'=== [{n:02d}] rc={rc} ({time.time() - t0:.1f}s) ===', flush=True)
        if rc != 0:
            return rc
    return subprocess.call(
        [sys.executable, '-m', 'audits.homology_and_splits.active_builders.export_reference', '--sync'],
        env=env,
    )


if __name__ == '__main__':
    raise SystemExit(main())
