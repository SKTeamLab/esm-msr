"""End-to-end runner for the cross-set homology and training-data leakage audit.

Executes the complete dependency DAG in topological order so every intermediate table and final
output in ``docs/benchmarks/`` and ``analysis/reference/`` can be regenerated cleanly from a single
command:

    python analysis/leak_audit/run_all.py
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    '00_build_inventories.py',
    '00b_extract_ifum_split.py',
    '01_megascale.py',
    '02_testsets.py',
    '04_overlap_repo_models.py',
    '05b_inventory_proteins.py',
    '05_proteins.py',
    '05c_tmpnnd_inventory.py',
    '05d_structure_homology.py',
    '00c_prepare_mgnify.py',
    '06_families.py',
    '07_relations.py',
    '09_homology_audit.py',
    '10_overlap_engine.py',
    '11_homology_classes.py',
    '12_code_exact.py',
    '15_mgnify.py',
    '19_esm3dg_exact.py',
    '14_pg.py',
    '13_matrix.py',
    '18_db.py',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-step', default=None, help='Start execution at the first step matching this substring')
    ap.add_argument('--skip-upstream', action='store_true', help='Skip 00/00b upstream inventory extraction if CSVs exist')
    args = ap.parse_args()

    started = args.from_step is None
    t0_all = time.time()
    for step in STEPS:
        if not started:
            if args.from_step in step:
                started = True
            else:
                continue
        if args.skip_upstream and step in ('00_build_inventories.py', '00b_extract_ifum_split.py'):
            continue
        p = os.path.join(SCRIPT_DIR, step)
        print(f'\n================ [{step}] ================', flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, p], check=True)
        print(f'---------------- [{step} done in {time.time() - t0:.1f}s] ----------------', flush=True)
    print(f'\nAll steps completed in {time.time() - t0_all:.1f}s.')


if __name__ == '__main__':
    main()
