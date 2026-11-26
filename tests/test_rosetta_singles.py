"""inference_scripts/baselines/rosetta: tasks for the missing constituent singles, and collect.py adding them to pred_additive."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROSETTA = Path(__file__).resolve().parents[1] / 'inference_scripts' / 'baselines' / 'rosetta'


def run(script, *args):
    subprocess.run([sys.executable, str(ROSETTA / script), *args], check=True, cwd=ROSETTA, capture_output=True)


def write_ddg(out, chunk, scores):
    d = out / 'ddg' / f'chunk_{chunk:05d}'
    d.mkdir(parents=True, exist_ok=True)
    lines = [f'COMPLEX:   Round1: WT:  0.0\n'] + [f'COMPLEX:   Round1: {lab}:  {v}\n' for lab, v in scores.items()]
    (d / f'chunk_{chunk:05d}.ddg').write_text(''.join(lines))
    (d / 'DONE').touch()


def test_singles_tasks_and_additive(tmp_path):
    src, dest = tmp_path / 'src', tmp_path / 'singles'
    (src / 'tasks').mkdir(parents=True)
    pd.DataFrame([dict(code='d1', mut_structure='-', mut_type=m, chunk=0, label=lab, uid=f'd1_{m}') for m, lab in
                  [('A1G', 'MUT_1GLY'), ('A1G:C3D', 'MUT_1GLY_3ASP'), ('C3D:E5F', 'MUT_3ASP_5PHE')]]).to_csv(src / 'tasks' / 'mutations.csv', index=False)
    pd.DataFrame([dict(chunk=0, code='d1', mut_structure='-', n_mut=3, cost=3, n_res=9)]).to_csv(src / 'tasks' / 'chunks.csv', index=False)
    run('prepare_single_tasks.py', '--src', str(src), '--dest', str(dest), '--chunk_cost', '1')
    new = pd.read_csv(dest / 'tasks' / 'mutations.csv')
    assert list(new.mut_type) == ['C3D', 'E5F']                      # A1G is already computed
    assert list(new.label) == ['MUT_3ASP', 'MUT_5PHE']
    assert (dest / 'tasks' / 'mutfiles' / 'chunk_00001.mut').read_text() == 'total 1\n1\nE 5 F\n'
    assert list(pd.read_csv(dest / 'tasks' / 'chunks.csv').n_res) == [9, 9]

    write_ddg(src, 0, {'MUT_1GLY': 1.0, 'MUT_1GLY_3ASP': 4.0, 'MUT_3ASP_5PHE': 7.0})
    write_ddg(dest, 0, {'MUT_3ASP': 2.0})                            # E5F not computed
    out = tmp_path / 'pred.csv'
    run('collect.py', '--out', str(src), '--singles', str(dest), '--dest', str(out))
    d = pd.read_csv(out).set_index('mut_type')
    assert list(d.index) == ['A1G', 'A1G:C3D', 'C3D:E5F']            # the singles run adds no rows
    assert d.pred_additive['A1G:C3D'] == -3.0
    assert np.isnan(d.pred_additive['C3D:E5F'])
    assert (dest / 'tasks' / 'missing_chunks.txt').read_text() == '1\n'
