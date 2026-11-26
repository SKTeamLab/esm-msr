"""Task files for the constituent singles of a finished run's multi-mutants that were never computed (so collect.py can build pred_additive).

  python prepare_single_tasks.py --src rosetta_predictions_rocklin --dest rosetta_predictions_rocklin_singles [--chunk_cost 8]

Same layout as prepare_tasks.py, so relax.sh / ddg.sh / drive.sh run on --dest unchanged (OUT=<dest>).  The singles are scored on the SAME relaxed
structures: link <dest>/relaxed -> <src>/relaxed on the cluster (inputs/ is not needed because nothing is relaxed again).  A chunk holds one
design's singles, ordered by position.  Positions and wild types come from the multi-mutants' own components.
"""
import argparse
import os

import pandas as pd

from prepare_tasks import label


def missing_singles(muts):
    """{code: [single, ...]} (codes in table order, singles by position then residue) needed by a multi-mutant and not in the table."""
    multi = muts.mut_type.str.contains(':')
    have = set(zip(muts.code[~multi], muts.mut_type[~multi]))
    need = {}
    for code, mt in zip(muts.code[multi], muts.mut_type[multi]):
        need.setdefault(code, set()).update(mt.split(':'))
    out = {}
    for code in dict.fromkeys(muts.code):
        todo = sorted((s for s in need.get(code, ()) if (code, s) not in have), key=lambda s: (int(s[1:-1]), s[-1]))
        if todo:
            wts = {}
            for s in need[code]:
                assert wts.setdefault(int(s[1:-1]), s[0]) == s[0], f'{code}: two wild types at position {s[1:-1]}'
            out[code] = todo
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='rosetta_predictions_rocklin')
    p.add_argument('--dest', default='rosetta_predictions_rocklin_singles')
    p.add_argument('--chunk_cost', type=int, default=8)   # ~420 s per single on average -> ~56 min/chunk (3 h limit)
    a = p.parse_args()
    muts = pd.read_csv(f'{a.src}/tasks/mutations.csv', keep_default_na=False)
    n_res = pd.read_csv(f'{a.src}/tasks/chunks.csv', keep_default_na=False).drop_duplicates('code').set_index('code').n_res
    os.makedirs(f'{a.dest}/tasks/mutfiles', exist_ok=True)
    todo = missing_singles(muts)
    rows, chunks = [], []
    for code, singles in todo.items():
        for i in range(0, len(singles), a.chunk_cost):
            cur, k = singles[i:i + a.chunk_cost], len(chunks)
            with open(f'{a.dest}/tasks/mutfiles/chunk_{k:05d}.mut', 'w') as f:
                f.write(f'total {len(cur)}\n' + ''.join(f'1\n{m[0]} {m[1:-1]} {m[-1]}\n' for m in cur))
            chunks.append(dict(chunk=k, code=code, mut_structure='-', n_mut=len(cur), cost=len(cur), n_res=n_res[code]))
            rows += [dict(code=code, mut_structure='-', mut_type=m, chunk=k, label=label(m), uid=f'{code}_{m}') for m in cur]
    pd.DataFrame({'code': list(todo)}).to_csv(f'{a.dest}/tasks/structures.csv', index=False)
    pd.DataFrame(chunks).to_csv(f'{a.dest}/tasks/chunks.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{a.dest}/tasks/mutations.csv', index=False)
    print(f'{len(todo)} structures, {len(rows)} singles, {len(chunks)} chunks of <= {a.chunk_cost}')


if __name__ == '__main__':
    main()
