"""Rosetta (relax + cartesian_ddg) task files for the scaffolds of data/mgnify (login-node light; pandas only).  Adapted from narval's
cluster_inference_scripts/capped/prepare_capped_tasks.py (same outputs, same flags downstream), with one change: a scaffold here is its own
structure file (no mutant-background step), so structures are keyed by `code`.

  python prepare_tasks.py --mgnify_dir <data/mgnify> --source rocklin_denovo --out rosetta_predictions_rocklin [--chunk_cost 10]

Writes under --out:
  inputs/<code>.pdb         first model, chain A, heavy ATOM records only (Rosetta rebuilds hydrogens)
  tasks/structures.csv      code                      -> relax.sh   (array index = line - 2)
  tasks/chunks.csv          chunk,code,mut_structure,n_mut,cost,n_res -> ddg.sh (array index = line - 2; mut_structure is always '-')
  tasks/mutfiles/chunk_NNNNN.mut
  tasks/mutations.csv       code, mut_structure, mut_type, chunk, label, uid -> collect.py
Backbone-mutated scaffolds (backbone_mutation set) are refused: they need the background step of the capped pipeline.
"""
import argparse
import os

import pandas as pd

THREE = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS',
         'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
         'W': 'TRP', 'Y': 'TYR'}
ONE = {v: k for k, v in THREE.items()}


def clean_pdb(src, dst, chain='A'):
    seq, out = {}, []
    for line in open(src):
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM  ') and line[21] == chain:
            elem = line[76:78].strip() or line[12:16].strip()[0]
            if elem == 'H':
                continue
            out.append(line)
            seq[int(line[22:26])] = ONE.get(line[17:20], 'X')
    assert out, f'no ATOM records for chain {chain} in {src}'
    with open(dst, 'w') as f:
        f.writelines(out)
        f.write('TER\nEND\n')
    return seq


def label(mut_type):
    """cartesian_ddg's name for a mutant in its .ddg output: MUT_<pos><AAA>[_<pos><AAA>...], positions ascending."""
    ms = sorted(mut_type.split(':'), key=lambda m: int(m[1:-1]))
    return 'MUT_' + '_'.join(f'{m[1:-1]}{THREE[m[-1]]}' for m in ms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mgnify_dir', required=True)
    p.add_argument('--table', default=None, help='default: <mgnify_dir>/mgnify_mapped.csv[.gz]')
    p.add_argument('--source', default='rocklin_denovo')
    p.add_argument('--out', required=True)
    p.add_argument('--chunk_cost', type=int, default=10)   # mutants per chunk: ~8 min each at 70 residues -> ~80 min/chunk (3 h limit)
    a = p.parse_args()
    t = a.table or next(f for f in (os.path.join(a.mgnify_dir, 'mgnify_mapped.csv.gz'), os.path.join(a.mgnify_dir, 'mgnify_mapped.csv')) if os.path.exists(f))
    d = pd.read_csv(t, usecols=['uid', 'source', 'code', 'chain', 'pdb_file', 'backbone_mutation', 'mut_type_renumbered', 'wt_seq'], low_memory=False)
    d = d[d.source == a.source]
    assert (d.chain == 'A').all() and d.backbone_mutation.isna().all(), 'chain A, wild-type backbones only'
    for sub in ['inputs', 'tasks/mutfiles']:
        os.makedirs(f'{a.out}/{sub}', exist_ok=True)
    codes = list(dict.fromkeys(d.code))
    seqs = {}
    for c, f in d.drop_duplicates('code').set_index('code').pdb_file.items():
        seqs[c] = clean_pdb(os.path.join(a.mgnify_dir, f), f'{a.out}/inputs/{c}.pdb')
        assert sorted(seqs[c]) == list(range(1, len(seqs[c]) + 1)), f'{c}: residues not numbered 1..N'
    for c, ws in d.drop_duplicates('code').set_index('code').wt_seq.items():
        assert ''.join(seqs[c][i] for i in sorted(seqs[c])) == ws, f'{c}: model sequence != library wild type'
    pd.DataFrame({'code': codes}).to_csv(f'{a.out}/tasks/structures.csv', index=False)
    rows, chunks = [], []
    for c, g in d.groupby('code', sort=False):
        cur, uids = [], []
        mts = list(g.mut_type_renumbered)
        for i, (mt, u) in enumerate(zip(mts, g.uid)):
            cur.append(mt); uids.append(u)
            if len(cur) >= a.chunk_cost or i == len(mts) - 1:
                k = len(chunks)
                with open(f'{a.out}/tasks/mutfiles/chunk_{k:05d}.mut', 'w') as f:
                    f.write(f'total {sum(m.count(":") + 1 for m in cur)}\n')
                    for m in cur:
                        parts = m.split(':')
                        f.write(f'{len(parts)}\n' + ''.join(f'{x[0]} {x[1:-1]} {x[-1]}\n' for x in parts))
                chunks.append(dict(chunk=k, code=c, mut_structure='-', n_mut=len(cur), cost=len(cur), n_res=len(seqs[c])))
                rows += [dict(code=c, mut_structure='-', mut_type=m, chunk=k, label=label(m), uid=uu) for m, uu in zip(cur, uids)]
                cur, uids = [], []
    pd.DataFrame(chunks).to_csv(f'{a.out}/tasks/chunks.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{a.out}/tasks/mutations.csv', index=False)
    print(f'{len(codes)} structures, {len(rows)} mutations, {len(chunks)} chunks of <= {a.chunk_cost}')


if __name__ == '__main__':
    main()
