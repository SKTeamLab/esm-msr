"""Re-run the external-dataset structure steps of esm_msr.preprocess.main on a few structures, in a separate folder, and
compare each stage with the reference files in data/structures.

  --phase download   (login node)   RCSB pdb + fasta  ->  <root>/raw
  --phase process    (cpu job)      caps -> heteroatoms -> noncanonical -> renumber -> repair -> add_lines, then compare
"""
import argparse, os, shutil, sys, traceback
import pandas as pd
from pathlib import Path
from esm_msr import preprocess as P

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--phase', required=True, choices=['download', 'process'])
ap.add_argument('--root', required=True)
ap.add_argument('--n', type=int, default=8)
a = ap.parse_args()
root = Path(a.root); raw = root / 'raw'; out = root / 'structures'

t = pd.read_csv(REPO / 'data/preprocessed/ssym_mapped.csv')
pairs = t[['code', 'chain']].drop_duplicates().reset_index(drop=True)
pairs = pairs.iloc[::max(1, len(pairs) // a.n)].head(a.n)
print(pairs.to_string())

def atoms(path):
    return [l[:54] for l in open(path) if l.startswith(('ATOM', 'HETATM'))]

if a.phase == 'download':
    for code, chain in pairs.itertuples(index=False):
        P.download_pdb(code, str(raw), dataset='SSYM')
    sys.exit()

out.mkdir(parents=True, exist_ok=True)
for code, chain in pairs.itertuples(index=False):
    s = f'{code}_{chain}'
    f = lambda kind: str(out / f'{s}_{kind}.pdb')
    ref = lambda kind: REPO / f'data/structures/{s}_{kind}.pdb'
    print(f'\n===== {s}')
    try:
        shutil.copy(raw / f'{code.lower()}.pdb', f('original'))
        P.remove_caps(f('original'))
        P.remove_heteroatoms(f('original'), f('intermediate'))
        steps = {'original': 'original'}
        P.fix_noncanonical_residues(f('intermediate'), f('intermediate'))
        P.renumber_pdb(f('intermediate'), f('intermediate'))
        steps['intermediate'] = 'intermediate'
        P.repair_pdb(f('intermediate'), f('processed'), sequence_file=str(raw / f'{code.lower()}.fasta'), chain_id=chain, verbose=False)
        P.add_lines(f('processed'))
        steps['processed'] = 'processed'
    except Exception:
        traceback.print_exc(limit=2)
    for kind in ('original', 'intermediate', 'processed'):
        if not os.path.exists(f(kind)):
            print(f'  {kind:12s} NOT PRODUCED'); continue
        if not ref(kind).exists():
            print(f'  {kind:12s} no reference'); continue
        same_bytes = open(f(kind), 'rb').read() == open(ref(kind), 'rb').read()
        same_atoms = atoms(f(kind)) == atoms(ref(kind))
        print(f'  {kind:12s} bytes-identical={same_bytes}  atom-records-identical={same_atoms}  ({len(atoms(f(kind)))} vs {len(atoms(ref(kind)))})')
