"""Extract protein sequences and PDB residue-number maps for code/position inventories
(DDMut S9028 + SM1242, DDGemb S2450, GeoDDG S8754).

Prefers ``_original.pdb`` over ``_processed.pdb`` so author residue numbers in S9028 and SM1242
align 1-to-1 with PDB ATOM records for downstream sequence-pair exact matching in ``12_code_exact.py``.
"""
import collections
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import SOFTWARE_DIR, WORK_DATA_DIR, data_file

OUT = os.path.join(WORK_DATA_DIR, 'inventory_proteins')
os.makedirs(OUT, exist_ok=True)

AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}
STDIR = os.environ.get('STRUCTURES_DIR', os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'structures'))


def pdb_chain_residues(path: str):
    """Return {chain: [(author_resi_str, aa_letter), ...]} in chain order."""
    out = collections.defaultdict(list)
    seen = set()
    for line in open(path, errors='ignore'):
        if line.startswith(('ATOM  ', 'HETATM')):
            res = line[17:20].strip()
            ch = line[21]
            resi = line[22:27].strip()
            key = (ch, resi)
            if key in seen:
                continue
            seen.add(key)
            out[ch].append((resi, AA.get(res, 'X')))
    return out


def find_pdb(code: str, chain: str):
    """Prefer _original.pdb so author residue numbers match S9028 / SM1242 tables."""
    for pat in [f'{code}_{chain}_original.pdb', f'{code}_{chain}.pdb', f'{code}_{chain}_processed.pdb']:
        p = os.path.join(STDIR, pat)
        if os.path.exists(p):
            return p
    if os.path.isdir(STDIR):
        for f in sorted(os.listdir(STDIR)):
            if f.upper().startswith(code.upper() + '_'):
                return os.path.join(STDIR, f)
    return None


# DDMut S9028
S = pd.read_csv(data_file('S9028.csv'))
recs = {}
for x in S['pdb_mut_chain'].astype(str):
    m = re.match(r'^([0-9A-Za-z]{4})_[A-Z]\d+[A-Z]_([A-Za-z0-9]+)$', x)
    if m:
        recs[(m.group(1).upper(), m.group(2))] = None

s9028 = []
resmaps = {}
for code, chain in recs:
    p = find_pdb(code, chain)
    if not p:
        s9028.append((f'inv:ddmut9028:{code}_{chain}', '', code, chain, ''))
        continue
    ch_map = pdb_chain_residues(p)
    chosen_ch = chain if chain in ch_map else (max(ch_map, key=lambda k: len(ch_map[k])) if ch_map else '')
    rl = ch_map.get(chosen_ch, [])
    seq = ''.join(aa for _, aa in rl)
    resmaps[f'{code}_{chain}'] = rl
    s9028.append((f'inv:ddmut9028:{code}_{chain}', seq, code, chain, p))
print('S9028 proteins:', len(s9028), 'with seq:', sum(1 for r in s9028 if r[1]))

# DDMut SM1242
sm = pd.read_csv(data_file('DD_SM1242.csv'))
recs2 = {}
for x in sm['struc'].astype(str):
    m = re.match(r'^([0-9A-Za-z]{4})_([A-Za-z0-9]+)$', x)
    if m:
        recs2[(m.group(1).upper(), m.group(2))] = None

sm2 = []
for code, chain in recs2:
    p = find_pdb(code, chain)
    if not p:
        continue
    ch_map = pdb_chain_residues(p)
    chosen_ch = chain if chain in ch_map else (max(ch_map, key=lambda k: len(ch_map[k])) if ch_map else '')
    rl = ch_map.get(chosen_ch, [])
    seq = ''.join(aa for _, aa in rl)
    resmaps[f'{code}_{chain}'] = rl
    sm2.append((f'inv:ddmutsm1242:{code}_{chain}', seq, code, chain, p))
print('SM1242 proteins:', len(sm2))

# DDGemb S2450 fasta
fa = []
h = ''
for line in open(data_file('S2450.fasta')):
    if line.startswith('>'):
        h = line[1:].strip()
    elif line.strip():
        fa.append((f'inv:ddgemb2450:{h}', line.strip(), h, '', ''))
print('S2450 proteins:', len(fa))

# GeoDDG S8754: reuse mapped S8754 sequences (training file is the same dataset)
gs = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', 's8754.pkl'))
gsprot = gs[['pid', 'seq']].drop_duplicates('pid')
geo = []
for p, s in gsprot.itertuples(index=False):
    pc = p.split(':')[-1]
    parts = pc.split('_', 1)
    c_id, ch_id = parts[0], (parts[1] if len(parts) > 1 else 'A')
    pdb_p = find_pdb(c_id, ch_id) or ''
    geo.append(('inv:geoddg8754:' + pc, s, c_id, ch_id, pdb_p))
print('S8754 proteins:', len(geo))

all_inv = s9028 + sm2 + fa + geo
with open(os.path.join(OUT, 'inventory_proteins.fasta'), 'w') as fh:
    for pid, seq, code, chain, _ in all_inv:
        if seq:
            fh.write(f'>{pid}\n{seq}\n')
pd.DataFrame(all_inv, columns=['pid', 'seq', 'code', 'chain', 'pdb_path']).to_pickle(
    os.path.join(OUT, 'inventory_proteins.pkl')
)
pd.to_pickle(resmaps, os.path.join(OUT, 'pdb_resmaps.pkl'))
print('written inventory_proteins.{fasta,pkl} and pdb_resmaps.pkl')
