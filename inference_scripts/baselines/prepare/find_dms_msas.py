"""Match each DMS assay's structure sequence to an existing ColabFold MSA by exact query-sequence equality."""
import glob, os, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths, dms_source
import paths  # noqa: E402
import dms_source  # noqa: E402
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
P = dms_source.PRE
PROTS = {'DLG4': 'DLG4_HUMAN_Faure_2021_abundance_domain', 'GRB2': 'GRB2_HUMAN_Faure_2021_abundance_domain',
         'MYO': 'MYO_HUMAN_Kung_2025_display', 'ESTA': 'ESTA_BACSU_Nutschel_2020_dTm',
         'GB1': 'GB1_Wu_2016_binding_domain'}

def chain_seq(path, chain):
    seq, seen = [], set()
    for l in open(path):
        if l.startswith('ATOM') and l[12:16].strip() == 'CA' and l[21] == chain:
            r = l[22:27]
            if r not in seen:
                seen.add(r); seq.append(AA3.get(l[17:20].strip(), 'X'))
    return ''.join(seq)

def query_of(path):
    """First record's sequence: ColabFold writes the query first (a leading '#' line is the paired-msa header)."""
    name = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                if name is not None:
                    break
                name = line[1:].strip(); seq = ''
            elif name is not None:
                seq += line.strip()
    return seq.replace('-', '').upper() if name is not None else None

targets = {}
for short, prot in PROTS.items():
    d = pd.read_csv(f'{P}/{prot}.csv', low_memory=False, nrows=5)
    pf = paths.fix_pdb_path(d['pdb_file'].iloc[0])
    targets[short] = chain_seq(pf, d['chain'].iloc[0])

# directories searched for a precomputed MSA whose query equals the structure's sequence (data inputs, not in the repo)
ROOTS = [str(paths.THIRD_PARTY_DIR / 'MutateEverything' / 'data' / 'dms'), str(paths.THIRD_PARTY_DIR / 'MutateEverything' / 'data' / 'dms_old'),
         str(paths.THIRD_PARTY_DIR / 'ColabFold')]
hits = {k: [] for k in targets}
n = 0
for root in ROOTS:
    for p in glob.glob(os.path.join(root, '**', '*.a3m'), recursive=True):
        n += 1
        try:
            q = query_of(p)
        except Exception:
            continue
        for short, seq in targets.items():
            if q == seq:
                hits[short].append(p)
print(f'scanned {n} a3m files\n')
for short, seq in targets.items():
    print(f'== {short} (len {len(seq)}) -> {len(hits[short])} exact match(es)')
    for p in sorted(hits[short])[:6]:
        print(f'     {p}')
