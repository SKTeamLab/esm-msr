"""Check, per domain, that the domainome mutations line up with the structure and that an exact-match MSA exists."""
import glob, json, os, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths
import paths  # noqa: E402
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
TAB = str(paths.DOMAINOME_TABLE)
FORK = os.environ.get('MSR_ME_DOMAINOME_MSAS', str(paths.THIRD_PARTY_DIR / 'MutateEverything' / 'data' / 'domainome_2026'))

def chain_seq(path, chain='A'):
    seq, seen = [], set()
    for line in open(path):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
            rid = line[22:27]
            if rid not in seen:
                seen.add(rid); seq.append(AA3.get(line[17:20].strip(), 'X'))
    return ''.join(seq)

def query_of(path):
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
    return seq.replace('-', '').upper()

df = pd.read_csv(TAB, usecols=['domain_ID', 'mut_type', 'scaled_fitness', 'pdb_file', 'position', 'wild_type', 'mutation'],
                 low_memory=False)
df['code'] = df['domain_ID'].str.replace('/', '_', regex=False)
df = df.dropna(subset=['pdb_file', 'position'])
df['pdb_file'] = paths.fix_pdb_column(df['pdb_file'])

# every MSA in the fork's domainome_2026 tree, indexed by exact query sequence
msas = {}
for p in glob.glob(os.path.join(FORK, 'msa', '**', '*.a3m'), recursive=True):
    try:
        msas.setdefault(query_of(p), p)
    except Exception:
        pass
print(f'indexed {len(msas)} distinct MSA queries from {FORK}/msa', flush=True)

ok = {'struct_match': 0, 'struct_mismatch': 0, 'no_pdb': 0}
dom = {}
for code, g in df.groupby('code'):
    pf = g['pdb_file'].iloc[0]
    if not os.path.exists(pf):
        ok['no_pdb'] += len(g); dom[code] = {'status': 'no_pdb'}; continue
    seq = chain_seq(pf)
    good = sum(1 for p, w in zip(g.position.astype(int), g.wild_type) if 1 <= p <= len(seq) and seq[p-1] == w)
    ok['struct_match'] += good
    ok['struct_mismatch'] += len(g) - good
    dom[code] = {'status': 'ok' if good == len(g) else 'partial', 'n': int(len(g)), 'matched': int(good),
                 'len': len(seq), 'msa': msas.get(seq), 'pdb': pf}

n_msa = sum(1 for v in dom.values() if v.get('msa'))
print(f'domains {len(dom)} | rows matching structure {ok["struct_match"]} | mismatched {ok["struct_mismatch"]} | no pdb {ok["no_pdb"]}')
print(f'domains fully consistent: {sum(1 for v in dom.values() if v.get("status") == "ok")}')
print(f'domains with an exact-match MSA: {n_msa} of {len(dom)}')
bad = {k: v for k, v in dom.items() if v.get('status') != 'ok' or not v.get('msa')}
print(f'domains needing attention: {len(bad)}')
for k, v in list(bad.items())[:10]:
    print('   ', k, v.get('status'), 'matched', v.get('matched'), 'of', v.get('n'), 'msa', bool(v.get('msa')))
out = os.environ.get('MSR_DOMAINOME_AUDIT', str(paths.REPO_ROOT / 'outputs' / 'baselines' / 'domainome_audit.json'))
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(dom, open(out, 'w'), indent=1)
print('wrote domainome_audit.json')
