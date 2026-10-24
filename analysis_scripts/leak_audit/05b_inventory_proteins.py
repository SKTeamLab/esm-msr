"""Extract protein sequences for inventories that are only distributed as PDB codes."""
import os, re, sys, collections
import pandas as pd
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
OUT=os.path.join(D,'inventory_proteins'); os.makedirs(OUT, exist_ok=True)
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'U','PYL':'O'}
STDIR='/home/sareeves/software/esm-msr/data/structures'

def pdb_seqs(path):
    out=collections.defaultdict(list); seen=set()
    for line in open(path, errors='ignore'):
        if line.startswith(('ATOM  ','HETATM')):
            res=line[17:20].strip(); ch=line[21]; resi=line[22:27]
            key=(ch,resi)
            if key in seen: continue
            seen.add(key)
            out[ch].append(AA.get(res,'X'))
    return {c:''.join(s) for c,s in out.items()}

def find_pdb(code, chain):
    for pat in [f'{code}_{chain}_processed.pdb', f'{code}_{chain}_original.pdb', f'{code}_{chain}.pdb']:
        p=os.path.join(STDIR,pat)
        if os.path.exists(p): return p
    # any file starting with code
    for f in os.listdir(STDIR):
        if f.upper().startswith(code.upper()+'_'):
            return os.path.join(STDIR,f)
    return None

# DDMut S9028 / SM1242
S=pd.read_csv(os.path.join(D,'raw','S9028.csv'))
recs={}
for x in S['pdb_mut_chain'].astype(str):
    m=re.match(r'^([0-9A-Za-z]{4})_[A-Z]\d+[A-Z]_([A-Za-z0-9]+)$', x)
    if m:
        recs[(m.group(1).upper(), m.group(2))]=None
s9028=[]
for code,chain in recs:
    p=find_pdb(code,chain)
    if not p:
        s9028.append((f'inv:ddmut9028:{code}_{chain}','',code,chain)); continue
    sq=pdb_seqs(p)
    seq=sq.get(chain) or (max(sq.values(),key=len) if sq else '')
    s9028.append((f'inv:ddmut9028:{code}_{chain}',seq,code,chain))
print('S9028 proteins',len(s9028),'with seq',sum(1 for r in s9028 if r[1]))

sm=pd.read_csv(os.path.join(D,'raw','DD_SM1242.csv'))
recs2={}
for x in sm['struc'].astype(str):
    m=re.match(r'^([0-9A-Za-z]{4})_([A-Za-z0-9]+)$', x)
    if m: recs2[(m.group(1).upper(), m.group(2))]=None
sm2=[]
for code,chain in recs2:
    p=find_pdb(code,chain)
    if not p: continue
    sq=pdb_seqs(p); seq=sq.get(chain) or (max(sq.values(),key=len) if sq else '')
    sm2.append((f'inv:ddmutsm1242:{code}_{chain}',seq,code,chain))
print('SM1242 proteins',len(sm2))

# DDGemb S2450 fasta
fa=[]
for line in open(os.path.join(D,'raw','S2450.fasta')):
    if line.startswith('>'):
        h=line[1:].strip()
    else:
        fa.append((f'inv:ddgemb2450:{h}', line.strip(), h, ''))
print('S2450 proteins',len(fa))

# GeoDDG S8754: reuse the mapped test sequences (the training file is the same dataset)
gs=pd.read_pickle(os.path.join(D,'testsets','s8754.pkl'))
gsprot=gs[['pid','seq']].drop_duplicates('pid')
geo=[('inv:geoddg8754:'+p.split(':')[-1], s, p, '') for p,s in gsprot.itertuples(index=False)]
print('S8754 proteins',len(geo))

with open(os.path.join(OUT,'inventory_proteins.fasta'),'w') as fh:
    for pid,seq,code,chain in s9028+sm2+fa+geo:
        if seq: fh.write(f'>{pid}\n{seq}\n')
pd.DataFrame(s9028+sm2+fa+geo, columns=['pid','seq','code','chain']).to_pickle(os.path.join(OUT,'inventory_proteins.pkl'))
print('written')
