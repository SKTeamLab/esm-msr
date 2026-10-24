"""Protein corpus + homology graph.

Collects every protein sequence that appears on the test side or in a training inventory,
runs MMseqs2 all-vs-all, and derives:
  - same-protein families (identity >= 0.95, coverage >= 0.9, single linkage)
  - homolog edges (MMseqs E <= 1e-3 or identity >= 0.30 with coverage >= 0.6)
  - weak/ambiguous edges (identity >= 0.20 or E <= 1e-1)
Outputs data/proteins.pkl, data/mmseqs_hits.csv, data/protein_families.pkl, data/protein_homologs.pkl
"""
import os, sys, pickle, subprocess, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import hash_seq

ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
os.makedirs(D, exist_ok=True)
MM=os.environ.get('MMSEQS','/usr/bin/mmseqs')

prots={}
def add(pid, seq, source):
    if not isinstance(seq,str) or not seq or seq!=seq: return
    prots[pid]=(seq, source)

# test-side proteins
for f in sorted(os.listdir(os.path.join(D,'testsets'))):
    d=pd.read_pickle(os.path.join(D,'testsets',f))
    for pid,seq in d[['pid','seq']].drop_duplicates().itertuples(index=False):
        add(pid, seq, 'test:'+f[:-4])
# inventory proteins (DDGemb S2450, DDMut S9028/SM1242, GeoDDG S8754)
inv_fa=os.path.join(D,'inventory_proteins','inventory_proteins.fasta')
if os.path.exists(inv_fa):
    pid=None
    for line in open(inv_fa):
        line=line.strip()
        if line.startswith('>'):
            pid=line[1:]; add(pid,'', 'inventory')  # seq filled below
        elif pid:
            prots[pid]=(line,'inventory')
# megascale libraries (all 479, not only the split)
lib=pd.read_pickle(os.path.join(D,'megascale_libraries.pkl'))
for c,seq in lib['wt_seq'].items():
    add('lib:'+c, seq, 'megascale-library')
print('proteins', len(prots))

pdf=pd.DataFrame([(k,v[0],v[1]) for k,v in prots.items()], columns=['pid','seq','source'])
pdf['seq_hash']=pdf['seq'].map(hash_seq)
pdf.to_pickle(os.path.join(D,'proteins.pkl'))
fa=os.path.join(D,'proteins.fasta')
with open(fa,'w') as fh:
    for pid,seq in zip(pdf.pid,pdf.seq): fh.write(f'>{pid}\n{seq}\n')

work=os.path.join(D,'mmseqs')
os.makedirs(work, exist_ok=True)
out=os.path.join(D,'mmseqs_hits.tsv')
if not os.path.exists(out):
    cmd=[MM,'easy-search',fa,fa,out,os.path.join(work,'tmp'),'-s','7.5','-e','10.0',
         '--max-seqs','20000','--format-output',
         'query,target,fident,alnlen,qlen,tlen,evalue,bits','--threads','8']
    print(' '.join(cmd)); subprocess.run(cmd, check=True)
hits=pd.read_csv(out, sep='\t', header=None, names=['a','b','fident','alnlen','qlen','tlen','evalue','bits'])
hits=hits[hits.a!=hits.b].copy()
# symmetrize: best evidence per unordered pair
hits['p']=hits[['a','b']].min(axis=1); hits['q']=hits[['a','b']].max(axis=1)
agg=hits.groupby(['p','q'], as_index=False).agg(fident=('fident','max'), alnlen=('alnlen','max'),
        qlen=('qlen','max'), tlen=('tlen','max'), evalue=('evalue','min'), bits=('bits','max'))
agg['coverage']=agg['alnlen']/agg[['qlen','tlen']].min(axis=1)
agg.to_csv(os.path.join(D,'mmseqs_pairs.csv'), index=False)
print('pairs', len(agg))
fam=agg[(agg['fident']>=0.95)&(agg['coverage']>=0.9)]
import networkx as nx
g=nx.Graph(); g.add_nodes_from(pdf.pid)
g.add_edges_from(zip(fam.p, fam.q))
comp={}
for i,c in enumerate(nx.connected_components(g)):
    for n in c: comp[n]=i
pdf['family']=pdf.pid.map(comp)
pdf.to_pickle(os.path.join(D,'proteins.pkl'))
famtab=pdf.groupby('family').size().sort_values(ascending=False)
print('families', famtab.size, 'largest', famtab.head(5).to_dict())
print('families with >1 member', (famtab>1).sum())

edges=pd.DataFrame({
 'a':agg.p,'b':agg.q,'fident':agg['fident'],'cov':agg['coverage'],'evalue':agg['evalue'],
 'homolog': ((agg['evalue']<=1e-3)|((agg['fident']>=0.30)&(agg['coverage']>=0.6))),
 'weak': ((agg['fident']>=0.20)|(agg['evalue']<=0.1)),
})
edges.to_pickle(os.path.join(D,'protein_homologs.pkl'))
print('homolog edges', int(edges.homolog.sum()), 'weak', int(edges.weak.sum()))
