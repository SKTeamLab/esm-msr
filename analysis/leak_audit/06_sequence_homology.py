"""Protein corpus and MMseqs2 sequence-homology graph.

Collects every protein sequence that appears on the test side or in a training inventory,
runs MMseqs2 all-vs-all (with SHA-1 cache invalidation on ``proteins.fasta``), and derives:
  - ``same-protein`` families: ``fident >= 0.95``, shorter-sequence coverage ``cov_min >= 0.90``,
    and length ratio ``min(len_p, len_q) / max(len_p, len_q) >= 0.80`` (single-linkage connected
    components), merging signal-peptide/construct variants symmetrically regardless of search
    direction while preventing isolated domains from chaining full-length multidomain proteins.
  - coarse ``homolog`` edges: ``evalue <= 1e-3`` or (``fident >= 0.30`` and ``cov_min >= 0.60``)
  - coarse ``weak`` edges: ``fident >= 0.20`` or ``evalue <= 0.1``
    (Note: the benchmark leak matrix uses the per-test-protein ``H``/``h`` classifications in
    ``11_homology_classes.py``, which require coverage over the *test* protein.)

Outputs ``$LEAK_DATA_DIR/{proteins.pkl, mmseqs_pairs.csv, protein_homologs.pkl}``.
"""
import hashlib
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import DERIVED_TESTSETS, MMSEQS_BIN, THREADS, WORK_DATA_DIR, hash_seq

os.makedirs(WORK_DATA_DIR, exist_ok=True)

prots = {}


def add(pid: str, seq: str, source: str):
    if not isinstance(seq, str) or not seq:
        return
    prots[pid] = (seq, source)


# 1. Test-side proteins
test_dir = os.path.join(WORK_DATA_DIR, 'testsets')
for f in sorted(os.listdir(test_dir)):
    if not f.endswith('.pkl') or f[:-4] in DERIVED_TESTSETS:
        continue
    d = pd.read_pickle(os.path.join(test_dir, f))
    for pid, seq in d[['pid', 'seq']].drop_duplicates().itertuples(index=False):
        add(pid, seq, 'test:' + f[:-4])

# 2. Inventory proteins (DDMut S9028/SM1662, DDGemb S2450, GeoDDG S8754) — require 05_inventory_proteins.py output
inv_fa = os.path.join(WORK_DATA_DIR, 'inventory_proteins', 'inventory_proteins.fasta')
if not os.path.exists(inv_fa):
    raise FileNotFoundError(
        f'Missing {inv_fa}; run 05_inventory_proteins.py before 06_sequence_homology.py so code-table inventories are indexed.'
    )
cur_pid = None
for line in open(inv_fa):
    line = line.strip()
    if line.startswith('>'):
        cur_pid = line[1:]
    elif cur_pid and line:
        add(cur_pid, line, 'inventory')
        cur_pid = None

# 3. MegaScale libraries (all library codes + synthesized mutant-background base scaffolds)
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
for c, seq in lib['wt_seq'].items():
    add('lib:' + str(c), seq, 'megascale-library')
print('proteins:', len(prots))

pdf = pd.DataFrame([(k, v[0], v[1]) for k, v in prots.items()], columns=['pid', 'seq', 'source'])
pdf['seq_hash'] = pdf['seq'].map(hash_seq)
pdf['seq_len'] = pdf['seq'].str.len()
fa = os.path.join(WORK_DATA_DIR, 'proteins.fasta')
with open(fa, 'w') as fh:
    for pid, seq in zip(pdf['pid'], pdf['seq']):
        fh.write(f'>{pid}\n{seq}\n')

fa_sha1 = hashlib.sha1(open(fa, 'rb').read()).hexdigest()
work = os.path.join(WORK_DATA_DIR, 'mmseqs')
os.makedirs(work, exist_ok=True)
out_tsv = os.path.join(WORK_DATA_DIR, 'mmseqs_hits.tsv')
sha_file = out_tsv + '.sha1'
cached_sha1 = open(sha_file).read().strip() if os.path.exists(sha_file) else ''

if not os.path.exists(out_tsv) or cached_sha1 != fa_sha1:
    tmp_dir = os.path.join(work, 'tmp')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
    cmd = [
        MMSEQS_BIN, 'easy-search', fa, fa, out_tsv, tmp_dir,
        '-s', '7.5', '-e', '10.0', '--max-seqs', '20000',
        '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits',
        '--threads', THREADS,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    with open(sha_file, 'w') as fh:
        fh.write(fa_sha1 + '\n')
else:
    print('reusing cached mmseqs_hits.tsv (sha1 match)')

hits = pd.read_csv(
    out_tsv, sep='\t', header=None,
    names=['a', 'b', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits'],
)
hits = hits[hits['a'] != hits['b']].copy()

# Symmetrize unordered pairs using exact sequence lengths from proteins.pkl (avoiding swapped qlen/tlen max bug)
len_map = dict(zip(pdf['pid'], pdf['seq_len']))
hits['p'] = np.where(hits['a'] < hits['b'], hits['a'], hits['b'])
hits['q'] = np.where(hits['a'] < hits['b'], hits['b'], hits['a'])
agg = hits.groupby(['p', 'q'], as_index=False).agg(
    fident=('fident', 'max'),
    alnlen=('alnlen', 'max'),
    evalue=('evalue', 'min'),
    bits=('bits', 'max'),
)
agg['len_p'] = agg['p'].map(len_map).astype(float)
agg['len_q'] = agg['q'].map(len_map).astype(float)
min_len = np.minimum(agg['len_p'], agg['len_q'])
max_len = np.maximum(agg['len_p'], agg['len_q'])
agg['cov_min'] = agg['alnlen'] / min_len
agg['cov_max'] = agg['alnlen'] / max_len
agg['len_ratio'] = min_len / max_len
agg['coverage'] = agg['cov_min']
agg.to_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'), index=False)
print('unordered pairs:', len(agg))

fam_edges = agg[(agg['fident'] >= 0.95) & (agg['cov_min'] >= 0.90) & (agg['len_ratio'] >= 0.80)]
g = nx.Graph()
g.add_nodes_from(pdf['pid'])
g.add_edges_from(zip(fam_edges['p'], fam_edges['q']))
comp = {}
for i, c in enumerate(nx.connected_components(g)):
    for n in c:
        comp[n] = i
pdf['family'] = pdf['pid'].map(comp)
pdf.to_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))

famtab = pdf.groupby('family').size().sort_values(ascending=False)
print('families:', famtab.size, 'largest:', famtab.head(5).to_dict())
print('families with >1 member:', int((famtab > 1).sum()))

edges = pd.DataFrame({
    'a': agg['p'],
    'b': agg['q'],
    'fident': agg['fident'],
    'cov': agg['cov_min'],
    'cov_max': agg['cov_max'],
    'evalue': agg['evalue'],
    'homolog': (agg['evalue'] <= 1e-3) | ((agg['fident'] >= 0.30) & (agg['cov_min'] >= 0.60)),
    'weak': (agg['fident'] >= 0.20) | (agg['evalue'] <= 0.1),
})
edges.to_pickle(os.path.join(WORK_DATA_DIR, 'protein_homologs.pkl'))
print('homolog edges:', int(edges['homolog'].sum()), 'weak:', int(edges['weak'].sum()))
