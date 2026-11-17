#!/usr/bin/env python
"""Homology of every MGnify scaffold to ESM-MSR's capped-split training set.

The audit's steps 13-15 ask the reverse question (benchmarks vs ESM3dG's MGnify training index).  ESM-MSR never trained on
MGnify, so here the training side is the 172 ``splits_oct06_capped`` train scaffolds (all their library WTs, mutant
backbones included) and the query side is the 180,367 MGnify scaffold WTs of ``data/mgnify/mgnify_mapped.csv``.

Classes follow ``lib.classify_hits`` (sequence evidence only; no Foldseek here): ``sameprot`` / ``H`` / ``h`` / ``low`` /
``none``, with coverage taken over the query (MGnify) protein.  Output ``mgnify_homology_classes.csv`` in
``$LEAK_OUT_DIR``: code_wt, wt_split, class, best_fident, best_evalue, best_qcov, best_target,
exact_wt (WT sequence identical to a training library WT).
"""
import argparse
import os
import pickle
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import MMSEQS_BIN, OUT_DIR, WORK_DATA_DIR, classify_hits, repo_data

ap = argparse.ArgumentParser()
ap.add_argument('--mgnify', default=None)
ap.add_argument('--out_dir', default=OUT_DIR)
ap.add_argument('--threads', type=int, default=4)
a = ap.parse_args()
mg_path = a.mgnify or repo_data('mgnify', 'mgnify_mapped.csv')
wd = os.path.join(a.out_dir, 'mgnify_vs_capped')
os.makedirs(wd, exist_ok=True)

train_codes = set(pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))['train'])
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
tr = lib[lib['code_wt'].isin(train_codes)].drop_duplicates('wt_seq')
tr_fa = os.path.join(wd, 'capped_train.fa')
with open(tr_fa, 'w') as fh:
    for r in tr.itertuples():
        fh.write(f'>{r.Index}\n{r.wt_seq}\n')
print(f'{len(train_codes)} train scaffolds -> {len(tr)} distinct library WT sequences')

mg = pd.read_csv(mg_path, usecols=['code_wt', 'wt_seq', 'wt_split']).drop_duplicates('code_wt')
q_fa = os.path.join(wd, 'mgnify_scaffolds.fa')
with open(q_fa, 'w') as fh:
    for r in mg.itertuples():
        fh.write(f'>{r.code_wt}\n{r.wt_seq}\n')
m8 = os.path.join(wd, 'mgnify_vs_capped_train.m8')
if not os.path.exists(m8):
    tmp = os.path.join(wd, 'tmp')
    subprocess.run([MMSEQS_BIN, 'easy-search', q_fa, tr_fa, m8, tmp, '-s', '7.5', '-e', '10', '--max-seqs', '500',
                    '--threads', str(a.threads), '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits,qcov'],
                   check=True)
h = pd.read_csv(m8, sep='\t', names=['q', 't', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits', 'qcov'])
print(len(h), 'hits for', h.q.nunique(), 'scaffolds')
hits = {q: g for q, g in h.groupby('q')}
trseq = set(tr['wt_seq'])
rows = []
for r in mg.itertuples():
    g = hits.get(r.code_wt)
    if g is None:
        cls, b = 'none', (None,) * 4
    else:
        cls = classify_hits(zip(g.fident, g.qcov, g.evalue))
        i = g.sort_values(['evalue', 'bits'], ascending=[True, False]).iloc[0]
        b = (i.fident, i.evalue, i.qcov, i.t)
    rows.append((r.code_wt, r.wt_split, cls, *b, r.wt_seq in trseq))
out = pd.DataFrame(rows, columns=['code_wt', 'wt_split', 'cls', 'best_fident', 'best_evalue', 'best_qcov', 'best_target', 'exact_wt'])
out.to_csv(os.path.join(a.out_dir, 'mgnify_homology_classes.csv'), index=False)
print(out.cls.value_counts().to_string())
print(out.groupby(['wt_split', 'cls']).size().unstack(fill_value=0).to_string())
print('exact WT matches:', int(out.exact_wt.sum()))
