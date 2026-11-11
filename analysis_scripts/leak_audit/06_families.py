"""Align every protein to its family representative and store 1-based position maps.

A ``family`` (from ``05_proteins.py``) is a single-linkage connected component at >=95% sequence
identity, >=90% shorter-sequence coverage, and >=0.80 length ratio—i.e. structures/constructs of
the same protein. Exact-mutation comparisons across datasets are evaluated in the representative's
residue coordinates.
"""
import os
import pickle
import sys

import pandas as pd
from Bio import Align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import WORK_DATA_DIR

p = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
al = Align.PairwiseAligner()
al.mode = 'global'
al.open_gap_score = -10
al.extend_gap_score = -1
al.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')

reps = {}
for fam, g in p.groupby('family'):
    if len(g) == 1:
        reps[fam] = g.iloc[0]['pid']
        continue
    med = int(g['seq'].str.len().median())
    g2 = g.assign(d=(g['seq'].str.len() - med).abs()).sort_values(['d', 'pid'])
    reps[fam] = g2.iloc[0]['pid']

maps = {}  # pid -> (family, {pos_in_pid: pos_in_rep}, rep_seq)
seqs = dict(zip(p['pid'], p['seq']))
for fam, g in p.groupby('family'):
    rep = reps[fam]
    rs = seqs[rep]
    for pid in g['pid']:
        s = seqs[pid]
        if pid == rep:
            mp = {i: i for i in range(1, len(s) + 1)}
        else:
            try:
                a = next(iter(al.align(rs, s)))
            except StopIteration:
                mp = {}
            else:
                mp = {}
                ri = si = 0
                for x, y in zip(str(a[0]), str(a[1])):
                    if x != '-':
                        ri += 1
                    if y != '-':
                        si += 1
                    if x != '-' and y != '-':
                        mp[si] = ri
        maps[pid] = (fam, mp, rs)

print('maps:', len(maps))
with open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'wb') as f:
    pickle.dump({'reps': reps, 'maps': maps, 'seqs': seqs}, f)

bad = 0
for pid, (fam, mp, rs) in maps.items():
    s = seqs[pid]
    for sp_i, rp_i in list(mp.items())[:5]:
        if 1 <= rp_i <= len(rs) and s[sp_i - 1] != rs[rp_i - 1]:
            bad += 1
print('sample wt-consistency mismatches (substitution variants across constructs):', bad)
print('families:', len(reps), 'multi-member:', sum(1 for _, g in p.groupby('family') if len(g) > 1))
