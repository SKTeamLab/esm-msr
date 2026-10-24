"""Align every protein to its family representative and store position maps.

A ``family`` (from 05_proteins.py) is a single-linkage cluster at >=95% identity over >=90%
coverage, i.e. structures/constructs of the same protein.  Exact-mutation comparisons across
datasets must be done in the representative's coordinates.
"""
import os, sys, pickle, collections
import pandas as pd, numpy as np
from Bio import Align
sys.path.insert(0, os.path.dirname(__file__))

ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
p=pd.read_pickle(os.path.join(D,'proteins.pkl'))
al=Align.PairwiseAligner(); al.mode='global'; al.open_gap_score=-10; al.extend_gap_score=-1
al.substitution_matrix=Align.substitution_matrices.load('BLOSUM62')

reps={}
for fam,g in p.groupby('family'):
    if len(g)==1: reps[fam]=g.iloc[0]['pid']; continue
    # representative: the sequence closest to the family's modal length (ties: first)
    med=int(g['seq'].str.len().median())
    g2=g.assign(d=(g['seq'].str.len()-med).abs()).sort_values(['d','pid'])
    reps[fam]=g2.iloc[0]['pid']

maps={}  # pid -> (family, {pos_in_pid: pos_in_rep})
seqs=dict(zip(p.pid,p.seq))
for fam,g in p.groupby('family'):
    rep=reps[fam]; rs=seqs[rep]
    for pid in g.pid:
        s=seqs[pid]
        if pid==rep:
            mp={i:i for i in range(1,len(s)+1)}
        else:
            try:
                a=next(iter(al.align(rs,s)))
            except StopIteration:
                mp={}
            else:
                mp={}; ri=si=0
                for (r0,r1),(s0,s1) in zip(a.aligned[0], a.aligned[1]):
                    # a.aligned gives blocks; iterate columns
                    pass
                # simpler: walk the aligned strings
                A,B=str(a[0]),str(a[1])
                for i,(x,y) in enumerate(zip(A,B), start=1):
                    if x!='-': ri+=1
                    if y!='-': si+=1
                    if x!='-' and y!='-': mp[si]=ri
        maps[pid]=(fam,mp,rs)
print('maps',len(maps))
with open(os.path.join(D,'family_maps.pkl'),'wb') as f:
    pickle.dump({'reps':reps,'maps':maps,'seqs':seqs},f)
# sanity: identities of maps
bad=0
for pid,(fam,mp,rs) in maps.items():
    s=seqs[pid]
    for sp,rp in list(mp.items())[:5]:
        if 1<=rp<=len(rs) and s[sp-1]!=rs[rp-1]: bad+=1
print('sample wt-consistency failures',bad)
print('families',len(reps),'multi-member',sum(1 for f,g in p.groupby('family') if len(g)>1))
