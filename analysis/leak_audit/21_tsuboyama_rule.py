#!/usr/bin/env python
"""The split's four-part homology filter applied to every held-out benchmark: which ProteinGym assays, Domainome domains, MGnify scaffolds and Rocklin designs are homologous to the training set.

The rule that built the Homology-Restricted MegaScale split (``docs/SPLITTING_AND_HOMOLOGY_AUDIT.md``, ``preprocessing/homology.py: homology_edge_mask``) joins two proteins when ANY of four criteria holds:

  1. MMseqs2 E-value <= 1e-3        (``mmseqs easy-search -s 7.5 -e 10.0 --max-seqs 10000``, as ``homology.mmseqs_all_vs_all``)
  2. identity > 0.40                 (BLOSUM62 local alignment, open -10 / extend -1, matches / length of the SHORTER sequence: ``homology.pairwise_identity``, computed for EVERY pair, not only
                                     for search hits)
  3. Foldseek E-value <= 1e-3        (``foldseek easy-search --exhaustive-search 1 -e 10``)
  4. Foldseek TM-score > 0.70        (normalised by the shorter chain: max of the query and target TM-scores)

Here every benchmark protein is compared with the 172 training libraries of the split (training only: the validation libraries are not used) and flagged when any of the four criteria holds.  E-values depend
on the size of the searched database, so the benchmark proteins are searched against the split's own search corpus (the 1,147 proteins of ``$HOMOLOGY_DIR/single_chain_pdbs``: the MegaScale libraries and the
external quarantine chains), exactly the database whose all-vs-all search gave the split's E-values, and only the hits to training libraries are kept (forward direction).  The reverse direction (a training library
searched against the benchmark proteins, the other half of the split's symmetrised pair table) is searched in the benchmark database and its E-values are rescaled by the ratio of database sizes (target residues,
corpus / benchmark); a TM-score counts only for hits that pass E <= 10 on that common scale.  Check: the validation and test libraries run through this same code against the training libraries come out
unflagged, as the split guarantees.  The same rule, applied the same way to all four sets, replaces the earlier flags
(sequence-only for MGnify and Rocklin, MMseqs2's default alignment-column identity, the leak audit's P / H classes for Domainome and ProteinGym).

  ProteinGym assays, Domainome domains   sequences of ``$LEAK_DATA_DIR/proteins.pkl`` (the assayed sequence), AlphaFold models of ``$LEAK_DATA_DIR/foldseek_dom_pg/queries``
  MGnify scaffolds, Rocklin designs      ``data/mgnify/mgnify_mapped.csv`` (``wt_seq``) and the structures it points to (``data/mgnify/structures``, ``structures_af2_denovo``)
  training libraries                     the 172 ``train`` libraries of ``data/splits_oct06_capped.pkl`` (``single_chain_pdbs`` of the split run, ``$HOMOLOGY_DIR``)

Output ``reference/tsuboyama_rule_flags.csv``: dataset (proteingym / domainome / mgnify / rocklin_denovo), item_id, homologous, the four values (min_evalue = MMseqs2, max_identity, min_struct_evalue,
max_tm), the four flags (f_seq_e, f_identity, f_struct_e, f_tm) and ``reason`` (letters E, I, S, T of the criteria met).  Heavy steps are cached in ``$LEAK_OUT_DIR/homology_filter`` (``--refresh``).

    python analysis/leak_audit/21_tsuboyama_rule.py [--threads 8] [--refresh]
"""
import argparse
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import FOLDSEEK_BIN, MMSEQS_BIN, OUT_DIR, REPO_ROOT, repo_data  # noqa: E402

REF = HERE.parent / 'reference'
WD = Path(OUT_DIR) / 'homology_filter'
HOM = Path(os.environ.get('HOMOLOGY_DIR', Path.home() / 'playground' / 'esm-msr-devel' / 'analysis' / 'salvaged_2026-10-16' / 'structure-homology-splits-e5ca04' / 'data' / 'tsuboyama' / 'homology'))
LEAK_DATA = Path(os.environ.get('LEAK_DATA_DIR', Path.home() / 'playground' / 'esm-msr-devel' / 'leak_audit' / 'data'))
QUERIES = LEAK_DATA / 'foldseek_dom_pg' / 'queries'
E_MAX, ID_MAX, TM_MAX = 1e-3, 0.40, 0.70
SEQ, ALIGNER = {}, None


def pdb_seq(path):
    chain = next(next(PDBParser(QUIET=True).get_structure('x', str(path)).get_models()).get_chains())
    return ''.join(seq1(r.get_resname()) for r in chain if r.id[0] == ' ' and 'CA' in r)


def items():
    """(train {name: (seq, pdb)}, bench {key: (dataset, item_id, seq, pdb)}); bench keys are '<dataset>::<item_id>'."""
    sp = pickle.load(open(Path(REPO_ROOT) / 'data' / 'splits_oct06_capped.pkl', 'rb'))
    train = {}
    for f in sp['train']:
        p = HOM / 'single_chain_pdbs' / f
        train['lib::' + f.replace('.pdb', '')] = (pdb_seq(p), p)
    corpus = dict(train)                                     # the split's search corpus: every library and external chain of the split run
    for p in sorted((HOM / 'single_chain_pdbs').glob('*.pdb')):
        k = 'lib::' + p.stem
        if k not in corpus:
            corpus[k] = (pdb_seq(p), p)
    prot = pd.read_pickle(LEAK_DATA / 'proteins.pkl').set_index('pid')['seq']
    bench = {}
    for f in sorted(QUERIES.glob('pg__*.pdb')):
        i = f.stem[4:]
        bench[f'proteingym::{i}'] = ('proteingym', i, prot[f'pg:{i}'], f)
    for f in sorted(QUERIES.glob('dom__*.pdb')):
        i = f.stem[5:]
        bench[f'domainome::{i}'] = ('domainome', i, prot[f'dom:{i}'], f)
    mg = pd.read_csv(repo_data('mgnify', 'mgnify_mapped.csv'), usecols=['code_wt', 'source', 'wt_seq', 'pdb_file']).drop_duplicates('code_wt')
    for src, ds in (('mgnify', 'mgnify'), ('rocklin_denovo', 'rocklin_denovo')):
        d = mg[mg.source == src]
        for code, seq, pf in zip(d.code_wt.astype(str), d.wt_seq, d.pdb_file):
            bench[f'{ds}::{code}'] = (ds, code, seq, Path(repo_data('mgnify', str(pf))))
    return train, corpus, bench


def write_fa(path, d):
    with open(path, 'w') as fh:
        for k, v in d.items():
            fh.write(f'>{k}\n{v}\n')


def mmseqs(train, corpus, bench, threads, refresh):
    """min MMseqs2 E-value of every benchmark protein against the training libraries, both directions."""
    out = WD / 'mmseqs_min_e_v2.pkl'
    if out.exists() and not refresh:
        return pickle.load(open(out, 'rb'))
    write_fa(WD / 'train.fa', {k: v[0] for k, v in train.items()})
    write_fa(WD / 'corpus.fa', {k: v[0] for k, v in corpus.items()})
    write_fa(WD / 'bench.fa', {k: v[2] for k, v in bench.items()})
    scale = sum(len(v[0]) for v in corpus.values()) / sum(len(v[2]) for v in bench.values())          # E-values grow with the target database: reverse hits are put on the corpus scale
    best = {}
    for name, q, t in (('fwd', 'bench.fa', 'corpus.fa'), ('rev', 'train.fa', 'bench.fa')):
        m8 = WD / f'mmseqs_{name}_v2.m8'
        if refresh or not m8.exists():
            subprocess.run([MMSEQS_BIN, 'easy-search', str(WD / q), str(WD / t), str(m8), str(WD / f'tmp_mm_{name}'), '-s', '7.5', '-e', '10.0', '--max-seqs', '10000', '--threads', str(threads),
                            '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits'], check=True)
        h = pd.read_csv(m8, sep='\t', names=['q', 't', 'f', 'aln', 'ql', 'tl', 'e', 'bits'])
        if name == 'fwd':
            h = h[h.t.isin(train)]                                   # hits to training libraries only (the benchmark protein is the query)
            h = h[h.q != h.t]
            key, e = h['q'], h['e']
        else:
            key, e = h['t'], h['e'] * scale
        for k, v in e.groupby(key).min().items():
            best[k] = min(best.get(k, np.inf), v)
    pickle.dump(best, open(out, 'wb'))
    return best


def _init():
    global ALIGNER
    a = Align.PairwiseAligner()
    a.mode, a.open_gap_score, a.extend_gap_score = 'local', -10, -1
    a.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    ALIGNER = a


def _max_identity(args):
    """Max over the training libraries of matches / shorter length of the BLOSUM62 local alignment (``homology._local_identity``)."""
    key, train_keys = args
    s1 = SEQ[key]
    best = 0.0
    for t in train_keys:
        s2 = SEQ[t]
        if not s1 or not s2:
            continue
        try:
            aln = next(iter(ALIGNER.align(s1, s2)))
        except StopIteration:
            continue
        m = sum(1 for x, y in zip(aln[0], aln[1]) if x == y and x != '-') / min(len(s1), len(s2))
        best = max(best, m)
    return key, best


def identity(train, bench, threads, refresh):
    out = WD / 'identity_max_v2.pkl'
    if out.exists() and not refresh:
        return pickle.load(open(out, 'rb'))
    for k, v in train.items():
        SEQ[k] = v[0]
    for k, v in bench.items():
        SEQ[k] = v[2]
    tk = list(train)
    with Pool(threads, initializer=_init) as pool:
        res = dict(pool.imap(_max_identity, [(k, tk) for k in bench], chunksize=200))
    pickle.dump(res, open(out, 'wb'))
    return res


def foldseek(train, corpus, bench, threads, refresh):
    """min Foldseek E-value and max TM-score (shorter-chain normalised) of every benchmark protein against the training libraries, both directions (exhaustive)."""
    out = WD / 'foldseek_best_v2.pkl'
    if out.exists() and not refresh:
        return pickle.load(open(out, 'rb'))
    dirs = {'train': WD / 'structs_train', 'corpus': WD / 'structs_corpus', 'bench': WD / 'structs_bench'}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for dname, src in (('train', {k: v[1] for k, v in train.items()}), ('corpus', {k: v[1] for k, v in corpus.items()}), ('bench', {k: v[3] for k, v in bench.items()})):
        for k, pth in src.items():
            t = dirs[dname] / (k.replace('::', '__') + '.pdb')
            if not t.exists():
                t.symlink_to(pth)
    scale = sum(len(v[0]) for v in corpus.values()) / sum(len(v[2]) for v in bench.values())
    bestE, bestT = {}, {}
    for name, q, t in (('fwd', 'bench', 'corpus'), ('rev', 'train', 'bench')):
        m8 = WD / f'foldseek_{name}_v2.m8'
        if refresh or not m8.exists():
            subprocess.run([FOLDSEEK_BIN, 'easy-search', str(dirs[q]), str(dirs[t]), str(m8), str(WD / f'tmp_fs_{name}'), '--exhaustive-search', '1', '-e', '10', '--threads', str(threads),
                            '--format-output', 'query,target,evalue,qtmscore,ttmscore'], check=True)
        for ch in pd.read_csv(m8, sep='\t', names=['q', 't', 'e', 'qtm', 'ttm'], chunksize=2_000_000):
            ch['tm'] = ch[['qtm', 'ttm']].max(axis=1)
            ch['q'], ch['t'] = ch['q'].str.replace('__', '::', n=1, regex=False), ch['t'].str.replace('__', '::', n=1, regex=False)
            if name == 'fwd':
                ch = ch[ch.t.isin(train) & (ch.q != ch.t)]
                key = ch['q']
            else:
                ch = ch.assign(e=ch['e'] * scale)
                ch = ch[ch.e <= 10.0]                               # a hit counts only if it passes the search's E <= 10 on the corpus scale
                key = ch['t']
            for k, e in ch.groupby(key)['e'].min().items():
                bestE[k] = min(bestE.get(k, np.inf), e)
            for k, v in ch.groupby(key)['tm'].max().items():
                bestT[k] = max(bestT.get(k, 0.0), v)
    pickle.dump((bestE, bestT), open(out, 'wb'))
    return bestE, bestT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    WD.mkdir(parents=True, exist_ok=True)
    train, corpus, bench = items()
    print(f'{len(train)} training libraries; benchmark proteins: ' + ', '.join(f'{d} {sum(1 for v in bench.values() if v[0] == d):,}' for d in ('proteingym', 'domainome', 'mgnify', 'rocklin_denovo')), flush=True)
    e_seq = mmseqs(train, corpus, bench, a.threads, a.refresh)
    print('MMseqs2 done', flush=True)
    ident = identity(train, bench, a.threads, a.refresh)
    print('identity done', flush=True)
    e_str, tm = foldseek(train, corpus, bench, a.threads, a.refresh)
    print('Foldseek done', flush=True)
    rows = []
    for k, (ds, iid, _, _) in bench.items():
        r = dict(dataset=ds, item_id=iid, min_evalue=e_seq.get(k, np.nan), max_identity=ident.get(k, np.nan), min_struct_evalue=e_str.get(k, np.nan), max_tm=tm.get(k, np.nan))
        r['f_seq_e'] = bool(r['min_evalue'] <= E_MAX)
        r['f_identity'] = bool(r['max_identity'] > ID_MAX)
        r['f_struct_e'] = bool(r['min_struct_evalue'] <= E_MAX)
        r['f_tm'] = bool(r['max_tm'] > TM_MAX)
        r['homologous'] = r['f_seq_e'] or r['f_identity'] or r['f_struct_e'] or r['f_tm']
        r['reason'] = ''.join(c for c, f in (('E', r['f_seq_e']), ('I', r['f_identity']), ('S', r['f_struct_e']), ('T', r['f_tm'])) if f)
        rows.append(r)
    res = pd.DataFrame(rows)[['dataset', 'item_id', 'homologous', 'min_evalue', 'max_identity', 'min_struct_evalue', 'max_tm', 'f_seq_e', 'f_identity', 'f_struct_e', 'f_tm', 'reason']]
    res.to_csv(REF / 'tsuboyama_rule_flags.csv', index=False, float_format='%.6g')
    g = res.groupby('dataset')
    print(pd.DataFrame({'items': g.size(), 'homologous': g.homologous.sum(), 'E': g.f_seq_e.sum(), 'identity': g.f_identity.sum(), 'struct_E': g.f_struct_e.sum(), 'TM': g.f_tm.sum()}).to_string())


if __name__ == '__main__':
    main()
