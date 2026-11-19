#!/usr/bin/env python
"""Maximum sequence identity and maximum TM-score between the protein sets of the split and of the held-out benchmarks: the two matrices of the split-construction supplementary figure.

Sets (rows = columns): the Homology-Restricted MegaScale libraries ``train`` / ``val`` / ``test`` (``data/splits_oct06_capped.pkl``); the nine thermostability sets (ssym, ptmul, s461, k2369, q3421, s571, s783,
s2648, s8754; the proteins that survived the split's external quarantine); and the held-out versions of the four large benchmarks used in the summary figure (``reference/tsuboyama_rule_flags.csv``): ProteinGym assays,
Domainome domains, MGnify scaffolds, Rocklin designs that are not homologous to the training set under the split's own rule.

``identity``   max over all pairs of two sets of the MMseqs2 identity normalised by the length of the SHORTER sequence (``--seq-id-mode 1``: matches / shorter sequence, the same whole-protein normalisation as
               the split's own identity, ``preprocessing/homology.py: pairwise_identity`` = BLOSUM62 local-alignment matches / shorter length; on the 23,485 library pairs the split's pair table holds, the two agree
               at r = 0.98).  Search as in ``homology.mmseqs_all_vs_all`` (``mmseqs easy-search -s 7.5 -e 10.0 --max-seqs 10000``, ``--alignment-mode 3``); every protein of every set is a query against all
               non-MGnify sequences, MGnify scaffolds are queries only, so every pair of sets is covered; the larger of the two directions is kept.  MMseqs2's default ``fident`` (identity over the aligned columns,
               ``--seq-id-mode 0``) is NOT used: it is 0.73 for an 11-14 residue local match and says nothing about the whole protein.  The split itself applies E <= 1e-3 or this identity > 0.40.
``tm``         max over all pairs of Foldseek's TM-score normalised by the shorter chain (max of the query and target TM-scores) over every structure: the libraries, thermostability chains, ProteinGym and
               Domainome structures (exhaustive all-vs-all, ``--exhaustive-search 1 -e 10``, as in ``07_structure_homology.py``) and the Rocklin designs (AF2 models, ``data/mgnify/structures_af2_denovo``) and MGnify
               scaffolds (``data/mgnify/structures``, the structures ESM-MSR inference reads) searched against all non-MGnify structures with Foldseek's sensitive search (``--max-seqs 3000``; an exhaustive search
               of 158 k scaffolds against 2.4 k structures is hours).

Heavy steps are cached in ``$LEAK_OUT_DIR/set_matrix`` (``--refresh`` redoes them).  Output ``reference/split_set_overlap_{identity,tm}.csv``; ``analysis/20b_split_overlap_heatmaps.py`` draws them.

    python analysis/leak_audit/22_set_overlap_matrix.py [--threads 8] [--refresh]
"""
import argparse
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import FOLDSEEK_BIN, MMSEQS_BIN, OUT_DIR, REPO_ROOT, repo_data  # noqa: E402

REF = HERE.parent / 'reference'
WD = Path(OUT_DIR) / 'set_matrix'
HOM = Path(os.environ.get('HOMOLOGY_DIR', Path.home() / 'playground' / 'esm-msr-devel' / 'analysis' / 'salvaged_2026-10-16' / 'structure-homology-splits-e5ca04' / 'data' / 'tsuboyama' / 'homology'))
QUERIES = Path(os.environ.get('LEAK_ROOT', Path.home() / 'playground' / 'esm-msr-devel')) / 'leak_audit' / 'data' / 'foldseek_dom_pg' / 'queries'
THERMO = ['ssym', 'ptmul', 's461', 'k2369', 'q3421', 's571', 's783', 's2648', 's8754']
E_SIG = 1e-3
NEW = ['ProteinGym', 'Domainome', 'MGnify', 'Rocklin designs']
SETS = ['train', 'val', 'test'] + THERMO + NEW
PREFIX = {'lib': 'lib', 'th': 'th', 'ProteinGym': 'pg', 'Domainome': 'dom', 'MGnify': 'mg', 'Rocklin designs': 'rk'}
ALIGNER = None


def pdb_seq(path):
    chain = next(next(PDBParser(QUIET=True).get_structure('x', str(path)).get_models()).get_chains())
    return ''.join(seq1(r.get_resname()) for r in chain if r.id[0] == ' ' and 'CA' in r)


def members():
    """{set: [protein key]}, {key: sequence}, {key: pdb path or None}; keys are '<prefix>::<name>'."""
    flags = pd.read_csv(REF / 'tsuboyama_rule_flags.csv')
    bad = {d: set(g.loc[g.homologous, 'item_id'].astype(str)) for d, g in flags.groupby('dataset')}
    sets, seqs, pdbs = defaultdict(list), {}, {}
    sp = pickle.load(open(Path(REPO_ROOT) / 'data' / 'splits_oct06_capped.pkl', 'rb'))
    for s in ('train', 'val', 'test'):
        for f in sp[s]:
            k = 'lib::' + f.replace('.pdb', '')
            sets[s].append(k)
            pdbs[k] = HOM / 'single_chain_pdbs' / f
    surviving = {f.replace('.pdb', '') for f in sp['thermostability']}
    for d in THERMO:
        df = pd.read_csv(repo_data('preprocessed', f'{d}_mapped.csv'), usecols=['code', 'chain'])
        for code, chain in df[['code', 'chain']].drop_duplicates().itertuples(index=False):
            name = f'test_{code}_{chain}'
            if name in surviving:
                k = 'th::' + name
                sets[d].append(k)
                pdbs[k] = HOM / 'single_chain_pdbs' / f'{name}.pdb'
    pg_ids = pd.read_csv(HERE.parent / 'proteingym' / 'p0_sigma' / 'results' / 'per_assay_rho.csv', index_col=0).index.astype(str)
    for i in pg_ids:
        if i not in bad['proteingym'] and (QUERIES / f'pg__{i}.pdb').exists():
            sets['ProteinGym'].append('pg::' + i)
            pdbs['pg::' + i] = QUERIES / f'pg__{i}.pdb'
    for f in sorted(QUERIES.glob('dom__*.pdb')):
        i = f.stem[5:]
        if i not in bad['domainome']:
            sets['Domainome'].append('dom::' + i)
            pdbs['dom::' + i] = f
    mg = pd.read_csv(repo_data('mgnify', 'mgnify_mapped.csv'), usecols=['code_wt', 'source', 'wt_seq', 'pdb_file']).drop_duplicates('code_wt')
    for name, src, flag in (('MGnify', 'mgnify', 'mgnify'), ('Rocklin designs', 'rocklin_denovo', 'rocklin_denovo')):
        d = mg[(mg.source == src) & ~mg.code_wt.astype(str).isin(bad[flag])]
        for code, seq, pf in zip(d.code_wt.astype(str), d.wt_seq, d.pdb_file):
            k = f'{PREFIX[name]}::{code}'
            sets[name].append(k)
            seqs[k] = seq
            pdbs[k] = Path(repo_data('mgnify', str(pf)))
    for s in SETS:
        sets[s] = sorted(set(sets[s]))
    for k in {k for s in SETS for k in sets[s]}:
        if k not in seqs:
            seqs[k] = pdb_seq(pdbs[k])
    return sets, seqs, pdbs


def sequence_matrix(sets, seqs, threads, refresh):
    """{(set a, set b): max MMseqs2 identity} over the hits (E <= 10) of every protein against all non-MGnify sequences."""
    cache = WD / 'identity_mmseqs_shorter.pkl'
    if cache.exists() and not refresh:
        return pickle.load(open(cache, 'rb'))
    WD.mkdir(parents=True, exist_ok=True)
    key_sets = defaultdict(list)
    for s_ in SETS:
        for k in sets[s_]:
            key_sets[k].append(s_)
    non_mg = [k for k in key_sets if not k.startswith('mg::')]
    for name, keys in (('targets', non_mg), ('queries', list(key_sets))):
        with open(WD / f'{name}.fa', 'w') as fh:
            for k in keys:
                fh.write(f'>{k}\n{seqs[k]}\n')
    m8 = WD / 'seq_hits_shorter.m8'
    if refresh or not m8.exists():
        subprocess.run([MMSEQS_BIN, 'easy-search', str(WD / 'queries.fa'), str(WD / 'targets.fa'), str(m8), str(WD / 'tmp_mm'), '-s', '7.5', '-e', '10.0', '--max-seqs', '10000', '--seq-id-mode', '1', '--alignment-mode', '3', '--threads', str(threads),
                        '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits'], check=True)
    best = {'raw': defaultdict(float), 'sig': defaultdict(float)}
    for ch in pd.read_csv(m8, sep='\t', names=['q', 't', 'f', 'aln', 'ql', 'tl', 'e', 'bits'], chunksize=2_000_000):
        ch = ch[ch.q != ch.t]
        ch['qs'], ch['ts'] = ch.q.map(key_sets), ch.t.map(key_sets)
        ch = ch.explode('qs').explode('ts')
        ch = ch[ch.qs != ch.ts]
        for kind, sub in (('raw', ch), ('sig', ch[ch.e <= E_SIG])):
            for (a_, b_), v in sub.groupby(['qs', 'ts']).f.max().items():
                best[kind][(a_, b_)] = max(best[kind][(a_, b_)], v)
                best[kind][(b_, a_)] = max(best[kind][(b_, a_)], v)
    out = {k: dict(v) for k, v in best.items()}
    pickle.dump(out, open(cache, 'wb'))
    return out


def foldseek(qdir, tdir, out, threads, exhaustive):
    cmd = [FOLDSEEK_BIN, 'easy-search', str(qdir), str(tdir), str(out), str(WD / 'tmp_fs'), '-e', '10', '--threads', str(threads), '--format-output', 'query,target,evalue,qtmscore,ttmscore']
    cmd += ['--exhaustive-search', '1'] if exhaustive else ['--max-seqs', '3000']
    subprocess.run(cmd, check=True)


def tm_matrix(sets, pdbs, threads, refresh):
    """{(set a, set b): max TM-score} from Foldseek over every structure of the sets."""
    cache = WD / 'tm_pairs.pkl'
    if cache.exists() and not refresh:
        return pickle.load(open(cache, 'rb'))
    ks = defaultdict(list)
    for s_ in SETS:
        for k in sets[s_]:
            ks[k].append(s_)
    dirs = {'nonmg': WD / 'structs_nonmg', 'mg': WD / 'structs_mg'}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for k in ks:
        if pdbs[k] is None:
            continue
        tgt = dirs['mg' if k.startswith('mg::') else 'nonmg'] / (k.replace('::', '__') + '.pdb')
        if not tgt.exists():
            tgt.symlink_to(pdbs[k])
    h_non, h_mg = WD / 'tm_hits_nonmg.m8', WD / 'tm_hits_mg.m8'
    if refresh or not h_non.exists():
        foldseek(dirs['nonmg'], dirs['nonmg'], h_non, threads, True)
    if refresh or not h_mg.exists():
        foldseek(dirs['mg'], dirs['nonmg'], h_mg, threads, False)
    best = defaultdict(float)
    for f in (h_non, h_mg):
        for ch in pd.read_csv(f, sep='\t', names=['q', 't', 'e', 'qtm', 'ttm'], chunksize=2_000_000):
            ch['q'] = ch.q.str.replace('.pdb', '', regex=False).str.replace('__', '::', n=1, regex=False)
            ch['t'] = ch.t.str.replace('.pdb', '', regex=False).str.replace('__', '::', n=1, regex=False)
            ch['tm'] = ch[['qtm', 'ttm']].max(axis=1)
            ch = ch[ch.q != ch.t]
            ch['qs'], ch['ts'] = ch.q.map(ks), ch.t.map(ks)
            ch = ch.explode('qs').explode('ts')
            ch = ch[ch.qs != ch.ts]
            for (a, b), v in ch.groupby(['qs', 'ts']).tm.max().items():
                best[(a, b)] = max(best[(a, b)], v)
                best[(b, a)] = max(best[(b, a)], v)
    pickle.dump(dict(best), open(cache, 'wb'))
    return dict(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    WD.mkdir(parents=True, exist_ok=True)
    sets, seqs, pdbs = members()
    print({s_: len(sets[s_]) for s_ in SETS})
    out = {}
    ident = sequence_matrix(sets, seqs, a.threads, a.refresh)
    for kind, best in (('identity', ident['raw']), ('tm', tm_matrix(sets, pdbs, a.threads, a.refresh))):
        m = pd.DataFrame(np.nan, index=SETS, columns=SETS)
        for (r, c), v in best.items():
            m.loc[r, c] = v
        m.to_csv(REF / f'split_set_overlap_{kind}.csv', float_format='%.4f')
        out[kind] = m
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        for k, m in out.items():
            print(k)
            print(m.round(2).to_string())


if __name__ == '__main__':
    main()
