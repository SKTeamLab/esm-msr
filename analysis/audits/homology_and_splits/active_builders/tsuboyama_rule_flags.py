#!/usr/bin/env python
"""Canonical four-part homology filter applied across ProteinGym, Domainome, MGnify, and Rocklin designs.

Applies the exact edge rule of the Homology-Restricted MegaScale split (`preprocessing/homology.py`):
  1. MMseqs2 E-value <= 1e-3
  2. BLOSUM62 local-alignment identity > 0.40 (matches / length of SHORTER sequence)
  3. Foldseek E-value <= 1e-3
  4. Foldseek TM-score > 0.70 (max of query and target TM-score)

Writes ``analysis/reference/tsuboyama_rule_flags.csv`` (served downstream by ``msr_analysis.heldout``).
Heavy search intermediates are cached in ``$LEAK_OUT_DIR/homology_filter`` (``--refresh`` recomputes).

Usage: ``python -m audits.homology_and_splits.active_builders.tsuboyama_rule_flags [--threads 8] [--refresh]``
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

from ..lib import FOLDSEEK_BIN, MMSEQS_BIN, OUT_DIR, WORK_DATA_DIR, pdb_ca_sequence, repo_data
from msr_analysis.paths import DATA_DIR, REFERENCE, SALVAGED_HOMOLOGY_CSV, atomic_pickle_dump, atomic_to_csv

WD = Path(OUT_DIR) / 'homology_filter'
HOM = Path(os.environ.get('HOMOLOGY_DIR', SALVAGED_HOMOLOGY_CSV.parent.parent / 'homology')).expanduser()
LEAK_DATA = Path(WORK_DATA_DIR)
QUERIES = LEAK_DATA / 'foldseek_dom_pg' / 'queries'
E_MAX, ID_MAX, TM_MAX = 1e-3, 0.40, 0.70
SEQ: dict[str, str] = {}
ALIGNER = None


def items():
    """Return ``(train, corpus, bench)`` sequence/PDB dictionaries."""
    with open(DATA_DIR / 'splits_oct06_capped.pkl', 'rb') as fh:
        sp = pickle.load(fh)
    train = {}
    for f in sp['train']:
        p = HOM / 'single_chain_pdbs' / f
        train['lib::' + f.replace('.pdb', '')] = (pdb_ca_sequence(p), p)
    corpus = dict(train)
    for p in sorted((HOM / 'single_chain_pdbs').glob('*.pdb')):
        k = 'lib::' + p.stem
        if k not in corpus:
            corpus[k] = (pdb_ca_sequence(p), p)
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


def write_fa(path: Path, d: dict[str, str]) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        for k, v in d.items():
            fh.write(f'>{k}\n{v}\n')


def mmseqs(train, corpus, bench, threads: int, refresh: bool) -> dict[str, float]:
    out = WD / 'mmseqs_min_e_v2.pkl'
    if out.exists() and not refresh:
        with open(out, 'rb') as fh:
            return pickle.load(fh)
    write_fa(WD / 'train.fa', {k: v[0] for k, v in train.items()})
    write_fa(WD / 'corpus.fa', {k: v[0] for k, v in corpus.items()})
    write_fa(WD / 'bench.fa', {k: v[2] for k, v in bench.items()})
    scale = sum(len(v[0]) for v in corpus.values()) / sum(len(v[2]) for v in bench.values())
    best: dict[str, float] = {}
    for name, q, t in (('fwd', 'bench.fa', 'corpus.fa'), ('rev', 'train.fa', 'bench.fa')):
        m8 = WD / f'mmseqs_{name}_v2.m8'
        if refresh or not m8.exists():
            subprocess.run(
                [MMSEQS_BIN, 'easy-search', str(WD / q), str(WD / t), str(m8), str(WD / f'tmp_mm_{name}'),
                 '-s', '7.5', '-e', '10.0', '--max-seqs', '10000', '--threads', str(threads),
                 '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits'],
                check=True,
            )
        h = pd.read_csv(m8, sep='\t', names=['q', 't', 'f', 'aln', 'ql', 'tl', 'e', 'bits'])
        if name == 'fwd':
            h = h[h.t.isin(train) & (h.q != h.t)]
            key, e = h['q'], h['e']
        else:
            key, e = h['t'], h['e'] * scale
        for k, v in e.groupby(key).min().items():
            best[k] = min(best.get(k, np.inf), float(v))
    atomic_pickle_dump(best, out)
    return best


def _init() -> None:
    global ALIGNER
    a = Align.PairwiseAligner()
    a.mode, a.open_gap_score, a.extend_gap_score = 'local', -10, -1
    a.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    ALIGNER = a


def _max_identity(args):
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


def identity(train, bench, threads: int, refresh: bool) -> dict[str, float]:
    out = WD / 'identity_max_v2.pkl'
    if out.exists() and not refresh:
        with open(out, 'rb') as fh:
            return pickle.load(fh)
    for k, v in train.items():
        SEQ[k] = v[0]
    for k, v in bench.items():
        SEQ[k] = v[2]
    tk = list(train)
    with Pool(threads, initializer=_init) as pool:
        res = dict(pool.imap(_max_identity, [(k, tk) for k in bench], chunksize=200))
    atomic_pickle_dump(res, out)
    return res


def foldseek(train, corpus, bench, threads: int, refresh: bool):
    out = WD / 'foldseek_best_v2.pkl'
    if out.exists() and not refresh:
        with open(out, 'rb') as fh:
            return pickle.load(fh)
    dirs = {'train': WD / 'structs_train', 'corpus': WD / 'structs_corpus', 'bench': WD / 'structs_bench'}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for dname, src in (
        ('train', {k: v[1] for k, v in train.items()}),
        ('corpus', {k: v[1] for k, v in corpus.items()}),
        ('bench', {k: v[3] for k, v in bench.items()}),
    ):
        for k, pth in src.items():
            t = dirs[dname] / (k.replace('::', '__') + '.pdb')
            if not t.exists():
                t.symlink_to(pth)
    scale = sum(len(v[0]) for v in corpus.values()) / sum(len(v[2]) for v in bench.values())
    bestE: dict[str, float] = {}
    bestT: dict[str, float] = {}
    for name, q, t in (('fwd', 'bench', 'corpus'), ('rev', 'train', 'bench')):
        m8 = WD / f'foldseek_{name}_v2.m8'
        if refresh or not m8.exists():
            subprocess.run(
                [FOLDSEEK_BIN, 'easy-search', str(dirs[q]), str(dirs[t]), str(m8), str(WD / f'tmp_fs_{name}'),
                 '--exhaustive-search', '1', '-e', '10', '--threads', str(threads),
                 '--format-output', 'query,target,evalue,qtmscore,ttmscore'],
                check=True,
            )
        for ch in pd.read_csv(m8, sep='\t', names=['q', 't', 'e', 'qtm', 'ttm'], chunksize=2_000_000):
            ch['tm'] = ch[['qtm', 'ttm']].max(axis=1)
            ch['q'], ch['t'] = ch['q'].str.replace('__', '::', n=1, regex=False), ch['t'].str.replace('__', '::', n=1, regex=False)
            if name == 'fwd':
                ch = ch[ch.t.isin(train) & (ch.q != ch.t)]
                key = ch['q']
            else:
                ch = ch.assign(e=ch['e'] * scale)
                ch = ch[ch.e <= 10.0]
                key = ch['t']
            for k, e in ch.groupby(key)['e'].min().items():
                bestE[k] = min(bestE.get(k, np.inf), float(e))
            for k, v in ch.groupby(key)['tm'].max().items():
                bestT[k] = max(bestT.get(k, 0.0), float(v))
    atomic_pickle_dump((bestE, bestT), out)
    return bestE, bestT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args(argv)
    WD.mkdir(parents=True, exist_ok=True)
    train, corpus, bench = items()
    print(f'train libraries: {len(train)}; search corpus: {len(corpus)}; benchmark items: {len(bench)}')
    me = mmseqs(train, corpus, bench, a.threads, a.refresh)
    mi = identity(train, bench, a.threads, a.refresh)
    fe, ft = foldseek(train, corpus, bench, a.threads, a.refresh)
    rows = []
    for k, (ds, iid, _, _) in sorted(bench.items()):
        se, idv, ste, tm = me.get(k, np.inf), mi.get(k, 0.0), fe.get(k, np.inf), ft.get(k, 0.0)
        f1, f2, f3, f4 = se <= E_MAX, idv > ID_MAX, ste <= E_MAX, tm > TM_MAX
        why = ''.join(c for c, fl in zip('EIST', (f1, f2, f3, f4)) if fl)
        rows.append(dict(
            dataset=ds, item_id=iid, homologous=bool(why), reason=why,
            min_evalue=se if np.isfinite(se) else np.nan, max_identity=round(idv, 4),
            min_struct_evalue=ste if np.isfinite(ste) else np.nan, max_tm=round(tm, 4),
            f_seq_e=f1, f_identity=f2, f_struct_e=f3, f_tm=f4,
        ))
    df = pd.DataFrame(rows)
    atomic_to_csv(df, REFERENCE / 'tsuboyama_rule_flags.csv', index=False)
    for ds, g in df.groupby('dataset'):
        print(
            f'{ds:16s}  total={len(g):6d}  homologous={int(g.homologous.sum()):5d}  kept={int((~g.homologous).sum()):6d}'
            f'  (E={int(g.f_seq_e.sum())} I={int(g.f_identity.sum())} S={int(g.f_struct_e.sum())} T={int(g.f_tm.sum())})'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
