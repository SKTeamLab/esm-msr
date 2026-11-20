#!/usr/bin/env python
"""Maximum sequence identity and maximum Foldseek TM-score matrices across split and held-out benchmark sets.

Writes ``analysis/reference/split_set_overlap_{identity,tm}.csv`` (plotted by ``pipelines/p10_splits_and_leakage.py``).
Completely decoupled from downstream ``results/`` directories (uses ``msr_analysis.heldout.flags()`` directly).

Usage: ``python -m audits.homology_and_splits.active_builders.set_overlap_matrix [--threads 8] [--refresh]``
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import FOLDSEEK_BIN, MMSEQS_BIN, OUT_DIR, WORK_DATA_DIR, pdb_ca_sequence, repo_data
from msr_analysis import heldout
from msr_analysis.paths import DATA_DIR, REFERENCE, SALVAGED_HOMOLOGY_CSV, atomic_pickle_dump, atomic_to_csv

WD = Path(OUT_DIR) / 'set_matrix'
HOM = Path(os.environ.get('HOMOLOGY_DIR', SALVAGED_HOMOLOGY_CSV.parent.parent / 'homology')).expanduser()
QUERIES = Path(WORK_DATA_DIR) / 'foldseek_dom_pg' / 'queries'
THERMO = ['ssym', 'ptmul', 's461', 'k2369', 'q3421', 's571', 's783', 's2648', 's8754']
E_SIG = 1e-3
NEW = ['ProteinGym', 'Domainome', 'MGnify', 'Rocklin designs']
SETS = ['train', 'val', 'test'] + THERMO + NEW
PREFIX = {'lib': 'lib', 'th': 'th', 'ProteinGym': 'pg', 'Domainome': 'dom', 'MGnify': 'mg', 'Rocklin designs': 'rk'}


def members():
    """Return ``(sets, seqs, pdbs)`` without reading any downstream ``results/`` file."""
    flags = heldout.flags()
    bad = {d: set(g.loc[g.homologous, 'item_id'].astype(str)) for d, g in flags.groupby('dataset')}
    pg_all_ids = sorted(flags.loc[flags['dataset'] == 'proteingym', 'item_id'].astype(str).unique())

    sets, seqs, pdbs = defaultdict(list), {}, {}
    with open(DATA_DIR / 'splits_oct06_capped.pkl', 'rb') as fh:
        sp = pickle.load(fh)
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
    for i in pg_all_ids:
        if i not in bad.get('proteingym', set()) and (QUERIES / f'pg__{i}.pdb').exists():
            sets['ProteinGym'].append('pg::' + i)
            pdbs['pg::' + i] = QUERIES / f'pg__{i}.pdb'
    for f in sorted(QUERIES.glob('dom__*.pdb')):
        i = f.stem[5:]
        if i not in bad.get('domainome', set()):
            sets['Domainome'].append('dom::' + i)
            pdbs['dom::' + i] = f
    mg = pd.read_csv(repo_data('mgnify', 'mgnify_mapped.csv'), usecols=['code_wt', 'source', 'wt_seq', 'pdb_file']).drop_duplicates('code_wt')
    for name, src, flag in (('MGnify', 'mgnify', 'mgnify'), ('Rocklin designs', 'rocklin_denovo', 'rocklin_denovo')):
        d = mg[(mg.source == src) & ~mg.code_wt.astype(str).isin(bad.get(flag, set()))]
        for code, seq, pf in zip(d.code_wt.astype(str), d.wt_seq, d.pdb_file):
            k = f'{PREFIX[name]}::{code}'
            sets[name].append(k)
            seqs[k] = seq
            pdbs[k] = Path(repo_data('mgnify', str(pf)))
    for s in SETS:
        sets[s] = sorted(set(sets[s]))
    for k in {k for s in SETS for k in sets[s]}:
        if k not in seqs:
            seqs[k] = pdb_ca_sequence(pdbs[k])
    return sets, seqs, pdbs


def sequence_matrix(sets, seqs, threads: int, refresh: bool):
    cache = WD / 'identity_mmseqs_shorter.pkl'
    if cache.exists() and not refresh:
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    WD.mkdir(parents=True, exist_ok=True)
    key_sets = defaultdict(list)
    for s_ in SETS:
        for k in sets[s_]:
            key_sets[k].append(s_)
    non_mg = [k for k in key_sets if not k.startswith('mg::')]
    for name, keys in (('targets', non_mg), ('queries', list(key_sets))):
        with open(WD / f'{name}.fa', 'w', encoding='utf-8') as fh:
            for k in keys:
                fh.write(f'>{k}\n{seqs[k]}\n')
    m8 = WD / 'seq_hits_shorter.m8'
    if refresh or not m8.exists():
        subprocess.run(
            [MMSEQS_BIN, 'easy-search', str(WD / 'queries.fa'), str(WD / 'targets.fa'), str(m8), str(WD / 'tmp_mm'),
             '-s', '7.5', '-e', '10.0', '--max-seqs', '10000', '--seq-id-mode', '1', '--alignment-mode', '3',
             '--threads', str(threads), '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits'],
            check=True,
        )
    best = {'raw': defaultdict(float), 'sig': defaultdict(float)}
    for ch in pd.read_csv(m8, sep='\t', names=['q', 't', 'f', 'aln', 'ql', 'tl', 'e', 'bits'], chunksize=2_000_000):
        ch = ch[ch.q != ch.t]
        ch['qs'], ch['ts'] = ch.q.map(key_sets), ch.t.map(key_sets)
        ch = ch.explode('qs').explode('ts')
        ch = ch[ch.qs != ch.ts]
        for kind, sub in (('raw', ch), ('sig', ch[ch.e <= E_SIG])):
            for (a_, b_), v in sub.groupby(['qs', 'ts']).f.max().items():
                best[kind][(a_, b_)] = max(best[kind][(a_, b_)], float(v))
                best[kind][(b_, a_)] = max(best[kind][(b_, a_)], float(v))
    out = {k: dict(v) for k, v in best.items()}
    atomic_pickle_dump(out, cache)
    return out


def _run_foldseek(qdir: Path, tdir: Path, out: Path, threads: int, exhaustive: bool) -> None:
    cmd = [
        FOLDSEEK_BIN, 'easy-search', str(qdir), str(tdir), str(out), str(WD / 'tmp_fs'),
        '-e', '10', '--threads', str(threads), '--format-output', 'query,target,evalue,qtmscore,ttmscore',
    ]
    cmd += ['--exhaustive-search', '1'] if exhaustive else ['--max-seqs', '3000']
    subprocess.run(cmd, check=True)


def tm_matrix(sets, pdbs, threads: int, refresh: bool):
    cache = WD / 'tm_pairs.pkl'
    if cache.exists() and not refresh:
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
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
        _run_foldseek(dirs['nonmg'], dirs['nonmg'], h_non, threads, True)
    if refresh or not h_mg.exists():
        _run_foldseek(dirs['mg'], dirs['nonmg'], h_mg, threads, False)
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
                best[(a, b)] = max(best[(a, b)], float(v))
                best[(b, a)] = max(best[(b, a)], float(v))
    out = dict(best)
    atomic_pickle_dump(out, cache)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args(argv)
    WD.mkdir(parents=True, exist_ok=True)
    sets, seqs, pdbs = members()
    print({s_: len(sets[s_]) for s_ in SETS})
    ident = sequence_matrix(sets, seqs, a.threads, a.refresh)
    for kind, best in (('identity', ident['raw']), ('tm', tm_matrix(sets, pdbs, a.threads, a.refresh))):
        m = pd.DataFrame(np.nan, index=SETS, columns=SETS)
        for (r, c), v in best.items():
            m.loc[r, c] = v
        atomic_to_csv(m, REFERENCE / f'split_set_overlap_{kind}.csv', float_format='%.4f')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
