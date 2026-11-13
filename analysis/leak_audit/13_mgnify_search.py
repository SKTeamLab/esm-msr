#!/usr/bin/env python
"""Build ``mgnify_train.fa``, ``test_queries.fa``, and ``test_vs_mgnify_s75.m8`` from ``mgnify_index.csv``.

Must run after ``06_sequence_homology.py`` (which creates ``proteins.pkl``) and before ``14_esm3dg_homology.py`` whenever the test
protein corpus changes or when regenerating from raw Zenodo 19411306 ``mgnify_training_index.csv``.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import MMSEQS_BIN, THREADS, WORK_DATA_DIR, data_file


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_csv', default=data_file('raw', 'mgnify_index.csv'))
    ap.add_argument('--proteins_pkl', default=os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
    ap.add_argument('--threads', type=int, default=int(THREADS))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    raw_dir = os.path.join(WORK_DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    train_fa = os.path.join(raw_dir, 'mgnify_train.fa')
    queries_fa = os.path.join(raw_dir, 'test_queries.fa')
    out_m8 = os.path.join(raw_dir, 'test_vs_mgnify_s75.m8')
    stamp_file = out_m8 + '.sha1'

    if (a.force or not os.path.exists(train_fa)) and os.path.exists(a.index_csv):
        hdr = pd.read_csv(a.index_csv, nrows=0).columns
        seq_col = 'seq' if 'seq' in hdr else 'sequence'
        idx = pd.read_csv(a.index_csv, usecols=['name', 'split', seq_col] if 'name' in hdr else ['split', seq_col])
        trn = idx[idx['split'] == 'train']
        with open(train_fa, 'w') as fh:
            for i, row in enumerate(trn.itertuples(index=False)):
                nm = getattr(row, 'name', f'mgnify_{i}')
                sq = getattr(row, seq_col)
                if isinstance(sq, str) and sq:
                    fh.write(f'>{nm}\n{sq}\n')
        print(f'Wrote {len(trn)} MGnify train sequences -> {train_fa}')

    pdf = pd.read_pickle(a.proteins_pkl)
    tq = pdf[pdf['source'].str.startswith('test:')].copy()
    with open(queries_fa, 'w') as fh:
        for pid, seq in zip(tq['pid'], tq['seq']):
            fh.write(f'>{pid}\n{seq}\n')
    q_hash = sha1_file(queries_fa)

    prev_hash = open(stamp_file).read().strip() if os.path.exists(stamp_file) else ''
    if a.force or not os.path.exists(out_m8) or prev_hash != q_hash:
        if not os.path.exists(train_fa):
            raise FileNotFoundError(f'Missing {train_fa} (download Zenodo 19411306 mgnify_training_index.csv to {a.index_csv})')
        tmp_dir = os.path.join(WORK_DATA_DIR, 'mmseqs_mgnify', 'tmp')
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        cmd = [
            MMSEQS_BIN, 'easy-search', queries_fa, train_fa, out_m8, tmp_dir,
            '-s', '7.5', '-e', '10', '--max-seqs', '500', '--threads', str(a.threads),
            '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits',
        ]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        with open(stamp_file, 'w') as fh:
            fh.write(q_hash + '\n')
        print(f'Wrote {out_m8}')
    else:
        print(f'Up to date: {out_m8} (queries SHA-1={q_hash[:12]})')


if __name__ == '__main__':
    main()
