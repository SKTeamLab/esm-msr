"""MutateEverything inputs for the MGnify Stability scaffolds and the Rocklin designs (data/mgnify/mgnify_mapped.csv[.gz]).

Upstream MutateEverything reads <root>/mutations/<set>.csv, <root>/fasta/<id>/<id>.fasta and <root>/msa/<id>/<id>.a3m, with one
AlphaFold-style MSA per protein. The root written here is $MUTATE_EVERYTHING_DIR/data/mgnify_me (--out); rows are the table's
rows in the library wild-type numbering (mut_type_renumbered, the numbering ESM-MSR, SPURS and the harness tables use; on a
backbone-mutated scaffold the wild type is still the library sequence), pdb_id = the scaffold code.

  fasta   MGnify only: the distinct wild-type sequences in chunk_NNN.fasta files for ColabFold (colabfold_search), headers
          = the first code with that sequence, plus seq_index.csv (code, seq_id, length) mapping every code to its header.
  build   --source rocklin_denovo --single_sequence_msa   the designs have no homologs worth searching: the MSA is the query alone
          --source mgnify --msa_dir DIR                  the ColabFold a3m files of `fasta` (DIR/<seq_id>.a3m, or below it);
                                                           scaffolds whose MSA is missing are left out and counted in build_stats.json
          One mutations file per --scaffolds_per_set scaffolds: rocklin_denovo.csv, or mgnify_NNN.csv.

Rows follow build_pgym_me_data.py: mutations sorted by position, rows with two substitutions at one position dropped (ME
cannot represent them), duplicates dropped, each component single of a multi-mutant that is not itself a row added as a
`component` row (ddg 0.0) so every checkpoint has its own additive prediction. ddg = -ddG_ML only marks which entries ME
predicts (the metric is dropped), 0.0 where the label is missing. `table_uid` keeps the table's uid for joining predictions back.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402
from build_pgym_me_data import canon, me_row  # noqa: E402

COLS = ['uid', 'source', 'code', 'mut_type_renumbered', 'n_mut', 'wt_seq', 'ddG_ML']


def load(table, source):
    df = pd.read_csv(table, usecols=COLS, low_memory=False)
    df = df[df.source == source].copy()
    assert len(df), f'no rows with source {source!r} in {table}'
    assert (df.groupby('code').wt_seq.nunique() == 1).all(), 'a code with two wild-type sequences'
    return df


def write_fasta(args):
    df = load(args.table, 'mgnify')
    seqs = df.drop_duplicates('code')[['code', 'wt_seq']]
    first = seqs.drop_duplicates('wt_seq')
    seq_id = dict(zip(first.wt_seq, first.code))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.DataFrame({'code': seqs.code, 'seq_id': seqs.wt_seq.map(seq_id), 'length': seqs.wt_seq.str.len()})
    idx.to_csv(out / 'seq_index.csv', index=False)
    first = first.assign(L=first.wt_seq.str.len()).sort_values(['L', 'code'])      # length-sorted chunks: even work per chunk
    n = 0
    for k in range(0, len(first), args.chunk):
        part = first.iloc[k:k + args.chunk]
        with open(out / f'chunk_{k // args.chunk:03d}.fasta', 'w') as f:
            for code, seq in zip(part.code, part.wt_seq):
                f.write(f'>{code}\n{seq}\n')
        n += 1
    print(f'{len(seqs)} scaffolds, {len(first)} distinct sequences -> {n} fasta chunks of <= {args.chunk} in {out}')


def find_msas(msa_dir):
    return {p.stem: p for p in Path(msa_dir).rglob('*.a3m')}


def build(args):
    df = load(args.table, args.source)
    root = Path(args.out)
    for sub in ('mutations', 'fasta', 'msa'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if args.source == 'mgnify':
        assert args.msa_dir, '--source mgnify needs --msa_dir (the ColabFold output)'
        idx = pd.read_csv(Path(args.msa_dir) / 'seq_index.csv') if (Path(args.msa_dir) / 'seq_index.csv').exists() \
            else pd.read_csv(args.seq_index)
        seq_id = dict(zip(idx.code, idx.seq_id))
        msas = find_msas(args.msa_dir)
    else:
        assert args.single_sequence_msa, '--source rocklin_denovo needs --single_sequence_msa'
        seq_id, msas = None, None
    df['mut_info'] = df.mut_type_renumbered.map(canon)
    pos = df.mut_info.map(lambda s: [int(m[1:-1]) for m in s.split(':')])
    same_pos = pos.map(lambda p: len(set(p)) != len(p))
    codes = list(dict.fromkeys(df.code))
    stats = {'rows_in_table': int(len(df)), 'dropped_same_position': int(same_pos.sum()), 'scaffolds': len(codes)}
    df = df[~same_pos]
    keep_codes, no_msa = [], []
    for code in codes:
        if args.source == 'mgnify' and seq_id.get(code) not in msas:
            no_msa.append(code)
        else:
            keep_codes.append(code)
    stats.update(scaffolds_without_msa=len(no_msa), scaffolds_built=len(keep_codes))
    g = {c: d for c, d in df.groupby('code', sort=False)}
    sets, n_rows, n_comp = {}, 0, 0
    for i, code in enumerate(keep_codes):
        d = g[code].drop_duplicates('mut_info')
        seq = d.wt_seq.iloc[0]
        bad = d.mut_info.map(lambda s: any(seq[int(m[1:-1]) - 1] != m[0] for m in s.split(':')))
        assert not bad.any(), f'{code}: {int(bad.sum())} rows disagree with the wild-type sequence'
        rows = [me_row(code, seq, mi, -float(y) if pd.notna(y) else 0.0) for mi, y in zip(d.mut_info, d.ddG_ML)]
        for r, u in zip(rows, d.uid):
            r['table_uid'] = u
        have = set(d.mut_info)
        comps = sorted({m for mi in d.mut_info if ':' in mi for m in mi.split(':')} - have, key=lambda m: (int(m[1:-1]), m))
        crows = [me_row(code, seq, m, 0.0, component=True) for m in comps]
        for r in crows:
            r['table_uid'] = ''
        n_rows += len(rows); n_comp += len(crows)
        name = 'rocklin_denovo' if args.source == 'rocklin_denovo' else f'mgnify_{i // args.scaffolds_per_set:03d}'
        sets.setdefault(name, []).extend(rows + crows)
        (root / 'fasta' / code).mkdir(exist_ok=True)
        (root / 'fasta' / code / f'{code}.fasta').write_text(f'>{code}\n{seq}\n')
        (root / 'msa' / code).mkdir(exist_ok=True)
        dst = root / 'msa' / code / f'{code}.a3m'
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if args.source == 'rocklin_denovo':
            dst.write_text(f'>{code}\n{seq}\n')                       # single-sequence MSA: the query alone
        else:
            dst.symlink_to(msas[seq_id[code]].resolve())
    for name, rows in sets.items():
        out = pd.DataFrame(rows)
        for c in ('mut_fr_list', 'mut_pos_list', 'mut_to_list'):
            if c not in out.columns:
                out[c] = None
        out.to_csv(root / 'mutations' / f'{name}.csv', index=False)
    stats.update(rows_built=n_rows, component_rows=n_comp, sets=sorted(sets), source=args.source,
                 msa='single sequence' if args.source == 'rocklin_denovo' else str(args.msa_dir))
    json.dump(stats, open(root / f'build_stats_{args.source}.json', 'w'), indent=2)
    print(json.dumps(stats))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('fasta')
    a.add_argument('--table', required=True)
    a.add_argument('--out_dir', required=True)
    a.add_argument('--chunk', type=int, default=10000)
    b = sub.add_parser('build')
    b.add_argument('--table', required=True)
    b.add_argument('--source', choices=['rocklin_denovo', 'mgnify'], required=True)
    b.add_argument('--single_sequence_msa', action='store_true')
    b.add_argument('--msa_dir')
    b.add_argument('--seq_index')
    b.add_argument('--scaffolds_per_set', type=int, default=2000)
    b.add_argument('--out', default=str(paths.MUTATE_EVERYTHING_DIR / 'data' / 'mgnify_me'))
    args = ap.parse_args()
    write_fasta(args) if args.cmd == 'fasta' else build(args)
