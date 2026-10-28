"""Write the ProteinGym substitution benchmark in upstream MutateEverything's layout (data/pgym_oct06).

Rows are the pgym_inputs tables written by preprocessing/pgym_preprocess.py -- the rows ESM-MSR, ProteinMPNN, SPURS and
ThermoMPNN score -- in AF2-structure numbering (mut_type_renumbered). pdb_id is the structure's name (PIN1_HUMAN for
PIN1_HUMAN.pdb), so DMS on one protein share one fasta and one MSA, and the wild-type sequence is the structure's chain
A, as in build_dms_me_data.py. ddg = -DMS_score: MutateEverything only uses it to mark which entries to predict
(predict_dms.py drops upstream's metric, the one place its sign matters).

Multi-mutant DMS also get each component single that is not itself a row ('component' rows, ddg 0.0), so every
checkpoint has its own additive prediction. Rows with two substitutions at one position cannot be represented and are
dropped (counted in build_stats.json); upstream asserts pos1 < pos2, so mutations are sorted by position.

MSAs are reused only on an exact query-sequence match (prepare/index_msas.py). Structures without one are written to
--missing_fasta for ColabFold; re-run after the MSAs exist and pass their directory to --msa_roots.

Usage: python build_pgym_me_data.py --msa_index INDEX.json [--missing_fasta FASTA]
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths
import paths  # noqa: E402

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
       'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}
PGYM = str(paths.PGYM_INPUTS)
STRUCTS = str(paths.PGYM_STRUCTURES)
OUT = str(paths.MUTATE_EVERYTHING_DIR / 'data' / 'pgym_oct06')


def chain_seq(path, chain='A'):
    seq, seen = [], set()
    for line in open(path):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
            rid = line[22:27]
            if rid not in seen:
                seen.add(rid); seq.append(AA3.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def canon(s):
    return ':'.join(sorted(s.split(':'), key=lambda m: int(m[1:-1])))


def me_row(pdb_id, seq, mi, ddg, component=False):
    ms = mi.split(':')
    row = {'pdb_id': pdb_id, 'mut_info': mi, 'ddg': ddg, 'wt_seq': seq, 'uid': f'{pdb_id}_{mi}',
           'component': component}
    for i, m in enumerate(ms, 1):
        row.update({f'fr{i}': m[0], f'pos{i}': int(m[1:-1]), f'to{i}': m[-1]})
    if len(ms) > 2:   # upstream's arbitrary-order path reads these three with eval()
        row['mut_fr_list'] = str([m[0] for m in ms])
        row['mut_pos_list'] = str([int(m[1:-1]) for m in ms])
        row['mut_to_list'] = str([m[-1] for m in ms])
    return row


def main(args):
    for sub in ['mutations', 'fasta', 'msa']:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    index = json.load(open(args.msa_index))
    manifest = pd.read_csv(os.path.join(PGYM, 'manifest.csv'))
    seqs, stats = {}, {}
    for did, pdb_file in zip(manifest.DMS_id, manifest.pdb_file):
        pdb_id = os.path.splitext(pdb_file)[0]
        if pdb_id not in seqs:
            seqs[pdb_id] = chain_seq(os.path.join(STRUCTS, pdb_file))
        seq = seqs[pdb_id]
        df = pd.read_csv(os.path.join(PGYM, f'{did}.csv'), usecols=['mut_type_renumbered', 'DMS_score'])
        df['mut_info'] = df.mut_type_renumbered.map(canon)
        pos = df.mut_info.map(lambda s: [int(m[1:-1]) for m in s.split(':')])
        same_pos = pos.map(lambda p: len(set(p)) != len(p))
        bad_wt = df.mut_info.map(lambda s: any(seq[int(m[1:-1]) - 1] != m[0] for m in s.split(':')))
        assert not bad_wt.any(), f'{did}: {int(bad_wt.sum())} rows disagree with the {pdb_id} chain A sequence'
        keep = df[~same_pos].drop_duplicates('mut_info')
        rows = [me_row(pdb_id, seq, mi, -float(s)) for mi, s in zip(keep.mut_info, keep.DMS_score)]
        have = set(keep.mut_info)
        comps = sorted({m for mi in keep.mut_info if ':' in mi for m in mi.split(':')} - have,
                       key=lambda m: (int(m[1:-1]), m))
        rows += [me_row(pdb_id, seq, m, 0.0, component=True) for m in comps]
        out = pd.DataFrame(rows)
        for c in ['mut_fr_list', 'mut_pos_list', 'mut_to_list']:
            if c not in out.columns:
                out[c] = None
        out.to_csv(os.path.join(OUT, 'mutations', f'{did}.csv'), index=False)
        n = keep.mut_info.str.count(':') + 1
        stats[did] = {'pdb_id': pdb_id, 'len': len(seq), 'rows': len(keep), 'component_rows': len(comps),
                      'dropped_same_position': int(same_pos.sum()), 'duplicates': int((~same_pos).sum() - len(keep)),
                      'order': {int(k): int(v) for k, v in n.value_counts().sort_index().items()},
                      'msa': seq in index}
        print(f'{did[:44]:46s} {pdb_id:14s} L={len(seq):5d} rows {len(keep):7d} +{len(comps):5d} comp  '
              f'dropped {int(same_pos.sum())}  msa={"yes" if seq in index else "MISSING"}', flush=True)

    missing = {p: s for p, s in seqs.items() if s not in index}
    for pdb_id, seq in seqs.items():
        os.makedirs(os.path.join(OUT, 'fasta', pdb_id), exist_ok=True)
        with open(os.path.join(OUT, 'fasta', pdb_id, f'{pdb_id}.fasta'), 'w') as f:
            f.write(f'>{pdb_id}\n{seq}\n')
        if pdb_id in missing:
            continue
        os.makedirs(os.path.join(OUT, 'msa', pdb_id), exist_ok=True)
        dst = os.path.join(OUT, 'msa', pdb_id, f'{pdb_id}.a3m')
        if not (os.path.exists(dst) and os.path.samefile(index[seq], dst)):   # the index may list OUT's own MSAs
            shutil.copyfile(index[seq], dst)
    if args.missing_fasta:
        with open(args.missing_fasta, 'w') as f:
            for pdb_id, seq in sorted(missing.items(), key=lambda kv: len(kv[1])):
                f.write(f'>{pdb_id}\n{seq}\n')
        print(f'wrote {len(missing)} sequences without an MSA to {args.missing_fasta}')
    json.dump(stats, open(os.path.join(OUT, 'build_stats.json'), 'w'), indent=2)
    print(f'{len(stats)} DMS, {len(seqs)} structures, {len(seqs) - len(missing)} with an exact-match MSA, '
          f'{len(missing)} missing -> {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--msa_index', required=True, help='index_msas.py output covering every MSA root to reuse')
    ap.add_argument('--missing_fasta', default=None)
    main(ap.parse_args())
