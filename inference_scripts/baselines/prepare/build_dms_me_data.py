"""Write the seven double-mutant DMS assays in upstream MutateEverything's layout.

Rows come from ESM-MSR's own preprocessed tables (the same rows esm_msr_testing.py's DMS block scores, keyed
code_mut_info, deduplicated, target ddG_ML) -- NOT from software/MutateEverything/data/dms/mutations, which is the
untrusted local fork's and is where `ddg` was previously found left un-negated. ddg = -ddG_ML, so positive is
destabilizing, as upstream expects.

MSAs are reused from the fork only after an exact query-sequence match (see find_dms_msas.py). GB1 deliberately uses
the GB1A MSA, never any of the six 1PGA-derived ones.

Orders >2 (GB1's 147,193 triples/quadruples) go through upstream's arbitrary-order path, which needs
mut_fr_list/mut_pos_list/mut_to_list; doubles use fr1..to2 and the dense pair tensor. Both are written.
"""
import ast
import json
import os
import shutil

import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths, dms_source
import paths  # noqa: E402
import dms_source  # noqa: E402

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
       'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}
# precomputed MSAs of the DMS query sequences (ColabFold; a data input, not in the repo), reused after an exact sequence match
FORK = os.environ.get('MSR_ME_DMS_MSAS', str(paths.THIRD_PARTY_DIR / 'MutateEverything' / 'data' / 'dms'))
OUT = str(paths.MUTATE_EVERYTHING_DIR / 'data' / 'dms_oct06')
# assay -> (pdb_id used for the fasta/msa dirs). DLG4 and GRB2 each have two assays on one sequence.
ASSAYS = {'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4_HUMAN',
          'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4_HUMAN',
          'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2_HUMAN',
          'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2_HUMAN',
          'MYO_HUMAN_Kung_2025_display': 'MYO_HUMAN',
          'ESTA_BACSU_Nutschel_2020_dTm': 'ESTA_BACSU',
          'GB1_Wu_2016_binding_domain': 'GB1A'}


def chain_seq(path, chain):
    seq, seen = [], set()
    for line in open(path):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
            rid = line[22:27]
            if rid not in seen:
                seen.add(rid); seq.append(AA3.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def query_of(path):
    name = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                if name is not None:
                    break
                name = line[1:].strip(); seq = ''
            elif name is not None:
                seq += line.strip()
    return seq.replace('-', '').upper()


def main():
    for sub in ['mutations', 'fasta', 'msa']:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    stats = {}
    seqs = {}

    for assay, pdb_id in ASSAYS.items():
        df = dms_source.load(assay)
        pf = paths.fix_pdb_path(df['pdb_file'].iloc[0])
        chain = df['chain'].iloc[0]
        seq = chain_seq(pf, chain)
        if pdb_id in seqs:
            assert seqs[pdb_id] == seq, f'{pdb_id}: two assays disagree on the wild-type sequence'
        seqs[pdb_id] = seq

        df['mut_info'] = df['mut_info'].map(lambda s: ':'.join(sorted(str(s).split(':'), key=lambda m: int(m[1:-1]))))
        df['id'] = df['code'] + '_' + df['mut_info']
        df = df[~df['id'].duplicated(keep='first')].reset_index(drop=True)

        rows, dropped = [], 0
        for mi, ddg_ml in zip(df['mut_info'], df['ddG_ML']):
            if pd.isna(ddg_ml):
                dropped += 1
                continue
            ms = mi.split(':')
            pos = [int(m[1:-1]) for m in ms]
            if any(p < 1 or p > len(seq) or seq[p - 1] != m[0] for p, m in zip(pos, ms)):
                dropped += 1
                continue
            if len(set(pos)) != len(pos):          # two substitutions at one position cannot be represented
                dropped += 1
                continue
            mut = list(seq)
            for p, m in zip(pos, ms):
                mut[p - 1] = m[-1]
            row = {'pdb_id': pdb_id, 'mut_info': mi, 'ddg': -float(ddg_ml), 'ddG_ML': float(ddg_ml),
                   'wt_seq': seq, 'mut_seq': ''.join(mut), 'uid': f'{pdb_id}_{mi}'}
            for i, (p, m) in enumerate(zip(pos, ms), 1):
                row.update({f'fr{i}': m[0], f'pos{i}': p, f'to{i}': m[-1]})
            if len(ms) > 2:                        # upstream's arbitrary-order path reads these three with eval()
                row['mut_fr_list'] = str([m[0] for m in ms])
                row['mut_pos_list'] = str(pos)
                row['mut_to_list'] = str([m[-1] for m in ms])
            rows.append(row)

        out = pd.DataFrame(rows)
        # every row carries the list columns (NaN for order<=2) so the column exists for the whole assay
        for c in ['mut_fr_list', 'mut_pos_list', 'mut_to_list']:
            if c not in out.columns:
                out[c] = None
        n = out['mut_info'].str.count(':') + 1
        path = os.path.join(OUT, 'mutations', f'{assay}.csv')
        out.to_csv(path, index=False)
        stats[assay] = {'rows': len(out), 'dropped': dropped, 'pdb_id': pdb_id, 'len': len(seq),
                        'order': {int(k): int(v) for k, v in n.value_counts().sort_index().items()},
                        'frac_destabilizing': round(float((out.ddg > 0).mean()), 4)}
        print(f'{assay[:42]:44s} {len(out):7d} rows (dropped {dropped:5d})  orders {stats[assay]["order"]}  '
              f'ddg>0 {stats[assay]["frac_destabilizing"]:.3f}', flush=True)
        # sanity: the list columns must round-trip for every order>2 row
        hi = out[n > 2]
        if len(hi):
            assert all(len(ast.literal_eval(x)) == len(m.split(':'))
                       for x, m in zip(hi['mut_pos_list'], hi['mut_info'])), f'{assay}: bad mut_pos_list'

    for pdb_id, seq in seqs.items():
        os.makedirs(os.path.join(OUT, 'fasta', pdb_id), exist_ok=True)
        with open(os.path.join(OUT, 'fasta', pdb_id, f'{pdb_id}.fasta'), 'w') as f:
            f.write(f'>{pdb_id}\n{seq}\n')
        src = os.path.join(FORK, 'msa', pdb_id, f'{pdb_id}.a3m')
        assert os.path.exists(src), f'no MSA for {pdb_id} at {src}'
        q = query_of(src)
        assert q == seq, f'{pdb_id}: MSA query does not match the structure sequence ({len(q)} vs {len(seq)})'
        os.makedirs(os.path.join(OUT, 'msa', pdb_id), exist_ok=True)
        shutil.copyfile(src, os.path.join(OUT, 'msa', pdb_id, f'{pdb_id}.a3m'))
        depth = sum(1 for line in open(src) if line.startswith('>'))
        print(f'{pdb_id:12s} len {len(seq):4d}  MSA query matches  depth {depth}', flush=True)

    json.dump(stats, open(os.path.join(OUT, 'build_stats.json'), 'w'), indent=2)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
