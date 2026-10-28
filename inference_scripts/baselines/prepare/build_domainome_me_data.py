"""Write the domainome VAMP assay in upstream MutateEverything's layout.

Rows come from ESM-MSR's own domainome_mapped_2026.csv -- the same rows esm_msr_testing.py's domainome block scores
(code = domain_ID with '/'->'_', target scaled_fitness, rows without a structure or position dropped) -- not from
software/MutateEverything/data/domainome_2026/mutations, which is the untrusted fork's. ddg = -scaled_fitness, so
positive is destabilizing as upstream expects; scaled_fitness is 76% negative, so this does NOT trip upstream's
sign-flip heuristic.

Fastas and MSAs are reused from the fork only where the MSA's query sequence equals the structure's chain-A sequence
exactly (audit_domainome.py checked all 522 domains: every row matches its structure and every domain has a match).
All 576,403 rows are single mutants, so only the L x 20 tensor is used; no pair or arbitrary-order columns are needed.
"""
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
TAB = str(paths.DOMAINOME_TABLE)
# precomputed MSAs / fastas per domain (ColabFold; a data input, not in the repo)
FORK = os.environ.get('MSR_ME_DOMAINOME_MSAS', str(paths.THIRD_PARTY_DIR / 'MutateEverything' / 'data' / 'domainome_2026'))
OUT = str(paths.MUTATE_EVERYTHING_DIR / 'data' / 'domainome_oct06')
AUDIT = os.environ.get('MSR_DOMAINOME_AUDIT', str(paths.REPO_ROOT / 'outputs' / 'baselines' / 'domainome_audit.json'))   # prepare/audit_domainome.py


def chain_seq(path, chain='A'):
    seq, seen = [], set()
    for line in open(path):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
            rid = line[22:27]
            if rid not in seen:
                seen.add(rid); seq.append(AA3.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def main():
    for sub in ['mutations', 'fasta', 'msa']:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    audit = json.load(open(AUDIT))

    df = pd.read_csv(TAB, usecols=['domain_ID', 'mut_type', 'scaled_fitness', 'pdb_file', 'position', 'wild_type',
                                   'mutation'], low_memory=False)
    df['code'] = df['domain_ID'].str.replace('/', '_', regex=False)
    df = df.dropna(subset=['pdb_file', 'position'])
    df['pdb_file'] = paths.fix_pdb_column(df['pdb_file'])
    df = df.dropna(subset=['scaled_fitness'])
    print(f'{len(df)} rows, {df.code.nunique()} domains', flush=True)

    rows, dropped, missing_msa = [], 0, []
    for code, g in df.groupby('code'):
        seq = chain_seq(g['pdb_file'].iloc[0])
        pdb_id = code + 'A' if os.path.isdir(os.path.join(FORK, 'fasta', code + 'A')) else code
        src = audit.get(code, {}).get('msa')
        if src is None or not os.path.exists(src):
            missing_msa.append(code); continue
        g = g[~g[['code', 'mut_type']].duplicated(keep='first')]
        for mt, pos, wt, mu, fit in zip(g.mut_type, g.position.astype(int), g.wild_type, g.mutation, g.scaled_fitness):
            if not (1 <= pos <= len(seq)) or seq[pos - 1] != wt:
                dropped += 1
                continue
            mut = list(seq); mut[pos - 1] = mu
            rows.append({'pdb_id': pdb_id, 'mut_info': mt, 'ddg': -float(fit), 'scaled_fitness': float(fit),
                         'wt_seq': seq, 'mut_seq': ''.join(mut), 'uid': f'{code}_{mt}', 'code': code,
                         'fr1': wt, 'pos1': pos, 'to1': mu})
        # fasta written from the structure sequence, MSA copied after the audit's exact-query check
        os.makedirs(os.path.join(OUT, 'fasta', pdb_id), exist_ok=True)
        with open(os.path.join(OUT, 'fasta', pdb_id, f'{pdb_id}.fasta'), 'w') as f:
            f.write(f'>{pdb_id}\n{seq}\n')
        os.makedirs(os.path.join(OUT, 'msa', pdb_id), exist_ok=True)
        shutil.copyfile(src, os.path.join(OUT, 'msa', pdb_id, f'{pdb_id}.a3m'))

    out = pd.DataFrame(rows)
    assert out['pdb_id'].nunique() == df['code'].nunique() - len(missing_msa)
    # upstream requires exactly one wt_seq per pdb_id or it silently drops the protein
    assert (out.groupby('pdb_id').wt_seq.nunique() == 1).all(), 'a pdb_id has more than one wild-type sequence'
    dest = os.path.join(OUT, 'mutations', 'domainome.csv')
    out.to_csv(dest, index=False)
    print(f'wrote {len(out)} rows, {out.pdb_id.nunique()} proteins -> {dest}')
    print(f'dropped {dropped} rows (position/wild-type mismatch), {len(missing_msa)} domains without an MSA')
    print(f'ddg>0 fraction {float((out.ddg > 0).mean()):.3f}  (upstream flips the sign if mean(ddg<0) > 0.5)')


if __name__ == '__main__':
    main()
