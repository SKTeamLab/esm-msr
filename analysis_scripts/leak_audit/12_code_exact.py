"""Exact-mutation and constituent-mutation overlaps for code/sequence-table inventories:
DDMut (S9028 + SM1242), GeoDDG (S8754), and DDGemb (S2450).

Combines three complementary matchers so identical variants are caught across alternate PDB IDs,
construct/signal-peptide truncations, and mutant-crystal reverse rows:
  1. Direct ``(code, chain, author_pos, wt, mt)`` key match (when test dataset carries PDB author numbering).
  2. Exact ``(start_seq, target_seq)`` full-sequence pair match (using PDB residue maps from ``05b`` for
     S9028 Forward + Reverse rows and SM1242 F + R rows, ``GS_S8754`` / ``s8754`` sequences, and ``S2450``).
  3. Semiglobal alignment coordinate mapping (free end gaps on ``>=90%``-identity, ``>=80%``-shorter-coverage
     MMseqs2 pairs) from each inventory protein into each test protein's 1-based sequence coordinates,
     validating both wild-type and mutant amino acids at the aligned position(s).
"""
import collections
import os
import re
import sys

import numpy as np
import pandas as pd
from Bio import Align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (
    OUT_DIR,
    REPO_ROOT,
    WORK_DATA_DIR,
    apply_mutations,
    check_against_seq,
    data_file,
    parse_mut_str,
)

os.makedirs(OUT_DIR, exist_ok=True)
PREP_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed')

prots = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
pid_to_seq = dict(zip(prots['pid'], prots['seq']))
resmaps = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'inventory_proteins', 'pdb_resmaps.pkl'))

# Build semiglobal position maps (inv_pid -> {test_pid: {inv_pos_1based: test_pos_1based}})
# prefiltered by mmseqs_pairs (fident >= 0.90 & cov_min >= 0.80)
mm = pd.read_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'))
cand = mm[(mm['fident'] >= 0.90) & (mm['cov_min'] >= 0.80)]

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.end_open_gap_score = 0
aligner.end_extend_gap_score = 0
aligner.open_gap_score = -10
aligner.extend_gap_score = -1
aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')

inv_to_test_pos = collections.defaultdict(dict)


def register_pair(inv_pid: str, test_pid: str):
    iseq = pid_to_seq.get(inv_pid, '')
    tseq = pid_to_seq.get(test_pid, '')
    if not iseq or not tseq:
        return
    if iseq == tseq:
        inv_to_test_pos[inv_pid][test_pid] = {i: i for i in range(1, len(iseq) + 1)}
        return
    a = next(iter(aligner.align(iseq, tseq)))
    A, B = str(a[0]), str(a[1])
    cols = [(x, y) for x, y in zip(A, B) if x != '-' and y != '-']
    if not cols:
        return
    fid = sum(1 for x, y in cols if x == y) / len(cols)
    cov = len(cols) / min(len(iseq), len(tseq))
    if fid < 0.90 or cov < 0.80:
        return
    ii = ti = 0
    mp = {}
    for x, y in zip(A, B):
        if x != '-':
            ii += 1
        if y != '-':
            ti += 1
        if x != '-' and y != '-':
            mp[ii] = ti
    inv_to_test_pos[inv_pid][test_pid] = mp


# Self-sequence matches (identical hashes, which MMseqs skips only when pid==pid, not across pids, but ensure 100%)
seq_to_inv = collections.defaultdict(list)
for pid, s in pid_to_seq.items():
    if str(pid).startswith('inv:'):
        seq_to_inv[s].append(pid)
for pid, s in pid_to_seq.items():
    if not str(pid).startswith(('inv:', 'lib:')):
        for ipid in seq_to_inv.get(s, ()):
            inv_to_test_pos[ipid][pid] = {i: i for i in range(1, len(s) + 1)}

for p_id, q_id in zip(cand['p'], cand['q']):
    ps, qs = str(p_id), str(q_id)
    if ps.startswith('inv:') and not qs.startswith(('inv:', 'lib:')):
        register_pair(ps, qs)
    elif qs.startswith('inv:') and not ps.startswith(('inv:', 'lib:')):
        register_pair(qs, ps)

print('aligned inventory->test protein maps:', sum(len(v) for v in inv_to_test_pos.values()))


def project_inv_variants(inv_variants_by_pid):
    """Given {inv_pid: [tuple_of_(wt, inv_pos_1based, mt), ...]}, project onto every test_pid:
    returns (test_variants_by_pid, test_singles_by_pid)."""
    t_vars = collections.defaultdict(set)
    t_sgls = collections.defaultdict(set)
    for ipid, mut_lists in inv_variants_by_pid.items():
        for tpid, pos_map in inv_to_test_pos.get(ipid, {}).items():
            tseq = pid_to_seq[tpid]
            for mlist in mut_lists:
                proj = []
                ok = True
                for wt, ipos, mt in mlist:
                    tpos = pos_map.get(ipos)
                    if tpos is None or tpos < 1 or tpos > len(tseq) or tseq[tpos - 1] != wt:
                        ok = False
                        break
                    proj.append((wt, tpos, mt))
                if ok and proj:
                    fs = frozenset(proj)
                    t_vars[tpid].add(fs)
                    t_sgls[tpid].update(fs)
    return t_vars, t_sgls


# ---------------------------------------------------------------- 1. Build S9028 inventory
S9 = pd.read_csv(data_file('S9028.csv'))
s9028_author = set()
s9028_seq_pairs = set()
s9028_by_pid = collections.defaultdict(list)

for x, rtype in zip(S9['pdb_mut_chain'].astype(str), S9['type'].astype(str)):
    m = re.match(r'^([0-9A-Za-z]{4})_([A-Z])(-?\d+[A-Za-z]?)([A-Z])_([A-Za-z0-9]+)$', x)
    if not m:
        continue
    code, wt_m, resi_s, mt_m, ch = m.group(1).upper(), m.group(2), m.group(3), m.group(4), m.group(5)
    try:
        s9028_author.add((code, ch, int(resi_s), wt_m, mt_m))
    except ValueError:
        pass
    rl = resmaps.get(f'{code}_{ch}')
    if not rl:
        continue
    idx_map = {r_str: idx for idx, (r_str, _) in enumerate(rl)}
    if resi_s not in idx_map:
        continue
    idx = idx_map[resi_s]
    pdb_seq = ''.join(aa for _, aa in rl)
    ipid = f'inv:ddmut9028:{code}_{ch}'
    if rtype.lower().startswith('f') and pdb_seq[idx] == wt_m:
        start_s = pdb_seq
        targ_s = pdb_seq[:idx] + mt_m + pdb_seq[idx + 1:]
        s9028_seq_pairs.add((start_s, targ_s))
        s9028_by_pid[ipid].append([(wt_m, idx + 1, mt_m)])
    elif rtype.lower().startswith('r') and pdb_seq[idx] == mt_m:
        start_s = pdb_seq[:idx] + wt_m + pdb_seq[idx + 1:]
        targ_s = pdb_seq
        s9028_seq_pairs.add((start_s, targ_s))
        # Also project Reverse mutations onto mutant-crystal test proteins (e.g. ssym/s669 reverse rows)
        s9028_by_pid[ipid].append([(wt_m, idx + 1, mt_m)])

s9028_tvars, s9028_tsgls = project_inv_variants(s9028_by_pid)
print(f'S9028: author_keys={len(s9028_author)}, seq_pairs={len(s9028_seq_pairs)}, projected_test_pids={len(s9028_tvars)}')

# ---------------------------------------------------------------- 2. Build SM1242 inventory
SM = pd.read_csv(data_file('DD_SM1242.csv'))
sm1242_author = set()
sm1242_seq_pairs = set()
sm1242_by_pid = collections.defaultdict(list)

for struc, mmlab, rtype in zip(SM['struc'].astype(str), SM['mm_label'].astype(str), SM['type'].astype(str)):
    m_st = re.match(r'^([0-9A-Za-z]{4})_([A-Za-z0-9]+)$', struc.strip())
    if not m_st:
        continue
    code, ch = m_st.group(1).upper(), m_st.group(2)
    parts = [p for p in re.split(r'[_:;]', mmlab.strip()) if p]
    parsed_tokens = [re.match(r'^([A-Z])(-?\d+[A-Za-z]?)([A-Z])$', p) for p in parts]
    if not parsed_tokens or any(t is None for t in parsed_tokens):
        continue
    sm1242_author.add((f'{code}_{ch}', tuple(sorted(parts))))
    rl = resmaps.get(f'{code}_{ch}')
    if not rl:
        continue
    idx_map = {r_str: idx for idx, (r_str, _) in enumerate(rl)}
    if any(t.group(2) not in idx_map for t in parsed_tokens):
        continue
    pdb_seq = ''.join(aa for _, aa in rl)
    ipid = f'inv:ddmutsm1242:{code}_{ch}'
    if rtype.strip() == 'F' and all(pdb_seq[idx_map[t.group(2)]] == t.group(1) for t in parsed_tokens):
        arr = list(pdb_seq)
        mlist = []
        for t in parsed_tokens:
            ix = idx_map[t.group(2)]
            arr[ix] = t.group(3)
            mlist.append((t.group(1), ix + 1, t.group(3)))
        sm1242_seq_pairs.add((pdb_seq, ''.join(arr)))
        sm1242_by_pid[ipid].append(mlist)
    elif rtype.strip() == 'R' and all(pdb_seq[idx_map[t.group(2)]] == t.group(3) for t in parsed_tokens):
        arr = list(pdb_seq)
        mlist = []
        for t in parsed_tokens:
            ix = idx_map[t.group(2)]
            arr[ix] = t.group(1)
            mlist.append((t.group(1), ix + 1, t.group(3)))
        sm1242_seq_pairs.add((''.join(arr), pdb_seq))
        sm1242_by_pid[ipid].append(mlist)

sm1242_tvars, sm1242_tsgls = project_inv_variants(sm1242_by_pid)
print(f'SM1242: author_keys={len(sm1242_author)}, seq_pairs={len(sm1242_seq_pairs)}, projected_test_pids={len(sm1242_tvars)}')

# ---------------------------------------------------------------- 3. Build GeoDDG S8754 inventory
gs_author = set()
gs_seq_pairs = set()
gs_by_pid = collections.defaultdict(list)
gs_raw_csv = data_file('GS_S8754.csv')
if os.path.exists(gs_raw_csv):
    GS = pd.read_csv(gs_raw_csv)
    s8_df = pd.read_csv(os.path.join(PREP_DIR, 's8754_mapped.csv'), low_memory=False)
    for x, ws, ms in zip(GS['name'].astype(str), GS['wt_seq'].astype(str), GS['mut_seq'].astype(str)):
        gs_seq_pairs.add((ws, ms))
        m = re.match(r'^rcsb_([0-9A-Za-z]{4})_([A-Za-z0-9]+)_([A-Z])(-?\d+)([A-Z])', x)
        if m:
            gs_author.add((m.group(1).upper(), m.group(2), int(m.group(4)), m.group(3), m.group(5)))
    for r in s8_df.itertuples(index=False):
        ipid = f'inv:geoddg8754:{r.code}_{r.chain}'
        if pd.notna(r.seq_pos):
            gs_by_pid[ipid].append([(str(r.wild_type), int(r.seq_pos), str(r.mutation))])
else:
    # Fallback: GeoDDG's training table is S8754 itself
    s8_df = pd.read_csv(os.path.join(PREP_DIR, 's8754_mapped.csv'), low_memory=False)
    for r in s8_df.itertuples(index=False):
        gs_seq_pairs.add((str(r.wt_seq), str(r.mut_seq)))
        if pd.notna(r.position):
            gs_author.add((str(r.code).upper(), str(r.chain), int(r.position), str(r.wild_type), str(r.mutation)))
        if pd.notna(r.seq_pos):
            ipid = f'inv:geoddg8754:{r.code}_{r.chain}'
            gs_by_pid[ipid].append([(str(r.wild_type), int(r.seq_pos), str(r.mutation))])

gs_tvars, gs_tsgls = project_inv_variants(gs_by_pid)
print(f'GS_S8754: author_keys={len(gs_author)}, seq_pairs={len(gs_seq_pairs)}, projected_test_pids={len(gs_tvars)}')

# ---------------------------------------------------------------- 4. Build DDGemb S2450 inventory
s24_fa = {}
cur_h = ''
for line in open(data_file('S2450.fasta')):
    if line.startswith('>'):
        cur_h = line[1:].strip()
    elif line.strip():
        s24_fa[cur_h] = line.strip()

s24_tsv = pd.read_csv(data_file('S2450.tsv'), sep='\t')
s24_seq_pairs = set()
s24_by_pid = collections.defaultdict(list)
for r in s24_tsv.itertuples(index=False):
    hid = str(r.PDB)
    wt_c, pos_i, mt_c = str(r.WT), int(r.POS), str(r.MT)
    sq = s24_fa.get(hid, '')
    if sq and 1 <= pos_i <= len(sq) and sq[pos_i - 1] == wt_c:
        msq = sq[:pos_i - 1] + mt_c + sq[pos_i:]
        s24_seq_pairs.add((sq, msq))
        s24_by_pid[f'inv:ddgemb2450:{hid}'].append([(wt_c, pos_i, mt_c)])

s24_tvars, s24_tsgls = project_inv_variants(s24_by_pid)
print(f'S2450: seq_pairs={len(s24_seq_pairs)}, projected_test_pids={len(s24_tvars)}')

# ---------------------------------------------------------------- Evaluate across datasets
EXT_SINGLE_CSVS = {
    's571': 's571_mapped.csv',
    's783': 's783_mapped.csv',
    's2648': 's2648_mapped.csv',
    's8754': 's8754_mapped.csv',
    's669': 's669_mapped.csv',
    's461': 's461_mapped.csv',
    'ssym': 'ssym_mapped.csv',
    'q3421': 'q3421_mapped.csv',
    'k3822': 'k3822_mapped.csv',
    'k2369': 'k2369_mapped.csv',
}
EXT_MULTI_CSVS = {
    'ptmul': 'ptmul_mapped.csv',
    'ptmuld': 'ptmuld_mapped.csv',
}

ALL_DS = [
    's571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369',
    'ptmul', 'ptmuld', 'tsu_capped_val', 'tsu_capped_test',
    'DLG4 abund.', 'DLG4 bind.', 'GRB2 abund.', 'GRB2 bind.', 'MYO', 'ESTA', 'GB1',
    'domainome',
]

res = []


def add_row(inv, ds, n, exact, const=0, note=''):
    res.append(dict(
        inventory=inv,
        dataset=ds,
        n=int(n),
        exact=int(exact),
        constituent=int(const),
        frac=exact / max(n, 1),
        note=note,
    ))
    if exact > 0 or const > 0 or ds in EXT_SINGLE_CSVS or ds in EXT_MULTI_CSVS:
        print(f'{inv:16s} {ds:18s} exact={exact:6d}/{n:6d} const={const:6d} ({exact / max(n, 1):.3f}) {note}')


def eval_on_dataset(ds: str, author_single_set, author_multi_set, seq_pairs_set, tvars, tsgls):
    tdf = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', f'{ds}.pkl'))
    n = len(tdf)

    # Map row_id -> author key if external dataset
    rid_to_author_single = {}
    rid_to_author_multi = {}
    if ds in EXT_SINGLE_CSVS and author_single_set is not None:
        raw = pd.read_csv(os.path.join(PREP_DIR, EXT_SINGLE_CSVS[ds]), low_memory=False)
        for i, r in raw.iterrows():
            if pd.notna(r.get('position')):
                try:
                    rid_to_author_single[str(i)] = (
                        str(r['code']).upper(), str(r['chain']), int(r['position']), str(r['wild_type']), str(r['mutation'])
                    )
                except ValueError:
                    pass
    elif ds in EXT_MULTI_CSVS and author_multi_set is not None:
        raw = pd.read_csv(os.path.join(PREP_DIR, EXT_MULTI_CSVS[ds]), low_memory=False)
        for i, r in raw.iterrows():
            muts_a = tuple(sorted(
                x for x in re.split(r'[:;]', str(r.get('mut_info', ''))) if re.match(r'^[A-Z]-?\d+[A-Za-z]?[A-Z]$', x)
            ))
            rid_to_author_multi[str(i)] = (f"{str(r['code']).upper()}_{str(r['chain'])}", muts_a)

    exact = const = 0
    for pid, seq, mut_str, rid in zip(tdf['pid'], tdf['seq'], tdf['mutation'], tdf['row_id']):
        muts = parse_mut_str(mut_str) or []
        if not muts:
            continue
        ok, _ = check_against_seq(seq, muts)
        targ_seq = apply_mutations(seq, muts) if ok else ''
        fs = frozenset(muts)

        is_ex = False
        if author_single_set is not None and str(rid) in rid_to_author_single:
            if rid_to_author_single[str(rid)] in author_single_set:
                is_ex = True
        if not is_ex and author_multi_set is not None and str(rid) in rid_to_author_multi:
            if rid_to_author_multi[str(rid)] in author_multi_set:
                is_ex = True
        if not is_ex and targ_seq and (seq, targ_seq) in seq_pairs_set:
            is_ex = True
        if not is_ex and fs in tvars.get(pid, ()):
            is_ex = True

        if is_ex:
            exact += 1
        elif len(fs) >= 1 and fs <= tsgls.get(pid, set()):
            # Only multi-mutants can be constituent (for single mutants len(fs)==1 -> fs in tsgls implies exact on singles-only inventories,
            # or constituent of a multi-mutant training row)
            if len(fs) > 1:
                const += 1
    return n, exact, const


for ds in ALL_DS:
    # DDMut S9028 (combined with SM1242 for DDMut's multi-mutant head on multi-mutant sets in 13_matrix.py)
    n, ex9, co9 = eval_on_dataset(ds, s9028_author, None, s9028_seq_pairs, s9028_tvars, s9028_tsgls)
    add_row('ddmut_s9028', ds, n, ex9, co9)

    n, exm, com = eval_on_dataset(ds, None, sm1242_author, sm1242_seq_pairs, sm1242_tvars, sm1242_tsgls)
    if exm > 0 or com > 0 or ds in ('ptmul', 'ptmuld', 'GB1'):
        add_row('ddmut_sm1242', ds, n, exm, com)

    n, exg, cog = eval_on_dataset(ds, gs_author, None, gs_seq_pairs, gs_tvars, gs_tsgls)
    add_row('geoddg_s8754', ds, n, exg, cog)

    n, ex24, co24 = eval_on_dataset(ds, None, None, s24_seq_pairs, s24_tvars, s24_tsgls)
    if ds == 's2648':
        note24 = f'S2450 = S2648 - 198 documented (2450; sequence-aligned={ex24})'
        ex24 = max(2450, ex24)
    else:
        note24 = 'verified via S2450.tsv + S2450.fasta alignment'
    add_row('ddgemb_s2450', ds, n, ex24, co24, note=note24)

pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, 'code_inventory_exact.csv'), index=False)
print('saved code_inventory_exact.csv:', len(res), 'rows')
