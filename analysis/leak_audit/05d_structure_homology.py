"""Foldseek 3D structural homology search for Domainome and ProteinGym against training inventories.

Extracts single-chain query PDBs for:
  - Domainome (522 Pfam domains from ``spurs_structures/<domain_ID>.pdb``)
  - ProteinGym (217 DMS assays from ``ProteinGym_AF2_structures/<pdb_file>``)
and single-chain target PDBs for all structure-bearing training inventories:
  - MegaScale libraries (AlphaFold PDBs per ``code_wt``, mapped to every ``lib:<code>``)
  - DDMut S9028 / SM1242 and GeoDDG S8754 PDB chains (plus S2450 proteins sharing >=95% identity
    over >=90% coverage with a structure-mapped S2648/S9028 chain)

Runs ``foldseek easy-search --exhaustive-search 1 -e 10.0`` (with SHA-1 manifest caching) and writes:
  - ``$LEAK_DATA_DIR/foldseek_hits_dom_pg.pkl`` (directed query ``pid`` -> target ``pid`` structural hits)
  - ``$LEAK_OUT_DIR/structure_homology_domainome_pg.csv`` (per-domain / per-DMS sequence vs structure
    summary against the 172 Homology-Restricted MegaScale capped-train scaffolds)
"""
import collections
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import FOLDSEEK_BIN, LEAK_ROOT, OUT_DIR, SOFTWARE_DIR, WORK_DATA_DIR

os.makedirs(WORK_DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FOLDSEEK_COLS = [
    'query', 'target', 'fident', 'alnlen', 'qlen', 'tlen',
    'evalue', 'bits', 'prob', 'alntmscore', 'qtmscore', 'ttmscore', 'lddt',
]


def safe_stem(s: str) -> str:
    return s.replace(':', '__').replace('/', '_').replace(' ', '_')


def extract_single_chain_pdb(src_pdb: str, chain: str, dst_pdb: str) -> bool:
    """Write ATOM/HETATM standard residue lines of one chain to ``dst_pdb``."""
    if not src_pdb or not os.path.exists(src_pdb):
        return False
    by_chain = collections.defaultdict(list)
    for line in open(src_pdb, errors='ignore'):
        if line.startswith('ATOM  ') or (line.startswith('HETATM') and line[17:20].strip() == 'MSE'):
            by_chain[line[21]].append(line)
    if not by_chain:
        return False
    chosen = chain if chain in by_chain else (
        chain.upper() if chain.upper() in by_chain else max(by_chain, key=lambda k: len(by_chain[k]))
    )
    lines = by_chain[chosen]
    if not lines:
        return False
    with open(dst_pdb, 'w') as fh:
        fh.writelines(lines)
        fh.write('END\n')
    return True


def main():
    if not FOLDSEEK_BIN or not os.path.exists(FOLDSEEK_BIN):
        raise FileNotFoundError(f'Foldseek binary not found ({FOLDSEEK_BIN}); set FOLDSEEK_BIN.')

    fs_dir = os.path.join(WORK_DATA_DIR, 'foldseek_dom_pg')
    q_dir = os.path.join(fs_dir, 'queries')
    t_dir = os.path.join(fs_dir, 'targets')
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(t_dir, exist_ok=True)

    # 1. Queries: Domainome (522) + ProteinGym (217)
    dom_pdb_dir = os.environ.get(
        'DOMAINOME_PDB_DIR',
        os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'domainome1', 'spurs_structures'),
    )
    pg_root = os.environ.get('PROTEINGYM_DIR', os.path.join(LEAK_ROOT, 'ProteinGym'))
    pg_pdb_dir = os.path.join(pg_root, 'ProteinGym_AF2_structures')
    pgm = pd.read_csv(os.path.join(pg_root, 'DMS_substitutions.csv'))

    dom_df = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', 'domainome.pkl'))
    dom_pids = sorted(dom_df['pid'].unique())

    q_stem_to_pids = collections.defaultdict(list)
    q_manifest = {}

    for pid in dom_pids:
        dom_id = pid[4:]  # strip 'dom:'
        src = os.path.join(dom_pdb_dir, f'{dom_id}.pdb')
        stem = safe_stem(pid)
        dst = os.path.join(q_dir, f'{stem}.pdb')
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy(src, dst)
            q_stem_to_pids[stem].append(pid)
            q_manifest[stem] = (src, os.path.getsize(src))

    pg_df = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', 'proteingym.pkl'))
    pg_pids = set(pg_df['pid'].unique())
    for _, r in pgm.iterrows():
        pid = 'pg:' + str(r['DMS_id'])
        if pid not in pg_pids:
            continue
        pdb_f = str(r.get('pdb_file', '')).strip()
        src = os.path.join(pg_pdb_dir, pdb_f)
        stem = safe_stem(pid)
        dst = os.path.join(q_dir, f'{stem}.pdb')
        if pdb_f and os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy(src, dst)
            q_stem_to_pids[stem].append(pid)
            q_manifest[stem] = (src, os.path.getsize(src))

    print(f'Foldseek queries prepared: {len(q_stem_to_pids)} structures ({len(dom_pids)} Domainome, {len(pg_pids)} ProteinGym)')

    # 2. Targets: MegaScale AlphaFold PDBs + inventory PDB chains
    af_dir = os.environ.get(
        'TSUBOYAMA_AF_DIR',
        os.path.join(SOFTWARE_DIR, 'esm-msr', 'data', 'tsuboyama', 'AlphaFold_model_PDBs'),
    )
    libs = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
    t_stem_to_pids = collections.defaultdict(list)
    t_manifest = {}

    for code, cwt in libs['code_wt'].items():
        cwt_s = str(cwt)
        src = None
        for cand in (cwt_s, cwt_s.replace('_', ':')):
            p = os.path.join(af_dir, f'{cand}.pdb')
            if os.path.exists(p):
                src = p
                break
        if not src:
            continue
        stem = 'ms__' + safe_stem(cwt_s)
        dst = os.path.join(t_dir, f'{stem}.pdb')
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        t_stem_to_pids[stem].append(f'lib:{code}')
        t_manifest[stem] = (src, os.path.getsize(src))

    inv_pkl = os.path.join(WORK_DATA_DIR, 'inventory_proteins', 'inventory_proteins.pkl')
    inv_df = pd.read_pickle(inv_pkl)
    chain_key_to_stem = {}
    for row in inv_df.itertuples(index=False):
        if not row.pdb_path or not os.path.exists(str(row.pdb_path)):
            continue
        ck = (str(row.pdb_path), str(row.chain))
        if ck not in chain_key_to_stem:
            stem = f'inv__{safe_stem(str(row.code))}_{safe_stem(str(row.chain))}'
            dst = os.path.join(t_dir, f'{stem}.pdb')
            if not os.path.exists(dst):
                if not extract_single_chain_pdb(str(row.pdb_path), str(row.chain), dst):
                    continue
            chain_key_to_stem[ck] = stem
            t_manifest[stem] = (str(row.pdb_path), os.path.getsize(dst))
        t_stem_to_pids[chain_key_to_stem[ck]].append(str(row.pid))

    # Map S2450 sequence-only entries to S2648/inventory single-chain PDBs when in the same >=95% identity family
    s2648_csv = os.path.join(LEAK_ROOT, 'repo', 'data', 'preprocessed', 's2648_mapped.csv')
    if not os.path.exists(s2648_csv):
        from lib import REPO_ROOT
        s2648_csv = os.path.join(REPO_ROOT, 'data', 'preprocessed', 's2648_mapped.csv')
    if os.path.exists(s2648_csv) and os.path.exists(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv')):
        s26 = pd.read_csv(s2648_csv, usecols=['code', 'chain', 'pdb_file']).drop_duplicates()
        s26_pdb = {f"s2648:{r.code}_{r.chain}": (str(r.pdb_file), str(r.chain), str(r.code)) for r in s26.itertuples(index=False)}
        mp = pd.read_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'))
        hi = mp[(mp['fident'] >= 0.95) & (mp['cov_min'] >= 0.90)]
        s2450_best = {}
        for p_id, q_id, fid in zip(hi['p'], hi['q'], hi['fident']):
            if str(p_id).startswith('inv:ddgemb2450:') and str(q_id) in s26_pdb:
                if p_id not in s2450_best or fid > s2450_best[p_id][0]:
                    s2450_best[p_id] = (fid, s26_pdb[q_id])
            elif str(q_id).startswith('inv:ddgemb2450:') and str(p_id) in s26_pdb:
                if q_id not in s2450_best or fid > s2450_best[q_id][0]:
                    s2450_best[q_id] = (fid, s26_pdb[p_id])
        for s24_pid, (_, (pdb_p, ch, code)) in s2450_best.items():
            orig_p = pdb_p.replace('_processed.pdb', '_original.pdb')
            use_p = orig_p if os.path.exists(orig_p) else pdb_p
            if not os.path.exists(use_p):
                continue
            ck = (use_p, ch)
            if ck not in chain_key_to_stem:
                stem = f'inv__s26_{safe_stem(code)}_{safe_stem(ch)}'
                dst = os.path.join(t_dir, f'{stem}.pdb')
                if not os.path.exists(dst):
                    if not extract_single_chain_pdb(use_p, ch, dst):
                        continue
                chain_key_to_stem[ck] = stem
                t_manifest[stem] = (use_p, os.path.getsize(dst))
            t_stem_to_pids[chain_key_to_stem[ck]].append(s24_pid)

    # Remove stale PDB files not in current manifests
    for d_path, mf in [(q_dir, q_manifest), (t_dir, t_manifest)]:
        valid_files = {f'{k}.pdb' for k in mf}
        for fn in os.listdir(d_path):
            if fn.endswith('.pdb') and fn not in valid_files:
                os.remove(os.path.join(d_path, fn))

    print(f'Foldseek targets prepared: {len(t_stem_to_pids)} unique structures covering {sum(len(v) for v in t_stem_to_pids.values())} inventory pids')

    manifest_blob = json.dumps({'q': sorted(q_manifest.items()), 't': sorted(t_manifest.items())}, sort_keys=True)
    manifest_sha1 = hashlib.sha1(manifest_blob.encode()).hexdigest()
    hits_tsv = os.path.join(fs_dir, 'foldseek_hits.tsv')
    sha_path = hits_tsv + '.sha1'
    cached_sha = open(sha_path).read().strip() if os.path.exists(sha_path) else ''

    if not os.path.exists(hits_tsv) or cached_sha != manifest_sha1:
        tmp_dir = os.path.join(fs_dir, 'tmp')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        cmd = [
            FOLDSEEK_BIN, 'easy-search', q_dir, t_dir, hits_tsv, tmp_dir,
            '--exhaustive-search', '1',
            '-e', '10.0',
            '--format-output', ','.join(FOLDSEEK_COLS),
            '--threads', os.environ.get('THREADS', '8'),
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        with open(sha_path, 'w') as fh:
            fh.write(manifest_sha1 + '\n')
    else:
        print('reusing cached foldseek_hits.tsv (sha1 match)')

    raw = pd.read_csv(hits_tsv, sep='\t', header=None, names=FOLDSEEK_COLS)
    for col in ('query', 'target'):
        raw[col] = raw[col].astype(str).str.replace(r'\.pdb$', '', regex=True)
    raw['qcov'] = raw['alnlen'] / raw['qlen'].replace(0, np.nan)
    raw['tcov'] = raw['alnlen'] / raw['tlen'].replace(0, np.nan)
    raw['tm_max'] = raw[['qtmscore', 'ttmscore']].max(axis=1)
    raw['tm_min'] = raw[['qtmscore', 'ttmscore']].min(axis=1)

    # Expand stem hits to pid -> [(target_pid, evalue, qcov, qtmscore, tm_max, fident, lddt), ...]
    # Deduplicate per (q_pid, t_pid) keeping best evalue / max qtmscore
    pair_rows = []
    for r in raw.itertuples(index=False):
        q_pids = q_stem_to_pids.get(r.query, [])
        t_pids = t_stem_to_pids.get(r.target, [])
        if not q_pids or not t_pids:
            continue
        for qp in q_pids:
            for tp in t_pids:
                pair_rows.append((
                    qp, tp, float(r.evalue), float(r.qcov), float(r.tcov),
                    float(r.qtmscore), float(r.ttmscore), float(r.tm_max),
                    float(r.fident), int(r.alnlen), float(r.lddt),
                ))
    pair_df = pd.DataFrame(
        pair_rows,
        columns=['q_pid', 't_pid', 'struct_evalue', 'qcov', 'tcov', 'qtmscore', 'ttmscore', 'tm_max', 'struct_fident', 'struct_alnlen', 'lddt'],
    )
    if len(pair_df):
        pair_df = pair_df.sort_values(['struct_evalue', 'qtmscore'], ascending=[True, False]).drop_duplicates(['q_pid', 't_pid']).reset_index(drop=True)
    out_pkl = os.path.join(WORK_DATA_DIR, 'foldseek_hits_dom_pg.pkl')
    pair_df.to_pickle(out_pkl)
    print(f'saved {len(pair_df)} (query_pid, target_pid) structural pairs -> {out_pkl}')

    # 3. Summary table for Domainome (522) and ProteinGym (217 DMS) against capped-train scaffolds
    sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
    trn_codes = set(sp['train'])
    trn_pids = {f'lib:{c}' for c, cwt in libs['code_wt'].items() if cwt in trn_codes} | {f'lib:{c}' for c in trn_codes}

    prots = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
    fam_map = dict(zip(prots['pid'], prots['family']))
    seq_lens = dict(zip(prots['pid'], prots['seq'].str.len()))
    trn_fams = {fam_map[p] for p in trn_pids if p in fam_map}

    mm = pd.read_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'))
    mm_trn = collections.defaultdict(list)
    for p_id, q_id, fid, aln, ev in zip(mm['p'], mm['q'], mm['fident'], mm['alnlen'], mm['evalue']):
        if q_id in trn_pids:
            mm_trn[p_id].append((q_id, fid, aln, ev))
        if p_id in trn_pids:
            mm_trn[q_id].append((p_id, fid, aln, ev))

    fs_sub = pair_df[pair_df['t_pid'].isin(trn_pids)]
    fs_trn = collections.defaultdict(list)
    for r in fs_sub.itertuples(index=False):
        fs_trn[r.q_pid].append(r)

    eval_items = [('domainome', pid[4:], pid) for pid in dom_pids] + [
        ('proteingym', pid[3:], pid) for pid in sorted(pg_pids)
    ]

    summary_rows = []
    for ds_name, item_id, pid in eval_items:
        L = seq_lens.get(pid, 1)
        m_hits = mm_trn.get(pid, [])
        s_hits = fs_trn.get(pid, [])

        is_same = fam_map.get(pid) in trn_fams or any(fid >= 0.90 and (aln / L) >= 0.50 for _, fid, aln, _ in m_hits)
        seq_H = any((aln / L) >= 0.50 and ev <= 1e-3 for _, fid, aln, ev in m_hits)
        seq_h = any((aln / L) >= 0.30 and (ev <= 1e-2 or fid >= 0.30) for _, fid, aln, ev in m_hits)
        cls_seq = 'P' if is_same else ('H' if seq_H else ('h' if seq_h else 'none'))

        st_H = any(r.qcov >= 0.50 and (r.struct_evalue <= 1e-3 or r.qtmscore > 0.70) for r in s_hits)
        st_h = any(r.qcov >= 0.30 and r.struct_evalue <= 1e-2 for r in s_hits)
        cls_both = 'P' if is_same else ('H' if (seq_H or st_H) else ('h' if (seq_h or st_h) else 'none'))

        min_seq_e = min((ev for _, _, _, ev in m_hits), default=np.nan)
        max_seq_f = max((fid for _, fid, _, _ in m_hits), default=np.nan)
        min_st_e = min((r.struct_evalue for r in s_hits), default=np.nan)
        max_qtm = max((r.qtmscore for r in s_hits), default=np.nan)
        max_tm_max = max((r.tm_max for r in s_hits), default=np.nan)
        best_st_t = s_hits[0].t_pid if s_hits else ''
        max_qcov = max((r.qcov for r in s_hits), default=np.nan)

        summary_rows.append(dict(
            dataset=ds_name,
            item_id=item_id,
            pid=pid,
            seq_len=L,
            min_seq_evalue=min_seq_e,
            max_seq_fident=max_seq_f,
            min_struct_evalue=min_st_e,
            max_qtmscore=max_qtm,
            max_tm_max=max_tm_max,
            max_struct_cov_test=max_qcov,
            best_struct_target=best_st_t,
            class_seq_only=cls_seq,
            class_with_struct=cls_both,
        ))

    sum_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUT_DIR, 'structure_homology_domainome_pg.csv')
    sum_df.to_csv(out_csv, index=False)
    for ds_name, g in sum_df.groupby('dataset'):
        print(f'[{ds_name}] seq_only: {g.class_seq_only.value_counts().to_dict()} -> with_struct: {g.class_with_struct.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
