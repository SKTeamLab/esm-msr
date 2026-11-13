"""Per-DMS ProteinGym overlap of the ESM-MSR capped-train inventory (exact, constituent, same protein, and sequence +
Foldseek structure homology by ``lib.classify_hits``), and the effect of removing the overlapping assays on the
ESM-MSR (devel) ProteinGym aggregates.

The aggregate block (``esm_msr_pg_clean.json``, quoted in LEAKAGE_AUDIT_OCT2026.md section 6.1) needs the per-seed
ProteinGym summaries in ``$LEAK_ROOT/pgym_results/unmasked-esm-msr-seed{1,2,3}/sigma1.0/summary.csv`` and
``$LEAK_ROOT/ProteinGym/DMS_substitutions.csv``; it is skipped when they are absent.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, OUT_DIR, WORK_DATA_DIR, classify_hits, expand_lib_pids, make_canon, parse_mut_str

os.makedirs(OUT_DIR, exist_ok=True)
fm = pickle.load(open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'rb'))
maps = fm['maps']
pid_fam = {p: v[0] for p, v in maps.items()}
canon = make_canon(maps)

prots = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
seq_lens = dict(zip(prots['pid'], prots['seq'].str.len()))

mm = pd.read_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'))
mm_adj = collections.defaultdict(list)
for a, b, f, e, al in zip(mm['p'], mm['q'], mm['fident'], mm['evalue'], mm['alnlen']):
    mm_adj[a].append((b, float(f), float(e), int(al)))
    mm_adj[b].append((a, float(f), float(e), int(al)))

fs_adj = collections.defaultdict(list)
fs_pkl = os.path.join(WORK_DATA_DIR, 'foldseek_hits_dom_pg.pkl')
if os.path.exists(fs_pkl):
    fs_df = pd.read_pickle(fs_pkl)
    for r in fs_df.itertuples(index=False):
        fs_adj[r.q_pid].append((r.t_pid, float(r.struct_evalue), float(r.qcov), float(r.qtmscore)))


def canon_keys(pid, mutstr):
    keys = set()
    for wt, pos, mt in parse_mut_str(mutstr) or []:
        c = canon(pid, wt, pos, mt)
        if c is None:
            return None
        keys.add(c)
    return frozenset(keys)


pg = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', 'proteingym.pkl'))
pg['DMS'] = pg['pid'].str.replace('pg:', '', regex=False)

msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
msv = msv[msv['mutation'].notna()]
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))


def add_ms(scaffolds):
    sc_set = set(scaffolds)
    v = collections.defaultdict(set)
    s = collections.defaultdict(set)
    pids = expand_lib_pids(lib, sc_set)
    fams = {pid_fam[p] for p in pids if p in pid_fam}
    sub = msv[msv['code_wt'].isin(sc_set) | msv['code'].isin(sc_set)]
    for code, mutstr in zip(sub['code'], sub['mutation']):
        pid = 'lib:' + str(code)
        if pid not in maps:
            continue
        keys = canon_keys(pid, mutstr)
        if not keys:
            continue
        f = pid_fam[pid]
        v[f].add(keys)
        s[f].update(keys)
        fams.add(f)
    return v, s, fams, pids


INV = {'esm_msr_capped': add_ms(sp['train'])}

pgkey = [canon_keys(pid, mut) if pid in maps else None for pid, mut in zip(pg['pid'], pg['mutation'])]
pg['keys'] = pgkey
pg['family'] = pg['pid'].map(pid_fam)


def dms_homology_flags(dms_id: str, inv_fams: set, inv_pids: set):
    """``(sameprot, H, h, min seq E, min struct E, max struct qTM)`` of one assay's protein against the inventory."""
    pid = 'pg:' + dms_id
    L = seq_lens.get(pid, 1)
    seq = [(q, f, e, al) for q, f, e, al in mm_adj.get(pid, ()) if q in inv_pids]
    st = [(t, e, cov, qtm) for t, e, cov, qtm in fs_adj.get(pid, ()) if t in inv_pids]
    min_se = min((e for _, _, e, _ in seq), default=np.nan)
    min_ste = min((e for _, e, _, _ in st), default=np.nan)
    max_qtm = max((qtm for _, _, _, qtm in st), default=np.nan)
    c = 'sameprot' if pid_fam.get(pid) in inv_fams else classify_hits(
        [(f, al / L, e) for _, f, e, al in seq], [(e, cov, qtm) for _, e, cov, qtm in st])
    return int(c == 'sameprot'), int(c == 'H'), int(c == 'h'), min_se, min_ste, max_qtm


rows = []
for iid, (v, s, fams, pids) in INV.items():
    exact = [0] * len(pg)
    const = [0] * len(pg)
    for i, (fam, k) in enumerate(zip(pg['family'], pg['keys'])):
        if fam in fams and k is not None and k:
            if k in v.get(fam, ()):
                exact[i] = 1
            elif k <= s.get(fam, set()):
                const[i] = 1
    pg['exact'] = exact
    pg['const'] = const
    g = pg.groupby('DMS').agg(n=('exact', 'size'), exact=('exact', 'sum'), const=('const', 'sum')).reset_index()
    g['frac'] = (g['exact'] + g['const']) / g['n']
    g['inventory'] = iid

    h_cols = [dms_homology_flags(dms_id, fams, pids) for dms_id in g['DMS']]
    g['sameprot'] = [t[0] for t in h_cols]
    g['H'] = [t[1] for t in h_cols]
    g['h'] = [t[2] for t in h_cols]
    g['min_seq_evalue'] = [t[3] for t in h_cols]
    g['min_struct_evalue'] = [t[4] for t in h_cols]
    g['max_struct_qtm'] = [t[5] for t in h_cols]

    rows.append(g)
    leak = g[g['frac'] >= 0.5]
    hom_h = g[g['H'] == 1]
    n_dms = len(g)
    print(
        f'{iid:20s} DMS with >=50% rows trained: {len(leak):3d}/{n_dms}; '
        f'strong homolog (H) DMS: {len(hom_h):3d}/{n_dms}; '
        f'rows exact+const {int(g.exact.sum() + g.const.sum()):8d}/{int(g.n.sum())}'
    )

pg_out = pd.concat(rows, ignore_index=True)
pg_out.to_csv(os.path.join(OUT_DIR, 'pg_leak_per_dms.csv'), index=False)

# Optional: effect on ESM-MSR devel aggregates when local per-DMS summaries exist
pg_root = os.environ.get('PROTEINGYM_DIR', os.path.join(LEAK_ROOT, 'ProteinGym'))
pgym_res_dir = os.path.join(LEAK_ROOT, 'pgym_results')
if os.path.exists(os.path.join(pg_root, 'DMS_substitutions.csv')) and os.path.isdir(pgym_res_dir):
    m = pd.read_csv(os.path.join(pg_root, 'DMS_substitutions.csv'))[['DMS_id', 'coarse_selection_type']]
    summ = {}
    for s_idx in (1, 2, 3):
        p_csv = os.path.join(pgym_res_dir, f'unmasked-esm-msr-seed{s_idx}', 'sigma1.0', 'summary.csv')
        if os.path.exists(p_csv):
            d = pd.read_csv(p_csv)
            col = [c for c in d.columns if 'spearman' in c][0]
            summ[s_idx] = d.set_index('DMS_id')[col]
    if summ:
        df = pd.concat(summ.values(), axis=1).mean(axis=1).rename('rho').reset_index().rename(columns={'index': 'DMS_id'})
        df = df.merge(m, on='DMS_id')
        cap_pg = pg_out[pg_out['inventory'] == 'esm_msr_capped']
        leaked_exact = set(cap_pg.loc[cap_pg['frac'] >= 0.5, 'DMS'])
        leaked_hom = set(cap_pg.loc[(cap_pg['sameprot'] == 1) | (cap_pg['H'] == 1), 'DMS'])

        def agg(x):
            cats = x.groupby('coarse_selection_type')['rho'].mean()
            return dict(
                PGavg=float(cats.mean()),
                DMSmean=float(x['rho'].mean()),
                n=int(len(x)),
                **{c: float(cats.get(c, np.nan)) for c in ('Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability')},
            )

        out = {
            'all': agg(df),
            'clean': agg(df[~df['DMS_id'].isin(leaked_exact)]),
            'clean_no_H': agg(df[~df['DMS_id'].isin(leaked_hom)]),
            'leaked': agg(df[df['DMS_id'].isin(leaked_exact)]),
            'n_leaked': len(leaked_exact),
            'n_leaked_or_H': len(leaked_hom),
        }
        with open(os.path.join(OUT_DIR, 'esm_msr_pg_clean.json'), 'w') as fh:
            json.dump(out, fh, indent=2, default=float)
        print(json.dumps(out, indent=2, default=float))
