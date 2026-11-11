"""Per-DMS ProteinGym leak coverage (exact, constituent, and sequence + Foldseek structure homology)
and the effect of removing contaminated DMS assays on ESM-MSR aggregates."""
import collections
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, OUT_DIR, WORK_DATA_DIR, data_file, norm_wt_name, parse_mut_str

os.makedirs(OUT_DIR, exist_ok=True)
fm = pickle.load(open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'rb'))
maps = fm['maps']
pid_fam = {p: v[0] for p, v in maps.items()}

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

pg = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', 'proteingym.pkl'))
pg['DMS'] = pg['pid'].str.replace('pg:', '', regex=False)

_pos = {}


def canon_one(pid, wt, pos, mt):
    k = (pid, pos, wt)
    if k in _pos:
        rp = _pos[k]
    else:
        fam, mp, rs = maps[pid]
        rp = mp.get(pos)
        if rp is not None and (rp < 1 or rp > len(rs) or rs[rp - 1] != wt):
            alt = None
            for d in (1, -1, 2, -2, 3, -3):
                q = rp + d
                if 1 <= q <= len(rs) and rs[q - 1] == wt:
                    alt = q
                    break
            rp = alt
        _pos[k] = rp
    return None if rp is None else (maps[pid][0], rp, wt, mt)


msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
msv = msv[msv['mutation'].notna()].copy()
msv['n'] = msv['mutation'].str.count(';') + 1
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))


def expand_lib_pids(scaffolds):
    sc = set(scaffolds)
    out = set()
    for c, cwt in lib['code_wt'].items():
        if c in sc or cwt in sc:
            out.add('lib:' + str(c))
    for c in sc:
        if c in lib.index:
            out.add('lib:' + str(c))
    return out


INV = {}


def add_ms(iid, scaffolds, keep=None):
    sc_set = set(scaffolds)
    v = collections.defaultdict(set)
    s = collections.defaultdict(set)
    pids = expand_lib_pids(sc_set)
    fams = {pid_fam[p] for p in pids if p in pid_fam}
    sub = msv[msv['code_wt'].isin(sc_set) | msv['code'].isin(sc_set)]
    if keep is not None:
        sub = sub[keep(sub)]
    for code, mutstr in zip(sub['code'], sub['mutation']):
        pid = 'lib:' + str(code)
        if pid not in maps:
            continue
        keys = set()
        ok = True
        for wt, pos, mt in parse_mut_str(mutstr) or []:
            c = canon_one(pid, wt, pos, mt)
            if c is None:
                ok = False
                break
            keys.add(c)
        if not ok:
            continue
        f = pid_fam[pid]
        v[f].add(frozenset(keys))
        s[f].update(keys)
        fams.add(f)
    INV[iid] = (v, s, fams, pids)


_td = pd.read_csv(data_file('tmpnnd_train_proteins.csv'))
tmpnnd_scaffolds = set(_td.loc[_td['retained'], 'code_wt'].astype(str))

add_ms('esm_msr_capped', sp['train'])
add_ms('me_stage1', sp['train'], lambda d: d['n'] == 1)
add_ms('me_stage2', sp['train'])
add_ms('tmpnn', sp['train'], lambda d: d['n'] == 1)
add_ms('tmpnnd', tmpnnd_scaffolds, lambda d: d['n'] >= 2)
add_ms('spurs_retrain_single', sp['train'], lambda d: d['n'] == 1)
add_ms('spurs_retrain_multi', sp['train'])

tm_up = pd.read_csv(data_file('thermompnn_upstream_train_proteins.csv'))
tmtest = {norm_wt_name(x)[1] for x in tm_up.loc[tm_up['split'] == 'test', 'code'].astype(str)}
add_ms('esm3dg_released', [cwt for cwt in set(lib['code_wt']) if cwt not in tmtest])

spurs_gz = os.path.join(WORK_DATA_DIR, 'spurs_train_mutations.csv.gz')
if os.path.exists(spurs_gz):
    spm = pd.read_csv(spurs_gz)
    for iid, pcsv in [('spurs_single', 'spurs_single_train_proteins.csv'), ('spurs_multi', 'spurs_multi_train_proteins.csv')]:
        v = collections.defaultdict(set)
        s = collections.defaultdict(set)
        sm_df = pd.read_csv(data_file(pcsv))
        pids = expand_lib_pids(norm_wt_name(x)[1] for x in sm_df.loc[sm_df['retained'] == True, 'code_wt'].astype(str))
        fams = {pid_fam[p] for p in pids if p in pid_fam}
        sub = spm[spm['model'] == iid]
        ccol = 'code' if 'code' in sub.columns else 'code_wt'
        for code, mutstr in zip(sub[ccol], sub['mutation']):
            pid = 'lib:' + str(code)
            if pid not in maps:
                continue
            keys = set()
            ok = True
            for wt, pos, mt in parse_mut_str(mutstr) or []:
                c = canon_one(pid, wt, pos, mt)
                if c is None:
                    ok = False
                    break
                keys.add(c)
            if not ok:
                continue
            f = pid_fam[pid]
            v[f].add(frozenset(keys))
            s[f].update(keys)
            fams.add(f)
        INV[iid] = (v, s, fams, pids)

pgkey = []
for pid, mut in zip(pg['pid'], pg['mutation']):
    if pid not in maps:
        pgkey.append(None)
        continue
    keys = set()
    ok = True
    for wt, pos, mt in parse_mut_str(mut) or []:
        c = canon_one(pid, wt, pos, mt)
        if c is None:
            ok = False
            break
        keys.add(c)
    pgkey.append(frozenset(keys) if ok else None)
pg['keys'] = pgkey
pg['family'] = pg['pid'].map(pid_fam)


def dms_homology_flags(dms_id: str, inv_fams: set, inv_pids: set):
    pid = 'pg:' + dms_id
    L = seq_lens.get(pid, 1)
    same = int(pid_fam.get(pid) in inv_fams)
    is_H = False
    is_h = False
    min_se = np.nan
    for q, f, e, al in mm_adj.get(pid, ()):
        if q not in inv_pids:
            continue
        if np.isnan(min_se) or e < min_se:
            min_se = e
        cov = al / L
        if f >= 0.90 and cov >= 0.50:
            same = 1
        elif cov >= 0.50 and e <= 1e-3:
            is_H = True
        elif cov >= 0.30 and (e <= 1e-2 or f >= 0.30):
            is_h = True
    min_ste = np.nan
    max_qtm = np.nan
    for t_pid, st_e, qcov, qtm in fs_adj.get(pid, ()):
        if t_pid not in inv_pids:
            continue
        if np.isnan(min_ste) or st_e < min_ste:
            min_ste = st_e
        if np.isnan(max_qtm) or qtm > max_qtm:
            max_qtm = qtm
        if qcov >= 0.50 and (st_e <= 1e-3 or qtm > 0.70):
            is_H = True
        elif qcov >= 0.30 and st_e <= 1e-2:
            is_h = True
    if same:
        return 1, 0, 0, min_se, min_ste, max_qtm
    if is_H:
        return 0, 1, 0, min_se, min_ste, max_qtm
    if is_h:
        return 0, 0, 1, min_se, min_ste, max_qtm
    return 0, 0, 0, min_se, min_ste, max_qtm


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
