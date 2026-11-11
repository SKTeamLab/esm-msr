"""Refined per-test-protein homology classification against every training inventory.

Evaluates *all* sequence hits (MMseqs2) and, for Domainome and ProteinGym, *all* 3D structural hits
(Foldseek from ``05d_structure_homology.py``) without E-value shadowing (a short low-coverage domain/motif
hit can never hide a longer qualifying homolog alignment):

  - ``sameprot`` : same >=95%-identity protein family, or any MMseqs2 hit with ``fident >= 0.90``
                   covering ``>= 50%`` of the test protein.
  - ``H`` (strong): any MMseqs2 hit covering ``>= 50%`` of the test protein with ``E <= 1e-3``, OR
                    (on Domainome / ProteinGym) any Foldseek hit covering ``>= 50%`` of the test
                    protein with ``E_struct <= 1e-3`` or test-normalized ``TM > 0.70``.
  - ``h`` (weak)  : if not ``H``, any MMseqs2 hit covering ``>= 30%`` of the test protein with
                    ``E <= 1e-2`` or ``fident >= 0.30``, OR (on Domainome / ProteinGym) any Foldseek
                    hit covering ``>= 30%`` of the test protein with ``E_struct <= 1e-2``.
  - ``low``       : only hits below the ``h`` thresholds (not counted as leakage).
  - ``none``      : no sequence or structural hit to any protein in the inventory.
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, WORK_DATA_DIR, data_file, norm_wt_name

os.makedirs(OUT_DIR, exist_ok=True)
p = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
seqs = dict(zip(p['pid'], p['seq']))
seq_lens = {pid: len(s) for pid, s in seqs.items()}
fam = dict(zip(p['pid'], p['family']))

pairs = pd.read_csv(os.path.join(WORK_DATA_DIR, 'mmseqs_pairs.csv'))
mm_adj = collections.defaultdict(list)
for a, b, f, e, al in zip(pairs['p'], pairs['q'], pairs['fident'], pairs['evalue'], pairs['alnlen']):
    mm_adj[a].append((b, float(f), float(e), int(al)))
    mm_adj[b].append((a, float(f), float(e), int(al)))

fs_adj = collections.defaultdict(list)
fs_pkl = os.path.join(WORK_DATA_DIR, 'foldseek_hits_dom_pg.pkl')
if os.path.exists(fs_pkl):
    fs_df = pd.read_pickle(fs_pkl)
    for r in fs_df.itertuples(index=False):
        fs_adj[r.q_pid].append((r.t_pid, float(r.struct_evalue), float(r.qcov), float(r.qtmscore)))
    print(f'loaded {len(fs_df)} Foldseek structural pairs for Domainome / ProteinGym')

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
INV['esm_msr_capped'] = expand_lib_pids(sp['train'])
for _k in ('me_stage1', 'me_stage2', 'tmpnn', 'spurs_retrain_single', 'spurs_retrain_multi'):
    INV[_k] = INV['esm_msr_capped']

_td = pd.read_csv(data_file('tmpnnd_train_proteins.csv'))
INV['tmpnnd'] = expand_lib_pids(_td.loc[_td['retained'], 'code_wt'].astype(str))

for iid, pcsv in [
    ('spurs_single', 'spurs_single_train_proteins.csv'),
    ('spurs_multi', 'spurs_multi_train_proteins.csv'),
]:
    sm = pd.read_csv(data_file(pcsv))
    INV[iid] = expand_lib_pids(norm_wt_name(x)[1] for x in sm.loc[sm['retained'] == True, 'code_wt'].astype(str))

et = pd.read_csv(data_file('esmtherm_train_proteins.csv'))
INV['esmtherm'] = expand_lib_pids(norm_wt_name(x)[1] for x in et.loc[et['split'].isin(['train', 'val']), 'code'].astype(str))

ifum_f = data_file('ifum_train_proteins.csv')
if os.path.exists(ifum_f):
    fi = pd.read_csv(ifum_f)
    INV['ifum'] = expand_lib_pids(norm_wt_name(x)[1] for x in fi.loc[fi['split'].isin(['train', 'val']), 'code'].astype(str))

tm_up = pd.read_csv(data_file('thermompnn_upstream_train_proteins.csv'))
tm_trn_val = {norm_wt_name(x)[1] for x in tm_up.loc[tm_up['split'].isin(['train', 'val']), 'code'].astype(str)}
tm_test = {norm_wt_name(x)[1] for x in tm_up.loc[tm_up['split'] == 'test', 'code'].astype(str)}
INV['esm3dg_released'] = {f'lib:{c}' for c, cwt in lib['code_wt'].items() if cwt not in tm_test}

INV['ddgemb_s2450'] = {x for x in p['pid'] if str(x).startswith('inv:ddgemb2450:')}
INV['ddmut_s9028'] = {x for x in p['pid'] if str(x).startswith('inv:ddmut9028:')}
INV['ddmut_sm1242'] = {x for x in p['pid'] if str(x).startswith('inv:ddmutsm1242:')}
INV['geoddg_s8754'] = {x for x in p['pid'] if str(x).startswith('inv:geoddg8754:')}

for k in INV:
    print(f'inventory {k:22s}: {len(INV[k]):5d} pids')

tests = {}
tdir = os.path.join(WORK_DATA_DIR, 'testsets')
for f in sorted(os.listdir(tdir)):
    if not f.endswith('.pkl'):
        continue
    ds = f[:-4]
    if ds.startswith('tsu_hyperopt'):
        continue
    tests[ds] = pd.read_pickle(os.path.join(tdir, f))

STRUCT_DATASETS = {'domainome', 'proteingym'}


def classify_pid(pid: str, inv_set: set, inv_fams: set, use_struct: bool) -> str:
    if fam.get(pid) in inv_fams:
        return 'sameprot'
    L = seq_lens.get(pid, 0)
    if L <= 0:
        return 'none'

    has_any = False
    is_H = False
    is_h = False

    for q, f, e, al in mm_adj.get(pid, ()):
        if q not in inv_set:
            continue
        has_any = True
        cov = al / L
        if f >= 0.90 and cov >= 0.50:
            return 'sameprot'
        if cov >= 0.50 and e <= 1e-3:
            is_H = True
        elif cov >= 0.30 and (e <= 1e-2 or f >= 0.30):
            is_h = True

    if use_struct and not is_H:
        for t_pid, st_e, qcov, qtm in fs_adj.get(pid, ()):
            if t_pid not in inv_set:
                continue
            has_any = True
            if qcov >= 0.50 and (st_e <= 1e-3 or qtm > 0.70):
                is_H = True
                break
            elif qcov >= 0.30 and st_e <= 1e-2:
                is_h = True
    elif use_struct and is_H:
        pass

    if is_H:
        return 'H'
    if is_h:
        return 'h'
    return 'low' if has_any else 'none'


rows = []
for iid, inv_set in INV.items():
    inv_fams = {fam[x] for x in inv_set if x in fam}
    for ds, d in tests.items():
        use_struct = ds in STRUCT_DATASETS
        rel = {pid: classify_pid(pid, inv_set, inv_fams, use_struct) for pid in d['pid'].unique()}
        rc = d['pid'].map(rel).value_counts().to_dict()
        rows.append(dict(
            inventory=iid,
            dataset=ds,
            n=len(d),
            **{'n_' + k: rc.get(k, 0) for k in ('sameprot', 'H', 'h', 'low', 'none')},
        ))
        print(
            f'{iid:20s} {ds:20s} P={rc.get("sameprot", 0):7d} '
            f'H={rc.get("H", 0):7d} h={rc.get("h", 0):7d} low={rc.get("low", 0):7d} none={rc.get("none", 0):7d}'
        )

df = pd.DataFrame(rows).fillna(0)
df.to_csv(os.path.join(OUT_DIR, 'homology_classes.csv'), index=False)
print('saved homology_classes.csv')
