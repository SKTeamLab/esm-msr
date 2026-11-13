"""Refined per-test-protein homology classification against every training inventory.

Evaluates *all* sequence hits (MMseqs2) and, for Domainome and ProteinGym, *all* 3D structural hits
(Foldseek from ``07_structure_homology.py``, whose targets are the 172 capped-train scaffold structures) without E-value shadowing (a short low-coverage domain/motif
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

The rule itself lives in ``lib.classify_hits`` and is shared with ``14_esm3dg_homology.py`` and ``17_proteingym.py``.
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR, classify_hits, data_file, expand_lib_pids, load_testsets, norm_wt_name

os.makedirs(OUT_DIR, exist_ok=True)
p = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
seq_lens = {pid: len(s) for pid, s in zip(p['pid'], p['seq'])}
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
hyp = pd.read_pickle(os.path.join(REPO_ROOT, 'data', 'hyperopt_splits.pkl'))


def split_scaffolds(csv_name, splits=('train', 'val')):
    d = pd.read_csv(data_file(csv_name))
    return {norm_wt_name(x)[1] for x in d.loc[d['split'].isin(splits), 'code'].astype(str)}


INV = {
    'esm_msr_capped': expand_lib_pids(lib, sp['train']),
    'esm_msr_small_hyperopt': expand_lib_pids(lib, {norm_wt_name(x)[1] for x in hyp['train']}),
    'esmtherm': expand_lib_pids(lib, split_scaffolds('esmtherm_train_proteins.csv')),
    'ifum': expand_lib_pids(lib, split_scaffolds('ifum_train_proteins.csv')),
}
for iid, prefix in [('ddgemb_s2450', 'inv:ddgemb2450:'), ('ddmut_s9028', 'inv:ddmut9028:'),
                    ('ddmut_sm1662', 'inv:ddmutsm1662:'), ('geoddg_s8754', 'inv:geoddg8754:')]:
    INV[iid] = {x for x in p['pid'] if str(x).startswith(prefix)}
INV['ddmut'] = INV['ddmut_s9028'] | INV['ddmut_sm1662']  # the deployed DDMut: both models' training proteins
for k, v in INV.items():
    print(f'inventory {k:24s}: {len(v):5d} pids')

tests = load_testsets()

# GeoDDG-3D's encoder starts from GeoFitness, trained on 74 DMS fitness sets (52 MaveDB + 22 DeepSequence; GeoStab
# data/dms/dms.csv, 70/10/20 per-mutation split within each protein). Only the wild types are audited
# (raw/geofitness_dms_wt.fasta): a test protein whose sequence equals, contains or is contained in one (>= 30 aa) is
# the same protein (P) for the GeoDDG-3D inventory.
gf, name = {}, ''
for line in open(data_file('raw', 'geofitness_dms_wt.fasta')):
    if line.startswith('>'):
        name = line[1:].strip()
    elif line.strip():
        gf[name] = gf.get(name, '') + line.strip()
GEOFIT = set()
grows = []
for ds, d in tests.items():
    prot = d[['pid', 'seq']].drop_duplicates('pid')
    hit = {pid for pid, s in prot.itertuples(index=False) if len(s) >= 30 and any(s in g or g in s for g in gf.values())}
    GEOFIT |= hit
    m = d['pid'].isin(hit)
    grows.append(dict(inventory='geofitness_dms', dataset=ds, n=len(d), n_sameprot=int(m.sum()), n_proteins=len(hit)))
pd.DataFrame(grows).to_csv(os.path.join(OUT_DIR, 'geofitness_proteins.csv'), index=False)
print('GeoFitness DMS proteins among test proteins:', len(GEOFIT))

STRUCT_DATASETS = {'domainome', 'domainome_low_id', 'proteingym'}


def classify_pid(pid, inv_set, inv_fams, use_struct, extra_same=frozenset()):
    if fam.get(pid) in inv_fams or pid in extra_same:
        return 'sameprot'
    L = seq_lens.get(pid, 0)
    if L <= 0:
        return 'none'
    seq_hits = [(f, al / L, e) for q, f, e, al in mm_adj.get(pid, ()) if q in inv_set]
    struct_hits = [(e, cov, qtm) for t, e, cov, qtm in fs_adj.get(pid, ()) if t in inv_set] if use_struct else []
    return classify_hits(seq_hits, struct_hits)


rows = []
for iid, inv_set in INV.items():
    inv_fams = {fam[x] for x in inv_set if x in fam}
    extra = GEOFIT if iid == 'geoddg_s8754' else frozenset()
    for ds, d in tests.items():
        rel = {pid: classify_pid(pid, inv_set, inv_fams, ds in STRUCT_DATASETS, extra) for pid in d['pid'].unique()}
        rc = d['pid'].map(rel).value_counts().to_dict()
        rows.append(dict(inventory=iid, dataset=ds, n=len(d), **{'n_' + k: rc.get(k, 0) for k in ('sameprot', 'H', 'h', 'low', 'none')}))
        print(f'{iid:24s} {ds:42s} P={rc.get("sameprot", 0):7d} H={rc.get("H", 0):7d} h={rc.get("h", 0):7d}')

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'homology_classes.csv'), index=False)
print('saved homology_classes.csv')
