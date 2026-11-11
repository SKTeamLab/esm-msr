"""ESM3dG (MGnify Stability fine-tune) vs the benchmark: exact-sequence and non-shadowed homology overlap."""
import collections
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, WORK_DATA_DIR, data_file, norm_wt_name

os.makedirs(OUT_DIR, exist_ok=True)
h = lambda s: hashlib.sha1(s.encode()).hexdigest()

m8_path = data_file('test_vs_mgnify_s75.m8')
if not os.path.exists(m8_path):
    m8_path = data_file('test_vs_mgnify.m8')
m8 = pd.read_csv(
    m8_path, sep='\t', header=None,
    names=['q', 't', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits'],
)
print('MGnify MMseqs2 hits:', len(m8))
m8['cov'] = m8['alnlen'] / m8['qlen'].replace(0, np.nan)

train_fa = data_file('mgnify_train.fa')
train_hashes = set()
for line in open(train_fa):
    if not line.startswith('>'):
        train_hashes.add(h(line.strip()))

p = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
tq = p[p['source'].str.startswith('test:')].copy()
tq['h'] = tq['seq'].map(h)
exact = set(tq.loc[tq['h'].isin(train_hashes), 'pid'])
print('test proteins with an exact MGnify sequence:', len(exact), sorted(exact)[:10])

# Evaluate all hits per query pid without E-value shadowing
hits_by_q = collections.defaultdict(list)
for q, f, c, e in zip(m8['q'], m8['fident'], m8['cov'], m8['evalue']):
    hits_by_q[q].append((float(f), float(c), float(e)))

cls = {}
best_qual_hit = {}
for q, qhits in hits_by_q.items():
    if q in exact:
        cls[q] = 'P'
        best_qual_hit[q] = min(qhits, key=lambda x: (x[2], -x[0]))
        continue
    h_strong = [x for x in qhits if x[1] >= 0.50 and x[2] <= 1e-3]
    if h_strong:
        cls[q] = 'H'
        best_qual_hit[q] = min(h_strong, key=lambda x: (x[2], -x[0]))
        continue
    h_weak = [x for x in qhits if x[1] >= 0.30 and (x[2] <= 1e-2 or x[0] >= 0.30)]
    if h_weak:
        cls[q] = 'h'
        best_qual_hit[q] = min(h_weak, key=lambda x: (x[2], -x[0]))
        continue
    cls[q] = 'low'
    best_qual_hit[q] = min(qhits, key=lambda x: (x[2], -x[0]))

for q in exact:
    cls[q] = 'P'

rows = []
tdir = os.path.join(WORK_DATA_DIR, 'testsets')
for f in sorted(os.listdir(tdir)):
    if not f.endswith('.pkl'):
        continue
    ds = f[:-4]
    if ds.startswith('tsu_hyperopt'):
        continue
    d = pd.read_pickle(os.path.join(tdir, f))
    d_cls = d['pid'].map(lambda x: cls.get(x, 'none'))
    rc = d_cls.value_counts().to_dict()
    rows.append(dict(
        inventory='esm3dg_mgnify',
        dataset=ds,
        n=len(d),
        n_sameprot=rc.get('P', 0),
        n_H=rc.get('H', 0),
        n_h=rc.get('h', 0),
        n_low=rc.get('low', 0),
        n_none=rc.get('none', 0),
    ))
    print(
        f'{ds:22s} n={len(d):8d} P={rc.get("P", 0):6d} H={rc.get("H", 0):6d} '
        f'h={rc.get("h", 0):6d} low={rc.get("low", 0):6d} none={rc.get("none", 0):6d}'
    )

df = pd.DataFrame(rows)
hc_csv = os.path.join(OUT_DIR, 'homology_classes.csv')
old = pd.read_csv(hc_csv)
old = old[old['inventory'] != 'esm3dg_mgnify']
df_all = pd.concat([old, df], ignore_index=True)
df_all.to_csv(hc_csv, index=False)

# Per-scaffold summary for the 33 Homology-Restricted MegaScale test scaffolds (leak_audit_esm3dg_libraries.csv)
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
ttest = pd.read_pickle(os.path.join(tdir, 'tsu_capped_test.pkl'))
ttest['code_wt'] = [lib['code_wt'].get(x[4:], norm_wt_name(x[4:])[1]) for x in ttest['pid']]
rank_order = {'P': 4, 'H': 3, 'h': 2, 'low': 1, 'none': 0}

lib_rows = []
for sc in sorted(sp['test']):
    sub = ttest[ttest['code_wt'] == sc]
    n_rows = int(len(sub))
    if n_rows == 0:
        lib_rows.append(dict(library=sc, n_rows=0, best_query='', fident=np.nan, evalue=np.nan, cov_test=np.nan, **{'class': 'none'}))
        continue
    pids = sorted(sub['pid'].unique())
    best_pid = max(
        pids,
        key=lambda qp: (
            rank_order.get(cls.get(qp, 'none'), 0),
            -(best_qual_hit[qp][2] if qp in best_qual_hit else 1e9),
        ),
    )
    c_label = cls.get(best_pid, 'none')
    if best_pid in best_qual_hit:
        fid_v, cov_v, ev_v = best_qual_hit[best_pid]
        lib_rows.append(dict(
            library=sc, n_rows=n_rows, best_query=best_pid,
            fident=fid_v, evalue=ev_v, cov_test=cov_v, **{'class': c_label},
        ))
    else:
        lib_rows.append(dict(library=sc, n_rows=n_rows, best_query='', fident=np.nan, evalue=np.nan, cov_test=np.nan, **{'class': 'none'}))

pd.DataFrame(lib_rows).to_csv(os.path.join(OUT_DIR, 'esm3dg_tsu_libraries.csv'), index=False)
print('saved updated homology_classes.csv and esm3dg_tsu_libraries.csv')
