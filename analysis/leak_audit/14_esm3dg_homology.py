"""ESM3dG (MGnify Stability fine-tune) vs the benchmark: protein-level relation of every test protein to the
960,216 MGnify training sequences, by the same rule as ``11_homology_classes.py`` (``lib.classify_hits``; an exact
sequence match also counts as same protein). Writes ``esm3dg_homology.csv`` (same columns as ``homology_classes.csv``)
and the per-scaffold summary over the 33 capped-test scaffolds (``esm3dg_tsu_libraries.csv``).
Mutation-level (exact variant sequence) overlap is ``15_esm3dg_exact.py``."""
import collections
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, WORK_DATA_DIR, classify_hits, data_file, load_testsets, norm_wt_name

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

# Evaluate all hits per query pid (no E-value shadowing)
hits_by_q = collections.defaultdict(list)
for q, f, c, e in zip(m8['q'], m8['fident'], m8['cov'], m8['evalue']):
    hits_by_q[q].append((float(f), float(c), float(e)))

cls = {q: classify_hits(qhits) for q, qhits in hits_by_q.items()}
cls.update({q: 'sameprot' for q in exact})
LABEL = {'sameprot': 'P', 'H': 'H', 'h': 'h', 'low': 'low', 'none': 'none'}
cls = {q: LABEL[c] for q, c in cls.items()}


def best_hit(q):
    """Lowest-E hit among those that set the class (for the per-scaffold table)."""
    qhits = hits_by_q.get(q, [])
    c = cls.get(q, 'none')
    keep = {
        'P': [x for x in qhits if x[0] >= 0.90 and x[1] >= 0.50],
        'H': [x for x in qhits if x[1] >= 0.50 and x[2] <= 1e-3],
        'h': [x for x in qhits if x[1] >= 0.30 and (x[2] <= 1e-2 or x[0] >= 0.30)],
    }.get(c) or qhits
    return min(keep, key=lambda x: (x[2], -x[0])) if keep else None


best_qual_hit = {q: best_hit(q) for q in hits_by_q}

rows = []
tests = load_testsets()
for ds, d in tests.items():
    rc = d['pid'].map(lambda x: cls.get(x, 'none')).value_counts().to_dict()
    rows.append(dict(inventory='esm3dg_mgnify', dataset=ds, n=len(d), n_sameprot=rc.get('P', 0), n_H=rc.get('H', 0),
                     n_h=rc.get('h', 0), n_low=rc.get('low', 0), n_none=rc.get('none', 0)))
    print(f'{ds:42s} n={len(d):8d} P={rc.get("P", 0):6d} H={rc.get("H", 0):6d} h={rc.get("h", 0):6d}')
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'esm3dg_homology.csv'), index=False)

# Per-scaffold summary for the 33 Homology-Restricted MegaScale test scaffolds (leak_audit_esm3dg_libraries.csv)
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
ttest = tests['tsu_capped_test'].copy()
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
print('saved esm3dg_homology.csv and esm3dg_tsu_libraries.csv')
