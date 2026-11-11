"""Coarse per-test-protein relation to the capped-train scaffold set using the family/homolog graph.

Note: This summary table is a quick sanity check on capped-train scaffolds; the benchmark leak
matrix uses the per-inventory ``H``/``h`` classifications from ``11_homology_classes.py``.
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, WORK_DATA_DIR

os.makedirs(OUT_DIR, exist_ok=True)
p = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
edges = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'protein_homologs.pkl'))
libs = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
split = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
train_scaffolds = set(split['train'])
train_pids = {f'lib:{c}' for c, cwt in libs['code_wt'].items() if cwt in train_scaffolds} | {
    f'lib:{c}' for c in train_scaffolds
}

fam2libs = collections.defaultdict(set)
for pid, fam in zip(p['pid'], p['family']):
    if pid in train_pids:
        fam2libs[fam].add(pid)

h_by_pid = collections.defaultdict(list)
for a, b, f, e, cv, h, w in zip(
    edges['a'], edges['b'], edges['fident'], edges['evalue'], edges['cov'], edges['homolog'], edges['weak']
):
    h_by_pid[a].append((b, f, e, cv, h, w))
    h_by_pid[b].append((a, f, e, cv, h, w))


def relation(pid):
    fam = p.loc[p['pid'] == pid, 'family'].iloc[0]
    if fam in fam2libs:
        return 'family', 1.0, 0.0, sorted(fam2libs[fam])[0]
    best = None
    for q, f, e, cv, h, w in h_by_pid.get(pid, []):
        if q in train_pids:
            if best is None or e < best[2]:
                best = (q, f, e, cv, h, w)
    if best is None:
        return 'none', 0.0, 1.0, None
    q, f, e, cv, h, w = best
    if h:
        return 'homolog', f, e, q
    if w:
        return 'weak', f, e, q
    return 'none', f, e, q


rows = []
for f in sorted(os.listdir(os.path.join(WORK_DATA_DIR, 'testsets'))):
    if not f.endswith('.pkl'):
        continue
    ds = f[:-4]
    if ds.startswith('tsu_hyperopt'):
        continue
    d = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'testsets', f))
    rel = {pid: relation(pid) for pid in d['pid'].unique()}
    d['rel'] = d['pid'].map(lambda x: rel[x][0])
    d['bestID'] = d['pid'].map(lambda x: rel[x][1])
    g = d.groupby('rel').size().to_dict()
    rows.append(dict(dataset=ds, n=len(d), **g))
    print(f'{ds:22s} rows={len(d):8d} ' + ' '.join(f'{k}={v}' for k, v in sorted(g.items())))

pd.DataFrame(rows).fillna(0).to_csv(os.path.join(OUT_DIR, 'relation_to_capped_train.csv'), index=False)
