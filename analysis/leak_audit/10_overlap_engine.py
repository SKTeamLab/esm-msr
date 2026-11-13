"""Mutation-level overlap of the Megascale-trained inventories (ESM-MSR, ESM-MSR-small, EsmTherm, IFUM) with every
benchmark dataset.

Inventories (every substitution row of every training scaffold, mutant-background libraries included; an upper
bound on what any one checkpoint consumed, so exact counts err upward):
  - ``esm_msr_capped``         : the 172 capped-train scaffolds of ``data/splits_oct06_capped.pkl`` (ESM-MSR released,
                                 baseline and devel checkpoints, every head and sigma).
  - ``esm_msr_small_hyperopt`` : the train libraries of ``data/hyperopt_splits.pkl`` (ESM-MSR-small, March reference).

For each test row (canonicalised into its >=95%-identity family by ``08_families.py``) the engine counts:
  - ``exact``       : identical variant (same family, identical canonical mutation set) in training
  - ``constituent`` : every mutation of the row trained in the same family, but not this exact combination
  - ``sameprot``    : the family is in training, its mutations are not
  - ``unmapped``    : the test protein has no family map (not related to any inventory protein)

EsmTherm and IFUM are evaluated by an exact join on the full mutant sequence (see the block at the end): ``exact``
plus ``val`` (rows only in the model's validation data). Protein-level homology (``P`` / ``H`` / ``h``) is classified
separately by ``11_homology_classes.py``.
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (OUT_DIR, REPO_ROOT, WORK_DATA_DIR, apply_mutations, check_against_seq, data_file, expand_lib_pids,
                 load_testsets, make_canon, norm_wt_name, parse_mut_str)

os.makedirs(OUT_DIR, exist_ok=True)
fm = pickle.load(open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'rb'))
maps = fm['maps']
pid_fam = {pid: v[0] for pid, v in maps.items()}
canon = make_canon(maps)


def canon_keys(pid, mutstr):
    """Frozenset of canonical mutation keys, or None when any mutation cannot be mapped."""
    keys = set()
    for wt, pos, mt in parse_mut_str(mutstr) or []:
        c = canon(pid, wt, pos, mt)
        if c is None:
            return None
        keys.add(c)
    return frozenset(keys)


msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
msv = msv[msv['mutation'].notna()]
lib = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
hyp = pd.read_pickle(os.path.join(REPO_ROOT, 'data', 'hyperopt_splits.pkl'))

SCAFFOLDS = {
    'esm_msr_capped': set(sp['train']),
    'esm_msr_small_hyperopt': {norm_wt_name(x)[1] for x in hyp['train']},
}


class Inventory:
    def __init__(self, iid, scaffolds):
        self.iid = iid
        self.variants = collections.defaultdict(set)
        self.singles = collections.defaultdict(set)
        self.families = {pid_fam[p] for p in expand_lib_pids(lib, scaffolds) if p in pid_fam}
        sub = msv[msv['code_wt'].isin(scaffolds) | msv['code'].isin(scaffolds)]
        for code, mutstr in zip(sub['code'], sub['mutation']):
            pid = 'lib:' + str(code)
            if pid not in maps:
                continue
            keys = canon_keys(pid, mutstr)
            if not keys:
                continue
            fam = pid_fam[pid]
            self.variants[fam].add(keys)
            self.singles[fam].update(keys)
            self.families.add(fam)


INV = [Inventory(iid, sc) for iid, sc in SCAFFOLDS.items()]
for inv in INV:
    print(f'{inv.iid:24s} families={len(inv.families):4d} variants={sum(len(v) for v in inv.variants.values()):7d}')
inv_fams = set().union(*(inv.families for inv in INV))

rows = []
for ds, d in load_testsets().items():
    fams = [pid_fam.get(pid) for pid in d['pid']]
    keys = [canon_keys(pid, mut) if fam in inv_fams else None for pid, mut, fam in zip(d['pid'], d['mutation'], fams)]
    for inv in INV:
        c = collections.Counter()
        for fam, k in zip(fams, keys):
            if fam is None:
                c['unmapped'] += 1
            elif fam not in inv.families:
                c['other'] += 1
            elif k and k in inv.variants.get(fam, ()):
                c['exact'] += 1
            elif k and k <= inv.singles.get(fam, set()):
                c['constituent'] += 1
            else:
                c['sameprot'] += 1
        rows.append(dict(inventory=inv.iid, dataset=ds, n=len(d), **{k: c.get(k, 0) for k in ('exact', 'constituent', 'sameprot', 'other', 'unmapped')}))
        print(f'  {inv.iid:24s} {ds:42s} n={len(d):8d} exact={c["exact"]:7d} constituent={c["constituent"]:6d} sameprot={c["sameprot"]:7d}')

# ---------------------------------------------------------------- EsmTherm and IFUM: exact variant sequences
# EsmTherm publishes every training / validation sequence (PLoS Comput Biol 2024, S2 Spreadsheet
# pcbi.1012248.s006.csv: 527,785 rows with split + sequence), so its identical-variant overlap is verified by an
# exact join on the full mutant sequence. IFUM publishes only its domain split (322 train / 86 val-or-test
# Dataset2 domains); every Dataset2 row of a train domain is in its training pool, so the same join against those
# rows gives the identical-variant ceiling (reported as M?; row filters and the released checkpoint's exact split are
# not public). Rows that hit only the validation sequences are V (EsmTherm) or V? (IFUM's merged val/test block).
def mutant_seq(seq, mutstr):
    muts = parse_mut_str(mutstr)
    if not muts or not check_against_seq(seq, muts)[0]:
        return None
    return apply_mutations(seq, muts)


et = pd.read_csv(data_file('raw', 'esmtherm_pcbi.1012248.s006.csv'), usecols=['split', 'sequence'])
et_train, et_val = set(et.loc[et['split'] == 'train', 'sequence']), set(et.loc[et['split'] == 'val', 'sequence'])
print(f'EsmTherm S2: {len(et)} rows, {len(et_train)} train / {len(et_val)} val sequences')

ifum = pd.read_csv(data_file('ifum_train_proteins.csv'))
d2 = msv[pd.to_numeric(msv['dG_ML'], errors='coerce').notna()]
d2_seq = [mutant_seq(lib.at[c, 'wt_seq'], m) if c in lib.index else None for c, m in zip(d2['code'], d2['mutation'])]
d2 = d2.assign(mseq=d2_seq)
if_train = set(d2.loc[d2['code_wt'].isin(set(ifum.loc[ifum['split'] == 'train', 'code'])), 'mseq'].dropna())
if_valtest = set(d2.loc[d2['code_wt'].isin(set(ifum.loc[ifum['split'] != 'train', 'code'])), 'mseq'].dropna())
print(f'IFUM Dataset2 pool: {len(if_train)} train-domain / {len(if_valtest)} val-or-test-domain variant sequences')

for ds, d in load_testsets().items():
    ms = [mutant_seq(s, m) for s, m in zip(d['seq'], d['mutation'])]
    for iid, tr, va in (('esmtherm', et_train, et_val), ('ifum', if_train, if_valtest)):
        ex = sum(1 for x in ms if x and x in tr)
        vo = sum(1 for x in ms if x and x in va and x not in tr)
        rows.append(dict(inventory=iid, dataset=ds, n=len(d), exact=ex, constituent=0, sameprot=0, other=0, unmapped=0, val=vo))
        if ex or vo:
            print(f'  {iid:24s} {ds:42s} n={len(d):8d} exact={ex:7d} val-only={vo:7d}')

pd.DataFrame(rows).fillna({'val': 0}).to_csv(os.path.join(OUT_DIR, 'inventory_overlaps.csv'), index=False)
print('saved inventory_overlaps.csv')
