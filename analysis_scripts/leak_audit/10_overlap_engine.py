"""Core leak-overlap engine for MegaScale-based training inventories.

For every MegaScale training inventory and every benchmark dataset, counts the test rows whose:
  - ``exact``       : identical variant (same protein family + identical canonical mutation set) was trained
  - ``constituent`` : every mutation in the row was trained somewhere in the same protein family, but not
                      this exact combination (e.g. a test double whose two singles were trained)
  - ``sameprot``    : the protein family is in training, none of its mutations were
  - ``homolog``     : coarse family-graph homolog (superseded in the final matrix by ``11_homology_classes.py``)
  - ``weak``        : coarse family-graph weak hit (superseded in the final matrix by ``11_homology_classes.py``)
  - ``none``        : no detected relation
"""
import collections
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR, data_file, norm_wt_name, parse_mut_str

os.makedirs(OUT_DIR, exist_ok=True)
fm = pickle.load(open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'rb'))
maps, seqs, reps = fm['maps'], fm['seqs'], fm['reps']
pid_fam = {pid: v[0] for pid, v in maps.items()}

_pos_cache = {}


def canon_one(pid, wt, pos, mt):
    """Canonical ``(family, rep_pos, wt, mt)`` or ``None``."""
    key = (pid, pos, wt)
    if key in _pos_cache:
        rp = _pos_cache[key]
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
        _pos_cache[key] = rp
    if rp is None:
        return None
    return (maps[pid][0], rp, wt, mt)


_parsed = {}


def parse_muts(s):
    if s in _parsed:
        return _parsed[s]
    m = parse_mut_str(s) or []
    _parsed[s] = m
    return m


class Inventory:
    def __init__(self, iid, description, provenance, variants=None, protein_pids=None, exact_status='verified'):
        self.iid = iid
        self.description = description
        self.provenance = provenance
        self.exact_status = exact_status
        self.variants = collections.defaultdict(set)
        self.singles = collections.defaultdict(set)
        self.protein_pids = set(protein_pids or [])
        self.families = set()
        for pid in self.protein_pids:
            if pid in pid_fam:
                self.families.add(pid_fam[pid])
        if variants:
            for pid, mutstr in variants:
                if pid not in maps:
                    continue
                muts = parse_muts(mutstr)
                if not muts:
                    continue
                keys = set()
                ok = True
                for wt, pos, mt in muts:
                    c = canon_one(pid, wt, pos, mt)
                    if c is None:
                        ok = False
                        break
                    keys.add(c)
                if not ok:
                    continue
                fam = maps[pid][0]
                fs = frozenset(keys)
                self.variants[fam].add(fs)
                self.singles[fam].update(keys)
                self.families.add(fam)
        self.singles_fams = set(self.singles)


msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
msv = msv[msv['mutation'].notna()].copy()
msv['n'] = msv['mutation'].str.count(';') + 1
sp = pickle.load(open(os.path.join(WORK_DATA_DIR, 'capped_split_sets.pkl'), 'rb'))
capped = set(sp['train'])
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


INV = []


def ms_inv(iid, desc, codes, keep=None, prov='', exact_status='verified', prot_codes=None):
    codes_set = set(codes)
    sub = msv[msv['code_wt'].isin(codes_set) | msv['code'].isin(codes_set)]
    if keep is not None:
        sub = sub[keep(sub)]
    variants = [('lib:' + c, m) for c, m in zip(sub['code'], sub['mutation'])]
    prot = expand_lib_pids(prot_codes if prot_codes is not None else codes_set)
    return Inventory(iid, desc, prov, variants=variants, protein_pids=prot, exact_status=exact_status)


_td = pd.read_csv(data_file('tmpnnd_train_proteins.csv'))
tmpnnd_scaffolds = set(_td.loc[_td['retained'], 'code_wt'].astype(str))

INV.append(ms_inv(
    'esm_msr_capped',
    'ESM-MSR (released & devel), all substitutions of the 172 capped-train scaffolds',
    capped,
    prov='splits_oct06_capped.pkl train + Tsuboyama2023 Dataset2+3',
))
INV.append(ms_inv(
    'me_stage1',
    'MutateEverything stage 1 (singles), capped-train scaffolds',
    capped,
    keep=lambda d: d['n'] == 1,
    prov='capped train; ME recipe stage1 = singles',
))
INV.append(ms_inv(
    'me_stage2',
    'MutateEverything stage 2 (singles then doubles), capped-train scaffolds',
    capped,
    prov='capped train; stage2 initialised from stage1',
))
INV.append(ms_inv(
    'tmpnn',
    'ThermoMPNN additive (singles), capped-train scaffolds',
    capped,
    keep=lambda d: d['n'] == 1,
    prov='capped train; train_single.yaml',
))
INV.append(ms_inv(
    'tmpnnd',
    'ThermoMPNN-D (doubles + Rosetta single-augmented scaffolds), 134 of 172 capped-train scaffolds',
    tmpnnd_scaffolds,
    keep=lambda d: d['n'] >= 2,
    prot_codes=tmpnnd_scaffolds,
    prov='capped train; train_epistatic.yaml; 05c_tmpnnd_inventory.py',
))
INV.append(ms_inv(
    'spurs_retrain_single',
    'SPURS retrained (single model: single + native_cond rows), capped-train scaffolds',
    capped,
    keep=lambda d: d['n'] == 1,
    prov='capped train; datamodule items, singles',
))
INV.append(ms_inv(
    'spurs_retrain_multi',
    'SPURS-multi retrained (initialised from same-seed single model, then doubles), capped-train scaffolds',
    capped,
    prov='capped train; datamodule items, doubles after single-model initialisation',
))
INV.append(ms_inv(
    'esm3dg_released',
    'ESM3dG (hypothetical MegaScale hypothesis; superseded by esm3dg_mgnify)',
    list(lib.index),
    prov='superseded diagnostic inventory',
    exact_status='likely-unverified',
))

# Upstream SPURS checkpoints (kept for historical comparison with retrained SPURS)
spurs_muts_gz = os.path.join(WORK_DATA_DIR, 'spurs_train_mutations.csv.gz')
if os.path.exists(spurs_muts_gz):
    spm_all = pd.read_csv(spurs_muts_gz)
    for iid, desc, pcsv in [
        ('spurs_single', 'Published SPURS single-mutant checkpoint (upstream ThermoMPNN split + m8 de-leak)', 'spurs_single_train_proteins.csv'),
        ('spurs_multi', 'Published SPURS multi-mutant checkpoint (upstream ThermoMPNN split + m8 de-leak)', 'spurs_multi_train_proteins.csv'),
    ]:
        sm = pd.read_csv(data_file(pcsv))
        use_scaffolds = {norm_wt_name(x)[1] for x in sm.loc[sm['retained'] == True, 'code_wt'].astype(str)}
        muts = spm_all[spm_all['model'] == iid]
        ccol = 'code' if 'code' in muts.columns else 'code_wt'
        variants = [('lib:' + str(c), m) for c, m in zip(muts[ccol], muts['mutation'])]
        INV.append(Inventory(iid, desc, 'SPURS_mod upstream split', variants=variants, protein_pids=expand_lib_pids(use_scaffolds)))

# EsmTherm (protein-level)
et = pd.read_csv(data_file('esmtherm_train_proteins.csv'))
et_scaffolds = {norm_wt_name(x)[1] for x in et.loc[et['split'].isin(['train', 'val']), 'code'].astype(str)}
INV.append(Inventory(
    'esmtherm',
    'EsmTherm (K50_dG Dataset1+2; mutant-level 80/10/10, train+val domains)',
    'datasets/split/wildtype_split.csv; mutation-level rows unavailable',
    protein_pids=expand_lib_pids(et_scaffolds),
    exact_status='protein-only',
))

# IFUM (protein-level)
ifum_csv = data_file('ifum_train_proteins.csv')
if os.path.exists(ifum_csv):
    fi = pd.read_csv(ifum_csv)
    if_scaffolds = {norm_wt_name(x)[1] for x in fi.loc[fi['split'].isin(['train', 'val']), 'code'].astype(str)}
    INV.append(Inventory(
        'ifum',
        'IFUM (MegaScale Dataset2; clustered split, 322 train, 86 val/test scaffolds)',
        'IFUM published dataset_splits.pdf',
        protein_pids=expand_lib_pids(if_scaffolds),
        exact_status='protein-only',
    ))

# Legacy hyperopt ESM-MSR-small
hyp = pd.read_pickle(os.path.join(REPO_ROOT, 'data', 'hyperopt_splits.pkl'))
hcodes = {norm_wt_name(x)[1] for x in hyp['train']}
INV.append(ms_inv(
    'esm_msr_small_hyperopt',
    'ESM-MSR-small (March reference), hyperopt-train libraries',
    hcodes,
    prov='hyperopt_splits.pkl train (legacy split; out of Oct-split audit scope)',
))

with open(os.path.join(WORK_DATA_DIR, 'inventories_megascale.pkl'), 'wb') as f:
    pickle.dump(INV, f)
print('inventories:', [(i.iid, len(i.families), len(i.variants)) for i in INV])

edges = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'protein_homologs.pkl'))
fam_h = collections.defaultdict(set)
fam_w = collections.defaultdict(set)
for a, b, h, w in zip(edges['a'], edges['b'], edges['homolog'], edges['weak']):
    fa, fb = pid_fam.get(a), pid_fam.get(b)
    if fa is None or fb is None or fa == fb:
        continue
    if h:
        fam_h[fa].add(fb)
        fam_h[fb].add(fa)
    if w:
        fam_w[fa].add(fb)
        fam_w[fb].add(fa)

all_inv_fams = set()
for inv in INV:
    all_inv_fams |= inv.families


def canon_testsets():
    out = {}
    tdir = os.path.join(WORK_DATA_DIR, 'testsets')
    for f in sorted(os.listdir(tdir)):
        if not f.endswith('.pkl'):
            continue
        ds = f[:-4]
        if ds.startswith('tsu_hyperopt'):
            continue
        d = pd.read_pickle(os.path.join(tdir, f))
        fams = [pid_fam.get(pid) for pid in d['pid']]
        keys = [None] * len(d)
        for i, (pid, mut, fam) in enumerate(zip(d['pid'], d['mutation'], fams)):
            if fam is None or fam not in all_inv_fams:
                continue
            muts = parse_muts(mut)
            acc = set()
            ok = True
            for wt, pos, mt in muts:
                c = canon_one(pid, wt, pos, mt)
                if c is None:
                    ok = False
                    break
                acc.add(c)
            keys[i] = frozenset(acc) if ok else None
        d['family'] = fams
        d['keys'] = keys
        out[ds] = d
        print('  canonicalised', ds, len(d), 'rows;', sum(1 for x in keys if x is not None), 'mutation-checked')
    return out


tests = canon_testsets()
_hc = {}
_wc = {}


def fam_h_union(inv):
    if inv.iid not in _hc:
        s = set()
        for fam in inv.families:
            s |= fam_h.get(fam, set())
        _hc[inv.iid] = s
    return _hc[inv.iid]


def fam_w_union(inv):
    if inv.iid not in _wc:
        s = set()
        for fam in inv.families:
            s |= fam_w.get(fam, set())
        _wc[inv.iid] = s
    return _wc[inv.iid]


def eval_inventory(inv, ds):
    d = tests[ds]
    counts = collections.Counter()
    hom = fam_h_union(inv)
    wk = fam_w_union(inv)
    for fam, keys in zip(d['family'], d['keys']):
        if fam is None:
            counts['unmapped'] += 1
            continue
        if fam in inv.families:
            if keys is not None and keys:
                if keys in inv.variants.get(fam, ()):
                    counts['exact'] += 1
                elif keys <= inv.singles.get(fam, set()):
                    counts['constituent'] += 1
                else:
                    counts['sameprot'] += 1
            else:
                counts['sameprot'] += 1
        elif fam in hom:
            counts['homolog'] += 1
        elif fam in wk:
            counts['weak'] += 1
        else:
            counts['none'] += 1
    return dict(n=len(d), **counts)


rows = []
for inv in INV:
    for ds in tests:
        c = eval_inventory(inv, ds)
        rows.append(dict(inventory=inv.iid, dataset=ds, **c))
ov = pd.DataFrame(rows).fillna(0)
for c in ('exact', 'constituent', 'sameprot', 'homolog', 'weak', 'none', 'unmapped'):
    if c not in ov:
        ov[c] = 0
ov['frac_exact'] = ov['exact'] / ov['n']
ov['frac_any_mut'] = (ov['exact'] + ov['constituent']) / ov['n']
ov.to_csv(os.path.join(OUT_DIR, 'inventory_overlaps.csv'), index=False)
print('saved inventory_overlaps.csv')
