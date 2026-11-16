#!/usr/bin/env python
"""Provenance of the mega_splits Tsuboyama sets: what the split files are, how they relate to the capped split and to the baselines' own
training data, and whether the test set is held out from train/val by the capped split's homology rule. Writes CSVs next to this script
(``provenance/``); the README here reports them.

Inputs (defaults; every optional one is skipped when missing):
  data/mega_splits.pkl                      ours (repo/data); MSR_THERMOMPNN_SPLITS / MSR_SPURS_SPLITS: the upstream copies to md5-compare
  MSR_TSUBOYAMA_CSV                         Tsuboyama2023_Dataset2_Dataset3_20230416.csv (the SPURS and ThermoMPNN input)
  MSR_HOMOLOGY_PAIRS                        the all-vs-all library homology table the capped split was built from (homology_pairs.csv)
  MSR_ME_CDNA                               MutateEverything's data/cdna/mutations/ (cdna_train.csv, cdna1_test.csv, cdna2_test.csv)
  MSR_MEGA_ITEMS                            the retraining tables (outputs/baselines/mega_splits_allbl/data on the cluster: items_{train,val,test}.csv)
and the task tables _tasks/tsuboyama_{mega,capped}_{val,test}.csv.gz (``00_build_tasks.py``).

    python tsuboyama/mega_splits/02_split_provenance.py
"""
import hashlib
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from msr_analysis.registry import TASKS  # noqa: E402

OUT = HERE / 'provenance'
OUT.mkdir(exist_ok=True)
REPO = HERE.parents[2]
ENV = lambda k, d=None: Path(os.environ[k]).expanduser() if os.environ.get(k) else (Path(d).expanduser() if d else None)
PKL = REPO / 'data' / 'mega_splits.pkl'
CAPPED_PKL = REPO / 'data' / 'splits_oct06_capped.pkl'
CSV = ENV('MSR_TSUBOYAMA_CSV', '~/software/SPURS/data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
PAIRS = ENV('MSR_HOMOLOGY_PAIRS', '~/playground/esm-msr-devel/analysis/salvaged_2026-10-16/structure-homology-splits-e5ca04/data/tsuboyama/homology/homology_pairs.csv')
CDNA = ENV('MSR_ME_CDNA')
ITEMS = ENV('MSR_MEGA_ITEMS')
strip = lambda x: str(x)[:-4] if str(x).endswith('.pdb') else str(x)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def kind(c):
    if re.match(r'^[0-9][A-Za-z0-9]{3}$', c):
        return 'natural'
    if 'TrROS_Hall' in c:
        return 'de novo (trRosetta hallucination)'
    if re.match(r'^(v2_)?(HHH|HEEH|EHEE|EEHEE|EHE|HEE)', c):
        return 'de novo (Rocklin design)'
    return 'other (redesign of a natural domain)' if c.startswith('v2_') else 'other'


def main():
    sp = pickle.load(open(PKL, 'rb'))
    parts = {k: {strip(x) for x in sp[k]} for k in ('train', 'val', 'test')}
    cap = pickle.load(open(CAPPED_PKL, 'rb'))
    cparts = {k: {strip(x) for x in cap[k]} for k in ('train', 'val', 'test')}
    print('mega_splits:', {k: len(v) for k, v in parts.items()}, '| capped:', {k: len(v) for k, v in cparts.items()})

    # 1. the split file against the upstream copies
    rows = [dict(file='ours (repo/data/mega_splits.pkl)', md5=md5(PKL))]
    for name, key, d in (('ThermoMPNN dataset_splits/mega_splits.pkl', 'MSR_THERMOMPNN_SPLITS', None),
                         ('SPURS data/dataset/megascale/mega_splits.pkl', 'MSR_SPURS_SPLITS', '~/software/SPURS/data/dataset/megascale/mega_splits.pkl')):
        p = ENV(key, d)
        if p and p.exists():
            rows.append(dict(file=name, md5=md5(p)))
    pd.DataFrame(rows).to_csv(OUT / 'split_file_md5.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. test-set composition, mega against capped
    comp = []
    for nm, ps, ds in (('mega_splits', parts, 'mega'), ('capped', cparts, 'capped')):
        t = pd.read_csv(TASKS / f'tsuboyama_{ds}_test.csv.gz', low_memory=False)
        t['n'] = t.mut_type.astype(str).str.count(':') + 1
        k = pd.Series([kind(c) for c in t.code_wt.unique()]).value_counts()
        comp.append(dict(split=nm, train_libs=len(ps['train']), val_libs=len(ps['val']), test_libs=t.code_wt.nunique(), test_backbones=t.code.nunique(),
                         rows=len(t), measured=int((t.cens == 0).sum()), singles=int((t.n == 1).sum()), doubles=int((t.n == 2).sum()),
                         mutant_backbone_libs=int((t.code != t.code_wt).groupby(t.code).first().sum()), **{f'test_{a}': int(b) for a, b in k.items()}))
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / 'test_composition.csv', index=False)
    mt, ct = set(pd.read_csv(TASKS / 'tsuboyama_mega_test.csv.gz', usecols=['code_wt']).code_wt), set(pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', usecols=['code_wt']).code_wt)
    print(comp.T.to_string(), '\nshared test libraries:', sorted(mt & ct), '\nmega test libraries in the capped train set:', sorted(mt & cparts['train']))

    # 3. mutant-background libraries: raw file, split membership and what our tables hold
    if CSV and CSV.exists():
        raw = pd.read_csv(CSV, usecols=['ddG_ML', 'mut_type', 'WT_name'], low_memory=False)
        raw = raw[raw.ddG_ML.astype(str) != '-']
        raw = raw[~raw.mut_type.astype(str).str.contains('ins|del|wt')]
        raw['n'] = raw.mut_type.str.count(':') + 1
        mbl = sorted(w for w in raw.WT_name.unique() if '.pdb_' in w)
        parent = lambda w: w.split('.pdb_')[0]
        where = lambda c: next((k for k in ('train', 'val', 'test') if c in parts[k]), 'not in mega_splits')
        mb = pd.DataFrame([dict(library=w.replace('.pdb_', '_'), parent=parent(w), parent_split=where(parent(w)), rows=int((raw.WT_name == w).sum())) for w in mbl])
        # upstream ThermoMPNN / SPURS keep a row only when WT_name is exactly a name in the split: no mutant-background library is ever kept
        mb['kept_by_upstream_loaders'] = [w in {x + '.pdb' for k in parts for x in parts[k]} for w in mbl]
        # the same Tsuboyama counts as the upstream loaders (numeric ddG_ML, no ins/del/wt), per split
        up = {k: raw[raw.WT_name.isin({x + '.pdb' for x in parts[k]})] for k in parts}
        rec = [dict(split=k, upstream_rows=len(u), upstream_singles=int((u.n == 1).sum()), upstream_doubles=int((u.n == 2).sum())) for k, u in up.items()]
        rec = pd.DataFrame(rec).set_index('split')
        if ITEMS and (ITEMS / 'items_train.csv').exists():
            for k in ('train', 'val', 'test'):
                d = pd.read_csv(ITEMS / f'items_{k}.csv', usecols=['code', 'code_wt', 'mut_type'], low_memory=False)
                d['n'] = d.mut_type.astype(str).str.count(':') + 1
                rec.loc[k, ['our_rows', 'our_singles', 'our_doubles', 'our_mutant_backbone_rows']] = [len(d), (d.n == 1).sum(), (d.n == 2).sum(), (d.code != d.code_wt).sum()]
                mbk = d[d.code != d.code_wt].groupby('code').size()
                mb[f'rows_in_our_{k}'] = mb.library.map(mbk).fillna(0).astype(int)
        mb.to_csv(OUT / 'mutant_background_libraries.csv', index=False)
        rec.to_csv(OUT / 'row_reconciliation.csv')
        print(f'{len(mb)} mutant-background libraries in the raw file; kept by the upstream loaders: {int(mb.kept_by_upstream_loaders.sum())}')
        print(mb[mb.parent_split != 'not in mega_splits'].to_string(index=False))
        print(rec.to_string())

    # 4. ME's own cDNA sets against the ThermoMPNN split
    if CDNA and (CDNA / 'cdna_train.csv').exists():
        rows = []
        for f in ('cdna_train', 'cdna1_test', 'cdna2_test'):
            d = pd.read_csv(CDNA / f'{f}.csv', low_memory=False)
            d['n'] = d.mut_info.astype(str).str.count(':') + 1
            p = {x.upper() for x in d.pdb_id.astype(str)}
            up = {k: {c.upper() for c in v} for k, v in parts.items()}
            rows.append(dict(me_set=f, rows=len(d), singles=int((d.n == 1).sum()), doubles=int((d.n == 2).sum()), proteins=len(p), in_mega_train=len(p & up['train']),
                             in_mega_val=len(p & up['val']), in_mega_test=len(p & up['test']), in_no_mega_split=len(p - up['train'] - up['val'] - up['test'])))
        pd.DataFrame(rows).to_csv(OUT / 'me_cdna_vs_mega_splits.csv', index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    # 5. homology of the mega test (and val) libraries to train/val under the capped split's rule
    if PAIRS and PAIRS.exists():
        P = pd.read_csv(PAIRS)
        edge = ((P.seq_evalue <= 1e-3) | (P.struct_evalue <= 1e-3) | (P.identity > 0.40) | (P.struct_tm_max > 0.70)).fillna(False)
        E = P[edge]
        out, lib = [], []
        for nm, ps in (('mega_splits', parts), ('capped', cparts)):
            for other in ('train', 'val'):
                x = E[(E.a.isin(ps['test']) & E.b.isin(ps[other])) | (E.b.isin(ps['test']) & E.a.isin(ps[other]))]
                tl = (set(x.a) | set(x.b)) & ps['test']
                out.append(dict(split=nm, against=other, homologous_pairs=len(x), test_libraries_linked=len(tl), test_libraries=len(ps['test']),
                                seq_evalue=int((x.seq_evalue <= 1e-3).sum()), struct_evalue=int((x.struct_evalue <= 1e-3).sum()), identity_gt_0_4=int((x.identity > 0.4).sum()),
                                tm_gt_0_7=int((x.struct_tm_max > 0.7).sum()), max_identity=float(x.identity.max()) if len(x) else np.nan,
                                max_tm=float(x.struct_tm_max.max()) if len(x) else np.nan))
        for t in sorted(parts['test']):
            x = E[((E.a == t) & E.b.isin(parts['train'] | parts['val'])) | ((E.b == t) & E.a.isin(parts['train'] | parts['val']))]
            o = np.where(x.a == t, x.b, x.a)
            lib.append(dict(test_library=t, kind=kind(t), n_homologs=len(x), max_identity=x.identity.max() if len(x) else np.nan, max_tm=x.struct_tm_max.max() if len(x) else np.nan,
                            closest_by_identity=o[int(np.argmax(x.identity.values))] if len(x) else ''))
        pd.DataFrame(out).to_csv(OUT / 'homology_test_vs_trainval.csv', index=False)
        L = pd.DataFrame(lib).sort_values('max_identity', ascending=False)
        L.to_csv(OUT / 'mega_test_library_homology.csv', index=False)
        print(pd.DataFrame(out).to_string(index=False), '\n', L.to_string(index=False))

        # 6. upstream SPURS drops a hard-coded list of train / val scaffolds as homologs of independent sets
    if CSV and CSV.exists():
        spurs = Path(os.path.expanduser('~/software/SPURS/spurs/datamodules/datasets/megascale.py'))
        if spurs.exists():
            src = spurs.read_text()
            i = src.index('to_remove = [', src.index('def remove_homologs_from_scaffold'))
            lst = {x.split('.pdb')[0] for x in re.findall(r"'([^']+)'", src[i:src.index(']', i)])}
            r = [dict(part=k, libraries=len(parts[k]), on_spurs_removal_list=len(lst & parts[k])) for k in ('train', 'val', 'test')]
            if ITEMS and (ITEMS / 'items_train.csv').exists():
                for x in r[:2]:
                    d = pd.read_csv(ITEMS / f'items_{x["part"]}.csv', usecols=['code_wt', 'mut_type'], low_memory=False)
                    d['n'] = d.mut_type.astype(str).str.count(':') + 1
                    m = d.code_wt.isin(lst)
                    x.update(singles=int((d.n == 1).sum()), singles_removed=int(((d.n == 1) & m).sum()), doubles=int((d.n == 2).sum()), doubles_removed=int(((d.n == 2) & m).sum()))
            pd.DataFrame(r).to_csv(OUT / 'spurs_removal_list.csv', index=False)
            print(f'SPURS removal list: {len(lst)} entries\n', pd.DataFrame(r).to_string(index=False))


if __name__ == '__main__':
    main()
