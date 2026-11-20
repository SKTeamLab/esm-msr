#!/usr/bin/env python
"""Provenance audit of ``mega_splits`` vs ``splits_oct06_capped`` and upstream SPURS/ThermoMPNN/MutateEverything splits.

Writes 8 CSVs under ``analysis/audits/homology_and_splits/provenance/`` (including ``mega_test_library_homology.csv``,
consumed by ``pipelines/p10_splits_and_leakage.py`` for the 12-clean vs 16-homologous test stratification figure).

Usage: ``python -m audits.homology_and_splits.active_builders.mega_split_provenance``
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import PROVENANCE_DIR, SOFTWARE_DIR
from msr_analysis.paths import DATA_DIR, SALVAGED_HOMOLOGY_CSV, TASKS, atomic_to_csv


def _env_path(key: str, default: str | Path | None = None) -> Path | None:
    if os.environ.get(key):
        return Path(os.environ[key]).expanduser()
    return Path(default).expanduser() if default is not None else None


PKL = DATA_DIR / 'mega_splits.pkl'
CAPPED_PKL = DATA_DIR / 'splits_oct06_capped.pkl'
CSV = _env_path('MSR_TSUBOYAMA_CSV', Path(SOFTWARE_DIR) / 'SPURS' / 'data' / 'dataset' / 'megascale' / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
PAIRS = _env_path(
    'MSR_HOMOLOGY_PAIRS',
    SALVAGED_HOMOLOGY_CSV if SALVAGED_HOMOLOGY_CSV.exists() else SALVAGED_HOMOLOGY_CSV.parent.parent / 'homology' / 'homology_pairs.csv',
)
CDNA = _env_path('MSR_ME_CDNA')
ITEMS = _env_path('MSR_MEGA_ITEMS')


def _strip_pdb(x) -> str:
    s = str(x)
    return s[:-4] if s.endswith('.pdb') else s


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def kind(c: str) -> str:
    if re.match(r'^[0-9][A-Za-z0-9]{3}$', c):
        return 'natural'
    if 'TrROS_Hall' in c:
        return 'de novo (trRosetta hallucination)'
    if re.match(r'^(v2_)?(HHH|HEEH|EHEE|EEHEE|EHE|HEE)', c):
        return 'de novo (Rocklin design)'
    return 'other (redesign of a natural domain)' if c.startswith('v2_') else 'other'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.parse_args(argv)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    with open(PKL, 'rb') as fh:
        sp = pickle.load(fh)
    parts = {k: {_strip_pdb(x) for x in sp[k]} for k in ('train', 'val', 'test')}
    with open(CAPPED_PKL, 'rb') as fh:
        cap = pickle.load(fh)
    cparts = {k: {_strip_pdb(x) for x in cap[k]} for k in ('train', 'val', 'test')}

    rows = [dict(file='ours (data/mega_splits.pkl)', md5=md5(PKL))]
    for name, key, d in (
        ('ThermoMPNN dataset_splits/mega_splits.pkl', 'MSR_THERMOMPNN_SPLITS', None),
        ('SPURS data/dataset/megascale/mega_splits.pkl', 'MSR_SPURS_SPLITS', Path(SOFTWARE_DIR) / 'SPURS' / 'data' / 'dataset' / 'megascale' / 'mega_splits.pkl'),
    ):
        p = _env_path(key, d)
        if p and p.exists():
            rows.append(dict(file=name, md5=md5(p)))
    atomic_to_csv(pd.DataFrame(rows), PROVENANCE_DIR / 'split_file_md5.csv', index=False)

    comp = []
    for nm, ps, ds in (('mega_splits', parts, 'mega'), ('capped', cparts, 'capped')):
        t = pd.read_csv(TASKS / f'tsuboyama_{ds}_test.csv.gz', low_memory=False)
        t['n'] = t.mut_type.astype(str).str.count(':') + 1
        k = pd.Series([kind(c) for c in t.code_wt.unique()]).value_counts()
        comp.append(dict(
            split=nm, train_libs=len(ps['train']), val_libs=len(ps['val']),
            test_libs=t.code_wt.nunique(), test_backbones=t.code.nunique(),
            rows=len(t), measured=int((t.cens == 0).sum()), singles=int((t.n == 1).sum()), doubles=int((t.n == 2).sum()),
            mutant_backbone_libs=int((t.code != t.code_wt).groupby(t.code).first().sum()),
            **{f'test_{a}': int(b) for a, b in k.items()},
        ))
    atomic_to_csv(pd.DataFrame(comp), PROVENANCE_DIR / 'test_composition.csv', index=False)

    if CSV and CSV.exists():
        raw = pd.read_csv(CSV, usecols=['ddG_ML', 'mut_type', 'WT_name'], low_memory=False)
        raw = raw[raw.ddG_ML.astype(str) != '-']
        raw = raw[~raw.mut_type.astype(str).str.contains('ins|del|wt')]
        raw['n'] = raw.mut_type.str.count(':') + 1
        mbl = sorted(w for w in raw.WT_name.unique() if '.pdb_' in w)
        parent = lambda w: w.split('.pdb_')[0]
        where = lambda c: next((k for k in ('train', 'val', 'test') if c in parts[k]), 'not in mega_splits')
        mb = pd.DataFrame([dict(library=w.replace('.pdb_', '_'), parent=parent(w), parent_split=where(parent(w)), rows=int((raw.WT_name == w).sum())) for w in mbl])
        mb['kept_by_upstream_loaders'] = [w in {x + '.pdb' for k in parts for x in parts[k]} for w in mbl]
        up = {k: raw[raw.WT_name.isin({x + '.pdb' for x in parts[k]})] for k in parts}
        rec = pd.DataFrame([dict(split=k, upstream_rows=len(u), upstream_singles=int((u.n == 1).sum()), upstream_doubles=int((u.n == 2).sum())) for k, u in up.items()]).set_index('split')
        if ITEMS and (ITEMS / 'items_train.csv').exists():
            for k in ('train', 'val', 'test'):
                d = pd.read_csv(ITEMS / f'items_{k}.csv', usecols=['code', 'code_wt', 'mut_type'], low_memory=False)
                d['n'] = d.mut_type.astype(str).str.count(':') + 1
                rec.loc[k, ['our_rows', 'our_singles', 'our_doubles', 'our_mutant_backbone_rows']] = [len(d), (d.n == 1).sum(), (d.n == 2).sum(), (d.code != d.code_wt).sum()]
                mbk = d[d.code != d.code_wt].groupby('code').size()
                mb[f'rows_in_our_{k}'] = mb.library.map(mbk).fillna(0).astype(int)
        atomic_to_csv(mb, PROVENANCE_DIR / 'mutant_background_libraries.csv', index=False)
        atomic_to_csv(rec.reset_index(), PROVENANCE_DIR / 'row_reconciliation.csv', index=False)

    if CDNA and (CDNA / 'cdna_train.csv').exists():
        c_rows = []
        for f in ('cdna_train', 'cdna1_test', 'cdna2_test'):
            d = pd.read_csv(CDNA / f'{f}.csv', low_memory=False)
            d['n'] = d.mut_info.astype(str).str.count(':') + 1
            p = {x.upper() for x in d.pdb_id.astype(str)}
            up = {k: {c.upper() for c in v} for k, v in parts.items()}
            c_rows.append(dict(
                me_set=f, rows=len(d), singles=int((d.n == 1).sum()), doubles=int((d.n == 2).sum()), proteins=len(p),
                in_mega_train=len(p & up['train']), in_mega_val=len(p & up['val']), in_mega_test=len(p & up['test']),
                in_no_mega_split=len(p - up['train'] - up['val'] - up['test']),
            ))
        atomic_to_csv(pd.DataFrame(c_rows), PROVENANCE_DIR / 'me_cdna_vs_mega_splits.csv', index=False)

    if PAIRS and PAIRS.exists():
        P = pd.read_csv(PAIRS)
        edge = ((P.seq_evalue <= 1e-3) | (P.struct_evalue <= 1e-3) | (P.identity > 0.40) | (P.struct_tm_max > 0.70)).fillna(False)
        E = P[edge]
        out, lib = [], []
        for nm, ps in (('mega_splits', parts), ('capped', cparts)):
            for other in ('train', 'val'):
                x = E[(E.a.isin(ps['test']) & E.b.isin(ps[other])) | (E.b.isin(ps['test']) & E.a.isin(ps[other]))]
                tl = (set(x.a) | set(x.b)) & ps['test']
                out.append(dict(
                    split=nm, against=other, homologous_pairs=len(x), test_libraries_linked=len(tl), test_libraries=len(ps['test']),
                    seq_evalue=int((x.seq_evalue <= 1e-3).sum()), struct_evalue=int((x.struct_evalue <= 1e-3).sum()),
                    identity_gt_0_4=int((x.identity > 0.4).sum()), tm_gt_0_7=int((x.struct_tm_max > 0.7).sum()),
                    max_identity=float(x.identity.max()) if len(x) else np.nan, max_tm=float(x.struct_tm_max.max()) if len(x) else np.nan,
                ))
        for t in sorted(parts['test']):
            x = E[((E.a == t) & E.b.isin(parts['train'] | parts['val'])) | ((E.b == t) & E.a.isin(parts['train'] | parts['val']))]
            o = np.where(x.a == t, x.b, x.a)
            lib.append(dict(
                test_library=t, kind=kind(t), n_homologs=len(x),
                max_identity=x.identity.max() if len(x) else np.nan,
                max_tm=x.struct_tm_max.max() if len(x) else np.nan,
                closest_by_identity=o[int(np.argmax(x.identity.values))] if len(x) else '',
            ))
        atomic_to_csv(pd.DataFrame(out), PROVENANCE_DIR / 'homology_test_vs_trainval.csv', index=False)
        L = pd.DataFrame(lib).sort_values('max_identity', ascending=False)
        atomic_to_csv(L, PROVENANCE_DIR / 'mega_test_library_homology.csv', index=False)

    spurs_py = _env_path('MSR_SPURS_MEGASCALE_PY', Path(SOFTWARE_DIR) / 'SPURS' / 'spurs' / 'datamodules' / 'datasets' / 'megascale.py')
    if spurs_py and spurs_py.exists():
        src = spurs_py.read_text(encoding='utf-8')
        i = src.index('to_remove = [', src.index('def remove_homologs_from_scaffold'))
        lst = {x.split('.pdb')[0] for x in re.findall(r"'([^']+)'", src[i:src.index(']', i)])}
        r = [dict(part=k, libraries=len(parts[k]), on_spurs_removal_list=len(lst & parts[k])) for k in ('train', 'val', 'test')]
        atomic_to_csv(pd.DataFrame(r), PROVENANCE_DIR / 'spurs_removal_list.csv', index=False)

    print('Provenance CSVs verified/updated in', PROVENANCE_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
