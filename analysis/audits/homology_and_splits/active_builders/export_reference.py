#!/usr/bin/env python
"""Sync / verify exported audit tables in ``analysis/reference/`` and ``docs/benchmarks/``.

Consolidates ``leak_audit/19_export.py`` + ``mgnify_homology_classes.csv.gz`` export, and provides ``--verify`` to
audit every committed reference CSV and static auxiliary table without requiring external cluster scratch directories.

Usage:
    python -m audits.homology_and_splits.active_builders.export_reference --verify
    python -m audits.homology_and_splits.active_builders.export_reference --sync
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

from ..lib import OUT_DIR, PROVENANCE_DIR, REPO_ROOT
from msr_analysis.paths import REFERENCE, atomic_to_csv

BENCH_DIR = Path(REPO_ROOT) / 'docs' / 'benchmarks'

EXPECTED_REFERENCE_FILES = {
    # Active canonical 4-part split & overlap builders
    'tsuboyama_rule_flags.csv': 'active_builders/tsuboyama_rule_flags.py',
    'split_set_overlap_identity.csv': 'active_builders/set_overlap_matrix.py',
    'split_set_overlap_tm.csv': 'active_builders/set_overlap_matrix.py',
    # Oct-2026 leakage & homology audit exports
    'leak_audit_matrix.csv': 'active_builders/export_reference.py (from oct2026_leak_audit/18_matrix.py)',
    'leak_audit_inventories.csv': 'active_builders/export_reference.py',
    'leak_audit_pg_per_dms.csv': 'active_builders/export_reference.py (from oct2026_leak_audit/17_proteingym.py)',
    'leak_audit_structure_homology.csv': 'active_builders/export_reference.py (from oct2026_leak_audit/07_structure_homology.py)',
    'mgnify_homology_classes.csv.gz': 'active_builders/export_reference.py (from oct2026_leak_audit/20_mgnify_vs_capped_train.py)',
    # Static committed auxiliary reference tables (documented provenance)
    'leak_public_spurs.csv': 'Static committed audit of published SPURS checkpoint mega_splits training overlap (consumed by p10_splits_and_leakage.py)',
    'spurs_public_checkpoint.csv': 'Static committed Spearman rho of published SPURS checkpoint vs retrained SPURS (consumed by p10_splits_and_leakage.py)',
    'pgym_thermompnn_corrected_s1_per_dms.csv': 'Static committed seed-1 corrected ThermoMPNN-D fallback on ProteinGym',
}

INV_DESC = pd.DataFrame([
    ('esm_msr_capped', 'ESM-MSR FINAL_p0, released, baseline and devel checkpoints: all substitutions of the 172 capped-train scaffolds; selected on capped val', 'data/splits_oct06_capped.pkl + Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'row-level verified'),
    ('esm_msr_small_hyperopt', 'ESM-MSR-small (March reference): train libraries of the legacy hyperopt split; selected on its 26 val libraries', 'data/hyperopt_splits.pkl', 'row-level verified'),
    ('esmtherm', 'EsmTherm: K50_dG Dataset1+2; 47 test-only domains, the other 414 split 80/10/10 per sequence', 'PLoS Comput Biol 2024 S2 Spreadsheet pcbi.1012248.s006.csv + wildtype_split.csv', 'row-level verified'),
    ('ifum', 'IFUM: Megascale Dataset2, MMseqs2-30% cluster split; 322 train domains, 86 in a merged val/test block', 'IFUM dataset_splits.pdf (02_ifum_inventory.py)', 'protein-level (M?, V? are ceilings)'),
    ('ddgemb_s2450', 'DDGemb: S2450 = S2648 minus 198 S669-homologous mutations, doubled with reverse variants', 'S2450.tsv + S2450.fasta', 'row-level verified'),
    ('ddmut', 'DDMut as deployed (union of S9028 and SM1662)', 'S9028.csv + DD_SM1218.csv + DD_SM444.csv', 'row-level verified'),
    ('ddmut_s9028', 'DDMut single-mutant model: S9028 = S2648 + FireProtDB (4,514 forward + 4,514 reverse)', 'S9028.csv', 'row-level verified'),
    ('ddmut_sm1662', 'DDMut multi-mutant model: SM1218 training + SM444 blind test (831 forward + 831 reverse)', 'DD_SM1218.csv + DD_SM444.csv', 'row-level verified'),
    ('geoddg_s8754', 'GeoDDG-3D: fine-tuned on S8754', 'GeoStab data/ddG/S8754.csv', 'row-level verified'),
    ('geofitness_dms', 'GeoDDG-3D encoder pretraining: GeoFitness, 74 DMS fitness sets', 'GeoStab data/dms/dms.csv', 'protein-level'),
    ('esm3dg_mgnify', 'ESM3dG: MGnify Stability fine-tune, 960,216 train sequences', 'Zenodo 19411306 mgnify_training_index.csv', 'row-level verified (0 exact sequences)'),
    ('rosetta_beta_nov16', 'Rosetta cartesian_ddg (beta_nov16_cart): 20 reference energies fit on Kellogg-2011 + Rocklin-2017; 1/2.94 scale on Kellogg', 'Park et al. 2016 JCTC SI; kellogg2011_1210.csv', 'calibration (V)'),
    ('corpus', 'ProSST / ProteinMPNN / ESM3 zero-shot and sigma=0 readout: unsupervised pretraining only', 'papers', 'documented (no supervised data)'),
], columns=['inventory', 'description', 'provenance', 'evidence'])


def verify() -> int:
    missing = []
    for fname, src_desc in EXPECTED_REFERENCE_FILES.items():
        p = REFERENCE / fname
        if not p.exists():
            missing.append((fname, src_desc))
        else:
            df = pd.read_csv(p)
            print(f'  OK  reference/{fname:42s} ({len(df):7,d} rows) <- {src_desc}')
    prov_csv = PROVENANCE_DIR / 'mega_test_library_homology.csv'
    if not prov_csv.exists():
        missing.append((str(prov_csv), 'active_builders/mega_split_provenance.py'))
    else:
        df_p = pd.read_csv(prov_csv)
        print(f'  OK  provenance/{prov_csv.name:41s} ({len(df_p):7,d} rows)')
    if missing:
        for fn, desc in missing:
            print(f'  MISSING: {fn} ({desc})')
        return 1
    return 0


def sync_from_out_dir() -> int:
    out_p = Path(OUT_DIR)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    mx = pd.read_csv(out_p / 'leak_matrix.csv')
    pg = pd.read_csv(out_p / 'pg_leak_per_dms.csv')
    hom = pd.concat([pd.read_csv(out_p / 'homology_classes.csv'), pd.read_csv(out_p / 'esm3dg_homology.csv')], ignore_index=True)
    cex = pd.read_csv(out_p / 'code_inventory_exact.csv')
    ros = pd.read_csv(out_p / 'rosetta_overlap.csv')

    for dst_dir in (BENCH_DIR, REFERENCE):
        atomic_to_csv(mx, dst_dir / 'leak_audit_matrix.csv', index=False)
        atomic_to_csv(INV_DESC, dst_dir / 'leak_audit_inventories.csv', index=False)
        atomic_to_csv(pg, dst_dir / 'leak_audit_pg_per_dms.csv', index=False)
        st_src = out_p / 'structure_homology_domainome_pg.csv'
        if st_src.exists():
            atomic_to_csv(pd.read_csv(st_src), dst_dir / 'leak_audit_structure_homology.csv', index=False)

    atomic_to_csv(hom, BENCH_DIR / 'leak_audit_homology.csv', index=False)
    atomic_to_csv(cex, BENCH_DIR / 'leak_audit_code_exact.csv', index=False)
    atomic_to_csv(ros, BENCH_DIR / 'leak_audit_rosetta.csv', index=False)

    mg_hom = out_p / 'mgnify_homology_classes.csv'
    if mg_hom.exists():
        atomic_to_csv(pd.read_csv(mg_hom), REFERENCE / 'mgnify_homology_classes.csv.gz', index=False, compression='gzip')

    db = BENCH_DIR / 'leak_audit.sqlite'
    if db.exists():
        os.remove(db)
    con = sqlite3.connect(str(db))
    INV_DESC.to_sql('inventories', con, index=False)
    mx.rename(columns={'P': 'same_protein', 'H': 'strong_homolog', 'h': 'weak_homolog', 'D': 'documented', 'V': 'validation', 'bad': 'invalid'}).to_sql('cells', con, index=False)
    pg.rename(columns={'H': 'strong_homolog', 'h': 'weak_homolog'}).to_sql('pg_per_dms', con, index=False)
    hom.rename(columns={'n_H': 'n_strong_homolog', 'n_h': 'n_weak_homolog'}).to_sql('homology', con, index=False)
    cex.to_sql('code_exact', con, index=False)
    ros.to_sql('rosetta', con, index=False)
    con.execute('CREATE INDEX ix_cells ON cells(model, dataset)')
    con.commit()
    con.close()
    print('Synced exports to docs/benchmarks/ and analysis/reference/')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--verify', action='store_true', default=True, help='Verify committed reference & provenance files (default)')
    ap.add_argument('--sync', action='store_true', help='Re-export tables from $LEAK_OUT_DIR into reference/ and docs/benchmarks/')
    args = ap.parse_args(argv)
    if args.sync:
        return sync_from_out_dir()
    return verify()


if __name__ == '__main__':
    raise SystemExit(main())
