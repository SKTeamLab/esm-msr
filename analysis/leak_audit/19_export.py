"""Build the SQLite audit database and sync machine-readable CSV/JSON outputs into
``docs/benchmarks/`` and ``analysis/reference/``."""
import os
import shutil
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT

BENCH_DIR = os.path.join(REPO_ROOT, 'docs', 'benchmarks')
REF_DIR = os.path.join(REPO_ROOT, 'analysis', 'reference')
os.makedirs(BENCH_DIR, exist_ok=True)

mx = pd.read_csv(os.path.join(OUT_DIR, 'leak_matrix.csv'))
pg = pd.read_csv(os.path.join(OUT_DIR, 'pg_leak_per_dms.csv'))
hom = pd.read_csv(os.path.join(OUT_DIR, 'homology_classes.csv'))
cex = pd.read_csv(os.path.join(OUT_DIR, 'code_inventory_exact.csv'))
hom = pd.concat([hom, pd.read_csv(os.path.join(OUT_DIR, 'esm3dg_homology.csv'))], ignore_index=True)
ros = pd.read_csv(os.path.join(OUT_DIR, 'rosetta_overlap.csv'))

# One row per audited training inventory: what it is, where it came from, and how strong the evidence is.
# evidence tiers: row-level verified (the training rows are public and every overlap count is a verified match);
# protein-level (only the protein / domain list is public; mutation-level counts are ceilings, marked ?);
# documented (described in the paper, no rows); calibration (global parameters tuned on the data: V, not M/P).
inv_desc = pd.DataFrame([
    ('esm_msr_capped', 'ESM-MSR released, baseline and devel checkpoints (every head, every sigma): all substitutions of the 172 capped-train scaffolds; checkpoints selected on capped val', 'data/splits_oct06_capped.pkl + Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'row-level verified'),
    ('esm_msr_small_hyperopt', 'ESM-MSR-small (March reference): train libraries of the legacy hyperopt split; selected on its 26 val libraries', 'data/hyperopt_splits.pkl (= LoRA_models/esm-msr-small/hparams.yaml split_file)', 'row-level verified'),
    ('esmtherm', 'EsmTherm: K50_dG Dataset1+2; 47 test-only domains, the other 414 split 80/10/10 per sequence', 'PLoS Comput Biol 2024 S2 Spreadsheet pcbi.1012248.s006.csv (527,785 sequences with split) + datasets/split/wildtype_split.csv', 'row-level verified'),
    ('ifum', 'IFUM: Megascale Dataset2, MMseqs2-30% cluster split; 322 train domains, 86 in a merged val/test block (no row lists)', 'IFUM dataset_splits.pdf (02_ifum_inventory.py)', 'protein-level (M?, V? are ceilings)'),
    ('ddgemb_s2450', 'DDGemb: S2450 = S2648 minus 198 S669-homologous mutations, doubled with reverse variants', 'ddgemb.biocomp.unibo.it S2450.tsv + S2450.fasta', 'row-level verified'),
    ('ddmut', 'DDMut as deployed (union of the two rows below); used for every DDMut cell', 'S9028.csv + DD_SM1218.csv + DD_SM444.csv', 'row-level verified'),
    ('ddmut_s9028', 'DDMut single-mutant model: S9028 = S2648 + FireProtDB (4,514 forward + 4,514 reverse)', 'biosig.lab.uq.edu.au/ddmut/data S9028.csv', 'row-level verified'),
    ('ddmut_sm1662', 'DDMut multi-mutant model (deployed): SM1218 training + SM444 blind test (831 forward + 831 reverse)', 'biosig.lab.uq.edu.au/ddmut/data DD_SM1218.csv + DD_SM444.csv', 'row-level verified'),
    ('geoddg_s8754', 'GeoDDG-3D: fine-tuned on S8754 (its own test set S8754 is #)', 'GeoStab data/ddG/S8754.csv (GS_S8754.csv)', 'row-level verified'),
    ('geofitness_dms', 'GeoDDG-3D encoder pretraining: GeoFitness, 74 DMS fitness sets (52 MaveDB + 22 DeepSequence)', 'GeoStab data/dms/dms.csv (wild types only: geofitness_dms_wt.fasta)', 'protein-level'),
    ('esm3dg_mgnify', 'ESM3dG: MGnify Stability fine-tune, 960,216 train sequences (augmented checkpoints use a subset); not Megascale', 'Zenodo 19411306 mgnify_training_index.csv', 'row-level verified (0 exact sequences)'),
    ('rosetta_beta_nov16', 'Rosetta cartesian_ddg (2019, beta_nov16_cart): no per-variant training; 20 reference energies fit with Kellogg-2011 ProTherm ddG + Rocklin-2017 design stability in the objective; 1/2.94 scale fit on Kellogg', 'docs.rosettacommons.org Updates-beta-nov16; Park et al. 2016 JCTC SI; Kortemme-Lab/ddg kellogg.csv', 'calibration (V)'),
    ('corpus', 'ProSST / ProteinMPNN / ESM3 zero-shot and the sigma=0 readout: unsupervised pretraining only', 'papers', 'documented (no supervised data)'),
], columns=['inventory', 'description', 'provenance', 'evidence'])

# Write docs/benchmarks CSVs & JSON
mx.to_csv(os.path.join(BENCH_DIR, 'leak_audit_matrix.csv'), index=False)
inv_desc.to_csv(os.path.join(BENCH_DIR, 'leak_audit_inventories.csv'), index=False)
pg.to_csv(os.path.join(BENCH_DIR, 'leak_audit_pg_per_dms.csv'), index=False)
hom.to_csv(os.path.join(BENCH_DIR, 'leak_audit_homology.csv'), index=False)
cex.to_csv(os.path.join(BENCH_DIR, 'leak_audit_code_exact.csv'), index=False)
ros.to_csv(os.path.join(BENCH_DIR, 'leak_audit_rosetta.csv'), index=False)
shutil.copy(os.path.join(OUT_DIR, 'homology_audit.json'), os.path.join(BENCH_DIR, 'split_homology_audit.json'))

for src_name, dst_name in [
    ('structure_homology_domainome_pg.csv', 'leak_audit_structure_homology.csv'),
    ('esm3dg_tsu_libraries.csv', 'leak_audit_esm3dg_libraries.csv'),
    ('esm3dg_exact.csv', 'esm3dg_exact.csv'),
    ('esm3dg_pg_exact_per_dms.csv', 'esm3dg_pg_exact_per_dms.csv'),
]:
    sp = os.path.join(OUT_DIR, src_name)
    if os.path.exists(sp):
        shutil.copy(sp, os.path.join(BENCH_DIR, dst_name))

# Sync analysis/reference copies if directory exists
if os.path.isdir(REF_DIR):
    mx.to_csv(os.path.join(REF_DIR, 'leak_audit_matrix.csv'), index=False)
    inv_desc.to_csv(os.path.join(REF_DIR, 'leak_audit_inventories.csv'), index=False)
    pg.to_csv(os.path.join(REF_DIR, 'leak_audit_pg_per_dms.csv'), index=False)
    st_src = os.path.join(OUT_DIR, 'structure_homology_domainome_pg.csv')
    if os.path.exists(st_src):
        shutil.copy(st_src, os.path.join(REF_DIR, 'leak_audit_structure_homology.csv'))

# Build SQLite database
db = os.path.join(BENCH_DIR, 'leak_audit.sqlite')
if os.path.exists(db):
    os.remove(db)
con = sqlite3.connect(db)
inv_desc.to_sql('inventories', con, index=False)
mx.rename(columns={
    'P': 'same_protein', 'H': 'strong_homolog', 'h': 'weak_homolog',
    'D': 'documented', 'V': 'validation', 'bad': 'invalid',
}).to_sql('cells', con, index=False)
pg.rename(columns={'H': 'strong_homolog', 'h': 'weak_homolog'}).to_sql('pg_per_dms', con, index=False)
hom.rename(columns={'n_H': 'n_strong_homolog', 'n_h': 'n_weak_homolog'}).to_sql('homology', con, index=False)
cex.to_sql('code_exact', con, index=False)
ros.to_sql('rosetta', con, index=False)
for name in ('esm3dg_tsu_libraries', 'esm3dg_exact', 'esm3dg_pg_exact_per_dms', 'geofitness_proteins'):
    src = os.path.join(OUT_DIR, name + '.csv')
    if os.path.exists(src):
        pd.read_csv(src).to_sql(name, con, index=False)
st_csv = os.path.join(OUT_DIR, 'structure_homology_domainome_pg.csv')
if os.path.exists(st_csv):
    pd.read_csv(st_csv).to_sql('structure_homology', con, index=False)
con.execute('CREATE INDEX ix_cells ON cells(model, dataset)')
con.commit()
con.close()
print('sqlite db:', db, os.path.getsize(db), 'bytes; CSVs synced to docs/benchmarks/ and analysis/reference/')
