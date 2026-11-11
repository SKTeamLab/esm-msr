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

inv_desc = pd.DataFrame([
    ('esm_msr_capped', 'ESM-MSR (released) and (devel) checkpoints; all substitutions of the 172 capped-train scaffolds', 'splits_oct06_capped.pkl train + Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'verified'),
    ('me_stage1', 'MutateEverything stage 1 (singles) retrained on the 172 capped-train scaffolds', 'capped split; ME recipe stage 1', 'verified'),
    ('me_stage2', 'MutateEverything stage 2 (singles then doubles) retrained on the 172 capped-train scaffolds', 'capped split; stage 2 initialised from stage 1', 'verified'),
    ('tmpnn', 'ThermoMPNN additive (singles) retrained on the 172 capped-train scaffolds', 'capped split; train_single.yaml', 'verified'),
    ('tmpnnd', 'ThermoMPNN-D (doubles) retrained on the capped-train scaffolds it sees: 52 with measured doubles plus 82 with a Rosetta-modelled single (134 of 172)', 'capped split; train_epistatic.yaml; 05c_tmpnnd_inventory.py', 'verified'),
    ('spurs_retrain_single', 'SPURS single-mutant model retrained on the 172 capped-train scaffolds (singles; same rows as ME stage 1)', 'capped split; training_scripts/baselines/slurm_spurs_train.sh', 'verified'),
    ('spurs_retrain_multi', 'SPURS multi-mutant model retrained on the 172 capped-train scaffolds (same-seed single model, then singles+doubles; same rows as ESM-MSR)', 'capped split; training_scripts/baselines/slurm_spurs_train.sh', 'verified'),
    ('spurs_single', 'PUBLISHED SPURS single-mutant checkpoint (historical comparison only; not in primary BENCHMARKS tables): upstream ThermoMPNN 8:1:1 split + m8 filter', 'SPURS_mod data', 'verified'),
    ('spurs_multi', 'PUBLISHED SPURS multi-mutant checkpoint (historical comparison only; not in primary BENCHMARKS tables): upstream ThermoMPNN 8:1:1 split + m8 filter', 'SPURS_mod data', 'verified'),
    ('esmtherm', 'EsmTherm: K50_dG Dataset1+2 domains, mutant-level 80/10/10 split', 'datasets/split/wildtype_split.csv', 'protein-level (mutation rows unavailable)'),
    ('ifum', 'IFUM: MegaScale Dataset2 train domains (322 train, 86 val/test scaffolds)', 'IFUM published dataset_splits.pdf', 'protein-level (mutation rows unavailable)'),
    ('esm3dg_mgnify', 'ESM3dG: 960,216-sequence MGnify Stability fine-tune (not MegaScale)', 'Zenodo 19411306 mgnify_training_index.csv', 'verified (0 exact sequences)'),
    ('ddgemb_s2450', 'DDGemb: S2450 = S2648 minus 198 S669-homologous mutations', 'ddgemb.biocomp.unibo.it S2450.tsv + S2450.fasta', 'verified'),
    ('ddmut_s9028', 'DDMut: S9028 = S2648 + FireProtDB (4,514 forward + 4,514 reverse)', 'biosig.lab.uq.edu.au S9028.csv', 'verified'),
    ('ddmut_sm1242', 'DDMut multi-mutant branch: SM1242 (621 forward + 621 reverse)', 'biosig.lab.uq.edu.au DD_SM1242.csv', 'verified'),
    ('geoddg_s8754', 'GeoDDG-3D: S8754 (its own training set)', 'GeoStab GS_S8754.csv', 'verified'),
    ('corpus', 'ProSST / ProteinMPNN / ESM3 zero-shot & σ=0 readout: unsupervised pretraining only (reported as –)', 'papers', 'no supervised ΔΔG data'),
    ('none', 'Rosetta Cartesian ΔΔG: physics-based energy function (reported as –)', 'physics-based', 'n/a'),
], columns=['inventory', 'description', 'provenance', 'evidence'])

# Write docs/benchmarks CSVs & JSON
mx.to_csv(os.path.join(BENCH_DIR, 'leak_audit_matrix.csv'), index=False)
inv_desc.to_csv(os.path.join(BENCH_DIR, 'leak_audit_inventories.csv'), index=False)
pg.to_csv(os.path.join(BENCH_DIR, 'leak_audit_pg_per_dms.csv'), index=False)
hom.to_csv(os.path.join(BENCH_DIR, 'leak_audit_homology.csv'), index=False)
cex.to_csv(os.path.join(BENCH_DIR, 'leak_audit_code_exact.csv'), index=False)
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

# Build SQLite database
db = os.path.join(BENCH_DIR, 'leak_audit.sqlite')
if os.path.exists(db):
    os.remove(db)
con = sqlite3.connect(db)
inv_desc.to_sql('inventories', con, index=False)
mx.rename(columns={
    'P': 'same_protein', 'H': 'strong_homolog', 'h': 'weak_homolog',
    'C': 'corpus', 'D': 'documented', 'U': 'unknown', 'V': 'validation', 'bad': 'invalid',
}).to_sql('cells', con, index=False)
pg.rename(columns={'H': 'strong_homolog', 'h': 'weak_homolog'}).to_sql('pg_per_dms', con, index=False)
hom.rename(columns={'n_H': 'n_strong_homolog', 'n_h': 'n_weak_homolog'}).to_sql('homology', con, index=False)
cex.to_sql('code_exact', con, index=False)
st_csv = os.path.join(OUT_DIR, 'structure_homology_domainome_pg.csv')
if os.path.exists(st_csv):
    pd.read_csv(st_csv).to_sql('structure_homology', con, index=False)
con.execute('CREATE INDEX ix_cells ON cells(model, dataset)')
con.commit()
con.close()
print('sqlite db:', db, os.path.getsize(db), 'bytes; CSVs synced to docs/benchmarks/ and analysis/reference/')
