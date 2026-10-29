"""Build the audit database and the machine-readable tables committed with the repo."""
import os, sqlite3, json, shutil
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data'); OUT=os.path.join(ROOT,'leak_audit','out')
PAY=os.path.join(ROOT,'leak_audit','repo_payload')
os.makedirs(os.path.join(PAY,'docs','benchmarks'),exist_ok=True)
mx=pd.read_csv(os.path.join(OUT,'leak_matrix.csv'))
pg=pd.read_csv(os.path.join(OUT,'pg_leak_per_dms.csv'))
inv_desc=pd.DataFrame([
 ('esm_msr_capped','ESM-MSR (released) and (devel) checkpoints; all substitutions of the 172 capped-train libraries','splits_oct06_capped.pkl train + Tsuboyama2023_Dataset2_Dataset3_20230416.csv','verified'),
 ('me_stage1','MutateEverything stage 1 (singles) on the capped train libraries','capped split; ME recipe stage 1','verified'),
 ('me_stage2','MutateEverything stage 2 (singles then doubles) on the capped train libraries','capped split; stage 2 initialised from stage 1','verified'),
 ('tmpnn','ThermoMPNN additive (singles) retrained on the capped train libraries','capped split; train_single.yaml','verified'),
 ('tmpnnd','ThermoMPNN-D (doubles) retrained on the capped train libraries it can see: the 52 with measured doubles, which all have a Rosetta model too, plus 82 more with a Rosetta-modelled single (134 of 172)','capped split; train_epistatic.yaml; 05c_tmpnnd_inventory.py','verified'),
 ('spurs_retrain_single','SPURS single-mutant model retrained on the capped train libraries (singles; same rows as ME stage 1)','capped split; training_scripts/baselines/slurm_spurs_train.sh','verified'),
 ('spurs_retrain_multi','SPURS multi-mutant model retrained on the capped train libraries (same-seed single model, then singles+doubles; same rows as ESM-MSR)','capped split; training_scripts/baselines/slurm_spurs_train.sh','verified'),
 ('spurs_single','PUBLISHED SPURS single-mutant checkpoint (not retrained; not in the BENCHMARKS tables): Megascale train+val libraries after the 25%-identity blacklist','SPURS_mod data; ThermoMPNN 8:1:1 split; m8 filter','verified'),
 ('spurs_multi','PUBLISHED SPURS multi-mutant checkpoint (not retrained; not in the BENCHMARKS tables): same libraries, doubles','SPURS_mod data','verified'),
 ('esmtherm','EsmTherm: K50_dG Dataset1+2 domains, mutant-level 80/10/10 split','datasets/split/wildtype_split.csv','protein-level (mutation rows unavailable)'),
 ('ifum','IFUM: Megascale Dataset2 train domains (322 train, 86 val/test)','IFUM published dataset_splits.pdf','protein-level (mutation rows unavailable)'),
 ('esm3dg_mgnify','ESM3dG: 960,216-sequence MGnify Stability fine-tune (not Megascale)','Zenodo 19411306 mgnify_training_index.csv','verified (0 exact sequences)'),
 ('ddgemb_s2450','DDGemb: S2450 = S2648 minus 198 S669-homologous mutations','ddgemb.biocomp.unibo.it S2450.tsv/.fasta','verified'),
 ('ddmut_s9028','DDMut: S9028 = S2648 + FireProtDB (4,514 forward + reverse)','biosig.lab.uq.edu.au S9028.csv','verified'),
 ('ddmut_sm1242','DDMut multi-mutant branch: SM1242','biosig.lab.uq.edu.au SM1242.csv','verified'),
 ('geoddg_s8754','GeoDDG: S8754 (its own training set)','GeoStab S8754.csv','verified'),
 ('corpus','ProSST / ProteinMPNN / ESM3 zero-shot: pretraining corpus only','papers','documented only'),
 ('none','Rosetta Cartesian ΔΔG: no training data','physics-based','n/a'),
], columns=['inventory','description','provenance','evidence'])
# write CSVs
mx.to_csv(os.path.join(PAY,'docs','benchmarks','leak_audit_matrix.csv'),index=False)
inv_desc.to_csv(os.path.join(PAY,'docs','benchmarks','leak_audit_inventories.csv'),index=False)
pg.to_csv(os.path.join(PAY,'docs','benchmarks','leak_audit_pg_per_dms.csv'),index=False)
pd.read_csv(os.path.join(OUT,'homology_classes.csv')).to_csv(os.path.join(PAY,'docs','benchmarks','leak_audit_homology.csv'),index=False)
pd.read_csv(os.path.join(OUT,'code_inventory_exact.csv')).to_csv(os.path.join(PAY,'docs','benchmarks','leak_audit_code_exact.csv'),index=False)
shutil.copy(os.path.join(OUT,'homology_audit.json'), os.path.join(PAY,'docs','benchmarks','split_homology_audit.json'))
# sqlite
db=os.path.join(PAY,'docs','benchmarks','leak_audit.sqlite')
if os.path.exists(db): os.remove(db)
con=sqlite3.connect(db)
inv_desc.to_sql('inventories',con,index=False)
mx.rename(columns={'P':'same_protein','H':'strong_homolog','h':'weak_homolog','C':'corpus','D':'documented','U':'unknown','V':'validation','bad':'invalid'}).to_sql('cells',con,index=False)
pg.to_sql('pg_per_dms',con,index=False)
pd.read_csv(os.path.join(OUT,'homology_classes.csv')).rename(columns={'n_H':'n_strong_homolog','n_h':'n_weak_homolog'}).to_sql('homology',con,index=False)
pd.read_csv(os.path.join(OUT,'code_inventory_exact.csv')).to_sql('code_exact',con,index=False)
con.execute('CREATE INDEX ix_cells ON cells(model,dataset)')
con.commit(); con.close()
print('db', db, os.path.getsize(db))
print('csvs written')
