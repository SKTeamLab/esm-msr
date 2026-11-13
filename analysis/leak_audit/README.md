# Leak-audit pipeline (`analysis/leak_audit/`)

This pipeline reproduces [`docs/LEAKAGE_AUDIT_OCT2026.md`](../../docs/LEAKAGE_AUDIT_OCT2026.md) and the
machine-readable audit artifacts in [`docs/benchmarks/`](../../docs/benchmarks/). The ones the analysis code reads are
mirrored to [`analysis/reference/`](../reference/).

| artifact | contents |
|---|---|
| `leak_audit_matrix.csv` | per `(model, dataset)` row counts, compact codes and superscripts |
| `leak_audit_inventories.csv` | every audited training inventory, with provenance and evidence tier |
| `leak_audit_pg_per_dms.csv` | ESM-MSR capped-train overlap with each of the 217 ProteinGym assays |
| `leak_audit_homology.csv` | `P` / `H` / `h` row counts per inventory × dataset |
| `leak_audit_structure_homology.csv` | Foldseek 3D hits on Domainome and ProteinGym |
| `leak_audit_code_exact.csv` | verified exact / constituent rows for DDGemb, DDMut and GeoDDG-3D |
| `leak_audit_rosetta.csv` | rows in Rosetta's `beta_nov16` tuning data |
| `leak_audit_esm3dg_libraries.csv`, `esm3dg_exact.csv`, `esm3dg_pg_exact_per_dms.csv` | ESM3ΔG homology and exact joins |
| `split_homology_audit.json` | checks of the `splits_oct06_capped` quarantine, bridge pruning and threshold sensitivity |
| `leak_audit.sqlite` | all of the above as one database |

The retrained transfer-learning baselines (MutateEverything, ThermoMPNN / ThermoMPNN-D, SPURS) are not audited
separately. They train on a subset of ESM-MSR's capped-train scaffolds, so they inherit its codes (doc, scope note).

## Run

```bash
python analysis/leak_audit/run_all.py
```

- `run_all.py` runs every numbered script in file-name order. The number is the dependency order.
- `--from-step 11` resumes from a step; `--skip-upstream` reuses the tracked EsmTherm / IFUM CSVs.
- `THREADS=4` caps MMseqs2 and Foldseek.
- A full run takes about 4 minutes; all searches are cached by input hash.

| step | script | output |
|---|---|---|
| 01 | `01_esmtherm_inventory.py` | `data/esmtherm_train_proteins.csv` (EsmTherm wild-type split) |
| 02 | `02_ifum_inventory.py` | `data/ifum_train_proteins.csv` (parsed from IFUM's `dataset_splits.pdf`) |
| 03 | `03_megascale.py` | Megascale library / variant tables; `capped_split_sets.pkl` |
| 04 | `04_testsets.py` | canonical `(pid, seq, mutation)` tables for every benchmark dataset |
| 05 | `05_inventory_proteins.py` | sequences and PDB residue maps of the literature inventories |
| 06 | `06_sequence_homology.py` | protein corpus + MMseqs2 all-vs-all |
| 07 | `07_structure_homology.py` | Foldseek search of Domainome and ProteinGym structures |
| 08 | `08_families.py` | ≥ 95 % identity families and position maps |
| 09 | `09_split_audit.py` | `split_homology_audit.json` |
| 10 | `10_overlap_engine.py` | `M` / `m` / `P` (ESM-MSR, ESM-MSR-small); exact-sequence `M` / `V` (EsmTherm, IFUM) |
| 11 | `11_homology_classes.py` | `P` / `H` / `h` per inventory (rule: `lib.classify_hits`) |
| 12 | `12_literature_exact.py` | DDGemb / DDMut / GeoDDG-3D `M` / `m` |
| 13 | `13_mgnify_search.py` | MMseqs2 vs the 960k ESM3ΔG MGnify training sequences |
| 14 | `14_esm3dg_homology.py` | ESM3ΔG `P` / `H` / `h` |
| 15 | `15_esm3dg_exact.py` | ESM3ΔG exact WT / mutant sequence join |
| 16 | `16_rosetta.py` | Rosetta `V` (Kellogg 2011 and Rocklin 2017 rows) |
| 17 | `17_proteingym.py` | per-assay ProteinGym table; ESM-MSR clean aggregates |
| 18 | `18_matrix.py` | leak codes |
| 19 | `19_export.py` | `docs/benchmarks/`, SQLite, `analysis/reference/` |

## Inputs

Small reference tables are tracked in [`data/`](data/). Large inputs and intermediates live in `$LEAK_DATA_DIR`
(default `$LEAK_ROOT/leak_audit/data`) and `$LEAK_OUT_DIR` (default `$LEAK_ROOT/leak_audit/out`). `$LEAK_ROOT`
defaults to the directory above the repository checkout, found correctly from inside a git worktree too.

| tracked file in `data/` | source |
|---|---|
| `S2450.tsv`, `S2450.fasta` | DDGemb training set ([ddgemb.biocomp.unibo.it/datasets](https://ddgemb.biocomp.unibo.it/datasets/)) |
| `S9028.csv`, `DD_SM1218.csv`, `DD_SM444.csv` | DDMut single-mutant and (deployed) multi-mutant training sets ([biosig.lab.uq.edu.au/ddmut/data](https://biosig.lab.uq.edu.au/ddmut/data)) |
| `ifum_train_proteins.csv` | IFUM supplementary `dataset_splits.pdf` (step 02) |
| `esmtherm_split.csv`, `esmtherm_train_proteins.csv` | EsmTherm `datasets/split/wildtype_split.csv` (step 01) |
| `kellogg2011_1210.csv` | Kellogg et al. 2011 ProTherm set, Kortemme-lab curated copy ([Kortemme-Lab/ddg `input/csv/kellogg.csv`](https://github.com/Kortemme-Lab/ddg/blob/master/input/csv/kellogg.csv)), parsed to one row per substitution |

| expected in `$LEAK_DATA_DIR/raw/` | source |
|---|---|
| `GS_S8754.csv` | GeoStab `data/ddG/S8754.csv` |
| `geofitness_dms_wt.fasta` | wild types of GeoStab `data/dms/dms.csv` (74 GeoFitness DMS sets) |
| `esmtherm_pcbi.1012248.s006.csv` | EsmTherm S2 Spreadsheet (PLoS Comput Biol 2024; 527,785 sequences with split) |
| `mgnify_index.csv` | ESM3ΔG Zenodo 19411306 `mgnify_training_index.csv` |
| `pdb_chain_uniprot.tsv.gz` | SIFTS ([ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/)) |
| `Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv` | Domainome 1.0 (Beltran et al. 2025); otherwise read from the copy in a local SPURS clone |

Other inputs:
- **Repository (tracked):** `data/splits_oct06_capped.pkl`, `data/hyperopt_splits.pkl`, `data/preprocessed/*` and
  `docs/benchmarks/all_by_all_long.csv`.
- **Repository (untracked):** `data/domainome1/alphafold_structures/`. In a worktree this is read from the main
  checkout.
- **Under `$LEAK_ROOT`:**
  - `ProteinGym/`: `DMS_substitutions.csv`, the assays and AF2 structures;
  - `remote_data/splits_oct06_capped-{val,test}.csv.gz`;
  - `analysis/salvaged_2026-10-16/…/homology_pairs.csv` (step 09);
  - `pgym_results/` (optional, step 17 aggregates).
- **Under `$SOFTWARE_DIR/esm-msr/data/tsuboyama/`:** the Tsuboyama CSV and its AlphaFold models.
