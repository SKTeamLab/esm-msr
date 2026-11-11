# Preprocessing Pipeline

This directory contains the standalone scripts for downloading raw datasets and structures, preparing benchmark mutation tables, constructing homology-controlled train/validation/test splits, and extracting biophysical features. Core library preprocessors invoked during training, evaluation, and tensor-cache construction live in `src/esm_msr/`.

---

## Overview of Components

| Component | Purpose | Primary Outputs |
|---|---|---|
| `preprocessing/download_tsuboyama.py` | Download MegaScale (Tsuboyama et al., *Nature* 2023) ΔΔG dataset and AlphaFold2 PDBs from Zenodo (`7992926`) | `data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`, `data/tsuboyama/AlphaFold_model_PDBs/*.pdb` |
| `src/esm_msr/preprocess_megascale.py` | Parse, filter, deduplicate, and standardize MegaScale single/double/epistasis stability splits | In-memory DataFrames for training, evaluation (`inference_scripts/esm_msr_testing.py`), and `src/esm_msr/cache_build.py` |
| `src/esm_msr/preprocess.py` | Map external ΔΔG (`q3421`, `s669`, `ssym`, `k3822`, `fireprot`, `protherm`, `ddgun`, `ptmul`) and functional DMS (`gb1`, `trpb`, `ube4b`, `avgfp`, `pab1`, `bgl3`) datasets onto cleaned/repaired PDB structures | `data/preprocessed/<dataset>_mapped.csv`, `data/structures/*.pdb` |
| `preprocessing/download_domainome_structures.py` | Download and extract the 522 Domainome 1.0 domain structures distributed with SPURS (and optional Supplementary Table 5 TSV) | `data/domainome1/alphafold_structures/<domain_ID>.pdb`, `data/domainome1/SupplementaryTable5.txt` |
| `preprocessing/preprocess_domainome.py` | Truncate/renumber Domainome 1.0 structures to 1-based construct coordinates and map missense mutations | `data/domainome1/alphafold_structures/<domain_ID>_truncated_renumbered.pdb`, `data/domainome1/domainome_mapped_2026.csv` |
| `preprocessing/pgym_preprocess.py` | Map the 217 ProteinGym v1.3 DMS substitution assays onto AlphaFold2 structures | `pgym_inputs/<DMS_id>.csv`, `pgym_inputs/manifest.csv` |
| `preprocessing/split_tsuboyama.py` & `preprocessing/homology.py` | Build sequence- and structure-homology-controlled MegaScale splits against held-out benchmarks via MMseqs2, Foldseek, local BLOSUM62 identity, and bridge pruning | `data/splits_oct06_capped.pkl` (`.csv` optional) |
| `preprocessing/split_overlap_report.py` | Audit an existing MegaScale split file against held-out benchmarks and generate leakage summary CSVs | `<out_prefix>_summary_by_benchmark.csv`, `<out_prefix>_flagged_pairs.csv` |
| `preprocessing/compute_features.py` | Optional biophysical & MSA feature extraction (DSSP `rel_ASA`/`SS`, QuantiProt AAIndex, interface, AliStat conservation) for property-category analysis | `data/features/<dataset>_mapped_feats.csv` |

> **Baseline-specific data preparation:** Input bundles for external baseline models (Mutate Everything, ThermoMPNN, ProteinMPNN, SPURS) are built from the preprocessed benchmark tables above using the scripts in `inference_scripts/baselines/prepare/`.

---

## 1. MegaScale Training & Evaluation Data (Tsuboyama et al., 2023)

Download the raw MegaScale dataset and AlphaFold2 domain models from [Zenodo record 7992926](https://zenodo.org/records/7992926) (~2 GB download, ~5 GB extracted):

```bash
python preprocessing/download_tsuboyama.py --out data/tsuboyama
```

This populates:
- `data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`
- `data/tsuboyama/AlphaFold_model_PDBs/*.pdb`

No intermediate CSV rewrite is required for MegaScale: `esm_msr.preprocess_megascale.MegaScaleDatasetPreprocessor` filters unreliable `dG_ML` entries, resolves construct sequences against the AlphaFold2 structures, aggregates replicates, and splits single vs. double mutants directly at training, evaluation, or tensor-cache build time (`esm_msr.cache_build`).

---

## 2. External ΔΔG & Combinatorial DMS Benchmarks (`data/preprocessed/`)

External thermostability benchmarks (`s669`, `ssym`, `q3421`, `k3822`, `fireprot`, `protherm`, `ddgun`, `ptmul` / `ptmuld`) and combinatorial functional DMS datasets (`gb1`, `trpb`, `ube4b`, `avgfp`, `pab1`, `bgl3`) are mapped from raw tables in `data/external_datasets/` onto cleaned, 1-indexed, missing-atom-repaired PDB chains in `data/structures/` using `src/esm_msr/preprocess.py`:

```bash
PYTHONPATH=src python -m esm_msr.preprocess \
    --dataset s669 \
    --output data/preprocessed
```

Each run fetches the corresponding RCSB PDB or UniProt AlphaFold structure as needed, strips heteroatoms, repairs missing heavy atoms with PDBFixer (or Modeller when `--modeller` is passed), renumbers the target chain starting at residue `1`, verifies wild-type sequence agreement at every mutated site, and writes `data/preprocessed/<dataset>_mapped.csv` alongside repaired PDBs in `data/structures/`.

---

## 3. Domainome 1.0 Benchmark (Beltran et al., 2025)

### Provenance of Domainome Benchmark Structures

**Domainome benchmarking (both ESM-MSR evaluation in `inference_scripts/esm_msr_testing.py` and all baseline evaluations in `inference_scripts/baselines/`) uses the 522 pre-sliced domain PDB structures distributed in the SPURS paper.**

Specifically:
- SPURS distributes all 522 Domainome domain structures inside `data.tar.gz` on HuggingFace ([`cyclization9/SPURS`](https://huggingface.co/cyclization9/SPURS/resolve/main/data.tar.gz), under `data/dataset/Domainome/files_to_reproduce_analyses/analysis_files/pdb_files/<domain_ID>.pdb`), mirroring the `files_to_reproduce_analyses/analysis_files/pdb_files/` structure set from the Domainome 1.0 release ([Zenodo record 14356805](https://zenodo.org/records/14356805) / DOI `10.5281/zenodo.11043642`).
- Unlike generic single-entry UniProt AlphaFold DB downloads—which omit de novo designed Rocklin domains (`*_rockdoms_1`), non-human controls, and human proteins exceeding 2,700 residues (23 of the 522 domains)—this structure set provides complete coverage of all 522 assayed domains.

### Step 3a: Download Domainome Structures and Raw Table

Download and extract the 522 `<domain_ID>.pdb` structures into `data/domainome1/alphafold_structures` (default `--out` path) and fetch Domainome Supplementary Table 5 (`data/domainome1/SupplementaryTable5.txt`):

```bash
python preprocessing/download_domainome_structures.py --download_table
```

Useful options:
- `--out <dir>`: Target directory for extracted `<domain_ID>.pdb` files (defaults to `<repo_root>/data/domainome1/alphafold_structures`).
- `--source {spurs-hf,spurs-dropbox,zenodo}`: Upstream archive (`spurs-hf` downloads `cyclization9/SPURS` `data.tar.gz`; `zenodo` downloads the smaller 4.1 GB `files_to_reproduce_analyses.tar.gz` directly from Zenodo record `14356805`).
- `--archive /path/to/data.tar.gz`: Extract from an already-downloaded archive without network transfer.

### Step 3b: Truncate, Renumber, and Map Mutations

Run `preprocessing/preprocess_domainome.py` to align each construct to its domain PDB, write 1-based `<domain_ID>_truncated_renumbered.pdb` files, validate every missense mutation, and generate `data/domainome1/domainome_mapped_2026.csv`:

```bash
python preprocessing/preprocess_domainome.py \
    --raw_table data/domainome1/SupplementaryTable5.txt \
    --structures_dir data/domainome1/alphafold_structures \
    --out data/domainome1/domainome_mapped_2026.csv
```

Across all 522 domains, this produces 576,441 missense rows (576,403 structure-mapped rows; 540,265 rows with non-null `scaled_fitness`), with `pdb_file` pointing to the generated `<domain_ID>_truncated_renumbered.pdb` files. Downstream scripts locate this table automatically at `<data_root>/domainome1/domainome_mapped_2026.csv` or via `MSR_DOMAINOME_TABLE=/path/to/domainome_mapped_2026.csv`.

---

## 4. ProteinGym v1.3 Benchmark Preparation

Download `DMS_ProteinGym_substitutions.zip`, `ProteinGym_AF2_structures.zip`, and `DMS_substitutions.csv` from [Zenodo record 15293562](https://zenodo.org/records/15293562) into a single directory, then map all 217 DMS substitution assays to AlphaFold2 structure coordinates:

```bash
python preprocessing/pgym_preprocess.py \
    --proteingym_dir /path/to/ProteinGym_v1.3 \
    --out pgym_inputs
```

For multi-fragment proteins >2,700 aa (`BRCA2_HUMAN`, `TTN_HUMAN`, etc.), `pgym_preprocess.py` automatically selects the overlapping AlphaFold2 fragment and applies the appropriate residue coordinate offset, emitting `pgym_inputs/<DMS_id>.csv` and `pgym_inputs/manifest.csv` for `inference_scripts/esm_msr_testing.py --protein_gym` and baseline scorers.

---

## 5. Homology-Controlled MegaScale Splits & Overlap Auditing

MegaScale train/validation/test splits (`data/splits_oct06_capped.pkl`) exclude homologs of held-out thermostability and functional benchmarks using sequence search (`mmseqs`), structure alignment (`foldseek`), exact BLOSUM62 local Smith–Waterman verification, and connected-component bridge pruning (`preprocessing/homology.py`).

Requires external CLI binaries `mmseqs` (MMseqs2) and `foldseek`. Running with default parameters (`censor_range=False`, `--homology both`, `--seq_evalue 1e-3`, `--struct_evalue 1e-3`, `--max_identity 0.40`, `--max_tm 0.70`, `--tm_norm max`, `--seed 42`) reproduces `data/splits_oct06_capped.pkl`, `data/splits_oct06_capped.csv`, and `data/splits_oct06_capped_library_assignment.csv` byte-for-byte:

```bash
PYTHONPATH=src:preprocessing python preprocessing/split_tsuboyama.py \
    --output splits_oct06_capped \
    --data_root data \
    --mmseqs_bin mmseqs \
    --foldseek_bin foldseek
```

To audit one or more split pickles against the pairwise homology table emitted by `split_tsuboyama.py`:

```bash
PYTHONPATH=src:preprocessing python preprocessing/split_overlap_report.py \
    --pairs data/splits_oct06_capped_homology_pairs.csv \
    --library_table data/splits_oct06_capped_library_assignment.csv \
    --splits capped=data/splits_oct06_capped.pkl \
    --out_prefix data/split_overlap_audit
```

---

## 6. Biophysical & MSA Feature Extraction (Optional)

For downstream biophysical property-category analysis (`analysis/14_property_category_gain.py`), `preprocessing/compute_features.py` annotates mapped ΔΔG tables (`q3421`, `fireprot`, `s669`, `ssym`, `k3822`) with DSSP solvent accessibility (`rel_ASA`) and secondary structure (`SS`), QuantiProt AAIndex deltas (`kdh_*`, `vol_*`, `chg_*`), inter-chain interface proximity, and MSA conservation metrics:

```bash
python preprocessing/compute_features.py \
    --dataset s669 \
    --alistat_loc /path/to/AliStat \
    --msa_dir data/tsuboyama/tsuboyama_msas \
    --reduced_msa_dir data/tsuboyama/tsuboyama_msas_reduced \
    --output_root .
```

Optional external requirements for this step: `quantiprot` (`pip install quantiprot`), `mkdssp` on `PATH`, and [AliStat](https://github.com/thomaskf/AliStat).
