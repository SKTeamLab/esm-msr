# Homology, Split Construction & Leakage Provenance Audits (`analysis/audits/homology_and_splits/`)

This package consolidates all offline homology searches, split-construction matrices, `mega_splits` provenance tables, and literature training-set leakage audits into two clearly separated tiers so active figure pipelines in `analysis/pipelines/` never execute external MMseqs2/Foldseek jobs or cross-import audit scripts.

## Directory Layout

1. **`active_builders/` — Canonical Split, Holdout & Provenance Builders**
   - `tsuboyama_rule_flags.py`: Applies the four-part Homology-Restricted MegaScale split rule (`MMseqs2 E <= 1e-3`, `BLOSUM62 shorter-sequence identity > 0.40`, `Foldseek E <= 1e-3`, `Foldseek TM-score > 0.70`) against the 172 capped training libraries across **ProteinGym** (41/217 flagged; 40/213 after `PG_DROP` -> **173 held-out assays**), **Domainome** (86/522 flagged -> **436 held-out domains**), **MGnify** (27,687 flagged), and **Rocklin de novo designs** (258 flagged). Writes `analysis/reference/tsuboyama_rule_flags.csv` (consumed by `msr_analysis.heldout`).
   - `set_overlap_matrix.py`: Computes pairwise maximum shorter-sequence MMseqs2 identity and Foldseek TM-score across the `train`/`val`/`test` MegaScale splits, 9 surviving external thermostability sets, and 4 held-out large benchmarks. Writes `analysis/reference/split_set_overlap_{identity,tm}.csv` (plotted in `analysis/results/00_supplementary_figs/split_construction/` by `pipelines/p10_splits_and_leakage.py`).
   - `build_mega_tasks.py`: Constructs canonical task spine tables `_tasks/tsuboyama_mega_{val,test}.csv.gz` from `FINAL_p0_mega` seed 1–3 prediction tables, verifying seed agreement and backbone sequence reconstruction via `msr_analysis.structures.parse_mutations`.
   - `mega_split_provenance.py`: Audits `data/mega_splits.pkl` against upstream SPURS/ThermoMPNN/MutateEverything splits and evaluates the 28 `mega_splits` test libraries under the four-part homology rule (**16 of 28 test libraries linked to `mega_splits` train/val**, leaving **12 homology-clean test libraries**). Writes 8 provenance CSVs to `provenance/` (consumed by `pipelines/p10_splits_and_leakage.py` -> `analysis/results/01_megascale/mega_splits/provenance/`).
   - `export_reference.py`: Verifies (`--verify`) or re-syncs (`--sync`) all exported audit tables in `analysis/reference/` and `docs/benchmarks/`, including static committed auxiliary tables (`pgym_thermompnn_corrected_s1_per_dms.csv`).

2. **`oct2026_leak_audit/` — Literature Training-Data Leakage Audit (`01`..`20`)**
   - Audits row-level mutation (`M`/`V`) and coverage-gated protein/homolog (`P`/`H`/`h`, requiring `>= 50%` test-protein coverage) overlaps between published model training corpora (EsmTherm, IFUM, DDGemb, DDMut, GeoDDG-3D, GeoFitness, ESM3ΔG, Rosetta `beta_nov16`) and every benchmark test set.
   - Outputs feed `analysis/reference/leak_audit_{matrix,inventories,pg_per_dms,structure_homology}.csv` and `mgnify_homology_classes.csv.gz` via `active_builders/export_reference.py --sync`.

3. **`data/` & `provenance/`**
   - `data/`: Small tracked literature training split/protein inventories (`S2450`, `S9028`, `DD_SM1218`, `DD_SM444`, `esmtherm`, `ifum`, `kellogg2011_1210`).
   - `provenance/`: Committed `mega_splits` reconciliation and test-library homology CSVs (`mega_test_library_homology.csv`, `homology_test_vs_trainval.csv`, `test_composition.csv`, etc.).

## Usage

```bash
# Verify all committed reference & provenance CSVs (no external raw search dirs needed):
PYTHONPATH=src:analysis python -m audits.homology_and_splits.active_builders.export_reference --verify

# Recompute mega_splits provenance CSVs:
PYTHONPATH=src:analysis python -m audits.homology_and_splits.active_builders.mega_split_provenance
```
