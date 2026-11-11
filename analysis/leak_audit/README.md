# Leak-Audit Pipeline (`analysis/leak_audit/`)

End-to-end reproducible pipeline producing [`docs/LEAKAGE_AUDIT_OCT2026.md`](../../docs/LEAKAGE_AUDIT_OCT2026.md),
[`docs/SPLITTING_AND_HOMOLOGY_AUDIT.md`](../../docs/SPLITTING_AND_HOMOLOGY_AUDIT.md),
[`docs/BENCHMARKS_OCT2026.md`](../../docs/BENCHMARKS_OCT2026.md) §6, and all machine-readable audit artifacts in
[`docs/benchmarks/`](../../docs/benchmarks/) (synced to [`analysis/reference/`](../reference/)):

- `docs/benchmarks/leak_audit_matrix.csv` — per-`(model, dataset)` row counts, leak fractions, compact codes, and superscripts
- `docs/benchmarks/leak_audit_inventories.csv` — provenance and scope of all 18 training inventories
- `docs/benchmarks/leak_audit_pg_per_dms.csv` — per-assay ProteinGym exact/constituent/same-protein/sequence+Foldseek homology breakdown (all 217 DMS)
- `docs/benchmarks/leak_audit_homology.csv` — protein-level (`P` / `H` / `h` / `low` / `none`) row counts across all `(inventory, dataset)` pairs
- `docs/benchmarks/leak_audit_structure_homology.csv` — Foldseek 3D vs MMseqs2 sequence homology summary on Domainome (522 domains) and ProteinGym (all 217 DMS)
- `docs/benchmarks/leak_audit_code_exact.csv` — verified mutation-level overlaps for `DDGemb` (`S2450`), `DDMut` (`S9028` + `SM1242`), and `GeoDDG` (`S8754`)
- `docs/benchmarks/leak_audit_esm3dg_libraries.csv` — per-scaffold MGnify homology audit across the 33 capped-test scaffolds
- `docs/benchmarks/split_homology_audit.json` — verification of the `splits_oct06_capped` quarantine, bridge pruning, and threshold sensitivity sweep
- `docs/benchmarks/leak_audit.sqlite` — consolidated SQLite database containing all audit tables

---

## Directory Layout & Tracked Reference Data

Small plain-text upstream split and training reference tables (**< 800 KB total**) are tracked directly under
[`analysis/leak_audit/data/`](data/), while large raw datasets, intermediate pickles, FASTA databases, and
MMseqs2/Foldseek alignments live in `$LEAK_DATA_DIR` (default `../leak_audit/data`) and `$LEAK_OUT_DIR`
(default `../leak_audit/out`) and are excluded via [`data/.gitignore`](data/.gitignore):

| Tracked file in `data/` | Size | Source / Generator |
|---|---:|---|
| `S2450.tsv`, `S2450.fasta` | 87 KB | DDGemb training set ([Savojardo et al. 2024](https://ddgemb.biocomp.unibo.it/datasets/)) |
| `S9028.csv`, `DD_SM1242.csv` | 386 KB | DDMut single- and multi-mutant training sets ([Zhou et al. 2023](https://biosig.lab.uq.edu.au/ddmut/data)) |
| `ifum_train_proteins.csv` | 27 KB | Parsed from IFUM supplementary `dataset_splits.pdf` (`00b_extract_ifum_split.py`) |
| `esmtherm_split.csv`, `esmtherm_train_proteins.csv` | 206 KB | EsmTherm mutant-level split (`00_build_inventories.py`) |
| `tmpnnd_train_proteins.csv` | 5 KB | 134 of 172 capped-train scaffolds seen by retrained ThermoMPNN-D (`05c_tmpnnd_inventory.py`) |
| `spurs_single_train_proteins.csv`, `spurs_multi_train_proteins.csv` | 43 KB | Upstream SPURS MegaScale protein inventories (`00_build_inventories.py`) |
| `thermompnn_upstream_train_proteins.csv` | 20 KB | Upstream ThermoMPNN Zenodo MegaScale/FireProt protein index (`00_build_inventories.py`) |

Large external inputs expected in `$LEAK_DATA_DIR` / `$SOFTWARE_DIR` for a full cold rebuild (all public; URLs in
[`docs/LEAKAGE_AUDIT_OCT2026.md`](../../docs/LEAKAGE_AUDIT_OCT2026.md) §3):
- Tsuboyama MegaScale CSV + `AlphaFold_model_PDBs/` (`$SOFTWARE_DIR/esm-msr/data/tsuboyama/`)
- Preprocessed benchmark CSVs and original/processed PDBs (`repo/data/preprocessed/`, `$SOFTWARE_DIR/esm-msr/data/structures/`)
- GeoDDG `GS_S8754.csv` (`$LEAK_DATA_DIR/raw/GS_S8754.csv`, 2.7 MB)
- ESM3ΔG `mgnify_index.csv` (`$LEAK_DATA_DIR/raw/mgnify_index.csv`, 138 MB; 960,216 `train` sequences)
- Domainome supplementary table & structures (`$SOFTWARE_DIR/esm-msr/data/domainome1/`)
- ProteinGym v1.3 substitutions & AF2 structures (`$LEAK_ROOT/ProteinGym/`, Zenodo record `15293562`)

---

## Regenerating Everything

Run the complete pipeline in topological dependency order with a single command:

```bash
python analysis/leak_audit/run_all.py
```

Or resume from any step substring (e.g. `--from-step 09` or `--skip-upstream`):

```bash
python analysis/leak_audit/run_all.py --from-step 09
```

### Execution DAG (`run_all.py`)

```text
00_build_inventories.py     # Upstream SPURS, EsmTherm, and ThermoMPNN CSVs (with .pdb_<mut> normalization)
00b_extract_ifum_split.py   # Parse IFUM dataset_splits.pdf -> data/ifum_train_proteins.csv
01_megascale.py             # MegaScale library & variant tables (synthesizing base code_wt entries)
02_testsets.py              # Parse all 23 benchmark datasets into canonical (pid, seq, muts) tables
04_overlap_repo_models.py   # Direct MegaScale split sanity check -> capped_split_sets.pkl
05b_inventory_proteins.py   # Extract WT sequences & author->0-idx PDB residue maps (_original.pdb preferred)
05_proteins.py              # Unified protein corpus + MMseqs2 all-vs-all (-s 7.5, SHA-1 cache-invalidated)
05c_tmpnnd_inventory.py     # Determine the 134/172 capped-train scaffolds seen by retrained ThermoMPNN-D
05d_structure_homology.py   # Foldseek exhaustive 3D structural search on Domainome (522) & ProteinGym (217)
00c_prepare_mgnify.py       # Build mgnify_train.fa + test_queries.fa and run MMseqs2 (-s 7.5)
06_families.py              # Symmetric >=95% id / >=90% cov / >=0.80 len_ratio families & position maps
07_relations.py             # Quick relation summary against the 172 capped-train scaffolds
09_homology_audit.py        # Verify splits_oct06_capped quarantine, bridge cuts, and threshold sensitivity
10_overlap_engine.py        # Exact (M), constituent (m), and same-protein (P) engine for MegaScale models
11_homology_classes.py      # Non-shadowed P / H / h classification (MMseqs2 + Foldseek on Domainome & PG)
12_code_exact.py            # Sequence-verified M / m counts for DDGemb S2450, DDMut S9028+SM1242, GeoDDG S8754
15_mgnify.py                # Non-shadowed ESM3dG MGnify homology + 33-scaffold capped-test breakdown
19_esm3dg_exact.py          # Exact WT & mutant SHA-1 check against the 960,216 MGnify train sequences
14_pg.py                    # Per-DMS ProteinGym overlap + Foldseek homology -> leak_audit_pg_per_dms.csv
13_matrix.py                # Assemble per-(model, dataset) leak codes (retrained SPURS/TMPNN-D; '–' for zero-shot)
18_db.py                    # Export leak_audit.sqlite and CSV/JSON artifacts to docs/benchmarks/ & analysis/reference/
```
