# `analysis/`: Modular Benchmark Evaluation, Figure Pipelines, Mechanistic Studies, and Homology Audits

CPU-only post-inference evaluation, statistical bootstrapping, figure/table generation, and training-set homology auditing for ESM-MSR (`FINAL_p0`) and all retrained, supervised, and zero-shot baselines.

All 45 legacy top-level numbered scripts (`00_*.py`–`99_*.py`) and 7 legacy sub-study directories (`ablation_vs_p0/`, `combined_vs_wt/`, `faure_epistasis/`, `leak_audit/`, `pg_activity/`, `proteingym/`, `tsuboyama/mega_splits/`) have been consolidated into:
- **`msr_analysis/`**: 19-module shared library (paths, immutable model roster, holdout filtering, prediction frames, joint domain/seed bootstrap engine, epistasis hierarchy, Tobit link, calibration, screening, structures, plotting primitives, and table formatters).
- **`pipelines/`**: 12 self-contained, ordered figure & table pipelines (`p01`–`p12`) with zero cross-script `importlib` or `sys.path` hacks.
- **`studies/`**: 2 standalone mechanistic studies (`faure_epistasis`, `sigma_mechanisms`).
- **`audits/`**: Offline homology, split-provenance, and literature leakage builders (`homology_and_splits`, `baseline_bridge`).
- **`scoring/`**: Cluster prediction collection, baseline run scoring, and fixed-slot hardware resource summaries.

Detailed methodological and audit notes live in [`docs/`](docs/):
- [`docs/ARCHITECTURE_AND_FINDINGS.md`](docs/ARCHITECTURE_AND_FINDINGS.md) — Data-flow architecture, methodological/statistical fixes, scientific findings, and composite figure suite.
- [`docs/SEED_AUDIT.md`](docs/SEED_AUDIT.md) — Triplicate seed averaging and joint `[cluster × seed]` bootstrap resampling across every pipeline and study.
- [`docs/HOMOLOGY_FILTER_AUDIT.md`](docs/HOMOLOGY_FILTER_AUDIT.md) — Canonical four-part sequence + 3D structure holdout filter (`msr_analysis/heldout.py`) vs historical coverage-gated literature leak codes.

---

## Quick Start & Unified CLI

Run from the repository root using the `msr_venv` Python 3.11 environment (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `biopython`, `pytest`). Every pipeline runs on CPU from precomputed prediction CSVs under `MSR_ROOT` (default `~/playground/esm-msr-devel`).

```bash
# List all 12 pipeline stages and 2 optional mechanistic studies
PYTHONPATH=.:src:analysis python -m analysis.run --list

# Run the complete main + supplementary pipeline suite (B = 1000 bootstrap draws)
./analysis/run_all.sh

# Fast smoke-test pass across all stages (sets B = 100)
./analysis/run_all.sh --fast

# Run specific pipeline stages by code or stage name
PYTHONPATH=.:src:analysis python -m analysis.run --only p01,p03,p05,p12
PYTHONPATH=.:src:analysis python -m analysis.run --only tsuboyama --split mega

# Include standalone mechanistic studies (faure_epistasis, sigma_mechanisms)
PYTHONPATH=.:src:analysis python -m analysis.run --studies --B 1000

# Run unit tests across tests/ and analysis/tests/
pytest
```

### CLI Options (`python -m analysis.run`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--list` | `False` | Print all registered `p01`–`p12` stages and `study:*` targets in execution order and exit. |
| `--only STAGE [...]` | all `p01`–`p12` | Comma- or space-separated stage codes (`p01`), stage names (`tsuboyama`), or study targets (`study:sigma_mechanisms`). |
| `--skip STAGE [...]` | none | Comma- or space-separated stages to skip. |
| `--studies` | `False` | Also execute `studies/faure_epistasis` and `studies/sigma_mechanisms` after the selected pipelines. |
| `--split {capped,mega}` | `capped` | MegaScale split passed explicitly to split-aware pipelines (`p02`, `p06`, `p07`, `p09`); `--split mega` writes to `results/01_megascale/mega_splits/` without mutating environment variables. |
| `--B INT` | `1000` | Number of bootstrap replicates forwarded to every stage. |
| `--fast` | `False` | Smoke-test preset (`--B 100`). |

Every individual stage or study module also runs directly via `python -m analysis.pipelines.<module> [--B 1000] [--fast]`.

---

## Directory Layout

```text
analysis/
├── README.md                     # Architecture, CLI reference, module guide, and results hierarchy
├── run.py                        # Unified Python CLI orchestrator (`python -m analysis.run`)
├── run_all.sh                    # Thin shell wrapper setting PYTHONPATH and invoking `analysis.run "$@"`
├── msr_analysis/                 # Core shared Python library (19 modules; zero script cross-imports)
├── pipelines/                    # 12 ordered benchmark & figure pipelines (`p01` .. `p12`)
├── studies/                      # Self-contained mechanistic deep-dives
│   ├── faure_epistasis/          #   BindingPCA/AbundancePCA block geometry, SNR & noise ceilings
│   └── sigma_mechanisms/         #   FINAL_p0 LoRA scale (σ ∈ {0, 0.5, 1}) mechanistic decomposition
├── audits/                       # Offline homology searches, split provenance & cluster inventory
│   ├── homology_and_splits/      #   `active_builders/` (4-part flags, overlap matrices, mega provenance)
│   │                             #   + `oct2026_leak_audit/` (steps 01..20) + `data/` + `provenance/`
│   └── baseline_bridge/          #   Local vs cluster training/checkpoint inventory (`inventory.py`)
├── scoring/                      # Cluster rsync/manifest collector, baseline scorers & GPU slot profiler
├── reference/                    # Committed reference tables (`tsuboyama_rule_flags.csv`, `leak_audit_*.csv`,
│                                 #   `benchmark_tables/T*.csv`, `pg_activity/*.csv`, `dG_wt.json`, ...)
├── cache/                        # Git-ignored Frame & bootstrap caches (`frame_*.pkl`, `boot_ext_mutation.pkl`,
│                                 #   `core_rho_bootstrap.csv`, `pgym_ablation_matrices_v1.pkl`; atomic temp + os.replace)
├── results/                      # Canonical output hierarchy (see tree below)
├── docs/                         # Audit & architecture documentation
│   ├── ARCHITECTURE_AND_FINDINGS.md
│   ├── SEED_AUDIT.md
│   └── HOMOLOGY_FILTER_AUDIT.md
└── tests/                        # Unit tests for bootstrap, roster, frames, holdout, links, epistasis & screening
```

### Canonical `analysis/results/` Hierarchy

All generated figures (`.png` + `.pdf`), CSVs, Markdown/LaTeX tables, and manifests follow a strict dataset-first, cross-dataset-second layout. Numbered `figX_` / `fig_` prefixes are stripped everywhere outside `00_main_text_figs/` and `00_supplementary_figs/`, and all 3-DMS functional assay outputs (`DLG4`, `GRB2`, `MYO`) are isolated under `99_misc/dms3/`:

```text
analysis/results/
├── 00_for_reviewers/                       # Reviewer-facing ablations & split comparisons
│   ├── baseline_nobb_ablation.{pdf,png}
│   ├── split_best_partner_hyperopt.{csv,pdf,png}
│   └── reference/                          # Published reference & historical table cell verification
│       ├── published_reference.{pdf,png}
│       ├── spurs_published_vs_retrained.{csv,pdf,png}
│       └── validation_vs_benchmarks.csv
├── 00_main_text_figs/                      # Main-text figures (`figure3`, `fig2_fig3_stacked`, `epistasis_levels_e`, `epistasis_main_composite`)
├── 00_supplementary_figs/                  # Supplementary composites, `pg_ours_vs_published`, `holdout_filter`, `masking_schemes`, `split_construction/`
├── 01_megascale/                           # Homology-Restricted MegaScale (`splits_oct06_capped`; `megascale_*` stems)
│   ├── epistasis/                          #   MegaScale-only epistasis hierarchy & mechanisms (`dG_additive_vs_pred_dddG/`, `dddG_scatters/`, `pair_heatmaps/`)
│   ├── gain_fit/                           #   Logarithmic virtual-screening gain fits (`g(K) = a - b ln K`)
│   ├── screening/                          #   MegaScale virtual-screening curves, per-protein CSVs & `property_gain/`
│   └── mega_splits/                        #   Mirrored `--split mega` hierarchy (`epistasis/`, `epistasis/pair_heatmaps/`, `screening/`, `gain_fit/`, `provenance/`)
├── 02_thermostability/                     # 12 external ΔΔG thermostability benchmarks (`EXT12`)
│   ├── rmse/                               #   Single-donor linear scale-and-bias calibration & RMSE/MAE
│   └── screening/                          #   Pooled external thermostability virtual screening
├── 03_proteingym/                          # Held-out ProteinGym (173 canonical assays across 5 categories)
│   └── screening/                          #   ProteinGym Activity & enzyme virtual-screening curves
├── 04_domainome/                           # Held-out Domainome 1.0 (436 human domains; Domainome only)
│   └── screening/                          #   Domainome per-protein virtual-screening CSVs & curves
├── 05_mgnify/                              # Held-out MGnify stability & Rocklin de novo (`new_data_epistasis/`, `rocklin_designs/`)
├── 06_summary/                             # Cross-dataset thermostability & functional benchmark summaries + paired p-values
├── 07_screening_overview/                  # Cross-dataset single- & multi-mutant screening overviews
├── 08_sequence_length/                     # Sequence-length distributions & performance scaling across main benchmarks
├── 09_ablation/                            # Unified `FINAL_*` architecture/loss ablation tables & `ablation_architecture_suite`
├── 10_inference_resources/                 # Fixed-slot RTX A4500 wall-clock & peak RAM/VRAM summaries
├── 99_misc/                                # Auxiliary benchmarks & standalone mechanistic deep-dives
│   ├── dms3/                               #   All 3-DMS (`DLG4`, `GRB2`, `MYO`) benchmark, epistasis, length & ablation outputs
│   ├── dual_view/                          #   Dual-view (`comb`) vs WT-view (`wt`) regime analysis across `FINAL_*` families
│   ├── esm_msr_vs_esm3dg/                  #   Direct ESM-MSR vs `ESM3ΔG` comparison across held-out sets
│   ├── faure_epistasis/                    #   Faure et al. PCA vs MegaScale combinatorial block geometry & SNR ceilings
│   ├── rosetta_multi_vs_additive/          #   Rosetta direct Cartesian multi-mutant vs additive single-mutant ΔΔG
│   └── sigma_mechanisms/                   #   `FINAL_p0` LoRA scale (`σ ∈ {0, 0.5, 1}`) mechanistic decomposition
├── manifest_outputs.csv
└── table_figure_map.csv
```

---

## Core Library Guide (`analysis/msr_analysis/`)

| Module | Responsibility & Key Exports |
| :--- | :--- |
| `paths.py` | Repo-relative (`REPO_ROOT`, `ANALYSIS_ROOT`, `DATA_DIR`, `CACHE`, `REFERENCE`, `RESULTS`, `CORE_RHO_BOOTSTRAP_CSV`, `VALIDATION_VS_BENCHMARKS_CSV`, `CANONICAL_RESULT_TOPS`) and external `MSR_ROOT` paths (`MSR`, `FINAL_BENCH`, `FINAL_CTX`, `FINAL_PGYM_*`, `FINAL_MGNIFY*`, `BASE`, `TASKS`, `PGYM_INPUTS`, `TSUBOYAMA_PDB_DIR`, `TSUBOYAMA_FEATS_CSV`, `SALVAGED_HOMOLOGY_CSV`), split-aware `results_dir(name, split)` (enforcing canonical result folders and rejecting retired paths), `strip_fig_prefix()`, `source_signature()`, and symlink-safe atomic writers (`atomic_pickle_dump`, `atomic_to_csv`, `atomic_write_text`). |
| `models.py` | Single immutable visual & key roster: `Model`, `StyleSpec` (frozen mapping with `.with_overrides()`), `ROSTER`, `BY_NAME`, `ORDER`, `UNIFIED`, `ABLATION_STYLES` (all 11 `FINAL_*` families + 2 cross-mask arms), `CATEGORY_COLOR`, `VARIANT_COLOR`, `column()`. |
| `registry.py` | Dataset cohorts (`EXT12`, `DMS3`, `DMS_AUX`, `TSU_CAPPED`, `TSU_MEGA`, `MAIN_BENCHMARK_DATASETS`, `ALL_DATASETS`, `PER_PROTEIN`, `LABEL`), `tsu_datasets(split)`, `main_benchmark_datasets(split)`, `all_datasets(split)`, canonical result path re-exports, and read-only `MODELS` (`MappingProxyType`) registering every `.single`, `.multi`, `.add`, `.epi`, and display-name style. |
| `heldout.py` | Canonical four-part sequence + structure holdout filter backed by `reference/tsuboyama_rule_flags.csv`: `flags()`, `excluded(dataset)` (`proteingym=41`, `domainome=86`, `mgnify=27,687`, `rocklin_denovo=258`), and `keep_ids(dataset)`. |
| `data.py` | `Frame(name, spine, preds, notes)`, `build_spine()`, `load_frame(ds, refresh=False, heldout=True, check_sources=False)`, `sum_singles()`, `singles_of_doubles()`, `synthesize_missing_epi()` (derives `:epi` on the fly from `pred - pred+` across cached pickles), and `add_esm3_masked()`. |
| `final_runs.py` | Loader for all 11 `FINAL_*` ablation families (`FINAL_p0` reference, `p1`, `p1_marg`, `pw0`, `norank`, `rcthr0`, `norange`, `mse`, `noseqhwm`, `r2a4`, `mega`) plus cross-masking arms (`FINAL_p0@independent`, `FINAL_p1@unmasked`): `load_final_frame()`, `pgym_per_dms()`, `pgym_per_dms_seeds()`. |
| `mgnify.py` | Loader for the 766,790-variant MGnify stability benchmark and 1,000 Rocklin *de novo* designs (`load_mgnify`), joining v2 `mgnify_mapped.csv` labels, applying 4-part holdout masking (`heldout=True`), and computing both `latent` and post-hoc Tobit-linked (`<head>~`) predictions. |
| `proteingym.py` | Authoritative ProteinGym loader & scorer over the **173 canonical held-out assays** (`PG_DROP` + `heldout.keep_ids`: `Activity=38, Binding=13, Expression=18, OrganismalFitness=67, Stability=37`): `pgym_reference()`, `load_final_pgym_seeds()`, `load_sigma_pgym()`, `load_all_models_pgym()`, `boot_pgym_categories()` (joint `[category-stratified assay × 3-seed]` bootstrap), `sigma_delta_pvalues()`, and `draw_property_sigma_panel()`. |
| `external.py` | Unified external benchmark helpers: `model_key_for_dataset()`, `seed_frame()`, `load_mutation_boot()` (`cache/boot_ext_mutation.pkl` with triplicate seed resampling), and `fresh_frames()`. |
| `common.py` | Dataset/frame utilities: `has_multi()`, `primary_key()`, `out_dir(name, split)`, `load_rho_df()` (reads `cache/core_rho_bootstrap.csv`), `primary_ci()`, `epi_rows()` (8,494 val / 12,455 test doubles with both constituent singles measured), `subframe()`, and `leak_table()`. |
| `boot.py` | Exact $O(n)$ count-weighted `SpearmanEngine` and `DatasetBoot` supporting protein-, site-, and mutation-level resampling (`bootstrap_rho`) with exact per-seed-count multinomial draws for arbitrary `n_seeds` ($1, 2, 3, \dots$) and true site-level resampling on `PER_PROTEIN` datasets. |
| `stats.py` | Unified statistical testing: `holm()`, `boot_pvalue()` ($1/(B+1)$ finite-sample floor), `fmt_p()`, `seed_sd()` (`ddof=1`), `paired_boot_test()`, `make_domain_seed_draws()`, `boot_domain_seeds()`, and `ndcg_at_k(mode='rectified'|'shifted')`. |
| `epistasis.py` | Adapter over `esm_msr.epi_metrics.Table`: `make_epi_table()`, `compute_epi_metrics()` (6 core + 16 extended metrics), `boot_epistasis_models()` (parallelized domain-cloning + 3-seed bootstrap), and `smooth_on()`. |
| `tobit.py` | `FINAL_p0` post-hoc Tobit observation link `linked(latent, dg_wt)` ($\sigma_y=0.28, \mu_f=-0.14, \sigma_f=0.5$) via `esm_msr.val_stats.tobit_link`, plus analytical `softplus_link` and `leaky_link` curves for floor-saturation diagnostics. |
| `calib.py` | Single-donor OLS linear scale-and-bias calibration (`fit_scale_bias`, `rmse`, `mae`, `calibration_key`, `fit_frame_scale_bias`, `calibrated_preds`), automatically selecting `:comb` on multi-mutant sets (`ptmul`, `ptmuld`) and `:wt` on singles sets. |
| `screening.py` | Experimental virtual-screening metrics (`protein_screen`, `eligible_proteins`, `per_protein_table`, `per_protein_curve`, `pooled_screen`) with fixed protein cohorts across $K$-sweeps, deterministic tie-breaking, and explicit separation of `rectified_gain` ($\sum \max(y,0)$) from unrectified `mean_y`. |
| `structures.py` | PDB/AF2 coordinate loader (`load_pdb_residues` with virtual Glycine $C_\beta$ reconstruction), construct sequence alignment (`align_domain_residues`), and $C_\beta$ / heavy-atom inter-residue distance matrices (`cb_distance_matrix`, `heavy_atom_distance_matrix`). |
| `style.py` | Publication figure styling: immutable `style_of(key, overrides=None)`, `dot_panel`, `bar_panel`, `forest_overview`, `raincloud` (with `[lo, hi]` 95% CI whiskers), `density_scatter`, `multi_seed_kde_grid`, and `finish_figure`. |
| `tables.py` | Benchmark table parser (`read_table`, `numeric`) and unified `.csv` + `.md` + `.tex` statistical table exporter (`write_stat_table`, `format_stat_table_md`, `format_stat_table_tex`). |

---

## Key Methodological Conventions

1. **One Immutable Look per Model (`msr_analysis/models.py`, `registry.py`)**:
   - Models appear in a single canonical order across figures: zero-shot (`ProSST`, `ProteinMPNN`, `ESM3-small-open`), models retrained on `splits_oct06_capped` (`ESM-MSR`, `MutateEverything`, `ThermoMPNN`, `SPURS`), supervised models not trained on MegaScale (`GeoDDG-3D`, `Rosetta`, `ESM3ΔG`), and—in supplementary comparisons only—MegaScale-trained models (`ESMtherm*`, `IFUM*`, `SPURS published*`, `DDGemb`, `DDMut`).
   - **ESM-MSR** is always the final deployed checkpoint (`FINAL_p0`, key `msrp0`, seeds 1–3) and `ESM3-small-open` is its $\sigma = 0$ readout (`msrrel@0`). Legacy development keys (`msr`, `msrdev`, `msrrel`) are retired from `registry.MODELS` and exist in raw `Frame` dicts solely for historical `BENCHMARKS_OCT2026.md` cell verification (`p12_manifest`, `p10_splits_and_leakage`).
   - **Hollow markers (dashed lines)** denote the single-mutant version (`msrp0:wt`, `msrrel@0:wt`, `me1`, `tmpnn`, `spurs`, `+` additive sums); **filled markers (solid lines)** denote the multi-mutant version (`msrp0:comb`, `msrrel@0:comb`, `me2`, `tmpnnd`, `spursm`) or single-version models. Each benchmark scores the deployed version matching its composition (`wt` on singles-only sets, `comb` when multi-mutants are present).
   - **ESM3ΔG placement**: `ESM3ΔG` is omitted from Tsuboyama val/test (`NO_ESM3DG`, since its MGnify Stability training data overlap MegaScale homologs) and marked non-held-out on MGnify/Rocklin, while **included** in Figure 3b (`p05_summary_main`, `LEFT_OUT = set()`) on ProteinGym and Domainome with a red training-overlap ring on its ProteinGym average and Stability category.

2. **Consistent $1\times$ Generalized Epistasis (`:epi`)**:
   - Because the deployed multi-mutant readout is `comb_AB = 0.5 * (MT_AB + WT_AB)` and `WT_AB = WT_A + WT_B` holds only under unmasked wild-type scoring, every epistasis readout subtracts explicit constituent singles (`msr_analysis.data.sum_singles`):
     $$\Delta\Delta\Delta G_{\text{pred}} = \tfrac{1}{2}(\text{MT}_{AB} + \text{WT}_{AB}) - \text{WT}_A - \text{WT}_B = \text{comb}_{AB} - \text{WT}_A - \text{WT}_B$$
   - Public and zero-shot models carrying `pred` and `pred_additive` (`+`) expose `pred - pred+` as `<key>:epi` automatically via `synthesize_missing_epi()`.

3. **Canonical Four-Part Holdout Filter (`msr_analysis/heldout.py`)**:
   - ProteinGym, Domainome, MGnify, and Rocklin *de novo* designs are filtered identically against the **172 capped training libraries** (`audits/homology_and_splits/active_builders/tsuboyama_rule_flags.py` $\rightarrow$ `reference/tsuboyama_rule_flags.csv`): any protein meeting `MMseqs2 E <= 1e-3`, `BLOSUM62 shorter-sequence identity > 0.40`, `Foldseek E <= 1e-3`, or `Foldseek TM-score > 0.70` is excluded (`41/217` ProteinGym assays, leaving **173 held-out assays** after dropping the 4 `PG_DROP` $>1{,}800$-aa assays: `Activity=38` [including **16 enzymes**], `Binding=13`, `Expression=18`, `OrganismalFitness=67`, `Stability=37`; **436/522** Domainome domains; **152,680/180,367** MGnify scaffolds; **742/1,000** Rocklin designs).

4. **Triplicate Seeds & Joint Resampling Everywhere (`docs/SEED_AUDIT.md`)**:
   - Every retrained model (`FINAL_p0`, all `FINAL_*` ablations, `MutateEverything`, `ThermoMPNN`, `SPURS`, and our 3 `ProteinMPNN` runs on ProteinGym / Activity screening) uses **seeds 1–3** (point estimate = mean of per-seed statistics; sample SD uses `ddof=1`; bootstrap resamples seeds jointly with proteins, domains, sites, or assays). The sole data-completeness exception is `FINAL_p1` combined on ProteinGym (seeds 1 and 3; seed 2 covers 126/219 assays).
   - In ProteinGym Activity screening (`p09_screening`), `ProteinMPNN` uses **our 3 per-variant runs** and `ESM3-small-open` uses **our unmasked $\sigma = 0$ readout**, whereas `ProSST` uses ProteinGym's shipped zero-shot per-variant scores (`p03_proteingym` audits our runs against ProteinGym's shipped scores across all categories and Activity curves).

---

## Pipeline Stages, Studies, Audits & Output Mapping

### 1. Active Pipelines (`analysis/pipelines/`)

| Stage | Module | Scope & Benchmarks Covered | Key Outputs (`analysis/results/` & `analysis/cache/`) |
| :--- | :--- | :--- | :--- |
| **`p01`** | `p01_benchmarks_external.py` (`benchmarks_external`) | Core Spearman $\rho$ bootstrap cache, 12 external $\Delta\Delta G$ thermostability benchmarks (`EXT12`), held-out Domainome 1.0 (`436` domains), and 3 functional DMS assays (`DMS3`). | `cache/core_rho_bootstrap.csv`, `02_thermostability/{thermostability_rho.csv, thermo_external_rho.*, thermo_overview.*, thermo_interval_types.*}`, `04_domainome/{domainome_vs_retrains.*, domainome_sigma_vs_zero_shot.*, domainome_sigma_sweep.*}`, `99_misc/dms3/{dms_vs_retrains.*, dms_sigma_vs_zero_shot.*, dms_sigma_sweep.*}` |
| **`p02`** | `p02_tsuboyama.py` (`tsuboyama`, `--split capped\|mega`) | Homology-Restricted MegaScale (`splits_oct06_capped`) and `mega_splits` $\Delta\Delta G$ ranking, stabilizing/destabilizing subsets, singles vs doubles, transfer vs overall, and backbone-feature ablation. | `01_megascale/{megascale_overview.*, megascale_per_library.*, megascale_singles_doubles.*, megascale_scatter.*, spearman_stab_destab*.*, ndcg96_singles_doubles*.*, doubles_vs_singles_on_singles.*, overall_vs_transfer_*.*, esm3_scoring_variants.*}` (mirrored in `01_megascale/mega_splits/`), `00_supplementary_figs/ndcg96_singles_doubles*.*`, `00_for_reviewers/baseline_nobb_ablation.*` |
| **`p03`** | `p03_proteingym.py` (`proteingym`) | Held-out ProteinGym (`173` canonical assays across 5 categories), $\sigma \in \{0, 0.5, 1\}$ category sweeps, multi-property trajectories, and zero-shot verification against published ProteinGym scores. | `03_proteingym/{pg_per_dms.csv, pg_all_models_per_dms.csv, pg_summary*.csv, pg_sigma_summary.csv, property_delta_pvalues.csv, pg_table_heatmap.*, pg_overview.*, pg_per_dms_scatter.*, pg_property_sigma_summary*.*, pg_multi_property_trajectories.*}`, `00_supplementary_figs/pg_ours_vs_published.*`, `99_misc/dms3/{per_assay_sigma_rho.csv, dms_multi_property_index.csv, dms_multi_property_trajectories.*}` |
| **`p04`** | `p04_mgnify_rocklin.py` (`mgnify_rocklin`) | Held-out MGnify stability (`152,680` scaffolds), Rocklin *de novo* (`742` designs), latent vs Tobit-linked $\Delta\Delta G$, homology strata, and mutation-count breakdowns. | `05_mgnify/{mgnify_summary.csv, mgnify_rmse.csv, mgnify_per_scaffold_rho.csv, mgnify_models_per_scaffold.csv, mgnify_models_pooled.csv, mgnify_by_mutation_count.*, mgnify_overview.*, rocklin_overview.*, mgnify_homology.*, mgnify_per_scaffold.*, mgnify_scatter.*, table_*.md, rocklin_designs/*, new_data_epistasis/*}` |
| **`p05`** | `p05_summary_main.py` (`summary_main`) | Main-text Figure 3 composite, cross-dataset thermostability summaries & paired p-values, and direct ESM-MSR vs `ESM3ΔG` comparison. | `00_main_text_figs/figure3.{png,pdf}`, `06_summary/{pg_categories.*, pg_categories_per_scaffold.*, thermostability_sets.*, thermostability_sets_pvalues.{csv,md}, thermostability_sets_best.csv, thermostability_sets_msr_heads.csv}`, `99_misc/esm_msr_vs_esm3dg/esm_msr_vs_esm3dg.*` |
| **`p06`** | `p06_epistasis_hierarchy.py` (`epistasis_hierarchy`, `--split`) | Four-level $\Delta\Delta\Delta G$ epistasis hierarchy (`naive_rho`, `beyond_rho`, `line`, `cell_mag`), epistatic vs additive double ranking, masking-scheme ablations, and $\lambda$-blend scan. | `00_main_text_figs/{fig2_fig3_stacked.*, epistasis_levels_e.*, epistasis_main_composite.*}`, `00_supplementary_figs/{fig5a_epistasis_levels_bars_test_{p0,masking}.*, fig5_epistasis_levels_*, masking_schemes*.*}`, `01_megascale/epistasis/{epistasis_levels_bars_{val,test}*.*, epistasis_naive_recomputed.*, epistasis_scatter_*, ddg_ranking*.*, lambda_scan.*, spearman_doubles_additive_vs_epistatic*.*, naive_epistasis_pos_neg_*.*, epistatic_vs_additive*.*}` (and `mega_splits/epistasis/`), `99_misc/dms3/epistasis_{naive_recomputed,scatter_dddG,lambda_scan}.*` |
| **`p07`** | `p07_epistasis_mechanisms.py` (`epistasis_mechanisms`, `--split`) | Thermodynamic floor-saturation mechanism, Tobit link binned curves, $20\times 20$ residue-pair epistasis heatmaps, $\Delta\Delta\Delta G$ scatters, and Rosetta multi-mutant vs additive singles. | `00_supplementary_figs/floor_saturation_mechanism.{png,pdf}`, `01_megascale/epistasis/{dG_additive_vs_dddG.*, ddG_additive_vs_dddG_*.*, alt_observed_additive_dG.*, esm_msr_*dddG_pred_vs_true*.*, epistasis_scatters.*, epistasis_types_by_additive_dG.*, link_binned.csv, dddG_scatters/*, dG_additive_vs_pred_dddG/*, pair_heatmaps/*}`, `99_misc/rosetta_multi_vs_additive/rosetta_multi_vs_additive.*` |
| **`p08`** | `p08_calibration_length.py` (`calibration_length`) | Single-donor OLS linear scale-and-bias $\Delta\Delta G$ calibration (RMSE/MAE) and protein sequence-length scaling analysis. | `02_thermostability/rmse/{calibration_candidates*.csv, rmse_by_dataset*.csv, calibration_transfer*.csv, rmse_recalibrated*.*}`, `08_sequence_length/{length_summary.csv, sequence_length_distribution.*, per_protein_metrics.csv, length_vs_performance.*, length_correlation_summary.*}`, `99_misc/dms3/{length_summary.csv, length_per_protein_metrics.csv}` |
| **`p09`** | `p09_screening.py` (`screening`, `--split`) | Virtual-screening hit rates, best $\Delta\Delta G$ vs $K$, biophysical property-gain rainclouds, $g(K) = a - b\ln K$ gain fits, Domainome & ProteinGym Activity screening, and cross-dataset overviews. | `00_supplementary_figs/screening_gain_law_composite.{png,pdf}`, `01_megascale/screening/{best_ddg_vs_k*.*, suggested_hit_rate*.*, top6_recovery_*.*, screening_per_protein_*.csv, property_gain/*}`, `01_megascale/gain_fit/gain_fit_*.*`, `02_thermostability/screening/*`, `03_proteingym/screening/*`, `04_domainome/screening/{domainome_screening_curves.*, screening_per_protein*.csv}`, `07_screening_overview/*` |
| **`p10`** | `p10_splits_and_leakage.py` (`splits_and_leakage`) | Four-part holdout filter breakdown, split best-partner homology distributions, cross-set identity/TM heatmaps, `mega_splits` clean-vs-homologous audit, and published reference comparisons. | `00_supplementary_figs/{holdout_filter.*, split_construction/{split_best_partner_capped.*, split_max_{identity,tm}_heatmap.*, mega_clean_vs_homologous_test.*}}`, `00_for_reviewers/split_best_partner_hyperopt.*`, `00_for_reviewers/reference/{published_reference.*, spurs_published_vs_retrained.*}`, `01_megascale/mega_splits/provenance/*` |
| **`p11`** | `p11_ablations_and_heads.py` (`ablations_and_heads`) | Architecture/loss ablations across all 11 `FINAL_*` families + cross-masking arms, and Dual-View (`comb`) vs WT-View (`wt`) regime law across held-out units. | `09_ablation/{ablation_architecture_suite.*, cells_long*.csv, coverage.csv, summary.csv, delta_paired.csv, paired_vs_p0*.csv, head_choice.csv, informativeness.csv, table_main_delta.{csv,md,tex}, table_wt_delta.{csv,md,tex}, table_composites.{csv,md}, table_counts.{csv,md}, layout_*.md}`, `99_misc/dual_view/{dual_view_summary.*, *.csv, table_all_families.md}`, `99_misc/dms3/ablation_*.csv` |
| **`p12`** | `p12_manifest.py` (`manifest`) | Optional `--reparse` of `docs/BENCHMARKS_OCT2026.md` reference tables, cell-by-cell validation against `reference/all_by_all_long.csv`, table-to-figure coverage verification, and output manifest. | `00_for_reviewers/reference/validation_vs_benchmarks.csv` (records both 436-domain held-out `rho` and 522-domain legacy `rho_unfiltered`), `table_figure_map.csv`, `manifest_outputs.csv` |

### 2. Mechanistic Studies (`analysis/studies/`)

| Study Target | Directory | Scope | Key Outputs (`analysis/results/99_misc/`) |
| :--- | :--- | :--- | :--- |
| `study:faure_epistasis` | [`studies/faure_epistasis/`](studies/faure_epistasis/README.md) | Combinatorial $3\times 3+$ block geometry, assay floor saturation, and DiMSum vs cDNA proteolysis SNR ceilings across Faure et al. PCA (`DLG4`, `GRB2`) and MegaScale. | `99_misc/faure_epistasis/{faure_vs_megascale_epistasis_snr.*, proximity_distribution.*, epistasis_snr.*, cell_mag_ceiling.*, *.csv}` |
| `study:sigma_mechanisms` | [`studies/sigma_mechanisms/`](studies/sigma_mechanisms/README.md) | 3-seed `FINAL_p0` mechanistic decomposition of LoRA scaling ($\sigma \in \{0, 0.5, 1\}$) across the 173 canonical held-out ProteinGym assays. | `99_misc/sigma_mechanisms/{sigma_mechanisms_composite.*, aa_category_heatmaps.*, site_vs_sub_decomposition.csv, aa_rank_shifts.csv, tail_selection_vs_filtering.csv, pareto_multi_property.csv}` |

### 3. Offline Audits & Scoring Utilities (`analysis/audits/`, `analysis/scoring/`)

| Package / Tool | Purpose |
| :--- | :--- |
| [`audits/homology_and_splits/`](audits/homology_and_splits/README.md) | `active_builders/` (`tsuboyama_rule_flags.py`, `set_overlap_matrix.py`, `build_mega_tasks.py`, `mega_split_provenance.py`, `export_reference.py`) and `oct2026_leak_audit/` (`01`..`20` + `run_all.py`). |
| [`audits/baseline_bridge/`](audits/baseline_bridge/README.md) | Audits local and cluster baseline training/evaluation directories via `msr_analysis.paths.TRAINING_LOGS`. |
| `scoring/collect_predictions.py` | Mirrors `msr_predictions/`, `final_*`, `baselines/`, `mega_splits/`, and `resources_tracked/` from cluster runs with atomic `MANIFEST.tsv` verification. |
| `scoring/score_runs.py` (+ `score_baseline_runs.py`, `score_me_proteingym.py` wrappers) | Unified baseline & ProteinGym run scorer importing `esm_msr.stats` and `msr_analysis.heldout` directly (`ddof=1`). |
| `scoring/resources.py` (+ `resource_report.py`, `resource_summary.py` wrappers) | Summarizes fixed-slot RTX A4500 resource-tracker scopes into `results/10_inference_resources/` and updates `docs/INFERENCE_RESOURCES.md` (`--docs`). |
