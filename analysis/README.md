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

# Run unit tests (485 passing tests across tests/ and analysis/tests/)
pytest
```

### CLI Options (`python -m analysis.run`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--list` | `False` | Print all registered `p01`–`p12` stages and `study:*` targets in execution order and exit. |
| `--only STAGE [...]` | all `p01`–`p12` | Comma- or space-separated stage codes (`p01`), stage names (`tsuboyama`), or study targets (`study:sigma_mechanisms`). |
| `--skip STAGE [...]` | none | Comma- or space-separated stages to skip. |
| `--studies` | `False` | Also execute `studies/faure_epistasis` and `studies/sigma_mechanisms` after the selected pipelines. |
| `--split {capped,mega}` | `capped` | MegaScale split passed explicitly to split-aware pipelines (`p02`, `p06`, `p07`, `p09`); `--split mega` writes to `results/03_tsuboyama/mega_splits/` without mutating environment variables. |
| `--B INT` | `1000` | Number of bootstrap replicates forwarded to every stage. |
| `--fast` | `False` | Smoke-test preset (`--B 100`). |

Every individual stage or study module also runs directly via `python -m analysis.pipelines.<module> [--B 1000] [--fast]`.

---

## Directory Layout

```text
analysis/
├── README.md                     # Architecture, CLI reference, module guide, and legacy-to-pipeline map
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
├── cache/                        # Git-ignored Frame & bootstrap caches (written atomically via temp + os.replace)
├── results/                      # Generated figures (.png + .pdf), CSVs, Markdown/LaTeX tables & manifests
├── docs/                         # Audit & architecture documentation
│   ├── ARCHITECTURE_AND_FINDINGS.md
│   ├── SEED_AUDIT.md
│   └── HOMOLOGY_FILTER_AUDIT.md
└── tests/                        # Unit tests for bootstrap, roster, frames, holdout, links, epistasis & screening
```

---

## Core Library Guide (`analysis/msr_analysis/`)

| Module | Responsibility & Key Exports |
| :--- | :--- |
| `paths.py` | Repo-relative (`REPO_ROOT`, `ANALYSIS_ROOT`, `DATA_DIR`, `CACHE`, `REFERENCE`, `RESULTS`) and external `MSR_ROOT` paths (`MSR`, `FINAL_BENCH`, `FINAL_CTX`, `FINAL_PGYM_*`, `FINAL_MGNIFY*`, `BASE`, `TASKS`, `PGYM_INPUTS`, `TSUBOYAMA_PDB_DIR`, `TSUBOYAMA_FEATS_CSV`, `SALVAGED_HOMOLOGY_CSV`), canonical dataset-to-folder maps, `results_dir(name, split)`, `source_signature()`, and symlink-safe atomic writers (`atomic_pickle_dump`, `atomic_to_csv`, `atomic_write_text`). |
| `models.py` | Single immutable visual & key roster: `Model`, `StyleSpec` (frozen mapping with `.with_overrides()`), `ROSTER`, `BY_NAME`, `ORDER`, `UNIFIED`, `ABLATION_STYLES` (all 11 `FINAL_*` families + 2 cross-mask arms), `CATEGORY_COLOR`, `VARIANT_COLOR`, `column()`. |
| `registry.py` | Dataset cohorts (`EXT12`, `DMS3`, `DMS_AUX`, `TSU_CAPPED`, `TSU_MEGA`, `PER_PROTEIN`, `LABEL`), `tsu_datasets(split)`, `all_datasets(split)`, and read-only `MODELS` (`MappingProxyType`) registering every `.single`, `.multi`, `.add`, `.epi`, and display-name style. |
| `heldout.py` | Canonical four-part sequence + structure holdout filter backed by `reference/tsuboyama_rule_flags.csv`: `flags()`, `excluded(dataset)` (`proteingym=41`, `domainome=86`, `mgnify=27,687`, `rocklin_denovo=258`), and `keep_ids(dataset)`. |
| `data.py` | `Frame(name, spine, preds, notes)`, `build_spine()`, `load_frame(ds, refresh=False, heldout=True, check_sources=False)`, `sum_singles()`, `singles_of_doubles()`, `synthesize_missing_epi()` (derives `:epi` on the fly from `pred - pred+` across cached pickles), and `add_esm3_masked()`. |
| `final_runs.py` | Loader for all 11 `FINAL_*` ablation families (`FINAL_p0` reference, `p1`, `p1_marg`, `pw0`, `norank`, `rcthr0`, `norange`, `mse`, `noseqhwm`, `r2a4`, `mega`) plus cross-masking arms (`FINAL_p0@independent`, `FINAL_p1@unmasked`): `load_final_frame()`, `pgym_per_dms()`, `pgym_per_dms_seeds()`. |
| `mgnify.py` | Loader for the 766,790-variant MGnify stability benchmark and 1,000 Rocklin *de novo* designs (`load_mgnify`), joining v2 `mgnify_mapped.csv` labels, applying 4-part holdout masking (`heldout=True`), and computing both `latent` and post-hoc Tobit-linked (`<head>~`) predictions. |
| `proteingym.py` | Authoritative ProteinGym loader & scorer over the **173 canonical held-out assays** (`PG_DROP` + `heldout.keep_ids`: `Activity=38, Binding=13, Expression=18, OrganismalFitness=67, Stability=37`): `pgym_reference()`, `load_final_pgym_seeds()`, `load_sigma_pgym()`, `load_all_models_pgym()`, `boot_pgym_categories()` (joint `[category-stratified assay × 3-seed]` bootstrap), `sigma_delta_pvalues()`, and `draw_property_sigma_panel()`. |
| `external.py` | Unified external benchmark helpers: `model_key_for_dataset()`, `seed_frame()`, `load_mutation_boot()` (`cache/boot_ext_mutation.pkl` with triplicate seed resampling), and `fresh_frames()`. |
| `common.py` | Dataset/frame utilities: `has_multi()`, `primary_key()`, `out_dir(name, split)`, `load_rho_df()`, `primary_ci()`, `epi_rows()` (8,494 val / 12,455 test doubles with both constituent singles measured), `subframe()`, and `leak_table()`. |
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

## Pipeline Stages, Studies, Audits & Legacy Mapping

### 1. Active Pipelines (`analysis/pipelines/`)

| Stage | Module | Legacy Scripts & Sub-Studies Replaced | Key `analysis/results/` Outputs |
| :--- | :--- | :--- | :--- |
| **`p01`** | `p01_benchmarks_external.py` (`benchmarks_external`) | `01_core_rho.py`, `02_thermostability.py`, `04_dms_domainome.py` | `00_core/rho_bootstrap.csv`, `02_thermostability/{thermostability_rho.csv, thermo_external_rho.*, thermo_overview.*, thermo_interval_types.*}`, `04_dms_domainome/{dms_vs_retrains.*, dms_sigma_vs_zero_shot.*, dms_sigma_sweep.*}` |
| **`p02`** | `p02_tsuboyama.py` (`tsuboyama`, `--split capped\|mega`) | `03_tsuboyama.py`, `11_publication_figures.py` (figs 1–2), `11e` (fig 2), `11i_doubles_vs_singles_pvalues.py`, `11j_overall_vs_transfer.py`, `11k_ndcg96_supplement.py`, `tsuboyama/mega_splits/{01_rho_overview, 03_version_comparison}.py` | `03_tsuboyama/{tsuboyama_overview.*, tsuboyama_per_library.*, tsuboyama_singles_doubles.*, tsuboyama_scatter.*, fig1_*, fig2_*, fig2b_*, doubles_vs_singles_on_singles.*, overall_vs_transfer_*.*}`, `00_supplementary_figs/ndcg96_singles_doubles.*`, `03_tsuboyama/mega_splits/03_tsuboyama/*` |
| **`p03`** | `p03_proteingym.py` (`proteingym`) | `05_proteingym.py`, `23_pg_ours_vs_published.py`, `proteingym/p0_sigma/{01..05}` | `05_proteingym/{pg_per_dms.csv, pg_all_models_per_dms.csv, pg_summary.csv, pg_sigma_summary.csv, property_delta_pvalues.csv, pg_table_heatmap.*, pg_overview.*, pg_per_dms_scatter.*, pg_property_sigma_summary.*, pg_multi_property_trajectories.*}`, `23_pg_ours_vs_published/*` |
| **`p04`** | `p04_mgnify_rocklin.py` (`mgnify_rocklin`) | `18_mgnify.py`, `18b_mgnify_models.py`, `18c_mgnify_by_mutation_count.py`, `scoring/mgnify_{rdenovo_stats, epistasis_beyond, by_homology, baselines_compare, labels, link}.py` | `18_mgnify/{mgnify_summary.csv, mgnify_rmse.csv, mgnify_per_scaffold_rho.csv, mgnify_models_per_scaffold.csv, mgnify_models_pooled.csv, mgnify_by_mutation_count.*, mgnify_overview.*, rocklin_overview.*, mgnify_homology.*, mgnify_per_scaffold.*, mgnify_scatter.*, rocklin_designs/*, new_data_epistasis/*}` |
| **`p05`** | `p05_summary_main.py` (`summary_main`) | `01_summary.py`, `01b_thermostability_pvalues.py`, `22_esm_msr_vs_esm3dg.py` | `00_main_text_figs/figure3.{png,pdf}`, `01_summary/{pg_categories.*, pg_categories_per_scaffold.*, thermostability_sets.*, thermostability_sets_pvalues.{csv,md}, thermostability_sets_best.csv, thermostability_sets_msr_heads.csv}`, `22_esm_msr_vs_esm3dg/esm_msr_vs_esm3dg.*` |
| **`p06`** | `p06_epistasis_hierarchy.py` (`epistasis_hierarchy`, `--split`) | `06_epistasis.py`, `11_publication_figures.py` (figs 3–4), `11e` (fig 3), `11b_publication_epistasis_levels.py`, `11d_epistasis_levels_bootstrap.py`, `11h_stacked_stab_epistatic.py`, `11l_epistatic_vs_additive_pvalues.py`, `11m_masking_schemes.py`, `11n_epistasis_levels_p0.py`, `scoring/spurs_epi_table_cells.py` | `00_main_text_figs/{fig2_fig3_stacked.*, epistasis_levels_e.*, epistasis_main_composite.*}`, `06_epistasis/{epistasis_naive_recomputed.*, epistasis_scatter_*, ddg_ranking.*, lambda_scan.*, fig3_*, fig4_*, fig5_*, fig5a_*, epistatic_vs_additive*.*}`, `00_supplementary_figs/{fig5a_epistasis_levels_bars_test_{p0,masking}.*, masking_schemes.*}` |
| **`p07`** | `p07_epistasis_mechanisms.py` (`epistasis_mechanisms`, `--split`) | `11c_publication_epistasis_scatters.py`, `11f_epistasis_types.py`, `11g_link_figure.py`, `12_pair_epistasis_heatmaps.py`, `13_dddG_correlation_scatters.py`, `19_rosetta_multi_vs_additive.py` | `00_supplementary_figs/floor_saturation_mechanism.{png,pdf}`, `06_epistasis/{fig6a..fig6e.*, fig7_*, fig8_link_binned.csv, dddG_scatters/*, dG_additive_vs_pred_dddG/*}`, `12_pair_heatmaps/{pairs.csv, agreement.csv}`, `19_rosetta_multi_vs_additive/rosetta_multi_vs_additive.*` |
| **`p08`** | `p08_calibration_length.py` (`calibration_length`) | `07_rmse_calibration.py`, `09_sequence_length.py` | `07_rmse/{calibration_candidates*.csv, rmse_by_dataset*.csv, calibration_transfer*.csv, rmse_recalibrated*.*}`, `09_sequence_length/{length_summary.csv, sequence_length_distribution.*, per_protein_metrics.csv, length_vs_performance.*, length_correlation_summary.*}` |
| **`p09`** | `p09_screening.py` (`screening`, `--split`) | `08_screening.py`, `14_property_category_gain.py`, `15_tsuboyama_single_screening.py`, `16_screening_overview.py`, `17_gain_per_variant_fit.py`, `pg_activity/{01_wt_reference, 02_activity_screening}.py` | `00_supplementary_figs/screening_gain_law_composite.{png,pdf}`, `08_screening/{screening_pooled_external.*, screening_per_protein_*.csv, fig_*, property_gain/*, pg_activity/*}`, `16_screening_overview/*`, `17_gain_fit/*` |
| **`p10`** | `p10_splits_and_leakage.py` (`splits_and_leakage`) | `10_leakage_and_reference.py`, `20_split_best_partner.py`, `20b_split_overlap_heatmaps.py`, `21b_holdout_filter_figure.py`, `tsuboyama/mega_splits/02_split_provenance.py` | `10_leakage_and_reference/*`, `00_supplementary_figs/holdout_filter.*`, `00_supplementary_figs/split_construction/{split_best_partner_capped.*, split_max_{identity,tm}_heatmap.*, mega_clean_vs_homologous_test.*}`, `for_reviewers/{split_best_partner_hyperopt.*, baseline_nobb_ablation.*}` |
| **`p11`** | `p11_ablations_and_heads.py` (`ablations_and_heads`) | `98_final_ablation_table.py`, `ablation_vs_p0/{01..06}`, `combined_vs_wt/{01..08}` | `98_ablation/{cells_long.csv, summary.csv, delta_paired.csv, table_main_delta.{csv,md,tex}, table_wt_delta.{csv,md,tex}, head_choice.csv, informativeness.csv}`, `studies/ablations/{ablation_architecture_suite.*, ...}`, `studies/dual_view/{dual_view_summary.*, ...}` |
| **`p12`** | `p12_manifest.py` (`manifest`) | `00_parse_benchmark_tables.py` (`--reparse`), `00_validate_against_tables.py`, `99_table_coverage.py` | `00_core/validation_vs_benchmarks.csv` (records both 436-domain held-out `rho` and 522-domain legacy `rho_unfiltered`), `table_figure_map.csv` (verifies 27/27 benchmark tables mapped), `manifest_outputs.csv` |

### 2. Mechanistic Studies (`analysis/studies/`)

| Study Target | Directory | Legacy Sub-Study Replaced | Key Outputs (`analysis/results/studies/`) |
| :--- | :--- | :--- | :--- |
| `study:faure_epistasis` | [`studies/faure_epistasis/`](studies/faure_epistasis/README.md) | `analysis/faure_epistasis/{01..05, faure.py}` | `studies/faure_epistasis/{faure_vs_megascale_epistasis_snr.*, proximity_distribution.*, epistasis_snr.*, cell_mag_ceiling.*, *.csv}` |
| `study:sigma_mechanisms` | [`studies/sigma_mechanisms/`](studies/sigma_mechanisms/README.md) | `analysis/proteingym/sigma_tradeoff/` (upgraded from Oct-27 1-seed baseline to 3-seed `FINAL_p0` on 173 held-out assays) | `studies/sigma_mechanisms/{sigma_mechanisms_composite.*, aa_category_heatmaps.*, site_vs_sub_decomposition.csv, aa_rank_shifts.csv, tail_selection_vs_filtering.csv, pareto_multi_property.csv}` |

### 3. Offline Audits & Scoring Utilities (`analysis/audits/`, `analysis/scoring/`)

| Package / Tool | Legacy Counterparts | Purpose |
| :--- | :--- | :--- |
| [`audits/homology_and_splits/`](audits/homology_and_splits/README.md) | `leak_audit/{01..22, lib.py, run_all.py, data/}`, `tsuboyama/mega_splits/{00_build_tasks, 02_split_provenance}.py` | `active_builders/` (`tsuboyama_rule_flags.py`, `set_overlap_matrix.py`, `build_mega_tasks.py`, `mega_split_provenance.py`, `export_reference.py`) and `oct2026_leak_audit/` (`01`..`20` + `run_all.py`). |
| [`audits/baseline_bridge/`](audits/baseline_bridge/README.md) | `analysis/baseline_bridge/inventory.py` | Audits local and cluster baseline training/evaluation directories via `msr_analysis.paths.TRAINING_LOGS`. |
| `scoring/collect_predictions.py` | `scoring/collect_predictions.py` | Mirrors `msr_predictions/`, `final_*`, `baselines/`, `mega_splits/`, and `resources_tracked/` from cluster runs with atomic `MANIFEST.tsv` verification. |
| `scoring/score_runs.py` (+ `score_baseline_runs.py`, `score_me_proteingym.py` wrappers) | `score_baseline_runs.py`, `score_me_proteingym.py`, `summarize_spurs_runs.py`, `pgym_public_scores.py`, `pgym_rosetta_singles.py` | Unified baseline & ProteinGym run scorer importing `esm_msr.stats` and `msr_analysis.heldout` directly (`ddof=1`). |
| `scoring/resources.py` (+ `resource_report.py`, `resource_summary.py` wrappers) | `resource_report.py`, `resource_summary.py` | Summarizes fixed-slot RTX A4500 resource-tracker scopes into `results/inference_resources/` and updates `docs/INFERENCE_RESOURCES.md` (`--docs`). |
