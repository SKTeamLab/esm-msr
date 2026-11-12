# `analysis/`: Benchmark Scoring, Figures, and Leak Audit

Benchmark scoring, figure generation, and homology leak auditing for the repository. All figures and CSV tables in `results/` are computed by the top-level numbered scripts from `msr_predictions/` and `baselines/{retrained,public}/`.

Run all scripts (CPU only, using `msr_venv`):
```bash
./run_all.sh
```

---

## Subfolders

| Subfolder | Contents |
|---|---|
| `msr_analysis/` | Shared Python package: `registry.py` (datasets, model metadata, colors), `data.py` (spine & prediction loaders, `sum_singles`, generalized epistasis computation, frame cache), `boot.py` (exact multiplicity Spearman bootstrap: protein, site, mutation), `calib.py` (one-donor OLS recalibration for RMSE), `link.py` (Tobit softplus link & post-link epistasis using `y_floor.json`), `screening.py` (hit-rate, top-$K$ recovery, best-$\Delta\Delta G$ curves), `style.py` (matplotlib formatting), `tables.py` (CSV summary exporters), `common.py` (shared helpers). |
| `scoring/` | Prediction collection and benchmark table assembly (`collect_predictions.py`, `benchmark_matrix.py`, `score_baseline_runs.py`, `score_me_proteingym.py`, `pgym_public_scores.py`, `pgym_rosetta_singles.py`, `resource_report.py`). |
| `leak_audit/` | Sequence (MMseqs2) and structure (Foldseek TM-align) homology and training-data overlap audit (`run_all.py`, steps `00_*.py`–`19_*.py`; see `leak_audit/README.md`). |
| `reference/` | Committed reference inputs: `benchmark_tables/*.csv` (parsed from `docs/BENCHMARKS_OCT2026.md`), `all_by_all_long.csv`, `y_floor.json` (per-library Tobit assay floors), `ctx_wt_rev_devel.csv.gz` (mutant-context WT-adapter reverse pass $\widetilde{\text{WT}}$), `leak_*.csv`, and `spurs_public_checkpoint.csv`. |
| `results/` | Generated figures (`.png` + `.pdf`) and backing tables (`.csv`), organized into per-script subdirectories (`00_core/` through `12_pair_heatmaps/`). |
| `cache/` | Pickled `Frame` objects (`frame_<dataset>.pkl`) and bootstrap replicate arrays (`boot_<unit>.pkl`); git-ignored and rebuilt automatically. |
| `notebooks/` | Archived exploratory notebooks (`*.ipynb`, `analysis_utils_msr.py`); not read by the analysis pipeline. |
| `tests/` | Unit tests verifying `msr_analysis.boot` against `scipy.stats.spearmanr`. |

---

## Python Scripts: Figures and Tables Created

| Script | Output Subfolder | Figures Created (`.png` / `.pdf`) | Tables Created (`.csv`) |
|---|---|---|---|
| `00_parse_benchmark_tables.py` | `reference/benchmark_tables/` | — | `<section>.<k>.csv`, `tables_index.csv` (all markdown tables extracted from `docs/BENCHMARKS_OCT2026.md`). |
| `00_validate_against_tables.py` | `results/00_core/` | — | `validation_vs_benchmarks.csv` (recomputes all 509 documented benchmark cells and verifies agreement to within rounding). |
| `01_core_rho.py` | `results/00_core/` & `cache/` | — | `rho_bootstrap.csv` (Spearman $\rho$ and 95% CIs across all datasets and models under protein, site, and mutation resampling; caches replicate matrices in `cache/boot_*.pkl`). |
| `02_thermostability.py` | `results/02_thermostability/` | `thermo_all_models`, `thermo_vs_retrains`, `thermo_interval_types`, `thermo_scatter_external` | `thermo_all_models.csv`, `thermo_vs_retrains.csv`, `thermo_vs_retrains_paired_diff.csv`, `thermo_interval_types.csv`, `thermo_scatter_external_stats.csv` |
| `03_tsuboyama.py` | `results/03_tsuboyama/` | `tsuboyama_overview`, `tsuboyama_per_library`, `tsuboyama_singles_doubles`, `tsuboyama_scatter` | `tsuboyama_overview.csv`, `tsuboyama_per_library.csv`, `tsuboyama_singles_doubles.csv`, `tsuboyama_scatter_stats.csv` |
| `04_dms_domainome.py` | `results/04_dms_domainome/` | `dms_sigma_vs_zero_shot`, `dms_sigma_sweep`, `dms_vs_retrains` | `dms_sigma_vs_zero_shot.csv`, `dms_sigma_sweep.csv`, `dms_vs_retrains.csv`, `dms_vs_retrains_paired_diff.csv` |
| `05_proteingym.py` | `results/05_proteingym/` | `pg_categories`, `pg_per_dms_scatter`, `pg_table_heatmap` | `pg_summary.csv` (leak-filtered 190/213), `pg_summary_all213.csv`, `pg_per_dms.csv`, `pg_sigma_paired.csv`, `pg_sigma_paired_all213.csv` |
| `06_epistasis.py` | `results/06_epistasis/` | `epistasis_authoritative_metrics`, `epistasis_hierarchy_val`, `epistasis_hierarchy_test`, `epistasis_naive_recomputed`, `epistasis_scatter_dddG`, `epistasis_scatter_ddG_doubles`, `ddg_ranking`, `lambda_scan`, `nobb_ablation` | `epistasis_naive_recomputed.csv`, `epistasis_rmse.csv`, `epistasis_scatter_dddG_stats.csv`, `doubles_pooled_vs_per_library.csv`, `ddg_ranking.csv`, `lambda_scan.csv` |
| `07_rmse_calibration.py` | `results/07_rmse/` | `rmse_recalibrated` (S461 donor), `rmse_recalibrated_ssym` (Ssym donor) | `rmse_by_dataset.csv`, `rmse_by_dataset_ssym.csv`, `calibration_candidates.csv`, `calibration_transfer.csv`, `calibration_transfer_ssym.csv` |
| `08_screening.py` | `results/08_screening/` | `screening_external_top10pct`, `screening_per_protein_curves_singles_only`, `screening_per_protein_curves_combined`, `screening_per_protein_curves` | `screening_external_top10pct.csv`, `screening_per_protein_curves_singles_only.csv`, `screening_per_protein_curves_combined.csv`, `screening_per_protein_curves.csv` |
| `09_sequence_length.py` | `results/09_sequence_length/` | `sequence_length_distribution`, `length_vs_performance`, `length_correlation_summary` | `length_summary.csv`, `length_vs_performance_per_protein.csv`, `length_vs_performance_stats.csv` |
| `10_leakage_and_reference.py` | `results/10_leakage_and_reference/` | `leak_matrix_external`, `leak_matrix_other`, `leak_overlap_tsuboyama_test`, `clean_means`, `spurs_published_vs_retrained`, `published_reference`, `leak_audit_text_tables` | `clean_means.csv`, `spurs_published_vs_retrained.csv` |
| `11_publication_figures.py` | `results/03_tsuboyama/` & `results/06_epistasis/` | `fig1_ndcg96_singles_doubles`, `fig2_spearman_stab_destab[_pooled][_mutboot]`, `fig2b_spearman_stab_destab_singles_doubles[_pooled][_mutboot]` (`03_tsuboyama/`); `fig3_spearman_doubles_additive_vs_epistatic`, `fig4_naive_epistasis_pos_neg_{grouped,pooled}` (`06_epistasis/`) | Accompanying `<fig>.csv`, `<fig>_pvalues.csv`, `fig3_..._paired_difference.csv`, `best_other_ddg_model.csv` |
| `11b_publication_epistasis_levels.py` | `results/06_epistasis/` | `fig5a_epistasis_levels_bars_{test,val}[_cell_mag|_cell_compare]` (5-level epistasis hierarchy bar charts) | Reads `fig5_epistasis_levels_bootstrap_{test,val}.csv` from `11d`. |
| `11c_publication_epistasis_scatters.py` | `results/06_epistasis/` | `fig6a_dG_additive_vs_dddG`, `fig6b_esm_msr{,_baseline,_p0,_models}_dddG_pred_vs_true`, `fig6c_..._dG_additive_lt0`, `fig6d_ddG_additive_vs_dddG_dG_additive_gt0`, `fig6e_ddG_additive_vs_dddG_by_singles`, `fig6f_option_{A,B,C}_*` | `fig6_epistasis_scatters.csv`, `fig6de_ddG_additive_kinds.csv` |
| `11d_epistasis_levels_bootstrap.py` | `results/06_epistasis/` | — | `fig5_epistasis_levels_bootstrap_{test,val}.csv` (domain-bootstrapped `esm_msr.epi_metrics` hierarchy for all models). |
| `11e_esm3_scoring_variants.py` | `results/03_tsuboyama/` & `results/06_epistasis/` | `fig2_esm3_scoring_variants` (`03_tsuboyama/`), `fig3_esm3_scoring_variants` (`06_epistasis/`) | `fig2_esm3_scoring_variants.csv`, `fig3_esm3_scoring_variants.csv` |
| `11f_epistasis_types.py` | `results/06_epistasis/` | `fig7_epistasis_types_by_additive_dG` | `fig7_epistasis_types_by_additive_dG.csv` |
| `11g_link_figure.py` | `results/06_epistasis/` | `fig8_link_option_{A,B,C,D,E}_*` | `fig8_link_binned.csv` |
| `12_pair_epistasis_heatmaps.py` | `results/12_pair_heatmaps/` | `<raw\|saturation_corrected>/<domain>/<wt><pos>_<wt><pos>.png` (20×20 residue pair heatmaps across 15 complete test pairs) | `pairs.csv`, `agreement.csv` |
| `13_dddG_correlation_scatters.py` | `results/06_epistasis/` | `dddG_scatters/<all_dG_additive\|dG_additive_positive\|dG_additive_gt1>/<model>.png`, `dG_additive_vs_pred_dddG/<model>.png`, `dG_additive_vs_pred_dddG/esm_msr_models_dG_additive_vs_pred_dddG`, `dddG_scatters/latent_vs_linked_gt1.png` | `dddG_scatters/stats.csv`, `dddG_scatters/latent_vs_observed.csv`, `dddG_scatters/rosetta_unit_check.csv`, `dddG_scatters/bias_*.csv`, `dG_additive_vs_pred_dddG/stats.csv` |
| `14_property_category_gain.py` | `results/08_screening/property_gain/` | `gain_aa_stacked_k{24,96}`, `gain_{asa_hydrophobicity,volume,conservation}_k{24,96}[_boxes][_top_only][_allpicks]`, `gain_rule_comparison` | `<fig>.csv` alongside each figure |
| `15_tsuboyama_single_screening.py` | `results/08_screening/` | `fig_suggested_hit_rate_with_counts[_nocys]`, `fig_suggested_hit_rate[_pooled][_nocys]`, `fig_top6_recovery_k6_k24_k96[_atleast1|_both][_nocys]`, `fig_best_ddg_vs_k[_panels][_nocys]` | Accompanying `.csv` and `_pvalues.csv` for each figure |
| `99_table_coverage.py` | `results/` | — | `table_figure_map.csv` (verifies that every result table in `BENCHMARKS_OCT2026.md` is plotted). |

---

## Generalized Epistasis Definition

Earlier scripts wrote ESM-MSR's predicted epistasis as $\text{MT}_{AB} - \text{WT}_{AB}$ (or $\text{comb}_{AB} - \text{WT}_{AB}$). That shorthand has two issues:
1. **Masking generality**: $\text{WT}_A + \text{WT}_B = \text{WT}_{AB}$ holds **only** when the wild-type pass is unmasked so single-site log-odds sum exactly on the double row. Under independent single-position masking (where only position $A$ or position $B$ is masked at a time on the wild-type sequence), $\text{WT}_{AB}$ is either undefined or not equal to $\text{WT}_A + \text{WT}_B$. Every additive baseline and epistasis calculation in `analysis/` therefore looks up the component single-mutant rows explicitly via `msr_analysis.data.sum_singles`.
2. **2× magnitude**: Because the deployed ESM-MSR predictor on a double is the 50/50 ensemble $\text{comb}_{AB} = \frac{1}{2}(\text{MT}_{AB} + \text{WT}_{AB})$, the raw head difference $\text{MT}_{AB} - \text{WT}_{AB} = 2(\text{comb}_{AB} - \text{WT}_{AB})$ has twice the magnitude of the model's actual deviation from WT-single additivity.

All scripts and cached frames (`msr_analysis/data.py`) compute and expose the consistent 1× generalization for every MSR/ESM3 model family (`msr`, `msrdev`, `msrp0`, `msrrel`, `msrrel@0`, `esm3avg`):
$$\Delta\Delta\Delta G_{\text{pred}} = \frac{1}{2}\big(\text{MT}_{AB} + \text{WT}_{AB}\big) - \text{WT}_A - \text{WT}_B = \text{comb}_{AB} - \text{WT}_A - \text{WT}_B$$
stored as `:epi`, with corresponding additive expectation `:wt+` $= \text{WT}_A + \text{WT}_B$. (For the mutant-context control `ctx ctrl`, $\text{MT}_{AB}$ is replaced by the reverse pass $\widetilde{\text{WT}}_{AB}$: $\frac{1}{2}(\widetilde{\text{WT}}_{AB} + \text{WT}_{AB}) - \text{WT}_A - \text{WT}_B$.)

---

## Resolved Discrepancies and Notes

1. **All 509 benchmark validation cells match (`00_validate_against_tables.py`)**:
   - Every cell in `reference/all_by_all_long.csv` reproduces to within 3-decimal rounding (`median |diff| = 0.00000`, `max |diff| = 0.0005`, `0` cells $> 0.005$).
   - Parent-dataset scoring rules from `BENCHMARKS_OCT2026.md` §1.1 are enforced in `msr_analysis/data.py`: ESM3 Forge models and MutateEverything score `S461`, `K2369`, and `PTMUL-D` by slicing predictions from their supersets (`S669`, `K3822`, `PTMUL`), and MutateEverything stage 2's additive column on `GB1` sums singles from ME's complete 149,360-row `GB1` table (which includes the 3 singles absent from the harness label file).
2. **Why `BENCHMARKS_OCT2026.md` Tables 2.0–2.2 had 2× `global_err` and unlinked residual metrics for `devel`**:
   - In `src/esm_msr/epi_metrics.py`, `Table.__init__` applied the Tobit link $P$ **only** to `naive_rho` (`0.643` val / `0.464` test when `lat['epi'] = MT_AB` is passed through the link with `reference/y_floor.json`), whereas all other columns (`beyond_rho`, `pair_rho`, `line_rho`, `cell_mag`, `cell_rank`, `global_err`) called `predicted_dddg` on the **unlinked latent `epi` head** ($\text{MT}_{AB} - \text{WT}_A - \text{WT}_B$, 2× magnitude).
   - That 2× magnitude explains why Tables 2.0–2.2 reported `global_err = 0.717 / 0.954` (`baseline` val/test) and `0.824 / 1.079` (`devel` val/test), whereas the 1× definition $\frac{1}{2}(\text{MT}_{AB}+\text{WT}_{AB}) - \text{WT}_A - \text{WT}_B$ gives `RMSE = 0.733 / 0.797` (`baseline` val/test) and `0.687 / 0.784` (`devel` val/test).
   - Additionally, `msr_predictions/` had inconsistent `epi_pred` CSV columns across checkpoints (`baseline` wrote 2× `MT - WT`, `msrp0` wrote 1× `comb - WT`, `msrdev` omitted `epi_pred`). `msr_analysis/data.py` now ignores the raw `epi_pred` CSV column and computes `:epi` directly from `comb`, `wt`, and `sum_singles`.
3. **MutateEverything stage 2 epistasis hierarchy (`11d` vs Table 2.2)**:
   - On Tsuboyama test, `11d_epistasis_levels_bootstrap.py` gives `(naive, beyond, pair, line, cell_rank) = (0.409, 0.227, 0.132, 0.229, 0.110)` vs Table 2.2's `(0.405, 0.223, 0.124, 0.225, 0.112)`.
   - Cause: `baselines/scoring/score_baselines.py` took ME stage 1's single-mutant predictions from `tsuboyama_capped_testcomp.csv` (the 35,484-row singles-only pass, which reproduces `(0.405, 0.223, 0.124, 0.225, 0.110)` to 3 decimals), whereas `msr_analysis/data.py` reads `me1` from `tsuboyama_capped_test.csv` (the 48,740-row combined pass; ME has ~0.99 run-to-run correlation between separate forward passes).
4. **Assay floor file (`reference/y_floor.json`) and recomputability**:
   - `reference/y_floor.json` is committed in `analysis/reference/y_floor.json` and is used by default in `msr_analysis/link.py`, `06_epistasis.py`, `11d_epistasis_levels_bootstrap.py`, and `12_pair_epistasis_heatmaps.py`.
   - Both the epistasis hierarchy (`11d` / `11b`) and the homology leak audit (`leak_audit/run_all.py`) are fully recomputable in `analysis/`. Only the mutant-background ablation (`2.5.1.csv`), the earlier hyperopt published-paper reference (`1.5.1.csv`), the third-party models' epistasis hierarchy rows in `06_epistasis.py` (`2.1.1.csv`, `2.2.1.csv`), and frozen public SPURS numbers (`spurs_public_checkpoint.csv`) are read from `reference/`.
5. **Unit & screening conventions**:
   - `S571` labels are not in kcal/mol ($\text{ SD } = 8.9$) and are excluded from RMSE and screening figures (rank correlation $\rho$ is unaffected).
   - Rosetta $\Delta\Delta G$ predictions are scaled by $1 / 2.94$ (REU to kcal/mol) in `msr_analysis/data.py`.
   - Domainome screening (`08_screening.py`) defines a hit as $\Delta\text{fitness} > 0$ (abundance > wild type) and filters out the 77 Domainome domains with sequence identity $> 25\%$ (`fident > 0.25`) or Foldseek `alntmscore >= 0.5` (`qtmscore >= 0.5` and `ttmscore >= 0.5`) to the 195 Tsuboyama training + validation domains (leaving 445 non-redundant domains).
