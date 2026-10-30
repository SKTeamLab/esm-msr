# analysis/: the October 2026 benchmark, re-done from the per-variant predictions

This directory replaces `analysis_notebooks/` for the benchmark. Every number and figure is computed by a script
here from the predictions collected under `msr_predictions/` and `baselines/{retrained,public}/` (nothing is read from
`analysis_notebooks/`), and every result table of `docs/BENCHMARKS_OCT2026.md` is drawn in at least one figure
(`results/table_figure_map.csv`, checked by `99_table_coverage.py`).

```
analysis/
  msr_analysis/        the library: registry (datasets, models, colours), data (loaders), boot (bootstrap), calib, screening, style, tables
  00_parse_benchmark_tables.py   BENCHMARKS markdown tables -> reference/benchmark_tables/*.csv (committed; the doc need not be present)
  00_validate_against_tables.py  recompute 509 documented cells from the predictions (the gate for everything else)
  11d_epistasis_levels_bootstrap.py  the epistasis suite (esm_msr.epi_metrics) on every model, bootstrapped over domains -> results/11_publication/fig5_epistasis_levels_bootstrap_*.csv (the ctx run's reverse leg: reference/ctx_wt_rev_devel.csv.gz)
  01_core_rho.py                 rho of every model on every dataset, protein / site / mutation bootstrap
  02_ ... 10_*.py                one script per theme; each writes figures (png + pdf) and the csv behind them to results/<NN_theme>/
  11_*.py, 11b, 11c              the publication figures (results/11_publication); 12_pair_epistasis_heatmaps.py the pair heat maps (results/12_pair_heatmaps)
  99_table_coverage.py           every BENCHMARKS table appears in a figure
  reference/                     small committed inputs: parsed doc tables, all_by_all_long.csv, leak audit matrix
  results/                       the output, one subfolder per script
  cache/                         pickled frames and bootstrap replicates (git-ignored, rebuilt on demand)
  tests/                         the bootstrap engine against scipy
```

Run: `./run_all.sh` (CPU only; `01_core_rho.py` is the slow step, about 15 minutes on one core). Environment: numpy, pandas, scipy, matplotlib
(`msr_venv` has them). Data root: `MSR_ROOT` (default `~/playground/esm-msr-devel`). The GPU is not used.

## How the numbers are computed

* **Scoring is BENCHMARKS' scoring**: stabilizing-positive for every model; seed means of per-seed rho; one pooled rho per external set and
  DMS; the mean over proteins of per-protein rho for Tsuboyama (per wild-type library: 23 validation, 33 test; measured rows, all backbones)
  and Domainome (per domain). `00_validate_against_tables.py` reproduces 509 documented cells: median |difference| < 1e-5, one cell off (below).
* **Bootstrap** (`msr_analysis/boot.py`): Spearman under integer row multiplicities, exactly scipy's `spearmanr` on the expanded sample (`tests/`).
  `protein`: whole proteins resampled, one draw over the union of proteins of the 12 external sets (a protein is its wild-type entry; Ssym's 357
  variant structures are its 15 wild types), so models and sets are paired and "mean of 12" has an interval; Tsuboyama libraries and Domainome
  domains resampled on their own. `site`: first mutated position, for the single-protein DMS where proteins cannot be resampled. `mutation`:
  rows (within a protein for the per-protein means). All three are in `results/00_core/rho_bootstrap.csv`; figures use protein-level where it exists
  and say which. The protein interval is much wider than the mutation interval (`02_thermostability/thermo_interval_types`).
* **ESM-MSR head rule** (`common.primary_key`): the WT head where only single mutants are scored, the combined prediction where multi-mutants are
  (PTMUL, PTMUL-D, DLG4/GRB2/GB1, Tsuboyama doubles). Where both matter (all-models figure, retrains figure, Tsuboyama, ProteinGym) both are drawn.
  Baseline = `release-baseline` triplicate; "devel" = `dc8-floorlink` triplicate; "released" = the single `esm-msr-baseline-w2a4-m16a16-capped` checkpoint
  (σ sweeps, ProteinGym, §2.6). σ is the LoRA scale (alpha).
* **Predicted epistasis**: ESM-MSR `epi_pred` = MT(AB) − WT(AB); ME stage 2 = ME2(AB) − ΣME1(single), same seed; ThermoMPNN-D = TM-D(AB) − ΣTM(single);
  SPURS multi = multi(AB) − Σsingle. The epistasis tables score doubles with both singles measured (8,494 val / 12,455 test); table 2.3 scores every double with a measured dddG.
* **RMSE**: ESM-MSR (and, for a like-for-like comparison, every trained model) is recalibrated by a scale and bias fitted on **one sacrificial set, S461**;
  S461 and its superset S669 are starred. See `07_rmse/`.
* **Screening**: a hit is ddG > 0.5 kcal/mol (`--tau`). Hit rate = fraction of suggested variants that are hits; P(recover ≥ 1) = chance that K screened variants of a
  protein contain a hit (over proteins that have one); best ddG in the top K, against the oracle and K random picks; mean ddG of the selection (the notebooks' "gain per
  selected mutant", here a plain mean over the selected variants, not the notebooks' rectified/thresholded variants). Proteins are the resampling unit.

## Where each request is met

| request | where |
|---|---|
| 1 scripts in `analysis/`, outputs in subfolders | this layout; `results/<NN_theme>/` |
| 2 every BENCHMARKS table in a figure | `results/table_figure_map.csv` (23 result tables drawn; T7.1 is a table of cluster paths, not a result) |
| 3 notebook figures with protein / mutation bootstrap | per-set Spearman dot plots → `02`, `03`, `04`; ProteinGym → `05`; RMSE → `07`; gain per selected mutant → `08`; epistasis bars and density scatters → `06`; NDCG and the AUPRC classification bars of the notebooks are **not** reproduced (the hit-rate / recovery statistics of `08` replace them) |
| 4 style kept, simpler | same dot-and-whisker panels with grey bands, same density scatters; one message per figure, ≤ 8 models per dot panel except the deliberate all-models figure |
| 5 scaffolds | ESM-MSR vs retrains: `02/thermo_vs_retrains`, `03`, `04/dms_vs_retrains`, `05/pg_categories` (lower panel); vs lower σ and zero-shot: `04/dms_sigma_vs_zero_shot`, `05/pg_categories` (upper); WT head on singles / combined on multi-mutants: the head rule above; **all models on all 12 thermostability sets**: `02/thermo_all_models` |
| 6 recalibrated RMSE with a sacrificial set | `07_rmse/rmse_recalibrated` (donor S461) and `rmse_recalibrated_ssym` (donor Ssym, which has both signs: 5-95% range −3.1 to +3.1, sd 1.83 against 1.25); the choice of donor is tested in panel C and `calibration_transfer*.csv` |
| 7 more scatter and raw data, especially dddG | `06/epistasis_scatter_dddG` (3 sets × 5 predictors), `06/epistasis_scatter_ddG_doubles`, `03/tsuboyama_scatter`, `03/tsuboyama_per_library`, `02/thermo_scatter_external`, `05/pg_per_dms_scatter`, `09/length_vs_performance` |
| 8 screening statistics | `08_screening/` (pooled over each external set and per protein as K grows) |
| 9 sizes of the full-sized proteins | `09_sequence_length/length_summary.csv` (median / min / max per dataset, single-protein DMS included), `sequence_length_distribution` (external sets, Domainome, Tsuboyama; the single-protein DMS are in its footnote: DLG4 84, GRB2 56, MYO 154, ESTA 181, GB1 56 residues), `length_vs_performance` and `length_correlation_summary` (ESM-MSR WT head, ME stage 1, ThermoMPNN, SPURS multi; the ten single-mutant sets) |

## Publication figures (`11_publication_figures.py`, `results/11_publication/`)

The notebooks' headline bar charts on the capped Tsuboyama **test** libraries (33 proteins), recomputed with intervals from resampling libraries (proteins) instead of the spread over three models. Columns come in three labelled categories:

* **Retrained on the capped split**: ESM-MSR WT head and combined head (both always shown: the WT head is the cheap one and the focus of the paper) and one column per retrained baseline, scored by the model that fits the variant: ME stage 1 for singles and stage 2 for doubles, ThermoMPNN for singles and ThermoMPNN-D for doubles, SPURS for singles and SPURS multi for doubles (MutateEverything(1+2), ThermoMPNN(-D), SPURS(-multi) on the axis; the panels of figure 2b name the model they use).
* **Zero-shot**: ESM3-small-open (the σ = 0 control of an ESM-MSR checkpoint, so the calibration head is kept) scored by **canonical masked marginal** (all mutated positions masked at once in the wild-type sequence); ProteinMPNN; ProSST.
* **Supervised, not trained on Megascale**: GeoDDG-3D (better than DDGemb on mean per-library singles ρ; trained on S8754) and Rosetta, on the far right. ESM3ΔG is left out on purpose: its MGnify fine-tune is recent and overlaps the test proteins.

Every legend gives the number of variants and libraries behind each split; the black "overall" line is drawn behind the bars. The ESM3 masked runs are `cluster/esm3_masked_gamut.sh` (one job per strategy on the fixed gpu4 slot, ESM-MSR checkpoint at `--sigma 0`, no ProteinGym); their outputs are copied to
`msr_predictions/esm3_small_open_sigma0_masked/<strategy>/` and read by `data.add_esm3_masked`. Without them the ESM3 column falls back to the unmasked σ = 0 readout and is marked ‡. `--esm3 independent` draws independent masking (each mutated position masked on its own, partner visible on the mutant pass) in figures 1 and 2; the other strategy stays in the CSVs.

| file | what |
|---|---|
| `fig1_ndcg96_singles_doubles` | NDCG@96 (rectified relevance max(ddG, 0), per library, as in the notebooks), singles against doubles (the doubles bars use the 8 libraries with a stabilizing double: NDCG is undefined without one); the black line is the overall NDCG@96 of a library with its singles and doubles ranked together (a baseline's singles model on the singles and its doubles model on the doubles), 0.01 to 0.03 below the singles bar in every column, so it is drawn; the script drops it for a footnote when it matches the singles bar to within 0.01 in 9 of 10 columns |
| `fig2_spearman_stab_destab` (+ `_pooled`) | Spearman ρ over all variants (singles and doubles; a baseline's singles model on the singles, its doubles model on the doubles) for stabilizing (measured ddG > 0) against destabilizing (< 0) variants; the legend gives the variants counted (7,417 stabilizing, 41,323 destabilizing) and the domains (33 each); black line = overall ρ; per-domain mean over domains with ≥ 10 variants in the subset, and pooled over domains |
| `fig2b_spearman_stab_destab_singles_doubles` (+ `_pooled`) | the same as one panel for singles and one for doubles, each with the model that fits it |
| `fig3_spearman_doubles_additive_vs_epistatic` | ρ on doubles, the additive version (the singles model summed over the double's substitutions: WT head, ME stage 1, ThermoMPNN, SPURS, Rosetta, GeoDDG-3D and ProteinMPNN sums of singles, ESM3 WT pass) against the version that scores the double (combined head, stage 2, ThermoMPNN-D, SPURS multi, the direct double, ESM3 mutant pass); ProSST is additive by construction; `…_paired_difference.csv` has the paired differences |
| `fig4_naive_epistasis_pos_neg_{grouped,pooled}` | naive epistasis ρ (predicted against measured dddG, doubles with both singles measured) for measured positive against negative epistasis; the predicted epistasis is always the double prediction of one model minus the singles of another (or another head), spelled out under each column; for ESM3-small-open it is the independent-masking mutant pass minus the masked-marginal WT pass (`esm3ind - esm3mm`; independent minus independent is in the CSV: naive ρ 0.21 against 0.11) |
| `fig5a_epistasis_levels_bars_{test,val}` (`11b_publication_epistasis_levels.py`, numbers from `11d_epistasis_levels_bootstrap.py`) | the epistasis hierarchy of BENCHMARKS tables 2.0-2.2 recomputed with the suite itself and bootstrapped over domains: three bars per model for the levels from coarse to fine (pair ρ, line ρ, cell rank), two horizontal dashes across them for naive ρ (raw dddG, saturation included) and saturation-subtracted ρ (the tables' "beyond ρ"); 95% intervals on every bar and, as whiskers at the dashes' ends, on the dashes. Models: ESM-MSR devel's MT − WT (the epistasis head) and ~WT − WT (the WT adapter read on the mutated sequence: a control for what the backbone and single-mutant training know), ME stage 2, ThermoMPNN-D, SPURS multi, ESM3-small-open (independent-masking mutant pass − masked-marginal WT pass), ProteinMPNN, GeoDDG-3D, Rosetta |
| `fig6a_dG_additive_vs_dddG`, `fig6b_esm_msr_dddG_pred_vs_true` (`11c_publication_epistasis_scatters.py`) | the notebooks' density scatters with marginal histograms on the 12,455 test doubles with both singles measured: −(additive dG = dG_wt + ddG_A + ddG_B) against the measured dddG (ρ 0.64; a double lost to the assay floor, dG ≈ 0, sits on the x = y line whatever its domain's stability, which is what the line shows), and ESM-MSR's predicted dddG (the MT head minus the WT head, mean of the three baseline seeds) against the measured dddG (ρ 0.46); the only line is x = y (the notebooks drew a dashed linear fit); the colour bar sits clear of the tick labels |

Reading them (test domains): ESM-MSR's overall values reproduce the benchmark (doubles ρ 0.709 / 0.766 for WT / combined, naive ρ 0.452). NDCG@96 (singles / doubles): ESM-MSR WT 0.58 / 0.60 and combined 0.62 / 0.71 against 0.41 to 0.55 / 0.43 to 0.51 for the
retrained baselines, 0.32 to 0.34 / 0.31 to 0.50 for the zero-shot models and 0.36 to 0.37 / 0.29 to 0.40 for GeoDDG-3D and Rosetta; the doubles intervals are wide (8 libraries with a stabilizing double). Stabilizing variants are ranked far worse than destabilizing ones by every model
(all variants, ESM-MSR WT 0.27 against 0.80 and combined 0.33 against 0.82; baselines 0.16 to 0.27 against 0.75 to 0.77; zero-shot 0.03 to 0.06 against 0.50 to 0.59; GeoDDG-3D 0.19 / 0.57, Rosetta 0.08 / 0.65). On doubles the epistatic version helps ESM-MSR (0.709 to 0.766, paired difference +0.057 [0.030, 0.082], the only interval that excludes zero) and ME (+0.035 [−0.006, 0.074]) and not ThermoMPNN, SPURS, GeoDDG-3D or ProteinMPNN; for ESM3-small-open the
mutant pass is no better than its WT pass (0.408 to 0.381, independent masking). No model ranks *negative* epistasis (naive ρ between −0.25 and +0.15), because predicted epistasis is mostly the saturation of the readout; positive-epistasis ρ is 0.32 to 0.49 for ESM-MSR, ME stage 2, SPURS multi and Rosetta and 0.05 to 0.24 for the others (0.40 and 0.49 for ESM-MSR and ME).
ESM3-small-open's scoring barely matters: singles ρ is 0.574 under masked marginals and under independent masking (the same input for one mutated position), doubles 0.398 (marginal), 0.408 (independent, WT pass), 0.381 (independent, mutant pass) against 0.421 for the unmasked readout of the older σ = 0 files.
Figure 5 (levels, test; the intervals are wide because only 13 domains have doubles: ESM-MSR's MT − WT pair ρ 0.61 is [0.31, 0.91]; ESM-MSR devel three-seed means): MT − WT keeps the most at every level (pair ρ 0.61, line ρ 0.38, cell rank 0.21) and its saturation-subtracted ρ (0.49) is above its naive ρ (0.46); ME stage 2 and Rosetta fall from naive 0.41 / 0.38 to 0.22 / 0.21 once saturation is subtracted, and ThermoMPNN-D has essentially no pair or cell signal. The context control ~WT − WT, which has never seen a double, already holds 91% of the pair level (0.55), 95% of the line level (0.36) and 73% of the cell rank (0.15), and sits above every other model at the pair and line levels (ESM3-small-open, zero-shot, has pair ρ 0.46, line ρ 0.19, cell rank 0.10, ProteinMPNN 0.47 / 0.13 / 0.06); the MT head's addition is largest in relative terms at the finest level (validation: 94% / 77% / 71%).

## Pair-epistasis heat maps (`12_pair_epistasis_heatmaps.py`, `results/12_pair_heatmaps/`)

The notebooks' `plot_joint_with_marginals` (a 20 × 20 map of dddG over the residue identities of a position pair, with the single-substitution scores of each position as bars on the margins), for **every
test position pair whose 361 identity pairs are all in the data** (15 of the 45 test pairs, in 7 domains: 2D1U, 2KRU, 2KVS, 2L09, 2LC2, 3DKM, 3ZGK; `pairs.csv`; 6 of them have a usable measurement in every cell, the others
have grey × where the double sits at an assay bound or a single is missing). One figure per pair, one panel for the measurement and one for each of 13 models (ESM-MSR baseline and devel, ME stage 2, ThermoMPNN-D, SPURS multi,
ESM3-small-open, ProteinMPNN, GeoDDG-3D, DDGemb, Rosetta, and the Megascale-trained ESMtherm\*, IFUM WT backbone\*, IFUM ESMFold\*; ESM3ΔG is left out as elsewhere). Layout:

```
results/12_pair_heatmaps/raw/<domain>/<wt><pos>_<wt><pos>.png                  dddG as is (measured) and the model's predicted epistasis
results/12_pair_heatmaps/saturation_corrected/<domain>/<wt><pos>_<wt><pos>.png the same minus E[dddG | additive expectation]
results/12_pair_heatmaps/pairs.csv, agreement.csv                              the pairs; the Spearman ρ of every model against the measured cells, per pair and mode
```

*Saturation-corrected* is the beyond-global residual of the epistasis suite (`esm_msr/epi_metrics.py`, BENCHMARKS table 2.2): each side minus its mean given its own additive expectation, E[dddG | x] with x = dG_wt + ddG_A + ddG_B for the
measurement and x' = dG_wt + the model's own additive prediction of the double for a model, the 25-quantile-bin smoother fitted on all 12,455 test doubles. What it removes is the assay floor and any diminishing returns; what is left is
pair, line and cell epistasis. Blue is positive epistasis, red negative, a grey × a cell with no usable value (the wild-type residue of each position is a row and a column of ×). One colour scale for every panel of every figure of a mode, symmetric about zero: ±3.19 kcal/mol raw and ±1.38 saturation-corrected (98th percentile of |measured| over the usable cells of the 15 pairs); a model's values beyond it are clipped (Rosetta's mostly are), and each panel title carries the panel's ρ. ESM-MSR's epistasis is always the MT head minus the WT head. Mean over the 15 pairs of the per-pair ρ
(raw / corrected): ESM-MSR 0.53 / 0.35, ESM-MSR devel 0.53 / 0.37, ME stage 2 0.41 / 0.23, SPURS multi 0.40 / 0.24, ESM3-small-open 0.38 / 0.16, ThermoMPNN-D 0.26 / 0.18, Rosetta 0.17 / 0.09, GeoDDG-3D 0.07 / 0.04;
ESMtherm\* 0.34 / 0.41 and IFUM\* 0.13 to 0.15 / 0.19 to 0.21 are the only models whose ρ rises once saturation is removed (they are trained on these domains).

## What the figures say (all intervals 95%)

* **ESM-MSR against the retrained baselines on the 12 thermostability sets** (`02`): ahead on average by 0.02 (ME stage 1; interval −0.01 to 0.04, not separated),
  0.04 (ME stage 2; 0.01 to 0.06), 0.05 (ThermoMPNN, ThermoMPNN-D, SPURS) and 0.08 (SPURS multi); the per-set differences are mostly inside the protein-level intervals; against most baselines the
  intervals exclude zero on PTMUL, K2369, K3822, PTMUL-D and S461 (not against ME stage 1, which is level with ESM-MSR on every set but PTMUL-D).
* **Tsuboyama** (`03`): 0.855 / 0.847 (val / test), SPURS retrained 0.803 / 0.801, ThermoMPNN 0.787 / 0.794; ESM-MSR is above each retrained baseline in 30 to 32 of the 33 test libraries.
* **Lower LoRA scale** (`04`, `05`): σ = 0.75 is the only setting that beats σ = 1 on ProteinGym (all 213 assays: +0.0053 per assay, interval 0.0024 to 0.0082, better on 135; without the trained Stability assays: +0.0098 [0.0069, 0.0126], better on 135 of 190); on the DMS σ = 0.5 has the highest rho on 5 of the 7 assays, their mean and Domainome (a choice made on these assays: an upper bound); ProSST is level with ESM-MSR on ProteinGym only because of the 23 Tsuboyama assays that are in the training split: without them ProSST's PG avg (0.514) is ahead of ESM-MSR devel σ = 1 (0.503) and level with σ = 0.75 (0.510).
* **Epistasis** (`06`): the combined head's dddG ranking, 0.505 (val) and 0.574 (test), is 0.15 to 0.23 above ME stage 2; λ = 0.5 is within 0.006 of the best constant weight except on GB1 (baseline triplicate: best at λ = 0, 0.022 above).
* **RMSE** (`07`): recalibration changes ESM-MSR little (S461 donor: scale 1.15, bias −0.19; Ssym donor: 1.27, +0.19); its RMSE is 0.83* on S461 and 1.1 to 1.9 elsewhere, within the protein-level intervals of the retrained baselines, with either donor.
* **Screening** (`08`): on Tsuboyama test libraries ESM-MSR's top-10 hit rate is 0.46 against 0.30 to 0.35 for the baselines and P(recover ≥ 1 in 10) = 0.92 against 0.77 to 0.85; on the external sets the pooled
  top-10% hit rates are mostly inside each other's intervals; ESM-MSR is ahead on PTMUL (and on K3822 against ThermoMPNN) and SPURS is ahead on S783 and S8754.
* **Length** (`09`): the external sets' proteins have a median length of 108 to 140 residues (range 20 to 782); per-protein rho falls with length for each of the four models and per-protein RMSE rises (numbers in `length_vs_performance_stats.csv`).

## Conventions added after review (2026-10-29)

* **Megascale-trained models** (ESMtherm and IFUM in both modes; list in `registry.MEGASCALE_TRAINED`) carry an asterisk, sit last in the forest figures and the epistasis hierarchy, and are greyed.
  ESM3ΔG is not among them: it is trained on the MGnify stability set and Megascale is its held-out test (BENCHMARKS section 6.3); it keeps its homology caveat in the leak maps.
* **ProteinGym**: the 23 Stability assays whose proteins the retrained models trained on (Tsuboyama libraries inside the capped train split; the leak audit gives the same 23 for ESM-MSR, ME, ThermoMPNN
  and the retrained SPURS) are removed from every model in the figures. Stability* is labelled with what is left, (43/66), and PG avg* and DMS mean* (190/213) are over the remaining assays. The documented all-213
  numbers are in `results/05_proteingym/pg_summary_all213.csv` and `pg_sigma_paired_all213.csv` (table 3.2 reproduces: σ = 0.75 is +0.0053, better on 135 of 213). "ours" and "ProteinGym reported"
  are separate rows for ProteinMPNN and ESM3-small-open (σ = 0).
* **SPURS**: the retrained SPURS and SPURS multi are unlabelled (they are the models of every table); the public checkpoint, whose values are frozen in `reference/spurs_public_checkpoint.csv` and
  `leak_public_spurs.csv` because the document no longer carries them, is labelled "(public checkpoint)".
* **Domainome screening** counts a hit as any increase in expression (> 0); the stability sets use > 0.5 kcal/mol.
* **Pooled versus per-library ranking of doubles** (`06/epistasis_scatter_ddG_doubles`, `doubles_pooled_vs_per_library.csv`): ESM-MSR leads the benchmark's per-library mean rho on the Tsuboyama test doubles
  (0.766 baseline, 0.781 devel against 0.652 ME stage 2) but not the rho pooled over libraries (0.469 baseline, 0.471 released, 0.514 devel latent, 0.560 devel after the link, against 0.580 ME stage 2,
  0.648 ME stage 1, 0.632 ThermoMPNN). The link (`msr_analysis/link.py`, parameters read from the three devel checkpoints) is monotone, so it cannot change per-library rho and does raise the pooled one. It is rebuilt
  with the checkpoints' one learned floor because the per-item floors (`y_floor.json`) are not collected, so post-link values are illustrative: used for this comparison only, they do not reproduce the document's
  devel epistasis scores (post-link naive rho 0.42 on validation against 0.643). The document's devel epistasis rows match the *latent* MT − WT difference (global error 0.824 / 1.079 to three decimals), so
  they appear not to have been put through the link; that is worth confirming.

## Known gaps and discrepancies

* **Every recomputed cell now agrees with the document to within 0.005** (`00_validate_against_tables.py`, 509 cells). The document's ME stage 1 GB1 cells were corrected on 2026-10-29 (0.299 direct, 0.294 additive, from ME's re-run
  of Oct 13 on the complete table); `reference/` was refreshed from that version (the ESM3ΔG leak codes of the hold-out verification came with it). Two cells that earlier looked unreproducible are explained and reproduce:
  S461 (0.721 / 0.710) is scored from ME's S669 predictions, never from ME's own S461 file (a different input: other constructs and MSAs), and ME stage 2 additive on GB1 (0.281) needs ME's own complete
  149,360-row GB1 file (the harness label table lacks three GB1 singles).
* **ME stage 2's pair and line levels differ from the document** (suite on the collected predictions, three-seed mean: pair ρ 0.17, line ρ 0.18; document table 2.2: 0.12, 0.23); its naive, saturation-subtracted ρ and cell rank agree (0.41 / 0.22 / 0.11), and so do all five metrics of every other model to within 0.005 (`11d`'s console check). Figure 5 uses the recomputation.
* **Not recomputable from the collected files** (drawn from the document's own numbers): the epistasis suite beyond naive rho in the heat maps (tables 2.0-2.2, `06/epistasis_hierarchy_*`; figure 5 recomputes it with `esm_msr.epi_metrics`, which lives
  with the training code, and the naive rho and global error columns *are* recomputed and match), the mutant-background ablation (table 2.5), the published SPURS checkpoints (tables 1.1-1.3
  rows, drawn in `10/spurs_published_vs_retrained` from `reference/spurs_public_checkpoint.csv`), the published-paper evaluation (table 1.5), and the leak codes (table 6).
* **S571's labels are not in kcal/mol** (standard deviation 8.9, 5th percentile −20): it is excluded from the RMSE and screening figures; its rho is unaffected.
* ESM-MSR (devel) predictions are on the latent scale; rank metrics within a library are unaffected, and scale-dependent metrics for it are marked as such (see the conventions above).
* The ESM3 Forge models are scored for S461, K2369 and PTMUL-D from their parents' predictions (BENCHMARKS section 1.1); ME is scored for K2369 and PTMUL-D the same way.

## From the old notebooks

| notebook | now |
|---|---|
| `compare_supervised_external_summarize*` (Spearman, RMSE, gain, NDCG by set) | `02`, `03`, `04`, `07`, `08` |
| `compare_supervised_epistasis_rho_{val,test,variants,msrs}` (hierarchy bars, density scatters, classification) | `06`, `03` |
| `analyze_features_precomputed`, `variation_table` | not ported (feature analysis and the variation table are outside the BENCHMARKS tables) |
