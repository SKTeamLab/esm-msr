# analysis/: the October 2026 benchmark, re-done from the per-variant predictions

This directory replaces `analysis_notebooks/` for the benchmark. Every number and figure is computed by a script
here from the predictions collected under `msr_predictions/` and `baselines/{retrained,public}/` (nothing is read from
`analysis_notebooks/`), and every table of `docs/BENCHMARKS_OCT2026.md` is drawn in at least one figure
(`results/table_figure_map.csv`, checked by `99_table_coverage.py`).

```
analysis/
  msr_analysis/        the library: registry (datasets, models, colours), data (loaders), boot (bootstrap), calib, screening, style, tables
  00_parse_benchmark_tables.py   BENCHMARKS markdown tables -> reference/benchmark_tables/*.csv (committed; the doc need not be present)
  00_validate_against_tables.py  recompute 465 documented cells from the predictions (the gate for everything else)
  01_core_rho.py                 rho of every model on every dataset, protein / site / mutation bootstrap
  02_ ... 10_*.py                one script per theme; each writes figures (png + pdf) and the csv behind them to results/<NN_theme>/
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
  and Domainome (per domain). `00_validate_against_tables.py` reproduces 465 documented cells: median |difference| 1e-5, three cells off (below).
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
| 2 every BENCHMARKS table in a figure | `results/table_figure_map.csv` (26 result tables drawn; T7.1 is a table of cluster paths, replaced by the section below) |
| 3 notebook figures with protein / mutation bootstrap | per-set Spearman dot plots → `02`, `03`, `04`; ProteinGym → `05`; RMSE → `07`; gain per selected mutant → `08`; epistasis bars and density scatters → `06`; NDCG and the AUPRC classification bars of the notebooks are **not** reproduced (the hit-rate / recovery statistics of `08` replace them) |
| 4 style kept, simpler | same dot-and-whisker panels with grey bands, same density scatters; one message per figure, ≤ 8 models per dot panel except the deliberate all-models figure |
| 5 scaffolds | ESM-MSR vs retrains: `02/thermo_vs_retrains`, `03`, `04/dms_vs_retrains`, `05/pg_categories` (lower panel); vs lower σ and zero-shot: `04/dms_sigma_vs_zero_shot`, `05/pg_categories` (upper); WT head on singles / combined on multi-mutants: the head rule above; **all models on all 12 thermostability sets**: `02/thermo_all_models` |
| 6 recalibrated RMSE with a sacrificial set | `07_rmse/rmse_recalibrated`; the choice of donor is tested in panel C and `calibration_transfer.csv` |
| 7 more scatter and raw data, especially dddG | `06/epistasis_scatter_dddG` (3 sets × 5 predictors), `06/epistasis_scatter_ddG_doubles`, `03/tsuboyama_scatter`, `03/tsuboyama_per_library`, `02/thermo_scatter_external`, `05/pg_per_dms_scatter`, `09/length_vs_performance` |
| 8 screening statistics | `08_screening/` (pooled over each external set and per protein as K grows) |
| 9 sizes of the full-sized proteins | `09_sequence_length/length_summary.csv` (median / min / max per dataset), `sequence_length_distribution`, `length_vs_performance`, `length_correlation_summary` |

## What the figures say (all intervals 95%)

* **ESM-MSR against the retrained baselines on the 12 thermostability sets** (`02`): ahead on average by 0.03 (ME stage 1; interval −0.01 to 0.05, not separated),
  0.04 (ME stage 2), 0.05 (ThermoMPNN, ThermoMPNN-D, SPURS) and 0.08 (SPURS multi); the per-set differences are mostly inside the protein-level intervals; against most baselines the
  intervals exclude zero on S461, PTMUL, K2369, K3822 and PTMUL-D.
* **Tsuboyama** (`03`): 0.855 / 0.847 (val / test), SPURS retrained 0.803 / 0.801, ThermoMPNN 0.787 / 0.794; ESM-MSR is above each retrained baseline in 30 to 32 of the 33 test libraries.
* **Lower LoRA scale** (`04`, `05`): σ = 0.75 is the only setting that beats σ = 1 on ProteinGym (+0.0053 per assay, interval 0.0024 to 0.0082, better on 135 of 213); on the DMS σ = 0.5 has the highest rho on 5 of the 7 assays, their mean and Domainome (a choice made on these assays: an upper bound); ProSST is level with ESM-MSR on ProteinGym only because of the 23 Tsuboyama assays that are in the training split.
* **Epistasis** (`06`): the combined head's dddG ranking, 0.505 (val) and 0.574 (test), is 0.15 to 0.23 above ME stage 2; λ = 0.5 is within 0.006 of the best constant weight except on GB1 (baseline triplicate: best at λ = 0, 0.022 above).
* **RMSE** (`07`): recalibration changes ESM-MSR little (scale 1.15, bias −0.19 on S461); its RMSE is 0.83* on S461 and 1.1 to 1.9 elsewhere, within the protein-level intervals of the retrained baselines; the result does not depend on the donor set.
* **Screening** (`08`): on Tsuboyama test libraries ESM-MSR's top-10 hit rate is 0.46 against 0.30 to 0.35 for the baselines and P(recover ≥ 1 in 10) = 0.92 against 0.77 to 0.85; on the external sets the pooled
  top-10% hit rates are mostly inside each other's intervals; ESM-MSR is ahead on PTMUL (and on K3822 against ThermoMPNN) and SPURS is ahead on S783 and S8754.
* **Length** (`09`): the external sets' proteins have a median length of 108 to 140 residues (range 20 to 782); per-protein rho falls with length for every model (ESM-MSR −0.34 [−0.44, −0.22]; retrained −0.24 to −0.29) and per-protein RMSE rises.

## Known gaps and discrepancies

* **Three recomputed cells differ from the document by more than 0.005**: ME stage 1 / stage 2 on S461 (0.678 / 0.657 here, 0.721 / 0.710 in the table; ME's own S461 file scored
  against its own labels also gives 0.679, so the document's value is not reproduced and its provenance is unknown) and ME stage 2 additive on GB1 (0.270 against 0.281; the additive
  readout needs both singles in the file and 13 % of GB1 doubles lack one).
* **Not recomputable from the collected files** (drawn from the document's own numbers): the epistasis suite beyond naive rho (tables 2.0-2.2, `06/epistasis_hierarchy_*`; the suite lives
  with the training code, and the naive rho and global error columns *are* recomputed and match), the mutant-background ablation (table 2.5), the published SPURS checkpoints (tables 1.1-1.3
  rows, drawn in `10/spurs_published_vs_retrained` from `reference/all_by_all_long.csv`), the published-paper evaluation (table 1.5), and the leak codes (table 6).
  The SPURS multi row of tables 2.1 / 2.2 is the published checkpoint; `06/epistasis_naive_recomputed` shows the retrained one.
* **S571's labels are not in kcal/mol** (standard deviation 8.9, 5th percentile −20): it is excluded from the RMSE and screening figures; its rho is unaffected.
* ESM-MSR (devel) is on the latent scale (its link is not applied): rank metrics are unaffected, and no scale-dependent metric is computed for it. `msr_predictions/README.md` explains.
* The ESM3 Forge models are scored for S461, K2369 and PTMUL-D from their parents' predictions (BENCHMARKS section 1.1); ME is scored for K2369 and PTMUL-D the same way.

## From the old notebooks

| notebook | now |
|---|---|
| `compare_supervised_external_summarize*` (Spearman, RMSE, gain, NDCG by set) | `02`, `03`, `04`, `07`, `08` |
| `compare_supervised_epistasis_rho_{val,test,variants,msrs}` (hierarchy bars, density scatters, classification) | `06`, `03` |
| `analyze_features_precomputed`, `variation_table` | not ported (feature analysis and the variation table are outside the BENCHMARKS tables) |
