# Tsuboyama analyses on the mega_splits retrains

Every Tsuboyama (MegaScale) analysis of `analysis/`, rerun on the models retrained on the full MegaScale split (`mega_splits`, the split
the baselines' own papers use) instead of the Homology-Restricted one (`splits_oct06_capped`). Results: `analysis/results/03_tsuboyama/mega_splits/`,
one folder per shared script, named as in `analysis/results/`.

```bash
bash analysis/tsuboyama/mega_splits/run_all.sh        # PYTHON=... to pick the interpreter; rebuilds results/03_tsuboyama/mega_splits/ from scratch
```

## Models (three seeds each)

| Key | Model | Source |
|---|---|---|
| `msrp0:wt`, `msrp0:comb`, `msrp0:epi` | ESM-MSR = FINAL_p0_mega (the FINAL_p0 recipe trained on mega_splits) | testing queue, `final_bench/predictions/mega_splits-{val,test}/final_checkpoints/FINAL_p0_mega/s{1,2,3}/` |
| `me1`, `me2`, `me2:epi` | MutateEverything stage 1 / stage 2 | cluster `outputs/tracked/me_mega_st{1,2}_s{k}/results_mega_splits_allbl_{val,test,testcomp}.csv` |
| `tmpnn`, `tmpnnd`, `tmpnnd:epi` | ThermoMPNN / ThermoMPNN + ThermoMPNN-D | cluster `outputs/tracked/thermompnn_mega_s{k}/predictions_{val,test}.csv` (`pred_single_additive`, `pred_combined`) |
| `spurs`, `spursm`, `spursm:epi` | SPURS / SPURS multi | cluster `outputs/tracked/spurs_mega_s{k}/tsuboyama_{val,test}.csv` |

The cluster runs are collected to `<MSR_ROOT>/baselines/retrained/mega_splits/tracked/<run>/` (`registry.MEGA_RAW`). Epistasis
definitions match the capped split: ESM-MSR comb(AB) − WT(A) − WT(B); ME st2 minus the same seed's st1 singles (the test run's
`testcomp` file supplies the constituent singles that are not measured); ThermoMPNN-D and SPURS multi minus their own single-additive reading.
No zero-shot or public supervised model was scored on mega_splits, so those rows and panels are empty here.

## How it works

* `00_build_tasks.py` writes `_tasks/tsuboyama_mega_{val,test}.csv.gz`, the label tables in the capped tables' layout, from FINAL_p0_mega's
  prediction file (the seeds are checked to agree on rows and labels; `ref_seq` is rebuilt per library and reproduces the capped table's on 100% of rows).
  val: 39,252 rows (37,610 measured), 31 libraries; test: 53,981 rows (49,048 measured), 28 libraries (30 backbones).
* `MSR_TSU_SPLIT=mega` (set by `run_all.sh`): `registry.TSU` becomes `['tsuboyama_mega_val', 'tsuboyama_mega_test']`, `RESULTS` becomes
  `results/03_tsuboyama/mega_splits/`, `data.load_frame` reads FINAL_p0_mega under the `msrp0` keys (three seeds) and the mega baselines
  (`data._load_mega`), and `style.save` rewrites the capped split's name and "ESM-MSR (p0)" in figure text. Scripts drop what has no mega
  counterpart: the document's tables (06: 2.0–2.2, 2.5), the devel link and ctx control (06, 11d, 13), Rosetta checks (13), the
  external and Domainome screening (08).
* `01_rho_overview.py`: the `03_tsuboyama.py` figures for the mega models, plus `tsuboyama_capped_vs_mega` (each model on its own split's test set).
* Then the shared scripts: `06_epistasis`, `11d` + `11b` (the `epi_metrics` hierarchy, bootstrapped over domains), `11c`, `11f`, `12`, `13`, `08`, `15`, `17`.

## Results (test set; seed means; bootstrap B = 200 for the ρ overview, 300 for the epistasis hierarchy)

**ddG ranking** (mean per-library Spearman ρ, `03_tsuboyama/table_mega_split.md`):

| Model | val | test | test singles | test doubles | same model, Homology-Restricted test |
|---|---|---|---|---|---|
| ESM-MSR combined | 0.868 | 0.827 | 0.830 | 0.662 | 0.845 |
| ESM-MSR WT head | 0.845 | 0.812 | 0.821 | 0.569 | 0.827 |
| SPURS | 0.813 | 0.799 | 0.796 | 0.567 | 0.801 |
| ThermoMPNN + D | 0.788 | 0.779 | 0.769 | 0.615 | 0.784 |
| ThermoMPNN | 0.782 | 0.777 | 0.769 | 0.557 | 0.794 |
| MutateEverything st1 | 0.790 | 0.774 | 0.775 | 0.509 | 0.771 |
| SPURS multi | 0.768 | 0.771 | 0.738 | 0.598 | 0.772 |
| MutateEverything st2 | 0.781 | 0.767 | 0.758 | 0.595 | 0.760 |

ESM-MSR combined is above every baseline in every paired library bootstrap draw and higher on 24–27 of the 28 test libraries. Training on
all of MegaScale moves no model by more than 0.02 against its Homology-Restricted score (different test sets, 28 vs 33 libraries).
Pooled over libraries (not the benchmark metric), ESM-MSR ranks doubles worst (ρ 0.42 vs 0.52–0.55, `06_epistasis/doubles_pooled_vs_per_library.csv`):
its per-library offsets are less consistent than the baselines'.

**Epistasis hierarchy** (`epi_metrics`, 16,923 test doubles with both singles measured in 20 domains; [95% domain bootstrap], `06_epistasis/fig5_*`):

| Model | naive ρ | beyond ρ | pair ρ | line ρ | cell rank | cell mag |
|---|---|---|---|---|---|---|
| ESM-MSR | 0.507 [0.41, 0.65] | 0.455 [0.27, 0.60] | 0.560 | 0.308 [0.25, 0.39] | 0.108 [0.07, 0.16] | 0.158 [0.10, 0.22] |
| ME stage 2 | 0.480 [0.40, 0.57] | 0.362 [0.21, 0.52] | 0.417 | 0.248 [0.21, 0.30] | 0.076 [0.05, 0.10] | 0.121 [0.07, 0.15] |
| SPURS multi | 0.397 [0.30, 0.50] | 0.309 [0.15, 0.45] | 0.394 | 0.289 [0.22, 0.35] | 0.049 [0.02, 0.08] | 0.039 [−0.02, 0.09] |
| ThermoMPNN-D | 0.354 [0.21, 0.52] | 0.305 [0.14, 0.48] | 0.418 | 0.215 [0.14, 0.28] | 0.016 [0.00, 0.03] | 0.003 [−0.03, 0.05] |

ESM-MSR has the highest point value at every level on test, and at every level but pair ρ on validation (val pair ρ 0.219 vs 0.299
for ME st2, both intervals spanning ~0 to 0.6; val cell rank 0.206 vs 0.090). The intervals overlap at the coarse levels; the finest
(line, cell) levels separate ESM-MSR from SPURS multi and ThermoMPNN-D. On doubles clear of
the assay floor (additive dG > 1, `06_epistasis/dddG_scatters/stats.csv`) ESM-MSR keeps ρ 0.571 while ME st2 falls to 0.267,
ThermoMPNN-D to 0.197 and SPURS multi to −0.067: the baselines' epistasis signal on mega_splits is mostly the floor. The measured −dG_add
control alone reaches naive ρ 0.618, above every model.

**Screening** (Tsuboyama test, Cys excluded, K = 10 per protein, `08_screening/screening_per_protein_*.csv`): SPURS leads on hit rate
(0.42) and rectified gain (5.6 kcal/mol singles; ESM-MSR 5.3, ME 4.5, ThermoMPNN 4.5); ESM-MSR ties ME on P(≥ 1 hit) (0.885).
The gain per variant still follows a − b ln K, but less tightly than on the capped split (rmse 0.018 vs 0.004 for ESM-MSR, `17_gain_fit/`).
