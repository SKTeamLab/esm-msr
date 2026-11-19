# Three-seed audit of the analysis (Nov 18)

**Rule.** Every retrained model has triplicates (ESM-MSR p0 and the ablation families `FINAL_*`, MutateEverything, ThermoMPNN, SPURS). Every figure and table uses all three seeds: the point value is the seed mean
(for a per-protein / per-assay statistic: the statistic of each seed, then the mean) and the bootstrap resamples the seeds together with the proteins, sites or assays (`msr_analysis.boot.bootstrap_rho`: one set of
seed weights per replicate, shared by every model; `pub.boot_reps`). Zero-shot and published models (ESM3-small-open, ProteinMPNN, ProSST, GeoDDG-3D, Rosetta, ESMtherm, IFUM, DDGemb) are single runs and stay as they are.
A scatter plot shows the prediction averaged over the seeds (its statistic is then the correlation of the average; the figure says so).

## What was single-seed, and what it is now

| where | before | now |
|---|---|---|
| `msr_analysis/data.py::_load_msr` (every `load_frame`; key `msrp0:*`) | FINAL_p0 seed 1 (the registry said "seed 1 only"), baselines and others 3 seeds | FINAL_p0 seeds 1-3 for every dataset (22 folders in `final_bench/predictions` all have s1-s3); the frame cache was rebuilt |
| `msr_analysis/external.py::seed_frame` (was `_seed1_frame`; thermostability, DMS, Domainome, `01b`) | `preds[k][:1]` for **every** model: ME, ThermoMPNN, SPURS were also seed 1 in the figures | all rows; `point_rho` = seed mean; boot cache `boot_ext_mutation.pkl` (version 4) |
| `02_thermostability`, `04_dms_domainome`, `01_summary` panel a, `01b_thermostability_pvalues` (+ its head test) | seed 1 | seeds 1-3, seeds resampled; the seed-1 / seeds 1-3 pair of the head test is now just the three-seed row |
| `05_proteingym` (per-assay rho of every model, Domainome frame, sigma 0.5 Domainome) | s1 files of p0 (WT and combined), ME st1 / st2, ThermoMPNN, SPURS, ESM-MSR devel / baseline; sigma 0.5 Domainome s1 | per-assay rho of every seed, then the mean (`_rho_seeds`); cache `pg_all_models.csv`; sigma 0.5 Domainome seeds 1-3 |
| `proteingym/p0_sigma/*` (panel c of `figure3`, `01`, `02`, `05`) | FINAL_p0 s1 at sigma 1 and 0.5 | per-assay rho per seed then the mean (`pgsigma.at_seed`); the per-assay bootstrap of `02` resamples variants jointly across sigma and seeds; `05` computes every column per seed |
| `pg_activity/02_activity_screening.py` | p0, ME st1, ThermoMPNN, SPURS: seed 1 | seeds 1-3 of the four, per-protein screening table per seed then the mean |
| `08_screening`, `15_tsuboyama_single_screening` | already iterate over the seeds of a frame (they were fed one for p0) | 3 seeds for p0 through the frame |
| `17_gain_per_variant_fit` | `preds[k][0]` | per-protein table of each seed, averaged |
| `combined_vs_wt/02_pg_per_assay` -> `04`, `06`, `ablation_vs_p0/02` | p0 s1 and p1 s1 columns | `p0_*` / `p1_*` = mean over the complete seeds (p0 1-3; p1 combined: 1 and 3, its seed 2 holds 126 of 219 assays and is left out); single-seed columns kept as `p0_s<k>_*` |
| `combined_vs_wt/03_unit_rho` | WT-vs-MT head correlation at seed 1 | per seed, then the mean |
| `ablation_vs_p0/02_pg_vs_p0` | WT head s1 of every family | `final_runs.pgym_per_dms_seeds` (mean over seeds 1-3) |
| `98_final_ablation_table` | paired deltas "at seed 1" (`delta_s1.csv`): ProteinGym assays, the benchmark frames, the epistasis suite | `delta_paired.csv`: seeds resampled in every group (the epistasis suite draws domain and seed weights); the standard error of a bold cell is the bootstrap's, since it contains the seed noise (the seed SD was added on top before: double counting) |
| `03_tsuboyama`, `06_epistasis`, `tsuboyama/mega_splits/01` scatters | `preds[k][0]` | prediction averaged over the seeds |
| `11d_epistasis_levels_bootstrap` | p0 seeds only if the frame had them (it had seed 1) | seeds 1-3 through the frame |

Already three seeds before this audit: `98` cells (seed mean +- SD), `11h` / `11i` / `11j` / `11k` / `11l` / `11m` / `11n` (final frames), `ablation_vs_p0/01` and `combined_vs_wt/01`, `05`, `08`, `18`, `18b`, `18c` (MGnify: p0 and the three SPURS / ThermoMPNN seeds), `faure_epistasis`.

## What stays at one seed, and why

| where | reason |
|---|---|
| FINAL_p1 combined on ProteinGym | seed 2 holds 126 of 219 assays: seeds 1 and 3 |
| `proteingym/sigma_tradeoff` | the baseline sigma sweep exists as one run |
| `scoring/resource_summary.py` | runtime / memory of one run (not an accuracy) |
| ESM3-small-open, ProteinMPNN, ProSST, GeoDDG-3D, Rosetta, ESMtherm, IFUM, DDGemb, public SPURS | single runs |

## What the change did to the numbers (seed 1 -> seeds 1-3)

| quantity | change |
|---|---|
| ESM-MSR p0, thermostability sets (`02`, panel a of `figure3`) | at most 0.006 per set (Ssym 0.783 -> 0.778, S783 0.713 -> 0.707, PTMUL combined 0.586 -> 0.592) |
| ME stage 2 on PTMUL / PTMUL-D | -0.010 to -0.027 (seed 1 was a high seed); ThermoMPNN-D on PTMUL-D -0.013; the other retrained models within 0.006 |
| sigma 0.5 combined readout on ProteinGym (panel c) | the local queue finished seed 3 on Nov 19 12:21 (`final_pgym_comb_sigma0.5/FINAL_p0` s1-s3, 71 files each = 69 assays + summary + log): the 69 multi-mutant assays are now three-seed at sigma 0.5 too |
| ThermoMPNN-D on ProteinGym (`05`, panel b) | seeds 2 and 3 were run on gpu4 on Nov 19 (jobs 390049 / 390050, `submit_pgym_thermompnn.sh`) and fetched with seed 1 into `retrained/thermompnn/predictions/epistatic_corrected/seed<k>/proteingym` (seed-1 files reproduce the saved table exactly): the column is now a three-seed mean |
| ProteinGym average (`05`) | 0.003 or less for every model; Domainome: SPURS +0.016, ThermoMPNN -0.006, ESM-MSR -0.003, ME -0.004 |
| per-assay rho at sigma 0.5 / 1 (`p0_sigma/01`, panel c) | +0.001 on average (SD 0.005-0.008, largest 0.044); the medians of panel c move by 0.001-0.007 (Stability sigma 1: +0.184 -> +0.191) |
| ProteinGym combined - WT (`combined_vs_wt/04`) | p0: Stability +0.016 (Holm p = 0.022), Activity -0.014 and Organismal fitness -0.021 (Holm p = 0.0005); p1 (seeds 1, 3): none significant |
| paired ablation deltas (`98`, `delta_paired.csv`, 755 rows) | the median 95% interval widens from 0.020 to 0.023 (the seeds are resampled), the number of intervals that exclude 0 falls from 267 to 190 (139 of 755 change side): single-seed "significant" ablations were partly seed noise |
| ProteinGym ablation vs p0 (`ablation_vs_p0/02`) | full MegaScale +0.014 on the average (Stability +0.038, Organismal fitness +0.013, all Holm p = 0.002); p1 worse on Organismal fitness (-0.013); the rank-threshold-0 gain on Stability is no longer significant (+0.002, Holm p = 0.16) |

`msr_analysis/external_s1.py` was renamed `external.py` (it no longer selects seed 1; `_seed1_frame` -> `seed_frame`, `load_s1_mutation_boot` -> `load_mutation_boot`).

## Check

```bash
grep -rnE "preds\[[^]]*\]\[(0|:1)\b|\[:1\]|[Ss]eed ?1 only" analysis --include=*.py | grep -v "leak_audit\|scoring"
```
lists only the `[:1]` of list slicing in `11g` / `11n` (legend handles, footnotes); the Rosetta scatter (`13`) indexes the single Rosetta run.
