# Cluster benchmark campaign, October 2026 (subhak-gpu)

Consolidated 2026-10-21 from the four Claude sessions that ran on `sreeves@subhak-gpu.uwaterloo.ca`
between 2026-10-04 and 2026-10-21, their notes (`~/project_oct06_baselines/{README,RUNS}.md`,
`~/stability-baselines/PROGRESS.md`, `~/esm-msr-devel/baselines/README.md`) and the result files they wrote.
Training of ESM-MSR itself (the `o6`, link, `dc` series) ran on the local 5090 and is logged in
`EXPERIMENTS.md`, not here. This document covers what the cluster produced: **every comparison of
ESM-MSR against other models**.

All paths below are on the cluster unless they start with a repo-relative path.

## 1. What was done

| workstream | dates | session | where the results are |
|---|---|---|---|
| A. MutateEverything (ME) and ThermoMPNN / ThermoMPNN-D (TMPNN) **retrained on our split** (`splits_oct06_capped.pkl`), 3 seeds each; scored on Tsuboyama val/test, 12 external sets, 7 DMS, Domainome, with the full epistasis suite | 10-06 → 10-21 | `842004cb` (cwd `~`) | `~/esm-msr-devel/outputs/baselines_capped/`, index `~/project_oct06_baselines/` |
| A'. Same, **without mutant-background libraries** in train/val (the "nobb" ablation) | 10-13 → 10-15 | `842004cb` | `outputs/baselines_capped/*_nobb.csv`, `RUNS.md` |
| B. The **dc8-floorlink** triplicate (`esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}`) through the same gamut | 10-20 → 10-21 | `842004cb` | `outputs/baselines_capped/*dc8*.csv` |
| C. **ProteinGym v1.3** (217 DMS) for ESM-MSR (capped, ε sweep, dc8 × 3 seeds × 4 ε), SPURS, ProteinMPNN, ME, ThermoMPNN | 10-15 → 10-21 | `0e3d7fac` (cwd `~/esm-msr-devel`) | `~/esm-msr-devel/outputs/pgym/` |
| D. **Third-party stability predictors** run natively: ESMtherm, DDGemb (server), DDMut (server), IFUM, GeoDDG, ProSST, ESM3ΔG; UniMutStab blocked | 10-04 → 10-21 | `796d0b37` (background job) | `~/stability-baselines/results/FINAL_TABLE.md` |
| E. One ESM-MSR training run on the cluster (`v6_censor0_rank05`, Comet `esm-msr-agent`) | 10-04 | `823e327b` | cancelled at 91 % of epoch 0; **no result** |

Common ground for A–C: the ESM-MSR reference checkpoint is `LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`
("capped", one seed); dc8 is the three-seed `dc8-floorlink-canon` series (same split, same LoRA config;
`val_rho_combined_avg` 0.840 / 0.839 / 0.841).

## 2. Retrained baselines vs ESM-MSR on the Tsuboyama split

**Rule throughout: neither baseline was modified, only its data layer.** ME follows the upstream README
recipe (stage 1 = singles, 20 epochs; stage 2 = doubles, 100 epochs from stage 1). ThermoMPNN is two
upstream configs, i.e. two models: `train_single.yaml` (singles) and `train_epistatic.yaml` (ThermoMPNN-D,
siamese, doubles only). The baselines' additive head is their own singles summed; their `mt` head is ME's
direct double head or ThermoMPNN-D. `comb ≡ mt` for both; `ctx` does not transfer (no separable adapter).

### 2.1 Validation, epistasis hierarchy (valfull: 34,957 rows, 8,494 doubles, 24 pairs, 2,509 flips)

Seed means; ESM-MSR capped is one checkpoint. Metric definitions: `validation_metrics.md`.

| model / head | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| ME stage 1 / mt | −0.024 | −0.007 | 0.124 | −0.010 | −0.016 | 0.007 | 0.517 | 1.000 |
| ME stage 2 / mt | **0.464** | 0.340 | 0.610 | 0.242 | 0.083 | 0.061 | 0.700 | **0.498** |
| TMPNN-D / mt | 0.166 | 0.095 | 0.155 | 0.179 | 0.040 | 0.018 | 0.601 | 0.650 |
| ESM-MSR capped / comb | 0.351 | 0.485 | 0.614 | 0.390 | 0.316 | 0.156 | **0.861** | 0.884 |
| ESM-MSR capped / mt | 0.351 | **0.513** | 0.644 | 0.410 | **0.326** | **0.206** | 0.860 | 0.593 |
| dc8 / comb (3 seeds) | 0.518 | 0.481 | **0.667** | **0.415** | 0.246 | 0.155 | 0.855 | 0.626 |
| dc8 / mt (3 seeds) | 0.373 | 0.502 | 0.641 | 0.401 | 0.307 | 0.192 | 0.855 | 0.543 |

Additive heads score exactly 0 / 0.5 on every residual and rank metric — the suite's own correctness check —
and `global_err` 1.105 for both baselines.

### 2.2 Test, epistasis hierarchy (12,455 doubles, 43 pairs, 4,236 flips)

| model / head | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| ME stage 2 / mt | **0.460** | 0.311 | 0.299 | 0.310 | 0.140 | 0.110 | 0.645 | **0.403** |
| TMPNN-D / mt | 0.037 | 0.035 | −0.068 | 0.218 | 0.019 | 0.011 | 0.591 | 0.600 |
| ESM-MSR capped / mt | 0.339 | 0.480 | 0.619 | 0.400 | 0.239 | 0.199 | 0.744 | 0.560 |
| dc8 / mt (3 seeds) | 0.342 | **0.511 ± 0.009** | **0.660 ± 0.017** | **0.429** | **0.272** | **0.210 ± 0.005** | **0.754** | 0.541 |
| dc8 / comb (3 seeds) | 0.337 | 0.489 | 0.627 | 0.417 | 0.250 | 0.173 | 0.751 | 0.696 |

Every destabilized-backbone test row is a single, so these columns are identical whether or not those
libraries are kept (verified: max abs diff 0).

### 2.3 ddG ranking (mean per-protein Spearman)

| | ME st1 | ME st2 | TMPNN | TMPNN+D | ESM-MSR capped comb | dc8 comb (3 seeds) |
|---|---|---|---|---|---|---|
| val `spearman_all` | 0.751 | 0.727 | 0.785 | 0.782 | 0.854 | 0.854 ± 0.001 |
| test `spearman_all` | 0.771 | 0.762 | 0.794 | 0.783 | 0.843 | **0.849 ± 0.001** |
| test `spearman_doubles` | 0.617 | 0.660 | 0.673 | 0.672 | 0.763 | — |
| test `spearman_doubles_epi` (dddG ρ) | 0.020 | 0.434 | — | 0.181 | 0.554 | — |

dc8 vs capped: test combined +0.006, test MT +0.007, val WT −0.005 — a modest gain on test, level on val.
These dc8 numbers are **on measured rows only** (§6.3); with bounded rows they would read 0.858 and look like
+0.015.

### 2.4 External sets, DMS, Domainome (baselines = 3-seed means, ESM-MSR = capped)

- **12 external sets, mean ρ:** ESM-MSR combined **0.647**, MT 0.617, ME stage 1 0.615, ME stage 2 0.599,
  TMPNN 0.585. ESM-MSR combined is best on 9 of 12 (ME stage 1 on s2648, s8754, q3421).
- **7 DMS** (`spearman_all`): ESM-MSR leads DLG4 binding (WT 0.728), GB1 (combined 0.346) and MYO (combined
  0.639); ME's additive variants lead DLG4 abundance (stage 1+ 0.729 vs 0.726), GRB2 abundance (0.690 vs
  0.672), GRB2 binding (stage 2+ 0.635 vs 0.633) and ESTA (0.483 vs 0.479) — mostly by < 0.02. **Every
  model's additive variant beats its own direct multi-mutant prediction** on these assays.
- **dddG on DMS:** ME stage 2 and ESM-MSR MT/combined are comparable on DLG4 (≈ 0.45–0.47); ESM-MSR leads GRB2
  abundance (0.424 vs 0.328); TMPNN-D leads GB1 (0.469 vs ESM-MSR 0.316); ME stage 1 is reliably negative.
- **Domainome** (singles): ESM-MSR combined 0.542 ≈ WT 0.540 > ME stage 1 0.520 = ESM-MSR MT 0.520 > ME stage 2
  0.511 > TMPNN 0.450.

### 2.5 Mutant-background ablation (nobb)

Dropping the 3 train / 5 val mutant-background libraries (all `native_cond` singles; zero doubles) and
re-scoring on the identical wild-type-backbone test rows:

| head | Δ spearman_all | Δ spearman_doubles | Δ beyond ρ | Δ pair ρ | Δ flipacc |
|---|---|---|---|---|---|
| ME stage 2 / mt | **−0.017** | **−0.022** | +0.004 | +0.026 | +0.033 |
| ME stage 1 / mt | +0.001 | −0.000 | +0.006 | +0.006 | −0.004 |
| TMPNN / add | 0.000 | +0.005 | 0 | 0 | 0 |

Those 3,543 training rows buy ME stage 2 a little ddG ranking on wild-type backbones and nothing on the
interaction metrics. ThermoMPNN-D was not retrained: its training data is byte-identical in both conditions.

## 3. ProteinGym v1.3

213 DMS that every model scored (BRCA1, BRCA2, POLG_CXB3N and SCN5A excluded: > 1,800 residues, out of
memory for at least one model). **PG avg** = mean of the five category means (a plain mean over DMS within a
category, not ProteinGym's per-UniProt weighting); **DMS mean** = mean over the 213. Unmasked scoring,
ProteinGym's own AF2 structures.

| model | Activity (43) | Binding (13) | Expression (18) | Org. fitness (73) | Stability (66) | PG avg | DMS mean |
|---|---|---|---|---|---|---|---|
| **dc8 full, ε = 0.75** (3 seeds) | 0.439 | 0.364 | 0.515 | 0.410 | 0.858 | **0.517** | **0.560** |
| ESM-MSR capped, WT only, ε = 0.75 | 0.449 | 0.363 | 0.512 | 0.425 | 0.813 | 0.512 | 0.553 |
| dc8 full, ε = 1 (3 seeds) | 0.426 | 0.347 | 0.517 | 0.395 | 0.869 | 0.511 | 0.555 |
| ESM-MSR capped, full | 0.426 | 0.346 | 0.516 | 0.395 | **0.870** | 0.510 | 0.556 |
| ESM-MSR capped, WT only, ε = 1 | 0.436 | 0.349 | 0.515 | 0.412 | 0.826 | 0.508 | 0.550 |
| ESM-MSR capped, WT only, ε = 0.5 | 0.450 | 0.377 | 0.493 | 0.425 | 0.779 | 0.505 | 0.542 |
| SPURS single-additive | 0.384 | 0.334 | 0.490 | 0.368 | 0.837 | 0.483 | 0.525 |
| ESM-MSR capped, WT only, ε = 0.25 | 0.438 | 0.382 | 0.466 | 0.411 | 0.712 | 0.482 | 0.512 |
| MutateEverything single (3 seeds) | 0.371 | 0.336 | 0.472 | 0.352 | 0.808 | 0.468 | 0.506 |
| ESM-MSR, ε = 0 (no LoRA) | 0.417 | 0.374 | 0.439 | 0.391 | 0.621 | 0.448 | 0.470 |
| MutateEverything double (3 seeds) | 0.348 | 0.296 | 0.437 | 0.338 | 0.806 | 0.445 | 0.491 |
| ThermoMPNN (3 seeds) | 0.351 | 0.299 | 0.439 | 0.337 | 0.781 | 0.441 | 0.484 |
| SPURS multi | 0.326 | 0.267 | 0.410 | 0.320 | 0.808 | 0.426 | 0.477 |
| ProteinMPNN (3 decoding orders) | 0.343 | 0.323 | 0.437 | 0.310 | 0.627 | 0.408 | 0.426 |

- **dc8 ε = 1 matches capped** (paired Δ −0.0002); **dc8 ε = 0.75 beats capped** by +0.0047 (better on 133 of
  213 DMS, paired t = 2.83). Seed sd ≤ 0.002.
- **The ε trend:** Stability rises monotonically to ε = 1; Activity, Binding and Organismal fitness peak at
  ε = 0.25–0.75. Scaling the LoRA down trades stability for general fitness.
- **dc8 at ε = 0.5 and 0.25** were 177/213 complete (s3 ε = 0.5 and all three ε = 0.25) on 2026-10-21 and are
  left out above; on the 177 DMS they have, ε = 0.5 ≈ ε = 0.75 on PG avg and ε = 0.25 is lower.
- **MT-pass ablation** (`--mt_pass_adapter wt`, dc8 s1 ε = 1): the WT adapter scoring the mutant sequence
  costs −0.016 ρ on the first 24 DMS (worse on 18). Verified by hand on BCHB: unmasked, the WT adapter sees the
  mutant residue it is scoring and its LLR collapses toward zero (mean −0.85 vs −3.44 on the WT sequence).
  Still running on an RTX 6000 at ~4× the A4500 time.
- **ProteinMPNN test-time ensembling:** averaging three decoding orders +0.0003; a 0.2 Å backbone-noise
  ensemble of 8 +0.0017 on a 73-DMS pilot (50/73 up, t = 1.29). Not worth a full run.

## 4. Third-party stability predictors (`~/stability-baselines`)

Mean Spearman, positive = stabilizing. External sets and DMS pooled; Tsuboyama and Domainome are the mean of
per-protein ρ. "ESM-MSR-small" is the March reference model (crystal / AF2 wild-type structures).

| set | ESMtherm | DDGemb | DDMut | IFUM | GeoDDG-3D | ProSST (zero-shot) | ESM3ΔG | ESM-MSR-small |
|---|---|---|---|---|---|---|---|---|
| 12 external sets | 0.365 | 0.697 | 0.599 | 0.470 | **0.767** | 0.423 | 0.582 | 0.639 |
| Tsuboyama val | 0.691 | 0.610 | (26 %) | 0.746 / 0.778 native | 0.610 | 0.607 | 0.790 † | — |
| Tsuboyama test | 0.707 | 0.524 | (4 %) | 0.697 / 0.738 native | 0.557 | 0.594 | 0.744 † | — |
| Domainome | 0.201 | 0.351 | — | 0.464 | 0.413 | **0.519** | 0.512 | — |
| DLG4 abund. / bind. | 0.09 / 0.14 | 0.10 / 0.12 | — | 0.77 / 0.77 | 0.35 / 0.43 | **0.78 / 0.80** | 0.76 / 0.77 | — |
| GRB2 abund. / bind. | 0.48 / 0.47 | 0.46 / 0.47 | — | 0.65 / 0.57 | 0.60 / 0.48 | 0.71 / **0.70** | **0.75** / 0.65 | — |
| GB1 | 0.214 | 0.154 | — | 0.262 | 0.200 | 0.258 | **0.324** | — |

- **Read the external-set column with care:** S2648, S8754, Q3421, K3822 and Ssym overlap the training data of
  DDGemb, DDMut and GeoDDG, which is why GeoDDG (0.767) and DDGemb (0.697) lead there and fall behind on
  Tsuboyama and the DMS. **†** ESM3ΔG trains on Megascale, so its Tsuboyama numbers are not held out.
- **UniMutStab:** no released weights or inference code (pseudocode only); not run.
- **DDMut** Tsuboyama doubles were still being submitted to the server on 2026-10-21 (6,861 + 8,321 of
  ~20,151); its Tsuboyama column is partial.
- **AF2 vs crystal structures for ESM-MSR-small** (protein-level bootstrap, 2,000 resamples): mean Δ over the
  10 single-mutant sets +0.003 [−0.006, +0.010], over the 2 multi-mutant sets −0.016 [−0.040, +0.001]; only
  Ssym (−0.036) clearly moves. AF2 structures are a safe substitute.
- **Indexing audit:** all 944,018 unique variants over 22 datasets, 0 wild-type mismatches; a shifted-by-one
  negative control flags 94 %.

## 5. Where the code went

Committed to `devel-revision` (this consolidation and the 2026-10-16 merge `65e9067`):

- `inference_scripts/pgym_runner.py` — the one ProteinGym loop shared by every model; now records per-DMS
  time, peak VRAM, GPU and host.
- ProteinGym modes for `esm_msr_testing.py`, `proteinmpnn_testing.py` (fast scorer; optional decoding-order /
  backbone-noise ensemble: `--pgym_n_samples`, `--pgym_coord_noise`), `spurs_testing.py`,
  `thermompnnd_testing.py`.
- `esm_msr_testing.py`: Tsuboyama and DMS data read from the repo; reported stats exclude bounded rows;
  `--mt_pass_adapter wt`.

**Not in the repo** (left on the cluster, by the 2026-10-16 decision that baseline retraining, MutateEverything,
Slurm and cluster-specific scripts stay out):

- `~/esm-msr-devel/baselines/` — data preparation for the baselines (`prepare_oct06_data.py`, the data
  contract), ME/TMPNN training wrappers, `epi_baselines.py` / `epi_dms.py` (the epistasis suite applied to
  other models), `score_baselines.py`, `final_tables.py`, `restat_bounded.py`, `dms_source.py`, Slurm scripts.
  Its `README.md` documents it. **These AST-load functions out of `esm_msr_testing.py`** and resolve
  `epi_items` / `_by_head` from `devel-experimental`; a change to those names in the repo breaks them.
- `~/stability-baselines/` — adapters and harness for the third-party predictors (not a git repository).
- External repos: `~/software/MutateEverything-oct06`, `~/software/ThermoMPNN-D`, `~/software/SPURS`
  (conda env `spurs_higher`), ColabFold MSAs.
- `~/project_oct06_baselines/` — a symlink index over all of workstream A, with `README.md` and `RUNS.md`.

## 6. Data and scoring problems found along the way

These change numbers, so anything scored before the fix date is suspect.

1. **Broken GB1 copy (fixed 10-13).** `~/software/esm-msr/data/preprocessed/GB1_Wu_2016_binding_domain.csv`
   is missing V39A/C/D (73 of 76 singles), which drops 152 of GB1's 2,091 scorable doubles. The repo's
   `data/preprocessed` copy is complete (149,360 rows; same md5 locally and on the cluster). The bad file is
   quarantined as `*.BROKEN_MISSING_SINGLES_DO_NOT_USE`; GB1 was re-predicted for all ten models.
2. **`dG_wt` was wrong in the old code (fixed 10-20).** Averaging the `wt` rows matched the library's reference
   constant in only 7 of 412 libraries (median error 0.015, max 0.197 kcal/mol). `devel-revision` recovers it as
   `dG_ML − ddG_ML`. Re-scoring moved seed-mean epistasis cells by ≤ 0.011; `naive_rho`, `cell_rank` and
   `cell_flipacc` did not move at all (they do not use `dG_wt`).
3. **Bounded rows inflate ρ by ~0.01 (fixed 10-21).** `censor_range` (default on in `devel-revision`) brings in
   rows the assay only bounded (`<-1` / `>5`): 1.5 % of val, 6.4 % of test. They sit at the extremes and are easy
   to rank. Predict them, report without them.
4. **Validation population.** The baselines' prepared val table applies training rules (33,333 rows);
   ESM-MSR scores 34,957. Use `valfull` (`build_valfull.py`) for any cross-model val comparison; the filtered
   table is kept as `epi_val_filtered_DO_NOT_USE.csv`.
5. **Destabilized backbones in test.** 7 of 37 test libraries carry a background mutation, and only models
   trained with mutant-background rows have seen the pattern. The baseline ddG tables default to
   `--test_backbones wt_only` (40,866 rows); epistasis columns are unaffected (§2.2).
6. **Upstream ME flips the ground-truth sign** when more than half of `ddg` is negative (Ssym, GB1). All
   reported scoring reads `ddG_ML` from ESM-MSR's truth instead.
7. **Subset relations** used to align external sets: S461 ⊂ S669, K2369 ⊂ K3822, PTMUL-D ⊂ PTMUL.
8. **The link is training-only.** `esm_msr_testing.py` writes **latent** ddG for link-trained checkpoints
   (dc8); `epi_baselines.msr_link` rebuilds the link from the checkpoint for the observed-scale epistasis
   metrics. Rank metrics within a DMS are unaffected.
9. **`splits_oct06_capped.pkl`.** The cluster checkout switched to `origin/devel-revision`, which did not track
   it, and the file disappeared mid-campaign; it was restored from `devel-experimental` (blob `10c6aec1`). Local
   `devel-revision` tracks the identical blob.

## 7. Open items as of 2026-10-21

- dc8 ProteinGym ε = 0.5 (s3) and ε = 0.25 (s1–s3): 177/213, expected to finish late 10-21. Then a combined
  dc8 per-DMS CSV still has to be written into `outputs/pgym/`.
- MT-pass ablation job 386255: ~6 days on gpu3 at the current speed; moving it to an A4500 (~33 h) needs a
  `scancel` decision.
- DDMut Tsuboyama doubles: still submitting to the server.
- The cluster checkout `~/esm-msr-devel` is one commit ahead of `origin/devel-revision` (now consolidated
  here) with uncommitted edits (also consolidated here) and staged `data/preprocessed/*_mapped.csv` changes that
  only rewrite structure paths to cluster locations — do not commit those.
- Cleanup awaiting approval: `~/esm-msr-rev` (old work clone) and `outputs/pgym_verify_rev/`.
