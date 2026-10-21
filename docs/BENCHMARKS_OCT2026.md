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
| F. **All-by-all fill-in** (§2): every third-party model on the capped Tsuboyama split and ProteinGym; SPURS, ProteinMPNN and ESM3-small-open (ε = 0) on the thermostability gamut | 10-21 → | this consolidation | queued on gpu4; ledger in `GPU4_INFERENCE_LEDGER.md` |

Common ground for A–C: the ESM-MSR reference checkpoint is `LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`
("capped", one seed); dc8 is the three-seed `dc8-floorlink-canon` series (same split, same LoRA config;
`val_rho_combined_avg` 0.840 / 0.839 / 0.841).

## 2. All-by-all: every model on every dataset

One Spearman ρ per model and dataset, positive = stabilizing / fitter for every model. External sets and DMS are a
pooled ρ per set; Tsuboyama and Domainome are the mean of per-protein ρ (Tsuboyama on measured rows only); ProteinGym
as in §4. Multi-seed rows are seed means (per-seed values and sds are in the CSV). Regenerate with
`analysis_scripts/benchmark_matrix.py` (run on the cluster); the long table is
[`benchmarks/all_by_all_long.csv`](benchmarks/all_by_all_long.csv) (model, dataset, rho, sd, n_seeds, source).

**Legend.** **†** the test set overlaps that model's documented training data (DDGemb trained on S2450 ⊂ S2648;
DDMut on S9028 ⊃ S2648 + FireProtDB; GeoDDG on S8754 ⊃ S2648, and S783 only has a leakage-excluded column upstream;
the Megascale-trained models on any Tsuboyama split). The list follows each paper's stated training sets and is
not exhaustive. **‡** the model was trained on Megascale (Tsuboyama), so none of its Tsuboyama numbers are held
out. **\*** partial coverage (DDMut: S669 98 %, S461 97 %, Tsuboyama val 26 %, test 4 %). **q** queued on gpu4 on
2026-10-21 (jobs 386274–386282, see [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md)). **–** not run, with the
reason in [`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md).

Which ESM-MSR / baseline numbers are comparable to which: the **capped** Tsuboyama columns are the split every
October model was trained on (`splits_oct06_capped.pkl`). The **hyperopt** columns are the older split
(`hyperopt_splits.pkl`) the third-party harness was first run on; it shares only 4 of 23 val and 9 of 33 test
libraries with capped, so the two are not comparable. The queued runs put every third-party model on capped.

### 2.1 Thermostability: the 12 external sets

| model | s571 | s783 | s2648 | s8754 | s669 | s461 | ssym | q3421 | k3822 | k2369 | ptmul | ptmuld | mean of 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR capped (combined) | 0.456 | 0.717 | 0.665 | 0.615 | 0.565 | 0.756 | 0.751 | 0.617 | 0.715 | 0.714 | 0.595 | 0.601 | 0.647 |
| ESM-MSR capped, WT head | 0.442 | 0.703 | 0.657 | 0.606 | 0.547 | 0.748 | 0.752 | 0.613 | 0.714 | 0.711 | 0.568 | 0.586 | 0.637 |
| ESM-MSR capped, MT head | 0.451 | 0.697 | 0.634 | 0.596 | 0.561 | 0.725 | 0.714 | 0.589 | 0.681 | 0.682 | 0.545 | 0.533 | 0.617 |
| ESM-MSR dc8 (combined, 3 seeds) | 0.445 | 0.725 | 0.677 | 0.625 | 0.560 | 0.753 | 0.773 | 0.626 | 0.721 | 0.722 | 0.594 | 0.597 | 0.651 |
| ESM-MSR dc8, WT head | 0.431 | 0.713 | 0.675 | 0.618 | 0.551 | 0.753 | 0.776 | 0.623 | 0.720 | 0.717 | 0.580 | 0.584 | 0.645 |
| ESM-MSR dc8, MT head | 0.438 | 0.703 | 0.639 | 0.603 | 0.539 | 0.709 | 0.737 | 0.595 | 0.688 | 0.694 | 0.528 | 0.524 | 0.616 |
| MutateEverything st1 (3 seeds) | 0.439 | 0.706 | 0.701 | 0.623 | 0.559 | 0.721 | 0.674 | 0.638 | 0.696 | 0.674 | 0.472 | 0.480 | 0.615 |
| MutateEverything st2 (3 seeds) | 0.424 | 0.687 | 0.669 | 0.598 | 0.541 | 0.709 | 0.632 | 0.615 | 0.673 | 0.657 | 0.491 | 0.490 | 0.599 |
| ThermoMPNN (additive, 3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.433 | 0.499 | 0.585 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.427 | 0.502 | 0.584 |
| SPURS single-additive ‡ | q | q | q | q | q | q | q | q | q | q | q | q | q |
| SPURS multi ‡ | q | q | q | q | q | q | q | q | q | q | q | q | q |
| ESMtherm ‡ | 0.349 | 0.434 | 0.371 | 0.347 | 0.414 | 0.474 | 0.380 | 0.348 | 0.366 | 0.393 | 0.235 | 0.270 | 0.365 |
| DDGemb (server) | 0.484 | 0.689 | 0.895† | 0.684† | 0.570 | 0.654 | 0.891 | 0.752 | 0.745 | 0.702 | 0.616 | 0.678 | 0.697 |
| DDMut (server) | 0.350 | 0.687 | 0.731† | 0.638† | 0.402* | 0.534* | 0.363 | 0.653† | 0.709† | 0.681† | 0.725 | 0.718 | – |
| IFUM, WT backbone ‡ | 0.376 | 0.498 | 0.485 | 0.477 | 0.434 | 0.531 | 0.574 | 0.431 | 0.536 | 0.547 | 0.350 | 0.405 | 0.470 |
| IFUM, ESMFold per variant ‡ | 0.385 | 0.471 | 0.484 | 0.451 | 0.409 | 0.542 | 0.580 | 0.433 | 0.508 | 0.512 | 0.391 | 0.443 | 0.467 |
| GeoDDG-3D (FASPR) | 0.526 | 0.865† | 0.892† | 0.841† | 0.615 | 0.664 | 0.914 | 0.822 | 0.852 | 0.831 | 0.717 | 0.669 | 0.767 |
| ESM3ΔG ‡ | 0.511 | 0.653 | 0.642 | 0.542 | 0.564 | 0.681 | 0.735 | 0.572 | 0.589 | 0.598 | 0.409 | 0.482 | 0.582 |
| ProSST (zero-shot) | 0.310 | 0.564 | 0.436 | 0.428 | 0.369 | 0.561 | 0.519 | 0.416 | 0.476 | 0.497 | 0.182 | 0.317 | 0.423 |
| ProteinMPNN (zero-shot) | q | q | q | q | q | q | q | q | q | q | q | q | q |
| ESM3-small-open, ε = 0 (zero-shot) | q | q | q | q | q | q | q | q | q | q | q | q | q |
| ESM3-large (Forge, zero-shot) | 0.220 | 0.353 | 0.267 | 0.291 | 0.272 | – | 0.539 | 0.306 | 0.318 | – | 0.146 | – | – |
| ESM3-medium (Forge, zero-shot) | 0.332 | 0.552 | 0.439 | 0.438 | 0.342 | – | 0.633 | 0.450 | 0.483 | – | 0.295 | – | – |
| ESM-MSR-small (March reference) | 0.447 | 0.714 | 0.677 | 0.614 | 0.542 | 0.743 | 0.781 | 0.621 | 0.701 | 0.704 | 0.543 | 0.577 | 0.639 |

- **On all 12 sets GeoDDG (0.767) and DDGemb (0.697) lead, then ESM-MSR dc8 (0.651) and capped (0.647).** Most of the
  gap sits on the flagged sets: GeoDDG scores 0.84–0.89 on S783, S2648 and S8754, which it trained on or next to.
- **On the six sets with no documented overlap for any model** (S571, S669, S461, Ssym, PTMUL, PTMUL-D) the order is
  GeoDDG 0.684, DDGemb 0.649, ESM-MSR capped 0.621, dc8 0.620, ESM-MSR-small 0.605, ESM3ΔG 0.564, ME st1 0.557,
  ThermoMPNN 0.532, IFUM 0.445–0.458, ProSST 0.376, ESMtherm 0.354. GeoDDG and DDGemb are still ahead; both were
  trained on curated literature ΔΔG (ProTherm-derived), the same source as these sets, so homology to the training
  proteins is likely even where the exact variants are not shared.
- **Among the models trained on Tsuboyama/Megascale, ESM-MSR leads 9 of 12 sets.** ME stage 1 leads S2648 (0.701 vs
  0.677) and Q3421 (0.638 vs 0.626), and ESM3ΔG leads S571 (0.511 vs 0.456).

### 2.2 Tsuboyama

| model | capped val | capped test | hyperopt val | hyperopt test |
|---|---|---|---|---|
| ESM-MSR capped (combined) | 0.854 | 0.843 | – | – |
| ESM-MSR capped, WT head | 0.841 | 0.824 | – | – |
| ESM-MSR capped, MT head | 0.817 | 0.813 | – | – |
| ESM-MSR dc8 (combined, 3 seeds) | 0.854 | 0.849 | – | – |
| ESM-MSR dc8, WT head | 0.836 | 0.827 | – | – |
| ESM-MSR dc8, MT head | 0.825 | 0.820 | – | – |
| MutateEverything st1 (3 seeds) | 0.751 | 0.771 | – | – |
| MutateEverything st2 (3 seeds) | 0.727 | 0.762 | – | – |
| ThermoMPNN (additive, 3 seeds) | 0.785 | 0.794 | – | – |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.782 | 0.783 | – | – |
| SPURS single-additive ‡ | q | q | – | – |
| SPURS multi ‡ | q | q | – | – |
| ESMtherm ‡ | q | q | 0.691† | 0.707† |
| DDGemb (server) | – | – | 0.611 | 0.524 |
| DDMut (server) | – | – | 0.542* | 0.582* |
| IFUM, WT backbone ‡ | q | q | 0.746† | 0.697† |
| IFUM, ESMFold per variant ‡ | – | – | 0.778† | 0.738† |
| GeoDDG-3D (FASPR) | q | q | 0.610 | 0.557 |
| ESM3ΔG ‡ | q | q | 0.790† | 0.744† |
| ProSST (zero-shot) | q | q | 0.607 | 0.594 |
| ProteinMPNN (zero-shot) | q | q | – | – |
| ESM3-small-open, ε = 0 (zero-shot) | q | q | – | – |

The published-paper evaluation (below) has ESM-MSR, ProteinMPNN, Rosetta and ESM3 on the hyperopt split. SPURS and
ProteinMPNN cannot represent a mutant background (they read the sequence from the wild-type PDB), so their capped
numbers will be on the wild-type-background libraries only; the cell will say so when it lands.

### 2.3 DMS and Domainome

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR capped (combined) | 0.726 | 0.725 | 0.673 | 0.619 | 0.639 | 0.479 | 0.346 | 0.542 |
| ESM-MSR capped, WT head | 0.723 | 0.728 | 0.669 | 0.633 | 0.637 | 0.466 | 0.338 | 0.540 |
| ESM-MSR capped, MT head | 0.705 | 0.696 | 0.627 | 0.560 | 0.622 | 0.467 | 0.302 | 0.520 |
| ESM-MSR dc8 (combined, 3 seeds) | 0.745 | 0.742 | 0.682 | 0.636 | 0.639 | 0.481 | 0.337 | 0.543 |
| ESM-MSR dc8, WT head | 0.749 | 0.752 | 0.685 | 0.656 | **0.641** | 0.478 | **0.365** | **0.548** |
| ESM-MSR dc8, MT head | 0.718 | 0.707 | 0.632 | 0.570 | 0.616 | 0.457 | 0.243 | 0.509 |
| MutateEverything st1 (3 seeds) | 0.707 | 0.681 | 0.646 | 0.581 | 0.580 | **0.483** | 0.296 | 0.520 |
| MutateEverything st1, additive | 0.729 | 0.714 | 0.690 | 0.615 | 0.603 | 0.483 | 0.287 | – |
| MutateEverything st2 (3 seeds) | 0.655 | 0.644 | 0.624 | 0.601 | 0.566 | 0.441 | 0.284 | 0.511 |
| MutateEverything st2, additive | 0.710 | 0.695 | 0.675 | 0.635 | 0.597 | 0.441 | 0.285 | – |
| ThermoMPNN (additive, 3 seeds) | 0.626 | 0.632 | 0.613 | 0.569 | 0.570 | 0.405 | 0.319 | 0.450 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.574 | 0.588 | 0.565 | 0.561 | 0.540 | 0.405 | 0.318 | – |
| SPURS single-additive ‡ | q | q | q | q | q | q | q | q |
| SPURS multi ‡ | q | q | q | q | q | q | q | q |
| ESMtherm ‡ | 0.093 | 0.136 | 0.484 | 0.466 | 0.229 | −0.000 | 0.214 | 0.201 |
| DDGemb (server) | 0.103 | 0.123 | 0.457 | 0.469 | 0.475 | 0.188 | 0.154 | 0.351 |
| IFUM, WT backbone ‡ | 0.766 | 0.768 | 0.648 | 0.565 | 0.507 | 0.190 | 0.262 | 0.464 |
| IFUM, ESMFold per variant ‡ | 0.762 | 0.747 | 0.697 | 0.575 | 0.523 | 0.217 | 0.284 | 0.473 |
| GeoDDG-3D (FASPR) | 0.351 | 0.431 | 0.604 | 0.481 | 0.395 | 0.240 | 0.200 | 0.413 |
| ESM3ΔG ‡ | 0.763 | 0.766 | **0.749** | 0.646 | 0.594 | 0.280 | 0.324 | 0.512 |
| ProSST (zero-shot) | **0.776** | **0.796** | 0.711 | **0.702** | 0.538 | 0.356 | 0.258 | 0.519 |
| ProteinMPNN (zero-shot) | q | q | q | q | q | q | q | q |
| ESM3-small-open, ε = 0 (zero-shot) | q | q | q | q | q | q | q | q |
| ESM3-large (Forge, zero-shot) | 0.624 | 0.606 | 0.557 | 0.582 | 0.458 | 0.432 | – | – |

- **Mean over the 7 DMS:** ESM-MSR dc8 WT 0.618, dc8 combined 0.609, capped 0.601, ProSST 0.591, ME st1 additive
  0.589, ESM3ΔG 0.589, IFUM 0.53–0.54, ThermoMPNN 0.533, GeoDDG 0.386, DDGemb 0.281, ESMtherm 0.232.
- **The literature-ΔΔG models invert here:** GeoDDG and DDGemb, which lead the external sets, are near the bottom on
  the deep-mutational scans; ProSST, a zero-shot model, leads three of the four Faure assays.
- **Domainome:** dc8 WT 0.548, ESM-MSR ≈ 0.54, then ME st1 0.520, ProSST 0.519, ESM3ΔG 0.512.

### 2.4 ProteinGym (213 DMS)

| model | PG avg | Stability | DMS mean |
|---|---|---|---|
| ESM-MSR dc8 (combined, 3 seeds) | 0.511 | 0.869 | 0.555 |
| ESM-MSR capped (combined) | 0.510 | **0.870** | **0.556** |
| ESM-MSR capped, WT head | 0.508 | 0.826 | 0.550 |
| SPURS single-additive ‡ | 0.483 | 0.837 | 0.525 |
| MutateEverything st1 (3 seeds) | 0.468 | 0.808 | 0.506 |
| ESM3-small-open, ε = 0 (WT pass only) | 0.448 | 0.621 | 0.470 |
| MutateEverything st2 (3 seeds) | 0.445 | 0.806 | 0.491 |
| ThermoMPNN (additive, 3 seeds) | 0.441 | 0.781 | 0.484 |
| SPURS multi ‡ | 0.426 | 0.808 | 0.477 |
| ProteinMPNN (zero-shot) | 0.408 | 0.627 | 0.426 |
| ESMtherm ‡ | q | q | q |
| IFUM, WT backbone ‡ | q | q | q |
| GeoDDG-3D (FASPR) | q | q | q |
| ESM3ΔG ‡ | q | q | q |
| ProSST (zero-shot) | q | q | q |

The ε sweep and per-category breakdown are in §4. DDGemb and DDMut cannot be run on ProteinGym (web servers;
see the outstanding log).

### 2.5 Published-paper evaluation (reference; earlier models and row sets)

From `analysis_notebooks/bootstrap_external_results/rho_all_models_replicate1.csv`: the models of the
published comparison, trained on the hyperopt split. Its row sets are de-duplicated (S8754 → 6,411 rows) and it has
neither S669 nor K3822 (it uses their subsets S461 and K2369), so compare within this table only. It adds Rosetta and
the ESM3 family, which the October runs do not have.

| dataset | ESM-MSR (published) | ProteinMPNN | Rosetta cart. ΔΔG | ESM3-small-open | ESM3-small | ESM3-medium | ESM3-large | ME (hyperopt) | ME additive (hyperopt) | ThermoMPNN (hyperopt) | ThermoMPNN-D (hyperopt) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ssym | 0.750 | 0.694 | 0.621 | 0.584 | 0.594 | 0.617 | 0.498 | 0.669 | 0.722 | 0.727 | – |
| S461 | 0.750 | 0.441 | 0.617 | 0.402 | 0.445 | 0.422 | 0.371 | 0.713 | 0.720 | 0.647 | – |
| S783 | 0.725 | 0.559 | 0.648 | 0.480 | 0.526 | 0.510 | 0.330 | 0.690 | 0.717 | 0.691 | – |
| K2369 | 0.705 | 0.435 | 0.618 | 0.438 | 0.488 | 0.477 | 0.318 | 0.671 | 0.689 | 0.659 | – |
| S8754 | 0.617 | 0.437 | 0.571 | 0.379 | 0.410 | 0.409 | 0.268 | 0.630 | 0.657 | 0.611 | – |
| S2648 | 0.665 | 0.508 | 0.606 | 0.374 | 0.432 | 0.407 | 0.261 | 0.679 | 0.719 | 0.658 | – |
| Q3421 | 0.613 | 0.495 | 0.554 | 0.377 | 0.429 | 0.414 | 0.282 | 0.622 | 0.654 | 0.604 | – |
| PTMUL-D | 0.587 | 0.545 | 0.472 | 0.299 | 0.442 | 0.339 | 0.145 | 0.493 | 0.446 | 0.502 | 0.563 |
| PTMUL | 0.545 | 0.377 | 0.540 | 0.191 | 0.316 | 0.250 | 0.043 | 0.539 | 0.470 | 0.437 | 0.452 |
| S571 | 0.451 | 0.336 | 0.357 | 0.274 | 0.249 | 0.341 | 0.196 | 0.440 | 0.465 | 0.420 | – |
| mean of 10 external | 0.641 | 0.483 | 0.561 | 0.380 | 0.433 | 0.418 | 0.271 | 0.615 | 0.626 | 0.595 | 0.507 |
| Megascale val (per-domain avg.) | 0.835 | 0.595 | 0.666 | 0.561 | 0.592 | 0.583 | 0.516 | 0.744 | 0.744 | 0.768 | 0.763 |
| Megascale test (per-domain avg.) | 0.804 | 0.529 | 0.634 | 0.551 | 0.606 | 0.590 | 0.553 | 0.673 | 0.678 | 0.740 | 0.732 |
| DLG4 abundance | 0.721 | 0.621 | – | 0.678 | 0.745 | 0.723 | 0.583 | 0.645 | 0.702 | 0.629 | 0.530 |
| DLG4 binding | 0.739 | 0.615 | – | 0.731 | 0.755 | 0.762 | 0.562 | 0.638 | 0.722 | 0.651 | 0.545 |
| GRB2 abundance | 0.734 | 0.668 | – | 0.613 | 0.639 | 0.617 | 0.549 | 0.618 | 0.711 | 0.649 | 0.640 |
| GRB2 binding | 0.623 | 0.538 | – | 0.709 | 0.717 | 0.680 | 0.520 | 0.593 | 0.571 | 0.592 | 0.587 |
| MYO display | 0.636 | 0.499 | – | 0.527 | 0.557 | 0.521 | 0.463 | 0.562 | 0.586 | 0.577 | 0.528 |
| ESTA ΔT50 | 0.517 | 0.546 | – | 0.484 | 0.486 | 0.497 | 0.413 | 0.465 | 0.501 | 0.470 | – |
| Domainome (per-domain avg.) | 0.548 | 0.548 | – | 0.533 | 0.548 | 0.532 | 0.508 | 0.516 | 0.523 | 0.462 | – |

## 3. Retrained baselines vs ESM-MSR on the Tsuboyama split

**Rule throughout: neither baseline was modified, only its data layer.** ME follows the upstream README
recipe (stage 1 = singles, 20 epochs; stage 2 = doubles, 100 epochs from stage 1). ThermoMPNN is two
upstream configs, i.e. two models: `train_single.yaml` (singles) and `train_epistatic.yaml` (ThermoMPNN-D,
siamese, doubles only). The baselines' additive head is their own singles summed; their `mt` head is ME's
direct double head or ThermoMPNN-D. `comb ≡ mt` for both; `ctx` does not transfer (no separable adapter).

### 3.1 Validation, epistasis hierarchy (valfull: 34,957 rows, 8,494 doubles, 24 pairs, 2,509 flips)

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

### 3.2 Test, epistasis hierarchy (12,455 doubles, 43 pairs, 4,236 flips)

| model / head | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| ME stage 2 / mt | **0.460** | 0.311 | 0.299 | 0.310 | 0.140 | 0.110 | 0.645 | **0.403** |
| TMPNN-D / mt | 0.037 | 0.035 | −0.068 | 0.218 | 0.019 | 0.011 | 0.591 | 0.600 |
| ESM-MSR capped / mt | 0.339 | 0.480 | 0.619 | 0.400 | 0.239 | 0.199 | 0.744 | 0.560 |
| dc8 / mt (3 seeds) | 0.342 | **0.511 ± 0.009** | **0.660 ± 0.017** | **0.429** | **0.272** | **0.210 ± 0.005** | **0.754** | 0.541 |
| dc8 / comb (3 seeds) | 0.337 | 0.489 | 0.627 | 0.417 | 0.250 | 0.173 | 0.751 | 0.696 |

Every destabilized-backbone test row is a single, so these columns are identical whether or not those
libraries are kept (verified: max abs diff 0).

### 3.3 ddG ranking (mean per-protein Spearman)

| | ME st1 | ME st2 | TMPNN | TMPNN+D | ESM-MSR capped comb | dc8 comb (3 seeds) |
|---|---|---|---|---|---|---|
| val `spearman_all` | 0.751 | 0.727 | 0.785 | 0.782 | 0.854 | 0.854 ± 0.001 |
| test `spearman_all` | 0.771 | 0.762 | 0.794 | 0.783 | 0.843 | **0.849 ± 0.001** |
| test `spearman_doubles` | 0.617 | 0.660 | 0.673 | 0.672 | 0.763 | — |
| test `spearman_doubles_epi` (dddG ρ) | 0.020 | 0.434 | — | 0.181 | 0.554 | — |

dc8 vs capped: test combined +0.006, test MT +0.007, val WT −0.005 — a modest gain on test, level on val.
These dc8 numbers are **on measured rows only** (§7.3); with bounded rows they would read 0.858 and look like
+0.015.

### 3.4 External sets, DMS, Domainome (baselines = 3-seed means, ESM-MSR = capped)

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

### 3.5 Mutant-background ablation (nobb)

Dropping the 3 train / 5 val mutant-background libraries (all `native_cond` singles; zero doubles) and
re-scoring on the identical wild-type-backbone test rows:

| head | Δ spearman_all | Δ spearman_doubles | Δ beyond ρ | Δ pair ρ | Δ flipacc |
|---|---|---|---|---|---|
| ME stage 2 / mt | **−0.017** | **−0.022** | +0.004 | +0.026 | +0.033 |
| ME stage 1 / mt | +0.001 | −0.000 | +0.006 | +0.006 | −0.004 |
| TMPNN / add | 0.000 | +0.005 | 0 | 0 | 0 |

Those 3,543 training rows buy ME stage 2 a little ddG ranking on wild-type backbones and nothing on the
interaction metrics. ThermoMPNN-D was not retrained: its training data is byte-identical in both conditions.

## 4. ProteinGym v1.3

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

## 5. Third-party stability predictors (`~/stability-baselines`)

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

## 6. Where the code went

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

## 7. Data and scoring problems found along the way

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
   `--test_backbones wt_only` (40,866 rows); epistasis columns are unaffected (§3.2).
6. **Upstream ME flips the ground-truth sign** when more than half of `ddg` is negative (Ssym, GB1). All
   reported scoring reads `ddG_ML` from ESM-MSR's truth instead.
7. **Subset relations** used to align external sets: S461 ⊂ S669, K2369 ⊂ K3822, PTMUL-D ⊂ PTMUL.
8. **The link is training-only.** `esm_msr_testing.py` writes **latent** ddG for link-trained checkpoints
   (dc8); `epi_baselines.msr_link` rebuilds the link from the checkpoint for the observed-scale epistasis
   metrics. Rank metrics within a DMS are unaffected.
9. **`splits_oct06_capped.pkl`.** The cluster checkout switched to `origin/devel-revision`, which did not track
   it, and the file disappeared mid-campaign; it was restored from `devel-experimental` (blob `10c6aec1`). Local
   `devel-revision` tracks the identical blob.

## 8. Open items

Moved to [`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md), which tracks what is queued, blocked or
incomplete; where each inference ran is in [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md).
