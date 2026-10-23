# ESM-MSR benchmark results, October 2026

**Every comparison of ESM-MSR against another model.** ESM-MSR's own training history (the `o6`, link and `dc`
series) is in [`EXPERIMENTS.md`](EXPERIMENTS.md); this document is the results.

**The two ESM-MSR checkpoints compared throughout.** "capped" is
`LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`, one seed. "dc8" is the three-seed
`esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}` series — same split, same LoRA config,
`val_rho_combined_avg` 0.840 / 0.839 / 0.841. Both were trained on `splits_oct06_capped.pkl`, as were the
retrained baselines, so all of them are scored on identical rows.

**Metric conventions.** One Spearman ρ per model and dataset, positive = stabilizing / fitter for every model.
External sets and DMS are a pooled ρ per set; Tsuboyama and Domainome are the mean of per-protein ρ, on measured
rows only; ProteinGym is per-DMS ρ summarised as in §3. Multi-seed rows are seed means.

**Regenerating.** `analysis_scripts/benchmark_matrix.py collect|tables`, run on the cluster (it reads the result
trees in place). The long table is [`benchmarks/all_by_all_long.csv`](benchmarks/all_by_all_long.csv)
(model, dataset, rho, sd, n_seeds, source). Companion documents: what is still outstanding is in
[`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md); which hardware each run used, and the inference-speed
findings, in [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md).

## 1. All-by-all: every model on every dataset

**Legend.** **†** the test set overlaps that model's documented training data (DDGemb trained on S2450 ⊂ S2648;
DDMut on S9028 ⊃ S2648 + FireProtDB; GeoDDG on S8754 ⊃ S2648, and S783 only has a leakage-excluded column
upstream; the Megascale-trained models on any Tsuboyama split). The list follows each paper's stated training
sets and is not exhaustive. **‡** the model was trained on Megascale (Tsuboyama), so none of its Tsuboyama
numbers are held out. **\*** partial coverage (DDMut: S669 98 %, S461 97 %, Tsuboyama val 26 %, test 4 %).
**q** run in progress. **–** not run; the reason is in `OUTSTANDING_EXPERIMENTS.md`. **Bold** marks the best
value in each dataset column, taken over the cells that are a like-for-like comparison — flagged (†, ‡) and
partial (\*) cells are excluded from that choice, and the aggregate column is left unbolded because it averages
over sets that are flagged for some models and not others.

ESM-MSR's **MT head is not listed here**: it is an internal head, not a deployable model, and it never wins a
column. Its WT / MT / combined decomposition is what §2 is about, so it is kept there.

**Which Tsuboyama columns are comparable.** The **capped** columns are the split every October model was trained
on. The **hyperopt** columns are the older split the third-party harness was first run on; it shares only 4 of 23
val and 9 of 33 test libraries with capped, so the two are not comparable with each other.

### 1.1 Thermostability: the 12 external sets

| model | s571 | s783 | s2648 | s8754 | s669 | s461 | ssym | q3421 | k3822 | k2369 | ptmul | ptmuld | mean of 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR capped (combined) | 0.456 | 0.717 | 0.665 | 0.615 | 0.565 | **0.756** | 0.751 | 0.617 | 0.715 | 0.714 | 0.595 | 0.601 | 0.647 |
| ESM-MSR capped, WT head | 0.442 | 0.703 | 0.657 | 0.606 | 0.547 | 0.748 | 0.752 | 0.613 | 0.714 | 0.711 | 0.568 | 0.586 | 0.637 |
| ESM-MSR dc8 (combined, 3 seeds) | 0.445 | **0.725** | 0.677 | 0.625 | 0.560 | 0.753 | 0.773 | 0.626 | 0.721 | 0.722 | 0.594 | 0.597 | 0.651 |
| ESM-MSR dc8, WT head | 0.431 | 0.713 | 0.675 | 0.618 | 0.551 | 0.753 | 0.776 | 0.623 | 0.720 | 0.717 | 0.580 | 0.584 | 0.645 |
| MutateEverything st1 (3 seeds) | 0.439 | 0.706 | **0.701** | 0.623 | 0.559 | 0.721 | 0.674 | 0.638 | 0.696 | 0.674 | 0.472 | 0.480 | 0.615 |
| MutateEverything st2 (3 seeds) | 0.424 | 0.687 | 0.669 | 0.598 | 0.541 | 0.709 | 0.632 | 0.615 | 0.673 | 0.657 | 0.491 | 0.490 | 0.599 |
| ThermoMPNN (additive, 3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.433 | 0.499 | 0.585 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.427 | 0.502 | 0.584 |
| SPURS single-additive ‡ | 0.468 | 0.712 | 0.682 | **0.626** | 0.502 | 0.654 | 0.784 | 0.614 | 0.681 | 0.698 | 0.399 | 0.495 | 0.610 |
| SPURS multi ‡ | 0.468 | 0.712 | 0.682 | **0.626** | 0.502 | 0.654 | 0.784 | 0.614 | 0.681 | 0.698 | 0.359 | 0.466 | 0.604 |
| ESMtherm ‡ | 0.349 | 0.434 | 0.371 | 0.347 | 0.414 | 0.474 | 0.380 | 0.348 | 0.366 | 0.393 | 0.235 | 0.270 | 0.365 |
| DDGemb (server) | 0.484 | 0.689 | 0.895† | 0.684† | 0.570 | 0.654 | 0.891 | 0.752 | 0.745 | 0.702 | 0.616 | 0.678 | 0.697 |
| DDMut (server) | 0.350 | 0.687 | 0.731† | 0.638† | 0.402* | 0.534* | 0.363 | 0.653† | 0.709† | 0.681† | **0.725** | **0.718** | – |
| IFUM, WT backbone ‡ | 0.376 | 0.498 | 0.485 | 0.477 | 0.434 | 0.531 | 0.574 | 0.431 | 0.536 | 0.547 | 0.350 | 0.405 | 0.470 |
| IFUM, ESMFold per variant ‡ | 0.385 | 0.471 | 0.484 | 0.451 | 0.409 | 0.542 | 0.580 | 0.433 | 0.508 | 0.512 | 0.391 | 0.443 | 0.467 |
| GeoDDG-3D (FASPR) | **0.526** | 0.865† | 0.892† | 0.841† | **0.615** | 0.664 | **0.914** | **0.822** | **0.852** | **0.831** | 0.717 | 0.669 | 0.767 |
| ESM3ΔG ‡ | 0.511 | 0.653 | 0.642 | 0.542 | 0.564 | 0.681 | 0.735 | 0.572 | 0.589 | 0.598 | 0.409 | 0.482 | 0.582 |
| ProSST (zero-shot) | 0.310 | 0.564 | 0.436 | 0.428 | 0.369 | 0.561 | 0.519 | 0.416 | 0.476 | 0.497 | 0.182 | 0.317 | 0.423 |
| ProteinMPNN (zero-shot) | 0.324 | 0.561 | 0.506 | 0.438 | 0.329 | 0.442 | 0.696 | 0.494 | 0.420 | 0.436 | 0.382 | 0.546 | 0.464 |
| ESM3-small-open, ε = 0 (zero-shot) | 0.268 | 0.487 | 0.379 | 0.374 | 0.299 | 0.404 | 0.583 | 0.382 | 0.423 | 0.442 | 0.190 | 0.304 | 0.378 |
| ESM3-large (Forge, zero-shot) | 0.220 | 0.353 | 0.267 | 0.291 | 0.272 | – | 0.539 | 0.306 | 0.318 | – | 0.146 | – | – |
| ESM3-medium (Forge, zero-shot) | 0.332 | 0.552 | 0.439 | 0.438 | 0.342 | – | 0.633 | 0.450 | 0.483 | – | 0.295 | – | – |
| ESM-MSR-small (March reference) | 0.447 | 0.714 | 0.677 | 0.614 | 0.542 | 0.743 | 0.781 | 0.621 | 0.701 | 0.704 | 0.543 | 0.577 | 0.639 |

- **On all 12 sets GeoDDG (0.767) and DDGemb (0.697) lead, then ESM-MSR dc8 (0.651) and capped (0.647).** Most of
  the gap sits on the flagged sets: GeoDDG scores 0.84–0.89 on S783, S2648 and S8754, which it trained on or next
  to.
- **On the six sets with no documented overlap for any model** (S571, S669, S461, Ssym, PTMUL, PTMUL-D): GeoDDG
  0.684, DDGemb 0.649, ESM-MSR capped 0.621, dc8 0.620, ESM-MSR-small 0.605, ESM3ΔG 0.564, ME st1 0.557, SPURS
  0.550, ThermoMPNN 0.532, IFUM 0.445–0.458, ProteinMPNN 0.453, ProSST 0.376, ESMtherm 0.354, ESM3-small-open
  0.341. GeoDDG and DDGemb are still ahead; both were trained on curated literature ΔΔG (ProTherm-derived), the
  same source as these sets, so homology to the training proteins is likely even where the exact variants are not
  shared.
- **Among the models trained on Tsuboyama/Megascale, ESM-MSR leads 9 of 12 sets.** ME stage 1 leads S2648 (0.701
  vs 0.677) and Q3421 (0.638 vs 0.626), and ESM3ΔG leads S571 (0.511 vs 0.456).
- **The ε = 0 row is a control, not a model:** the ESM-MSR architecture with the LoRA scaled to zero, i.e. plain
  ESM3-small-open read through the calibration head. Its 12-set mean (0.378) sits on the published
  ESM3-small-open figure (0.380 over that table's 10 sets, §1.5), so the LoRA-off path behaves as expected and the
  LoRA is worth about +0.27 ρ.
- **SPURS single-additive and SPURS multi are identical on the 10 single-mutant sets**, as they must be: the two
  differ only in how a multi-mutant is scored, so they separate only on PTMUL and PTMUL-D.

### 1.2 Tsuboyama

| model | capped val | capped test | hyperopt val | hyperopt test |
|---|---|---|---|---|
| ESM-MSR capped (combined) | **0.854** | 0.843 | – | – |
| ESM-MSR capped, WT head | 0.841 | 0.824 | – | – |
| ESM-MSR dc8 (combined, 3 seeds) | **0.854** | **0.849** | – | – |
| ESM-MSR dc8, WT head | 0.836 | 0.827 | – | – |
| MutateEverything st1 (3 seeds) | 0.751 | 0.771 | – | – |
| MutateEverything st2 (3 seeds) | 0.727 | 0.762 | – | – |
| ThermoMPNN (additive, 3 seeds) | 0.785 | 0.794 | – | – |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.782 | 0.783 | – | – |
| SPURS single-additive ‡ | 0.845† | 0.866† | – | – |
| SPURS multi ‡ | 0.870† | 0.874† | – | – |
| ESMtherm ‡ | 0.716† | 0.774† | 0.691† | 0.707† |
| DDGemb (server) | 0.490 | 0.508 | **0.611** | 0.524 |
| DDMut (server) | – | – | 0.542* | 0.582* |
| IFUM, WT backbone ‡ | 0.666† | 0.711† | 0.746† | 0.697† |
| IFUM, ESMFold per variant ‡ | – | – | 0.778† | 0.738† |
| GeoDDG-3D (FASPR) | 0.595 | 0.609 | 0.610 | 0.557 |
| ESM3ΔG ‡ | 0.811† | 0.811† | 0.790† | 0.744† |
| ProSST (zero-shot) | q | q | 0.607 | **0.594** |
| ProteinMPNN (zero-shot) | 0.532 | 0.537 | – | – |
| ESM3-small-open, ε = 0 (zero-shot) | 0.565 | 0.578 | – | – |

- **SPURS leads the capped split (multi 0.874, single-additive 0.866), above ESM-MSR dc8 (0.849).** Treat this as
  not yet interpretable: SPURS trains on Megascale, and `esm_msr_testing.py` carries a `--remove_spurs_homologs`
  flag written for exactly this comparison, which the harness adapters do not apply. The honest version of this
  row needs the homolog-filtered rerun; the same caveat applies to ESM3ΔG (0.811) and IFUM (0.711).
- **Among models with no Tsuboyama in their training data, ESM-MSR leads:** dc8 0.849 and capped 0.843 on test,
  then ThermoMPNN 0.794, ME stage 1 0.771, GeoDDG 0.609, ProteinMPNN 0.537, DDGemb 0.508.
- **The zero-shot control** (ESM3-small-open, ε = 0) reaches 0.578 on capped test, so the fine-tuning is worth
  about +0.27 ρ here too.
- SPURS and ProteinMPNN cannot represent a mutant background (they read the sequence from the wild-type PDB), so
  their capped numbers cover the wild-type-background libraries only — 5 of 25 val and 7 of 37 test libraries are
  excluded for them, all singles.

### 1.3 DMS and Domainome

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR capped (combined) | 0.726 | 0.725 | 0.673 | 0.619 | 0.639 | 0.479 | 0.346 | 0.542 |
| ESM-MSR capped, WT head | 0.723 | 0.728 | 0.669 | 0.633 | 0.637 | 0.466 | 0.338 | 0.540 |
| ESM-MSR dc8 (combined, 3 seeds) | 0.745 | 0.742 | 0.682 | 0.636 | 0.639 | 0.481 | 0.337 | 0.543 |
| ESM-MSR dc8, WT head | 0.749 | 0.752 | 0.685 | 0.656 | **0.641** | 0.478 | **0.365** | **0.548** |
| MutateEverything st1 (3 seeds) | 0.707 | 0.681 | 0.646 | 0.581 | 0.580 | 0.483 | 0.296 | 0.520 |
| MutateEverything st1, additive | 0.729 | 0.714 | 0.690 | 0.615 | 0.603 | 0.483 | 0.287 | – |
| MutateEverything st2 (3 seeds) | 0.655 | 0.644 | 0.624 | 0.601 | 0.566 | 0.441 | 0.284 | 0.511 |
| MutateEverything st2, additive | 0.710 | 0.695 | 0.675 | 0.635 | 0.597 | 0.441 | 0.285 | – |
| ThermoMPNN (additive, 3 seeds) | 0.626 | 0.632 | 0.613 | 0.569 | 0.570 | 0.405 | 0.319 | 0.450 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.574 | 0.588 | 0.565 | 0.561 | 0.540 | 0.405 | 0.318 | – |
| SPURS single-additive ‡ | 0.689 | 0.711 | 0.736 | 0.635 | 0.602 | 0.535 | 0.330 | 0.513 |
| SPURS multi ‡ | 0.626 | 0.645 | 0.600 | 0.504 | 0.576 | 0.535 | 0.258 | 0.513 |
| ESMtherm ‡ | 0.093 | 0.136 | 0.484 | 0.466 | 0.229 | -0.000 | 0.214 | 0.201 |
| DDGemb (server) | 0.103 | 0.123 | 0.457 | 0.469 | 0.475 | 0.188 | 0.154 | 0.351 |
| IFUM, WT backbone ‡ | 0.766 | 0.768 | 0.648 | 0.565 | 0.507 | 0.190 | 0.262 | 0.464 |
| IFUM, ESMFold per variant ‡ | 0.762 | 0.747 | 0.697 | 0.575 | 0.523 | 0.217 | 0.284 | 0.473 |
| GeoDDG-3D (FASPR) | 0.351 | 0.431 | 0.604 | 0.481 | 0.395 | 0.240 | 0.200 | 0.413 |
| ESM3ΔG ‡ | 0.763 | 0.766 | **0.749** | 0.646 | 0.594 | 0.280 | 0.324 | 0.512 |
| ProSST (zero-shot) | **0.776** | **0.796** | 0.711 | 0.702 | 0.538 | 0.356 | 0.258 | 0.519 |
| ProteinMPNN (zero-shot) | 0.619 | 0.613 | 0.667 | 0.538 | 0.498 | **0.544** | 0.119 | **0.548** |
| ESM3-small-open, ε = 0 (zero-shot) | 0.689 | 0.742 | 0.627 | **0.723** | 0.536 | 0.483 | 0.259 | 0.540 |
| ESM3-large (Forge, zero-shot) | 0.624 | 0.606 | 0.557 | 0.582 | 0.458 | 0.432 | – | – |

- **Mean over the 7 DMS:** ESM-MSR dc8 WT 0.618, dc8 combined 0.609, SPURS single-additive 0.605, capped 0.601,
  ProSST 0.591, ME st1 additive 0.589, ESM3ΔG 0.589, ESM3-small-open ε = 0 0.580, ThermoMPNN 0.533,
  ProteinMPNN 0.514, GeoDDG 0.386, DDGemb 0.281, ESMtherm 0.232.
- **The literature-ΔΔG models invert here:** GeoDDG and DDGemb, which lead the external sets, are near the bottom
  on the deep-mutational scans; ProSST, a zero-shot model, leads three of the four Faure assays.
- **These assays discriminate much less than the stability benchmarks do:** the zero-shot ε = 0 control (0.580) is
  within 0.04 of the best supervised model, against a 0.27 gap on Tsuboyama. Rankings this tight should not be
  read as model quality without interval estimates (§1.5 has them for the published subset).
- **Domainome:** dc8 WT 0.548, ESM-MSR ≈ 0.54, then ME st1 0.520, ProSST 0.519, ESM3ΔG 0.512.

### 1.4 ProteinGym (213 DMS)

| model | PG avg | Stability | DMS mean |
|---|---|---|---|
| ESM-MSR capped (combined) | 0.510 | **0.870** | **0.556** |
| ESM-MSR capped, WT head | 0.508 | 0.826 | 0.550 |
| ESM-MSR dc8 (combined, 3 seeds) | **0.511** | 0.869 | 0.555 |
| MutateEverything st1 (3 seeds) | 0.468 | 0.808 | 0.506 |
| MutateEverything st2 (3 seeds) | 0.445 | 0.806 | 0.491 |
| ThermoMPNN (additive, 3 seeds) | 0.441 | 0.781 | 0.484 |
| SPURS single-additive ‡ | 0.483 | 0.837 | 0.525 |
| SPURS multi ‡ | 0.426 | 0.808 | 0.477 |
| ESMtherm ‡ | q | q | q |
| GeoDDG-3D (FASPR) | q | q | q |
| ESM3ΔG ‡ | q | q | q |
| ProSST (zero-shot) | q | q | q |
| ProteinMPNN (zero-shot) | 0.408 | 0.627 | 0.426 |
| ESM3-small-open, ε = 0 (zero-shot) | 0.448 | 0.621 | 0.470 |

The ε sweep and per-category breakdown are in §3. A model is summarised only once all 213 common DMS are in:
ESMtherm, GeoDDG and ESM3ΔG are mid-run (176, 67 and 141 of 213 on 2026-10-22). IFUM's ProteinGym run was
stopped as not worth the time it needed, and DDGemb and DDMut are web services that cannot be run at this scale.

### 1.5 Published-paper evaluation (reference; earlier models and row sets)


From `analysis_notebooks/bootstrap_external_results/rho_all_models_replicate1.csv`: the models of the
published comparison, trained on the hyperopt split. Its row sets are de-duplicated (S8754 → 6,411 rows) and it has
neither S669 nor K3822 (it uses their subsets S461 and K2369), so compare within this table only. It adds Rosetta and
the ESM3 family, which the October runs do not have.

| dataset | ESM-MSR (published) | ProteinMPNN | Rosetta cart. ΔΔG | ESM3-small-open | ESM3-small | ESM3-medium | ESM3-large | ME (hyperopt) | ME additive (hyperopt) | ThermoMPNN (hyperopt) | ThermoMPNN-D (hyperopt) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ssym | **0.750** | 0.694 | 0.621 | 0.584 | 0.594 | 0.617 | 0.498 | 0.669 | 0.722 | 0.727 | – |
| S461 | **0.750** | 0.441 | 0.617 | 0.402 | 0.445 | 0.422 | 0.371 | 0.713 | 0.720 | 0.647 | – |
| S783 | **0.725** | 0.559 | 0.648 | 0.480 | 0.526 | 0.510 | 0.330 | 0.690 | 0.717 | 0.691 | – |
| K2369 | **0.705** | 0.435 | 0.618 | 0.438 | 0.488 | 0.477 | 0.318 | 0.671 | 0.689 | 0.659 | – |
| S8754 | 0.617 | 0.437 | 0.571 | 0.379 | 0.410 | 0.409 | 0.268 | 0.630 | **0.657** | 0.611 | – |
| S2648 | 0.665 | 0.508 | 0.606 | 0.374 | 0.432 | 0.407 | 0.261 | 0.679 | **0.719** | 0.658 | – |
| Q3421 | 0.613 | 0.495 | 0.554 | 0.377 | 0.429 | 0.414 | 0.282 | 0.622 | **0.654** | 0.604 | – |
| PTMUL-D | **0.587** | 0.545 | 0.472 | 0.299 | 0.442 | 0.339 | 0.145 | 0.493 | 0.446 | 0.502 | 0.563 |
| PTMUL | **0.545** | 0.377 | 0.540 | 0.191 | 0.316 | 0.250 | 0.043 | 0.539 | 0.470 | 0.437 | 0.452 |
| S571 | 0.451 | 0.336 | 0.357 | 0.274 | 0.249 | 0.341 | 0.196 | 0.440 | **0.465** | 0.420 | – |
| mean of 10 external | **0.641** | 0.483 | 0.561 | 0.380 | 0.433 | 0.418 | 0.271 | 0.615 | 0.626 | 0.595 | 0.507 |
| Megascale val (per-domain avg.) | **0.835** | 0.595 | 0.666 | 0.561 | 0.592 | 0.583 | 0.516 | 0.744 | 0.744 | 0.768 | 0.763 |
| Megascale test (per-domain avg.) | **0.804** | 0.529 | 0.634 | 0.551 | 0.606 | 0.590 | 0.553 | 0.673 | 0.678 | 0.740 | 0.732 |
| DLG4 abundance | 0.721 | 0.621 | – | 0.678 | **0.745** | 0.723 | 0.583 | 0.645 | 0.702 | 0.629 | 0.530 |
| DLG4 binding | 0.739 | 0.615 | – | 0.731 | 0.755 | **0.762** | 0.562 | 0.638 | 0.722 | 0.651 | 0.545 |
| GRB2 abundance | **0.734** | 0.668 | – | 0.613 | 0.639 | 0.617 | 0.549 | 0.618 | 0.711 | 0.649 | 0.640 |
| GRB2 binding | 0.623 | 0.538 | – | 0.709 | **0.717** | 0.680 | 0.520 | 0.593 | 0.571 | 0.592 | 0.587 |
| MYO display | **0.636** | 0.499 | – | 0.527 | 0.557 | 0.521 | 0.463 | 0.562 | 0.586 | 0.577 | 0.528 |
| ESTA ΔT50 | 0.517 | **0.546** | – | 0.484 | 0.486 | 0.497 | 0.413 | 0.465 | 0.501 | 0.470 | – |
| Domainome (per-domain avg.) | **0.548** | **0.548** | – | 0.533 | **0.548** | 0.532 | 0.508 | 0.516 | 0.523 | 0.462 | – |


## 2. Epistasis: ESM-MSR vs the retrained baselines


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
| ME stage 2 / mt | 0.464 | 0.340 | 0.610 | 0.242 | 0.083 | 0.061 | 0.700 | **0.498** |
| TMPNN-D / mt | 0.166 | 0.095 | 0.155 | 0.179 | 0.040 | 0.018 | 0.601 | 0.650 |
| ESM-MSR capped / comb | 0.351 | 0.485 | 0.614 | 0.390 | 0.316 | 0.156 | **0.861** | 0.884 |
| ESM-MSR capped / mt | 0.351 | **0.513** | 0.644 | 0.410 | **0.326** | **0.206** | 0.860 | 0.593 |
| dc8 / comb (3 seeds) | **0.518** | 0.481 | **0.667** | **0.415** | 0.246 | 0.155 | 0.855 | 0.626 |
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
| val `spearman_all` | 0.751 | 0.727 | 0.785 | 0.782 | **0.854** | **0.854 ± 0.001** |
| test `spearman_all` | 0.771 | 0.762 | 0.794 | 0.783 | 0.843 | **0.849 ± 0.001** |
| test `spearman_doubles` | 0.617 | 0.660 | 0.673 | 0.672 | **0.763** | — |
| test `spearman_doubles_epi` (dddG ρ) | 0.020 | 0.434 | — | 0.181 | **0.554** | — |

dc8 vs capped: test combined +0.006, test MT +0.007, val WT −0.005 — a modest gain on test, level on val.
These dc8 numbers are **on measured rows only** (§5.3); with bounded rows they would read 0.858 and look like
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


### 2.6 How much predicted epistasis should a multi-mutant prediction keep?

Write **A** = `wt_lora_pred` (additive by construction) and **M** = `mt_lora_pred` (the direct multi-mutant
head). Then

    (1 − λ)·A + λ·M  ≡  A + λ·(M − A)

so the ensemble weight **λ is exactly the fraction of the MT head's predicted interaction that is kept**. The
shipped `combined_pred = 0.5·(A + M)` is already λ = 0.5 — a 50 % shrinkage of predicted epistasis, not an
unweighted average of two independent models. The proposal `(N−1)/N · WT + 1/N · MT` is the schedule λ = 1/N.

Rescored offline from the saved per-variant predictions (`analysis_scripts/ensemble_weight_scan.py`; no GPU
needed, since every prediction file carries both heads and the mutation count). Multi-mutants only, capped
checkpoint:

| dataset | n | N | λ = 0 (additive) | λ = 0.5 (shipped) | λ = 1 (MT only) | best λ | ρ at best |
|---|---|---|---|---|---|---|---|
| Tsuboyama test | 13,319 | 2 | 0.6996 | 0.7628 | 0.6771 | 0.6 | **0.7662** |
| PTMUL | 914 | 2–10 | 0.5676 | **0.5954** | 0.5448 | 0.5 | 0.5954 |
| PTMUL-D | 536 | 2 | 0.5861 | 0.6015 | 0.5329 | 0.4 | **0.6022** |
| GRB2 abundance | 62,332 | 2 | 0.6635 | 0.6671 | 0.6214 | 0.33 | **0.6684** |
| DLG4 binding | 6,810 | 2 | 0.7091 | 0.7038 | 0.6700 | 0.1 | **0.7093** |
| GB1 | 149,284 | 2–4 | 0.3363 | 0.3450 | 0.3017 | 0.6 | **0.3453** |

**λ = 1/N does not help, for three separate reasons.**

1. **It is a no-op on doubles.** 1/N = 0.5 at N = 2, which is the shipped recipe, and doubles are essentially all
   of the multi-mutant data here: Tsuboyama, both Faure assays and PTMUL-D contain nothing else.
2. **Where N ≥ 3 exists it is worse.** Paired bootstrap of ρ(λ = 1/N) − ρ(λ = 0.5) on multi-mutants:
   PTMUL −0.0140 [−0.0203, −0.0075] (capped) and −0.0107 [−0.0173, −0.0041] (dc8 s1); GB1 −0.0044
   [−0.0065, −0.0023] (capped), though dc8 disagrees at +0.0111. Both PTMUL intervals exclude zero.
3. **The N-dependence points the other way.** On PTMUL the best λ *rises* with N — ρ(λ = 0.6) − ρ(λ = 0.3) is
   −0.0127 at N = 2, −0.0021 at N = 3 and +0.0269 at N = 4 (capped). The N = 4 interval still includes zero
   (n = 98), so this is a trend rather than a result, but nothing in the data supports shrinking M as N grows.
   A mechanism fits it: the additive baseline accumulates N single-prediction errors and misses every
   interaction, while the MT head reads the whole mutant context in one pass, so M gets *relatively* more
   trustworthy as N grows, not less.

Also note that λ = 1/N sends λ → 1 at N = 1, i.e. pure MT for single mutants, which costs 0.024 ρ on the 35,421
Tsuboyama test singles (0.8019 → 0.7774). Any N-schedule has to be restricted to N ≥ 2.

**What the data does say.** The optimal λ tracks the *assay*, not the mutation count: ≈ 0.5–0.6 on in-distribution
stability (Tsuboyama), ≈ 0.4–0.5 on PTMUL, and ≈ 0.1–0.33 on the functional Faure assays. So the MT head's
epistasis is genuinely overweighted — but on out-of-distribution functional readouts, where λ should fall to
roughly a third, not as a function of N. In-distribution it is slightly *under*weighted (best λ = 0.6 > 0.5).

**Recommendation: keep λ = 0.5.** It is within 0.004 ρ of the best constant λ on every set except GB1, and no
N-schedule is justified. The one case worth revisiting is GB1 at N = 2, where λ = 0 gains +0.047 (0.5847 →
0.6321) — consistent with GB1 being the assay whose epistasis sign disagrees with Tsuboyama throughout these
tables. If a per-assay λ is ever wanted, it needs to be fitted on held-out assays rather than on these.

## 3. ProteinGym in detail


213 DMS that every model scored (BRCA1, BRCA2, POLG_CXB3N and SCN5A excluded: > 1,800 residues, out of
memory for at least one model). **PG avg** = mean of the five category means (a plain mean over DMS within a
category, not ProteinGym's per-UniProt weighting); **DMS mean** = mean over the 213. Unmasked scoring,
ProteinGym's own AF2 structures.

| model | Activity (43) | Binding (13) | Expression (18) | Org. fitness (73) | Stability (66) | PG avg | DMS mean |
|---|---|---|---|---|---|---|---|
| **dc8 full, ε = 0.75** (3 seeds) | 0.439 | 0.364 | 0.515 | 0.410 | 0.858 | **0.517** | **0.560** |
| ESM-MSR capped, WT only, ε = 0.75 | 0.449 | 0.363 | 0.512 | **0.425** | 0.813 | 0.512 | 0.553 |
| dc8 full, ε = 1 (3 seeds) | 0.426 | 0.347 | **0.517** | 0.395 | 0.869 | 0.511 | 0.555 |
| dc8 full, ε = 0.5 (3 seeds) | 0.443 | 0.381 | 0.499 | 0.413 | 0.822 | 0.512 | 0.551 |
| ESM-MSR capped, full | 0.426 | 0.346 | 0.516 | 0.395 | **0.870** | 0.510 | 0.556 |
| ESM-MSR capped, WT only, ε = 1 | 0.436 | 0.349 | 0.515 | 0.412 | 0.826 | 0.508 | 0.550 |
| ESM-MSR capped, WT only, ε = 0.5 | **0.450** | 0.377 | 0.493 | **0.425** | 0.779 | 0.505 | 0.542 |
| SPURS single-additive | 0.384 | 0.334 | 0.490 | 0.368 | 0.837 | 0.483 | 0.525 |
| dc8 full, ε = 0.25 (3 seeds) | 0.430 | **0.385** | 0.468 | 0.400 | 0.742 | 0.485 | 0.517 |
| ESM-MSR capped, WT only, ε = 0.25 | 0.438 | 0.382 | 0.466 | 0.411 | 0.712 | 0.482 | 0.512 |
| MutateEverything single (3 seeds) | 0.371 | 0.336 | 0.472 | 0.352 | 0.808 | 0.468 | 0.506 |
| ESM-MSR, ε = 0 (no LoRA) | 0.417 | 0.374 | 0.439 | 0.391 | 0.621 | 0.448 | 0.470 |
| MutateEverything double (3 seeds) | 0.348 | 0.296 | 0.437 | 0.338 | 0.806 | 0.445 | 0.491 |
| ThermoMPNN (3 seeds) | 0.351 | 0.299 | 0.439 | 0.337 | 0.781 | 0.441 | 0.484 |
| SPURS multi | 0.326 | 0.267 | 0.410 | 0.320 | 0.808 | 0.426 | 0.477 |
| ProteinMPNN (3 decoding orders) | 0.343 | 0.323 | 0.437 | 0.310 | 0.627 | 0.408 | 0.426 |

Paired against the capped checkpoint over the 213 common DMS (3 seeds × 4 ε, every DMS scored):

| dc8 setting | PG avg | paired Δ vs capped | better on | seed sd (PG avg) |
|---|---|---|---|---|
| ε = 1 | 0.5107 | −0.0002 | 108 / 213 | 0.0003 |
| **ε = 0.75** | **0.5170** | **+0.0047** | **133 / 213** | 0.0007 |
| ε = 0.5 | 0.5115 | −0.0048 | 98 / 213 | 0.0007 |
| ε = 0.25 | 0.4853 | −0.0385 | 70 / 213 | 0.0005 |

- **dc8 ε = 1 matches capped** and **ε = 0.75 is the only setting that beats it** (+0.0047, better on 133 of 213,
  paired t = 2.83). Seed sd ≤ 0.0007, so the ε = 0.75 gain is ~7 seed-sds and the ε = 1 null is real.
- **The ε trend:** Stability falls monotonically as ε drops (0.869 → 0.742) while Binding rises monotonically
  (0.347 → 0.385), with Activity and Organismal fitness peaking near ε = 0.5. Scaling the LoRA down trades
  stability skill for general fitness, and ε = 0.75 is where the two balance.
- **MT-pass ablation** (`--mt_pass_adapter wt`, dc8 s1 ε = 1): the WT adapter scoring the mutant sequence
  costs −0.016 ρ on the first 24 DMS (worse on 18). Verified by hand on BCHB: unmasked, the WT adapter sees the
  mutant residue it is scoring and its LLR collapses toward zero (mean −0.85 vs −3.44 on the WT sequence).
  This is a partial result: 26 of 213 DMS before the run was stopped.
- **ProteinMPNN test-time ensembling:** averaging three decoding orders +0.0003; a 0.2 Å backbone-noise
  ensemble of 8 +0.0017 on a 73-DMS pilot (50/73 up, t = 1.29). Not worth a full run.


## 4. Third-party predictor notes

- **The external-set leaders trained on that kind of data.** S2648, S8754, Q3421, K3822 and Ssym overlap the
  training data of DDGemb, DDMut and GeoDDG, which is why they lead §1.1 and fall behind on Tsuboyama and the DMS.
- **IFUM has two modes.** §1 reports the WT-backbone mode. Its native mode folds every variant with ESMFold and
  scores it on its own structure, which is slightly better on Tsuboyama (val 0.778 vs 0.746, test 0.738 vs 0.697)
  but costs ~3.9 s per variant sequence to fold, so it was run on the external sets and DMS only.
- **UniMutStab could not be run:** the repository has no trained weights and no inference code, only pseudocode.
- **DDMut is rate-limited** to 20–30 s per variant through its REST API, so only the external sets are complete;
  its Tsuboyama column is partial (`*`).
- **AF2 vs crystal structures** (ESM-MSR-small, protein-level bootstrap, 2,000 resamples): mean Δ over the 10
  single-mutant sets +0.003 [−0.006, +0.010], over the 2 multi-mutant sets −0.016 [−0.040, +0.001]; only Ssym
  (−0.036) clearly moves. AF2 structures are a safe substitute.
- **Indexing audit:** all 944,018 unique variants over 22 datasets, 0 wild-type mismatches, with a
  shifted-by-one negative control flagging 94 % — so the mutation-to-structure mapping is not a source of error
  anywhere in these tables.

## 5. Caveats that bind these numbers

1. **GB1 must come from the repo's own `data/preprocessed`.** The copy under `~/software/esm-msr` is missing
   V39A/C/D (73 of 76 singles of a 4-site saturation library), which costs 152 of GB1's 2,091 scorable doubles,
   since a double needs both its singles measured. The repo copy is complete (149,360 rows) and is what every
   number here uses; the bad file is quarantined.
2. **`dG_wt` is the identity `dG_ML − ddG_ML`,** not the mean of the `wt` rows — the old definition matched the
   library constant in only 7 of 412 libraries (median error 0.015, max 0.197 kcal/mol), and the error is a
   per-library constant, so it shifts that library's whole position on the global curve. Re-scoring with the fixed
   definition moved seed-mean epistasis cells by ≤ 0.011, and left `naive_rho`, `cell_rank` and `cell_flipacc`
   exactly unchanged, since those do not use `dG_wt`.
3. **Rows the assay only bounded are predicted but not reported.** `<-1` / `>5` rows are 1.5 % of val and 6.4 % of
   test; their `ddG_ML` is a library bound rather than a measurement, and they sit at the extremes where ranking
   is easy, so including them inflates ρ by about 0.01.
4. **Cross-model validation comparisons use `valfull` (34,957 rows).** The baselines' prepared val table applies
   item-level *training* rules and is a strict subset (33,333), which is not comparable with ESM-MSR's own val
   numbers.
5. **Test-set ddG tables exclude destabilized backbones** (`--test_backbones wt_only`, 40,866 rows): 7 of 37 test
   libraries carry a background mutation that only the models trained on that pattern have seen. Every such row is
   a single, so the epistasis columns are identical either way (verified: max abs diff 0).
6. **Ground truth is always ESM-MSR's `ddG_ML`.** Upstream MutateEverything silently flips the sign of its own
   ground truth when more than half of `ddg` is negative, which would bite Ssym and GB1.
7. **External sets are aligned by subset relation:** S461 ⊂ S669, K2369 ⊂ K3822, PTMUL-D ⊂ PTMUL, so a model only
   has to predict the superset.
8. **The link is applied in training only.** For link-trained checkpoints (dc8), `esm_msr_testing.py` writes
   *latent* ddG; `epi_baselines.msr_link` rebuilds the link from the checkpoint for the observed-scale epistasis
   metrics. Rank metrics within a dataset are unaffected.

## 6. Where the code and results live

In the repo, on `devel-revision`: `inference_scripts/pgym_runner.py` (the one ProteinGym loop every model shares,
recording per-DMS time, peak VRAM, GPU and host), the ProteinGym modes of `esm_msr_testing.py`,
`proteinmpnn_testing.py`, `spurs_testing.py` and `thermompnnd_testing.py`, and
`analysis_scripts/benchmark_matrix.py`.

On the cluster, deliberately not in the repo (baseline retraining, MutateEverything, Slurm and cluster-specific
scripts):

| path | what |
|---|---|
| `~/esm-msr-devel/outputs/baselines_capped/` | the retrained-baseline and dc8 result tables (§2) |
| `~/esm-msr-devel/outputs/pgym/` | every ProteinGym run, plus `pgym_all_models_per_dms.csv` and `pgym_dc8_per_dms.csv` |
| `~/esm-msr-devel/analysis_notebooks/stats/` | ESM-MSR's own per-dataset statistics |
| `~/esm-msr-devel/baselines/` | baseline data prep (`prepare_oct06_data.py`), training wrappers, and the epistasis suite applied to other models (`epi_baselines.py`, `epi_dms.py`) |
| `~/stability-baselines/` | the third-party harness, adapters and results (not a git repository) |
| `~/project_oct06_baselines/` | a symlink index over the retrained-baseline work, with its own README |

`baselines/` AST-loads functions out of `esm_msr_testing.py` and resolves `epi_items` / `_by_head` from
`devel-experimental`, so renaming those in the repo breaks it.
