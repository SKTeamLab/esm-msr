# ESM-MSR benchmark results, October 2026

**Comprehensive evaluation of ESM-MSR against retrained baselines and published stability and fitness predictors.**

**The ESM-MSR checkpoints compared throughout.** **ESM-MSR (baseline)** is the standard unmasked dual-adapter recipe retrained on the
`splits_oct06_capped` split as a **three-seed triplicate** ($\sigma = 1.0$); §1.1–1.3 and §2 report its three-seed mean.
Rows labelled **ESM-MSR (released)** denote the single released capped checkpoint (`n_seeds = 1`), used in the §1.3 LoRA-scale sweep,
§1.4 single-checkpoint ProteinGym rows, §2.6, and §3 (in `benchmarks/all_by_all_long.csv`, rows labelled `"ESM-MSR (released)"` with
`n_seeds = 3` correspond to the three-seed baseline triplicate, whereas `n_seeds = 1` rows correspond to the single released checkpoint).
**ESM-MSR (devel)** is the three-seed latent-floor triplicate trained on `splits_oct06_capped` (`val_rho_combined_avg`
0.840 / 0.839 / 0.841; checkpoint metadata and MD5 checksums in [`benchmarks/checkpoints.csv`](benchmarks/checkpoints.csv)).
All ESM-MSR checkpoints and all retrained baselines were trained on `data/splits_oct06_capped.pkl` and evaluated on identical held-out rows.
Throughout this document, "capped" refers strictly to the `splits_oct06_capped` data split.

**The retrained baselines.** MutateEverything (ME, stage 1 and stage 2), ThermoMPNN / ThermoMPNN-D, and SPURS / SPURS multi
follow their upstream training recipes retrained from scratch across three seeds on `data/splits_oct06_capped.pkl`, using all
**valid numeric `ddG_ML` labels** (both mutant and wild-type unfolding free energy within the assay's $[-1, 5]$ kcal/mol dynamic
range, excluding bounded rows) without applying any ESM-MSR-specific censoring or item filters, and retaining stabilizing
mutant-background libraries. Full training-inventory accounting and protocol details are given in §1.6 and §6.3.

**Metric conventions.** Every table reports Spearman rank correlation $\rho$ per model and dataset, oriented so positive predictions
correspond to higher stability or fitness for every model. External $\Delta\Delta G$ sets and the 7 functional DMS assays report
pooled Spearman $\rho$ per dataset; Tsuboyama and Domainome report the equal-weight mean of per-protein Spearman $\rho$ across
uncensored measured variants; ProteinGym reports per-DMS Spearman $\rho$ aggregated by functional category and overall mean (§3).
Multi-seed entries report the arithmetic mean across seeds; sample standard deviations ($n - 1$) across seeds are recorded in
[`benchmarks/all_by_all_long.csv`](benchmarks/all_by_all_long.csv) and [`benchmarks/seed_sd_by_table_cell.csv`](benchmarks/seed_sd_by_table_cell.csv).
Companion documentation: [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md) (training–test overlap audit),
[`SPLITTING_AND_HOMOLOGY_AUDIT.md`](SPLITTING_AND_HOMOLOGY_AUDIT.md) (splitting and homology methodology), and
[`validation_metrics.md`](validation_metrics.md) (stability and epistasis metric definitions).

## 1. All-by-all: every model on every dataset

**Leak notation (§6).** Every number carries a superscript for the worst supervised training-data overlap it has with its test set, and the full code sits in the matrix directly below each table. **ᴹ** identical mutation(s) in training; **ᵐ** the test variant's mutations were trained but not this exact combination (additive models, doubles made of trained singles); **ᴾ** same protein, other mutations; **ᴴ** strong homolog in training (MMseqs2 E ≤ 1e-3 over ≥ 50 % of the test protein, or Foldseek E ≤ 1e-3 / qTM > 0.70 over ≥ 50 % on Domainome and ProteinGym); **ʰ** weak/ambiguous similarity (E ≤ 1e-2 or ≥ 30 % identity over ≥ 30 %, or Foldseek E ≤ 1e-2 over ≥ 30 %); **ᴰ** documented training on the same source dataset, rows not reconstructable; **#** the number is not a test result (the set is in training); **ⱽ** the rows were used to validate, select or calibrate the model (capped val for models trained on the split, a third-party model's own validation rows, Rosetta's energy-function tuning data). Zero-shot / unsupervised models without mutation-level supervision carry **–** (no supervised overlap). A code such as `M.59 P.35 H.02` gives the fraction of dataset rows in each class, `M?` / `V?` mark ceilings inferred from a published protein-level split (IFUM), and `V` marks rows used for validation, model selection or calibration (`ⱽ` in the numeric tables). Superscripts are omitted below 0.5 % of rows; the companion matrix keeps them. A star (`*`) marks partial coverage (DDMut: S669 98 %, S461 97 %); **_italic_** a different variant set (Rosetta on ProteinGym: single mutants only, shown but never compared); **q** a run in progress or queued; **–** not run (or no overlap in the leak-code matrix). The audit, the inventories and the assumptions are in §6; the hyperopt-split rows of §1.2 and §1.5 predate the Oct06 split and are out of its scope.

ESM-MSR's **MT head is not listed anywhere**: it is an internal head, not a deployable model. ESM-MSR appears by
its combined prediction, which is what is deployed, and in §1 also by its WT head, which is the additive reading.

Provenance of the rows that were not produced by `esm_msr_testing.py`: ProSST's ProteinGym numbers are the
authors' own published ProSST-2048 scores (ProteinGym ships every benchmarked model's predictions), scored on the
same variants as everything else; Rosetta's capped-split numbers are `cartesian_ddg` runs (FastRelaxed
structures, 3 iterations, doubles scored directly) and its ProteinGym numbers are single-mutant `cartesian_ddg`
predictions; the ESM3 zero-shot rows are the Forge API runs.

**Which Tsuboyama columns are comparable.** The **capped** columns are the split every October model was trained
on. The **hyperopt** columns are the older split the third-party harness was first run on; it shares only 4 of 23
val and 9 of 33 test libraries with capped, so the two are not comparable with each other.

### 1.1 Thermostability: the 12 external sets

| model | s571 | s783 | s2648 | s8754 | s669 | s461 | ssym | q3421 | k3822 | k2369 | ptmul | ptmuld | mean of 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | **0.455** | 0.713 | 0.669 | 0.616 | 0.558 | 0.750 | 0.765 | 0.621 | 0.714 | 0.715 | 0.583 | 0.593 | 0.646 |
| ESM-MSR (baseline), WT head (3 seeds) | 0.441 | 0.698 | 0.662 | 0.607 | 0.543 | 0.744 | 0.764 | 0.615 | 0.715 | 0.712 | 0.571 | 0.577 | 0.637 |
| ESM-MSR (devel), combined (3 seeds) | 0.445 | **0.725** | 0.677 | **0.625** | 0.560 | **0.753** | 0.773 | 0.626 | **0.721** | **0.722** | **0.594** | **0.597** | 0.652 |
| ESM-MSR (devel), WT head | 0.430 | 0.713 | 0.675 | 0.618 | 0.551 | **0.753** | **0.776** | 0.623 | 0.719 | 0.717 | 0.583 | 0.585 | 0.645 |
| MutateEverything st1 (3 seeds) | 0.439 | 0.706 | **0.701** | 0.623 | 0.559 | 0.721 | 0.674 | **0.638** | 0.696 | 0.674 | 0.472 | 0.480 | 0.615 |
| MutateEverything st2 (3 seeds) | 0.426 | 0.694 | 0.676 | 0.606 | 0.542 | 0.710 | 0.636 | 0.620 | 0.683 | 0.662 | 0.510 | 0.489 | 0.605 |
| ThermoMPNN (additive, 3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.433 | 0.499 | 0.585 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.432 | 0.514 | 0.586 |
| SPURS (retrained, 3 seeds) | 0.449 | 0.675 | 0.655 | 0.581 | 0.497 | 0.644 | 0.741 | 0.593 | 0.621 | 0.635 | 0.427 | 0.537 | 0.588 |
| SPURS multi (retrained, 3 seeds) | 0.426 | 0.641 | 0.620 | 0.557 | 0.471 | 0.612 | 0.696 | 0.566 | 0.598 | 0.605 | 0.403 | 0.481 | 0.556 |
| ESMtherm | 0.349ᴹ | 0.434ᴾ | 0.371ᴾ | 0.347ᴹ | 0.414ᴾ | 0.474ᴾ | 0.380 | 0.348ᴾ | 0.366ᴹ | 0.393ᴹ | 0.235ᴾ | 0.270ᴾ | 0.365 |
| DDGemb (server) | 0.484ᴹ | 0.689ᴾ | 0.895ᴹ | 0.684ᴹ | **0.570** | 0.654 | 0.891ᴹ | 0.752ᴹ | 0.745ᴹ | 0.702ᴹ | 0.616ᵐ | 0.678ᵐ | 0.697 |
| DDMut (server) | 0.350ᴹ | 0.687ᴹ | 0.731ᴹ | 0.638ᴹ | 0.402*ᴹ | 0.534*ᴹ | 0.363ᴹ | 0.653ᴹ | 0.709ᴹ | 0.681ᴹ | 0.725ᴹ | 0.718ᴹ | – |
| IFUM, WT backbone | 0.376ᴹ | 0.498ᴹ | 0.485ᴹ | 0.477ᴹ | 0.434ᴾ | 0.531ᴾ | 0.574 | 0.431ᴹ | 0.536ᴹ | 0.547ᴹ | 0.350ᴾ | 0.405ᴾ | 0.470 |
| IFUM, ESMFold per variant | 0.385ᴹ | 0.471ᴹ | 0.484ᴹ | 0.451ᴹ | 0.409ᴾ | 0.542ᴾ | 0.580 | 0.433ᴹ | 0.508ᴹ | 0.512ᴹ | 0.391ᴾ | 0.443ᴾ | 0.467 |
| GeoDDG-3D (FASPR) | 0.526ᴹ | 0.865ᴹ | 0.892ᴹ | 0.841# | 0.615ᴹ | 0.664ᴾ | 0.914ᴹ | 0.822ᴹ | 0.852ᴹ | 0.831ᴹ | 0.717ᵐ | 0.669ᵐ | 0.767 |
| ESM3ΔG | 0.511ᴾ | 0.653ʰ | 0.642ᴴ | 0.542ᴴ | 0.564ʰ | 0.681ʰ | 0.735ʰ | 0.572ʰ | 0.589ᴴ | 0.598ᴴ | 0.409ᴴ | 0.482ᴴ | 0.582 |
| ProSST (zero-shot) | 0.310 | 0.564 | 0.436 | 0.428 | 0.369 | 0.561 | 0.519 | 0.416 | 0.476 | 0.497 | 0.182 | 0.317 | 0.423 |
| ProteinMPNN (zero-shot) | 0.324 | 0.561 | 0.506 | 0.438 | 0.329 | 0.442 | 0.696 | 0.494 | 0.420 | 0.436 | 0.382 | 0.546 | 0.464 |
| ESM3-small-open, σ = 0 (zero-shot) | 0.268 | 0.487 | 0.379 | 0.374 | 0.299 | 0.404 | 0.583 | 0.382 | 0.423 | 0.442 | 0.190 | 0.304 | 0.378 |
| ESM3-large (Forge, zero-shot) | 0.220 | 0.353 | 0.267 | 0.291 | 0.272 | 0.334 | 0.539 | 0.306 | 0.318 | 0.364 | 0.017 | 0.126 | 0.284 |
| ESM3-medium (Forge, zero-shot) | 0.332 | 0.552 | 0.439 | 0.438 | 0.342 | 0.445 | 0.633 | 0.450 | 0.483 | 0.503 | 0.264 | 0.353 | 0.436 |
| ESM3-small (Forge, zero-shot) | 0.236 | 0.548 | 0.441 | 0.425 | 0.336 | 0.457 | 0.588 | 0.441 | 0.475 | 0.500 | 0.304 | 0.406 | 0.430 |
| ESM-MSR-small (March reference) | 0.447 | 0.714 | 0.677ᴴ | 0.614ᴴ | 0.542 | 0.743 | 0.781 | 0.621ᴴ | 0.701 | 0.704 | 0.543ᴴ | 0.577ᴴ | 0.639 |

| model | s571 (leak code) | s783 (leak code) | s2648 (leak code) | s8754 (leak code) | s669 (leak code) | s461 (leak code) | ssym (leak code) | q3421 (leak code) | k3822 (leak code) | k2369 (leak code) | ptmul (leak code) | ptmuld (leak code) | mean of 12 (leak code) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (baseline), WT head (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (devel), combined (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (devel), WT head | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| MutateEverything st1 (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| MutateEverything st2 (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ThermoMPNN (additive, 3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| SPURS (retrained, 3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| SPURS multi (retrained, 3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESMtherm | M.02 P.10 H.03 V.009 | M.004 P.19 h.04 | P.05 H.02 h.01 | M.01 P.21 H.02 h.008 V.002 | P.24 H.003 h.09 | P.35 h.14 | – | M<.001 P.06 H.03 h.02 | M.02 P.29 H.03 h.02 V.002 | M.03 P.36 H.03 h.02 V.003 | P.13 H.06 h.01 V.001 | P.06 H.06 h.02 | – |
| DDGemb (server) | M.08 P.21 H.03 h.01 | P.71 H.02 | M.92 P.008 H.02 h.001 D | M.44 P.23 H.04 h.004 | – | – | M.60 P.40 | M.63 P.21 H.02 | M.38 P.19 H.03 h.006 | M.33 P.17 H.03 h.003 | m.48 P.32 H.03 h.003 | m.54 P.32 H.04 h.004 | – |
| DDMut (server) | M.21 P.26 H.05 | M.61 P.35 H.005 | M.98 P.01 H.007 D | M.69 P.21 H.03 h.002 D | M.10 P.19 H.02 h.004 | M.15 P.24 H.004 h.004 | M.88 P.12 | M.84 P.11 H.01 | M.72 P.15 H.03 | M.74 P.14 H.03 | M.80 m.11 P.08 H.02 | M.95 m.002 P.03 H.02 | – |
| IFUM, WT backbone | M?.04 P.01 H.08 | M?.03 P.006 H.03 h.11 | M?.006 P.05 H.02 h.006 | M?.12 P.10 H.02 h.003 V?.001 | P.14 H.003 h.09 | P.21 h.14 | – | M?.01 P.05 H.02 h.01 | M?.15 P.14 H.02 h.02 V?.002 | M?.21 P.17 H.02 h.01 V?.003 | M?.001 P.15 H.03 h.001 | P.07 H.04 h.002 | – |
| IFUM, ESMFold per variant | M?.04 P.01 H.08 | M?.03 P.006 H.03 h.11 | M?.006 P.05 H.02 h.006 | M?.12 P.10 H.02 h.003 V?.001 | P.14 H.003 h.09 | P.21 h.14 | – | M?.01 P.05 H.02 h.01 | M?.15 P.14 H.02 h.02 V?.002 | M?.21 P.17 H.02 h.01 V?.003 | M?.001 P.15 H.03 h.001 | P.07 H.04 h.002 | – |
| GeoDDG-3D (FASPR) | M.41 P.09 H.04 | M.88 h.07 | M1 | M1 # | M.01 P.07 H.03 h.03 | P.11 H.004 h.03 | M.50 P.50 | M.98 P.002 | M.89 P.002 H.002 h.002 | M.91 P.003 H.003 h.002 | m.73 P.24 H.004 | m.78 P.20 H.006 | – |
| ESM3ΔG | P.01 H.005 h.33 | h.44 | H.02 h.33 | H.01 h.31 | h.55 | h.64 | h.72 | H.004 h.37 | H.009 h.29 | H.007 h.26 | H.009 h.34 | H.01 h.32 | – |
| ProSST (zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ProteinMPNN (zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ESM3-small-open, σ = 0 (zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ESM3-large (Forge, zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ESM3-medium (Forge, zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ESM3-small (Forge, zero-shot) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ESM-MSR-small (March reference) | – | H.001 | H.02 h.003 | H.01 h.001 | H.003 | – | – | H.01 h.002 | H.003 h.003 | H.004 h.004 | H.05 h.009 | H.03 h.01 | – |
- **The 12-set ranking is dominated by training overlap.** GeoDDG (0.767) and DDGemb (0.697) lead only on cells that are partly or wholly their training data: S8754 *is* GeoDDG's training set (`M 1.00`, `#`), S2648 is 92.5 % in DDGemb's S2450, and all twelve of GeoDDG's cells carry an identical-mutation, constituent-mutation or same-protein flag (S461 through the GeoFitness DMS pretraining of its encoder; §6.5). Over the cells with no such flag (§6.5) the 12-set mean is ESM-MSR (devel) 0.652, ESM-MSR (baseline) 0.646, ME stage 1 0.615, ME stage 2 0.605, SPURS (retrained) 0.588, ThermoMPNN 0.585, SPURS multi (retrained) 0.556, ProteinMPNN 0.464 and ProSST 0.423; none of them loses a cell to the filter. On the two sets that are same-protein-clean for ESM-MSR, ME, ThermoMPNN, retrained SPURS and DDGemb (S669, S461), DDGemb is about 0.01 ahead on S669 (0.570 against ESM-MSR 0.558 baseline, 0.560 devel) and ESM-MSR leads S461 (0.750 baseline, 0.753 devel, against 0.654).
- **There is no set with no overlap for every model.** The overlap lattice is measured, not assumed: even on S669 and S461, DDMut shares 10.0 % (67/669) and 14.5 % (67/461) exact training mutations via FireProtDB proteins in S9028, and GeoDDG shares 1.2 % (8/669) exact mutations and 7 % same-protein rows on S669 (0 exact mutations on S461, making S461 the sole full-coverage set with zero exact mutations, but 11 % same-protein rows through GeoDDG's GeoFitness DMS pretraining). S571 is 41 % identical-mutation overlap with GeoDDG and 21 % with DDMut; Ssym is 88 % with DDMut (including reverse mutant structures), 60 % with DDGemb (S2450 forward + reverse) and 50 % with GeoDDG (all 342 forward halves); PTMUL is 80 % exact + 11 % constituent with DDMut (S9028 + the SM1218 + SM444 multi-mutant set its deployed model was trained on), 73 % constituent + 24 % same-protein with GeoDDG, and 48 % constituent + 32 % same-protein with DDGemb. A ranking over the raw 12 sets therefore compares different models on different subsets. The per-model means over their own clean cells are in §6.5; only ESM-MSR, ME, ThermoMPNN, the retrained SPURS and SPURS multi, ProSST, ProteinMPNN and the zero-shot control are clean on all twelve, and ESM-MSR (baseline 0.646, devel 0.652) leads them.
- **Among the models trained on Tsuboyama/Megascale, ESM-MSR (devel) leads 9 of 12 sets and the baseline triplicate 8, and all but two of those models are clean comparators.** ESMtherm trained on 51 % of the capped test rows as identical variants (verified against the sequences of its published split; 85 % same-protein in all) and IFUM on the domains behind up to 46 % (`M?`, a ceiling from its published domain split); ESM3ΔG is not Megascale-trained at all (see below). ESM-MSR, MutateEverything, ThermoMPNN and the retrained SPURS and SPURS multi have zero exact-mutation and zero same-protein overlap with all 33 test libraries (§6.5). ME stage 1 takes S2648 and Q3421 from both ESM-MSR rows; the baseline triplicate also trails ME stage 1 on S8754 (0.616 against 0.623) and, by 0.001, on S669; devel gives S571 to the baseline (0.455 against 0.445) and to SPURS (0.449). The retrained SPURS leads no set.
- **The σ = 0 row is a control, not a model:** the ESM-MSR architecture with the LoRA scaled to zero, i.e. plain
  ESM3-small-open read through the calibration head. Its 12-set mean (0.378) sits on the published
  ESM3-small-open figure (0.380 over that table's 10 sets, §1.5), so the LoRA-off path behaves as expected and the
  LoRA is worth about +0.27 ρ.
- **SPURS is two models, and both are retrained.** SPURS proper is the single-mutant model, summed over substitutions for a multi-mutant. **SPURS multi** is the
  multi-mutant model applied to *every* variant, singles included. Both were retrained on the capped split, three seeds each and paired by seed (the multi model of
  seed *s* starts from the single model of seed *s*; recipe in §1.6), so they are held out like ESM-MSR, ME and ThermoMPNN. Held out, SPURS is level with ThermoMPNN
  on the 12-set mean (0.588 against 0.585) and 0.06 behind ESM-MSR (devel 0.652). SPURS multi is the weaker predictor on singles: 12-set mean 0.556 against 0.588,
  7-DMS mean 0.523 against 0.578, Domainome 0.393 against 0.489 (0.383 vs 0.477 on the 458 low-id subset), capped test 0.772 against 0.801 (S669 0.471 against 0.497, K3822 0.598 against 0.621). It is
  also lower on the multi-mutant sets (PTMUL 0.403 against 0.427, PTMUL-D 0.481 against 0.537), so the multi model is not buying anything even where it was
  designed to be used.
- **ESM3 through Forge** (small, medium, large). Predictions exist only for the parent sets, so S461, K2369 and
  PTMUL-D are scored from S669, K3822 and PTMUL: every child row is found in its parent by (code, mutation) with
  the same reference sequence and the same PDB (461/461, 2,369/2,369, 536/536), and a zero-shot score is a function
  of structure and mutation alone. Only the *labels* are the child's own, and they are not all the parent's: S461
  revised 40 of its 461 labels relative to S669 (12 of them sign flips), PTMUL-D differs from PTMUL on 41 of 536.
  Larger is not better here: large is the worst of the three on every set, and medium (0.436) ≈ small (0.430)
  on the 12-set mean against 0.284 for large.
- **PTMUL for ESM3 is scored on all 914 rows** (571 doubles and 343 triple-or-higher mutants), like every other
  model. The Forge file has a score for every row but a label for only the 571 doubles, so its own statistics file
  reports 0.146 for ESM3-large on the doubles alone; against the full labels it is 0.017. The same applies to
  medium and small. The low value is real, not an indexing artefact: every one of the 914 Forge rows links to a
  source variant by sequence identity (its mutation string applied to the wild-type sequence reproduces a source
  mutated sequence, 571/571 doubles through 1/1 ten-mutants), and
  ESM3-large is genuinely anti-informative on the higher-order mutants (ρ −0.06 over the 343 with three or more
  substitutions, −0.28 at four) while positive on the doubles (0.145).

- **MutateEverything stage 2 on PTMUL (capped retrain vs earlier hyperopt-split retrain).** Across all 914 PTMUL rows,
  the three-seed capped-split stage-2 retrain scores 0.520 / 0.450 / 0.561 (mean 0.510, sd 0.056; stage 1: 0.533 / 0.415 /
  0.467, mean 0.472), compared with 0.558 on the older hyperopt split (§1.5; doubles 0.474 vs 0.503, triples 0.589 vs 0.677,
  4+ 0.582 vs 0.621). Beyond the stricter homology quarantine of `splits_oct06_capped`, the earlier hyperopt-split ME run
  had passed `ddg = +ddG_ML` (positive = stabilizing) instead of upstream ME's sign convention (positive = destabilizing);
  because ME's stage-2 loss subsamples ~25% of positive-signed doubles per epoch and weights negative-signed doubles $4\times$,
  the capped retrain restores upstream ME's intended stabilizing upweighting.

### 1.2 Tsuboyama

| model | capped val | capped test |
|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | **0.855ʰ** | 0.847ʰ |
| ESM-MSR (baseline), WT head (3 seeds) | 0.841ʰ | 0.829ʰ |
| ESM-MSR (devel), combined (3 seeds) | 0.854ʰ | **0.849ʰ** |
| ESM-MSR (devel), WT head | 0.837ʰ | 0.828ʰ |
| MutateEverything st1 (3 seeds) | 0.754ʰ | 0.771ʰ |
| MutateEverything st2 (3 seeds) | 0.730ʰ | 0.760ʰ |
| ThermoMPNN (additive, 3 seeds) | 0.787ʰ | 0.794ʰ |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.785ʰ | 0.784ʰ |
| SPURS (retrained, 3 seeds) | 0.803ʰ | 0.801ʰ |
| SPURS multi (retrained, 3 seeds) | 0.763ʰ | 0.772ʰ |
| ESMtherm | 0.716ᴹ | 0.774ᴹ |
| DDGemb (server) | 0.490 | 0.508ʰ |
| DDMut (server) | – | – |
| IFUM, WT backbone | 0.666ᴹ | 0.711ᴹ |
| IFUM, ESMFold per variant | 0.713ᴹ | 0.752ᴹ |
| GeoDDG-3D (FASPR) | 0.595 | 0.609ʰ |
| ESM3ΔG | 0.811ᴾ | 0.811ᴴ |
| ProSST (zero-shot) | 0.593 | 0.596 |
| ProteinMPNN (zero-shot) | 0.541 | 0.542 |
| ESM3-small-open, σ = 0 (zero-shot) | 0.565 | 0.578 |
| ESM3-large (Forge, zero-shot) | – | – |

*The legacy hyperopt val/test columns are omitted here: those rows predate the Oct06 split and are out of this audit's scope (their values are in §1.5).*
| model | capped val (leak code) | capped test (leak code) |
|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | h.20 V | h.16 |
| ESM-MSR (baseline), WT head (3 seeds) | h.20 V | h.16 |
| ESM-MSR (devel), combined (3 seeds) | h.20 V | h.16 |
| ESM-MSR (devel), WT head | h.20 V | h.16 |
| MutateEverything st1 (3 seeds) | h.20 V | h.16 |
| MutateEverything st2 (3 seeds) | h.20 V | h.16 |
| ThermoMPNN (additive, 3 seeds) | h.20 V | h.16 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | h.07 V | h.12 |
| SPURS (retrained, 3 seeds) | h.20 V | h.16 |
| SPURS multi (retrained, 3 seeds) | h.20 V | h.16 |
| ESMtherm | M.44 P.46 H.03 h.02 V.05 | M.51 P.34 H.04 h.03 V.06 |
| DDGemb (server) | – | h.02 |
| DDMut (server) | – | – |
| IFUM, WT backbone | M?.70 P.01 h.07 V?.28 | M?.46 P.02 H.07 h.06 V?.47 |
| IFUM, ESMFold per variant | M?.70 P.01 h.07 V?.28 | M?.46 P.02 H.07 h.06 V?.47 |
| GeoDDG-3D (FASPR) | – | h.07 |
| ESM3ΔG | P.19 H.30 h.24 | H.41 h.50 |
| ProSST (zero-shot) | – | – |
| ProteinMPNN (zero-shot) | – | – |
| ESM3-small-open, σ = 0 (zero-shot) | – | – |
| ESM3-large (Forge, zero-shot) | – | – |
- **SPURS and SPURS multi are retrained on the capped split and held out.** Three seeds each (recipe in §1.6): 0.803 / 0.801 on capped val / test for SPURS and 0.763 / 0.772 for SPURS multi, level with ThermoMPNN (0.787 / 0.794) and ahead of ME stage 1 (0.754 / 0.771), and 0.04–0.05 below the ESM-MSR baseline (0.855 / 0.847). They trained on exactly the libraries of ESM-MSR, ME and ThermoMPNN, so they carry the same leak codes (§6).
- **Among the models whose test set is held out** (ESM-MSR, MutateEverything and ThermoMPNN, all retrained on the
  same capped training split, plus the models that never saw Megascale), ESM-MSR leads: devel 0.849 and baseline
  0.847 on test, then ThermoMPNN 0.794, ME stage 1 0.771, **Rosetta 0.663**, GeoDDG 0.609, ProteinMPNN 0.542,
  DDGemb 0.508.
- **Rosetta Cartesian ΔΔG** (physics-based, no per-variant training; but the `beta_nov16` reference energies it runs on were partly fit to the stability of the Rocklin 2017 designs, and 7 of the 33 capped-test and 3 of the 23 val scaffolds are such designs: `V.12` / `V.07`, §6) scores 0.683 on validation and 0.663 on test at full
  coverage, above GeoDDG, ProteinMPNN, DDGemb and the zero-shot control and below every learned model trained on
  this data. The earlier published comparison had Rosetta at 0.666 / 0.634 on the older hyperopt split (§1.5),
  so the new run agrees.
- **The zero-shot control** (ESM3-small-open, σ = 0) reaches 0.578 on capped test, so the fine-tuning is worth
  about +0.27 ρ here too.
- **SPURS and ProteinMPNN on the mutant-background libraries.** Because wild-type PDB files carry the unmutated residue at stabilizing background positions (6,026 val and 7,931 test rows across 12 libraries, all single mutants relative to the background), the SPURS and ProteinMPNN inference adapters (`parse_pdb_with_seq`) overwrite the PDB-parsed sequence with the mutant-background reference sequence while retaining the wild-type backbone coordinates, verifying that the sequence differs from the PDB strictly at the declared background site. The SPURS adapter matches patched upstream SPURS to within $7 \times 10^{-6}$ across all Tsuboyama rows; on the 12 mutant-background libraries, per-library Spearman (seed means) spans 0.74–0.89 for SPURS and 0.71–0.83 for SPURS multi, versus 0.44–0.70 for ProteinMPNN.
- **IFUM is not held out either.** Its published split puts 21 of the 33 capped-test libraries in train and the other 12 in its unseparated val/test block; 25,296 test rows (49 %, including mutant-background variants of trained scaffolds) are on proteins it trained on, and 24,200 of them (46 %, `M?`) are the very Dataset2 rows of its train domains; another 24,608 rows (47 %) sit in its val/test block (`V?`). Both IFUM modes are `ᴹ`-flagged on the capped split, and its GB1 row is same-protein (IFUM trained on the Megascale 1PGA/GB1 library; only 111 GB1 variants are Megascale rows). Its per-variant ESMFold mode still beats its WT-backbone mode, but neither is a clean test.

### 1.3 DMS and Domainome

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome | Domainome (low-id, 458) |
|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | 0.737 | 0.737 | 0.675 | 0.627 | 0.638 | 0.480 | 0.318 | 0.543ᴴ | 0.536ʰ |
| ESM-MSR (baseline), WT head (3 seeds) | 0.733 | 0.736 | 0.670 | 0.636 | 0.637 | 0.462 | 0.341 | 0.541ᴴ | 0.534ʰ |
| ESM-MSR (devel), combined (3 seeds) | 0.745 | 0.742 | 0.681 | 0.636 | 0.639 | 0.481 | 0.337 | 0.543ᴴ | 0.536ʰ |
| ESM-MSR (devel), WT head | 0.751 | 0.754 | 0.684 | 0.655 | **0.640** | 0.478 | **0.364** | 0.548ᴴ | 0.542ʰ |
| MutateEverything st1 (3 seeds) | 0.707 | 0.681 | 0.646 | 0.581 | 0.580 | 0.483 | 0.299 | 0.520ᴴ | 0.511ʰ |
| MutateEverything st1, additive | 0.729 | 0.714 | 0.690 | 0.615 | 0.603 | 0.483 | 0.294 | – | – |
| MutateEverything st2 (3 seeds) | 0.645 | 0.635 | 0.605 | 0.598 | 0.560 | 0.439 | 0.253 | 0.509ᴴ | 0.499ʰ |
| MutateEverything st2, additive | 0.715 | 0.699 | 0.669 | 0.642 | 0.599 | 0.439 | 0.281 | – | – |
| ThermoMPNN (additive, 3 seeds) | 0.626 | 0.632 | 0.613 | 0.569 | 0.570 | 0.405 | 0.319 | 0.450ᴴ | 0.437ʰ |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.568 | 0.582 | 0.560 | 0.549 | 0.535 | 0.405 | 0.318 | – | – |
| SPURS (retrained, 3 seeds) | 0.715 | 0.729 | 0.674 | 0.589 | 0.583 | 0.449 | 0.308 | 0.489ᴴ | 0.477ʰ |
| SPURS multi (retrained, 3 seeds) | 0.658 | 0.673 | 0.576 | 0.545 | 0.539 | 0.335 | 0.333 | 0.393ᴴ | 0.383ʰ |
| ESMtherm | 0.093 | 0.136 | 0.484 | 0.466 | 0.229 | -0.000 | 0.214ᴾ | 0.201ᴾ | 0.183ᴾ |
| DDGemb (server) | 0.103 | 0.123 | 0.457 | 0.469 | 0.475ᴴ | 0.188 | 0.154 | 0.351ᴾ | 0.343ᴾ |
| IFUM, WT backbone | 0.766 | 0.768 | 0.648 | 0.565 | 0.507 | 0.190 | 0.262ᴾ | 0.464ᴾ | 0.459ᴹ |
| IFUM, ESMFold per variant | 0.762 | 0.747 | 0.697 | 0.575 | 0.523 | 0.217 | 0.284ᴾ | 0.473ᴾ | 0.467ᴹ |
| GeoDDG-3D (FASPR) | 0.351ᴾ | 0.431ᴾ | 0.604 | 0.481 | 0.395ᴾ | 0.240 | 0.200ᵐ | 0.413ᴾ | 0.412ᴾ |
| ESM3ΔG | 0.763 | 0.766 | **0.749** | 0.646 | 0.594ʰ | 0.280ʰ | 0.324 | 0.512ᴾ | 0.502ᴾ |
| ProSST (zero-shot) | **0.776** | **0.796** | 0.711 | 0.702 | 0.538 | 0.356 | 0.258 | 0.519 | 0.526 |
| ProteinMPNN (zero-shot) | 0.619 | 0.613 | 0.667 | 0.538 | 0.498 | **0.544** | 0.119 | **0.548** | 0.544 |
| ESM3-small-open, σ = 0 (zero-shot) | 0.689 | 0.742 | 0.627 | 0.723 | 0.536 | 0.483 | 0.259 | 0.540 | **0.545** |
| ESM3-large (Forge, zero-shot) | 0.624 | 0.606 | 0.557 | 0.582 | 0.458 | 0.432 | – | 0.510 | 0.516 |
| ESM3-medium (Forge, zero-shot) | 0.734 | 0.779 | 0.664 | – | – | – | – | – | – |
| ESM3-small (Forge, zero-shot) | 0.769 | 0.785 | 0.673 | **0.731** | 0.568 | 0.498 | – | – | – |

| model | DLG4 abund. (leak code) | DLG4 bind. (leak code) | GRB2 abund. (leak code) | GRB2 bind. (leak code) | MYO (leak code) | ESTA dTm (leak code) | GB1 (leak code) | Domainome (leak code) | Domainome (low-id, 458) (leak code) |
|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| ESM-MSR (baseline), WT head (3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| ESM-MSR (devel), combined (3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| ESM-MSR (devel), WT head | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| MutateEverything st1 (3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| MutateEverything st1, additive | – | – | – | – | – | – | – | – | – |
| MutateEverything st2 (3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| MutateEverything st2, additive | – | – | – | – | – | – | – | – | – |
| ThermoMPNN (additive, 3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | – | – | – | – | – | – | – | – | – |
| SPURS (retrained, 3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| SPURS multi (retrained, 3 seeds) | – | – | – | – | – | – | – | M.004 P.004 H.11 h.02 | h.03 |
| ESMtherm | – | – | – | – | – | – | M<.001 P1 V<.001 | M.001 P.05 H.37 h.04 V<.001 | M.002 P.04 H.30 h.04 V<.001 |
| DDGemb (server) | – | – | – | – | H1 | – | – | M<.001 P.03 H.17 h.07 | M<.001 P.03 H.19 h.08 |
| IFUM, WT backbone | – | – | – | – | – | – | M?<.001 P1 | M?.004 P.04 H.36 h.04 V?.001 | M?.005 P.04 H.29 h.04 V?.002 |
| IFUM, ESMFold per variant | – | – | – | – | – | – | M?<.001 P1 | M?.004 P.04 H.36 h.04 V?.001 | M?.005 P.04 H.29 h.04 V?.002 |
| GeoDDG-3D (FASPR) | P1 | P1 | – | – | M<.001 P1 | – | M<.001 m.41 P.59 | M.001 P.05 H.28 h.08 | M.001 P.06 H.31 h.08 |
| ESM3ΔG | – | – | – | – | h1 | h1 | – | P.005 H.22 h.51 | P.006 H.14 h.56 |
| ProSST (zero-shot) | – | – | – | – | – | – | – | – | – |
| ProteinMPNN (zero-shot) | – | – | – | – | – | – | – | – | – |
| ESM3-small-open, σ = 0 (zero-shot) | – | – | – | – | – | – | – | – | – |
| ESM3-large (Forge, zero-shot) | – | – | – | – | – | – | – | – | – |
| ESM3-medium (Forge, zero-shot) | – | – | – | – | – | – | – | – | – |
| ESM3-small (Forge, zero-shot) | – | – | – | – | – | – | – | – | – |
- **Mean over the 7 DMS** (models with all seven): ESM-MSR (devel) WT head 0.618, (devel) combined 0.609, ESM-MSR (baseline) 0.602, ProSST 0.591, ME st1 additive 0.590, ESM3ΔG 0.589, ESM3-small-open σ = 0 0.580, SPURS (retrained) 0.578, ThermoMPNN 0.533, SPURS multi (retrained) 0.523, ProteinMPNN 0.514, GeoDDG 0.386, DDGemb 0.281, ESMtherm 0.232. **The GB1 and MYO columns are not clean for all third-party models:** on GB1, ESMtherm and IFUM trained on the Megascale 1PGA/GB1 library and GeoDDG trained on 1,538 1PGA/GB1 single mutants in S8754 (61 of the Wu rows exact and 61,880 constituent); on MYO, GeoDDG trained on human myoglobin (`3RGK` in S8754; 3 exact rows, rest `P`) and DDGemb trained on sperm whale myoglobin (`H1`). ESM-MSR/ME/ThermoMPNN and the retrained SPURS are clean on all seven because GB1 and MYO were quarantined from the capped split. GB1 is the lowest of the seven for every model, so dropping it moves the means up by 0.00–0.07 and leaves the top two unchanged (ESM-MSR devel WT 0.660, devel 0.654) with ESM-MSR (baseline) third (0.649); ProSST moves from fifth to fourth, ahead of ESM-MSR (released).
- **Retrained SPURS on Domainome and GB1.** On Domainome (mean per-domain ρ) the retrained SPURS is at 0.489, ahead of ThermoMPNN (0.450) and behind ME stage 1 (0.520), ME stage 2 (0.509) and the ESM-MSR baseline (0.543); SPURS multi is at 0.393. On the 458 low-homology Domainome subset (`Domainome (low-id, 458)`), the ordering is unchanged: SPURS 0.477 ahead of ThermoMPNN (0.437) and behind ME stage 1 (0.511), ME stage 2 (0.499) and ESM-MSR baseline (0.536), with SPURS multi at 0.383. SPURS multi is below SPURS on every DMS column except GB1 (0.333 against 0.308).
- **The literature-ΔΔG models invert here:** GeoDDG and DDGemb, which lead the external sets, are near the bottom
  on the deep-mutational scans; ProSST, a zero-shot model, leads both DLG4 assays (0.776 and 0.796) among the
  models that cover all seven. ESM3-small through Forge is also ahead of ESM-MSR (devel) on the Faure assays it
  covers (DLG4 0.769 and 0.785, GRB2 binding 0.731, against 0.745, 0.742 and 0.636; it has no GB1).
- **These assays discriminate much less than the stability benchmarks do:** the zero-shot σ = 0 control (0.580) is
  within 0.04 of the best supervised model, against a 0.27 gap on Tsuboyama. Rankings this tight should not be
  read as model quality without interval estimates (§1.5 has them for the published subset).
- **Domainome is not clean either — companion low-homology column `Domainome (low-id, 458)`.** Domainome 1.0 evaluations and 3D homology checks use the 522 domain PDB structures and Supplementary Table 5 from Beltran et al. (*Nature* 2025; Zenodo record `14356805`, DOI `10.5281/zenodo.11043642`). Two Domainome domains match capped-train Megascale libraries at 100 % sequence identity: `O95155_PF04564_1229` (UBE4B = `3L1X`, 1,134 of 1,206 rows exact) and `Q9H161_PF00046_216` (ALX4 homeodomain = `2M0C`, 889 of 1,038 rows exact), giving 2,023 identical mutations (0.38 % of Domainome) in ESM-MSR's, ME's, ThermoMPNN's and SPURS's training data, plus 2,288 more same-protein rows (0.43 %) across four domains (`M.004 P.004`). Combined MMseqs2 sequence + Foldseek 3D structure homology flags 60 of the 522 domains (58,422 rows, 10.9 %, `H.11`; 40 by sequence plus 20 additional 3D structural homologs) as strong homologs of capped-train scaffolds and 12 domains (2.3 %, `h.02`) as weak homologs. ESMtherm (25,405 rows, 768 of them identical variants: `M.001 P.05`), IFUM (26,039 rows, up to 2,388 identical: `M?.004 P.04`), DDGemb (`P.03`) and GeoDDG (`P.05`) are same-protein on other domains, and ESM3ΔG has same-protein (≥ 90 % identity, none identical) homologs for 0.5 % of rows, strong ones for 22.2 % and weak ones for 51.2 % in its MGnify fine-tune. Removing the 4 `P` + 60 `H` capped-train overlap domains leaves **458 low-homology domains** (471,652 audit rows; weak `h.03` only across all capped-train models): supervised models drop by 0.006–0.013 ρ (ESM-MSR devel WT 0.548 → 0.542, baseline/devel combined 0.543 → 0.536, ME st1 0.520 → 0.511, ME st2 0.509 → 0.499, SPURS 0.489 → 0.477, ThermoMPNN 0.450 → 0.437, ESM3ΔG 0.512 → 0.502), while zero-shot models hold or rise (ESM3-small-open σ = 0: 0.540 → 0.545 combined / 0.544 → 0.549 WT head; ProteinMPNN 0.548 → 0.544; ProSST 0.519 → 0.526; ESM3-large 0.510 → 0.516).

#### ESM-MSR LoRA scale σ on the DMS and Domainome

ProteinGym showed that the right LoRA scale depends on the assay (§3), so the DMS and Domainome blocks are evaluated at
σ = 0.75, 0.5 and 0.25 for both checkpoints (`released` is the single released capped checkpoint; `devel` is the three-seed
mean; the combined head is shown).

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome | Domainome (low-id, 458) |
|---|---|---|---|---|---|---|---|---|---|
| released, σ = 1 | 0.728 | 0.727 | 0.673 | 0.620 | **0.639** | 0.478 | 0.345 | 0.542 | 0.535 |
| released, σ = 0.75 | 0.751 | 0.764 | 0.707 | 0.660 | **0.639** | 0.510 | 0.340 | 0.560 | 0.554 |
| released, σ = 0.5 | 0.762 | 0.791 | **0.718** | 0.696 | 0.627 | **0.526** | 0.342 | **0.572** | 0.567 |
| released, σ = 0.25 | 0.748 | 0.793 | 0.688 | 0.721 | 0.596 | 0.516 | 0.320 | 0.568 | **0.568** |
| devel, σ = 1 | 0.745 | 0.742 | 0.681 | 0.636 | **0.639** | 0.481 | 0.337 | 0.543 | 0.536 |
| devel, σ = 0.75 | 0.762 | 0.773 | 0.713 | 0.674 | **0.639** | 0.510 | 0.345 | 0.560 | 0.554 |
| devel, σ = 0.5 | **0.768** | **0.796** | 0.716 | 0.705 | 0.627 | 0.525 | **0.349** | 0.570 | 0.566 |
| devel, σ = 0.25 | 0.749 | 0.793 | 0.687 | **0.726** | 0.596 | 0.516 | 0.318 | 0.567 | 0.567 |
| σ = 0 (zero-shot control) | 0.689 | 0.743 | 0.627 | 0.723 | 0.536 | 0.483 | 0.260 | 0.540 | 0.545 |

- **Released, a lower LoRA scale is clearly better on these assays.** At σ = 0.5 the 7-DMS mean is 0.637 against
  0.601 at σ = 1 (0.624 at 0.75, 0.626 at 0.25), and σ = 0.5 beats σ = 1 on 7 of the 9 columns,
  including Domainome (0.572 against 0.542), Domainome (low-id, 458) (0.567 against 0.535; 0.568 at σ = 0.25), DLG4 binding (0.791 against 0.727) and GRB2 binding (0.696 against
  0.620; 0.721 at σ = 0.25). Only MYO and GB1 prefer σ = 1, and by at most 0.012. This is the same direction as
  ProteinGym's functional categories (§3), and at σ = 0.5 ESM-MSR (released) is ahead of ProSST (0.591) on the 7-DMS
  mean, while reaching 0.567–0.568 on the 458 low-homology Domainome subset (+0.022–0.024 over σ = 0 and ProteinMPNN).
- **The caveat that goes with it:** the best σ per assay is chosen by looking at these same assays, so these numbers
  are an upper bound on what tuning the scale buys, not a held-out result. The stability benchmarks (§1.1, §1.2) use
  σ = 1, fixed in advance, and a fair version of this comparison fixes one σ before looking.
- The σ = 0 control (LoRA off) shows the other end: it scores 0.723 on GRB2 binding, above both checkpoints at
  σ = 1, and ties ESM-MSR (devel) on DLG4 binding (0.742); fine-tuning on stability data is what costs these assays.
- **Devel follows the same shape.** Its 7-DMS mean is 0.641 at σ = 0.5, 0.631 at 0.75, 0.626 at 0.25 and 0.609 at
  σ = 1, so the intermediate scales beat σ = 1 by 0.017–0.032 and the best devel row (σ = 0.5, 0.641) edges the
  best released row (0.637). Devel at σ = 0.5 takes DLG4 abundance (0.768), DLG4 binding (0.796) and GB1 (0.349);
  released at σ = 0.25 takes GRB2 binding (0.726). The two checkpoints are level within about 0.01 on every column
  and on the mean, so the gain from a lower LoRA scale is a property of the scale, not of the checkpoint. At
  σ = 0.5 the devel row is also ahead of ProSST (0.591) on the 7-DMS mean, as the released row is.

### 1.4 ProteinGym (213 DMS)

| model | PG avg | Stability | DMS mean |
|---|---|---|---|
| ESM-MSR (released), combined | 0.510ᴹ | 0.870ᴹ | 0.555ᴹ |
| ESM-MSR (released), WT head | 0.507ᴹ | 0.828ᴹ | 0.550ᴹ |
| ESM-MSR (devel), combined (3 seeds) | 0.511ᴹ | 0.869ᴹ | 0.555ᴹ |
| MutateEverything st1 (3 seeds) | 0.468ᴹ | 0.808ᴹ | 0.506ᴹ |
| MutateEverything st2 (3 seeds) | 0.441ᴹ | 0.801ᴹ | 0.489ᴹ |
| ThermoMPNN (additive, 3 seeds) | 0.441ᴹ | 0.781ᴹ | 0.484ᴹ |
| SPURS (retrained, 3 seeds) | 0.462ᴹ | 0.805ᴹ | 0.501ᴹ |
| SPURS multi (retrained, 3 seeds) | 0.418ᴹ | 0.772ᴹ | 0.462ᴹ |
| ESMtherm | 0.213ᴹ | 0.720ᴹ | 0.283ᴹ |
| GeoDDG-3D (FASPR) | 0.383 | 0.638 | 0.422 |
| ESM3ΔG | 0.380ᴴ | 0.788ᴴ | 0.441ᴴ |
| ProSST (zero-shot) | **0.514** | **0.653** | **0.528** |
| Rosetta Cartesian ΔΔG | _0.375_ | _0.640_ | _0.409_ |
| ProteinMPNN (zero-shot) | 0.408 | 0.627 | 0.426 |
| ProteinMPNN, ProteinGym published recipe | 0.257 | 0.565 | 0.298 |
| ESM3-small-open, σ = 0 (zero-shot) | 0.448 | 0.620 | 0.470 |
| ESM3-open, ProteinGym published recipe | 0.466 | 0.641 | 0.486 |

| model | PG avg (leak code) | Stability (leak code) | DMS mean (leak code) |
|---|---|---|---|
| ESM-MSR (released), combined | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 |
| ESM-MSR (released), WT head | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 |
| ESM-MSR (devel), combined (3 seeds) | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 |
| MutateEverything st1 (3 seeds) | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 |
| MutateEverything st2 (3 seeds) | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 |
| ThermoMPNN (additive, 3 seeds) | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 |
| SPURS (retrained, 3 seeds) | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 | M.010 m.009 P<.001 H.002 h.003 |
| SPURS multi (retrained, 3 seeds) | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 | M.02 m<.001 H.002 h.003 |
| ESMtherm | M.02 P.02 H.004 h<.001 V.003 | M.02 P.02 H.004 h<.001 V.003 | M.02 P.02 H.004 h<.001 V.003 |
| GeoDDG-3D (FASPR) | n.a. | n.a. | n.a. |
| ESM3ΔG | P.003 H.02 h.33 | P.003 H.02 h.33 | P.003 H.02 h.33 |
| ProSST (zero-shot) | – | – | – |
| Rosetta Cartesian ΔΔG | V.002 | V.002 | V.002 |
| ProteinMPNN (zero-shot) | – | – | – |
| ProteinMPNN, ProteinGym published recipe | – | – | – |
| ESM3-small-open, σ = 0 (zero-shot) | – | – | – |
| ESM3-open, ProteinGym published recipe | – | – | – |
- **MutateEverything stage 2 on ProteinGym (3 seeds).** PG avg / Stability / DMS mean are 0.441 / 0.801 / 0.489 (seed sd 0.006 / 0.005 / 0.005). Each seed scored 214 of the 217 assays, including all 213 common assays plus `SCN5A` (`BRCA1` and `POLG_CXB3N` exceeded GPU memory in PyTorch 1.12, and `BRCA2` failed an upstream input-loader assertion). ME stage 2 sits below ME stage 1 (0.468 / 0.808 / 0.506) in every category (Binding −0.054, Expression −0.037, Activity −0.022, Organismal fitness −0.013, Stability −0.008), ties ThermoMPNN on PG avg (0.441), and exceeds SPURS multi by +0.023 / +0.029 / +0.027. On the 23 trained Stability assays it scores 0.900, and on the remaining 190 assays 0.439.
- **Retrained SPURS on ProteinGym.** PG avg / Stability / DMS mean are 0.462 / 0.805 / 0.501 for SPURS and 0.418 / 0.772 / 0.462 for SPURS multi (three seeds each). SPURS is ahead of ThermoMPNN (0.441 / 0.781 / 0.484), 0.003–0.006 behind ME stage 1 (0.468 / 0.808 / 0.506) and 0.05–0.07 behind ESM-MSR (released and devel); SPURS multi is 0.044 / 0.033 / 0.039 below SPURS.
- **ProSST ties ESM-MSR on the average only because of 23 fully memorised DMS.** 23 of the 217 ProteinGym assays (all 23 inside the 213 common multi-model intersection) are Tsuboyama DMS whose every row is in the capped train split used by ESM-MSR, ME and ThermoMPNN: 45,625 of 2,465,767 rows across all 217 DMS (1.9 % of the benchmark, 10.6 % of the 217 assays; 45,623 of 2,446,730 rows on the 213-DMS subset). They are all Stability assays and ESM-MSR scores 0.897 on them. Removing them across all 217 assays (ESM-MSR devel 3-seed mean, 194 DMS) moves ESM-MSR's DMS mean from 0.557 to 0.516, its Stability category mean from 0.880 to 0.871, and its PG average from 0.515 to 0.513 (on the 213-DMS common subset, 190 DMS: DMS mean 0.561 → 0.520, PG average 0.516 → 0.514); additionally removing the 3 Foldseek 3D structural-homolog Stability assays (`AMFR_HUMAN_Tsuboyama_2023_4G3O`, `RAD_ANTMA_Tsuboyama_2023_2CJJ`, `VG08_BPP22_Tsuboyama_2023_2GP8`; 4,607 rows, `H.002`) leaves 191 of 217 DMS (DMS mean 0.510, PG average 0.513; or 187 of 213 DMS, DMS mean 0.514, PG average 0.514). ProSST, zero-shot on those assays, is 0.528 on the all-213 DMS mean and 0.514 on the all-213 PG average; after removing the leaked assays the two PG averages are no longer directly comparable (§6.5). The same 23 DMS are in the training data of ME stage 1/2 (24,252 exact + 21,331 constituent) and ThermoMPNN (24,252 exact singles), and of the retrained SPURS (as ME stage 1) and SPURS multi (as ESM-MSR: 45,623 exact + 2 constituent).
- **Rosetta (italic) is singles only.** It covers ProteinGym's single mutants, 97.8 % of them scored, and 212 of the
  213 DMS (POLG_HCVJF has none). Its numbers are on a different variant set from every other row, which score all
  variants including multi-mutants, so they are shown but neither bolded nor compared.
- **Our ProteinMPNN is not the ProteinGym leaderboard recipe.** It scores the mutated position with the rest of the
  sequence visible (the unmasked WT/MUT conditional score) and averages 0.424 over all 217 DMS; the authors'
  published ProteinMPNN, which scores the whole sequence's likelihood, averages 0.295. The two track each other
  across DMS (rank correlation 0.90) and ours is higher on 208 of 217, so this is a different and stronger readout,
  not an error.
- **Published recipe against ours, for the two models we can read both ways.** ProteinGym's ProteinMPNN column
  averages 0.257 on the 213 common DMS (Stability 0.565, DMS mean 0.298) against 0.408 (0.627, 0.426) for our
  readout, so the choice of readout is worth +0.15 on the average. For ESM3 the direction reverses and the gap is
  small: the published ESM3-open column gives 0.466 (Stability 0.641, DMS mean 0.486) against 0.448 (0.621, 0.470)
  for our ESM-MSR readout with the LoRA off. Same weights in both rows (ESM3-small-open); the published scores are
  the authors' own and are scored here on exactly the variants every other row sees, all 213 DMS fully covered.
  The published ESM3 column is additive, like ProSST's (every multi-mutant equals the sum of its singles to 7e-15 over
  the same 1,499,350 variants), whereas the published ProteinMPNN column is not (differences up to 52), so part of
  the ESM3 comparison is a readout that cannot represent an interaction against one that can.
- **ESMtherm is Stability-only**, and the table shows how much: 0.720 on Stability against 0.213 on the five-category
  average (0.283 DMS mean), the opposite shape to ProSST.
- **ESM3ΔG completes the row stability-led, like the other stability-trained models.** 0.788 on Stability against a
  0.380 five-category average and 0.441 on the DMS mean. Measured as the Stability-minus-average gap that is 0.41,
  second only to ESMtherm's 0.51 and well above GeoDDG's 0.26; ProSST leans the same way but only by 0.14, so the
  row's shape is set by how much weight each model puts on the stability assays rather than by which side of zero.
- **GeoDDG inverts the way its external-set lead predicts.** It scores 0.383 on the five-category average with 0.638
  on Stability and 0.422 on the DMS mean: strong on the stability-led slice, weak on the functional assays, the
  mirror image of ESMtherm and the same inversion §1.3 records for the literature-ΔΔG models. Its mutation-level ProteinGym overlap
  composition has not been split into `M`/`m`/`P` (at the protein level its inventory, S8754 plus the 74 GeoFitness DMS proteins its encoder was pretrained on, shares 67 % `P` and 2 % `H` of ProteinGym rows; the GeoFitness proteins alone cover 74 assays and 38 % of rows), so its row in the leak-code table above is `n.a.` (not audited at mutation level, so not eligible for bold).
- **Every row in this table covers all 213 common DMS assays** (212 for single-mutant Rosetta, since `POLG_HCVJF` contains only multi-mutants). IFUM is omitted from the 213-DMS benchmark due to the computational cost of per-variant structure prediction across 2.45 million variants, and DDGemb and DDMut are rate-limited web servers.

**Row note:** `ESM-MSR (released)` in this table denotes the single released capped checkpoint (`n_seeds = 1`), while `ESM-MSR (devel)` reports the three-seed mean. The full LoRA-scale $\sigma$ sweep and per-category breakdown are in §3.

### 1.5 Published-paper evaluation (reference; earlier models and row sets)

Reference benchmark table for models trained on the earlier `hyperopt` split. Its external row sets are de-duplicated (`S8754` → 6,411 rows) and evaluate the curated subsets `S461` and `K2369` rather than `S669` and `K3822`, so comparisons should be made strictly within this table. Its Rosetta column reflects the earlier `hyperopt` Tsuboyama evaluation (consistent with the `splits_oct06_capped` evaluation in §1.2), and its MutateEverything columns correspond to the pre-October models trained with `ddg = +ddG_ML` (§1.1, §5 item 13).

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


### 1.6 SPURS retrained on the capped split

SPURS and SPURS multi were retrained on `data/splits_oct06_capped.pkl` so that they are held out identically to ESM-MSR, MutateEverything, and ThermoMPNN. Their rows in §1.1–§1.4, §2, and §3 report three-seed means (where the multi-mutant model of seed $s$ initializes from the single-mutant model of seed $s$) and carry the same leak codes as the other retrained models (§6). Tsuboyama is evaluated on **uncensored measured rows only** (`cens == 0`), and ProteinGym follows §1.4 / §3.

- **Data.** Capped training split only: 172 scaffolds (175 library entries including 3 mutant-background libraries), 154,358 single and mutant-background-single rows
  for SPURS; 52 libraries with doubles, 53,188 doubles, for SPURS multi (validation: 23 scaffolds, 26,410 singles, and 9 libraries / 8,547 doubles).
- **Recipe.** The upstream SPURS config (AdamW, lr 1e-4, weight decay 1e-3, plateau scheduler, early stopping with patience 30 on the pooled
  validation Spearman, one GPU), with the **pretrained ProteinMPNN `v_48_020` initialisation restored**: the public repository omitted loading it in later upstream revisions,
  and without it a retrain on SPURS's own `mega_splits` peaks at validation 0.741, against 0.809 for the released
  checkpoint and about 0.80 with the initialisation restored. The structural adapter (ESM layer 30) is left at its random initialisation,
  which matches the released checkpoints (it received no gradient in upstream training); one seed with the adapter trained gives
  the same results within noise (adapter bullet below). SPURS converged in 43, 67, and 46 epochs across seeds 1, 2, and 3. SPURS multi is initialised from the
  same-seed SPURS checkpoint, matching upstream protocol, and trained on doubles only (lr 1e-4, weight decay 1e-4, patience 10).
- **Epistasis.** Predicted epistasis is SPURS multi(AB) − SPURS(A) − SPURS(B), with the same-seed single model's predictions of A and B; the baseline's definition and the numbers are in §2.
- **Adapter trained or frozen** makes no measurable difference at this seed count (one trained-adapter seed: 0.798 / 0.799, 12-set 0.586, DMS 0.583, Domainome 0.494, against the frozen three-seed means; `docs/benchmarks/spurs_retrain_oct06.csv`).

The results are reported in the SPURS rows of §1.1 (12 external sets), §1.2 (Tsuboyama), §1.3 (DMS and Domainome), §1.4 (ProteinGym), and §2 (epistasis), with summary CSVs in [`benchmarks/spurs_retrain_oct06.csv`](benchmarks/spurs_retrain_oct06.csv), [`benchmarks/all_by_all_long.csv`](benchmarks/all_by_all_long.csv), and [`benchmarks/checkpoints.csv`](benchmarks/checkpoints.csv), and reproducible retraining code in `training_scripts/baselines/`.


## 2. Epistasis: ESM-MSR vs the retrained baselines and the third-party models

**Leak note (§6).** The capped val/test rows used here are clean for ESM-MSR, MutateEverything, ThermoMPNN and the retrained SPURS multi (identical mutations 0, same protein 0). ESMtherm trained on 51 % of the test rows as identical variants (verified) and IFUM on up to 46 % (`M?`); ESM3ΔG is zero-shot with 40.7 % strong/50.3 % weak homologs. Read the ‡ rows (documented Megascale training; §6.5 quantifies it) accordingly; the "Among models that did not train on Tsuboyama" comparison is the only like-for-like interaction comparison in this section. Model names carry the Tsuboyama leak superscript of §6.


**Rule throughout: neither baseline was modified, only its data layer.** ME follows the upstream README
recipe (stage 1 = singles, 20 epochs; stage 2 = doubles, 100 epochs from stage 1). ThermoMPNN is two
upstream configs, i.e. two models: `train_single.yaml` (singles) and `train_epistatic.yaml` (ThermoMPNN-D,
siamese; the measured doubles plus upstream's augmented doubles). Each baseline is scored by its direct double prediction against an additive expectation made
of a model that never trained on doubles, which is the role the WT head plays for ESM-MSR: ME stage 2 against the stage-1
singles of the same seed (predicted epistasis ME2(AB) − ME1(A) − ME1(B)), ThermoMPNN-D against the single ThermoMPNN's
(TMD(AB) − TM(A) − TM(B)), ME stage 1 (untrained double head) against its own singles; ESM-MSR by its combined head.
With ME stage 2's own singles as the baseline instead, test naive ρ is 0.477 rather than 0.405 and the other levels
move by up to 0.1 (§2.2); the stage-1 baseline is the one that parallels ESM-MSR and is used throughout. Rosetta scores a double directly with
`cartesian_ddg`, so it has both as well (evaluated through `src/esm_msr/epi_metrics.py` `assemble` and
`score`). So does every third-party model that scores a double itself (ESMtherm, IFUM in both modes,
GeoDDG-3D, ESM3ΔG, ProteinMPNN, DDGemb), each against the sum of its own singles. SPURS multi is scored like ME stage 2 against stage 1 and ThermoMPNN-D against ThermoMPNN: its double is SPURS multi(AB) and the baseline the sum of the same-seed SPURS single model's predictions of A and B, so predicted epistasis is SPURS multi(AB) − SPURS(A) − SPURS(B) (against the multi model's own singles the same predictions are nearly additive: validation naive ρ 0.083, flat pair and line levels). ProSST is left out because its double *is* the sum of its singles on every row, so it has no
interaction to score. That is the published method and not an adapter artefact: the paper's zero-shot score
(Appendix A.1, Eq. 8) is Σ over the mutated positions of log P(x_i = F_i | x, s) − log P(x_i = w_i | x, s) with x and
s the wild-type sequence and structure tokens, the authors' `zero_shot/proteingym_benchmark.py` does exactly that in
one forward pass, and their published ProteinGym predictions agree with the sum of their own singles to 2e-14 over
1,499,350 multi-mutants in 69 DMS. A non-additive reading of the same network (scoring the mutant sequence, or
conditioning each position on the other substitutions) is possible but is not the published recipe and was not run.
DDMut and UniMutStab have no predictions on this split. Models marked ‡ (EsmTherm, IFUM) trained on Megascale,
which includes these libraries' doubles, so they are shown but never bolded. `ctx` does not transfer to the
baselines (no separable adapter). Rows are models; seed means. Metric definitions: `validation_metrics.md`. The
table is regenerated by `analysis/scoring/benchmark_matrix.py epi`. The superscripts on the model names are the Tsuboyama leak classes of §6.

### 2.0 Extended epistasis metrics

Full twelve-metric epistasis suite (`src/esm_msr/epi_metrics.py`, `docs/validation_metrics.md`). For ESM-MSR, the
epistasis head is **`epi` = ~MT(AB) − WT(AB)**: the MT pass evaluated against the WT adapter's additive baseline,
which under `--per_mutation_bias` is additive by construction (the 50/50 `comb` head halves the epistatic contrast and
is reported separately for overall $\Delta\Delta G$ ranking in §2.3). Point-wise error and bias (`global_err`,
`global_bias`) evaluate predicted epistasis $\Delta\Delta\Delta G'$ directly against measured $\Delta\Delta\Delta G$ on
double mutants. Beyond the eight core columns shared with §2.1–2.2, this table reports `beyond_lib` (residual Spearman
after subtracting each library's mean residual, removing between-library offsets), `pair-avg ρ` (`beyond_rho_pairavg`,
unweighted mean within-position-pair Spearman across pairs with $\ge 20$ variants), three-class residual accuracy
(negative $< -0.6$, neutral, positive $> +0.6\text{ kcal/mol}$), and `global_bias`.

Every ESM-MSR and retrained baseline row (3 seeds per series) is scored from its saved predictions on the identical
populations used in §2.1–2.2: censored rows excluded, 8,494 validation doubles and 12,455 test doubles.

**Validation (valfull: 8,494 doubles, 24 pairs, 2,509 flips)**

| model | naive ρ | beyond ρ | beyond_lib | pair-avg ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | class3 acc | global_err ↓ | global_bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), 3 seeds | **0.654** | 0.586 | 0.434 | 0.377 | 0.745 | 0.395 | 0.309 | 0.193 | 0.845 | 0.683 | **0.717** | 0.177 |
| ESM-MSR (devel), 3 seeds | 0.643 | 0.585 | 0.437 | 0.406 | 0.734 | 0.434 | 0.315 | 0.195 | 0.854 | 0.657 | 0.824 | 0.275 |
| ESM-MSR (trim/mask), 3 seeds | 0.598 | **0.637** | **0.530** | **0.433** | **0.851** | **0.475** | **0.320** | **0.217** | **0.906** | 0.619 | 0.830 | **-0.117** |
| MutateEverything stage 2 (vs stage-1 singles), 3 seeds | 0.421 | 0.311 | 0.141 | 0.153 | 0.467 | 0.161 | 0.093 | 0.072 | 0.736 | 0.668 | 0.803 | -0.127 |
| ThermoMPNN-D (vs ThermoMPNN), 3 seeds | 0.190 | 0.085 | 0.108 | 0.150 | 0.060 | 0.203 | 0.048 | 0.023 | 0.655 | **0.692** | 0.979 | -0.435 |

**Test (12,455 doubles, 43 pairs, 4,236 flips)**

| model | naive ρ | beyond ρ | beyond_lib | pair-avg ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | class3 acc | global_err ↓ | global_bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), 3 seeds | 0.452 | 0.484 | 0.343 | 0.304 | **0.637** | 0.361 | 0.240 | 0.217 | 0.761 | **0.604** | 0.954 | 0.280 |
| ESM-MSR (devel), 3 seeds | **0.464** | **0.487** | 0.361 | 0.322 | 0.606 | 0.382 | 0.236 | 0.209 | 0.754 | 0.590 | 1.079 | 0.405 |
| ESM-MSR (trim/mask), 3 seeds | 0.444 | 0.460 | **0.398** | **0.374** | 0.589 | **0.417** | **0.266** | **0.224** | **0.803** | 0.567 | 0.937 | **0.083** |
| MutateEverything stage 2 (vs stage-1 singles), 3 seeds | 0.405 | 0.223 | 0.165 | 0.211 | 0.124 | 0.225 | 0.134 | 0.112 | 0.665 | 0.595 | **0.922** | 0.255 |
| ThermoMPNN-D (vs ThermoMPNN), 3 seeds | 0.077 | 0.071 | 0.124 | 0.166 | -0.003 | 0.197 | 0.022 | 0.016 | 0.601 | 0.578 | 1.030 | -0.224 |

On validation, the trim/mask seeds lead the interaction columns (`beyond ρ`, `beyond_lib`, pair-avg, pair, line,
`cell_mag`, `cell_rank`, `flipacc`) while trailing on naive ρ (0.598 vs 0.654), `class3 acc`, and `global_err`;
`beyond_lib` (0.530 vs 0.437) strips each library's mean residual, confirming that the gain is not a between-library
offset. On test, they lead `beyond_lib`, pair-avg, line, `cell_mag`, `cell_rank` (by less than one seed sd,
`benchmarks/seed_sd_by_table_cell.csv`), and `flipacc` while trailing on `beyond ρ` (0.460 vs 0.487, devel) and pair ρ
(0.589 vs 0.637, baseline), where the pooled correlation retains per-library level agreement that `beyond_lib` removes.
Negative `global_bias` indicates under-predicted epistatic shift relative to measured $\Delta\Delta\Delta G$. All models
in §2.1–2.2 are evaluated on the shared eight-metric subset using head `epi` (direct double prediction minus the
corresponding additive single-mutant baseline).

### 2.1 Validation, epistasis hierarchy (valfull: 34,957 rows, 8,494 doubles, 24 pairs, 2,509 flips)

| model | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| MutateEverything stage 1 ʰ | -0.024 | -0.007 | 0.124 | -0.010 | -0.016 | 0.007 | 0.517 | 1.189 |
| MutateEverything stage 2 ʰ | 0.421 | 0.311 | 0.467 | 0.161 | 0.093 | 0.072 | 0.736 | 0.803 |
| ThermoMPNN-D ʰ | 0.190 | 0.085 | 0.060 | 0.203 | 0.048 | 0.023 | 0.655 | 0.979 |
| Rosetta Cartesian ΔΔG ⱽ | 0.269 | 0.267 | 0.421 | 0.168 | 0.068 | 0.060 | 0.620 | 3.066 |
| ESMtherm ‡ ᴹ | 0.486 | 0.471 | 0.697 | 0.384 | 0.258 | 0.171 | 0.884 | 1.069 |
| IFUM, WT backbone ‡ ᴹ | 0.300 | 0.351 | 0.594 | 0.248 | 0.121 | 0.076 | 0.759 | 1.084 |
| IFUM, ESMFold per variant ‡ ᴹ | 0.292 | 0.316 | 0.657 | 0.189 | 0.097 | 0.076 | 0.771 | 1.067 |
| GeoDDG-3D (FASPR) | 0.137 | 0.112 | 0.240 | 0.080 | 0.089 | 0.073 | 0.736 | 1.277 |
| ESM3ΔG ᴾ | 0.305 | 0.447 | 0.587 | 0.346 | 0.211 | 0.149 | 0.856 | 1.044 |
| SPURS multi (retrained, 3 seeds) ʰ | 0.422 | 0.261 | 0.450 | 0.262 | -0.026 | 0.023 | 0.533 | 0.815 |
| ProteinMPNN (zero-shot) | 0.059 | 0.152 | 0.123 | 0.151 | 0.141 | 0.077 | 0.669 | 1.096 |
| DDGemb (server) | 0.164 | 0.370 | 0.728 | 0.143 | 0.062 | 0.045 | 0.616 | 1.295 |
| ESM-MSR (baseline) ʰ | **0.654** | **0.586** | **0.745** | 0.395 | 0.309 | 0.193 | 0.845 | **0.717** |
| ESM-MSR (devel) (3 seeds) ʰ | 0.643 | 0.585 | 0.734 | **0.434** | **0.315** | **0.195** | **0.854** | 0.824 |

### 2.2 Test, epistasis hierarchy (12,455 doubles, 43 pairs, 4,236 flips)

| model | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| MutateEverything stage 1 ʰ | 0.029 | 0.018 | -0.059 | 0.038 | -0.024 | -0.010 | 0.519 | 1.098 |
| MutateEverything stage 2 ʰ | 0.405 | 0.223 | 0.124 | 0.225 | 0.134 | 0.112 | 0.665 | 0.922 |
| ThermoMPNN-D ʰ | 0.077 | 0.071 | -0.003 | 0.197 | 0.022 | 0.016 | 0.601 | 1.030 |
| Rosetta Cartesian ΔΔG ⱽ | 0.381 | 0.207 | 0.258 | 0.195 | 0.094 | 0.066 | 0.556 | 3.083 |
| ESMtherm ‡ ᴹ | 0.495 | 0.507 | 0.635 | 0.550 | 0.250 | 0.178 | 0.790 | 0.806 |
| IFUM, WT backbone ‡ ᴹ | 0.002 | 0.247 | 0.315 | 0.254 | 0.107 | 0.068 | 0.617 | 1.113 |
| IFUM, ESMFold per variant ‡ ᴹ | 0.102 | 0.304 | 0.503 | 0.230 | 0.093 | 0.069 | 0.649 | 1.132 |
| GeoDDG-3D (FASPR) ʰ | 0.055 | 0.046 | 0.083 | 0.087 | 0.077 | 0.054 | 0.589 | 1.195 |
| ESM3ΔG ᴴ | 0.413 | 0.462 | 0.571 | 0.384 | 0.289 | 0.203 | 0.794 | 0.956 |
| SPURS multi (retrained, 3 seeds) ʰ | 0.308 | 0.278 | 0.399 | 0.270 | 0.009 | 0.041 | 0.522 | **0.883** |
| ProteinMPNN (zero-shot) | 0.314 | 0.273 | 0.473 | 0.125 | 0.125 | 0.064 | 0.592 | 0.992 |
| DDGemb (server) ʰ | 0.031 | 0.122 | 0.242 | -0.019 | 0.028 | 0.028 | 0.550 | 1.108 |
| ESM-MSR (baseline) ʰ | 0.452 | 0.484 | **0.637** | 0.361 | **0.240** | **0.217** | **0.761** | 0.954 |
| ESM-MSR (devel) (3 seeds) ʰ | **0.464** | **0.487** | 0.606 | **0.382** | 0.236 | 0.209 | 0.754 | 1.079 |

Largest seed sd per model, validation: ME stage 1 0.364, ME stage 2 0.077, ThermoMPNN-D 0.134, ESM-MSR (devel) 0.057; test:
ME stage 1 0.201, ME stage 2 0.062, ThermoMPNN-D 0.076, ESM-MSR (devel) 0.035. A difference smaller than these is not
a result.

*What the baseline rows are.* ME stage 2: predicted epistasis ME2(AB) − ME1(A) − ME1(B) with the stage-1 singles of the
same seed (with its own singles instead, test: naive ρ 0.477, beyond ρ 0.316, pair ρ 0.322, line ρ 0.283, global_err 0.745,
cell_rank and flip accuracy unchanged). ME stage 1: its untrained double head against its own singles, so about zero
by construction. ThermoMPNN-D: TMD(AB) − TM(A) − TM(B). Every external/retrained predictor is passed to `epi_metrics.score()` under head `epi` (direct double prediction minus the corresponding additive single-mutant baseline).

- **Summary of rankings across clean comparators.** A bold cell is the best value among the models
  without more than minor overlap with training (§6.6: no ‡, no `M`/`m`/`P`/`H` flag). On test that is (devel) naive ρ
  0.464, beyond ρ 0.487 and line ρ 0.382, and (baseline) pair ρ 0.637, cell_mag 0.240, cell_rank 0.217 and flip accuracy
  0.761, with the lowest global error among non-SPURS models going to ME stage 2 (0.922 vs 0.954 for ESM-MSR baseline; SPURS multi 0.883). The ‡ models
  and ESM3ΔG show higher raw values in several columns (ESMtherm leads line ρ at 0.550, ESM3ΔG cell_mag at 0.289),
  and stay unbolded because they trained on Tsuboyama or have strong homologs in
  training. On validation ESM-MSR leads all eight columns: (baseline) naive ρ 0.654, beyond ρ 0.586, pair ρ 0.745 and
  global error 0.717, (devel) line ρ 0.434, cell_mag 0.315, cell_rank 0.195 and flip accuracy 0.854.
- **Among models that did not train on Tsuboyama, ESM-MSR leads the interaction metrics.** On test, beyond ρ is
  0.487 (devel) and 0.484 (baseline) against 0.223 for ME stage 2, 0.273 for ProteinMPNN and 0.207 for Rosetta;
  cell_rank is 0.209–0.217 against 0.112 (ME stage 2) and 0.064–0.066 (ProteinMPNN, Rosetta). No model outside
  ESM-MSR reaches 0.21 on cell_rank. ME stage 2 is the best retrained baseline, and its predicted epistasis tracks
  the measured one (naive ρ 0.405 on test and 0.421 on validation, against 0.452–0.464 and 0.643–0.654 for ESM-MSR),
  which a model trained on doubles should do; naive ρ is mostly saturation, though, and its within-column order (cell_rank
  0.112, flip accuracy 0.665) is far below ESM-MSR's. ThermoMPNN-D's naive ρ is 0.077 on test and 0.190 on validation.
- **Megascale-trained predictors vs MGnify-trained ESM3ΔG.** On test ESMtherm trained on 51 % of the rows as identical variants (verified against its published split) and IFUM on up to 46 % (`M?`); both trained on the Megascale libraries this split uses (`‡`). By contrast, ESM3ΔG's released weights are fine-tuned on the authors' 960,216-sequence MGnify Stability set, not on Megascale: 0 benchmark row's wild-type or mutant sequence is in training, while 8 of the 33 libraries have a strong homolog (40.7 % of rows, `ᴴ`) and 21 a weak one (50.3 %). Its 0.811/0.811 is a genuine zero-shot result on Megascale with a homology caveat rather than training-set overlap.
- **Pair ρ is noisy for the models that are not trained here.** ProteinMPNN scores 0.123 on validation and 0.473 on
  test; DDGemb scores 0.728 (the best validation value of any model) and 0.242. There are only 24 (validation) and
  43 (test) pairs, so a few of them move it; read the within-column and flip metrics instead.
- **GeoDDG's doubles carry little usable interaction; the retrained SPURS multi carries some.** GeoDDG beyond ρ is 0.046 (test) and 0.112 (validation). SPURS multi (retrained, against the same-seed SPURS singles) has naive ρ 0.422 (validation) and 0.308 (test), beyond ρ 0.261 / 0.278 and pair ρ 0.450 / 0.399, but cell_mag near zero (−0.026, 0.009): it orders the interaction at the pair and line level and not within the cell. Its direct double correlates 0.999 with the sum of its own singles (seed 1, validation doubles), so this signal is the difference between the multi and the single model, as for ME stage 2 against stage 1 and ThermoMPNN-D against ThermoMPNN; against its own singles the same predictions score naive ρ 0.083 / 0.022 with flat pair and line levels.
- **Rosetta carries real interaction signal** from a physics model that knows nothing of this data: test beyond ρ
  0.207 and pair ρ 0.258, validation 0.267 and 0.421. That is above ThermoMPNN-D on beyond ρ, pair ρ and cell_mag on both
  halves (test 0.207 / 0.258 / 0.094 against 0.071 / −0.003 / 0.022; validation 0.267 / 0.421 / 0.068 against 0.085 /
  0.060 / 0.048). On test it is also above ME stage 2 on pair ρ (0.258 against 0.124), and below ME stage 2 and ESM-MSR
  on the other levels and on naive ρ (0.381 against 0.405 and 0.452–0.464).
- **What the ESM-MSR rows score.** They score the WT-referenced epistasis head `epi` (`~MT(AB) − WT(AB)`): on test (devel), beyond ρ is 0.487 ± 0.020, pair ρ 0.606 ± 0.034, and cell_rank 0.209 ± 0.006. The combined 50/50 prediction (`comb`) is evaluated for overall $\Delta\Delta G$ ranking in §2.3, and the combination weight $\lambda$ is analysed in §2.6.
- Additive heads score exactly 0 / 0.5 on every residual and rank metric, the suite's own correctness check, and
  `global_err` equal to the RMS of the measured dddG, since their predicted dddG is exactly 0 by construction.
- Every destabilized-backbone test row is a single, so these epistasis columns are identical with or without those
  libraries (max difference 0 over all 17 `epi_*` columns); only
  the pooled ddG metrics move, by up to 0.01 on `spearman_all`.

### 2.3 ddG ranking (mean per-protein Spearman)

Each ESM-MSR checkpoint is shown by its WT (additive) head and by its combined prediction, beside the retrained
baselines. Baselines are 3-seed means; the ESM-MSR columns are means over the three seeds of each series (their seed sd: `benchmarks/seed_sd_by_table_cell.csv`). TMPNN is the singles-only ThermoMPNN, which makes no pair-specific prediction, so it has no dddG: that is the
only gap. Regenerated by `analysis/scoring/benchmark_matrix.py ddg`. Validation numbers of all models are evaluated on the full 34,957-row `valfull` population.

| | ME st1 | ME st2 | TMPNN | TMPNN+D | ESM-MSR (baseline) WT ʰ | ESM-MSR (baseline) combined ʰ | ESM-MSR (devel) WT ʰ | ESM-MSR (devel) combined ʰ |
|---|---|---|---|---|---|---|---|---|
| val all variants (`spearman_all`) | 0.754 | 0.730 | 0.787 | 0.785 | 0.841 | **0.855** | 0.836 | 0.854 |
| val doubles (`spearman_doubles`) | 0.571 | 0.629 | 0.630 | 0.688 | 0.687 | 0.720 | 0.693 | **0.727** |
| val dddG of doubles (`spearman_doubles_epi`) | -0.013 | 0.341 | – | 0.225 | -0.023 | **0.505** | 0.009 | 0.492 |
| test all variants (`spearman_all`) | 0.771 | 0.760 | 0.794 | 0.784 | 0.829 | 0.847 | 0.827 | **0.849** |
| test doubles (`spearman_doubles`) | 0.617 | 0.652 | 0.673 | 0.668 | 0.709 | 0.766 | 0.717 | **0.781** |
| test dddG of doubles (`spearman_doubles_epi`) | 0.020 | 0.346 | – | 0.184 | -0.045 | 0.574 | -0.022 | **0.575** |

- **The combined prediction beats the WT head for both checkpoints on every metric, and almost all of the gain is
  the interaction term.** On test, all-variant ρ rises 0.829 → 0.847 (baseline) and 0.827 → 0.849 (devel); doubles
  0.709 → 0.766 and 0.717 → 0.781; and the dddG of doubles goes from about nothing (−0.045 ± 0.033 and
  −0.022 ± 0.026) to 0.574 ± 0.013 and 0.575 ± 0.007. Validation reads the same way (dddG −0.023 ± 0.031 → 0.505
  ± 0.010 baseline, 0.009 ± 0.068 → 0.492 ± 0.006 devel).
  The WT head carries no usable pair signal by itself, as an additive-style head should not.
- **Devel against baseline, combined:** test +0.002 on all variants, +0.015 on doubles, +0.001 on dddG; validation
  −0.001 on all variants, +0.007 on doubles and −0.013 on dddG (0.492 ± 0.006 against 0.505 ± 0.010). Both series
  carry a seed sd, and the dddG difference is 1.3 sd, so it cannot be called either way. The devel
  numbers are **on measured rows only** (§5.3); with bounded rows they would read 0.858 and look like +0.015.
- **Against the retrained baselines** an ESM-MSR combined column is the best value in all six rows. ME stage 2 is the
  best baseline on dddG (0.346 test, 0.341 validation, against the stage-1 singles) and ESM-MSR is ahead of it by 0.15–0.23;
  ThermoMPNN-D reaches 0.184 on test and 0.225 on validation, and ME stage 1 and the WT heads are near zero. With its own
  singles ME stage 2's dddG is 0.462 (test) and 0.438 (validation).

### 2.4 External sets, DMS, Domainome (baselines = 3-seed means, ESM-MSR = baseline triplicate)

- **12 external sets, mean ρ:** ESM-MSR (baseline) **0.646**, ME stage 1 0.615, ME stage 2 0.605, ThermoMPNN
  0.585, ThermoMPNN + D 0.586. ESM-MSR (baseline) is best of these on 8 of 12 (ME stage 1 takes s2648, s8754 and q3421, and s669 by 0.001); ESM-MSR (devel) is best on 10 of 12.
- **7 DMS** (`spearman_all`): the baseline triplicate leads the retrained baselines on DLG4 abundance (0.737 against
  ME stage 1+ 0.729), DLG4 binding (0.737 against 0.714) and MYO (0.638 against 0.603; ESM-MSR (devel) 0.639); ME's
  additive variants lead GRB2 abundance (stage 1+ 0.690 vs 0.675), GRB2 binding (stage 2+ 0.642 vs 0.627) and ESTA
  (0.483 vs 0.480, inside the seed sd of 0.007) — mostly by < 0.02. On GB1 the baseline combined row (0.318 ± 0.012)
  ties ThermoMPNN (0.319); the WT heads lead (baseline 0.341, devel 0.364). **The additive variants of ME and
  ThermoMPNN beat their own direct multi-mutant predictions** on the four Faure assays and MYO, by 0.02–0.07; ESTA is an
  exact tie and GB1 is mixed (ME stage 1 additive 0.294 against 0.299 direct).
- **dddG on DMS:** ME stage 2 (against the stage-1 singles) scores 0.477 and 0.462 on DLG4 abundance and
  binding, 0.349 on GRB2 abundance, 0.116 on GRB2 binding, 0.087 on MYO and 0.177 on GB1; ThermoMPNN-D (against
  ThermoMPNN) scores 0.203 and 0.192, 0.203, 0.034, 0.071 and 0.445 on the same six. So ME stage 2 and ESM-MSR are comparable on DLG4
  (≈ 0.45–0.47) and ESM-MSR leads GRB2 abundance (0.424 against 0.349 and 0.203), while ThermoMPNN-D leads GB1 (0.445
  against ESM-MSR's 0.316 and ME stage 2's 0.177). ME stage 1 is reliably negative.
- **Domainome** (singles): ESM-MSR (baseline) combined 0.543 ≈ WT head 0.541 > ME stage 1 0.520 > ME stage 2 0.509
  > ThermoMPNN 0.450 (ME stage 2: 0.512 / 0.509 / 0.505 over the three seeds; all 536,145 predictions merge onto
  the Domainome table, none missing). On the 458 low-homology subset (`Domainome (low-id, 458)`), the 3-seed ordering is ESM-MSR (baseline) combined 0.536 ≈ WT head 0.534 > ME stage 1 0.511 > ME stage 2 0.499 > SPURS 0.477 > ThermoMPNN 0.437.

### 2.5 Mutant-background ablation (nobb)

Dropping the 3 train / 5 val mutant-background libraries (all `native_cond` singles; zero doubles) and
re-scoring on the identical wild-type-backbone test rows:

| model | Δ spearman_all | Δ spearman_doubles | Δ beyond ρ | Δ pair ρ | Δ flipacc |
|---|---|---|---|---|---|
| MutateEverything stage 2 | **−0.017** | **−0.022** | +0.004 | +0.026 | +0.033 |
| MutateEverything stage 1 | +0.001 | −0.000 | +0.006 | +0.006 | −0.004 |
| ThermoMPNN | 0.000 | +0.005 | 0 | 0 | 0 |

Those 3,543 mutant-background training rows provide ME stage 2 a modest improvement in overall $\Delta\Delta G$ ranking on wild-type backbones (−0.017 / −0.022 when omitted) and no improvement on the interaction metrics. ThermoMPNN's row is unchanged (singles only); ThermoMPNN-D was not included in this comparison.


### 2.6 How much predicted epistasis should a multi-mutant prediction keep?

Write **A** = `wt_lora_pred` (additive by construction) and **M** = `mt_lora_pred` (the direct multi-mutant
head). Then

    (1 − λ)·A + λ·M  ≡  A + λ·(M − A)

so the ensemble weight **λ is exactly the fraction of the MT head's predicted interaction that is kept**. The
shipped `combined_pred = 0.5·(A + M)` is already λ = 0.5 — a 50 % shrinkage of predicted epistasis, not an
unweighted average of two independent models. The proposal `(N−1)/N · WT + 1/N · MT` is the schedule λ = 1/N.

Evaluated offline from the saved per-variant predictions (no GPU
needed, since every prediction file carries both heads and the mutation count). Multi-mutants only, ESM-MSR
(released):

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
   PTMUL −0.0140 [−0.0203, −0.0075] (released) and −0.0107 [−0.0173, −0.0041] (devel seed 1); GB1 −0.0044
   [−0.0065, −0.0023] (released), though devel seed 1 disagrees at +0.0111. Both PTMUL intervals exclude zero.
3. **The N-dependence points the other way.** On PTMUL the best λ *rises* with N — ρ(λ = 0.6) − ρ(λ = 0.3) is
   −0.0127 at N = 2, −0.0021 at N = 3 and +0.0269 at N = 4 (released). The N = 4 interval still includes zero
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

**Leak note (§6).** All 217 ProteinGym assays (2,465,767 rows) are evaluated in the leak and homology audit (`docs/benchmarks/leak_audit_pg_per_dms.csv`, `docs/benchmarks/leak_audit_structure_homology.csv`): 23 of the 217 assays (45,625 rows; all Stability, all 23 within the 213 common assays below) are inside the capped train split and 3 more (4,607 rows) are Foldseek 3D structural homologs (§6.5). The σ sweep below is unaffected in ranking (a lower σ is better on exactly these Stability assays, so removing them strengthens rather than weakens the σ = 0.75 conclusion) but its absolute values are upper bounds for ESM-MSR.


213 DMS that every model scored (BRCA1, BRCA2, POLG_CXB3N and SCN5A excluded: > 1,800 residues, exceeding GPU memory for at least one evaluated predictor). **PG avg** = mean of the five category means (a plain mean over DMS within a
category, not ProteinGym's per-UniProt weighting); **DMS mean** = mean over the 213. Unmasked scoring,
ProteinGym's own AF2 structures. ProSST is the authors' published ProSST-2048 scores, not a rerun. Rosetta is
italic because its predictions cover single mutants only (212 of the 213 DMS) and the rest of the column does not.

| model | Activity (43) | Binding (13) | Expression (18) | Org. fitness (73) | Stability (66) | PG avg | DMS mean |
|---|---|---|---|---|---|---|---|
| ESM-MSR (devel), σ = 0.75 (3 seeds) ᴹ | 0.439 | 0.364 | 0.515 | 0.410 | 0.858 | 0.517 | 0.560 |
| ProSST (zero-shot; published scores) | **0.486** | **0.449** | **0.530** | **0.453** | **0.653** | **0.514** | **0.528** |
| ESM-MSR (devel), σ = 0.5 (3 seeds) ᴹ | 0.443 | 0.381 | 0.499 | 0.413 | 0.822 | 0.512 | 0.551 |
| ESM-MSR (released), WT head only, σ = 0.75 ᴹ | 0.447 | 0.362 | 0.512 | 0.424 | 0.815 | 0.512 | 0.554 |
| ESM-MSR (devel), σ = 1 (3 seeds) ᴹ | 0.426 | 0.347 | 0.517 | 0.395 | 0.869 | 0.511 | 0.555 |
| ESM-MSR (released), σ = 1 ᴹ | 0.425 | 0.345 | 0.515 | 0.394 | 0.870 | 0.510 | 0.555 |
| ESM-MSR (released), WT head only, σ = 1 ᴹ | 0.434 | 0.349 | 0.515 | 0.411 | 0.828 | 0.507 | 0.550 |
| ESM-MSR (released), WT head only, σ = 0.5 ᴹ | 0.449 | 0.377 | 0.493 | 0.425 | 0.780 | 0.505 | 0.543 |
| ESM-MSR (devel), σ = 0.25 (3 seeds) ᴹ | 0.430 | 0.385 | 0.468 | 0.400 | 0.742 | 0.485 | 0.517 |
| ESM-MSR (released), WT head only, σ = 0.25 ᴹ | 0.436 | 0.381 | 0.466 | 0.411 | 0.713 | 0.481 | 0.512 |
| MutateEverything single (3 seeds) ᴹ | 0.371 | 0.336 | 0.472 | 0.352 | 0.808 | 0.468 | 0.506 |
| SPURS (retrained, 3 seeds) ᴹ | 0.373 | 0.320 | 0.472 | 0.340 | 0.805 | 0.462 | 0.501 |
| ESM-MSR, σ = 0 (no LoRA) | 0.416 | 0.374 | 0.439 | 0.391 | 0.620 | 0.448 | 0.470 |
| ThermoMPNN (3 seeds) ᴹ | 0.351 | 0.299 | 0.439 | 0.337 | 0.781 | 0.441 | 0.484 |
| MutateEverything double (3 seeds) ᴹ | 0.349 | 0.283 | 0.434 | 0.339 | 0.801 | 0.441 | 0.489 |
| SPURS multi (retrained, 3 seeds) ᴹ | 0.333 | 0.269 | 0.413 | 0.303 | 0.772 | 0.418 | 0.462 |
| ProteinMPNN (3 decoding orders) | 0.343 | 0.323 | 0.437 | 0.310 | 0.627 | 0.408 | 0.426 |
| Rosetta Cartesian ΔΔG (*singles only*) | _0.311_ | _0.291_ | _0.336_ | _0.296_ | _0.640_ | _0.375_ | _0.409_ |

Paired against the released checkpoint over the 213 common DMS (3 seeds × 4 σ, every DMS scored):

| devel setting | PG avg | paired Δ vs released (mean over DMS) | better on | seed sd (PG avg) |
|---|---|---|---|---|
| σ = 1 | 0.5107 | +0.0003 | 112 / 213 | 0.0003 |
| **σ = 0.75** | **0.5170** | **+0.0053** | **135 / 213** | 0.0007 |
| σ = 0.5 | 0.5115 | −0.0043 | 102 / 213 | 0.0007 |
| σ = 0.25 | 0.4853 | −0.0380 | 71 / 213 | 0.0005 |

- **Devel at σ = 1 matches released** and **σ = 0.75 is the only setting that beats it** (+0.0053, better on 135 of 213,
  paired t = 3.04). Seed sd ≤ 0.0007, so the σ = 0.75 gain is ~8 seed-sds and the σ = 1 null is real.
- **The σ trend:** Stability falls monotonically as σ drops (0.869 → 0.742) while Binding rises monotonically
  (0.347 → 0.385), with Activity and Organismal fitness peaking near σ = 0.5. Scaling the LoRA down trades
  stability skill for general fitness, and σ = 0.75 is where the two balance.
- **ProSST wins four of five categories outright** (Activity, Binding, Expression, Organismal fitness) against every ESM-MSR setting, not just σ = 1; the 0.22 it concedes on Stability is what brings the averages level (§1.4). **Under the audit the Stability comparison is the contaminated one:** all 23 fully trained DMS are Stability assays, so the category where ESM-MSR wins by 0.22 is the only one where it has direct training data. Removing them puts ESM-MSR's Stability at 0.871 and ProSST's 0.653 unchanged; the four-category lead is untouched and the global tie is restored without ESM-MSR's training advantage.
- **ProteinMPNN test-time ensembling:** averaging three decoding orders yields +0.0003; a 0.2 Å backbone-noise
  ensemble of 8 yields +0.0017 on a 73-DMS subset (50/73 up, t = 1.29).


## 4. Third-party predictor notes

- **The external-set leaders trained on that kind of data.** S2648, S8754, Q3421, K3822 and Ssym overlap the
  training data of DDGemb, DDMut and GeoDDG, which is why they lead §1.1 and fall behind on Tsuboyama and the DMS.
- **IFUM has two modes.** §1 reports the WT-backbone mode. Its native mode folds every variant with ESMFold and
  scores it on its own structure, which is slightly better on the hyperopt Tsuboyama split (val 0.778 vs 0.746,
  test 0.738 vs 0.697). For the capped split, 64,008 mutant sequences (mean length 58, ~6 GPU-hours) were folded with ESMFold prior to scoring (§2.1–2.2).
- **Rosetta Cartesian ΔΔG protocol:** Rosetta 2019, FastRelax in cartesian mode (`beta_nov16_cart`; its 20 reference
  energies were fit partly to Kellogg-2011 ProTherm ΔΔG ranks and Rocklin-2017 design stabilities, coded `V`, §6), mutant
  backgrounds built first for the 12 mutant-background libraries, then `cartesian_ddg` with 3 iterations over 87,506
  unique mutations in 8,775 chunks, doubles scored directly. All 87,544 capped rows have a prediction.
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

1. **GB1 uses the complete 149,360-row table in `data/preprocessed/`.** All 76 single mutants of the 4-site saturation library (including V39A/C/D) are present so that all 2,091 scorable double mutants have measured constituent singles.
2. **`dG_wt` is defined by the exact library identity `dG_ML − ddG_ML`,** avoiding per-library constant offsets that arise if `dG_wt` is approximated by averaging noisy `wt` replicate rows across Tsuboyama libraries.
3. **Rows the assay only bounded are predicted but not reported.** `<-1` / `>5` rows are 1.5 % of val and 6.4 % of
   test; their `ddG_ML` is a library bound rather than a measurement, and they sit at the extremes where ranking
   is easy, so including them inflates ρ by about 0.01.
4. **Cross-model validation comparisons use `valfull` (34,957 rows).** Every model in §1.2, §2.0, §2.1 and §2.3 is evaluated on the identical 34,957-row uncensored validation population without training-time item filters.
5. **Test-set ddG tables exclude destabilized backbones** (`--test_backbones wt_only`, 40,866 rows): 7 of 37 test
   libraries carry a background mutation that only the models trained on that pattern have seen. Every such row is
   a single, so the epistasis columns are identical either way (max abs difference 0 over every `epi_*` column).
6. **Ground truth is always ESM-MSR's `ddG_ML`.** Upstream MutateEverything flips the sign of its ground truth when more than half of `ddg` is negative, which would invert Ssym and GB1 if not standardized.
7. **External sets are aligned by subset relation:** S461 ⊂ S669, K2369 ⊂ K3822, PTMUL-D ⊂ PTMUL, so a model only
   has to predict the superset. The link is exact for *predictions* (same code, mutation, reference sequence and PDB
   on every child row) but not for *labels*: S461 revised 40 of its 461 labels (12 sign flips) and PTMUL-D differs
   from PTMUL on 41 of 536, so a child is always scored against its own labels.
8. **Inference outputs latent ddG.** `esm_msr_testing.py` writes calibrated *latent* ddG; rank metrics within a dataset are invariant to any monotone observation map.
9. **Check the label column, and link by variant, before trusting a statistic from someone else's prediction
   file.** The ESM3 Forge files carry no label for 343 of PTMUL's 914 rows (all triple-or-higher mutants), so their
   own statistics silently score the 571 doubles; the numbers here score all 914 against the harness labels. Their
   `id` column cannot be used to join: it matches the source's PDB-numbered `uid` on only 283 rows and the
   sequence-numbered key on 343, so rows are linked by mutation string plus PDB code and verified against the
   source's wild-type and mutated sequences (which carry both numberings).
10. **SPURS is two models** (§1.1); the single-model and multi-model rows are both reported, from the same tracked run of the two same-seed models.
11. **Every model is scored on the mutant-background libraries.** SPURS and ProteinMPNN use the mutant-background sequence (§1.2), leaving wild-type-background groups untouched. The epistasis tables are unaffected either way, because every mutant-background row is a single.
12. **ESM-MSR calibration convention (`per_mutation_bias = True`, independent WT/MT scales).** Calibration bias is applied once per mutated position ($k \times \text{bias}$ for a $k$-mutant), making the WT pass exact-additive across mutation order ($\text{WT}(AB) = \text{WT}(A) + \text{WT}(B)$). Within a fixed mutation order $k$, rank metrics are invariant to a constant shift; across mixed-order datasets (such as ProteinGym and PTMUL), per-mutation bias ensures consistent additive scaling across single and higher-order mutants.
13. **Retrained baseline conventions.** (a) MutateEverything stage 1/2, ThermoMPNN, ThermoMPNN-D, SPURS, and SPURS multi were trained on all valid uncensored rows of the 172 capped-train libraries (`splits_oct06_capped.pkl`), without item-level epistasis filters (§6.3). (b) Retrained MutateEverything uses upstream ME's sign convention (`ddg = −ddG_ML`, positive = destabilizing) so stage-2 loss subsampling and $4\times$ stabilizing weighting match the published recipe (§1.1). (c) Each multi-mutant baseline's predicted epistasis is its direct double prediction minus an additive expectation from a same-seed model that never trained on doubles (ME stage 2 vs stage 1, ThermoMPNN-D vs ThermoMPNN, SPURS multi vs SPURS), passed to `epi_metrics.score()` under head `epi`.

## 6. Data-leakage audit (October 2026; revised 2026-11-12)

Full method, evidence tiers and the revision log are in [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md); this section summarises it.

**Scope.** Every model row in §1–§3, every dataset column, and the capped Oct06 split. The legacy hyperopt split
(§1.2 hyperopt columns, §1.5) is out of scope: it predates the audit and is not the split the headline results use.
The audit answers two questions per cell — *was the exact variant trained?* — and, if not, *was the same protein, a
homolog, or the source dataset trained?* — and records the answer as a code. It also validates the homology tooling
the split was built with, and measures the overlap the split was never designed to catch (ProteinGym, Domainome).

### 6.1 Method

1. **Normalise every test row** to a protein sequence and a set of 1-based `(WT, position, mutant)` substitutions
   (all 2,465,767 ProteinGym rows across all 217 DMS, 534,385 Domainome, 149,360 GB1, 87,544 Tsuboyama capped val+test, 25,632 external-set
   rows, 7 DMS assays; 3,382,799 rows in all). Every protein is aligned to its same-protein family representative (≥95 % identity, ≥90 %
   of shorter-sequence coverage, single linkage) so mutations can be compared across PDB entries, chain numbering and construct
   differences.
2. **Reconstruct each model's training inventory** from published papers, repositories, and released split manifests
   (Table 6.3). Inventories are recorded at the mutation level
   where the data exist (ESM-MSR, ESM-MSR-small, EsmTherm through the sequences of its published split, DDGemb, DDMut,
   GeoDDG, ESM3ΔG) and at the protein level where only domains or proteins are public (IFUM; GeoDDG's GeoFitness
   pretraining). The retrained baselines train on a subset of ESM-MSR's capped-train rows and inherit its codes.
3. **Classify each test row** as exact (`M`), constituent (`m`: all its mutations were trained in the same protein but
   not this combination), same protein (`P`), strong homolog (`H`), weak/ambiguous similarity (`h`), documented-only (`D`),
   or none (`–`, including zero-shot / unsupervised pretraining-only models). Homology is evaluated over *all* qualifying
   MMseqs2 hits partitioned by query-coverage tier, and on Domainome and ProteinGym is augmented with exhaustive Foldseek 3D
   structural alignments (`07_structure_homology.py`). The classification is per row, then aggregated per dataset.
4. **Validate the split's homology tools** on the pair table the split was actually built from (408,330 pairs across
   370 Megascale libraries and all 776 external nodes) and re-run MMseqs2/Foldseek on the benchmark structures (§6.4).

### 6.2 Notation

| code | meaning | evidence |
|---|---|---|
| `M` | identical variant row in training (same protein family, identical mutation set, including reverse structures) | verified |
| `m` | every mutation in the row appears in training in the same protein, but not this combination (e.g. a test double whose two singles were trained) | verified |
| `P` | same protein (≥95 % identity over ≥90 % shorter-sequence coverage, or ≥90 % identity over ≥50 %) in training, no mutation match | verified |
| `H` | strong homolog in training: MMseqs2 E ≤ 1e-3 over ≥50 % of the test protein (plus Foldseek E ≤ 1e-3 or qTM > 0.70 over ≥50 % on Domainome and ProteinGym) | verified |
| `h` | weak/ambiguous: MMseqs2 E ≤ 1e-2 or ≥30 % identity over ≥30 % of the test protein (plus Foldseek E ≤ 1e-2 over ≥30 % on Domainome and ProteinGym) | measured |
| `D` | documented training on the same source dataset; rows not reconstructable | documented |
| `–` | no supervised ΔΔG training overlap (includes physics-based and unsupervised pretraining-only models) | verified / documented |
| `V` | the rows were used for validation, model selection or calibration (Rosetta's energy-function tuning) | split file / paper |
| `#` | the number is not a test result (the dataset is the training set) | verified |
| `?` | ceiling, not verified: inferred from a published protein-level split because the training rows are not public (IFUM `M?`, `V?`) | inferred |
| `n.a.` | not audited: no statement about overlap is made, and the cell is not eligible for bold | – |

Fractions are of the dataset's reported rows. A cell carries every class that occurs, e.g. `M.59 P.35 H.02`.
The numeric tables keep only the worst class in a superscript; the matrix below each table keeps the full code.
Superscripts below 0.5 % of rows are dropped in the numeric tables and retained in the companion matrix.

### 6.3 Training inventories

| inventory | training data | source | evidence |
|---|---|---|---|
| ESM-MSR (released), ESM-MSR (devel), σ sweep, WT/MT heads | all substitutions of the 172 capped-train libraries (229,721 rows) | `splits_oct06_capped.pkl` + `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | verified |
| MutateEverything st1 | the 172 capped-train libraries: 154,358 valid singles (150,815 wild-type-backbone + 3,543 mutant-background) | capped split; ME README recipe, stage 1 (20 epochs) | verified |
| MutateEverything st2 | the same singles plus all 53,188 measured doubles (207,546 rows); stage 2 (100 epochs) initialised from the same seed's stage 1 | capped split; ME README recipe | verified |
| ThermoMPNN (additive) | the same 154,358 singles | capped split; `train_single.yaml` | verified |
| ThermoMPNN-D | 134 of the 172 capped-train libraries: the 53,188 measured doubles (52 libraries) plus 118,963 Rosetta-augmented doubles (82 additional natural libraries) | capped split; `train_epistatic.yaml` (October audit; the retrained-baseline inventories were retired from the pipeline on 2026-11-12, LEAKAGE_AUDIT_OCT2026.md §11) | verified |
| SPURS (retrained) | the 172 capped-train libraries: 154,358 valid singles (150,815 wild-type-backbone + 3,543 mutant-background) | capped split; patched SPURS training code, three seeds | verified |
| SPURS multi (retrained) | the same singles (through its same-seed SPURS initialisation) and the 53,188 measured doubles | capped split; patched SPURS training code, three seeds | verified |
| EsmTherm | `K50_dG_Dataset1_Dataset2`: 47 test-only domains; the other 414 split 80/10/10 per sequence (527,785 sequences with split labels) | S2 Spreadsheet `pcbi.1012248.s006.csv` (PLoS Comput Biol 2024) + `wildtype_split.csv` | verified (exact-sequence join) |
| IFUM | Megascale Dataset2, MMseqs2-30 % cluster split: 322 train domains, 86 in an unseparated val/test block | IFUM published `dataset_splits.pdf` | protein-level (`M?` / `V?` ceilings from the public Dataset2 rows of those domains) |
| DDGemb | S2450 = S2648 − 18 sequences/198 mutations >25 % identity to S669; 2,450 mutations / 115 UniProt, doubled with reverse variants | `ddgemb.biocomp.unibo.it` `S2450.tsv` + `S2450.fasta` | verified (sequence-aligned, forward + reverse) |
| DDMut | single-mutant model: S9028 = S2648 + FireProtDB, 4,514 forward + 4,514 reverse; deployed multi-mutant model: SM1218 training + SM444 blind test (831 + 831 reverse) | `biosig.lab.uq.edu.au` `S9028.csv`, `DD_SM1218.csv`, `DD_SM444.csv` | verified (forward + reverse, union of both models) |
| GeoDDG-3D | S8754 = 8,754 single mutations / 301 proteins (ProThermDB + ThermoMutDB); encoder pretrained as GeoFitness on 74 DMS fitness sets (52 MaveDB + 22 DeepSequence) | `GeoStab` `S8754.csv`; `data/dms/dms.csv` | verified (construct-aligned); GeoFitness protein-level |
| ESM3ΔG | 960,216-sequence **MGnify Stability** fine-tune (525,127 WT + 435,089 point mutants); Megascale is its held-out test | Zenodo 19411306 `mgnify_training_index.csv` | verified (0 exact sequences) |
| ProSST | 18.8 M AlphaFoldDB structures (masked-LM); no ΔΔG fine-tune (`–`) | ProSST paper | documented only |
| ProteinMPNN | PDB assemblies deposited before 2021-08-02; no mutation-level supervision (`–`) | ProteinMPNN paper | documented only |
| ESM3 (small/medium/large, and the σ = 0 control) | UniRef/MGnify/JGI + PDB pre-2020-05-01 + AFDB v4; no mutation-level supervision (`–`) | ESM3 paper | documented only |
| Rosetta Cartesian ΔΔG (2019, `beta_nov16_cart`) | no per-variant training; 20 reference energies fit with Kellogg-2011 ProTherm ΔΔG ranks and Rocklin-2017 design stabilities in the objective, REU → kcal/mol 1/2.94 fit on Kellogg | Rosetta docs `Updates-beta-nov16`; Park et al. 2016 JCTC SI | calibration (`V`) |
| ESM-MSR-small (March reference) | the 120 train libraries of the legacy hyperopt split; selected on its 26 val libraries | `data/hyperopt_splits.pkl` | verified |

Two key findings follow from this table. **ESM3ΔG is not Megascale-trained** (its released
`*_augmented_*` weights are fine-tuned on the authors' MGnify Stability set; 0 benchmark row's wild-type or mutant
sequence is in it, and 8 of the 33 capped test libraries have a strong homolog), and **S8754 is GeoDDG's training set**, not a held-out external set.

**How the retrained baselines were built.** *Tables:* the standard ESM-MSR preprocessor with `censor_range`
off and no SPURS override, on `splits_oct06_capped.pkl`. A row survives only if its `ddG_ML` is numeric; in the raw
table that means the mutant and the wild type both lie inside dG −1 to 5, so no bounded row enters. Replicates are
averaged; `wt`, insertion and deletion rows and doubles carrying a wild-type-to-wild-type "hidden single" are dropped
(6,434 train rows); nothing else is: no `dG_wt + ddG_A + ddG_B` rule, no requirement that a double's singles be
measured (ME's loss masks such doubles by itself), no rank-only items, no floor. Rows: train 207,546 = 150,815 singles +
3,543 mutant-background singles + 53,188 doubles; val 34,957; test 48,740. All 172 / 23 / 33 libraries of the split are
covered, and every singles row of the raw table that survives the filters is present (154,358 in train). *Recipes:* ME
stage 1 for 20 epochs on singles and stage 2 for 100 epochs from stage 1 (3 GPUs, batch 1, lr 3e-4; stage 1 λ_single 1
and λ_double 0, stage 2 λ_single 0.1 and λ_double 1); ThermoMPNN's `train_single.yaml` and `train_epistatic.yaml` unchanged, with the checkpoint chosen by best validation
Spearman as upstream does; seeds 1, 2, 3 for every stage. Mutant-background libraries are libraries of their own, with
their own sequence and, for ME, their own MSA. *What still differs from upstream:* (i) the ThermoMPNN-D augmentation pool
holds only the singles that have a Rosetta model, 118,963 of 150,815 (19,170 of 20,474 in validation); the 36 designed
libraries have none, and upstream's own splits never contained them; (ii) replicates are averaged, where upstream ME
keeps duplicates and upstream ThermoMPNN-D keeps the first replicate of a double.

### 6.4 Do the split's homology tools hold up?

The split was built with `preprocessing/split_tsuboyama.py` and `preprocessing/homology.py` (MMseqs2 + Foldseek +
a BLOSUM62 local-identity cap), and its edge rule is: a library may not be split away from a protein if **any** of
MMseqs2 E ≤ 1e-3, Foldseek E ≤ 1e-3, identity > 0.40, or TM-score > 0.70 holds. Over all 370 Megascale libraries (172 train, 23 val, 33 test, 142 excluded) and
all 776 external quarantine nodes (771 structures from the 9 classical sets + 5 functional DMS structures), the saved
pair table contains 408,330 pairs and 25,027 default homology edges (`09_split_audit.py`).

- **The rule is enforced with zero leaks.** Not a single edge joins two different splits or a kept split and any of the
  776 external nodes: **0 of 25,027** default-rule edges cross a real boundary (`0` internal train/val/test, `0`
  external). All **880** default edges that cross a group label involve *excluded* libraries, which is the quarantine
  working. Of the 142 excluded libraries, **132** have a direct homology edge to an external benchmark protein (560
  direct homology edges, 91 at identity > 0.90 and 149 at TM > 0.90), **8** more belong to external-touching connected
  components through transitive homology, and **2** (`HEEH_KT_rd6_0872`, `HHH_rd4_0124`) were removed by component
  bridge pruning (`max_component_size = 50`).
- **Foldseek adds value.** 550 pairs are structure-only (no MMseqs2 hit, identity ≤ 0.40); none of them crosses a
  kept boundary, i.e. Foldseek drove exclusions rather than introducing leaks.
- **The 40 % identity / 0.70 TM-score cap leaves a documented twilight zone.** Tightening one criterion at a time over
  kept libraries (`cross_real = internal + external`): at 30 % identity, 1,220 pairs newly cross a boundary (206
  internal + 1,014 external); at 25 % identity, 6,453 (560 + 5,893); at TM > 0.50, 667 (145 + 522); at E ≤ 1e-2, 10
  (4 + 6). The capped split contains 8,114 capped-test rows (15.6 %) and 7,210 capped-val rows (20.3 %) whose protein
  has a 30–40 % identity match to a train library (`h`), not `H`.
- **Quarantine scope is intentional.** Nine classical external sets (`Ssym`, `S461`, `S783`, `K2369`, `S8754`, `S2648`,
  `Q3421`, `PTMUL`, `S571`) and five functional DMS (`DLG4`, `GRB2`, `ESTA`, `MYO`, `GB1`) were placed in the split
  quarantine: `S669`- and `K3822`-specific proteins outside `S461` and `K2369` were intentionally left unquarantined,
  `PTMUL-D` is a subset of `PTMUL`, and `ProteinGym` and `Domainome` are too large to exclude without collapsing the
  training pool (so their overlaps are quantified post-hoc by this audit, including Foldseek 3D homology).

### 6.5 What the audit finds

| finding | size | affected rows |
|---|---|---|
| 23 ProteinGym assays are fully inside the capped train split (+ 3 strong 3D structural homologs) | 23/217 `M` + 3/217 `H` DMS, all Stability | 45,623 `M` + 2 `m` + 4,607 `H` / 2,465,767 rows |
| ME stage 1/2, ThermoMPNN and retrained SPURS on those assays | singles (24,252 exact) + constituent (21,331) for ME/SPURS; 24,252 exact for TMPNN | same 23 `M` + 3 `H` DMS |
| Two Domainome domains (`O95155_PF04564_1229` = `3L1X`, `Q9H161_PF00046_216` = `2M0C`) match capped train at 100 % identity (+ 60 strong 2D/3D homolog domains) | 2 `M` + 2 `P` + 60 `H` of 522 domains (458 low-id clean of `M`/`P`/`H`) | 2,023 exact + 2,288 `P` (0.8 %) + 58,422 `H` rows (10.9 %); `h.03` on the 458-domain subset |
| EsmTherm's 80/10/10 is per sequence, not per domain: 26,789 capped-test rows are identical training variants, verified against its published sequences | 51.4 % identical (`M.51`) + 34 % same protein; 3,155 rows only in its validation split (`V.06`) | `M.51 P.34 H.04 h.03 V.06` |
| IFUM's train domains cover 21/33 test libraries (including mutant backgrounds) | 24,200 rows are Dataset2 rows of its train domains (`M?.46`, a ceiling); 24,608 more sit in its val/test block (`V?.47`) | `M?.46 P.02 H.07 h.06 V?.47` |
| ESM3ΔG is zero-shot on Megascale | 0 exact sequences anywhere in the benchmark | 40.7 % strong (`8/33` libs) + 50.3 % weak (`21/33`) homologs |
| DDGemb = S2450 forward + reverse (sequence-aligned `S2450.tsv` + `S2450.fasta`) | 91.7 % of S2648; 63.4 % of Q3421; 43.6 % of S8754; 38.3 % of K3822; 33.0 % of K2369; 59.6 % of Ssym; `m.48`/`m.54` on PTMUL/PTMUL-D; 0 % on S783, S669, S461 | verified exact / constituent |
| DDMut = S9028 + SM1218 + SM444 (forward + reverse structures) | 98.2 % of S2648; 88.3 % of Ssym; 83.5 % of Q3421; 74.3 % of K2369; 72.2 % of K3822; 69.5 % of S8754; 60.9 % of S783; 79.5 % + 11.3 % constituent / 95.0 % of PTMUL/PTMUL-D; 10.0 %/14.5 % of S669/S461 (plus 19 %/24 % same protein) | verified exact |
| GeoDDG = S8754 (construct-aligned) | S8754 is its training set (`#`); 100 % of S2648; 97.9 % of Q3421; 91.2 % of K2369; 89.1 % of K3822; 88.4 % of S783; 50.0 % of Ssym; `m.73`/`m.78` on PTMUL/PTMUL-D; 1.2 % of S669; GeoFitness pretraining proteins: DLG4 abundance/binding (`P1`), 11 % of S461, 7 % of S669 | verified exact / constituent; GeoFitness protein-level |
| the "six clean sets" claim | only S461 has zero identical mutations across all full-coverage models | DDMut has `M.10`/`M.15` on S669/S461; GeoDDG has `M.01 P.07` on S669 and `P.11` on S461 |
| Rosetta's `beta_nov16` tuning data | Kellogg 2011 mutations in 4–28 % of the classical sets (Q3421 `V.28`, S783 `V.22`, Ssym `V.19`, K3822 `V.18`, S2648 `V.16`); Rocklin-2017 designs = 7/33 capped-test and 3/23 val scaffolds | `V.12` test, `V.07` val, `V.002` ProteinGym (protein level) |
| ESM-MSR-small (March reference) homology was never computed | strong homologs on 9 of 12 external sets, ≥ 0.5 % on S2648, S8754, Q3421, PTMUL, PTMUL-D | `H.02`, `H.01`, `H.01`, `H.05`, `H.03` |

**Clean comparators on Tsuboyama.** ESM-MSR (baseline and devel), MutateEverything st1/st2, ThermoMPNN(-D) and the retrained SPURS and SPURS multi have
zero exact mutations and zero same-protein overlap with all 33 capped-test and all 23 capped-val libraries; their only
caveat is the 15.6 %/20.3 % weak-homology rows and validation selection (`V` on the val column). That is the group the
headline Tsuboyama numbers should be read within. At the other end, EsmTherm (`M.51`, verified) and IFUM (`M?.46`) are not held out on this split, and ESM3ΔG is held out (0 exact wild-type or
mutant sequences anywhere in the benchmark) but homology-caveated (8/33 libraries, 40.7 % strong + 50.3 % weak rows).

**The retrained baselines inherit the split, not extra leakage.** MutateEverything stage 1/2, ThermoMPNN and SPURS were retrained on the same 172 libraries as ESM-MSR, so their `P`/`H`/`h` columns are identical by construction and only the
mutation classes differ (stage 1 = singles, stage 2 = singles+doubles, ThermoMPNN-D = doubles plus augmented doubles on 134 scaffolds, SPURS = singles, SPURS multi = singles+doubles). The single
weak-homology row in S669/S461 (`1IOJ` vs `HHH_rd2_0155`, 37 % identity over 61 % of the test protein, E = 5.5) and the single weak-homology row in K3822/K2369 (`1RTP` vs `HHH_rd2_0134`) are present for every retrained model, ThermoMPNN-D included (both libraries have Rosetta models, so its augmentation sees them). ThermoMPNN-D is the one exception to the identical columns elsewhere: it never sees 38 of the 172 libraries (36 designed ones and `2M7O`, `5GU9`: no measured double and no Rosetta model for its augmentation), so its weak-homology fractions are lower on Tsuboyama (`h.07` val, `h.12` test, against `h.20` and `h.16`). Retraining introduced no overlap: the upstream models train on the whole Megascale pool (upstream
ThermoMPNN: 19/33 test libraries, 15,918/36,526 singles; upstream SPURS: 27/33 test libraries, 44,507/52,071 rows), which is strictly more than the capped train split.

**Ssym is not a Megascale set.** None of Ssym's 357 PDB chains or 15 protein families is a Megascale library, so the
Megascale-trained models (EsmTherm, IFUM, ESM3ΔG) have no same-protein overlap with it, and neither have the retrained models.

**ProteinGym.** The 23 fully trained assays are `ARGR_ECOLI_Tsuboyama_2023_1AOY`, `CBX4_HUMAN_Tsuboyama_2023_2K28`,
`CSN4_MOUSE_Tsuboyama_2023_1UFM`, `CUE1_YEAST_Tsuboyama_2023_2MYX`, `DNJA1_HUMAN_Tsuboyama_2023_2LO1`,
`FKBP3_HUMAN_Tsuboyama_2023_2KFV`, `HCP_LAMBD_Tsuboyama_2023_2L6Q`, `ILF3_HUMAN_Tsuboyama_2023_2L33`,
`ISDH_STAAW_Tsuboyama_2023_2LHR`, `NKX31_HUMAN_Tsuboyama_2023_2L9R`, `NUSG_MYCTU_Tsuboyama_2023_2MI6`,
`PITX2_HUMAN_Tsuboyama_2023_2L7M`, `PSAE_PICP2_Tsuboyama_2023_1PSE`, `RCD1_ARATH_Tsuboyama_2023_5OAO`,
`RFAH_ECOLI_Tsuboyama_2023_2LCL`, `RS15_GEOSE_Tsuboyama_2023_1A32`, `SBI_STAAM_Tsuboyama_2023_2JVG`,
`SDA_BACSU_Tsuboyama_2023_1PV0`, `SR43C_ARATH_Tsuboyama_2023_2N88`, `UBE4B_HUMAN_Tsuboyama_2023_3L1X`,
`YAIA_ECOLI_Tsuboyama_2023_2KVT`, `POLG_PESV_Tsuboyama_2023_2MXD`, `YNZC_BACSU_Tsuboyama_2023_2JVD`.
Foldseek 3D homology additionally flags 3 Tsuboyama Stability DMS as strong structural homologs (`H`: `AMFR_HUMAN_Tsuboyama_2023_4G3O` ↔ `2MYX` qTM 0.74, `RAD_ANTMA_Tsuboyama_2023_2CJJ` ↔ `2L9R` qTM 0.73, `VG08_BPP22_Tsuboyama_2023_2GP8` ↔ `HHH_rd1_0606` qTM 0.83; 4,607 rows) and 2 DMS as weak homologs (`h`: `RNC_ECOLI_Weeks_2023` ↔ `2L33`, `THIO_ECOLI_Tsuboyama_2023_2TRX` ↔ `2MI6`; 6,305 rows); none of the 4 >1,800-residue assays (`BRCA1`, `BRCA2`, `POLG_CXB3N`, `SCN5A`) have any `M`, `P`, `H`, or `h` overlap with capped train.
Removing the 23 `M` assays from the all-217 ESM-MSR (devel) 3-seed evaluations (194 DMS): DMS mean 0.557 → 0.516,
Stability 0.880 → 0.871, PG average 0.515 → 0.513, and 191 DMS after also removing the 3 `H` assays: DMS mean 0.510, PG average 0.513 (on the 213-DMS subset: 190 DMS at 0.561 → 0.520 DMS mean and 0.516 → 0.514 PG average; 187 DMS at 0.514 / 0.514). ProSST, zero-shot on all of them, is 0.528 on the clean DMS mean
and 0.514 on the clean PG average.

**Clean external-set means.** Mean over the 12 external sets of §1.1, per model, over the cells with no `M`, `m`, `P`, `H`, `D` or `#`
class at 0.5 % of rows or more (the number in brackets is the number of clean sets it was computed on; S461, K2369 and PTMUL-D are
subsets of S669, K3822 and PTMUL, so the sets overlap):

| model | clean mean (n of 12) | all-12 mean | cells excluded |
|---|---|---|---|
| ESM-MSR (baseline) | 0.646 (12) | 0.646 | none |
| ESM-MSR (devel) | **0.652 (12)** | 0.652 | none |
| ME stage 1 | 0.615 (12) | 0.615 | none |
| ME stage 2 | 0.605 (12) | 0.605 | none |
| ThermoMPNN | 0.585 (12) | 0.585 | none |
| ThermoMPNN + D | 0.586 (12) | 0.586 | none |
| SPURS (retrained) | 0.588 (12) | 0.588 | none |
| SPURS multi (retrained) | 0.556 (12) | 0.556 | none |
| EsmTherm | 0.380 (1) | 0.365 | s571, s783, s2648, s8754, s669, s461, q3421, k3822, k2369, ptmul, ptmuld |
| DDGemb | 0.612 (2) | 0.697 | s571, s783, s2648, s8754, ssym, q3421, k3822, k2369, ptmul, ptmuld |
| GeoDDG-3D | – (0) | 0.767 | all twelve (S461 is `P.11` through GeoFitness pretraining) |
| ESM3ΔG | 0.641 (5) | 0.582 | s571, s2648, s8754, k3822, k2369, ptmul, ptmuld |
| ProSST | 0.423 (12) | 0.423 | none |
| ProteinMPNN | 0.464 (12) | 0.464 | none |
| ESM3-small-open, σ = 0 | 0.378 (12) | 0.378 | none |
| ESM-MSR-small (March reference) | 0.662 (7) | 0.639 | s2648, s8754, q3421, ptmul, ptmuld (`H`, hyperopt-split homologs) |

Only ESM-MSR (baseline and devel), ME stage 1/2, ThermoMPNN(+D), retrained SPURS/SPURS multi, ProSST, ProteinMPNN and the σ = 0 control are clean on all twelve; ESM-MSR
(devel 0.652, baseline 0.646) leads that group, ahead of ME stage 1 0.615, ME stage 2 0.605, SPURS (retrained) 0.588, ThermoMPNN 0.585, SPURS multi (retrained) 0.556,
ProteinMPNN 0.464 and ProSST 0.423. The other means are over fewer sets and over different ones, so they are not comparable with these or
with each other: DDGemb's clean cells are S669 and S461, where it is about 0.01 ahead of ESM-MSR on S669, and GeoDDG has no clean cell.

**External sets.** With identical mutations, same protein and strong homologs excluded, the surviving comparisons are
narrow: on S669 and S461 the clean supervised models are ESM-MSR (0.558 / 0.750 baseline; 0.560 / 0.753 devel),
ME stage 1 (0.559 / 0.721), SPURS retrained (0.497 / 0.644), ThermoMPNN (0.490 / 0.655) and DDGemb (0.570 / 0.654). DDGemb is about 0.01 ahead on
S669; ESM-MSR leads S461 by 0.10. The former 12-set leaders, GeoDDG and DDGemb, are not clean on the sets that made
them leaders.

**The ESM3ΔG hold-out, verified.** The 28,172-mutant held-out set is the ThermoMPNN 28-protein split, not ours, so
its existence says nothing about our 33 libraries. Two checks close that gap. **(i) Exact:** a SHA-1 join of every
benchmark row's wild-type *and* mutant sequence against the 960,216-sequence MGnify train split (`15_esm3dg_exact.py`) gives **0 matches** --
0/87,544 Tsuboyama capped val+test rows, 0/2,465,767 ProteinGym rows across all 217 DMS (including all 117,811 Tsuboyama DMS rows),
0/534,385 Domainome rows, 0/25,632 external-set rows and 0/269,471 DMS-assay rows. No exact variant or exact protein
of ours is in ESM3ΔG's training data. **(ii) Homology:** high-sensitivity MMseqs2 (s = 7.5, e = 10, max-seqs 500)
gives 8 of the 33 capped test libraries a strong homolog (41-68 % identity over 78-96 % of the test protein, E 1e-20
to 1e-6; 21,185 / 52,071 = 40.7 % of test rows), 21 a weak/ambiguous local match (26,204 / 52,071 = 50.3 %) and 4 none (4,682 rows = 9.0 %). The strong hits are all natural
PDB libraries (1GYZ, 3DKM, 2MKX, 2D1U, 2KRU, 2K52, 2L09, 1ZHC); the designs are mostly weak or absent. **(iii)
Cost:** the full comparison is not expensive. Against the 960k corpus the 33-library search takes 4.7-5.8 s including
target indexing (1.3 GB RSS), and the full 2,583-protein test set at the same sensitivity takes 9.7 s (1.5 GB RSS,
42,219 hits). The audit's recipe is exact hash join first, then MMseqs2 at s = 7.5 for the rows it does not resolve.
The per-library table is `docs/benchmarks/leak_audit_esm3dg_libraries.csv`.

### 6.6 Bold-eligibility rules and table annotations

- Superscripts and companion matrices in §1–§3 are derived directly from `docs/benchmarks/leak_audit_matrix.csv`. A cell is eligible for bold when it has no more
  than minor overlap with training: no `M`, `M?`, `m`, `P`, `H`, `D` or `#` class at 0.5 % of rows or more (the same
  threshold that decides whether a superscript is printed). `h` (weak similarity), `–` (unsupervised pretraining only) and `V` / `V?`
  (validation, model selection or calibration: capped val for the models trained on the split, a third-party model's own validation rows,
  Rosetta's energy-function tuning data; superscript `ⱽ`) do not disqualify, because a 30 %-identity local match,
  generic pretraining exposure or validation model selection does not constitute supervised mutation overlap.
  Tables that compare settings of one model (the σ sweeps) bold within it; summary-mean columns and
  singles-only Rosetta rows on multi-mutant sets are never bolded. Consequently, on §1.1 the clean leaders are s571 ESM-MSR (baseline)
  0.455 (GeoDDG 0.526 is `M`), s783 ESM-MSR (devel) 0.725 (GeoDDG `M`, DDGemb `P`), s2648 ME stage 1 0.701 (GeoDDG `M`), s8754
  ESM-MSR (devel) 0.625 (GeoDDG `#`, DDGemb/DDMut `M`), S669 DDGemb 0.570 (GeoDDG/DDMut `M`), and S461 ESM-MSR (devel) 0.753; on §1.3
  Domainome, ProteinMPNN leads at 0.548 on all 522 domains (ESM-MSR `H.11`, `M.004 P.004`), and on `Domainome (low-id, 458)` ESM3-small-open, σ = 0 leads at 0.545 ahead of ProteinMPNN (0.544) and ESM-MSR (devel) WT head (0.542ʰ, weak `h.03` only, eligible); and on §1.4 ProteinGym, ProSST leads at 0.514 / 0.653 / 0.528 (all trained rows carry `M` on the 23 Tsuboyama DMS).
- **Model-specific external-set overlap.** Across the 12 classical external sets, DDMut and GeoDDG overlap S571, Ssym, S669 and PTMUL; DDGemb overlaps S571, Ssym, Q3421, K3822 and
  K2369 at the mutation level (and PTMUL at the constituent level); and EsmTherm/IFUM are same-protein on most classical sets (EsmTherm with 1–3 % verified identical variants on S571, S8754, K3822 and K2369; IFUM with `M?` ceilings of 3–21 % on S571, S783, S8754, K3822 and K2369) through the
  Megascale natural libraries. Only S461 has zero identical mutations across all full-coverage predictors (§1.1).
- **ESM3ΔG is zero-shot on Megascale (`ᴴ`; `ᴾ` on capped val, where 19 % of rows have a ≥ 90 %-identity MGnify homolog, none identical).** Its overlap on `splits_oct06_capped` is homolog-level (8/33 test libraries have a
  strong homolog covering 40.7 % of test rows, 21 have a weak one covering 50.3 %), with 0 exact wild-type or mutant sequence matches across its 960,216-sequence training corpus.
- **S8754 is GeoDDG's training set (`#`).** GeoDDG's 0.841 on S8754 is an in-sample training score rather than a held-out evaluation.
- **ProteinGym Stability overlap.** ESM-MSR's 0.510 vs ProSST's 0.514 global average reflects 23 Tsuboyama Stability assays inside the capped training split; on the 190/187 held-out DMS, ProSST leads the unweighted DMS mean while the category-balanced PG average remains tied (0.514 vs 0.514).

**Consistency across models trained on `splits_oct06_capped`.** Every model trained on the 172 capped-train libraries (ESM-MSR, MutateEverything stage 1/2, ThermoMPNN, retrained SPURS, and SPURS multi) shares the exact same scaffold set: `H` and `h` fractions are identical across all 25 dataset columns, and `M`/`m`/`P` fractions are identical within the singles-only group (ME stage 1, ThermoMPNN, SPURS) and within the singles+doubles group (ESM-MSR, ME stage 2, SPURS multi). ThermoMPNN-D differs only because its training inventory spans the 134 scaffolds with measured doubles or Rosetta augmentation models. On Domainome, two domains (`Q9H161_PF00046_216` = `2M0C` and `O95155_PF04564_1229` = `3L1X`) match capped-train scaffolds at 100 % sequence identity (`M.004 P.004`), and combined MMseqs2 + Foldseek 3D homology flags 60 domains (`H.11`, superscript `ᴴ`); excluding those 64 domains leaves the 458-domain low-homology subset (`h.03` weak similarity only).

### 6.7 Reproducibility, assumptions and limits

**Artifacts.** `docs/benchmarks/leak_audit.sqlite` (tables `inventories`, `cells`, `pg_per_dms`, `homology`,
`structure_homology`, `code_exact`, `rosetta`, `esm3dg_*`, `geofitness_proteins`), `leak_audit_rosetta.csv`, `leak_audit_matrix.csv` (every model×dataset cell with counts and code), `leak_audit_inventories.csv`,
`leak_audit_pg_per_dms.csv`, `leak_audit_homology.csv`, `leak_audit_structure_homology.csv`, `leak_audit_code_exact.csv`, `leak_audit_esm3dg_libraries.csv`, `split_homology_audit.json`.
Scripts: `analysis/leak_audit/` (`run_all.py` executes `01_esmtherm_inventory.py` → `19_export.py` in dependency order; plain-text upstream reference tables are tracked in `analysis/leak_audit/data/`).

**Assumptions and limits.**
1. A model is treated as having trained on *all* rows of a library it trained on (ESM-MSR, ME, ThermoMPNN, SPURS,
   ESM3ΔG). Model-specific filters (censored rows, early stopping) can
   only remove rows; the audit is therefore an upper bound on exact overlap, which is the conservative direction.
2. EsmTherm publishes every training sequence, so its `M` is an exact-sequence join. IFUM publishes domains, not rows:
   its `M?` counts test rows whose mutant sequence is a Dataset2 row of an IFUM train domain, and `V?` those of its
   val/test domains; both are ceilings, not measurements. Mutant-background libraries (`.pdb_<mut>`) are normalised to their wild-type scaffold before checking domain membership.
3. The ESM3ΔG inventory uses its published 960,216-sequence MGnify index. An exact SHA-1 join of every benchmark
   wild-type and mutant sequence returns 0 matches; the homology fractions use the high-sensitivity MMseqs2 run
   (s = 7.5, max-seqs 500) evaluated across all qualifying hits partitioned by query-coverage tier. Because the `*_augmented_*` checkpoints were trained on a subset of that index, the
   homology fractions are a conservative upper bound.
4. `P` is judged at ≥95 % identity/≥90 % shorter-sequence coverage, with ≥90 %/≥50 % as a secondary rule for literature sets;
   near-identical constructs (missing purification tags, residue offsets) count as the same protein.
5. Sequence homology uses MMseqs2 across all inventories, and 3D structure homology (exhaustive Foldseek over query PDBs vs the 172 capped-train structures) is additionally computed for Domainome 1.0 (522 domain PDBs from Beltran et al., 2025; Zenodo `14356805`) and ProteinGym (`07_structure_homology.py`). Foldseek was not run on the sequence-only 960,216-entry MGnify corpus or against third-party literature training sets (`S2450`, `S9028`).
6. `h` is deliberately permissive (≥30 % identity or E ≤ 1e-2 over ≥30 % of the test protein) and should be read as
   "not clean, evidence weak", not as a demonstrated leak.
7. `DDGemb` (`S2450`, forward + reverse), `DDMut` (`S9028 + SM1218 + SM444`) and `GeoDDG` (`S8754`) exact (`M`) and constituent (`m`) overlaps are verified directly by construct/family sequence alignment (including reverse mutant structures `mt_pos_wt` in `S9028` and `Ssym`).
8. Rosetta is coded `V`, not `M`/`P`: about 20 amino-acid-type offsets and one scale factor were tuned on Kellogg-2011
   and Rocklin-2017 data, which cannot encode any one variant, but its numbers on those rows are not fully held out.
9. GeoDDG's GeoFitness pretraining is matched by wild-type sequence only; its `dms.csv` rows were not audited.
10. No attempt is made to estimate how much of a score a leak is *worth*. A 100 %-trained DMS can still score below a
   zero-shot model (the learned map differs), and a same-protein overlap need not help at all. The codes say what was
   trained, not what it bought.

## 7. Repository code and benchmark artifacts

- **Benchmark tables and audit artifacts (`docs/benchmarks/`):**
  - `all_by_all_long.csv`, `seed_sd_by_table_cell.csv`, `spurs_retrain_oct06.csv`, `checkpoints.csv`: machine-readable cell means, multi-seed standard deviations, retrained SPURS evaluations, and checkpoint MD5 manifests.
  - `leak_audit.sqlite`, `leak_audit_*.csv`, `esm3dg_*.csv`, `split_homology_audit.json`: row-level leakage database, model×dataset overlap matrices, ESM3ΔG exact/homology verification outputs, and Tsuboyama split homology audit summary.
- **Inference and resource profiling (`inference_scripts/`):**
  - `esm_msr_testing.py`: ESM-MSR evaluation across Tsuboyama splits, external datasets, functional DMS assays, Domainome, and ProteinGym (`--lo_scale` for LoRA scaling $\sigma$).
  - `pgym_runner.py` and `resource_tracker.py`: shared ProteinGym and gamut runner recording per-protein wall time, CPU utilization, host RSS, and peak CUDA memory (`resources.csv` and `run_meta.json`).
  - `inference_scripts/baselines/` and `training_scripts/baselines/`: unified evaluation and retraining entrypoints on `splits_oct06_capped` for ThermoMPNN / ThermoMPNN-D, MutateEverything (stages 1 and 2), SPURS / SPURS multi, and ProteinMPNN.
- **Scoring, epistasis evaluation, and downstream analysis (`src/esm_msr/`, `scripts/`, `analysis/`):**
  - `src/esm_msr/epi_metrics.py` and `scripts/epi_from_dump.py`: double-mutant epistasis assembly and four-level evaluation hierarchy (`docs/validation_metrics.md`).
  - `analysis/`: reproducible figure and table verification pipeline (`analysis/run_all.sh`), cross-set leakage audit (`analysis/leak_audit/run_all.py`), and prediction scoring utilities (`analysis/scoring/`).
