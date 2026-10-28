# ESM-MSR benchmark results, October 2026

**Every comparison of ESM-MSR against another model.** ESM-MSR's own training history (the `o6`, link and `dc`
series) is in [`EXPERIMENTS.md`](EXPERIMENTS.md); this document is the results.

**The two ESM-MSR checkpoints compared throughout.** **ESM-MSR (baseline)** is the released recipe retrained on the
capped split as a **three-seed triplicate** (`LoRA_models/release-baseline/s{1,2,3}`, unmasked, σ = 1); §1.1–1.3 carry
its seed mean ± the standard deviation over the three seeds. It replaces the single
`LoRA_models/esm-msr-baseline-w2a4-m16a16-capped` checkpoint, which is now quoted only by the §1.4 ProteinGym row;
those came from the earlier run, and the triplicate's ProteinGym pass was still in flight when this table was built.
§2 epistasis is rescored from the triplicate too (`analysis_scripts/rescore_epistasis.py`). **ESM-MSR (devel)** is the three-seed
floor-link series from `devel-revision` (`esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}`,
`val_rho_combined_avg` 0.840 / 0.839 / 0.841). Both were trained on `splits_oct06_capped.pkl`, as were the
retrained baselines, so all of them are scored on identical rows. "Capped" on its own always means that
**split**, never a checkpoint.

**The retrained baselines (corrected 2026-10-27).** MutateEverything (ME, stage 1 and stage 2) and ThermoMPNN /
ThermoMPNN-D are the upstream recipes retrained on the capped split, on **valid ddG labels only** (`ddG_ML` numeric, i.e.
the mutant and the wild type both inside the assay's dG range of −1 to 5 kcal/mol; no bounded or censored row), with none
of ESM-MSR's rules (no item filters, rank-only items, floor, link or censoring) and with the mutant-background libraries
kept, which is the one departure from the upstream recipes (they never saw those libraries). An earlier set of stage-2 and
ThermoMPNN-D runs had applied ESM-MSR's item filters and lost 14,388 of the 53,188 measured doubles; those runs are
withdrawn, and every ME stage-2 and ThermoMPNN-D number below is from the corrected runs (3 seeds), except where a cell
says `q` (queued). The recipe, the data accounting and the deviations that remain are in §6.3.

**Metric conventions.** One Spearman ρ per model and dataset, positive = stabilizing / fitter for every model.
External sets and DMS are a pooled ρ per set; Tsuboyama and Domainome are the mean of per-protein ρ, on measured
rows only; ProteinGym is per-DMS ρ summarised as in §3. Multi-seed rows are seed means.

**Regenerating.** `analysis_scripts/benchmark_matrix.py collect|tables`, run on the cluster (it reads the result
trees in place). The long table is [`benchmarks/all_by_all_long.csv`](benchmarks/all_by_all_long.csv)
(model, dataset, rho, sd, n_seeds, source). Companion documents: what is still outstanding is in
[`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md); which hardware each run used, and the inference-speed
findings, in [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md).

## 1. All-by-all: every model on every dataset

**Leak notation (§6).** Every number carries a superscript for the worst training-data overlap it has with its test set, and the full code sits in the matrix directly below each table. **ᴹ** identical mutation(s) in training; **ᵐ** the test variant's mutations were trained but not this exact combination (additive models, doubles made of trained singles); **ᴾ** same protein, other mutations; **ᴴ** strong homolog in training (MMseqs2 E ≤ 1e-3 over ≥ 50 % of the test protein); **ʰ** weak/ambiguous similarity (E ≤ 1e-2 or ≥ 30 % identity over ≥ 30 %); **ᴰ** documented training on the same source dataset, rows not reconstructable; **ᶜ** pretraining-corpus exposure only (WT sequence/structure, no mutation supervision); **#** the number is not a test result (the set is in training). A code such as `M.59 P.35 H.02` gives the fraction of dataset rows in each class, `M?` marks a probable but unverified identical-mutation overlap, and `V` marks a dataset used for validation or model selection. Superscripts are omitted below 0.5 % of rows; the companion matrix keeps them. A star (`*`) marks partial coverage (DDMut: S669 98 %, S461 97 %); **_italic_** a different variant set (Rosetta on ProteinGym: single mutants only, shown but never compared); **q** a run in progress or queued; **–** not run. The audit, the inventories and the assumptions are in §6; the hyperopt-split rows of §1.2 and §1.5 predate the Oct06 split and are out of its scope.

ESM-MSR's **MT head is not listed anywhere**: it is an internal head, not a deployable model. ESM-MSR appears by
its combined prediction, which is what is deployed, and in §1 also by its WT head, which is the additive reading.

Provenance of the rows that were not produced by `esm_msr_testing.py`: ProSST's ProteinGym numbers are the
authors' own published ProSST-2048 scores (ProteinGym ships every benchmarked model's predictions), scored on the
same variants as everything else; Rosetta's capped-split numbers are `cartesian_ddg` runs from Narval (relaxed
structures, 3 iterations, doubles scored directly) and its ProteinGym numbers are a September 2024 singles-only
file; the ESM3 zero-shot rows are the Forge API runs.

**Which Tsuboyama columns are comparable.** The **capped** columns are the split every October model was trained
on. The **hyperopt** columns are the older split the third-party harness was first run on; it shares only 4 of 23
val and 9 of 33 test libraries with capped, so the two are not comparable with each other.

### 1.1 Thermostability: the 12 external sets

| model | s571 | s783 | s2648 | s8754 | s669 | s461 | ssym | q3421 | k3822 | k2369 | ptmul | ptmuld | mean of 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | **0.455±0.003** | 0.713±0.004 | 0.669±0.005 | 0.616±0.003 | 0.558±0.003 | 0.750±0.004 | 0.765±0.004 | 0.621±0.003 | 0.714±0.002 | 0.715±0.002 | 0.583±0.003 | 0.593±0.005 | 0.646±0.002 |
| ESM-MSR (baseline), WT head (3 seeds) | 0.441±0.006 | 0.698±0.004 | 0.662±0.003 | 0.607±0.004 | 0.543±0.000 | 0.744±0.004 | 0.764±0.003 | 0.615±0.001 | 0.715±0.001 | 0.712±0.001 | 0.571±0.003 | 0.577±0.001 | 0.637±0.001 |
| ESM-MSR (devel), combined (3 seeds) | 0.445 | **0.725** | 0.677 | **0.625** | 0.560 | **0.753** | 0.773 | 0.626 | **0.721** | **0.722** | **0.594** | **0.597** | 0.652 |
| ESM-MSR (devel), WT head | 0.430 | 0.713 | 0.675 | 0.618 | 0.551 | **0.753** | 0.776 | 0.623 | 0.719 | 0.717 | 0.583 | 0.585 | 0.645 |
| MutateEverything st1 (3 seeds) | 0.439 | 0.706 | **0.701** | 0.623 | 0.559 | 0.721 | 0.674 | **0.638** | 0.696 | 0.674 | 0.472 | 0.480 | 0.615 |
| MutateEverything st2 (3 seeds) | 0.426 | 0.694 | 0.676 | 0.606 | 0.542 | 0.710 | 0.636 | 0.620 | 0.683 | 0.662 | 0.510 | 0.489 | 0.605 |
| ThermoMPNN (additive, 3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.433 | 0.499 | 0.585 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.406 | 0.689 | 0.648 | 0.602 | 0.490 | 0.655 | 0.711 | 0.591 | 0.643 | 0.649 | 0.432 | 0.514 | 0.586 |
| SPURS (single model) | 0.468ᴾ | 0.712ᴾ | 0.682ᴾ | 0.626ᴾ | 0.502 | 0.654 | **0.784** | 0.614ᴾ | 0.681ᴾ | 0.698ᴾ | 0.399ᵐ | 0.495ᵐ | 0.610 |
| SPURS multi (multi-mutant model) | 0.430ᵐ | 0.701ᴾ | 0.642ᴾ | 0.594ᴾ | 0.459ᵐ | 0.610ᵐ | 0.758 | 0.594ᴾ | 0.627ᴾ | 0.650ᴾ | 0.359ᴾ | 0.466ᴾ | 0.574 |
| ESMtherm | 0.349ᴾ | 0.434ᴾ | 0.371ᴾ | 0.347ᴾ | 0.414ᴾ | 0.474ᴾ | 0.380 | 0.348ᴾ | 0.366ᴾ | 0.393ᴾ | 0.235ᴾ | 0.270ᴾ | 0.365 |
| DDGemb (server) | 0.484ᴹ | 0.689ᴹ | 0.895ᴹ | 0.684ᴹ | **0.570** | 0.654 | 0.891ᴹ | 0.752ᴹ | 0.745ᴹ | 0.702ᴹ | 0.616ᴾ | 0.678ᴾ | 0.697 |
| DDMut (server) | 0.350ᴹ | 0.687ᴹ | 0.731ᴹ | 0.638ᴹ | 0.402*ᴾ | 0.534*ᴾ | 0.363ᴹ | 0.653ᴹ | 0.709ᴹ | 0.681ᴹ | 0.725ᴹ | 0.718ᴹ | – |
| IFUM, WT backbone | 0.376ᴾ | 0.498ᴾ | 0.485ᴾ | 0.477ᴾ | 0.434ᴾ | 0.531ᴾ | 0.574 | 0.431ᴾ | 0.536ᴾ | 0.547ᴾ | 0.350ᴾ | 0.405ᴾ | 0.470 |
| IFUM, ESMFold per variant | 0.385ᴾ | 0.471ᴾ | 0.484ᴾ | 0.451ᴾ | 0.409ᴾ | 0.542ᴾ | 0.580 | 0.433ᴾ | 0.508ᴾ | 0.512ᴾ | 0.391ᴾ | 0.443ᴾ | 0.467 |
| GeoDDG-3D (FASPR) | 0.526ᴹ | 0.865ᴹ | 0.892ᴹ | 0.841# | 0.615ᴾ | 0.664ʰ | 0.914ᴹ | 0.822ᴹ | 0.852ᴹ | 0.831ᴹ | 0.717ᴾ | 0.669ᴾ | 0.767 |
| ESM3ΔG | 0.511ᴴ | 0.653ʰ | 0.642ᴴ | 0.542ᴴ | 0.564ʰ | 0.681ʰ | 0.735ʰ | 0.572ʰ | 0.589ᴴ | 0.598ᴴ | 0.409ᴴ | 0.482ᴴ | 0.582 |
| ProSST (zero-shot) | 0.310ᶜ | 0.564ᶜ | 0.436ᶜ | 0.428ᶜ | 0.369ᶜ | 0.561ᶜ | 0.519ᶜ | 0.416ᶜ | 0.476ᶜ | 0.497ᶜ | 0.182ᶜ | 0.317ᶜ | 0.423 |
| ProteinMPNN (zero-shot) | 0.324ᶜ | 0.561ᶜ | 0.506ᶜ | 0.438ᶜ | 0.329ᶜ | 0.442ᶜ | 0.696ᶜ | 0.494ᶜ | 0.420ᶜ | 0.436ᶜ | 0.382ᶜ | 0.546ᶜ | 0.464 |
| ESM3-small-open, σ = 0 (zero-shot) | 0.268ᶜ | 0.487ᶜ | 0.379ᶜ | 0.374ᶜ | 0.299ᶜ | 0.404ᶜ | 0.583ᶜ | 0.382ᶜ | 0.423ᶜ | 0.442ᶜ | 0.190ᶜ | 0.304ᶜ | 0.378 |
| ESM3-large (Forge, zero-shot) | 0.220ᶜ | 0.353ᶜ | 0.267ᶜ | 0.291ᶜ | 0.272ᶜ | 0.334ᶜ | 0.539ᶜ | 0.306ᶜ | 0.318ᶜ | 0.364ᶜ | 0.017ᶜ | 0.126ᶜ | 0.284 |
| ESM3-medium (Forge, zero-shot) | 0.332ᶜ | 0.552ᶜ | 0.439ᶜ | 0.438ᶜ | 0.342ᶜ | 0.445ᶜ | 0.633ᶜ | 0.450ᶜ | 0.483ᶜ | 0.503ᶜ | 0.264ᶜ | 0.353ᶜ | 0.436 |
| ESM3-small (Forge, zero-shot) | 0.236ᶜ | 0.548ᶜ | 0.441ᶜ | 0.425ᶜ | 0.336ᶜ | 0.457ᶜ | 0.588ᶜ | 0.441ᶜ | 0.475ᶜ | 0.500ᶜ | 0.304ᶜ | 0.406ᶜ | 0.430 |
| ESM-MSR-small (March reference) | 0.447 | 0.714 | 0.677 | 0.614 | 0.542 | 0.743 | 0.781 | 0.621 | 0.701 | 0.704 | 0.543 | 0.577 | 0.639 |

| model | s571 (leak code) | s783 (leak code) | s2648 (leak code) | s8754 (leak code) | s669 (leak code) | s461 (leak code) | ssym (leak code) | q3421 (leak code) | k3822 (leak code) | k2369 (leak code) | ptmul (leak code) | ptmuld (leak code) | mean of 12 (leak code) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (baseline), WT head (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (devel), combined (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| ESM-MSR (devel), WT head | – | – | – | – | h.001 | h.002 | – | – | h<.001 | h<.001 | – | – | – |
| MutateEverything st1 (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | – | – | – | – | – |
| MutateEverything st2 (3 seeds) | – | – | – | – | h.001 | h.002 | – | – | – | – | – | – | – |
| ThermoMPNN (additive, 3 seeds) | – | – | – | – | h.001 | h.002 | – | – | – | – | – | – | – |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| SPURS (single model) | P.08 H.03 | P.07 h.08 | M.001 P.02 H.02 h.01 | M.002 P.20 H.02 h.006 | H.003 h.001 | h.002 | – | M.002 P.02 H.02 h.008 | M.001 P.23 H.02 h.01 | M.002 P.34 H.02 h.01 | m.02 P.09 H.05 h.02 | m.006 P.02 H.04 h.02 | – |
| SPURS multi (multi-mutant model) | m.005 P.08 H.03 | P.07 h.11 | m<.001 P.04 H.01 h.01 | m.002 P.21 H.01 h.006 | m.01 P.13 H.003 h.09 | m.02 P.19 h.13 | – | m<.001 P.04 H.02 h.02 | m.002 P.27 H.02 h.02 | m.002 P.35 H.02 h.02 | P.10 H.04 h.02 | P.02 H.04 h.02 | – |
| ESMtherm | P.12 M?.09 H.03 | P.19 M?.15 h.04 | P.05 M?.04 H.02 h.008 | P.23 M?.18 H.02 h.006 | P.24 M?.19 H.003 h.09 | P.35 M?.28 h.14 | – | P.06 M?.05 H.03 h.02 | P.31 M?.24 H.03 h.02 | P.38 M?.31 H.03 h.02 | P.13 M?.10 H.06 h.01 | P.06 M?.05 H.06 h.02 | – |
| DDGemb (server) | M.09 P.20 H.03 h.01 | M.01 P.70 H.02 | M.93 H.02 h.001 D | M.30 P.37 H.04 h.004 | – | – | M.28 P.72 | M.45 P.39 H.02 | M.26 P.31 H.03 h.006 | M.24 P.26 H.03 h.003 | P.79 H.03 h.003 | P.86 H.04 h.004 | – |
| DDMut (server) | M.20 P.20 H.06 | M.42 P.49 H.02 | M.89 P.11 H<.001 D | M.60 P.30 H.03 h.001 D | P.11 H.02 h.004 | P.16 H.004 h.004 | M.30 P.70 | M.66 P.27 H.009 | M.57 P.28 H.03 h.002 | M.65 P.22 H.03 h.003 | M.59 P.37 H.01 | M.36 P.59 H.02 | – |
| IFUM, WT backbone | P.05 M?.05 H.08 | P.04 M?.04 H.03 h.11 | P.06 M?.06 H.02 | P.22 M?.22 H.02 h.001 | P.14 M?.14 H.003 h.09 | P.21 M?.21 h.14 | – | P.06 M?.06 H.02 h.009 | P.29 M?.29 H.02 h.02 | P.38 M?.38 H.02 h.01 | P.15 M?.15 H.03 h.001 | P.07 M?.07 H.04 h.002 | – |
| IFUM, ESMFold per variant | P.05 M?.05 H.08 | P.04 M?.04 H.03 h.11 | P.06 M?.06 H.02 | P.22 M?.22 H.02 h.001 | P.14 M?.14 H.003 h.09 | P.21 M?.21 h.14 | – | P.06 M?.06 H.02 h.009 | P.29 M?.29 H.02 h.02 | P.38 M?.38 H.02 h.01 | P.15 M?.15 H.03 h.001 | P.07 M?.07 H.04 h.002 | – |
| GeoDDG-3D (FASPR) | M.39 P.11 H.04 | M.88 h.07 | M.98 P.02 | M1 # | P.01 H.03 h.03 | H.004 h.03 | M.37 P.63 | M.69 P.29 | M.69 P.20 H.002 h.002 | M.76 P.15 H.003 h.002 | P.97 H.004 | P.97 H.006 | – |
| ESM3ΔG | H.02 h.07 | h.08 | H.02 h.12 | H.01 h.08 | h.23 | h.31 | h.04 | H.004 h.11 | H.009 h.10 | H.007 h.09 | H.009 h.14 | H.01 h.12 | – |
| ProSST (zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ProteinMPNN (zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ESM3-small-open, σ = 0 (zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ESM3-large (Forge, zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ESM3-medium (Forge, zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ESM3-small (Forge, zero-shot) | C | C | C | C | C | C | C | C | C | C | C | C | – |
| ESM-MSR-small (March reference) | – | – | – | – | – | – | – | – | – | – | – | – | – |
- **The 12-set ranking is dominated by training overlap.** GeoDDG (0.767) and DDGemb (0.697) lead only on cells that are partly or wholly their training data: S8754 *is* GeoDDG's training set (`M 1.00`, `#`), S2648 is 92.5 % of DDGemb's S2450, and every one of GeoDDG's twelve cells carries an identical-mutation, same-protein or strong-homolog flag (§6.5). Over the cells with no such flag (§6.5) the 12-set mean is ESM-MSR (devel) 0.651, ME stage 1 0.615, ME stage 2 0.605, ThermoMPNN 0.585, ProteinMPNN 0.464 and ProSST 0.423; none of them loses a cell to the filter. On the two sets that are same-protein-clean for every supervised model (S669, S461) ESM-MSR and DDGemb are level on S669 (0.565 vs 0.570) and ESM-MSR leads S461 (0.756 vs 0.654).
- **There is no set with no overlap for every model.** The overlap lattice is measured, not assumed: S669 and S461 are the only sets with zero identical mutations for every model, yet DDMut still shares 11 %/16 % of them at the same-protein level (its S9028 includes S2648+FireProtDB proteins), and GeoDDG has a same-protein or strong-homolog trace on both (1 % of S669, 3 % of S461). S571 is 39 % identical-mutation overlap with GeoDDG and 20 % with DDMut; Ssym is 37 % with GeoDDG, 30 % with DDMut and 28 % with DDGemb; PTMUL is 59 % with DDMut's multi-mutant training file SM1242 and 97 % same-protein with GeoDDG/DDGemb. A ranking over the raw 12 sets therefore compares different models on different subsets. The per-model means over their own clean cells are in §6.5; only ESM-MSR, ME, ThermoMPNN, ProSST, ProteinMPNN and the zero-shot control are clean on all twelve, and ESM-MSR (released 0.647, devel 0.651) leads them.
- **Among the models trained on Tsuboyama/Megascale, ESM-MSR leads 9 of 12 sets, but only two of those models are clean comparators.** SPURS trained on 24 of the 33 capped test libraries (21,315 of 36,526 singles, 58.4 % exact), ESMtherm on 27 of 33 (82 % of test rows same-protein, mutant-level split) and IFUM on 21 of 33 (45 % same-protein); ESM3ΔG is not Megascale-trained at all (see below). Only ESM-MSR, MutateEverything and ThermoMPNN have zero exact-mutation and zero same-protein overlap with all 33 test libraries (§6.5).
- **The σ = 0 row is a control, not a model:** the ESM-MSR architecture with the LoRA scaled to zero, i.e. plain
  ESM3-small-open read through the calibration head. Its 12-set mean (0.378) sits on the published
  ESM3-small-open figure (0.380 over that table's 10 sets, §1.5), so the LoRA-off path behaves as expected and the
  LoRA is worth about +0.27 ρ.
- **SPURS is two models.** SPURS proper is the single-mutant model, summed over substitutions for a multi-mutant.
  **SPURS multi** is the multi-mutant model applied to *every* variant, singles included. The first harness run used
  the single model for singles and the multi model for multi-mutants, so on a singles-only benchmark it was the single
  model throughout and the multi model had never been scored on a single mutant outside ProteinGym. Scored properly,
  SPURS multi is clearly the weaker predictor on singles: 12-set mean 0.574 against 0.610, DMS mean 0.504 against
  0.605, Domainome 0.406 against 0.513, capped test 0.806 against 0.864 (S669 0.459 against 0.502, K3822 0.627 against
  0.681). It is also lower on the multi-mutant sets (PTMUL 0.359 against 0.399, PTMUL-D 0.466 against 0.495), so the
  multi model is not buying anything even where it was designed to be used.
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
  mutated sequence, 571/571 doubles through 1/1 ten-mutants; `analysis_scripts/forge_ptmul_link_check.py`), and
  ESM3-large is genuinely anti-informative on the higher-order mutants (ρ −0.06 over the 343 with three or more
  substitutions, −0.28 at four) while positive on the doubles (0.145).

- **Corrected ME stage 2 against the pre-October ME on PTMUL (3 seeds each, all 914 rows).** PTMUL ρ: pre-October
  (hyperopt split) 0.539 / 0.580 / 0.555, mean 0.558; item-filtered capped run 0.501 / 0.427 / 0.544, mean 0.491;
  corrected capped run 0.520 / 0.450 / 0.561, mean 0.510. Restoring the 14,388 filtered doubles improves every seed
  (+0.019 / +0.023 / +0.017; mean +0.020, bootstrap over variants +0.005 to +0.034), and the gain is entirely in
  multi-mutants: doubles +0.001, triples +0.038, 4+ +0.061. The remaining −0.048 to the pre-October model is not the
  filtering. Stage 1, whose singles are identical in both tables, already sits at 0.472 against 0.499 (new seeds
  0.533 / 0.415 / 0.467), the stage 2 gain over stage 1 is +0.039 against +0.059, and the weak seed 2 is weak from stage 1
  on (0.415 → 0.450). Per order, corrected against pre-October: doubles 0.474 / 0.503, triples 0.589 / 0.677, 4+ 0.582 /
  0.621. One more difference is known: the pre-October ME was trained on `ddg = +ddG_ML` (positive = stabilizing), the
  opposite of the upstream convention (positive = destabilizing) that the retrained models use. ME's stage-2 loss is
  sign-dependent: it keeps only about a quarter of the destabilizing doubles per epoch and weights the stabilizing ones 4×.
  With the flipped sign it kept every double and put 99 % of the weight on the destabilizing ones. The new seeds' sd on
  PTMUL is 0.056; the old 0.020 understates it, because the three old stage-2 runs all used seed 0. The remaining −0.048
  is about one standard error.

### 1.2 Tsuboyama

| model | capped val | capped test |
|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | **0.855±0.001ʰ** | 0.847±0.001ʰ |
| ESM-MSR (baseline), WT head (3 seeds) | 0.841±0.002ʰ | 0.829±0.002ʰ |
| ESM-MSR (devel), combined (3 seeds) | 0.854ʰ | **0.849ʰ** |
| ESM-MSR (devel), WT head | 0.837ʰ | 0.828ʰ |
| MutateEverything st1 (3 seeds) | 0.754 | 0.771 |
| MutateEverything st2 (3 seeds) | 0.730 | 0.760 |
| ThermoMPNN (additive, 3 seeds) | 0.787 | 0.794 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.785 | 0.784 |
| SPURS (single model) | 0.841ᴹ | 0.864ᴹ |
| SPURS multi (multi-mutant model) | 0.787ᴹ | 0.806ᴹ |
| ESMtherm | 0.716ᴾ | 0.774ᴾ |
| DDGemb (server) | 0.490 | 0.508ʰ |
| DDMut (server) | – | – |
| IFUM, WT backbone | 0.666ᴾ | 0.711ᴾ |
| IFUM, ESMFold per variant | 0.713ᴾ | 0.752ᴾ |
| GeoDDG-3D (FASPR) | 0.595 | 0.609ʰ |
| ESM3ΔG | 0.811ᴴ | 0.811ᴴ |
| ProSST (zero-shot) | 0.593ᶜ | 0.596ᶜ |
| ProteinMPNN (zero-shot) | 0.541ᶜ | 0.542ᶜ |
| ESM3-small-open, σ = 0 (zero-shot) | 0.565ᶜ | 0.578ᶜ |
| ESM3-large (Forge, zero-shot) | – | – |

*The legacy hyperopt val/test columns are omitted here: those rows predate the Oct06 split and are out of this audit's scope (their values are in §1.5).*
| model | capped val (leak code) | capped test (leak code) |
|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | h.20 V | h.16 |
| ESM-MSR (baseline), WT head (3 seeds) | h.20 V | h.16 |
| ESM-MSR (devel), combined (3 seeds) | h.20 V | h.16 |
| ESM-MSR (devel), WT head | h.20 V | h.16 |
| MutateEverything st1 (3 seeds) | V | – |
| MutateEverything st2 (3 seeds) | V | – |
| ThermoMPNN (additive, 3 seeds) | V | – |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | V | – |
| SPURS (single model) | M.48 m.21 P.10 V | M.41 m.13 P.35 |
| SPURS multi (multi-mutant model) | M.21 m.03 P.30 H.02 h.02 V | M.12 m.03 P.43 H.07 h.02 |
| ESMtherm | P.87 M?.70 H.03 h.02 V | P.82 M?.66 H.04 h.03 |
| DDGemb (server) | V | h.02 |
| DDMut (server) | – | – |
| IFUM, WT backbone | P.68 M?.68 h.07 V | P.45 M?.45 H.07 h.06 |
| IFUM, ESMFold per variant | P.68 M?.68 h.07 V | P.45 M?.45 H.07 h.06 |
| GeoDDG-3D (FASPR) | V | h.07 |
| ESM3ΔG | H.49 h.05 V | H.41 h.25 |
| ProSST (zero-shot) | C.73 | C.78 |
| ProteinMPNN (zero-shot) | C.73 | C.78 |
| ESM3-small-open, σ = 0 (zero-shot) | C.73 | C.78 |
| ESM3-large (Forge, zero-shot) | – | – |
- **SPURS is not held out, and that explains the comparison.** SPURS's Megascale fit uses the ThermoMPNN 8:1:1 split, not ours: 24 of our 33 capped-test libraries and 21,315 of 36,526 singles are in its fit (58.4 % exact; 41.2 % of the doubles for the multi model), and its val is in the fit too. Its 0.866/0.844 is therefore not a held-out result. Under the audit it is `M.41 m.13 P.35` on test; the identical-mutation part alone is larger than any difference between the models.
- **Among the models whose test set is held out** (ESM-MSR, MutateEverything and ThermoMPNN, all retrained on the
  same capped training split, plus the models that never saw Megascale), ESM-MSR leads: devel 0.849 and released
  0.843 on test, then ThermoMPNN 0.794, ME stage 1 0.771, **Rosetta 0.663**, GeoDDG 0.609, ProteinMPNN 0.542,
  DDGemb 0.508.
- **Rosetta Cartesian ΔΔG** (physics-based; nothing to leak) scores 0.683 on validation and 0.663 on test at full
  coverage, above GeoDDG, ProteinMPNN, DDGemb and the zero-shot control and below every learned model trained on
  this data. The earlier published comparison had Rosetta at 0.666 / 0.634 on the older hyperopt split (§1.5),
  so the new run agrees.
- **The zero-shot control** (ESM3-small-open, σ = 0) reaches 0.578 on capped test, so the fine-tuning is worth
  about +0.27 ρ here too.
- **SPURS and ProteinMPNN on the mutant-background libraries.** Both models read the sequence from the wild-type
  PDB, which has the wild-type residue at the background position, and the first runs skipped those libraries (6,026
  val and 7,931 test rows, all singles). The adapters now overwrite the parsed sequence with the corrected
  mutant-background sequence (the structure stays the wild-type backbone's), asserting that it differs from the PDB
  sequence at exactly the background positions. The reruns of capped val and test (jobs 386549–386551) are in the
  numbers above. They moved the capped values only a little (SPURS 0.845 / 0.866 → 0.841 / 0.864, SPURS multi 0.789 / 0.806 →
  0.787 / 0.806, ProteinMPNN 0.532 / 0.537 → 0.541 / 0.542), and left the wild-type-background rows bit-for-bit unchanged. On the
  mutant-background libraries themselves the per-library Spearman is 0.70–0.87 for SPURS and 0.44–0.70 for ProteinMPNN.
  No other model had this gap. (The SPURS single-model rerun first wrote to a mislabelled directory and these numbers lagged it
  by a few hours; see `GPU4_INFERENCE_LEDGER.md` §4a.)
- **IFUM is not held out either.** Its published split puts 21 of the 33 capped-test libraries in train and the other 12 in its unseparated val/test block; 23,518 test rows (45 %) are on proteins it trained on, and the mutations are the Megascale rows themselves. Both IFUM modes are `P`-flagged on the capped split, and its GB1 row is same-protein (IFUM trained on the Megascale 1PGA/GB1 library). Its per-variant ESMFold mode still beats its WT-backbone mode, but neither is a clean test.

### 1.3 DMS and Domainome

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | 0.737±0.003 | 0.737±0.002 | 0.675±0.015 | 0.627±0.009 | 0.638±0.002 | 0.480±0.006 | 0.318±0.010 | 0.543±0.001ᴾ |
| ESM-MSR (baseline), WT head (3 seeds) | 0.733±0.001 | 0.736±0.002 | 0.670±0.012 | 0.636±0.004 | 0.637±0.001 | 0.462±0.008 | 0.341±0.010 | 0.541±0.003ᴾ |
| ESM-MSR (devel), combined (3 seeds) | 0.745 | 0.742 | 0.681 | 0.636 | 0.639 | 0.481 | 0.337 | 0.543ᴾ |
| ESM-MSR (devel), WT head | 0.751 | 0.754 | 0.684 | 0.655 | **0.640** | 0.478 | **0.364** | **0.548ᴾ** |
| MutateEverything st1 (3 seeds) | 0.707 | 0.681 | 0.646 | 0.581 | 0.580 | 0.483 | 0.296 | 0.520 |
| MutateEverything st1, additive | 0.729 | 0.714 | 0.690 | 0.615 | 0.603 | 0.483 | 0.287 | – |
| MutateEverything st2 (3 seeds) | 0.645 | 0.635 | 0.605 | 0.598 | 0.560 | 0.439 | 0.253 | 0.509 |
| MutateEverything st2, additive | 0.715 | 0.699 | 0.669 | 0.642 | 0.599 | 0.439 | 0.281 | – |
| ThermoMPNN (additive, 3 seeds) | 0.626 | 0.632 | 0.613 | 0.569 | 0.570 | 0.405 | 0.319 | 0.450 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | 0.568 | 0.582 | 0.560 | 0.549 | 0.535 | 0.405 | 0.318 | – |
| SPURS (single model) | 0.689 | 0.711 | 0.736 | 0.635 | 0.602 | 0.535 | 0.330ᴾ | 0.513ᴹ |
| SPURS multi (multi-mutant model) | 0.612 | 0.624 | 0.600 | 0.502 | 0.556 | 0.375 | 0.258ᴾ | 0.406ᴾ |
| ESMtherm | 0.093 | 0.136 | 0.484 | 0.466 | 0.229 | -0.000 | 0.214ᴾ | 0.201ᴾ |
| DDGemb (server) | 0.103 | 0.123 | 0.457 | 0.469 | 0.475 | 0.188 | 0.154 | 0.351ᴾ |
| IFUM, WT backbone | 0.766 | 0.768 | 0.648 | 0.565 | 0.507 | 0.190 | 0.262ᴾ | 0.464ᴾ |
| IFUM, ESMFold per variant | 0.762 | 0.747 | 0.697 | 0.575 | 0.523 | 0.217 | 0.284ᴾ | 0.473ᴾ |
| GeoDDG-3D (FASPR) | 0.351 | 0.431 | 0.604 | 0.481 | 0.395 | 0.240 | 0.200ᵐ | 0.413ᴾ |
| ESM3ΔG | 0.763 | 0.766 | **0.749** | 0.646 | 0.594 | 0.280 | 0.324 | 0.512ᴴ |
| ProSST (zero-shot) | **0.776** | **0.796** | 0.711 | 0.702 | 0.538 | 0.356 | 0.258 | 0.519ᶜ |
| ProteinMPNN (zero-shot) | 0.619 | 0.613 | 0.667 | 0.538 | 0.498 | **0.544** | 0.119 | **0.548ᶜ** |
| ESM3-small-open, σ = 0 (zero-shot) | 0.689 | 0.742 | 0.627 | 0.723 | 0.536 | 0.483 | 0.259 | 0.540ᶜ |
| ESM3-large (Forge, zero-shot) | 0.624 | 0.606 | 0.557 | 0.582 | 0.458 | 0.432 | – | 0.510ᶜ |
| ESM3-medium (Forge, zero-shot) | 0.734 | 0.779 | 0.664 | – | – | – | – | – |
| ESM3-small (Forge, zero-shot) | 0.769 | 0.785 | 0.673 | **0.731** | 0.568 | 0.498 | – | – |

| model | DLG4 abund. (leak code) | DLG4 bind. (leak code) | GRB2 abund. (leak code) | GRB2 bind. (leak code) | MYO (leak code) | ESTA dTm (leak code) | GB1 (leak code) | Domainome (leak code) |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), combined (3 seeds) | – | – | – | – | – | – | – | M.002 P.006 H.08 h.02 |
| ESM-MSR (baseline), WT head (3 seeds) | – | – | – | – | – | – | – | M.002 P.006 H.08 h.02 |
| ESM-MSR (devel), combined (3 seeds) | – | – | – | – | – | – | – | M.002 P.006 H.08 h.02 |
| ESM-MSR (devel), WT head | – | – | – | – | – | – | – | M.002 P.006 H.08 h.02 |
| MutateEverything st1 (3 seeds) | – | – | – | – | – | – | – | M.002 P<.001 |
| MutateEverything st1, additive | – | – | – | – | – | – | – | – |
| MutateEverything st2 (3 seeds) | – | – | – | – | – | – | – | M.002 P<.001 |
| MutateEverything st2, additive | – | – | – | – | – | – | – | – |
| ThermoMPNN (additive, 3 seeds) | – | – | – | – | – | – | – | M.002 P<.001 |
| ThermoMPNN + ThermoMPNN-D (3 seeds) | – | – | – | – | – | – | – | – |
| SPURS (single model) | – | – | – | – | – | – | P1 | M.01 P.03 H.24 h.05 |
| SPURS multi (multi-mutant model) | – | – | – | – | – | – | P1 | m.002 P.03 H.22 h.06 |
| ESMtherm | – | – | – | – | – | – | P1 | P.05 M?.04 H.26 h.06 |
| DDGemb (server) | – | – | – | – | – | – | – | P.03 H.14 h.06 |
| IFUM, WT backbone | – | – | – | – | – | – | P1 | P.05 M?.05 H.26 h.05 |
| IFUM, ESMFold per variant | – | – | – | – | – | – | P1 | P.05 M?.05 H.26 h.05 |
| GeoDDG-3D (FASPR) | – | – | – | – | – | – | m.41 M<.001 | P.05 H.22 h.08 |
| ESM3ΔG | – | – | – | – | – | – | – | H.20 h.32 |
| ProSST (zero-shot) | – | – | – | – | – | – | – | C |
| ProteinMPNN (zero-shot) | – | – | – | – | – | – | – | C |
| ESM3-small-open, σ = 0 (zero-shot) | – | – | – | – | – | – | – | C |
| ESM3-large (Forge, zero-shot) | – | – | – | – | – | – | – | C |
| ESM3-medium (Forge, zero-shot) | – | – | – | – | – | – | – | – |
| ESM3-small (Forge, zero-shot) | – | – | – | – | – | – | – | – |
- **Mean over the 7 DMS** (models with all seven): ESM-MSR (devel) WT head 0.618, (devel) combined 0.609, SPURS 0.605, ESM-MSR (released) 0.601, ProSST 0.591, ME st1 additive 0.589, ESM3ΔG 0.589, ESM3-small-open σ = 0 0.580, ThermoMPNN 0.533, ProteinMPNN 0.514, SPURS multi 0.504, GeoDDG 0.386, DDGemb 0.281, ESMtherm 0.232. **The GB1 column is not clean for four of these:** SPURS (same-protein via a near-identical Megascale design), ESMtherm and IFUM (trained on the Megascale 1PGA/GB1 library) and GeoDDG (1,538 1PGA/GB1 single mutants in S8754; 61 of the Wu rows exact and 61,880 constituent); ESM-MSR/ME/ThermoMPNN are clean on it because GB1 was quarantined from the capped split. GB1 is the lowest of the seven for every model, so dropping it moves the means up by 0.00–0.07 and leaves the top three unchanged (ESM-MSR devel WT 0.660, devel 0.654, SPURS 0.651); ProSST moves from fifth to fourth, ahead of ESM-MSR (released).
- **The literature-ΔΔG models invert here:** GeoDDG and DDGemb, which lead the external sets, are near the bottom
  on the deep-mutational scans; ProSST, a zero-shot model, leads both DLG4 assays (0.776 and 0.796) among the
  models that cover all seven. ESM3-small through Forge is also ahead of ESM-MSR (devel) on the Faure assays it
  covers (DLG4 0.769 and 0.785, GRB2 binding 0.731, against 0.745, 0.742 and 0.636; it has no GB1).
- **These assays discriminate much less than the stability benchmarks do:** the zero-shot σ = 0 control (0.580) is
  within 0.04 of the best supervised model, against a 0.27 gap on Tsuboyama. Rankings this tight should not be
  read as model quality without interval estimates (§1.5 has them for the published subset).
- **Domainome is not clean either.** Domainome's domain `O95155_PF04564_1229` (UBE4B) is the same protein as Megascale library 3L1X, which is in the capped train split: 1,134 of its 1,206 rows (0.21 % of Domainome) are identical mutations in ESM-MSR's, ME's and ThermoMPNN's training data, and 3,177 more rows are same-protein. SPURS adds 0.1 % exact. ESMtherm (15,335 rows), IFUM (16,991), DDGemb (P 3 %) and GeoDDG (P 5 %) are same-protein on other domains, and ESM3ΔG has strong homologs for 19.8 % of rows in its MGnify fine-tune. The 0.548 leaderboard is therefore an upper bound for several models; the fully memorised domain is 1 of 522 in a per-domain mean, so its direct effect on that mean is at most about 0.001 ρ, but its own score should be discarded.

#### ESM-MSR LoRA scale σ on the DMS and Domainome

ProteinGym showed that the right LoRA scale depends on the assay (§3), so the DMS and Domainome blocks are rerun at
σ = 0.75, 0.5 and 0.25 for both checkpoints (devel rows are seed means over all three seeds, shown only once every
seed has finished; the combined head is shown).

| model | DLG4 abund. | DLG4 bind. | GRB2 abund. | GRB2 bind. | MYO | ESTA dTm | GB1 | Domainome |
|---|---|---|---|---|---|---|---|---|
| released, σ = 1 | 0.726 | 0.725 | 0.673 | 0.619 | **0.639** | 0.479 | 0.346 | 0.542 |
| released, σ = 0.75 | 0.751 | 0.764 | 0.707 | 0.660 | **0.639** | 0.510 | 0.340 | 0.560 |
| released, σ = 0.5 | 0.762 | 0.791 | **0.718** | 0.696 | 0.627 | **0.526** | 0.342 | **0.572** |
| released, σ = 0.25 | 0.748 | 0.793 | 0.688 | 0.721 | 0.596 | 0.516 | 0.320 | 0.568 |
| devel, σ = 1 | 0.745 | 0.742 | 0.682 | 0.636 | **0.639** | 0.481 | 0.337 | 0.543 |
| devel, σ = 0.75 | 0.762 | 0.773 | 0.713 | 0.674 | **0.639** | 0.510 | 0.345 | 0.560 |
| devel, σ = 0.5 | **0.768** | **0.796** | 0.716 | 0.705 | 0.627 | 0.525 | **0.349** | 0.570 |
| devel, σ = 0.25 | 0.749 | 0.793 | 0.687 | **0.726** | 0.596 | 0.516 | 0.318 | 0.567 |
| σ = 0 (zero-shot control) | 0.689 | 0.743 | 0.627 | 0.723 | 0.536 | 0.483 | 0.260 | 0.540 |

- **Released, a lower LoRA scale is clearly better on these assays.** At σ = 0.5 the 7-DMS mean is 0.637 against
  0.601 at σ = 1 (0.624 at 0.75, 0.626 at 0.25), and σ = 0.5 beats σ = 1 on 6 of the 8 columns,
  including Domainome (0.572 against 0.542), DLG4 binding (0.791 against 0.725) and GRB2 binding (0.696 against
  0.619; 0.721 at σ = 0.25). Only MYO and GB1 prefer σ = 1, and by at most 0.012. This is the same direction as
  ProteinGym's functional categories (§3), and at σ = 0.5 ESM-MSR (released) is ahead of ProSST (0.591) on the 7-DMS
  mean.
- **The caveat that goes with it:** the best σ per assay is chosen by looking at these same assays, so these numbers
  are an upper bound on what tuning the scale buys, not a held-out result. The stability benchmarks (§1.1, §1.2) use
  σ = 1, fixed in advance, and a fair version of this comparison fixes one σ before looking.
- The σ = 0 control (LoRA off) shows the other end: it scores 0.723 on GRB2 binding, above both checkpoints at
  σ = 1, and ties ESM-MSR (devel) on DLG4 binding (0.742); fine-tuning on stability data is what costs these assays.
  Devel at the three intermediate scales is now complete (all three seeds).
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
| MutateEverything st2 (3 seeds) | q | q | q |
| ThermoMPNN (additive, 3 seeds) | 0.441ᴹ | 0.781ᴹ | 0.484ᴹ |
| SPURS (single model) | 0.483ᴹ | 0.837ᴹ | 0.525ᴹ |
| SPURS multi (multi-mutant model) | 0.426ᴹ | 0.808ᴹ | 0.477ᴹ |
| ESMtherm | 0.213ᴾ | 0.720ᴾ | 0.283ᴾ |
| GeoDDG-3D (FASPR) | 0.383 | 0.638 | 0.422 |
| ESM3ΔG | 0.380 | 0.788 | 0.441 |
| ProSST (zero-shot) | **0.514ᶜ** | **0.653ᶜ** | **0.528ᶜ** |
| Rosetta Cartesian ΔΔG | _0.375_ | _0.640_ | _0.409_ |
| ProteinMPNN (zero-shot) | 0.408ᶜ | 0.627ᶜ | 0.426ᶜ |
| ProteinMPNN, ProteinGym published recipe | 0.257ᶜ | 0.565ᶜ | 0.298ᶜ |
| ESM3-small-open, σ = 0 (zero-shot) | 0.448ᶜ | 0.620ᶜ | 0.470ᶜ |
| ESM3-open, ProteinGym published recipe | 0.466ᶜ | 0.641ᶜ | 0.486ᶜ |

| model | PG avg (leak code) | Stability (leak code) | DMS mean (leak code) |
|---|---|---|---|
| ESM-MSR (released), combined | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 |
| ESM-MSR (released), WT head | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 |
| ESM-MSR (devel), combined (3 seeds) | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 | M.02 m<.001 h<.001 |
| MutateEverything st1 (3 seeds) | M.010 m.009 P<.001 | M.010 m.009 P<.001 | M.010 m.009 P<.001 |
| MutateEverything st2 (3 seeds) | M.02 m<.001 | M.02 m<.001 | M.02 m<.001 |
| ThermoMPNN (additive, 3 seeds) | M.010 m.009 P<.001 | M.010 m.009 P<.001 | M.010 m.009 P<.001 |
| SPURS (single model) | M.02 m.02 P.008 h<.001 | M.02 m.02 P.008 h<.001 | M.02 m.02 P.008 h<.001 |
| SPURS multi (multi-mutant model) | M.02 m.002 P.02 h<.001 | M.02 m.002 P.02 h<.001 | M.02 m.002 P.02 h<.001 |
| ESMtherm | P.04 M?.03 H<.001 h<.001 | P.04 M?.03 H<.001 h<.001 | P.04 M?.03 H<.001 h<.001 |
| GeoDDG-3D (FASPR) | n.a. | n.a. | n.a. |
| ESM3ΔG | n.a. | n.a. | n.a. |
| ProSST (zero-shot) | C | C | C |
| Rosetta Cartesian ΔΔG | – | – | – |
| ProteinMPNN (zero-shot) | C | C | C |
| ProteinMPNN, ProteinGym published recipe | C | C | C |
| ESM3-small-open, σ = 0 (zero-shot) | C | C | C |
| ESM3-open, ProteinGym published recipe | C | C | C |
- **ME stage 2 is `q`: its ProteinGym run predates the data correction.** The item-filtered stage 2 scored 0.445 / 0.806 /
  0.491 (PG avg / Stability / DMS mean); the corrected seed-1 run (job 387353, one GPU, pinned to gpu4) and a matching
  ThermoMPNN-D run (job 387352) are queued and also record per-DMS time and peak memory.
- **ProSST ties ESM-MSR on the average only because of 23 fully memorised DMS.** 23 of the 213 ProteinGym assays are Tsuboyama DMS whose every row is in the capped train split used by ESM-MSR, ME and ThermoMPNN: 45,623 of 2,446,730 rows (1.9 % of the benchmark, 10.8 % of the assays). They are all Stability assays and ESM-MSR scores 0.897 on them. Removing them (local devel-seed reruns, 190 DMS) moves ESM-MSR's DMS mean from 0.561 to 0.520, its Stability category mean from 0.880 to 0.871, and its PG average from 0.516 to 0.514. ProSST, zero-shot on those assays, is 0.528 on the all-213 DMS mean and 0.514 on the all-213 PG average; after removing the 23 assays ESM-MSR is at 0.520 and the two PG averages are no longer directly comparable (§6.5). The same 23 DMS are in the training data of ME stage 1/2 (24,252 exact + 21,331 constituent) and ThermoMPNN (24,252 exact singles), and SPURS is worse: 51 of 213 DMS have ≥ 50 % of their rows in its fit.
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
  mirror image of ESMtherm and the same inversion §1.3 records for the literature-ΔΔG models. Its ProteinGym overlap
  composition has not been audited, so its row in the leak-code table below is `n.a.` (not audited, so not eligible for bold),
  like ESM3ΔG's.
- **Every row in this table is complete.** ESM3ΔG finished on 2026-10-26 (386364, 3 d 23 h, 213 of 213; the last
  dataset, `SPG1_STRSG_Olson_2014`, took 25.5 h on its own) and GeoDDG on 2026-10-25 (386804, 1 d 08 h). GeoDDG's
  first attempt had been killed by its own two-day wall clock at 148 DMS, and because the harness skips datasets whose
  output already exists, the resumed job redid none of that work. IFUM's ProteinGym run was stopped as not worth the
  time it needed, and DDGemb and DDMut are web services that cannot be run at this scale.

**Row caveat:** the ESM-MSR rows in this table are the *single* released checkpoint, not the §1.1–1.3 triplicate —
see the header note. The triplicate's ProteinGym pass was still running when this table was last built.

The σ sweep and per-category breakdown are in §3.

### 1.5 Published-paper evaluation (reference; earlier models and row sets)


From `analysis_notebooks/bootstrap_external_results/rho_all_models_replicate1.csv`: the models of the
published comparison, trained on the hyperopt split. Its row sets are de-duplicated (S8754 → 6,411 rows) and it has
neither S669 nor K3822 (it uses their subsets S461 and K2369), so compare within this table only. Its Rosetta column
is the earlier run on the older split; the October capped-split run (§1.2) is consistent with it. The ME columns are the pre-October models, trained with the
label sign flipped relative to the upstream recipe (§5, item 13).

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


## 2. Epistasis: ESM-MSR vs the retrained baselines and the third-party models

**Leak note (§6).** The capped val/test rows used here are clean for ESM-MSR, MutateEverything and ThermoMPNN (identical mutations 0, same protein 0). SPURS multi is the only third-party model that predicts doubles directly *and* was trained on these libraries: 6,400 of its 15,545 test doubles are exact (41.2 %) and 1,717 more are constituent. ESMtherm is same-protein on 82 % of test rows and IFUM on 45 %; ESM3ΔG is zero-shot with 40.7 % strong/24.6 % weak homologs. Read the ‡ rows (documented Megascale training; §6.5 quantifies it) accordingly; the "Among models that did not train on Tsuboyama" comparison is the only like-for-like interaction comparison in this section. Model names carry the Tsuboyama leak superscript of §6.


**Rule throughout: neither baseline was modified, only its data layer.** ME follows the upstream README
recipe (stage 1 = singles, 20 epochs; stage 2 = doubles, 100 epochs from stage 1). ThermoMPNN is two
upstream configs, i.e. two models: `train_single.yaml` (singles) and `train_epistatic.yaml` (ThermoMPNN-D,
siamese; the measured doubles plus upstream's augmented doubles). Each baseline is scored by its direct double prediction against an additive expectation made
of a model that never trained on doubles, which is the role the WT head plays for ESM-MSR: ME stage 2 against the stage-1
singles of the same seed (predicted epistasis ME2(AB) − ME1(A) − ME1(B)), ThermoMPNN-D against the single ThermoMPNN's
(TMD(AB) − TM(A) − TM(B)), ME stage 1 (untrained double head) against its own singles; ESM-MSR by its combined head.
With ME stage 2's own singles as the baseline instead, test naive ρ is 0.477 rather than 0.405 and the other levels
move by up to 0.1 (§2.2); the stage-1 baseline is the one that parallels ESM-MSR and is used throughout. Rosetta scores a double directly with
`cartesian_ddg`, so it has both as well (`analysis_scripts/epi_rosetta.py`, through the same `assemble` and
`score` as the others). So does every third-party model that scores a double itself (ESMtherm, IFUM in both modes,
GeoDDG-3D, ESM3ΔG, SPURS multi, ProteinMPNN, DDGemb; `analysis_scripts/epi_harness.py`), each against the sum of its
own singles. ProSST is left out because its double *is* the sum of its singles on every row, so it has no
interaction to score. That is the published method and not an adapter artefact: the paper's zero-shot score
(Appendix A.1, Eq. 8) is Σ over the mutated positions of log P(x_i = F_i | x, s) − log P(x_i = w_i | x, s) with x and
s the wild-type sequence and structure tokens, the authors' `zero_shot/proteingym_benchmark.py` does exactly that in
one forward pass, and their published ProteinGym predictions agree with the sum of their own singles to 2e-14 over
1,499,350 multi-mutants in 69 DMS. A non-additive reading of the same network (scoring the mutant sequence, or
conditioning each position on the other substitutions) is possible but is not the published recipe and was not run.
DDMut and UniMutStab have no predictions on this split. Models marked ‡ (EsmTherm, IFUM, SPURS multi) trained on Megascale,
which includes these libraries' doubles, so they are shown but never bolded. `ctx` does not transfer to the
baselines (no separable adapter). Rows are models; seed means. Metric definitions: `validation_metrics.md`. The
table is regenerated by `benchmark_matrix.py epi`. The superscripts on the model names are the Tsuboyama leak classes of §6.

### 2.0 Authoritative epistasis metrics (2026-10-24)

The metric set changed on 2026-10-24 and this is now the authoritative one (`src/esm_msr/epi_metrics.py`,
`docs/validation_metrics.md`). The epistasis head is **`epi` = ~MT(AB) − WT(AB)**: the MT pass read against the WT
adapter's additive baseline, which under `--per_mutation_bias` is additive by construction. `comb` (the 50/50 WT+MT
mix) is no longer used for the residual levels, because halving the epistatic contrast compresses every residual by
2x and pushes >97 % of variants into the neutral band. `global_err`/`global_bias` are anchored to `epi` and evaluate the predicted
epistasis dddG' against the *measured* dddG on double mutants, no longer against observed singles (which confounded
epistasis bias with single-mutant shrinkage). Three statistics are new: `beyond_lib` (each library's mean removed from the residuals, so
between-library offsets do not count), `beyond_rho_pairavg` (mean within-position-pair Spearman), and a three-class
residual accuracy (negative < −0.6, neutral, positive > +0.6 kcal/mol).

Every ESM-MSR row here was rescored from its saved predictions — nothing was re-run, because `epi_pred =
mt_lora_pred − wt_lora_pred` is already carried in the prediction CSVs. The retrained ME stage 2 and ThermoMPNN-D rows
are scored by the same suite from their saved predictions (3 seeds, corrected runs). The population is unchanged and the numbers
are comparable to §2.1–2.2: same splits, censored rows excluded, 8,494 and 12,455 doubles.

**Validation (valfull: 8,494 doubles, 24 pairs, 2,509 flips)**

| model | naive ρ | beyond ρ | beyond_lib | pair-avg ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | class3 acc | global_err ↓ | global_bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), 3 seeds | **0.654±0.003** | 0.586±0.018 | 0.434±0.033 | 0.377±0.045 | 0.745±0.024 | 0.395±0.050 | 0.309±0.004 | 0.193±0.004 | 0.845±0.009 | 0.683±0.012 | 0.717±0.036 | 0.177±0.081 |
| ESM-MSR (devel), 3 seeds | 0.643±0.032 | 0.585±0.017 | 0.437±0.031 | 0.406±0.011 | 0.734±0.023 | 0.434±0.011 | 0.315±0.014 | 0.195±0.007 | 0.854±0.023 | 0.657±0.011 | 0.824±0.019 | 0.275±0.030 |
| ESM-MSR (trim/mask), 3 seeds | 0.371 | **0.659** | **0.554** | **0.460** | **0.868** | **0.514** | **0.328** | **0.217** | **0.906** | 0.685 | 0.915 | -0.313 |
| MutateEverything stage 2 (vs stage-1 singles), 3 seeds | 0.421 | 0.311 | 0.141 | 0.153 | 0.467 | 0.161 | 0.093 | 0.072 | 0.736 | 0.668 | **0.803** | **-0.127** |
| ThermoMPNN-D (vs ThermoMPNN), 3 seeds | 0.190 | 0.085 | 0.108 | 0.150 | 0.060 | 0.203 | 0.048 | 0.023 | 0.655 | **0.692** | 0.979 | -0.435 |

**Test (12,455 doubles, 43 pairs, 4,236 flips)**

| model | naive ρ | beyond ρ | beyond_lib | pair-avg ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | class3 acc | global_err ↓ | global_bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (baseline), 3 seeds | **0.452±0.013** | 0.484±0.016 | 0.343±0.018 | 0.304±0.018 | 0.637±0.006 | 0.361±0.026 | 0.240±0.008 | 0.217±0.006 | 0.761±0.003 | 0.604±0.011 | 0.954±0.042 | 0.280±0.109 |
| ESM-MSR (devel), 3 seeds | 0.464±0.015 | **0.487±0.020** | 0.361±0.030 | 0.322±0.021 | **0.606±0.034** | 0.382±0.026 | 0.236±0.003 | 0.209±0.006 | 0.754±0.008 | **0.590±0.008** | **1.079±0.024** | 0.405±0.021 |
| ESM-MSR (trim/mask), 3 seeds | 0.411 | 0.463 | **0.400** | **0.376** | 0.594 | **0.416** | **0.260** | **0.224** | **0.803** | 0.589 | 0.865 | **-0.088** |
| MutateEverything stage 2 (vs stage-1 singles), 3 seeds | 0.405 | 0.223 | 0.165 | 0.211 | 0.124 | 0.225 | 0.134 | 0.112 | 0.665 | 0.595 | 0.922 | 0.255 |
| ThermoMPNN-D (vs ThermoMPNN), 3 seeds | 0.077 | 0.071 | 0.124 | 0.166 | -0.003 | 0.197 | 0.022 | 0.016 | 0.601 | 0.578 | 1.030 | -0.224 |

Two things this set makes sharper than the old one. On validation the trim seeds lead every interaction column, with
the largest margin on `beyond_lib` (0.554 against 0.440) — so it is not a between-library offset. On test they lead
`cell_mag`, `cell_rank`, `flipacc` and `beyond_lib` while trailing on `beyond ρ` (0.463 against 0.494): the pooled
correlation still carries the per-library level agreement that `beyond_lib` strips out. `global_err` and
`global_bias` evaluate the point-wise RMSE and mean bias of the predicted epistasis $\sim\text{MT}(AB) - \text{WT}(AB)$
directly against measured epistasis $\delta\Delta\Delta G$ on double mutants; negative bias reflects under-predicted epistasis.

The change is specific to ESM-MSR: it redefines how that model's WT and MT passes are compared, and no other model has
that pair of heads. The baselines and third-party rows of §2.1–2.2 enter the suite as head `epi` (the model's direct double
against its own additive expectation). Rescoring them on 2026-10-27 reproduced their earlier naive, beyond, pair, line,
cell and flip columns to within 0.001 and changed only `global_err`, which they had from before the RMSE fix; those
values are updated. Those sections now also carry the ESM-MSR rows under the authoritative metrics above; §2.0 is where the columns that did not
exist before live (`beyond_lib`, `beyond_rho_pairavg`, three-class accuracy, `global_bias`).

### 2.1 Validation, epistasis hierarchy (valfull: 34,957 rows, 8,494 doubles, 24 pairs, 2,509 flips)

| model | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| MutateEverything stage 1 | -0.024 | -0.007 | 0.124 | -0.010 | -0.016 | 0.007 | 0.517 | 1.189 |
| MutateEverything stage 2 | 0.421 | 0.311 | 0.467 | 0.161 | 0.093 | 0.072 | 0.736 | **0.803** |
| ThermoMPNN-D | 0.190 | 0.085 | 0.060 | 0.203 | 0.048 | 0.023 | 0.655 | 0.979 |
| Rosetta Cartesian ΔΔG | 0.269 | 0.267 | 0.421 | 0.168 | 0.068 | 0.060 | 0.620 | 3.066 |
| ESMtherm ‡ ᴾ | 0.486 | 0.471 | 0.697 | 0.384 | 0.258 | 0.171 | 0.884 | 1.069 |
| IFUM, WT backbone ‡ ᴾ | 0.300 | 0.351 | 0.594 | 0.248 | 0.121 | 0.076 | 0.759 | 1.084 |
| IFUM, ESMFold per variant ‡ ᴾ | 0.292 | 0.316 | 0.657 | 0.189 | 0.097 | 0.076 | 0.771 | 1.067 |
| GeoDDG-3D (FASPR) ʰ | 0.137 | 0.112 | 0.240 | 0.080 | 0.089 | 0.073 | 0.736 | 1.277 |
| ESM3ΔG ᴴ | 0.305 | 0.447 | 0.587 | 0.346 | 0.211 | 0.149 | 0.856 | 1.044 |
| SPURS multi (multi-mutant model) ‡ ᴹ | 0.230 | 0.098 | 0.602 | -0.143 | 0.148 | 0.059 | 0.513 | 0.925 |
| ProteinMPNN (zero-shot) ᶜ | 0.059 | 0.152 | 0.123 | 0.151 | 0.141 | 0.077 | 0.669 | 1.096 |
| DDGemb (server) ʰ | 0.164 | 0.370 | 0.728 | 0.143 | 0.062 | 0.045 | 0.616 | 1.295 |
| ESM-MSR (baseline) ʰ | **0.654±0.003** | **0.586±0.018** | **0.745±0.024** | 0.395±0.050 | **0.309±0.004** | 0.193±0.004 | **0.845±0.009** | 0.717±0.036 |
| ESM-MSR (devel) (3 seeds) ʰ | 0.643±0.032 | 0.585±0.017 | 0.734±0.023 | **0.434±0.011** | 0.315±0.014 | 0.195±0.007 | 0.854±0.023 | 0.824±0.019 |

### 2.2 Test, epistasis hierarchy (12,455 doubles, 43 pairs, 4,236 flips)

| model | naive ρ | beyond ρ | pair ρ | line ρ | cell_mag | cell_rank | flipacc | global_err ↓ |
|---|---|---|---|---|---|---|---|---|
| MutateEverything stage 1 | 0.029 | 0.018 | -0.059 | 0.038 | -0.024 | -0.010 | 0.519 | 1.098 |
| MutateEverything stage 2 | 0.405 | 0.223 | 0.124 | 0.225 | 0.134 | 0.112 | 0.665 | 0.922 |
| ThermoMPNN-D | 0.077 | 0.071 | -0.003 | 0.197 | 0.022 | 0.016 | 0.601 | 1.030 |
| Rosetta Cartesian ΔΔG | 0.381 | 0.207 | 0.258 | 0.195 | 0.094 | 0.066 | 0.556 | 3.083 |
| ESMtherm ‡ ᴾ | 0.495 | 0.507 | 0.635 | 0.550 | 0.250 | 0.178 | 0.790 | 0.806 |
| IFUM, WT backbone ‡ ᴾ | 0.002 | 0.247 | 0.315 | 0.254 | 0.107 | 0.068 | 0.617 | 1.113 |
| IFUM, ESMFold per variant ‡ ᴾ | 0.102 | 0.304 | 0.503 | 0.230 | 0.093 | 0.069 | 0.649 | 1.132 |
| GeoDDG-3D (FASPR) ʰ | 0.055 | 0.046 | 0.083 | 0.087 | 0.077 | 0.054 | 0.589 | 1.195 |
| ESM3ΔG ᴴ | 0.413 | 0.462 | 0.571 | 0.384 | 0.289 | 0.203 | 0.794 | 0.956 |
| SPURS multi (multi-mutant model) ‡ ᴹ | 0.137 | 0.132 | 0.494 | -0.052 | 0.092 | 0.082 | 0.549 | 0.894 |
| ProteinMPNN (zero-shot) ᶜ | 0.314 | 0.273 | 0.473 | 0.125 | 0.125 | 0.064 | 0.592 | 0.992 |
| DDGemb (server) ʰ | 0.031 | 0.122 | 0.242 | -0.019 | 0.028 | 0.028 | 0.550 | 1.108 |
| ESM-MSR (baseline) ʰ | **0.452±0.013** | 0.484±0.016 | 0.637±0.006 | 0.361±0.026 | **0.240±0.008** | 0.217±0.006 | 0.761±0.003 | 0.954±0.042 |
| ESM-MSR (devel) (3 seeds) ʰ | 0.464±0.015 | **0.487±0.020** | **0.606±0.034** | **0.382±0.026** | 0.236±0.003 | **0.209±0.006** | **0.754±0.008** | **1.079±0.024** |

Largest seed sd per model, validation: ME stage 1 0.364, ME stage 2 0.077, ThermoMPNN-D 0.134, ESM-MSR (devel) 0.057; test:
ME stage 1 0.201, ME stage 2 0.062, ThermoMPNN-D 0.076, ESM-MSR (devel) 0.035. A difference smaller than these is not
a result.

*What the baseline rows are.* ME stage 2: predicted epistasis ME2(AB) − ME1(A) − ME1(B) with the stage-1 singles of the
same seed (with its own singles instead, test: naive ρ 0.477, beyond ρ 0.316, pair ρ 0.322, line ρ 0.283, global_err 0.745,
cell_rank and flip accuracy unchanged). ME stage 1: its untrained double head against its own singles, so about zero
by construction. ThermoMPNN-D: TMD(AB) − TM(A) − TM(B). A baseline must be passed to the suite as head `epi`; under the older
head name `mt` the suite returns the raw double prediction as the predicted epistasis, which turns ME stage 2's naive ρ
from +0.405 into −0.164.

- **The metric overhaul moves the ESM-MSR numbers, not the ranking.** A bold cell is the best value among the models
  without more than minor overlap with training (§6.6: no ‡, no `M`/`m`/`P`/`H` flag). On test that is (devel) beyond ρ
  0.494, pair ρ 0.618, line ρ 0.384, cell_rank 0.209, flip accuracy 0.754 and the lowest global error 0.852, and
  (released) naive ρ 0.452 and cell_mag 0.247. The ‡ models and ESM3ΔG show higher raw values in several columns (ESMtherm
  leads line ρ at 0.550, ESM3ΔG cell_mag at 0.289); they did so before the overhaul too, and they stay unbolded because
  they trained on Tsuboyama or have strong homologs in training. The one ordering the overhaul does change is *inside*
  the ESM-MSR rows: the cell_rank lead moves from (released) 0.176 to (devel) 0.209. On validation ESM-MSR leads seven of
  the eight columns — (released) naive ρ 0.675, beyond ρ 0.598, pair ρ 0.730, cell_mag 0.327, cell_rank 0.207 and flip
  accuracy 0.860, (devel) line ρ 0.440 — with global error going to ME stage 2 (0.803 against 0.849 for ESM-MSR released).
- **Among models that did not train on Tsuboyama, ESM-MSR still leads the interaction metrics.** On test, beyond ρ is
  0.494 (devel) and 0.488 (released) against 0.223 for ME stage 2, 0.273 for ProteinMPNN and 0.207 for Rosetta;
  cell_rank is 0.198–0.209 against 0.112 (ME stage 2) and 0.064–0.066 (ProteinMPNN, Rosetta). No model outside
  ESM-MSR reaches 0.21 on cell_rank. ME stage 2 is the best retrained baseline, and its predicted epistasis does track
  the measured one (naive ρ 0.405 on test and 0.421 on validation, against 0.435–0.452 and 0.370–0.675 for ESM-MSR),
  which a model trained on doubles should do; naive ρ is mostly saturation, though, and its within-column order (cell_rank
  0.112, flip accuracy 0.665) is far below ESM-MSR's. ThermoMPNN-D's naive ρ is 0.077 on test and 0.190 on validation.
- **The Megascale-trained models are not a held-out comparison, and one of them should not have been flagged at all.** On test SPURS multi has 41.2 % of doubles exact, ESMtherm 82 % of rows same-protein (mutant-level split) and IFUM 45 % same-protein; all three are trained on the Megascale libraries this split uses. ESM3ΔG was labelled ‡ but its released weights are fine-tuned on the authors' own 960,216-sequence MGnify Stability set, not on Megascale: 0 of the 33 test libraries are in training, 8 have a strong homolog (40.7 % of test rows) and 12 a weak one. Its 0.811/0.811 is a genuine zero-shot result with a homology caveat, not a training leak.
- **Pair ρ is noisy for the models that are not trained here.** ProteinMPNN scores 0.123 on validation and 0.473 on
  test; DDGemb scores 0.728 (the best validation value of any model) and 0.242. There are only 24 (validation) and
  43 (test) pairs, so a few of them move it; read the within-column and flip metrics instead.
- **GeoDDG's and SPURS multi's doubles carry little usable interaction.** GeoDDG beyond ρ is 0.046 (test) and 0.112
  (validation). SPURS multi has pair ρ 0.494 and 0.602 but a *negative* line ρ (−0.052, −0.143): it separates pairs
  of residues and then orders the substitutions within a line backwards.
- **Rosetta carries real interaction signal** from a physics model that knows nothing of this data: test beyond ρ
  0.207 and pair ρ 0.258, validation 0.267 and 0.421. That is above ThermoMPNN-D on beyond ρ, pair ρ and cell_mag on both
  halves (test 0.207 / 0.258 / 0.094 against 0.071 / −0.003 / 0.022; validation 0.267 / 0.421 / 0.068 against 0.085 /
  0.060 / 0.048). On test it is also above ME stage 2 on pair ρ (0.258 against 0.124), and below ME stage 2 and ESM-MSR
  on the other levels and on naive ρ (0.381 against 0.405 and 0.435–0.452).
- **ESM-MSR is shown by its combined head.** Its MT head alone scores higher on the within-column levels (devel,
  test: beyond ρ 0.511, pair ρ 0.660, cell_rank 0.210 against 0.489, 0.627, 0.173 combined) and worse on global
  error; the choice of the combination weight is §2.6.
- Additive heads score exactly 0 / 0.5 on every residual and rank metric, the suite's own correctness check, and
  `global_err` equal to the RMS of the measured dddG, since their predicted dddG is exactly 0 by construction.
- Every destabilized-backbone test row is a single, so these epistasis columns are identical with or without those
  libraries (re-verified with the current `dG_wt`: max difference 0 over all 17 `epi_*` columns and 18 rows); only
  the pooled ddG metrics move, by up to 0.01 on `spearman_all`.

### 2.3 ddG ranking (mean per-protein Spearman)

Each ESM-MSR checkpoint is shown by its WT (additive) head and by its combined prediction, beside the retrained
baselines. Baselines are 3-seed means; the devel columns are means ± sd over 3 seeds; the released checkpoint is one
run. TMPNN is the singles-only ThermoMPNN, which makes no pair-specific prediction, so it has no dddG: that is the
only gap. Regenerated by `benchmark_matrix.py ddg`, with the ME stage 2 and ThermoMPNN+D columns replaced on 2026-10-27 by the corrected runs. The validation numbers of all
retrained baselines are on the full 34,957-row validation population (ME stage 1 and ThermoMPNN were rescored on it on 2026-10-28).

| | ME st1 | ME st2 | TMPNN | TMPNN+D | ESM-MSR (released) WT ʰ | ESM-MSR (released) combined ʰ | ESM-MSR (devel) WT ʰ | ESM-MSR (devel) combined ʰ |
|---|---|---|---|---|---|---|---|---|
| val all variants (`spearman_all`) | 0.754 | 0.730 | 0.787 | 0.785 | 0.841 | **0.854** | 0.836 ± 0.001 | **0.854 ± 0.001** |
| val doubles (`spearman_doubles`) | 0.571 | 0.629 | 0.630 | 0.688 | 0.691 | 0.720 | 0.693 ± 0.002 | **0.727 ± 0.011** |
| val dddG of doubles (`spearman_doubles_epi`) | -0.013 | 0.341 | – | 0.225 | 0.062 | **0.513** | 0.009 ± 0.068 | 0.492 ± 0.006 |
| test all variants (`spearman_all`) | 0.771 | 0.760 | 0.794 | 0.784 | 0.824 | 0.843 | 0.827 ± 0.002 | **0.849 ± 0.001** |
| test doubles (`spearman_doubles`) | 0.617 | 0.652 | 0.673 | 0.668 | 0.700 | 0.763 | 0.717 ± 0.005 | **0.781 ± 0.008** |
| test dddG of doubles (`spearman_doubles_epi`) | 0.020 | 0.346 | – | 0.184 | 0.042 | 0.554 | -0.022 ± 0.026 | **0.575 ± 0.007** |

- **The combined prediction beats the WT head for both checkpoints on every metric, and almost all of the gain is
  the interaction term.** On test, all-variant ρ rises 0.824 → 0.843 (released) and 0.827 → 0.849 (devel); doubles
  0.700 → 0.763 and 0.717 → 0.781; and the dddG of doubles goes from about nothing (0.042 and −0.022 ± 0.026) to
  0.554 and 0.575. Validation reads the same way (dddG 0.062 → 0.513 released, 0.009 ± 0.068 → 0.492 ± 0.006 devel).
  The WT head carries no usable pair signal by itself, as an additive-style head should not.
- **Devel against released, combined:** test +0.006 on all variants, +0.018 on doubles, +0.021 on dddG; validation
  level on all variants (0.854 both), +0.007 on doubles and −0.021 on dddG (0.492 ± 0.006 against 0.513). The
  released checkpoint has no seed sd, so a difference of about 0.02 on dddG cannot be called either way. The devel
  numbers are **on measured rows only** (§5.3); with bounded rows they would read 0.858 and look like +0.015.
- **Against the retrained baselines** an ESM-MSR combined column is the best value in all six rows. ME stage 2 is the
  best baseline on dddG (0.346 test, 0.341 validation, against the stage-1 singles) and ESM-MSR is ahead of it by 0.15–0.23;
  ThermoMPNN-D reaches 0.184 on test and 0.225 on validation, and ME stage 1 and the WT heads are near zero. With its own
  singles ME stage 2's dddG is 0.462 (test) and 0.438 (validation).

### 2.4 External sets, DMS, Domainome (baselines = 3-seed means, ESM-MSR = released)

- **12 external sets, mean ρ:** ESM-MSR (baseline) **0.646**, ME stage 1 0.615, ME stage 2 0.605, ThermoMPNN
  0.585, ThermoMPNN + D 0.586. ESM-MSR is best on 9 of 12 (ME stage 1 on s2648, s8754, q3421).
- **7 DMS** (`spearman_all`): ESM-MSR (devel) leads DLG4 binding (0.742) and GB1 (0.337); the baseline triplicate reaches 0.737 on DLG4 binding and 0.318 on GB1 and MYO
  (combined 0.639); ME's additive variants lead DLG4 abundance (stage 1+ 0.729 vs 0.726), GRB2 abundance (0.690 vs
  0.672), GRB2 binding (stage 2+ 0.642 vs 0.633) and ESTA (0.483 vs 0.479) — mostly by < 0.02. **Every
  model's additive variant beats its own direct multi-mutant prediction** on these assays.
- **dddG on DMS:** ME stage 2 (corrected, against the stage-1 singles) scores 0.477 and 0.462 on DLG4 abundance and
  binding, 0.349 on GRB2 abundance, 0.116 on GRB2 binding, 0.087 on MYO and 0.177 on GB1; ThermoMPNN-D (corrected, against
  ThermoMPNN) scores 0.203 and 0.192, 0.203, 0.034, 0.071 and 0.445 on the same six. So ME stage 2 and ESM-MSR are comparable on DLG4
  (≈ 0.45–0.47) and ESM-MSR leads GRB2 abundance (0.424 against 0.349 and 0.203), while ThermoMPNN-D leads GB1 (0.445
  against ESM-MSR's 0.316 and ME stage 2's 0.177). ME stage 1 is reliably negative.
- **Domainome** (singles): ESM-MSR (baseline) combined 0.543 ≈ WT head 0.541 > ME stage 1 0.520 > ME stage 2 0.509
  > ThermoMPNN 0.450 (ME stage 2: 0.512 / 0.509 / 0.505 over the three corrected seeds; all 536,145 predictions merge onto
  the Domainome table, none missing).

### 2.5 Mutant-background ablation (nobb)

Dropping the 3 train / 5 val mutant-background libraries (all `native_cond` singles; zero doubles) and
re-scoring on the identical wild-type-backbone test rows:

| model | Δ spearman_all | Δ spearman_doubles | Δ beyond ρ | Δ pair ρ | Δ flipacc |
|---|---|---|---|---|---|
| MutateEverything stage 2 | **−0.017** | **−0.022** | +0.004 | +0.026 | +0.033 |
| MutateEverything stage 1 | +0.001 | −0.000 | +0.006 | +0.006 | −0.004 |
| ThermoMPNN | 0.000 | +0.005 | 0 | 0 | 0 |

Those 3,543 training rows buy ME stage 2 a little ddG ranking on wild-type backbones and nothing on the
interaction metrics. This ablation predates the data correction: its ME stage 2 rows compare two item-filtered stage-2
runs, and it was not repeated on the corrected data. ThermoMPNN's row is unchanged (singles only, identical in both
conditions); ThermoMPNN-D was not part of it.


### 2.6 How much predicted epistasis should a multi-mutant prediction keep?

Write **A** = `wt_lora_pred` (additive by construction) and **M** = `mt_lora_pred` (the direct multi-mutant
head). Then

    (1 − λ)·A + λ·M  ≡  A + λ·(M − A)

so the ensemble weight **λ is exactly the fraction of the MT head's predicted interaction that is kept**. The
shipped `combined_pred = 0.5·(A + M)` is already λ = 0.5 — a 50 % shrinkage of predicted epistasis, not an
unweighted average of two independent models. The proposal `(N−1)/N · WT + 1/N · MT` is the schedule λ = 1/N.

Rescored offline from the saved per-variant predictions (`analysis_scripts/ensemble_weight_scan.py`; no GPU
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

**Leak note (§6).** The ME stage-2 row (`q`) is the item-filtered run, withdrawn with the other stage-2 numbers; the corrected run is queued (§1.4). 23 of the 213 assays are fully inside the capped train split (45,623 rows; all Stability); they are listed in `docs/benchmarks/leak_audit_pg_per_dms.csv` and marked in the per-model `M` fractions of §6.5. The σ sweep below is unaffected in ranking (a lower σ is better on exactly these Stability assays, so removing them strengthens rather than weakens the σ = 0.75 conclusion) but its absolute values are upper bounds for ESM-MSR. SPURS's 51/213 contaminated assays make its 0.483 worse than ESM-MSR's 0.510, not better.


213 DMS that every model scored (BRCA1, BRCA2, POLG_CXB3N and SCN5A excluded: > 1,800 residues, out of
memory for at least one model; ESM-MSR itself could not score them on the cluster's cards and they are queued for
the local GPU). **PG avg** = mean of the five category means (a plain mean over DMS within a
category, not ProteinGym's per-UniProt weighting); **DMS mean** = mean over the 213. Unmasked scoring,
ProteinGym's own AF2 structures. ProSST is the authors' published ProSST-2048 scores, not a rerun. Rosetta is
italic because its predictions cover single mutants only (212 of the 213 DMS) and the rest of the column does not.

| model | Activity (43) | Binding (13) | Expression (18) | Org. fitness (73) | Stability (66) | PG avg | DMS mean |
|---|---|---|---|---|---|---|---|
| ESM-MSR (devel), σ = 0.75 (3 seeds) ᴹ | 0.439 | 0.364 | 0.515 | 0.410 | 0.858 | 0.517 | 0.560 |
| ProSST (zero-shot; published scores) ᶜ | **0.486** | **0.449** | **0.530** | **0.453** | **0.653** | **0.514** | **0.528** |
| ESM-MSR (devel), σ = 0.5 (3 seeds) ᴹ | 0.443 | 0.381 | 0.499 | 0.413 | 0.822 | 0.512 | 0.551 |
| ESM-MSR (released), WT head only, σ = 0.75 ᴹ | 0.447 | 0.362 | 0.512 | 0.424 | 0.815 | 0.512 | 0.554 |
| ESM-MSR (devel), σ = 1 (3 seeds) ᴹ | 0.426 | 0.347 | 0.517 | 0.395 | 0.869 | 0.511 | 0.555 |
| ESM-MSR (released), σ = 1 ᴹ | 0.425 | 0.345 | 0.515 | 0.394 | 0.870 | 0.510 | 0.555 |
| ESM-MSR (released), WT head only, σ = 1 ᴹ | 0.434 | 0.349 | 0.515 | 0.411 | 0.828 | 0.507 | 0.550 |
| ESM-MSR (released), WT head only, σ = 0.5 ᴹ | 0.449 | 0.377 | 0.493 | 0.425 | 0.780 | 0.505 | 0.543 |
| ESM-MSR (devel), σ = 0.25 (3 seeds) ᴹ | 0.430 | 0.385 | 0.468 | 0.400 | 0.742 | 0.485 | 0.517 |
| SPURS (single model) ᴹ | 0.384 | 0.334 | 0.490 | 0.368 | 0.837 | 0.483 | 0.525 |
| ESM-MSR (released), WT head only, σ = 0.25 ᴹ | 0.436 | 0.381 | 0.466 | 0.411 | 0.713 | 0.481 | 0.512 |
| MutateEverything single (3 seeds) ᴹ | 0.371 | 0.336 | 0.472 | 0.352 | 0.808 | 0.468 | 0.506 |
| ESM-MSR, σ = 0 (no LoRA) ᶜ | 0.416 | 0.374 | 0.439 | 0.391 | 0.620 | 0.448 | 0.470 |
| MutateEverything double (3 seeds) ᴹ | q | q | q | q | q | q | q |
| ThermoMPNN (3 seeds) ᴹ | 0.351 | 0.299 | 0.439 | 0.337 | 0.781 | 0.441 | 0.484 |
| SPURS multi (multi-mutant model) ᴹ | 0.326 | 0.267 | 0.410 | 0.320 | 0.808 | 0.426 | 0.477 |
| ProteinMPNN (3 decoding orders) ᶜ | 0.343 | 0.323 | 0.437 | 0.310 | 0.627 | 0.408 | 0.426 |
| Rosetta Cartesian ΔΔG (*singles only*) | _0.311_ | _0.291_ | _0.336_ | _0.296_ | _0.640_ | _0.375_ | _0.409_ |

Paired against the released checkpoint over the 213 common DMS (3 seeds × 4 σ, every DMS scored):

| devel setting | PG avg | paired Δ vs released (mean over DMS) | better on | seed sd (PG avg) |
|---|---|---|---|---|
| σ = 1 | 0.5107 | +0.0003 | 112 / 213 | 0.0003 |
| **σ = 0.75** | **0.5170** | **+0.0053** | **135 / 213** | 0.0007 |
| σ = 0.5 | 0.5115 | −0.0043 | 102 / 213 | 0.0007 |
| σ = 0.25 | 0.4853 | −0.0380 | 71 / 213 | 0.0005 |

- **Devel at σ = 1 matches released** and **σ = 0.75 is the only setting that beats it** (+0.0053, better on 135 of 213,
  paired t = 3.04). Seed sd ≤ 0.0007, so the σ = 0.75 gain is ~8 seed-sds and the σ = 1 null is real. (Before the
  2026-10-23 bias correction these read +0.0047, 133 of 213, t = 2.83; the conclusion did not move.)
- **The σ trend:** Stability falls monotonically as σ drops (0.869 → 0.742) while Binding rises monotonically
  (0.347 → 0.385), with Activity and Organismal fitness peaking near σ = 0.5. Scaling the LoRA down trades
  stability skill for general fitness, and σ = 0.75 is where the two balance.
- **MT-pass ablation** (`--mt_pass_adapter wt`, devel seed 1, σ = 1): the WT adapter scoring the mutant sequence
  costs −0.017 ρ on the 26 DMS done so far (worse on 20, better on 6; −0.0165 on the first 24, against −0.016 before the
  2026-10-23 bias correction). Verified by hand on BCHB: unmasked, the WT adapter sees the
  mutant residue it is scoring and its LLR collapses toward zero (mean −0.85 vs −3.44 on the WT sequence).
  This is still a partial result, but it is being completed: the 26 stored files were corrected to the per-mutation bias
  (both heads of this mode shift by the WT bias, 0.0987; checked against a real rerun of 3 of them, Spearman within 8e-4) and
  the remaining 187 DMS are running under the new defaults (job 386637, resumed from those files; 177 of the 213 were
  in at 2026-10-24 16:10).
- **ProSST wins four of five categories outright** (Activity, Binding, Expression, Organismal fitness) against every ESM-MSR setting, not just σ = 1; the 0.22 it concedes on Stability is what brings the averages level (§1.4). **Under the audit the Stability comparison is the contaminated one:** all 23 fully trained DMS are Stability assays, so the category where ESM-MSR wins by 0.22 is the only one where it has direct training data. Removing them puts ESM-MSR's Stability at 0.871 and ProSST's 0.653 unchanged; the four-category lead is untouched and the global tie is restored without ESM-MSR's training advantage.
- **ProteinMPNN test-time ensembling:** averaging three decoding orders +0.0003; a 0.2 Å backbone-noise
  ensemble of 8 +0.0017 on a 73-DMS pilot (50/73 up, t = 1.29). Not worth a full run.


## 4. Third-party predictor notes

- **The external-set leaders trained on that kind of data.** S2648, S8754, Q3421, K3822 and Ssym overlap the
  training data of DDGemb, DDMut and GeoDDG, which is why they lead §1.1 and fall behind on Tsuboyama and the DMS.
- **IFUM has two modes.** §1 reports the WT-backbone mode. Its native mode folds every variant with ESMFold and
  scores it on its own structure, which is slightly better on the hyperopt Tsuboyama split (val 0.778 vs 0.746,
  test 0.738 vs 0.697). Folding is the cost, so the first Tsuboyama run reused structures already on disk; for the
  capped split 64,008 new sequences (mean length 58, about 6 GPU-hours) are being folded before scoring.
- **Rosetta** comes from a separate session on Narval: FastRelax in cartesian mode (`beta_nov16_cart`), mutant
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
   numbers. The corrected retrained baselines carry no item rules, so their val table is the full 34,957 rows, and the
   validation numbers of ME stage 1 and 2, ThermoMPNN and ThermoMPNN-D are all on those rows (ME stage 1 and ThermoMPNN
   were rescored on them on 2026-10-28; their earlier values, 0.751 and 0.785, were on the 33,333-row subset).
5. **Test-set ddG tables exclude destabilized backbones** (`--test_backbones wt_only`, 40,866 rows): 7 of 37 test
   libraries carry a background mutation that only the models trained on that pattern have seen. Every such row is
   a single, so the epistasis columns are identical either way (re-verified 2026-10-22 against the current
   `dG_wt`: max abs difference 0 over every `epi_*` column). A file scored before the `dG_wt` fix will differ from a
   current one by up to 0.05 on pair ρ, which is that fix and not the backbone filter.
6. **Ground truth is always ESM-MSR's `ddG_ML`.** Upstream MutateEverything silently flips the sign of its own
   ground truth when more than half of `ddg` is negative, which would bite Ssym and GB1.
7. **External sets are aligned by subset relation:** S461 ⊂ S669, K2369 ⊂ K3822, PTMUL-D ⊂ PTMUL, so a model only
   has to predict the superset. The link is exact for *predictions* (same code, mutation, reference sequence and PDB
   on every child row) but not for *labels*: S461 revised 40 of its 461 labels (12 sign flips) and PTMUL-D differs
   from PTMUL on 41 of 536, so a child is always scored against its own labels.
8. **The link is applied in training only.** For link-trained checkpoints (devel), `esm_msr_testing.py` writes
   *latent* ddG; `epi_baselines.msr_link` rebuilds the link from the checkpoint for the observed-scale epistasis
   metrics. Rank metrics within a dataset are unaffected.
9. **Check the label column, and link by variant, before trusting a statistic from someone else's prediction
   file.** The ESM3 Forge files carry no label for 343 of PTMUL's 914 rows (all triple-or-higher mutants), so their
   own statistics silently score the 571 doubles; the numbers here score all 914 against the harness labels. Their
   `id` column cannot be used to join: it matches the source's PDB-numbered `uid` on only 283 rows and the
   sequence-numbered key on 343, so rows are linked by mutation string plus PDB code and verified against the
   source's wild-type and mutated sequences (which carry both numberings).
10. **SPURS is two models** and the first harness run mixed them (§1.1); the single-model and multi-model rows are
    both reported, from separate runs.
11. **Every model is scored on the mutant-background libraries.** SPURS and ProteinMPNN needed their parsed sequence
    patched to do so (§1.2); the patch leaves wild-type-background groups untouched. The epistasis tables are
    unaffected either way, because every mutant-background row is a single.
12. **ESM-MSR calibration defaults changed on 2026-10-23: `tie_mt_scale` off, `per_mutation_bias` on, everywhere.** The
    released and dc8 checkpoints record neither key (they were trained with an independent MT scale and one bias per
    item), so inference now scores them with the bias applied once per mutated position: a k-mutant's calibrated
    prediction moves by (k − 1) × bias, singles do not move. Estimated offline from the stored capped-test predictions
    (shift = each head's calibration bias; WT +0.099 to +0.130, MT −0.035 to −0.140, combined +0.048 for released and
    within ±0.011 for the dc8 seeds), per-protein Spearman changes by at most 0.0008 (all variants) and 0.0000
    (doubles only), because every double in a protein shifts by the same constant. What does move is anything that
    reads levels (global error, bias, RMSE, the dddG of a double against its singles) and any set that mixes mutation
    counts, most likely ProteinGym (its variants carry many substitutions; not yet measured). Every ESM-MSR number on a set with multi-mutants (Tsuboyama doubles, PTMUL, the DMS
    assays, ProteinGym) computed before this date used one bias per item; singles-only sets
    (Domainome, most external sets) are unaffected. **ProteinGym was corrected offline, not rerun** (a rerun is ~250 GPU-hours): the 18
    stored run directories were rewritten with `analysis_scripts/pgym_per_mutation_bias.py` (31.9 M rows shifted, originals kept in
    `outputs/pgym_pre_per_mutation_bias/`, a manifest in each run directory). The correction was checked against a real rerun of
    three DMS (GCN4, with up to 44 substitutions, and two Tsuboyama sets) at σ = 1 and 0.5 in full and WT-only mode: the MT head's
    shift agrees to within 0.001, the Spearman of the corrected values to within 5e-4 of the rerun's. The old code's WT pass for
    multi-mutants differed slightly between a full and a WT-only run (up to 0.015 on a double, 0.04 on a deep multi-mutant, bias aside);
    the correction keeps that offset. Effect on the ProteinGym table: released combined PG avg 0.510 unchanged, DMS mean 0.556 → 0.555, WT
    head 0.508 → 0.507, devel unchanged at the printed precision; per-DMS changes reach 0.025–0.036 for the released checkpoint and
    0.005 for a devel seed. Tsuboyama, PTMUL and the DMS assays are being rerun for real; until a seed mean is complete under one
    convention it is shown as `q` (`benchmark_matrix.py` classifies each file by the mean of its `wt_lora_dddg_pred`).
13. **ME and ThermoMPNN-D were retrained on valid labels only (2026-10-26/27), and two conventions matter.** (a) The first
    October stage-2 and ThermoMPNN-D runs applied ESM-MSR's item filters and lost 14,388 of the 53,188 measured doubles;
    they are withdrawn, and the corrected runs (same recipes, every measured double, no item filter) are the ones in every
    table. ME stage 1 and the single ThermoMPNN were unaffected (their singles are identical in both tables) and were not
    retrained. (b) The pre-October ME (the hyperopt columns of §1.5) was trained on `ddg = +ddG_ML`, positive =
    stabilizing, the opposite of the upstream convention; stage 2's loss subsamples and weights by that sign (§1.1), so
    those models were not the upstream recipe. The retrained models use the upstream sign. (c) A baseline's predicted
    epistasis is its direct double minus an additive expectation from a model that never trained on doubles: ME stage 2
    against the stage-1 singles of the same seed, ThermoMPNN-D against ThermoMPNN (§2). The epistasis suite must be given
    that model as head `epi`; under the older head name `mt` it silently treats the raw double prediction as if it were
    already an epistasis term.

## 6. Data-leakage audit (October 2026)

**Scope.** Every model row in §1–§3, every dataset column, and the capped Oct06 split. The legacy hyperopt split
(§1.2 hyperopt columns, §1.5) is out of scope: it predates the audit and is not the split the headline results use.
The audit answers two questions per cell — *was the exact variant trained?* — and, if not, *was the same protein, a
homolog, or the source dataset trained?* — and records the answer as a code. It also validates the homology tooling
the split was built with, and measures the overlap the split was never designed to catch (ProteinGym, Domainome).

### 6.1 Method

1. **Normalise every test row** to a protein sequence and a set of 1-based `(WT, position, mutant)` substitutions
   (2,446,730 ProteinGym rows, 534,385 Domainome, 149,360 GB1, 87,544 Tsuboyama capped val+test, 25,632 external-set
   rows, 7 DMS assays). Every protein is aligned to its same-protein family representative (≥95 % identity, ≥90 %
   coverage, single linkage) so mutations can be compared across PDB entries, chain numbering and construct
   differences.
2. **Reconstruct each model's training inventory** from the paper/code/split where public, and from the local
   clone where the training data ships with the model (Table 6.3). Inventories are recorded at the mutation level
   where the data exist (ESM-MSR, ME, ThermoMPNN, SPURS, DDGemb, DDMut, GeoDDG, ESM3ΔG) and at the protein level
   where only domains are public (EsmTherm, IFUM).
3. **Classify each test row** as exact (`M`), constituent (`m`: all its mutations were trained in the same protein but
   not this combination), same protein (`P`), strong homolog (`H`), weak/ambiguous similarity (`h`), corpus-only (`C`),
   documented-only (`D`), or none. The classification is per row, then aggregated per dataset.
4. **Validate the split's homology tools** on the pair table the split was actually built from (408,330 pairs) and
   re-run MMseqs2/Foldseek on the same structures (§6.4).

### 6.2 Notation

| code | meaning | evidence |
|---|---|---|
| `M` | identical variant row in training (same protein family, identical mutation set) | verified |
| `m` | every mutation in the row appears in training in the same protein, but not this combination (e.g. a test double whose two singles were trained) | verified |
| `P` | same protein (≥95 % identity over ≥90 % coverage, or ≥90 % identity over ≥50 %) in training, no mutation match | verified |
| `H` | strong homolog in training: MMseqs2 E ≤ 1e-3 over ≥50 % of the test protein | verified |
| `h` | weak/ambiguous: E ≤ 1e-2 or ≥30 % identity over ≥30 % of the test protein | measured |
| `D` | documented training on the same source dataset; rows not reconstructable | documented |
| `C` | pretraining-corpus exposure only (WT sequence/structure, no mutation supervision) | documented |
| `U` | training data not disclosed | documented |
| `V` | the dataset was used for validation or model selection | documented |
| `#` | the number is not a test result (the dataset is the training set) | verified |
| `?` | probable but unverified (e.g. `M?`: same protein and same source dataset, row-level file not public) | inferred |
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
| ThermoMPNN-D | the 53,188 measured doubles plus 118,963 augmented doubles (upstream `double-aug`, built from the singles that have a Rosetta model: 79 % of them) | capped split; `train_epistatic.yaml` | verified |
| SPURS single | 212 train + 29 val Megascale libraries after a >25 %-identity blacklist; 216,336 singles | `SPURS_mod` data, ThermoMPNN 8:1:1 split, `mmseq_mut_search_0.25.m8` | verified |
| SPURS multi | same libraries, 132,950 doubles | `SPURS_mod` | verified |
| EsmTherm | `K50_dG_Dataset1_Dataset2` domains; 414 wildtypes contribute train mutants under a per-row 80/10/10 split; only 47 domains are test-only | `EsmTherm/datasets/split/wildtype_split.csv` | protein-level (mutation rows unavailable) |
| IFUM | Megascale Dataset2: 322 train libraries, 86 in an unseparated val/test block | IFUM published `dataset_splits.pdf` | protein-level |
| DDGemb | S2450 = S2648 − 18 sequences/198 mutations >25 % identity to S669; 2,450 mutations / 115 UniProt | `ddgemb.biocomp.unibo.it` S2450.tsv/.fasta | verified |
| DDMut | S9028 = S2648 + FireProtDB, 4,514 forward + 4,514 reverse; multi-mutant branch SM1242 | `biosig.lab.uq.edu.au` S9028.csv, SM1242.csv | verified |
| GeoDDG-3D | S8754 = 8,754 single mutations / 301 proteins (ProThermDB + ThermoMutDB) | `GeoStab` S8754.csv | verified |
| ESM3ΔG | 960,216-sequence **MGnify Stability** fine-tune (525,127 WT + 435,089 point mutants); Megascale is its held-out test | Zenodo 19411306 `mgnify_training_index.csv` | verified (0 exact sequences) |
| ProSST | 18.8 M AlphaFoldDB structures (masked-LM); no ΔΔG fine-tune | ProSST paper | documented only |
| ProteinMPNN | PDB assemblies deposited before 2021-08-02; no mutation-level supervision | ProteinMPNN paper | documented only |
| ESM3 (small/medium/large, and the σ = 0 control) | UniRef/MGnify/JGI + PDB pre-2020-05-01 + AFDB v4 | ESM3 paper | documented only |
| Rosetta Cartesian ΔΔG | no training data | physics-based | n/a |

Two corrections to the old legend follow from this table. **ESM3ΔG is not Megascale-trained** (its released
`*_augmented_*` weights are fine-tuned on the authors' MGnify Stability set; 0 of the 33 capped test libraries are in
it by sequence), and **S8754 is GeoDDG's training set**, not a held-out external set.

**How the retrained baselines were built (2026-10-27).** *Tables:* the standard ESM-MSR preprocessor with `censor_range`
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
MMseqs2 E ≤ 1e-3, Foldseek E ≤ 1e-3, identity > 0.40, or TM-score > 0.70 holds. The saved pair table gives 25,027
edges, and `make_homology_split` re-checks every edge.

- **The rule is enforced.** No edge joins two different splits or a split and an external protein: 0 of 25,027
  default-rule edges cross a real boundary. The 366 edges that do cross *involve excluded libraries*, which is the
  quarantine working. The excluded set is doing real work: all 142 excluded libraries have a direct pair to an
  external benchmark protein, 90 of those pairs at identity > 0.9 and 117 at TM > 0.9 (near-identical structures).
- **Foldseek adds value.** 550 pairs are structure-only (no MMseqs hit, identity ≤ 0.40); none of them crosses a
  kept boundary, i.e. Foldseek drove exclusions rather than introducing leaks.
- **The 40 % identity cap leaves a grey zone.** If the cap were 30 %, 218 train/val/test↔external pairs would newly
  cross a boundary; at 25 %, 610; at TM ≥ 0.5, 177; at E ≤ 1e-2, only 4. So the E-value criterion is doing most of
  the work and the identity cap is the weaker half of the rule. The capped split contains 8,114 capped-test rows
  (15.6 %) and 7,210 capped-val rows (20.3 %) whose protein has a 30–40 % identity match to a train library that the
  split did not exclude; this audit terms those `h` (ambiguous), not `H`.
- **Exclusion is not transitive.** 134 train libraries sit in a connected component that contains an external
  protein through an excluded (bridge) library. All 134 have a direct pair to an external protein just below the
  thresholds (max identity 0.395, max TM 0.6998), so no strong pair was missed, but a stricter cap would have
  removed them.
- **The split only ever guarded the nine classical external sets and the five functional proteins.** ProteinGym and
  Domainome were deliberately not in the quarantine (Domainome would have bridged unrelated clusters). That is the
  largest gap: 23 of 213 ProteinGym assays and 1 of 522 Domainome domains are *inside* the capped train split at the
  mutation level, and dozens more have homologs.

### 6.5 What the audit finds

| finding | size | affected rows |
|---|---|---|
| 23 ProteinGym assays are fully inside the capped train split | 23/213 DMS, all Stability | 45,623/2,446,730 rows (100 % of those 23 DMS) |
| ME stage 1/2 and ThermoMPNN on those assays | singles (24,252 exact) + constituent (21,331) for ME; 24,252 exact for TMPNN | same 23 DMS |
| Domainome domain `O95155_PF04564_1229` (UBE4B) is Megascale library 3L1X | 1 of 522 domains | 1,134 exact + 3,177 same-protein rows (0.8 % of Domainome) |
| SPURS single trained on 24/33 capped test libraries | 58.4 % of test singles | 21,315 exact + 6,747 constituent of 52,071 rows |
| SPURS multi trained on 9/13 double-bearing test libraries | 41.2 % of test doubles | 6,400 exact + 1,717 constituent |
| EsmTherm's 80/10/10 is per row, not per domain: 27/33 test libraries contribute training mutants | 82.2 % of test rows same-protein | `P`, `M?` expected ≈ 0.8 |
| IFUM trained on 21/33 test libraries | 45.2 % of test rows same-protein | `P`, `M?` = same rows |
| ESM3ΔG is zero-shot on Megascale | 0/33 exact | 40.7 % strong + 24.6 % weak homologs |
| DDGemb = S2450 | 92.5 % of S2648; 30 % of S8754; 28 % of Ssym; 45 % of Q3421; 26 % of K3822; 24 % of K2369 | upper bounds via S2648 |
| DDMut = S9028 + SM1242 | 88.8 % of S2648; 60.3 % of S8754; 42.1 % of S783; 19.6 % of S571; 29.7 % of Ssym; 58.9 % of PTMUL; 36.2 % of PTMUL-D | exact |
| GeoDDG = S8754 | S8754 is its training set (`#`); 97.7 % of S2648; 88.3 % of S783; 38.5 % of S571; 68.7 % of Q3421; 69.3 % of K3822; 76.1 % of K2369 | exact |
| the "six clean sets" claim | only S669 and S461 have zero identical mutations for every model | DDMut shares 11 %/16 % of them at protein level, GeoDDG 1 %/3 % |

**Clean comparators on Tsuboyama.** ESM-MSR (released and devel), MutateEverything st1/st2 and ThermoMPNN(-D) have
zero exact mutations and zero same-protein overlap with all 33 capped-test and all 23 capped-val libraries; their only
caveat is the 15.6 %/20.3 % weak-homology rows and validation selection (`V` on the val column). That is the group the
headline Tsuboyama numbers should be read within. At the other end, SPURS single, SPURS multi, EsmTherm and IFUM are
not held out on this split, and ESM3ΔG is held out but homology-caveated.

**The retrained baselines inherit the split, not extra leakage.** MutateEverything stage 1/2 and ThermoMPNN were
retrained on the same 172 libraries as ESM-MSR, so their `P`/`H`/`h` columns are identical by construction and only the
mutation classes differ (stage 1 = singles, stage 2 = singles+doubles, ThermoMPNN-D = doubles plus augmented doubles). The single
weak-homology row in S669/S461 (`1IOJ` vs `HHH_rd2_0155`, 37 % identity over 61 % of the test protein, E = 5.5) is
therefore present for ESM-MSR, ME stage 1/2 and ThermoMPNN, and absent only for ThermoMPNN-D, which never sees that library (it has no measured doubles and no Rosetta model for
augmentation). It is below the 0.5 % display threshold in the numeric tables and visible in the
companion matrix. Retraining introduced no overlap: the upstream models train on the whole Megascale pool (upstream
ThermoMPNN: 19/33 test libraries, 15,918/36,526 singles), which is strictly more than the capped train split.

**Ssym is not a Megascale set.** None of Ssym's 357 PDB chains or 14 protein families is a Megascale library, so the
Megascale-trained models (SPURS, EsmTherm, IFUM, ESM3ΔG) have no same-protein overlap with it. The best sequence match
between an Ssym protein and a SPURS train library is 70 % identity over 23 % of a lysozyme (E = 3); a Foldseek search of
the 357 Ssym structures against the 269 SPURS train structures gives 0 pairs at E ≤ 1e-3 or TM > 0.7, and the 303 pairs
at TM 0.50–0.53 are fold-level matches to small designs (lddt 0.35–0.43). SPURS's 0.784 on Ssym is clean and stands
ahead of ESM-MSR's 0.773.

**ProteinGym.** The 23 fully trained assays are `ARGR_ECOLI_Tsuboyama_2023_1AOY`, `CBX4_HUMAN_Tsuboyama_2023_2K28`,
`CSN4_MOUSE_Tsuboyama_2023_1UFM`, `CUE1_YEAST_Tsuboyama_2023_2MYX`, `DNJA1_HUMAN_Tsuboyama_2023_2LO1`,
`FKBP3_HUMAN_Tsuboyama_2023_2KFV`, `HCP_LAMBD_Tsuboyama_2023_2L6Q`, `ILF3_HUMAN_Tsuboyama_2023_2L33`,
`ISDH_STAAW_Tsuboyama_2023_2LHR`, `NKX31_HUMAN_Tsuboyama_2023_2L9R`, `NUSG_MYCTU_Tsuboyama_2023_2MI6`,
`PITX2_HUMAN_Tsuboyama_2023_2L7M`, `PSAE_PICP2_Tsuboyama_2023_1PSE`, `RCD1_ARATH_Tsuboyama_2023_5OAO`,
`RFAH_ECOLI_Tsuboyama_2023_2LCL`, `RS15_GEOSE_Tsuboyama_2023_1A32`, `SBI_STAAM_Tsuboyama_2023_2JVG`,
`SDA_BACSU_Tsuboyama_2023_1PV0`, `SR43C_ARATH_Tsuboyama_2023_2N88`, `UBE4B_HUMAN_Tsuboyama_2023_3L1X`,
`YAIA_ECOLI_Tsuboyama_2023_2KVT`, `POLG_PESV_Tsuboyama_2023_2MXD`, `YNZC_BACSU_Tsuboyama_2023_2JVD`.
All are Stability assays. Removing them from the local devel-seed reruns (190 DMS): DMS mean 0.561 → 0.520,
Stability 0.880 → 0.871, PG average 0.516 → 0.514. ProSST, zero-shot on all of them, is 0.528 on the clean DMS mean
and 0.514 on the clean PG average. SPURS is more contaminated than ESM-MSR here (51/213 assays with ≥50 % of rows
trained), so its 0.483 PG average is not a clean comparison either.

**Clean external-set means.** Mean over the 12 external sets of §1.1, per model, over the cells with no `M`, `m`, `P`, `H`, `D`, `U` or `#`
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
| SPURS single | 0.647 (3) | 0.610 | s571, s783, s2648, s8754, q3421, k3822, k2369, ptmul, ptmuld |
| SPURS multi | 0.758 (1) | 0.574 | s571, s783, s2648, s8754, s669, s461, q3421, k3822, k2369, ptmul, ptmuld |
| EsmTherm | 0.380 (1) | 0.365 | s571, s783, s2648, s8754, s669, s461, q3421, k3822, k2369, ptmul, ptmuld |
| DDGemb | 0.612 (2) | 0.697 | s571, s783, s2648, s8754, ssym, q3421, k3822, k2369, ptmul, ptmuld |
| GeoDDG-3D | 0.664 (1) | 0.767 | s571, s783, s2648, s8754, s669, ssym, q3421, k3822, k2369, ptmul, ptmuld |
| ESM3ΔG | 0.641 (5) | 0.582 | s571, s2648, s8754, k3822, k2369, ptmul, ptmuld |
| ProSST | 0.423 (12) | 0.423 | none |
| ProteinMPNN | 0.464 (12) | 0.464 | none |
| ESM3-small-open, σ = 0 | 0.378 (12) | 0.378 | none |

Only ESM-MSR (released and devel), ME stage 1/2, ThermoMPNN(+D), ProSST, ProteinMPNN and the σ = 0 control are clean on all twelve; ESM-MSR
(devel 0.651, released 0.647) leads that group, ahead of ME stage 1 0.615, ME stage 2 0.605, ThermoMPNN 0.585,
ProteinMPNN 0.464 and ProSST 0.423. The other means are over fewer sets and over different ones, so they are not comparable with these or
with each other: DDGemb's clean cells are S669 and S461, where it is level with ESM-MSR on S669.

**External sets.** With identical mutations, same protein and strong homologs excluded, the surviving comparisons are
narrow: on S669 and S461 the clean supervised models are ESM-MSR (0.565 / 0.756, released; 0.560 / 0.753 devel),
ME stage 1 (0.559 / 0.721), ThermoMPNN (0.490 / 0.655) and DDGemb (0.570 / 0.654). ESM-MSR and DDGemb are level on
S669; ESM-MSR leads S461 by 0.10. The former 12-set leaders, GeoDDG and DDGemb, are not clean on the sets that made
them leaders.

### 6.6 What changes in the tables

- Numbers are unchanged; the superscripts and companion matrices are new. A cell is eligible for bold when it has no more
  than minor overlap with training: no `M`, `m`, `P`, `H`, `D`, `U` or `#` class at 0.5 % of rows or more (the same
  threshold that decides whether a superscript is printed). `h` (weak similarity), `C` (pretraining only) and `V` (the
  split's own validation role, which every in-house model shares) do not disqualify, because a 30 %-identity local match,
  generic pretraining exposure or having been used for model selection is not the same kind of evidence as a trained
  mutation. Tables that compare settings of one model (the σ sweeps) bold within it; summary-mean columns, the
  singles-only Rosetta rows and `q` cells are never bolded. This is why several bold values moved: §1.1 s571 ESM-MSR
  0.456 (GeoDDG 0.526 is `M`), s783 ESM-MSR (devel) 0.725 (GeoDDG `M`), s2648 ME stage 1 0.701 (GeoDDG `M`), s8754
  ESM-MSR (devel) 0.625 (GeoDDG `#`, DDGemb/DDMut `M`), S669 DDGemb 0.570 (GeoDDG/DDMut `P`), S461 ESM-MSR 0.756; §1.3
  Domainome ProteinMPNN 0.548 (ESM-MSR `P`, `H.08`); §1.4 ProSST 0.514/0.653/0.528 (all trained rows `M`); §1.2 the clean
  held-out group is unchanged. On 2026-10-27 every superscript was re-derived from its matrix with the 0.5 % threshold
  (SPURS's tiny `M`/`m` fractions no longer print where a larger `P` is present, and the Domainome cells of ME and
  ThermoMPNN print nothing), the DMS columns of §1.3 were bolded, the stray bold on the released GB1 σ = 1 cell was removed,
  and the §3 table now bolds only ProSST, because every ESM-MSR, ME and ThermoMPNN row carries `M`.
- **The "six sets with no documented overlap for any model" claim is withdrawn.** The overlap lattice is measured,
  and it is model-specific: DDMut and GeoDDG are on S571, Ssym and PTMUL; DDGemb is on S571, Ssym, Q3421, K3822 and
  K2369 at the mutation level; EsmTherm/IFUM/SPURS are same-protein on most of the classical sets through the
  Megascale natural libraries. The corrected statement is in the §1.1 bullet.
- **ESM3ΔG's ‡ flag is removed.** It is zero-shot on Megascale; its caveat is homology (8/33 test libraries, 40.7 %
  of test rows), not trained mutations.
- **S8754's GeoDDG cell carries `#`.** It is the training set; the 0.841 is a training-set score.
- **ProteinGym's 0.510/0.514 tie is a trained-vs-zero-shot comparison on 23 Stability assays.** The clean DMS mean
  puts ProSST ahead; the PG average is level.

### 6.7 Reproducibility, assumptions and limits

**Artifacts.** `docs/benchmarks/leak_audit.sqlite` (tables `inventories`, `cells`, `pg_per_dms`, `homology`,
`code_exact`), `leak_audit_matrix.csv` (every model×dataset cell with counts and code), `leak_audit_inventories.csv`,
`leak_audit_pg_per_dms.csv`, `leak_audit_homology.csv`, `leak_audit_code_exact.csv`, `split_homology_audit.json`.
Scripts: `analysis_scripts/leak_audit/` (01 megascale → 18 database; run in order, `MMSEQS`/`FOLDSEEK` paths at the
top of the scripts). Inputs are the repo's `data/`, the public files listed in Table 6.3, and the saved pair table
from `preprocessing/split_tsuboyama.py`.

**Assumptions and limits.**
1. A model is treated as having trained on *all* rows of a library it trained on (ESM-MSR, ME, ThermoMPNN, SPURS,
   ESM3ΔG). The baselines' private filters (censored rows, ME's stage-1/2 split, ThermoMPNN's own early stopping) can
   only remove rows; the audit is therefore an upper bound on exact overlap, which is the conservative direction.
2. EsmTherm and IFUM publish domains, not rows. `M?` is the expected exact fraction under their per-domain or
   per-row split (IFUM 1.0, EsmTherm 0.8); it is an inference, not a measurement.
3. The ESM3ΔG inventory uses its published 960,216-sequence MGnify index. The `*_augmented_*` checkpoints we scored
   were trained on "a more limited subset"; the audit's homology fractions are therefore an upper bound.
4. `P` is judged at ≥95 % identity/≥90 % coverage, with ≥90 %/≥50 % as a secondary rule for literature sets;
   near-identical constructs (missing purification tags, one-residue offsets) count as the same protein.
5. Homology is MMseqs2 over sequences. Foldseek was validated on the split's own pair table (no new cross-set
   leaks) but was not run for the 960,216-sequence MGnify corpus or the literature sets; a structure-based homolog
   could therefore be missed.
6. `h` is deliberately permissive (≥30 % identity or E ≤ 1e-2 over ≥30 % of the test protein) and should be read as
   "not clean, evidence weak", not as a demonstrated leak.
7. The S2450/DGGemb inherited overlaps are upper bounds via S2648 (S2450 = S2648 − 198 mutations; the removed rows
   cannot all be located in the mapped copies).
8. The S8754 file and the benchmark's mapped S8754 agree on 8,745/8,754 names; nine rows differ by normalisation and
   are treated as exact.
9. No attempt is made to estimate how much of a score a leak is *worth*. A 100 %-trained DMS can still score below a
   zero-shot model (the learned map differs), and a same-protein overlap need not help at all. The codes say what was
   trained, not what it bought.


ts `wt_lora_dddg_pred`).

## 7. Where the code and results live

In the repo, on `devel-revision`: `inference_scripts/pgym_runner.py` (the one ProteinGym loop every model shares,
recording per-DMS time, peak VRAM, GPU and host), the ProteinGym modes of `esm_msr_testing.py`,
`proteinmpnn_testing.py`, `spurs_testing.py` and `thermompnnd_testing.py`, and in `analysis_scripts/`:
`benchmark_matrix.py` (the all-by-all tables and the epistasis tables; it still reads the ME stage-2 and ThermoMPNN-D numbers from the withdrawn item-filtered exports `outputs/baselines_capped/{combined,final}_*_table.csv`, so regenerating a table brings those numbers back until it is repointed to the corrected result trees listed below), `ensemble_weight_scan.py` (§2.6),
`pgym_public_scores.py` (ProSST and ProteinMPNN from ProteinGym's published scores), `pgym_rosetta_singles.py` and
`epi_rosetta.py` (Rosetta on ProteinGym and on the epistasis suite), `forge_ptmul_link_check.py` (the PTMUL
indexing check above).

On the cluster, deliberately not in the repo (baseline retraining, MutateEverything, Slurm and cluster-specific
scripts):

| path | what |
|---|---|
| `~/esm-msr-devel/outputs/baselines_capped/` | the retrained-baseline and devel-checkpoint result tables (§2), and `epi_rosetta_*.csv` |
| `~/esm-msr-devel/outputs/baselines_corrected/`, `outputs/baselines_capped_allbl/` | the corrected ThermoMPNN-D checkpoints and evaluations, and the corrected (unfiltered) item tables |
| `~/me_audit/` | the corrected-data builder (`prepare_allbl.py`), ME stage-2 and evaluation launchers, scorers, the epistasis rescoring with the `epi` head fix (`epi/`), and the corrected score tables (`out/`) |
| `~/software/MutateEverything-oct06/` | the upstream ME clone (034dd91) with the oct06 data, `logs/` (checkpoints) and `logs_eval/` (predictions) |
| `~/esm-msr-devel/outputs/pgym/` | every ProteinGym run, plus `pgym_all_models_per_dms.csv` and `pgym_dc8_per_dms.csv` |
| `~/esm-msr-devel/analysis_notebooks/stats/` | ESM-MSR's own per-dataset statistics |
| `~/esm-msr-devel/baselines/` | baseline data prep (`prepare_oct06_data.py`), training wrappers, and the epistasis suite applied to other models (`epi_baselines.py`, `epi_dms.py`) |
| `~/stability-baselines/` | the third-party harness, adapters and results (not a git repository) |
| `~/project_oct06_baselines/` | a symlink index over the retrained-baseline work, with its own README |
| `~/stability-baselines/results/rosetta/` | the Rosetta capped-split predictions in harness layout, with their provenance; the raw Narval outputs are in `~/playground/narval-recovery/predictions/` on the local machine |
| `~/stability-baselines/pgym_public/` | ProteinGym's published zero-shot scores (1.9 GB zip) |

`baselines/` AST-loads functions out of `esm_msr_testing.py` and resolves `epi_items` / `_by_head` from
`devel-experimental`, so renaming those in the repo breaks it.
