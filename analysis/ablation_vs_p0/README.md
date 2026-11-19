# Ablations against FINAL_p0: paired comparisons on every benchmark

Each FINAL_* family differs from FINAL_p0 (the ESM-MSR recipe) in one hparam group (`msr_analysis/final_runs.py::FAMILIES`). This folder compares every ablation with p0 under the **same readout** (WT head; combined prediction 1/2 (MT + WT)) on **the same variants** of every benchmark set, as paired differences delta rho = rho(family) - rho(p0) with bootstrap intervals, and asks in particular whether anything beats p0 on the thermostability benchmarks. (The deployed-readout ablation table of `../98_final_ablation_table.py` uses seed-1 intervals on a few headline columns; this is the set-by-set version with all seeds and both heads.)

Ablations: **masking** p1 (independent masking of both passes), p1 + MT marginal; **loss** pw0 (no pair loss, `block_pair_weight` 0.38 -> 0), no rank loss, rank censor threshold 0; **architecture** no sequence head, MT LoRA rank 2; **data** full MegaScale (trained on `mega_splits`, not homology-restricted; not scored on the Homology-Restricted MegaScale sets). Cross-masking evaluations (`FINAL_p0@independent`, `FINAL_p1@unmasked`) are not ablations and are not included.

Sets: the **thermostability benchmarks** = the 10 external single-mutant sets (S571, S783, S2648, S8754, S669, S461, Ssym, Q3421, K3822, K2369) and PTMUL / PTMUL-D; Homology-Restricted MegaScale val and test (all variants, single mutants, doubles, and the predicted epistasis against dddG); Domainome; the DMS benchmark (DLG4 binding, GRB2 binding, MYO); the 194 ProteinGym assays by category (the 23 stability assays whose variants are in the training data are left out).

```bash
python analysis/ablation_vs_p0/01_paired_vs_p0.py        # results/paired_vs_p0.csv: every set / variant class x family x head: rho, delta, 95% interval, p (B = 1000 by default; the files in results/ were made at 100)
python analysis/ablation_vs_p0/02_pg_vs_p0.py            # results/paired_vs_p0_pg.csv: ProteinGym by category (WT head for every family, combined for p1; 20000 resamples over assays)
python analysis/ablation_vs_p0/03_fig_heatmap.py         # ablation_heatmap
python analysis/ablation_vs_p0/04_fig_thermostability.py # thermostability_vs_p0
python analysis/ablation_vs_p0/05_fig_forest.py          # ablation_forest_wt, ablation_forest_comb
python analysis/ablation_vs_p0/06_tables.py              # table_composites.md, table_counts.md, table_significant.md
```

| figure | content |
|---|---|
| `ablation_heatmap` | rows = the eight ablations, columns = the sets in six groups, cell = delta rho x 1000, outlined when the 95% interval excludes 0; top: WT head, bottom: combined prediction |
| `thermostability_vs_p0` | the key question: per ablation, the composite of the 10 external single-mutant sets, of PTMUL + PTMUL-D and (for reference) Homology-Restricted MegaScale test, with the intervals, p and the individual sets as grey dots; WT head and combined prediction |
| `ablation_forest_wt`, `ablation_forest_comb` | every row of the heatmap with its interval, one panel per ablation |

## Does anything beat p0 on the thermostability benchmarks? No.

* Of the 192 set x ablation x readout cells of the 12 thermostability sets, **one** has an interval above 0: S8754, no rank loss, WT head, +0.012 [+0.001, +0.027] (p = 0.04). With 192 cells about five would be expected above 0 by chance alone at the one-sided 2.5% level. **None of the 32 composite cells** (the composite of the 10 single-mutant sets and of PTMUL / PTMUL-D, 8 ablations, 2 readouts) is above 0. **Twenty cells are below 0**, 16 of them for the two masked families (p1, p1 + MT marginal).
* Composite of the 10 external single-mutant sets (WT head / combined): pw0 +0.001 / +0.000, no rank loss +0.003 / -0.002, rank censor threshold 0 -0.001 / -0.001, no sequence head +0.001 / -0.002, MT LoRA rank 2 -0.001 / -0.002, full MegaScale -0.006 / +0.000: all indistinguishable from p0. The masked families are worse: p1 -0.007 / -0.013 [-0.020, -0.003], p1 + MT marginal -0.010 [-0.017, -0.003] / -0.016 [-0.024, -0.007].
* PTMUL / PTMUL-D (914 and 536 variants of few proteins) have wide intervals (+-0.02 to +-0.04): nothing is above 0; the most negative point estimates are those of the masked families (-0.021 to -0.034) and, on the WT head, of the no sequence head and full MegaScale (-0.018, -0.020).
* On the in-distribution Homology-Restricted MegaScale test set, rank censor threshold 0 on the WT head is above 0 on all variants (+0.003 [+0.001, +0.006]) and on single mutants (+0.002 [+0.001, +0.004]); on the test doubles and the predicted epistasis nothing is above 0 (p1 + MT marginal is -0.056 and -0.050). On the validation set p1 is above 0 on single mutants (WT head, +0.006) and on doubles (combined, +0.018 [+0.001, +0.044]), the opposite sign of its test doubles (-0.030).

## What does beat p0, and where

* **DMS benchmark (composite of DLG4 binding, GRB2 binding, MYO):** several ablations beat p0 on the combined readout: no rank loss +0.019 [+0.011, +0.028], p1 +0.014, p1 + MT marginal +0.013, full MegaScale +0.013 [+0.000, +0.028], rank censor threshold 0 +0.006. These are the ablations that damp or change the MT head, and p0's combined prediction is worse than its WT head on these sets (`../combined_vs_wt`), so the gain is the recovered cost; on the WT head the gains are +0.004 to +0.009 (p1 + MT marginal, no rank loss, rank censor threshold 0), mostly GRB2 binding.
* **Domainome:** full MegaScale +0.013 [+0.006, +0.021] (WT head), no sequence head +0.005, p1 +0.009 on the combined readout (p0's combined prediction is -0.011 below its WT head there).
* **ProteinGym (WT head, seeds 1-3 averaged per assay, held-out assays only):** full MegaScale beats p0, +0.014 on the average of the categories (Stability +0.038, Organismal fitness +0.013; Holm p = 0.002 for all three; Activity +0.005, Binding +0.008, Expression +0.007 are not significant). The family is not homology-restricted, so MegaScale homologs among the ProteinGym stability assays (and Domainome domains) are in its training data: these gains are not clean out-of-distribution gains. p1 is worse on Organismal fitness (-0.013, Holm p = 0.002), p1 + MT marginal on Organismal fitness (-0.011, p = 0.009) and no rank loss on Organismal fitness (-0.006, p = 0.002); rank censor threshold 0 is level with p0 (Stability +0.002, Holm p = 0.16). The p1 combined read is level with p0 on the average (+0.003; Stability -0.006).

## pw0 against p0 (both combined)

No difference anywhere: of the 21 combined-readout cells of pw0 none has an interval excluding 0. Composites (pw0 - p0, combined): thermostability singles +0.000 [-0.004, +0.005], PTMUL / PTMUL-D +0.004 [-0.007, +0.013], MegaScale test all -0.002 [-0.006, +0.003], doubles -0.009 [-0.031, +0.010], predicted epistasis -0.038 [-0.127, +0.016] (the largest point estimate, interval includes 0), DMS +0.005 [-0.001, +0.012]. On the WT head DLG4 binding is above 0 (+0.004 [+0.000, +0.008]) and MYO display (-0.002) and the ProteinGym average (-0.001) are below 0: effects of at most 0.004. pw0 has no combined Domainome or ProteinGym run.

**WT head** (delta rho = family - p0; * = interval excludes 0)

| set | p1 | p1 marg. | pw0 | no rank | rank thr 0 | no seq. head | MT r2 | full Mega |
|---|---|---|---|---|---|---|---|---|
| Thermostability singles (10 sets, composite) | -0.007 [-0.016, +0.000] | -0.010 [-0.017, -0.003]* | +0.001 [-0.003, +0.004] | +0.003 [-0.005, +0.012] | -0.001 [-0.004, +0.001] | +0.001 [-0.004, +0.010] | -0.001 [-0.004, +0.001] | -0.006 [-0.016, +0.004] |
| PTMUL / PTMUL-D (composite) | -0.021 [-0.042, +0.011] | -0.031 [-0.053, +0.003] | -0.006 [-0.020, +0.002] | +0.000 [-0.016, +0.024] | -0.002 [-0.014, +0.007] | -0.018 [-0.035, +0.002] | -0.003 [-0.008, +0.001] | -0.020 [-0.036, +0.009] |
| MegaScale test, all | -0.000 [-0.008, +0.006] | -0.000 [-0.008, +0.006] | +0.002 [-0.001, +0.006] | -0.002 [-0.008, +0.003] | +0.003 [+0.001, +0.006]* | +0.001 [-0.003, +0.005] | -0.001 [-0.003, +0.001] | - |
| MegaScale test, doubles | -0.011 [-0.032, +0.010] | -0.011 [-0.033, +0.010] | +0.003 [-0.011, +0.010] | -0.001 [-0.021, +0.014] | +0.004 [-0.002, +0.010] | +0.000 [-0.009, +0.010] | -0.004 [-0.019, +0.003] | - |
| MegaScale test, epistasis | - | - | - | - | - | - | - | - |
| Domainome | +0.001 [-0.011, +0.010] | +0.001 [-0.002, +0.004] | -0.001 [-0.002, +0.000] | +0.002 [-0.001, +0.005] | +0.001 [+0.000, +0.002]* | +0.005 [+0.001, +0.008]* | +0.002 [+0.001, +0.004]* | +0.013 [+0.006, +0.021]* |
| DMS benchmark (3, composite) | +0.006 [-0.001, +0.014] | +0.009 [+0.004, +0.015]* | +0.001 [-0.004, +0.007] | +0.009 [+0.003, +0.016]* | +0.004 [+0.001, +0.006]* | +0.001 [-0.004, +0.007] | +0.001 [-0.001, +0.004] | +0.002 [-0.011, +0.015] |
| ProteinGym average (194 assays) | -0.006 [-0.010, -0.003]* | -0.004 [-0.008, +0.001] | -0.001 [-0.001, -0.000]* | -0.003 [-0.005, -0.001]* | +0.001 [-0.000, +0.001] | +0.001 [-0.000, +0.003] | -0.001 [-0.001, +0.000] | +0.019 [+0.015, +0.023]* |

**Combined prediction** (delta rho = family - p0; * = interval excludes 0)

| set | p1 | p1 marg. | pw0 | no rank | rank thr 0 | no seq. head | MT r2 | full Mega |
|---|---|---|---|---|---|---|---|---|
| Thermostability singles (10 sets, composite) | -0.013 [-0.020, -0.003]* | -0.016 [-0.024, -0.007]* | +0.000 [-0.004, +0.005] | -0.002 [-0.010, +0.009] | -0.001 [-0.004, +0.004] | -0.002 [-0.007, +0.009] | -0.002 [-0.005, +0.003] | +0.000 [-0.008, +0.010] |
| PTMUL / PTMUL-D (composite) | -0.025 [-0.052, +0.012] | -0.034 [-0.069, +0.011] | +0.004 [-0.007, +0.013] | +0.000 [-0.019, +0.017] | +0.006 [-0.005, +0.019] | -0.007 [-0.021, +0.003] | -0.002 [-0.010, +0.006] | -0.011 [-0.029, +0.014] |
| MegaScale test, all | -0.010 [-0.018, -0.002]* | -0.009 [-0.015, -0.002]* | -0.002 [-0.006, +0.003] | -0.005 [-0.011, +0.002] | -0.001 [-0.003, +0.001] | -0.003 [-0.008, +0.002] | -0.008 [-0.014, -0.003]* | - |
| MegaScale test, doubles | -0.030 [-0.061, -0.007]* | -0.056 [-0.101, -0.023]* | -0.009 [-0.031, +0.010] | -0.012 [-0.037, +0.013] | -0.005 [-0.014, +0.006] | -0.015 [-0.038, +0.002] | -0.025 [-0.062, -0.001]* | - |
| MegaScale test, epistasis | -0.033 [-0.072, +0.045] | -0.050 [-0.125, -0.008]* | -0.038 [-0.127, +0.016] | -0.031 [-0.110, +0.013] | -0.021 [-0.103, +0.022] | -0.016 [-0.075, +0.023] | -0.027 [-0.098, +0.021] | - |
| Domainome | +0.009 [+0.006, +0.013]* | - | - | - | - | - | - | - |
| DMS benchmark (3, composite) | +0.014 [+0.010, +0.022]* | +0.013 [+0.007, +0.020]* | +0.005 [-0.001, +0.012] | +0.019 [+0.011, +0.028]* | +0.006 [+0.003, +0.009]* | +0.003 [-0.004, +0.008] | +0.000 [-0.004, +0.004] | +0.013 [+0.000, +0.028]* |
| ProteinGym average (194 assays) | +0.004 [-0.000, +0.008] | - | - | - | - | - | - | - |

Cells with an interval above / below 0 per ablation and readout (every set cell, and the thermostability cells alone; `results/table_counts.md`; every such cell is listed in `results/table_significant.md`):

| ablation | head | cells | interval above 0 | interval below 0 | thermostability cells | above 0 | below 0 |
|---|---|---|---|---|---|---|---|
| p1 | wt | 27 | 2 | 3 | 12 | 0 | 1 |
| p1 | comb | 27 | 7 | 7 | 12 | 0 | 4 |
| p1 | epi | 2 | 0 | 0 | - | - | - |
| p1 marg. | wt | 27 | 1 | 8 | 12 | 0 | 6 |
| p1 marg. | comb | 21 | 2 | 7 | 12 | 0 | 5 |
| p1 marg. | epi | 2 | 0 | 2 | - | - | - |
| pw0 | wt | 27 | 1 | 1 | 12 | 0 | 0 |
| pw0 | comb | 21 | 0 | 0 | 12 | 0 | 0 |
| pw0 | epi | 2 | 0 | 0 | - | - | - |
| no rank | wt | 27 | 2 | 2 | 12 | 1 | 0 |
| no rank | comb | 21 | 2 | 2 | 12 | 0 | 1 |
| no rank | epi | 2 | 0 | 0 | - | - | - |
| rank thr 0 | wt | 27 | 5 | 0 | 12 | 0 | 0 |
| rank thr 0 | comb | 21 | 2 | 0 | 12 | 0 | 0 |
| rank thr 0 | epi | 2 | 0 | 0 | - | - | - |
| no seq. head | wt | 27 | 2 | 0 | 12 | 0 | 0 |
| no seq. head | comb | 21 | 0 | 2 | 12 | 0 | 0 |
| no seq. head | epi | 2 | 0 | 0 | - | - | - |
| MT r2 | wt | 27 | 1 | 1 | 12 | 0 | 1 |
| MT r2 | comb | 21 | 0 | 6 | 12 | 0 | 1 |
| MT r2 | epi | 2 | 0 | 0 | - | - | - |
| full Mega | wt | 21 | 7 | 1 | 12 | 0 | 1 |
| full Mega | comb | 15 | 1 | 0 | 12 | 0 | 0 |

## Statistics

* Intervals: the bootstrap resamples the same units for p0 and for every family of a cell (`msr_analysis.boot.bootstrap_rho`, one draw per replicate, triplicate seeds resampled too), so each is the interval of the paired difference. Units: proteins (external sets, PTMUL), sites (DLG4 binding, GRB2 binding, MYO), proteins with the per-protein rho averaged (Homology-Restricted MegaScale, Domainome); composites average the rho of their sets over one shared draw. B = 100 (draft): the smallest possible p is 0.01, and the intervals are percentile intervals of 100 replicates (coarse in the tails).
* No multiple-comparison correction is applied to the set-level cells; the counts above are meant to be read against the number of cells (one above 0 in 192 is what chance gives). The ProteinGym p-values are Holm-adjusted over the families x categories of a head.
* Every family has seeds 1-3 on the benchmark sets and on ProteinGym (WT head), and the bootstraps resample the seeds with the assays; only p0 and p1 have a combined ProteinGym run (p1: seeds 1 and 3; its seed 2 covers 126 of 219 assays).
