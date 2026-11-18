# Combined versus WT head: when does the combined prediction pay?

ESM-MSR has two heads on every variant: the **WT head** (`wt_lora_pred`: the wild-type pass; on a multi-mutant row, its own prediction of that row) and the **combined prediction** (`combined_pred` = 1/2 (MT + WT), the MT head reads the mutant sequence). The WT head cannot represent an interaction on its own; the combined prediction can. This folder asks, for every kind of assay and variant, whether the combined prediction ranks the variants better than the WT head, and under which training recipe.

All comparisons are **paired** (the same variants, checkpoint, seed and sigma = 1 for both readouts), reported as delta rho = rho(combined) - rho(WT head).

| family | what it is | seeds | benchmarks with a combined run |
|---|---|---|---|
| p0 | FINAL_p0 = ESM-MSR: unmasked WT and MT passes | 1-3 (ProteinGym 1-3, figures use 1) | all |
| p1 | FINAL_p1: independent single-site masking of both passes | 1-3 (ProteinGym: 1) | all |
| pw0 | FINAL_p0_pw0: FINAL_p0 without the pair loss (`block_pair_weight` 0.38 -> 0) | 1-3 | all benchmark sets except Domainome (WT head only) and ProteinGym (no combined run) |

Inputs are the testing queue's prediction files (`msr_analysis.final_runs`; no GPU). The 23 ProteinGym stability assays whose variants are in the training data are always left out (194 assays; `pgsigma.train_overlap`).

```bash
python analysis/combined_vs_wt/01_benchmark_delta.py   # results/benchmark_delta.csv, epistasis_rho.csv: per set, variant class and family, rho of the three heads, delta rho with a paired bootstrap interval (B = 100, draft)
python analysis/combined_vs_wt/02_pg_per_assay.py      # results/pg_per_assay.csv: per ProteinGym assay, rho of WT / MT / combined on all, single and multi-mutant rows
python analysis/combined_vs_wt/03_unit_rho.py          # results/unit_rho.csv: per Homology-Restricted MegaScale library and Domainome domain, the same
python analysis/combined_vs_wt/04_fig_proteingym.py    # pg_combined_vs_wt + results/pg_delta_pvalues.csv
python analysis/combined_vs_wt/05_fig_overview.py      # overview_forest, by_mutation_count
python analysis/combined_vs_wt/06_fig_when.py          # when_rule + results/when_rule.csv
python analysis/combined_vs_wt/07_tables.py            # table_sets.md, table_groups.md, table_nmut.md
python analysis/combined_vs_wt/08_all_families.py      # results/table_all_families.md: combined - WT head inside every FINAL_* family (DMS, external sets, MegaScale)
```

| figure | content |
|---|---|
| `overview_forest` | delta rho of every set (p0 circle, p1 diamond, pw0 square, 95% interval, filled when it excludes 0) in four groups: stability / abundance single mutants, stability multi-mutants, the DMS benchmark sets, ProteinGym by category; the p0 rho of the WT head and of the combined prediction at the right |
| `by_mutation_count` | delta rho against the number of mutations for every set with multi-mutants; last panel: the rho of the predicted epistasis comb(AB) - WT(A) - WT(B) with the measured dddG (Homology-Restricted MegaScale doubles; the WT head has none) |
| `pg_combined_vs_wt` | ProteinGym assays: delta rho per category (box, dots, mean, Holm-adjusted bootstrap p), and the single mutants and multi-mutants of the assays that have both, for Stability and for the other categories |
| `when_rule` | delta_comb against delta_mt = rho(MT head) - rho(WT head) per protein / assay / set, with the least-squares line and its break-even |

## What the figures say

delta rho = rho(combined) - rho(WT head), seed means; the group table averages the sets of a group with equal weight (`results/table_groups.md`; every set with its interval is in `results/table_sets.md`, the counts by number of mutations in `results/table_nmut.md`).

| group | sets | delta p0 | delta p1 | delta pw0 |
|---|---|---|---|---|
| Stability / abundance, single mutants | 13 | +0.007 | +0.002 | - |
| Homology-Restricted MegaScale, single mutants | 2 | +0.013 | +0.008 | +0.011 |
| External single-mutant sets (10) | 10 | +0.007 | +0.001 | +0.006 |
| Stability, multi-mutants (MegaScale doubles, PTMUL, PTMUL-D) | 4 | +0.027 | +0.021 | +0.031 |
| Homology-Restricted MegaScale, doubles | 2 | +0.047 | +0.038 | +0.043 |
| DMS benchmark sets (3, all variants) | 3 | -0.020 | -0.012 | -0.016 |
| DMS sets, single mutants only | 3 | -0.017 | -0.007 | -0.016 |
| DMS sets, multi-mutants only | 3 | -0.022 | -0.013 | -0.017 |

| ProteinGym group | assays | delta p0 | p (Holm) | delta p1 | p (Holm) |
|---|---|---|---|---|---|
| Stability (all variants) | 43 | +0.019 | p=0.009 | +0.012 | p=0.29 |
| Expression / abundance (all variants) | 18 | -0.007 | p=0.77 | +0.002 | p=0.77 |
| Activity (all variants) | 43 | -0.016 | p<0.001 | +0.003 | p=0.77 |
| Binding (all variants) | 13 | -0.011 | p=0.77 | -0.005 | p=0.79 |
| Organismal fitness (all variants) | 77 | -0.024 | p<0.001 | +0.000 | p=0.93 |
| Stability, single | 34 | +0.007 | p=0.004 | +0.008 | p<0.001 |
| Stability, multi | 34 | +0.082 | p<0.001 | +0.057 | p<0.001 |
| Other categories, single | 18 | -0.033 | p<0.001 | +0.001 | p=0.80 |
| Other categories, multi | 18 | -0.035 | p<0.001 | -0.010 | p=0.21 |

1. **Multi-mutants of stability phenotypes: combine.** The gain is the largest there and consistent: Homology-Restricted MegaScale doubles +0.054 (test, interval +0.030 to +0.080) and +0.041 (val) for p0, +0.035 / +0.041 for p1, +0.042 / +0.045 for pw0; ProteinGym stability assays, multi-mutant rows: +0.082 (p0) and +0.057 (p1), 30 of 34 assays gain, both p < 0.001. On the ESM-MSR deployed readout the combined prediction is also the only one that carries epistasis: rho of comb(AB) - WT(A) - WT(B) with the measured dddG is 0.628 (val) and 0.479 (test) for p0, 0.535 / 0.446 for p1 and 0.543 / 0.441 for pw0 (intervals overlap). PTMUL and PTMUL-D (914 and 536 variants of few proteins) are positive but their intervals include 0 (+0.003 and +0.012 for p0).
2. **Single mutants of stability phenotypes: a small gain.** +0.013 / +0.014 on Homology-Restricted MegaScale val / test (intervals exclude 0), +0.007 on the 10 external single-mutant sets on average (S669 +0.020 and S8754 +0.011 exclude 0, the others do not), +0.007 / +0.008 for the ProteinGym stability assays' single mutants (p0 / p1, Holm p < 0.005).
3. **Function, binding, abundance and fitness phenotypes: use the WT head under p0.** The three DMS benchmark sets lose -0.020 on average (DLG4 binding -0.029, GRB2 binding -0.029, MYO display -0.003); Domainome -0.011 (interval -0.016 to -0.006); ProteinGym Activity -0.016 and Organismal fitness -0.024 (both p < 0.001), Binding -0.011 and Expression -0.007 (not significant, 13 and 18 assays); on the 18 non-stability assays with both single and multi-mutants the loss is the same on either: -0.033 (singles) and -0.035 (multi-mutants). The cost is similar for the single and double mutants of DLG4 binding (-0.031 / -0.031) and GRB2 binding (-0.024 / -0.029); the gain on Homology-Restricted MegaScale grows from singles to doubles (`by_mutation_count`).
4. **Masking (p1) removes most of the cost and keeps most of the gain.** Under p1 the DMS sets lose -0.012 on average, ProteinGym Activity, Binding, Expression and Organismal fitness are within +-0.005 of 0 (all p > 0.7), Domainome -0.002; the stability gain stays on multi-mutants (+0.057 on the ProteinGym assays' multi-mutants, +0.035 / +0.041 on the MegaScale doubles) and on singles (+0.008). Masked, the combined prediction is within 0.005 of the WT head on the ProteinGym function categories and costs at most 0.02 on the DMS sets (DLG4 binding -0.014, GRB2 binding -0.018), and it is better on stability multi-mutants; unmasked, it is better by a wider margin on stability multi-mutants and worse on function phenotypes.
5. **pw0 behaves like p0.** Without the pair loss the combined prediction gains +0.010 / +0.012 on MegaScale singles, +0.042 / +0.045 on its doubles, and loses -0.016 on average on the DMS sets (p0: +0.013 / +0.014, +0.054 / +0.041, -0.020); on PTMUL and PTMUL-D it gains more (+0.014, +0.022, intervals include 0). So the pair loss is not what makes the MT head help on stability or hurt on function; it changes the MegaScale doubles gain by +0.012 (test) and -0.004 (val), within the intervals.

**Why: the MT head's deficit and the disagreement between the heads (`when_rule`, `results/when_rule.csv`).** Over the 772 ProteinGym assays, MegaScale libraries and Domainome domains of p0, delta_comb = +0.007 + 0.31 delta_mt (R^2 0.56), where delta_mt = rho(MT head) - rho(WT head): the combined prediction wins when the MT head is within 0.024 rho of the WT head (p1: 0.014). The MT head is 0.04 to 0.08 rho worse than the WT head on the ProteinGym function categories (Activity -0.060, Binding -0.059, Organismal fitness -0.081, Expression -0.043) and 0.050 worse on the stability assays, but the stability assays also have the lowest correlation between the two heads (Spearman r 0.80 against 0.84 to 0.90), so averaging them adds information. Adding the disagreement as a second predictor, delta_comb = -0.006 + 0.42 delta_mt + 0.18 (1 - r), R^2 0.84 (p1: 0.77). This is a description of the 772 units, not a causal test.

**Does an ablation make the combined prediction beat its own WT head on the DMS sets? No** (`08_all_families.py`, combined - WT head of the same family, * = interval excludes 0). On the DMS composite every family still loses: p0 -0.020, no rank loss -0.011 [-0.015, -0.007], p1 -0.011, full MegaScale -0.009, MT LoRA rank 2 -0.021; DLG4 binding and GRB2 binding stay significantly negative in every family except full MegaScale on GRB2 binding (-0.002). On the stability sets every family gains (single mutants +0.007 to +0.014; doubles +0.033 to +0.054, the intervals of p1, p1 + MT marginal and MT LoRA rank 2 on the doubles include 0).

| set | p0 (reference) | p1 masked | p1 MT marginal | no pair loss | no rank loss | rank censor 0 | no seq. head | MT LoRA r2 | full MegaScale |
|---|---|---|---|---|---|---|---|---|---|
| DMS benchmark (3, composite) | -0.020 [-0.026, -0.014]* | -0.011 [-0.018, -0.006]* | -0.016 [-0.020, -0.011]* | -0.016 [-0.020, -0.012]* | -0.011 [-0.015, -0.007]* | -0.018 [-0.023, -0.012]* | -0.018 [-0.025, -0.012]* | -0.021 [-0.028, -0.015]* | -0.009 [-0.014, -0.004]* |
| DLG4 binding | -0.029 [-0.044, -0.016]* | -0.014 [-0.024, -0.002]* | -0.018 [-0.030, -0.005]* | -0.022 [-0.034, -0.012]* | -0.014 [-0.022, -0.007]* | -0.026 [-0.038, -0.016]* | -0.026 [-0.043, -0.014]* | -0.035 [-0.050, -0.021]* | -0.018 [-0.030, -0.007]* |
| GRB2 binding | -0.029 [-0.038, -0.021]* | -0.018 [-0.027, -0.012]* | -0.026 [-0.034, -0.019]* | -0.026 [-0.034, -0.016]* | -0.018 [-0.027, -0.011]* | -0.025 [-0.034, -0.018]* | -0.027 [-0.037, -0.017]* | -0.028 [-0.034, -0.021]* | -0.002 [-0.009, +0.006] |
| MYO display | -0.003 [-0.011, +0.004] | -0.002 [-0.007, +0.003] | -0.004 [-0.009, -0.000]* | -0.001 [-0.008, +0.006] | -0.000 [-0.008, +0.006] | -0.002 [-0.011, +0.005] | -0.002 [-0.009, +0.004] | +0.000 [-0.008, +0.005] | -0.007 [-0.016, +0.001] |
| External single-mutant sets (10, composite) | +0.007 [+0.001, +0.013]* | +0.001 [-0.004, +0.008] | +0.002 [-0.003, +0.007] | +0.006 [+0.000, +0.013]* | +0.002 [-0.003, +0.007] | +0.008 [+0.001, +0.014]* | +0.004 [-0.002, +0.011] | +0.007 [+0.002, +0.013]* | +0.013 [+0.006, +0.020]* |
| PTMUL / PTMUL-D (composite) | +0.007 [-0.024, +0.030] | +0.003 [-0.028, +0.028] | +0.005 [-0.023, +0.028] | +0.018 [-0.017, +0.047] | +0.007 [-0.027, +0.033] | +0.016 [-0.023, +0.044] | +0.018 [-0.025, +0.049] | +0.009 [-0.023, +0.034] | +0.017 [-0.018, +0.044] |
| MegaScale test, single mutants | +0.014 [+0.008, +0.019]* | +0.007 [+0.003, +0.010]* | +0.007 [+0.003, +0.011]* | +0.010 [+0.006, +0.015]* | +0.012 [+0.008, +0.016]* | +0.011 [+0.007, +0.015]* | +0.011 [+0.006, +0.015]* | +0.008 [+0.004, +0.013]* | - |
| MegaScale test, doubles | +0.054 [+0.030, +0.079]* | +0.035 [-0.006, +0.067] | +0.009 [-0.041, +0.040] | +0.042 [+0.010, +0.068]* | +0.043 [+0.009, +0.069]* | +0.045 [+0.015, +0.070]* | +0.039 [+0.002, +0.063]* | +0.033 [-0.014, +0.064] | - |

## Statistics

* Benchmark sets (`01_benchmark_delta.py`): the bootstrap resamples the same units for the three heads and the three families of a cell (`msr_analysis.boot.bootstrap_rho`, seeds weighted as in the rest of `analysis/`), so the interval is that of the paired difference. Units: proteins (external sets, PTMUL), sites (DLG4 binding, GRB2 binding, MYO: first mutated position), proteins with the per-protein rho averaged (Homology-Restricted MegaScale, Domainome). B = 100 (draft sizes).
* ProteinGym (`04_fig_proteingym.py`): seed 1 of p0 and p1 (p1 seed 2 is incomplete); the mean over p0's three seeds of delta rho is in `results/pg_delta_pvalues.csv` (Stability +0.019, Expression -0.005, Activity -0.015, Binding -0.008, Organismal fitness -0.021; seed 1: +0.019, -0.007, -0.016, -0.012, -0.024). p-values: two-sided bootstrap over assays (20000 resamples, `msr_analysis.boot.paired_diff`'s convention), Holm-adjusted over the group x family comparisons of a figure row; a group needs 10 assays to get a p-value; the Wilcoxon signed-rank p is in the csv. Assays are treated as independent, so several assays of one protein count separately.
* pw0 has no combined ProteinGym or Domainome run (WT head only), so it is missing from those rows.
