# AlphaFold2 vs. Crystal Structures on External Thermostability Benchmarks (ESM-MSR (FINAL_p0, 3 seeds))

Matched evaluation of ColabFold AF2 structures (762 unique WT sequences, 5 models, 3 recycles, Amber relax) against experimental crystal structures across all 12 external thermostability sets (25,632 rows; canonical deployed head: WT-view on single-mutant sets, Dual-view on PTMUL / PTMUL-D; seeds 1–3 averaged; joint [cluster × seed] and [mutation × seed] bootstraps, $B = 10,000$, two-sided $p$-values).

| Dataset | n (clust.) | Head | Crystal ρ | AF2 ρ | Δρ (AF2 − Cryst.) [95% prot. CI] | p_prot (Holm) | p_mut | Crystal / AF2 r | Crystal / AF2 RMSE | ρ(AF2, Cryst.) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K2369 | 2,369 (181) | WT-view | 0.719±0.002 | 0.727±0.001 | +0.008 [−0.005, +0.017] | 0.290 (1.000) | 0.004 | 0.755 / 0.753 | 1.062 / 1.084 | 0.984 | Main-text, single; ⊂ K3822 |
| K3822 | 3,822 (201) | WT-view | 0.722±0.003 | 0.731±0.004 | +0.010 [−0.001, +0.019] | 0.071 (0.637) | <10^-4 | 0.744 / 0.748 | 1.048 / 1.043 | 0.982 | Additional, single; ⊃ K2369 |
| PTMUL | 914 (90) | Dual-view | 0.591±0.008 | 0.570±0.006 | −0.021 [−0.042, +0.001] | 0.063 (0.628) | <10^-4 | 0.598 / 0.585 | 1.889 / 1.944 | 0.982 | Main-text, multi; ⊃ PTMUL-D |
| PTMUL-D | 536 (83) | Dual-view | 0.596±0.006 | 0.556±0.006 | −0.040 [−0.068, −0.009] | 0.015 (0.180) | <10^-4 | 0.549 / 0.516 | 1.882 / 1.945 | 0.972 | Additional, multi; ⊂ PTMUL |
| Q3421 | 3,421 (148) | WT-view | 0.618±0.003 | 0.612±0.005 | −0.005 [−0.018, +0.008] | 0.415 (1.000) | 0.096 | 0.542 / 0.536 | 1.662 / 1.680 | 0.980 | Main-text, single; 98% in S8754 |
| S461 | 461 (48) | WT-view | 0.768±0.004 | 0.773±0.005 | +0.004 [−0.030, +0.029] | 0.773 (1.000) | 0.656 | 0.767 / 0.765 | 0.839 / 0.816 | 0.963 | Main-text, single; ⊂ S669 |
| S571 | 571 (37) | WT-view | 0.415±0.003 | 0.417±0.002 | +0.002 [−0.017, +0.032] | 0.801 (1.000) | 0.792 | 0.325 / 0.329 | 9.030 / 8.975°C | 0.969 | Additional, single; ΔTm (°C) |
| S669 | 669 (94) | WT-view | 0.536±0.005 | 0.533±0.008 | −0.003 [−0.032, +0.019] | 0.827 (1.000) | 0.742 | 0.489 / 0.489 | 1.444 / 1.463 | 0.967 | Additional, single; ⊃ S461 |
| S783 | 783 (55) | WT-view | 0.707±0.008 | 0.701±0.008 | −0.006 [−0.015, +0.004] | 0.201 (1.000) | 0.108 | 0.733 / 0.735 | 1.376 / 1.355 | 0.989 | Main-text, single; 90% in S8754 |
| S2648 | 2,648 (131) | WT-view | 0.664±0.006 | 0.659±0.008 | −0.006 [−0.021, +0.011] | 0.508 (1.000) | 0.140 | 0.634 / 0.625 | 1.147 / 1.184 | 0.979 | Main-text, single; ⊂ S8754 |
| S8754 | 8,754 (301) | WT-view | 0.617±0.003 | 0.617±0.003 | −0.001 [−0.013, +0.009] | 0.932 (1.000) | 0.716 | 0.599 / 0.596 | 1.593 / 1.607 | 0.984 | Main-text, single; ⊃ S2648, Ssym-dir |
| Ssym | 684 (15) | WT-view | 0.778±0.004 | 0.754±0.005 | −0.024 [−0.067, −0.000] | 0.045 (0.499) | 0.0002 | 0.790 / 0.772 | 1.156 / 1.194 | 0.980 | Main-text, single; 342 dir + 342 rev |
| **Mean (10 single-mutant sets)** | 24,182 (443) | WT-view | 0.654±0.004 | 0.652±0.005 | −0.002 [−0.012, +0.006] | 0.609 | 0.295 | 0.638 / 0.635 | 1.259 / 1.270 | 0.978 | Unweighted macro-mean (RMSE excl. S571) |
| **Mean (8 main-text sets)** | 20,034 (403) | Deployed | 0.683±0.003 | 0.676±0.003 | −0.006 [−0.017, +0.002] | 0.160 | 0.001 | 0.677 / 0.671 | 1.340 / 1.358 | 0.980 | Fig. 3a cohort (excl. S571, S669, K3822, PTMUL-D) |
| **Mean (2 multi-mutant sets)** | 1,450 (90) | Dual-view | 0.594±0.004 | 0.563±0.003 | −0.030 [−0.053, −0.004] | 0.023 | <10^-4 | 0.574 / 0.550 | 1.886 / 1.945 | 0.977 | PTMUL & PTMUL-D |
| **Mean (all 12 sets)** | 25,632 (451) | Deployed | 0.644±0.003 | 0.637±0.004 | −0.007 [−0.017, +0.002] | 0.141 | <10^-4 | 0.627 / 0.621 | 1.373 / 1.392 | 0.978 | All 12 external benchmarks (RMSE excl. S571) |
