# AlphaFold2 vs. Crystal Structures on External Thermostability Benchmarks (ESM-MSR-small)

Matched evaluation of ColabFold AF2 structures (762 unique WT sequences, 5 models, 3 recycles, Amber relax) against experimental crystal structures across all 12 external thermostability sets (25,632 rows; canonical deployed head: WT-view on single-mutant sets, Dual-view on PTMUL / PTMUL-D; paired joint protein-cluster and mutation-level bootstraps, $B = 10,000$, two-sided $p$-values).

| Dataset | n (clust.) | Head | Crystal ρ | AF2 ρ | Δρ (AF2 − Cryst.) [95% prot. CI] | p_prot (Holm) | p_mut | Crystal / AF2 r | Crystal / AF2 RMSE | ρ(AF2, Cryst.) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K2369 | 2,369 (181) | WT-view | 0.702 | 0.715 | +0.013 [−0.004, +0.022] | 0.179 (1.000) | <10^-4 | 0.723 / 0.726 | 1.127 / 1.139 | 0.981 | Main-text, single; ⊂ K3822 |
| K3822 | 3,822 (201) | WT-view | 0.700 | 0.715 | +0.015 [+0.000, +0.026] | 0.044 (0.477) | <10^-4 | 0.718 / 0.726 | 1.097 / 1.082 | 0.978 | Additional, single; ⊃ K2369 |
| PTMUL | 914 (90) | Dual-view | 0.543 | 0.541 | −0.001 [−0.033, +0.018] | 0.779 (1.000) | 0.796 | 0.561 / 0.560 | 1.938 / 1.976 | 0.978 | Main-text, multi; ⊃ PTMUL-D |
| PTMUL-D | 536 (83) | Dual-view | 0.577 | 0.546 | −0.030 [−0.069, −0.002] | 0.035 (0.415) | 0.002 | 0.541 / 0.511 | 1.886 / 1.925 | 0.967 | Additional, multi; ⊂ PTMUL |
| Q3421 | 3,421 (148) | WT-view | 0.619 | 0.611 | −0.008 [−0.020, +0.004] | 0.199 (1.000) | 0.008 | 0.545 / 0.539 | 1.659 / 1.663 | 0.979 | Main-text, single; 98% in S8754 |
| S461 | 461 (48) | WT-view | 0.739 | 0.756 | +0.017 [−0.031, +0.051] | 0.482 (1.000) | 0.203 | 0.743 / 0.759 | 0.882 / 0.820 | 0.950 | Main-text, single; ⊂ S669 |
| S571 | 571 (37) | WT-view | 0.441 | 0.450 | +0.009 [−0.012, +0.038] | 0.380 (1.000) | 0.386 | 0.392 / 0.392 | 9.013 / 8.953°C | 0.960 | Additional, single; ΔTm (°C) |
| S669 | 669 (94) | WT-view | 0.532 | 0.545 | +0.014 [−0.025, +0.046] | 0.538 (1.000) | 0.199 | 0.498 / 0.514 | 1.426 / 1.412 | 0.957 | Additional, single; ⊃ S461 |
| S783 | 783 (55) | WT-view | 0.714 | 0.706 | −0.008 [−0.020, +0.005] | 0.225 (1.000) | 0.070 | 0.737 / 0.736 | 1.400 / 1.379 | 0.986 | Main-text, single; 90% in S8754 |
| S2648 | 2,648 (131) | WT-view | 0.673 | 0.664 | −0.009 [−0.023, +0.007] | 0.242 (1.000) | 0.010 | 0.649 / 0.641 | 1.120 / 1.133 | 0.977 | Main-text, single; ⊂ S8754 |
| S8754 | 8,754 (301) | WT-view | 0.606 | 0.606 | +0.000 [−0.014, +0.012] | 0.968 (1.000) | 0.970 | 0.573 / 0.570 | 1.632 / 1.637 | 0.982 | Main-text, single; ⊃ S2648, Ssym-dir |
| Ssym | 684 (15) | WT-view | 0.772 | 0.744 | −0.027 [−0.080, −0.001] | 0.043 (0.477) | <10^-4 | 0.792 / 0.772 | 1.165 / 1.198 | 0.976 | Main-text, single; 342 dir + 342 rev |
| **Mean (10 single-mutant sets)** | 24,182 (443) | WT-view | 0.650 | 0.651 | +0.002 [−0.011, +0.011] | 0.898 | 0.472 | 0.637 / 0.637 | 1.279 / 1.274 | 0.972 | Unweighted macro-mean (RMSE excl. S571) |
| **Mean (8 main-text sets)** | 20,034 (403) | Deployed | 0.671 | 0.668 | −0.003 [−0.016, +0.006] | 0.444 | 0.187 | 0.665 / 0.663 | 1.365 / 1.368 | 0.976 | Fig. 3a cohort (excl. S571, S669, K3822, PTMUL-D) |
| **Mean (2 multi-mutant sets)** | 1,450 (90) | Dual-view | 0.560 | 0.544 | −0.016 [−0.050, +0.006] | 0.161 | 0.005 | 0.551 / 0.535 | 1.912 / 1.950 | 0.973 | PTMUL & PTMUL-D |
| **Mean (all 12 sets)** | 25,632 (451) | Deployed | 0.635 | 0.633 | −0.001 [−0.014, +0.008] | 0.670 | 0.542 | 0.623 / 0.620 | 1.394 / 1.397 | 0.972 | All 12 external benchmarks (RMSE excl. S571) |
