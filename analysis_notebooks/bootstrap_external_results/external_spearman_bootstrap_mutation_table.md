## Mutation-level bootstrap

B = 10000; 95% percentile CIs; one-sided paired test ESM-MSR > next-best non-SPURS model; Bonferroni within each family (N in the family header); * = adjusted p < 0.05.

### External thermostability, σ=1 only: Bonferroni N=10

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Ssym | σ=1 | 684 | 15 | 0.750 [0.704, 0.792] | 1 | ThermoMPNN | 0.727 [0.681, 0.769] | +0.024 [-0.017, +0.066] | 0.130 | 1.000 | 1.000 | 1.000 |  |
| S461 | σ=1 | 461 | 48 | 0.750 [0.698, 0.796] | 1 | MutateEverything_additive | 0.720 [0.662, 0.771] | +0.030 [-0.008, +0.071] | 0.063 | 0.627 | 0.627 | 0.627 |  |
| S783 | σ=1 | 783 | 55 | 0.725 [0.682, 0.762] | 1 | MutateEverything_additive | 0.717 [0.673, 0.757] | +0.008 [-0.023, +0.040] | 0.308 | 1.000 | 1.000 | 1.000 |  |
| K2369 | σ=1 | 2369 | 181 | 0.705 [0.682, 0.727] | 1 | MutateEverything_additive | 0.689 [0.664, 0.713] | +0.015 [-0.003, +0.035] | 0.059 | 0.591 | 0.591 | 0.591 |  |
| S8754 | σ=1 | 6411 | 301 | 0.617 [0.599, 0.635] | 3 | MutateEverything_additive | 0.657 [0.641, 0.673] | -0.040 [-0.052, -0.027] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| S2648 | σ=1 | 2648 | 131 | 0.665 [0.639, 0.689] | 3 | MutateEverything_additive | 0.719 [0.697, 0.739] | -0.054 [-0.075, -0.032] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Q3421 | σ=1 | 3421 | 148 | 0.613 [0.588, 0.637] | 3 | MutateEverything_additive | 0.654 [0.630, 0.676] | -0.041 [-0.060, -0.022] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| PTMUL-D | σ=1 | 536 | 83 | 0.587 [0.520, 0.648] | 1 | ThermoMPNN-D | 0.563 [0.494, 0.626] | +0.024 [-0.025, +0.074] | 0.167 | 1.000 | 1.000 | 1.000 |  |
| PTMUL | σ=1 | 912 | 90 | 0.545 [0.493, 0.596] | 1 | Rosetta Cartesian DDG | 0.540 [0.486, 0.590] | +0.005 [-0.049, +0.061] | 0.434 | 1.000 | 1.000 | 1.000 |  |
| S571 | σ=1 | 557 | 37 | 0.451 [0.375, 0.525] | 2 | MutateEverything_additive | 0.465 [0.392, 0.536] | -0.014 [-0.069, +0.042] | 0.691 | 1.000 | 1.000 | 1.000 |  |

### Single-protein DMS, σ∈{0.5, 1}: Bonferroni N=6×2=12

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Faure DLG4 Abundance | σ=1 | 6976 | 1 | 0.721 [0.711, 0.731] | 5 | ESM3-small_additive | 0.765 [0.756, 0.774] | -0.044 [-0.054, -0.035] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Faure DLG4 Abundance | σ=0.5 | 6976 | 1 | 0.757 [0.747, 0.765] | 2 | ESM3-small_additive | 0.765 [0.756, 0.774] | -0.009 [-0.015, -0.002] | 0.996 | 1.000 | 1.000 | 1.000 |  |
| Faure DLG4 Binding | σ=1 | 8251 | 1 | 0.739 [0.730, 0.748] | 6 | ESM3-small_additive | 0.781 [0.773, 0.788] | -0.041 [-0.050, -0.033] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Faure DLG4 Binding | σ=0.5 | 8251 | 1 | 0.792 [0.785, 0.799] | 1 | ESM3-small_additive | 0.781 [0.773, 0.788] | +0.011 [+0.006, +0.017] | 3.0e-04 | 0.004 | 0.005 | 0.004 | * |
| Faure GRB2 Abundance | σ=1 | 63366 | 1 | 0.734 [0.730, 0.738] | 1 | MutateEverything_additive | 0.711 [0.707, 0.715] | +0.023 [+0.020, +0.025] | 1.0e-04 | 0.001 | 0.002 | 0.001 | * |
| Faure GRB2 Abundance | σ=0.5 | 63366 | 1 | 0.747 [0.743, 0.750] | 1 | MutateEverything_additive | 0.711 [0.707, 0.715] | +0.035 [+0.033, +0.038] | 1.0e-04 | 0.001 | 0.002 | 0.001 | * |
| Faure GRB2 Binding | σ=1 | 33441 | 1 | 0.623 [0.616, 0.630] | 7 | ESM3-small-open_additive | 0.740 [0.735, 0.745] | -0.117 [-0.122, -0.112] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Faure GRB2 Binding | σ=0.5 | 33441 | 1 | 0.692 [0.686, 0.698] | 6 | ESM3-small-open_additive | 0.740 [0.735, 0.745] | -0.048 [-0.051, -0.044] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Küng Myoglobin Display | σ=1 | 5905 | 1 | 0.636 [0.619, 0.655] | 1 | MutateEverything_additive | 0.586 [0.567, 0.604] | +0.051 [+0.040, +0.062] | 1.0e-04 | 0.001 | 0.002 | 0.001 | * |
| Küng Myoglobin Display | σ=0.5 | 5905 | 1 | 0.627 [0.609, 0.645] | 1 | MutateEverything_additive | 0.586 [0.567, 0.604] | +0.041 [+0.029, +0.053] | 1.0e-04 | 0.001 | 0.002 | 0.001 | * |
| Nutschel EstA ΔT_50 | σ=1 | 2172 | 1 | 0.517 [0.483, 0.550] | 2 | ProteinMPNN | 0.546 [0.513, 0.577] | -0.029 [-0.057, +0.000] | 0.973 | 1.000 | 1.000 | 1.000 |  |
| Nutschel EstA ΔT_50 | σ=0.5 | 2172 | 1 | 0.542 [0.509, 0.574] | 2 | ProteinMPNN | 0.546 [0.513, 0.577] | -0.004 [-0.029, +0.021] | 0.619 | 1.000 | 1.000 | 1.000 |  |

### Domainome (522 domains): N=2 (σ)

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Beltran Domainome Abundance (per-domain avg.) | σ=1 | 536145 | 522 | 0.548 [0.545, 0.549] | 4 | ESM3-small_additive | 0.551 [0.548, 0.553] | -0.004 [-0.005, -0.002] | 1.000 | 1.000 | 1.000 | 1.000 |  |
| Beltran Domainome Abundance (per-domain avg.) | σ=0.5 | 536145 | 522 | 0.573 [0.570, 0.574] | 1 | ESM3-small_additive | 0.551 [0.548, 0.553] | +0.022 [+0.020, +0.023] | 1.0e-04 | 2.0e-04 | 2.0e-04 | 2.0e-04 | * |

### Overall (macro-average of Δρ across datasets in the family)

Per family and σ: one test vs each dataset's own next-best model, plus one per method present in every dataset (Holm across methods). Adjusted p additionally multiplies by the number of σ values tested (`p adj`; `p adj (σ=3)` counts three σ values for the DMS family).

| Family | σ | Comparator | datasets won | mean Δρ [95% CI] | p | p adj | p adj (σ=3) |
|---|---|---|---:|---|---:|---:|---:|
| stability | σ=1 | best other (per dataset) | 6/10 | -0.004 [-0.016, +0.008] | 0.754 | 0.754 | 0.754 |
| stability | σ=1 | ThermoMPNN | 10/10 | +0.045 [+0.033, +0.058] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | MutateEverything_additive | 6/10 | +0.015 [+0.003, +0.026] | 0.005 | 0.005 | 0.005 |
| stability | σ=1 | ProteinMPNN | 10/10 | +0.158 [+0.142, +0.175] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | MutateEverything | 7/10 | +0.026 [+0.015, +0.038] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | Rosetta Cartesian DDG | 10/10 | +0.080 [+0.065, +0.096] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-medium | 10/10 | +0.222 [+0.207, +0.237] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-medium_additive | 10/10 | +0.220 [+0.204, +0.235] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small | 10/10 | +0.208 [+0.193, +0.222] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small_additive | 10/10 | +0.202 [+0.188, +0.216] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small-open | 10/10 | +0.261 [+0.246, +0.276] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small-open_additive | 10/10 | +0.259 [+0.244, +0.274] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-large_additive | 10/10 | +0.357 [+0.336, +0.378] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-large | 10/10 | +0.370 [+0.348, +0.391] | 1.0e-04 | 0.001 | 0.001 |
| dms | σ=1 | best other (per dataset) | 2/6 | -0.026 [-0.032, -0.021] | 1.000 | 1.000 | 1.000 |
| dms | σ=1 | ESM3-small_additive | 3/6 | -0.001 [-0.006, +0.005] | 0.585 | 1.000 | 1.000 |
| dms | σ=1 | ESM3-small | 3/6 | +0.012 [+0.007, +0.018] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-medium_additive | 3/6 | +0.012 [+0.007, +0.018] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-medium | 3/6 | +0.029 [+0.023, +0.035] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | MutateEverything_additive | 6/6 | +0.030 [+0.025, +0.034] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-small-open_additive | 4/6 | +0.020 [+0.014, +0.025] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-small-open | 5/6 | +0.038 [+0.033, +0.044] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | MutateEverything | 6/6 | +0.075 [+0.070, +0.080] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ThermoMPNN | 6/6 | +0.067 [+0.062, +0.072] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ProteinMPNN | 5/6 | +0.081 [+0.074, +0.087] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-large | 6/6 | +0.147 [+0.139, +0.155] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=1 | ESM3-large_additive | 6/6 | +0.138 [+0.130, +0.146] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | best other (per dataset) | 3/6 | +0.005 [-0.000, +0.009] | 0.031 | 0.063 | 0.094 |
| dms | σ=0.5 | ESM3-small_additive | 4/6 | +0.030 [+0.026, +0.034] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-small | 5/6 | +0.043 [+0.039, +0.047] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-medium_additive | 5/6 | +0.043 [+0.039, +0.047] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-medium | 6/6 | +0.059 [+0.055, +0.064] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | MutateEverything_additive | 6/6 | +0.060 [+0.055, +0.066] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-small-open_additive | 5/6 | +0.051 [+0.047, +0.055] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-small-open | 5/6 | +0.069 [+0.065, +0.073] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | MutateEverything | 6/6 | +0.106 [+0.100, +0.111] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ThermoMPNN | 6/6 | +0.098 [+0.093, +0.103] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ProteinMPNN | 5/6 | +0.112 [+0.106, +0.117] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-large | 6/6 | +0.178 [+0.171, +0.185] | 1.0e-04 | 0.002 | 0.004 |
| dms | σ=0.5 | ESM3-large_additive | 6/6 | +0.169 [+0.162, +0.176] | 1.0e-04 | 0.002 | 0.004 |
