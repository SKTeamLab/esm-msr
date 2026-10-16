## Protein-level bootstrap

B = 10000; 95% percentile CIs; one-sided paired test ESM-MSR > next-best non-SPURS model; Bonferroni within each family (N in the family header); * = adjusted p < 0.05.

### External thermostability, σ=1 only: Bonferroni N=10

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Ssym | σ=1 | 684 | 15 | 0.750 [0.458, 0.848] | 1 | ThermoMPNN | 0.727 [0.522, 0.790] | +0.024 [-0.110, +0.090] | 0.265 | 1.000 | 1.000 | 1.000 |  |
| S461 | σ=1 | 461 | 48 | 0.750 [0.637, 0.807] | 1 | MutateEverything_additive | 0.720 [0.590, 0.782] | +0.030 [-0.007, +0.083] | 0.050 | 0.504 | 0.504 | 0.504 |  |
| S783 | σ=1 | 783 | 55 | 0.725 [0.626, 0.790] | 1 | MutateEverything_additive | 0.717 [0.635, 0.769] | +0.008 [-0.060, +0.066] | 0.402 | 1.000 | 1.000 | 1.000 |  |
| K2369 | σ=1 | 2369 | 181 | 0.705 [0.668, 0.762] | 1 | MutateEverything_additive | 0.689 [0.647, 0.751] | +0.015 [-0.021, +0.054] | 0.170 | 1.000 | 1.000 | 1.000 |  |
| S8754 | σ=1 | 6411 | 301 | 0.617 [0.563, 0.670] | 3 | MutateEverything_additive | 0.657 [0.614, 0.698] | -0.040 [-0.076, -0.004] | 0.986 | 1.000 | 1.000 | 1.000 |  |
| S2648 | σ=1 | 2648 | 131 | 0.665 [0.598, 0.718] | 3 | MutateEverything_additive | 0.719 [0.679, 0.748] | -0.054 [-0.108, +0.001] | 0.973 | 1.000 | 1.000 | 1.000 |  |
| Q3421 | σ=1 | 3421 | 148 | 0.613 [0.520, 0.679] | 3 | MutateEverything_additive | 0.654 [0.577, 0.709] | -0.041 [-0.091, +0.007] | 0.948 | 1.000 | 1.000 | 1.000 |  |
| PTMUL-D | σ=1 | 536 | 83 | 0.587 [0.478, 0.681] | 1 | ThermoMPNN-D | 0.563 [0.450, 0.674] | +0.024 [-0.066, +0.097] | 0.304 | 1.000 | 1.000 | 1.000 |  |
| PTMUL | σ=1 | 912 | 90 | 0.545 [0.450, 0.665] | 1 | Rosetta Cartesian DDG | 0.540 [0.399, 0.649] | +0.005 [-0.093, +0.143] | 0.404 | 1.000 | 1.000 | 1.000 |  |
| S571 | σ=1 | 557 | 37 | 0.451 [0.278, 0.599] | 2 | MutateEverything_additive | 0.465 [0.305, 0.610] | -0.014 [-0.126, +0.054] | 0.672 | 1.000 | 1.000 | 1.000 |  |

### Single-protein DMS, σ∈{0.5, 1}: Bonferroni N=6×2=12

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Faure DLG4 Abundance | σ=1 | 6976 | 1 | 0.721 | 5 | ESM3-small_additive | 0.765 | -0.044 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure DLG4 Abundance | σ=0.5 | 6976 | 1 | 0.757 | 2 | ESM3-small_additive | 0.765 | -0.009 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure DLG4 Binding | σ=1 | 8251 | 1 | 0.739 | 6 | ESM3-small_additive | 0.781 | -0.041 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure DLG4 Binding | σ=0.5 | 8251 | 1 | 0.792 | 1 | ESM3-small_additive | 0.781 | +0.011 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure GRB2 Abundance | σ=1 | 63366 | 1 | 0.734 | 1 | MutateEverything_additive | 0.711 | +0.023 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure GRB2 Abundance | σ=0.5 | 63366 | 1 | 0.747 | 1 | MutateEverything_additive | 0.711 | +0.035 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure GRB2 Binding | σ=1 | 33441 | 1 | 0.623 | 7 | ESM3-small-open_additive | 0.740 | -0.117 | n/a | n/a | n/a | n/a | n/a: single protein |
| Faure GRB2 Binding | σ=0.5 | 33441 | 1 | 0.692 | 6 | ESM3-small-open_additive | 0.740 | -0.048 | n/a | n/a | n/a | n/a | n/a: single protein |
| Küng Myoglobin Display | σ=1 | 5905 | 1 | 0.636 | 1 | MutateEverything_additive | 0.586 | +0.051 | n/a | n/a | n/a | n/a | n/a: single protein |
| Küng Myoglobin Display | σ=0.5 | 5905 | 1 | 0.627 | 1 | MutateEverything_additive | 0.586 | +0.041 | n/a | n/a | n/a | n/a | n/a: single protein |
| Nutschel EstA ΔT_50 | σ=1 | 2172 | 1 | 0.517 | 2 | ProteinMPNN | 0.546 | -0.029 | n/a | n/a | n/a | n/a | n/a: single protein |
| Nutschel EstA ΔT_50 | σ=0.5 | 2172 | 1 | 0.542 | 2 | ProteinMPNN | 0.546 | -0.004 | n/a | n/a | n/a | n/a | n/a: single protein |

### Domainome (522 domains): N=2 (σ)

| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| Beltran Domainome Abundance (per-domain avg.) | σ=1 | 536145 | 522 | 0.548 [0.535, 0.560] | 4 | ESM3-small_additive | 0.551 [0.539, 0.563] | -0.004 [-0.010, +0.003] | 0.864 | 1.000 | 1.000 | 1.000 |  |
| Beltran Domainome Abundance (per-domain avg.) | σ=0.5 | 536145 | 522 | 0.573 [0.560, 0.585] | 1 | ESM3-small_additive | 0.551 [0.539, 0.563] | +0.022 [+0.017, +0.026] | 1.0e-04 | 2.0e-04 | 2.0e-04 | 2.0e-04 | * |

### Overall (macro-average of Δρ across datasets in the family)

Per family and σ: one test vs each dataset's own next-best model, plus one per method present in every dataset (Holm across methods). Adjusted p additionally multiplies by the number of σ values tested (`p adj`; `p adj (σ=3)` counts three σ values for the DMS family).

| Family | σ | Comparator | datasets won | mean Δρ [95% CI] | p | p adj | p adj (σ=3) |
|---|---|---|---:|---|---:|---:|---:|
| stability | σ=1 | best other (per dataset) | 6/10 | -0.004 [-0.036, +0.025] | 0.584 | 0.584 | 0.584 |
| stability | σ=1 | ThermoMPNN | 10/10 | +0.045 [+0.008, +0.080] | 0.011 | 0.032 | 0.032 |
| stability | σ=1 | MutateEverything_additive | 6/10 | +0.015 [-0.021, +0.044] | 0.199 | 0.199 | 0.199 |
| stability | σ=1 | ProteinMPNN | 10/10 | +0.158 [+0.100, +0.214] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | MutateEverything | 7/10 | +0.026 [-0.006, +0.057] | 0.051 | 0.102 | 0.102 |
| stability | σ=1 | Rosetta Cartesian DDG | 10/10 | +0.080 [+0.042, +0.121] | 3.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-medium | 10/10 | +0.222 [+0.174, +0.263] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-medium_additive | 10/10 | +0.220 [+0.172, +0.254] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small | 10/10 | +0.208 [+0.164, +0.249] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small_additive | 10/10 | +0.202 [+0.159, +0.241] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small-open | 10/10 | +0.261 [+0.213, +0.298] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-small-open_additive | 10/10 | +0.259 [+0.211, +0.292] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-large_additive | 10/10 | +0.357 [+0.293, +0.409] | 1.0e-04 | 0.001 | 0.001 |
| stability | σ=1 | ESM3-large | 10/10 | +0.370 [+0.302, +0.427] | 1.0e-04 | 0.001 | 0.001 |
| dms | σ=1 | best other (per dataset) | | n/a: single-protein datasets | | | |
| dms | σ=0.5 | best other (per dataset) | | n/a: single-protein datasets | | | |
