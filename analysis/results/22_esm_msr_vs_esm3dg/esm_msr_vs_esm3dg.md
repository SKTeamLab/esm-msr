| Benchmark | Metric | n | ESM-MSR ρ [95% CI] | ESM3ΔG ρ [95% CI] | Δρ = ESM-MSR − ESM3ΔG [95% CI, paired] | ESM3ΔG overlap (leak audit) |
|---|---|---|---|---|---|---|
| **Thermostability sets** | | | | | | |
| Ssym | pooled rho | 684 | 0.778  [0.738, 0.818] | 0.735  [0.686, 0.781] | +0.044  [+0.013, +0.075] | h.72 |
| S461 | pooled rho | 461 | 0.768  [0.718, 0.810] | 0.682  [0.623, 0.733] | +0.086  [+0.044, +0.134] | h.64 |
| K2369 | pooled rho | 2,369 | 0.719  [0.695, 0.741] | 0.598  [0.570, 0.625] | +0.121  [+0.098, +0.143] | H.007 h.26 |
| S783 | pooled rho | 783 | 0.707  [0.665, 0.748] | 0.653  [0.603, 0.701] | +0.053  [+0.012, +0.098] | h.44 |
| S2648 | pooled rho | 2,648 | 0.664  [0.640, 0.690] | 0.642  [0.616, 0.667] | +0.022  [+0.000, +0.044] | H.02 h.33 |
| Q3421 | pooled rho | 3,421 | 0.618  [0.593, 0.644] | 0.572  [0.545, 0.597] | +0.046  [+0.024, +0.068] | H.004 h.37 |
| S8754 | pooled rho | 8,754 | 0.617  [0.601, 0.633] | 0.543  [0.524, 0.559] | +0.075  [+0.061, +0.088] | H.01 h.31 |
| S571 | pooled rho | 571 | 0.415  [0.342, 0.494] | 0.511  [0.435, 0.582] | -0.096  [-0.147, -0.043] | P.01 H.005 h.33 |
| PTMUL | pooled rho | 914 | 0.592  [0.545, 0.639] | 0.409  [0.355, 0.465] | +0.183  [+0.143, +0.226] | H.009 h.34 |
| PTMUL-D | pooled rho | 536 | 0.596  [0.527, 0.658] | 0.482  [0.407, 0.553] | +0.115  [+0.070, +0.161] | H.01 h.32 |
| **DMS sets** | | | | | | |
| DLG4 bind. | pooled rho | 8,251 | 0.734  [0.722, 0.747] | 0.766  [0.758, 0.774] | -0.032  [-0.041, -0.021] | – |
| GRB2 bind. | pooled rho | 33,441 | 0.631  [0.624, 0.639] | 0.646  [0.639, 0.652] | -0.014  [-0.019, -0.009] | – |
| MYO | pooled rho | 5,905 | 0.638  [0.620, 0.656] | 0.594  [0.575, 0.612] | +0.044  [+0.034, +0.054] | h1 |
| **Domainome** | | | | | | |
| Domainome (held-out domains) | mean per-domain rho | 436 | 0.540  [0.526, 0.553] | 0.501  [0.484, 0.519] | +0.038  [+0.026, +0.051] | P.005 H.22 h.51 |
| **ProteinGym (held-out assays)** | | | | | | |
| PG average | mean per-assay rho | 173 | 0.509  [0.476, 0.539] | 0.381  [0.340, 0.421] | +0.128  [+0.103, +0.154] | P.003 H.02 h.33 |
| Stability | mean per-assay rho | 37 | 0.827  [0.783, 0.866] | 0.788  [0.727, 0.834] | +0.039  [+0.017, +0.067] | P.003 H.02 h.33 |
| Expression | mean per-assay rho | 18 | 0.519  [0.430, 0.584] | 0.320  [0.219, 0.400] | +0.199  [+0.135, +0.263] | – |
| Activity | mean per-assay rho | 38 | 0.434  [0.384, 0.483] | 0.304  [0.254, 0.355] | +0.130  [+0.094, +0.165] | – |
| Binding | mean per-assay rho | 13 | 0.349  [0.233, 0.450] | 0.209  [0.052, 0.351] | +0.140  [+0.047, +0.226] | – |
| OrganismalFitness | mean per-assay rho | 67 | 0.415  [0.375, 0.455] | 0.285  [0.230, 0.336] | +0.130  [+0.096, +0.167] | – |
| **MGnify / Rocklin (not held out for ESM3ΔG)** | | | | | | |
| MGnify | pooled rho | 612,406 | 0.644  [0.640, 0.647] | 0.686  [0.682, 0.690] | -0.042  (separate intervals) | same source |
| Rocklin designs | pooled rho | 109,581 | 0.454  [0.427, 0.482] | 0.384  [0.355, 0.422] | +0.070  (separate intervals) | same source |
| MGnify | mean per-scaffold rho | 5,505 | 0.677  [0.670, 0.684] | 0.731  [0.725, 0.737] | -0.054  (separate intervals) | same source |
| Rocklin designs | mean per-scaffold rho | 618 | 0.273  [0.253, 0.294] | 0.362  [0.342, 0.384] | -0.089  (separate intervals) | same source |

ESM-MSR: deployed readout (WT-view on singles, Dual-view where multi-mutants are scored), seeds 1-3 averaged; ESM3ΔG: one run, direct prediction. Δρ intervals: paired bootstrap (B = 1,000, seeds resampled; mutation level for thermostability & DMS, domain level for Domainome, category-stratified assays × seeds for ProteinGym; MGnify & Rocklin: separate scaffold bootstraps). ESM3ΔG is fine-tuned on MGnify stability data (960,216 sequences), so MGnify & Rocklin are not held out for it. Green / red: Δρ interval excludes 0 (ESM-MSR higher / lower).
