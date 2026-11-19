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
| **Domainome, MegaScale val / test** | | | | | | |
| Domainome (held-out domains) | mean per-domain rho | 436 | 0.540  [0.525, 0.554] | 0.501  [0.483, 0.519] | +0.038  [+0.027, +0.051] | P.005 H.22 h.51 |
| Tsuboyama val | mean per-library rho | 23 | 0.853  [0.825, 0.881] | 0.811  [0.772, 0.845] | +0.043  [+0.026, +0.063] | P.19 H.30 h.24 |
| Tsuboyama test | mean per-library rho | 33 | 0.845  [0.807, 0.876] | 0.811  [0.770, 0.843] | +0.034  [+0.004, +0.067] | H.41 h.50 |
| Tsuboyama test, singles | mean per-library rho | 33 | 0.833  [0.803, 0.860] | 0.814  [0.786, 0.838] | +0.018  [-0.010, +0.043] | H.41 h.50 |
| Tsuboyama test, doubles | mean per-library rho | 13 | 0.777  [0.687, 0.829] | 0.745  [0.631, 0.826] | +0.032  [-0.050, +0.141] | H.41 h.50 |
| **ProteinGym (held-out assays)** | | | | | | |
| PG average | mean per-assay rho | 173 | 0.509  [0.477, 0.539] | 0.381  [0.343, 0.421] | +0.128  [+0.102, +0.151] | P.003 H.02 h.33 |
| Stability | mean per-assay rho | 37 | 0.827  [0.781, 0.862] | 0.788  [0.734, 0.835] | +0.039  [+0.019, +0.062] | P.003 H.02 h.33 |
| Expression | mean per-assay rho | 18 | 0.519  [0.437, 0.582] | 0.320  [0.226, 0.401] | +0.199  [+0.137, +0.261] | – |
| Activity | mean per-assay rho | 38 | 0.434  [0.386, 0.485] | 0.304  [0.255, 0.359] | +0.130  [+0.094, +0.166] | – |
| Binding | mean per-assay rho | 13 | 0.349  [0.231, 0.454] | 0.209  [0.064, 0.356] | +0.140  [+0.045, +0.234] | – |
| OrganismalFitness | mean per-assay rho | 67 | 0.415  [0.371, 0.458] | 0.285  [0.225, 0.335] | +0.130  [+0.097, +0.164] | – |
| **MGnify / Rocklin (not held out)** | | | | | | |
| MGnify | pooled rho | 612,406 | 0.644  [0.640, 0.647] | 0.686  [0.683, 0.690] | -0.042  (separate intervals) | same source |
| Rocklin designs | pooled rho | 109,581 | 0.454  [0.416, 0.481] | 0.384  [0.347, 0.416] | +0.070  (separate intervals) | same source |
| MGnify | mean per-scaffold rho | 5,505 | 0.677  [0.670, 0.684] | 0.731  [0.725, 0.737] | -0.054  (separate intervals) | same source |
| Rocklin designs | mean per-scaffold rho | 618 | 0.273  [0.255, 0.293] | 0.362  [0.341, 0.383] | -0.089  (separate intervals) | same source |

ESM-MSR: deployed readout (WT-view on singles, Dual-view where multi-mutants are scored), seeds 1-3 averaged; ESM3ΔG: one run, direct prediction. Both on the same rows. Δρ intervals: paired bootstrap (draws shared by both models, seeds resampled; mutation level for the thermostability and DMS sets, domain level for Domainome and the MegaScale sets, assays within category for ProteinGym; MGnify and Rocklin: separate scaffold bootstraps). ESM3ΔG is fine-tuned on MGnify stability data (960,216 sequences, not Megascale): MGnify and Rocklin are not held out for it. Overlap = the leak audit's share of the set's rows whose protein has a homolog in that training data (P same protein, H strong homolog, h weak homolog), before the four-part held-out filter. Green / red: Δρ interval excludes 0 (ESM-MSR higher / lower).
