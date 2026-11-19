| Benchmark | Metric | n | ESM-MSR ρ [95% CI] | ESM3ΔG ρ [95% CI] | Δρ = ESM-MSR − ESM3ΔG [95% CI, paired] | ESM3ΔG overlap (leak audit) |
|---|---|---|---|---|---|---|
| **Thermostability sets** | | | | | | |
| Ssym | pooled rho | 684 | 0.778  [0.733, 0.810] | 0.735  [0.685, 0.776] | +0.044  [+0.013, +0.076] | h.72 |
| S461 | pooled rho | 461 | 0.768  [0.716, 0.811] | 0.682  [0.622, 0.733] | +0.086  [+0.042, +0.134] | h.64 |
| K2369 | pooled rho | 2,369 | 0.719  [0.694, 0.741] | 0.598  [0.573, 0.625] | +0.121  [+0.097, +0.144] | H.007 h.26 |
| S783 | pooled rho | 783 | 0.707  [0.666, 0.746] | 0.653  [0.603, 0.702] | +0.053  [+0.016, +0.096] | h.44 |
| S2648 | pooled rho | 2,648 | 0.664  [0.639, 0.690] | 0.642  [0.619, 0.665] | +0.022  [-0.001, +0.045] | H.02 h.33 |
| Q3421 | pooled rho | 3,421 | 0.618  [0.596, 0.643] | 0.572  [0.546, 0.596] | +0.046  [+0.024, +0.066] | H.004 h.37 |
| S8754 | pooled rho | 8,754 | 0.617  [0.600, 0.633] | 0.543  [0.524, 0.557] | +0.075  [+0.061, +0.088] | H.01 h.31 |
| S571 | pooled rho | 571 | 0.415  [0.348, 0.501] | 0.511  [0.435, 0.582] | -0.096  [-0.144, -0.041] | P.01 H.005 h.33 |
| PTMUL | pooled rho | 914 | 0.592  [0.545, 0.633] | 0.409  [0.359, 0.465] | +0.183  [+0.142, +0.220] | H.009 h.34 |
| PTMUL-D | pooled rho | 536 | 0.596  [0.526, 0.651] | 0.482  [0.404, 0.546] | +0.115  [+0.072, +0.160] | H.01 h.32 |
| **DMS sets** | | | | | | |
| DLG4 bind. | pooled rho | 8,251 | 0.734  [0.723, 0.746] | 0.766  [0.759, 0.774] | -0.032  [-0.041, -0.023] | – |
| GRB2 bind. | pooled rho | 33,441 | 0.631  [0.624, 0.638] | 0.646  [0.639, 0.653] | -0.014  [-0.019, -0.010] | – |
| MYO | pooled rho | 5,905 | 0.638  [0.618, 0.651] | 0.594  [0.573, 0.609] | +0.044  [+0.035, +0.053] | h1 |
| **Domainome** | | | | | | |
| Domainome (held-out domains) | mean per-domain rho | 436 | 0.540  [0.526, 0.553] | 0.501  [0.484, 0.519] | +0.038  [+0.027, +0.051] | P.005 H.22 h.51 |
| **ProteinGym (held-out assays, no Stability)** | | | | | | |
| PG average (4 categories) | mean per-assay rho | 173 | 0.429  [0.395, 0.461] | 0.280  [0.238, 0.328] | +0.150  [+0.119, +0.178] | P.003 H.02 h.33 |
| Expression | mean per-assay rho | 18 | 0.519  [0.441, 0.594] | 0.320  [0.235, 0.403] | +0.199  [+0.140, +0.257] | – |
| Activity | mean per-assay rho | 38 | 0.434  [0.383, 0.489] | 0.304  [0.254, 0.360] | +0.130  [+0.098, +0.170] | – |
| Binding | mean per-assay rho | 13 | 0.349  [0.235, 0.450] | 0.209  [0.067, 0.332] | +0.140  [+0.060, +0.220] | – |
| OrganismalFitness | mean per-assay rho | 67 | 0.415  [0.378, 0.454] | 0.285  [0.234, 0.336] | +0.130  [+0.105, +0.163] | – |
| **MGnify / Rocklin (not held out)** | | | | | | |
| MGnify | pooled rho | 612,406 | 0.644  [0.640, 0.647] | 0.686  [0.683, 0.690] | -0.042  (separate intervals) | same source |
| Rocklin designs | pooled rho | 109,581 | 0.454  [0.416, 0.481] | 0.384  [0.347, 0.416] | +0.070  (separate intervals) | same source |
| MGnify | mean per-scaffold rho | 5,505 | 0.677  [0.670, 0.684] | 0.731  [0.725, 0.737] | -0.054  (separate intervals) | same source |
| Rocklin designs | mean per-scaffold rho | 618 | 0.273  [0.255, 0.293] | 0.362  [0.341, 0.383] | -0.089  (separate intervals) | same source |

ESM-MSR: deployed readout (WT-view on singles, Dual-view where multi-mutants are scored), seeds 1-3 averaged; ESM3ΔG: one run, direct prediction. Both on the same rows. Δρ intervals: paired bootstrap (draws shared by both models, seeds resampled; mutation level for the thermostability and DMS sets, domain level for Domainome, assays within category for ProteinGym (no Stability: Megascale overlap); MGnify and Rocklin: separate scaffold bootstraps). ESM3ΔG is not scored on Megascale (Tsuboyama) data, which its training data overlap. It is fine-tuned on MGnify stability data (960,216 sequences): MGnify and Rocklin are not held out for it. Overlap = the leak audit's share of the set's rows whose protein has a homolog in that training data (P same protein, H strong homolog, h weak homolog), before the four-part held-out filter. Green / red: Δρ interval excludes 0 (ESM-MSR higher / lower).
