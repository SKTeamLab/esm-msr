| ESM-MSR against | ρ ESM-MSR | ρ other | Δρ [95% CI] | Domains where ESM-MSR is higher | p one-sided | p one-sided (Holm) | p two-sided | p two-sided (Holm) |
|---|---|---|---|---|---|---|---|---|
| MutateEverything (Stage 1 & Stage 2) | 0.832 | 0.770 | +0.062  [+0.041, +0.081] | 32 / 33 | < 1.0e-04 | < 0.001 | < 1.0e-04 | < 0.001 |
| ThermoMPNN (& ThermoMPNN-D) | 0.832 | 0.784 | +0.048  [+0.031, +0.064] | 28 / 33 | < 1.0e-04 | < 0.001 | < 1.0e-04 | < 0.001 |
| SPURS (& SPURS-multi) | 0.832 | 0.801 | +0.031  [+0.010, +0.051] | 30 / 33 | 0.002 | 0.002 | 0.005 | 0.005 |

Tsuboyama capped test set, all variants (singles and doubles), each model read as deployed (singles version on singles, doubles-trained version on doubles). ρ: mean over test domains of the per-domain Spearman ρ, seeds averaged; Δρ = ESM-MSR − other. Paired bootstrap over domains and seeds (10,000 draws shared by all models). One-sided p = P(Δρ ≤ 0); two-sided p = twice the smaller tail. Holm over the three comparisons.
