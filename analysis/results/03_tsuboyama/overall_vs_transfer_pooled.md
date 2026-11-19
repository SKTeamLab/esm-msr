| ESM-MSR p0 (Tobit-linked) against | ρ ESM-MSR | ρ other | Δρ [95% CI] mutation resampling | p one-sided (Holm) | p two-sided (Holm) | Δρ [95% CI] domain resampling | p one-sided (Holm) | p two-sided (Holm) |
|---|---|---|---|---|---|---|---|---|
| MutateEverything | 0.757 | 0.768 | -0.011  [-0.015, -0.007] | 1.000 | 0.003 | -0.011  [-0.104, +0.077] | 1.000 | 1.000 |
| ThermoMPNN(-D) | 0.757 | 0.747 | +0.011  [+0.006, +0.016] | 0.003 | 0.003 | +0.011  [-0.053, +0.077] | 0.867 | 1.000 |
| SPURS(-multi) | 0.757 | 0.782 | -0.025  [-0.029, -0.021] | 1.000 | 0.003 | -0.025  [-0.116, +0.061] | 1.000 | 1.000 |

Tsuboyama capped test set, all variants, each model read as deployed. ρ: one Spearman ρ over all test variants, seeds averaged; ESM-MSR p0 with the post hoc Tobit link h(dG_wt + latent) − dG_wt, the other models raw. Δρ = ESM-MSR − other. Bootstrap of 1,000 draws shared by the models: mutation resampling resamples the rows inside every domain (domain composition fixed); domain resampling resamples whole domains. One-sided p = P(Δρ ≤ 0); Holm over the three comparisons.
