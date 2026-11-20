| ESM-MSR (Tobit-linked) against | ρ ESM-MSR | ρ other | Δρ [95% CI] mutation resampling | p one-sided (Holm) | p two-sided (Holm) | Δρ [95% CI] domain resampling | p one-sided (Holm) | p two-sided (Holm) |
|---|---|---|---|---|---|---|---|---|
| MutateEverything (Stage 1 & Stage 2) | 0.757 | 0.768 | -0.011  [-0.020, -0.004] | 1.00 | 0.003 | -0.011  [-0.111, +0.077] | 1.00 | 1.00 |
| ThermoMPNN (& ThermoMPNN-D) | 0.757 | 0.747 | +0.011  [+0.005, +0.017] | 0.003 | 0.003 | +0.011  [-0.057, +0.076] | 0.92 | 1.00 |
| SPURS (& SPURS-multi) | 0.757 | 0.782 | -0.025  [-0.045, +0.005] | 1.00 | 0.11 | -0.025  [-0.127, +0.066] | 1.00 | 1.00 |

Tsuboyama capped test set, all variants, each model read as deployed. ρ: one Spearman ρ over all test variants, seeds averaged; ESM-MSR with the post hoc Tobit link h(dG_wt + latent) − dG_wt, the other models raw. Δρ = ESM-MSR − other. Joint bootstrap of 1,000 draws shared by all models (resampling triplicate seeds jointly): mutation resampling resamples rows inside every domain (domain composition fixed); domain resampling resamples whole domains. One-sided p = P(Δρ ≤ 0); Holm over the three comparisons.
