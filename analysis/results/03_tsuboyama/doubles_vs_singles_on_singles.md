| Model | Singles version → trained with doubles | ρ singles | ρ trained with doubles | Δρ [95% CI] | Domains lower | p | p (Holm) | Doubles-trained version |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR p0 | WT head → combined head | 0.833 | 0.846 | +0.014  [+0.008, +0.020] | 3 / 33 | < 1e-5 | 3.0e-05 | higher, *** |
| MutateEverything | stage 1 → stage 2 | 0.771 | 0.756 | -0.015  [-0.023, -0.008] | 28 / 33 | 6.0e-05 | 1.2e-04 | lower, *** |
| ThermoMPNN | ThermoMPNN → ThermoMPNN-D | 0.786 | 0.786 | identical predictions | – | – | – | same network on singles, not tested |
| SPURS (trained adapter) | SPURS single → SPURS multi | 0.804 | 0.761 | -0.043  [-0.062, -0.022] | 29 / 33 | 2.2e-04 | 2.2e-04 | lower, *** |

Tsuboyama capped test set, single mutants only. ρ = mean over test domains (≥ 10 singles) of the per-domain Spearman ρ, seeds averaged; Δρ = doubles-trained − singles. Paired bootstrap over domains (100,000 draws, shared by all models); p two-sided; Holm over the tested models (ThermoMPNN is not tested: ThermoMPNN-D gives the same single-mutant predictions as ThermoMPNN).
