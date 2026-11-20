| Model | Singles version → trained with doubles | ρ singles | ρ trained with doubles | Δρ [95% CI] | Domains lower | p | p (Holm) | Doubles-trained version |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR | ESM-MSR WT-view → ESM-MSR Dual-view | 0.833 | 0.846 | +0.014  [+0.008, +0.020] | 3 / 33 | < 1.0e-04 | < 0.001 | higher, *** |
| MutateEverything | MutateEverything Stage 1 → MutateEverything Stage 2 | 0.771 | 0.756 | -0.015  [-0.023, -0.008] | 28 / 33 | < 1.0e-04 | < 0.001 | lower, *** |
| ThermoMPNN | ThermoMPNN → ThermoMPNN-D | 0.786 | 0.786 | identical predictions | – | – | – | same network on singles, not tested |
| SPURS | SPURS → SPURS-multi | 0.804 | 0.761 | -0.043  [-0.062, -0.022] | 29 / 33 | < 1.0e-04 | < 0.001 | lower, *** |

Tsuboyama capped test set, single mutants. ρ: mean over test domains (≥ 10 singles) of the per-domain Spearman ρ, seeds averaged; Δρ = trained with doubles − singles version. p: two-sided, paired bootstrap over domains and seeds (10,000 draws shared by all models); Holm over the tested models (ThermoMPNN-D scores singles with the ThermoMPNN network, so it is not tested). Red: significantly lower; green: significantly higher (p < 0.05).
