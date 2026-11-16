| Model | val ρ | test ρ | test singles | test doubles | Homology-Restricted test ρ | P(ESM-MSR comb > model), test |
|---|---|---|---|---|---|---|
| ESM-MSR combined | 0.868 [0.833, 0.894] | 0.827 [0.779, 0.865] | 0.830 | 0.662 | 0.845 |  |
| ESM-MSR WT head | 0.845 [0.802, 0.874] | 0.812 [0.768, 0.849] | 0.821 | 0.569 | 0.827 | 1.00 |
| SPURS | 0.813 [0.761, 0.846] | 0.799 [0.759, 0.834] | 0.796 | 0.567 | 0.801 | 1.00 |
| ThermoMPNN + ThermoMPNN-D | 0.788 [0.748, 0.815] | 0.779 [0.739, 0.810] | 0.769 | 0.615 | 0.784 | 1.00 |
| ThermoMPNN | 0.782 [0.738, 0.810] | 0.777 [0.736, 0.811] | 0.769 | 0.557 | 0.794 | 1.00 |
| MutateEverything st1 | 0.790 [0.745, 0.831] | 0.774 [0.742, 0.811] | 0.775 | 0.509 | 0.771 | 1.00 |
| SPURS multi | 0.768 [0.726, 0.800] | 0.771 [0.735, 0.805] | 0.738 | 0.598 | 0.772 | 1.00 |
| MutateEverything st2 | 0.781 [0.747, 0.816] | 0.767 [0.728, 0.801] | 0.758 | 0.595 | 0.760 | 1.00 |

Mean per-library Spearman ρ, seed mean of three seeds per model; [95% interval] from resampling libraries and seeds (B = 200). mega_splits test: 28 libraries, 49,048 measured variants.
