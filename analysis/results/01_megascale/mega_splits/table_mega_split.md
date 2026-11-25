| Model | val ρ | test ρ | test singles | test doubles | test median ρ (all variants) | test median ρ, WT-backbone singles (paper statistic) | Homology-Restricted test ρ | P(ESM-MSR comb > model), test |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR combined | 0.868 [0.836, 0.894] | 0.827 [0.776, 0.868] | 0.830 | 0.662 | 0.873 | 0.860 | 0.845 |  |
| ESM-MSR WT head | 0.845 [0.799, 0.877] | 0.812 [0.765, 0.852] | 0.821 | 0.569 | 0.844 | 0.846 | 0.827 | 1.00 |
| SPURS, trained adapter (retrain) | 0.812 [0.758, 0.849] | 0.800 [0.759, 0.838] | 0.796 | 0.570 | 0.819 | 0.812 | – | 1.00 |
| SPURS, frozen adapter (retrain) | 0.813 [0.761, 0.849] | 0.799 [0.758, 0.839] | 0.796 | 0.567 | 0.823 | 0.817 | 0.800 | 1.00 |
| ThermoMPNN-D | 0.788 [0.750, 0.819] | 0.779 [0.739, 0.815] | 0.769 | 0.615 | 0.782 | 0.773 | 0.784 | 1.00 |
| ThermoMPNN | 0.782 [0.740, 0.815] | 0.777 [0.737, 0.813] | 0.769 | 0.557 | 0.782 | 0.773 | 0.794 | 1.00 |
| ThermoMPNN, published (3 ensemble members) | 0.787 [0.746, 0.818] | 0.777 [0.738, 0.812] | 0.765 | 0.564 | 0.790 | 0.766 | – | 1.00 |
| ThermoMPNN-D, published (3 ensemble members) | 0.792 [0.753, 0.823] | 0.775 [0.734, 0.812] | 0.765 | 0.611 | 0.782 | 0.766 | – | 1.00 |
| MutateEverything Stage 1 | 0.790 [0.740, 0.829] | 0.774 [0.735, 0.810] | 0.775 | 0.509 | 0.782 | 0.779 | 0.771 | 1.00 |
| SPURS multi, frozen adapter (retrain) | 0.768 [0.728, 0.806] | 0.771 [0.733, 0.807] | 0.738 | 0.598 | 0.797 | 0.753 | 0.775 | 1.00 |
| MutateEverything Stage 2 | 0.781 [0.744, 0.815] | 0.767 [0.728, 0.802] | 0.758 | 0.595 | 0.773 | 0.769 | 0.760 | 1.00 |
| SPURS-multi, trained adapter (retrain) | 0.755 [0.718, 0.786] | 0.761 [0.723, 0.796] | 0.721 | 0.612 | 0.793 | 0.736 | – | 1.00 |

Mean per-library Spearman ρ, seed mean of three seeds per model; [95% interval] from resampling libraries and seeds (B = 1,000). mega_splits test: 28 libraries, 49,048 measured variants.
