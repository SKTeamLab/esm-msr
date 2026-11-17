| Model | val ρ | test ρ | test singles | test doubles | test median ρ (all variants) | test median ρ, WT-backbone singles (paper statistic) | Homology-Restricted test ρ | P(ESM-MSR comb > model), test |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR combined | 0.868 [0.836, 0.896] | 0.827 [0.798, 0.869] | 0.830 | 0.662 | 0.873 | 0.860 | 0.845 |  |
| ESM-MSR WT head | 0.845 [0.796, 0.873] | 0.812 [0.780, 0.857] | 0.821 | 0.569 | 0.844 | 0.846 | 0.827 | 1.00 |
| SPURS, trained adapter (retrain) | 0.812 [0.760, 0.842] | 0.800 [0.771, 0.839] | 0.796 | 0.570 | 0.819 | 0.812 | – | 1.00 |
| SPURS, frozen adapter (retrain) | 0.813 [0.763, 0.841] | 0.799 [0.771, 0.839] | 0.796 | 0.567 | 0.823 | 0.817 | 0.801 | 1.00 |
| SPURS, published checkpoint | 0.813 [0.761, 0.845] | 0.794 [0.762, 0.830] | 0.798 | 0.543 | 0.821 | 0.827 | – | 1.00 |
| ThermoMPNN + ThermoMPNN-D | 0.788 [0.757, 0.818] | 0.779 [0.755, 0.813] | 0.769 | 0.615 | 0.782 | 0.773 | 0.784 | 1.00 |
| ThermoMPNN | 0.782 [0.744, 0.809] | 0.777 [0.749, 0.813] | 0.769 | 0.557 | 0.782 | 0.773 | 0.794 | 1.00 |
| ThermoMPNN, published (3 ensemble members) | 0.787 [0.748, 0.814] | 0.777 [0.748, 0.812] | 0.765 | 0.564 | 0.790 | 0.766 | – | 1.00 |
| ThermoMPNN-D, published (3 ensemble members) | 0.792 [0.758, 0.819] | 0.775 [0.751, 0.809] | 0.765 | 0.611 | 0.782 | 0.766 | – | 1.00 |
| MutateEverything st1 | 0.790 [0.741, 0.830] | 0.774 [0.746, 0.811] | 0.775 | 0.509 | 0.782 | 0.779 | 0.771 | 1.00 |
| SPURS multi, frozen adapter (retrain) | 0.768 [0.726, 0.798] | 0.771 [0.742, 0.800] | 0.738 | 0.598 | 0.797 | 0.753 | 0.772 | 1.00 |
| MutateEverything st2 | 0.781 [0.745, 0.815] | 0.767 [0.743, 0.805] | 0.758 | 0.595 | 0.773 | 0.769 | 0.760 | 1.00 |
| SPURS multi, published checkpoint | 0.765 [0.723, 0.800] | 0.762 [0.737, 0.796] | 0.726 | 0.617 | 0.788 | 0.740 | – | 1.00 |
| SPURS multi, trained adapter (retrain) | 0.755 [0.715, 0.789] | 0.761 [0.738, 0.796] | 0.721 | 0.612 | 0.793 | 0.736 | – | 1.00 |

Medians: median over the test libraries of the per-library ρ (per seed, then the mean over seeds; the seed range is in `tsuboyama_mega_test_medians.csv`). The paper statistic is the SPURS paper's: median per protein over the wild-type-backbone singles (SPURS 0.83, ThermoMPNN 0.77 as reported there).
Mean per-library Spearman ρ, seed mean of three seeds per model; [95% interval] from resampling libraries and seeds (B = 50). mega_splits test: 28 libraries, 49,048 measured variants.
