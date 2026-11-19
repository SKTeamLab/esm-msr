| Model | Additive → epistatic | NDCG@96 additive | NDCG@96 epistatic | ΔNDCG@96 [95% CI] | Domains epistatic higher | p one-sided | p one-sided (Holm) | p two-sided |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR p0 | WT_A + WT_B → combined head | 0.895 | 0.911 | +0.016  [-0.004, +0.033] | 11 / 13 | 0.050 | 0.397 | 0.099 |
| MutateEverything | stage 1 → stage 2 | 0.865 | 0.882 | +0.017  [-0.003, +0.044] | 9 / 13 | 0.053 | 0.397 | 0.106 |
| ThermoMPNN | ThermoMPNN → ThermoMPNN-D | 0.868 | 0.852 | -0.016  [-0.046, +0.010] | 6 / 13 | 0.873 | 1.000 | 0.255 |
| SPURS (trained adapter) | SPURS single → SPURS multi | 0.903 | 0.862 | -0.040  [-0.064, -0.020] | 3 / 13 | 1.000 | 1.000 | 8.0e-05 |
| ESM3-small-open | WT_A + WT_B → combined head | 0.784 | 0.769 | -0.015  [-0.026, -0.005] | 2 / 13 | 0.999 | 1.000 | 0.002 |
| ProteinMPNN | sum of singles → direct double | 0.808 | 0.795 | -0.013  [-0.033, +0.003] | 6 / 13 | 0.935 | 1.000 | 0.129 |
| GeoDDG-3D | sum of singles → direct double | 0.788 | 0.782 | -0.006  [-0.019, +0.008] | 4 / 13 | 0.802 | 1.000 | 0.397 |
| Rosetta | sum of singles → direct double | 0.779 | 0.782 | +0.004  [-0.015, +0.019] | 8 / 13 | 0.318 | 1.000 | 0.637 |
