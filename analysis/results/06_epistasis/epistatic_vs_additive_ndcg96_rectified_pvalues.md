| Model | Additive → epistatic | NDCG@96 additive | NDCG@96 epistatic | ΔNDCG@96 [95% CI] | Domains epistatic higher | p one-sided | p one-sided (Holm) | p two-sided |
|---|---|---|---|---|---|---|---|---|
| ESM-MSR p0 | WT_A + WT_B → combined head | 0.605 | 0.714 | +0.108  [+0.005, +0.277] | 6 / 8 | 0.017 | 0.135 | 0.034 |
| MutateEverything | stage 1 → stage 2 | 0.501 | 0.506 | +0.005  [-0.126, +0.157] | 4 / 8 | 0.495 | 1.000 | 0.990 |
| ThermoMPNN | ThermoMPNN → ThermoMPNN-D | 0.536 | 0.431 | -0.106  [-0.314, +0.062] | 3 / 8 | 0.871 | 1.000 | 0.258 |
| SPURS (trained adapter) | SPURS single → SPURS multi | 0.607 | 0.433 | -0.175  [-0.377, -0.036] | 1 / 8 | 0.997 | 1.000 | 0.006 |
| ESM3-small-open | WT_A + WT_B → combined head | 0.385 | 0.395 | +0.010  [-0.032, +0.060] | 4 / 8 | 0.350 | 1.000 | 0.700 |
| ProteinMPNN | sum of singles → direct double | 0.480 | 0.504 | +0.023  [-0.077, +0.142] | 3 / 8 | 0.352 | 1.000 | 0.704 |
| GeoDDG-3D | sum of singles → direct double | 0.414 | 0.401 | -0.013  [-0.032, +0.006] | 3 / 8 | 0.911 | 1.000 | 0.178 |
| Rosetta | sum of singles → direct double | 0.293 | 0.290 | -0.003  [-0.072, +0.060] | 3 / 8 | 0.523 | 1.000 | 0.955 |
