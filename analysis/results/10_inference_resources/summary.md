| Model | Variant | Seed | GPU | Inference h | Proteins (failed) | Peak RAM GB | Peak GPU GB | Tsuboyama h | External h | DMS h | Domainome h | ProteinGym h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR | baseline recipe | 1 | RTX A4500 | 35.17 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.17 | 0.42 | 0.94 | 33.51 |
| ESM-MSR | dc8 (devel) | 1 | RTX A4500 | 34.82 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.17 | 0.41 | 0.93 | 33.18 |
| ESM-MSR | released checkpoint | — | RTX A4500 | 34.77 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.17 | 0.41 | 0.92 | 33.14 |
| ESM-MSR | fl0 (PG WT-only) | 1 | RTX A4500 | 1.67 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.17 | 0.41 | 0.93 | 0.03 |
| ESM-MSR | fl0 (PG WT-only) | 2 | RTX A4500 | 1.68 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.17 | 0.41 | 0.94 | 0.03 |
| ESM-MSR | fl0 (PG WT-only) | 3 | RTX A4500 | 1.68 | 2538 (4) | 7.1 | 19.5 | 0.13 | 0.18 | 0.41 | 0.93 | 0.03 |
| ESM-MSR | FINAL_p0, WT head only (all sets) | 1 | RTX A4500 | 0.10 | 2538 (4) | 7.1 | 19.5 | 0.00 | 0.06 | 0.00 | 0.01 | 0.03 |
| MutateEverything | stage 1 | 1 | Quadro RTX 6000 | 8.10 | 2223 (1) | 15.4 | 23.1 | 0.06 | 3.03 | 0.01 | 0.32 | 4.68 |
| MutateEverything | stage 1 | 1 | RTX A4500 | 7.25 | 2222 (2) | 10.2 | 19.7 | 0.04 | 3.04 | 0.01 | 0.23 | 3.93 |
| MutateEverything | stage 2 | 1 | RTX A4500 | 7.53 | 2312 (2) | 11.7 | 19.7 | 0.04 | 3.24 | 0.01 | 0.23 | 4.00 |
| MutateEverything | stage 2 | 2 | RTX A4500 | 7.53 | 2312 (2) | 12.2 | 19.7 | 0.04 | 3.24 | 0.01 | 0.23 | 4.00 |
| MutateEverything | stage 2 | 3 | Quadro RTX 6000 | 7.81 | 2312 (2) | 13.3 | 23.6 | 0.06 | 3.18 | 0.02 | 0.32 | 4.22 |
| ThermoMPNN | single + ThermoMPNN-D | 1 | RTX A4500 | 6.56 | 2875 (0) | 15.2 | 18.2 | 0.09 | 0.03 | 0.45 | 0.30 | 5.69 |
| SPURS | retrained (single + multi) | 1 | RTX A4500 | 0.29 | 2542 (0) | 9.6 | 10.0 | 0.00 | 0.06 | 0.00 | 0.02 | 0.20 |
| SPURS | retrained (single + multi) | 2 | RTX A4500 | 0.29 | 2542 (0) | 9.6 | 10.0 | 0.00 | 0.06 | 0.00 | 0.02 | 0.20 |
| SPURS | retrained (single + multi) | 3 | RTX A4500 | 0.29 | 2542 (0) | 9.7 | 10.0 | 0.00 | 0.06 | 0.00 | 0.02 | 0.20 |
| SPURS † | published checkpoint | — | RTX A4500 | 0.29 | 2542 (0) | 7.2 | 10.0 | 0.00 | 0.06 | 0.00 | 0.02 | 0.20 |
| ProteinMPNN † | zero-shot | — | RTX A4500 | 1.81 | 2542 (0) | 4.5 | 17.7 | 0.00 | 0.02 | 0.01 | 0.01 | 1.77 |

† not transfer-learned by us (published checkpoint / zero-shot).
