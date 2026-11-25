# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB + 30/40 GB VRAM Extrapolation)

## Estimated Maximum Protein Length ($B = 1$ PyTorch-Allocated VRAM)

| Model Family | Measured Max `L` | VRAM @ Max `L` | A4500 OOM `L` | Est. `L_max` @ 20 GB | Est. `L_max` @ ~30 GB (32 GB usable) | Est. `L_max` @ 40 GB |
|---|---|---|---|---|---|---|
| MutateEverything (St1 / St2) | 1,024 | 12.41 GB | 2,048 | 1,442 | 1,881 | 2,249 |
| ESM-MSR (all variants) | 2,048 | 23.62 GB | 2,048 | 1,832 | 2,381 | 2,825 |
| ProSST-2048 | 4,096 | 5.69 GB | 8,192 | 8,057 | 9,928 | 11,499 |
| SPURS (single / multi) | 8,192 | 13.44 GB | — | 10,406 | 13,076 | 15,287 |
| ThermoMPNN / -D | 16,384 | 10.97 GB | 32,768 | 25,818 | 34,416 | 41,848 |
| ProteinMPNN | 16,384 | 9.02 GB | 32,768 | 28,292 | 36,581 | 43,613 |

## Full Mutational Screening Cost (RTX A4500 20 GB Fixed Slot)

| Model | Max `L` ok (A4500) | First OOM `L` | Weights GB | VRAM @ `L=128` | VRAM @ `L=512` | VRAM @ `L=1024` | 1-mut @ `L=256` | 1-mut @ `L=1024` | 2-mut @ `L=256` | 2-mut @ `L=1024` |
|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (WT-view) | 1,024 | 2048 | 5.44 | 5.60 GB | 6.59 GB | 10.00 GB | 178 ms | 1.1 s | — | — |
| ESM-MSR (Dual-view) | 1,024 | 2048 | 5.44 | 5.60 GB | 6.61 GB | 10.03 GB | 2.2 m | 1.1 h | 3.8 d (cap) | impractical (>24 h) |
| ESM-MSR (Indep. masking) | 1,024 | 2048 | 5.44 | 5.60 GB | 6.60 GB | 10.01 GB | 14.2 s | 7.0 m | 9.4 h | impractical (>24 h) |
| ESM-MSR (Masked marginal) | 1,024 | 2048 | 5.44 | 5.60 GB | 6.60 GB | 10.01 GB | 14.3 s | 7.0 m | 15.1 m | 1.2 d (cap) |
| SPURS (single) | 8,192 | >8192 | 2.47 | 2.54 GB | 2.74 GB | 3.02 GB | 63 ms | 253 ms | — | — |
| SPURS-multi | 8,192 | >8192 | 2.48 | 2.54 GB | 2.73 GB | 3.00 GB | 70 ms | 285 ms | 1.1 m | 27.2 m |
| ThermoMPNN (SSM single) | 16,384 | 32768 | 0.02 | 0.08 GB | 0.27 GB | 0.53 GB | 14 ms | 30 ms | — | — |
| ThermoMPNN-D (SSM double) | 16,384 | 32768 | 0.01 | 0.08 GB | 0.28 GB | 0.55 GB | — | — | 3.6 s | 1.0 m |
| ProteinMPNN (zero-shot) | 16,384 | 32768 | 0.02 | 0.09 GB | 0.28 GB | 0.54 GB | 398 ms | 6.2 s | 31.3 m | 1.4 d (cap) |
| MutateEverything (Stage 1) | 1,024 | 2048 | 0.37 | 0.90 GB | 5.80 GB | 12.41 GB | 12.8 s | 5.9 m | — | — |
| MutateEverything (Stage 2) | 1,024 | 2048 | 0.37 | 2.81 GB | 5.80 GB | 12.41 GB | 12.8 s | 5.9 m | 12.8 s | 5.9 m |
| ProSST-2048 (zero-shot) | 4,096 | 8192 | 0.47 | 0.68 GB | 1.34 GB | 1.65 GB | 519 ms | 4.9 s | — | — |
