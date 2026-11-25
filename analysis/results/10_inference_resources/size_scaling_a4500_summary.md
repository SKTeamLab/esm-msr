# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB + On-Card 30 GB Probes & 40 GB Extrapolation)

## Estimated Maximum Protein Length ($B = 1$ PyTorch-Allocated VRAM)

| Model Family | Highest On-Card Measured `L` | Measured VRAM | Est. `L_max` @ 16 GB | Est. `L_max` @ 24 GB | Est. `L_max` @ 32 GB | Est. `L_max` @ 40 GB |
|---|---:|---:|---:|---:|---:|---:|
| MutateEverything (St1 / St2) | 1,650 | 26.44 GB | 1,211 | 1,558 | 1,844 | 2,094 |
| ESM-MSR (all variants) | 2,200 | 26.40 GB | 1,560 | 2,070 | 2,477 | 2,826 |
| ProSST-2048 | 9,200 | 26.17 GB | 7,136 | 8,800 | 10,195 | 11,422 |
| SPURS (single / multi) | 12,800 | 28.90 GB | 9,118 | 11,541 | 13,536 | 15,273 |
| ProteinMPNN | 28,000 | 26.31 GB | 21,691 | 26,745 | 30,985 | 34,711 |
| ThermoMPNN / -D | 26,000 | 20.80 GB | 21,627 | 28,672 | 34,715 | 40,091 |

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
