# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB + 95%-Cap Half-Step Probes & 40 GB Quadratic Fit)

## Estimated Maximum Protein Length ($B = 1$ PyTorch-Allocated VRAM, Final-3-Point Quadratic Fit)

| Model Family | Final 3 Empirical Points `(L, GB)` | Est. `L_max` @ 16 GB | Est. `L_max` @ 24 GB | Est. `L_max` @ 32 GB | Est. `L_max` @ 40 GB |
|---|---|---:|---:|---:|---:|
| MutateEverything (St1 / St2) | (512, 5.80), (1,024, 12.41), (1,536, 23.13) | 1,219 | 1,570 | 1,858 | 2,108 |
| ESM-MSR (all variants) | (512, 6.59), (1,024, 10.00), (2,048, 23.62) | 1,560 | 2,069 | 2,476 | 2,825 |
| ProSST-2048 | (2,048, 5.21), (4,096, 5.69), (8,192, 20.89) | 7,056 | 8,947 | 10,503 | 11,856 |
| SPURS (single / multi) | (4,096, 5.33), (8,192, 13.44), (12,288, 26.86) | 9,114 | 11,536 | 13,534 | 15,275 |
| ProteinMPNN | (8,192, 3.67), (16,384, 9.02), (24,576, 20.27) | 21,772 | 27,029 | 31,419 | 35,267 |
| ThermoMPNN / -D | (8,192, 4.74), (16,384, 10.97), (24,576, 18.69) | 21,898 | 29,450 | 36,023 | 41,920 |

## Full Mutational Screening Cost (RTX A4500 20 GB Fixed Slot + 6-Core CPU ESM-MSR WT)

| Model | Max `L` ok | First OOM `L` | Weights GB | VRAM/RAM @ `L=128` | VRAM/RAM @ `L=512` | VRAM/RAM @ `L=1024` | 1-mut @ `L=256` | 1-mut @ `L=1024` | 2-mut @ `L=256` | 2-mut @ `L=1024` |
|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR (WT-view) | 1,024 | 2048 | 5.44 | 5.60 GB | 6.59 GB | 10.00 GB | 178 ms | 1.1 s | — | — |
| ESM-MSR (WT-view, 6-core CPU) | 2,048 | 4096 (60G RAM) | — | 7.73 GB | 8.48 GB | 11.86 GB | 4.7 s | 19.6 s | — | — |
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
