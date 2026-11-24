# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB)

| Model | Max L ok | First OOM L | Weights GB | VRAM @ L=128 | VRAM @ L=512 | VRAM @ L=1024 | 1-mut @ L=256 | 1-mut @ L=1024 | 2-mut @ L=256 | 2-mut @ L=1024 |
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
