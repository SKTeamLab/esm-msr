| Ablation | Change from p0 | Seeds | Tsuboyama test | Tsuboyama doubles | Epistasis beyond ρ | Epistasis cell mag. | External singles (10) | PTMUL / PTMUL-D | Domainome low-id | ProteinGym avg |
|---|---|---|---|---|---|---|---|---|---|---|
| **p0 (reference)** | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.845 | 0.777 | 0.487 | 0.212 | 0.654 | 0.594 | 0.540 | 0.509‡ |
| *Masking* | | | | | | | | | | |
| p1 masked | block_mask_wt/mt 0 -> 1 | 1,2,3 | **−0.010** | **−0.030** | +0.015 | +0.020 | −0.007 | −0.025 | +0.001 | −0.004 |
| p1 MT marginal | p1 + MT pass marginal | 1,2,3 | **−0.009** | **−0.056** | −0.056 | **−0.225** | **−0.010** | −0.034 | +0.001 | — |
| *Loss* | | | | | | | | | | |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | −0.002 | −0.009 | −0.071 | +0.003 | +0.001 | +0.004 | −0.001 | −0.001 |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | −0.005 | −0.012 | −0.045 | −0.016 | +0.003 | 0.000 | +0.002 | −0.003 |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | −0.001 | −0.005 | −0.051 | +0.007 | −0.001 | +0.006 | +0.001 | +0.001 |
| *Architecture* | | | | | | | | | | |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | −0.003 | −0.015 | −0.021 | −0.003 | +0.001 | −0.007 | +0.005 | 0.000 |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | **−0.008** | −0.025 | **−0.063** | −0.005 | −0.001 | −0.002 | +0.002 | −0.001 |
| *Cross-masking (evaluation only)* | | | | | | | | | | |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | — | — | — | — | — | — | — | — | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — | — | — | — | — |

p0 row: Spearman ρ (seed mean; seeds listed). Other rows: Δρ = family seed mean − p0, deployed readout (WT head on singles-only sets, 1/2(MT + WT) where multi-mutants are scored). Bold: |Δ| > 2 SE, where SE combines seed noise (family SD, or the pooled seed SD of the column for a single-seed run) and the paired seed-1 bootstrap over proteins / sites / assays. Epistasis columns: comb(AB) − WT(A) − WT(B) against measured ΔΔΔG on the 12,455 Tsuboyama test doubles with both singles measured (esm_msr.epi_metrics; beyond ρ = Spearman above the global saturation curve, cell mag. = double-centred residuals within complete blocks of each position pair), sampling noise by a paired bootstrap over the 13 test domains. ‡: fewer seeds than the rest of the row (the cell's seeds are in table_main_delta.csv). —: not yet run.
