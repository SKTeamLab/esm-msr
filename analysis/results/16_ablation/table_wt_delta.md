| Ablation | Change from p0 | Seeds | Tsuboyama test | Tsuboyama singles | Tsuboyama doubles | External singles (10) | PTMUL / PTMUL-D | Domainome low-id | ProteinGym avg |
|---|---|---|---|---|---|---|---|---|---|
| **p0 (reference)** | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.827 | 0.833 | 0.723 | 0.654 | 0.587 | 0.540 | 0.509‡ |
| *Masking* | | | | | | | | | |
| p1 masked | block_mask_wt/mt 0 -> 1 | 1,2,3 | 0.000 | +0.001 | −0.011 | −0.007 | −0.021 | +0.001 | −0.004 |
| *Loss* | | | | | | | | | |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | −0.002 | −0.003 | −0.001 | +0.003 | 0.000 | +0.002 | −0.003 |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | +0.003 | +0.002 | +0.004 | −0.001 | −0.002 | +0.001 | +0.001 |
| *Architecture* | | | | | | | | | |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | +0.001 | +0.001 | 0.000 | +0.001 | **−0.018** | +0.005 | 0.000 |
| *Cross-masking (evaluation only)* | | | | | | | | | |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | — | — | — | — | — | — | — | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — | — | — | — |

WT head alone on every row (wt_lora_pred). p0 row: Spearman ρ (seed mean; seeds listed). Other rows: Δρ = family seed mean − p0 under the same readout. Left out: ablations that reach the MT head only (no pair loss, MT LoRA rank 2, p1 MT marginal), whose WT heads differ from p0 by less than p0's own seeds do, and the epistasis columns (the WT head is additive by construction). Bold: |Δ| > 2 SE, where SE combines seed noise (family SD, or the pooled seed SD of the column for a single-seed run) and the paired seed-1 bootstrap over proteins / sites / assays. ‡: fewer seeds than the rest of the row (the cell's seeds are in table_wt_delta.csv). —: not yet run.
