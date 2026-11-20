| Ablation | Change from ESM-MSR | Seeds | Tsuboyama test | Tsuboyama singles | Tsuboyama doubles | External singles (10) | PTMUL / PTMUL-D | Domainome low-id | ProteinGym avg |
|---|---|---|---|---|---|---|---|---|---|
| **p0 (reference)** | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.827 | 0.833 | 0.723 | 0.654 | 0.587 | 0.540 | 0.509 |
| *Masking* | | | | | | | | | |
| p1 masked | block_mask_wt/mt 0 -> 1 | 1,2,3 | 0.000 | +0.001 | −0.011 | −0.007 | −0.021 | 0.000 | **−0.005** |
| *Loss* | | | | | | | | | |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | −0.002 | −0.003 | −0.001 | +0.003 | 0.000 | +0.002 | −0.003 |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | **+0.003** | **+0.002** | +0.004 | −0.001 | −0.002 | **+0.001** | +0.001 |
| no range-tagged cells | Tobit regression without the assay's range-tagged (<-1 / >5) cells | — | — | — | — | — | — | — | — |
| MSE (no range-tagged) | Tobit NLL -> plain MSE on ddG, range-tagged cells excluded | — | — | — | — | — | — | — | — |
| *Architecture* | | | | | | | | | |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | +0.001 | +0.001 | 0.000 | +0.001 | −0.018 | **+0.005** | 0.000 |
| *Cross-masking (evaluation only)* | | | | | | | | | |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | 1,2,3 | **−0.010** | **−0.009** | −0.028 | −0.006 | **−0.024** | **+0.012** | **−0.005** |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | 1,2,3 | −0.005 | **−0.008** | −0.003 | **−0.013** | −0.008 | **−0.018** | **−0.005**‡ |

WT head alone on every row (wt_lora_pred). ESM-MSR row: Spearman ρ (seed mean). Other rows: Δρ = family seed mean − ESM-MSR under the WT head (joint [cluster/assay × 3-seed] bootstrap). Left out: MT-only ablations (no pair loss, MT LoRA rank 2, p1 MT marginal) and epistasis columns. Bold: |Δ| > 2 SE, where SE is the paired bootstrap over proteins / sites / assays with triplicate seeds resampled jointly (B = 1,000; pooled column seed SD fallback where no bootstrap exists). ‡: fewer seeds than the rest of the row (exact seeds in table_wt_delta.csv). —: not run.
