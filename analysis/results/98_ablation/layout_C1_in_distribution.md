### C1_in_distribution

| Family | Change | Seeds | Tsu test | Tsu test singles | Tsu test doubles | Tsu test ΔΔΔG |
|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.845 ± 0.000 | 0.833 ± 0.003 | 0.777 ± 0.005 | 0.479 ± 0.009 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.835 ± 0.001 (**-0.010**) | 0.834 ± 0.003 (+0.001) | 0.747 ± 0.004 (**-0.030**) | 0.447 ± 0.034 (-0.033) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.836 ± 0.002 (**-0.009**) | 0.835 ± 0.002 (+0.002) | 0.721 ± 0.006 (**-0.056**) | 0.429 ± 0.018 (-0.050) |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.843 ± 0.002 (-0.002) | 0.835 ± 0.002 (+0.002) | 0.768 ± 0.002 (-0.009) | 0.441 ± 0.018 (-0.038) |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.840 ± 0.005 (-0.005) | 0.830 ± 0.003 (-0.003) | 0.765 ± 0.008 (-0.012) | 0.448 ± 0.007 (-0.031) |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.844 ± 0.001 (-0.001) | 0.835 ± 0.002 (**+0.002**) | 0.772 ± 0.005 (-0.005) | 0.458 ± 0.012 (-0.021) |
| no range-tagged cells | Tobit regression without the assay's range-tagged (<-1 / >5) cells (--reg_exclude_range) | — | — | — | — | — |
| MSE (no range-tagged) | Tobit NLL -> plain MSE on ddG, range-tagged cells excluded (--reg_nll none) | — | — | — | — | — |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.842 ± 0.001 (-0.003) | 0.834 ± 0.001 (+0.001) | 0.763 ± 0.004 (-0.015) | 0.463 ± 0.013 (-0.016) |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.837 ± 0.004 (**-0.008**) | 0.832 ± 0.003 (-0.001) | 0.752 ± 0.008 (**-0.025**) | 0.452 ± 0.020 (-0.027) |
| full MegaScale | train split splits_oct06_capped -> mega_splits (evaluated on its own test split) | 1,2,3 | 0.827 ± 0.002† | 0.821 ± 0.000† | 0.662 ± 0.014† | 0.507 ± 0.014† |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | 1,2,3 | 0.820 ± 0.001 (**-0.025**) | 0.824 ± 0.003 (**-0.009**) | 0.704 ± 0.003 (**-0.073**) | 0.427 ± 0.012 (**-0.052**) |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | 1,2,3 | 0.819 ± 0.003 (**-0.026**) | 0.825 ± 0.001 (**-0.008**) | 0.729 ± 0.006 (**-0.048**) | 0.448 ± 0.033 (-0.031) |

Cells: seed mean ± seed SD (ddof=1; Spearman ρ). Parentheses: paired bootstrap Δ vs ESM-MSR across seeds 1–3 (B = 1,000; bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits test set.
