### A_compact_primary

| Family | Change | Seeds | Tsu test | Tsu test ΔΔΔG | Ext. singles (10) | PTMUL/-D | Domainome low-id | PG avg |
|---|---|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.845 ± 0.000 | 0.479 ± 0.009 | 0.654 ± 0.004 | 0.594 ± 0.003 | 0.540 ± 0.004 | 0.509 ± 0.003 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.835 ± 0.001 (**-0.010**) | 0.447 ± 0.034 (-0.033) | 0.647 ± 0.005 (-0.007) | 0.569 ± 0.007 (-0.025) | 0.540 ± 0.006 (+0.000) | 0.505 ± 0.002 (**-0.004**) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.836 ± 0.002 (**-0.009**) | 0.429 ± 0.018 (-0.050) | 0.644 ± 0.004 (**-0.010**) | 0.560 ± 0.013 (-0.034) | 0.540 ± 0.006 (+0.001) | — |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.843 ± 0.002 (-0.002) | 0.441 ± 0.018 (-0.038) | 0.656 ± 0.003 (+0.001) | 0.598 ± 0.009 (+0.004) | 0.539 ± 0.004 (-0.001) | — |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.840 ± 0.005 (-0.005) | 0.448 ± 0.007 (-0.031) | 0.658 ± 0.002 (+0.003) | 0.594 ± 0.004 (+0.000) | 0.541 ± 0.003 (+0.002) | — |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.844 ± 0.001 (-0.001) | 0.458 ± 0.012 (-0.021) | 0.653 ± 0.005 (-0.001) | 0.600 ± 0.006 (+0.006) | 0.541 ± 0.004 (**+0.001**) | — |
| no range-tagged cells | Tobit regression without the assay's range-tagged (<-1 / >5) cells (--reg_exclude_range) | — | — | — | — | — | — | — |
| MSE (no range-tagged) | Tobit NLL -> plain MSE on ddG, range-tagged cells excluded (--reg_nll none) | 3 | 0.843 (-0.003) | 0.445 (-0.034) | 0.654 (-0.000) | 0.591 (-0.003) | 0.537 (-0.003) | — |
| deep floor (-1, 1) | Tobit floor mu_f/sigma_f -0.14/0.5 -> -1.0/1.0 (also the posterior teacher-forcing label); sigma_y unchanged | — | — | — | — | — | — | — |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.842 ± 0.001 (-0.003) | 0.463 ± 0.013 (-0.016) | 0.656 ± 0.004 (+0.001) | 0.587 ± 0.010 (-0.007) | 0.544 ± 0.004 (**+0.005**) | — |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.837 ± 0.004 (**-0.008**) | 0.452 ± 0.020 (-0.027) | 0.653 ± 0.005 (-0.001) | 0.592 ± 0.001 (-0.002) | 0.541 ± 0.004 (**+0.002**) | — |
| full MegaScale | train split splits_oct06_capped -> mega_splits (evaluated on its own test split) | 1,2,3 | 0.827 ± 0.002† | 0.507 ± 0.014† | 0.649 ± 0.003 (-0.006) | 0.583 ± 0.012 (-0.011) | 0.553 ± 0.006 (**+0.014**) | — |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | 1,2,3 | 0.820 ± 0.001 (**-0.025**) | 0.427 ± 0.012 (**-0.052**) | 0.648 ± 0.005 (-0.006) | 0.574 ± 0.002 (-0.020) | 0.551 ± 0.003 (**+0.012**) | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | 1,2,3 | 0.819 ± 0.003 (**-0.026**) | 0.448 ± 0.033 (-0.031) | 0.642 ± 0.004 (**-0.013**) | 0.578 ± 0.007 (-0.016) | 0.521 ± 0.005 (**-0.018**) | — |

Cells: seed mean ± seed SD (ddof=1; Spearman ρ). Parentheses: paired bootstrap Δ vs ESM-MSR across seeds 1–3 (B = 1,000; bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits test set.
