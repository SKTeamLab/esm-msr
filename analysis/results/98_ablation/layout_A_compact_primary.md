### A_compact_primary

| Family | Change | Seeds | Tsu test | Tsu test ΔΔΔG | Ext. singles (10) | PTMUL/-D | DMS (3) | Domainome low-id | PG avg |
|---|---|---|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.845 ± 0.000 | 0.479 ± 0.009 | 0.654 ± 0.004 | 0.594 ± 0.003 | 0.668 ± 0.005 | 0.540 ± 0.004 | 0.509 ± 0.002 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.835 ± 0.001 (**-0.010**) | 0.446 ± 0.034 (-0.023) | 0.647 ± 0.005 (-0.007) | 0.569 ± 0.007 (-0.019) | 0.682 ± 0.007 (**+0.015**) | 0.540 ± 0.006 (+0.002) | 0.504 ± 0.002 (**-0.005**) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.836 ± 0.002 (**-0.009**) | 0.429 ± 0.018 (**-0.049**) | 0.644 ± 0.004 (**-0.010**) | 0.560 ± 0.013 (-0.027) | 0.681 ± 0.002 (**+0.014**) | 0.540 ± 0.006 (+0.000) | 0.505 ± 0.001 (**-0.004**) |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.843 ± 0.002 (-0.002) | 0.441 ± 0.018 (-0.046) | 0.656 ± 0.003 (+0.001) | 0.598 ± 0.009 (+0.005) | 0.673 ± 0.002 (+0.005) | 0.539 ± 0.004 (-0.001) | 0.509 ± 0.001 (-0.000) |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.840 ± 0.005 (-0.005) | 0.448 ± 0.007 (-0.038) | 0.658 ± 0.002 (+0.003) | 0.594 ± 0.004 (-0.002) | 0.687 ± 0.001 (**+0.019**) | 0.541 ± 0.003 (+0.002) | 0.507 ± 0.000 (**-0.003**) |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.844 ± 0.001 (-0.001) | 0.458 ± 0.012 (-0.023) | 0.653 ± 0.005 (-0.001) | 0.600 ± 0.006 (+0.004) | 0.674 ± 0.002 (**+0.006**) | 0.541 ± 0.004 (**+0.001**) | 0.510 ± 0.003 (**+0.001**) |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.842 ± 0.001 (-0.003) | 0.463 ± 0.013 (-0.017) | 0.656 ± 0.004 (+0.001) | 0.587 ± 0.010 (-0.006) | 0.671 ± 0.003 (+0.003) | 0.544 ± 0.004 (**+0.005**) | 0.510 ± 0.001 (+0.000) |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.837 ± 0.004 (**-0.008**) | 0.452 ± 0.020 (-0.030) | 0.653 ± 0.005 (-0.001) | 0.592 ± 0.001 (-0.001) | 0.668 ± 0.007 (+0.001) | 0.541 ± 0.004 (**+0.002**) | 0.509 ± 0.003 (**-0.001**) |
| full MegaScale | train split splits_oct06_capped -> mega_splits (not homology-restricted; its Tsuboyama numbers are on its own test split) | 1,2,3 | 0.827 ± 0.002† | 0.507 ± 0.014† | 0.649 ± 0.003 (-0.006) | 0.583 ± 0.012 (-0.008) | 0.681 ± 0.004 (**+0.012**) | 0.553 ± 0.006 (**+0.014**) | 0.523 ± 0.003 (**+0.014**) |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | 1 | 0.820 (**-0.025**) | 0.427 (-0.052) | 0.652 (-0.002) | 0.573 (-0.016) | 0.683 (**+0.016**) | 0.554 (**+0.015**) | 0.504 (**-0.005**) |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — | — | — | — |

Cells: seed mean ± seed SD (Spearman ρ). In parentheses: paired bootstrap Δ vs p0, seeds 1-3 (bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits Tsuboyama split (not comparable). ΔΔΔG = ρ of comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG.
