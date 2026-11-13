### A_compact_primary

| Family | Change | Seeds | Tsu test | Tsu test ΔΔΔG | Ext. singles (10) | PTMUL/-D | DMS (7) | Domainome low-id | PG avg |
|---|---|---|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1 | 0.845 | 0.475 | 0.658 | 0.591 | 0.597 | 0.543 | 0.509 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.835 ± 0.001 (**-0.008**) | 0.446 ± 0.034 (+0.009) | 0.647 ± 0.005 (-0.007) | 0.569 ± 0.007 (-0.011) | 0.609 ± 0.006 (**+0.007**) | 0.540 ± 0.006 (**-0.010**) | 0.504 ± 0.002 (**-0.006**) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.836 ± 0.002 (**-0.011**) | 0.429 ± 0.018 (-0.043) | 0.644 ± 0.004 (**-0.010**) | 0.560 ± 0.013 (-0.030) | 0.611 ± 0.001 (**+0.012**) | 0.541 ± 0.006 (**+0.004**) | — |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.843 ± 0.002 (-0.003) | 0.441 ± 0.018 (-0.034) | 0.656 ± 0.003 (+0.001) | 0.598 ± 0.009 (**+0.013**) | 0.607 ± 0.003 (**+0.011**) | 0.539 ± 0.004 (**+0.000**) | 0.508 ± 0.002 (**-0.001**) |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.840 ± 0.005 (+0.001) | 0.448 ± 0.007 (-0.020) | 0.658 ± 0.002 (-0.002) | 0.594 ± 0.004 (+0.002) | 0.609 ± 0.003 (**+0.010**) | 0.542 ± 0.003 (**-0.002**) | 0.506 ± 0.000 (**-0.003**) |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.844 ± 0.001 (-0.001) | 0.458 ± 0.012 (-0.007) | 0.653 ± 0.005 (+0.001) | 0.600 ± 0.006 (+0.012) | 0.608 ± 0.003 (**+0.011**) | 0.541 ± 0.004 (**+0.002**) | 0.509 ± 0.002 (+0.001) |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.842 ± 0.001 (**-0.005**) | 0.463 ± 0.013 (-0.004) | 0.656 ± 0.004 (-0.002) | 0.587 ± 0.010 (-0.001) | 0.607 ± 0.003 (**+0.013**) | 0.545 ± 0.004 (**+0.006**) | 0.508 ± 0.001 (+0.001) |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.837 ± 0.004 (**-0.007**) | 0.452 ± 0.020 (-0.014) | 0.653 ± 0.005 (+0.001) | 0.592 ± 0.001 (+0.002) | 0.604 ± 0.008 (**+0.009**) | 0.542 ± 0.004 (**+0.000**) | 0.508 ± 0.003 (**-0.001**) |
| full MegaScale | train split splits_oct06_capped -> mega_splits (not homology-restricted; its Tsuboyama numbers are on its own test split) | 1,2 | 0.827 ± 0.003† | 0.515 ± 0.003† | 0.648 ± 0.005 (**-0.013**) | 0.577 ± 0.005 (-0.006) | 0.624 ± 0.006 (**+0.032**) | 0.552 ± 0.006 (**+0.012**) | 0.522 ± 0.004 (**+0.016**) |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | — | — | — | — | — | — | — | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — | — | — | — |

Cells: seed mean ± seed SD (Spearman ρ). In parentheses: paired bootstrap Δ vs p0 at seed 1 (bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits Tsuboyama split (not comparable). ΔΔΔG = ρ of comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG.
