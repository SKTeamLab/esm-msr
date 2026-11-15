### C1_in_distribution

| Family | Change | Seeds | Tsu test | Tsu test singles | Tsu test doubles | Tsu test ΔΔΔG |
|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.845 ± 0.000 | 0.833 ± 0.003 | 0.777 ± 0.005 | 0.479 ± 0.009 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.835 ± 0.001 (**-0.008**) | 0.834 ± 0.003 (+0.002) | 0.747 ± 0.004 (-0.021) | 0.446 ± 0.034 (+0.009) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.836 ± 0.002 (**-0.011**) | 0.835 ± 0.002 (-0.002) | 0.721 ± 0.006 (**-0.055**) | 0.429 ± 0.018 (-0.043) |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.843 ± 0.002 (-0.003) | 0.835 ± 0.002 (**-0.002**) | 0.768 ± 0.002 (-0.005) | 0.441 ± 0.018 (-0.034) |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.840 ± 0.005 (+0.001) | 0.830 ± 0.003 (-0.002) | 0.765 ± 0.008 (+0.002) | 0.448 ± 0.007 (-0.020) |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.844 ± 0.001 (-0.001) | 0.835 ± 0.002 (+0.001) | 0.772 ± 0.005 (-0.005) | 0.458 ± 0.012 (-0.007) |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.842 ± 0.001 (**-0.005**) | 0.834 ± 0.001 (-0.002) | 0.763 ± 0.004 (-0.014) | 0.463 ± 0.013 (-0.004) |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.837 ± 0.004 (**-0.007**) | 0.832 ± 0.003 (-0.000) | 0.752 ± 0.008 (-0.021) | 0.452 ± 0.020 (-0.014) |
| full MegaScale | train split splits_oct06_capped -> mega_splits (not homology-restricted; its Tsuboyama numbers are on its own test split) | 1,2 | 0.827 ± 0.003† | 0.821 ± 0.000† | 0.667 ± 0.016† | 0.515 ± 0.003† |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | — | — | — | — | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — |

Cells: seed mean ± seed SD (Spearman ρ). In parentheses: paired bootstrap Δ vs p0 at seed 1 (bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits Tsuboyama split (not comparable). ΔΔΔG = ρ of comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG.
