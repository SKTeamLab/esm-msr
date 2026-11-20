### E_multi_readouts

| Family | Change | Seeds | Tsu test doubles [wt] | Tsu test doubles [wt+] | Tsu test doubles [comb] | PTMUL [wt] | PTMUL [wt+] | PTMUL [comb] | PTMUL-D [wt] | PTMUL-D [wt+] | PTMUL-D [comb] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1,2,3 | 0.723 ± 0.004 | 0.723 ± 0.004 | 0.777 ± 0.005 | 0.589 ± 0.004 | — | 0.592 ± 0.008 | 0.584 ± 0.002 | — | 0.596 ± 0.005 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.712 ± 0.009 (-0.011) | 0.713 ± 0.009 | 0.747 ± 0.004 (**-0.030**) | 0.558 ± 0.017 (-0.030) | — | 0.560 ± 0.010 (-0.032) | 0.572 ± 0.006 (-0.012) | — | 0.577 ± 0.004 (-0.019) |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.713 ± 0.004 (-0.011) | 0.713 ± 0.003 | 0.721 ± 0.006 (**-0.056**) | 0.551 ± 0.018 (-0.037) | — | 0.558 ± 0.015 (-0.034) | 0.560 ± 0.010 (-0.025) | — | 0.563 ± 0.011 (-0.033) |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.726 ± 0.003 (+0.003) | 0.726 ± 0.004 | 0.768 ± 0.002 (-0.009) | 0.581 ± 0.014 (-0.008) | — | 0.594 ± 0.011 (+0.003) | 0.580 ± 0.005 (-0.005) | — | 0.601 ± 0.008 (+0.005) |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.723 ± 0.010 (-0.001) | 0.721 ± 0.011 | 0.765 ± 0.008 (-0.012) | 0.584 ± 0.004 (-0.005) | — | 0.591 ± 0.005 (-0.001) | 0.590 ± 0.005 (+0.005) | — | 0.597 ± 0.004 (+0.001) |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.727 ± 0.002 (+0.004) | 0.727 ± 0.002 | 0.772 ± 0.005 (-0.005) | 0.587 ± 0.006 (-0.002) | — | 0.601 ± 0.006 (+0.009) | 0.582 ± 0.007 (-0.002) | — | 0.600 ± 0.007 (+0.004) |
| no range-tagged cells | Tobit regression without the assay's range-tagged (<-1 / >5) cells (--reg_exclude_range) | — | — | — | — | — | — | — | — | — | — |
| MSE (no range-tagged) | Tobit NLL -> plain MSE on ddG, range-tagged cells excluded (--reg_nll none) | 3 | 0.719 (-0.004) | 0.719 | 0.774 (-0.003) | 0.572 (-0.017) | — | 0.592 (+0.001) | 0.574 (-0.011) | — | 0.590 (-0.006) |
| deep floor (-1, 1) | Tobit floor mu_f/sigma_f -0.14/0.5 -> -1.0/1.0 (also the posterior teacher-forcing label); sigma_y unchanged | — | — | — | — | — | — | — | — | — | — |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.724 ± 0.005 (+0.000) | 0.724 ± 0.005 | 0.763 ± 0.004 (-0.015) | 0.568 ± 0.013 (-0.021) | — | 0.584 ± 0.010 (-0.007) | 0.570 ± 0.010 (-0.015) | — | 0.589 ± 0.010 (-0.008) |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.720 ± 0.005 (-0.004) | 0.719 ± 0.005 | 0.752 ± 0.008 (**-0.025**) | 0.586 ± 0.001 (-0.003) | — | 0.590 ± 0.003 (-0.002) | 0.582 ± 0.002 (-0.003) | — | 0.595 ± 0.001 (-0.002) |
| full MegaScale | train split splits_oct06_capped -> mega_splits (evaluated on its own test split) | 1,2,3 | 0.569 ± 0.015† | 0.564 ± 0.014† | 0.662 ± 0.014† | 0.553 ± 0.006 (-0.035) | — | 0.564 ± 0.011 (-0.028) | 0.580 ± 0.010 (-0.004) | — | 0.603 ± 0.012 (+0.006) |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | 1,2,3 | 0.695 ± 0.004 (**-0.028**) | 0.697 ± 0.004 | 0.704 ± 0.003 (**-0.073**) | 0.563 ± 0.004 (-0.026) | — | 0.571 ± 0.005 (-0.020) | 0.562 ± 0.003 (-0.022) | — | 0.577 ± 0.006 (-0.019) |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | 1,2,3 | 0.720 ± 0.004 (-0.003) | 0.719 ± 0.004 | 0.729 ± 0.006 (**-0.048**) | 0.568 ± 0.017 (-0.021) | — | 0.568 ± 0.011 (-0.024) | 0.590 ± 0.005 (+0.005) | — | 0.588 ± 0.003 (-0.009) |

Cells: seed mean ± seed SD (ddof=1; Spearman ρ). Parentheses: paired bootstrap Δ vs ESM-MSR across seeds 1–3 (B = 1,000; bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits test set.
