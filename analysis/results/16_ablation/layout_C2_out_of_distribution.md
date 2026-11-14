### C2_out_of_distribution

| Family | Change | Seeds | Ext. singles (10) | PTMUL/-D | DMS (7) | Domainome | Domainome low-id | PG avg | PG avg low-id | PG Stab. low-id |
|---|---|---|---|---|---|---|---|---|---|---|
| p0 (reference) | reference recipe: unmasked WT and MT passes | 1/1,2,3 | 0.654 ± 0.004 | 0.594 ± 0.003 | 0.601 ± 0.004 | 0.546 ± 0.004 | 0.540 ± 0.004 | 0.509 | 0.507 | 0.810 |
| p1 masked | block_mask_wt/mt 0 -> 1 (independent single-site masking) | 1,2,3 | 0.647 ± 0.005 (-0.007) | 0.569 ± 0.007 (-0.011) | 0.609 ± 0.006 (**+0.007**) | 0.546 ± 0.007 (**-0.010**) | 0.540 ± 0.006 (**-0.010**) | 0.504 ± 0.002 (**-0.006**) | 0.501 ± 0.002 | 0.810 ± 0.002 |
| p1 MT marginal | p1 + MT pass marginal (block_marginal_mt) | 1,2,3 | 0.644 ± 0.004 (**-0.010**) | 0.560 ± 0.013 (-0.030) | 0.611 ± 0.001 (**+0.012**) | 0.546 ± 0.007 (**+0.004**) | 0.541 ± 0.006 (**+0.004**) | — | — | — |
| no pair loss | block_pair_weight 0.38 -> 0 | 1,2,3 | 0.656 ± 0.003 (+0.001) | 0.598 ± 0.009 (**+0.013**) | 0.607 ± 0.003 (**+0.011**) | 0.545 ± 0.005 (**+0.000**) | 0.539 ± 0.004 (**+0.000**) | 0.508 ± 0.002 (**-0.001**) | 0.506 ± 0.001 | 0.811 ± 0.002 |
| no rank loss | lambda_rank_wt/mt 0.5 -> 0 | 1,2,3 | 0.658 ± 0.002 (-0.002) | 0.594 ± 0.004 (+0.002) | 0.609 ± 0.003 (**+0.010**) | 0.548 ± 0.003 (**-0.002**) | 0.542 ± 0.003 (**-0.002**) | 0.506 ± 0.000 (**-0.003**) | 0.504 ± 0.001 | 0.812 ± 0.002 |
| rank censor 0 | rank_censor_threshold -1 -> 0 | 1,2,3 | 0.653 ± 0.005 (+0.001) | 0.600 ± 0.006 (+0.012) | 0.608 ± 0.003 (**+0.011**) | 0.547 ± 0.004 (**+0.002**) | 0.541 ± 0.004 (**+0.002**) | 0.509 ± 0.002 (+0.001) | 0.507 ± 0.003 | 0.813 ± 0.002 |
| no seq. head | incl_sequence_head_wt/mt True -> False | 1,2,3 | 0.656 ± 0.004 (-0.002) | 0.587 ± 0.010 (-0.001) | 0.607 ± 0.003 (**+0.013**) | 0.550 ± 0.004 (**+0.006**) | 0.545 ± 0.004 (**+0.006**) | 0.508 ± 0.001 (+0.001) | 0.507 ± 0.001 | 0.811 ± 0.002 |
| MT LoRA r2 | lora_rank/alpha_mt 16/16 -> 2/4 | 1,2,3 | 0.653 ± 0.005 (+0.001) | 0.592 ± 0.001 (+0.002) | 0.604 ± 0.008 (**+0.009**) | 0.548 ± 0.005 (**+0.001**) | 0.542 ± 0.004 (**+0.000**) | 0.508 ± 0.003 (**-0.001**) | 0.506 ± 0.003 | 0.810 ± 0.003 |
| full MegaScale | train split splits_oct06_capped -> mega_splits (not homology-restricted; its Tsuboyama numbers are on its own test split) | 1,2 | 0.648 ± 0.005 (**-0.013**) | 0.577 ± 0.005 (-0.006) | 0.624 ± 0.006 (**+0.032**) | 0.556 ± 0.006 (**+0.011**) | 0.552 ± 0.006 (**+0.012**) | 0.522 ± 0.004 (**+0.016**) | 0.522 ± 0.004 | 0.851 ± 0.008 |
| p0, tested masked | FINAL_p0 checkpoint evaluated with independent masking | — | — | — | — | — | — | — | — | — |
| p1, tested unmasked | FINAL_p1 checkpoint evaluated unmasked | — | — | — | — | — | — | — | — | — |

Cells: seed mean ± seed SD (Spearman ρ). In parentheses: paired bootstrap Δ vs p0 at seed 1 (bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits Tsuboyama split (not comparable). ΔΔΔG = ρ of comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG.
