#!/bin/bash
# Gradient-balanced regression (--reg_balance) on the oct06 split. NOT meant to be launched whole: run ONE arm at a time with
#   scripts/run_with_retry.sh NAME 6 1 <flags>        (paste a line below) and choose the next from the result.
#
# What --reg_balance does: every regression term is multiplied by a constant (training.REG_BALANCE, measured with scripts/grad_share_probe.py on
# 48 + 48 batches) so that a weight of 1 gives it about as much gradient (rms, on the adapters) as the rank loss (--lambda_rank_wt /
# --lambda_mt_colrank, both left at 1). Accurate to about a factor of 2: per-batch ratios are heavy-tailed and the two checkpoints measured differ.
#   weight meaning:  --lambda_reg_wt      WT regression (singles)
#                    --lambda_mt_cell     ALL of the MT regression (blocks and the remaining cells)
#                    --mt_comp_offset / _subst / _int   relative weights of the pair offset / substitution effects / interaction parts of the MT
#                                         block error; they multiply lambda_mt_cell.
#   So "0.2,0.2,0.2" = --lambda_reg_wt 0.2 --lambda_mt_cell 0.2 with components 1,1,1; "0.2,0.2,0" = the same with --mt_comp_int 0.
#
# Why the earlier o6b1 / o6f1 runs are not comparable and were weak: their weights in rank-gradient units were tiny. Old (unbalanced) weight w
# is about w / K here:  WT regression 1 -> 0.06;  MT plain regression 1 -> 0.01;  components 3,3,1 -> 0.13, 0.05, 0.002;  and the old
# --mt_comp_int 850 (i3_lam300, the only earlier run that moved the epistasis metrics) -> 1.6. Rank losses, by contrast, were at 1.
#
# Reference run: o6g1_bal111 (launched 2026-10-06 21:54 on the capped split).
S=$(dirname "$(readlink -f "$0")")
echo "queue_o6g.sh is a menu: copy ONE run_with_retry line into a shell (o6g1 is already running). Not launching anything." >&2; exit 1
B="--include_out_of_range --reg_balance"

# --- 1. the weight grid (decides the base for the mask arm). Already queued, in this order, with scripts/after.sh -------------------------------------
$S/run_with_retry.sh o6g1_bal111      6 1 $B                                                           # 1: every regression term worth one rank loss (RUNNING)
$S/run_with_retry.sh o6g2_bal01       6 1 $B --lambda_reg_wt 0.1 --lambda_mt_cell 0.1                  # 2: a tenth of the rank gradient (QUEUED)
$S/run_with_retry.sh o6g3_bal0        6 1 $B --lambda_reg_wt 0 --lambda_mt_cell 0 --mt_single_anchor_weight 0   # 3: rank losses only (QUEUED). The MT single anchors are regression
                                                                                                        #    terms and must be off, or the compose step asserts. With no regression the
                                                                                                        #    calibration scales and the link get no gradient (they stay at their initial
                                                                                                        #    values), so judge this arm on rank metrics, not RMSE / pair offset.

# --- 2. structure masking on the best of 1-3 (launched when arm 3 is done; weights flags of the winner after $B) -------------------------------
# $S/run_with_retry.sh o6gm1_maskstruct 6 1 $B --mask_structure <winning weight flags>
#   then: the REMAINDER below, on the better of (mask arm, best non-mask arm); add --mask_structure to each if the mask arm wins.

# --- 3. what the pair offset needs (the o6f1 offset deficit; weights now strong enough to test it) --------------------------------------------
# $S/run_with_retry.sh o6g_off3         6 1 $B --mt_comp_offset 3                                    # offset part 3x the rank gradient-equivalent
# $S/run_with_retry.sh o6g_off3_int0    6 1 $B --mt_comp_offset 3 --mt_comp_int 0

# --- 4. the other rank-vs-regression questions, now meaningful because regression is no longer ~1/50 of the rank gradient -------------------
# $S/run_with_retry.sh o6g_colrank0     6 1 $B --lambda_mt_colrank 0                                 # does the MT column rank loss help?
# $S/run_with_retry.sh o6g_wtrank0      6 1 $B --lambda_rank_wt 0
# $S/run_with_retry.sh o6g_colrank3     6 1 $B --lambda_mt_colrank 3                                 # dose response; only if colrank0 shows the loss matters
# $S/run_with_retry.sh o6g_colrank03    6 1 $B --lambda_mt_colrank 0.3
# $S/run_with_retry.sh o6g_int16        6 1 $B --mt_comp_int 1.6                                     # ~ the old i3_lam300 (exploits the link leak: read with care)

# --- 5. link and data options (as in queue_o6b.sh, rebased) ----------------------------------------------------------------------------------
# $S/run_with_retry.sh o6g_fixbounds    6 1 $B --no-link_learn_bounds
# $S/run_with_retry.sh o6g_frac025      6 1 $B --mt_single_anchor_frac 0.25                          # cost lever
# $S/run_with_retry.sh o6g_floor0       6 1 $B --censor_floor 0.0
# $S/run_with_retry.sh o6g_nooor        6 1 --reg_balance                                            # no out-of-range items

# Every arm writes val_dump_<tag>.npz (zero-shot 'zs' and per epoch 'e<N>') into its training_logs dir, for the pair-offset / saturation
# decomposition (see the scratchpad scripts offdec2.py / satbins.py).
