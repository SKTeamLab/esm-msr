#!/bin/bash
# Phase B of docs/hparam_review.md §5. Do NOT launch blind: set BASE (and RANK) to the winner of phase A first, then delete the arms you do not want.
S=$(dirname "$(readlink -f "$0")")
BASE="--link softclamp --include_out_of_range"          # + the phase-A winner, e.g. --lora_rank_mt 4
$S/run_with_retry.sh p3_mask 6 1 $BASE --mask_structure
$S/run_with_retry.sh p3_fixbounds 6 1 $BASE --no-link_learn_bounds
$S/run_with_retry.sh p4_rank0_int30 6 1 $BASE --lambda_rank_mt 0 --lambda_int_mt 30
$S/run_with_retry.sh p4_floor0 6 1 $BASE --censor_floor 0.0
$S/run_with_retry.sh p5_nooor 6 1 --link softclamp                                  # no --include_out_of_range
$S/run_with_retry.sh p6_frac025 6 1 $BASE --mt_single_anchor_frac 0.25              # cost lever; if neutral, later arms are ~1/3 cheaper
# only if phase A shows that MT capacity / step size matters:
# $S/run_with_retry.sh p2_a8  6 1 $BASE --lora_alpha_mt 8
# $S/run_with_retry.sh p2_a32 6 1 $BASE --lora_alpha_mt 32
