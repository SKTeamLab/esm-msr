#!/bin/bash
# Second queue on the oct06 split (replaces the pending part of queue_o6.sh). Waits for the arm that is already running, then runs in priority order.
# Every arm is o6f1_comp331 (the reference, components 3/3/1) plus ONE change, except where marked. Results are comparable only with o6f1 / o6b1.
# Arm names: o6f* = the original foundation list, o6m* = the historical "phase B" arms (scripts/queue_plan2.sh) rebased onto o6f1.
S=$(dirname "$(readlink -f "$0")")
D=/home/sareeves/playground/esm-msr-devel
WAIT_PID=${1:-}                                      # run_with_retry.sh of the arm in flight
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
B="--include_out_of_range"
C="$B --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 1"
# 1  structure masking in the MT pass (paradigm question: does the model get a better pair signal when the mutated sites' geometry is blanked?)
$S/run_with_retry.sh o6m1_maskstruct   6 1 $C --mask_structure
# 2  is the interaction component doing anything?
$S/run_with_retry.sh o6f4_comp330      6 1 $B --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 0
# 3  does the MT column-rank loss help?
$S/run_with_retry.sh o6f3_colrank0     6 1 $C --lambda_mt_colrank 0
# 4  link bounds fixed at (-1, 5): is the slack in the learned bounds hiding or creating saturation?
$S/run_with_retry.sh o6m2_fixbounds    6 1 $C --no-link_learn_bounds
# 5  historical rank-off + lambda_int_mt 30 (= --mt_comp_int 85); relative to arm 3 this isolates the large interaction weight (exploits the link leak, see artifact)
$S/run_with_retry.sh o6m3_rank0_int85  6 1 $C --lambda_mt_colrank 0 --mt_comp_int 85
# 6  cost lever: a quarter of the MT single anchors per step (if neutral, later arms are cheaper)
$S/run_with_retry.sh o6m6_frac025      6 1 $C --mt_single_anchor_frac 0.25
# 7  does the WT head need its own rank loss?
$S/run_with_retry.sh o6f2_wtrank0      6 1 $C --lambda_rank_wt 0
# 8  rank losses treat items below 0 dG as ties
$S/run_with_retry.sh o6m4_floor0       6 1 $C --censor_floor 0.0
# 9  no out-of-range items at all (different training set; the validation set is unchanged)
$S/run_with_retry.sh o6m5_nooor        6 1 --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 1
# 10-11  dose response for the MT column-rank loss; only informative if arm 3 shows the loss matters
$S/run_with_retry.sh o6f5_colrank3     6 1 $C --lambda_mt_colrank 3
$S/run_with_retry.sh o6f6_colrank03    6 1 $C --lambda_mt_colrank 0.3
