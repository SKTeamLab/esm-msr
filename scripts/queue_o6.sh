#!/bin/bash
# First sweep on the 2026-10-06 structure-homology split (identity- and TM-capped) (data/splits_oct06_capped.pkl; Comet project esm-msr-agent-oct06).
# MT head matched to the WT head (rank 2, alpha 4: the run_arm.sh defaults). Link on (default), out-of-range items included, micro-batch 64.
# Results are NOT comparable with runs on the old hyperopt split. o6b1 is the plain-regression reference for everything else.
S=$(dirname "$(readlink -f "$0")")
B="--include_out_of_range"
C="$B --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 1"
$S/run_with_retry.sh o6b1_base 6 1 $B
$S/run_with_retry.sh o6f1_comp331 6 1 $C
$S/run_with_retry.sh o6f2_wtrank0 6 1 $C --lambda_rank_wt 0
$S/run_with_retry.sh o6f3_colrank0 6 1 $C --lambda_mt_colrank 0
$S/run_with_retry.sh o6f4_comp330 6 1 $B --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 0
$S/run_with_retry.sh o6f5_colrank3 6 1 $C --lambda_mt_colrank 3
$S/run_with_retry.sh o6f6_colrank03 6 1 $C --lambda_mt_colrank 0.3
