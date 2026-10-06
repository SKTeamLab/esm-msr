#!/bin/bash
# Foundation run with weighted epistasis components, then the two rank-loss ablations on top of it (docs/hparam_review.md).
# Base = link (default) + out-of-range items, micro-batch 64 (3 columns per pair unit). Components: pair offset x3, substitution effects x3, interaction x1.
S=$(dirname "$(readlink -f "$0")")
BASE="--include_out_of_range --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 1"
$S/run_with_retry.sh f1_comp331 6 1 $BASE
$S/run_with_retry.sh f2_comp331_wtrank0 6 1 $BASE --lambda_rank_wt 0
$S/run_with_retry.sh f3_comp331_colrank0 6 1 $BASE --lambda_mt_colrank 0
