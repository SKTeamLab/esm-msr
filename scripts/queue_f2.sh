#!/bin/bash
# Second batch, chained after queue_f1.sh: the interaction component off, and the rest of the MT column-rank sweep (0 is f3).
S=$(dirname "$(readlink -f "$0")")
BASE="--include_out_of_range --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 1"
$S/run_with_retry.sh f4r2_comp330 6 1 --include_out_of_range --mt_comp_offset 3 --mt_comp_subst 3 --mt_comp_int 0
$S/run_with_retry.sh f5r2_colrank3 6 1 $BASE --lambda_mt_colrank 3
$S/run_with_retry.sh f6r2_colrank03 6 1 $BASE --lambda_mt_colrank 0.3
