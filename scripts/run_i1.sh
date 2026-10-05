#!/bin/bash
# Usage: scripts/run_i1.sh [NAME] [LAMBDA_INT]   the I-series design: link + out-of-range items + interaction loss (pair groups and aligned
# micro-batches are now implied by --lambda_int_mt and --micro_batch_size 64).
NAME=${1:-i1_link_cens_int}; LAM=${2:-1.0}
exec "$(dirname "$(readlink -f "$0")")/run_with_retry.sh" $NAME 6 1 --link softclamp --include_out_of_range --lambda_int_mt $LAM
