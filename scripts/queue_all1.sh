#!/bin/bash
# i3 (lambda 300, requeued) then phase A of docs/hparam_review.md. Link is the default now; Comet project esm-msr-agent-2.
S=$(dirname "$(readlink -f "$0")")
$S/run_with_retry.sh i3_lam300 6 1 --include_out_of_range --lambda_int_mt 300
$S/queue_plan1.sh
