#!/bin/bash
# Phase A of docs/hparam_review.md §5. Base = link + out-of-range items, rank loss only (no interaction loss), micro-batch 64.
# Run from the merged branch, after the GPU smoke of the cleaned code, and only when the lambda scan (queue_lam.sh) has finished.
# The MT rank sweep holds lora_alpha_mt at 16: the adapters use rsLoRA (scale = alpha / sqrt(rank)), which keeps the update size rank-independent.
S=$(dirname "$(readlink -f "$0")")
BASE="--link softclamp --include_out_of_range"
$S/run_with_retry.sh p1_base_s1 6 1 $BASE
$S/run_with_retry.sh p1_base_s2 6 2 $BASE
$S/run_with_retry.sh p2_mtr4 6 1 $BASE --lora_rank_mt 4
$S/run_with_retry.sh p2_mtr2 6 1 $BASE --lora_rank_mt 2
