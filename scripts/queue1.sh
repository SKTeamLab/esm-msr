#!/bin/bash
R=/home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts/run_arm.sh
$R w0_base 3 1 --shared_bias_init 0
$R w2_cens 3 1 --shared_bias_init 0 --include_out_of_range
$R l0_link 3 1 --link softclamp
$R l1_link_cens 3 1 --link softclamp --include_out_of_range
