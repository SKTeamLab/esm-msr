#!/bin/bash
# Interaction-loss weight scan (I1 design, micro 64, initial validation on): lambda_int_mt 30, then 300.
S=/home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts
$S/run_i1.sh i2_lam30 30
$S/run_i1.sh i3_lam300 300
