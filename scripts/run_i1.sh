#!/bin/bash
# I1 as designed (micro-batch 64, pair groups of 3, aligned micro-batches); retries ONLY on a crash, always with the same configuration.
D=/home/sareeves/playground/esm-msr-devel
R=$D/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts/run_arm.sh
for attempt in 1 2 3; do
  echo "$(date) I1 START(attempt $attempt)" >> $D/queue_history.txt
  rm -rf $D/training_logs/i1_link_cens_int
  $R i1_link_cens_int 6 1 --link softclamp --include_out_of_range --flip_pair_groups 3 --flip_align_units --lambda_int_mt 1.0
  tail -1 $D/run_logs/i1_link_cens_int.log | grep -aq "DONE rc=0" && break
  cp $D/run_logs/i1_link_cens_int.log $D/run_logs/crashed/i1_mb64_try$attempt.log
  echo "$(date) I1 END crashed (attempt $attempt)" >> $D/queue_history.txt
  sleep 120
done
echo "$(date) I1 END" >> $D/queue_history.txt
