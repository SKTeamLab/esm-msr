#!/bin/bash
# Usage: scripts/run_i1.sh [NAME] [LAMBDA_INT]   I1 as designed (micro-batch 64, pair groups of 3, aligned micro-batches).
# Runs initial validation. Retries ONLY on a crash, always with the same configuration; CUDA memory capped at 95% (clean OOM, not a driver error).
NAME=${1:-i1_link_cens_int}; LAM=${2:-1.0}
D=/home/sareeves/playground/esm-msr-devel
R=$D/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts/run_arm.sh
export MSR_MEM_FRACTION=${MSR_MEM_FRACTION:-0.95}
for attempt in 1 2 3; do
  echo "$(date) $NAME START(attempt $attempt) lambda_int_mt=$LAM" >> $D/queue_history.txt
  rm -rf $D/training_logs/$NAME
  $R $NAME 6 1 --link softclamp --include_out_of_range --flip_pair_groups 3 --flip_align_units --lambda_int_mt $LAM
  tail -1 $D/run_logs/$NAME.log | grep -aq "DONE rc=0" && break
  cp $D/run_logs/$NAME.log $D/run_logs/crashed/${NAME}_try$attempt.log
  echo "$(date) $NAME END crashed (attempt $attempt)" >> $D/queue_history.txt
  sleep 120
done
echo "$(date) $NAME END" >> $D/queue_history.txt
