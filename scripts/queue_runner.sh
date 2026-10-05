#!/bin/bash
# Runs arms from queue.txt one at a time; each line: RUN_NAME EPOCHS SEED [flags...]. Edit the file to change what runs next.
D=/home/sareeves/playground/esm-msr-devel
R=$D/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts/run_arm.sh
until grep -aq "^DONE" $D/run_logs/w0_base.log; do sleep 20; done   # let the already-running W0 finish
while true; do
  L=$(head -1 $D/queue.txt)
  if [ -z "$L" ]; then sleep 60; continue; fi
  sed -i 1d $D/queue.txt
  echo "$(date) START $L" >> $D/queue_history.txt
  $R $L
  echo "$(date) END $L" >> $D/queue_history.txt
done
