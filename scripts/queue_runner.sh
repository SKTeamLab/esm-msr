#!/bin/bash
# Runs arms from queue.txt one at a time; each line: RUN_NAME EPOCHS SEED [flags...]. Edit the file to change what runs next.
# A crashed arm (log ends 'DONE rc!=0') is retried: attempt 2 unchanged, attempt 3 with micro_batch_size 32 and the default allocator.
D=/home/sareeves/playground/esm-msr-devel
R=$D/repo/.claude/worktrees/censored-margin-ranking-9b239c/scripts/run_arm.sh
while true; do
  L=$(head -1 $D/queue.txt)
  if [ -z "$L" ]; then sleep 60; continue; fi
  sed -i 1d $D/queue.txt
  NAME=$(echo $L | cut -d' ' -f1)
  for attempt in 1 2 3; do
    echo "$(date) START(attempt $attempt) $L" >> $D/queue_history.txt
    rm -rf $D/training_logs/$NAME
    if [ $attempt = 3 ]; then ALLOC_CONF=garbage_collection_threshold:0.8 $R $L --micro_batch_size 32; else $R $L; fi
    echo "$(date) END $NAME $(tail -1 $D/run_logs/$NAME.log | cut -c1-30)" >> $D/queue_history.txt
    if tail -1 $D/run_logs/$NAME.log | grep -aq "DONE rc=0"; then break; fi
    mkdir -p $D/run_logs/crashed; cp $D/run_logs/$NAME.log $D/run_logs/crashed/${NAME}_try$attempt.log
    sleep 120
  done
done
