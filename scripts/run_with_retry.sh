#!/bin/bash
# Usage: scripts/run_with_retry.sh NAME EPOCHS SEED [training flags...]
# Runs scripts/run_arm.sh and retries ONLY on a crash (the intermittent WSL2/Windows GPU "device not ready" errors), always with the SAME
# configuration. CUDA memory is capped at 95% so a genuine OOM is a clean error rather than a driver fault. Initial validation is on.
NAME=$1; EPOCHS=$2; SEED=$3; shift 3
D=/home/sareeves/playground/esm-msr-devel
R=$(dirname "$(readlink -f "$0")")/run_arm.sh
export MSR_MEM_FRACTION=${MSR_MEM_FRACTION:-0.95}
mkdir -p $D/run_logs/crashed
for attempt in 1 2 3; do
  echo "$(date) $NAME START(attempt $attempt) $*" >> $D/queue_history.txt
  rm -rf $D/training_logs/$NAME
  $R $NAME $EPOCHS $SEED "$@"
  tail -1 $D/run_logs/$NAME.log | grep -aq "DONE rc=0" && break
  cp $D/run_logs/$NAME.log $D/run_logs/crashed/${NAME}_try$attempt.log
  echo "$(date) $NAME END crashed (attempt $attempt)" >> $D/queue_history.txt
  sleep 120
done
echo "$(date) $NAME END" >> $D/queue_history.txt
