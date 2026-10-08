#!/bin/bash
# Usage: baseline/start_full_when_ready.sh "PID PID ..." RUN_NAME EPOCHS SEED [flags]
# Waits for the given processes (cache shards, a test run) to exit, checks that EVERY library of the capped split has its cache, then runs the baseline.
PIDS=$1; shift
D=/home/sareeves/playground/esm-msr-devel
W=$(dirname "$(dirname "$(readlink -f "$0")")")
while true; do
  alive=0; for p in $PIDS; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break; sleep 30
done
export PYTHONPATH=$W/src
missing=$(/home/sareeves/miniconda3/envs/msr_venv/bin/python $W/baseline/missing_caches.py 2>/dev/null | tail -1)
if [ "$missing" != "0" ]; then
  echo "$(date) $2 NOT STARTED: $missing libraries of the capped split still lack a cache (see baseline/missing_caches.py)" >> $D/queue_history.txt; exit 1
fi
exec $W/baseline/run_with_retry.sh "$@"
