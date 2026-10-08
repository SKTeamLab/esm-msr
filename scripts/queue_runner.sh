#!/bin/bash
# Usage: scripts/queue_runner.sh QUEUE_FILE [WAIT_PID]
# Runs the arms of QUEUE_FILE one after another on THIS worktree's code (scripts/run_with_retry.sh -> scripts/run_arm.sh), starting once
# WAIT_PID (the run currently on the GPU) has exited. One arm per line: NAME EPOCHS SEED [training flags...]; blank lines and '#' lines are
# skipped. The file is re-read before every arm, so arms can be edited, reordered, added or removed while the queue runs (do NOT edit this
# script while it runs: bash re-reads it). A started arm is commented out in place ('# started <time>: ...') so it never runs twice.
# To stop after the current arm: touch QUEUE_FILE.stop
Q=$(readlink -f "$1"); WAIT_PID=$2
D=/home/sareeves/playground/esm-msr-devel
HERE=$(dirname "$(readlink -f "$0")")
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
while true; do
  if [ -e "$Q.stop" ]; then echo "$(date) QUEUE $(basename "$Q") stopped ($Q.stop exists)" >> $D/queue_history.txt; exit 0; fi
  line=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$Q" | head -1)
  if [ -z "$line" ]; then echo "$(date) QUEUE $(basename "$Q") finished" >> $D/queue_history.txt; exit 0; fi
  # never start on a busy card (another job, or the previous one still releasing memory)
  for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 4000 ] && break
    sleep 30
  done
  awk -v d="$(date '+%F %T')" '!done && $0 !~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*$/ { print "# started " d ": " $0; done = 1; next } { print }' \
    "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  set -- $line
  "$HERE/run_with_retry.sh" "$@"
  sleep 20
done
