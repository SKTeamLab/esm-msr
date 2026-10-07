#!/bin/bash
# Usage: scripts/after.sh WAIT_PID NAME EPOCHS SEED [training flags...]
# Waits until process WAIT_PID has exited, then runs scripts/run_with_retry.sh with the rest. Chains one run behind another so the GPU never idles.
# NOTE: if the run you wait for is killed, this starts anyway (a kill looks like an exit); kill the waiter first.
WAIT_PID=$1; shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
exec "$(dirname "$(readlink -f "$0")")/run_with_retry.sh" "$@"
