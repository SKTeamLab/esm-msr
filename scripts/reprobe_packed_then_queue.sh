#!/bin/bash
# Usage: scripts/reprobe_packed_then_queue.sh WAIT_PID QUEUE_FILE
# After WAIT_PID exits: the packed gradient-share probe again (the first one ran before the autocast-cache fix of --pack_pair_matrices and
# recorded no MT rank-loss gradient, so no *_packed constant could be formed), the constants file rewritten from the plain probes of
# 2026-10-08 (unchanged values) plus the new packed one, then the queue continues (scripts/queue_runner.sh QUEUE_FILE).
WAIT_PID=$1; Q=$(readlink -f "$2")
D=/home/sareeves/playground/esm-msr-devel
HERE=$(dirname "$(readlink -f "$0")"); WT=$(dirname "$HERE")
P=$D/probes
log() { echo "$(date) $*" >> $D/queue_history.txt; }
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
for _ in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 12000 ] && break; sleep 30; done
export PYTHONPATH=$WT/src HF_HUB_OFFLINE=1 PROBE_GPU_FRACTION=0.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
mv -f $P/probe_ref_packed.json $P/probe_ref_packed_before_fix.json 2>/dev/null
mv -f $P/probe_ref_packed.json.log $P/probe_ref_packed_before_fix.json.log 2>/dev/null
log "PROBE packed re-run (after the autocast-cache fix, devel 9d629d1)"
for micro in 64 38; do
  timeout 5400 $PY $HERE/grad_share_probe.py $P/probe_ref_packed.json --gpu --batches 20 --proteins 40 --micro $micro \
      --ckpt $D/training_logs/o6g6_c10_hier/0/last.ckpt --batch_size 400 --min_cond 200 --extra="--pack_pair_matrices" > $P/probe_ref_packed.json.log 2>&1
  grep -q PROBE_DONE $P/probe_ref_packed.json.log && break
  sleep 20
done
grep -q PROBE_DONE $P/probe_ref_packed.json.log && log "PROBE probe_ref_packed.json done" || log "PROBE probe_ref_packed.json FAILED (packed arms keep the plain constants)"
cp $P/reg_balance_oct08.json $P/reg_balance_oct08_before_packed_fix.json
$PY $HERE/balance_from_probe.py $P/reg_balance_oct08.json --plain $P/probe_ref_plain.json $P/probe_mt16_plain.json \
    --packed $P/probe_ref_packed.json > $P/balance.log 2>&1
log "PROBE constants rewritten -> $P/reg_balance_oct08.json: $(tr -d '\n ' < $P/balance.log | cut -c1-500)"
exec "$HERE/queue_runner.sh" "$Q"
