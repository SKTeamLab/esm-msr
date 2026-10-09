#!/bin/bash
# Usage: scripts/probe_then_queue.sh WAIT_PID QUEUE_FILE
# After WAIT_PID exits: gradient-share probes on the GPU (the reference checkpoint o6g6_c10_hier with micro-batch slices and with
# --pack_pair_matrices, and the rank-16 o6q01_mt16 with slices), then scripts/balance_from_probe.py writes the --reg_balance constants to
# $D/probes/reg_balance_oct08.json (defaults kept for anything not measured), then the queue starts (scripts/queue_runner.sh QUEUE_FILE).
# A failed probe is retried at micro-batch 38 and otherwise skipped; the queue starts in every case. Everything is logged in queue_history.txt.
WAIT_PID=$1; Q=$(readlink -f "$2")
D=/home/sareeves/playground/esm-msr-devel
HERE=$(dirname "$(readlink -f "$0")"); WT=$(dirname "$HERE")
P=$D/probes; mkdir -p $P
log() { echo "$(date) $*" >> $D/queue_history.txt; }
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
for _ in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 12000 ] && break; sleep 30; done
export PYTHONPATH=$WT/src HF_HUB_OFFLINE=1 PROBE_GPU_FRACTION=0.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
probe() {
  local out=$1; shift
  for micro in 64 38; do
    timeout 5400 $PY $HERE/grad_share_probe.py $out --gpu --batches 20 --proteins 40 --micro $micro "$@" > $out.log 2>&1
    grep -q PROBE_DONE $out.log && { log "PROBE $(basename $out) done (micro $micro)"; return 0; }
    sleep 20
  done
  log "PROBE $(basename $out) FAILED (see $out.log)"; return 1
}
log "PROBE start: grad-share probes for the --reg_balance constants (scripts/probe_then_queue.sh)"
probe $P/probe_ref_plain.json --ckpt $D/training_logs/o6g6_c10_hier/0/last.ckpt --batch_size 256
probe $P/probe_ref_packed.json --ckpt $D/training_logs/o6g6_c10_hier/0/last.ckpt --batch_size 400 --min_cond 200 --extra="--pack_pair_matrices"
probe $P/probe_mt16_plain.json --ckpt $D/training_logs/o6q01_mt16/0/last.ckpt --batch_size 256 --extra="--lora_rank_mt 16 --lora_alpha_mt 16"
$PY $HERE/balance_from_probe.py $P/reg_balance_oct08.json --plain $P/probe_ref_plain.json $P/probe_mt16_plain.json \
    --packed $P/probe_ref_packed.json > $P/balance.log 2>&1
log "PROBE constants -> $P/reg_balance_oct08.json: $(tr -d '\n ' < $P/balance.log | cut -c1-400)"
exec "$HERE/queue_runner.sh" "$Q"
