#!/bin/bash
# Usage: scripts/premask_next.sh MASK_ARM_PID CPU_BUILD_PID
# Waits for the structure-mask arm to end, then finishes the pre-encoder (--premask_mt_structure) cache on the GPU (stopping the slower CPU build
# first and deleting the cache file it may have been writing), and runs the pre-encoder arm with the same weights as the mask arm (10,10,10).
D=/home/sareeves/playground/esm-msr-devel
S=$(dirname "$(readlink -f "$0")")
export COMET_PROJECT=esm-msr-agent-oct06-capped PYTHONPATH=$S/../src
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "$(date) mask arm ended; finishing the premask cache on the GPU" >> $D/queue_history.txt
if kill -0 "$2" 2>/dev/null; then
  kill -9 "$2"; sleep 3
  newest=$(ls -t $D/cache_v7/*_mtpremask.pkl 2>/dev/null | head -1); [ -n "$newest" ] && rm -f "$newest"    # possibly truncated by the kill
fi
$PY $S/build_cache.py --device cuda --threads 4 --premask_mt_structure > $D/run_logs/build_premask_cache_gpu.log 2>&1
if ! tail -3 $D/run_logs/build_premask_cache_gpu.log | grep -q BUILD_DONE; then
  echo "$(date) premask cache build FAILED; o6gp1_premask not started (see run_logs/build_premask_cache_gpu.log)" >> $D/queue_history.txt; exit 1
fi
exec $S/run_with_retry.sh o6gp1_premask 6 1 --include_out_of_range --reg_balance --premask_mt_structure --mt_comp_offset 10 --mt_comp_subst 10 --mt_comp_int 10
