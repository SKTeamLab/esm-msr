#!/bin/bash
# Resubmits the chunks listed in logs/rocklin_missing.txt (timed out at 3 h) with a 6 h limit, as soon as the queue has room (limit 1000, driver gates at 900).
ROOT=/lustre07/scratch/sareeves/esm-msr; cd $ROOT
LIST=$(paste -sd, logs/rocklin_missing.txt); N=$(wc -l < logs/rocklin_missing.txt)
while [ $(( $(squeue -u $USER -h -r | wc -l) + N )) -gt 998 ]; do sleep 30; done
J=$(sbatch --parsable --time=6:00:00 --array=$LIST cluster_inference_scripts/rocklin/ddg.sh 0)
echo "$(date) retry of $N timed-out chunks, 6h, job $J" >> logs/rocklin_drive.log
