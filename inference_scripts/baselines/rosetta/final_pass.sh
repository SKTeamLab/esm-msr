#!/bin/bash
# After the driver has submitted everything and the queue has drained: resubmit every chunk without a DONE marker (3 h timeouts, crashes) once, with a 6 h limit.
ROOT=/lustre07/scratch/sareeves/esm-msr; cd $ROOT; OUT=rosetta_predictions_rocklin
while pgrep -u $USER -f "rocklin/drive.sh" > /dev/null; do sleep 120; done
while [ "$(squeue -u $USER -h -r -n ros-ddg-mg | wc -l)" -gt 0 ]; do sleep 120; done
find $OUT/ddg -maxdepth 2 -name DONE | sed "s#.*/chunk_0*\([0-9]*\)/DONE#\1#; s#^\$#0#" | sort > logs/rocklin_done_ids.txt
NC=$(( $(wc -l < $OUT/tasks/chunks.csv) - 1 ))
seq 0 $((NC-1)) | sort | comm -23 - logs/rocklin_done_ids.txt | sort -n > logs/rocklin_missing_final.txt
N=$(wc -l < logs/rocklin_missing_final.txt); echo "$(date) $N chunks without DONE"
[ $N -gt 0 ] || exit 0
J=$(sbatch --parsable --time=6:00:00 --array=$(paste -sd, logs/rocklin_missing_final.txt) cluster_inference_scripts/rocklin/ddg.sh 0)
echo "$(date) final pass: $N chunks, 6h, job $J"
